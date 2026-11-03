import os
import hashlib
import contextlib
import io
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.strata_kaggle import redact, safe_extract_runtime, _sha256, opencode_config_snippet, _cached_model_path, run_streaming, restore_model_cache, _copy_verified
from scripts.strata_kaggle import (_drafting_args, drafting_status, mtp_layer_path,
                                   restore_draft_layer, _draft_layer_entries)


class StrataKaggleTests(unittest.TestCase):
    def test_redacts_strata_key(self):
        text = "API key sk-strata-very-secret123\n"
        self.assertIn("sk-strata-<redacted>", redact(text))
        self.assertNotIn("very-secret123", redact(text))

    def test_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "f"
            path.write_bytes(b"hello")
            self.assertEqual(_sha256(path), hashlib.sha256(b"hello").hexdigest())

    def test_runtime_extractor_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "bad.tar"
            with tarfile.open(archive, "w:") as tf:
                info = tarfile.TarInfo("../evil")
                info.size = 1
                import io
                tf.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                safe_extract_runtime(archive, td_path / "out")
            self.assertFalse((td_path / "evil").exists())

    def test_runtime_extractor_rejects_cross_tree_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "cross-tree.tar"
            with tarfile.open(archive, "w:") as tf:
                info = tarfile.TarInfo("terminal/escape")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../components/engine"
                tf.addfile(info)
            with self.assertRaises(RuntimeError):
                safe_extract_runtime(archive, td_path / "out")


    def test_runtime_extractor_rewrites_internal_absolute_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            runtime_root = td_path / "strata-root"
            archive = td_path / "runtime.tar"
            with tarfile.open(archive, "w:") as tf:
                for directory in (
                    "terminal/0.2.6",
                    "terminal/0.2.6/bin",
                ):
                    info = tarfile.TarInfo(directory)
                    info.type = tarfile.DIRTYPE
                    tf.addfile(info)

                binary = tarfile.TarInfo("terminal/0.2.6/bin/strata")
                payload = b"strata-binary"
                binary.size = len(payload)
                tf.addfile(binary, io.BytesIO(payload))

                link = tarfile.TarInfo("terminal/current")
                link.type = tarfile.SYMTYPE
                link.linkname = str(runtime_root / "terminal" / "0.2.6")
                tf.addfile(link)

            destination = td_path / "staging"
            with patch("scripts.strata_kaggle.STRATA_ROOT", runtime_root):
                safe_extract_runtime(archive, destination)

            current = destination / "terminal" / "current"
            self.assertTrue(current.is_symlink())
            self.assertEqual(os.readlink(current), "0.2.6")
            self.assertEqual(
                (current / "bin" / "strata").read_bytes(),
                b"strata-binary",
            )

    def test_runtime_extractor_rejects_external_absolute_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "external-link.tar"
            with tarfile.open(archive, "w:") as tf:
                link = tarfile.TarInfo("terminal/escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tf.addfile(link)

            with patch("scripts.strata_kaggle.STRATA_ROOT", td_path / "strata-root"):
                with self.assertRaisesRegex(RuntimeError, "Unsafe absolute link"):
                    safe_extract_runtime(archive, td_path / "out")

    def test_cached_model_path_selects_expected_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models" / "gyro-Gyro-S"
            model_dir.mkdir(parents=True)
            (model_dir / "model.gguf").write_bytes(b"model")
            found = _cached_model_path(root, "gyro", "gyro-s")
            self.assertEqual(found, model_dir)


    def test_run_streaming_prints_subprocess_output(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_streaming(
                [sys.executable, "-c", "print('progress-visible'); print('done')"],
                timeout=10,
            )
        self.assertIn("progress-visible", buffer.getvalue())
        self.assertIn("done", result.stdout)


    def test_run_streaming_reports_heartbeat(self):
        with tempfile.TemporaryDirectory() as td:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = run_streaming(
                    [sys.executable, "-c", "import time; time.sleep(0.7)"],
                    timeout=10,
                    heartbeat_interval=0.1,
                    heartbeat_label="test launch",
                    progress_paths=[Path(td)],
                )
            self.assertEqual(result.returncode, 0)
            self.assertIn("test launch still running", buffer.getvalue())
            self.assertIn("0.00 GiB", buffer.getvalue())


    def test_copy_verified_reports_byte_progress_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "source.bin"
            dst = root / "dest.bin"
            payload = b"progress-bytes-123"
            src.write_bytes(payload)
            events = []

            _copy_verified(
                src,
                dst,
                lambda amount, path, event: events.append((amount, path, event)),
            )

            self.assertEqual(dst.read_bytes(), payload)
            self.assertEqual(
                sum(amount for amount, _path, event in events if event == "progress"),
                len(payload),
            )
            self.assertEqual(events[0][2], "start")
            self.assertEqual(events[-1][2], "complete")

    def test_restore_model_cache_copy_mode_reports_byte_progress(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "test-owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            payload = b"model-progress-data"
            flattened = dataset / "0001__model.gguf"
            flattened.write_bytes(payload)
            manifest = {
                "format": "devzonex.strata-model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{
                    "dataset_name": flattened.name,
                    "relative_path": "models/gyro-Gyro-S/model.gguf",
                    "size": len(payload),
                }],
            }
            (dataset / "manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            output = io.StringIO()
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with contextlib.redirect_stdout(output):
                    restored = restore_model_cache(
                        "test-owner",
                        "strata-model-gyro-gyro-s",
                        root / "local-model",
                        family="gyro",
                        variant="gyro-s",
                        restore_mode="copy",
                    )

            self.assertTrue(restored)
            self.assertIn("Model restore:", output.getvalue())
            self.assertIn("100.0%", output.getvalue())
            self.assertIn("MiB/s", output.getvalue())

    def test_restore_model_cache_symlink_mode_avoids_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "test-owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            payload = b"model-bytes"
            flattened = dataset / "0001__model.gguf"
            flattened.write_bytes(payload)
            manifest = {
                "format": "devzonex.strata-model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{
                    "dataset_name": flattened.name,
                    "relative_path": "models/gyro-Gyro-S/model.gguf",
                    "size": len(payload),
                }],
            }
            (dataset / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
            target_root = root / "local-model"
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_model_cache(
                    "test-owner", "strata-model-gyro-gyro-s", target_root,
                    family="gyro", variant="gyro-s", restore_mode="symlink",
                )
            linked = target_root / "models" / "gyro-Gyro-S" / "model.gguf"
            self.assertTrue(restored)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.read_bytes(), payload)
            self.assertEqual(linked.resolve(), flattened.resolve())

    def test_open_code_snippet_has_provider_wrapper_and_no_key(self):
        value = opencode_config_snippet("https://sample.trycloudflare.com", "qwen3.8-flash-next-gyro-gyro-s")
        import json
        parsed = json.loads(value)
        self.assertIn("strata", parsed)
        self.assertEqual(parsed["strata"]["settings"]["apiKey"], "{file:~/.config/opencode/strata-key.txt}")
        self.assertEqual(
            parsed["strata"]["models"]["qwen3.8-flash-next-gyro-gyro-s"]["reasoningEffort"],
            "none",
        )
        self.assertNotIn("sk-strata-", value)


class DraftingTests(unittest.TestCase):
    def test_drafting_args_defaults_to_the_draft_layer(self):
        self.assertEqual(_drafting_args("auto"), ["--mtp", "on", "--drafting", "auto"])
        self.assertEqual(_drafting_args("MTP"), ["--mtp", "on", "--drafting", "mtp"])
        self.assertEqual(_drafting_args("off"), ["--mtp", "off", "--drafting", "off"])
        self.assertEqual(_drafting_args("dflash2"), ["--mtp", "off", "--drafting", "dflash2"])
        self.assertEqual(_drafting_args(None), ["--mtp", "on", "--drafting", "auto"])

    def test_drafting_args_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            _drafting_args("sometimes")

    def test_drafting_status_reports_missing_layer(self):
        with tempfile.TemporaryDirectory() as td:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = drafting_status(Path(td), "auto")
            self.assertFalse(status["ready"])
            self.assertTrue(status["needs_mtp_layer"])
            self.assertEqual(status["mtp_fetch_gb"], 5.2)
            self.assertIn("5.2 GB", output.getvalue())

    def test_drafting_status_skips_layer_when_drafting_off(self):
        with tempfile.TemporaryDirectory() as td:
            status = drafting_status(Path(td), "off")
            self.assertTrue(status["ready"])
            self.assertFalse(status["needs_mtp_layer"])
            self.assertEqual(status["mtp_fetch_gb"], 0.0)

    def test_drafting_status_finds_existing_layer(self):
        with tempfile.TemporaryDirectory() as td:
            layer = Path(td) / "mtp" / "rt"
            layer.mkdir(parents=True)
            (layer / "experts.bin").write_bytes(b"x")
            self.assertEqual(mtp_layer_path(Path(td)), layer)
            status = drafting_status(Path(td), "auto")
            self.assertTrue(status["ready"])
            self.assertEqual(status["mtp_fetch_gb"], 0.0)

    def test_draft_layer_entries_reads_both_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "mtp_rt__experts.bin").write_bytes(b"a")
            (root / "mtp_rt__draft_vocab.bin").write_bytes(b"b")
            entries = _draft_layer_entries(root)
            self.assertEqual(sorted(entries), ["draft_vocab.bin", "experts.bin"])

            nested = root / "nested"
            (nested / "mtp" / "rt").mkdir(parents=True)
            (nested / "mtp" / "rt" / "experts.bin").write_bytes(b"c")
            self.assertEqual(list(_draft_layer_entries(nested)), ["experts.bin"])

    def test_restore_draft_layer_copies_flattened_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "clothespin" / "strata-qfn-assets-lite"
            dataset.mkdir(parents=True)
            (dataset / "mtp_rt__experts.bin").write_bytes(b"e" * 4096)
            (dataset / "mtp_rt__draft_vocab.bin").write_bytes(b"v" * 16)
            local = root / "local"
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_draft_layer(
                    "clothespin", "strata-qfn-assets-lite", local, restore_mode="copy",
                    min_experts_bytes=1)
            self.assertTrue(restored)
            self.assertEqual((local / "mtp" / "rt" / "experts.bin").read_bytes(), b"e" * 4096)
            self.assertEqual((local / "mtp" / "rt" / "draft_vocab.bin").read_bytes(), b"v" * 16)

    def test_restore_draft_layer_symlinks_when_asked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "clothespin" / "strata-qfn-assets-lite"
            dataset.mkdir(parents=True)
            source = dataset / "mtp_rt__experts.bin"
            source.write_bytes(b"e" * 4096)
            local = root / "local"
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_draft_layer(
                    "clothespin", "strata-qfn-assets-lite", local, restore_mode="symlink",
                    min_experts_bytes=1)
            target = local / "mtp" / "rt" / "experts.bin"
            self.assertTrue(restored)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())

    def test_restore_draft_layer_skips_when_already_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            layer = root / "mtp" / "rt"
            layer.mkdir(parents=True)
            (layer / "experts.bin").write_bytes(b"x")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                restored = restore_draft_layer("owner", "slug", root)
            self.assertFalse(restored)
            self.assertIn("already present", output.getvalue())

    def test_restore_draft_layer_reports_a_missing_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            input_root.mkdir(parents=True)
            output = io.StringIO()
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with contextlib.redirect_stdout(output):
                    restored = restore_draft_layer("owner", "slug", root / "local")
            self.assertFalse(restored)
            self.assertIn("not attached", output.getvalue())

    def test_restore_draft_layer_rejects_a_dataset_without_a_layer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "owner" / "slug"
            dataset.mkdir(parents=True)
            (dataset / "mtp_rt__draft_vocab.bin").write_bytes(b"v")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with self.assertRaises(RuntimeError):
                    restore_draft_layer("owner", "slug", root / "local")

    def test_restore_draft_layer_default_threshold_rejects_tiny_experts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "owner" / "slug"
            dataset.mkdir(parents=True)
            (dataset / "mtp_rt__experts.bin").write_bytes(b"e" * 4096)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with self.assertRaises(RuntimeError) as caught:
                    restore_draft_layer("owner", "slug", root / "local")
            self.assertIn("bytes are required", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
