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
from scripts.strata_kaggle import (agent_smoke, benchmark_api, environment_report,
                                   publish_connection, session_identity, serve_until_idle,
                                   validate_config)
from scripts.strata_kaggle import find_attached_dataset, attached_datasets, dataset_mount_report


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
        value = opencode_config_snippet("https://sample.trycloudflare.com", "qwen3.8-flash-next-gyro-gyro-s",
                                        context_tokens=65536)
        import json
        parsed = json.loads(value)
        self.assertIn("strata", parsed)
        settings = parsed["strata"]["settings"]
        self.assertEqual(settings["apiKey"], "{file:~/.config/opencode/strata-key.txt}")
        self.assertEqual(settings["headerTimeout"], 900000)
        self.assertEqual(settings["chunkTimeout"], 900000)
        model = parsed["strata"]["models"]["qwen3.8-flash-next-gyro-gyro-s"]
        self.assertEqual(model["compatibility"]["reasoningField"], "reasoning_content")
        self.assertEqual(model["limit"]["context"], 65536)
        self.assertTrue(model["capabilities"]["tools"])
        self.assertNotIn("reasoningEffort", model)
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


class ConfigValidationTests(unittest.TestCase):
    def test_validate_config_returns_context_tokens(self):
        config = validate_config("gyro", "gyro-s", "32k", "auto", "off", "copy", "0,1")
        self.assertEqual(config["context_tokens"], 32768)
        self.assertEqual(config["gpu_ids"], "0,1")
        self.assertEqual(config["family"], "gyro")

    def test_validate_config_rejects_bad_values(self):
        cases = [
            dict(family="", variant="gyro-s", context="32k"),
            dict(family="gyro", variant="", context="32k"),
            dict(family="gyro", variant="gyro-s", context="40k"),
            dict(family="gyro", variant="gyro-s", context="32k", drafting="turbo"),
            dict(family="gyro", variant="gyro-s", context="32k", vision="maybe"),
            dict(family="gyro", variant="gyro-s", context="32k", restore_mode="move"),
            dict(family="gyro", variant="gyro-s", context="32k", gpu_list="0;1"),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    validate_config(**case)

    def test_validate_config_allows_blank_gpu_list(self):
        self.assertEqual(validate_config("qwen36", "UD-Q4_K_XL", "16k")["gpu_ids"], "")

    def test_environment_report_fails_without_a_gpu(self):
        with patch("scripts.strata_kaggle.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                environment_report(require_cuda=True)
        with patch("scripts.strata_kaggle.shutil.which", return_value=None):
            self.assertEqual(environment_report(require_cuda=False)["gpus"], [])


class SessionTests(unittest.TestCase):
    def test_session_identity_reads_kaggle_container_name(self):
        with patch.dict(os.environ, {"KAGGLE_CONTAINER_NAME": "someone-kernel-12345-webtier"}):
            self.assertEqual(session_identity()["session_id"], 12345)

    def test_session_identity_is_zero_when_unknown(self):
        with patch.dict(os.environ, {"KAGGLE_CONTAINER_NAME": ""}, clear=False):
            self.assertEqual(session_identity()["session_id"], 0)

    def test_serve_until_idle_stops_after_the_idle_window(self):
        stopped = []
        activity = {"requests": 1, "last_request_at": 10, "in_flight": 0}
        with patch("scripts.strata_kaggle.server_activity", return_value=activity), \
                patch("scripts.strata_kaggle.time.sleep", return_value=None), \
                patch("scripts.strata_kaggle.time.monotonic", side_effect=range(0, 40)):
            summary = serve_until_idle(idle_seconds=5, serve_seconds=0, poll_seconds=1,
                                       key="k", stop=lambda: stopped.append(True))
        self.assertEqual(summary["reason"], "idle_timeout")
        self.assertEqual(stopped, [True])
        self.assertEqual(summary["requests_served"], 1)

    def test_serve_until_idle_stops_at_the_serve_deadline(self):
        stopped = []
        activity = {"requests": 7, "last_request_at": 10, "in_flight": 1}
        with patch("scripts.strata_kaggle.server_activity", return_value=activity), \
                patch("scripts.strata_kaggle.time.sleep", return_value=None), \
                patch("scripts.strata_kaggle.time.monotonic", side_effect=range(0, 40)):
            summary = serve_until_idle(idle_seconds=0, serve_seconds=3, poll_seconds=1,
                                       key="k", stop=lambda: stopped.append(True))
        self.assertEqual(summary["reason"], "serve_deadline")
        self.assertEqual(stopped, [True])

    def test_serve_until_idle_refuses_to_wait_forever(self):
        with self.assertRaises(ValueError):
            serve_until_idle(idle_seconds=0, serve_seconds=0)

    def test_serve_until_idle_needs_activity_or_a_deadline(self):
        stopped = []
        with patch("scripts.strata_kaggle.server_activity", return_value=None):
            with self.assertRaises(RuntimeError):
                serve_until_idle(idle_seconds=300, serve_seconds=0,
                                 stop=lambda: stopped.append(True))
        self.assertEqual(stopped, [True])


class QualityGateTests(unittest.TestCase):
    def _reply(self, content=None, tool_calls=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"message": message, "tool_calls": tool_calls or [], "usage": {},
                "timings": {}, "finish_reason": "tool_calls" if tool_calls else "stop",
                "ttft_s": 1.0, "elapsed_s": 2.0}

    def test_benchmark_averages_server_timings(self):
        replies = [
            self._reply("a") | {"timings": {"prompt_per_second": 100, "predicted_per_second": 20},
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            self._reply("b") | {"timings": {"prompt_per_second": 200, "predicted_per_second": 30},
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        ]
        with patch("scripts.strata_kaggle.stream_chat", side_effect=replies):
            result = benchmark_api(words=(128,), repeats=2)
        self.assertEqual(result["runs"][0]["generation_tokens_per_second"], 20)
        row = result["summary"][0]
        self.assertEqual(row["prompt_tokens_per_second"], 150)
        self.assertEqual(row["generation_tokens_per_second"], 25)
        self.assertEqual(row["samples"], 2)

    def test_benchmark_falls_back_to_wall_time(self):
        reply = {"timings": {}, "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                 "ttft_s": 1.0, "elapsed_s": 3.0, "message": {}, "tool_calls": [],
                 "finish_reason": "stop"}
        with patch("scripts.strata_kaggle.stream_chat", return_value=reply):
            row = benchmark_api(words=(64,))["runs"][0]
        self.assertEqual(row["prompt_tokens_per_second"], 50.0)
        self.assertEqual(row["generation_tokens_per_second"], 5.0)

    def test_agent_smoke_completes_read_write_test(self):
        replies = [
            self._reply(tool_calls=[{"id": "1", "type": "function",
                                     "function": {"name": "read_file", "arguments": "{}"}}]),
            self._reply(tool_calls=[{"id": "2", "type": "function",
                                     "function": {"name": "write_file",
                                                  "arguments": '{"content": "def answer():\\n    return 42\\n"}'}}]),
            self._reply(tool_calls=[{"id": "3", "type": "function",
                                     "function": {"name": "run_test", "arguments": "{}"}}]),
            self._reply("Fixed and tested."),
        ]
        with patch("scripts.strata_kaggle.stream_chat", side_effect=replies):
            result = agent_smoke()
        self.assertTrue(result["ok"])
        self.assertEqual(result["runs"][0]["local_tools"],
                         ["read_file", "write_file", "run_test"])

    def test_agent_smoke_rejects_an_edit_before_the_read(self):
        reply = self._reply(tool_calls=[{"id": "1", "type": "function",
                                         "function": {"name": "write_file",
                                                      "arguments": '{"content": "x"}'}}])
        with patch("scripts.strata_kaggle.stream_chat", return_value=reply):
            with self.assertRaises(RuntimeError):
                agent_smoke(rounds=2)

    def test_agent_smoke_gives_up_after_the_turn_limit(self):
        empty = self._reply("thinking")
        with patch("scripts.strata_kaggle.stream_chat", return_value=empty):
            with self.assertRaises(RuntimeError):
                agent_smoke(rounds=2)


class ConnectionTests(unittest.TestCase):
    def test_publish_connection_without_a_key_shows_only_the_url(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            connection = publish_connection("https://sample.trycloudflare.com", api_key="sk-strata-x",
                                            model="m")
        self.assertEqual(connection["base_url"], "https://sample.trycloudflare.com")
        self.assertNotIn("sk-strata-x", output.getvalue())
        self.assertIn("STRATA_CONNECTION=", output.getvalue())

    @unittest.skipUnless(__import__("shutil").which("openssl"), "OpenSSL is not installed")
    def test_publish_connection_encrypts_the_key_for_the_client(self):
        import json
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            key_path = Path(td) / "client-key.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "RSA",
                            "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(key_path)],
                           check=True, capture_output=True)
            public = subprocess.run(["openssl", "pkey", "-in", str(key_path), "-pubout"],
                                    check=True, capture_output=True, text=True).stdout
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                publish_connection("https://sample.trycloudflare.com",
                                   client_public_key=public, api_key="sk-strata-roundtrip",
                                   model="m")
            payload = output.getvalue().split("STRATA_CONNECTION=", 1)[1].strip().splitlines()[0]
            connection = json.loads(payload)
            self.assertNotIn("sk-strata-roundtrip", output.getvalue())
            decrypted = subprocess.run(
                ["openssl", "pkeyutl", "-decrypt", "-inkey", str(key_path),
                 "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256",
                 "-pkeyopt", "rsa_mgf1_md:sha256"],
                input=__import__("base64").b64decode(connection["encrypted_api_key"]),
                capture_output=True, check=True).stdout
            self.assertEqual(decrypted, b"sk-strata-roundtrip")

    def test_publish_connection_rejects_a_non_pem_key(self):
        with self.assertRaises(ValueError):
            publish_connection("https://sample.trycloudflare.com", client_public_key="nope",
                               api_key="k", model="m")


class ModelManifestSafetyTests(unittest.TestCase):
    def test_restore_model_cache_rejects_a_traversing_dataset_name(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "test-owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            (root / "input" / "escape.gguf").write_bytes(b"x")
            manifest = {
                "format": "devzonex.strata-model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{"dataset_name": "../../escape.gguf",
                           "relative_path": "models/gyro-Gyro-S/model.gguf", "size": 1}],
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with self.assertRaises(RuntimeError):
                    restore_model_cache("test-owner", "strata-model-gyro-gyro-s",
                                        root / "local-model", family="gyro",
                                        variant="gyro-s", restore_mode="copy")


class DatasetMountTests(unittest.TestCase):
    def test_finds_a_dataset_mounted_without_the_owner_folder(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            input_root.mkdir(parents=True)
            flat = Path(td) / "input" / "strata-model-gyro-gyro-s"
            flat.mkdir(parents=True)
            (flat / "manifest.json").write_text("{}", encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                found = find_attached_dataset("owner", "strata-model-gyro-gyro-s", "manifest.json")
            self.assertEqual(found, flat)

    def test_finds_the_owner_slug_layout(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            nested = input_root / "owner" / "strata-runtime-cache"
            nested.mkdir(parents=True)
            (nested / "runtime-manifest.json").write_text("{}", encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                found = find_attached_dataset("owner", "strata-runtime-cache",
                                              "runtime-manifest.json")
            self.assertEqual(found, nested)

    def test_missing_dataset_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            input_root.mkdir(parents=True)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                self.assertIsNone(find_attached_dataset("owner", "absent", "manifest.json"))

    def test_attached_datasets_lists_both_mount_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            flat = Path(td) / "input" / "strata-runtime-cache"
            nested = input_root / "owner" / "strata-model-gyro-gyro-s"
            for path in (flat, nested):
                path.mkdir(parents=True)
                (path / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                found = attached_datasets()
            self.assertEqual(sorted(str(path) for path in found),
                             sorted([str(flat), str(nested)]))

    def test_attached_datasets_ignores_folders_without_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            (input_root / "owner" / "not-a-dataset").mkdir(parents=True)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                self.assertEqual(attached_datasets(), [])
                report = dataset_mount_report()
            self.assertEqual(report["attached"], [])
            self.assertIn(str(input_root), report["roots"])

    def test_attached_datasets_does_not_walk_dataset_contents(self):
        # A huge dataset should be reported without reading its files.
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            dataset = input_root / "owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            (dataset / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            (dataset / "manifest.json").write_text("{}", encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with patch.object(Path, "rglob", side_effect=AssertionError("walked the mount")):
                    self.assertEqual(len(attached_datasets()), 1)

    def test_draft_layer_is_found_from_a_flat_mount(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            input_root.mkdir(parents=True)
            flat = Path(td) / "input" / "strata-qfn-assets-lite"
            flat.mkdir(parents=True)
            (flat / "mtp_rt__experts.bin").write_bytes(b"e" * 4096)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_draft_layer("clothespin", "strata-qfn-assets-lite",
                                               Path(td) / "local", min_experts_bytes=1)
            self.assertTrue(restored)
            self.assertTrue((Path(td) / "local" / "mtp" / "rt" / "experts.bin").is_file())


if __name__ == "__main__":
    unittest.main()
