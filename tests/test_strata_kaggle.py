import ast
import os
import hashlib
import contextlib
import io
import json
import subprocess
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
from scripts.strata_kaggle import find_model_cache, describe_inputs
from scripts.strata_kaggle import adopt_model_bundle, _model_source_args
from scripts.strata_kaggle import (upload_model_cache, upload_draft_layer, cache_status,
                                   DRAFT_LAYER_PREFIX, DRAFT_LAYER_FORMAT,
                                   setup_model_name, strata_terminal_python,
                                   find_strata_setup, prepare_model_on_cpu)
import scripts.strata_kaggle as strata_kaggle


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


class ModelDiscoveryTests(unittest.TestCase):
    def _manifest(self, entries, family="gyro", variant="gyro-s"):
        return {"format": "devzonex.strata-model-cache.v1",
                "model": {"family": family, "variant": variant},
                "files": entries}

    def test_model_cache_is_found_inside_another_dataset(self):
        # A dataset named after the runtime cache can also carry the model cache.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "owner" / "strata-runtime-cache"
            dataset.mkdir(parents=True)
            (dataset / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            payload = b"weights-in-runtime-dataset"
            (dataset / "0001__model.gguf").write_bytes(payload)
            (dataset / "manifest.json").write_text(json.dumps(self._manifest([
                {"dataset_name": "0001__model.gguf",
                 "relative_path": "models/gyro-Gyro-S/model.gguf", "size": len(payload)}])),
                encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_model_cache("owner", "strata-model-gyro-gyro-s",
                                               root / "local", family="gyro",
                                               variant="gyro-s", restore_mode="copy")
            self.assertTrue(restored)
            self.assertEqual(
                (root / "local" / "models" / "gyro-Gyro-S" / "model.gguf").read_bytes(), payload)

    def test_manifest_one_level_down_finds_files_at_the_dataset_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "owner" / "strata-runtime-cache"
            nested = dataset / "strata-model-gyro-gyro-s"
            nested.mkdir(parents=True)
            payload = b"nested-manifest"
            (dataset / "0001__model.gguf").write_bytes(payload)
            (nested / "manifest.json").write_text(json.dumps(self._manifest([
                {"dataset_name": "0001__model.gguf",
                 "relative_path": "models/gyro-Gyro-S/model.gguf", "size": len(payload)}])),
                encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                found = find_model_cache("owner", "strata-model-gyro-gyro-s")
            self.assertIsNotNone(found)
            self.assertEqual(found[1], nested)
            self.assertEqual(found[2]["format"], "devzonex.strata-model-cache.v1")

    def test_a_dataset_without_a_model_manifest_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            dataset = input_root / "owner" / "strata-runtime-cache"
            dataset.mkdir(parents=True)
            (dataset / "manifest.json").write_text(json.dumps({"format": "something-else"}),
                                                   encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                self.assertIsNone(find_model_cache("owner", "strata-model-gyro-gyro-s"))

    def test_the_expected_dataset_wins_over_another_one(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            expected = input_root / "owner" / "strata-model-gyro-gyro-s"
            other = input_root / "owner" / "strata-runtime-cache"
            for path in (expected, other):
                path.mkdir(parents=True)
                (path / "manifest.json").write_text(
                    json.dumps(self._manifest([])), encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                found = find_model_cache("owner", "strata-model-gyro-gyro-s")
            self.assertEqual(found[0], expected)


    def test_attached_datasets_detects_mounts_without_metadata(self):
        # Newer Kaggle mounts contain no dataset-metadata.json; files are enough.
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            owner = input_root / "kvinodhkumar"
            runtime = owner / "strata-runtime-cache"
            runtime.mkdir(parents=True)
            (runtime / "strata-runtime-cache.bin").write_bytes(b"r" * 10)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                self.assertEqual(attached_datasets(), [runtime])

    def test_describe_inputs_reports_what_each_dataset_holds(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            runtime = input_root / "strata-runtime-cache"
            runtime.mkdir(parents=True)
            (runtime / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            (runtime / "strata-runtime-cache.bin").write_bytes(b"r" * 100)
            model = input_root / "alpha-orionis-xs-betelgeuse"
            model.mkdir(parents=True)
            (model / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            payload = b"w" * 50
            (model / "0001__model.gguf").write_bytes(payload)
            (model / "manifest.json").write_text(json.dumps({
                "format": "devzonex.strata-model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{"dataset_name": "0001__model.gguf",
                           "relative_path": "models/gyro-Gyro-S/model.gguf",
                           "size": len(payload)}]}), encoding="utf-8")
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                report = {entry["name"]: entry for entry in describe_inputs()}
            self.assertIn("runtime_cache_gib", report["strata-runtime-cache"])
            self.assertEqual(report["alpha-orionis-xs-betelgeuse"]["model_cache"]["family"],
                             "gyro")
            self.assertEqual(report["alpha-orionis-xs-betelgeuse"]["model_cache"]["files"], 1)


class ModelBundleTests(unittest.TestCase):
    def _bundle(self, root, files):
        input_root = root / "input" / "datasets"
        bundle = input_root / "owner" / "alpha-orionis-xs-betelgeuse"
        bundle.mkdir(parents=True)
        for name, payload in files.items():
            (bundle / name).write_bytes(payload)
        return input_root, bundle

    def test_adopted_bundle_makes_the_launch_use_a_local_model_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root, _bundle = self._bundle(root, {
                "Qwen3.8-Flash-Next-Dense.gguf": b"dense",
                "Qwen3.8-Flash-Next-PLE.gguf": b"ple",
                "experts.bin": b"experts",
                "profile-qwen4exp.bin": b"profile",
            })
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                target = adopt_model_bundle("owner", "alpha-orionis-xs-betelgeuse",
                                            root / "local", "gyro", "gyro-s")
                self.assertEqual(target, root / "local" / "models" / "gyro-gyro-s")
                self.assertTrue((target / "experts.bin").is_symlink())
                self.assertEqual((target / "experts.bin").read_bytes(), b"experts")
                self.assertEqual(sorted(path.name for path in target.iterdir()),
                                 ["Qwen3.8-Flash-Next-Dense.gguf",
                                  "Qwen3.8-Flash-Next-PLE.gguf",
                                  "experts.bin", "profile-qwen4exp.bin"])
                self.assertEqual(_model_source_args(root / "local", "gyro", "gyro-s"),
                                 ["--model-path", str(target)])

    def test_adopted_bundle_can_be_copied(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root, _bundle = self._bundle(root, {
                "model.gguf": b"w" * 32, "experts.bin": b"e" * 8})
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                target = adopt_model_bundle("owner", "alpha-orionis-xs-betelgeuse",
                                            root / "local", "gyro", "gyro-s",
                                            restore_mode="copy")
            self.assertFalse((target / "model.gguf").is_symlink())
            self.assertEqual((target / "model.gguf").read_bytes(), b"w" * 32)

    def test_adopting_a_bundle_that_does_not_fit_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root, _bundle = self._bundle(root, {"model.gguf": b"w" * 32})
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root), \
                    patch("scripts.strata_kaggle.shutil.disk_usage") as usage:
                usage.return_value = type("Usage", (), {"free": 8})()
                with self.assertRaises(RuntimeError) as caught:
                    adopt_model_bundle("owner", "alpha-orionis-xs-betelgeuse",
                                       root / "local", "gyro", "gyro-s",
                                       restore_mode="copy")
            self.assertIn("symlink", str(caught.exception))

    def test_a_kaggle_model_mount_is_found_by_owner_and_slug(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            version = (root / "input" / "models" / "buianhtruc"
                       / "strata-qwen3-8-flash-next-gyro-m-tq2-0"
                       / "gguf" / "default" / "1")
            version.mkdir(parents=True)
            (version / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf").write_bytes(b"g" * 8)
            (version / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf.done").write_text("done")
            with patch.object(strata_kaggle, "INPUT_ROOT", input_root):
                found = find_attached_dataset(
                    "buianhtruc", "strata-qwen3-8-flash-next-gyro-m-tq2-0")
            self.assertEqual(found, version)

    def test_an_attached_kaggle_model_can_be_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            version = (root / "input" / "models" / "buianhtruc"
                       / "strata-qwen3-8-flash-next-gyro-m-tq2-0"
                       / "gguf" / "default" / "1")
            version.mkdir(parents=True)
            (version / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf").write_bytes(b"g" * 8)
            (version / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf.done").write_text("done")
            with patch.object(strata_kaggle, "INPUT_ROOT", input_root):
                target = adopt_model_bundle(
                    "buianhtruc", "strata-qwen3-8-flash-next-gyro-m-tq2-0",
                    root / "local", "gyro", "gyro-m")
            self.assertTrue((target / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf").is_symlink())
            self.assertEqual((target / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf").read_bytes(),
                             b"g" * 8)

    def test_adopting_a_missing_dataset_explains_what_to_do(self):
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "input" / "datasets"
            input_root.mkdir(parents=True)
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with self.assertRaises(RuntimeError) as caught:
                    adopt_model_bundle("owner", "absent", Path(td) / "local",
                                       "gyro", "gyro-s")
            self.assertIn("not attached", str(caught.exception))

    def test_a_bundle_without_a_gguf_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root, _bundle = self._bundle(root, {"experts.bin": b"e"})
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with self.assertRaises(RuntimeError) as caught:
                    adopt_model_bundle("owner", "alpha-orionis-xs-betelgeuse",
                                       root / "local", "gyro", "gyro-s")
            self.assertIn("No .gguf", str(caught.exception))


class UploadTests(unittest.TestCase):
    """The model, packs and draft layer are published as separate datasets."""

    def _data_dir(self, root: Path) -> Path:
        data = root / "strata-local"
        (data / "models" / "gyro-s").mkdir(parents=True)
        (data / "models" / "gyro-s" / "experts.bin").write_bytes(b"model")
        (data / "packs").mkdir()
        (data / "packs" / "dense.pack").write_bytes(b"pack")
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_bytes(b"draft")
        (data / "mtp" / "rt" / "dense.bin").write_bytes(b"dense")
        return data

    def test_model_upload_excludes_the_draft_layer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = self._data_dir(root)
            stages = []
            with patch("scripts.strata_kaggle.WORKING", root), \
                    patch("scripts.strata_kaggle.find_attached_dataset", return_value=None), \
                    patch("scripts.strata_kaggle._upload_create_or_version",
                          side_effect=lambda stage, *a, **k: stages.append(stage)):
                uploaded = upload_model_cache(data, "owner", "strata-model-gyro-gyro-s",
                                              "gyro", "gyro-s", {}, license_confirmed=True)
            self.assertTrue(uploaded)
            manifest = json.loads((stages[0] / "manifest.json").read_text())
            paths = sorted(item["relative_path"] for item in manifest["files"])
            self.assertEqual(paths, ["models/gyro-s/experts.bin", "packs/dense.pack"])
            self.assertFalse(any(path.startswith("mtp/") for path in paths))
            self.assertEqual(manifest["format"], "devzonex.strata-model-cache.v1")
            self.assertEqual(sorted(p.name for p in stages[0].iterdir()),
                             ["0001__models__gyro-s__experts.bin", "0002__packs__dense.pack",
                              "dataset-metadata.json", "manifest.json"])

    def test_model_upload_needs_the_license_confirmation(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._data_dir(Path(td))
            with patch("scripts.strata_kaggle._upload_create_or_version") as upload:
                uploaded = upload_model_cache(data, "owner", "slug", "gyro", "gyro-s", {},
                                              license_confirmed=False)
            self.assertFalse(uploaded)
            upload.assert_not_called()

    def test_draft_layer_upload_uses_the_flat_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = self._data_dir(root)
            stages = []
            with patch("scripts.strata_kaggle.WORKING", root), \
                    patch("scripts.strata_kaggle.find_attached_dataset", return_value=None), \
                    patch("scripts.strata_kaggle._upload_create_or_version",
                          side_effect=lambda stage, *a, **k: stages.append(stage)):
                uploaded = upload_draft_layer(data, "owner", "strata-mtp-rt", {})
            self.assertTrue(uploaded)
            manifest = json.loads((stages[0] / "manifest.json").read_text())
            self.assertEqual(manifest["format"], DRAFT_LAYER_FORMAT)
            self.assertTrue(manifest["model_agnostic"])
            for item in manifest["files"]:
                self.assertTrue(item["dataset_name"].startswith(DRAFT_LAYER_PREFIX))
                self.assertIn(item["relative_path"], {"mtp/rt/experts.bin", "mtp/rt/dense.bin"})
            self.assertTrue((stages[0] / "mtp_rt__experts.bin").is_symlink())

    def test_draft_layer_upload_is_skipped_when_already_attached(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = self._data_dir(root)
            attached = root / "input"
            attached.mkdir()
            manifest = {"format": DRAFT_LAYER_FORMAT, "model_agnostic": True, "files": [
                {"relative_path": "mtp/rt/experts.bin", "size": 5},
                {"relative_path": "mtp/rt/dense.bin", "size": 5},
            ]}
            (attached / "manifest.json").write_text(json.dumps(manifest))
            with patch("scripts.strata_kaggle.WORKING", root), \
                    patch("scripts.strata_kaggle.find_attached_dataset", return_value=attached), \
                    patch("scripts.strata_kaggle._upload_create_or_version") as upload:
                uploaded = upload_draft_layer(data, "owner", "strata-mtp-rt", {})
            self.assertFalse(uploaded)
            upload.assert_not_called()

    def test_draft_layer_upload_without_a_layer_reports_and_stops(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "strata-local"
            (data / "models").mkdir(parents=True)
            with patch("scripts.strata_kaggle._upload_create_or_version") as upload:
                uploaded = upload_draft_layer(data, "owner", "strata-mtp-rt", {})
            self.assertFalse(uploaded)
            upload.assert_not_called()


class CacheStatusTests(unittest.TestCase):
    def test_cache_status_counts_symlinks_and_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "strata-local"
            (data / "models").mkdir(parents=True)
            (data / "models" / "real.bin").write_bytes(b"x" * 10)
            source = root / "dataset.bin"
            source.write_bytes(b"y" * 5)
            (data / "models" / "linked.bin").symlink_to(source)
            status = cache_status(data)
            self.assertEqual(status["parts"]["models"]["files"], 2)
            self.assertEqual(status["parts"]["models"]["links"], 1)
            self.assertEqual(status["parts"]["models"]["bytes"], 15)
            self.assertEqual(status["parts"]["packs"]["files"], 0)
            self.assertFalse(status["parts"]["packs"]["exists"])

    def test_environment_report_reports_the_requested_filesystem(self):
        with tempfile.TemporaryDirectory() as td, \
                patch("scripts.strata_kaggle.shutil.which", return_value=None):
            report = environment_report(require_cuda=False, path=td)
            self.assertEqual(report["disk_path"], td)
            self.assertIsNotNone(report["disk_free_gb"])


class DefaultRestoreModeTests(unittest.TestCase):
    def test_symlink_is_the_default_restore_mode(self):
        self.assertEqual(validate_config("gyro", "gyro-s", "32k")["restore_mode"],
                         "symlink")


class NotebookIntegrityTests(unittest.TestCase):
    def test_every_notebook_code_cell_parses(self):
        notebooks = Path(__file__).resolve().parent.parent / "notebooks"
        for path in sorted(notebooks.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for number, cell in enumerate(notebook["cells"], 1):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=path.name, cell=number):
                    ast.parse("".join(cell["source"]))


class CpuPrepareTests(unittest.TestCase):
    def test_terminal_python_is_read_from_version_line(self):
        with tempfile.TemporaryDirectory() as td:
            interpreter = Path(td) / "python3.12"
            interpreter.write_text("", encoding="utf-8")
            fake = subprocess.CompletedProcess(
                ["strata", "--version"], 0,
                "strata 0.2.5 (build 123; python 3.12.7 at " + str(interpreter) + ")\n", "")
            with patch.object(strata_kaggle.subprocess, "run", return_value=fake), \
                    patch.object(strata_kaggle.shutil, "which", return_value="/usr/bin/strata"):
                self.assertEqual(strata_terminal_python(), str(interpreter))

    def test_terminal_python_falls_back_to_the_kernel(self):
        fake = subprocess.CompletedProcess(["strata", "--version"], 0, "not the expected line\n", "")
        with patch.object(strata_kaggle.subprocess, "run", return_value=fake), \
                patch.object(strata_kaggle.shutil, "which", return_value="/usr/bin/strata"):
            self.assertEqual(strata_terminal_python(), sys.executable)

    def test_find_strata_setup_requires_the_tools_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "terminal" / "current").mkdir(parents=True)
            setup = root / "terminal" / "current" / "setup.py"
            setup.write_text("# setup\n", encoding="utf-8")
            with patch.object(strata_kaggle, "STRATA_ROOT", root), \
                    patch.object(strata_kaggle.shutil, "which", return_value=None), \
                    patch.object(strata_kaggle, "STRATA_BIN", root / "missing"):
                with self.assertRaises(RuntimeError):
                    find_strata_setup()
                tools = setup.parent / "tools"
                tools.mkdir()
                (tools / "iq_pack.py").write_text("# tool\n", encoding="utf-8")
                self.assertEqual(find_strata_setup(), setup)

    def test_prepare_model_on_cpu_passes_the_setup_spelling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = root / "setup.py"
            setup.write_text("# setup\n", encoding="utf-8")
            seen = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                return subprocess.CompletedProcess(
                    command, 0, 'CPU_PREPARE_RESULT {"model": "/data/models/gyro-Gyro-S/x.gguf"}\n', None)

            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable), \
                    patch.object(strata_kaggle, "run_streaming", side_effect=fake_run):
                summary = prepare_model_on_cpu(root / "data", "gyro", "gyro-s")
            config = json.loads(seen["command"][2])
            self.assertEqual(config["model"], "Gyro-S")
            self.assertEqual(config["family"], "gyro")
            self.assertTrue(config["download_model"])
            self.assertEqual(summary["model"], "/data/models/gyro-Gyro-S/x.gguf")
            self.assertEqual(summary["returncode"], 0)
            script = (root / "working" / "strata_cpu_prepare.py").read_text(encoding="utf-8")
            self.assertIn("setup.download(", script)
            self.assertIn("fetch_mtp", script)

    def test_prepare_model_on_cpu_reports_a_failed_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = root / "setup.py"
            setup.write_text("# setup\n", encoding="utf-8")
            failed = subprocess.CompletedProcess(["python"], 1, "the download stopped\n", None)
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable), \
                    patch.object(strata_kaggle, "run_streaming", return_value=failed):
                with self.assertRaises(RuntimeError) as caught:
                    prepare_model_on_cpu(root / "data", "gyro", "gyro-s")
            self.assertIn("the download stopped", str(caught.exception))

    def test_prepare_model_on_cpu_rejects_download_with_a_model_path(self):
        with self.assertRaises(ValueError):
            prepare_model_on_cpu(Path("/tmp"), "gyro", "gyro-s",
                                 download_model=True, model_path=Path("/tmp/bundle"))

    def test_prepare_model_on_cpu_needs_a_variant(self):
        with self.assertRaises(ValueError):
            prepare_model_on_cpu(Path("/tmp"), "gyro", "")

    def test_prepare_model_on_cpu_forwards_an_adopted_model_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = root / "setup.py"
            setup.write_text("# setup\n", encoding="utf-8")
            seen = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                return subprocess.CompletedProcess(command, 0, "CPU_PREPARE_RESULT {}\n", None)

            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable), \
                    patch.object(strata_kaggle, "run_streaming", side_effect=fake_run):
                prepare_model_on_cpu(
                    root / "data", "gyro", "gyro-m",
                    download_model=False, build_pack=True, fetch_draft=False,
                    model_path=root / "input" / "bundle")
            config = json.loads(seen["command"][2])
            self.assertEqual(config["model_path"], str(root / "input" / "bundle"))
            self.assertFalse(config["download_model"])
            self.assertFalse(config["fetch_draft"])

    def test_generated_script_builds_a_pack_from_an_adopted_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "input" / "bundle"
            bundle.mkdir(parents=True)
            (bundle / "model.gguf").write_bytes(b"g" * 8)
            (bundle / "model.gguf.done").write_text("done", encoding="utf-8")
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "from pathlib import Path\n"
                "FAMILIES = {'gyro': {'tag': 'gyro-', 'hf': 'http://example.invalid/', 'file': '{q}.gguf'}}\n"
                "def model_file(fam, model, i): return 'model.gguf'\n"
                "def get_llama_cpp(): return '.'\n"
                "def run(cmd, env=None):\n"
                "    out = Path(cmd[cmd.index('--out') + 1])\n"
                "    (out / 'tokenizer').mkdir(parents=True, exist_ok=True)\n"
                "    (out / 'tokenizer' / 'vocab.json').write_text('{}')\n"
                "    (out / 'native_experts.txt').write_text('')\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                summary = prepare_model_on_cpu(
                    root / "data", "gyro", "gyro-m",
                    download_model=False, build_pack=True, fetch_draft=False,
                    model_path=bundle)
            self.assertEqual(summary["returncode"], 0)
            self.assertTrue(summary["pack_ok"])
            self.assertEqual(summary["model_bytes"], 8)

    def test_generated_script_uses_the_first_shard_of_a_multi_shard_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "input" / "bundle"
            bundle.mkdir(parents=True)
            (bundle / "model-00001-of-00002.gguf").write_bytes(b"a" * 8)
            (bundle / "model-00002-of-00002.gguf").write_bytes(b"b" * 4)
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "from pathlib import Path\n"
                "FAMILIES = {'qwen': {'tag': '', 'hf': 'http://example.invalid/', 'file': '{q}-0000{i}-of-00002.gguf'}}\n"
                "def model_file(fam, model, i): return 'model-0000%d-of-00002.gguf' % i\n"
                "def get_llama_cpp(): return '.'\n"
                "def run(cmd, env=None):\n"
                "    out = Path(cmd[cmd.index('--out') + 1])\n"
                "    (out / 'tokenizer').mkdir(parents=True, exist_ok=True)\n"
                "    (out / 'tokenizer' / 'vocab.json').write_text('{}')\n"
                "    (out / 'native_experts.txt').write_text('')\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                summary = prepare_model_on_cpu(
                    root / "data", "qwen", "iq3-xxs",
                    download_model=False, build_pack=True, fetch_draft=False,
                    model_path=bundle)
            self.assertEqual(summary["returncode"], 0)
            self.assertTrue(summary["pack_ok"])
            self.assertEqual(summary["model_bytes"], 8)

    def test_generated_script_accepts_a_lowercase_adopted_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            model_dir = data / "models" / "gyro-gyro-m"
            model_dir.mkdir(parents=True)
            (model_dir / "model.gguf").write_bytes(b"g" * 8)
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "from pathlib import Path\n"
                "FAMILIES = {'gyro': {'tag': 'gyro-', 'hf': 'http://example.invalid/', 'file': '{q}.gguf'}}\n"
                "def model_file(fam, model, i): return 'model.gguf'\n"
                "def get_llama_cpp(): return '.'\n"
                "def run(cmd, env=None):\n"
                "    out = Path(cmd[cmd.index('--out') + 1])\n"
                "    (out / 'tokenizer').mkdir(parents=True, exist_ok=True)\n"
                "    (out / 'tokenizer' / 'vocab.json').write_text('{}')\n"
                "    (out / 'native_experts.txt').write_text('')\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                summary = prepare_model_on_cpu(
                    data, "gyro", "gyro-m",
                    download_model=False, build_pack=True, fetch_draft=False)
            self.assertEqual(summary["returncode"], 0)
            self.assertTrue(summary["pack_ok"])
            self.assertEqual(summary["model_bytes"], 8)

    def test_generated_script_reports_a_missing_helper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "FAMILIES = {'gyro': {'tag': 'gyro-', 'hf': 'http://example.invalid/', 'file': '{q}.gguf'}}\n"
                "def model_file(fam, model, i): return 'x.gguf'\n"
                "def get_llama_cpp(): return '.'\n"
                "def download(url, dst): dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(b'x')\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                with self.assertRaises(RuntimeError) as caught:
                    prepare_model_on_cpu(root / "data", "gyro", "gyro-s")
            self.assertIn("has no", str(caught.exception))
            self.assertIn("fetch_mtp", str(caught.exception))

    def test_draft_only_pass_runs_without_the_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "from pathlib import Path\n"
                "FAMILIES = {'gyro': {'tag': 'gyro-', 'hf': 'http://example.invalid/', 'file': '{q}.gguf'}}\n"
                "def model_file(fam, model, i): return 'x.gguf'\n"
                "def get_llama_cpp(): return '.'\n"
                "def fetch_mtp(mtp, env):\n"
                "    (Path(mtp) / 'rt').mkdir(parents=True, exist_ok=True)\n"
                "    (Path(mtp) / 'rt' / 'experts.bin').write_bytes(b'x' * 4)\n"
                "def run(cmd, env=None): return None\n"
                "def refresh_draft_vocab(rt, choice='cjk'): return None\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                summary = prepare_model_on_cpu(
                    root / "data", "gyro", "gyro-s",
                    download_model=False, build_pack=False, fetch_draft=True)
            self.assertEqual(summary["returncode"], 0)
            self.assertEqual(summary["model_bytes"], 0)
            self.assertFalse(summary["pack_ok"])
            self.assertTrue(summary["draft_ok"])
            self.assertEqual(summary["warnings"], [])

    def test_generated_script_keeps_the_model_when_the_pack_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tools").mkdir()
            (root / "tools" / "iq_pack.py").write_text("# tool\n", encoding="utf-8")
            (root / "llama" / "gguf-py").mkdir(parents=True)
            fake_setup = root / "setup.py"
            fake_setup.write_text(
                "from pathlib import Path\n"
                "FAMILIES = {'gyro': {'title': 'Gyro', 'tag': 'gyro-', "
                "'hf': 'http://example.invalid/', 'file': '{q}.gguf', "
                "'pack_args': ['--compat-bf16'], 'shards': 1}}\n"
                "MODELS = {'Gyro-S': {'file': 'Qwen-Gyro-S.gguf', 'shards': 1}}\n"
                "def model_file(fam, model, i): return MODELS[model]['file']\n"
                "def download(url, dst):\n"
                "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
                "    dst.write_bytes(b'x' * 10)\n"
                "def get_llama_cpp(): return Path(__file__).parent / 'llama'\n"
                "def run(cmd, env=None): raise SystemExit(3)\n"
                "def fetch_mtp(mtp, env): print('fetch_mtp')\n"
                "def refresh_draft_vocab(rt, choice='cjk'): print('vocab')\n",
                encoding="utf-8")
            with patch.object(strata_kaggle, "WORKING", root / "working"), \
                    patch.object(strata_kaggle, "find_strata_setup", return_value=fake_setup), \
                    patch.object(strata_kaggle, "strata_terminal_python", return_value=sys.executable):
                summary = prepare_model_on_cpu(root / "data", "gyro", "gyro-s")
            self.assertEqual(summary["returncode"], 0)
            self.assertEqual(summary["model_bytes"], 10)
            self.assertFalse(summary["pack_ok"])
            self.assertFalse(summary["draft_ok"])
            self.assertTrue(any("pack build stopped" in item for item in summary["warnings"]))


class SetupModelNameTests(unittest.TestCase):
    def test_launch_spelling_maps_to_setup_spelling(self):
        self.assertEqual(setup_model_name("gyro-s"), "Gyro-S")
        self.assertEqual(setup_model_name("GYRO-M"), "Gyro-M")
        self.assertEqual(setup_model_name("q2_0"), "Q2_0")
        self.assertEqual(setup_model_name("ud-q4_k_xl"), "UD-Q4_K_XL")

    def test_unknown_names_pass_through(self):
        self.assertEqual(setup_model_name("tomorrow-model"), "tomorrow-model")
        self.assertEqual(setup_model_name(""), "")


if __name__ == "__main__":
    unittest.main()
