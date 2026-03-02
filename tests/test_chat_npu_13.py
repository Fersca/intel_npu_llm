import io
import json
import os
import pathlib
import shutil
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".tmp_tests"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide lightweight stubs so importing the app does not require external deps.
fake_ov = types.ModuleType("openvino_genai")
fake_ov.LLMPipeline = object
fake_hf = types.ModuleType("huggingface_hub")
fake_hf.snapshot_download = lambda **kwargs: None

with mock.patch.dict("sys.modules", {"openvino_genai": fake_ov, "huggingface_hub": fake_hf}):
    import chat_npu_13 as app


class TestChatNPU(unittest.TestCase):
    def setUp(self):
        self.base = TMP_ROOT / f"test_chat_npu_{time.time_ns()}"
        self.base.mkdir(parents=True, exist_ok=False)
        self.cache = self.base / "ov_models"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.auth = self.cache / "hf_auth.json"
        self.stats = self.cache / "stats.json"
        self.prompts_file = self.cache / "benchmark_prompts.json"
        self.orig_cache = app.CACHE_DIR
        self.orig_auth = app.AUTH_FILE
        self.orig_stats = app.STATS_FILE
        self.orig_prompts = app.BENCHMARK_PROMPTS_FILE
        app.CACHE_DIR = self.cache
        app.AUTH_FILE = self.auth
        app.STATS_FILE = self.stats
        app.BENCHMARK_PROMPTS_FILE = self.prompts_file
        os.environ.pop("HF_TOKEN", None)

    def tearDown(self):
        app.CACHE_DIR = self.orig_cache
        app.AUTH_FILE = self.orig_auth
        app.STATS_FILE = self.orig_stats
        app.BENCHMARK_PROMPTS_FILE = self.orig_prompts
        os.environ.pop("HF_TOKEN", None)
        shutil.rmtree(self.base, ignore_errors=True)

    def test_ensure_auth_file_creates_default_json(self):
        app.ensure_auth_file()
        self.assertTrue(self.auth.exists())
        self.assertEqual(json.loads(self.auth.read_text(encoding="utf-8")), {"hf_token": ""})

    def test_load_hf_token_prefers_environment(self):
        os.environ["HF_TOKEN"] = "hf_env_token"
        token = app.load_hf_token()
        self.assertEqual(token, "hf_env_token")

    def test_load_hf_token_reads_auth_file(self):
        self.auth.write_text(json.dumps({"hf_token": "hf_file_token"}), encoding="utf-8")
        token = app.load_hf_token()
        self.assertEqual(token, "hf_file_token")
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_file_token")

    def test_load_hf_token_prompts_when_missing(self):
        with (
            mock.patch("builtins.input", return_value="hf_prompted_token"),
            redirect_stdout(io.StringIO()),
        ):
            token = app.load_hf_token()
        self.assertEqual(token, "hf_prompted_token")
        saved = json.loads(self.auth.read_text(encoding="utf-8"))
        self.assertEqual(saved["hf_token"], "hf_prompted_token")

    def test_dir_size_bytes_and_human_bytes(self):
        model_dir = self.cache / "model"
        model_dir.mkdir()
        (model_dir / "a.bin").write_bytes(b"a" * 10)
        (model_dir / "b.xml").write_bytes(b"b" * 14)
        self.assertEqual(app.dir_size_bytes(model_dir), 24)
        self.assertEqual(app.human_bytes(0), "—")
        self.assertEqual(app.human_bytes(10), "10 B")
        self.assertEqual(app.human_bytes(2048), "2.00 KB")

    def test_is_downloaded_and_model_menu_label(self):
        model_dir = self.cache / "m1"
        model = {"display": "X", "params": "1B", "local": model_dir}
        self.assertFalse(app.is_downloaded(model_dir))
        label = app.model_menu_label(model)
        self.assertIn("(1B, —)", label)
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("x", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"x")
        self.assertTrue(app.is_downloaded(model_dir))
        self.assertIn("B)", app.model_menu_label(model))

    def test_download_model_invokes_snapshot_download(self):
        dest = self.cache / "dest"
        with (
            mock.patch.object(app, "load_hf_token", return_value="hf_any"),
            mock.patch.object(app, "snapshot_download") as mocked_download,
            mock.patch.dict(os.environ, {"HF_TOKEN": "hf_any"}, clear=False),
            redirect_stdout(io.StringIO()),
        ):
            app.download_model("owner/repo", dest)
        mocked_download.assert_called_once()
        kwargs = mocked_download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "owner/repo")
        self.assertEqual(kwargs["local_dir"], str(dest))
        self.assertEqual(kwargs["token"], "hf_any")

    def test_choose_model_interactive_cancel(self):
        with (
            mock.patch.object(app, "MODELS", [{"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}]),
            mock.patch("builtins.input", side_effect=["0"]),
            redirect_stdout(io.StringIO()),
        ):
            self.assertIsNone(app.choose_model_interactive(False, "Title"))

    def test_choose_model_interactive_downloads_when_needed(self):
        m = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        with (
            mock.patch.object(app, "MODELS", [m]),
            mock.patch.object(app, "is_downloaded", return_value=False),
            mock.patch.object(app, "download_model") as mocked_download,
            mock.patch("builtins.input", side_effect=["1"]),
            redirect_stdout(io.StringIO()),
        ):
            picked = app.choose_model_interactive(True, "Title")
        self.assertEqual(picked["repo"], "r/a")
        mocked_download.assert_called_once_with("r/a", self.cache / "a")

    def test_delete_model_files_missing_and_outside_cache(self):
        missing = {"display": "M", "params": "1B", "local": self.cache / "missing", "repo": "r/m"}
        with redirect_stdout(io.StringIO()):
            self.assertFalse(app.delete_model_files(missing))

        outside = self.base / "outside"
        outside.mkdir()
        (outside / "x.bin").write_bytes(b"x")
        blocked = {"display": "B", "params": "1B", "local": outside, "repo": "r/b"}
        with redirect_stdout(io.StringIO()):
            self.assertFalse(app.delete_model_files(blocked))
        self.assertTrue(outside.exists())

    def test_delete_model_files_success(self):
        inside = self.cache / "inside"
        inside.mkdir()
        (inside / "x.xml").write_text("x", encoding="utf-8")
        model = {"display": "S", "params": "1B", "local": inside, "repo": "r/s"}
        with redirect_stdout(io.StringIO()):
            self.assertTrue(app.delete_model_files(model))
        self.assertFalse(inside.exists())

    def test_load_save_record_stats_and_mean(self):
        self.assertEqual(app.load_stats(), {"models": {}})
        stats = {"models": {}}
        app.record_stats(stats, "repo/a", "Model A", 1.5, 12.0)
        app.record_stats(stats, "repo/a", "Model A", 2.5, 8.0)
        self.assertEqual(stats["models"]["repo/a"]["runs"], 2)
        self.assertEqual(app.mean([1.0, 3.0]), 2.0)
        app.save_stats(stats)
        loaded = app.load_stats()
        self.assertEqual(loaded["models"]["repo/a"]["runs"], 2)

    def test_print_stats_table_outputs_headers(self):
        stats = {"models": {"repo/a": {"name": "Model A", "runs": 1, "ttft_s": [1.0], "tps": [10.0]}}}
        out = io.StringIO()
        with redirect_stdout(out):
            app.print_stats_table(stats)
        text = out.getvalue()
        self.assertIn("Model", text)
        self.assertIn("TTFT avg(s)", text)
        self.assertIn("Model A", text)

    def test_command_helpers(self):
        self.assertTrue(app.is_command("/help"))
        self.assertTrue(app.is_command("models"))
        self.assertFalse(app.is_command("hello"))
        self.assertEqual(app.normalize_command("/stats"), "stats")
        self.assertTrue(app.is_command("current_model"))
        self.assertTrue(app.is_command("benchmark 1"))
        self.assertTrue(app.is_command("start_server"))

    def test_load_pipeline_calls_openvino_constructor(self):
        selected = {"display": "X", "params": "1B", "local": self.cache / "x", "repo": "r/x"}
        with (
            mock.patch.object(app.ov_genai, "LLMPipeline", return_value="PIPE") as mocked_pipe,
            redirect_stdout(io.StringIO()),
        ):
            result = app.load_pipeline(selected)
        self.assertEqual(result, "PIPE")
        mocked_pipe.assert_called_once_with(str(self.cache / "x"), app.DEVICE, PERFORMANCE_HINT="LATENCY")

    def test_main_full_flow(self):
        model_dir = self.cache / "ready_model"
        model_dir.mkdir()
        model = {"display": "Ready", "params": "1B", "local": model_dir, "repo": "repo/ready"}

        class FakePipe:
            def generate(self, prompt, max_new_tokens, temperature, top_p, streamer):
                streamer("hola")
                streamer(" mundo")

        inputs = iter(["hola", "help", "stats", "current_model", "models", "pregunta", "delete", "exit"])

        def fake_input(_prompt):
            return next(inputs)

        out = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=fake_input),
            mock.patch.object(app, "ensure_auth_file"),
            mock.patch.object(app, "load_stats", return_value={"models": {}}),
            mock.patch.object(app, "choose_model_interactive", side_effect=[model, model]),
            mock.patch.object(app, "load_pipeline", return_value=FakePipe()),
            mock.patch.object(app, "delete_model_files", return_value=True),
            mock.patch.object(app, "save_stats") as mocked_save,
            redirect_stdout(out),
        ):
            app.main()

        text = out.getvalue()
        self.assertIn("No model loaded", text)
        self.assertIn("Commands:", text)
        self.assertIn("TTFT:", text)
        self.assertIn("You deleted the active model", text)
        self.assertIn("Bye.", text)
        self.assertTrue(mocked_save.called)

    def test_benchmark_models_all_downloaded(self):
        m1 = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        m2 = {"display": "B", "params": "2B", "local": self.cache / "b", "repo": "r/b"}

        class FakePipe:
            def generate(self, prompt, max_new_tokens, temperature, top_p, streamer):
                streamer("ok")

        out = io.StringIO()
        with (
            mock.patch.object(app, "MODELS", [m1, m2]),
            mock.patch.object(app, "is_downloaded", return_value=True),
            mock.patch.object(app, "load_pipeline", return_value=FakePipe()) as mocked_load,
            mock.patch.object(app, "save_stats") as mocked_save,
            redirect_stdout(out),
        ):
            app.benchmark_models({"models": {}}, ["p1", "p2", "p3", "p4", "p5"])

        self.assertEqual(mocked_load.call_count, 2)
        self.assertTrue(mocked_save.called)

    def test_benchmark_models_single_model_number(self):
        m1 = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        m2 = {"display": "B", "params": "2B", "local": self.cache / "b", "repo": "r/b"}

        class FakePipe:
            def generate(self, prompt, max_new_tokens, temperature, top_p, streamer):
                streamer("ok")

        with (
            mock.patch.object(app, "MODELS", [m1, m2]),
            mock.patch.object(app, "is_downloaded", return_value=True),
            mock.patch.object(app, "load_pipeline", return_value=FakePipe()) as mocked_load,
            mock.patch.object(app, "save_stats"),
            redirect_stdout(io.StringIO()),
        ):
            app.benchmark_models({"models": {}}, ["p1", "p2", "p3", "p4", "p5"], model_number=2)

        mocked_load.assert_called_once_with(m2)

    def test_collect_benchmark_prompts(self):
        with (
            mock.patch("builtins.input", side_effect=["a", "b", "c", "d", "e"]),
            redirect_stdout(io.StringIO()),
        ):
            prompts = app.collect_benchmark_prompts(5)
        self.assertEqual(prompts, ["a", "b", "c", "d", "e"])

    def test_save_and_load_benchmark_prompts(self):
        app.save_benchmark_prompts(["one", "two", "three"])
        self.assertTrue(self.prompts_file.exists())
        loaded = app.load_saved_benchmark_prompts()
        self.assertEqual(loaded, ["one", "two", "three"])

    def test_collect_benchmark_prompts_reuses_saved(self):
        app.save_benchmark_prompts(["p1", "p2", "p3", "p4", "p5"])
        with (
            mock.patch("builtins.input", side_effect=["y"]),
            redirect_stdout(io.StringIO()),
        ):
            prompts = app.collect_benchmark_prompts(5)
        self.assertEqual(prompts, ["p1", "p2", "p3", "p4", "p5"])

    def test_prompt_yes_no_retries_invalid_input(self):
        out = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=["maybe", "n"]),
            redirect_stdout(out),
        ):
            result = app.prompt_yes_no("Use saved prompts?", default_yes=True)
        self.assertFalse(result)
        self.assertIn("Please answer with 'y' or 'n'.", out.getvalue())

    def test_build_chat_prompt(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = app.build_chat_prompt(messages)
        self.assertIn("System: You are helpful", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertTrue(prompt.endswith("\nAssistant:"))

    def test_create_openai_chat_response_shape(self):
        response = app.create_openai_chat_response("repo/test", "sample response")
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "repo/test")
        self.assertEqual(response["choices"][0]["message"]["content"], "sample response")

    def test_start_openai_compatible_server_starts_background_server(self):
        state = {"pipe": None, "current": None}
        fake_server = mock.MagicMock()
        fake_thread = mock.MagicMock()

        with (
            mock.patch.object(app, "ThreadingHTTPServer", return_value=fake_server) as mocked_http,
            mock.patch.object(app.threading, "Thread", return_value=fake_thread) as mocked_thread,
        ):
            result = app.start_openai_compatible_server(state)

        self.assertIs(result, fake_server)
        mocked_http.assert_called_once()
        mocked_thread.assert_called_once()
        fake_thread.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
