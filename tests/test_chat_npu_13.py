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

# Lightweight stubs to avoid external dependencies during import.
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
        self.compat_file = self.cache / "device_compat.json"
        self.models_file = self.cache / "models.json"

        self.orig_cache = app.CACHE_DIR
        self.orig_auth = app.AUTH_FILE
        self.orig_stats = app.STATS_FILE
        self.orig_prompts = app.BENCHMARK_PROMPTS_FILE
        self.orig_compat = app.DEVICE_COMPAT_FILE
        self.orig_models_file = app.MODELS_FILE
        self.orig_models = app.MODELS

        app.CACHE_DIR = self.cache
        app.AUTH_FILE = self.auth
        app.STATS_FILE = self.stats
        app.BENCHMARK_PROMPTS_FILE = self.prompts_file
        app.DEVICE_COMPAT_FILE = self.compat_file
        app.MODELS_FILE = self.models_file
        app.MODELS = []

        os.environ.pop("HF_TOKEN", None)

    def tearDown(self):
        app.CACHE_DIR = self.orig_cache
        app.AUTH_FILE = self.orig_auth
        app.STATS_FILE = self.orig_stats
        app.BENCHMARK_PROMPTS_FILE = self.orig_prompts
        app.DEVICE_COMPAT_FILE = self.orig_compat
        app.MODELS_FILE = self.orig_models_file
        app.MODELS = self.orig_models
        os.environ.pop("HF_TOKEN", None)
        shutil.rmtree(self.base, ignore_errors=True)

    def test_auth_file_and_token_sources(self):
        app.ensure_auth_file()
        self.assertTrue(self.auth.exists())
        self.assertEqual(json.loads(self.auth.read_text(encoding="utf-8")), {"hf_token": ""})

        os.environ["HF_TOKEN"] = "hf_env"
        self.assertEqual(app.load_hf_token(), "hf_env")
        os.environ.pop("HF_TOKEN", None)

        self.auth.write_text(json.dumps({"hf_token": "hf_file"}), encoding="utf-8")
        self.assertEqual(app.load_hf_token(), "hf_file")

    def test_human_bytes_and_downloaded(self):
        self.assertEqual(app.human_bytes(0), "—")
        self.assertEqual(app.human_bytes(10), "10 B")

        model_dir = self.cache / "m"
        model = {"display": "X", "params": "1B", "local": model_dir}
        self.assertFalse(app.is_downloaded(model_dir))
        self.assertIn("(1B, —)", app.model_menu_label(model))

        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("x", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"x")
        self.assertTrue(app.is_downloaded(model_dir))

    def test_models_json_load_and_save(self):
        models = app.load_models()
        self.assertTrue(self.models_file.exists())
        self.assertGreaterEqual(len(models), 1)

        models.append(
            {
                "display": "X",
                "params": "1B",
                "repo": "owner/x",
                "local": self.cache / "x",
            }
        )
        app.save_models(models)
        loaded = app.load_models()
        self.assertTrue(any(m["repo"] == "owner/x" for m in loaded))

    def test_stats_schema_and_record_by_mode(self):
        stats = app.normalize_stats_schema({"models": {}})

        app.record_stats(
            stats,
            "repo/a",
            "Model A",
            "CPU",
            1.0,
            10.0,
            mode=app.STATS_MODE_NORMAL,
        )
        app.record_stats(
            stats,
            "repo/a",
            "Model A",
            "GPU",
            2.0,
            20.0,
            mode=app.STATS_MODE_BENCHMARK,
        )

        self.assertEqual(
            stats["models"]["repo/a"]["modes"]["normal"]["devices"]["CPU"]["runs"],
            1,
        )
        self.assertEqual(
            stats["models"]["repo/a"]["modes"]["benchmark"]["devices"]["GPU"]["runs"],
            1,
        )

        app.save_stats(stats)
        loaded = app.load_stats()
        self.assertIn("modes", loaded["models"]["repo/a"])

    def test_print_stats_table_shows_two_sections(self):
        stats = app.normalize_stats_schema({"models": {}})
        app.record_stats(stats, "repo/a", "Model A", "CPU", 1.0, 10.0, mode=app.STATS_MODE_NORMAL)
        app.record_stats(
            stats,
            "repo/a",
            "Model A",
            "NPU",
            1.5,
            25.0,
            mode=app.STATS_MODE_BENCHMARK,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            app.print_stats_table(stats)
        text = out.getvalue()
        self.assertIn("Normal stats", text)
        self.assertIn("Benchmark stats", text)
        self.assertIn("?) Model A", text)

    def test_clear_stats_device_removes_from_both_modes(self):
        m1 = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        with mock.patch.object(app, "MODELS", [m1]):
            stats = app.normalize_stats_schema({"models": {}})
            app.record_stats(stats, "r/a", "A", "CPU", 1.0, 10.0, mode=app.STATS_MODE_NORMAL)
            app.record_stats(stats, "r/a", "A", "CPU", 1.0, 10.0, mode=app.STATS_MODE_BENCHMARK)

            with redirect_stdout(io.StringIO()):
                app.clear_stats(stats, model_number=1, device="CPU")

            self.assertNotIn("r/a", stats["models"])

    def test_command_helpers_slash_only(self):
        self.assertTrue(app.is_command("/help"))
        self.assertFalse(app.is_command("help"))
        self.assertTrue(app.is_command("/benchmark 2"))
        self.assertEqual(app.normalize_command("/stats"), "stats")

    def test_device_compat_save_load_and_badges(self):
        compat = {}
        app.mark_model_device_compat(compat, "repo/a", "CPU", True)
        app.mark_model_device_compat(compat, "repo/a", "GPU", False)
        app.save_device_compat(compat)
        loaded = app.load_device_compat()
        self.assertTrue(loaded["repo/a"]["CPU"])
        self.assertFalse(loaded["repo/a"]["GPU"])
        badges = app.model_device_badges(loaded, "repo/a")
        self.assertIn("CPU:✅", badges)
        self.assertIn("GPU:❌", badges)

    def test_choose_model_interactive_displays_device_badges(self):
        model = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        compat = {"r/a": {"CPU": True, "GPU": False, "NPU": True}}
        out = io.StringIO()
        with (
            mock.patch.object(app, "MODELS", [model]),
            mock.patch("builtins.input", side_effect=["0"]),
            redirect_stdout(out),
        ):
            app.choose_model_interactive(False, "Title", compat=compat)
        text = out.getvalue()
        self.assertIn("CPU:✅", text)
        self.assertIn("GPU:❌", text)

    def test_add_model_interactive(self):
        models = []
        with (
            mock.patch("builtins.input", side_effect=["My Model", "4B", "owner/my-model", ""]),
            redirect_stdout(io.StringIO()),
        ):
            created = app.add_model_interactive(models)
        self.assertIsNotNone(created)
        self.assertEqual(created["repo"], "owner/my-model")
        self.assertTrue(self.models_file.exists())
        stored = json.loads(self.models_file.read_text(encoding="utf-8"))
        self.assertTrue(any(x["repo"] == "owner/my-model" for x in stored))

    def test_benchmark_models_all_models_all_devices(self):
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
            app.benchmark_models({"models": {}}, ["p1", "p2", "p3", "p4", "p5"])

        self.assertEqual(mocked_load.call_count, 6)

    def test_benchmark_models_single_model_runs_three_devices(self):
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

        self.assertEqual(mocked_load.call_count, 3)

    def test_benchmark_models_only_missing_models(self):
        m1 = {"display": "A", "params": "1B", "local": self.cache / "a", "repo": "r/a"}
        m2 = {"display": "B", "params": "2B", "local": self.cache / "b", "repo": "r/b"}

        class FakePipe:
            def generate(self, prompt, max_new_tokens, temperature, top_p, streamer):
                streamer("ok")

        stats = app.normalize_stats_schema({"models": {}})
        app.record_stats(
            stats,
            "r/a",
            "A",
            "CPU",
            1.0,
            10.0,
            mode=app.STATS_MODE_BENCHMARK,
        )

        with (
            mock.patch.object(app, "MODELS", [m1, m2]),
            mock.patch.object(app, "is_downloaded", return_value=True),
            mock.patch.object(app, "load_pipeline", return_value=FakePipe()) as mocked_load,
            mock.patch.object(app, "save_stats"),
            redirect_stdout(io.StringIO()),
        ):
            app.benchmark_models(
                stats,
                ["p1", "p2", "p3", "p4", "p5"],
                only_missing_models=True,
            )

        self.assertEqual(mocked_load.call_count, 3)

    def test_main_flow_slash_commands(self):
        model_dir = self.cache / "ready_model"
        model_dir.mkdir()
        model = {"display": "Ready", "params": "1B", "local": model_dir, "repo": "repo/ready"}

        class FakePipe:
            def generate(self, prompt, max_new_tokens, temperature, top_p, streamer):
                streamer("hola")
                streamer(" mundo")

        inputs = iter([
            "hola",
            "/help",
            "/stats",
            "/current_model",
            "/models",
            "1",
            "1",
            "pregunta",
            "/delete",
            "/exit",
        ])

        def fake_input(_prompt):
            return next(inputs)

        out = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=fake_input),
            mock.patch.object(app, "ensure_auth_file"),
            mock.patch.object(app, "load_stats", return_value=app.normalize_stats_schema({"models": {}})),
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

    def test_main_marks_device_incompatible_when_load_fails(self):
        model_dir = self.cache / "m"
        model_dir.mkdir()
        model = {"display": "M", "params": "1B", "local": model_dir, "repo": "repo/m"}
        inputs = iter(["/models", "1", "1", "/exit"])

        def fake_input(_prompt):
            return next(inputs)

        with (
            mock.patch("builtins.input", side_effect=fake_input),
            mock.patch.object(app, "ensure_auth_file"),
            mock.patch.object(app, "load_stats", return_value=app.normalize_stats_schema({"models": {}})),
            mock.patch.object(app, "choose_model_interactive", return_value=model),
            mock.patch.object(app, "load_pipeline", side_effect=RuntimeError("boom")),
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        compat = app.load_device_compat()
        self.assertIn("repo/m", compat)
        self.assertFalse(compat["repo/m"]["CPU"])


if __name__ == "__main__":
    unittest.main()
