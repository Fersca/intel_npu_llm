import gc
import json
import os
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

# =========================
# Config
# =========================
DEFAULT_DEVICE = "NPU"
DEFAULT_PERFORMANCE_HINT = "LATENCY"

DEVICE_OPTIONS = [
    "CPU",
    "GPU",
    "NPU",
    "AUTO",
    "AUTO:NPU,GPU,CPU",
    "MULTI:NPU,GPU",
    "HETERO:NPU,GPU,CPU",
]

PERFORMANCE_HINT_OPTIONS = [
    "LATENCY",
    "THROUGHPUT",
    "CUMULATIVE_THROUGHPUT",
    "UNDEFINED",
]

BENCHMARK_DEVICES = ["CPU", "GPU", "NPU"]
STATS_MODE_NORMAL = "normal"
STATS_MODE_BENCHMARK = "benchmark"

ACTIVE_DEVICE = DEFAULT_DEVICE
ACTIVE_PERFORMANCE_HINT = DEFAULT_PERFORMANCE_HINT

CACHE_DIR = Path("ov_models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS_FILE = CACHE_DIR / "stats.json"
AUTH_FILE = CACHE_DIR / "hf_auth.json"  # {"hf_token": "hf_..."}
BENCHMARK_PROMPTS_FILE = CACHE_DIR / "benchmark_prompts.json"
DEVICE_COMPAT_FILE = CACHE_DIR / "device_compat.json"
MODELS_FILE = CACHE_DIR / "models.json"

# =========================
# Model list
# =========================
DEFAULT_MODELS_DATA = [
    {
        "display": "Phi-4 mini instruct INT4",
        "params": "?3.8B",
        "repo": "FluidInference/phi-4-mini-instruct-int4-ov-npu",
        "local_dir": "phi-4-mini-instruct-int4-ov-npu",
    },
    {
        "display": "Qwen2.5 1.5B instruct INT4",
        "params": "1.5B",
        "repo": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "local_dir": "qwen2.5-1.5b-instruct-int4-ov",
    },
    {
        "display": "TinyLlama 1.1B Chat INT4",
        "params": "1.1B",
        "repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "local_dir": "tinyllama-1.1b-chat-int4-ov",
    },
    {
        "display": "Phi-3 Medium 4k Instruct INT4",
        "params": "3.8B",
        "repo": "OpenVINO/Phi-3-medium-4k-instruct-int4-ov",
        "local_dir": "phi-3-medium-4k-instruct-int4-ov",
    },
    {
        "display": "Qwen2.5 7B INT4 NPU OV",
        "params": "7B",
        "repo": "FluidInference/qwen2.5-7b-int4-npu-ov",
        "local_dir": "qwen2.5-7b-int4-npu-ov",
    },
    {
        "display": "Llama 3.1 8B Instruct INT4 NPU OV",
        "params": "8B",
        "repo": "llmware/llama-3.1-8b-instruct-npu-ov",
        "local_dir": "llama-3.1-8b-instruct-npu-ov",
    },
]

MODELS = []

# =========================
# HF token handling
# =========================
def ensure_auth_file() -> None:
    if not AUTH_FILE.exists():
        AUTH_FILE.write_text(json.dumps({"hf_token": ""}, indent=2), encoding="utf-8")


def load_hf_token() -> str | None:
    """
    Look for a token in:
      1) env HF_TOKEN
      2) ov_models/hf_auth.json  ({"hf_token": "hf_..."})
    If the file exists but is empty, ask for one and save it.
    """
    ensure_auth_file()

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token

    token = ""
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8") or "{}")
        token = (data.get("hf_token") or "").strip()
    except Exception:
        token = ""

    if not token:
        print("\nâš ï¸ HF token not configured (optional).")
        print("Paste your token (starts with 'hf_') or press Enter to continue without one.\n")
        token = input("HF token: ").strip()
        AUTH_FILE.write_text(json.dumps({"hf_token": token}, indent=2), encoding="utf-8")

    if token:
        os.environ["HF_TOKEN"] = token
        return token

    return None


# =========================
# Disk size helpers
# =========================
def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def human_bytes(num: int) -> str:
    if num <= 0:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num)
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


# =========================
# Models: download/select/delete
# =========================
def is_downloaded(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    has_xml = any(model_dir.rglob("*.xml"))
    has_bin = any(model_dir.rglob("*.bin"))
    return has_xml and has_bin


def model_menu_label(m: dict) -> str:
    if is_downloaded(m["local"]):
        size = human_bytes(dir_size_bytes(m["local"]))
    else:
        size = "—"
    return f"{m['display']} ({m['params']}, {size})"


def slug_from_repo(repo: str) -> str:
    name = repo.strip().split("/")[-1].strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in name)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "model"


def model_to_storage_entry(model: dict) -> dict:
    return {
        "display": model["display"],
        "params": model["params"],
        "repo": model["repo"],
        "local_dir": model["local"].name,
    }


def parse_model_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    display = str(entry.get("display", "")).strip()
    params = str(entry.get("params", "")).strip()
    repo = str(entry.get("repo", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not display or not params or not repo:
        return None
    if not local_dir:
        local_dir = slug_from_repo(repo)
    return {
        "display": display,
        "params": params,
        "repo": repo,
        "local": CACHE_DIR / local_dir,
    }


def save_models(models: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [model_to_storage_entry(m) for m in models]
    MODELS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_models() -> list[dict]:
    if MODELS_FILE.exists():
        try:
            raw = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                parsed = [parse_model_entry(x) for x in raw]
                models = [x for x in parsed if x is not None]
                if models:
                    return models
        except Exception:
            pass

    defaults = []
    for entry in DEFAULT_MODELS_DATA:
        parsed = parse_model_entry(entry)
        if parsed is not None:
            defaults.append(parsed)
    save_models(defaults)
    return defaults


def add_model_interactive(models: list[dict]) -> dict | None:
    print("\nAdd model\n")
    display = input("Display name: ").strip()
    if not display:
        print("\nCancelled: display name is required.\n")
        return None
    params = input("Params (e.g. 7B): ").strip()
    if not params:
        print("\nCancelled: params are required.\n")
        return None
    repo = input("HF repo (owner/name): ").strip()
    if not repo:
        print("\nCancelled: repo is required.\n")
        return None

    if any(m["repo"] == repo for m in models):
        print("\n⚠️ A model with that repo already exists.\n")
        return None

    default_local = slug_from_repo(repo)
    local_dir = input(f"Local folder [{default_local}]: ").strip() or default_local
    new_model = {
        "display": display,
        "params": params,
        "repo": repo,
        "local": CACHE_DIR / local_dir,
    }
    models.append(new_model)
    save_models(models)
    print(f"\n✅ Model added as #{len(models)}: {model_menu_label(new_model)}\n")
    return new_model


def load_device_compat() -> dict:
    try:
        if DEVICE_COMPAT_FILE.exists():
            data = json.loads(DEVICE_COMPAT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_device_compat(compat: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE_COMPAT_FILE.write_text(json.dumps(compat, indent=2), encoding="utf-8")


def mark_model_device_compat(compat: dict, model_repo: str, device: str, is_ok: bool) -> None:
    model_entry = compat.setdefault(model_repo, {})
    model_entry[device.upper()] = bool(is_ok)


def model_device_badges(compat: dict, model_repo: str) -> str:
    model_entry = compat.get(model_repo, {})
    badges = []
    for device in BENCHMARK_DEVICES:
        value = model_entry.get(device)
        mark = "✅" if value is True else ("❌" if value is False else "❔")
        badges.append(f"{device}:{mark}")
    return " ".join(badges)


def download_model(repo_id: str, local_dir: Path) -> None:
    load_hf_token()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 Downloading model: {repo_id}")
    print(f"   -> destination: {local_dir}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("\n✅ Download complete.\n")


def choose_model_interactive(allow_download: bool, title: str, compat: dict | None = None) -> dict | None:
    print(f"\n{title}\n")
    compat = compat or {}
    for i, m in enumerate(MODELS, 1):
        status = "✅" if is_downloaded(m["local"]) else "⬇️"
        badges = model_device_badges(compat, m["repo"])
        print(f"  {i}) {status} {model_menu_label(m)} | {badges}")
    print("  0) Cancel\n")

    while True:
        choice = input("Option: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            m = MODELS[int(choice) - 1]
            if allow_download and not is_downloaded(m["local"]):
                download_model(m["repo"], m["local"])
            return m
        print("Invalid option.")


def delete_model_files(m: dict) -> bool:
    path = m["local"]
    if not path.exists():
        print("\nℹ️ That model is not on disk.\n")
        return False

    try:
        resolved = path.resolve()
        cache = CACHE_DIR.resolve()
        if cache not in resolved.parents and resolved != cache:
            print("\n❌ Security check failed: path is outside ov_models, skipping delete.\n")
            return False
    except Exception:
        print("\nâŒ Could not resolve paths for safe deletion.\n")
        return False

    size_before = human_bytes(dir_size_bytes(path))
    print(f"\n🗑️ Deleting: {model_menu_label(m)}")
    print(f"   Path: {path}")
    print(f"   Size: {size_before}\n")

    try:
        shutil.rmtree(path)
        print("✅ Delete complete.\n")
        return True
    except Exception as e:
        print(f"\nâŒ Delete error: {e}\n")
        return False


# =========================
# Stats persistence
# =========================
def load_stats() -> dict:
    try:
        if STATS_FILE.exists():
            raw = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            return normalize_stats_schema(raw)
    except Exception:
        pass
    return normalize_stats_schema({"models": {}})


def save_stats(stats: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def normalize_stats_schema(stats: dict) -> dict:
    models = stats.setdefault("models", {})
    for entry in models.values():
        if not isinstance(entry, dict):
            continue

        modes = entry.get("modes")
        if isinstance(modes, dict):
            for mode_name in (STATS_MODE_NORMAL, STATS_MODE_BENCHMARK):
                mode_entry = modes.setdefault(mode_name, {})
                mode_entry.setdefault("devices", {})
            continue

        normal_devices = {}
        legacy_devices = entry.get("devices", {})
        if isinstance(legacy_devices, dict):
            normal_devices = legacy_devices
        else:
            runs = int(entry.get("runs", 0) or 0)
            ttft_s = list(entry.get("ttft_s", []))
            tps = list(entry.get("tps", []))
            if runs or ttft_s or tps:
                normal_devices["UNKNOWN"] = {"runs": runs, "ttft_s": ttft_s, "tps": tps}

        entry["modes"] = {
            STATS_MODE_NORMAL: {"devices": normal_devices},
            STATS_MODE_BENCHMARK: {"devices": {}},
        }
    return stats


def get_mode_devices(entry: dict, mode: str, create: bool = False) -> dict:
    modes = entry.get("modes")
    if not isinstance(modes, dict):
        if not create:
            return {}
        entry["modes"] = {
            STATS_MODE_NORMAL: {"devices": {}},
            STATS_MODE_BENCHMARK: {"devices": {}},
        }
        modes = entry["modes"]

    mode_entry = modes.get(mode)
    if not isinstance(mode_entry, dict):
        if not create:
            return {}
        mode_entry = {"devices": {}}
        modes[mode] = mode_entry

    devices = mode_entry.get("devices")
    if not isinstance(devices, dict):
        if not create:
            return {}
        mode_entry["devices"] = {}
        devices = mode_entry["devices"]
    return devices


def record_stats(
    stats: dict,
    model_key: str,
    model_name: str,
    device: str,
    ttft_s: float,
    tps: float,
    mode: str = STATS_MODE_NORMAL,
) -> None:
    models = stats.setdefault("models", {})
    entry = models.setdefault(
        model_key,
        {
            "name": model_name,
            "modes": {
                STATS_MODE_NORMAL: {"devices": {}},
                STATS_MODE_BENCHMARK: {"devices": {}},
            },
        },
    )
    entry["name"] = model_name
    devices = get_mode_devices(entry, mode, create=True)
    device_entry = devices.setdefault(device, {"runs": 0, "ttft_s": [], "tps": []})
    device_entry["runs"] += 1
    device_entry["ttft_s"].append(ttft_s)
    device_entry["tps"].append(tps)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def build_stats_rows(stats: dict, mode: str) -> list[dict]:
    models = stats.get("models", {})
    model_index_by_repo = {m["repo"]: i for i, m in enumerate(MODELS, 1)}
    rows = []
    for k, v in models.items():
        model_name = v.get("name", k)
        model_index = model_index_by_repo.get(k)
        model_label = f"{model_index}) {model_name}" if model_index is not None else f"?) {model_name}"
        devices = get_mode_devices(v, mode, create=False)
        for device, d in devices.items():
            ttfts = d.get("ttft_s", [])
            tpss = d.get("tps", [])
            rows.append(
                {
                    "model": model_label,
                    "device": device,
                    "runs": d.get("runs", 0),
                    "ttft_avg": mean(ttfts),
                    "tps_avg": mean(tpss),
                    "ttft_last": ttfts[-1] if ttfts else 0.0,
                    "tps_last": tpss[-1] if tpss else 0.0,
                }
            )
    rows.sort(key=lambda r: (r["model"].lower(), -r["tps_avg"], r["device"].lower()))
    return rows


def print_stats_mode_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no stats)")
        return

    def f(x):
        return f"{x:0.3f}"

    headers = ["Model", "Device", "n", "TTFT avg(s)", "TPS avg", "TTFT last(s)", "TPS last"]
    colw = [56, 18, 4, 12, 10, 13, 9]

    def cut(s, w):
        s = str(s)
        return s if len(s) <= w else s[: w - 1] + "…"

    line = (
        f"{cut(headers[0], colw[0]):<{colw[0]}} "
        f"{cut(headers[1], colw[1]):<{colw[1]}} "
        f"{headers[2]:>{colw[2]}} "
        f"{headers[3]:>{colw[3]}} "
        f"{headers[4]:>{colw[4]}} "
        f"{headers[5]:>{colw[5]}} "
        f"{headers[6]:>{colw[6]}}"
    )
    sep = "-" * len(line)

    print("\n" + line)
    print(sep)
    last_model = None
    for r in rows:
        model_cell = r["model"] if r["model"] != last_model else ""
        last_model = r["model"]
        print(
            f"{cut(model_cell, colw[0]):<{colw[0]}} "
            f"{cut(r['device'], colw[1]):<{colw[1]}} "
            f"{r['runs']:>{colw[2]}} "
            f"{f(r['ttft_avg']):>{colw[3]}} "
            f"{f(r['tps_avg']):>{colw[4]}} "
            f"{f(r['ttft_last']):>{colw[5]}} "
            f"{f(r['tps_last']):>{colw[6]}}"
        )
    print("")


def print_stats_table(stats: dict) -> None:
    normal_rows = build_stats_rows(stats, STATS_MODE_NORMAL)
    benchmark_rows = build_stats_rows(stats, STATS_MODE_BENCHMARK)
    if not normal_rows and not benchmark_rows:
        print("\n(no stats yet)\n")
        return
    print_stats_mode_table(normal_rows, "Normal stats")
    print_stats_mode_table(benchmark_rows, "Benchmark stats")


def clear_stats(stats: dict, model_number: int | None = None, device: str | None = None) -> None:
    models = stats.setdefault("models", {})

    if model_number is None:
        if not models:
            print("\n(no stats to clear)\n")
            return
        stats["models"] = {}
        save_stats(stats)
        print("\n✅ All stats were cleared.\n")
        return

    if model_number < 1 or model_number > len(MODELS):
        print(f"\n⚠️ Invalid model number. Choose 1..{len(MODELS)}.\n")
        return

    model = MODELS[model_number - 1]
    model_key = model["repo"]
    model_label = model_menu_label(model)
    entry = models.get(model_key)
    if entry is None:
        print(f"\n(no stats found for model {model_number}: {model_label})\n")
        return

    if device is None:
        del models[model_key]
        save_stats(stats)
        print(f"\n✅ Cleared all stats for model {model_number}: {model_label}\n")
        return

    normalized_device = device.strip().upper()
    if not normalized_device:
        print("\n⚠️ Device cannot be empty.\n")
        return

    deleted_any = False
    available_devices = set()
    for mode in (STATS_MODE_NORMAL, STATS_MODE_BENCHMARK):
        mode_devices = get_mode_devices(entry, mode, create=False)
        for existing_device in list(mode_devices.keys()):
            available_devices.add(str(existing_device))
            if str(existing_device).upper() == normalized_device:
                del mode_devices[existing_device]
                deleted_any = True

    if not deleted_any:
        available = ", ".join(sorted(available_devices)) or "(none)"
        print(f"\n⚠️ Device not found for that model. Available: {available}\n")
        return

    normal_left = get_mode_devices(entry, STATS_MODE_NORMAL, create=False)
    benchmark_left = get_mode_devices(entry, STATS_MODE_BENCHMARK, create=False)
    if not normal_left and not benchmark_left:
        del models[model_key]
    save_stats(stats)
    print(f"\n✅ Cleared stats for model {model_number} on device {normalized_device}.\n")


# =========================
# Pipeline
# =========================
def load_pipeline(selected_model: dict) -> ov_genai.LLMPipeline:
    model_path = selected_model["local"]
    print(
        f"\n✅ Model loaded on {ACTIVE_DEVICE}: {model_menu_label(selected_model)} "
        f"(PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT})"
    )
    return ov_genai.LLMPipeline(
        str(model_path),
        ACTIVE_DEVICE,
        PERFORMANCE_HINT=ACTIVE_PERFORMANCE_HINT,
    )


def choose_from_options(title: str, options: list[str], current_value: str) -> str:
    print(f"\n{title}")
    for i, option in enumerate(options, 1):
        marker = "(current)" if option == current_value else ""
        print(f"  {i}) {option} {marker}".rstrip())

    while True:
        choice = input("Option: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid option.")


def configure_runtime() -> None:
    global ACTIVE_DEVICE, ACTIVE_PERFORMANCE_HINT

    print("\n⚙️ Runtime configuration")
    ACTIVE_DEVICE = choose_from_options("Select DEVICE:", DEVICE_OPTIONS, ACTIVE_DEVICE)
    ACTIVE_PERFORMANCE_HINT = choose_from_options(
        "Select PERFORMANCE_HINT:",
        PERFORMANCE_HINT_OPTIONS,
        ACTIVE_PERFORMANCE_HINT,
    )

    print(
        f"\n✅ Runtime updated: DEVICE={ACTIVE_DEVICE}, "
        f"PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
    )


# =========================
# UI / Commands
# =========================
HELP_TEXT = """\
Commands:
  /help                     Show this help
  /models                   Select and load a model (download if missing)
  /add_model                Add a new model to ov_models/models.json
  /delete                   Delete model files from disk (keeps it in the list)
  /stats                    Show separate tables: normal stats and benchmark stats
  /clear_stats              Clear all stats
  /clear_stats <n>          Clear all stats for model number <n>
  /clear_stats <n> <device> Clear stats only for model <n> and device <device>
  /current_model            Show the currently loaded model and runtime config
  /benchmark                Ask: all models or only missing models (always CPU/GPU/NPU)
  /benchmark <number>       Run benchmark on CPU/GPU/NPU only for model number <number>
  /start_server             Start OpenAI-compatible chat server on port 1311
  /exit                     Exit

Usage:
  - First run '/models' to load one.
  - Then type any regular prompt to chat.
"""


def is_command(s: str) -> bool:
    s = s.strip()
    return s.startswith("/")


def normalize_command(s: str) -> str:
    s = s.strip()
    if s.startswith("/"):
        s = s[1:]
    return s



def load_saved_benchmark_prompts() -> list[str]:
    try:
        if BENCHMARK_PROMPTS_FILE.exists():
            data = json.loads(BENCHMARK_PROMPTS_FILE.read_text(encoding="utf-8"))
            prompts = data.get("prompts", []) if isinstance(data, dict) else []
            return [str(p) for p in prompts if str(p).strip()]
    except Exception:
        pass
    return []


def save_benchmark_prompts(prompts: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"prompts": prompts}
    BENCHMARK_PROMPTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        answer = input(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default_yes
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def collect_benchmark_prompts(count: int = 5) -> list[str]:
    saved_prompts = load_saved_benchmark_prompts()
    if len(saved_prompts) >= count:
        print("\nSaved benchmark prompts were found:\n")
        for i, prompt in enumerate(saved_prompts[:count], 1):
            print(f"  {i}) {prompt}")
        if prompt_yes_no("Do you want to reuse the saved prompts?", default_yes=True):
            print("")
            return saved_prompts[:count]

    print(f"\nEnter {count} prompts for the benchmark:\n")
    prompts: list[str] = []
    for i in range(1, count + 1):
        prompt = input(f"Prompt {i}: ").strip()
        prompts.append(prompt)

    save_benchmark_prompts(prompts)
    print(f"\n✅ Saved {count} benchmark prompts to {BENCHMARK_PROMPTS_FILE}.\n")
    return prompts

def benchmark_models(
    stats: dict,
    prompts: list[str],
    model_number: int | None = None,
    only_missing_models: bool = False,
    compat: dict | None = None,
) -> None:
    global ACTIVE_DEVICE

    if not prompts:
        print("\nNo prompts were provided.\n")
        return

    model_candidates = list(enumerate(MODELS, 1))
    if model_number is not None:
        model_candidates = [x for x in model_candidates if x[0] == model_number]
        if not model_candidates:
            print("\nInvalid model number for benchmark.\n")
            return

    downloaded = [(idx, m) for idx, m in model_candidates if is_downloaded(m["local"])]
    if model_number is None and only_missing_models:
        filtered = []
        for idx, model in downloaded:
            entry = stats.get("models", {}).get(model["repo"], {})
            benchmark_devices = get_mode_devices(entry, STATS_MODE_BENCHMARK, create=False)
            if not benchmark_devices:
                filtered.append((idx, model))
        downloaded = filtered

    if not downloaded:
        print("\nNo downloaded models match this benchmark selection.\n")
        return

    mode_text = "missing models only" if only_missing_models else "all models"
    print(
        f"\nStarting benchmark on {len(downloaded)} model(s) "
        f"across devices ({mode_text}): {', '.join(BENCHMARK_DEVICES)}.\n"
    )

    previous_device = ACTIVE_DEVICE
    try:
        for idx, model in downloaded:
            for device in BENCHMARK_DEVICES:
                print(f"\n===== Benchmark model {idx} on {device}: {model_menu_label(model)} =====")
                ACTIVE_DEVICE = device
                try:
                    pipe = load_pipeline(model)
                    if compat is not None:
                        mark_model_device_compat(compat, model["repo"], device, True)
                        save_device_compat(compat)
                except Exception as exc:
                    if compat is not None:
                        mark_model_device_compat(compat, model["repo"], device, False)
                        save_device_compat(compat)
                    print(f"Failed to load model on {device}: {exc}")
                    continue

                for prompt_idx, prompt in enumerate(prompts, 1):
                    print(f"\n[{idx}.{device}.{prompt_idx}] Prompt: {prompt}")
                    t_start = time.perf_counter()
                    first_token_time = None
                    token_events = 0

                    def streamer(chunk: str):
                        nonlocal first_token_time, token_events
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_events += 1
                        print(chunk, end="", flush=True)

                    print("🤖 > ", end="", flush=True)
                    pipe.generate(
                        prompt,
                        max_new_tokens=300,
                        temperature=0.7,
                        top_p=0.9,
                        streamer=streamer,
                    )
                    t_end = time.perf_counter()
                    print("\n")

                    if first_token_time is None:
                        ttft = t_end - t_start
                        tps = 0.0
                    else:
                        ttft = first_token_time - t_start
                        decode_time = max(1e-9, t_end - first_token_time)
                        tps = token_events / decode_time

                    stats_name = model_menu_label(model)
                    record_stats(
                        stats,
                        model["repo"],
                        stats_name,
                        device,
                        ttft,
                        tps,
                        mode=STATS_MODE_BENCHMARK,
                    )
                    save_stats(stats)
                    print(f"TTFT: {ttft:0.3f}s | TPS~ {tps:0.2f} | events: {token_events}")

                del pipe
                gc.collect()
    finally:
        ACTIVE_DEVICE = previous_device


def build_chat_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "system":
            lines.append(f"System: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n".join(lines) + "\nAssistant:"


def create_openai_chat_response(model_name: str, content: str) -> dict:
    now = int(time.time())
    completion_tokens = max(1, len(content.split())) if content else 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }


def start_openai_compatible_server(state: dict) -> ThreadingHTTPServer:
    class OpenAICompatHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "Not found"}})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                data = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"error": {"message": "Invalid JSON body"}})
                return

            pipe = state.get("pipe")
            current = state.get("current")
            if pipe is None or current is None:
                self._send_json(400, {"error": {"message": "No model loaded. Use '/models' first."}})
                return

            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": {"message": "'messages' must be a non-empty list"}})
                return

            prompt = build_chat_prompt(messages)
            chunks: list[str] = []

            def streamer(chunk: str):
                chunks.append(chunk)

            try:
                pipe.generate(
                    prompt,
                    max_new_tokens=int(data.get("max_tokens", 300)),
                    temperature=float(data.get("temperature", 0.7)),
                    top_p=float(data.get("top_p", 0.9)),
                    streamer=streamer,
                )
            except Exception as exc:
                self._send_json(500, {"error": {"message": f"Generation failed: {exc}"}})
                return

            text = "".join(chunks)
            response = create_openai_chat_response(current["repo"], text)
            self._send_json(200, response)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", 1311), OpenAICompatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# =========================
# Main loop
# =========================
def main() -> None:
    global MODELS

    ensure_auth_file()
    MODELS = load_models()
    stats = load_stats()
    compat = load_device_compat()

    pipe = None
    current = None
    history = []
    server = None
    server_state = {"pipe": None, "current": None}

    print("Ready. Type '/help' to list commands.\n")

    while True:
        user_input = input("🧑 > ").strip()

        if not user_input:
            continue

        if is_command(user_input):
            cmd = normalize_command(user_input)

            if cmd == "help":
                print("\n" + HELP_TEXT)
                continue

            if cmd == "exit":
                break

            if cmd == "start_server":
                if server is not None:
                    print("\nℹ️ Server is already running on http://0.0.0.0:1311\n")
                    continue
                server = start_openai_compatible_server(server_state)
                print("\n✅ Server started at http://0.0.0.0:1311/v1/chat/completions\n")
                continue

            if cmd == "stats":
                print_stats_table(stats)
                continue

            if cmd.startswith("clear_stats"):
                parts = cmd.split()
                if len(parts) == 1:
                    clear_stats(stats)
                    continue
                if len(parts) == 2:
                    if not parts[1].isdigit():
                        print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                        continue
                    clear_stats(stats, model_number=int(parts[1]))
                    continue
                if len(parts) == 3:
                    if not parts[1].isdigit():
                        print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                        continue
                    clear_stats(stats, model_number=int(parts[1]), device=parts[2])
                    continue
                print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                continue

            if cmd == "current_model":
                if current is None:
                    print(
                        f"\n(no model loaded) | DEVICE={ACTIVE_DEVICE} | "
                        f"PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
                    )
                else:
                    print(
                        f"\nLoaded model: {model_menu_label(current)} | "
                        f"DEVICE={ACTIVE_DEVICE} | PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
                    )
                continue

            if cmd.startswith("benchmark"):
                parts = cmd.split(maxsplit=1)
                model_number = None
                only_missing_models = False
                if len(parts) == 2:
                    if not parts[1].isdigit():
                        print("\n⚠️ benchmark expects a model number, e.g. '/benchmark 2'.\n")
                        continue
                    model_number = int(parts[1])
                else:
                    only_missing_models = prompt_yes_no(
                        "Run only models missing benchmark stats?",
                        default_yes=True,
                    )

                prompts = collect_benchmark_prompts(5)
                benchmark_models(
                    stats,
                    prompts,
                    model_number=model_number,
                    only_missing_models=only_missing_models,
                    compat=compat,
                )
                continue

            if cmd == "add_model":
                add_model_interactive(MODELS)
                continue

            if cmd == "models":
                new_model = choose_model_interactive(
                    allow_download=True,
                    title="Choose a model to load:",
                    compat=compat,
                )
                if new_model is None:
                    print("\nCancelled.\n")
                    continue

                configure_runtime()

                # release previous
                if pipe is not None:
                    del pipe
                    gc.collect()

                current = new_model
                history = []
                try:
                    pipe = load_pipeline(current)
                    mark_model_device_compat(compat, current["repo"], ACTIVE_DEVICE, True)
                    save_device_compat(compat)
                    server_state["pipe"] = pipe
                    server_state["current"] = current
                except Exception as exc:
                    mark_model_device_compat(compat, current["repo"], ACTIVE_DEVICE, False)
                    save_device_compat(compat)
                    pipe = None
                    current = None
                    history = []
                    server_state["pipe"] = None
                    server_state["current"] = None
                    print(f"\n❌ Failed to load model on {ACTIVE_DEVICE}: {exc}\n")
                continue

            if cmd == "delete":
                to_delete = choose_model_interactive(
                    allow_download=False,
                    title="Choose a model to DELETE from disk (it remains in the list):",
                    compat=compat,
                )
                if to_delete is None:
                    print("\nCancelled.\n")
                    continue

                deleting_current = (current is not None and to_delete["repo"] == current["repo"])
                if deleting_current and pipe is not None:
                    del pipe
                    pipe = None
                    server_state["pipe"] = None
                    server_state["current"] = None
                    gc.collect()

                deleted = delete_model_files(to_delete)

                if deleting_current:
                    if deleted:
                        print("ℹ️ You deleted the active model. Load another one with '/models'.\n")
                        current = None
                        history = []
                        server_state["pipe"] = None
                        server_state["current"] = None
                    else:
                        print("ℹ️ Could not delete the active model.\n")
                continue

            print("\n⚠️ Unknown command. Use '/help'.\n")
            continue

        # Non-command: chat
        if pipe is None or current is None:
            print("⚠️ No model loaded. Use '/models' to load one (or '/help').\n")
            continue

        history.append(f"User: {user_input}")
        prompt = "\n".join(history) + "\nAssistant:"

        t_start = time.perf_counter()
        first_token_time = None
        token_events = 0

        def streamer(chunk: str):
            nonlocal first_token_time, token_events
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_events += 1
            print(chunk, end="", flush=True)

        print("🤖 > ", end="", flush=True)

        pipe.generate(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )

        t_end = time.perf_counter()
        print("\n")

        if first_token_time is None:
            ttft = t_end - t_start
            tps = 0.0
        else:
            ttft = first_token_time - t_start
            decode_time = max(1e-9, t_end - first_token_time)
            tps = token_events / decode_time

        stats_name = model_menu_label(current)
        record_stats(
            stats,
            current["repo"],
            stats_name,
            ACTIVE_DEVICE,
            ttft,
            tps,
            mode=STATS_MODE_NORMAL,
        )
        save_stats(stats)

        print(f"📈 TTFT: {ttft:0.3f}s | TPS≈ {tps:0.2f} | events: {token_events}")

        history.append("Assistant: (answer shown above)")

    if server is not None:
        server.shutdown()
        server.server_close()
    print("Bye.")


if __name__ == "__main__":
    main()
