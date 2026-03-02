import gc
import json
import os
import shutil
import time
from pathlib import Path

import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

# =========================
# Config
# =========================
DEVICE = "NPU"
CACHE_DIR = Path("ov_models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS_FILE = CACHE_DIR / "stats.json"
AUTH_FILE = CACHE_DIR / "hf_auth.json"  # {"hf_token": "hf_..."}

# =========================
# Model list
# =========================
MODELS = [
    {
        "display": "Phi-4 mini instruct INT4",
        "params": "≈3.8B",
        "repo": "FluidInference/phi-4-mini-instruct-int4-ov-npu",
        "local": CACHE_DIR / "phi-4-mini-instruct-int4-ov-npu",
    },
    {
        "display": "Qwen2.5 1.5B instruct INT4",
        "params": "1.5B",
        "repo": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "local": CACHE_DIR / "qwen2.5-1.5b-instruct-int4-ov",
    },
    {
        "display": "TinyLlama 1.1B Chat INT4",
        "params": "1.1B",
        "repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "local": CACHE_DIR / "tinyllama-1.1b-chat-int4-ov",
    },
    {
        "display": "Phi-3 Medium 4k Instruct INT4",
        "params": "3.8B",
        "repo": "OpenVINO/Phi-3-medium-4k-instruct-int4-ov",
        "local": CACHE_DIR / "phi-3-medium-4k-instruct-int4-ov",
    },
    {
        "display": "Qwen2.5 7B INT4 NPU OV",
        "params": "7B",
        "repo": "FluidInference/qwen2.5-7b-int4-npu-ov",
        "local": CACHE_DIR / "qwen2.5-7b-int4-npu-ov",
    },
    {
        "display": "Llama 3.1 8B Instruct INT4 NPU OV",
        "params": "8B",
        "repo": "llmware/llama-3.1-8b-instruct-npu-ov",
        "local": CACHE_DIR / "llama-3.1-8b-instruct-npu-ov",
    },
]

# =========================
# HF token handling
# =========================
def ensure_auth_file():
    if not AUTH_FILE.exists():
        AUTH_FILE.write_text(json.dumps({"hf_token": ""}, indent=2), encoding="utf-8")


def load_hf_token() -> str | None:
    """
    Busca token en:
      1) env HF_TOKEN
      2) ov_models/hf_auth.json  ({"hf_token": "hf_..."})
    Si el archivo existe pero está vacío, pide token y lo guarda.
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
        print("\n⚠️ HF token no configurado (opcional).")
        print("Pegá tu token (empieza con 'hf_') o Enter para seguir sin token.\n")
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


def download_model(repo_id: str, local_dir: Path):
    load_hf_token()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 Bajando modelo: {repo_id}")
    print(f"   -> destino: {local_dir}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("\n✅ Descarga completa.\n")


def choose_model_interactive(allow_download: bool, title: str) -> dict | None:
    print(f"\n{title}\n")
    for i, m in enumerate(MODELS, 1):
        status = "✅" if is_downloaded(m["local"]) else "⬇️"
        print(f"  {i}) {status} {model_menu_label(m)}")
    print("  0) Cancelar\n")

    while True:
        choice = input("Opción: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            m = MODELS[int(choice) - 1]
            if allow_download and not is_downloaded(m["local"]):
                download_model(m["repo"], m["local"])
            return m
        print("Opción inválida.")


def delete_model_files(m: dict) -> bool:
    path = m["local"]
    if not path.exists():
        print("\nℹ️ Ese modelo no está en disco.\n")
        return False

    try:
        resolved = path.resolve()
        cache = CACHE_DIR.resolve()
        if cache not in resolved.parents and resolved != cache:
            print("\n❌ Seguridad: el path no está dentro de ov_models, no borro.\n")
            return False
    except Exception:
        print("\n❌ No pude resolver paths para borrar de forma segura.\n")
        return False

    size_before = human_bytes(dir_size_bytes(path))
    print(f"\n🗑️ Borrando: {model_menu_label(m)}")
    print(f"   Path: {path}")
    print(f"   Tamaño: {size_before}\n")

    try:
        shutil.rmtree(path)
        print("✅ Borrado completo.\n")
        return True
    except Exception as e:
        print(f"\n❌ Error borrando: {e}\n")
        return False


# =========================
# Stats persistence
# =========================
def load_stats() -> dict:
    try:
        if STATS_FILE.exists():
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"models": {}}


def save_stats(stats: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def record_stats(stats: dict, model_key: str, model_name: str, ttft_s: float, tps: float):
    models = stats.setdefault("models", {})
    entry = models.setdefault(
        model_key,
        {"name": model_name, "runs": 0, "ttft_s": [], "tps": []},
    )
    entry["name"] = model_name
    entry["runs"] += 1
    entry["ttft_s"].append(ttft_s)
    entry["tps"].append(tps)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def print_stats_table(stats: dict):
    models = stats.get("models", {})
    if not models:
        print("\n(no hay stats todavía)\n")
        return

    rows = []
    for k, v in models.items():
        ttfts = v.get("ttft_s", [])
        tpss = v.get("tps", [])
        rows.append(
            {
                "model": v.get("name", k),
                "runs": v.get("runs", 0),
                "ttft_avg": mean(ttfts),
                "tps_avg": mean(tpss),
                "ttft_last": ttfts[-1] if ttfts else 0.0,
                "tps_last": tpss[-1] if tpss else 0.0,
            }
        )

    rows.sort(key=lambda r: r["tps_avg"], reverse=True)

    def f(x):
        return f"{x:0.3f}"

    headers = ["Model", "n", "TTFT avg(s)", "TPS avg", "TTFT last(s)", "TPS last"]
    colw = [74, 4, 12, 10, 13, 9]

    def cut(s, w):
        s = str(s)
        return s if len(s) <= w else s[: w - 1] + "…"

    line = (
        f"{cut(headers[0], colw[0]):<{colw[0]}} "
        f"{headers[1]:>{colw[1]}} "
        f"{headers[2]:>{colw[2]}} "
        f"{headers[3]:>{colw[3]}} "
        f"{headers[4]:>{colw[4]}} "
        f"{headers[5]:>{colw[5]}}"
    )
    sep = "-" * len(line)

    print("\n" + line)
    print(sep)
    for r in rows:
        print(
            f"{cut(r['model'], colw[0]):<{colw[0]}} "
            f"{r['runs']:>{colw[1]}} "
            f"{f(r['ttft_avg']):>{colw[2]}} "
            f"{f(r['tps_avg']):>{colw[3]}} "
            f"{f(r['ttft_last']):>{colw[4]}} "
            f"{f(r['tps_last']):>{colw[5]}}"
        )
    print("")


# =========================
# Pipeline
# =========================
def load_pipeline(selected_model: dict) -> ov_genai.LLMPipeline:
    model_path = selected_model["local"]
    print(f"\n✅ Modelo cargado en {DEVICE}: {model_menu_label(selected_model)}")
    return ov_genai.LLMPipeline(
        str(model_path),
        DEVICE,
        PERFORMANCE_HINT="LATENCY",
    )


# =========================
# UI / Commands
# =========================
HELP_TEXT = """\
Comandos:
  help                 Muestra esta ayuda
  /help                Alias de help
  models               Elegir y cargar un modelo (descarga si falta)
  /models              Alias de models
  delete               Borrar archivos de un modelo del disco (no lo saca de la lista)
  /delete              Alias de delete
  stats                Ver métricas (TTFT/TPS) acumuladas por modelo
  /stats               Alias de stats
  curre:model          Ver el modelo actualmente cargado
  /curre:model         Alias de curre:model
  exit                 Salir
  /exit                Alias de exit

Uso:
  - Primero corré 'models' para cargar uno.
  - Luego escribí cualquier prompt normal para chatear.
"""


def is_command(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s.startswith("/"):
        s = s[1:]
    return s in {"help", "models", "delete", "stats", "exit", "curre:model"}


def normalize_command(s: str) -> str:
    s = s.strip()
    if s.startswith("/"):
        s = s[1:]
    return s


# =========================
# Main loop
# =========================
def main():
    ensure_auth_file()
    stats = load_stats()

    pipe = None
    current = None
    history = []

    print("Listo. Escribí 'help' para ver comandos.\n")

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

            if cmd == "stats":
                print_stats_table(stats)
                continue

            if cmd == "curre:model":
                if current is None:
                    print("\n(no hay modelo cargado)\n")
                else:
                    print(f"\nModelo cargado: {model_menu_label(current)}\n")
                continue

            if cmd == "models":
                new_model = choose_model_interactive(
                    allow_download=True,
                    title="Elegí un modelo para cargar:",
                )
                if new_model is None:
                    print("\nCancelado.\n")
                    continue

                # liberar anterior
                if pipe is not None:
                    del pipe
                    gc.collect()

                current = new_model
                history = []
                pipe = load_pipeline(current)
                continue

            if cmd == "delete":
                to_delete = choose_model_interactive(
                    allow_download=False,
                    title="Elegí un modelo para BORRAR del disco (no se saca de la lista):",
                )
                if to_delete is None:
                    print("\nCancelado.\n")
                    continue

                deleting_current = (current is not None and to_delete["repo"] == current["repo"])
                if deleting_current and pipe is not None:
                    del pipe
                    pipe = None
                    gc.collect()

                deleted = delete_model_files(to_delete)

                if deleting_current:
                    if deleted:
                        print("ℹ️ Borraste el modelo activo. Cargá otro con 'models'.\n")
                        current = None
                        history = []
                    else:
                        print("ℹ️ No se pudo borrar el modelo activo.\n")
                continue

        # Si no es comando: chat
        if pipe is None or current is None:
            print("⚠️ No hay modelo cargado. Usá 'models' para cargar uno (o 'help').\n")
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
        record_stats(stats, current["repo"], stats_name, ttft, tps)
        save_stats(stats)

        print(f"📈 TTFT: {ttft:0.3f}s | TPS≈ {tps:0.2f} | events: {token_events}")

        history.append("Assistant: (respuesta arriba)")

    print("Bye.")


if __name__ == "__main__":
    main()