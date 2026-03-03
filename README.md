# intel_npu_llm

Local CLI to download, load, test, and compare OpenVINO-optimized LLMs on CPU/GPU/NPU, with persistent metrics and a GitHub Pages presentation site.

## Features
- Interactive model management (`/models`, `/delete`, `/add_model`).
- Automatic download from Hugging Face (`snapshot_download`).
- Model loading with `openvino_genai` and runtime configuration when switching models.
- Per-model, per-device compatibility tracking (`✅/❌/❔`) shown in the model menu.
- TTFT/TPS metrics separated by:
  - normal chat mode,
  - benchmark mode,
  - model,
  - device.
- Granular metric cleanup with `/clear_stats`.
- Multi-device benchmark (`CPU/GPU/NPU`) with option to run only missing models.
- OpenAI-compatible backend server (`POST /v1/chat/completions`).

## Project Structure
- `chat_npu_13.py`: main CLI app.
- `ov_models/models.json`: editable model catalog.
- `ov_models/stats.json`: metrics history.
- `ov_models/device_compat.json`: compatibility by model/device.
- `docs/`: static GitHub Pages site.

## Requirements
- Windows + Python 3.10+
- Compatible Intel NPU (for NPU runs)
- Dependencies:
  - `openvino-genai`
  - `huggingface_hub`

Quick install:

```powershell
python -m pip install openvino-genai huggingface_hub
```

## Run

```powershell
python .\chat_npu_13.py
```

Or:

```powershell
.\run.ps1
```

## CLI Commands
- `/help`: show help.
- `/models`: select/load a model (downloads if missing).
- `/add_model`: add a model interactively and save it to `ov_models/models.json`.
- `/delete`: delete local files for one model.
- `/stats`: show separate tables for `normal` and `benchmark` metrics.
- `/clear_stats`: clear all metrics.
- `/clear_stats <n>`: clear all metrics for model number `<n>`.
- `/clear_stats <n> <device>`: clear metrics only for that model/device pair.
- `/current_model`: show currently loaded model and active runtime config.
- `/benchmark`: prompts to run all models or only missing models; always runs `CPU/GPU/NPU`.
- `/benchmark <n>`: run benchmark for model `<n>` on `CPU/GPU/NPU`.
- `/start_server`: start OpenAI-compatible API at `http://0.0.0.0:1311/v1/chat/completions`.
- `/exit`: exit.

## Model Catalog (`models.json`)
Each entry:

```json
{
  "display": "Visible name",
  "params": "7B",
  "repo": "owner/repo",
  "local_dir": "folder-name"
}
```

You can edit this file manually or use `/add_model`.

## Website (GitHub Pages)
The project includes a site in `docs/`:
- `docs/index.html`
- `docs/styles.css`
- `docs/app.js`
- `docs/benchmarks.json`
- `docs/benchmarks-data.js`

Site capabilities:
- Project overview.
- Current results table in `Benchmark` and `Normal` modes.
- Model and device selection.
- Row highlight by selected model.
- Per-column filters.
- Sort by column (click headers).

### Publish on GitHub Pages
1. Go to `Settings` → `Pages`.
2. Source: `Deploy from a branch`.
3. Branch: `main` (or your branch), folder: `/docs`.
4. Save.

## Tests
Run:

```powershell
python -m pytest -q tests
```

Current local suite covers critical flows (mode-separated stats, benchmark behavior, device compatibility tracking, slash commands, and JSON model catalog).

## Notes
- `ov_models/` stores runtime data and can grow significantly.
- For private Hugging Face repos, use `HF_TOKEN` or `ov_models/hf_auth.json`.
