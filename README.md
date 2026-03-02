# intel_npu_llm

Python CLI to download, load, and run OpenVINO-optimized LLMs on Intel NPU, with basic runtime metrics (`TTFT` and `TPS`).

## Features
- Interactive menu to pick a model.
- Automatic model download from Hugging Face.
- Model loading on `NPU` via `openvino_genai`.
- Management commands (`models`, `delete`, `stats`, `benchmark`, `current_model`, `start_server`, `help`).
- Persisted metrics in `ov_models/stats.json`.

## Requirements
- Windows + Python 3.10+
- Compatible Intel NPU
- Dependencies:
  - `openvino-genai`
  - `huggingface_hub`

## Quick install
```powershell
python -m pip install openvino-genai huggingface_hub
```

## Run the app
```powershell
.\run.ps1
```

You can also run it directly:
```powershell
python .\chat_npu_13.py
```

## In-app commands
- `help`: show help
- `models`: choose/load model (downloads if missing)
- `delete`: remove local model files
- `stats`: show aggregated TTFT/TPS metrics
- `benchmark`: ask for 5 prompts and run them on all downloaded models (prompts are saved and can be reused in future runs)
- `benchmark <number>`: run benchmark on one model from the list
- `current_model`: show active loaded model
- `start_server`: start an OpenAI-compatible server on port `1311` (`POST /v1/chat/completions`)
- `exit`: quit

## Stats table
The app stores benchmark/chat runs in `ov_models/stats.json` and shows them in a table ordered by average TPS. Benchmark prompts are stored in `ov_models/benchmark_prompts.json`.

| Model | n | TTFT avg(s) | TPS avg | TTFT last(s) | TPS last |
|---|---:|---:|---:|---:|---:|
| Phi-4 mini instruct INT4 (≈3.8B, 8.40 GB) | 5 | 0.842 | 16.203 | 0.799 | 16.884 |
| Qwen2.5 1.5B instruct INT4 (1.5B, 3.20 GB) | 5 | 0.611 | 21.107 | 0.592 | 21.443 |

> Values above are an example format. Actual values come from your local runs.

## Tests and coverage
```powershell
.\test.ps1
```

This command runs tests and prints coverage in console (target: minimum `90%` on `chat_npu_13.py`).

## Notes
- Models are stored in `ov_models/` and are **not** versioned in Git.
- For private Hugging Face repos, set `HF_TOKEN` or use `ov_models/hf_auth.json`.
