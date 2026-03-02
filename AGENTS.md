# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally small:

- `chat_npu_13.py`: main interactive CLI for model download, selection, inference, and metrics.
- `ov_models/`: local model cache and runtime artifacts (`*.xml`, `*.bin`, tokenizer files).
- `ov_models/stats.json`: persisted TTFT/TPS benchmark history.
- `ov_models/hf_auth.json`: optional Hugging Face token store.

Treat `ov_models/` as generated/runtime data. Keep new source code at repo root or in a new `src/` package if the project grows.

## Build, Test, and Development Commands
Use Python 3.10+ in a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install openvino-genai huggingface_hub
python chat_npu_13.py
```

- `python chat_npu_13.py`: launches the local chat + model-management loop.
- Inside the app, use `help`, `models`, `stats`, and `delete` commands.

No Makefile or CI tasks are currently defined.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Prefer `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for constants (as in `DEVICE`, `CACHE_DIR`).
- Keep functions focused and side-effect boundaries clear (I/O, model loading, stats persistence).
- Add type hints for new/changed functions.
- Preserve existing CLI command naming and user-facing behavior unless intentionally changing UX.

## Testing Guidelines
There is no test suite yet. For new logic, add `pytest` tests under `tests/` with names like `test_stats.py` and `test_model_utils.py`.

Run when available:

```powershell
pytest -q
```

Prioritize tests for pure logic (`human_bytes`, stats aggregation, command parsing). Manually verify model flows by running the CLI.

## Commit & Pull Request Guidelines
This repository currently has no commit history; use Conventional Commits going forward:

- `feat: add model filtering command`
- `fix: guard delete path traversal`
- `docs: update setup steps`

PRs should include:
- clear summary and rationale,
- test evidence (manual/automated),
- linked issue (if applicable),
- screenshots or terminal snippets for CLI UX changes.
