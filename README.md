# intel_npu_llm

CLI en Python para descargar, cargar y probar modelos LLM optimizados para OpenVINO sobre NPU, con métricas básicas de rendimiento (`TTFT` y `TPS`).

## Características
- Menú interactivo para elegir modelo.
- Descarga automática desde Hugging Face.
- Carga de modelos en `NPU` con `openvino_genai`.
- Comandos de gestión (`models`, `delete`, `stats`, `help`).
- Persistencia de métricas en `ov_models/stats.json`.

## Requisitos
- Windows + Python 3.10+
- Intel NPU compatible
- Dependencias:
  - `openvino-genai`
  - `huggingface_hub`

## Instalación rápida
```powershell
python -m pip install openvino-genai huggingface_hub
```

## Ejecutar la app
```powershell
.\run.ps1
```

También podés correrla directo:
```powershell
python .\chat_npu_13.py
```

## Comandos dentro de la app
- `help`: ayuda
- `models`: elegir/cargar modelo (descarga si falta)
- `delete`: borrar modelo local
- `stats`: ver métricas acumuladas
- `curre:model`: ver modelo activo
- `exit`: salir

## Tests y coverage
```powershell
.\test.ps1
```

Este comando ejecuta tests y muestra cobertura en pantalla (objetivo mínimo: `90%` sobre `chat_npu_13.py`).

## Notas
- Los modelos se guardan en `ov_models/` y **no** se versionan en Git.
- Si necesitás modelos privados en Hugging Face, configurá `HF_TOKEN` o `ov_models/hf_auth.json`.
