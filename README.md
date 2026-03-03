# intel_npu_llm

CLI local para descargar, cargar, probar y comparar LLMs optimizados para OpenVINO en CPU/GPU/NPU, con métricas persistentes y una web de presentación para GitHub Pages.

## Features
- Gestión interactiva de modelos (`/models`, `/delete`, `/add_model`).
- Descarga automática desde Hugging Face (`snapshot_download`).
- Carga con `openvino_genai` y configuración de runtime por cambio de modelo.
- Compatibilidad por device persistida por modelo (`✅/❌/❔`) en menú de modelos.
- Métricas TTFT/TPS por:
  - modo normal de chat,
  - modo benchmark,
  - modelo,
  - device.
- Limpieza de métricas granular con `/clear_stats`.
- Benchmark multi-device (`CPU/GPU/NPU`) y opción de correr solo modelos faltantes.
- Servidor compatible con OpenAI (`POST /v1/chat/completions`).

## Estructura
- `chat_npu_13.py`: CLI principal.
- `ov_models/models.json`: catálogo editable de modelos.
- `ov_models/stats.json`: historial de métricas.
- `ov_models/device_compat.json`: compatibilidad por modelo/device.
- `docs/`: sitio estático para GitHub Pages.

## Requisitos
- Windows + Python 3.10+
- Intel NPU compatible (para pruebas NPU)
- Dependencias:
  - `openvino-genai`
  - `huggingface_hub`

Instalación rápida:

```powershell
python -m pip install openvino-genai huggingface_hub
```

## Ejecutar

```powershell
python .\chat_npu_13.py
```

O:

```powershell
.\run.ps1
```

## Comandos CLI
- `/help`: ayuda.
- `/models`: seleccionar/cargar modelo (descarga si falta).
- `/add_model`: alta interactiva de modelo y guardado en `ov_models/models.json`.
- `/delete`: borrar archivos locales de un modelo.
- `/stats`: muestra dos tablas separadas (`normal` y `benchmark`).
- `/clear_stats`: limpia métricas.
- `/clear_stats <n>`: limpia todas las métricas del modelo `<n>`.
- `/clear_stats <n> <device>`: limpia solo ese modelo/device.
- `/current_model`: muestra modelo cargado y runtime activo.
- `/benchmark`: pregunta si correr todos los modelos o solo faltantes; siempre en `CPU/GPU/NPU`.
- `/benchmark <n>`: benchmark del modelo `<n>` en `CPU/GPU/NPU`.
- `/start_server`: inicia API compatible OpenAI en `http://0.0.0.0:1311/v1/chat/completions`.
- `/exit`: salir.

## Catálogo de modelos (`models.json`)
Cada entrada:

```json
{
  "display": "Nombre visible",
  "params": "7B",
  "repo": "owner/repo",
  "local_dir": "folder-name"
}
```

Podés editar este archivo manualmente o usar `/add_model`.

## Sitio web (GitHub Pages)
El proyecto incluye un sitio en `docs/`:
- `docs/index.html`
- `docs/styles.css`
- `docs/app.js`
- `docs/benchmarks.json`
- `docs/benchmarks-data.js`

Funcionalidad del sitio:
- Presentación del proyecto.
- Tabla de resultados actuales en modos `Benchmark` y `Normal`.
- Selección de modelo y device.
- Resaltado de filas por modelo seleccionado.
- Filtro por cada columna.
- Ordenamiento por columna (click en encabezados).

### Publicar en GitHub Pages
1. Ir a `Settings` → `Pages`.
2. Source: `Deploy from a branch`.
3. Branch: `main` (o la que uses), folder: `/docs`.
4. Guardar.

## Tests
Ejecutar:

```powershell
python -m pytest -q tests
```

Actualmente la suite pasa en local con cobertura de flujos críticos (stats por modo, benchmark, compatibilidad device, slash commands y modelos JSON).

## Notas
- `ov_models/` contiene datos de runtime y puede crecer mucho.
- Para repos privados de Hugging Face, usar `HF_TOKEN` o `ov_models/hf_auth.json`.
