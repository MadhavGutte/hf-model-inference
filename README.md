# HF Model Inference API (FastAPI + vLLM)

Minimal Python project to host **any Hugging Face text generation model** (base or fine-tuned) using a single API.

- Backend options: `vllm` (recommended for GPU) or `transformers`
- API endpoint: `POST /generate`
- Health endpoint: `GET /health`
- Works with environment-based configuration for easy deployment

## 1) Project structure

```text
hf-model-server/
  app/
    config.py
    engine.py
    main.py
  .env.example
  docker-compose.yml
  Dockerfile
  requirements.txt
  README.md
```

## 2) Environment configuration

Create your runtime env file from template:

```bash
cp .env.example .env
```

Set at least:

- `MODEL_ID` (any HF model id)
- `INFERENCE_BACKEND=vllm` (or `transformers`)

Optional:

- `HF_TOKEN` for gated/private models
- `TENSOR_PARALLEL_SIZE` for multi-GPU
- `QUANTIZATION` for quantized loading

### Env injection options when deploying

You can inject env values in multiple ways:

1. **`.env` file** (used by docker compose)
2. `docker run -e KEY=value ...`
3. Platform-level env vars (Kubernetes/VM service manager/CI-CD)

The app always reads from environment, so you can use whichever deployment style you prefer.

## 3) Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 4) Run with Docker Compose (GPU instance)

Prerequisites:

- NVIDIA GPU drivers installed
- NVIDIA Container Toolkit configured
- Docker Compose v2+

Commands:

```bash
cp .env.example .env
# edit .env
docker compose up --build
```

API URL: `http://localhost:8000`

### Quantization (INT8 / INT4)

Use env variables to pick quantization without code changes.

Examples:

- `QUANTIZATION=int8`
- `QUANTIZATION=int4`

Notes:

- For `transformers` backend, `int8`/`int4` maps to bitsandbytes quantization.
- For `vllm`, `int8`/`int4` maps to `bitsandbytes` mode; you can also set `QUANTIZATION=awq` or `QUANTIZATION=gptq` if your model supports them.
- You can force transformers bits with `QUANTIZATION_BITS=4` or `QUANTIZATION_BITS=8`.

## 5) API usage

### Health check

```bash
curl http://localhost:8000/health
```

### Generate text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short intro about transformers.",
    "max_new_tokens": 120,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

Response example:

```json
{
  "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
  "backend": "vllm",
  "generated_text": "..."
}
```

## 6) Notes for model compatibility

- This project is for **text generation models** from Hugging Face.
- For gated models, set `HF_TOKEN` in env.
- If a model requires custom code, set `TRUST_REMOTE_CODE=true`.
- For non-generation tasks (embedding/classification/vision), you can keep the same FastAPI layout and change only `app/engine.py` task logic.

## 7) Guardrails

Guardrails are safety and control rules around model inference. In this project, they run before generation to keep API usage bounded and policy-aware.

Available guardrails (env-driven):

- `ENABLE_GUARDRAILS=true|false`
- `MAX_PROMPT_CHARS` to reject overlong prompts
- `MAX_REQUEST_NEW_TOKENS` to cap generation size per request
- `BLOCKED_TERMS` (comma-separated) to reject prompts containing specific terms

Example:

```dotenv
ENABLE_GUARDRAILS=true
MAX_PROMPT_CHARS=3000
MAX_REQUEST_NEW_TOKENS=256
BLOCKED_TERMS=malware source code,credit card dump
```

Behavior:

- Guardrail violations return HTTP `400` with a clear reason.
- This logic is implemented in `app/guardrails.py` and used by `POST /generate`.

## 8) Troubleshooting

### `ValueError: 'aimv2' is already used by a Transformers config`

This means your installed `transformers` version is incompatible with the pinned `vllm` build.

Run:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

Then restart the API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
