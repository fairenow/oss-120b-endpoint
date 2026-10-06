# GPT-OSS-120B Modal Endpoint

Deploy `openai/gpt-oss-120b` on Modal using Hugging Face Transformers Serve.

The deployed service exposes OpenAI-compatible Transformers endpoints, including:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/responses`
- `/v1/models`

## Architecture

```text
Client application
      |
      v
Modal HTTPS endpoint
      |
      v
transformers serve
      |
      v
openai/gpt-oss-120b
      |
      v
H100 GPU
```

A persistent Modal Volume is mounted as the Hugging Face cache so downloaded model weights survive container shutdowns.

## Install / update Modal locally

On macOS, prefer Python's module form rather than assuming the `pip` executable is on PATH:

```bash
python3 -m pip install -U modal openai
```

If your Python installation says that `pip` itself is missing:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install -U modal openai
```

Authenticate once:

```bash
modal setup
```

## Deploy

```bash
git pull
modal deploy modal_gpt_oss_120b.py
```

The deployment intentionally uses Modal's long-supported `@app.function()` + `@modal.web_server()` pattern rather than requiring the newer `@app.server()` primitive.

## Authentication

The web server uses:

```python
requires_proxy_auth=True
```

Create a Modal proxy token:

```bash
modal workspace proxy-tokens create
```

Use the returned key and secret together as the bearer token.

## Test with curl

```bash
export GPT_OSS_URL="https://YOUR-MODAL-SERVER-URL"
export MODAL_PROXY_TOKEN="wk-....ws-...."

curl "$GPT_OSS_URL/v1/chat/completions" \
  -H "Modal-Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "Explain mixture-of-experts models."}
    ],
    "max_tokens": 500
  }'
```

## Runtime configuration

- GPU: H100
- Model: `openai/gpt-oss-120b`
- Server: `transformers serve`
- Continuous batching enabled
- Persistent Hugging Face model cache
- 10-minute scale-down window
- Modal proxy authentication enabled

The model weights are not committed to this repository.
