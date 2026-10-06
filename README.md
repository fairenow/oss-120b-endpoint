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

The model is pinned server-side to `openai/gpt-oss-120b`. A persistent Modal Volume is mounted as the Hugging Face cache so downloaded model weights survive container shutdowns.

## Install

```bash
pip install -U modal
modal setup
```

## Deploy

From the repository root:

```bash
modal deploy modal_gpt_oss_120b.py
```

Modal will print the HTTPS endpoint after deployment.

## Authentication

The deployment is protected with:

```python
unauthenticated=False
```

Create a Modal proxy token:

```bash
modal workspace proxy-tokens create
```

Use the returned key and secret together as the bearer token:

```text
wk-....ws-....
```

## Test with curl

```bash
export GPT_OSS_URL="https://YOUR-MODAL-SERVER-URL"
export MODAL_PROXY_TOKEN="wk-....ws-...."

curl "$GPT_OSS_URL/v1/chat/completions" \
  -H "Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "Explain mixture-of-experts models."}
    ],
    "max_tokens": 500
  }'
```

## Test with the OpenAI Python client

```bash
pip install openai
export GPT_OSS_BASE_URL="https://YOUR-MODAL-SERVER-URL"
export MODAL_PROXY_TOKEN="wk-....ws-...."
python test_client.py
```

## Runtime configuration

- GPU: H100
- Model: `openai/gpt-oss-120b`
- Server: `transformers serve`
- Continuous batching enabled
- Persistent Hugging Face model cache
- 10-minute scale-down window
- Protected Modal HTTP endpoint

The model weights are not committed to this repository. Modal downloads them into the persistent cache volume at runtime.
