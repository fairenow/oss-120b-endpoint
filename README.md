# Multi-Model Modal Endpoints

Deploy several open-weight models behind a single Modal app. Each model runs as its
own GPU endpoint with the serving stack it actually needs.

| Endpoint | Model (Hugging Face repo) | Task | Serving stack | Route | GPU |
| --- | --- | --- | --- | --- | --- |
| `gpt_oss_120b` | `openai/gpt-oss-120b` | text chat | `transformers serve` (rollback) | `/v1/chat/completions` | H100 |
| `gpt_oss_120b_vllm` | `openai/gpt-oss-120b` | text chat + tools | **vLLM 0.31.0** (staging) | `/v1/chat/completions` | H100 |
| `deepseek_v4_1_flash` | `deepseek-ai/DeepSeek-V4.1-Flash` | text / image chat + tools | **vLLM 0.31.0** | `/v1/chat/completions` | H200:8 |
| `QwenImage2512` | `Qwen/Qwen-Image-2512` | text to image | `diffusers.QwenImagePipeline` | `/v1/images/generations` | H100 |
| `Ltx25` | `Lightricks/LTX-2.5-Diffusers` | text to video | `diffusers.LTX2Pipeline` | `/v1/videos` | H100 |
| `Wan22` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | text to video | `diffusers.WanPipeline` | `/v1/videos` | L40S |
| `HunyuanVideo15` | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | text to video | `diffusers.HunyuanVideo15Pipeline` | `/v1/videos` | H100 |

## Why "one app, multiple endpoints" instead of one shared endpoint

These models are two different modalities and cannot share one server:

- `transformers serve` is an OpenAI-compatible server for **text / multimodal LLM**
  inference. It does not expose image or video generation.
- Qwen-Image, LTX-2.5, Wan 2.2, and HunyuanVideo 1.5 are **diffusion** pipelines.
  They need `diffusers` and their own FastAPI routes (`/v1/images/generations`,
  `/v1/videos`).
- GPU memory is per-container. gpt-oss-120B fits an H100; DeepSeek-V4.1-Flash is a
  ~552B-backbone + ~196B-Engram multimodal MoE (~476 GiB of weights) that needs at
  least 8×H200 or a Blackwell node. It cannot run on H100 (80 GB) and cannot be
  co-located with gpt-oss.

So the app defines one endpoint per model. Modal scales each independently and each
mounts the shared Hugging Face cache volume so weights survive container shutdowns.

## Architecture

```text
Clients
   |
   +-- /v1/chat/completions ...... gpt-oss-120b (transformers serve, H100)  [rollback]
   +-- /v1/chat/completions ...... gpt-oss-120b (vLLM, H100)                [staging]
   +-- /v1/chat/completions ...... DeepSeek-V4.1-Flash (vLLM, H200:8)
   +-- /v1/images/generations .... Qwen-Image-2512 (FastAPI + diffusers, H100)
   +-- /v1/videos ................ LTX-2.5 (FastAPI + diffusers, H100)
   +-- /v1/videos ................ Wan 2.2 (FastAPI + diffusers, L40S)
   +-- /v1/videos ................ HunyuanVideo 1.5 (FastAPI + diffusers, H100)
                  |
             Modal proxy auth
                  |
        gpt-oss-120b-hf-cache + gpt-oss-120b-vllm-cache volumes
```

## GPT-OSS on vLLM (staging)

`transformers serve` does not parse gpt-oss's Harmony channels, so tool calls leaked
into `content` as text and no structured `tool_calls` were returned. The staging
endpoint `gpt_oss_120b_vllm` runs **vLLM `0.31.0`** with the documented GPT-OSS parsers:

```text
vllm serve openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --gpu-memory-utilization 0.95 --max-model-len 32768 --max-num-seqs 16
```

Parser names are validated against the installed version by `vllm_selftest`
(`modal run modal_models.py::vllm_selftest`): `openai` and `openai_gptoss` are both
registered in vLLM 0.31.0. No built-in/browsing/python tool server is enabled —
only user-supplied functions are parsed; executing them stays the caller's job.

The original `gpt_oss_120b` (`transformers serve`) endpoint is preserved as the
rollback option.

### Verified results (H100 SXM5, single GPU)

| Property | Value |
| --- | --- |
| Model ID (normalized) | `gpt-oss-120b` (`root: openai/gpt-oss-120b`) |
| Weights on disk / loaded | 60.77 GiB / 61.43 GiB |
| Engine init (load + compile + graph capture) | ~226 s |
| Cold start (first request, warm HF cache) | ~335 s |
| GPU KV cache | 11.72 GiB / 226,940 tokens (≈6.9 concurrent @32k) |
| Warm latency (chat / tool call) | ~0.5–0.7 s |
| Throughput (12 concurrent, 48 tok each) | ~553 completion tok/s, ~1,439 total tok/s |
| GPU cost | H100 @ $0.001097/s = $3.95/h ⇒ ≈$0.37 per cold start, ≈$2 per 1M output tokens |

Behavioral notes:

- Plain chat and reasoning prompts return clean `content` with no Harmony control
  tokens. For these prompts gpt-oss wrote chain-of-thought inline in `content`;
  `reasoning_content` was not populated.
- Structured tool calls work non-streaming and streaming (`finish_reason: "tool_calls"`,
  valid JSON `arguments`, `tool_call_id`), and tool-result round trips complete.
- Integer arguments are correctly coerced (`"noon"` → `{"hour": 12}`).
- gpt-oss-120b did **not** emit parallel tool calls in testing (one call per turn,
  even with two distinct tools). Callers should not assume parallelism.


## DeepSeek-V4.1-Flash serving audit

The original endpoint (`transformers serve` on `H100:8`) was never a supported
configuration and has been replaced by **vLLM 0.31.0 on `H200:8`**. Findings from the
audit (2026-10-09), done without paid GPU runs:

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash` (revision `2cba9e42aa02…`),
  architecture `DeepseekV41ForCausalLM` / `model_type deepseek_v41`, FP8 dense +
  MXFP4 MoE experts, image-text-to-text, 1M context.
- **Size:** ~552B backbone + ~196B Engram parameters (~476 GiB of weights; ~614 GB
  minimum VRAM with headroom). Serviceable on ≥8×H200 (tensor parallel + Engram CPU
  offload) or a Blackwell node (GB200 NVL4 768 GB / B200 DEP8). **Not supported on
  H100 (80 GB):** the expert kernels are MXFP4 (native only on Blackwell) and
  8×H100 = 640 GB leaves no room for KV cache. The weights were not in the HF cache.
- **Backend:** vLLM (recommended) over SGLang. The day-0 vLLM recipe is the most
  complete; NIM/Dynamo also validate SGLang as a fallback. Requires the exact
  `deepseek_v41` tokenizer/tool/reasoning parsers — **not** `deepseek_v4`.
- **Cold start:** first-boot kernel autotune can take tens of minutes (GB300: ~75–90
  min first boot, ~7 min warm); the first run also downloads ~511 GB of weights into
  the HF cache volume. `startup_timeout` is 90 min and scale-down 30 min.
- **Cost (Modal):** H200:8 ≈ $36.3/h, B200:8 ≈ $50/h; H100:8 (unsupported) ≈ $31.6/h.

### Configuration (`modal_models.py`)

```text
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --trust-remote-code --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tokenizer-mode deepseek_v41 \
  --gpu-memory-utilization 0.92 --max-model-len 32768
```

vLLM 0.31.0 registers all of these (verified by `vllm_selftest`):
`DeepseekV41ForCausalLM` is a supported architecture, and `deepseek_v41` exists as a
tool parser, reasoning parser, and tokenizer mode.

### Behavioral notes / gateway implications

- Thinking is on by default (reasoning effort ≈50) and arrives in
  `reasoning_content`.
- With too small an output budget the model returns **empty `content` +
  `finish_reason: "length"`** (the thinking block consumed the budget). This is not
  an error — callers must set an adequate `max_tokens` (and/or lower the reasoning
  effort), and the gateway should not treat `finish_reason: "length"` as a failure.
- Tool calls use DeepSeek's DSML tag blocks; tool results are returned as
  `<tool_result>`.

### Status

**Not run live yet.** No GPU test has been performed against this endpoint: the
supported configuration is expensive (≈$36–50/h) and the previous H100:8 +
`transformers serve` config could not load the model. The code and docs encode the
supported configuration; a paid staging run must verify load, tool calling,
reasoning, timing, and cost before any production GSW routing change.

## Install / update Modal locally

On macOS, prefer Python's module form rather than assuming the `pip` executable is on PATH:

```bash
python3 -m pip install -U modal openai requests
```

If your Python installation says that `pip` itself is missing:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install -U modal openai requests
```

Authenticate once:

```bash
modal setup
```

## Deploy

```bash
git pull
modal deploy modal_models.py
```

`modal deploy` prints a URL for every endpoint. Each endpoint can also be deployed
and scaled on its own.

The deployment uses Modal's long-supported `@app.function()` + `@modal.web_server()`
pattern for the text servers and `@app.cls()` + `@modal.asgi_app()` for the diffusion
servers. `modal_gpt_oss_120b.py` is the original single-model app and is superseded
by `modal_models.py`.

## Authentication

Every endpoint uses:

```python
requires_proxy_auth=True
```

Create a Modal proxy token (the `modal workspace` command is not in Modal CLI 1.2.4;
run it through an ephemeral latest CLI):

```bash
pipx run --spec modal modal workspace proxy-tokens create --name multi-model-endpoints
```

Use the returned key and secret together as the bearer token. Pass it either as
`Modal-Key` / `Modal-Secret` headers, or joined as an `Authorization: Bearer` token.

## Test

Text (OpenAI-compatible):

```bash
export GPT_OSS_BASE_URL="https://YOUR-GPT-OSS-URL"
export DEEPSEEK_BASE_URL="https://YOUR-DEEPSEEK-URL"
export MODAL_PROXY_TOKEN="wk-....ws-...."

python test_client.py gpt-oss
python test_client.py deepseek
```

Image and video:

```bash
export QWEN_IMAGE_BASE_URL="https://YOUR-QWEN-IMAGE-URL"
export LTX_BASE_URL="https://YOUR-LTX-URL"
export WAN_BASE_URL="https://YOUR-WAN-URL"
export HUNYUAN_BASE_URL="https://YOUR-HUNYUAN-URL"

python test_client.py image      # writes out.png
python test_client.py ltx        # writes ltx.mp4
python test_client.py wan        # writes wan.mp4
python test_client.py hunyuan    # writes hunyuan.mp4
```

Chat with curl:

```bash
curl "$GPT_OSS_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain mixture-of-experts models."}],
    "max_tokens": 500
  }'
```

Generate an image with curl:

```bash
curl "$QWEN_IMAGE_BASE_URL/v1/images/generations" \
  -H "Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red panda surfing a wave at sunset", "seed": 42}'
```

Generate a video with curl:

```bash
curl "$WAN_BASE_URL/v1/videos" \
  -H "Authorization: Bearer $MODAL_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A drone shot over a snowy mountain range at sunrise", "seed": 42}'
```

Both the image and video endpoints return base64-encoded bytes in
`data[0].b64_json` (`PNG` and `MP4` respectively).

## Runtime configuration

- Shared persistent Hugging Face cache volume: `gpt-oss-120b-hf-cache` (reused from
  the original single-model deployment, so its download is not repeated; the volume
  now caches every model in the app)
- vLLM compile cache volume: `gpt-oss-120b-vllm-cache` (`VLLM_CACHE_ROOT`)
- gpt-oss-120B (rollback): H100, `transformers serve`, continuous batching, 10-minute scale-down
- gpt-oss-120B (staging): H100, vLLM 0.31.0, `--enable-auto-tool-choice`, 10-minute scale-down
- DeepSeek-V4.1-Flash: H200:8, vLLM 0.31.0, `deepseek_v41` tokenizer/tool/reasoning
  parsers, 30-minute scale-down, 90-minute startup timeout
- Diffusion servers: FastAPI inside `@app.cls`, model loaded once in `@modal.enter()`
- Modal proxy authentication enabled on every endpoint

## Sizing and tuning notes

- **DeepSeek-V4.1-Flash** is much larger than gpt-oss (~476 GiB of weights) and is
  Blackwell-oriented (MXFP4 experts). It must run on ≥8×H200 or a Blackwell node —
  **not** H100. The vLLM config uses tensor parallelism (`--tensor-parallel-size 8`)
  and the `deepseek_v41` parsers. This is set up but has **not been run live**; see
  the audit section above for cost and cold-start expectations.
- **Video generation is slow** (tens of seconds to minutes per clip). Generation
  defaults are conservative; increase `num_frames`, resolution, or steps per request
  as your budget allows.
- The `build_image_app` / `build_video_app` helpers pass only the keyword arguments a
  pipeline's signature accepts, so model-family differences in call signatures are
  handled automatically. If a pipeline needs extra arguments (for example LTX audio
  output or Wan's second guidance scale), add them to the request model and the
  call site.
- GPU types and counts are per-endpoint and can be changed in `modal_models.py`.
