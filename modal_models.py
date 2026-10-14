import base64
import inspect
import io
import logging
import threading
import os
import subprocess
import tempfile
import time

import modal

APP_NAME = "multi-model-endpoints"
HF_HOME = "/root/.cache/huggingface"
PORT = 8000

GPT_OSS = "openai/gpt-oss-120b"
GPT_OSS_CANONICAL = "gpt-oss-120b"
QWEN36 = "Qwen/Qwen3.6-35B-A3B-FP8"
QWEN36_CANONICAL = "Qwen/Qwen3.6-35B-A3B"
DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash"
QWEN_IMAGE = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_EDIT = "Qwen/Qwen-Image-Edit-2511"
LTX = "Lightricks/LTX-2.5-Diffusers"
WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
HUNYUAN = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

VLLM_VERSION = "0.31.0"
VLLM_CACHE = "/root/.cache/vllm"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name(
    "gpt-oss-120b-hf-cache",
    create_if_missing=True,
)

vllm_cache = modal.Volume.from_name(
    "gpt-oss-120b-vllm-cache",
    create_if_missing=True,
)

_ENV = {
    "HF_HOME": HF_HOME,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONUNBUFFERED": "1",
}

diffusion_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "diffusers>=0.36.0",
        "transformers>=5.17.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "fastapi[standard]",
        "uvicorn",
        "huggingface_hub",
    )
    .env(_ENV)
)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub",
    )
    .env(
        {
            **_ENV,
            "VLLM_CACHE_ROOT": VLLM_CACHE,
        }
    )
)


def _serve_vllm():
    subprocess.Popen(
        [
            "vllm",
            "serve",
            GPT_OSS,
            "--served-model-name",
            GPT_OSS_CANONICAL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "openai",
            "--reasoning-parser",
            "openai_gptoss",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "16",
        ]
    )


def _serve_qwen36_vllm():
    # Qwen3.6-35B-A3B: 35B MoE (3B activated) with a vision encoder, hybrid
    # Gated DeltaNet + full attention layers. The FP8 checkpoint is served so a
    # single H100 keeps headroom for the KV cache; BF16 is ~70GB and does not.
    subprocess.Popen(
        [
            "vllm",
            "serve",
            QWEN36,
            "--served-model-name",
            QWEN36_CANONICAL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "16",
        ]
    )


def _serve_deepseek_vllm():
    # DeepSeek-V4.1-Flash: MXFP4 experts + FP8 dense + Engram memory. H100 80GB
    # cannot hold it (no native MXFP4; ~476 GiB weights). Requires >=8xH200 or
    # Blackwell, with the deepseek_v41 tokenizer/tool/reasoning parsers.
    subprocess.Popen(
        [
            "vllm",
            "serve",
            DEEPSEEK,
            "--served-model-name",
            DEEPSEEK,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--trust-remote-code",
            "--tensor-parallel-size",
            "8",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "deepseek_v41",
            "--reasoning-parser",
            "deepseek_v41",
            "--tokenizer-mode",
            "deepseek_v41",
            "--gpu-memory-utilization",
            "0.92",
            "--max-model-len",
            "32768",
        ]
    )


def _call(fn, **kwargs):
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(**kwargs)
    return fn(**{key: value for key, value in kwargs.items() if key in parameters})


def _extract_frames(result):
    frames = getattr(result, "frames", result)
    if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
        frames = frames[0]
    return frames


def build_image_app(pipeline, model_id):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class ImageRequest(BaseModel):
        prompt: str
        negative_prompt: str | None = None
        width: int = 1024
        height: int = 1024
        num_inference_steps: int = 30
        guidance_scale: float = 4.0
        seed: int | None = None

    api = FastAPI(title=model_id)
    generation_lock = threading.Lock()
    logger = logging.getLogger("qwen-image")

    @api.get("/health")
    def health():
        return {"status": "ready", "model": model_id}

    @api.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @api.post("/v1/images/generations")
    def generate(request: ImageRequest):
        import torch
        if not request.prompt.strip():
            raise HTTPException(status_code=422, detail="Prompt must not be empty")
        if request.width < 256 or request.height < 256 or request.width > 2048 or request.height > 2048 or request.width % 16 or request.height % 16:
            raise HTTPException(status_code=422, detail="Image dimensions must be multiples of 16 between 256 and 2048")
        if not 1 <= request.num_inference_steps <= 100:
            raise HTTPException(status_code=422, detail="Inference steps must be between 1 and 100")
        if not generation_lock.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Image worker busy; retry with the same job identifier")
        try:
            return _generate_locked(request, torch)
        finally:
            generation_lock.release()

    def _generate_locked(request, torch):
        started = time.monotonic()
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        result = _call(
            pipeline,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        images = getattr(result, "images", None)
        if not images or images[0] is None:
            raise HTTPException(status_code=502, detail="Image model returned no image")
        image = images[0]
        if not hasattr(image, "save"):
            raise HTTPException(status_code=502, detail="Image model returned invalid image")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        if not buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=502, detail="Image encoding was invalid")
        logger.info("image_generation_succeeded model=%s duration_ms=%d bytes=%d", model_id, int((time.monotonic()-started)*1000), buffer.tell())
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}],
        }

    return api


def build_image_edit_app(pipeline, model_id):
    """Edit an uploaded source image; never fall back to text-to-image generation."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from PIL import Image, UnidentifiedImageError

    class SourceImage(BaseModel):
        mime_type: str
        b64_json: str = Field(min_length=4, max_length=18_000_000)

    class EditRequest(BaseModel):
        model: str = "image-edit"
        prompt: str = Field(min_length=3, max_length=4000)
        image: SourceImage
        response_format: str = "b64_json"

    api = FastAPI(title=model_id)

    @api.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @api.post("/v1/images/edits")
    def edit(request: EditRequest):
        import binascii
        import torch

        if request.response_format != "b64_json" or request.image.mime_type not in (
            "image/png", "image/jpeg", "image/webp"
        ):
            raise HTTPException(status_code=422, detail="Unsupported image format")
        try:
            raw = base64.b64decode(request.image.b64_json, validate=True)
            if len(raw) > 12 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Image too large")
            original = Image.open(io.BytesIO(raw))
            original.verify()
            original = Image.open(io.BytesIO(raw)).convert("RGB")
            if original.width * original.height > 16_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions too large")
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail="Invalid source image") from exc
        result = pipeline(image=original, prompt=request.prompt, num_inference_steps=30)
        output = io.BytesIO()
        result.images[0].save(output, format="PNG")
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}],
        }

    return api


def build_video_app(pipeline, model_id, defaults):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class VideoRequest(BaseModel):
        prompt: str
        negative_prompt: str | None = None
        width: int = defaults["width"]
        height: int = defaults["height"]
        num_frames: int = defaults["num_frames"]
        num_inference_steps: int = defaults["num_inference_steps"]
        guidance_scale: float = defaults["guidance_scale"]
        fps: int = defaults.get("fps", 24)
        seed: int | None = None

    api = FastAPI(title=model_id)

    @api.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @api.post("/v1/videos")
    def generate(request: VideoRequest):
        import torch
        from diffusers.utils import export_to_video

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        result = _call(
            pipeline,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        frames = _extract_frames(result)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            path = handle.name
        export_to_video(frames, path, fps=request.fps)
        with open(path, "rb") as handle:
            video_bytes = handle.read()
        os.unlink(path)

        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(video_bytes).decode()}],
        }

    return api


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache, VLLM_CACHE: vllm_cache},
    timeout=1800,
    scaledown_window=1800,
    min_containers=0,
    max_containers=2,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(
    PORT,
    startup_timeout=1800,
    requires_proxy_auth=True,
)
def gpt_oss_120b_vllm():
    _serve_vllm()


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache, VLLM_CACHE: vllm_cache},
    timeout=1800,
    scaledown_window=1200,
    min_containers=0,
    max_containers=2,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(
    PORT,
    startup_timeout=1800,
    requires_proxy_auth=True,
)
def qwen36_35b_a3b():
    _serve_qwen36_vllm()


@app.function(image=vllm_image, timeout=600)
def vllm_selftest():
    import pathlib

    import vllm

    print("VLLM_VERSION:", vllm.__version__)

    try:
        from vllm.tool_parsers import ToolParserManager

        print("TOOL_PARSERS:", ToolParserManager.list_registered())
    except Exception as exc:  # noqa: BLE001
        print("TOOL_MANAGER_FAILED:", repr(exc))

    try:
        from vllm.reasoning import ReasoningParserManager

        print("REASONING_PARSERS:", ReasoningParserManager.list_registered())
    except Exception as exc:  # noqa: BLE001
        print("REASONING_MANAGER_FAILED:", repr(exc))

    try:
        from vllm.model_executor.models.registry import ModelRegistry

        archs = ModelRegistry.get_supported_archs()
        print("HAS_DeepseekV41ForCausalLM:", "DeepseekV41ForCausalLM" in archs)
        print(
            "HAS_Qwen3_5MoeForConditionalGeneration:",
            "Qwen3_5MoeForConditionalGeneration" in archs,
        )
    except Exception as exc:  # noqa: BLE001
        print("MODEL_REGISTRY_FAILED:", repr(exc))

    try:
        reg = pathlib.Path(vllm.__file__).parent / "tokenizers" / "registry.py"
        lines = reg.read_text(errors="ignore").splitlines()
        modes = [
            line.split(":")[0].strip().strip('"')
            for line in lines
            if line.strip().startswith('"') and '": (' in line
        ]
        print("TOKENIZER_MODES:", modes)
    except Exception as exc:  # noqa: BLE001
        print("TOKENIZER_MODE_PROBE_FAILED:", repr(exc))


@app.function(
    image=vllm_image,
    gpu="H200:8",
    volumes={HF_HOME: hf_cache, VLLM_CACHE: vllm_cache},
    timeout=7200,
    scaledown_window=1800,
 )
@modal.concurrent(max_inputs=8)
@modal.web_server(
    PORT,
    startup_timeout=5400,
    requires_proxy_auth=True,
)
def deepseek_v4_1_flash():
    _serve_deepseek_vllm()


@app.cls(
    image=diffusion_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache},
    timeout=1800,
    scaledown_window=1200,
    min_containers=0,
    max_containers=1,
)
class QwenImage2512:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImagePipeline

        self.pipeline = QwenImagePipeline.from_pretrained(
            QWEN_IMAGE,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return build_image_app(self.pipeline, QWEN_IMAGE)


@app.cls(
    image=diffusion_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache},
    timeout=1800,
    scaledown_window=600,
)
class QwenImageEdit2511:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_IMAGE_EDIT,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return build_image_edit_app(self.pipeline, QWEN_IMAGE_EDIT)


@app.cls(
    image=diffusion_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache},
    timeout=3600,
    scaledown_window=900,
)
class Ltx25:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import LTX2Pipeline

        self.pipeline = LTX2Pipeline.from_pretrained(
            LTX,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return build_video_app(
            self.pipeline,
            LTX,
            {
                "width": 768,
                "height": 512,
                "num_frames": 121,
                "num_inference_steps": 30,
                "guidance_scale": 3.0,
            },
        )


@app.cls(
    image=diffusion_image,
    gpu="L40S",
    volumes={HF_HOME: hf_cache},
    timeout=3600,
    scaledown_window=900,
)
class Wan22:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanPipeline

        self.pipeline = WanPipeline.from_pretrained(
            WAN,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return build_video_app(
            self.pipeline,
            WAN,
            {
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_scale": 5.0,
            },
        )


@app.cls(
    image=diffusion_image,
    gpu="H100",
    volumes={HF_HOME: hf_cache},
    timeout=3600,
    scaledown_window=900,
)
class HunyuanVideo15:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import HunyuanVideo15Pipeline

        self.pipeline = HunyuanVideo15Pipeline.from_pretrained(
            HUNYUAN,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return build_video_app(
            self.pipeline,
            HUNYUAN,
            {
                "width": 854,
                "height": 480,
                "num_frames": 121,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
            },
        )
