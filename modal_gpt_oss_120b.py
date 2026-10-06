import subprocess

import modal

APP_NAME = "gpt-oss-120b-transformers"
MODEL_ID = "openai/gpt-oss-120b"
PORT = 8000

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name(
    "gpt-oss-120b-hf-cache",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers[serving]>=5.17.0",
        "accelerate",
        "kernels",
        "triton>=3.4",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1800,
    scaledown_window=600,
    allow_concurrent_inputs=8,
)
@modal.web_server(
    PORT,
    startup_timeout=1800,
    requires_proxy_auth=True,
)
def transformers_server():
    subprocess.Popen(
        [
            "transformers",
            "serve",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--continuous-batching",
            "--dtype",
            "bfloat16",
            "--model-timeout",
            "-1",
        ]
    )
