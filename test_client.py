import base64
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

PROXY_TOKEN = os.environ["MODAL_PROXY_TOKEN"]
HEADERS = {"Authorization": f"Bearer {PROXY_TOKEN}"}


def text_client(env_var):
    base_url = os.environ[env_var].rstrip("/") + "/v1"
    return OpenAI(
        base_url=base_url,
        api_key="unused",
        default_headers=HEADERS,
    )


def test_text(env_var, model_id):
    response = text_client(env_var).chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": "Explain why the sky is blue in three sentences.",
            }
        ],
        max_tokens=300,
    )
    print(response.choices[0].message.content)


def test_image(env_var, output="out.png"):
    url = os.environ[env_var].rstrip("/") + "/v1/images/generations"
    response = requests.post(
        url,
        headers=HEADERS,
        json={
            "prompt": "A red panda surfing a wave at sunset, cinematic lighting",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "seed": 42,
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    Path(output).write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
    print(f"wrote {output}")


def test_video(env_var, output="out.mp4"):
    url = os.environ[env_var].rstrip("/") + "/v1/videos"
    response = requests.post(
        url,
        headers=HEADERS,
        json={
            "prompt": "A drone shot flying over a snowy mountain range at sunrise",
            "seed": 42,
        },
        timeout=1800,
    )
    response.raise_for_status()
    payload = response.json()
    Path(output).write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
    print(f"wrote {output}")


EXAMPLES = {
    "gpt-oss": lambda: test_text("GPT_OSS_BASE_URL", "openai/gpt-oss-120b"),
    "deepseek": lambda: test_text("DEEPSEEK_BASE_URL", "deepseek-ai/DeepSeek-V4.1-Flash"),
    "image": lambda: test_image("QWEN_IMAGE_BASE_URL"),
    "ltx": lambda: test_video("LTX_BASE_URL", "ltx.mp4"),
    "wan": lambda: test_video("WAN_BASE_URL", "wan.mp4"),
    "hunyuan": lambda: test_video("HUNYUAN_BASE_URL", "hunyuan.mp4"),
}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss"
    if target not in EXAMPLES:
        raise SystemExit(f"usage: python test_client.py [{'|'.join(EXAMPLES)}]")
    EXAMPLES[target]()
