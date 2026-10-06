import os

from openai import OpenAI

base_url = os.environ["GPT_OSS_BASE_URL"].rstrip("/") + "/v1"
modal_proxy_token = os.environ["MODAL_PROXY_TOKEN"]

client = OpenAI(
    base_url=base_url,
    api_key="unused",
    default_headers={
        "Authorization": f"Bearer {modal_proxy_token}",
    },
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain why the sky is blue in three sentences.",
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
