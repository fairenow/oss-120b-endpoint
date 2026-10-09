"""Verify the vLLM-backed GPT-OSS staging endpoint.

Stdlib only (urllib/json) so it runs with the system python3.

Usage:
    source env.local
    export GPT_OSS_VLLM_BASE_URL="https://ramon-williams-jr--multi-model-endpoints-gpt-oss-120b-vllm.modal.run"
    python3 verify_gpt_oss_vllm.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("GPT_OSS_VLLM_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("MODAL_PROXY_TOKEN", "")
MODEL = os.environ.get("GPT_OSS_MODEL_ID", "gpt-oss-120b")

if not BASE or not TOKEN:
    print("FAIL: set GPT_OSS_VLLM_BASE_URL and MODAL_PROXY_TOKEN")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

ALARM_TOOL = {
    "type": "function",
    "function": {
        "name": "set_alarm",
        "description": "Set an alarm at a given hour of the day.",
        "parameters": {
            "type": "object",
            "properties": {"hour": {"type": "integer"}},
            "required": ["hour"],
        },
    },
}

LEAK_MARKERS = ("analysis", "assistantfinal", "assistantcommentary", "<|")
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def post(body, stream=False, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=data, headers=HEADERS, method="POST"
    )
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        raw = resp.read()
        return json.loads(raw), start, time.time() - start

    chunks = []

    def iterator():
        for line in resp:
            line = line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
            yield payload

    first_seen = {"t": None}
    for _ in iterator():
        if first_seen["t"] is None:
            first_seen["t"] = time.time()
    return chunks, start, (first_seen["t"] or time.time()) - start


def warm_up(max_wait=1200):
    print(f"warming up {BASE} ...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            req = urllib.request.Request(f"{BASE}/v1/models", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.load(resp)
            print(f"warm after {time.time() - t0:.1f}s: {body}")
            return time.time() - t0, body
        except urllib.error.HTTPError as exc:
            if exc.code in (303, 302, 307, 308):
                loc = exc.headers.get("Location", "")
                if loc:
                    try:
                        req2 = urllib.request.Request(loc, headers=HEADERS)
                        urllib.request.urlopen(req2, timeout=60).read()
                    except Exception:  # noqa: BLE001
                        pass
                continue
            print("  warm retry", exc.code)
            time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            print("  warm retry", repr(exc))
            time.sleep(5)
    raise TimeoutError("endpoint did not become ready")


def build_tool_calls(chunks):
    acc = {}
    finish = None
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    return [acc[i] for i in sorted(acc)], finish


def check_no_leak(text):
    if not text:
        return True, ""
    lowered = text
    hits = [m for m in LEAK_MARKERS if m in lowered]
    return (not hits), f"leak markers: {hits}" if hits else ""


def parse_args(raw):
    try:
        return json.loads(raw), ""
    except json.JSONDecodeError as exc:
        return None, repr(exc)


def main():
    cold, _ = warm_up()

    # 1. plain chat, no harmony leakage
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 512,
    }
    resp, _, dt = post(body)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    ok_leak, leak_detail = check_no_leak(content)
    record(
        "plain_chat_clean",
        ok_leak and "pong" in content,
        f"content={content!r} model={resp.get('model')} {leak_detail} ({dt:.2f}s)",
    )

    # 1b. reasoning-heavy prompt: chain-of-thought must not leak into content
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "What is 17 * 23? Think step by step, then answer with just the number.",
            }
        ],
        "max_tokens": 1024,
    }
    resp, _, dt = post(body)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content")
    ok_leak, leak_detail = check_no_leak(content)
    record(
        "reasoning_prompt_clean",
        ok_leak and "391" in content,
        f"content={content!r} has_reasoning_content={reasoning is not None} "
        f"reasoning_len={len(reasoning or '')} {leak_detail} ({dt:.2f}s)",
    )

    # 2. model id normalization
    record("model_id_normalized", resp.get("model") == MODEL, f"model={resp.get('model')!r}")

    # 3. non-streaming tool call
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 512,
    }
    resp, _, dt = post(body)
    choice = resp["choices"][0]
    tcs = choice["message"].get("tool_calls") or []
    finish = choice.get("finish_reason")
    detail = f"finish={finish} n_tool_calls={len(tcs)} ({dt:.2f}s)"
    ok = finish == "tool_calls" and len(tcs) >= 1
    if ok:
        args, err = parse_args(tcs[0]["function"]["arguments"])
        ok = (
            tcs[0]["function"]["name"] == "get_weather"
            and args is not None
            and "city" in args
            and bool(tcs[0].get("id"))
        )
        detail += f" name={tcs[0]['function']['name']} args={args} id={tcs[0].get('id')} err={err}"
    record("tool_call_nonstreaming", ok, detail)

    # 4. streaming tool call
    resp_s, _, dt_s = post({**body, "stream": True}, stream=True)
    stcs, sfinish = build_tool_calls(resp_s)
    detail = f"finish={sfinish} n_tool_calls={len(stcs)} ({dt_s:.2f}s)"
    ok = sfinish == "tool_calls" and len(stcs) >= 1
    if ok:
        args, err = parse_args(stcs[0]["arguments"])
        ok = stcs[0]["name"] == "get_weather" and args is not None and "city" in args
        detail += f" name={stcs[0]['name']} args={args} id={stcs[0]['id']} err={err}"
    record("tool_call_streaming", ok, detail)

    # 5. full round trip using the streaming tool call
    call = stcs[0]
    roundtrip = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "The weather in Paris is 18C and sunny.",
            },
        ],
        "tools": [WEATHER_TOOL],
        "max_tokens": 512,
    }
    resp, _, dt = post(roundtrip)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    finish = resp["choices"][0].get("finish_reason")
    ok_leak, leak_detail = check_no_leak(content)
    ok = (
        finish == "stop"
        and not msg.get("tool_calls")
        and "18" in content
        and ok_leak
    )
    record(
        "tool_result_round_trip",
        ok,
        f"finish={finish} content={content!r} {leak_detail} ({dt:.2f}s)",
    )

    # 6. parallel tool calls
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the get_weather tool for Paris and use the get_weather tool "
                    "for Tokyo. You must make both tool calls together in the same "
                    "response."
                ),
            }
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 512,
        "parallel_tool_calls": True,
    }
    resp, _, dt = post(body)
    choice = resp["choices"][0]
    tcs = choice["message"].get("tool_calls") or []
    cities = []
    for tc in tcs:
        args, _ = parse_args(tc["function"]["arguments"])
        if isinstance(args, dict):
            cities.append(args.get("city"))
    ok = choice.get("finish_reason") == "tool_calls" and len(tcs) >= 2
    record(
        "parallel_tool_calls",
        ok,
        f"n={len(tcs)} cities={cities} finish={choice.get('finish_reason')} ({dt:.2f}s)",
    )

    # 7. malformed argument handling
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Set an alarm for tomorrow at noon. The hour must be an integer."}
        ],
        "tools": [ALARM_TOOL],
        "tool_choice": "auto",
        "max_tokens": 512,
    }
    resp, _, dt = post(body)
    choice = resp["choices"][0]
    tcs = choice["message"].get("tool_calls") or []
    detail = f"finish={choice.get('finish_reason')} ({dt:.2f}s)"
    ok = True
    if tcs:
        raw = tcs[0]["function"]["arguments"]
        args, err = parse_args(raw)
        detail += f" args={raw!r} parsed={args} int_ok={isinstance((args or {}).get('hour'), int)} err={err}"
        ok = args is not None
    else:
        detail += f" no tool call; content={(choice['message'].get('content') or '')!r}"
    record("malformed_arguments_handled", ok, detail)

    print()
    print(f"cold_start_seconds: {cold:.1f}")
    failed = [r for r in results if not r[1]]
    print(f"summary: {len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILURES:", [r[0] for r in failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
