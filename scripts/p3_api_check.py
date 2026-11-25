"""P3 Step 1: check the shared Qwen endpoint and learn what it supports (VS-P3-01).

Run from the repo root after filling QWEN_* in .env:
    python scripts/p3_api_check.py

1. raw chat completion (default request)
2. thinking disabled (chat_template_kwargs.enable_thinking=false, what QwenClient sends)
3. response_format json_object (is server-side JSON mode supported?)
4. QwenClient.complete() smoke test (the code path Planner/Critic use)

Never prints your API key.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from verisheet.config import get_settings  # noqa: E402
from verisheet.qwen_client import QwenClient, QwenError, _resolve_path, _secret  # noqa: E402

PROMPT = 'Reply ONLY with this JSON, computing n: {"ok": true, "n": 2+3}'


def main() -> int:
    settings = get_settings()
    base = settings.qwen_api_base.rstrip("/")
    key = _secret(settings.qwen_api_key) or ""
    if not base or not key or key == "REPLACE_ME" or "YOUR-" in base:
        print("QWEN_API_BASE / QWEN_API_KEY are not set. Copy .env.example to .env and fill them in.")
        return 1
    verify = True
    if settings.qwen_ca_cert:
        import ssl

        cert = _resolve_path(settings.qwen_ca_cert)
        if not cert.is_file():
            print(f"QWEN_CA_CERT not found: {cert}")
            return 1
        verify = ssl.create_default_context(cafile=str(cert))
    client = httpx.Client(verify=verify, headers={"Authorization": f"Bearer {key}"}, timeout=120)
    print(f"endpoint: {base}   model: {settings.qwen_model}")

    def call(label: str, extra: dict | None = None) -> None:
        body = {"model": settings.qwen_model, "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 200, "temperature": 0, **(extra or {})}
        print(f"\n--- {label} ---")
        started = time.time()
        try:
            response = client.post(f"{base}/chat/completions", json=body)
        except httpx.HTTPError as exc:  # certificate / network problems
            print("ERROR:", type(exc).__name__, str(exc)[:200])
            return
        print("status:", response.status_code, f"| {time.time() - started:.1f}s")
        print(response.text[:700].replace(key, "***"))

    call("1. default")
    call("2. thinking off", {"chat_template_kwargs": {"enable_thinking": False}})
    call("3. response_format json_object", {"response_format": {"type": "json_object"}})

    print("\n--- 4. QwenClient.complete() ---")
    try:
        reply = QwenClient().complete("You answer with JSON only.", PROMPT, max_tokens=100)
        print(f"content={reply.content!r} model={reply.model} tokens in/out="
              f"{reply.input_tokens}/{reply.output_tokens} latency={reply.latency_ms:.0f}ms "
              f"finish={reply.finish_reason}")
    except (QwenError, ValueError) as exc:
        print("ERROR:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
