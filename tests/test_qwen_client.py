"""Tests for the shared Qwen client (VS-P3-01). Offline: httpx.MockTransport."""
import json

import httpx
import pytest

from verisheet.config import Settings
from verisheet.qwen_client import (
    QwenClient,
    QwenConfigError,
    QwenError,
    extract_json,
    strip_think,
)
from verisheet.schemas import LLMResponse


def _reply(content: str, status: int = 200, finish: str = "stop"):
    body = {"model": "Qwen3-8B", "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return httpx.Response(status, json=body)


def make_client(handler, key="k"):
    return QwenClient(base_url="https://example.test/v1", api_key=key, model="Qwen3-8B",
                      transport=httpx.MockTransport(handler), sleep=lambda s: None)


# ---- text helpers ---------------------------------------------------------
def test_strip_think_closed_and_unclosed():
    assert strip_think("<think>hmm</think>{\"a\":1}") == '{"a":1}'
    assert strip_think('{"a":1}<think>cut off') == '{"a":1}'


def test_extract_json_handles_fences_prose_and_braces_in_strings():
    txt = 'Sure!\n```json\n{"a": {"b": "x}y"}}\n```\nDone'
    assert json.loads(extract_json(txt)) == {"a": {"b": "x}y"}}


def test_extract_json_truncated_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": [1, 2')


# ---- complete() -----------------------------------------------------------
def test_complete_returns_normalised_llmresponse_and_request_shape():
    seen = {}

    def handler(req):
        seen["body"], seen["auth"] = json.loads(req.content), req.headers["authorization"]
        return _reply("<think>x</think>hello")

    r = make_client(handler).complete("SYS", "USR", max_tokens=5000)
    assert isinstance(r, LLMResponse)
    assert (r.content, r.model, r.input_tokens, r.output_tokens) == ("hello", "Qwen3-8B", 10, 5)
    assert r.latency_ms >= 0 and r.finish_reason == "stop"
    b = seen["body"]
    assert b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert b["max_tokens"] == 1024 and b["temperature"] == 0.0
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["auth"] == "Bearer k"


def test_complete_default_max_tokens_is_large_enough_for_plans():
    seen = {}
    make_client(lambda r: (seen.update(b=json.loads(r.content)), _reply("ok"))[1]).complete("s", "u")
    assert seen["b"]["max_tokens"] == 900


def test_finish_reason_length_is_reported():
    assert make_client(lambda r: _reply("{", finish="length")).complete("s", "u").finish_reason == "length"


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] < 3 else _reply("ok")

    c = make_client(handler)
    assert c.complete("s", "u").content == "ok" and c.last_attempts == 3


def test_persistent_429_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        make_client(lambda r: httpx.Response(429)).complete("s", "u")


def test_400_and_401_are_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(QwenError):
        make_client(handler).complete("s", "u")
    assert calls["n"] == 1


def test_unexpected_response_shape_raises():
    with pytest.raises(QwenError, match="unexpected response"):
        make_client(lambda r: httpx.Response(200, json={"nope": 1})).complete("s", "u")


# ---- configuration and secrets (acceptance criteria) ----------------------
@pytest.mark.parametrize("base, key, match", [
    ("", "k", "QWEN_API_BASE"),
    ("https://x.test/v1", "", "QWEN_API_KEY"),
    ("https://YOUR-QWEN-ENDPOINT/v1", "k", "QWEN_API_BASE"),
    ("https://x.test/v1", "REPLACE_ME", "QWEN_API_KEY"),
])
def test_missing_or_placeholder_credentials_raise_valueerror(base, key, match):
    cfg = Settings(_env_file=None, qwen_api_base=base, qwen_api_key=key)
    with pytest.raises(ValueError, match=match) as info:      # stub documents ValueError
        QwenClient(settings=cfg)
    assert isinstance(info.value, QwenConfigError)


def test_defaults_come_from_settings():
    cfg = Settings(_env_file=None, qwen_api_base="https://x.test/v1/", qwen_api_key="k")
    c = QwenClient(settings=cfg)
    assert c.base_url == "https://x.test/v1" and c.model == "Qwen/Qwen3-8B"


def test_missing_ca_cert_file_is_rejected(tmp_path):
    with pytest.raises(QwenConfigError, match="CA certificate not found"):
        QwenClient(base_url="https://x.test/v1", api_key="k", ca_cert=str(tmp_path / "no.crt"))


def test_api_key_never_appears_in_errors():
    secret = "sk-super-secret-key"
    c = make_client(lambda r: httpx.Response(400, text=f"bad {secret}"), key=secret)
    with pytest.raises(QwenError) as info:
        c.complete("s", "u")
    assert secret not in str(info.value)


def test_settings_hide_the_api_key_from_repr_and_logs():
    secret = "sk-hidden-qwen-key"
    cfg = Settings(_env_file=None, qwen_api_base="https://x.test/v1", qwen_api_key=secret)
    assert secret not in repr(cfg) and secret not in str(cfg.model_dump())
    assert QwenClient(settings=cfg)._api_key == secret


def test_ca_cert_from_settings_is_used(tmp_path):
    cfg = Settings(_env_file=None, qwen_api_base="https://x.test/v1", qwen_api_key="k",
                   qwen_ca_cert=str(tmp_path / "qwen-ca.crt"))
    with pytest.raises(QwenConfigError, match="qwen-ca.crt"):
        QwenClient(settings=cfg)
