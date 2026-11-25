"""Shared OpenAI-compatible Qwen3-8B client used by Planner and semantic Critic (P3).

Public contract (unchanged from the repo stub):
    QwenClient(timeout_seconds=45).complete(system_prompt, user_prompt,
                                            temperature=0.0, max_tokens=None) -> LLMResponse

Behaviour required by the Track B API guide:
  * ONE active request per group key      -> process-wide lock
  * 8192 tokens in+out, max_tokens <= 1024 -> clamp; default 900 so plans are not cut off
  * Qwen3 <think> blocks eat the output cap -> thinking disabled in the request + stripped
  * Private CA                              -> QWEN_CA_CERT (never verify=False)
  * 429/503 under load                      -> retry 2s, 4s, 8s
The API key never appears in errors or logs.
"""
from __future__ import annotations

import os
import re
import ssl
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from verisheet.schemas import LLMResponse

_REQUEST_LOCK = threading.Lock()
RETRY_STATUS = {429, 503}
BACKOFF_SECONDS = (2, 4, 8)
MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_TOKENS = 900


class QwenError(RuntimeError):
    """Endpoint communication or response parsing failed (the stub's documented RuntimeError)."""


class QwenConfigError(QwenError, ValueError):
    """QWEN_* settings missing, placeholder, or invalid (the stub's documented ValueError)."""


# --------------------------------------------------------------------------- #
# Text helpers (also used by Critic for JSON replies)
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think> blocks, including a truncated, unclosed one."""
    text = _THINK_RE.sub("", text)
    i = text.lower().find("<think>")
    if i != -1:
        text = text[:i]
    return text.strip()


def extract_json(text: str) -> str:
    """Return the first balanced {...} object in text (tolerates fences and prose).

    Raises:
        ValueError: no object found, or the object is incomplete (truncated output).
    """
    text = _FENCE_RE.sub("", strip_think(text).strip())
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("JSON object is incomplete (output likely truncated)")


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _load_settings():
    try:
        from verisheet.config import get_settings
        return get_settings()
    except ImportError:
        return None


def _env(name: str) -> str | None:
    """Environment variable, falling back to .env (without mutating os.environ)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        from dotenv import dotenv_values
        return dotenv_values(".env").get(name) or None
    except ImportError:
        return None


def _secret(value: Any) -> str | None:
    """Unwrap pydantic SecretStr (Settings.qwen_api_key) or return the plain string."""
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def _resolve_path(value: str) -> Path:
    """Expand ~; resolve a relative path from the CWD, else from the project root."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parents[2] / path


def _require(value: str | None, name: str) -> str:
    if not value or value.strip() == "REPLACE_ME" or "YOUR-" in value:
        raise QwenConfigError(
            f"{name} is not configured. Copy .env.example to .env and set {name}.")
    return value.strip()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class QwenClient:
    """Thin, shared API wrapper; roles must supply separate system prompts."""

    def __init__(
        self,
        timeout_seconds: int = 45,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        ca_cert: str | None = None,
        disable_thinking: bool = True,
        settings: Any = None,                      # verisheet.config.Settings (tests)
        transport: httpx.BaseTransport | None = None,   # tests
        sleep=time.sleep,                                  # tests
    ) -> None:
        """Create a client from Settings (QWEN_API_BASE/KEY/MODEL, read from env or .env).

        Optional: QWEN_CA_CERT (private CA path; ~ expanded, relative paths resolve from the
        project root) and QWEN_TIMEOUT_SECONDS (overrides timeout_seconds).

        Raises:
            QwenConfigError (a ValueError): missing/placeholder credentials or bad CA path.
        """
        cfg = settings if settings is not None else _load_settings()
        self.base_url = _require(
            base_url or (cfg.qwen_api_base if cfg else os.getenv("QWEN_API_BASE")),
            "QWEN_API_BASE").rstrip("/")
        self._api_key = _require(
            api_key or _secret(cfg.qwen_api_key if cfg else os.getenv("QWEN_API_KEY")),
            "QWEN_API_KEY")
        self.model = model or (cfg.qwen_model if cfg else os.getenv("QWEN_MODEL")) or "Qwen/Qwen3-8B"
        cert = ca_cert or (getattr(cfg, "qwen_ca_cert", "") if cfg else "") or _env("QWEN_CA_CERT")
        verify: Any = True                                   # never disabled
        if cert:
            cert_path = _resolve_path(cert)
            if not cert_path.is_file():
                raise QwenConfigError(f"QWEN_CA_CERT: CA certificate not found: {cert_path}")
            verify = ssl.create_default_context(cafile=str(cert_path))
        timeout = (_env("QWEN_TIMEOUT_SECONDS")
                   or (getattr(cfg, "qwen_timeout_seconds", None) if cfg else None)
                   or timeout_seconds)
        self.disable_thinking = disable_thinking
        self.last_attempts = 0
        self._sleep = sleep
        self._http = httpx.Client(
            verify=verify,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=float(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call Qwen chat completions and return a normalised response with usage data.

        Args:
            system_prompt: Versioned role instructions (Planner or Critic).
            user_prompt: Bounded, sanitised task context serialized as text/JSON.
            temperature: Use 0 for reproducible planning/critique.
            max_tokens: Completion cap; default 900, clamped to the API maximum 1024.
        Returns:
            LLMResponse: content (think-blocks removed), model, token counts, latency_ms,
            finish_reason ("length" means the output was cut off).
        Raises:
            QwenError (a RuntimeError): endpoint failure after retries, or bad response shape.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max(1, min(int(max_tokens or DEFAULT_MAX_TOKENS), MAX_OUTPUT_TOKENS)),
        }
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.base_url}/chat/completions"
        attempts, last_err = 0, ""
        started = time.perf_counter()
        with _REQUEST_LOCK:                              # serialise all calls
            for delay in (0, *BACKOFF_SECONDS):
                if delay:
                    self._sleep(delay)
                attempts += 1
                try:
                    resp = self._http.post(url, json=body)
                except httpx.TransportError as exc:      # timeouts, connection errors
                    last_err = self._redact(f"transport error: {type(exc).__name__}")
                    continue
                if resp.status_code in RETRY_STATUS:
                    last_err = f"HTTP {resp.status_code}"
                    continue
                if resp.status_code >= 400:              # 400/401/... never retried
                    raise QwenError(self._redact(
                        f"Qwen completion failed: HTTP {resp.status_code}: {resp.text[:200]}"))
                break
            else:
                self.last_attempts = attempts
                raise QwenError(f"Qwen completion failed after {attempts} attempts: {last_err}")
        self.last_attempts = attempts
        elapsed_ms = (time.perf_counter() - started) * 1000

        try:
            data = resp.json()
            choice = data["choices"][0]
            raw = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise QwenError("Qwen completion failed: unexpected response shape") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            content=strip_think(raw),
            model=data.get("model") or self.model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=elapsed_ms,
            finish_reason=choice.get("finish_reason"),
        )
