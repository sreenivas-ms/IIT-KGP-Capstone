"""Append-only structured trace logging owned by Person 6; consumed by the Orchestrator.

One JSONL file per trace ID (``<root>/<trace_id>.jsonl``), one validated ``TraceEvent`` per
line: the same layout as the Orchestrator's fallback sink, so every trace reads the same way.
Before anything is written, summaries and metadata are redacted (API keys, bearer tokens,
e-mail addresses, phone numbers) and bounded in length.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verisheet.schemas import TraceEvent

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 2000
MAX_METADATA_STRING_CHARS = 4000
MAX_METADATA_DEPTH = 6
REDACTED_SECRET = "[REDACTED:secret]"

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"),                                   # OpenAI-style keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),                         # auth headers
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\",}]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                       # AWS access keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def safe_trace_filename(trace_id: str) -> str:
    """File stem for a trace ID: identical rule to the Orchestrator's fallback sink."""
    return "".join(ch for ch in trace_id if ch.isalnum() or ch in "-_")[:80] or "trace"


def _configured_secrets() -> list[str]:
    """Exact values of configured API keys, so they are redacted even without a known prefix."""
    try:
        from verisheet.config import get_settings

        settings = get_settings()
    except Exception:  # pragma: no cover - config unavailable
        return []
    values = []
    for name in ("qwen_api_key", "embedding_api_key"):
        value = getattr(settings, name, "")
        value = value.get_secret_value() if hasattr(value, "get_secret_value") else value
        if isinstance(value, str) and len(value) >= 8 and value != "REPLACE_ME":
            values.append(value)
    return values


def redact_text(text: str, max_chars: int | None = MAX_SUMMARY_CHARS,
                secrets: list[str] | None = None) -> str:
    """Mask secrets and personal data in free text, then bound its length."""
    if not text:
        return text
    for secret in secrets if secrets is not None else _configured_secrets():
        text = text.replace(secret, REDACTED_SECRET)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED_SECRET, text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _mask_phones(text)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"…[truncated {len(text) - max_chars} chars]"
    return text


def _mask_phones(text: str) -> str:
    try:  # reuse the profiler's phone detector (digit-count and not-a-date rules)
        from verisheet.profile import _PHONE_CANDIDATE_RE, _is_phone_match

        return _PHONE_CANDIDATE_RE.sub(
            lambda m: "[PHONE]" if _is_phone_match(m.group(0)) else m.group(0), text)
    except ImportError:  # pragma: no cover
        return text


def redact_value(value: Any, secrets: list[str], depth: int = 0) -> Any:
    """Recursively redact strings inside metadata (dict keys are kept, values redacted)."""
    if depth > MAX_METADATA_DEPTH:
        return "[TRUNCATED:depth]"
    if isinstance(value, str):
        return redact_text(value, MAX_METADATA_STRING_CHARS, secrets)
    if isinstance(value, dict):
        return {str(k): redact_value(v, secrets, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v, secrets, depth + 1) for v in value]
    return value


class TraceLogger:
    """Persist trace events as JSONL, one event per line, indexed by trace ID.

    Required fields include actor/event type, model/prompt version, tool/step status, token
    counts, latency, generated artifact references, verification verdict and failure category.
    Store bounded summaries and artifact paths/hashes rather than raw PII or whole workbooks.
    """

    def __init__(self, root_dir: str = "results/traces"):
        """Configure trace output directory and retention policy.

        The directory is created on first write. Files are append-only; nothing is deleted.
        """
        self.root_dir = Path(root_dir)
        self._lock = threading.Lock()
        self._secrets = _configured_secrets()

    # -- writing -----------------------------------------------------------------------------

    def log_event(self, event: TraceEvent) -> None:
        """Append and flush one validated TraceEvent to the trace file for its trace ID.

        Accepts a ``TraceEvent`` or an equivalent dict. Raises ``ValueError`` for invalid events.
        """
        if not isinstance(event, TraceEvent):
            event = TraceEvent.model_validate(event)
        clean = self.redact(event)
        line = clean.model_dump_json()
        path = self.trace_path(clean.trace_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def redact(self, event: TraceEvent) -> TraceEvent:
        """Return a copy with summaries/metadata redacted and bounded."""
        return event.model_copy(update={
            "input_summary": redact_text(event.input_summary, MAX_SUMMARY_CHARS, self._secrets),
            "output_summary": redact_text(event.output_summary, MAX_SUMMARY_CHARS, self._secrets),
            "metadata": redact_value(event.metadata, self._secrets),
        })

    # -- reading -----------------------------------------------------------------------------

    def trace_path(self, trace_id: str) -> Path:
        return self.root_dir / f"{safe_trace_filename(trace_id)}.jsonl"

    def load_trace(self, trace_id: str) -> list[TraceEvent]:
        """Load and validate all events belonging to a trace ID in timestamp order.

        Raises:
            FileNotFoundError: no trace file for this ID.
        Malformed lines are skipped (and logged) rather than hiding the rest of the trace.
        """
        path = self.trace_path(trace_id)
        if not path.is_file():
            raise FileNotFoundError(f"No trace found for '{safe_trace_filename(trace_id)}' in {self.root_dir}")
        events: list[tuple[int, TraceEvent]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                event = TraceEvent.model_validate_json(line)
            except ValidationError:
                logger.warning("skipping malformed trace line %d in %s", number + 1, path.name)
                continue
            if event.trace_id == trace_id:
                events.append((number, event))
        events.sort(key=lambda pair: (_as_utc(pair[1].timestamp), pair[0]))
        return [event for _, event in events]

    def list_traces(self, limit: int = 200) -> list[dict[str, Any]]:
        """Newest-first index of traces: id, time, event count, final status, question summary."""
        if not self.root_dir.is_dir():
            return []
        files = sorted(self.root_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        index = []
        for path in files[:limit]:
            first, final, count = None, None, 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                first = first or record
                if record.get("event_type") == "final_response":
                    final = record          # later events (e.g. experiment_config) may follow it
            if not first:
                continue
            index.append({
                "trace_id": first.get("trace_id", path.stem),
                "started": first.get("timestamp"),
                "events": count,
                "status": final.get("status") if final else "IN_PROGRESS/UNKNOWN",
                "question": (first.get("input_summary") or "")[:120],
                "session_id": (first.get("metadata") or {}).get("session_id"),
            })
        return index


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
