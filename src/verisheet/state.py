"""Session/workbook state management owned by Person 5.

State is persisted as one JSON file per (session ID, workbook fingerprint) pair, so:

* follow-up questions on the same workbook see the previously *verified* metric, period,
  filters and grouping;
* uploading a different workbook in the same session starts from a clean, isolated context
  (and switching back to the first workbook restores its context);
* only VERIFIED answers ever update accepted analytical facts. Unverified, failed or
  clarification turns are kept as short, bounded history summaries flagged as unverified.

Files are written atomically (temp file + ``os.replace``) and file names are derived from a
hash of the IDs, so a hostile session ID cannot escape the state directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verisheet.schemas import AgentState, AnalysisPlan, AnswerResponse, AnswerStatus, StateMessage

logger = logging.getLogger(__name__)

MAX_HISTORY = 20
MAX_SUMMARY_CHARS = 400
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def summarise_text(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Return a bounded, single-line summary safe to persist and to show to the Planner."""
    cleaned = _CONTROL_CHARS.sub(" ", str(text or "")).replace("\n", " ").strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


class StateStore:
    """Persist per-session context keyed by session ID and workbook fingerprint.

    Store only verified metric/period/filter context as accepted facts. Preserve enough recent
    turn summaries for multi-turn follow-up, but do not persist unverified answer text as truth.
    """

    def __init__(self, root_dir: str = "artifacts/sessions"):
        """Create the state directory (mode 0700 where supported) for JSON persistence."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root_dir, 0o700)
        except OSError:  # pragma: no cover - e.g. Windows or a read-only parent mount
            pass
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------------------

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """Return the session ID if it is a short token; raise ValueError otherwise."""
        if not isinstance(session_id, str) or not _SESSION_RE.match(session_id):
            raise ValueError("session_id must be 1-128 characters of letters, digits, '_', '-', '.' or ':'")
        return session_id

    def _path(self, session_id: str, workbook_id: str | None) -> Path:
        digest = hashlib.sha256(f"{session_id}\x00{workbook_id or '-'}".encode("utf-8")).hexdigest()[:32]
        return self.root_dir / f"state_{digest}.json"

    # ------------------------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------------------------

    def load(self, session_id: str, workbook_id: str | None = None) -> AgentState:
        """Load state and ensure the workbook fingerprint matches the active workbook context.

        Returns a fresh ``AgentState`` when nothing is stored, when the stored file is
        corrupt (it is quarantined as ``*.corrupt``), or when the stored workbook ID differs.
        """
        self.validate_session_id(session_id)
        path = self._path(session_id, workbook_id)
        fresh = AgentState(session_id=session_id, workbook_id=workbook_id)
        if not path.is_file():
            return fresh
        try:
            state = AgentState.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            logger.warning("state file for session is unreadable; starting fresh")
            try:
                path.replace(path.with_suffix(".corrupt"))
            except OSError:
                pass
            return fresh
        if state.session_id != session_id or state.workbook_id != workbook_id:
            return fresh  # never leak context across sessions/workbooks
        return state

    def save(self, state: AgentState) -> None:
        """Persist validated AgentState atomically to avoid partial/corrupt state files."""
        self.validate_session_id(state.session_id)
        validated = AgentState.model_validate(state.model_dump())
        payload = validated.model_dump_json(indent=2)
        path = self._path(validated.session_id, validated.workbook_id)
        with self._lock:
            fd, tmp = tempfile.mkstemp(prefix=".state_", suffix=".tmp", dir=self.root_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    def reset(self, session_id: str, workbook_id: str | None = None) -> AgentState:
        """Drop stored context for a session/workbook pair and return a fresh state."""
        self.validate_session_id(session_id)
        path = self._path(session_id, workbook_id)
        if path.exists():
            path.unlink()
        return AgentState(session_id=session_id, workbook_id=workbook_id)

    # ------------------------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------------------------

    @staticmethod
    def append_history(state: AgentState, role: str, text: str) -> AgentState:
        """Return a copy of ``state`` with one bounded turn summary appended."""
        history = list(state.history) + [StateMessage(role=role, content_summary=summarise_text(text))]
        return state.model_copy(update={"history": history[-MAX_HISTORY:]})

    def update_from_verified_response(self, state: AgentState, response: AnswerResponse, plan) -> AgentState:
        """Update reusable metric/period/filter state only when response is VERIFIED.

        For any other status the accepted facts are left untouched; only a short history entry
        marked as unverified is added so the Planner knows the turn happened.

        Args:
            state: current state for this session/workbook.
            response: terminal AnswerResponse of the turn.
            plan: the AnalysisPlan that produced ``response`` (may be None for early exits).
        Returns:
            The updated AgentState (not yet saved; call ``save``).
        """
        if response.workbook_id and state.workbook_id and response.workbook_id != state.workbook_id:
            raise ValueError("response belongs to a different workbook than the state")
        verified = (
            response.status == AnswerStatus.VERIFIED
            and response.verification is not None
            and response.verification.verified
        )
        if not verified:
            label = {
                AnswerStatus.CLARIFICATION_REQUIRED: "[clarification requested] ",
                AnswerStatus.NOT_VERIFIED: "[not verified; no facts retained] ",
                AnswerStatus.FAILED: "[failed; no facts retained] ",
            }.get(response.status, "[not verified] ")
            text = response.clarification_question if response.status == AnswerStatus.CLARIFICATION_REQUIRED \
                else ""
            return self.append_history(state, "assistant", label + (text or ""))

        update: dict[str, Any] = {
            "last_verification_id": response.verification.verification_id,
            "revision": state.revision + 1,
        }
        if isinstance(plan, AnalysisPlan):
            if plan.metric is not None:
                update["accepted_metric"] = plan.metric
            update["accepted_filters"] = list(plan.filters)
            update["accepted_group_by"] = list(plan.group_by)
            period = _verified_period(response, plan)
            if period:
                update["accepted_period"] = period
        new_state = state.model_copy(update=update)
        return self.append_history(new_state, "assistant", "[verified] " + response.answer)


def _verified_period(response: AnswerResponse, plan: AnalysisPlan) -> dict[str, str] | None:
    """Prefer the independently recomputed period; fall back to the plan's explicit range."""
    recomputed = response.verification.recomputed_values if response.verification else {}
    period = recomputed.get("period") if isinstance(recomputed, dict) else None
    if isinstance(period, dict) and period:
        return {str(k): str(v) for k, v in period.items() if v is not None}
    if plan.date_range:
        return {str(k): str(v) for k, v in plan.date_range.items() if v is not None}
    return None


def state_summary_for_prompt(state: AgentState) -> dict[str, Any]:
    """Compact JSON-safe view of accepted context for Planner prompts (no raw answers)."""
    return json.loads(state.model_copy(update={"history": state.history[-6:]}).model_dump_json(
        include={"accepted_metric", "accepted_period", "accepted_filters", "accepted_group_by", "history"}))
