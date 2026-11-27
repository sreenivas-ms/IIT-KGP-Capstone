"""E1 Direct baseline: Qwen answers from the profile and bounded, sanitised sample rows.

No planner, no executor, no verification: the answer is never VERIFIED. It exists to measure
what planning, execution and independent checking add (ablation baseline).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from verisheet.planner_context import compact_profile
from verisheet.profile import profile_workbook
from verisheet.qwen_client import QwenClient, extract_json
from verisheet.schemas import AnswerResponse, AnswerStatus, LLMResponse, TraceEvent
from verisheet.tools.inspect import sample_rows

ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = ROOT / "prompts" / "direct.txt"
MAX_SAMPLE_ROWS_PER_SHEET = 20
MAX_SAMPLE_CHARS = 9000


class DirectAnswerer:
    """``answer(question, workbook, session_id) -> AnswerResponse`` with one Qwen call."""

    def __init__(self, client: Any = None, trace_logger: Any = None):
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        match = re.search(r"VERSION\s+([\w.\-]+)", self.system_prompt)
        self.prompt_version = match.group(1) if match else "unknown"
        self.client = client or QwenClient()
        self.trace_logger = trace_logger
        self.last_llm: LLMResponse | None = None

    def answer(self, question: str, workbook: str, session_id: str) -> AnswerResponse:
        trace_id = f"trace-{uuid.uuid4().hex}"
        started = time.perf_counter()
        self._emit(trace_id, "orchestrator", "request_received", session_id, input_summary=question[:500])
        profile = profile_workbook(workbook)
        samples = {}
        for sheet in profile.sheets:
            if sheet.row_count and not sheet.hidden:
                samples[sheet.name] = sample_rows(workbook, sheet.name, n=MAX_SAMPLE_ROWS_PER_SHEET)
        sample_text = json.dumps(samples, default=str)[:MAX_SAMPLE_CHARS]
        user = (f"<question>\n{question}\n</question>\n<profile>\n{compact_profile(profile, 5000)}\n</profile>\n"
                f"<sample_rows>\n{sample_text}\n</sample_rows>\nReturn the JSON now.")
        llm = self.client.complete(self.system_prompt, user, temperature=0.0, max_tokens=400)
        self.last_llm = llm
        parsed = _parse(llm.content)
        self._emit(trace_id, "direct", "direct_answer", session_id, status="SUCCESS" if parsed else "PARSE_ERROR",
                   llm=llm, output_summary=llm.content[:1500])
        if parsed and parsed.get("clarification_needed") and parsed.get("clarification_question"):
            response = AnswerResponse(status=AnswerStatus.CLARIFICATION_REQUIRED, trace_id=trace_id,
                                      answer="I need more information before I can answer.",
                                      clarification_question=str(parsed["clarification_question"])[:500],
                                      workbook_id=profile.workbook_id)
        elif parsed:
            text = str(parsed.get("answer") or "")
            values = parsed.get("values")
            if values is not None:
                text += f"\n\nValues: {json.dumps(values, default=str)}"
            response = AnswerResponse(status=AnswerStatus.NOT_VERIFIED, trace_id=trace_id, answer=text[:3000],
                                      assumptions=["E1 direct baseline: model answer from sample rows, "
                                                   "not executed or independently verified."],
                                      workbook_id=profile.workbook_id)
        else:
            response = AnswerResponse(status=AnswerStatus.FAILED, trace_id=trace_id, workbook_id=profile.workbook_id,
                                      answer="The direct baseline did not return a valid JSON answer.")
        self._emit(trace_id, "orchestrator", "final_response", session_id, status=response.status.value,
                   output_summary=response.answer[:1500],
                   metadata={"actions": 1, "repairs": 0, "replans": 0,
                             "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                             "input_tokens": llm.input_tokens, "output_tokens": llm.output_tokens})
        return response

    def _emit(self, trace_id: str, actor: str, event_type: str, session_id: str, *, status: str | None = None,
              llm: LLMResponse | None = None, input_summary: str | None = None,
              output_summary: str | None = None, metadata: dict | None = None) -> None:
        if self.trace_logger is None:
            return
        self.trace_logger.log_event(TraceEvent(
            trace_id=trace_id, event_id=f"evt-{uuid.uuid4().hex[:8]}", actor=actor, event_type=event_type,
            status=status, input_summary=input_summary, output_summary=output_summary,
            model_name=llm.model if llm else None, prompt_version=self.prompt_version if llm else None,
            input_tokens=llm.input_tokens if llm else None, output_tokens=llm.output_tokens if llm else None,
            latency_ms=llm.latency_ms if llm else None,
            metadata={"session_id": session_id, "variant": "direct", **(metadata or {})}))


def _parse(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(extract_json(content))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
