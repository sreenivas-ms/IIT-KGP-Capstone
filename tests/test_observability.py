"""Person 6 observability tests: JSONL trace logger, redaction and trace dashboard data."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from p5_support import FakeQwen, write_superstore
from test_p5_integration import ACCEPT, ScriptedQwen, quarter_reply
from verisheet.critic import Critic
from verisheet.executor import Executor
from verisheet.observability.trace_dashboard import (
    artifact_refs,
    safe_artifact,
    summarize_trace,
    timeline_rows,
)
from verisheet.observability.trace_logger import TraceLogger, redact_text, safe_trace_filename
from verisheet.orchestrator import AgentOrchestrator
from verisheet.planner import Planner
from verisheet.profile import profile_workbook
from verisheet.schemas import AnswerStatus, TraceEvent
from verisheet.state import StateStore

T0 = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)


def event(n: int, offset_ms: int, trace="trace-abc", **kw) -> TraceEvent:
    return TraceEvent(trace_id=trace, event_id=f"evt-{n}", actor=kw.pop("actor", "executor"),
                      event_type=kw.pop("event_type", "execute_step"),
                      timestamp=T0 + timedelta(milliseconds=offset_ms), **kw)


def test_trace_logger_appends_and_loads_events_in_order(tmp_path):
    log = TraceLogger(str(tmp_path / "traces"))
    log.log_event(event(2, 50, status="SUCCESS"))
    log.log_event(event(1, 0, actor="orchestrator", event_type="request_received"))
    log.log_event(event(3, 90, actor="orchestrator", event_type="final_response", status="VERIFIED"))
    log.log_event(event(9, 10, trace="trace-other"))                         # different trace

    path = tmp_path / "traces" / "trace-abc.jsonl"
    assert path.is_file() and len(path.read_text().splitlines()) == 3      # append-only JSONL
    loaded = log.load_trace("trace-abc")
    assert [e.event_id for e in loaded] == ["evt-1", "evt-2", "evt-3"]     # timestamp order
    assert all(isinstance(e, TraceEvent) for e in loaded)
    index = log.list_traces()
    assert {r["trace_id"] for r in index} == {"trace-abc", "trace-other"}
    assert next(r for r in index if r["trace_id"] == "trace-abc")["status"] == "VERIFIED"


def test_trace_logger_uses_the_orchestrator_fallback_file_layout(tmp_path):
    assert safe_trace_filename("trace-1/../../etc") == "trace-1etc"        # no path traversal
    log = TraceLogger(str(tmp_path))
    log.log_event(event(1, 0, trace="trace-1/../x").model_dump())          # dicts accepted too
    assert (tmp_path / "trace-1x.jsonl").is_file()
    with pytest.raises(FileNotFoundError):
        log.load_trace("trace-missing")
    with pytest.raises(ValueError):
        log.log_event({"trace_id": "t"})                                    # invalid event


def test_trace_logger_skips_malformed_lines(tmp_path):
    log = TraceLogger(str(tmp_path))
    log.log_event(event(1, 0))
    with (tmp_path / "trace-abc.jsonl").open("a") as stream:
        stream.write("{not json\n")
    log.log_event(event(2, 5))
    assert [e.event_id for e in log.load_trace("trace-abc")] == ["evt-1", "evt-2"]


def test_trace_logger_redacts_sensitive_values(tmp_path, monkeypatch):
    configured = "groupkey-0123456789abcdef"
    monkeypatch.setattr("verisheet.observability.trace_logger._configured_secrets", lambda: [configured])
    log = TraceLogger(str(tmp_path))
    log.log_event(event(
        1, 0,
        input_summary="Email jane.doe@example.com or call +91 98765 43210 about 2025-01-10",
        output_summary="used Authorization: Bearer abcdefghijklmnop1234 and sk-ABCDEFGHIJKLMNOPQRST",
        metadata={"nested": {"note": f"key={configured}", "list": ["api_key=supersecretvalue99"]},
                  "count": 3, "long": "x" * 10_000}))

    raw = (tmp_path / "trace-abc.jsonl").read_text()
    for leaked in ("jane.doe@example.com", "98765 43210", "abcdefghijklmnop1234",
                   "sk-ABCDEFGHIJKLMNOPQRST", configured, "supersecretvalue99"):
        assert leaked not in raw
    stored = TraceLogger(str(tmp_path)).load_trace("trace-abc")[0]
    assert "[EMAIL]" in stored.input_summary and "[PHONE]" in stored.input_summary
    assert "2025-01-10" in stored.input_summary                              # dates are not phones
    assert stored.metadata["count"] == 3 and len(stored.metadata["long"]) < 4200   # bounded
    assert redact_text("plain text", secrets=[]) == "plain text"


def test_trace_dashboard_loads_real_execution_trace(tmp_path, monkeypatch):
    """A trace produced by the real orchestrator/profiler/planner/executor/verifier (only
    Qwen scripted) is written by TraceLogger and summarised by the dashboard functions."""
    monkeypatch.chdir(tmp_path)
    workbook = str(write_superstore(tmp_path / "superstore.xlsx"))
    artifacts = tmp_path / "artifacts"
    logger = TraceLogger(str(tmp_path / "traces"))
    orchestrator = AgentOrchestrator(
        profiler=profile_workbook, retriever=None, planner=Planner(client=ScriptedQwen(quarter_reply())),
        executor=Executor(artifact_dir=str(artifacts)), critic=Critic(client=FakeQwen(ACCEPT)),
        state_store=StateStore(str(tmp_path / "sessions")), trace_logger=logger,
        trace_dir=str(tmp_path / "fallback"), artifact_roots=[artifacts])
    response = orchestrator.answer_question("Compare profit in the last two completed quarters",
                                            workbook, "s-dash")
    assert response.status == AnswerStatus.VERIFIED
    assert not (tmp_path / "fallback").exists()                               # P6 logger was used

    events = logger.load_trace(response.trace_id)
    summary = summarize_trace(events, expected_model="Qwen/Qwen3-8B")
    assert summary["status"] == "VERIFIED" and summary["trace_id"] == response.trace_id
    assert summary["question"].startswith("Compare profit")
    assert summary["input_tokens"] >= 1200 and summary["llm_calls"] >= 1
    assert summary["prompt_versions"]["planner"] == Planner(client=FakeQwen()).prompt_version
    assert "ACCEPT" in summary["transitions"] and summary["verified"] == ["PASS"]
    assert summary["non_production_model"] is False

    rows = timeline_rows(events)
    assert rows[0]["event"] == "request_received" and rows[-1]["event"] == "final_response"
    assert [r["t+ms"] for r in rows] == sorted(r["t+ms"] for r in rows)
    kinds = {r["kind"] for r in artifact_refs(events)}
    assert {"script", "result_table", "chart"} <= kinds
    script = next(r for r in artifact_refs(events) if r["kind"] == "script")
    assert script["hash"].startswith("sha256:") and safe_artifact(script["path"], artifacts) is not None
    assert safe_artifact("/etc/hosts", artifacts) is None                     # outside artifact dir


def test_dashboard_flags_traces_from_a_non_configured_model():
    events = [event(1, 0, actor="planner", event_type="plan", model_name="fake", input_tokens=1, output_tokens=1)]
    assert summarize_trace(events, expected_model="Qwen/Qwen3-8B")["non_production_model"] is True
    assert summarize_trace([]) == {"events": 0}


def test_default_orchestrator_writes_through_trace_logger(tmp_path, monkeypatch):
    """build_default_orchestrator must now get a real TraceLogger, not the fallback None."""
    import verisheet.orchestrator as orch_mod

    created = {}

    class Recorder(TraceLogger):
        def __init__(self, root_dir="results/traces"):
            super().__init__(root_dir)
            created["root"] = root_dir

    monkeypatch.setattr("verisheet.observability.trace_logger.TraceLogger", Recorder)
    monkeypatch.setattr("verisheet.planner.Planner", lambda: object())
    orchestrator = orch_mod.build_default_orchestrator()
    assert isinstance(orchestrator.trace_logger, Recorder) and created["root"]
    json.dumps(created)  # path is plain data
