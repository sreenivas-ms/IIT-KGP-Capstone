"""Streamlit trace viewer owned by Person 6.

Reads only the real JSONL traces written by the Orchestrator/TraceLogger. Pure functions
(``summarize_trace``, ``timeline_rows``, ``artifact_refs``) hold the logic so they are
testable without Streamlit; ``render_trace_dashboard`` only displays their output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verisheet.observability.trace_logger import TraceLogger
from verisheet.schemas import TraceEvent

LLM_ACTORS = {"planner", "critic", "direct"}  # direct = E1 baseline
DECISION_EVENTS = {"critic_decision", "transition", "plan", "replan"}


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_trace(events: list[TraceEvent], expected_model: str | None = None) -> dict[str, Any]:
    """Headline facts for one trace: outcome, totals, models/prompts, decisions, failures."""
    if not events:
        return {"events": 0}
    final = next((e for e in reversed(events) if e.event_type == "final_response"), None)
    llm = [e for e in events if e.actor in LLM_ACTORS and e.model_name]
    models = sorted({e.model_name for e in llm if e.model_name})
    decisions = [e.status for e in events if e.actor == "critic" and e.event_type == "critic_decision" and e.status]
    transitions = [e.status for e in events if e.event_type == "transition" and e.status]
    verify = [e for e in events if e.actor == "verifier" and e.event_type == "verify"]
    request = next((e for e in events if e.event_type == "request_received"), events[0])
    meta_final = final.metadata if final else {}
    summary = {
        "trace_id": events[0].trace_id,
        "question": request.input_summary,
        "session_id": request.metadata.get("session_id"),
        "run_id": request.metadata.get("run_id"),
        "status": final.status if final else "IN_PROGRESS/UNKNOWN",
        "answer": final.output_summary if final else None,
        "failure_category": final.failure_category if final else None,
        "events": len(events),
        "started": events[0].timestamp.isoformat(),
        "finished": events[-1].timestamp.isoformat(),
        "llm_calls": len(llm),
        "input_tokens": int(sum(e.input_tokens or 0 for e in events)),
        "output_tokens": int(sum(e.output_tokens or 0 for e in events)),
        "llm_latency_ms": round(sum(_num(e.latency_ms) for e in llm), 1),
        "total_latency_ms": _num(meta_final.get("total_latency_ms")) or None,
        "models": models,
        "prompt_versions": {e.actor: e.prompt_version for e in llm if e.prompt_version},
        "critic_decisions": decisions,
        "transitions": transitions,
        "repairs": meta_final.get("repairs"),
        "replans": meta_final.get("replans"),
        "actions": meta_final.get("actions"),
        "verified": [e.status for e in verify],
        "failure_categories": sorted({e.failure_category for e in events if e.failure_category}),
        "verification_id": meta_final.get("verification_id"),
    }
    summary["non_production_model"] = bool(expected_model and models and any(m != expected_model for m in models))
    return summary


def timeline_rows(events: list[TraceEvent]) -> list[dict[str, Any]]:
    """One row per event in timestamp order, for the timeline table."""
    start = events[0].timestamp if events else None
    rows = []
    for e in events:
        rows.append({
            "t+ms": round((e.timestamp - start).total_seconds() * 1000, 1) if start else 0.0,
            "actor": e.actor,
            "event": e.event_type,
            "step": e.step_id or "",
            "status": e.status or "",
            "model": e.model_name or "",
            "prompt": e.prompt_version or "",
            "tokens in/out": f"{e.input_tokens}/{e.output_tokens}" if e.input_tokens is not None else "",
            "latency ms": round(_num(e.latency_ms), 1) if e.latency_ms is not None else None,
            "failure": e.failure_category or "",
        })
    return rows


def artifact_refs(events: list[TraceEvent]) -> list[dict[str, str]]:
    """Scripts, result tables and charts referenced by executor events (deduplicated)."""
    seen, refs = set(), []
    for e in events:
        for kind in ("script_path", "result_table_path", "chart_path"):
            path = e.metadata.get(kind)
            if path and path != "None" and path not in seen:
                seen.add(path)
                refs.append({"step": e.step_id or "", "kind": kind.replace("_path", ""), "path": str(path),
                             "hash": str(e.metadata.get("script_hash") or "") if kind == "script_path" else ""})
    return refs


def safe_artifact(path: str, artifact_root: str | Path) -> Path | None:
    """Resolve an artifact path only if it lies inside the artifact directory."""
    root = Path(artifact_root).resolve()
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    return candidate if candidate.is_file() and candidate.is_relative_to(root) else None


def _pretty(text: str | None) -> Any:
    if not text:
        return ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def render_trace_dashboard(trace_root: str = "results/traces", selected_trace_id: str | None = None,
                           artifact_root: str = "artifacts") -> None:
    """Render real trace executions, plans, script references, tool outputs and critic verdicts.

    Dashboard must show token counts, latency, model/prompt versions, failure categories and
    verification findings. It should never label mock/example traces as actual executions.
    """
    import pandas as pd
    import streamlit as st

    from verisheet.config import get_settings

    st.header("Trace dashboard")
    store = TraceLogger(trace_root)
    index = store.list_traces()
    if not index:
        st.info(f"No traces yet in `{trace_root}`. Ask a question on the Analyse page first.")
        return

    ids = [row["trace_id"] for row in index]
    typed = st.text_input("Trace ID", value=selected_trace_id or "", placeholder="trace-…")
    default = ids.index(typed) if typed in ids else 0
    choice = st.selectbox(
        "Recent traces (newest first)", ids, index=default,
        format_func=lambda tid: next(f"{r['status']:<22} {r['question'][:70]}  ·  {tid}" for r in index if r["trace_id"] == tid))
    trace_id = typed if typed and typed not in ids else choice
    try:
        events = store.load_trace(trace_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    summary = summarize_trace(events, expected_model=get_settings().qwen_model)
    if summary.get("non_production_model"):
        st.warning(f"This trace used model(s) {summary['models']}, not the configured "
                   f"{get_settings().qwen_model}. It may come from a test run, not a live execution.")
    st.subheader(summary["question"] or trace_id)
    cols = st.columns(6)
    cols[0].metric("Status", summary["status"])
    cols[1].metric("LLM calls", summary["llm_calls"])
    cols[2].metric("Tokens in / out", f"{summary['input_tokens']} / {summary['output_tokens']}")
    cols[3].metric("Total latency", f"{(summary['total_latency_ms'] or 0) / 1000:.1f} s")
    cols[4].metric("Repairs / replans", f"{summary['repairs'] or 0} / {summary['replans'] or 0}")
    cols[5].metric("Actions", summary["actions"] or 0)
    st.caption(f"Trace `{trace_id}` · session `{summary['session_id']}` · run `{summary['run_id']}` · "
               f"models {summary['models'] or '—'} · prompts {summary['prompt_versions'] or '—'}")
    if summary["answer"]:
        st.markdown(f"**Final answer:** {summary['answer']}")
    path = " → ".join(summary["transitions"] or summary["critic_decisions"]) or "—"
    st.markdown(f"**Decision path:** {path}")
    if summary["failure_categories"]:
        st.error(f"Failure categories: {', '.join(summary['failure_categories'])}")

    st.markdown("#### Timeline")
    st.dataframe(pd.DataFrame(timeline_rows(events)), hide_index=True, width="stretch")

    st.markdown("#### Events")
    for number, e in enumerate(events, 1):
        label = f"{number:02d}. {e.actor} · {e.event_type}" + (f" · {e.step_id}" if e.step_id else "") + \
                (f" · {e.status}" if e.status else "")
        with st.expander(label, expanded=e.event_type in {"critic_decision", "verify"} and e.status not in {"ACCEPT", "PASS"}):
            if e.input_summary:
                st.markdown("Input")
                st.json(_pretty(e.input_summary)) if isinstance(_pretty(e.input_summary), (dict, list)) else st.code(e.input_summary)
            if e.output_summary:
                st.markdown("Output")
                out = _pretty(e.output_summary)
                st.json(out) if isinstance(out, (dict, list)) else st.code(e.output_summary)
            st.markdown("Metadata")
            st.json(e.metadata)

    refs = artifact_refs(events)
    if refs:
        st.markdown("#### Artifacts")
        for ref in refs:
            artifact = safe_artifact(ref["path"], artifact_root)
            title = f"{ref['kind']} · {ref['step']} · `{ref['path']}`" + (f" · {ref['hash'][:23]}…" if ref["hash"] else "")
            with st.expander(title):
                if artifact is None:
                    st.caption("File not available (outside the artifact directory or deleted).")
                elif artifact.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                    st.image(str(artifact))
                elif artifact.suffix.lower() == ".csv":
                    st.dataframe(pd.read_csv(artifact), hide_index=True)
                else:
                    st.code(artifact.read_text(encoding="utf-8", errors="replace")[:20000], language="python")
