"""Run one system variant against the frozen development or held-out test questions.

Variants (VS-P6-03), identical model / temperature / tools / budgets / question file:
    direct                E1  Qwen answers from profile + sample rows (no plan, no execution)
    planner_executor      E2  Planner -> Executor, no critic (never VERIFIED)
    llm_critic            E3  + Qwen-only critic
    deterministic_critic  E4  + deterministic verification critic (no Qwen review)
    full                  E5  hybrid: deterministic verification + Qwen semantic review

Usage:
    python experiments/run.py --system full --split test
    python experiments/run.py --system all --split dev --questions experiments/smoke/questions.jsonl
Outputs results/experiments/<run_id>/{results.jsonl, run_metadata.json}; traces go to the trace dir.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from verisheet.config import get_settings
from verisheet.experiments.common import (
    DEFAULT_QUESTIONS,
    RESULTS_ROOT,
    environment_metadata,
    flatten_numbers,
    load_questions,
    numbers_in_table,
    numbers_in_text,
    sha256_file,
    utc_now,
    write_jsonl,
)
from verisheet.observability.trace_dashboard import summarize_trace
from verisheet.observability.trace_logger import TraceLogger
from verisheet.schemas import AnswerResponse, AnswerStatus, ExperimentConfig, ExperimentVariant, TraceEvent

ROOT = Path(__file__).resolve().parents[3]
CRITIC_MODE = {
    ExperimentVariant.PLANNER_EXECUTOR: None,
    ExperimentVariant.LLM_CRITIC: "llm_only",
    ExperimentVariant.DETERMINISTIC_CRITIC: "deterministic",
    ExperimentVariant.FULL: "hybrid",
}
VARIANT_LABEL = {
    ExperimentVariant.DIRECT: "E1", ExperimentVariant.PLANNER_EXECUTOR: "E2", ExperimentVariant.LLM_CRITIC: "E3",
    ExperimentVariant.DETERMINISTIC_CRITIC: "E4", ExperimentVariant.FULL: "E5",
}


class System:
    """Uniform interface: ``answer(question, workbook, session_id) -> (AnswerResponse, details)``."""

    def __init__(self, variant: ExperimentVariant, max_actions: int, max_repairs: int):
        self.variant = variant
        if variant == ExperimentVariant.DIRECT:
            from verisheet.experiments.direct import DirectAnswerer

            self._direct = DirectAnswerer(trace_logger=TraceLogger(get_settings().trace_dir))
            self._orchestrator = None
        else:
            from verisheet.orchestrator import build_default_orchestrator

            self._direct = None
            self._orchestrator = build_default_orchestrator(critic_mode=CRITIC_MODE[variant] or "deterministic")
            if CRITIC_MODE[variant] is None:
                self._orchestrator.critic = None          # E2: no critic -> never VERIFIED
            self._orchestrator.max_actions = max_actions
            self._orchestrator.max_repairs = max_repairs

    def prompt_versions(self) -> dict[str, str]:
        if self._direct is not None:
            return {"direct": self._direct.prompt_version}
        versions = {"planner": getattr(self._orchestrator.planner, "prompt_version", None)}
        if self._orchestrator.critic is not None:
            versions["critic"] = getattr(self._orchestrator.critic, "prompt_version", None)
        return {k: v for k, v in versions.items() if v}

    def answer(self, question: str, workbook: str, session_id: str) -> tuple[AnswerResponse, dict[str, Any]]:
        if self._direct is not None:
            return self._direct.answer(question, workbook, session_id), {}
        response = self._orchestrator.answer_question(question, workbook, session_id)
        plotted, chart_type, executed = None, None, []
        for result in self._orchestrator.last_results or []:
            path = result.result_table_path
            if path and str(path).endswith(".csv") and not result.chart_path:
                executed.append(path)          # computed tables, even if not shown to the user
            if result.plotted_data:
                plotted = result.plotted_data
                chart_type = result.chart_spec.chart_type.value if result.chart_spec else None
        if response.chart_spec is not None:
            chart_type = response.chart_spec.chart_type.value
        return response, {"plotted_data": plotted, "chart_type": chart_type, "executed_tables": executed}


def run_experiment(config: ExperimentConfig, out_dir: str | Path | None = None,
                   system_factory: Callable[[ExperimentConfig], Any] | None = None,
                   limit: int | None = None) -> list[dict]:
    """Execute every question in the selected split using one consistent system variant.

    Keep model version, temperature, prompt versions, tool library and test set consistent
    across E1–E5. Persist one structured trace/result record per question.
    """
    questions = load_questions(config.question_file, config.split)[: limit or None]
    if not questions:
        raise ValueError(f"no '{config.split}' questions in {config.question_file}")
    test_set_sha = sha256_file(config.question_file)
    run_id = f"{config.system.value}-{config.split}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = Path(out_dir) if out_dir else RESULTS_ROOT / run_id
    system = (system_factory or (lambda c: System(c.system, c.max_actions, c.max_repairs)))(config)
    trace_store = TraceLogger(get_settings().trace_dir)
    prompts = system.prompt_versions()
    metadata = {
        "run_id": run_id, "variant": config.system.value, "variant_label": VARIANT_LABEL[config.system],
        "split": config.split, "question_file": str(config.question_file), "test_set_sha256": test_set_sha,
        "question_ids": [q["id"] for q in questions], "model_name": config.model_name,
        "prompt_versions": prompts, "temperature": config.temperature, "max_actions": config.max_actions,
        "max_repairs": config.max_repairs, "started": utc_now(), **environment_metadata(),
    }
    records = []
    for question in questions:
        records.append(_run_one(system, question, config, metadata, trace_store))
        print(f"  {metadata['variant_label']} {question['id']:<6} {records[-1]['status']:<23} "
              f"{records[-1]['latency_ms'] or 0:>8.0f} ms  {(records[-1]['answer'] or '')[:70]!r}", flush=True)
    metadata["finished"] = utc_now()
    write_jsonl(out / "results.jsonl", records)
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return records


def _run_one(system, question: dict, config: ExperimentConfig, metadata: dict, trace_store: TraceLogger) -> dict:
    workbook = str(ROOT / question["workbook"]) if not Path(question["workbook"]).is_absolute() else question["workbook"]
    record: dict[str, Any] = {
        "run_id": metadata["run_id"], "variant": config.system.value, "question_id": question["id"],
        "family": question["family"], "split": question["split"], "must_refuse": question["must_refuse"],
        "test_set_sha256": metadata["test_set_sha256"], "error": None,
    }
    started = time.perf_counter()
    if not Path(workbook).is_file() or sha256_file(workbook) != question["workbook_sha256"]:
        response, details = AnswerResponse(status=AnswerStatus.FAILED, trace_id="none",
                                           answer="workbook missing or hash mismatch"), {}
        record["error"] = "workbook_hash_mismatch"
    else:
        session = f"exp-{metadata['run_id']}-{question['id']}"
        try:
            response, details = system.answer(question["question"], workbook, session)
        except Exception as exc:  # one failing question must not stop the run
            response, details = AnswerResponse(status=AnswerStatus.FAILED, trace_id="none",
                                               answer=f"{type(exc).__name__}"), {}
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    elapsed = round((time.perf_counter() - started) * 1000, 1)

    summary: dict[str, Any] = {}
    if response.trace_id.startswith("trace-"):
        try:
            summary = summarize_trace(trace_store.load_trace(response.trace_id))
        except FileNotFoundError:
            summary = {}
        trace_store.log_event(TraceEvent(
            trace_id=response.trace_id, event_id=f"evt-exp-{uuid.uuid4().hex[:8]}", actor="experiment",
            event_type="experiment_config", status=config.system.value,
            metadata={k: metadata[k] for k in ("run_id", "variant", "variant_label", "split", "test_set_sha256",
                                               "model_name", "prompt_versions", "temperature", "max_actions",
                                               "max_repairs")} | {"question_id": question["id"]}))
    record.update({
        "status": response.status.value, "answer": response.answer[:2000], "trace_id": response.trace_id,
        "clarification_question": response.clarification_question,
        "verified": bool(response.verification and response.verification.verified),
        "result_table_path": response.result_table_path, "chart_path": response.chart_path,
        "chart_type": details.get("chart_type"),
        "table_numbers": numbers_in_table(response.result_table_path),
        "executed_numbers": [n for p in details.get("executed_tables", []) for n in numbers_in_table(p)],
        "answer_numbers": numbers_in_text(response.answer),
        "plotted_numbers": flatten_numbers(details.get("plotted_data")) if details.get("plotted_data") else [],
        "latency_ms": summary.get("total_latency_ms") or elapsed,
        "input_tokens": summary.get("input_tokens", 0), "output_tokens": summary.get("output_tokens", 0),
        "llm_calls": summary.get("llm_calls", 0), "actions": summary.get("actions"),
        "repairs": summary.get("repairs") or 0, "replans": summary.get("replans") or 0,
        "transitions": summary.get("transitions", []), "failure_category": summary.get("failure_category"),
        "models": summary.get("models", []), "prompt_versions": summary.get("prompt_versions", {}),
    })
    return record


def _config(system: str, split: str, questions: str) -> ExperimentConfig:
    settings = get_settings()
    return ExperimentConfig(system=ExperimentVariant(system), split=split, question_file=questions,
                            model_name=settings.qwen_model, prompt_version="see run_metadata.prompt_versions",
                            temperature=0.0, max_actions=settings.max_agent_actions,
                            max_repairs=settings.max_repairs)


def main() -> None:
    """Parse `--system` and `--split` arguments and call run_experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=[v.value for v in ExperimentVariant] + ["all"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                        help="question_task JSONL (default: Person 1's frozen set)")
    parser.add_argument("--out", default=None, help="output dir (default results/experiments/<run_id>)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (debugging)")
    args = parser.parse_args()
    systems = [v.value for v in ExperimentVariant] if args.system == "all" else [args.system]
    for system in systems:
        config = _config(system, args.split, args.questions)
        print(f"== {VARIANT_LABEL[config.system]} {system} on {args.split} ({args.questions})")
        out = Path(args.out) / system if args.out and len(systems) > 1 else args.out
        records = run_experiment(config, out, limit=args.limit)
        print(f"   {len(records)} questions -> {Path(out) if out else RESULTS_ROOT}")


if __name__ == "__main__":
    main()
