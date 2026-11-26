"""Agent loop owned by Person 5; coordinates the Planner, Executor and Critic.

One user turn runs this bounded state machine::

    profile -> load state -> retrieve -> PLAN -> pre-execution checks -> EXECUTE
        -> independent VERIFY -> CRITIC -> ACCEPT | REPAIR | REPLAN | CLARIFY

* ACCEPT   -> VERIFIED answer only if the deterministic report passed (else NOT_VERIFIED).
* REPAIR   -> re-run only the failed step and its downstream dependants (max ``max_repairs``).
* REPLAN   -> ask the Planner for a revised plan with the Critic's feedback (max ``max_replans``).
* CLARIFY  -> return the clarification question; nothing is executed or persisted as fact.

Budgets: every Planner call, executed step and Critic evaluation is one *action*; the loop stops
at ``max_actions`` (default 10). Every transition emits a ``TraceEvent`` sharing one trace ID.
Answer text is composed deterministically from the independently recomputed values — never from
Qwen output — so the UI can display it without trusting the model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from verisheet.schemas import (
    AgentState, AnalysisPlan, AnswerResponse, AnswerStatus, ChartSpec, ChartType, CriticDecision,
    Decision, ExecutionResult, RetrievedContext, RunStatus, ToolError, TraceEvent,
    VerificationFinding, VerificationReport, WorkbookProfile,
)
from verisheet.state import StateStore, summarise_text
from verisheet.tools.verify import is_blocking, verify_execution, verify_plan_semantics

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 1e-6  # relative/absolute; counts compare exactly in practice
MAX_SUMMARY = 300


@dataclass
class _Usage:
    """LLM usage summary in the shape TraceEvent needs (model, tokens, latency)."""

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class ComponentUnavailable(RuntimeError):
    """A collaborator is not implemented/configured yet (e.g. a teammate's stub)."""


class _PlannerFailed(RuntimeError):
    """The Planner raised (malformed JSON, schema violation, endpoint error)."""


@dataclass
class _Turn:
    """Mutable bookkeeping for one user request."""

    trace_id: str
    run_id: str
    session_id: str
    question: str
    workbook_id: str | None = None
    actions: int = 0
    repairs: int = 0
    replans: int = 0
    events: list[TraceEvent] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


class _JsonlTraceSink:
    """Minimal append-only JSONL writer used only when the injected trace logger fails."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def log_event(self, event: TraceEvent) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch for ch in event.trace_id if ch.isalnum() or ch in "-_")[:80] or "trace"
        with (self.root_dir / f"{safe}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")


def _summary(value: Any, limit: int = MAX_SUMMARY) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return summarise_text(text, limit)


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 1 or value == 0 else f"{value:.4g}"
    return str(value)


class AgentOrchestrator:
    """Run the bounded Planner → Executor → Critic loop with state and structured trace events.

    Required transitions: ACCEPT returns; REPAIR retries the failed local step; REPLAN calls
    Planner with critic feedback; CLARIFY asks the user and stops. Enforce max 10 actions and
    at most 2 repairs by default. Never show unverified values as VERIFIED.
    """

    def __init__(self, profiler, retriever, planner, executor, critic, state_store, trace_logger, *,
                 verifier: Callable[..., VerificationReport] | None = None,
                 max_actions: int | None = None, max_repairs: int | None = None, max_replans: int = 2,
                 tolerance: float = DEFAULT_TOLERANCE, artifact_roots: list[str | Path] | None = None,
                 trace_dir: str | None = None):
        """Inject collaborators; use interfaces rather than importing their implementation details.

        Collaborators may be objects or plain callables:
          profiler:  ``profile_workbook(path)`` or callable(path) -> WorkbookProfile
          retriever: ``retrieve_context(q, profile)`` / ``search(q)`` / callable, or None
          planner:   ``generate_plan(...)`` and optionally ``revise_plan(plan, feedback, state)``
          executor:  ``execute_plan(plan, path, run_id)`` and optionally ``execute_step(...)``
          critic:    ``evaluate(question, plan, results, report, profile)`` or None (E2 variant)
          state_store: StateStore-compatible; trace_logger: ``log_event(TraceEvent)`` or None
        """
        self.profiler = profiler
        self.retriever = retriever
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.state_store = state_store
        self.trace_logger = trace_logger
        self.verifier = verifier or verify_execution
        settings = _settings()
        self.max_actions = int(max_actions if max_actions is not None else getattr(settings, "max_agent_actions", 10))
        self.max_repairs = int(max_repairs if max_repairs is not None else getattr(settings, "max_repairs", 2))
        self.max_replans = int(max_replans)
        self.tolerance = float(tolerance)
        self.artifact_roots = artifact_roots
        self._fallback_sink = _JsonlTraceSink(trace_dir or getattr(settings, "trace_dir", "results/traces"))
        self.last_events: list[TraceEvent] = []
        self.last_plan: AnalysisPlan | None = None
        self.last_results: list[ExecutionResult] = []
        if self.max_actions < 1 or self.max_repairs < 0 or self.max_replans < 0:
            raise ValueError("budgets must be non-negative and max_actions >= 1")

    # ====================================================================================
    # Public entry
    # ====================================================================================

    def answer_question(self, question: str, workbook_path: str, session_id: str) -> AnswerResponse:
        """Execute one user turn and return the canonical AnswerResponse.

        Steps: profile/fingerprint workbook; load matching state; retrieve definitions; plan;
        execute dependency steps; run independent verification; ask Critic for transition; repair,
        replan or clarify within budgets; persist only accepted verified context; return trace ID.
        """
        turn = _Turn(trace_id=f"trace-{uuid4().hex}", run_id=f"run-{uuid4().hex[:12]}",
                     session_id=session_id, question=summarise_text(question, 1000))
        self.last_events, self.last_plan, self.last_results = turn.events, None, []
        self._emit(turn, "orchestrator", "request_received", input_summary=turn.question)
        try:
            if not isinstance(question, str) or not question.strip():
                return self._finish(turn, self._make_failure_response(turn.trace_id, "Please enter a question."),
                                    None, None, "invalid_input")
            StateStore.validate_session_id(session_id)
            return self._run(turn, question.strip(), workbook_path, session_id)
        except ComponentUnavailable as exc:
            return self._finish(turn, self._make_failure_response(
                turn.trace_id, f"The {exc} component is not available yet, so no answer was produced.",
                turn.workbook_id), None, None, "component_unavailable")
        except Exception as exc:  # last-resort guard: never leak tracebacks to the user
            logger.exception("orchestrator failure")
            return self._finish(turn, self._make_failure_response(
                turn.trace_id, f"The request failed unexpectedly ({type(exc).__name__}).", turn.workbook_id),
                None, None, "orchestrator_error")

    # ====================================================================================
    # Loop
    # ====================================================================================

    def _run(self, turn: _Turn, question: str, workbook_path: str, session_id: str) -> AnswerResponse:
        profile = self._profile(turn, workbook_path)
        if profile is None:
            return self._finish(turn, self._make_failure_response(
                turn.trace_id, "The workbook could not be read. Upload a valid .xlsx or .csv file."),
                None, None, "upload_error")
        turn.workbook_id = profile.workbook_id
        state = self.state_store.load(session_id, profile.workbook_id)
        state = StateStore.append_history(state, "user", question)
        context = self._retrieve(turn, question, profile)

        try:
            plan = self._plan(turn, question, profile, context, state, feedback=None)
        except _PlannerFailed as exc:
            response = self._make_failure_response(
                turn.trace_id, f"The planner could not produce a valid plan for this question ({exc}). "
                "Try rephrasing it with the sheet, metric and period you need.", turn.workbook_id)
            return self._finish(turn, response, state, None, "planner_error")
        results: list[ExecutionResult] | None = None
        pending_repair: CriticDecision | None = None

        while True:
            self.last_plan = plan
            if plan.clarification_needed:
                return self._clarify(turn, state, plan, plan.clarification_question or "Please clarify the question.")

            if results is None and self.critic is not None:
                # Deterministic pre-execution gate: avoids spending actions on a plan we know is wrong.
                pre = verify_plan_semantics(question, plan, profile)
                self._emit(turn, "verifier", "plan_checks", status="PASS" if not any(map(is_blocking, pre)) else "FAIL",
                           output_summary=_summary([f["check_id"] for f in pre if not f["passed"]]))
                if any(is_blocking(f) for f in pre):
                    report = _pre_report(pre)
                    decision = self._critic(turn, question, plan, [], report, profile, count=False)
                    if decision.decision == Decision.REPAIR and decision.error_category == "inappropriate_chart_type":
                        if not self._budget_repair(turn):
                            return self._not_verified(turn, state, plan, report, decision, "repair budget exhausted")
                        plan = self._patch_plan_for_repair(plan, decision)
                        continue
                    outcome = self._transition(turn, state, plan, report, decision, question, profile, context)
                    if isinstance(outcome, AnswerResponse):
                        return outcome
                    plan, results = outcome, None
                    continue

            if results is None:
                if not self._spend(turn, len(plan.steps)):
                    return self._not_verified(turn, state, plan, None, None, "action budget exhausted")
                results = self._execute_plan(turn, plan, workbook_path)
            elif pending_repair is not None:
                results = self._handle_repair(plan, pending_repair, results, workbook_path, turn.run_id, turn=turn)
                pending_repair = None
                if results is None:
                    return self._not_verified(turn, state, plan, None, None, "action budget exhausted")

            self.last_results = results
            report = self._verify(turn, question, plan, results, profile, workbook_path)
            if not self._spend(turn, 1):
                return self._not_verified(turn, state, plan, report, None, "action budget exhausted")
            decision = self._critic(turn, question, plan, results, report, profile)

            if decision.decision == Decision.ACCEPT:
                return self._accept(turn, state, plan, results, report, decision)
            if decision.decision == Decision.REPAIR:
                if not self._budget_repair(turn):
                    return self._not_verified(turn, state, plan, report, decision, "repair budget exhausted")
                plan = self._patch_plan_for_repair(plan, decision)
                pending_repair = decision
                continue
            outcome = self._transition(turn, state, plan, report, decision, question, profile, context)
            if isinstance(outcome, AnswerResponse):
                return outcome
            plan, results = outcome, None

    def _transition(self, turn, state, plan, report, decision, question, profile, context):
        """Handle CLARIFY/REPLAN (and impossible REPAIRs); return a response or a new plan."""
        if decision.decision == Decision.CLARIFY:
            return self._clarify(turn, state, plan, decision.clarification_question
                                 or "Could you clarify the metric, period or definition you need?", decision)
        if decision.decision == Decision.REPLAN:
            if turn.replans >= self.max_replans:
                return self._not_verified(turn, state, plan, report, decision, "replan budget exhausted")
            turn.replans += 1
            self._emit(turn, "orchestrator", "transition", status="REPLAN",
                       failure_category=decision.error_category, output_summary=decision.instruction)
            feedback = {
                "decision": decision.decision.value, "failed_step_id": decision.failed_step_id,
                "error_category": decision.error_category, "instruction": decision.instruction,
                "evidence": decision.evidence, "previous_plan_id": plan.plan_id,
                # Planner.revise_plan falls back to these when it did not create the prior plan;
                # it removes them before building the prompt.
                "question": question, "workbook_profile": profile,
            }
            return self._revise(turn, question, profile, context, state, plan, feedback)
        return self._not_verified(turn, state, plan, report, decision, "unsupported transition")

    # ====================================================================================
    # Collaborator adapters
    # ====================================================================================

    def _profile(self, turn: _Turn, workbook_path: str) -> WorkbookProfile | None:
        started = time.perf_counter()
        try:
            fn = getattr(self.profiler, "profile_workbook", self.profiler)
            profile = fn(workbook_path)
            if not isinstance(profile, WorkbookProfile):
                profile = WorkbookProfile.model_validate(profile)
        except NotImplementedError as exc:
            raise ComponentUnavailable("profiling") from exc
        except (ValueError, OSError) as exc:
            self._emit(turn, "profiler", "profile", status="FAILED", failure_category="upload_error",
                       output_summary=summarise_text(str(exc), 200), started=started)
            return None
        self._emit(turn, "profiler", "profile", status="SUCCESS", started=started,
                   output_summary=_summary({"workbook_id": profile.workbook_id,
                                            "sheets": [s.name for s in profile.sheets],
                                            "warnings": [w.code for w in profile.warnings][:10]}))
        return profile

    def _retrieve(self, turn: _Turn, question: str, profile: WorkbookProfile) -> RetrievedContext:
        started = time.perf_counter()
        empty = RetrievedContext(query=question[:500], corpus_version="unavailable")
        if self.retriever is None:
            return empty
        try:
            if hasattr(self.retriever, "retrieve_context"):
                context = self.retriever.retrieve_context(question, profile)
            elif hasattr(self.retriever, "search"):
                context = self.retriever.search(question)
            else:
                context = self.retriever(question, profile)
            if not isinstance(context, RetrievedContext):
                context = RetrievedContext.model_validate(context)
        except Exception as exc:  # retrieval is advisory; the loop continues without it
            self._emit(turn, "retriever", "retrieve", status="FAILED", failure_category="retrieval_miss",
                       output_summary=type(exc).__name__, started=started)
            return empty.model_copy(update={"warnings": [f"RETRIEVAL_UNAVAILABLE: {type(exc).__name__}"]})
        self._emit(turn, "retriever", "retrieve", status="SUCCESS", started=started,
                   output_summary=_summary([i.source_path for i in context.items]),
                   metadata={"corpus_version": context.corpus_version})
        return context

    def _plan(self, turn, question, profile, context, state, feedback) -> AnalysisPlan:
        if not self._spend(turn, 1):
            raise RuntimeError("action budget exhausted before planning")
        started = time.perf_counter()
        try:
            fn = getattr(self.planner, "generate_plan", self.planner)
            plan = fn(question, profile, context, previous_state=state, feedback=feedback)
            if not isinstance(plan, AnalysisPlan):
                plan = AnalysisPlan.model_validate(plan)
        except NotImplementedError as exc:
            raise ComponentUnavailable("planner") from exc
        except Exception as exc:  # malformed JSON / schema violation / endpoint error
            self._emit(turn, "planner", "plan", status="FAILED", failure_category="planner_error",
                       output_summary=type(exc).__name__, started=started)
            raise _PlannerFailed(type(exc).__name__) from exc
        self._emit_plan(turn, plan, started, "plan")
        return plan

    def _revise(self, turn, question, profile, context, state, plan, feedback):
        if not self._spend(turn, 1):
            return self._not_verified(turn, state, plan, None, None, "action budget exhausted")
        started = time.perf_counter()
        try:
            if hasattr(self.planner, "revise_plan"):
                new_plan = self.planner.revise_plan(plan, feedback, state)
            else:
                fn = getattr(self.planner, "generate_plan", self.planner)
                new_plan = fn(question, profile, context, previous_state=state, feedback=feedback)
            if not isinstance(new_plan, AnalysisPlan):
                new_plan = AnalysisPlan.model_validate(new_plan)
        except NotImplementedError as exc:
            raise ComponentUnavailable("planner") from exc
        except Exception as exc:
            self._emit(turn, "planner", "replan", status="FAILED", failure_category="planner_error",
                       output_summary=type(exc).__name__, started=started)
            return self._not_verified(turn, state, plan, None, None, "the planner could not revise the plan")
        self._emit_plan(turn, new_plan, started, "replan")
        return new_plan

    def _emit_plan(self, turn, plan, started, event_type):
        llm = getattr(self.planner, "last_llm_response", None) or getattr(self.planner, "last_response", None)
        run = getattr(self.planner, "last_run", None)
        metadata: dict[str, Any] = {"plan_id": plan.plan_id, "steps": [s.step_id for s in plan.steps]}
        if llm is None and isinstance(run, dict) and run:
            # P3 Planner reports usage for all of its calls (incl. JSON/policy repairs) in last_run.
            llm = _Usage(model=run.get("model") or None, input_tokens=run.get("input_tokens"),
                         output_tokens=run.get("output_tokens"), latency_ms=run.get("latency_ms"))
            metadata.update({k: run.get(k) for k in ("calls", "json_repairs", "policy_repairs", "auto_fixes",
                                                      "state_ignored") if k in run})
        self._emit(turn, "planner", event_type, status="CLARIFY" if plan.clarification_needed else "SUCCESS",
                   started=started, output_summary=_summary(plan.model_dump(
                       mode="json", include={"plan_id", "goal", "metric", "date_policy", "chart_spec"})),
                   llm=llm, prompt_version=getattr(self.planner, "prompt_version", None), metadata=metadata)

    def _execute_plan(self, turn: _Turn, plan: AnalysisPlan, workbook_path: str) -> list[ExecutionResult]:
        started = time.perf_counter()
        try:
            fn = getattr(self.executor, "execute_plan", self.executor)
            raw = fn(plan, workbook_path, turn.run_id)
            results = [r if isinstance(r, ExecutionResult) else ExecutionResult.model_validate(r) for r in raw]
        except NotImplementedError as exc:
            raise ComponentUnavailable("executor") from exc
        except Exception as exc:
            results = [_failed_result(turn.run_id, s.step_id, exc) for s in plan.steps]
        for result in results:
            self._emit_result(turn, result, started)
        return results

    def _handle_repair(self, plan, decision, prior_results, workbook_path, run_id, *, turn: _Turn | None = None):
        """Re-run only the affected operation and downstream dependent artifacts after REPAIR.

        Returns the merged list of results (unaffected steps keep their earlier result), or
        None when the action budget cannot cover the re-run.
        """
        by_id = {r.step_id: r for r in prior_results}
        targets = _downstream(plan, decision.failed_step_id) if decision.failed_step_id else [s.step_id for s in plan.steps]
        if turn is not None:
            self._emit(turn, "orchestrator", "transition", status="REPAIR", step_id=decision.failed_step_id,
                       failure_category=decision.error_category, output_summary=decision.instruction,
                       metadata={"rerun_steps": targets, "repair_number": turn.repairs})
            if not self._spend(turn, len(targets)):
                return None
        step_runner = getattr(self.executor, "execute_step", None)
        _sync_executor_context(self.executor, plan)
        started = time.perf_counter()
        if step_runner is None or decision.failed_step_id is None:
            try:
                raw = getattr(self.executor, "execute_plan", self.executor)(plan, workbook_path, run_id)
                fresh = [r if isinstance(r, ExecutionResult) else ExecutionResult.model_validate(r) for r in raw]
            except NotImplementedError as exc:
                raise ComponentUnavailable("executor") from exc
            except Exception as exc:
                fresh = [_failed_result(run_id, sid, exc) for sid in targets]
            for result in fresh:
                if result.step_id in targets or decision.failed_step_id is None:
                    by_id[result.step_id] = result
                    if turn is not None:
                        self._emit_result(turn, result, started)
        else:
            steps = {s.step_id: s for s in plan.steps}
            for step_id in targets:
                try:
                    result = step_runner(steps[step_id], workbook_path, run_id, dict(by_id))
                    if not isinstance(result, ExecutionResult):
                        result = ExecutionResult.model_validate(result)
                except NotImplementedError as exc:
                    raise ComponentUnavailable("executor") from exc
                except Exception as exc:
                    result = _failed_result(run_id, step_id, exc)
                by_id[step_id] = result
                if turn is not None:
                    self._emit_result(turn, result, started)
        order = [s.step_id for s in plan.steps]
        return [by_id[sid] for sid in order if sid in by_id]

    def _emit_result(self, turn: _Turn, result: ExecutionResult, started: float) -> None:
        self._emit(turn, "executor", "execute_step", step_id=result.step_id, status=result.status.value,
                   started=started, failure_category=(result.errors[0].category if result.errors else None),
                   output_summary=_summary({"rows_out": result.row_count_out, "columns": result.result_columns,
                                            "errors": [e.code for e in result.errors]}),
                   metadata={"script_path": result.script_path, "script_hash": result.script_hash,
                             "result_table_path": result.result_table_path, "chart_path": result.chart_path,
                             "duration_seconds": result.duration_seconds, "tool_calls": result.tool_calls})

    def _verify(self, turn, question, plan, results, profile, workbook_path) -> VerificationReport:
        started = time.perf_counter()
        try:
            report = self.verifier(workbook_path, plan, results, profile, question, self.tolerance,
                                   artifact_roots=self.artifact_roots)
        except Exception as exc:  # verification failure is never a pass
            report = VerificationReport(
                verification_id=f"verify-{uuid4().hex[:12]}", verified=False, numeric_checks_passed=False,
                chart_checks_passed=False, semantic_checks_passed=False,
                findings=[VerificationFinding(check_id="plan/verifier", passed=False,
                                              category="not_independently_verifiable",
                                              message=f"Verifier error: {type(exc).__name__}")])
        failed = [f.check_id for f in report.findings if not f.passed]
        self._emit(turn, "verifier", "verify", status="PASS" if report.verified else "FAIL", started=started,
                   failure_category=next((f.category for f in report.findings if is_blocking(f)), None),
                   output_summary=_summary({"verified": report.verified, "failed_checks": failed[:10]}),
                   metadata={"verification_id": report.verification_id,
                             "numeric": report.numeric_checks_passed, "chart": report.chart_checks_passed,
                             "semantic": report.semantic_checks_passed})
        return report

    def _critic(self, turn, question, plan, results, report, profile, count: bool = True) -> CriticDecision:
        started = time.perf_counter()
        if self.critic is None:  # E2 variant: no critic; never VERIFIED
            decision = CriticDecision(decision=Decision.ACCEPT, evidence={"source": "no_critic"},
                                      independent_check_passed=False)
        else:
            try:
                decision = self.critic.evaluate(question, plan, results, report, profile)
            except NotImplementedError as exc:
                raise ComponentUnavailable("critic") from exc
        self._emit(turn, "critic", "critic_decision", status=decision.decision.value, step_id=decision.failed_step_id,
                   started=started, failure_category=decision.error_category,
                   output_summary=_summary({"instruction": decision.instruction, "evidence": decision.evidence}),
                   llm=getattr(self.critic, "last_llm_response", None),
                   prompt_version=getattr(self.critic, "prompt_version", None),
                   metadata={"independent_check_passed": decision.independent_check_passed, "counted": count})
        return decision

    # ====================================================================================
    # Repair helpers and budgets
    # ====================================================================================

    def _spend(self, turn: _Turn, n: int) -> bool:
        if turn.actions + n > self.max_actions:
            self._emit(turn, "orchestrator", "budget_exceeded", status="STOP", failure_category="action_budget",
                       metadata={"actions": turn.actions, "requested": n, "max_actions": self.max_actions})
            return False
        turn.actions += n
        return True

    def _budget_repair(self, turn: _Turn) -> bool:
        if turn.repairs >= self.max_repairs:
            return False
        turn.repairs += 1
        return True

    @staticmethod
    def _patch_plan_for_repair(plan: AnalysisPlan, decision: CriticDecision) -> AnalysisPlan:
        """Apply deterministic, local fixes a REPAIR can make without the Planner (chart type)."""
        suggested = decision.evidence.get("suggested_chart_type") if decision.evidence else None
        if decision.error_category != "inappropriate_chart_type" or not suggested:
            return plan
        try:
            chart_type = ChartType(suggested)
        except ValueError:
            return plan
        spec = plan.chart_spec or ChartSpec(chart_type=chart_type, title=plan.goal)
        updates: dict[str, Any] = {"chart_type": chart_type}
        if chart_type in (ChartType.BAR, ChartType.HORIZONTAL_BAR):
            updates["baseline"] = 0.0  # comparison bars must start at zero
        new_spec = spec.model_copy(update=updates)
        steps = []
        for step in plan.steps:
            if step.tool == "make_chart":
                inputs = dict(step.inputs)
                if "chart_type" in inputs:
                    inputs["chart_type"] = chart_type.value
                step = step.model_copy(update={"operation": chart_type.value, "inputs": inputs})
            steps.append(step)
        return plan.model_copy(update={"chart_spec": new_spec, "steps": steps})

    # ====================================================================================
    # Terminal responses
    # ====================================================================================

    def _accept(self, turn, state, plan, results, report, decision) -> AnswerResponse:
        from verisheet.critic import finalize_report  # local import avoids a hard Qwen dependency cycle

        final = finalize_report(report, decision)
        if not final.verified:
            return self._not_verified(turn, state, plan, final, decision,
                                      "the critic accepted but independent verification did not pass")
        analysis = next((r for s in plan.steps for r in results if r.step_id == s.step_id and s.tool == "run_analysis"), None)
        chart = next((r for s in plan.steps for r in results if r.step_id == s.step_id and s.tool == "make_chart"), None)
        response = AnswerResponse(
            status=AnswerStatus.VERIFIED,
            answer=compose_answer(plan, final),
            assumptions=_assumptions(plan, final),
            result_table_path=analysis.result_table_path if analysis else None,
            chart_path=chart.chart_path if chart else None,
            chart_spec=(chart.chart_spec if chart and chart.chart_spec else plan.chart_spec),
            verification=final,
            trace_id=turn.trace_id,
            workbook_id=turn.workbook_id,
        )
        self._emit(turn, "orchestrator", "transition", status="ACCEPT")
        return self._finish(turn, response, state, plan, None)

    def _clarify(self, turn, state, plan, question_text: str, decision: CriticDecision | None = None) -> AnswerResponse:
        response = AnswerResponse(
            status=AnswerStatus.CLARIFICATION_REQUIRED,
            answer="I need more information before I can answer without guessing.",
            assumptions=list(plan.assumptions) if plan else [],
            clarification_question=summarise_text(question_text, 500),
            trace_id=turn.trace_id, workbook_id=turn.workbook_id,
        )
        self._emit(turn, "orchestrator", "transition", status="CLARIFY",
                   failure_category=decision.error_category if decision else "clarification_requested")
        return self._finish(turn, response, state, plan, decision.error_category if decision else None)

    def _not_verified(self, turn, state, plan, report, decision, reason: str) -> AnswerResponse:
        detail = ""
        if decision is not None and decision.error_category:
            detail = f" Last issue: {decision.error_category.replace('_', ' ')}"
            if decision.instruction:
                detail += f" ({decision.instruction.rstrip('.')})"
            detail += "."
        report = report.model_copy(update={"verified": False}) if report is not None else None
        response = AnswerResponse(
            status=AnswerStatus.NOT_VERIFIED,
            answer=("I could not produce an independently verified answer, so no numbers are shown as verified. "
                    f"Reason: {reason}.{detail} The trace contains the attempted plan and checks."),
            assumptions=list(plan.assumptions) if plan else [],
            verification=report, trace_id=turn.trace_id, workbook_id=turn.workbook_id,
        )
        self._emit(turn, "orchestrator", "transition", status="NOT_VERIFIED", output_summary=reason,
                   failure_category=(decision.error_category if decision else reason.replace(" ", "_")))
        return self._finish(turn, response, state, plan, decision.error_category if decision else reason)

    def _make_failure_response(self, trace_id: str, message: str, workbook_id: str | None = None) -> AnswerResponse:
        """Build a safe FAILED/NOT_VERIFIED response without claiming numerical correctness."""
        return AnswerResponse(status=AnswerStatus.FAILED, answer=summarise_text(message, 500),
                              trace_id=trace_id, workbook_id=workbook_id)

    def _finish(self, turn, response: AnswerResponse, state: AgentState | None, plan, failure: str | None) -> AnswerResponse:
        if state is not None:
            try:
                updated = self.state_store.update_from_verified_response(state, response, plan)
                self.state_store.save(updated)
            except Exception as exc:  # state persistence must not hide the answer
                logger.warning("state save failed: %s", type(exc).__name__)
        self._emit(turn, "orchestrator", "final_response", status=response.status.value, failure_category=failure,
                   output_summary=summarise_text(response.answer, MAX_SUMMARY),
                   metadata={"actions": turn.actions, "repairs": turn.repairs, "replans": turn.replans,
                             "total_latency_ms": round((time.perf_counter() - turn.started) * 1000, 2),
                             "input_tokens": sum(e.input_tokens or 0 for e in turn.events),
                             "output_tokens": sum(e.output_tokens or 0 for e in turn.events),
                             "verification_id": response.verification.verification_id if response.verification else None})
        return response

    # ====================================================================================
    # Tracing
    # ====================================================================================

    def _emit(self, turn: _Turn, actor: str, event_type: str, *, status: str | None = None,
              step_id: str | None = None, input_summary: str | None = None, output_summary: str | None = None,
              failure_category: str | None = None, started: float | None = None, llm=None,
              prompt_version: str | None = None, metadata: dict | None = None) -> None:
        event = TraceEvent(
            trace_id=turn.trace_id, event_id=f"evt-{len(turn.events) + 1:03d}-{uuid4().hex[:6]}",
            actor=actor, event_type=event_type, step_id=step_id, status=status,
            input_summary=input_summary, output_summary=output_summary,
            model_name=getattr(llm, "model", None), prompt_version=prompt_version,
            input_tokens=getattr(llm, "input_tokens", None), output_tokens=getattr(llm, "output_tokens", None),
            latency_ms=(getattr(llm, "latency_ms", None) if llm is not None else
                        (round((time.perf_counter() - started) * 1000, 2) if started is not None else None)),
            failure_category=failure_category,
            metadata={"session_id": turn.session_id, "run_id": turn.run_id, "action_count": turn.actions,
                      **(metadata or {})},
        )
        turn.events.append(event)
        for sink in (self.trace_logger, self._fallback_sink):
            if sink is None:
                continue
            try:
                sink.log_event(event)
                return
            except Exception as exc:  # e.g. TraceLogger stub raising NotImplementedError
                logger.debug("trace sink failed: %s", type(exc).__name__)


# ========================================================================================
# Pure helpers
# ========================================================================================


def _settings():
    try:
        from verisheet.config import get_settings

        return get_settings()
    except Exception:  # pragma: no cover
        return None


def _failed_result(run_id: str, step_id: str, exc: Exception) -> ExecutionResult:
    """Wrap an exception raised by the Executor (rather than returned as a ToolError).

    The P4 Executor raises ``ValueError`` when plan validation fails (unknown sheet/column,
    cycle, budget). That is deterministic, so it is reported as non-retryable plan validation
    (-> REPLAN with the message); anything else is treated as a transient runtime fault.
    """
    deterministic = isinstance(exc, (ValueError, KeyError, TypeError))
    return ExecutionResult(run_id=run_id, step_id=step_id, status=RunStatus.FAILED, errors=[ToolError(
        code="PLAN_REJECTED" if deterministic else "EXECUTOR_EXCEPTION",
        category="plan_validation" if deterministic else "runtime",
        message=summarise_text(str(exc) or type(exc).__name__, 300),
        retryable=not deterministic, failed_step_id=step_id)])


def _sync_executor_context(executor: Any, plan: AnalysisPlan) -> None:
    """Give a step-level re-run the current plan's filters and chart spec.

    The P4 Executor keeps plan-level context (``_active_plan_filters``/``_active_chart_spec``)
    that ``execute_plan`` sets; a targeted ``execute_step`` after a repair must see the patched plan.
    """
    if hasattr(executor, "_active_plan_filters"):
        executor._active_plan_filters = plan.filters
    if hasattr(executor, "_active_chart_spec"):
        executor._active_chart_spec = plan.chart_spec


def _downstream(plan: AnalysisPlan, step_id: str) -> list[str]:
    """The failed step plus every step that (transitively) depends on it, in plan order."""
    affected = {step_id}
    changed = True
    while changed:
        changed = False
        for step in plan.steps:
            if step.step_id not in affected and affected.intersection(step.depends_on):
                affected.add(step.step_id)
                changed = True
    return [s.step_id for s in plan.steps if s.step_id in affected]


def _pre_report(findings: list[dict]) -> VerificationReport:
    return VerificationReport(
        verification_id=f"verify-pre-{uuid4().hex[:10]}", verified=False, numeric_checks_passed=False,
        chart_checks_passed=False, semantic_checks_passed=False,
        findings=[VerificationFinding.model_validate(f) for f in findings],
        verification_method="deterministic_plan_checks")


def _assumptions(plan: AnalysisPlan, report: VerificationReport) -> list[str]:
    seen: list[str] = []
    for text in list(plan.assumptions) + [n for p in report.recomputed_values.values()
                                          if isinstance(p, dict) for n in (p.get("notes") or [])]:
        if text and text not in seen:
            seen.append(str(text))
    return seen[:10]


def compose_answer(plan: AnalysisPlan, report: VerificationReport) -> str:
    """Build the user-facing answer only from independently recomputed values."""
    metric = plan.metric.name if plan.metric else "value"
    parts: list[str] = []
    for step in plan.steps:
        payload = report.recomputed_values.get(step.step_id)
        if not isinstance(payload, dict):
            continue
        scalars = payload.get("scalars") or {}
        if "previous_quarter" in scalars and "quarter" in scalars:
            line = (f"{metric.capitalize()} for {scalars['quarter']} was {_fmt(scalars.get('current'))}, "
                    f"compared with {_fmt(scalars.get('previous'))} in {scalars['previous_quarter']} "
                    f"(change {_fmt(scalars.get('absolute_change'))}")
            pct = scalars.get("pct_change")
            line += f", {pct:+.1f}% relative to |previous|)." if isinstance(pct, (int, float)) else \
                "; percentage change undefined because the previous value is 0)."
            parts.append(line)
            continue
        rows = payload.get("rows") or []
        keys = payload.get("key_columns") or []
        values = payload.get("value_columns") or []
        if not keys and scalars:
            parts.append("; ".join(f"{k}: {_fmt(v)}" for k, v in scalars.items()) + ".")
            continue
        shown = rows[:10]
        items = ["{}: {}".format(", ".join(str(r.get(k)) for k in keys),
                                 ", ".join(_fmt(r.get(v)) for v in values)) for r in shown]
        more = f" (+{payload.get('row_count', len(rows)) - len(shown)} more rows in the table)" \
            if payload.get("row_count", len(rows)) > len(shown) else ""
        parts.append(f"{', '.join(values) or metric} by {', '.join(keys)}: " + "; ".join(items) + more + ".")
    body = " ".join(parts) or "The analysis completed and passed independent verification."
    return body + " All figures were independently recomputed from the source workbook and the chart data was checked against them."


def build_default_orchestrator(critic_mode: str = "hybrid") -> AgentOrchestrator:
    """Wire the production collaborators. Raises ComponentUnavailable if a module is a stub."""
    from verisheet.config import get_settings
    from verisheet.critic import Critic
    from verisheet.profile import profile_workbook
    from verisheet.retrieval import retrieve_context

    settings = get_settings()
    components: dict[str, Any] = {}
    try:
        from verisheet.qwen_client import QwenConfigError
    except ImportError:  # pragma: no cover
        QwenConfigError = ValueError  # type: ignore[assignment]
    for name, factory in (
        ("planner", lambda: __import__("verisheet.planner", fromlist=["Planner"]).Planner()),
        ("executor", lambda: __import__("verisheet.executor", fromlist=["Executor"]).Executor(
            artifact_dir=settings.artifact_dir)),
        ("trace logger", lambda: __import__("verisheet.observability.trace_logger", fromlist=["TraceLogger"])
         .TraceLogger(settings.trace_dir)),
    ):
        try:
            components[name] = factory()
        except QwenConfigError as exc:
            raise ComponentUnavailable("Qwen endpoint (set QWEN_API_BASE, QWEN_API_KEY and QWEN_CA_CERT in .env)") from exc
        except NotImplementedError as exc:
            if name == "trace logger":
                components[name] = None  # orchestrator falls back to its JSONL sink
            else:
                raise ComponentUnavailable(name) from exc
    return AgentOrchestrator(
        profiler=profile_workbook, retriever=retrieve_context, planner=components["planner"],
        executor=components["executor"], critic=Critic(mode=critic_mode),
        state_store=StateStore(str(Path(settings.artifact_dir) / "sessions")),
        trace_logger=components["trace logger"], trace_dir=settings.trace_dir,
    )


def run_agent(
    question: str,
    workbook_reference: str,
    session_id: str,
    orchestrator: AgentOrchestrator | None = None,
) -> AnswerResponse:
    """Canonical application entrypoint returning one stable AnswerResponse.

    The UI calls this function (or an injected orchestrator service), not individual agents.
    It must preserve the bounded action/repair policy and trace all terminal outcomes.
    """
    if orchestrator is None:
        try:
            orchestrator = build_default_orchestrator()
        except ComponentUnavailable as exc:
            return AnswerResponse(status=AnswerStatus.FAILED, trace_id=f"trace-{uuid4().hex}",
                                  answer=f"The {exc} component is not available yet, so no answer was produced.")
        except Exception as exc:
            return AnswerResponse(status=AnswerStatus.FAILED, trace_id=f"trace-{uuid4().hex}",
                                  answer=f"The agent could not start ({type(exc).__name__}). Check the configuration.")
    return orchestrator.answer_question(question, workbook_reference, session_id)
