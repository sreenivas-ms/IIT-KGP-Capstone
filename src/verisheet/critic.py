"""Hybrid Critic owned by Person 5: deterministic checks plus a separate Qwen critic prompt.

Decision policy (highest priority first):

1. Planner asked for clarification                      -> CLARIFY
2. Deterministic findings routed to CLARIFY            -> CLARIFY  (missing data/definition)
3. Deterministic findings routed to REPLAN             -> REPLAN   (wrong approach/period/metric)
4. Deterministic findings routed to REPAIR             -> REPAIR   (local execution/chart defect)
5. Numbers could not be independently recomputed       -> REPLAN with ``not_independently_verifiable``
   (the Orchestrator ends with NOT_VERIFIED if it persists) — never ACCEPT
6. All deterministic checks pass -> Qwen semantic review (hybrid mode). Qwen may *veto*
   (REPAIR chart type / REPLAN intent / CLARIFY) but can never turn a failed or missing numeric
   check into ACCEPT. If Qwen is unavailable or returns malformed JSON, the deterministic
   verdict stands and a warning is recorded.

Modes support the experiment variants: ``hybrid`` (E5), ``deterministic`` (E4) and ``llm_only``
(E3, where ACCEPT can end the loop but the answer is never VERIFIED because no independent
numeric check exists).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from verisheet.qwen_client import QwenClient, strip_think
from verisheet.schemas import (
    AnalysisPlan, ChartType, CriticDecision, Decision, ExecutionResult, LLMResponse, Severity,
    VerificationFinding, VerificationReport, WorkbookProfile,
)
from verisheet.tools.verify import ROUTING, CAT_NOT_VERIFIABLE, is_blocking

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_PROMPT_CHARS = 12_000   # keeps prompt + 512 output tokens well inside the 8,192-token window
MAX_LLM_TOKENS = 512
MAX_EVIDENCE_FINDINGS = 8
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)

_CLARIFY_QUESTIONS = {
    "metric_unavailable": ("This workbook has no profit, cost or expense fields, so profit cannot be computed. "
                           "Can you provide cost/expense data, or should I report a different metric such as sales value?"),
    "proxy_metric_substitution": ("Profit needs cost data, and revenue is not profit. Can you provide cost/expense "
                                  "data, or should I report sales value instead?"),
    "period_not_in_data": ("The requested period is outside the dates in this workbook. Which period within the "
                           "available data should I use?"),
    "currency_conflict": ("The data contains more than one currency and no exchange rate. Which exchange rates "
                          "(and as-of date) should I use, or should I report each currency separately?"),
    "ambiguous_request": "Could you clarify the exact metric, period and grouping you want?",
}
_INSTRUCTIONS = {
    "execution_failed": "Re-run the failed step after fixing the runtime/tool error; keep the plan unchanged.",
    "missing_artifact": "Re-run the step so it writes its result table artifact.",
    "untrusted_artifact_path": "Write the result table inside the configured artifact directory.",
    "result_schema_mismatch": "Re-run the step so the result table has the planned key and value columns.",
    "value_mismatch": "Recompute the step exactly as planned (filters, aggregation, period) and rewrite the table.",
    "wrong_aggregation": "Recompute using the plan's declared aggregation, then re-render the chart.",
    "missing_or_extra_category": "Recompute the grouping so every group from the source appears exactly once.",
    "row_count_mismatch": "Recompute the step without duplicating or dropping rows.",
    "chart_value_mismatch": "Re-render the chart from the verified result table without altering values.",
    "chart_category_mismatch": "Re-render the chart so every table category is plotted exactly once.",
    "chart_field_mismatch": "Re-render the chart using the result table's actual x/y columns.",
    "chart_order": "Re-render with chronological order for time axes or ranked order for rankings.",
    "axis_truncation": "Re-render the bar chart with the value axis starting at zero.",
    "inappropriate_chart_type": "Re-render the chart with the suggested chart type.",
    "chart_missing": "Re-run the chart step so it exports machine-readable plotted_data.",
    "wrong_period": "Revise the plan's date policy/range to the correct (last completed or requested) quarter.",
    "wrong_metric": "Revise the plan's metric/aggregation to match the question.",
    "wrong_margin_formula": "Use margin = sum(Profit)/sum(Sales) via grouped_ratio, not a mean of row margins.",
    "unknown_sheet_or_column": "Revise the plan to use only sheets/columns present in the WorkbookProfile.",
    "invalid_plan": "Return a well-formed, acyclic plan within the action budget.",
    "unsafe_plan_content": "Use only typed operations; plans must not contain code.",
    "join_grain": "Aggregate the right-hand sheet to the join key before joining (aggregate_before_join=true).",
    "join_multiplication": "Fix the join grain: aggregate duplicates by key before joining.",
    "not_independently_verifiable": ("Use only canonical operations (group_aggregate, top_n, time_aggregate, "
                                     "quarter_comparison, grouped_ratio, filter_rows, safe_join) so results can be verified."),
    "semantic_mismatch": "Revise the plan so it answers the question that was asked.",
}


class CriticMode(str, Enum):
    """Critic variants used by the experiment runner."""

    HYBRID = "hybrid"
    DETERMINISTIC = "deterministic"
    LLM_ONLY = "llm_only"


def load_prompt(prompt_path: str | Path) -> tuple[str, str, str]:
    """Load a versioned system prompt; return (text, version, sha256-prefix).

    Relative paths are tried from the current directory first and then from the repository root.
    """
    path = Path(prompt_path)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError(f"critic prompt is empty: {candidate}")
            match = re.search(r"VERSION\s+([0-9][0-9A-Za-z.\-]*)", text.splitlines()[0])
            version = match.group(1) if match else "unversioned"
            return text, version, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    raise FileNotFoundError(f"critic prompt not found: {prompt_path}")


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (strips <think> blocks and fences)."""
    text = _FENCE_RE.sub("", strip_think(_THINK_RE.sub("", raw or ""))).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in critic response")


def _step_of(check_id: str) -> str | None:
    head = check_id.split("/", 1)[0]
    return None if head == "plan" else head


def _bounded(value: Any, limit: int = 600) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…"


class Critic:
    """Evaluate intent/plan semantics and independently verified execution evidence.

    Qwen may assess semantic match and chart appropriateness. Deterministic verification must
    control numeric correctness. Critic may ACCEPT only after all mandatory checks pass.
    """

    def __init__(
        self,
        client: QwenClient | None = None,
        prompt_path: str = "prompts/critic.txt",
        mode: CriticMode | str = CriticMode.HYBRID,
        max_tokens: int = MAX_LLM_TOKENS,
    ):
        """Initialise shared Qwen client and versioned critic prompt.

        The Qwen client is created lazily: missing credentials do not break deterministic
        verification, they only mark the semantic review as unavailable.
        """
        self.mode = CriticMode(mode)
        self.prompt_path = Path(prompt_path)
        self.system_prompt, self.prompt_version, self.prompt_hash = load_prompt(prompt_path)
        self.max_tokens = max(64, min(int(max_tokens), 1024))  # endpoint cap is 1,024
        self._client = client
        self._client_error: str | None = None
        self.last_llm_response: LLMResponse | None = None
        self.last_semantic_review: dict[str, Any] | None = None

    # ------------------------------------------------------------------------------------
    # Qwen access
    # ------------------------------------------------------------------------------------

    @property
    def client(self) -> QwenClient | None:
        """Shared Qwen client, or None when it cannot be configured."""
        if self._client is None and self._client_error is None and self.mode != CriticMode.DETERMINISTIC:
            try:
                self._client = QwenClient()
            except Exception as exc:  # missing credentials or bad configuration
                self._client_error = type(exc).__name__
                logger.warning("semantic critic unavailable: %s", self._client_error)
        return self._client

    # ------------------------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        plan: AnalysisPlan,
        execution_results: list[ExecutionResult],
        verification: VerificationReport,
        workbook_profile: WorkbookProfile,
    ) -> CriticDecision:
        """Return ACCEPT/REPAIR/REPLAN/CLARIFY with precise step and evidence.

        Never trust LLM arithmetic. When independent recomputation cannot be completed the
        decision is never ACCEPT (the Orchestrator reports NOT_VERIFIED). REPAIR is for local
        execution/artifact defects; REPLAN for a wrong analytical approach; CLARIFY for
        missing/ambiguous source information.
        """
        self.last_llm_response = None
        self.last_semantic_review = None
        if plan.clarification_needed:
            return CriticDecision(
                decision=Decision.CLARIFY, error_category="ambiguous_request",
                clarification_question=plan.clarification_question,
                evidence={"source": "planner"}, independent_check_passed=False)

        if self.mode == CriticMode.LLM_ONLY:
            return self._llm_only(question, plan, verification)

        failing = [f for f in verification.findings if is_blocking(f)]
        if failing:
            return self._deterministic_decision(failing, verification, plan)

        if not (verification.numeric_checks_passed and verification.chart_checks_passed):
            return CriticDecision(
                decision=Decision.REPLAN, error_category=CAT_NOT_VERIFIABLE,
                instruction=_INSTRUCTIONS[CAT_NOT_VERIFIABLE],
                evidence={"source": "deterministic", "numeric_checks_passed": verification.numeric_checks_passed,
                          "chart_checks_passed": verification.chart_checks_passed},
                independent_check_passed=False)

        if self.mode == CriticMode.DETERMINISTIC:
            return self._accept(verification, {"semantic_review": "skipped (deterministic mode)"})

        review = self.semantic_review(question, plan, self._answer_summary(plan, verification))
        if not review.get("available"):
            return self._accept(verification, {"semantic_review": "unavailable", "reason": review.get("reason")})
        veto = self._veto_from_review(review, plan)
        if veto is not None:
            return veto
        return self._accept(verification, {"semantic_review": "accepted", "rationale": review.get("rationale")})

    def _accept(self, verification: VerificationReport, extra: dict[str, Any]) -> CriticDecision:
        return CriticDecision(
            decision=Decision.ACCEPT,
            evidence={"source": "deterministic", "verification_id": verification.verification_id, **extra},
            independent_check_passed=bool(verification.verified),
        )

    def _deterministic_decision(self, failing: list[VerificationFinding], verification: VerificationReport,
                                plan: AnalysisPlan) -> CriticDecision:
        routed = {"CLARIFY": [], "REPLAN": [], "REPAIR": []}
        for finding in failing:
            routed[ROUTING.get(finding.category, "REPLAN")].append(finding)
        for name in ("CLARIFY", "REPLAN", "REPAIR"):
            if routed[name]:
                primary = routed[name][0]
                break
        decision = Decision(name)
        evidence: dict[str, Any] = {
            "source": "deterministic",
            "verification_id": verification.verification_id,
            "check_id": primary.check_id,
            "message": primary.message,
            "expected": _bounded(primary.expected),
            "observed": _bounded(primary.observed),
            "failed_checks": [f.check_id for f in failing][:MAX_EVIDENCE_FINDINGS],
        }
        failed_step = _step_of(primary.check_id)
        if primary.category == "inappropriate_chart_type":
            evidence["suggested_chart_type"] = primary.expected or ChartType.HORIZONTAL_BAR.value
            if failed_step is None:  # pre-execution finding: point REPAIR at the chart step
                failed_step = next((s.step_id for s in plan.steps if s.tool == "make_chart"), None)
        return CriticDecision(
            decision=decision,
            failed_step_id=failed_step,
            error_category=primary.category,
            evidence=evidence,
            instruction=_INSTRUCTIONS.get(primary.category, primary.message),
            clarification_question=(_CLARIFY_QUESTIONS.get(primary.category, _CLARIFY_QUESTIONS["ambiguous_request"])
                                    if decision == Decision.CLARIFY else None),
            independent_check_passed=False,
        )

    def _veto_from_review(self, review: dict[str, Any], plan: AnalysisPlan) -> CriticDecision | None:
        decision = review.get("decision")
        if decision == "ACCEPT" and review.get("intent_match", True) and review.get("chart_appropriate", True):
            return None
        evidence = {"source": "llm_semantic", "rationale": review.get("rationale"),
                    "llm_decision": decision}
        chart_step = next((s.step_id for s in plan.steps if s.tool == "make_chart"), None)
        suggested = review.get("suggested_chart_type")
        if decision == "CLARIFY" and review.get("clarification_question"):
            return CriticDecision(decision=Decision.CLARIFY, error_category=review.get("error_category") or "ambiguous_request",
                                  clarification_question=review["clarification_question"], evidence=evidence,
                                  independent_check_passed=False)
        if (decision == "REPAIR" or review.get("chart_appropriate") is False) and suggested and chart_step \
                and review.get("intent_match", True):
            evidence["suggested_chart_type"] = suggested
            return CriticDecision(decision=Decision.REPAIR, failed_step_id=chart_step,
                                  error_category="inappropriate_chart_type", evidence=evidence,
                                  instruction=review.get("instruction") or _INSTRUCTIONS["inappropriate_chart_type"],
                                  independent_check_passed=False)
        return CriticDecision(decision=Decision.REPLAN, failed_step_id=review.get("failed_step_id"),
                              error_category="semantic_mismatch", evidence=evidence,
                              instruction=review.get("instruction") or _INSTRUCTIONS["semantic_mismatch"],
                              independent_check_passed=False)

    def _llm_only(self, question: str, plan: AnalysisPlan, verification: VerificationReport) -> CriticDecision:
        review = self.semantic_review(question, plan, self._answer_summary(plan, verification))
        if not review.get("available"):
            return CriticDecision(decision=Decision.ACCEPT,
                                  evidence={"source": "llm_only", "semantic_review": "unavailable"},
                                  independent_check_passed=False)
        veto = self._veto_from_review(review, plan)
        if veto is not None:
            return veto.model_copy(update={"evidence": {**veto.evidence, "source": "llm_only"}})
        return CriticDecision(decision=Decision.ACCEPT, evidence={"source": "llm_only", "rationale": review.get("rationale")},
                              independent_check_passed=False)

    # ------------------------------------------------------------------------------------
    # Semantic review with Qwen
    # ------------------------------------------------------------------------------------

    def semantic_review(self, question: str, plan: AnalysisPlan, answer_summary: dict) -> dict:
        """Use Qwen critic prompt to judge intent match, assumptions and chart appropriateness.

        Returns a dict with ``available`` (bool) plus, when available, ``decision``,
        ``intent_match``, ``chart_appropriate``, ``failed_step_id``, ``error_category``,
        ``suggested_chart_type``, ``instruction``, ``clarification_question`` and ``rationale``.
        Numbers in the reply are ignored: this review is never numeric evidence.
        """
        client = self.client
        if client is None:
            review = {"available": False, "reason": self._client_error or "no client"}
            self.last_semantic_review = review
            return review
        user_prompt = self._user_prompt(question, plan, answer_summary)
        attempts = 0
        last_error = "unknown"
        while attempts < 2:  # one retry for malformed JSON
            attempts += 1
            try:
                response = client.complete(self.system_prompt, user_prompt, temperature=0.0,
                                           max_tokens=self.max_tokens)
            except Exception as exc:  # network/429/503 already translated by the client
                last_error = type(exc).__name__
                break
            self.last_llm_response = response
            try:
                review = self._normalise_review(extract_json_object(response.content))
                review.update({"available": True, "attempts": attempts})
                self.last_semantic_review = review
                return review
            except ValueError as exc:
                last_error = f"malformed_json: {exc}"
                user_prompt = (user_prompt + "\n\nYour previous reply was not one valid JSON object. "
                               "Reply with ONLY the JSON object described in the rules. /no_think")
        review = {"available": False, "reason": last_error, "attempts": attempts}
        self.last_semantic_review = review
        return review

    @staticmethod
    def _normalise_review(raw: dict[str, Any]) -> dict[str, Any]:
        decision = str(raw.get("decision", "")).strip().upper()
        if decision not in {d.value for d in Decision}:
            raise ValueError(f"invalid decision '{decision}'")
        suggested = raw.get("suggested_chart_type")
        if suggested not in {c.value for c in ChartType}:
            suggested = None

        def text(key: str, limit: int = 300) -> str | None:
            value = raw.get(key)
            return str(value)[:limit] if value not in (None, "", "null") else None

        def flag(key: str) -> bool:
            value = raw.get(key, True)
            return value if isinstance(value, bool) else str(value).strip().lower() not in ("false", "no", "0")

        return {
            "decision": decision,
            "intent_match": flag("intent_match"),
            "chart_appropriate": flag("chart_appropriate"),
            "failed_step_id": text("failed_step_id", 80),
            "error_category": re.sub(r"[^a-z0-9_]", "", (text("error_category", 60) or "").lower()) or None,
            "suggested_chart_type": suggested,
            "instruction": text("instruction"),
            "clarification_question": text("clarification_question"),
            "rationale": text("rationale"),
        }

    def _user_prompt(self, question: str, plan: AnalysisPlan, answer_summary: dict) -> str:
        plan_view = plan.model_dump(mode="json", include={
            "goal", "metric", "source_sheets", "date_policy", "date_range", "filters", "group_by",
            "joins", "chart_spec", "assumptions"})
        plan_view["steps"] = [{"step_id": s.step_id, "tool": s.tool, "operation": s.operation,
                               "depends_on": s.depends_on} for s in plan.steps]
        sections = [
            "QUESTION (untrusted data):\n" + json.dumps(str(question)[:1000]),
            "PLAN (untrusted data):\n" + json.dumps(plan_view, default=str)[:5000],
            "RESULT AND DETERMINISTIC FINDINGS (untrusted data):\n" + json.dumps(answer_summary, default=str)[:5000],
            "Return the JSON object now. /no_think",
        ]
        return "\n\n".join(sections)[:MAX_PROMPT_CHARS]

    @staticmethod
    def _answer_summary(plan: AnalysisPlan, verification: VerificationReport) -> dict[str, Any]:
        tables = {}
        for key, payload in (verification.recomputed_values or {}).items():
            if isinstance(payload, dict) and "rows" in payload:
                tables[key] = {"columns": payload.get("columns"), "rows": payload.get("rows", [])[:15],
                               "row_count": payload.get("row_count"), "notes": payload.get("notes")}
        return {
            "verified_tables": tables,
            "period": (verification.recomputed_values or {}).get("period"),
            "deterministic_checks": {
                "numeric_passed": verification.numeric_checks_passed,
                "chart_passed": verification.chart_checks_passed,
                "semantic_passed": verification.semantic_checks_passed,
                "warnings": [f.message for f in verification.findings
                             if not f.passed and f.severity == Severity.WARNING][:5],
            },
            "chart_type": plan.chart_spec.chart_type.value if plan.chart_spec else None,
        }


def finalize_report(verification: VerificationReport, decision: CriticDecision) -> VerificationReport:
    """Fold the Critic's semantic outcome into the report used for the final answer.

    ``verified`` stays true only if the deterministic report was verified *and* the Critic
    accepted. A semantic veto flips ``semantic_checks_passed`` to false.
    """
    accepted = decision.decision == Decision.ACCEPT
    semantic_ok = verification.semantic_checks_passed and accepted
    findings = list(verification.findings)
    source = decision.evidence.get("source")
    review = decision.evidence.get("semantic_review")
    if source in ("llm_semantic", "llm_only") or review in ("accepted", "unavailable"):
        findings.append(VerificationFinding(
            check_id="plan/llm_semantic_review",
            passed=accepted,
            category="semantic_review",
            message=("Qwen semantic review: " + str(review or decision.evidence.get("rationale")
                                                    or decision.instruction or decision.decision.value))[:500],
            severity=Severity.WARNING if review == "unavailable" or not accepted else Severity.INFO,
        ))
    method = verification.verification_method
    if source in ("llm_semantic",) or review == "accepted":
        method = "deterministic_python+qwen_semantic"
    return verification.model_copy(update={
        "semantic_checks_passed": semantic_ok,
        "verified": bool(verification.verified and accepted and decision.independent_check_passed),
        "findings": findings,
        "verification_method": method,
    })
