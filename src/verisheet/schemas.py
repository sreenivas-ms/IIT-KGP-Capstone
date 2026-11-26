"""Canonical API contracts shared by all six contributors.

Do not create parallel, module-specific versions of these payloads. Pydantic models are the
source of truth; JSON Schema files can be exported with scripts/export_schemas.py. Contract
changes require review from all modules that consume the changed model.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (``datetime.utcnow`` is deprecated since Python 3.12)."""
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base class for public payloads; reject undocumented fields to catch interface drift."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FileType(str, Enum):
    XLSX = "xlsx"
    CSV = "csv"


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REPAIR = "REPAIR"
    REPLAN = "REPLAN"
    CLARIFY = "CLARIFY"


class AnswerStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    FAILED = "FAILED"


class ChartType(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    SCATTER = "scatter"
    TABLE = "table"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ColumnProfile(StrictModel):
    """Column metadata from profiling; contains summaries, not unrestricted raw cell content.

    JSON schema fields: name, dtype, null_count, unique_count, candidate_roles, safe_examples.
    """
    name: str
    dtype: str
    null_count: int = 0
    unique_count: int | None = None
    candidate_roles: list[str] = Field(default_factory=list)
    safe_examples: list[str] = Field(default_factory=list, max_length=5)


class SheetProfile(StrictModel):
    """Profile of one worksheet or the synthetic `data` sheet created for a CSV."""
    name: str
    row_count: int
    columns: list[ColumnProfile]
    duplicate_rows: int = 0
    hidden: bool = False
    formula_cell_count: int = 0
    candidate_date_ranges: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class QualityWarning(StrictModel):
    """Machine-readable quality warning surfaced to planner and UI."""
    code: str
    severity: Severity
    message: str
    sheet: str | None = None
    column: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class WorkbookProfile(StrictModel):
    """Stable result of workbook inspection; consumed by Planner, Critic and UI.

    Contract: workbook_id is a SHA-256 fingerprint of the uploaded bytes. `sheets` contains
    every visible/hidden sheet that can be safely inspected. `warnings` carries quality issues.
    No workbook cell text is treated as executable instruction.
    """
    workbook_id: str
    file_name: str
    file_type: FileType
    size_bytes: int
    sheets: list[SheetProfile]
    warnings: list[QualityWarning] = Field(default_factory=list)
    inspected_at: datetime = Field(default_factory=_utcnow)
    profile_version: str = "1.0"


class RetrievedItem(StrictModel):
    """One retrieval hit with provenance for a definition/document chunk."""
    item_id: str
    source_path: str
    title: str
    content: str
    score: float
    source_type: str = "business_definition"


class RetrievedContext(StrictModel):
    """Top-k retrieved context; retrieved text is reference material, never an instruction."""
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    corpus_version: str
    warnings: list[str] = Field(default_factory=list)


class MetricSpec(StrictModel):
    """Canonical metric selection and aggregation chosen by Planner.

    `formula` is descriptive, not executable code. Actual formulas are implemented in an
    allowlisted operation registry and independently recomputed by verification code.
    """
    name: str
    source_columns: list[str]
    aggregation: str
    formula: str | None = None
    unit: str | None = None
    definition_source: str | None = None


class JoinSpec(StrictModel):
    """Join instruction with expected cardinality and explicit key mapping."""
    left_sheet: str
    right_sheet: str
    left_keys: list[str]
    right_keys: list[str]
    expected_cardinality: Literal["one_to_one", "one_to_many", "many_to_one"]
    aggregate_before_join: bool = False
    validate_row_count: bool = True


class ChartSpec(StrictModel):
    """Chart contract. Chart data is populated by Executor from result tables, not by Qwen."""
    chart_type: ChartType
    title: str
    x_column: str | None = None
    y_column: str | None = None
    series_columns: list[str] = Field(default_factory=list)
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    unit: str | None = None
    currency: str | None = None
    sort: str | None = None
    baseline: float | None = 0.0


class AnalysisStep(StrictModel):
    """One dependency-aware operation in a Planner-authored plan."""
    step_id: str
    tool: Literal["inspect_workbook", "sample_rows", "run_analysis", "make_chart"]
    operation: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    validation_checks: list[str] = Field(default_factory=list)


class AnalysisPlan(StrictModel):
    """Planner output consumed by Executor and Critic.

    A plan describes what to do; it does not contain computed answer values or executable
    Python. `clarification_needed=True` requires a non-empty clarification_question and no
    execution should start until the user responds.
    """
    plan_id: str
    goal: str
    metric: MetricSpec | None = None
    source_sheets: list[str] = Field(default_factory=list)
    date_policy: str | None = None
    date_range: dict[str, str] | None = None
    filters: list[dict[str, Any]] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    joins: list[JoinSpec] = Field(default_factory=list)
    steps: list[AnalysisStep] = Field(default_factory=list)
    chart_spec: ChartSpec | None = None
    checks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: str | None = None
    max_actions: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def validate_clarification(self) -> "AnalysisPlan":
        """Enforce a clarification prompt when the plan is explicitly unanswerable/ambiguous."""
        if self.clarification_needed and not self.clarification_question:
            raise ValueError("clarification_question is required when clarification_needed is true")
        return self


class ToolError(StrictModel):
    """Structured error returned by a tool; no raw secrets or unbounded workbook contents."""
    code: str
    category: str
    message: str
    retryable: bool = False
    failed_step_id: str | None = None
    safe_details: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(StrictModel):
    """Executor output for one step or one complete plan.

    Large tables/images are stored as artifact files. The payload contains paths, hashes,
    dimensions, scalar summaries and chart arrays needed for verification.
    """
    run_id: str
    step_id: str
    status: RunStatus
    result_table_path: str | None = None
    result_columns: list[str] = Field(default_factory=list)
    row_count_in: int | None = None
    row_count_out: int | None = None
    scalar_values: dict[str, float | int | str | None] = Field(default_factory=dict)
    script_path: str | None = None
    script_hash: str | None = None
    chart_path: str | None = None
    chart_spec: ChartSpec | None = None
    plotted_data: dict[str, list[Any]] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    tool_calls: list[str] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)


class VerificationFinding(StrictModel):
    """One deterministic or semantic verification finding."""
    check_id: str
    passed: bool
    category: str
    message: str
    expected: Any = None
    observed: Any = None
    severity: Severity = Severity.ERROR


class VerificationReport(StrictModel):
    """Critic verification evidence; VERIFIED is permitted only after required deterministic checks pass."""
    verification_id: str
    verified: bool
    numeric_checks_passed: bool
    chart_checks_passed: bool
    semantic_checks_passed: bool
    findings: list[VerificationFinding] = Field(default_factory=list)
    recomputed_values: dict[str, Any] = Field(default_factory=dict)
    tolerance: float = 1e-9
    verification_method: str = "deterministic_python"


class CriticDecision(StrictModel):
    """Action returned by Critic and interpreted by Orchestrator."""
    decision: Decision
    failed_step_id: str | None = None
    error_category: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = None
    clarification_question: str | None = None
    independent_check_passed: bool = False


class StateMessage(StrictModel):
    """Sanitised user/assistant turn summary used to preserve multi-turn context."""
    role: Literal["user", "assistant"]
    content_summary: str
    created_at: datetime = Field(default_factory=_utcnow)


class AgentState(StrictModel):
    """Session state. Only accepted/verified facts may be stored as reusable analytical context."""
    session_id: str
    workbook_id: str | None = None
    accepted_metric: MetricSpec | None = None
    accepted_period: dict[str, str] | None = None
    accepted_filters: list[dict[str, Any]] = Field(default_factory=list)
    accepted_group_by: list[str] = Field(default_factory=list)
    last_verification_id: str | None = None
    history: list[StateMessage] = Field(default_factory=list)
    revision: int = 0


class AnswerResponse(StrictModel):
    """Stable API response consumed by Streamlit and external callers."""
    status: AnswerStatus
    answer: str
    assumptions: list[str] = Field(default_factory=list)
    result_table_path: str | None = None
    chart_path: str | None = None
    chart_spec: ChartSpec | None = None
    verification: VerificationReport | None = None
    clarification_question: str | None = None
    trace_id: str
    workbook_id: str | None = None

    @model_validator(mode="after")
    def prevent_unverified_verified_status(self) -> "AnswerResponse":
        """Prevent VERIFIED status unless the attached deterministic report is passing."""
        if self.status == AnswerStatus.VERIFIED:
            if self.verification is None or not self.verification.verified:
                raise ValueError("VERIFIED response requires a passing VerificationReport")
        return self


class TraceEvent(StrictModel):
    """Append-only observability event; summaries must be PII-minimised and bounded."""
    trace_id: str
    event_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str
    event_type: str
    step_id: str | None = None
    status: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    failure_category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(StrictModel):
    """Normalised output from the shared Qwen API wrapper."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str | None = None


class ExperimentVariant(str, Enum):
    DIRECT = "direct"
    PLANNER_EXECUTOR = "planner_executor"
    LLM_CRITIC = "llm_critic"
    DETERMINISTIC_CRITIC = "deterministic_critic"
    FULL = "full"


class ExperimentConfig(StrictModel):
    """One reproducible evaluation run configuration."""
    system: ExperimentVariant
    split: Literal["dev", "test"]
    question_file: str
    model_name: str
    prompt_version: str
    temperature: float = 0.0
    max_actions: int = 10
    max_repairs: int = 2
