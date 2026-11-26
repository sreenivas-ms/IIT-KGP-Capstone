"""Deterministic verification tools owned by Person 5; independent from Qwen critique.

This module is the numeric authority of VeriSheet. It never trusts the Executor's numbers or
the Qwen critic's opinion. Instead it:

1. ``verify_plan_semantics`` - checks a plan *before* execution against the question and the
   WorkbookProfile (metric availability, aggregation words, margin formula, completed-quarter
   policy, explicit quarter/year requests, join grain, currency mixing, chart intent, plan
   structure and code-injection markers).
2. ``recompute_check`` - reloads the original workbook and recomputes a ``run_analysis`` step
   with *separately implemented* canonical operations, then compares the Executor's result
   table and scalars with the independent result (keys, values, period, row counts).
3. ``verify_chart_data`` - compares plotted x/y arrays with the verified table (categories,
   values, ordering, baseline, field mapping and chart-type appropriateness).
4. ``verify_execution`` - runs all of the above for a full plan and returns one
   ``VerificationReport`` whose ``verified`` flag is true only when every required check passed.

Finding conventions (consumed by ``critic.py``):

* ``check_id`` is ``"<step_id>/<check>"`` for step findings and ``"plan/<check>"`` for plan
  findings, so the Critic can name the failed step precisely.
* ``category`` is one of the ``CAT_*`` constants; ``ROUTING`` maps it to REPAIR/REPLAN/CLARIFY.

If a step cannot be recomputed (unknown operation, unreadable workbook, missing inputs) a
``not_independently_verifiable`` finding is emitted and ``verified`` stays false: an
unverifiable number is never reported as verified.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

from verisheet.schemas import (
    AnalysisPlan,
    AnalysisStep,
    ChartSpec,
    ChartType,
    ExecutionResult,
    RunStatus,
    Severity,
    VerificationFinding,
    VerificationReport,
    WorkbookProfile,
)

# --------------------------------------------------------------------------------------------
# Failure categories and routing
# --------------------------------------------------------------------------------------------

CAT_EXECUTION_FAILED = "execution_failed"
CAT_MISSING_ARTIFACT = "missing_artifact"
CAT_UNTRUSTED_ARTIFACT = "untrusted_artifact_path"
CAT_RESULT_SCHEMA = "result_schema_mismatch"
CAT_VALUE_MISMATCH = "value_mismatch"
CAT_WRONG_AGGREGATION = "wrong_aggregation"
CAT_WRONG_PERIOD = "wrong_period"
CAT_MISSING_CATEGORY = "missing_or_extra_category"
CAT_ROW_COUNT = "row_count_mismatch"
CAT_JOIN_MULTIPLICATION = "join_multiplication"
CAT_CHART_VALUE = "chart_value_mismatch"
CAT_CHART_CATEGORY = "chart_category_mismatch"
CAT_CHART_FIELD = "chart_field_mismatch"
CAT_CHART_ORDER = "chart_order"
CAT_AXIS_TRUNCATION = "axis_truncation"
CAT_CHART_TYPE = "inappropriate_chart_type"
CAT_CHART_MISSING = "chart_missing"
CAT_WRONG_METRIC = "wrong_metric"
CAT_WRONG_MARGIN = "wrong_margin_formula"
CAT_UNKNOWN_REFERENCE = "unknown_sheet_or_column"
CAT_INVALID_PLAN = "invalid_plan"
CAT_UNSAFE_PLAN = "unsafe_plan_content"
CAT_JOIN_GRAIN = "join_grain"
CAT_METRIC_UNAVAILABLE = "metric_unavailable"
CAT_PROXY_METRIC = "proxy_metric_substitution"
CAT_PERIOD_UNAVAILABLE = "period_not_in_data"
CAT_CURRENCY = "currency_conflict"
CAT_AMBIGUOUS = "ambiguous_request"
CAT_NOT_VERIFIABLE = "not_independently_verifiable"
CAT_SEMANTIC = "semantic_mismatch"
CAT_TOOL_REJECTED = "tool_rejected_inputs"

#: Category -> orchestrator transition. Unlisted failing categories are treated as REPLAN.
ROUTING: dict[str, str] = {
    CAT_EXECUTION_FAILED: "REPAIR",
    CAT_MISSING_ARTIFACT: "REPAIR",
    CAT_UNTRUSTED_ARTIFACT: "REPAIR",
    CAT_RESULT_SCHEMA: "REPAIR",
    CAT_VALUE_MISMATCH: "REPAIR",
    CAT_WRONG_AGGREGATION: "REPAIR",  # executor disagrees with the plan's declared aggregation
    CAT_MISSING_CATEGORY: "REPAIR",
    CAT_ROW_COUNT: "REPAIR",
    CAT_CHART_VALUE: "REPAIR",
    CAT_CHART_CATEGORY: "REPAIR",
    CAT_CHART_FIELD: "REPAIR",
    CAT_CHART_ORDER: "REPAIR",
    CAT_AXIS_TRUNCATION: "REPAIR",
    CAT_CHART_TYPE: "REPAIR",
    CAT_CHART_MISSING: "REPAIR",
    CAT_WRONG_PERIOD: "REPLAN",
    CAT_WRONG_METRIC: "REPLAN",
    CAT_WRONG_MARGIN: "REPLAN",
    CAT_UNKNOWN_REFERENCE: "REPLAN",
    CAT_INVALID_PLAN: "REPLAN",
    CAT_UNSAFE_PLAN: "REPLAN",
    CAT_JOIN_GRAIN: "REPLAN",
    CAT_JOIN_MULTIPLICATION: "REPLAN",
    CAT_NOT_VERIFIABLE: "REPLAN",
    CAT_SEMANTIC: "REPLAN",
    CAT_TOOL_REJECTED: "REPLAN",  # deterministic tool validation error: re-running cannot help
    CAT_METRIC_UNAVAILABLE: "CLARIFY",
    CAT_PROXY_METRIC: "CLARIFY",
    CAT_PERIOD_UNAVAILABLE: "CLARIFY",
    CAT_CURRENCY: "CLARIFY",
    CAT_AMBIGUOUS: "CLARIFY",
}

#: Canonical operations this verifier can recompute independently.
SUPPORTED_OPERATIONS = frozenset({
    "group_aggregate", "filter_rows", "top_n", "time_aggregate", "quarter_comparison",
    "safe_join", "grouped_ratio",
})

_AGG_ALIASES = {
    "sum": "sum", "total": "sum",
    "mean": "mean", "avg": "mean", "average": "mean",
    "count": "count", "size": "size",
    "min": "min", "max": "max", "median": "median",
    "nunique": "nunique", "count_distinct": "nunique", "distinct_count": "nunique",
}
_ALT_AGGS = ("sum", "mean", "count", "min", "max", "median", "nunique")
_FREQ_ALIASES = {
    "m": "M", "month": "M", "monthly": "M", "me": "M", "ms": "M",
    "q": "Q", "quarter": "Q", "quarterly": "Q", "qe": "Q", "qs": "Q",
    "y": "Y", "a": "Y", "year": "Y", "yearly": "Y", "annual": "Y", "ye": "Y", "ys": "Y",
}
_FILTER_OPS = {
    "==", "=", "eq", "!=", "ne", ">", "gt", ">=", "ge", "gte", "<", "lt", "<=", "le", "lte", "in", "not_in",
    "startswith", "not_startswith", "between", "is_null", "not_null", "contains",
}
_UNSAFE_MARKERS = re.compile(
    r"(__\w+__|\bimport\s+\w|\bexec\s*\(|\beval\s*\(|\bsubprocess\b|\bos\.system|\bopen\s*\("
    r"|\bsocket\b|\bpickle\b|\blambda\b)",
    re.IGNORECASE,
)

MAX_TABLE_ROWS = 200_000
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
MAX_EVIDENCE_ROWS = 50
DEFAULT_TOLERANCE = 1e-9


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _norm(name: Any) -> str:
    """Normalise a column/sheet name for tolerant matching (case, spaces, punctuation)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _finding(
    check_id: str,
    passed: bool,
    category: str,
    message: str,
    expected: Any = None,
    observed: Any = None,
    severity: Severity | str = Severity.ERROR,
) -> dict[str, Any]:
    """Build a VerificationFinding-compatible dict (validated by the shared schema)."""
    sev = severity if isinstance(severity, Severity) else Severity(severity)
    if passed and sev == Severity.ERROR:
        sev = Severity.INFO
    return VerificationFinding(
        check_id=check_id,
        passed=passed,
        category=category,
        message=message[:500],
        expected=_json_safe(expected),
        observed=_json_safe(observed),
        severity=sev,
    ).model_dump(mode="json")


def is_blocking(finding: dict[str, Any] | VerificationFinding) -> bool:
    """A finding blocks verification when it failed with error severity."""
    if isinstance(finding, VerificationFinding):
        return (not finding.passed) and finding.severity == Severity.ERROR
    return (not finding["passed"]) and finding["severity"] == Severity.ERROR.value


def _close(a: Any, b: Any, tolerance: float) -> bool:
    """Numeric equality with an explicit tolerance (relative and absolute)."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isnan(fa) or math.isnan(fb):
        return False
    tol = max(float(tolerance), DEFAULT_TOLERANCE)
    return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column; unparseable values become NaT (counted and disclosed by callers)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


_QUARTER_RE_1 = re.compile(r"(\d{4})\D{0,3}q([1-4])", re.IGNORECASE)
_QUARTER_RE_2 = re.compile(r"q([1-4])\D{0,3}(\d{4})", re.IGNORECASE)


def _norm_key(value: Any, kind: str) -> str:
    """Normalise a grouping key so that e.g. ``2024Q3``, ``2024-Q3`` and ``Q3 2024`` match."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<null>"
    if isinstance(value, pd.Period):
        value = str(value)
    text = str(value).strip()
    if kind == "Q":
        m = _QUARTER_RE_1.search(text)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"
        m = _QUARTER_RE_2.search(text)
        if m:
            return f"{m.group(2)}Q{m.group(1)}"
        ts = pd.to_datetime(text, errors="coerce")
        return str(ts.to_period("Q")) if not pd.isna(ts) else text.upper()
    if kind == "M":
        m = re.match(r"^(\d{4})-(\d{1,2})$", text)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
        ts = pd.to_datetime(text, errors="coerce")
        return ts.strftime("%Y-%m") if not pd.isna(ts) else text
    if kind == "Y":
        m = re.search(r"(\d{4})", text)
        return m.group(1) if m else text
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        return str(int(float(value)))
    return text


# --------------------------------------------------------------------------------------------
# Independent source loading
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_frames_cached(path_str: str, size: int, mtime_ns: int) -> dict[str, pd.DataFrame]:
    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        return {"data": pd.read_csv(path, low_memory=False)}
    frames = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return {str(name): frame for name, frame in frames.items()}


def load_source_frames(workbook_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the original workbook independently of the Executor.

    The shared upload guard (extension, size, macro and zip-bomb checks) runs first. Returns a
    copy of each sheet so callers cannot mutate the cache.

    Raises:
        ValueError: unsupported, oversized or unreadable workbook.
    """
    from verisheet.profile import validate_upload  # lazy: keeps this module import-light

    path = Path(workbook_path).expanduser().resolve()
    validate_upload(path)
    stat = path.stat()
    frames = _load_frames_cached(str(path), stat.st_size, stat.st_mtime_ns)
    return {name: frame.copy() for name, frame in frames.items()}


def _resolve_sheet(frames: dict[str, pd.DataFrame], name: str | None) -> tuple[str, pd.DataFrame]:
    if name is not None:
        if name in frames:
            return name, frames[name]
        for key, frame in frames.items():
            if _norm(key) == _norm(name):
                return key, frame
    if len(frames) == 1:  # CSV is exposed as the synthetic sheet "data"
        key = next(iter(frames))
        return key, frames[key]
    raise KeyError(f"sheet '{name}' not found" if name else "sheet not specified")


def _resolve_column(frame: pd.DataFrame, name: str) -> str:
    if name in frame.columns:
        return name
    matches = [c for c in frame.columns if _norm(c) == _norm(name)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"column '{name}' not found")


# --------------------------------------------------------------------------------------------
# Canonical recomputation
# --------------------------------------------------------------------------------------------


@dataclass
class CanonicalResult:
    """Independent result of one analysis step."""

    table: pd.DataFrame
    key_columns: list[str]
    value_columns: list[str]
    key_kind: str = "C"  # C=category, Q=quarter, M=month, Y=year
    scalars: dict[str, Any] = field(default_factory=dict)
    period: dict[str, str] | None = None
    row_count_in: int | None = None
    notes: list[str] = field(default_factory=list)
    data: pd.DataFrame | None = None  # filtered source rows (diagnoses wrong aggregations)
    group_columns: list[str] = field(default_factory=list)
    metric_columns: dict[str, str] = field(default_factory=dict)  # output -> source column
    positional: bool = False  # compare rows in order (raw-row rankings have no unique key)


def _get(inputs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in inputs and inputs[name] not in (None, ""):
            return inputs[name]
    return default


def _agg_name(value: Any) -> str:
    agg = _AGG_ALIASES.get(str(value or "sum").strip().lower())
    if agg is None:
        raise ValueError(f"unsupported aggregation '{value}'")
    return agg


def _apply_agg(target: Any, agg: str) -> Any:
    return target.size() if agg == "size" else target.agg(agg)


def _filters_for(plan: AnalysisPlan, step: AnalysisStep) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(plan.filters) + list(step.inputs.get("filters") or []):
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            filters.append(item)
    return filters


def _looks_like_date(value: Any) -> bool:
    if isinstance(value, (pd.Timestamp, datetime)):
        return True
    return bool(isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value))


def _apply_filters(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply typed filters with a closed operator set; unknown operators fail closed."""
    out = frame
    for item in filters:
        column = _resolve_column(out, str(_get(item, "column", "field", "col")))
        op = str(_get(item, "op", "operator", default="==")).strip().lower()
        if op not in _FILTER_OPS:
            raise ValueError(f"unsupported filter operator '{op}'")
        value = item.get("value")
        series = out[column]
        if op in ("==", "=", "eq"):
            mask = series == value
        elif op in ("!=", "ne"):
            mask = series != value
        elif op in (">", "gt"):
            mask = series > value
        elif op in (">=", "ge", "gte"):
            mask = series >= value
        elif op in ("<", "lt"):
            mask = series < value
        elif op in ("<=", "le", "lte"):
            mask = series <= value
        elif op == "in":
            mask = series.isin(list(value or []))
        elif op == "not_in":
            mask = ~series.isin(list(value or []))
        elif op == "startswith":
            mask = series.astype(str).str.startswith(str(value))
        elif op == "not_startswith":
            mask = ~series.astype(str).str.startswith(str(value))
        elif op == "contains":
            mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        elif op == "between":
            low, high = list(value)[:2]
            if pd.api.types.is_datetime64_any_dtype(series) or _looks_like_date(low):
                parsed = _parse_dates(series)
                mask = (parsed >= pd.Timestamp(low)) & (parsed <= pd.Timestamp(high))
            else:
                mask = (series >= low) & (series <= high)
        elif op == "is_null":
            mask = series.isna()
        else:  # not_null
            mask = series.notna()
        out = out.loc[mask.fillna(False).astype(bool)]
    return out


def _apply_date_range(frame: pd.DataFrame, plan: AnalysisPlan, date_col: str | None) -> pd.DataFrame:
    if not plan.date_range or not date_col:
        return frame
    start, end = plan.date_range.get("start"), plan.date_range.get("end")
    parsed = _parse_dates(frame[date_col])
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= parsed >= pd.Timestamp(start)
    if end:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():  # an inclusive end date covers the whole day
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= parsed <= end_ts
    return frame.loc[mask]


def _metrics_for(step: AnalysisStep, plan: AnalysisPlan, frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Return {output_name: (source_column, aggregation)} from step inputs or the plan metric."""
    inputs = step.inputs
    metrics: dict[str, tuple[str, str]] = {}
    raw = inputs.get("metrics")
    if isinstance(raw, dict) and raw:
        for column, agg in raw.items():
            metrics[str(column)] = (_resolve_column(frame, column), _agg_name(agg))
        return metrics
    if isinstance(raw, list) and raw:
        for item in raw:
            source = _resolve_column(frame, str(_get(item, "column", "source")))
            out = str(_get(item, "name", "output", default=item.get("column")))
            metrics[out] = (source, _agg_name(_get(item, "aggregation", "agg")))
        return metrics
    column = _get(inputs, "metric_column", "value_column", "column", "by")
    agg = _get(inputs, "aggregation", "agg")
    if column is None and plan.metric and plan.metric.source_columns:
        column = plan.metric.source_columns[0]
    if agg is None and plan.metric:
        agg = plan.metric.aggregation
    if column is None:
        raise ValueError("no metric column in step inputs or plan metric")
    source = _resolve_column(frame, str(column))
    metrics[str(source)] = (source, _agg_name(agg))  # outputs are named after the source column
    return metrics


def find_data_through(frames: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    """Return explicit reporting-close metadata (a ``Data_Through``-like column), if present."""
    candidates = {"datathrough", "reportingclose", "reportingcutoff", "datathroughdate", "cutoffdate"}
    found: list[pd.Timestamp] = []
    for frame in frames.values():
        for column in frame.columns:
            if _norm(column) in candidates:
                parsed = _parse_dates(frame[column]).dropna()
                if not parsed.empty:
                    found.append(parsed.max())
    return max(found) if found else None


def completed_quarter(dates: pd.Series, data_through: pd.Timestamp | None = None) -> tuple[pd.Period, str]:
    """Return the last completed quarter and the assumption to disclose.

    Explicit ``Data_Through`` metadata wins: its quarter counts as complete only when the date is
    the quarter's last day. Otherwise the quarter before the one containing max(observed date)
    is used (documented convention).
    """
    if data_through is not None and not pd.isna(data_through):
        quarter = data_through.to_period("Q")
        if data_through.normalize() < quarter.end_time.normalize():
            quarter = quarter - 1
        return quarter, f"Data_Through metadata ({data_through.date()}) defines the reporting close."
    parsed = _parse_dates(dates).dropna()
    if parsed.empty:
        raise ValueError("no valid dates to determine the completed quarter")
    return parsed.max().to_period("Q") - 1, (
        "No reporting-close metadata: the last completed quarter is the one before the quarter "
        f"containing the latest observed date ({parsed.max().date()})."
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def _canonical(operation: str, step: AnalysisStep, plan: AnalysisPlan,
               frames: dict[str, pd.DataFrame]) -> CanonicalResult:
    """Recompute one step with canonical logic independent from the Executor."""
    if operation == "safe_join":
        return _canonical_join(step, plan, frames)
    inputs = step.inputs
    sheet_name = _get(inputs, "sheet", "source_sheet")
    if sheet_name is None and plan.source_sheets:
        sheet_name = plan.source_sheets[0]
    _, frame = _resolve_sheet(frames, sheet_name)
    rows_in = len(frame)
    date_col_name = _get(inputs, "date_column", "date_col")
    date_col = _resolve_column(frame, str(date_col_name)) if date_col_name else None
    data = _apply_filters(frame, _filters_for(plan, step))
    if operation != "quarter_comparison":
        data = _apply_date_range(data, plan, date_col)

    if operation == "filter_rows":
        return CanonicalResult(table=pd.DataFrame({"row_count": [len(data)]}), key_columns=[],
                               value_columns=["row_count"], scalars={"row_count": len(data)},
                               row_count_in=rows_in, data=data)

    if operation in ("group_aggregate", "top_n"):
        group_cols = [_resolve_column(data, g) for g in _as_list(_get(inputs, "group_by", "groupby"))]
        if operation == "top_n" and not group_cols:
            return _canonical_raw_top_n(inputs, data, rows_in)
        metrics = _metrics_for(step, plan, data)
        if group_cols:
            grouped = data.groupby(group_cols, dropna=False)
            table = pd.DataFrame({o: _apply_agg(grouped[s], a) for o, (s, a) in metrics.items()}).reset_index()
        else:
            table = pd.DataFrame({o: [_apply_agg(data[s], a)] for o, (s, a) in metrics.items()})
        result = CanonicalResult(table=table, key_columns=list(group_cols), value_columns=list(metrics),
                                 row_count_in=rows_in, data=data, group_columns=list(group_cols),
                                 metric_columns={o: s for o, (s, _) in metrics.items()})
        if not group_cols:
            result.scalars = {o: table[o].iloc[0] for o in metrics}
        if operation == "top_n":
            n = int(_get(inputs, "n", "top_n", "limit", default=5))
            ascending = bool(_get(inputs, "ascending", default=False))
            by = _get(inputs, "by", "sort_by")
            sort_col = next((c for c in table.columns if by is not None and _norm(c) == _norm(by)),
                            list(metrics)[0])
            result.table = table.sort_values([sort_col] + group_cols,
                                             ascending=[ascending] + [True] * len(group_cols),
                                             kind="mergesort").head(n).reset_index(drop=True)
            result.notes.append(f"top {n} by {sort_col} ({'ascending' if ascending else 'descending'})")
        return result

    if operation == "time_aggregate":
        if date_col is None:
            raise ValueError("time_aggregate requires date_column")
        freq_raw = str(_get(inputs, "frequency", "freq", "period", default="M")).strip().lower()
        freq = _FREQ_ALIASES.get(freq_raw)
        if freq is None:
            raise ValueError(f"unsupported frequency '{freq_raw}'")
        metrics = _metrics_for(step, plan, data)
        parsed = _parse_dates(data[date_col])
        valid = data.loc[parsed.notna()].copy()
        period_col = {"M": "month", "Q": "quarter", "Y": "year"}[freq]
        valid[period_col] = parsed.loc[parsed.notna()].dt.to_period(freq).astype(str)
        extra_cols = [_resolve_column(valid, g) for g in _as_list(_get(inputs, "group_by"))]
        grouped = valid.groupby([period_col] + extra_cols, dropna=False)
        table = pd.DataFrame({o: _apply_agg(grouped[s], a) for o, (s, a) in metrics.items()}).reset_index()
        table = table.sort_values([period_col] + extra_cols, kind="mergesort").reset_index(drop=True)
        notes = []
        dropped = int(parsed.isna().sum())
        if dropped:
            notes.append(f"{dropped} rows with unparseable dates were excluded")
        period = None
        if not table.empty:
            period = {"start": str(table[period_col].iloc[0]), "end": str(table[period_col].iloc[-1]),
                      "frequency": freq}
        return CanonicalResult(table=table, key_columns=[period_col] + extra_cols, value_columns=list(metrics),
                               key_kind=freq, row_count_in=rows_in, notes=notes, period=period,
                               data=valid, group_columns=[period_col] + extra_cols,
                               metric_columns={o: s for o, (s, _) in metrics.items()})

    if operation == "quarter_comparison":
        if date_col is None:
            raise ValueError("quarter_comparison requires date_column")
        out_name, (src, agg) = next(iter(_metrics_for(step, plan, data).items()))
        parsed = _parse_dates(data[date_col])
        target, assumption = completed_quarter(parsed, find_data_through(frames))
        explicit = _get(inputs, "quarter", "target_quarter")
        if explicit:
            target = pd.Period(_norm_key(explicit, "Q"), freq="Q")
            assumption = f"Quarter {target} was requested explicitly."
        previous = target - 1
        quarters = parsed.dt.to_period("Q")
        values = {}
        for q in (previous, target):
            subset = data.loc[quarters == q, src]
            values[str(q)] = float(len(subset)) if agg == "size" else float(_apply_agg(subset, agg))
        prev_v, cur_v = values[str(previous)], values[str(target)]
        pct = None if abs(prev_v) < 1e-12 else 100.0 * (cur_v - prev_v) / abs(prev_v)
        table = pd.DataFrame({"quarter": [str(previous), str(target)], out_name: [prev_v, cur_v]})
        valid = data.loc[parsed.notna()].copy()
        valid["quarter"] = quarters.loc[parsed.notna()].astype(str)
        valid = valid.loc[valid["quarter"].isin([str(previous), str(target)])]
        return CanonicalResult(
            table=table, key_columns=["quarter"], value_columns=[out_name], key_kind="Q",
            scalars={"quarter": str(target), "current": cur_v, out_name: cur_v,
                     "previous_quarter": str(previous), "previous": prev_v,
                     "absolute_change": cur_v - prev_v, "pct_change": pct},
            period={"quarter": str(target), "previous_quarter": str(previous),
                    "start": str(target.start_time.date()), "end": str(target.end_time.date())},
            row_count_in=rows_in,
            notes=[assumption, "Percentage change uses |previous| as the denominator (documented "
                               "convention; undefined when the previous value is 0)."],
            data=valid, group_columns=["quarter"], metric_columns={out_name: src},
        )

    if operation == "grouped_ratio":
        group_cols = [_resolve_column(data, g) for g in _as_list(_get(inputs, "group_by"))]
        num = _resolve_column(data, str(_get(inputs, "numerator", "numerator_column")))
        den = _resolve_column(data, str(_get(inputs, "denominator", "denominator_column")))
        out = str(_get(inputs, "output_column", "output", "ratio_name", default="ratio"))
        if group_cols:
            sums = data.groupby(group_cols, dropna=False)[[num, den]].sum()
            table = (sums[num] / sums[den].replace(0, np.nan)).rename(out).reset_index()
        else:
            d = data[den].sum()
            table = pd.DataFrame({out: [data[num].sum() / d if d != 0 else np.nan]})
        return CanonicalResult(
            table=table, key_columns=group_cols, value_columns=[out], row_count_in=rows_in,
            notes=["Ratio computed as sum(numerator)/sum(denominator); zero denominators give null."],
            scalars={} if group_cols else {out: table[out].iloc[0]},
        )

    raise ValueError(f"operation '{operation}' has no canonical recomputation")


def _canonical_raw_top_n(inputs: dict[str, Any], data: pd.DataFrame, rows_in: int) -> CanonicalResult:
    """Executor contract for top_n without group_by: rank raw rows by ``by`` (stable order)."""
    by = _resolve_column(data, str(_get(inputs, "by", "metric_column", "sort_by")))
    n = int(_get(inputs, "n", "top_n", "limit", default=5))
    ascending = bool(_get(inputs, "ascending", default=False))
    table = data.sort_values(by, ascending=ascending, kind="mergesort", na_position="last").head(n)
    return CanonicalResult(table=table.reset_index(drop=True), key_columns=[], value_columns=[by],
                           row_count_in=rows_in, positional=True,
                           notes=[f"top {n} raw rows by {by} ({'ascending' if ascending else 'descending'}); "
                                  "rows are not aggregated"])


def _join_spec(step: AnalysisStep, plan: AnalysisPlan) -> dict[str, Any]:
    raw = step.inputs.get("join_spec")
    if isinstance(raw, dict):
        return raw
    spec = plan.joins[0].model_dump() if plan.joins else {}
    for key in ("left_sheet", "right_sheet", "left_keys", "right_keys", "expected_cardinality"):
        if step.inputs.get(key) is not None:
            spec[key] = step.inputs[key]
    return spec


def _canonical_join(step: AnalysisStep, plan: AnalysisPlan, frames: dict[str, pd.DataFrame]) -> CanonicalResult:
    """Canonical left join with independent key-uniqueness checks for the declared cardinality."""
    spec = _join_spec(step, plan)
    _, left = _resolve_sheet(frames, spec.get("left_sheet"))
    _, right = _resolve_sheet(frames, spec.get("right_sheet"))
    lk = [_resolve_column(left, k) for k in _as_list(spec.get("left_keys"))]
    rk = [_resolve_column(right, k) for k in _as_list(spec.get("right_keys"))]
    if not lk or len(lk) != len(rk):
        raise ValueError("join keys are required and must pair up")
    cardinality = spec.get("expected_cardinality") or "many_to_one"
    left_unique = not left.duplicated(subset=lk).any()
    right_unique = not right.duplicated(subset=rk).any()
    joined = left.merge(right, how="left", left_on=lk, right_on=rk)
    # "many_to_one"/"one_to_one" require unique RIGHT keys (the "one" side); "one_to_many" and
    # "one_to_one" require unique LEFT keys.
    grain_ok = (right_unique or cardinality == "one_to_many") and (left_unique or cardinality == "many_to_one")
    return CanonicalResult(
        table=pd.DataFrame({"row_count": [len(joined)]}), key_columns=[], value_columns=["row_count"],
        scalars={"row_count": len(joined), "left_row_count": len(left), "right_keys_unique": bool(right_unique),
                 "left_keys_unique": bool(left_unique), "cardinality": cardinality, "grain_ok": bool(grain_ok)},
        row_count_in=len(left),
        notes=[f"Declared {cardinality}; left keys unique={left_unique}, right keys unique={right_unique}."],
    )


# --------------------------------------------------------------------------------------------
# Artifact reading
# --------------------------------------------------------------------------------------------


def _artifact_roots(extra: Iterable[str | Path] | None) -> list[Path]:
    roots = [Path(r).expanduser().resolve() for r in (extra or [])]
    try:
        from verisheet.config import get_settings

        roots.append(Path(get_settings().artifact_dir).expanduser().resolve())
    except Exception:  # pragma: no cover - settings are optional for pure verification
        roots.append(Path("artifacts").resolve())
    return roots


def read_result_table(path: str | Path, roots: Iterable[str | Path] | None = None) -> pd.DataFrame:
    """Read an Executor result table (CSV/JSON/Parquet) from an approved artifact root.

    Raises:
        PermissionError: the path escapes every approved artifact root (path traversal guard).
        FileNotFoundError / ValueError: missing, oversized or unsupported artifact.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not any(resolved == root or root in resolved.parents for root in _artifact_roots(roots)):
        raise PermissionError("result table is outside the approved artifact directories")
    if not resolved.is_file():
        raise FileNotFoundError("result table not found")
    if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("result table exceeds the artifact size limit")
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved, nrows=MAX_TABLE_ROWS)
    if suffix == ".json":
        return pd.read_json(resolved)
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    raise ValueError(f"unsupported result table format '{suffix}'")


# --------------------------------------------------------------------------------------------
# Table comparison
# --------------------------------------------------------------------------------------------


def _match_columns(expected: CanonicalResult, observed: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Map expected key/value columns to observed column names; return (mapping, missing)."""
    mapping: dict[str, str] = {}
    missing: list[str] = []
    obs_norm = {_norm(c): c for c in observed.columns}
    for col in expected.key_columns + expected.value_columns:
        if col in observed.columns:
            mapping[col] = col
        elif _norm(col) in obs_norm:
            mapping[col] = obs_norm[_norm(col)]
        else:
            missing.append(col)
    # Positional fallback: one unmatched value column vs one spare numeric column (and keys alike).
    for wanted, numeric in ((expected.value_columns, True), (expected.key_columns, False)):
        unmatched = [c for c in missing if c in wanted]
        spare = [c for c in observed.columns if c not in mapping.values()
                 and pd.api.types.is_numeric_dtype(observed[c]) == numeric]
        if len(unmatched) == 1 and len(spare) == 1:
            mapping[unmatched[0]] = spare[0]
            missing.remove(unmatched[0])
    return mapping, missing


def _kinds(expected: CanonicalResult, n: int) -> list[str]:
    return [expected.key_kind if i == 0 else "C" for i in range(n)]


def _key_tuple(row: pd.Series, columns: list[str], kinds: list[str]) -> tuple[str, ...]:
    return tuple(_norm_key(row[c], k) for c, k in zip(columns, kinds))


def _diagnose_aggregation(expected: CanonicalResult, value_col: str, observed_values: dict,
                          tolerance: float) -> str | None:
    """If observed values equal another aggregation of the same rows, name it (e.g. mean(Sales))."""
    if expected.data is None or value_col not in expected.metric_columns:
        return None
    src = expected.metric_columns[value_col]
    kinds = _kinds(expected, len(expected.group_columns))
    for agg in _ALT_AGGS:
        try:
            if expected.group_columns:
                alt = expected.data.groupby(expected.group_columns, dropna=False)[src].agg(agg).reset_index()
                alt_map = {_key_tuple(r, expected.group_columns, kinds): r[src] for _, r in alt.iterrows()}
            else:
                alt_map = {(): expected.data[src].agg(agg)}
        except (TypeError, ValueError):
            continue
        common = [k for k in observed_values if k in alt_map]
        if common and all(_close(observed_values[k], alt_map[k], tolerance) for k in common):
            return f"{agg}({src})"
    return None


def compare_tables(expected: CanonicalResult, observed: pd.DataFrame, step_id: str,
                   tolerance: float) -> list[dict[str, Any]]:
    """Compare an Executor result table with the canonical recomputation."""
    findings: list[dict[str, Any]] = []
    mapping, missing = _match_columns(expected, observed)
    if missing:
        return [_finding(f"{step_id}/result_columns", False, CAT_RESULT_SCHEMA,
                         "Result table is missing required columns.",
                         expected=expected.key_columns + expected.value_columns,
                         observed=[str(c) for c in observed.columns])]
    findings.append(_finding(f"{step_id}/result_columns", True, CAT_RESULT_SCHEMA,
                             "Result table has the required columns."))
    if expected.positional:
        return findings + _compare_positional(expected, observed, mapping, step_id, tolerance)

    keys = expected.key_columns
    kinds = _kinds(expected, len(keys))
    obs_keys = [mapping[k] for k in keys]
    exp_rows = {_key_tuple(r, keys, kinds): r for _, r in expected.table.iterrows()}
    obs_rows: dict[tuple, pd.Series] = {}
    obs_order: list[tuple] = []
    duplicates = 0
    for _, r in observed.iterrows():
        key = _key_tuple(r, obs_keys, kinds)
        duplicates += key in obs_rows
        obs_rows[key] = r
        obs_order.append(key)
    if duplicates:
        findings.append(_finding(f"{step_id}/duplicate_keys", False, CAT_ROW_COUNT,
                                 "Result table repeats group keys (possible join/grain error).",
                                 observed=duplicates))

    missing_keys = [k for k in exp_rows if k not in obs_rows]
    extra_keys = [k for k in obs_rows if k not in exp_rows]
    if missing_keys or extra_keys:
        category = CAT_WRONG_PERIOD if expected.key_kind in ("Q", "M", "Y") else CAT_MISSING_CATEGORY
        findings.append(_finding(
            f"{step_id}/result_keys", False, category,
            "Result groups/periods differ from the independent recomputation.",
            expected=[list(k) for k in list(exp_rows)[:MAX_EVIDENCE_ROWS]],
            observed=[list(k) for k in list(obs_rows)[:MAX_EVIDENCE_ROWS]]))
    else:
        findings.append(_finding(f"{step_id}/result_keys", True, CAT_MISSING_CATEGORY,
                                 "Result groups/periods match the independent recomputation."))

    for value_col in expected.value_columns:
        obs_col = mapping[value_col]
        mismatches, observed_values = [], {}
        for key, exp_row in exp_rows.items():
            if key not in obs_rows:
                continue
            obs_v = obs_rows[key][obs_col]
            observed_values[key] = obs_v
            if not _close(obs_v, exp_row[value_col], tolerance):
                mismatches.append({"key": list(key), "expected": _json_safe(exp_row[value_col]),
                                   "observed": _json_safe(obs_v)})
        if not mismatches:
            findings.append(_finding(f"{step_id}/values:{value_col}", True, CAT_VALUE_MISMATCH,
                                     f"All '{value_col}' values match within tolerance {tolerance}."))
            continue
        diagnosis = _diagnose_aggregation(expected, value_col, observed_values, tolerance)
        src = expected.metric_columns.get(value_col, value_col)
        if diagnosis:
            findings.append(_finding(f"{step_id}/values:{value_col}", False, CAT_WRONG_AGGREGATION,
                                     f"Values equal {diagnosis}, not the planned aggregation of {src}.",
                                     expected=mismatches[:MAX_EVIDENCE_ROWS], observed=diagnosis))
        else:
            findings.append(_finding(f"{step_id}/values:{value_col}", False, CAT_VALUE_MISMATCH,
                                     f"{len(mismatches)} value(s) differ from the independent recomputation.",
                                     expected=[m["expected"] for m in mismatches[:MAX_EVIDENCE_ROWS]],
                                     observed=mismatches[:MAX_EVIDENCE_ROWS]))

    if keys and not missing_keys and not extra_keys:
        if expected.key_kind in ("Q", "M", "Y") and obs_order != sorted(obs_order):
            findings.append(_finding(f"{step_id}/order", False, CAT_CHART_ORDER,
                                     "Time periods are not in chronological order.",
                                     expected=[list(k) for k in sorted(obs_order)],
                                     observed=[list(k) for k in obs_order], severity=Severity.WARNING))
        elif any(n.startswith("top ") for n in expected.notes) and obs_order != list(exp_rows):
            findings.append(_finding(f"{step_id}/order", False, CAT_CHART_ORDER,
                                     "Ranking order differs from the canonical ranking.",
                                     expected=[list(k) for k in exp_rows], observed=[list(k) for k in obs_order]))
    return findings


def _compare_positional(expected: CanonicalResult, observed: pd.DataFrame, mapping: dict[str, str],
                        step_id: str, tolerance: float) -> list[dict[str, Any]]:
    """Row-by-row comparison for ordered raw-row results (e.g. top_n without grouping)."""
    if len(observed) != len(expected.table):
        return [_finding(f"{step_id}/row_count", False, CAT_ROW_COUNT, "Ranking returned a different number of rows.",
                         expected=len(expected.table), observed=len(observed))]
    findings = []
    for col in expected.value_columns:
        exp_vals = expected.table[col].tolist()
        obs_vals = observed[mapping[col]].tolist()
        bad = [{"rank": i + 1, "expected": _json_safe(e), "observed": _json_safe(o)}
               for i, (e, o) in enumerate(zip(exp_vals, obs_vals)) if not _close(o, e, tolerance)]
        if bad and sorted(map(float, obs_vals)) == sorted(map(float, exp_vals)):
            findings.append(_finding(f"{step_id}/order", False, CAT_CHART_ORDER,
                                     "Ranking order differs from the canonical ranking.",
                                     expected=exp_vals[:MAX_EVIDENCE_ROWS], observed=obs_vals[:MAX_EVIDENCE_ROWS]))
        elif bad:
            findings.append(_finding(f"{step_id}/values:{col}", False, CAT_VALUE_MISMATCH,
                                     f"{len(bad)} ranked value(s) differ from the independent recomputation.",
                                     expected=[b["expected"] for b in bad[:MAX_EVIDENCE_ROWS]],
                                     observed=bad[:MAX_EVIDENCE_ROWS]))
        else:
            findings.append(_finding(f"{step_id}/values:{col}", True, CAT_VALUE_MISMATCH,
                                     f"Ranked '{col}' values match in order."))
    return findings


def _compare_scalars(expected: CanonicalResult, result: ExecutionResult, tolerance: float) -> list[dict]:
    findings: list[dict[str, Any]] = []
    exp_norm = {_norm(k): (k, v) for k, v in expected.scalars.items()}
    for name, observed in result.scalar_values.items():
        match = exp_norm.get(_norm(name))
        if match is None:
            continue
        key, exp_value = match
        is_period = "quarter" in _norm(key)
        if exp_value is None or observed is None:
            ok = exp_value is None and observed is None
        elif is_period or isinstance(exp_value, str) or isinstance(observed, str):
            kind = "Q" if is_period else "C"
            ok = _norm_key(exp_value, kind) == _norm_key(observed, kind)
        else:
            ok = _close(observed, exp_value, tolerance)
        findings.append(_finding(
            f"{result.step_id}/scalar:{name}", ok, CAT_WRONG_PERIOD if is_period else CAT_VALUE_MISMATCH,
            f"Scalar '{name}' {'matches' if ok else 'differs from'} the independent recomputation.",
            expected=exp_value, observed=observed))
    return findings


# --------------------------------------------------------------------------------------------
# Public API: numeric recomputation
# --------------------------------------------------------------------------------------------


def _report(findings: list[dict], *, numeric: bool, chart: bool, semantic: bool,
            recomputed: dict, tolerance: float, method: str = "deterministic_python") -> VerificationReport:
    blocking = any(is_blocking(f) for f in findings)
    return VerificationReport(
        verification_id=f"verify-{uuid4().hex[:12]}",
        verified=bool(numeric and chart and semantic and not blocking),
        numeric_checks_passed=numeric,
        chart_checks_passed=chart,
        semantic_checks_passed=semantic,
        findings=[VerificationFinding.model_validate(f) for f in findings],
        recomputed_values=_json_safe(recomputed),
        tolerance=tolerance,
        verification_method=method,
    )


_REFERENCE_ERROR = re.compile(r"unknown (column|sheet|group_by|metric|dependenc)|not found|does not exist", re.I)


def tool_error_category(error: Any) -> str:
    """Map an Executor ToolError to a verification category.

    Retryable errors (timeouts, transient runtime faults) are local -> REPAIR. A deterministic
    validation error (the Executor raises ``ValueError``/``KeyError`` on bad inputs and reports
    ``retryable=False``) will fail identically on a re-run, so it is routed to REPLAN with the
    tool's message as feedback. ``DEPENDENCY_FAILED`` is attributed to the upstream step.
    """
    if error is None:
        return CAT_EXECUTION_FAILED
    if getattr(error, "code", "") == "DEPENDENCY_FAILED":
        return CAT_EXECUTION_FAILED
    if getattr(error, "retryable", False):
        return CAT_EXECUTION_FAILED
    category = str(getattr(error, "category", "")).lower()
    message = str(getattr(error, "message", ""))
    if category in ("schema", "validation", "unknown_column", "unknown_sheet", "plan_validation", "semantic",
                    "keyerror") or _REFERENCE_ERROR.search(message):
        return CAT_UNKNOWN_REFERENCE
    if category in ("valueerror", "typeerror", "dependency"):
        return CAT_TOOL_REJECTED
    return CAT_EXECUTION_FAILED


def recompute_step(
    workbook_path: str,
    plan: AnalysisPlan,
    execution_result: ExecutionResult,
    tolerance: float = DEFAULT_TOLERANCE,
    artifact_roots: Iterable[str | Path] | None = None,
) -> tuple[list[dict[str, Any]], CanonicalResult | None]:
    """Recompute one run_analysis step; return findings and the canonical result (if any)."""
    step_id = execution_result.step_id
    if execution_result.status != RunStatus.SUCCESS or execution_result.errors:
        error = execution_result.errors[0] if execution_result.errors else None
        category = tool_error_category(error)
        return [_finding(
            f"{step_id}/execution", False, category,
            f"Step did not complete: {error.message if error else execution_result.status.value}",
            observed={"code": error.code, "category": error.category, "retryable": error.retryable}
            if error else execution_result.status.value)], None

    step = next((s for s in plan.steps if s.step_id == step_id), None)
    if step is None:
        return [_finding(f"{step_id}/plan_step", False, CAT_INVALID_PLAN,
                         "Execution result refers to a step that is not in the plan.")], None
    operation = (step.operation or "").strip().lower()
    if operation not in SUPPORTED_OPERATIONS:
        return [_finding(f"{step_id}/recompute", False, CAT_NOT_VERIFIABLE,
                         f"Operation '{step.operation}' has no canonical recomputation; result stays NOT_VERIFIED.",
                         expected=sorted(SUPPORTED_OPERATIONS), observed=step.operation)], None
    try:
        expected = _canonical(operation, step, plan, load_source_frames(workbook_path))
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return [_finding(f"{step_id}/recompute", False, CAT_NOT_VERIFIABLE,
                         f"Independent recomputation could not run: {str(exc)[:200]}")], None

    findings: list[dict[str, Any]] = []
    if operation == "safe_join":
        observed_rows, exp_rows = execution_result.row_count_out, expected.scalars["row_count"]
        if not expected.scalars["grain_ok"]:
            findings.append(_finding(
                f"{step_id}/join_grain", False, CAT_JOIN_MULTIPLICATION,
                "Join keys are not unique on the side the declared cardinality requires; measures "
                "would be multiplied. Aggregate that side to the key grain first.",
                expected=expected.scalars["cardinality"],
                observed={"left_keys_unique": expected.scalars["left_keys_unique"],
                          "right_keys_unique": expected.scalars["right_keys_unique"],
                          "joined_rows": exp_rows, "left_rows": expected.scalars["left_row_count"]}))
            return findings, expected
        if observed_rows is None:
            findings.append(_finding(f"{step_id}/join_rows", False, CAT_NOT_VERIFIABLE,
                                     "Join result did not report row_count_out."))
        elif observed_rows > exp_rows or (observed_rows > expected.scalars["left_row_count"]
                                          and expected.scalars["cardinality"] != "one_to_many"):
            findings.append(_finding(f"{step_id}/join_rows", False, CAT_JOIN_MULTIPLICATION,
                                     "Join produced more rows than the left-table grain: measures would be inflated.",
                                     expected=exp_rows, observed=observed_rows))
        elif observed_rows != exp_rows:
            findings.append(_finding(f"{step_id}/join_rows", False, CAT_ROW_COUNT,
                                     "Join row count differs from the canonical join.",
                                     expected=exp_rows, observed=observed_rows))
        else:
            findings.append(_finding(f"{step_id}/join_rows", True, CAT_JOIN_MULTIPLICATION,
                                     "Join preserved the left-table grain."))
        return findings, expected

    if operation == "filter_rows":
        obs = execution_result.row_count_out
        ok = obs is not None and int(obs) == expected.scalars["row_count"]
        return [_finding(f"{step_id}/row_count", ok, CAT_ROW_COUNT,
                         "Filtered row count " + ("matches." if ok else "differs."),
                         expected=expected.scalars["row_count"], observed=obs)], expected

    if not execution_result.result_table_path:
        return [_finding(f"{step_id}/result_table", False, CAT_MISSING_ARTIFACT,
                         "Step produced no result table to verify.")], expected
    try:
        observed = read_result_table(execution_result.result_table_path, artifact_roots)
    except PermissionError as exc:
        return [_finding(f"{step_id}/result_table", False, CAT_UNTRUSTED_ARTIFACT, str(exc))], expected
    except (OSError, ValueError) as exc:
        return [_finding(f"{step_id}/result_table", False, CAT_MISSING_ARTIFACT,
                         f"Result table unreadable: {type(exc).__name__}")], expected

    findings.extend(compare_tables(expected, observed, step_id, tolerance))
    findings.extend(_compare_scalars(expected, execution_result, tolerance))
    if execution_result.row_count_out is not None and execution_result.row_count_out != len(observed):
        findings.append(_finding(f"{step_id}/row_count_out", False, CAT_ROW_COUNT,
                                 "Reported row_count_out does not match the result table.",
                                 expected=len(observed), observed=execution_result.row_count_out,
                                 severity=Severity.WARNING))
    return findings, expected


def canonical_payload(expected: CanonicalResult) -> dict[str, Any]:
    """Bounded, JSON-safe summary of a canonical result for reports, answers and traces."""
    table = expected.table.head(MAX_EVIDENCE_ROWS)
    return _json_safe({
        "columns": [str(c) for c in table.columns],
        "rows": table.to_dict(orient="records"),
        "row_count": int(len(expected.table)),
        "key_columns": expected.key_columns,
        "value_columns": expected.value_columns,
        "scalars": expected.scalars,
        "period": expected.period,
        "notes": expected.notes,
    })


def recompute_check(
    workbook_path: str,
    plan: AnalysisPlan,
    execution_result: ExecutionResult,
    tolerance: float = 1e-9,
    *,
    artifact_roots: Iterable[str | Path] | None = None,
) -> VerificationReport:
    """Independently recompute canonical numbers from source workbook and compare outputs.

    Uses canonical metric logic in this module, not the Executor's cached result or Qwen's
    explanation. Validates metric, period, filters, aggregation, join grain and ratio
    denominators. ``verified`` is true only if every blocking check passed; when the step
    cannot be recomputed a ``not_independently_verifiable`` finding keeps it false.

    Args:
        workbook_path: path to the original (read-only) workbook.
        plan: the plan whose step produced ``execution_result``.
        execution_result: Executor output for one ``run_analysis`` step.
        tolerance: relative/absolute numeric tolerance.
        artifact_roots: extra directories from which result tables may be read.
    Returns:
        VerificationReport with findings and ``recomputed_values[step_id]``.
    """
    findings, expected = recompute_step(workbook_path, plan, execution_result, tolerance, artifact_roots)
    recomputed = {execution_result.step_id: canonical_payload(expected)} if expected is not None else {}
    numeric_ok = expected is not None and not any(is_blocking(f) for f in findings)
    return _report(findings, numeric=numeric_ok, chart=True, semantic=True,
                   recomputed=recomputed, tolerance=tolerance)


# --------------------------------------------------------------------------------------------
# Public API: chart verification
# --------------------------------------------------------------------------------------------

RANKING_WORDS = re.compile(
    r"\b(top|bottom|rank|ranking|most|least|highest|lowest|best|worst|largest|smallest)\b", re.I)
TREND_WORDS = re.compile(r"\b(trend|over time|monthly|weekly|daily|each month|per month|by month|timeline)\b", re.I)
RELATION_WORDS = re.compile(r"\b(relate|relates|relationship|correlat\w*|versus|against)\b", re.I)


def _plotted_arrays(result: ExecutionResult, spec: ChartSpec) -> tuple[list | None, list | None]:
    data = result.plotted_data or {}
    norm = {_norm(k): v for k, v in data.items()}
    x = data.get("x")
    if x is None and spec.x_column:
        x = norm.get(_norm(spec.x_column))
    y = data.get("y")
    if y is None and spec.y_column:
        y = norm.get(_norm(spec.y_column))
    return x, y


def _table_column(table: pd.DataFrame, name: str | None, numeric: bool | None) -> str | None:
    if name is not None:
        for column in table.columns:
            if _norm(column) == _norm(name):
                return column
    if numeric is not None:
        candidates = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c]) == numeric]
        if len(candidates) == 1:
            return candidates[0]
    return None


def verify_chart_data(
    execution_result: ExecutionResult,
    tolerance: float = 1e-9,
    *,
    result_table: pd.DataFrame | str | None = None,
    question: str | None = None,
    key_kind: str = "C",
    artifact_roots: Iterable[str | Path] | None = None,
) -> list[dict]:
    """Compare x/y arrays embedded in the chart artifact with the verified result table.

    Checks field mapping, that every plotted point traces to one table row, missing/extra
    categories, values within ``tolerance``, chronological/ranked ordering, a zero baseline for
    bar charts and chart type versus intent. Discrepancies are reported, never repaired.

    Args:
        execution_result: ``make_chart`` result carrying ``chart_spec`` and ``plotted_data``
            (``{"x": [...], "y": [...]}`` or keyed by the chart's x/y column names).
        tolerance: numeric tolerance.
        result_table: verified table (DataFrame) or path; defaults to the result's table path.
        question: optional user question for chart-intent heuristics.
        key_kind: "Q"/"M"/"Y" when x is a time period, else "C".
    Returns:
        list of VerificationFinding-compatible dicts (``check_id`` = ``<step_id>/chart_*``).
    """
    sid = execution_result.step_id
    spec = execution_result.chart_spec
    if execution_result.status != RunStatus.SUCCESS or execution_result.errors:
        error = execution_result.errors[0] if execution_result.errors else None
        return [_finding(f"{sid}/chart_execution", False, CAT_EXECUTION_FAILED,
                         f"Chart step failed: {error.message if error else execution_result.status.value}")]
    if spec is not None and spec.chart_type == ChartType.TABLE:
        return [_finding(f"{sid}/chart_data", True, CAT_CHART_VALUE, "Table output; no plotted arrays.")]
    if spec is None or not execution_result.plotted_data:
        return [_finding(f"{sid}/chart_data", False, CAT_CHART_MISSING,
                         "Chart step has no chart_spec or machine-readable plotted_data.")]

    if isinstance(result_table, pd.DataFrame):
        table = result_table
    else:
        path = result_table or execution_result.result_table_path
        if not path:
            return [_finding(f"{sid}/chart_source", False, CAT_NOT_VERIFIABLE,
                             "No verified result table available to compare chart data against.")]
        try:
            table = read_result_table(path, artifact_roots)
        except PermissionError as exc:
            return [_finding(f"{sid}/chart_source", False, CAT_UNTRUSTED_ARTIFACT, str(exc))]
        except (OSError, ValueError) as exc:
            return [_finding(f"{sid}/chart_source", False, CAT_MISSING_ARTIFACT,
                             f"Chart source table unreadable: {type(exc).__name__}")]

    findings: list[dict[str, Any]] = []
    x_col = _table_column(table, spec.x_column, None if spec.chart_type == ChartType.SCATTER else False)
    y_col = _table_column(table, spec.y_column, True)
    if x_col is None or y_col is None:
        return [_finding(f"{sid}/chart_fields", False, CAT_CHART_FIELD,
                         "Chart x/y columns do not exist in the verified result table.",
                         expected=[str(c) for c in table.columns],
                         observed={"x_column": spec.x_column, "y_column": spec.y_column})]
    findings.append(_finding(f"{sid}/chart_fields", True, CAT_CHART_FIELD, "Chart fields map to table columns."))

    x, y = _plotted_arrays(execution_result, spec)
    if x is None or y is None:
        return findings + [_finding(f"{sid}/chart_data", False, CAT_CHART_MISSING,
                                    "plotted_data lacks x/y arrays for the chart fields.",
                                    observed=list(execution_result.plotted_data))]
    if len(x) != len(y):
        return findings + [_finding(f"{sid}/chart_data", False, CAT_CHART_VALUE,
                                    "Plotted x and y arrays have different lengths.",
                                    expected=len(x), observed=len(y))]

    if spec.chart_type == ChartType.SCATTER:
        def _key(p):
            return (str(p[0]), str(p[1]))
        pairs_table = sorted(zip(table[x_col].tolist(), table[y_col].tolist()), key=_key)
        pairs_plot = sorted(zip(x, y), key=_key)
        ok = len(pairs_table) == len(pairs_plot) and all(
            _close(a[0], b[0], tolerance) and _close(a[1], b[1], tolerance) for a, b in zip(pairs_table, pairs_plot))
        findings.append(_finding(f"{sid}/chart_values", ok, CAT_CHART_VALUE,
                                 "Scatter points " + ("match" if ok else "do not match") + " the result table.",
                                 expected=len(pairs_table), observed=len(pairs_plot)))
        return findings

    table_map = {_norm_key(xv, key_kind): yv for xv, yv in zip(table[x_col].tolist(), table[y_col].tolist())}
    plot_keys = [_norm_key(v, key_kind) for v in x]
    missing = [k for k in table_map if k not in plot_keys]
    extra = [k for k in plot_keys if k not in table_map]
    if missing or extra or len(plot_keys) != len(set(plot_keys)):
        findings.append(_finding(f"{sid}/chart_categories", False, CAT_CHART_CATEGORY,
                                 "Plotted categories differ from the verified table (missing, extra or duplicated).",
                                 expected=list(table_map)[:MAX_EVIDENCE_ROWS], observed=plot_keys[:MAX_EVIDENCE_ROWS]))
    else:
        findings.append(_finding(f"{sid}/chart_categories", True, CAT_CHART_CATEGORY,
                                 "Every plotted category traces to exactly one table row."))
    bad = [{"x": k, "expected": _json_safe(table_map[k]), "observed": _json_safe(v)}
           for k, v in zip(plot_keys, y) if k in table_map and not _close(v, table_map[k], tolerance)]
    findings.append(_finding(f"{sid}/chart_values", not bad, CAT_CHART_VALUE,
                             "Plotted values match the verified table." if not bad
                             else f"{len(bad)} plotted value(s) differ from the verified table.",
                             expected=[b["expected"] for b in bad] or None, observed=bad[:MAX_EVIDENCE_ROWS] or None))

    sort = (spec.sort or "").lower()
    is_time = key_kind in ("Q", "M", "Y") or "chrono" in sort
    numeric_y = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in y)
    if is_time and plot_keys != sorted(plot_keys):
        findings.append(_finding(f"{sid}/chart_order", False, CAT_CHART_ORDER, "Time axis is not chronological.",
                                 expected=sorted(plot_keys), observed=plot_keys))
    # Category sort order is a presentation issue (values are still faithful), so it is a
    # warning: it is shown to the user but does not block verification.
    elif numeric_y and sort in ("descending", "desc", "ranked", "rank") and list(y) != sorted(y, reverse=True):
        findings.append(_finding(f"{sid}/chart_order", False, CAT_CHART_ORDER, "Ranked chart is not sorted descending.",
                                 severity=Severity.WARNING))
    elif numeric_y and sort in ("ascending", "asc") and list(y) != sorted(y):
        findings.append(_finding(f"{sid}/chart_order", False, CAT_CHART_ORDER, "Chart is not sorted ascending.",
                                 severity=Severity.WARNING))

    if spec.chart_type in (ChartType.BAR, ChartType.HORIZONTAL_BAR) and (spec.baseline is None or float(spec.baseline) != 0.0):
        findings.append(_finding(f"{sid}/chart_baseline", False, CAT_AXIS_TRUNCATION,
                                 "Bar chart value axis does not start at zero (misleading truncation).",
                                 expected=0.0, observed=spec.baseline))

    if spec.chart_type == ChartType.LINE and not is_time and (
        (question and RANKING_WORDS.search(question)) or sort in ("descending", "desc", "ranked")
    ):
        findings.append(_finding(f"{sid}/chart_type", False, CAT_CHART_TYPE,
                                 "A ranking over categories is plotted as a line chart; use a (horizontal) bar.",
                                 expected=ChartType.HORIZONTAL_BAR.value, observed=spec.chart_type.value))
    return findings


# --------------------------------------------------------------------------------------------
# Public API: plan semantics (pre-execution)
# --------------------------------------------------------------------------------------------

_PROFIT_WORDS = re.compile(r"\b(profit|profits|profitable|cogs|cost of goods|operating income|net income|margin)\b", re.I)
_COST_COLUMN_WORDS = ("profit", "cost", "cogs", "expense", "margin")
_REVENUE_COLUMN_WORDS = ("sales", "unitprice", "price", "revenue", "quantity", "qty", "amount")
_COMPLETED_Q = re.compile(r"\b(last|latest|most recent|previous)\s+(fully\s+)?(complete|completed|full|closed)?\s*quarter", re.I)
_EXPLICIT_Q = re.compile(r"\bq([1-4])\b(?:\s*(?:of\s*)?(\d{4}))?", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SUM_WORDS = re.compile(r"\b(total|sum|overall|how much)\b", re.I)
_MEAN_WORDS = re.compile(r"\b(average|mean|avg|typical)\b", re.I)


def _profile_columns(profile: WorkbookProfile) -> dict[str, dict[str, Any]]:
    """sheet -> {normalised column name -> ColumnProfile}."""
    return {s.name: {_norm(c.name): c for c in s.columns} for s in profile.sheets}


def _profile_date(profile: WorkbookProfile, which: str, preferred: str | None = None) -> pd.Timestamp | None:
    best = None
    for sheet in profile.sheets:
        for column, rng in sheet.candidate_date_ranges.items():
            if preferred and _norm(column) != _norm(preferred):
                continue
            value = pd.to_datetime((rng or {}).get(which), errors="coerce")
            if value is None or pd.isna(value):
                continue
            if best is None or (value > best if which == "max" else value < best):
                best = value
    if best is None and preferred:
        return _profile_date(profile, which, None)
    return best


def _plan_date_column(plan: AnalysisPlan) -> str | None:
    return next((str(s.inputs["date_column"]) for s in plan.steps if s.inputs.get("date_column")), None)


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _all_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _all_strings(v)


def plan_quarter(plan: AnalysisPlan) -> str | None:
    """Return the single quarter a plan targets via step inputs or date_range, if any."""
    for step in plan.steps:
        explicit = step.inputs.get("quarter") or step.inputs.get("target_quarter")
        if explicit:
            return _norm_key(explicit, "Q")
    if plan.date_range and plan.date_range.get("start") and plan.date_range.get("end"):
        start = pd.to_datetime(plan.date_range["start"], errors="coerce")
        end = pd.to_datetime(plan.date_range["end"], errors="coerce")
        if not pd.isna(start) and not pd.isna(end) and start.to_period("Q") == end.to_period("Q"):
            return str(start.to_period("Q"))
    return None


def _has_cycle(plan: AnalysisPlan) -> bool:
    deps = {s.step_id: list(s.depends_on) for s in plan.steps}
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(d) for d in deps.get(node, []) if d in deps):
            return True
        state[node] = 2
        return False

    return any(visit(n) for n in deps)


def verify_plan_semantics(question: str, plan: AnalysisPlan, workbook_profile) -> list[dict]:
    """Run deterministic pre-execution checks on metric availability, date rules and join grain.

    Each check yields a VerificationFinding-compatible dict with ``check_id`` ``plan/<check>``:
      * structure: unique step IDs, known dependencies, no cycles, budget, no code-like content;
      * referenced sheets/columns exist in the WorkbookProfile;
      * profit asked of a workbook without profit/cost data -> CLARIFY (never substitute revenue);
      * aggregation matches "total"/"average" wording; margin is a ratio of sums;
      * "last completed quarter" uses the completed-quarter policy; an explicit Qn/year matches
        the plan's date range; a year outside the data -> CLARIFY;
      * joins on non-unique right keys require pre-aggregation (join grain);
      * mixed currencies without an exchange-rate field -> CLARIFY;
      * ranking questions are not drawn as line charts; trend/relationship chart hints.
    """
    q = question or ""
    findings: list[dict[str, Any]] = []
    if plan.clarification_needed:
        return [_finding("plan/clarification", False, CAT_AMBIGUOUS,
                         plan.clarification_question or "Planner requested clarification.")]

    # ---- structure -------------------------------------------------------------------------
    ids = [s.step_id for s in plan.steps]
    if not plan.steps:
        findings.append(_finding("plan/structure", False, CAT_INVALID_PLAN, "Plan has no steps."))
    if len(ids) != len(set(ids)):
        findings.append(_finding("plan/structure", False, CAT_INVALID_PLAN, "Duplicate step IDs.", observed=ids))
    unknown_deps = sorted({d for s in plan.steps for d in s.depends_on if d not in ids})
    if unknown_deps:
        findings.append(_finding("plan/dependencies", False, CAT_INVALID_PLAN,
                                 "Steps depend on unknown step IDs.", observed=unknown_deps))
    elif _has_cycle(plan):
        findings.append(_finding("plan/dependencies", False, CAT_INVALID_PLAN, "Step dependencies contain a cycle."))
    if len(plan.steps) > plan.max_actions:
        findings.append(_finding("plan/budget", False, CAT_INVALID_PLAN, "Plan has more steps than its action budget.",
                                 expected=plan.max_actions, observed=len(plan.steps)))
    unsafe = [s for s in _all_strings(plan.model_dump(mode="json", exclude={"assumptions", "goal"}))
              if _UNSAFE_MARKERS.search(s)]
    if unsafe:
        findings.append(_finding("plan/unsafe_content", False, CAT_UNSAFE_PLAN,
                                 "Plan contains code-like content; plans must use typed operations only.",
                                 observed=[u[:80] for u in unsafe[:5]]))

    profile: WorkbookProfile | None = workbook_profile
    if profile is None:
        return findings
    cols_by_sheet = _profile_columns(profile)
    all_cols = {k: v for cols in cols_by_sheet.values() for k, v in cols.items()}

    # ---- metric availability (checked first: CLARIFY beats REPLAN) -------------------------
    has_cost_data = any(any(w in name for w in _COST_COLUMN_WORDS) for name in all_cols)
    wants_profit = bool(_PROFIT_WORDS.search(q)) or bool(plan.metric and _PROFIT_WORDS.search(plan.metric.name))
    if wants_profit and not has_cost_data:
        findings.append(_finding("plan/metric_available", False, CAT_METRIC_UNAVAILABLE,
                                 "Profit/margin was requested but the workbook has no profit, cost or expense fields.",
                                 expected="a Profit field or cost/expense inputs",
                                 observed=sorted(c.name for c in all_cols.values())[:30]))
        return findings
    if plan.metric and _PROFIT_WORDS.search(plan.metric.name or ""):
        sources = [_norm(c) for c in plan.metric.source_columns]
        if sources and all(any(w in s for w in _REVENUE_COLUMN_WORDS) for s in sources) and not any(
                any(w in s for w in _COST_COLUMN_WORDS) for s in sources):
            findings.append(_finding("plan/metric_source", False, CAT_PROXY_METRIC,
                                     "Plan computes 'profit' from revenue fields only; revenue is not profit.",
                                     observed=plan.metric.source_columns))

    # ---- periods outside the data ----------------------------------------------------------
    max_date = _profile_date(profile, "max", _plan_date_column(plan))
    min_date = _profile_date(profile, "min")
    years = [int(y) for y in _YEAR.findall(q)]
    if years and max_date is not None and min_date is not None:
        outside = [y for y in years if y < min_date.year or y > max_date.year]
        if outside:
            findings.append(_finding("plan/period_available", False, CAT_PERIOD_UNAVAILABLE,
                                     "Requested year is outside the workbook's date coverage.",
                                     expected=f"{min_date.date()}..{max_date.date()}", observed=outside))
            return findings

    # ---- references ------------------------------------------------------------------------
    sheet_names = {_norm(s.name): s.name for s in profile.sheets}
    referenced_sheets = set(plan.source_sheets)
    for step in plan.steps:
        for key in ("sheet", "source_sheet", "left_sheet", "right_sheet"):
            if isinstance(step.inputs.get(key), str):
                referenced_sheets.add(step.inputs[key])
    for join in plan.joins:
        referenced_sheets.update([join.left_sheet, join.right_sheet])
    if len(profile.sheets) > 1:
        missing_sheets = sorted(s for s in referenced_sheets if _norm(s) not in sheet_names)
        if missing_sheets:
            findings.append(_finding("plan/sheets", False, CAT_UNKNOWN_REFERENCE,
                                     "Plan references sheets that are not in the workbook.",
                                     expected=sorted(sheet_names.values()), observed=missing_sheets))
    referenced_cols: set[str] = set(plan.group_by)
    if plan.metric:
        referenced_cols.update(plan.metric.source_columns)
    referenced_cols.update(str(f["column"]) for f in plan.filters if isinstance(f, dict) and f.get("column"))
    for step in plan.steps:
        if step.tool != "run_analysis":
            continue
        for key in ("date_column", "metric_column", "numerator", "denominator", "by"):
            if isinstance(step.inputs.get(key), str):
                referenced_cols.add(step.inputs[key])
        referenced_cols.update(_as_list(step.inputs.get("group_by")))
        if isinstance(step.inputs.get("metrics"), dict):
            referenced_cols.update(str(k) for k in step.inputs["metrics"])
    missing_cols = sorted(c for c in referenced_cols if _norm(c) not in all_cols)
    if missing_cols:
        findings.append(_finding("plan/columns", False, CAT_UNKNOWN_REFERENCE,
                                 "Plan references columns that are not in the workbook.",
                                 observed=missing_cols, expected=sorted(c.name for c in all_cols.values())[:50]))

    # ---- aggregation and margin ------------------------------------------------------------
    if plan.metric:
        agg = (plan.metric.aggregation or "").strip().lower()
        if re.search(r"\bmargin\b", q, re.I) or "margin" in (plan.metric.name or "").lower():
            formula = (plan.metric.formula or "").replace(" ", "").lower()
            ratio_ok = any((s.operation or "") == "grouped_ratio" for s in plan.steps) \
                or agg in ("ratio_of_sums", "sum_ratio", "ratio") or formula.count("sum(") >= 2
            if agg in ("mean", "avg", "average") or not ratio_ok:
                findings.append(_finding("plan/margin_formula", False, CAT_WRONG_MARGIN,
                                         "Margin must be sum(Profit)/sum(Sales), not an average of row margins.",
                                         expected="sum(Profit)/sum(Sales)", observed=plan.metric.formula or agg))
        else:
            canon = _AGG_ALIASES.get(agg, agg)
            mentions_sum, mentions_mean = bool(_SUM_WORDS.search(q)), bool(_MEAN_WORDS.search(q))
            if mentions_sum and not mentions_mean and canon in ("mean", "median"):
                findings.append(_finding("plan/aggregation", False, CAT_WRONG_METRIC,
                                         "Question asks for a total but the plan averages.",
                                         expected="sum", observed=agg))
            elif mentions_mean and not mentions_sum and canon == "sum":
                findings.append(_finding("plan/aggregation", False, CAT_WRONG_METRIC,
                                         "Question asks for an average but the plan sums (check the intended grain).",
                                         expected="mean", observed=agg, severity=Severity.WARNING))

    # ---- completed / explicit quarter ------------------------------------------------------
    explicit_q = _EXPLICIT_Q.search(q)
    target_q = plan_quarter(plan)
    policy = (plan.date_policy or "").lower()
    if _COMPLETED_Q.search(q) and not explicit_q:
        if target_q is None and not any(w in policy for w in ("complete", "previous_to", "prior", "data_through", "reporting")):
            findings.append(_finding("plan/date_policy", False, CAT_WRONG_PERIOD,
                                     "Question asks for the last completed quarter but the plan has no "
                                     "completed-quarter date policy.", observed=plan.date_policy))
        elif target_q is not None and max_date is not None and "data_through" not in policy:
            canonical = str(max_date.to_period("Q") - 1)
            if target_q != canonical:
                findings.append(_finding("plan/period", False, CAT_WRONG_PERIOD,
                                         "Plan's quarter is not the last completed quarter.",
                                         expected=canonical, observed=target_q))
    elif explicit_q and target_q is not None:
        wanted_q, wanted_year = int(explicit_q.group(1)), explicit_q.group(2)
        got = pd.Period(target_q, freq="Q")
        if got.quarter != wanted_q or (wanted_year and got.year != int(wanted_year)):
            findings.append(_finding("plan/period", False, CAT_WRONG_PERIOD,
                                     "Plan's quarter does not match the quarter named in the question.",
                                     expected=f"Q{wanted_q}" + (f" {wanted_year}" if wanted_year else ""),
                                     observed=target_q))

    # ---- joins -----------------------------------------------------------------------------
    sheet_rows = {_norm(s.name): s.row_count for s in profile.sheets}
    for join in plan.joins:
        right_cols = cols_by_sheet.get(sheet_names.get(_norm(join.right_sheet), join.right_sheet), {})
        rows = sheet_rows.get(_norm(join.right_sheet))
        if len(join.right_keys) == 1 and rows:
            col = right_cols.get(_norm(join.right_keys[0]))
            unique = col.unique_count if col is not None else None
            if unique is not None and unique < rows and not join.aggregate_before_join \
                    and join.expected_cardinality in ("one_to_one", "many_to_one"):
                findings.append(_finding(
                    "plan/join_grain", False, CAT_JOIN_GRAIN,
                    f"'{join.right_sheet}.{join.right_keys[0]}' is not unique; aggregate it before joining "
                    "or the join will multiply measures.",
                    expected="aggregate_before_join=true", observed={"unique": unique, "rows": rows}))

    # ---- currency --------------------------------------------------------------------------
    currency_cols = [c for name, c in all_cols.items() if "currency" in name or name in ("ccy", "curr")]
    has_fx = any(any(w in name for w in ("exchangerate", "fxrate", "conversionrate")) for name in all_cols)
    if any((c.unique_count or 0) > 1 for c in currency_cols) and not has_fx:
        findings.append(_finding("plan/currency", False, CAT_CURRENCY,
                                 "Workbook mixes currencies and has no exchange-rate source; values cannot be added.",
                                 observed=[c.name for c in currency_cols]))

    # ---- chart intent ----------------------------------------------------------------------
    chart = plan.chart_spec
    if chart is not None:
        if chart.chart_type == ChartType.LINE and RANKING_WORDS.search(q) and not TREND_WORDS.search(q):
            findings.append(_finding("plan/chart_type", False, CAT_CHART_TYPE,
                                     "Ranking question planned as a line chart; use a (horizontal) bar chart.",
                                     expected=ChartType.HORIZONTAL_BAR.value, observed=ChartType.LINE.value))
        elif TREND_WORDS.search(q) and chart.chart_type in (ChartType.SCATTER, ChartType.HORIZONTAL_BAR):
            findings.append(_finding("plan/chart_type", False, CAT_CHART_TYPE, "Trend question should use a line chart.",
                                     expected=ChartType.LINE.value, observed=chart.chart_type.value,
                                     severity=Severity.WARNING))
        elif RELATION_WORDS.search(q) and not RANKING_WORDS.search(q) \
                and chart.chart_type not in (ChartType.SCATTER, ChartType.TABLE):
            findings.append(_finding("plan/chart_type", False, CAT_CHART_TYPE,
                                     "Relationship question is usually best shown as a scatter plot.",
                                     expected=ChartType.SCATTER.value, observed=chart.chart_type.value,
                                     severity=Severity.WARNING))

    # ---- verifiability ---------------------------------------------------------------------
    unsupported = [s.step_id for s in plan.steps
                   if s.tool == "run_analysis" and (s.operation or "").lower() not in SUPPORTED_OPERATIONS]
    if unsupported:
        findings.append(_finding("plan/verifiable", False, CAT_NOT_VERIFIABLE,
                                 "Some analysis steps use operations without canonical recomputation.",
                                 expected=sorted(SUPPORTED_OPERATIONS), observed=unsupported, severity=Severity.WARNING))
    if not any(not f["passed"] for f in findings):
        findings.append(_finding("plan/semantics", True, CAT_SEMANTIC, "Deterministic plan checks passed."))
    return findings


# --------------------------------------------------------------------------------------------
# Public API: whole-plan verification
# --------------------------------------------------------------------------------------------


def verify_execution(
    workbook_path: str,
    plan: AnalysisPlan,
    execution_results: list[ExecutionResult],
    workbook_profile: WorkbookProfile | None = None,
    question: str = "",
    tolerance: float = 1e-9,
    *,
    artifact_roots: Iterable[str | Path] | None = None,
) -> VerificationReport:
    """Verify a full plan execution and return one VerificationReport.

    ``verified`` is true only if plan semantics pass, at least one analysis step was
    independently recomputed, every recomputation matched, every chart's plotted data matches
    the *independently recomputed* table, and no step failed or was unverifiable.
    ``recomputed_values`` holds a bounded canonical table per step plus ``period`` evidence.
    """
    findings: list[dict[str, Any]] = list(verify_plan_semantics(question, plan, workbook_profile))
    semantic_ok = not any(is_blocking(f) for f in findings)
    recomputed: dict[str, Any] = {}
    canonical: dict[str, CanonicalResult] = {}
    by_step = {r.step_id: r for r in execution_results}
    numeric_steps, numeric_ok = 0, True

    for step in plan.steps:
        if step.tool != "run_analysis":
            continue
        numeric_steps += 1
        result = by_step.get(step.step_id)
        if result is None:
            findings.append(_finding(f"{step.step_id}/execution", False, CAT_EXECUTION_FAILED,
                                     "Planned analysis step has no execution result."))
            numeric_ok = False
            continue
        step_findings, expected = recompute_step(workbook_path, plan, result, tolerance, artifact_roots)
        findings.extend(step_findings)
        if expected is None or any(is_blocking(f) for f in step_findings):
            numeric_ok = False
        if expected is not None:
            canonical[step.step_id] = expected
            recomputed[step.step_id] = canonical_payload(expected)
    if numeric_steps == 0:
        numeric_ok = False
        findings.append(_finding("plan/numeric", False, CAT_NOT_VERIFIABLE,
                                 "Plan has no analysis step that can be independently recomputed."))

    chart_ok = True
    chart_steps = [s for s in plan.steps if s.tool == "make_chart"]
    for step in chart_steps:
        result = by_step.get(step.step_id)
        if result is None:
            findings.append(_finding(f"{step.step_id}/chart_execution", False, CAT_EXECUTION_FAILED,
                                     "Planned chart step has no execution result."))
            chart_ok = False
            continue
        source_id = step.inputs.get("source_step_id") or (step.depends_on[0] if step.depends_on else None)
        source = canonical.get(source_id) if source_id else None
        if result.chart_spec is None and plan.chart_spec is not None:
            result = result.model_copy(update={"chart_spec": plan.chart_spec})
        if source is not None:
            chart_findings = verify_chart_data(result, tolerance, result_table=source.table, question=question,
                                               key_kind=source.key_kind, artifact_roots=artifact_roots)
        else:
            chart_findings = [_finding(f"{step.step_id}/chart_source", False, CAT_NOT_VERIFIABLE,
                                       "Chart source data was not independently verified.")]
        findings.extend(chart_findings)
        chart_ok = chart_ok and not any(is_blocking(f) for f in chart_findings)
    if plan.chart_spec is not None and plan.chart_spec.chart_type != ChartType.TABLE and not chart_steps:
        findings.append(_finding("plan/chart", False, CAT_CHART_MISSING,
                                 "Plan declares a chart but has no make_chart step.", severity=Severity.WARNING))

    period = next((p["period"] for p in recomputed.values() if p.get("period")), None)
    if period:
        recomputed["period"] = period
    return _report(findings, numeric=numeric_ok, chart=chart_ok, semantic=semantic_ok,
                   recomputed=recomputed, tolerance=tolerance)


__all__ = [
    "ROUTING", "SUPPORTED_OPERATIONS", "CanonicalResult", "canonical_payload", "compare_tables",
    "completed_quarter", "find_data_through", "is_blocking", "load_source_frames", "plan_quarter",
    "read_result_table", "recompute_check", "recompute_step", "verify_chart_data", "verify_execution",
    "verify_plan_semantics",
]
