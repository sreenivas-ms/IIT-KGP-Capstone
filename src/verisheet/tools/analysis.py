"""Allowlisted deterministic Pandas operations used by the VeriSheet Executor.

Only typed operation names and arguments are accepted. This module never evaluates
Python supplied by the Planner or spreadsheet cells.
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd

AGGREGATIONS = {"sum", "mean", "count", "min", "max", "nunique", "median"}
FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "between", "is_null", "not_null"}


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing}")


def _validate_aggregation(metrics: dict[str, str]) -> None:
    if not metrics:
        raise ValueError("At least one metric aggregation is required")
    invalid = {column: op for column, op in metrics.items() if op not in AGGREGATIONS}
    if invalid:
        raise ValueError(f"Unsupported aggregation(s): {invalid}")


def run_analysis(operation: str, frames: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    """Dispatch a typed operation name to a fixed implementation registry."""
    if not isinstance(inputs, dict) or not isinstance(frames, dict):
        raise TypeError("frames and inputs must be dictionaries")
    registry = {
        "group_aggregate": _op_group_aggregate,
        "filter_rows": _op_filter_rows,
        "top_n": _op_top_n,
        "time_aggregate": _op_time_aggregate,
        "quarter_comparison": _op_quarter_comparison,
        "safe_join": _op_safe_join,
        "grouped_ratio": _op_grouped_ratio,
    }
    handler = registry.get(operation)
    if handler is None:
        raise ValueError(f"Unsupported analysis operation: {operation}")
    result = handler(frames, inputs)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Operation {operation} did not return a DataFrame")
    result.attrs["operation"] = operation
    result.attrs["summary"] = {"operation": operation, "row_count_out": len(result), "columns": list(result.columns)}
    return result


def group_aggregate(frame: pd.DataFrame, group_by: list[str], metrics: dict[str, str]) -> pd.DataFrame:
    """Group rows by columns and apply approved aggregations."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    group_by = list(group_by or [])
    _require_columns(frame, group_by + list(metrics))
    _validate_aggregation(metrics)
    if group_by:
        result = frame.groupby(group_by, dropna=False, sort=True).agg(metrics).reset_index()
    else:
        result = frame.agg(metrics).to_frame().T
    return result


def filter_rows(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply typed filter predicates; reject unsupported operators and columns."""
    _require_columns(frame, [f.get("column") for f in filters if isinstance(f, dict) and f.get("column")])
    result = frame.copy()
    for rule in filters:
        if not isinstance(rule, dict) or not {"column", "operator"}.issubset(rule):
            raise ValueError("Each filter requires column and operator")
        column, operator, value = rule["column"], rule["operator"], rule.get("value")
        if operator not in FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {operator}")
        series = result[column]
        if operator == "eq": mask = series == value
        elif operator == "ne": mask = series != value
        elif operator == "gt": mask = series > value
        elif operator == "gte": mask = series >= value
        elif operator == "lt": mask = series < value
        elif operator == "lte": mask = series <= value
        elif operator == "in":
            if not isinstance(value, list): raise ValueError("'in' filter value must be a list")
            mask = series.isin(value)
        elif operator == "not_in":
            if not isinstance(value, list): raise ValueError("'not_in' filter value must be a list")
            mask = ~series.isin(value)
        elif operator == "contains": mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2: raise ValueError("'between' value must contain [lower, upper]")
            mask = series.between(value[0], value[1], inclusive="both")
        elif operator == "is_null": mask = series.isna()
        else: mask = series.notna()
        result = result.loc[mask.fillna(False)]
    return result.reset_index(drop=True)


def top_n(frame: pd.DataFrame, by: str, n: int, ascending: bool = False) -> pd.DataFrame:
    """Return a deterministic top/bottom N ranking, retaining stable tie order."""
    _require_columns(frame, [by])
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return frame.sort_values(by, ascending=ascending, kind="mergesort", na_position="last").head(n).reset_index(drop=True)


def time_aggregate(frame: pd.DataFrame, date_column: str, frequency: str, metrics: dict[str, str]) -> pd.DataFrame:
    """Aggregate approved metrics by calendar month, quarter or year."""
    _require_columns(frame, [date_column] + list(metrics))
    _validate_aggregation(metrics)
    freq_map = {"month": "M", "monthly": "M", "quarter": "Q", "quarterly": "Q", "year": "Y", "yearly": "Y"}
    if frequency.lower() not in freq_map:
        raise ValueError("frequency must be month, quarter, or year")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid = frame.loc[dates.notna()].copy()
    valid["period"] = dates.loc[dates.notna()].dt.to_period(freq_map[frequency.lower()]).astype(str)
    if valid.empty:
        return pd.DataFrame(columns=["period", *metrics.keys()])
    return valid.groupby("period", sort=True).agg(metrics).reset_index()


def safe_join(left: pd.DataFrame, right: pd.DataFrame, join_spec: Any) -> pd.DataFrame:
    """Join frames only after validating key uniqueness and declared cardinality."""
    get = (lambda key: getattr(join_spec, key)) if not isinstance(join_spec, dict) else join_spec.get
    left_keys, right_keys = list(get("left_keys")), list(get("right_keys"))
    cardinality = get("expected_cardinality")
    aggregate_before_join = get("aggregate_before_join")
    validate_row_count = get("validate_row_count") if not isinstance(join_spec, dict) else join_spec.get("validate_row_count", True)
    if aggregate_before_join:
        raise ValueError("aggregate_before_join requires an explicit pre-aggregation plan step; refusing an implicit grain change")
    if not left_keys or len(left_keys) != len(right_keys):
        raise ValueError("Join requires equally sized, non-empty key lists")
    _require_columns(left, left_keys); _require_columns(right, right_keys)
    if cardinality not in {"one_to_one", "one_to_many", "many_to_one"}:
        raise ValueError("Unsupported or unsafe join cardinality")
    if cardinality in {"one_to_one", "many_to_one"} and left.duplicated(left_keys).any():
        raise ValueError("Left join keys violate declared cardinality")
    if cardinality in {"one_to_one", "one_to_many"} and right.duplicated(right_keys).any():
        raise ValueError("Right join keys violate declared cardinality")
    before = len(left)
    merged = left.merge(right, left_on=left_keys, right_on=right_keys, how="left", validate=cardinality, suffixes=("", "_right"), indicator=True)
    if bool(validate_row_count) and len(merged) > before and cardinality != "one_to_many":
        raise ValueError("Join unexpectedly multiplied rows")
    return merged.drop(columns=["_merge"]).reset_index(drop=True)


def quarter_comparison(frame: pd.DataFrame, date_column: str, metric_column: str, aggregation: str = "sum") -> pd.DataFrame:
    """Compare the two completed quarters before the quarter containing max observed date.

    This implements the project's default observed-date policy. Percentage change is null
    when the previous quarter is zero or negative; the absolute change remains available.
    """
    _require_columns(frame, [date_column, metric_column])
    if aggregation not in AGGREGATIONS - {"nunique"}:
        raise ValueError(f"Unsupported quarter aggregation: {aggregation}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid = frame.loc[dates.notna()].copy()
    if valid.empty:
        raise ValueError("No valid dates available for quarter comparison")
    valid[date_column] = dates.loc[dates.notna()]
    current_quarter = valid[date_column].max().to_period("Q")
    quarters = [current_quarter - 2, current_quarter - 1]
    valid["_quarter"] = valid[date_column].dt.to_period("Q")
    valid = valid[valid["_quarter"].isin(quarters)]
    values = valid.groupby("_quarter")[metric_column].agg(aggregation)
    labels = [str(q) for q in quarters]
    previous_value = values.get(quarters[0], 0)
    latest_value = values.get(quarters[1], 0)
    previous = 0.0 if pd.isna(previous_value) else float(previous_value)
    latest = 0.0 if pd.isna(latest_value) else float(latest_value)
    change = latest - previous
    pct = (change / previous * 100.0) if previous > 0 else None
    return pd.DataFrame([
        {"quarter": labels[0], metric_column: previous, "period_role": "previous_completed", "change_vs_previous": None, "percent_change": None},
        {"quarter": labels[1], metric_column: latest, "period_role": "latest_completed", "change_vs_previous": change, "percent_change": pct},
    ])


def _frame(frames, inputs):
    name = inputs.get("sheet") or inputs.get("left_sheet") or (next(iter(frames)) if len(frames) == 1 else None)
    if name not in frames: raise ValueError(f"Unknown or ambiguous sheet: {name}")
    return frames[name]


def _op_group_aggregate(frames, i): return group_aggregate(_frame(frames, i), i.get("group_by", []), i.get("metrics", {}))
def _op_filter_rows(frames, i): return filter_rows(_frame(frames, i), i.get("filters", []))
def _op_top_n(frames, i): return top_n(_frame(frames, i), i["by"], i["n"], i.get("ascending", False))
def _op_time_aggregate(frames, i): return time_aggregate(_frame(frames, i), i["date_column"], i["frequency"], i["metrics"])
def _op_quarter_comparison(frames, i): return quarter_comparison(_frame(frames, i), i["date_column"], i["metric_column"], i.get("aggregation", "sum"))
def _op_safe_join(frames, i):
    spec = i.get("join_spec", i)
    return safe_join(frames[spec["left_sheet"]], frames[spec["right_sheet"]], spec)
def _op_grouped_ratio(frames, i):
    frame = _frame(frames, i)
    numerator, denominator = i["numerator"], i["denominator"]
    groups = i.get("group_by", [])
    _require_columns(frame, groups + [numerator, denominator])
    grouped = frame.groupby(groups, dropna=False, sort=True)[[numerator, denominator]].sum().reset_index() if groups else frame[[numerator, denominator]].sum().to_frame().T
    output = i.get("output_column", "ratio")
    grouped[output] = grouped[numerator].div(grouped[denominator].where(grouped[denominator] != 0))
    return grouped
