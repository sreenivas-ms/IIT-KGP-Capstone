"""Executor agent: validates typed plans and runs allowlisted spreadsheet operations."""
from __future__ import annotations
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any
import pandas as pd

from verisheet.config import get_settings
from verisheet.profile import list_sheet_names, read_sheet, validate_upload, profile_workbook
from verisheet.schemas import AnalysisPlan, ExecutionResult, RunStatus, ToolError
from verisheet.sandbox import RestrictedSandbox
from verisheet.tools.analysis import run_analysis, filter_rows
from verisheet.tools.chart import make_chart
from verisheet.tools.inspect import inspect_workbook, sample_rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "item"


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try: return value.item()
        except (ValueError, TypeError): pass
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False: return None
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


class Executor:
    """Run approved typed analysis steps and create bounded, reproducible artifacts."""
    def __init__(self, artifact_dir: str = "artifacts", sandbox=None):
        """Configure artifact output and restricted execution backend."""
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox or RestrictedSandbox(
            timeout_seconds=get_settings().execution_timeout_seconds,
            memory_mb=get_settings().execution_memory_mb,
        )
        self.operation_registry = {name: run_analysis for name in (
            "group_aggregate", "filter_rows", "top_n", "time_aggregate",
            "quarter_comparison", "safe_join", "grouped_ratio")}

    def execute_plan(self, plan: AnalysisPlan, workbook_path: str, run_id: str) -> list[ExecutionResult]:
        """Execute plan steps in dependency order; stop safely after a failed dependency."""
        if not isinstance(plan, AnalysisPlan): plan = AnalysisPlan.model_validate(plan)
        if plan.clarification_needed: return []
        if len(plan.steps) > min(plan.max_actions, get_settings().max_agent_actions):
            raise ValueError("Plan exceeds configured action budget")
        path = Path(workbook_path)
        self._active_plan_filters = plan.filters
        self._active_chart_spec = plan.chart_spec
        profile = profile_workbook(path)
        self.validate_plan_references(plan, profile)
        ordered = self._topological_steps(plan)
        prior: dict[str, ExecutionResult] = {}
        results: list[ExecutionResult] = []
        for step in ordered:
            if any(prior[dep].status != RunStatus.SUCCESS for dep in step.depends_on):
                error = ToolError(code="DEPENDENCY_FAILED", category="dependency", message="A required dependency did not complete successfully", failed_step_id=step.step_id)
                result = ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.FAILED, errors=[error])
            else:
                result = self.execute_step(step, str(path), run_id, prior)
            prior[step.step_id] = result; results.append(result)
            if result.status != RunStatus.SUCCESS: break
        return results

    def execute_step(self, step, workbook_path: str, run_id: str, prior_results: dict) -> ExecutionResult:
        """Execute one AnalysisStep using only registered tools and validated dependencies."""
        start = time.monotonic()
        folder = self.artifact_dir / _safe_name(run_id); folder.mkdir(parents=True, exist_ok=True)
        step_name = _safe_name(step.step_id)
        try:
            file_type = validate_upload(Path(workbook_path))
            if step.tool == "inspect_workbook":
                profile = inspect_workbook(workbook_path)
                payload = profile.model_dump(mode="json")
                result_path = folder / f"{step_name}_profile.json"
                result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                    result_table_path=str(result_path), result_columns=["workbook_profile"], row_count_out=len(profile.sheets),
                    duration_seconds=time.monotonic()-start, tool_calls=["inspect_workbook"])
            if step.tool == "sample_rows":
                sheet = step.inputs.get("sheet")
                if not sheet: raise ValueError("sample_rows requires inputs.sheet")
                rows = sample_rows(Path(workbook_path), sheet, int(step.inputs.get("n", 10)))
                result_path = folder / f"{step_name}_sample.json"
                result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                    result_table_path=str(result_path), result_columns=list(rows[0]) if rows else [], row_count_out=len(rows),
                    duration_seconds=time.monotonic()-start, tool_calls=["sample_rows"])
            if step.tool == "run_analysis":
                operation = step.operation
                if operation not in self.operation_registry: raise ValueError(f"Unsupported operation: {operation}")
                sheet_names = list_sheet_names(Path(workbook_path), file_type)
                frames = {name: read_sheet(Path(workbook_path), file_type, name) for name in sheet_names}
                inputs = dict(step.inputs)
                # Apply plan-level predicates to the explicitly selected source sheet before
                # the requested operation, so aggregation and ranking respect the same filters.
                if inputs.get("sheet") and getattr(self, "_active_plan_filters", None):
                    frames[inputs["sheet"]] = filter_rows(frames[inputs["sheet"]], self._active_plan_filters)
                if operation == "safe_join" and "join_spec" not in inputs:
                    raise ValueError("safe_join requires inputs.join_spec")
                frame = run_analysis(operation, frames, inputs)
                result_path = folder / f"{step_name}.csv"
                frame.to_csv(result_path, index=False)
                script_path = folder / f"{step_name}.py"
                # Audit/reproducibility record; this is not model-authored executable code.
                script_text = "# VeriSheet deterministic operation record\n# No dynamic code execution.\n" + f"OPERATION = {operation!r}\nINPUTS = {json.dumps(inputs, default=str, sort_keys=True)!r}\n"
                script_path.write_text(script_text, encoding="utf-8")
                script_hash = "sha256:" + hashlib.sha256(script_text.encode("utf-8")).hexdigest()
                scalars = {}
                if len(frame) == 1:
                    scalars = {str(k): _json_value(v) for k, v in frame.iloc[0].to_dict().items()}
                return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                    result_table_path=str(result_path), result_columns=[str(c) for c in frame.columns],
                    row_count_in=sum(len(f) for f in frames.values()), row_count_out=len(frame), scalar_values=scalars,
                    script_path=str(script_path), script_hash=script_hash, duration_seconds=time.monotonic()-start,
                    tool_calls=[operation])
            if step.tool == "make_chart":
                source_id = step.inputs.get("source_step_id")
                if source_id not in prior_results: raise ValueError("make_chart source_step_id must reference a completed step")
                source = prior_results[source_id]
                if source.status != RunStatus.SUCCESS or not source.result_table_path: raise ValueError("Chart source step has no successful result table")
                chart_spec = plan_chart_spec = getattr(self, "_active_chart_spec", None)
                if chart_spec is None: raise ValueError("Plan chart_spec is required for make_chart")
                chart = make_chart(source.result_table_path, chart_spec, str(folder / "charts"))
                return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                    result_table_path=source.result_table_path, result_columns=source.result_columns,
                    row_count_in=source.row_count_in, row_count_out=source.row_count_out,
                    chart_path=chart["chart_path"], chart_spec=chart_spec, plotted_data=chart["plotted_data"],
                    duration_seconds=time.monotonic()-start, tool_calls=["make_chart"])
            raise ValueError(f"Unsupported tool: {step.tool}")
        except Exception as exc:
            error = ToolError(code="EXECUTION_ERROR", category=type(exc).__name__, message=str(exc)[:500], retryable=isinstance(exc, (TimeoutError,)) , failed_step_id=step.step_id)
            return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.FAILED,
                duration_seconds=time.monotonic()-start, tool_calls=[step.tool], errors=[error])

    def validate_plan_references(self, plan: AnalysisPlan, workbook_profile) -> None:
        """Reject unknown sheets/columns, unsupported operations and dependency cycles."""
        sheet_map = {sheet.name: {column.name for column in sheet.columns} for sheet in workbook_profile.sheets}
        for name in plan.source_sheets:
            if name not in sheet_map: raise ValueError(f"Unknown source sheet: {name}")
        for join in plan.joins:
            if join.left_sheet not in sheet_map or join.right_sheet not in sheet_map: raise ValueError("Join references unknown sheet")
            if not set(join.left_keys).issubset(sheet_map[join.left_sheet]) or not set(join.right_keys).issubset(sheet_map[join.right_sheet]): raise ValueError("Join references unknown key column")
        for step in plan.steps:
            if step.tool == "run_analysis" and step.operation not in self.operation_registry: raise ValueError(f"Unsupported operation: {step.operation}")
            sheet = step.inputs.get("sheet")
            if sheet and sheet not in sheet_map: raise ValueError(f"Unknown sheet: {sheet}")
            if sheet:
                for key in ("date_column", "metric_column", "by", "numerator", "denominator"):
                    value = step.inputs.get(key)
                    if value and value not in sheet_map[sheet]: raise ValueError(f"Unknown column '{value}' in sheet '{sheet}'")
                for column in step.inputs.get("group_by", []):
                    if column not in sheet_map[sheet]: raise ValueError(f"Unknown group_by column '{column}'")
                for column in step.inputs.get("metrics", {}):
                    if column not in sheet_map[sheet]: raise ValueError(f"Unknown metric column '{column}'")
            if step.tool == "make_chart" and not plan.chart_spec: raise ValueError("make_chart requires plan.chart_spec")
        self._topological_steps(plan)

    @staticmethod
    def _topological_steps(plan: AnalysisPlan):
        """Return steps in stable topological order and reject missing/cyclic dependencies."""
        by_id = {step.step_id: step for step in plan.steps}
        if len(by_id) != len(plan.steps): raise ValueError("Duplicate step_id in plan")
        for step in plan.steps:
            missing = set(step.depends_on) - set(by_id)
            if missing: raise ValueError(f"Unknown dependencies for {step.step_id}: {sorted(missing)}")
        ordered, remaining = [], list(plan.steps)
        while remaining:
            ready = [step for step in remaining if all(dep in {s.step_id for s in ordered} for dep in step.depends_on)]
            if not ready: raise ValueError("Analysis plan contains a dependency cycle")
            ordered.extend(ready)
            ready_ids = {step.step_id for step in ready}
            remaining = [step for step in remaining if step.step_id not in ready_ids]
        return ordered


def execute_plan(plan: AnalysisPlan, workbook_path: str, run_id: str, executor: Executor | None = None) -> list[ExecutionResult]:
    """Canonical function wrapper around Executor.execute_plan."""
    service = executor or Executor()
    service._active_plan_filters = plan.filters
    service._active_chart_spec = plan.chart_spec
    return service.execute_plan(plan, workbook_path, run_id)
