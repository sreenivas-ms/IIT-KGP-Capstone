"""Chart rendering from executed result tables; plotted values are read from disk."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verisheet.schemas import ChartSpec, ChartType
from verisheet.tools.inspect import to_json_safe


def validate_chart_spec(columns: list[str], chart_spec: ChartSpec) -> None:
    """Reject missing columns and unsupported chart/axis combinations."""
    chart_type = chart_spec.chart_type.value if hasattr(chart_spec.chart_type, "value") else str(chart_spec.chart_type)
    if chart_type not in {item.value for item in ChartType}:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    if chart_type == "table":
        return
    if not chart_spec.x_column or chart_spec.x_column not in columns:
        raise ValueError("Chart x_column must exist in the result table")
    y_columns = [chart_spec.y_column] if chart_spec.y_column else list(chart_spec.series_columns)
    if not y_columns or any(column not in columns for column in y_columns):
        raise ValueError("Chart requires existing y_column or series_columns")
    if chart_type in {"line", "scatter", "bar", "horizontal_bar"} and not y_columns:
        raise ValueError("Chart requires at least one numeric series")


def make_chart(result_table_path: str, chart_spec: ChartSpec, output_dir: str = "artifacts/charts") -> dict:
    """Render a chart from a CSV result table and export exact plotted arrays as JSON."""
    path = Path(result_table_path)
    if not path.is_file(): raise FileNotFoundError(f"Result table not found: {path}")
    frame = pd.read_csv(path)
    validate_chart_spec(list(frame.columns), chart_spec)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    image_path, data_path = output / f"{stem}.png", output / f"{stem}_plotted_data.json"
    chart_type = chart_spec.chart_type.value if hasattr(chart_spec.chart_type, "value") else str(chart_spec.chart_type)
    if chart_type == "table":
        plotted = {column: [to_json_safe(v) for v in frame[column].tolist()] for column in frame.columns}
        fig, ax = plt.subplots(figsize=(10, max(2, min(12, 0.35 * len(frame) + 1))))
        ax.axis("off"); ax.table(cellText=frame.head(30).values, colLabels=frame.columns, loc="center")
    else:
        xcol = chart_spec.x_column
        ycols = [chart_spec.y_column] if chart_spec.y_column else list(chart_spec.series_columns)
        plotted = {xcol: [to_json_safe(v) for v in frame[xcol].tolist()]}
        for col in ycols:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            if numeric.isna().any() and frame[col].notna().any(): raise ValueError(f"Chart y column {col} contains non-numeric values")
            plotted[col] = [to_json_safe(v) for v in numeric.tolist()]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = frame[xcol].astype(str)
        for col in ycols:
            y = pd.to_numeric(frame[col], errors="coerce")
            if chart_type == "bar": ax.bar(x, y, label=col)
            elif chart_type == "horizontal_bar": ax.barh(x, y, label=col)
            elif chart_type == "line": ax.plot(x, y, marker="o", label=col)
            elif chart_type == "scatter": ax.scatter(x, y, label=col)
        if chart_spec.baseline is not None:
            if chart_type == "horizontal_bar": ax.axvline(chart_spec.baseline, linewidth=0.8)
            else: ax.axhline(chart_spec.baseline, linewidth=0.8)
        ax.set_xlabel(chart_spec.x_axis_label or xcol); ax.set_ylabel(chart_spec.y_axis_label or (ycols[0] if len(ycols) == 1 else "Value"))
        ax.set_title(chart_spec.title)
        if len(ycols) > 1: ax.legend()
        if len(frame) > 20: ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
    fig.savefig(image_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    data_path.write_text(json.dumps(plotted, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return {"chart_path": str(image_path), "plotted_data_path": str(data_path), "plotted_data": plotted, "row_count": len(frame), "columns": list(frame.columns)}
