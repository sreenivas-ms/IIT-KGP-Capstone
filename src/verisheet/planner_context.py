"""Compact a WorkbookProfile into a small JSON string for the Planner prompt (P3).

The API allows 8192 tokens in+out, so the full profile cannot be sent. This keeps
what the planner needs to choose sheets/columns/dates and to spot quality issues,
and drops the rest in stages until it fits the character budget.

Works with pydantic WorkbookProfile objects OR plain dicts (duck typed), using the
field names exercised by tests/test_profile.py: sheets[].name/row_count/hidden/
columns[].name/candidate_roles/safe_examples/candidate_date_ranges and
warnings[].code/sheet/column.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any

MAX_COLUMNS_PER_SHEET = 30
MAX_EXAMPLES = 2
MAX_EXAMPLE_CHARS = 30
MAX_WARNINGS = 20


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(value: Any) -> Any:
    """Enums -> their value, so json.dumps works and the prompt shows 'xlsx' not 'FileType.XLSX'."""
    return value.value if isinstance(value, Enum) else value


def _build(profile: Any, examples: bool, max_cols: int, max_warnings: int) -> dict:
    sheets = []
    for s in _get(profile, "sheets", []) or []:
        dates = _get(s, "candidate_date_ranges", {}) or {}
        cols = []
        for c in (_get(s, "columns", []) or [])[:max_cols]:
            col: dict[str, Any] = {
                "name": _get(c, "name"),
                "roles": [_plain(r) for r in (_get(c, "candidate_roles", []) or [])],
            }
            dtype = _get(c, "dtype")
            if dtype:
                col["dtype"] = str(_plain(dtype))
            if examples:
                ex = [str(e)[:MAX_EXAMPLE_CHARS] for e in (_get(c, "safe_examples", []) or [])]
                if ex:
                    col["examples"] = ex[:MAX_EXAMPLES]
            cols.append(col)
        entry: dict[str, Any] = {
            "name": _get(s, "name"),
            "rows": _get(s, "row_count"),
            "columns": cols,
        }
        if _get(s, "hidden"):
            entry["hidden"] = True
        if dates:
            entry["date_ranges"] = dates
        total = len(_get(s, "columns", []) or [])
        if total > max_cols:
            entry["columns_omitted"] = total - max_cols
        sheets.append(entry)

    warnings = [
        {k: v for k, v in {"code": _plain(_get(w, "code")),
                           "sheet": _get(w, "sheet"),
                           "column": _get(w, "column")}.items() if v}
        for w in (_get(profile, "warnings", []) or [])[:max_warnings]
    ]
    return {
        "file_type": _plain(_get(profile, "file_type")),
        "workbook_id": str(_get(profile, "workbook_id", ""))[:19],   # 'sha256:' + 12 hex
        "sheets": sheets,
        "warnings": warnings,
    }


def compact_profile(profile: Any, max_chars: int = 9000) -> str:
    """Return compact JSON for the prompt, shrinking in stages to fit max_chars.

    Stages: full -> no examples -> <=15 columns/sheet -> <=8 warnings.
    The last stage is returned even if still over budget; the planner then
    truncates (see planner.build_user_message).
    """
    stages = [
        {"examples": True, "max_cols": MAX_COLUMNS_PER_SHEET, "max_warnings": MAX_WARNINGS},
        {"examples": False, "max_cols": MAX_COLUMNS_PER_SHEET, "max_warnings": MAX_WARNINGS},
        {"examples": False, "max_cols": 15, "max_warnings": MAX_WARNINGS},
        {"examples": False, "max_cols": 15, "max_warnings": 8},
    ]
    text = ""
    for st in stages:
        text = json.dumps(_build(profile, **st), ensure_ascii=False, separators=(",", ":"),
                          default=str)
        if len(text) <= max_chars:
            break
    return text
