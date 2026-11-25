"""Tests for compact_profile (P3). Uses dicts shaped like tests/test_profile.py expects."""
import json
from enum import Enum

from verisheet.planner_context import compact_profile


class FT(Enum):
    XLSX = "xlsx"


def make_profile(n_cols=3, with_examples=True):
    cols = [{"name": f"c{i}", "candidate_roles": ["measure"],
             "safe_examples": ["abc" * 30] if with_examples else []} for i in range(n_cols)]
    return {
        "file_type": FT.XLSX, "workbook_id": "sha256:" + "a" * 64,
        "sheets": [{"name": "Orders", "row_count": 4, "hidden": False, "columns": cols,
                    "candidate_date_ranges": {"Order_Date": {"min": "2025-01-10", "max": "2025-03-01"}}},
                   {"name": "Hidden_Rates", "row_count": 2, "hidden": True, "columns": cols[:1]}],
        "warnings": [{"code": "HIDDEN_SHEET", "sheet": "Hidden_Rates", "column": None,
                      "evidence": {"big": "x" * 500}}],
    }


def test_compact_is_valid_json_and_keeps_key_facts():
    out = json.loads(compact_profile(make_profile()))
    assert out["file_type"] == "xlsx"                       # enum flattened
    assert out["workbook_id"] == "sha256:" + "a" * 12        # shortened
    orders = out["sheets"][0]
    assert orders["rows"] == 4 and orders["date_ranges"]["Order_Date"]["min"] == "2025-01-10"
    assert out["sheets"][1]["hidden"] is True
    assert out["warnings"] == [{"code": "HIDDEN_SHEET", "sheet": "Hidden_Rates"}]  # no evidence blob


def test_examples_are_truncated():
    out = json.loads(compact_profile(make_profile()))
    assert all(len(e) <= 30 for c in out["sheets"][0]["columns"] for e in c.get("examples", []))


def test_shrinks_to_fit_budget():
    big = make_profile(n_cols=60)
    small = compact_profile(big, max_chars=2500)
    full = compact_profile(big, max_chars=10**6)
    assert len(small) < len(full)
    assert json.loads(small)["sheets"][0].get("columns_omitted", 0) > 0
    assert "examples" not in small


def test_accepts_pydantic_like_objects():
    from types import SimpleNamespace as NS
    col = NS(name="Sales", candidate_roles=["measure"], safe_examples=[])
    sheet = NS(name="data", row_count=3, hidden=False, columns=[col], candidate_date_ranges={})
    prof = NS(file_type=FT.XLSX, workbook_id="sha256:" + "b" * 64, sheets=[sheet],
              warnings=[NS(code="NEGATIVE_VALUES", sheet="data", column="Sales")])
    out = json.loads(compact_profile(prof))
    assert out["sheets"][0]["columns"][0]["name"] == "Sales"
    assert out["warnings"][0]["code"] == "NEGATIVE_VALUES"
