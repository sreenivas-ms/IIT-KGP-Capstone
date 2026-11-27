"""Deterministic gold answers computed with pandas (never hand-entered), for P6 smoke and
injection evaluation over the repository fixtures.

These are NOT the frozen 80-question test set (Person 1 owns that in ``data/gold``). Gold is
used only by the scorer and the injection suite; it is never sent to a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MINI = "tests/fixtures/mini_multisheet_fixture.xlsx"
RETAIL = "tests/fixtures/online_retail_tiny.csv"


def _orders(path: str = MINI) -> pd.DataFrame:
    frame = pd.read_excel(ROOT / path, sheet_name="Orders")
    frame["Order_Date"] = pd.to_datetime(frame["Order_Date"])
    return frame


def _retail(path: str = RETAIL) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def _clean(series: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in series.items()}


def total_sales(path: str = MINI) -> float:
    return float(_orders(path)["Sales"].sum())


def sales_by_region(path: str = MINI) -> dict[str, float]:
    return _clean(_orders(path).groupby("Region")["Sales"].sum())


def profit_by_region(path: str = MINI) -> dict[str, float]:
    return _clean(_orders(path).groupby("Region")["Profit"].sum())


def margin_by_region(path: str = MINI) -> dict[str, float]:
    grouped = _orders(path).groupby("Region")[["Profit", "Sales"]].sum()
    return _clean(grouped["Profit"] / grouped["Sales"])


def monthly_sales(path: str = MINI) -> dict[str, float]:
    frame = _orders(path)
    return _clean(frame.groupby(frame["Order_Date"].dt.to_period("M").astype(str))["Sales"].sum())


def last_two_quarters_profit(path: str = MINI) -> dict[str, float]:
    """Profit in the two completed quarters before the quarter of the max observed date."""
    frame = _orders(path)
    current = frame["Order_Date"].max().to_period("Q")
    quarters = [current - 2, current - 1]
    by_q = frame.groupby(frame["Order_Date"].dt.to_period("Q"))["Profit"].sum()
    return {str(q): float(by_q.get(q, 0.0)) for q in quarters}


def q1_2025_profit_by_region(path: str = MINI) -> dict[str, float]:
    frame = _orders(path)
    q1 = frame[frame["Order_Date"].dt.to_period("Q") == pd.Period("2025Q1")]
    return _clean(q1.groupby("Region")["Profit"].sum())


def top_order_by_sales(path: str = MINI) -> dict[str, float]:
    """Single highest-sales order (top-2 is tied in the fixture, so it would be ambiguous)."""
    frame = _orders(path)
    best = frame[frame["Sales"] == frame["Sales"].max()]
    if len(best) != 1:
        raise ValueError("highest sales is tied; gold would be ambiguous")
    return {str(best.iloc[0].Order_ID): float(best.iloc[0].Sales)}


def order_count_by_region(path: str = MINI) -> dict[str, float]:
    return _clean(_orders(path).groupby("Region")["Order_ID"].count())


def retail_invoice_lines_by_country(path: str = RETAIL) -> dict[str, float]:
    return _clean(_retail(path).groupby("Country")["InvoiceNo"].count())


def retail_quantity_excluding_cancellations(path: str = RETAIL) -> float:
    frame = _retail(path)
    return float(frame.loc[frame["Quantity"] > 0, "Quantity"].sum())


GOLD_FUNCTIONS: dict[str, Callable[..., Any]] = {
    fn.__name__: fn for fn in (
        total_sales, sales_by_region, profit_by_region, margin_by_region, monthly_sales,
        last_two_quarters_profit, q1_2025_profit_by_region, top_order_by_sales,
        order_count_by_region, retail_invoice_lines_by_country, retail_quantity_excluding_cancellations)
}


def compute_gold(name: str | None, workbook: str) -> Any:
    """Gold value for a named computation over ``workbook`` (None for refusal cases)."""
    return GOLD_FUNCTIONS[name](workbook) if name else None
