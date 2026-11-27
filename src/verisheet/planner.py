"""Planner agent owned by Person 3; produces AnalysisPlan, never numeric answers.

How it works (VS-P3-02/03/04):
  * Prompt = versioned system text (prompts/planner.txt) + a schema outline generated from
    AnalysisPlan.model_json_schema() + the repo's contracts/examples/analysis_plan.json as a
    one-shot example, so the prompt cannot drift from schemas.py.
  * The 8B model only PROPOSES. Code overrides plan_id/max_actions, and `check_plan` enforces
    the policies the model is most likely to break. A violation triggers ONE targeted repair
    call; malformed JSON gets up to two repair calls; then PlanValidationError.
  * Only accepted/verified AgentState fields are shown to the model (never `history`), and
    state from a different workbook fingerprint is ignored.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verisheet.planner_context import compact_profile
from verisheet.qwen_client import QwenClient, extract_json, strip_think
from verisheet.tools.analysis import AGGREGATIONS, FILTER_OPERATORS
from verisheet.schemas import (
    AgentState,
    AnalysisPlan,
    LLMResponse,
    RetrievedContext,
    WorkbookProfile,
)

ROOT = Path(__file__).resolve().parents[2]
# P3-owned few-shot example, consistent with the executor's operations and result columns.
# (The shared contract example is kept for contract tests; it is not executor-consistent.)
EXAMPLE_PATH = ROOT / "prompts" / "planner_example.json"
CONTRACT_EXAMPLE_PATH = ROOT / "contracts" / "examples" / "analysis_plan.json"

INPUT_TOKEN_BUDGET = 8192 - 1024          # API: 8192 total, 1024 reserved for the answer
CHARS_PER_TOKEN = 3.5
MAX_CONTEXT_ITEMS = 5
MAX_CONTEXT_ITEM_CHARS = 500
PLAN_MAX_TOKENS = 1000
MAX_CALLS = 4                              # 1 + JSON repairs + policy repairs, hard cap
MAX_POLICY_REPAIRS = 2
TIME_GRAINS = {"day", "week", "month", "quarter", "year", "date", "fiscal_quarter"}
STATE_FIELDS = ("workbook_id", "accepted_metric", "accepted_period", "accepted_filters",
                "accepted_group_by", "last_verification_id")


# Executor operation catalogue (P4: src/verisheet/tools/analysis.py). Each run_analysis step
# reads workbook sheets directly; only make_chart consumes an earlier step (source_step_id).
# `result` lists the result-table columns so chart x/y columns can be checked.
OPERATIONS: dict[str, dict[str, Any]] = {
    "group_aggregate": {
        "required": ("sheet", "metrics"), "optional": ("group_by",),
        "example": {"sheet": "S", "group_by": ["Region"], "metrics": {"Sales": "sum"}},
        "result": "group_by columns + metric columns"},
    "time_aggregate": {
        "required": ("sheet", "date_column", "frequency", "metrics"), "optional": (),
        "example": {"sheet": "S", "date_column": "Order_Date", "frequency": "month",
                    "metrics": {"Sales": "sum"}},
        "result": "period + metric columns (frequency: month|quarter|year)"},
    "quarter_comparison": {
        "required": ("sheet", "date_column", "metric_column"), "optional": ("aggregation",),
        "example": {"sheet": "S", "date_column": "Order_Date", "metric_column": "Profit",
                    "aggregation": "sum"},
        "result": "quarter, <metric_column>, period_role, change_vs_previous, percent_change "
                  "(the two completed quarters before the quarter of the max observed date)"},
    "grouped_ratio": {
        "required": ("sheet", "numerator", "denominator"), "optional": ("group_by", "output_column"),
        "example": {"sheet": "S", "numerator": "Profit", "denominator": "Sales",
                    "group_by": ["Region"], "output_column": "margin"},
        "result": "group_by + numerator + denominator + output_column; use for margin = sum/sum"},
    "top_n": {
        "required": ("sheet", "by", "n"), "optional": ("ascending",),
        "example": {"sheet": "S", "by": "Sales", "n": 5, "ascending": False},
        "result": "the sheet's rows sorted by `by` (raw rows, not aggregated)"},
    "filter_rows": {
        "required": ("sheet", "filters"), "optional": (),
        "example": {"sheet": "S", "filters": [{"column": "Region", "operator": "eq",
                                               "value": "East"}]},
        "result": "the sheet's columns"},
    "safe_join": {
        "required": ("join_spec",), "optional": (),
        "example": {"join_spec": {"left_sheet": "Orders", "right_sheet": "Product_Costs",
                                  "left_keys": ["Product_ID"], "right_keys": ["Product_ID"],
                                  "expected_cardinality": "many_to_one"}},
        "result": "left columns + right columns"},
}
FREQUENCIES = {"month", "monthly", "quarter", "quarterly", "year", "yearly"}
_COLUMN_INPUTS = ("date_column", "metric_column", "by", "numerator", "denominator")


def operations_hint() -> str:
    """Prompt text listing every executor operation with an inputs example and result columns."""
    lines = [f"- {name}: inputs {json.dumps(spec['example'], separators=(',', ':'))} "
             f"-> result columns: {spec['result']}" for name, spec in OPERATIONS.items()]
    lines.append(f"aggregations: {'|'.join(sorted(AGGREGATIONS))}; "
                 f"filter operators: {'|'.join(sorted(FILTER_OPERATORS))}")
    return "\n".join(lines)


class PlanValidationError(ValueError):
    """No valid, policy-compliant plan could be produced within the call budget."""


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", str(obj))


def _json(obj: Any) -> str:
    return json.dumps(_dump(obj), ensure_ascii=False, separators=(",", ":"), default=str)


def _resolve(node: dict, defs: dict) -> dict:
    while "$ref" in node:
        node = defs.get(node["$ref"].split("/")[-1], {})
    return node


def _simplify(node: dict, defs: dict, depth: int = 0) -> Any:
    """Turn a JSON-Schema node into a compact outline (A|B = enum).

    Optional keys are NOT marked inline: Qwen copied "key?" markers into its JSON. Required
    keys are listed separately by ``schema_hint``.
    """
    node = _resolve(node, defs)
    if depth > 5:
        return "object"
    for key in ("anyOf", "oneOf"):
        if key in node:
            opts = [o for o in node[key] if _resolve(o, defs).get("type") != "null"]
            if len(opts) == 1:
                return _simplify(opts[0], defs, depth)
            return "|".join(str(_simplify(o, defs, depth + 1)) for o in opts)[:80]
    if "enum" in node:
        return "|".join(str(v) for v in node["enum"])
    typ = node.get("type")
    if typ == "object" or "properties" in node:
        if not node.get("properties"):
            return "object"
        out = {k: _simplify(v, defs, depth + 1) for k, v in node["properties"].items()}
        if "inputs" in out:  # live runs: Qwen closes `inputs` with an extra "}" -> keep it last
            out["inputs"] = out.pop("inputs")
        return out
    if typ == "array":
        return [_simplify(node.get("items", {}), defs, depth + 1)]
    return typ or "any"


def schema_hint(model_cls: Any = AnalysisPlan) -> str:
    """Compact outline of the model's JSON schema plus its required keys, for the prompt."""
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})
    outline = json.dumps(_simplify(schema, defs), separators=(",", ":"))
    required = [f"plan: {', '.join(schema.get('required', []))}"]
    for name in ("MetricSpec", "AnalysisStep", "JoinSpec", "ChartSpec"):
        if defs.get(name, {}).get("required"):
            required.append(f"{name}: {', '.join(defs[name]['required'])}")
    return (outline + "\nRequired keys -> " + "; ".join(required)
            + "\nKey names are exactly as shown. Omit optional keys you do not need.")


def _example_plan() -> str:
    for path in (EXAMPLE_PATH, CONTRACT_EXAMPLE_PATH):
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))
        except (OSError, ValueError):
            continue
    return "(no example available)"


def _format_context(context: Any, limit: int) -> str:
    rows = [{"item_id": it.item_id, "source": it.source_path, "title": it.title,
             "text": (it.content or "")[:MAX_CONTEXT_ITEM_CHARS]}
            for it in (getattr(context, "items", None) or [])[:MAX_CONTEXT_ITEMS]]
    text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _workbook_id(profile: Any) -> str | None:
    return profile.get("workbook_id") if isinstance(profile, dict) else getattr(profile, "workbook_id", None)


def usable_state(state: Any, profile: Any) -> tuple[dict | None, str | None]:
    """Reduce AgentState to accepted/verified facts. Returns (facts, reason_ignored).

    `history` (unverified summaries) is never passed on. State whose workbook_id differs
    from the active workbook is ignored.
    """
    if state is None:
        return None, None
    data = _dump(state)
    wb_state, wb_now = data.get("workbook_id"), _workbook_id(profile)
    if wb_state and wb_now and wb_state != wb_now:
        return None, "workbook_changed"
    facts = {k: data[k] for k in STATE_FIELDS if data.get(k) not in (None, [], {}, "")}
    facts.pop("workbook_id", None)
    return (facts or None), None


def build_prompts(
    system_prompt: str,
    question: str,
    profile: Any,
    context: Any,
    state_facts: dict | None = None,
    feedback: Any = None,
    prior_plan: Any = None,
    action_budget: int = 10,
) -> tuple[str, str]:
    """Return (system, user) strings sized to fit INPUT_TOKEN_BUDGET."""
    system = (system_prompt.replace("{{SCHEMA}}", schema_hint())
                           .replace("{{OPERATIONS}}", operations_hint())
                           .replace("{{EXAMPLE}}", _example_plan()))
    q = question.strip()
    prev = json.dumps(state_facts, default=str, separators=(",", ":")) if state_facts else ""
    fb = _json(feedback) if feedback else ""
    pp = _json(prior_plan) if prior_plan else ""
    fixed = _tokens(system) + sum(_tokens(x) for x in (q, prev, fb, pp)) + 60
    room = max((INPUT_TOKEN_BUDGET - fixed) * CHARS_PER_TOKEN, 3000)

    blocks = [f"<question>\n{q}\n</question>",
              f"<profile>\n{compact_profile(profile, max_chars=int(room * 0.65))}\n</profile>",
              f"<context>\n{_format_context(context, int(room * 0.35))}\n</context>",
              f"<budget>{action_budget}</budget>"]
    if prev:
        blocks.append(f"<previous>\n{prev[:int(room * 0.3)]}\n</previous>")
    if pp:
        blocks.append(f"<prior_plan>\n{pp}\n</prior_plan>")
    if fb:
        blocks.append(f"<feedback>\n{fb}\n</feedback>")
    blocks.append("Return the JSON plan now.")
    return system, "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Parsing and policy checks
# --------------------------------------------------------------------------- #
def parse_plan_response(raw_content: str) -> AnalysisPlan:
    """Parse Qwen output into AnalysisPlan; reject prose, missing fields and extra fields.

    Raises:
        ValueError: no JSON, prose around the JSON, or schema validation failure
            (pydantic.ValidationError is a ValueError; AnalysisPlan forbids extra fields).
    """
    return _parse_plan(raw_content)[0]


def _parse_plan(raw_content: str) -> tuple[AnalysisPlan, bool]:
    """Strict parse; if that fails on a reply that is *only* JSON, retry once after balancing
    brackets. Returns (plan, repaired). The repaired text must still pass the strict schema.
    """
    text = strip_think(raw_content).strip()
    try:
        body = extract_json(text)
        leftover = re.sub(r"```(?:json)?", "", text.replace(body, "", 1)).strip()
        if leftover:
            raise ValueError("model output contains prose outside the JSON object")
        return AnalysisPlan.model_validate(json.loads(body)), False
    except ValueError as strict_error:
        bare = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        if not (bare.startswith("{") and bare.endswith("}")):
            raise                                  # prose or truncated output: never guess
        try:
            return AnalysisPlan.model_validate(_first_object_with_empty_tail(balance_brackets(bare))), True
        except ValueError:
            raise strict_error from None


_EMPTY_STRING_RE = re.compile(r'"(?:\\[nrt]|\s)*"')
_JUNK_TAIL_RE = re.compile(r"[\s,:}\]]*")


def _first_object_with_empty_tail(text: str) -> Any:
    """Parse the first JSON value; allow only content-free junk after it.

    Live Qwen sometimes appends `,"\\n"}` after a complete plan. A tail made only of commas,
    closers and empty/whitespace strings carries no content and is dropped; anything else
    (words, numbers, keys) raises ValueError so prose is never silently discarded.
    """
    value, end = json.JSONDecoder().raw_decode(text)
    tail = _EMPTY_STRING_RE.sub("", text[end:])
    if not _JUNK_TAIL_RE.fullmatch(tail):
        raise ValueError("content after the JSON object")
    return value


def balance_brackets(text: str) -> str:
    """Insert missing closing brackets/braces in an otherwise complete JSON text.

    Fixes the slip seen in live Qwen runs: a key appearing while a list is still open (missing
    "]"). Strings are copied verbatim; a "}" with no object open at the top of the stack is
    dropped; unclosed brackets are closed at the end. Callers must re-validate the result.
    """
    out: list[str] = []
    stack: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            start, j = len(out), i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            k = j + 1
            while k < n and text[k].isspace():
                k += 1
            if k < n and text[k] == ":" and stack and stack[-1] == "[":
                closers = []
                while stack and stack[-1] == "[":
                    stack.pop()
                    closers.append("]")
                pos = start
                while pos > 0 and out[pos - 1].isspace():
                    pos -= 1
                if pos > 0 and out[pos - 1] == ",":
                    pos -= 1
                out.insert(pos, "".join(closers))
            i = j + 1
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if not stack or stack[-1] != "{":       # stray "}" (nothing or a list open): drop
                i += 1
                continue
            stack.pop()
        elif ch == "]":
            while stack and stack[-1] == "{":
                stack.pop()
                out.append("}")
            if not stack:
                i += 1
                continue
            stack.pop()
        out.append(ch)
        i += 1
    out.extend("}" if b == "{" else "]" for b in reversed(stack))
    return "".join(out)


_PROFIT_WORDS = re.compile(r"\b(profit|margin|cogs|cost of goods)\b", re.IGNORECASE)
_MARGIN_WORD = re.compile(r"\bmargin\b", re.IGNORECASE)
_QUARTER_WORDS = re.compile(
    r"\b(last|latest|previous|prior)\b.*\bquarter\b|\bquarter\b.*\b(last|latest)\b", re.IGNORECASE)
_AVG_AGGS = {"mean", "avg", "average"}
_ADD_AGGS = {"sum", "mean", "avg", "average"}


def _index_profile(profile: Any) -> tuple[dict[str, set[str]], set[str], set[str], set[str]]:
    """-> (columns per sheet, hidden sheets, identifier columns, warning codes)."""
    data = _dump(profile) or {}
    cols: dict[str, set[str]] = {}
    hidden: set[str] = set()
    idents: set[str] = set()
    for s in data.get("sheets", []):
        cols[s["name"]] = {c["name"] for c in s.get("columns", [])}
        if s.get("hidden"):
            hidden.add(s["name"])
        idents |= {c["name"] for c in s.get("columns", []) if "identifier" in c.get("candidate_roles", [])}
    return cols, hidden, idents, {w.get("code") for w in data.get("warnings", [])}


_NEW_PERIOD_RE = re.compile(
    r"\b(19|20)\d{2}\b|\bq[1-4]\b|\b(quarter|quarters|month|months|year|years|week|weeks|ytd|mtd|"
    r"all[- ]time|overall|since|between|until)\b", re.IGNORECASE)
_SAME_PERIOD_RE = re.compile(r"\b(same|that|this)\s+(quarter|month|year|week|period)\b", re.IGNORECASE)


def _follow_up_period(question: str, state_facts: dict | None) -> dict | None:
    """Verified period to carry into this plan, or None (no period / question names a new one)."""
    period = (state_facts or {}).get("accepted_period") or {}
    if not (period.get("start") and period.get("end")):
        return None
    if _NEW_PERIOD_RE.search(question or "") and not _SAME_PERIOD_RE.search(question or ""):
        return None
    return {"start": str(period["start"]), "end": str(period["end"])}


def _covers_period(p: dict, period: dict, date_cols: set[str]) -> bool:
    for f in p.get("filters", []):
        if (isinstance(f, dict) and f.get("column") in date_cols and f.get("operator") == "between"
                and [str(v) for v in (f.get("value") or [])] == [period["start"], period["end"]]):
            return True
    return False


def _plan_date_columns(p: dict, profile: Any) -> set[str]:
    """Date columns shared by every sheet the plan's run_analysis steps read."""
    data = _dump(profile) or {}
    by_sheet = {s["name"]: set((s.get("candidate_date_ranges") or {}).keys()) for s in data.get("sheets", [])}
    sheets = {(st.get("inputs") or {}).get("sheet") for st in p.get("steps", [])
              if st["tool"] == "run_analysis"} - {None}
    sheets = sheets or set(p.get("source_sheets", []))
    common = None
    for sheet in sheets:
        common = by_sheet.get(sheet, set()) if common is None else common & by_sheet.get(sheet, set())
    return common or set()


def carry_forward_period(plan: AnalysisPlan, profile: Any, question: str,
                         state_facts: dict | None) -> tuple[AnalysisPlan, list[str]]:
    """Keep the previously VERIFIED period on a follow-up that does not name a new period.

    Adds a plan-level ``between`` filter on the single date column (the Executor applies plan
    filters; the verifier applies them too) plus a matching ``date_range``. Ambiguous cases
    (several date columns) are left to ``check_plan`` so the model must choose.
    """
    period = _follow_up_period(question, state_facts)
    p = _dump(plan)
    if period is None or p.get("clarification_needed"):
        return plan, []
    date_cols = _plan_date_columns(p, profile)
    if _covers_period(p, period, date_cols) or len(date_cols) != 1:
        return plan, []
    column = next(iter(date_cols))
    rule = {"column": column, "operator": "between", "value": [period["start"], period["end"]]}
    updated = {**p, "filters": [*p.get("filters", []), rule], "date_range": dict(period)}
    note = f"kept verified period {period['start']}..{period['end']} as filter on {column}"
    updated["assumptions"] = [*p.get("assumptions", []), f"Follow-up: {note} (from accepted state)."]
    return AnalysisPlan.model_validate(updated), [note]


def check_plan(plan: AnalysisPlan, profile: Any, question: str = "",
               state_facts: dict | None = None) -> list[str]:
    """Deterministic policy checks; returns human-readable problems (empty list = OK).

    Checks: step ids/dependencies/budget, unknown or hidden sheets, unknown columns
    (metric, group_by, joins, filters), profit asked without profit/cost data, margin not
    sum/sum, identifier columns aggregated, ambiguous dates, undisclosed quarter assumption.
    Chart columns are NOT checked here because they name result-table columns.
    """
    p = _dump(plan)
    problems: list[str] = []

    # structure (cheap; Executor re-validates)
    seen: list[str] = []
    for st in p.get("steps", []):
        sid = st["step_id"]
        if sid in seen:
            problems.append(f"duplicate step_id '{sid}'.")
        for dep in st.get("depends_on", []):
            if dep not in seen:
                problems.append(f"step '{sid}' depends on '{dep}', which is not an earlier step.")
        seen.append(sid)
    if len(seen) > p.get("max_actions", 10):
        problems.append(f"plan has {len(seen)} steps but max_actions is {p.get('max_actions')}.")

    if p.get("clarification_needed"):
        return problems                         # a clarification plan needs no data checks
    if not any(st["tool"] == "run_analysis" for st in p.get("steps", [])):
        problems.append("the plan computes nothing: add a run_analysis step (sample_rows and "
                        "inspect_workbook only preview data and never answer the question).")

    cols, hidden, idents, warn_codes = _index_profile(profile)
    all_cols = set().union(*cols.values()) if cols else set()
    q = question or ""

    # sheets
    for s in p.get("source_sheets", []):
        if s not in cols:
            problems.append(f"sheet '{s}' is not in <profile> (sheets: {sorted(cols)}).")
        elif s in hidden and s.lower() not in q.lower():
            problems.append(f"sheet '{s}' is hidden and the user did not name it.")

    # columns
    refs: list[str] = list((p.get("metric") or {}).get("source_columns", []))
    refs += [g for g in p.get("group_by", []) if g.lower() not in TIME_GRAINS]
    refs += [f["column"] for f in p.get("filters", []) if isinstance(f.get("column"), str)]
    unknown = sorted({r for r in refs if r not in all_cols})
    for j in p.get("joins", []):
        for sheet, keys in ((j["left_sheet"], j["left_keys"]), (j["right_sheet"], j["right_keys"])):
            if sheet in cols:
                unknown += [k for k in keys if k not in cols[sheet]]
    if unknown:
        problems.append(f"these columns are not in <profile>: {sorted(set(unknown))}. "
                        "Use only profiled column names.")

    metric = p.get("metric") or {}
    agg = str(metric.get("aggregation", "")).lower()
    bad_ident = sorted(set(metric.get("source_columns", [])) & idents)
    if bad_ident and agg in _ADD_AGGS:
        problems.append(f"{bad_ident} are identifier columns; never sum or average an identifier.")

    # profit / margin
    if _PROFIT_WORDS.search(q):
        lower = {c.lower() for c in all_cols}
        has_profit = any("profit" in c for c in lower)
        has_cost = any(k in c for c in lower for k in ("cost", "cogs", "expense"))
        if not has_profit and not has_cost:
            problems.append("the workbook has no profit or cost data, so set clarification_needed=true "
                            "and ask for costs; do not use sales as profit.")
    formula = str(metric.get("formula") or "")
    if re.search(r"[A-Za-z_]\w*\s*[*x×]\s*[A-Za-z_]\w*", formula) and not p.get("clarification_needed"):
        problems.append(f"metric formula '{formula}' multiplies columns, but no OPERATION computes "
                        "a product of columns and no such column exists: set "
                        "clarification_needed=true and say the value cannot be computed yet.")
    if _MARGIN_WORD.search(q) and agg in _AVG_AGGS:
        problems.append("margin must be sum(Profit)/sum(Sales), not an average of row-level margins.")

    # executor compatibility (operations, inputs, filters, chart columns)
    problems += _check_steps(p, cols)

    # follow-up: keep the verified period unless the user names a new one (VS-P3-04)
    period = _follow_up_period(q, state_facts)
    if period:
        date_cols = _plan_date_columns(p, profile)
        if not _covers_period(p, period, date_cols):
            col = sorted(date_cols)[0] if date_cols else "<date column>"
            problems.append(
                f"this is a follow-up: keep the verified period by adding to plan filters "
                f'{{"column": "{col}", "operator": "between", "value": ["{period["start"]}", '
                f'"{period["end"]}"]}} and date_range {{"start": "{period["start"]}", "end": "{period["end"]}"}}.')

    # currency: summing amounts across several currencies needs a rate or a currency filter
    currency_cols = _multi_valued_currency_columns(profile)
    if currency_cols and agg in _ADD_AGGS:
        scoped = {f.get("column") for f in p.get("filters", []) if isinstance(f, dict)}
        scoped |= set(p.get("group_by", []))
        if not currency_cols & scoped:
            problems.append(f"the data has several currencies ({sorted(currency_cols)}): filter "
                            "or group by currency, or set clarification_needed=true and ask for "
                            "an exchange-rate source. Never add different currencies.")

    # dates
    if (p.get("date_policy") or p.get("date_range")) and warn_codes & {
            "AMBIGUOUS_DATE_FORMAT", "INCONSISTENT_DATE_FORMAT"}:
        problems.append("<profile> warns the date format is ambiguous or inconsistent: set "
                        "clarification_needed=true and ask which convention applies.")
    if _QUARTER_WORDS.search(q):
        text = " ".join(p.get("assumptions", [])).lower()
        if "quarter" not in text:
            problems.append("state in assumptions how the last completed quarter was chosen "
                            "(Data_Through, or the quarter before the max observed date).")
        elif any(_norm(c) == "data_through" for c in all_cols) and "data_through" not in _norm(text):
            problems.append("the workbook has a Data_Through field: base the last completed "
                            "quarter on Data_Through and say so in assumptions.")
    return problems


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower())


def _multi_valued_currency_columns(profile: Any) -> set[str]:
    data = _dump(profile) or {}
    found = set()
    for sheet in data.get("sheets", []):
        for col in sheet.get("columns", []):
            name = _norm(col["name"])
            if ("currency" in name or name in {"ccy", "cur", "curr"}) and (col.get("unique_count") or 0) > 1:
                found.add(col["name"])
    return found


def _filter_problems(filters: Any, where: str, known: set[str] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(filters, list):
        return [f"{where}: filters must be a list of {{column, operator, value}} objects."]
    for f in filters:
        if not isinstance(f, dict) or not {"column", "operator"} <= f.keys():
            problems.append(f"{where}: each filter needs 'column' and 'operator'.")
        elif f["operator"] not in FILTER_OPERATORS:
            problems.append(f"{where}: filter operator '{f['operator']}' is not allowed; use one "
                            f"of {sorted(FILTER_OPERATORS)}.")
        elif known is not None and f["column"] not in known:
            problems.append(f"{where}: filter column '{f['column']}' is not in that sheet.")
    return problems


def _result_columns(step: dict, cols: dict[str, set[str]]) -> set[str] | None:
    """Columns the executor will output for a run_analysis step (None = unknown)."""
    op, i = step.get("operation"), step.get("inputs") or {}
    metrics = list((i.get("metrics") or {}).keys()) if isinstance(i.get("metrics"), dict) else []
    group = list(i.get("group_by") or [])
    if op == "group_aggregate":
        return set(group + metrics)
    if op == "time_aggregate":
        return {"period", *metrics}
    if op == "quarter_comparison":
        return {"quarter", i.get("metric_column"), "period_role", "change_vs_previous", "percent_change"}
    if op == "grouped_ratio":
        return {*group, i.get("numerator"), i.get("denominator"), i.get("output_column", "ratio")}
    if op in ("top_n", "filter_rows"):
        return cols.get(i.get("sheet"))
    return None


def normalize_plan(plan: AnalysisPlan, profile: Any) -> tuple[AnalysisPlan, list[str]]:
    """Fix chart column names that differ from a real result column only by letter case.

    Only an unambiguous case-insensitive match is rewritten; every change is returned so it
    is visible in ``last_run["auto_fixes"]``. Nothing else in the plan is altered.
    """
    p = _dump(plan)
    spec = p.get("chart_spec")
    if not spec:
        return plan, []
    cols, *_ = _index_profile(profile)
    steps = {st["step_id"]: st for st in p.get("steps", [])}
    chart = next((st for st in p.get("steps", []) if st["tool"] == "make_chart"), None)
    source = steps.get(((chart or {}).get("inputs") or {}).get("source_step_id"))
    produced = {c for c in (_result_columns(source, cols) or set()) if c} if source else set()
    fixes: list[str] = []

    def fix(name: Any) -> Any:
        if not isinstance(name, str) or name in produced:
            return name
        matches = [c for c in produced if c.lower() == name.lower()]
        if len(matches) == 1:
            fixes.append(f"chart column '{name}' -> '{matches[0]}'")
            return matches[0]
        return name

    new_spec = {**spec, "x_column": fix(spec.get("x_column")), "y_column": fix(spec.get("y_column")),
                "series_columns": [fix(c) for c in spec.get("series_columns", [])]}
    if not fixes:
        return plan, []
    return AnalysisPlan.model_validate({**p, "chart_spec": new_spec}), fixes


def _check_steps(p: dict, cols: dict[str, set[str]]) -> list[str]:
    """Validate steps against the executor's operation catalogue and the profile."""
    problems = _filter_problems(p.get("filters", []), "plan filters", None)
    steps = {st["step_id"]: st for st in p.get("steps", [])}
    for st in p.get("steps", []):
        sid, tool, i = st["step_id"], st["tool"], st.get("inputs") or {}
        where = f"step '{sid}'"
        if tool == "run_analysis":
            op = st.get("operation")
            if op not in OPERATIONS:
                problems.append(f"{where}: operation '{op}' does not exist; use one of "
                                f"{sorted(OPERATIONS)}.")
                continue
            missing = [k for k in OPERATIONS[op]["required"] if k not in i]
            if missing:
                problems.append(f"{where}: {op} needs inputs {missing}; example "
                                f"{json.dumps(OPERATIONS[op]['example'])}.")
                continue
            if op == "safe_join":
                spec = i["join_spec"] if isinstance(i["join_spec"], dict) else {}
                need = {"left_sheet", "right_sheet", "left_keys", "right_keys", "expected_cardinality"}
                if not need <= spec.keys():
                    problems.append(f"{where}: join_spec needs {sorted(need)}.")
                elif spec.get("aggregate_before_join"):
                    problems.append(f"{where}: the executor cannot pre-aggregate inside a join; "
                                    "set aggregate_before_join=false or ask a clarification.")
                else:
                    for side in ("left", "right"):
                        sheet = spec[f"{side}_sheet"]
                        if sheet not in cols:
                            problems.append(f"{where}: join sheet '{sheet}' is not in <profile>.")
                        else:
                            bad = [k for k in spec[f"{side}_keys"] if k not in cols[sheet]]
                            if bad:
                                problems.append(f"{where}: join keys {bad} are not in '{sheet}'.")
                continue
            sheet = i.get("sheet")
            if sheet not in cols:
                problems.append(f"{where}: inputs.sheet '{sheet}' is not in <profile>.")
                continue
            known = cols[sheet]
            named = [i[k] for k in _COLUMN_INPUTS if isinstance(i.get(k), str)]
            named += [g for g in (i.get("group_by") or []) if isinstance(g, str)]
            metrics = i.get("metrics")
            if "metrics" in i and (not isinstance(metrics, dict) or not metrics):
                problems.append(f"{where}: metrics must be an object like {{\"Sales\": \"sum\"}}.")
                metrics = {}
            named += list((metrics or {}).keys())
            bad_cols = sorted({c for c in named if c not in known})
            if bad_cols:
                problems.append(f"{where}: columns {bad_cols} are not in sheet '{sheet}'.")
            bad_aggs = sorted({str(a) for a in (metrics or {}).values() if a not in AGGREGATIONS})
            if op == "quarter_comparison" and i.get("aggregation", "sum") not in AGGREGATIONS - {"nunique"}:
                bad_aggs.append(str(i.get("aggregation")))
            if bad_aggs:
                problems.append(f"{where}: aggregations {bad_aggs} are not allowed; use "
                                f"{sorted(AGGREGATIONS)}.")
            if op == "time_aggregate" and str(i.get("frequency", "")).lower() not in FREQUENCIES:
                problems.append(f"{where}: frequency must be month, quarter or year.")
            if op == "top_n" and (isinstance(i.get("n"), bool) or not isinstance(i.get("n"), int)
                                  or i.get("n", 0) < 1):
                problems.append(f"{where}: n must be a positive integer.")
            if op == "filter_rows":
                problems += _filter_problems(i.get("filters"), where, known)
        elif tool == "sample_rows":
            if i.get("sheet") not in cols:
                problems.append(f"{where}: sample_rows needs inputs.sheet naming a profiled sheet.")
        elif tool == "make_chart":
            source = i.get("source_step_id")
            if not p.get("chart_spec"):
                problems.append(f"{where}: make_chart needs a chart_spec.")
            if source not in steps or source not in st.get("depends_on", []):
                problems.append(f"{where}: make_chart needs inputs.source_step_id naming an "
                                "earlier step listed in depends_on.")
            elif p.get("chart_spec"):
                produced = _result_columns(steps[source], cols)
                spec = p["chart_spec"]
                wanted = [c for c in (spec.get("x_column"), spec.get("y_column"),
                                      *spec.get("series_columns", [])) if c]
                missing_cols = [c for c in wanted if produced is not None and c not in produced]
                if missing_cols:
                    problems.append(f"{where}: chart columns {missing_cols} are not in the result "
                                    f"of step '{source}' (it outputs {sorted(c for c in produced if c)}).")
    return problems


def _short_error(exc: Exception, limit: int = 500) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:6]]
        return "schema validation failed -> " + "; ".join(parts)
    return str(exc)[:limit]


def _max_actions() -> int:
    try:
        from verisheet.config import get_settings
        return get_settings().max_agent_actions
    except (ImportError, ValueError):   # config unavailable or invalid -> documented default
        return 10


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class Planner:
    """Convert a natural-language analytical question into a dependency-aware AnalysisPlan.

    Inputs are user query, WorkbookProfile, RetrievedContext and previously accepted AgentState.
    Output must validate against `AnalysisPlan` from schemas.py. Planner must not calculate
    numeric answers, write arbitrary Python, or mark any output verified.
    """

    def __init__(self, client: QwenClient | None = None, prompt_path: str = "prompts/planner.txt"):
        """Create planner with shared Qwen client and versioned system prompt.

        Raises:
            FileNotFoundError: prompt file missing.
            ValueError: prompt lacks a 'VERSION x.y' header or the {{SCHEMA}} placeholder.
        """
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.is_absolute() and not self.prompt_path.exists():
            self.prompt_path = ROOT / self.prompt_path
        self.system_prompt = self.prompt_path.read_text(encoding="utf-8")
        m = re.search(r"VERSION\s+([\w.\-]+)", self.system_prompt)
        if not m or "{{SCHEMA}}" not in self.system_prompt:
            raise ValueError(f"{self.prompt_path} needs a 'VERSION x.y' header and {{{{SCHEMA}}}}")
        self.prompt_version = m.group(1)
        self.client = client or QwenClient()
        self.last_run: dict[str, Any] = {}          # tokens/latency/repairs for traces (P6)
        self._contexts: dict[str, dict] = {}        # plan_id -> inputs, used by revise_plan

    def generate_plan(
        self,
        question: str,
        workbook_profile: WorkbookProfile,
        retrieved_context: RetrievedContext,
        previous_state: AgentState | None = None,
        feedback: dict | None = None,
    ) -> AnalysisPlan:
        """Ask Qwen for a strict JSON plan and validate it with the shared Pydantic schema.

        Detects unanswerable metrics (e.g. Online Retail profit) via check_plan and forces a
        CLARIFY plan instead of a proxy metric.

        Raises:
            PlanValidationError: no valid, policy-compliant plan within the call budget.
            QwenError: endpoint failure after retries.
        """
        facts, ignored = usable_state(previous_state, workbook_profile)
        system, user = build_prompts(self.system_prompt, question, workbook_profile,
                                     retrieved_context, facts, feedback,
                                     action_budget=_max_actions())
        plan = self._ask(system, user, workbook_profile, question,
                         plan_id=f"plan-{uuid.uuid4().hex[:8]}", state_ignored=ignored,
                         state_facts=facts)
        self._contexts[plan.plan_id] = {"question": question, "profile": workbook_profile,
                                        "context": retrieved_context}
        return plan

    def revise_plan(self, prior_plan: AnalysisPlan, critic_feedback: dict, state: AgentState) -> AnalysisPlan:
        """Produce a targeted revised plan after REPLAN; preserve validated steps where possible.

        The original question, profile and retrieved context are remembered per plan_id from
        generate_plan (use the same Planner instance). If this Planner did not create
        `prior_plan`, pass ``question`` and ``workbook_profile`` inside ``critic_feedback``.
        The revised plan keeps its lineage: ``plan-ab12`` -> ``plan-ab12.r1`` -> ``.r2``.

        Raises:
            ValueError: the original question/profile cannot be found.
            PlanValidationError / QwenError: as for generate_plan.
        """
        base = prior_plan.plan_id
        ctx = self._contexts.get(base) or {}
        fb = dict(critic_feedback or {})
        question = ctx.get("question") or fb.pop("question", None)
        profile = ctx.get("profile") or fb.pop("workbook_profile", None)
        fb.pop("question", None), fb.pop("workbook_profile", None)
        if not question or profile is None:
            raise ValueError("revise_plan needs the original question and profile: call "
                             "generate_plan on this Planner first, or include 'question' and "
                             "'workbook_profile' in critic_feedback")
        match = re.match(r"^(.*)\.r(\d+)$", base)
        new_id = f"{match.group(1)}.r{int(match.group(2)) + 1}" if match else f"{base}.r1"
        facts, ignored = usable_state(state, profile)
        system, user = build_prompts(self.system_prompt, question, profile, ctx.get("context"),
                                     facts, fb, prior_plan, _max_actions())
        plan = self._ask(system, user, profile, question, plan_id=new_id, state_ignored=ignored,
                         state_facts=facts)
        self._contexts[plan.plan_id] = {"question": question, "profile": profile,
                                        "context": ctx.get("context")}
        return plan

    # ---- internals --------------------------------------------------------
    def _ask(self, system: str, user: str, profile: Any, question: str,
             plan_id: str, state_ignored: str | None = None,
             state_facts: dict | None = None) -> AnalysisPlan:
        calls: list[LLMResponse] = []
        extra, last_error = "", ""
        json_repairs, policy_repairs = 0, 0
        plan: AnalysisPlan | None = None
        problems: list[str] = []
        auto_fixes: list[str] = []

        for _ in range(MAX_CALLS):
            resp = self.client.complete(system, user + extra, temperature=0.0,
                                        max_tokens=PLAN_MAX_TOKENS)
            calls.append(resp)
            try:
                if resp.finish_reason == "length":
                    raise ValueError("the reply was cut off; return a shorter, compact JSON plan")
                plan, repaired = _parse_plan(resp.content)
                if repaired:
                    auto_fixes.append("balanced JSON brackets in model reply")
            except ValueError as exc:
                last_error, plan = _short_error(exc), None
                json_repairs += 1
                extra = (f"\n<invalid_reply>\n{resp.content[:1200]}\n</invalid_reply>\n"
                         f"<error>{last_error}</error>\n"
                         "Return ONLY one corrected JSON object. No prose, no markdown.")
                continue
            plan, fixed = normalize_plan(plan, profile)
            auto_fixes += fixed
            plan, fixed = carry_forward_period(plan, profile, question, state_facts)
            auto_fixes += fixed
            problems = check_plan(plan, profile, question, state_facts)
            if not problems:
                break
            if policy_repairs >= MAX_POLICY_REPAIRS:
                break
            policy_repairs += 1
            compact = plan.model_dump_json(exclude_defaults=True)
            extra = (f"\n<previous_plan>\n{compact}\n</previous_plan>\n"
                     "<problems>\n- " + "\n- ".join(problems) + "\n</problems>\n"
                     "Fix these problems and return the full corrected JSON only.")

        self.last_run = {
            "prompt_version": self.prompt_version,
            "model": calls[-1].model if calls else "",
            "calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "latency_ms": round(sum(c.latency_ms for c in calls), 1),
            "json_repairs": json_repairs,
            "policy_repairs": policy_repairs,
            "auto_fixes": auto_fixes,
            "unresolved_problems": problems,
            "state_ignored": state_ignored,
        }
        if plan is None:
            raise PlanValidationError(f"model did not return a valid plan: {last_error}")
        if problems:
            raise PlanValidationError("; ".join(problems))
        return plan.model_copy(update={"plan_id": plan_id,
                                       "max_actions": min(plan.max_actions, _max_actions())})


def generate_plan(
    question: str,
    workbook_profile: WorkbookProfile,
    retrieved_context: RetrievedContext,
    previous_state: AgentState | None = None,
    feedback: dict | None = None,
    planner: Planner | None = None,
) -> AnalysisPlan:
    """Canonical function wrapper around Planner.generate_plan for dependency injection.

    `planner` should be supplied by Orchestrator/tests; when omitted, construct the default
    Planner. The public input/output schema is the same as the Planner method.
    """
    planner = planner or Planner()
    return planner.generate_plan(question, workbook_profile, retrieved_context,
                                 previous_state, feedback)
