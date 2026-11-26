# Critic, verification, orchestration and state (Person 5)

Files: `src/verisheet/tools/verify.py`, `src/verisheet/critic.py`, `src/verisheet/orchestrator.py`,
`src/verisheet/state.py`, `prompts/critic.txt` (VERSION 1.1). Tests: `tests/test_verify.py`,
`tests/test_critic.py`, `tests/test_orchestrator.py` (fakes in `tests/p5_support.py`).

```bash
python -m pytest -q tests/test_verify.py tests/test_critic.py tests/test_orchestrator.py
```

## 1. Independent verifier (`tools/verify.py`)

The verifier is the only numeric authority. It reloads the **original workbook** (after the shared
upload guard) and recomputes every `run_analysis` step with its own canonical implementation. It
never reads the Executor's numbers as truth and never asks Qwen.

| Function | Purpose |
|---|---|
| `verify_plan_semantics(question, plan, profile)` | Pre-execution checks: plan structure/cycles/budget, code-like content, unknown sheets/columns, profit without cost data, revenue used as profit, total-vs-average wording, margin = sum/sum, completed-quarter policy, explicit `Qn`/year, year outside data, join key uniqueness, mixed currencies, chart type vs intent. |
| `recompute_check(workbook, plan, result)` | Canonical recomputation of one step and comparison with the Executor's result table and scalars. |
| `verify_chart_data(result, result_table=...)` | Plotted x/y arrays vs the verified table: field mapping, missing/extra/duplicate categories, values, chronological/ranked order, zero baseline for bars, ranking drawn as line. |
| `verify_execution(workbook, plan, results, profile, question)` | Runs everything and returns one `VerificationReport`. Charts are compared with the *independently recomputed* table, not the Executor's table. |

Canonical operations: `group_aggregate`, `top_n`, `time_aggregate`, `quarter_comparison`,
`grouped_ratio`, `filter_rows`, `safe_join`. Any other operation yields
`not_independently_verifiable`, so the answer can never be VERIFIED.

### Step input conventions understood by the verifier

These match the P4 operation catalogue (`tools/analysis.py`, mirrored in `planner.OPERATIONS`).

| Operation | Inputs | Canonical result |
|---|---|---|
| `group_aggregate` | `sheet`, `metrics: {column: agg}`, optional `group_by` | group columns + one column per metric |
| `time_aggregate` | `sheet`, `date_column`, `frequency` (`month`/`quarter`/`year`), `metrics` | period + metric columns, chronological |
| `quarter_comparison` | `sheet`, `date_column`, `metric_column`, optional `aggregation` | the two completed quarters before the quarter of max(date); `Data_Through` wins when present |
| `grouped_ratio` | `sheet`, `numerator`, `denominator`, optional `group_by`, `output_column` (default `ratio`) | sum(numerator)/sum(denominator); zero denominator gives null |
| `top_n` | `sheet`, `by`, `n`, optional `ascending` | raw rows ranked by `by` (stable order), compared row by row; with `group_by` + `metrics`, an aggregated ranking |
| `filter_rows` | `sheet`, `filters` | filtered row count |
| `safe_join` | `join_spec` (`left_sheet`, `right_sheet`, `left_keys`, `right_keys`, `expected_cardinality`) | joined row count and key-uniqueness per declared cardinality |

Aggregations: sum, mean, count, min, max, median, nunique. `plan.filters` are applied to the step's
sheet first (as the Executor does). Filter operators: `eq ne gt gte lt lte in not_in contains between
is_null not_null` (plus `startswith`/`not_startswith`); unknown operators fail closed. Steps are
recomputed from step `inputs`; `plan.metric` is only a fallback when a step names no metric.

Result tables may name columns freely (matching is case/punctuation-insensitive, with a positional
fallback for a single value column). Period labels such as `2024Q3`, `2024-Q3` and `Q3 2024` are
treated as equal. Result tables are read only from the configured artifact directory (path-traversal
guard) and are size-capped.

### Business rules implemented

- Last completed quarter: `Data_Through`/reporting-close metadata wins (complete only if it is the
  quarter's last day); otherwise the quarter before the one containing max(date). The assumption is
  returned in `recomputed_values[step]["notes"]` and shown to the user.
- Growth: `(current - previous) / |previous|`; undefined (null) when previous is 0. Disclosed.
- Margin: ratio of sums, never the mean of row margins.
- Joins: a right-hand key that is not unique must be aggregated first; more output rows than the
  left grain is reported as `join_multiplication`.

## 2. Critic (`critic.py`)

Decision priority: Planner clarification → deterministic CLARIFY → REPLAN → REPAIR → "not
independently verifiable" (never ACCEPT) → Qwen semantic review.

| Category family | Decision |
|---|---|
| execution failure, missing artifact, value/aggregation mismatch, chart value/category/order/baseline/type | REPAIR (targeted step + dependants) |
| wrong period, wrong metric/margin, unknown column, invalid/unsafe plan, join grain/multiplication, unverifiable | REPLAN |
| metric unavailable (e.g. Online Retail profit), revenue-as-profit, period outside data, mixed currencies, ambiguity | CLARIFY |

Qwen (prompt `prompts/critic.txt`) only judges intent match and chart appropriateness. It is called
only after every deterministic check passed, and can veto (REPAIR chart type / REPLAN / CLARIFY) but
never approve a failed or missing numeric check. Qwen3 `<think>` blocks and code fences are
stripped, malformed JSON gets one retry, and if Qwen is down the deterministic verdict stands with a
warning. Modes: `hybrid` (E5), `deterministic` (E4), `llm_only` (E3: may end the loop, never VERIFIED).

## 3. Orchestrator (`orchestrator.py`)

`run_agent(question, workbook_reference, session_id, orchestrator=None) -> AnswerResponse`

Flow: profile → load state → retrieve → plan → deterministic plan gate → execute → verify →
critic → ACCEPT / REPAIR / REPLAN / CLARIFY. Budgets: `MAX_AGENT_ACTIONS` (10; every planner call,
executed step and critic evaluation counts) and `MAX_REPAIRS` (2); at most 2 replans. A plan that
fails the deterministic gate is replanned or clarified **before** any execution, and an
inappropriate chart type is fixed locally without calling the Planner.

Every terminal path returns an `AnswerResponse`:

- `VERIFIED` only when the final report is verified and the Critic accepted. The answer text is
  composed from the recomputed values, never from model text.
- `NOT_VERIFIED` shows no numbers as verified and names the last failure category.
- `CLARIFICATION_REQUIRED` carries `clarification_question`; nothing was executed.
- `FAILED` for unreadable uploads, invalid input or a teammate component that is still a stub.

Each step emits a `TraceEvent` with the shared `trace_id`, model/prompt version, token counts,
latency, failure category, artifact paths and script hashes. If the injected `TraceLogger` fails, the
orchestrator writes JSONL to `results/traces/<trace_id>.jsonl` itself.

Collaborators are injected and may be objects or callables (see the constructor docstring), so the
loop is testable with fakes and works with P2/P3/P4/P6 modules as they land.

## 4. State (`state.py`)

One JSON file per (session, workbook fingerprint), named by a SHA-256 of both IDs (no
user-controlled file names), written atomically, quarantined as `*.corrupt` if unreadable.
Accepted metric/period/filters/group-by are updated **only** for VERIFIED responses; other turns
add a bounded history entry tagged `[not verified]`, `[failed]` or `[clarification requested]`
with no numbers. A new workbook in the same session starts with an empty context.

## Integration notes (checked against the merged P2/P3/P4 code)

- `tests/test_p5_integration.py` runs the real `profile_workbook`, `Planner` (scripted Qwen replies)
  and `Executor`, including injected plot-value and wrong-aggregation faults that are caught and
  repaired, a plan that averages a total (replanned before execution), a ranking drawn as a line
  (fixed locally), Online Retail profit (clarification) and an unconfigured endpoint (FAILED).
- Planner usage (`Planner.last_run`: tokens, latency, repairs) is copied into the planner TraceEvent.
  `PlanValidationError` gives a FAILED answer; REPLAN reuses the same Planner instance and also
  passes `question`/`workbook_profile` in the feedback, as `revise_plan` documents.
- A targeted repair re-runs `Executor.execute_step` after syncing the Executor's plan context
  (`_active_plan_filters`, `_active_chart_spec`), so a chart re-render uses the patched chart spec.
- Executor exceptions raised during plan validation (`ValueError`) are non-retryable and route to
  REPLAN with the message; `retryable=True` ToolErrors route to REPAIR.
- Known upstream limits the verifier will (correctly) refuse to verify, for P4 to address:
  `quarter_comparison` ignores `Data_Through`; `safe_join` checks cardinality on the wrong side for
  `many_to_one` (also noted by P3); `make_chart` ignores `ChartSpec.sort` (reported as a warning).

## Security notes

- No credentials are read or logged by P5 code; Qwen access goes through the shared client.
- Plans containing code-like strings (`__import__`, `exec(`, `subprocess`, ...) are rejected.
- Workbook text, retrieved text and model replies are quoted as untrusted data in the critic prompt.
- Artifact reads are restricted to the artifact root; session IDs are validated.
