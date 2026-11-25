# Planner and Qwen client (Person 3)

Owner files: `src/verisheet/qwen_client.py`, `src/verisheet/planner.py`,
`src/verisheet/planner_context.py`, `prompts/planner.txt` (v1.5), `prompts/planner_example.json`, `scripts/p3_api_check.py`,
`scripts/p3_eval_planner.py`, `tests/test_planner*.py`, `tests/test_qwen_client.py`.

## Setup

Add to `.env` (never commit it):

```bash
QWEN_API_BASE=<endpoint from the coordinator, ending in /v1>
QWEN_API_KEY=<key>
QWEN_MODEL=Qwen/Qwen3-8B         # Track B guide: same id for every group
QWEN_CA_CERT=config/qwen-ca.crt  # if the endpoint uses a private CA (git-ignored)
QWEN_TIMEOUT_SECONDS=45
```

Then run:

```bash
python scripts/p3_api_check.py      # endpoint smoke test: thinking-off, JSON mode, QwenClient
python scripts/p3_eval_planner.py   # 11 representative questions through Planner + Executor
python -m pytest tests/test_planner.py tests/test_planner_context.py tests/test_qwen_client.py
```

## Qwen client (`QwenClient.complete(system, user, temperature=0.0, max_tokens=None) -> LLMResponse`)

- **Shared:** the Planner and the Critic use the same model and endpoint.
- **One request at a time:** a process-wide lock allows one active request per key.
- **Token limits:** `max_tokens` is clamped to 1024 and defaults to 900, within the 8192 total
  token budget.
- **Thinking off:** Qwen3 thinking is disabled in the request, and any `<think>` blocks are
  removed from the reply.
- **Retries:** 429 and 503 responses are retried after 2, 4 and 8 seconds. Other 4xx errors fail
  immediately.
- **Error types:** configuration errors raise `QwenConfigError` (a `ValueError`); endpoint errors
  raise `QwenError` (a `RuntimeError`).
- **Key safety:** the key is a `SecretStr` in `Settings` and is never logged. It is redacted from
  error text. TLS verification is never disabled.

## Planner

`Planner.generate_plan(question, profile, context, previous_state=None, feedback=None)` and
`Planner.revise_plan(prior_plan, critic_feedback, state)` return a validated `AnalysisPlan`.

**Prompt.** It is built from four parts:
1. the versioned system text
2. an outline generated from `AnalysisPlan.model_json_schema()`
3. the executor operation catalogue (`OPERATIONS`, with input examples and result columns)
4. `prompts/planner_example.json` as an example. This is P3-owned and matches the executor's
   operations and result-column names. The shared contract example uses `"y_column": "profit"`,
   which the executor never outputs, so it is used only as a fallback.

The user message wraps the question, a compacted profile, the retrieved definitions, previously
verified state, the action budget and any feedback in tags. Everything inside the tags is treated
as data. The whole prompt is sized to fit the 8192-token limit (see `planner_context.py`).

**Validation.**
- `parse_plan_response` rejects prose, extra or missing fields, and truncated JSON.
- `check_plan` then enforces deterministic policies:
  - unknown or hidden sheets and columns
  - identifiers that are summed or averaged
  - profit with no profit or cost data (a clarification is required)
  - margin computed as an average
  - ambiguous dates
  - an undisclosed quarter assumption, or one that ignores `Data_Through`
  - mixed currencies without a currency filter or group-by
- **Executor compatibility:** the operation must be in the catalogue, required inputs must be
  present, input columns must exist in the step's sheet, and aggregations, frequency, `n` and
  filter operators must be valid. Chart x/y columns must be in the source step's result.
- **Repairs:** malformed JSON gets repair calls. Policy violations get up to two targeted
  repairs. The hard cap is 4 calls, after which `PlanValidationError` is raised.
- **Deterministic auto-fixes** (logged in `last_run["auto_fixes"]`; they do not use a call):
  - chart columns that differ from a real result column only in letter case
  - missing closing brackets in a reply that is otherwise complete JSON

  Prose and truncated replies are never repaired, and repaired output must still pass the strict
  schema.

**Code-owned fields.** `plan_id` (`plan-xxxx`, then `.r1`, `.r2` after a REPLAN) and `max_actions`
(capped by settings) are set by code, not by the model.

**State.** Only accepted or verified `AgentState` fields are sent; `history` never is. State from
a different `workbook_id` is ignored and recorded as `last_run["state_ignored"]`.

## Live evaluation and prompt failure analysis (VS-P3-02/03)

Endpoint: Track B group 046-065 (`…:1034/v1`, `Qwen/Qwen3-8B`, vLLM). Command:
`python scripts/p3_eval_planner.py --out results/p3_planner_eval.json`. There are 11 questions over
the two fixtures. A case passes only if the outcome is right *and* the plan runs in the real
executor.

| Prompt | Passed | What failed, and the fix for the next version |
|---|---|---|
| v1.2 | 3/11 | Chart columns in lower case (`sales` instead of `Sales`), copied from the contract example; fixed with a P3 example and a case auto-fix. `"key?"` optional markers copied into JSON keys; fixed with a plain outline plus a required-keys list. Unrequested charts; fixed with a chart-only-when-asked rule. |
| v1.3 | 8/11 | An extra `}` after `inputs`; fixed by putting `inputs` last. A metric substitution, `sum(Quantity * UnitPrice)` run as a column sum; fixed with a derived-formula rule. |
| v1.4 | 9/11 | A plan that only previews data (`sample_rows`); fixed with a must-contain-`run_analysis` rule. A missing `]` that closes `steps`; fixed with bracket balancing. |
| v1.5 | **10/11** | The only remaining failure is the join, which is an executor bug (below). The Planner's plan is correct. |

Server-side JSON modes were tried and rejected. `response_format: json_object` forced Qwen's
slips into valid but wrong JSON. `json_schema` (strict) was slow (10–25 s) and ran out of output
tokens.

## Notes for integrators

- **P5 (orchestrator):**
  - Reuse the same `Planner` instance for `generate_plan` and `revise_plan`; it remembers the
    question and profile for each `plan_id`. Otherwise pass `question` and `workbook_profile` inside
    `critic_feedback`.
  - Catch `PlanValidationError`. It is a `ValueError` and means "could not produce a valid plan".
- **P6 (traces):** `planner.last_run` holds `prompt_version`, `model`, `calls`, `input_tokens`,
  `output_tokens`, `latency_ms`, `json_repairs`, `policy_repairs`, `unresolved_problems` and
  `state_ignored`. Put it in `TraceEvent.metadata`.
- **P4 (executor):**
  - **Bug:** `safe_join` checks cardinality backwards. For `many_to_one` it rejects duplicate
    *left* keys, but the left side is the "many" side; the *right* keys must be unique (pandas
    `validate="many_to_one"` agrees). Orders→Product_Costs fails because of this.
  - No operation computes derived columns (for example Quantity × UnitPrice), so Online Retail
    sales value cannot be computed yet. The Planner asks for clarification instead.
  - `OPERATIONS` mirrors the executor's registry; a test fails if they drift apart.
  - Each `run_analysis` step reads raw sheets, and step outputs are not chained. Questions that
    need "join, then aggregate" or "aggregate, then join" (for example, returns aggregated by
    Order_ID before joining to orders) cannot currently be executed in one plan.
- **P2:** `prompts/` is an approved retrieval source. Rebuild the index after prompt changes
  (`python scripts/build_index.py`).
