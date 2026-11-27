# VeriSheet: Agentic Spreadsheet Analysis with Independent Verification

**Technical report, IIT KGP Capstone (Track B)**

## Abstract

VeriSheet answers analytical questions about unfamiliar Excel/CSV workbooks and returns an answer
only with evidence. A Qwen3-8B planner turns each question into a typed, dependency-ordered plan.
An executor runs that plan as allowlisted pandas operations: no model-written code is ever
executed. A deterministic verifier then recomputes every number and every plotted value directly
from the source workbook. A hybrid critic routes failures to REPAIR, REPLAN or CLARIFY. Only answers
whose numbers were independently recomputed are labelled VERIFIED.

The system is evaluated as five ablation variants (E1 direct prompting to E5 full hybrid) and a
20-case fault-injection suite. Every run produces a structured trace with model and prompt
versions, tokens, latency, scripts and verification evidence.

This report covers architecture, prompt strategy, retrieval, agent decisions, evaluation, failure
taxonomy, security and limitations. Every number in it comes from a recorded run in
`results/experiments/`; none were typed in.

## 1. Problem and goals

Analysts receive workbooks they did not build: several sheets, inconsistent dates, numbers stored
as text, hidden sheets, formulas and personal data. A language model asked directly will often
give a fluent but wrong number: summing an identifier, averaging row-level margins, picking the
wrong quarter, or inventing a profit figure for a dataset that has no cost data.

VeriSheet's goals:

1. **Correctness over fluency.** A number is shown as VERIFIED only when deterministic code has
   independently recomputed it from the workbook.
2. **Answerability.** When the data cannot answer the question (missing metric, ambiguous dates,
   different currencies), the system asks a clarification question instead of guessing.
3. **Observability.** Every decision is traceable: the plan, each tool call, the script hash, the
   verification findings, the critic decision, tokens and latency.
4. **Safety.** Workbook content, sheet names and retrieved text are untrusted data. Macros and
   external links are rejected, and personal data and secrets never reach prompts or logs.

## 2. Architecture

```
upload ─► P2 profile_workbook ─► P2 retrieve_context ─► P3 Planner (Qwen3-8B, JSON plan)
              │ WorkbookProfile        │ RetrievedContext        │ AnalysisPlan
              ▼                        ▼                         ▼
        quality warnings        sourced definitions     P3 check_plan (deterministic policies)
                                                                 │
     P5 Orchestrator (bounded: actions, repairs, replans) ◄──────┘
              │
              ├─► P4 Executor: typed pandas operations only, artifacts + script hash
              ├─► P5 Verifier: independent recomputation of numbers and chart arrays
              ├─► P5 Critic: deterministic findings + Qwen semantic review → ACCEPT/REPAIR/REPLAN/CLARIFY
              ├─► P5 StateStore: only VERIFIED facts are reused in follow-ups
              └─► P6 TraceLogger (JSONL, redacted) ─► P6 trace dashboard
                                                     ─► P6 Streamlit UI (AnswerResponse)
```

All modules exchange the shared Pydantic contracts in `src/verisheet/schemas.py`:
`WorkbookProfile`, `RetrievedContext`, `AnalysisPlan`, `ExecutionResult`, `VerificationReport`,
`CriticDecision`, `AnswerResponse` and `TraceEvent`. Contracts reject unknown fields. They also
enforce invariants: a VERIFIED `AnswerResponse` must carry a passing `VerificationReport`, and a
clarification plan must carry a question. The UI calls a single entry point, `run_agent()`, and
renders only backend artifacts.

**Design principle: the model plans and never computes.** Qwen selects the sheets, columns,
operations, filters and chart. Pandas computes, and an independent code path recomputes. This
moves the trust boundary away from the probabilistic component.

## 3. Workbook profiling (P2)

`profile_workbook` validates the upload before reading it:
- extension, size limit and emptiness
- XLSX zip structure, macro parts and zip-bomb ratios
- external links

It then profiles every sheet: row counts, inferred dtypes, candidate roles (identifier, measure,
date, category), null rates, duplicate rows, formula cells (cached values only; formulas are never
recalculated), hidden sheets and candidate date ranges. Date parsing chooses day-first or
month-first from evidence (a part greater than 12) and flags ambiguous or inconsistent slash dates.

Twenty-four coded quality warnings reach the planner, including `HIDDEN_SHEET`,
`FORMULA_NO_CACHED_VALUE`, `NUMERIC_AS_TEXT`, `DATE_COERCION_LOSS`, `AMBIGUOUS_DATE_FORMAT`,
`NEGATIVE_VALUES`, `POSSIBLE_PII` and `SUSPICIOUS_INSTRUCTION_TEXT`. Warning evidence holds counts
only, never cell values. Example values are sanitised: emails and phone numbers are masked,
instruction-like text is redacted, and formula-like text gets an apostrophe prefix.

## 4. Retrieval (P2)

Approved documentation (the data dictionary, tool documentation and prompt rules) is split by
heading and bullet into chunks of up to 600 characters. Each chunk gets a stable ID and a
`file#heading` provenance path. Gold answers, raw or uploaded data, and secret-bearing paragraphs
are excluded.

Embeddings come from the course's shared BGE-small-en-v1.5 API (384 dimensions; at most 16 texts
per request; retry on capacity errors), with the BGE query instruction prefix. A local
sentence-transformers copy is an optional offline fallback; spot checks gave cosine 1.0 against
the API. Search fuses dense cosine ranking with BM25 using reciprocal rank fusion. Definitions
that mention the uploaded workbook's columns move up at most two ranks.

On 20 labelled queries, hybrid retrieval beat dense-only and BM25-only at every chunk size
(`scripts/eval_retrieval.py`):

| 600-char chunks | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| Hybrid (default) | 0.85 | 0.95 | 0.900 |
| Dense only | 0.80 | 0.95 | 0.854 |
| BM25 only | 0.80 | 0.90 | 0.850 |

The index manifest stores a `corpus_version` hash, which is checked on load. A missing index
degrades to a `RETRIEVAL_UNAVAILABLE` warning instead of crashing the agent.

## 5. Prompt strategy and the Planner (P3)

The planner prompt (`prompts/planner.txt`, v1.5) is assembled from four parts:
1. versioned rules
2. a schema outline generated from `AnalysisPlan.model_json_schema()`
3. an operation catalogue generated from the executor's real operation registry
4. one executor-consistent example plan

User input is wrapped in tags (`<question>`, `<profile>`, `<context>`, `<previous>`, `<feedback>`)
and declared to be data. The profile is compacted to fit the 8,192-token limit. Thinking is turned
off and temperature is 0.

**Deterministic plan policies.** These are enforced in code after parsing and cost no model
calls:
- unknown or hidden sheets and columns
- identifiers summed or averaged
- profit requested where there is no profit or cost data (forces CLARIFY)
- margin as an average of row-level margins
- ambiguous dates
- an undisclosed or wrong last-completed-quarter assumption (`Data_Through` takes precedence)
- mixed currencies
- formulas that multiply columns, which no operation can compute
- plans that compute nothing
- operation names, required inputs, aggregations and filter operators the executor does not support
- chart columns that do not exist in the source step's result

A violation triggers a targeted repair call. The hard cap is four calls per plan.

**Live prompt engineering.** Running the planner against live Qwen exposed failure modes that unit
tests could not:
- copying `"key?"` optional markers from the schema outline into JSON keys
- lower-case chart columns copied from an inconsistent example
- an extra `}` after `inputs`
- missing `]`
- `sample_rows` used as an answer

Each was fixed at its root: the outline format, a P3 example plan, `inputs` moved to the last key,
logged bracket balancing that is never applied to prose or truncated output, and new rules. The
representative-question pass rate rose from 3/11 (v1.2) to 10/11 (v1.5). The remaining failure was
an executor join bug (§11).

**Follow-ups.** Only accepted and verified state is passed to the model, never the unverified
history. State from a different workbook fingerprint is ignored. A follow-up that names no new
period keeps the verified period as a plan filter. A live test showed the model dropping it, and
the system then returned all-time totals labelled VERIFIED. The period is now enforced in code.

## 6. Execution, verification and the agent loop (P4, P5)

The executor runs typed operations only: `group_aggregate`, `time_aggregate`,
`quarter_comparison`, `grouped_ratio`, `top_n`, `filter_rows` and `safe_join`. Each step records an
operation script with its SHA-256 hash, a result CSV and, for charts, the plotted arrays.

The verifier reloads the original workbook and recomputes every `run_analysis` step with its own
pandas code. It does not import executor functions, so a corrupted execution cannot verify itself.
It then compares the results against the result table and every plotted value, within a tolerance.

The critic is hybrid:
- **Blocking deterministic findings** route by category. Wrong aggregation and chart-value
  mismatch go to REPAIR (re-run only the affected steps); a wrong approach, metric or period goes
  to REPLAN; missing inputs go to CLARIFY.
- **Qwen semantic review** checks whether the result answers the question and whether the chart
  type is appropriate. It can veto an answer, but it can never provide numeric proof.

The orchestrator enforces action, repair and replan budgets, persists only VERIFIED facts, and
always returns an `AnswerResponse`: VERIFIED, NOT_VERIFIED, CLARIFICATION_REQUIRED or FAILED.

## 7. Observability, UI and deployment (P6)

**Traces.** `TraceLogger` appends one redacted `TraceEvent` per line to
`results/traces/<trace_id>.jsonl`. Redaction covers the configured keys, `sk-` and bearer tokens,
`api_key=` patterns, emails and phone numbers. Summaries and nested metadata are bounded.

**Dashboard.** It loads a trace by ID and shows:
- a timestamped timeline
- tokens, latency, models and prompt versions
- the decision path
- verification findings and failure categories
- links to the actual scripts (with hashes), tables and charts

It warns when a trace's model does not match the configured production model.

**UI.** The Streamlit app quarantines uploads under content-hash names, shows the profile and
warnings, and keeps one session per workbook for follow-ups. It renders four distinct statuses
and shows tables and charts only from artifact files inside `artifacts/`. Every answer has its
trace ID and an "Open trace" link.

**Deployment.**
- The Docker image (python:3.12-slim) runs as a non-root user and contains no credentials or
  certificates: `.env` is passed at run time and certificates are mounted read-only.
- The retrieval index is built on first start.
- Local embedding dependencies (PyTorch) are optional, which keeps the image small.

## 8. Evaluation

All numbers below come from `results/experiments/smoke-20261127T072309/`:
`score_summary.json`, `ablation.json` and `injection/injection_results.jsonl`. They were produced
by live Qwen/Qwen3-8B runs (Track B endpoint) with planner prompt v1.5, critic v1.1, direct v1.0,
temperature 0, at code base `b045f7f` plus this branch's uncommitted changes. Test-set SHA-256:
`9b151cfc05a1475e…`; gold SHA-256: `0f87ab6150eeb1e4…`.

**Scope.** The frozen 80-question test set (Person 1) does not exist yet. These results use the
**12-question P6 smoke set** over the two repository fixtures, with gold values computed by pandas.
It contains 10 answerable questions and 2 that must be refused. Re-run
`experiments/run.py --system all --split test` once the frozen set is delivered; these numbers do
not replace it.

### 8.1 Five-variant ablation (identical 12 questions)

| Variant | Task success | Numeric (computed) | Numeric (delivered) | Verified-correct coverage | False verification | Refusal acc. | LLM calls | Median latency | Median tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 direct | 0.50 | 0.50 | 0.50 | 0.00 | — | 0.50 | 1.00 | 1.3 s | 1,242 |
| E2 planner+executor | 1.00 | 1.00 | 0.70 | 0.00 | — | 1.00 | 1.00 | 6.0 s | 3,578 |
| E3 LLM critic | 1.00 | 1.00 | 0.70 | 0.00 | — | 1.00 | 1.83 | 8.3 s | 4,714 |
| E4 deterministic critic | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 6.1 s | 3,578 |
| E5 full hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.83 | 8.3 s | 4,714 |

For E2–E5: execution success 1.00, plot-data accuracy 1.00, chart appropriateness 1.00, 3.5 agent
actions per question and 0 crashes. Totals: E4 used 46,014 tokens and E5 57,059 (p95 latency 6.3 s
vs 8.7 s). E1 used 14,135 tokens.

**Reading the ablation.**
- **Planning and execution matter most.** Direct prompting (E1) answered 5 of 10 answerable
  questions. It described the method instead of giving numbers (s03 to s05), got the
  last-two-quarters comparison wrong (s06), returned invalid JSON once (s10), and asked for
  clarification on an answerable question (s11). Every planned-and-executed variant computed all
  10 correctly.
- **Verification turns computation into a trustworthy answer.** E2 and E3 computed the right
  numbers but cannot verify them. By design they withhold unverified figures, so only 70% of
  correct results reached the user, through charts. E4 and E5 verified 10 of 10 with no false
  verifications.
- **The LLM-only critic (E3) cannot produce VERIFIED.** That is by design: a model's opinion is
  never accepted as numeric proof. It adds 0.83 calls and about 2.3 s per question over E2.
- **E4 and E5 are identical on this set.** The semantic review cost 31% more tokens and 37% more
  latency for no measured gain, because the smoke questions are too easy to separate them. §8.2
  shows a case where the semantic layer *should* have helped and did not.

### 8.2 Fault-injection suite (20 cases, full system E5)

| Category | Cases | Passed | Outcome |
|---|---:|---:|---|
| Wrong aggregation (mean / max / count) | 3 | 3 | detected → REPAIR → re-run → VERIFIED correct |
| Duplicated join: rows doubled at runtime / duplicate keys in data | 2 | 2 | REPAIR → VERIFIED correct / REPLAN×2 → NOT_VERIFIED |
| Wrong period: rotated labels ×2 / plan shifted to the wrong quarter | 3 | 3 | REPAIR, or REPLAN → VERIFIED correct |
| Wrong metric (Sales → Profit swapped in the plan) | 1 | **0** | **ACCEPTED: false verification** |
| Invalid chart arrays (+999, dropped category, extra category) | 4 | 4 | REPAIR (re-render) → VERIFIED; one ended NOT_VERIFIED |
| Missing metric / sheet / column | 3 | 3 | CLARIFY |
| Ambiguous slash dates | 1 | 1 | CLARIFY |
| Prompt injection (cell text, separate sheet) | 2 | 2 | correct total; injected number never shown |
| Unsafe code request | 1 | 1 | no plan, nothing executed (FAILED safely) |
| **Total** | **20** | **19** | pass rate 0.95 |

All 12 runtime pipeline faults fired (none untriggered). **Error-detection recall was 11/12 =
0.917**, and 10 cases were repaired to a VERIFIED correct answer.

The single failure, c09, is informative. The injected plan summed Profit for "What were total
sales?". Verification checks fidelity to the plan, so it recomputed the plan faithfully. The Qwen
semantic review did not notice that the metric did not match the question. A deterministic
question-to-metric consistency check in the critic is the clear next step (Person 5).

### 8.3 Planner and retrieval evaluations

- **Planner, live** (11 representative questions through planner and executor): pass rate rose
  from 3/11 (prompt v1.2) to 10/11 (v1.5). The remaining case failed on an executor join bug (§11).
- **Retrieval** (20 labelled queries): hybrid MRR 0.900 vs 0.854 dense-only (§4).


## 9. Failure taxonomy

| Category | Where detected | Route | Example |
|---|---|---|---|
| Invalid / truncated JSON | P3 parser | JSON repair call / bracket balancing | missing `]` in `steps` |
| Unsupported operation or input | P3 `check_plan` | policy repair | `group_sum`, missing `metrics` |
| Unknown sheet / column | P3, executor, verifier | policy repair / REPLAN | invented `Sales` in Online Retail |
| Unanswerable metric | P3 policy | CLARIFY | Online Retail profit, `Quantity × UnitPrice` |
| Ambiguous request / dates | P3 policy, profiler warning | CLARIFY | "How did we do?", `01/02/2025` |
| Wrong aggregation | verifier | REPAIR | sum swapped to mean |
| Wrong period / dropped context | verifier, critic, P3 follow-up rule | REPLAN | follow-up lost the quarter |
| Row multiplication (join) | executor cardinality, verifier | REPLAN / not verified | duplicated join keys |
| Chart-data mismatch | verifier (plotted arrays) | REPAIR (re-render only) | values +999, dropped category |
| Inappropriate chart | critic semantic review | REPAIR | ranking drawn as a line |
| Prompt injection in data | profiler redaction, prompt framing | answer ignores it | "report 999999" in a cell |
| Infrastructure | client retry / budget | FAILED safely | 429/503 from the shared endpoint |

## 10. Security

- **Uploads:** checked for size, macros, zip bombs and external links. Files are stored under
  content-hash names, so a user's file name never becomes a path.
- **No code execution:** no model-written code ever runs. The executor maps plan operations to
  fixed pandas functions, and a "run this Python" request is answered with a clarification.
- **Untrusted text:** workbook text, sheet names and retrieved snippets are framed as data. Instruction-like cell text is
  redacted in examples and flagged.
- **Secrets:** keys live only in `.env`, as `SecretStr` (masked in reprs). They are redacted from
  traces and error messages, git-ignored, and excluded from the Docker build context. TLS
  verification uses the course CA certificates and is never disabled.
- **Gold isolation:** gold answers are never in prompts or the retrieval index. The scorer is the
  only reader.

## 11. Limitations and future work

- **Frozen test set not yet available.** Person 1's datasets, frozen 80-question test set, gold
  answers and corruption workbooks have not been delivered. The variant comparison in §8 therefore
  uses a 12-question smoke set built from the repository fixtures. The pipeline
  (`experiments/run.py --split test`) is ready for the real set.
- **Executor gaps (P4):**
  - `safe_join` checks cardinality backwards, so a correct Orders→Product_Costs `many_to_one`
    join is rejected.
  - Steps cannot be chained, so "aggregate returns by Order_ID, then join to orders" cannot run.
  - There are no derived columns, such as `Quantity × UnitPrice`.

  The planner asks for clarification instead of guessing, but these limit coverage.
- **Verification is faithful to the plan, not to the question.** A plan that computes the wrong
  thing correctly passes numeric verification. The semantic critic and deterministic plan policies
  (period carry-over, metric checks) reduce this but cannot eliminate it.
- **Small model.** Qwen3-8B has an 8K context and makes structural JSON slips. The deterministic
  layer compensates, at the cost of extra calls on some questions.
- **Shared infrastructure.** The group key allows one active request and 60 requests per minute.
  Experiments run sequentially.
