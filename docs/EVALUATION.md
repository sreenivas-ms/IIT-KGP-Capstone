# Evaluation: variants, metrics and robustness suite (Person 6)

Code is in `src/verisheet/experiments/`. The CLIs in `experiments/` match the project brief.

## Data

| Set | File | Owner | Status |
|---|---|---|---|
| Frozen dev (20) and test (80) questions | `data/derived/questions.jsonl` (`question_task` schema) | Person 1 | **not yet delivered** |
| Gold answers | `data/gold/gold_answers.jsonl` (`gold_answer` schema) | Person 1 | **not yet delivered** |
| P6 smoke set (12 questions) | `experiments/smoke/{questions,gold}.jsonl` | Person 6 | built by `python experiments/build_smoke_set.py` |
| Injection suite (20 cases) | `experiments/injection_cases.json` | Person 6 | ready |

The smoke set uses the two repository fixtures. Its gold values are computed with pandas
(`verisheet.experiments.gold`), never typed in. It checks the whole pipeline and is **not** a
substitute for the frozen 80-question test set: report its numbers as smoke results.

Gold is read only by the scorer and the injection judge, never by a model. Every question file is
validated against `contracts/schemas/question_task.schema.json`, and every gold file against
`gold_answer.schema.json`.

## Variants (VS-P6-03)

| Label | `--system` | What runs | Can be VERIFIED? |
|---|---|---|---|
| E1 | `direct` | Qwen answers from the profile and up to 20 sanitised sample rows per sheet (`prompts/direct.txt`) | no |
| E2 | `planner_executor` | Planner then Executor; no critic | no |
| E3 | `llm_critic` | E2 plus a Qwen-only critic | yes |
| E4 | `deterministic_critic` | E2 plus independent deterministic verification; no Qwen review | yes |
| E5 | `full` | E4 plus Qwen semantic review (hybrid) | yes |

Every variant uses the same model (`QWEN_MODEL`), temperature 0, tool library, action and repair
budgets (`MAX_AGENT_ACTIONS`, `MAX_REPAIRS`) and question file.

Each run writes `results/experiments/<run_id>/results.jsonl` (one record per question) and
`run_metadata.json`. The metadata holds:
- the test-set SHA-256 and the question IDs
- the model and prompt versions
- temperature and budgets
- the git commit, Python, pandas and numpy versions

Each question's trace also gets an `experiment_config` event with the variant configuration.

## Metrics (VS-P6-04)

A = answerable questions (`must_refuse=false`); R = refusal questions; V = answers with status VERIFIED.

| Metric | Definition |
|---|---|
| numeric_accuracy | Share of A where every gold number is matched by a number the system **computed**: executed result tables, plotted data or answer text. Match tolerance: \|diff\| ≤ max(0.006, 1e-4·\|gold\|). "20%" also matches 0.2. |
| delivered_numeric_accuracy | The same match, using only what the user is **shown**: the answer text, response table and chart. E2/E3 compute numbers but withhold them when unverified, so this metric separates *computing* correctly from *delivering* correctly. |
| task_success | A: numeric match and status VERIFIED or NOT_VERIFIED. R: status CLARIFICATION_REQUIRED. |
| execution_success | Share of A that produced an answer (VERIFIED/NOT_VERIFIED, no crash). |
| plot_data_accuracy | Share of charted A answers whose plotted values contain every gold number. |
| chart_appropriateness | Share of charted answers whose chart type is in `accepted_charts`. |
| refusal_accuracy | Share of R answered with CLARIFICATION_REQUIRED. |
| false_verification_rate | Share of V that are not task successes, i.e. wrong but labelled VERIFIED. |
| verified_coverage | Share of A that ended VERIFIED. `verified_correct_coverage` additionally requires correctness. |
| repair_success | Share of questions with at least one repair or replan that ended as task successes. |
| tool efficiency | Mean agent actions and mean LLM calls per question. |
| latency / tokens | Per-question median and p95 latency (ms) and tokens; total tokens. |
| error_detection_recall | Injection suite: share of *triggered* pipeline faults that did not end VERIFIED-and-wrong. |

Report false-verification rate and verified coverage together. A system can reach 0% false
verification by never verifying anything, and coverage shows whether that is happening.

## Robustness / error-injection suite (20 cases)

`python experiments/inject_errors.py --suite 20_cases` runs every case through E5. There are three
kinds of case:
- **Pipeline faults**, injected at runtime into the Executor or Planner:
  - aggregation swapped to mean, max or count
  - rows doubled (simulating a duplicated join)
  - period labels rotated
  - plan period shifted to the wrong quarter
  - metric swapped from Sales to Profit
  - chart values offset by +999, a category dropped, or a fake category added

  The independent verifier is never patched. Each injector counts its activations, and a fault
  that never fired is reported as untriggered. It never counts as "detected".
- **Data faults**, using corrupted workbooks built for each run:
  - duplicated join keys in Product_Costs
  - prompt-injection text in a cell
  - prompt-injection text on a separate sheet
  - ambiguous slash dates
- **Question faults:** a missing metric (Online Retail profit), a missing sheet, a missing column,
  and a request to run arbitrary Python.

Expectations:
- `detect`: the run must not end VERIFIED with a wrong value.
- `clarify`: CLARIFICATION_REQUIRED or a safe FAILED.
- `not_verified`
- `ignore_injection`: the injected number never appears in the answer, and no wrong value is verified.

## Reproduce

```bash
python experiments/build_smoke_set.py
R=results/experiments/smoke-$(date +%Y%m%dT%H%M%S)
python experiments/run.py --system all --split dev --questions experiments/smoke/questions.jsonl --out $R
python experiments/inject_errors.py --suite 20_cases --out $R/injection
python experiments/score.py --results $R --gold experiments/smoke/gold.jsonl --out $R/score_summary.json
python experiments/ablate.py --runs $R --gold experiments/smoke/gold.jsonl --out $R/ablation.json
# when Person 1 delivers the frozen set:
python experiments/run.py --system all --split test --out results/experiments/test-<date>
python experiments/score.py --results results/experiments/test-<date> --out results/experiments/test-<date>/score_summary.json
```
