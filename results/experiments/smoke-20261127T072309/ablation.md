| Variant | task_success | numeric_accuracy | delivered_numeric_accuracy | verified_correct_coverage | false_verification_rate | refusal_accuracy | mean_llm_calls | latency_ms_median | tokens_median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 direct | 0.5 | 0.5 | 0.5 | 0.0 | — | 0.5 | 1 | 1289.0 | 1242.0 |
| E2 planner_executor | 1.0 | 1.0 | 0.7 | 0.0 | — | 1.0 | 1 | 5982.4 | 3578.0 |
| E3 llm_critic | 1.0 | 1.0 | 0.7 | 0.0 | — | 1.0 | 1.83 | 8278.7 | 4714.0 |
| E4 deterministic_critic | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1 | 6081.2 | 3578.0 |
| E5 full | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.83 | 8292.9 | 4714.0 |
