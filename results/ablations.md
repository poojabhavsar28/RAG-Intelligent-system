### Retrieval components

| config | recall@1 | recall@5 | MRR | nDCG@10 | p95_ms | leaks |
|---|---|---|---|---|---|---|
| dense only | 0.7351 | 0.9735 | 0.8273 | 0.8654 | 0.2118 | 0 |
| BM25 only | 0.9007 | 1.0 | 0.9454 | 0.9594 | 0.2452 | 0 |
| hybrid (RRF) | 0.8609 | 0.9801 | 0.9164 | 0.9343 | 0.5128 | 0 |
| hybrid + rerank | 0.9205 | 0.9934 | 0.9547 | 0.9662 | 1.1017 | 0 |

### Self-correcting loop

| config | answer_accuracy | wrongly_abstained | hallucination_unanswerable | answered_forbidden_q(near-miss) | leaked_forbidden_doc | p50_ms | p95_ms | n_ans/unans/forb |
|---|---|---|---|---|---|---|---|---|
| no self-check | 0.921 | 0.0 | 0.99 | 0.975 | 0 | 1.16 | 1.41 | 151/99/80 |
| judge + abstain (1 attempt) | 0.669 | 0.331 | 0.01 | 0.0 | 0 | 1.23 | 1.42 | 151/99/80 |
| judge + repair + retry (2 attempts) | 1.0 | 0.0 | 0.01 | 0.013 | 0 | 1.26 | 2.52 | 151/99/80 |
