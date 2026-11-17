| tenants | chunks | mode | ef_search | recall_vs_exact@10 | empty_results | p95_ms | index_MB | relations | build_s | pgvector |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 5724 | dedicated | 40 | 1.0 | 0 | 1.55 | 53.3 | 100 | 2.8 | 0.6.0 |
| 100 | 5724 | hybrid | 40 | 0.317 | 1 | 1.47 | 47.9 | 4 | 0.1 | 0.6.0 |
| 100 | 5724 | shared | 40 | 0.272 | 2 | 1.78 | 47.7 | 1 | 4.1 | 0.6.0 |
| 100 | 5724 | shared | 100 | 0.372 | 0 | 2.01 | 47.7 | 1 | 4.1 | 0.6.0 |
| 100 | 5724 | shared | 400 | 0.488 | 0 | 4.81 | 47.7 | 1 | 4.1 | 0.6.0 |
| 100 | 5724 | shared | 1000 | 0.657 | 0 | 7.78 | 47.7 | 1 | 4.1 | 0.6.0 |
| 1000 | 21924 | dedicated | 40 | 1.0 | 0 | 1.44 | 237.7 | 1000 | 11.6 | 0.6.0 |
| 1000 | 21924 | hybrid | 40 | 0.112 | 20 | 2.21 | 182.6 | 4 | 0.0 | 0.6.0 |
| 1000 | 21924 | shared | 40 | 0.106 | 19 | 3.1 | 182.4 | 1 | 29.6 | 0.6.0 |
| 1000 | 21924 | shared | 100 | 0.145 | 3 | 3.27 | 182.4 | 1 | 29.6 | 0.6.0 |
| 1000 | 21924 | shared | 400 | 0.223 | 0 | 7.87 | 182.4 | 1 | 29.6 | 0.6.0 |
| 1000 | 21924 | shared | 1000 | 0.27 | 0 | 18.77 | 182.4 | 1 | 29.6 | 0.6.0 |

Production query plan check:

- 100 tenants: uses HNSW = **False** (planner drives from documents(tenant, ACL) then exact-sorts the tenant's chunks)
- 1000 tenants: uses HNSW = **False** (planner drives from documents(tenant, ACL) then exact-sorts the tenant's chunks)
