# Phase 2 retrieval comparison (frozen labels, real pretrained models)

Units: 48 frozen pre-segmented passages (not production 800-word chunks). k=4.
Metrics (custom IR metrics, **not RAGAS**): Precision@4 = |rel∩top4|/4; Recall@4 = |rel∩top4|/|rel|;
MRR@10 = mean 1/rank of first relevant in top-10; latency = per-query CPU wall time incl. query encoding (ONNX, 4 threads).
Selection fixed on dev before held-out (commit `b9e7bb5`): max MRR@10, then Recall@4, then p50 latency.

## dev (n=24)

| System | P@4 | R@4 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| dense(bge-small) | 0.302 | 1.000 | 0.910 | 5.26 | 6.44 |
| bm25(k1=1.5,b=0.75) | 0.271 | 0.896 | 0.875 | 0.08 | 0.13 |
| hybrid-rrf(k=10; dense(bge-small) + bm25(k1=1.5,b=0.75)) **(selected)** | 0.302 | 1.000 | 0.920 | 5.87 | 7.47 |
| rerank(ms-marco-MiniLM-L6, top-10 of dense(bge-small)) | 0.302 | 1.000 | 0.903 | 106.23 | 144.42 |

## heldout (n=24)

| System | P@4 | R@4 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| dense(bge-small) | 0.281 | 0.938 | 0.865 | 6.30 | 6.88 |
| bm25(k1=1.5,b=0.75) | 0.250 | 0.833 | 0.760 | 0.09 | 0.13 |
| hybrid-rrf(k=10; dense(bge-small) + bm25(k1=1.5,b=0.75)) **(selected)** | 0.281 | 0.938 | 0.792 | 5.94 | 6.99 |
| rerank(ms-marco-MiniLM-L6, top-10 of dense(bge-small)) | 0.281 | 0.938 | 0.868 | 101.46 | 116.85 |

Full dev grid and per-query results: `dev.json`, `heldout.json`.
Outcome: the dev-selected hybrid did not generalise (held-out MRR 0.792 vs dense 0.865); it was not deployed.
