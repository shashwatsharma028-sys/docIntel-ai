# Phase 2 Retrieval Evaluation Set (prepared in Phase 1, NOT yet used)

Frozen synthetic corpus and labelled questions for the later evaluation of dense FAISS,
BM25, hybrid retrieval with reciprocal rank fusion, cross-encoder reranking, context
optimisation and RAGAS. **No retrieval system has been evaluated on this set.**

## Contents

| Path | What |
|---|---|
| `corpus/passages.jsonl` | 48 passages with stable IDs (`TRV24-03`, `MSA-01`, `LABJ-02`, …), doc id, source file name, domain, order |
| `corpus/*.txt` | 9 documents assembled from those passages (for PDF generation / ingestion) |
| `questions/dev.jsonl` | 24 development questions (may be used for tuning) |
| `questions/heldout.jsonl` | 24 held-out questions (**never** for tuning or method selection) |
| `LABELS.lock.json` | SHA-256 of every file above, counts, freeze statement |
| `build_corpus.py`, `build_questions.py` | Source of truth; regeneration is byte-identical (verified) |

Each question has `relevant_passage_ids`, `hard_negative_passage_ids`, `query_types` and a
`reference_answer` (ground truth for answer-level metrics). `tests/test_eval_labels.py`
fails if any frozen file changes, an ID is dangling, relevant and negative sets overlap,
a multi-passage question has fewer than two relevant passages, a distractor question names
no hard negative, or a numerical answer is not grounded in its relevant passages.

## Design

- 4 domains (enterprise, legal, medical, study), 9 documents, all fictitious.
- **Hard distractors by construction**: 2022 vs 2024 travel policy (different limits and
  approvers), Contoso vs Fabrikam contracts (different notice periods, caps, SLAs, breach
  windows), January vs July lab reports for the same patient (different values),
  respiration vs photosynthesis (parallel "cycle"/"equation" vocabulary).
- Query types across 48 questions (multi-label): numerical 27, distractor 20, factual 13,
  paraphrase 9, multi-passage 8.
- Labels were written from passage text only, before any system output existed.

## Local model investigation (no packages installed, no models downloaded)

Environment: 4 vCPU Xeon @ 2.3 GHz (AVX-512), 16 GB RAM, ~59 GB free disk, no GPU.

| Need | Candidate | Size / footprint | Notes |
|---|---|---|---|
| Local dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~22.7M params, ~90 MB weights, 384-d * | Fast on CPU; good baseline |
| | `BAAI/bge-small-en-v1.5` | ~33M params, ~130 MB, 384-d * | Usually stronger retrieval than MiniLM at similar cost |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22.7M params, ~90 MB * | Practical for top-20 reranking on CPU |
| | `BAAI/bge-reranker-base` | ~278M params, ~1.1 GB * | Higher quality, ~10x slower on CPU |
| Runtime (PyTorch) | `sentence-transformers` 6.3.0 + `torch` 2.15.0 | default PyPI torch wheel is the CUDA build: 539 MB + CUDA toolkit deps (several GB) | Needs the CPU-only torch index to stay reasonable |
| Runtime (ONNX) | `onnxruntime` 1.32.0 (+ optional `fastembed` 0.9.1) | 27.8 MB wheel | Much lighter; models must be available as ONNX |
| BM25 | `bm25s` 0.3.16 or `rank-bm25` 0.2.2 | < 0.2 MB, numpy only | No model download |
| RAGAS | `ragas` 0.4.4 | 0.5 MB + `datasets`, `tiktoken`, `pydantic`, … | Most RAGAS metrics need an LLM judge and embeddings -> conflicts with the no-live-API rule unless a local judge or an approved quota is used |

\* From the models' published cards; **not verified here**: `huggingface.co` is not reachable
from this environment (connection refused by the network policy), so model weights cannot be
downloaded without a network-policy change. PyPI metadata above was queried live.

**Blockers for Phase 2 (need approval):** (1) adding evaluation-only dependencies
(gate G4); (2) allowing `huggingface.co` or providing model files (network/G5);
(3) a decision on the RAGAS judge (local model vs approved provider quota; InsightPilot's
reserved quota must not be used).
