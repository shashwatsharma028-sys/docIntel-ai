#!/usr/bin/env python3
"""Fair caching baseline: B document-level incremental vs C chunk-level content-addressed reuse.

  B  embed every chunk of a document whose content hash is new; skip unchanged re-uploads and
     renamed duplicates (a competent document-level incremental indexer)
  C  embed only chunk texts not already in the content-addressed chunk cache (the shipped
     backend.embedding_cache keying: sha256 of chunk text)
  (A, full-corpus re-embedding, is the Phase 1 baseline and is not repeated here.)

Documents: real public paper text from QASPER (the 11 frozen dev papers; CC-BY-4.0).
Each edit is applied once to the original version of every paper, with the cache primed by the
original; workload composition and edit sizes are frozen in WORKLOAD below (written before the run).
Chunkers are identical for B and C: production words (800/100), fixed token windows (T=256, overlap 32),
structure-aware (T=256). Embedding COUNTS are exact; wall time is REAL local bge-small ONNX encoding
of exactly the chunks each method must embed (not provider calls; no billing claim).
"""
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qasper_eval as rq  # noqa: E402
from backend import ingest, structure_chunker  # noqa: E402

WORKLOAD = {
    "seed": 99,
    "sizes": {"small": "1 sentence (~25 words)", "medium": "1 paragraph", "substantial": "25% of paragraphs"},
    "edits": ["unique_new_document", "unchanged_reupload", "renamed_duplicate",
              "insert_beginning", "insert_middle", "append_end", "replace_paragraphs", "delete_paragraphs"],
    "insert_source": "paragraphs/sentences taken from a different QASPER dev paper (seeded)",
    "papers": "the 11 frozen QASPER dev papers",
}


def flatten(sections):
    return [w for h, ps in sections for w in (h.split() + " ".join(ps).split())]


def chunkers(tok):
    return {
        "words_800_100": lambda secs: ingest.chunk_text(" ".join(flatten(secs))),
        "fixed_T256": lambda secs: structure_chunker._windows(flatten(secs), 256, 32, tok.n),
        "structure_T256": lambda secs: structure_chunker.chunk_sections(secs, 256, tok.n),
    }


def edit(sections, kind, size, donor, rng):
    secs = [(h, list(ps)) for h, ps in sections]
    paras = [(i, j) for i, (_, ps) in enumerate(secs) for j in range(len(ps))]
    sentence = donor[0].split(". ")[0] + "."
    n_sub = max(1, len(paras) // 4)
    if kind in ("insert_beginning", "insert_middle", "append_end"):
        new = {"small": [sentence], "medium": [donor[0]], "substantial": donor[:n_sub]}[size]
        i, j = {"insert_beginning": paras[0], "insert_middle": paras[len(paras) // 2], "append_end": paras[-1]}[kind]
        if kind == "append_end":
            j += 1
        if size == "small":  # sentence edits stay inside an existing paragraph
            p = secs[i][1][min(j, len(secs[i][1]) - 1)]
            secs[i][1][min(j, len(secs[i][1]) - 1)] = (new[0] + " " + p) if kind != "append_end" else (p + " " + new[0])
        else:
            secs[i][1][j:j] = new
    elif kind == "replace_paragraphs":
        targets = {"small": [paras[len(paras) // 2]], "medium": [paras[len(paras) // 2]],
                   "substantial": rng.sample(paras, n_sub)}[size]
        for k, (i, j) in enumerate(targets):
            if size == "small":
                parts = secs[i][1][j].split(". ")
                parts[len(parts) // 2] = sentence.rstrip(".")
                secs[i][1][j] = ". ".join(parts)
            else:
                secs[i][1][j] = donor[k % len(donor)]
    elif kind == "delete_paragraphs":
        targets = {"small": [paras[len(paras) // 2]], "medium": [paras[len(paras) // 2]],
                   "substantial": rng.sample(paras, n_sub)}[size]
        for i, j in sorted(targets, reverse=True):
            if size == "small":
                parts = secs[i][1][j].split(". ")
                del parts[len(parts) // 2]
                secs[i][1][j] = ". ".join(parts) or secs[i][1][j]
            else:
                del secs[i][1][j]
    return secs


def main():
    frozen = json.loads(rq.FROZEN.read_text())
    rows = {r["id"]: r for r in rq.load_split("validation")}
    dense = rq.rre.Dense()
    from tokenizers import Tokenizer
    raw = Tokenizer.from_str(dense.m.tok.to_str())
    raw.no_truncation()
    raw.no_padding()
    tok = type("T", (), {"n": staticmethod(lambda t: len(raw.encode(t, add_special_tokens=False).ids))})
    papers = [rq.Paper(rows[p["paper_id"]], tok).sections for p in frozen["dev"]["papers"]]
    rng = random.Random(WORKLOAD["seed"])
    results = {}
    key = lambda t: hashlib.sha256(t.encode()).hexdigest()
    for cname, chunk in chunkers(tok).items():
        for kind in WORKLOAD["edits"]:
            for size in (["-"] if kind in ("unique_new_document", "unchanged_reupload", "renamed_duplicate")
                         else ["small", "medium", "substantial"]):
                agg = {"chunks_new_version": 0, "B_embed": 0, "C_embed": 0, "B_ms": 0.0, "C_ms": 0.0,
                       "B_vectors_stored": 0, "C_vectors_stored": 0, "max_abs_diff_reused": 0.0}
                for p_idx, secs in enumerate(papers):
                    donor_secs = papers[(p_idx + 1) % len(papers)]
                    donor = [p for _, ps in donor_secs for p in ps if len(p.split()) > 40][:20]
                    orig = chunk(secs)
                    new_secs = secs if kind in ("unchanged_reupload", "renamed_duplicate", "unique_new_document") \
                        else edit(secs, kind, size, donor, rng)
                    new = chunk(new_secs)
                    if kind == "unique_new_document":
                        b_embed, c_embed = new, new            # nothing cached for a brand-new document
                        b_store = c_store = len(new)
                    else:
                        doc_changed = key(" ".join(flatten(new_secs))) != key(" ".join(flatten(secs)))
                        b_embed = new if doc_changed else []
                        cached = {key(c) for c in orig}
                        c_embed = [c for c in dict.fromkeys(new) if key(c) not in cached]
                        b_store = len(orig) + (len(new) if doc_changed else 0)
                        c_store = len({key(c) for c in orig} | {key(c) for c in new})
                    t = time.perf_counter()
                    if b_embed:
                        dense.encode(b_embed)
                    agg["B_ms"] += (time.perf_counter() - t) * 1000
                    t = time.perf_counter()
                    if c_embed:
                        dense.encode(c_embed)
                    agg["C_ms"] += (time.perf_counter() - t) * 1000
                    reused = [c for c in new if c in set(orig)][:8]
                    if reused and kind not in ("unique_new_document",):
                        cached_vecs = dense.encode(orig)[[orig.index(c) for c in reused]]
                        fresh_vecs = dense.encode(new)[[new.index(c) for c in reused]]
                        agg["max_abs_diff_reused"] = max(agg["max_abs_diff_reused"], float(np.abs(cached_vecs - fresh_vecs).max()))
                    agg["chunks_new_version"] += len(new)
                    agg["B_embed"] += len(b_embed)
                    agg["C_embed"] += len(c_embed)
                    agg["B_vectors_stored"] += b_store
                    agg["C_vectors_stored"] += c_store
                agg["C_hit_rate"] = round(1 - agg["C_embed"] / agg["chunks_new_version"], 4)
                agg["C_vs_B_embed_reduction"] = round(1 - agg["C_embed"] / agg["B_embed"], 4) if agg["B_embed"] else None
                agg["B_ms"], agg["C_ms"] = round(agg["B_ms"], 1), round(agg["C_ms"], 1)
                results[f"{cname}|{kind}|{size}"] = agg
                print(f"{cname:15s} {kind:20s} {size:11s} chunks {agg['chunks_new_version']:4d}  B {agg['B_embed']:4d}  C {agg['C_embed']:4d}"
                      f"  C-vs-B {agg['C_vs_B_embed_reduction']}  B {agg['B_ms']:7.1f}ms C {agg['C_ms']:7.1f}ms"
                      f"  store B {agg['B_vectors_stored']} C {agg['C_vectors_stored']}  maxdiff {agg['max_abs_diff_reused']:.1e}", flush=True)
    (HERE / "cache_benchmark.json").write_text(json.dumps({"workload": WORKLOAD, "note": __doc__, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
