#!/usr/bin/env python3
"""Cache reconstruction check (offline, real local BGE): for edited documents, an index rebuilt
from CACHED vectors (unchanged chunks, encoded with the original version) + newly encoded chunks
is compared with a FRESH full re-encode of the edited document, on the 41 frozen dev questions.
Reports identical top-4 rankings and max |distance| difference."""
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cache_benchmark as cb  # noqa: E402
import run_qasper_eval as rq  # noqa: E402

EDITS = [("insert_middle", "medium"), ("append_end", "small"), ("replace_paragraphs", "medium"),
         ("delete_paragraphs", "medium"), ("insert_beginning", "substantial")]


def main():
    frozen = json.loads(rq.FROZEN.read_text())
    rows = {r["id"]: r for r in rq.load_split("validation")}
    dense = rq.rre.Dense()
    from tokenizers import Tokenizer
    raw = Tokenizer.from_str(dense.m.tok.to_str())
    raw.no_truncation()
    raw.no_padding()
    tok = type("T", (), {"n": staticmethod(lambda t: len(raw.encode(t, add_special_tokens=False).ids))})
    chunkers = cb.chunkers(tok)
    rng = random.Random(5)
    papers = [(p, rq.Paper(rows[p["paper_id"]], tok).sections) for p in frozen["dev"]["papers"]]
    out = {}
    for cname in ("fixed_T256", "structure_T256"):
        for kind, size in EDITS:
            agg = {"queries": 0, "top4_identical": 0, "max_abs_distance_diff": 0.0, "reused_chunks": 0}
            for idx, (p, secs) in enumerate(papers):
                donor = [x for _, ps in papers[(idx + 1) % len(papers)][1] for x in ps if len(x.split()) > 40][:20]
                orig = chunkers[cname](secs)
                new = chunkers[cname](cb.edit(secs, kind, size, donor, rng))
                cache = dict(zip(orig, dense.encode(orig)))            # vectors as cached at original indexing
                fresh = dense.encode(new)                              # full re-encode of the edited version
                todo = [c for c in dict.fromkeys(new) if c not in cache]
                newv = dict(zip(todo, dense.encode(todo))) if todo else {}
                rebuilt = np.stack([cache.get(c, newv.get(c)) for c in new])
                agg["reused_chunks"] += sum(c in cache for c in new)
                for q in p["questions"]:
                    qv = dense.encode([q["question"]])[0]
                    da, db = rebuilt @ qv, fresh @ qv
                    ta, tb = np.argsort(-da, kind="stable")[:4], np.argsort(-db, kind="stable")[:4]
                    agg["queries"] += 1
                    agg["top4_identical"] += bool((ta == tb).all())
                    agg["max_abs_distance_diff"] = max(agg["max_abs_distance_diff"], float(np.abs(da - db).max()))
            out[f"{cname}|{kind}|{size}"] = agg
            print(cname, kind, size, agg, flush=True)
    (HERE / "cache_topk_equivalence.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
