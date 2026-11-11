"""CI quality gate: fail when retrieval quality regresses or any leak appears.

    python -m tenantrag.eval.gate            # evaluate and compare to eval/baseline.json
    python -m tenantrag.eval.gate --update   # rewrite the baseline (intentional change only)

The evaluation is deterministic (seeded synthetic corpus + hashing embedder), so
the tolerance only needs to absorb floating-point noise, not run-to-run variance.
Latency is reported but never gated, because shared CI runners are too noisy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder

BASELINE = Path(__file__).resolve().parents[2] / "eval" / "baseline.json"
GATED = ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_10")
TOLERANCE = 0.02


def run_reference() -> dict:
    corpus = build_corpus(n_tenants=20, max_docs=300, queries_per_tenant=8, seed=11)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    r = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=corpus.authoritative_acl)
    rep = evaluate_retrieval(corpus.golden, lambda g: r.retrieve(g.principal, g.query, 10))
    return rep.as_row()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv)
    row = run_reference()
    print(json.dumps(row, indent=2))
    if row["leaks"] or row["tenant_violations"]:
        print(f"FAIL: isolation violated (leaks={row['leaks']}, tenant_violations={row['tenant_violations']})")
        return 1
    if args.update or not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({k: row[k] for k in GATED}, indent=2) + "\n")
        print(f"baseline written to {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    bad = [f"{k}: {row[k]:.4f} < baseline {base[k]:.4f} - {TOLERANCE}" for k in GATED if row[k] < base[k] - TOLERANCE]
    if bad:
        print("FAIL: quality regression\n  " + "\n  ".join(bad))
        return 1
    print("PASS: no regression vs baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
