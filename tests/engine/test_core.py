import math

import numpy as np
import pytest

from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.metrics import mrr, ndcg_at_k, recall_at_k
from tenantrag.retrieval.fusion import reciprocal_rank_fusion
from tenantrag.text import HashingEmbedder, tokenize
from tenantrag.types import Acl, Chunk, Hit, Principal


def P(uid=1, groups=(), tenant="t"):
    return Principal(tenant, uid, tuple(groups))


def test_acl_default_deny_and_deny_wins():
    assert not Acl().permits(P())
    assert Acl(is_public=True).permits(P())
    assert Acl(allow_groups=frozenset({5})).permits(P(groups=[5]))
    assert not Acl(allow_groups=frozenset({5})).permits(P(groups=[6]))
    assert Acl(allow_users=frozenset({9})).permits(P(uid=9))
    assert not Acl(is_public=True, allow_groups=frozenset({5}), deny_groups=frozenset({5})).permits(P(groups=[5]))


def test_hindi_and_gujarati_words_are_not_split_at_vowel_signs():
    assert tokenize("रिफंड नीति") == ["रिफंड", "नीति"]
    assert tokenize("રિફંડ નીતિ") == ["રિફંડ", "નીતિ"]
    assert tokenize("Refund-window: 30 days") == ["refund", "window", "30", "days"]


def test_embedder_deterministic_and_normalized():
    e = HashingEmbedder(64)
    a, b = e.embed_query("refund window"), e.embed_query("refund window")
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert math.isclose(ndcg_at_k(["x", "a"], ["a"], 5), 1 / math.log2(3))
    assert math.isnan(recall_at_k(["a"], [], 1))


def test_rrf_prefers_items_ranked_by_both_legs():
    c = lambda i: Chunk("t", i, f"d{i}", 1, 0, "x")
    a = [Hit(c(1), 1), Hit(c(2), 1), Hit(c(3), 1)]
    b = [Hit(c(3), 1), Hit(c(2), 1), Hit(c(4), 1)]
    order = [h.chunk.chunk_id for h in reciprocal_rank_fusion([a, b])]
    assert set(order[:2]) == {2, 3} and set(order[2:]) == {1, 4}


def test_corpus_deterministic_skewed_and_global_unique_ids():
    c1, c2 = build_corpus(12, max_docs=100, seed=3), build_corpus(12, max_docs=100, seed=3)
    assert [x.text for x in c1.chunks[:20]] == [x.text for x in c2.chunks[:20]]
    sizes = sorted((t["n_docs"] for t in c1.tenants.values()), reverse=True)
    assert sizes[0] >= 5 * sizes[-1]
    groups = [g for t in c1.tenants.values() for g in t["groups"]]
    users = [u.user_id for us in c1.users.values() for u in us]
    assert len(groups) == len(set(groups)) and len(users) == len(set(users))
    assert any(g.kind == "forbidden" for g in c1.golden)
