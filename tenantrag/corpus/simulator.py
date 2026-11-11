"""Synthetic multi-tenant corpus + ACL + golden-set generator.

Real client data cannot be published, so tenants are synthetic. The generator is
deliberately adversarial for shared indexes:

  * Tenants on the same topic reuse the same *subject names* ("refund-window of
    Plan Korvani") with different values. A shared index therefore contains many
    near-duplicate chunks across tenants, and only one is correct for a given
    tenant. This is what makes selective-filter ANN recall loss measurable.
  * Tenant sizes follow a Zipf-like skew (a few large tenants, a long tail).
  * Documents carry realistic ACL shapes: public, group-restricted,
    user-specific and deny-overridden.

All text is pseudo-words, so no retriever can lean on real-language semantics.
The benchmark therefore measures isolation and index behaviour, not embedding
quality. Swap in real corpora via ``load_directory_corpus`` for quality studies.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

from tenantrag.text import chunk_by_tokens
from tenantrag.types import Acl, Chunk, Principal

_SYL = ["ka", "ro", "vi", "mu", "ta", "len", "shi", "dor", "bel", "nu", "fa", "zen", "or", "pa", "qui", "lo", "mar", "tes", "ix", "ul"]
TOPICS = {
    "hr": ["notice period", "leave allowance", "probation length", "overtime rate", "review cycle"],
    "billing": ["refund window", "invoice due date", "late fee", "trial length", "seat limit"],
    "security": ["password rotation", "session timeout", "audit retention", "mfa grace period", "lockout threshold"],
    "support": ["response target", "escalation window", "ticket retention", "weekend coverage", "satisfaction goal"],
    "legal": ["contract term", "renewal notice", "liability cap", "data retention", "termination notice"],
    "ops": ["backup frequency", "deploy window", "rollback limit", "capacity buffer", "incident review"],
}
QUERY_TEMPLATES = [
    "What is the {attr} for {subject}?",
    "tell me the {attr} of {subject}",
    "{subject} {attr}?",
    "how long is the {attr} under {subject}",
]


def _word(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(_SYL) for _ in range(n))


@dataclass(slots=True)
class SimDoc:
    tenant_id: str
    doc_id: str
    topic: str
    subject: str
    attr: str
    value: str
    paragraphs: list[str]
    acl: Acl


@dataclass(slots=True)
class GoldenItem:
    tenant_id: str
    principal: Principal
    query: str
    relevant_doc_ids: tuple[str, ...]
    answer: str | None
    kind: str = "answerable"  # answerable | forbidden
    forbidden_doc_id: str | None = None
    canary: str | None = None


@dataclass
class Corpus:
    tenants: dict[str, dict] = field(default_factory=dict)
    docs: dict[str, SimDoc] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    users: dict[str, list[Principal]] = field(default_factory=dict)
    golden: list[GoldenItem] = field(default_factory=list)

    def chunks_for(self, tenant_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.tenant_id == tenant_id]

    def authoritative_acl(self, tenant_id: str, doc_id: str) -> Acl | None:
        d = self.docs.get(doc_id)
        return d.acl if d and d.tenant_id == tenant_id else None


def zipf_sizes(n_tenants: int, max_docs: int, min_docs: int, alpha: float) -> list[int]:
    return [max(min_docs, int(max_docs / ((i + 1) ** alpha))) for i in range(n_tenants)]


def build_corpus(
    n_tenants: int = 10,
    *,
    max_docs: int = 120,
    min_docs: int = 6,
    alpha: float = 0.9,
    queries_per_tenant: int = 6,
    seed: int = 7,
    subject_pool: int = 360,
    typo_rate: float = 0.35,
) -> Corpus:
    rng = random.Random(seed)
    corpus = Corpus()
    ids = itertools.count(1)
    gids = itertools.count(1)  # globally unique group ids
    uids = itertools.count(1)  # globally unique user ids
    topic_names = list(TOPICS)
    # Per-topic shared vocabularies (subjects are reused across tenants on purpose).
    # 40 base words x digits 1-9: many subjects differ only by a digit, so retrieval
    # has to separate confusable names, not just find a rare token.
    subjects = {}
    for t in topic_names:
        bases = [_word(rng) for _ in range(40)]
        combos = [f"{b}-{d}" for b in bases for d in range(1, 10)]
        rng.shuffle(combos)
        subjects[t] = combos[:subject_pool]
    filler = {t: [_word(rng, 2) for _ in range(70)] for t in topic_names}
    sizes = zipf_sizes(n_tenants, max_docs, min_docs, alpha)
    rng.shuffle(sizes)

    for ti in range(n_tenants):
        tid = f"tenant-{ti:04d}"
        topic = topic_names[ti % len(topic_names)]
        n_docs = min(sizes[ti], subject_pool)
        groups = [next(gids) for _ in range(4)]
        n_users = min(30, max(6, n_docs // 8))
        users = []
        for _ in range(n_users):
            g = tuple(rng.sample(groups, rng.choice([1, 1, 2])))
            users.append(Principal(tid, next(uids), g))
        corpus.users[tid] = users
        corpus.tenants[tid] = {"topic": topic, "n_docs": n_docs, "groups": groups}

        subs = rng.sample(subjects[topic], n_docs)
        for di, subject in enumerate(subs):
            attr = rng.choice(TOPICS[topic])
            value = f"{rng.randint(2, 90)} {rng.choice(['days', 'hours', 'months', 'weeks'])}"
            roll = rng.random()
            if roll < 0.5:
                acl = Acl(is_public=True)
            elif roll < 0.85:
                acl = Acl(allow_groups=frozenset({rng.choice(groups)}))
            elif roll < 0.95:
                acl = Acl(allow_users=frozenset({rng.choice(users).user_id}))
            else:
                g1, g2 = rng.sample(groups, 2)
                acl = Acl(allow_groups=frozenset({g1}), deny_groups=frozenset({g2}))
            doc_id = f"{tid}-doc-{di:04d}"
            paras = []
            fact_at = rng.randrange(3)
            for pi in range(3):
                # the fact paragraph is short and FAQ-like; other paragraphs are filler
                n_fill = 10 if pi == fact_at else 40
                words = " ".join(rng.choice(filler[topic]) for _ in range(n_fill))
                if pi == fact_at:
                    words += f". The {attr} of {subject} is {value}."
                paras.append(words)
            doc = SimDoc(tid, doc_id, topic, subject, attr, value, paras, acl)
            corpus.docs[doc_id] = doc
            for pi, p in enumerate(paras):
                corpus.chunks.append(Chunk(tid, next(ids), doc_id, 1, pi, p, "en", acl, {"subject": subject}))

        _make_golden(corpus, tid, rng, queries_per_tenant, typo_rate)
    return corpus


def _typo(subject: str, rng: random.Random) -> str:
    base, _, digit = subject.partition("-")
    if len(base) > 3:
        i = rng.randrange(1, len(base) - 1)
        base = base[:i] + base[i + 1 :]
    return f"{base}-{digit}"


def _make_golden(corpus: Corpus, tid: str, rng: random.Random, n: int, typo_rate: float = 0.0) -> None:
    docs = [d for d in corpus.docs.values() if d.tenant_id == tid]
    users = corpus.users[tid]
    for d in rng.sample(docs, min(n, len(docs))):
        subj = _typo(d.subject, rng) if rng.random() < typo_rate else d.subject
        q = rng.choice(QUERY_TEMPLATES).format(attr=d.attr, subject=subj)
        allowed = [u for u in users if d.acl.permits(u)]
        denied = [u for u in users if not d.acl.permits(u)]
        if allowed:
            corpus.golden.append(GoldenItem(tid, rng.choice(allowed), q, (d.doc_id,), d.value))
        if denied:
            corpus.golden.append(
                GoldenItem(tid, rng.choice(denied), q, (), None, "forbidden", forbidden_doc_id=d.doc_id)
            )


def load_directory_corpus(root: str | Path, *, seed: int = 7) -> Corpus:
    """Build tenants from ``root/<tenant>/*.txt`` (one tenant per folder, e.g. arXiv-by-field
    or Wikipedia-by-category). Every document is public inside its tenant; add ACLs afterwards.
    """
    rng = random.Random(seed)
    corpus = Corpus()
    ids = itertools.count(1)
    uid = itertools.count(1)
    for tdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        tid = tdir.name
        corpus.tenants[tid] = {"topic": tid, "n_docs": 0, "groups": []}
        corpus.users[tid] = [Principal(tid, next(uid), ())]
        for f in sorted(tdir.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            doc_id = f"{tid}-{hashlib.sha1(f.name.encode()).hexdigest()[:10]}"
            acl = Acl(is_public=True)
            corpus.docs[doc_id] = SimDoc(tid, doc_id, tid, f.stem, "", "", [text], acl)
            corpus.tenants[tid]["n_docs"] += 1
            for pi, ch in enumerate(chunk_by_tokens(text)):
                corpus.chunks.append(Chunk(tid, next(ids), doc_id, 1, pi, ch, "en", acl, {"title": f.stem}))
    _ = rng
    return corpus
