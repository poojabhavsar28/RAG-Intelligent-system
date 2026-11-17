# TenantRAG design notes

## Principles
1. **Default deny.** An ACL with no grants permits nobody; a missing authority record means invisible; ingestion must never default to public.
2. **Documents own policy, chunks own retrieval.** The authoritative ACL, lifecycle state, version and trust tier live on the document. Chunks carry denormalised copies used only for fast filtering.
3. **Revocation is immediate, grant is eventual.** The final retrieval decision uses the authoritative ACL, so a stale chunk copy that still says "allowed" is dropped (and counted as `stale_acl_hits`); a stale copy that still says "denied" only delays a grant. That is the fail-safe direction, and a version mismatch is a metric, not a hard drop.
4. **Globally unique group and user ids.** If a tenant filter were ever dropped, an ACL still could not match across tenants.
5. **tenant_id leads every key, index and foreign key; no cross-tenant unique constraint** (it would leak existence through error messages). A test scans `pg_indexes` for violations; a short allowlist covers hashed credentials and surrogate keys.
6. **Isolation in depth, tested one layer at a time.** Partition/index per tenant, RLS (`FORCE`, fail-closed on unset setting, `SET LOCAL` only), roles without BYPASSRLS, tenant-keyed doc store, per-tenant cache keys/encryption.

## Retrieval ACL semantics
`Acl.permits`: deny groups win; then public; then user grant; then group grant. Postgres encodes this in `document_acl` (`permission='deny'` added by `pg0005`); Qdrant encodes it as payload filters (`must tenant_id`, `should is_public|allow_groups|allow_users`, `must_not deny_groups`). Tests assert all three backends agree.

## Tenancy models
| | dedicated | shared + filter | hybrid |
|---|---|---|---|
| isolation | structural (own index) | filter must be correct | mixed |
| filtered-ANN recall | exact | degrades with selectivity | long tail still degrades |
| memory/cost per tenant | highest (per-index overhead) | lowest | middle |
| ops | many objects | one object | promotion logic |

In Postgres the physical basis is `PARTITION BY HASH(tenant_id)` for the tail and `LIST` partitions for large tenants (see `postgres-foundation.md`).

## Cache (see `tenantrag/cache/scoped.py` docstring)
Seven independent layers plus bug injection. Admission: only grounded, cited, non-PII, non-temporal answers from trusted, non-suspect sources; abstentions are never cached. Semantic hits additionally need equal constraint slots (digits, negations, comparators), two thresholds with a paraphrase check in the middle band, and exact-only for languages without validated slot extraction (Gujarati by default). A hit never replays a response: it is re-materialised and re-checked.

Residual risks that are *not* eliminated: timing inside a scope (mitigation: `hide_hits_ms` padding or scope=`user`), vector inversion if semantic vectors are stored in plaintext, and semantic false hits (a wrong-but-non-leaking answer).

## Prompt-injection defence
Ingestion scoring marks suspect chunks; suspect chunks are excluded from context and from cache admission; retrieved text is fenced as untrusted data with instruction-like lines neutralised; output is checked for foreign citations, canaries and PII. None of the isolation guarantees depend on the classifier: ACLs and tenant scoping are enforced before generation.
