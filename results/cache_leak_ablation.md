| injected_bug | attack | layers_off | leaked | stopped_by | security_events | as_expected |
|---|---|---|---|---|---|---|
| no bug (control) | cross_tenant | - | False | 1-3 key/keyspace isolation | - | True |
| key omits tenant | cross_tenant | - | False | 4 AEAD | decrypt_failure | True |
| key omits tenant, AEAD off | cross_tenant | aead | False | 5 tenant tag | tenant_tag_mismatch | True |
| key omits tenant, AEAD+tag off | cross_tenant | aead,tag_check | False | 6 validate-on-read | - | True |
| key omits tenant, AEAD+tag+validate off | cross_tenant | aead,tag_check,validate | False | 7 re-materialization | - | True |
| key omits tenant, layers 4-7 off | cross_tenant | aead,rematerialize,tag_check,validate | True | none (expected leak) | - | True |
| scope omitted from key | cross_user | - | False | 6 validate-on-read | - | True |
| scope omitted, validate off | cross_user | validate | False | 7 re-materialization | - | True |
| scope omitted, validate+remat off | cross_user | rematerialize,validate | True | none (expected leak) | - | True |
| shared semantic index, filter dropped | cross_tenant | - | False | 2/4-7 | decrypt_failure | True |
| ACL revoked, events lost | stale_acl | - | False | 6 validate-on-read | - | True |
| ACL revoked, events lost, validate off | stale_acl | validate | False | 7 re-materialization | - | True |
| ACL revoked, events lost, validate+remat off | stale_acl | rematerialize,validate | True | none (expected leak) | - | True |
