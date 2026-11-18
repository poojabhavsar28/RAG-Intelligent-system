import pytest

from scripts.migrate_chroma_to_pgvector import DIMENSIONS, _json_safe, _stable_uuid, _vector


def test_legacy_document_and_chunk_ids_are_deterministic():
    assert _stable_uuid("same-input") == _stable_uuid("same-input")
    assert _stable_uuid("same-input") != _stable_uuid("different-input")


def test_chroma_embedding_dimension_is_checked():
    values = [0.1] * DIMENSIONS
    assert len(_vector(values, tenant_id="tenant-a", collection_id="chunk-1")) == DIMENSIONS


def test_chroma_embedding_dimension_mismatch_fails_closed():
    with pytest.raises(ValueError, match="requires 768"):
        _vector([0.1] * 384, tenant_id="tenant-a", collection_id="chunk-1")


def test_chroma_metadata_is_json_safe():
    assert _json_safe({"n": 3, "labels": ("a", "b")}) == {"n": 3, "labels": ["a", "b"]}
