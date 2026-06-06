"""Tests for openmind vector builder."""

import os
import tempfile
import pytest

from openmind import VectorBuilder, DualVector, ingest_repo


@pytest.fixture
def sample_repo(tmp_path):
    """Create a minimal sample repo."""
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "mod.py").write_text('''
def authenticate(username: str, password: str) -> bool:
    """Authenticate a user."""
    return check_password(username, password)

def check_password(username: str, password: str) -> bool:
    """Check if password matches."""
    return True

def create_token(user_id: int) -> str:
    """Create an auth token."""
    return f"token_{user_id}"
''')
    return tmp_path


@pytest.fixture
def vector_builder(tmp_path):
    """Create a VectorBuilder with temp DB."""
    db_path = str(tmp_path / "test_vectors.db")
    return VectorBuilder(db_path=db_path)


def test_vector_builder_init(vector_builder):
    """Test VectorBuilder initializes."""
    assert vector_builder is not None
    assert os.path.exists(vector_builder.db_path)


def test_build_vectors(vector_builder, sample_repo):
    """Test building vectors from an ingest result."""
    result = ingest_repo(str(sample_repo))
    vectors = vector_builder.build_all(result)

    assert len(vectors) > 0
    assert all(isinstance(v, DualVector) for v in vectors)


def test_dual_vector_fields(vector_builder, sample_repo):
    """Test DualVector has expected fields."""
    result = ingest_repo(str(sample_repo))
    vectors = vector_builder.build_all(result)

    for v in vectors:
        assert len(v.input_vector) == 128  # default dim
        assert len(v.output_vector) == 128
        assert isinstance(v.function_name, str)
        assert isinstance(v.module, str)
        assert len(v.input_text) > 0
        assert len(v.output_text) > 0


def test_search_input(vector_builder, sample_repo):
    """Test input-side search."""
    result = ingest_repo(str(sample_repo))
    vector_builder.build_all(result)

    matches = vector_builder.search_input("authenticate user")
    assert len(matches) > 0
    assert all(isinstance(m, tuple) and len(m) == 2 for m in matches)
    # First match should have highest similarity
    assert matches[0][1] >= matches[-1][1]


def test_search_output(vector_builder, sample_repo):
    """Test output-side search."""
    result = ingest_repo(str(sample_repo))
    vector_builder.build_all(result)

    matches = vector_builder.search_output("create token")
    assert len(matches) > 0


def test_search_generic(vector_builder, sample_repo):
    """Test generic search method."""
    result = ingest_repo(str(sample_repo))
    vector_builder.build_all(result)

    matches = vector_builder.search("authenticate", top_k=3)
    assert len(matches) <= 3


def test_search_by_repo(vector_builder, sample_repo):
    """Test search filtering by repo_url."""
    result = ingest_repo(str(sample_repo))
    vector_builder.build_all(result)

    matches = vector_builder.search_input("test", repo_url=str(sample_repo))
    assert len(matches) > 0


def test_hash_embedding_deterministic():
    """Test that hash-based embedding is deterministic."""
    from openmind.induction.vectors import _simple_hash_embed

    v1 = _simple_hash_embed("hello world")
    v2 = _simple_hash_embed("hello world")
    assert v1 == v2


def test_cosine_similarity():
    """Test cosine similarity."""
    from openmind.induction.vectors import _cosine_similarity

    assert abs(_cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6
    assert abs(_cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-6


def test_custom_embed_fn(tmp_path):
    """Test VectorBuilder with custom embedding function."""
    def custom_embed(text):
        return [1.0] * 64

    db_path = str(tmp_path / "custom.db")
    builder = VectorBuilder(db_path=db_path, embed_fn=custom_embed)

    assert builder.embed_fn == custom_embed
