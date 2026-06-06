"""Tests for the muscle memory system."""

import json
import os
import tempfile
import pytest

from openmind.induction.ingester import (
    IngestResult, FunctionInfo, ClassInfo,
    _parse_python_file, _is_test_file,
)
from openmind.muscle import MuscleMemory, Chord, Reflex


# ── Fixtures ─────────────────────────────────────────────────────────────

SAMPLE_MODULE = '''
"""A sample module."""

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def process_data(data, threshold=0.5):
    """Process data with a threshold filter."""
    filtered = filter_values(data, threshold)
    results = transform(filtered)
    return results

def filter_values(data, threshold):
    return [x for x in data if x > threshold]

def transform(data):
    return [x * 2 for x in data]

class Calculator:
    """A calculator class."""
    def compute(self, x):
        return x * 2
'''

SAMPLE_TEST = '''
import sample_module

def test_add():
    assert sample_module.add(1, 2) == 3

def test_multiply():
    assert sample_module.multiply(2, 3) == 6
'''


@pytest.fixture
def temp_repo():
    """Create a temporary repo for testing."""
    tmpdir = tempfile.mkdtemp(prefix="openmind-test-")
    with open(os.path.join(tmpdir, "sample_module.py"), "w") as f:
        f.write(SAMPLE_MODULE)
    with open(os.path.join(tmpdir, "test_sample.py"), "w") as f:
        f.write(SAMPLE_TEST)
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_result(temp_repo):
    """Create a sample IngestResult."""
    from openmind.induction.ingester import ingest_repo
    return ingest_repo(temp_repo)


# ── Chord tests ──────────────────────────────────────────────────────────

class TestChord:
    def test_exact_match(self):
        chord = Chord(
            name="spi_write",
            module="drivers.spi",
            signature="def spi_write(data: bytes) -> None",
            intent_keywords=["spi", "write", "data", "bytes"],
            decision="hardcode",
            docstring_summary="Write data to SPI bus",
        )
        assert chord.matches("spi_write") == 1.0

    def test_partial_match(self):
        chord = Chord(
            name="spi_write",
            module="drivers.spi",
            signature="def spi_write(data: bytes) -> None",
            intent_keywords=["spi", "write", "data"],
            decision="hardcode",
            docstring_summary="Write data to SPI bus",
        )
        assert chord.matches("spi") == 0.8
        assert chord.matches("write") == 0.8

    def test_keyword_match(self):
        chord = Chord(
            name="process",
            module="core",
            signature="def process()",
            intent_keywords=["authentication", "token", "verify"],
            decision="model",
            docstring_summary="",
        )
        assert chord.matches("authentication") > 0.0

    def test_no_match(self):
        chord = Chord(
            name="foo",
            module="xyz_unique_module",
            signature="def foo()",
            intent_keywords=["x", "y"],
            decision="hardcode",
            docstring_summary="",
        )
        assert chord.matches("zzzzzz_no_match_qwerty") == 0.0


# ── MuscleMemory tests ──────────────────────────────────────────────────

class TestMuscleMemory:
    def test_build(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        assert mm.total_functions > 0
        assert len(mm.chords) > 0

    def test_recall(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        results = mm.recall("add")
        assert len(results) > 0
        assert any(c.name == "add" for c in results)

    def test_recall_top_k(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        results = mm.recall("process", top_k=2)
        assert len(results) <= 2

    def test_recall_one(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        chord = mm.recall_one("add")
        assert chord is not None
        assert chord.name == "add"

    def test_recall_none(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        chord = mm.recall_one("zzzzzz_no_such_function_qwerty")
        assert chord is None

    def test_flex_known(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        reflex = mm.flex("add")
        assert reflex.chord.name == "add"
        assert reflex.exec_strategy in ("direct", "cached", "generate", "hybrid")
        assert reflex.confidence > 0.0

    def test_flex_unknown(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        reflex = mm.flex("zzzzzz_no_such_function_qwerty")
        assert reflex.chord.name == "zzzzzz_no_such_function_qwerty"
        assert reflex.exec_strategy == "generate"
        assert reflex.confidence == 0.0

    def test_stats(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        stats = mm.stats()
        assert stats["total_chords"] > 0
        assert "decision_breakdown" in stats

    def test_save_load(self, sample_result):
        mm = MuscleMemory.build(sample_result)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mm.save(path)
            assert os.path.exists(path)

            mm2 = MuscleMemory.load(path)
            assert len(mm2.chords) == len(mm.chords)
            assert mm2.source_repo == mm.source_repo

            # Verify a chord survived round-trip
            for key in mm.chords:
                if key in mm2.chords:
                    assert mm.chords[key].name == mm2.chords[key].name
                    assert mm.chords[key].decision == mm2.chords[key].decision
                    break
        finally:
            os.unlink(path)

    def test_decision_distribution(self, sample_result):
        mm = MuscleMemory.build(sample_result)
        stats = mm.stats()
        breakdown = stats["decision_breakdown"]
        total = sum(breakdown.values())
        assert total == stats["total_chords"]


# ── Reflex tests ─────────────────────────────────────────────────────────

class TestReflex:
    def test_reflex_creation(self):
        chord = Chord(
            name="test",
            module="mod",
            signature="def test()",
            intent_keywords=[],
            decision="hardcode",
            docstring_summary="",
        )
        reflex = Reflex(chord=chord, exec_strategy="direct", confidence=0.9)
        assert reflex.exec_strategy == "direct"
        assert reflex.confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
