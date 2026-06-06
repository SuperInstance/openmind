"""Build vectors for both sides of inference.

For each function in the ingested repo:
- INPUT vector: what context triggers this function? (callers, arguments, types)
- OUTPUT vector: what does this function produce? (return type, side effects, calls)

These vectors enable:
- Induction: "given input X, what function handles it?" (search input vectors)
- Deduction: "given function Y, what does it produce?" (search output vectors)
- Hybrid: chain input→output across functions to predict entire flows

Usage:
    from openmind import VectorBuilder
    result = openmind.ingest("https://github.com/user/repo")
    builder = openmind.VectorBuilder()
    builder.build_all(result)
    matches = builder.search_input("handle authentication")
"""

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openmind.induction.ingester import FunctionInfo, IngestResult


@dataclass
class DualVector:
    """A pair of vectors representing a function's input and output semantics."""

    function_name: str
    module: str
    input_vector: list[float]
    output_vector: list[float]
    input_text: str  # The text that was embedded for input
    output_text: str  # The text that was embedded for output


def _simple_hash_embed(text: str, dim: int = 128) -> list[float]:
    """Simple deterministic embedding using hash-based projections.

    This is a fallback when no embedding model is available.
    Not semantically meaningful but provides consistent vectors for lookup.
    """
    vector = [0.0] * dim
    # Split into chunks, hash each chunk, project into vector
    words = text.lower().split()
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(dim):
            seed = h + j
            vector[j] += math.sin(seed * 0.1) * (1.0 / (i + 1))

    # Normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorBuilder:
    """Builds dual-side vectors (input + output) for all functions in a repo.

    Stores vectors in a local SQLite database for fast retrieval.
    Can optionally use a real embedding model via LiteLLM.
    """

    def __init__(self, db_path: Optional[str] = None, embed_fn=None):
        """Initialize the vector builder.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.openmind/vectors.db
            embed_fn: Optional embedding function. If None, uses hash-based fallback.
                      Signature: embed_fn(text: str) -> list[float]
        """
        if db_path is None:
            db_dir = os.path.expanduser("~/.openmind")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "vectors.db")

        self.db_path = db_path
        self.embed_fn = embed_fn or _simple_hash_embed
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT NOT NULL,
                    module TEXT NOT NULL,
                    repo_url TEXT,
                    input_vector TEXT NOT NULL,
                    output_vector TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_function_name ON vectors(function_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_module ON vectors(module)
            """)

    def _build_input_text(self, func: FunctionInfo) -> str:
        """Build the text representation for input-side embedding."""
        parts = []
        # Function name and signature describe what triggers it
        parts.append(f"function: {func.name}")
        parts.append(f"signature: {func.signature}")

        # Who calls it describes the context
        if func.called_by:
            parts.append(f"called_by: {', '.join(func.called_by[:10])}")

        # Arguments and types describe expected input
        for arg, typ in func.arg_types.items():
            parts.append(f"arg {arg}: {typ}")

        # Docstring describes purpose
        if func.docstring:
            parts.append(func.docstring)

        return " ".join(parts)

    def _build_output_text(self, func: FunctionInfo) -> str:
        """Build the text representation for output-side embedding."""
        parts = []
        # Return type
        if func.return_type:
            parts.append(f"returns: {func.return_type}")

        # What it calls (side effects / downstream behavior)
        if func.calls:
            parts.append(f"calls: {', '.join(func.calls[:10])}")

        # Docstring often describes what the function produces
        if func.docstring:
            parts.append(func.docstring)

        # Source code snippet (first few lines as behavioral hint)
        lines = func.source_code.split("\n")
        if len(lines) > 1:
            # Just the body, not the signature
            body = "\n".join(lines[1:4])  # First 3 lines of body
            parts.append(f"behavior: {body.strip()}")

        return " ".join(parts)

    def build_function_vectors(self, func: FunctionInfo, repo_url: str = "") -> DualVector:
        """Build dual vectors for a single function."""
        input_text = self._build_input_text(func)
        output_text = self._build_output_text(func)

        input_vector = self.embed_fn(input_text)
        output_vector = self.embed_fn(output_text)

        return DualVector(
            function_name=func.name,
            module=func.module,
            input_vector=input_vector,
            output_vector=output_vector,
            input_text=input_text,
            output_text=output_text,
        )

    def build_all(self, result: IngestResult) -> list[DualVector]:
        """Build and store vectors for all functions in an ingested repo."""
        vectors = []
        for func in result.functions:
            dv = self.build_function_vectors(func, result.repo_url)
            self._store_vector(dv, result.repo_url)
            vectors.append(dv)
            # Also update the function's embedding
            func.embedding = dv.input_vector

        return vectors

    def _store_vector(self, dv: DualVector, repo_url: str = ""):
        """Store a dual vector in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO vectors (function_name, module, repo_url, input_vector, output_vector, input_text, output_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    dv.function_name,
                    dv.module,
                    repo_url,
                    json.dumps(dv.input_vector),
                    json.dumps(dv.output_vector),
                    dv.input_text,
                    dv.output_text,
                ),
            )

    def search(self, query: str, repo_url: Optional[str] = None, top_k: int = 5, side: str = "input") -> list[tuple[DualVector, float]]:
        """Search for functions matching a query.

        Args:
            query: Search text
            repo_url: Optional repo URL to filter by
            top_k: Number of results to return
            side: "input" for induction search, "output" for deduction search

        Returns:
            List of (DualVector, similarity_score) tuples
        """
        query_vector = self.embed_fn(query)
        return self._search(query_vector, side, repo_url, top_k)

    def search_input(self, query: str, repo_url: Optional[str] = None, top_k: int = 5) -> list[tuple[DualVector, float]]:
        """Search for functions whose input context matches a query.

        This is INDUCTION: given input context, find what functions handle it.
        """
        return self.search(query, repo_url, top_k, side="input")

    def search_output(self, query: str, repo_url: Optional[str] = None, top_k: int = 5) -> list[tuple[DualVector, float]]:
        """Search for functions whose output behavior matches a query.

        This is DEDUCTION: given a description of behavior, find functions that produce it.
        """
        return self.search(query, repo_url, top_k, side="output")

    def _search(
        self, query_vector: list[float], side: str, repo_url: Optional[str], top_k: int
    ) -> list[tuple[DualVector, float]]:
        """Internal search across stored vectors."""
        results = []

        with sqlite3.connect(self.db_path) as conn:
            if repo_url:
                rows = conn.execute(
                    "SELECT function_name, module, input_vector, output_vector, input_text, output_text FROM vectors WHERE repo_url = ?",
                    (repo_url,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT function_name, module, input_vector, output_vector, input_text, output_text FROM vectors"
                ).fetchall()

            for row in rows:
                func_name, module, iv_json, ov_json, it, ot = row
                stored_vector = json.loads(iv_json if side == "input" else ov_json)
                sim = _cosine_similarity(query_vector, stored_vector)
                results.append(
                    (
                        DualVector(
                            function_name=func_name,
                            module=module,
                            input_vector=json.loads(iv_json),
                            output_vector=json.loads(ov_json),
                            input_text=it,
                            output_text=ot,
                        ),
                        sim,
                    )
                )

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
