"""Agent muscle memory — compressed callable functions as "chord shapes".

A guitarist's hand knows chord shapes without thinking. The mind sings.
This module does the same for agents: compress functions into callable
"chords" that the agent invokes by intent, without loading source context.

The agent's context window is conscious attention. Every chord we compress
is attention freed for higher-level thinking.

Architecture:
    1. Ingest a codebase → extract every function/class (the "chords")
    2. Compress each into a Chord: name, signature, decision, docstring summary
    3. Index by intent (name, keywords, callers) for O(1) lookup
    4. Agent calls flex("chord_name", args) → result, no source loading

Tripartite mapping:
    HARDCODE = muscle memory (hot path, deterministic, sub-ms)
    CACHED   = replay (same song, same fingering)
    HYBRID   = chord progression with a solo (mostly muscle + some improv)
    MODEL    = full improv (novel situation, LLM actually thinks)
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any

from openmind.induction.ingester import IngestResult, FunctionInfo
from openmind.induction.synchronizer import (
    TripartiteSynchronizer,
    TriHardwareProfile,
    TriApplicationProfile,
    TriUserProfile,
    Decision,
)


# ── Data models ──────────────────────────────────────────────────────────

@dataclass
class Chord:
    """A compressed, callable function shape — the agent's muscle memory.

    Like a guitarist's hand knowing a chord shape:
    - The agent doesn't think about finger positions (source code)
    - It just "plays the chord" by name/intent
    - The hand executes automatically (HARDCODE/CACHED)
    """

    name: str                          # Function name (the "chord name")
    module: str                        # Module path
    signature: str                     # Type signature (shape of the chord)
    intent_keywords: list[str]         # Words that trigger this chord
    decision: str                      # HARDCODE/MODEL/HYBRID/CACHED
    docstring_summary: str             # First line of docstring
    call_count: int = 0                # How many other functions call this
    called_by: list[str] = field(default_factory=list)
    has_tests: bool = False
    source_hash: str = ""              # Hash of source for change detection

    def matches(self, query: str) -> float:
        """How well does a query match this chord? Returns 0.0-1.0.

        Scoring:
        - Exact name match: 1.0
        - Name contains query: 0.8
        - Keyword match: 0.6 per keyword
        - Module contains query: 0.3
        """
        q = query.lower().strip()
        name_lower = self.name.lower()

        if name_lower == q:
            return 1.0
        if q in name_lower:
            return 0.8
        if name_lower in q:
            return 0.7

        # Keyword matching
        kw_matches = sum(1 for kw in self.intent_keywords if len(kw) >= 3 and (q in kw.lower() or kw.lower() in q))
        if kw_matches:
            return 0.6 * min(kw_matches / max(len(self.intent_keywords), 1), 1.0)

        # Module match
        if q in self.module.lower() and len(q) >= 3:
            return 0.3

        return 0.0


@dataclass
class Reflex:
    """A pre-computed execution plan for a chord.

    When the synchronizer decides HARDCODE or CACHED, we can pre-build
    the entire execution plan — no thinking required at runtime.

    Like muscle memory: the signal goes from ear → spinal cord → hand,
    bypassing the brain entirely.
    """

    chord: Chord
    exec_strategy: str           # "direct" | "cached" | "generate" | "hybrid"
    cached_result: Any = None    # For CACHED decisions
    generator_hint: str = ""     # For MODEL decisions: what to tell the LLM
    confidence: float = 1.0      # How confident in this reflex


@dataclass
class MuscleMemory:
    """The agent's full muscle memory — a repertoire of chords.

    Built by ingesting a codebase and compressing every function into
    a Chord, then deciding execution strategy for each.

    Usage:
        mm = MuscleMemory.build(ingest_result)
        chord = mm.recall("spi_write")
        print(chord.decision)  # "HARDCODE" — muscle memory
    """

    chords: dict[str, Chord] = field(default_factory=dict)
    source_repo: str = ""
    total_functions: int = 0
    total_classes: int = 0

    # ── Recall (lookup) ──────────────────────────────────────────────

    def recall(self, intent: str, top_k: int = 5) -> list[Chord]:
        """Recall chords matching an intent. Like a guitarist's hand
        finding the right chord shape from muscle memory.

        Args:
            intent: What you want to do (function name, keyword, description)
            top_k: How many results to return

        Returns:
            List of matching Chords, best match first
        """
        scored = [(chord, chord.matches(intent)) for chord in self.chords.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chord for chord, score in scored[:top_k] if score > 0.0]

    def recall_one(self, intent: str) -> Optional[Chord]:
        """Recall the single best match. Returns None if nothing matches."""
        results = self.recall(intent, top_k=1)
        return results[0] if results else None

    def flex(self, intent: str, **kwargs) -> Reflex:
        """Flex a muscle — get the execution plan for an intent.

        This is THE agent API. The agent says what it wants to do,
        and gets back a Reflex telling it HOW to do it:
        - HARDCODE: call the function directly (muscle memory)
        - CACHED: return the pre-computed result (replay)
        - MODEL: ask the LLM to generate (improvise)
        - HYBRID: try cache, fall back to model

        Args:
            intent: What you want to do
            **kwargs: Arguments to pass through

        Returns:
            Reflex with execution plan
        """
        chord = self.recall_one(intent)
        if chord is None:
            # Unknown chord — agent must improvise
            return Reflex(
                chord=Chord(
                    name=intent,
                    module="unknown",
                    signature="",
                    intent_keywords=[],
                    decision="MODEL",
                    docstring_summary=f"Unknown function: {intent}",
                ),
                exec_strategy="generate",
                generator_hint=f"Generate code for: {intent} with args {kwargs}",
                confidence=0.0,
            )

        # Build reflex from decision
        strategy_map = {
            "HARDCODE": "direct",
            "CACHED": "cached",
            "MODEL": "generate",
            "HYBRID": "hybrid",
        }

        return Reflex(
            chord=chord,
            exec_strategy=strategy_map.get(chord.decision, "generate"),
            confidence=0.9 if chord.has_tests else 0.5,
        )

    # ── Build ────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        result: IngestResult,
        hw_profile: Optional[TriHardwareProfile] = None,
        user_profile: Optional[TriUserProfile] = None,
    ) -> "MuscleMemory":
        """Build muscle memory from an ingest result.

        Args:
            result: Output of openmind.ingest()
            hw_profile: Hardware context (auto-detected if None)
            user_profile: User preferences (defaults if None)

        Returns:
            MuscleMemory with all functions compressed as chords
        """
        import hashlib

        if hw_profile is None:
            hw_profile = TriHardwareProfile()
        if user_profile is None:
            user_profile = TriUserProfile()

        sync = TripartiteSynchronizer()
        chords = {}

        for func in result.functions:
            qualified = f"{func.module}.{func.name}"

            # Build application profile per function
            is_hot = len(func.called_by) >= 2
            app = TriApplicationProfile(
                latency_requirement_ms=10 if is_hot else 200,
                accuracy_requirement=0.95 if func.has_tests else 0.7,
                safety_critical=len(func.called_by) >= 5,
                scale=max(1, len(func.called_by) * 10),
                deterministic=is_hot,
            )

            decision = sync.decide(hw_profile, app, user_profile)

            # Extract intent keywords from name, docstring, arg names
            keywords = _extract_keywords(func)

            # Source hash for change detection
            src_hash = hashlib.md5(func.source_code.encode()).hexdigest()[:12]

            # Docstring summary (first line)
            doc_summary = ""
            if func.docstring:
                doc_summary = func.docstring.split("\n")[0].strip()[:120]

            chord = Chord(
                name=func.name,
                module=func.module,
                signature=func.signature,
                intent_keywords=keywords,
                decision=decision.value,
                docstring_summary=doc_summary,
                call_count=len(func.calls),
                called_by=list(func.called_by),
                has_tests=func.has_tests,
                source_hash=src_hash,
            )
            chords[qualified] = chord

        return cls(
            chords=chords,
            source_repo=result.repo_url,
            total_functions=result.stats.get("total_functions", 0),
            total_classes=result.stats.get("total_classes", 0),
        )

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save muscle memory to a JSON file."""
        data = {
            "source_repo": self.source_repo,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "chords": {k: asdict(v) for k, v in self.chords.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MuscleMemory":
        """Load muscle memory from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        chords = {}
        for k, v in data.get("chords", {}).items():
            chords[k] = Chord(**v)

        return cls(
            chords=chords,
            source_repo=data.get("source_repo", ""),
            total_functions=data.get("total_functions", 0),
            total_classes=data.get("total_classes", 0),
        )

    # ── Statistics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return statistics about the muscle memory repertoire."""
        from collections import Counter
        decisions = Counter(c.decision for c in self.chords.values())
        tested = sum(1 for c in self.chords.values() if c.has_tests)
        return {
            "total_chords": len(self.chords),
            "muscle_memory": decisions.get("hardcode", 0) + decisions.get("cached", 0),
            "needs_thinking": decisions.get("model", 0) + decisions.get("hybrid", 0),
            "tested": tested,
            "untested": len(self.chords) - tested,
            "decision_breakdown": dict(decisions),
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_keywords(func: FunctionInfo) -> list[str]:
    """Extract intent-matching keywords from a function."""
    keywords = []

    # Function name parts (snake_case split)
    parts = func.name.split("_")
    keywords.extend(parts)

    # Argument names
    keywords.extend(func.arg_names)

    # Words from docstring
    if func.docstring:
        first_line = func.docstring.split("\n")[0].lower()
        words = [w for w in first_line.split() if len(w) > 3]
        keywords.extend(words[:5])  # Top 5 meaningful words

    # Called functions (what it does)
    keywords.extend(func.calls[:3])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)

    return unique[:15]  # Cap at 15 keywords
