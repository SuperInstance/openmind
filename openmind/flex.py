"""Flex — the agent's primary interface to muscle memory.

Usage:
    from openmind import ingest, MuscleMemory

    # One-shot: ingest + build muscle memory + flex
    result = ingest("./my-esp32-firmware")
    mm = MuscleMemory.build(result)

    # Flex a chord — get execution plan
    reflex = mm.flex("spi_write", data=b"\\x01\\x02\\x03")
    print(reflex.exec_strategy)  # "direct" (muscle memory)

    # Recall what's available
    chords = mm.recall("gpio")
    for c in chords:
        print(f"  {c.name} ({c.decision}): {c.docstring_summary}")

    # Save for later (no re-ingestion needed)
    mm.save("firmware_memory.json")

    # Load and flex
    mm2 = MuscleMemory.load("firmware_memory.json")
    reflex2 = mm2.flex("wifi_connect", ssid="my-network")
"""

# The flex() function is on MuscleMemory itself.
# This module provides convenience functions for common patterns.

from openmind.muscle import MuscleMemory, Chord, Reflex
from openmind.induction.ingester import ingest, ingest_repo

import json
import os
from typing import Optional


def quick_flex(
    source: str,
    intent: str,
    top_k: int = 5,
) -> list[Reflex]:
    """One-shot: ingest source → build muscle memory → flex an intent.

    Args:
        source: Repo URL or local path
        intent: What you want to do
        top_k: How many results

    Returns:
        List of Reflex objects (execution plans)
    """
    if os.path.isdir(source):
        result = ingest_repo(source)
    else:
        result = ingest(source)

    mm = MuscleMemory.build(result)
    chords = mm.recall(intent, top_k=top_k)
    return [mm.flex(c.name) for c in chords]


def load_and_flex(
    memory_path: str,
    intent: str,
) -> Reflex:
    """Load saved muscle memory and flex an intent.

    Args:
        memory_path: Path to saved .json muscle memory file
        intent: What you want to do

    Returns:
        Reflex with execution plan
    """
    mm = MuscleMemory.load(memory_path)
    return mm.flex(intent)


__all__ = ["quick_flex", "load_and_flex", "MuscleMemory", "Chord", "Reflex"]
