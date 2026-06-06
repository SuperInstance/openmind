"""openmind CLI — Agent muscle memory from the terminal.

Usage:
    openmind ingest <repo-url-or-path>    # Ingest and show summary
    openmind flex <repo> <intent>         # Flex a chord (get execution plan)
    openmind recall <repo> <query>        # Search for matching chords
    openmind probe                        # Hardware probe
    openmind save <repo> <output.json>    # Build and save muscle memory
    openmind stats <memory.json>          # Show muscle memory statistics
"""

import argparse
import json
import os
import sys

from openmind.induction.ingester import ingest, ingest_repo
from openmind.induction.hardware import probe_hardware
from openmind.muscle import MuscleMemory


def _ingest_source(source: str):
    """Ingest from URL or local path."""
    if os.path.isdir(source):
        return ingest_repo(source)
    return ingest(source)


def cmd_ingest(args):
    """Ingest a repository and show summary."""
    result = _ingest_source(args.source)

    print(f"\n  📦 Ingested: {result.repo_url}")
    print(f"  🔧 Functions: {len(result.functions)}")
    print(f"  📚 Classes:   {len(result.classes)}")
    print(f"  ✅ Test files: {len(result.test_files)}")

    tested = sum(1 for f in result.functions if f.has_tests)
    print(f"  🧪 Tested functions: {tested}/{len(result.functions)} ({100*tested/max(len(result.functions),1):.0f}%)")

    # Top connected functions
    from collections import Counter
    degree = Counter()
    for func in result.functions:
        q = f"{func.module}.{func.name}"
        degree[q] = len(func.calls) + len(func.called_by)

    if degree:
        print(f"\n  🌐 Top 5 most-connected functions:")
        for name, deg in degree.most_common(5):
            print(f"     {name} (degree={deg})")

    # Language breakdown
    langs = result.stats.get("languages", {})
    if langs:
        print(f"\n  🗣  Languages: {', '.join(f'{k} ({v})' for k, v in langs.items())}")

    print()


def cmd_flex(args):
    """Flex a chord — get the execution plan for an intent."""
    result = _ingest_source(args.source)
    mm = MuscleMemory.build(result)

    reflex = mm.flex(args.intent)
    chord = reflex.chord

    # Decision icon
    icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}
    icon = icons.get(chord.decision, "?")

    print(f"\n  {icon} {chord.name} ({chord.module})")
    print(f"  Strategy:  {reflex.exec_strategy}")
    print(f"  Decision:  {chord.decision}")
    print(f"  Confidence: {reflex.confidence:.0%}")
    print(f"  Signature: {chord.signature}")
    if chord.docstring_summary:
        print(f"  Summary:   {chord.docstring_summary}")
    print(f"  Has tests: {'✅' if chord.has_tests else '❌'}")
    print(f"  Called by: {len(chord.called_by)} functions")
    print()


def cmd_recall(args):
    """Search for chords matching a query."""
    result = _ingest_source(args.source)
    mm = MuscleMemory.build(result)

    chords = mm.recall(args.query, top_k=args.top)

    if not chords:
        print(f"\n  No chords matching '{args.query}'")
        return

    print(f"\n  🔍 Results for '{args.query}' ({len(chords)} matches):\n")
    icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}

    for chord in chords:
        icon = icons.get(chord.decision, "?")
        print(f"  {icon} {chord.name} ({chord.module})")
        print(f"     {chord.signature}")
        if chord.docstring_summary:
            print(f"     → {chord.docstring_summary}")
        print()


def cmd_probe(args):
    """Run hardware probe."""
    hw = probe_hardware()

    print(f"\n  🔍 Hardware Probe\n")
    print(f"  Device:    {hw.device_type}")
    print(f"  Arch:      {hw.arch}")
    print(f"  CPU cores: {hw.cpu_cores}")
    print(f"  RAM:       {hw.ram_gb:.1f} GB")

    if hw.gpu:
        print(f"  GPU:       ✅ {hw.gpu_name or 'detected'}")
        if hw.gpu_vram_mb:
            print(f"  GPU VRAM:  {hw.gpu_vram_mb} MiB")
    else:
        print(f"  GPU:       ❌ none")

    if hw.battery_pct is not None:
        status = "plugged" if hw.battery_plugged else "battery"
        print(f"  Battery:   {hw.battery_pct:.0f}% ({status})")

    print()


def cmd_save(args):
    """Build muscle memory and save to file."""
    result = _ingest_source(args.source)
    mm = MuscleMemory.build(result)
    mm.save(args.output)

    stats = mm.stats()
    print(f"\n  💾 Saved muscle memory to {args.output}")
    print(f"  Total chords: {stats['total_chords']}")
    print(f"  Muscle memory: {stats['muscle_memory']} (HARDCODE + CACHED)")
    print(f"  Needs thinking: {stats['needs_thinking']} (MODEL + HYBRID)")
    print()


def cmd_stats(args):
    """Show statistics from a saved muscle memory file."""
    mm = MuscleMemory.load(args.memory)
    stats = mm.stats()

    print(f"\n  📊 Muscle Memory Statistics\n")
    print(f"  Source repo: {mm.source_repo}")
    print(f"  Total chords: {stats['total_chords']}")
    print(f"  Tested: {stats['tested']} | Untested: {stats['untested']}")

    breakdown = stats["decision_breakdown"]
    total = stats["total_chords"]
    icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}

    print(f"\n  Decision breakdown:")
    for decision in ("hardcode", "cached", "hybrid", "model"):
        count = breakdown.get(decision, 0)
        pct = 100 * count / max(total, 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        icon = icons.get(decision, "?")
        print(f"  {icon} {decision:10s} {count:4d} ({pct:5.1f}%) {bar}")

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="openmind",
        description="Agent muscle memory — compress codebases into callable chord shapes",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a repository")
    p_ingest.add_argument("source", help="Repo URL or local path")

    # flex
    p_flex = subparsers.add_parser("flex", help="Flex a chord (get execution plan)")
    p_flex.add_argument("source", help="Repo URL or local path")
    p_flex.add_argument("intent", help="What you want to do")

    # recall
    p_recall = subparsers.add_parser("recall", help="Search for matching chords")
    p_recall.add_argument("source", help="Repo URL or local path")
    p_recall.add_argument("query", help="Search query")
    p_recall.add_argument("--top", type=int, default=5, help="Number of results")

    # probe
    subparsers.add_parser("probe", help="Run hardware probe")

    # save
    p_save = subparsers.add_parser("save", help="Build and save muscle memory")
    p_save.add_argument("source", help="Repo URL or local path")
    p_save.add_argument("output", help="Output JSON file path")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show muscle memory statistics")
    p_stats.add_argument("memory", help="Path to saved muscle memory JSON")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "flex":
        cmd_flex(args)
    elif args.command == "recall":
        cmd_recall(args)
    elif args.command == "probe":
        cmd_probe(args)
    elif args.command == "save":
        cmd_save(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
