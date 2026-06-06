"""OpenMind CLI — command-line interface for the induction engine.

Usage:
    openmind ingest <repo-url-or-path>    # Ingest and show summary
    openmind search <query>               # Search ingested repos
    openmind analyze <repo>               # Full pipeline: ingest → vectors → decisions
    openmind decide <function>            # Get execution strategy
    openmind graph <function>             # Show call graph
    openmind lever <repo>                 # Export to lever-runner format
    openmind nail <repo>                  # Export to pincherOS format
    openmind hardware                     # Show hardware probe results
    openmind profiles                     # List built-in profiles
"""

import argparse
import json
import sys
from typing import Optional

from openmind import __version__


def _fmt_decision(d) -> str:
    """Format a Decision enum with emoji."""
    emojis = {
        "hardcode": "⚙️  HARDCODE",
        "model": "🧠 MODEL",
        "hybrid": "🔄 HYBRID",
        "cached": "📦 CACHED",
    }
    val = d.value if hasattr(d, "value") else str(d)
    return emojis.get(val, f"❓ {val}")


def _print_summary(result):
    """Print a summary of an IngestResult."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Stats table
    table = Table(title=f"Ingestion Summary: {result.repo_url}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Functions", str(result.stats.get("total_functions", len(result.functions))))
    table.add_row("Classes", str(result.stats.get("total_classes", len(result.classes))))
    table.add_row("Test Files", str(len(result.test_files)))
    table.add_row("Tested Functions", str(result.stats.get("tested_functions", 0)))
    table.add_row("Call Graph Nodes", str(len(result.call_graph)))

    langs = result.stats.get("languages", {})
    if langs:
        table.add_row("Languages", ", ".join(f"{k} ({v})" for k, v in langs.items()))

    console.print(table)

    # Top functions by connections
    if result.functions:
        func_table = Table(title="Top Functions by Call Connections")
        func_table.add_column("Function", style="cyan")
        func_table.add_column("Module", style="dim")
        func_table.add_column("Calls", style="yellow")
        func_table.add_column("Called By", style="magenta")
        func_table.add_column("Tested", style="green")

        sorted_funcs = sorted(
            result.functions,
            key=lambda f: len(f.calls) + len(f.called_by),
            reverse=True,
        )
        for func in sorted_funcs[:15]:
            func_table.add_row(
                func.name,
                func.module,
                str(len(func.calls)),
                str(len(func.called_by)),
                "✓" if func.has_tests else "✗",
            )

        console.print(func_table)


def cmd_ingest(args):
    """Ingest a repo and show summary."""
    from openmind import ingest, ingest_repo
    import os

    target = args.repo
    if os.path.isdir(target) or os.path.isfile(target):
        result = ingest_repo(os.path.abspath(target))
    elif target.startswith("http") or target.endswith(".git"):
        result = ingest(target, cleanup=args.cleanup)
    else:
        result = ingest_repo(os.path.abspath(target))

    _print_summary(result)

    if args.json:
        data = {
            "repo_url": result.repo_url,
            "stats": result.stats,
            "functions": [
                {
                    "name": f.name,
                    "module": f.module,
                    "signature": f.signature,
                    "calls": f.calls,
                    "called_by": f.called_by,
                    "has_tests": f.has_tests,
                }
                for f in result.functions
            ],
            "call_graph": result.call_graph,
        }
        print(json.dumps(data, indent=2))


def cmd_search(args):
    """Search ingested vectors."""
    from openmind import VectorBuilder

    builder = VectorBuilder()
    results = builder.search_input(args.query, top_k=args.top)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Search: '{args.query}'")
    table.add_column("Function", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Score", style="green")
    table.add_column("Input Context", style="yellow", max_width=60)

    for dv, score in results:
        table.add_row(
            dv.function_name,
            dv.module,
            f"{score:.4f}",
            dv.input_text[:100] + "..." if len(dv.input_text) > 100 else dv.input_text,
        )

    console.print(table)


def cmd_analyze(args):
    """Full pipeline: ingest → vectors → decisions."""
    from openmind import ingest, ingest_repo, VectorBuilder, TripartiteSynchronizer, TriHardwareProfile, TriApplicationProfile, TriUserProfile
    from openmind.induction.hardware import probe_hardware
    import os

    target = args.repo
    if os.path.isdir(target):
        result = ingest_repo(os.path.abspath(target))
    elif target.startswith("http") or target.endswith(".git"):
        result = ingest(target, cleanup=args.cleanup)
    else:
        result = ingest_repo(os.path.abspath(target))

    _print_summary(result)

    # Build vectors
    builder = VectorBuilder()
    vectors = builder.build_all(result)

    # Tripartite decisions
    hw_caps = probe_hardware()
    hw = TriHardwareProfile(
        compute_power=0.5,
        gpu_available=hw_caps.gpu,
        memory_gb=hw_caps.ram_gb,
        device_type=hw_caps.device_type,
    )
    sync = TripartiteSynchronizer()

    decisions = {}
    for func in result.functions:
        qualified = f"{func.module}.{func.name}"
        is_hot = func.name in [f.name for f in sorted(result.functions, key=lambda f: len(f.calls), reverse=True)[:max(1, len(result.functions) // 5)]]
        app = TriApplicationProfile(
            latency_requirement_ms=10 if is_hot else 100,
            accuracy_requirement=0.9 if func.has_tests else 0.5,
            safety_critical=False,
            deterministic=func.has_tests,
        )
        user = TriUserProfile()
        d = sync.decide(hw, app, user)
        decisions[qualified] = d

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Tripartite Decisions")
    table.add_column("Function", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Decision", style="green")

    for qual, d in sorted(decisions.items()):
        parts = qual.rsplit(".", 1)
        name = parts[-1] if len(parts) > 1 else qual
        mod = parts[0] if len(parts) > 1 else ""
        table.add_row(name, mod, _fmt_decision(d))

    console.print(table)


def cmd_decide(args):
    """Get execution strategy for a function."""
    from openmind import TripartiteSynchronizer, TriHardwareProfile, TriApplicationProfile, TriUserProfile

    hw = TriHardwareProfile()
    app = TriApplicationProfile(
        safety_critical=args.safety,
        latency_requirement_ms=args.latency,
    )
    user = TriUserProfile(
        wants_creativity=args.creativity,
        wants_manual_control=args.manual,
    )

    if args.override:
        user.preference_override = args.override

    sync = TripartiteSynchronizer()
    d = sync.decide(hw, app, user)

    from rich.console import Console
    console = Console()
    console.print(f"\n[bold]Function:[/bold] {args.function}")
    console.print(f"[bold]Decision:[/bold] {_fmt_decision(d)}")
    console.print(f"[bold]Latency:[/bold] ≤{args.latency}ms")
    console.print(f"[bold]Safety-critical:[/bold] {args.safety}")
    console.print()


def cmd_graph(args):
    """Show call graph for a function."""
    from openmind import ingest_repo
    import os

    if not args.repo:
        print("Error: --repo is required for graph command")
        sys.exit(1)

    if os.path.isdir(args.repo):
        result = ingest_repo(os.path.abspath(args.repo))
    else:
        print(f"Error: {args.repo} is not a directory")
        sys.exit(1)

    target = args.function
    call_graph = result.call_graph

    from rich.console import Console
    from rich.tree import Tree

    console = Console()

    matches = [k for k in call_graph if k.endswith(f".{target}") or k == target]
    if not matches:
        print(f"Function '{target}' not found in call graph")
        print(f"Available: {', '.join(k.split('.')[-1] for k in list(call_graph.keys())[:30])}...")
        sys.exit(1)

    for match in matches:
        tree = Tree(f"[bold cyan]{match}[/bold cyan]")
        _build_tree(tree, match, call_graph, depth=0, max_depth=args.depth or 5)
        console.print(tree)


def _build_tree(tree, node, call_graph, depth, max_depth):
    """Recursively build a call tree."""
    if depth >= max_depth:
        return
    callees = call_graph.get(node, [])
    for callee in callees:
        qualified = callee
        for k in call_graph:
            if k.split(".")[-1] == callee:
                qualified = k
                break
        branch = tree.add(f"[yellow]{callee}[/yellow]")
        if qualified in call_graph:
            _build_tree(branch, qualified, call_graph, depth + 1, max_depth)


def cmd_export_lever(args):
    """Export to lever-runner format."""
    from openmind import ingest_repo, export_lever_pack
    import os

    if os.path.isdir(args.repo):
        result = ingest_repo(os.path.abspath(args.repo))
    else:
        print(f"Error: {args.repo} is not a directory")
        sys.exit(1)

    records = []
    for func in result.functions:
        if func.has_tests:
            rec = export_lever_pack(
                function_name=func.name,
                module_path=func.module,
                description=func.docstring or "",
                db_path=args.db if args.db else None,
            )
            records.append(rec)

    print(f"Exported {len(records)} lever-runner commands")
    if args.json:
        print(json.dumps(records, indent=2))


def cmd_export_nail(args):
    """Export to pincherOS .nail format."""
    from openmind import ingest_repo, export_nail
    import os

    if os.path.isdir(args.repo):
        result = ingest_repo(os.path.abspath(args.repo))
    else:
        print(f"Error: {args.repo} is not a directory")
        sys.exit(1)

    records = []
    for func in result.functions:
        manifest = export_nail(
            function_name=func.name,
            cached_output=func.signature,
            description=func.docstring or "",
            export_dir=args.dir if args.dir else None,
        )
        records.append(manifest)

    print(f"Exported {len(records)} .nail files")
    if args.json:
        print(json.dumps(records, indent=2))


def cmd_hardware(args):
    """Show hardware probe results."""
    from openmind import probe_hardware

    cap = probe_hardware()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Hardware Capabilities")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("GPU", f"{cap.gpu_name or 'None'} ({'available' if cap.gpu else 'absent'})")
    table.add_row("GPU VRAM", f"{cap.gpu_vram_mb} MiB" if cap.gpu_vram_mb else "N/A")
    table.add_row("RAM", f"{cap.ram_gb} GB")
    table.add_row("CPU Cores", str(cap.cpu_cores))
    table.add_row("Architecture", cap.arch)
    table.add_row("Device Type", cap.device_type)
    table.add_row("Battery", f"{cap.battery_pct}%" if cap.battery_pct else "Plugged in / N/A")

    console.print(table)


def cmd_profiles(args):
    """List built-in profiles."""
    from openmind.induction.profiles import (
        GAMING_PC, DEV_LAPTOP, RASPBERRY_PI, CAR_BRAKE_SYSTEM,
        NPC_BEHAVIOR, TERMINAL_COMMANDS,
    )
    from openmind import TripartiteSynchronizer

    scenarios = {
        "Gaming PC": GAMING_PC,
        "Dev Laptop": DEV_LAPTOP,
        "Raspberry Pi": RASPBERRY_PI,
        "Car Brake System": CAR_BRAKE_SYSTEM,
        "NPC Behavior": NPC_BEHAVIOR,
        "Terminal Commands": TERMINAL_COMMANDS,
    }

    sync = TripartiteSynchronizer()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Built-in Profiles")
    table.add_column("Scenario", style="cyan")
    table.add_column("Device", style="dim")
    table.add_column("Decision", style="green")

    for name, (hw, app, user) in scenarios.items():
        d = sync.decide(hw, app, user)
        table.add_row(name, hw.device_type, _fmt_decision(d))

    console.print(table)


def main():
    """Entry point for the openmind CLI."""
    parser = argparse.ArgumentParser(
        prog="openmind",
        description="OpenMind — Code induction engine",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a repo and show summary")
    p_ingest.add_argument("repo", help="Repo URL or local path")
    p_ingest.add_argument("--cleanup", action="store_true", help="Remove cloned repo after ingestion")
    p_ingest.add_argument("--json", action="store_true", help="Output as JSON")
    p_ingest.set_defaults(func=cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Search ingested repos")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Full pipeline: ingest → vectors → decisions")
    p_analyze.add_argument("repo", help="Repo URL or local path")
    p_analyze.add_argument("--cleanup", action="store_true", help="Remove cloned repo after analysis")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    # decide
    p_decide = subparsers.add_parser("decide", help="Get execution strategy for a function")
    p_decide.add_argument("function", help="Function name")
    p_decide.add_argument("--latency", type=float, default=100, help="Latency requirement (ms)")
    p_decide.add_argument("--safety", action="store_true", help="Safety-critical")
    p_decide.add_argument("--creativity", type=float, default=0.3, help="Creativity level (0-1)")
    p_decide.add_argument("--manual", action="store_true", help="Manual control wanted")
    p_decide.add_argument("--override", choices=["hardcode", "model"], help="Force a decision")
    p_decide.set_defaults(func=cmd_decide)

    # graph
    p_graph = subparsers.add_parser("graph", help="Show call graph for a function")
    p_graph.add_argument("function", help="Function name to trace")
    p_graph.add_argument("--repo", required=True, help="Repo path")
    p_graph.add_argument("--depth", type=int, default=5, help="Max graph depth")
    p_graph.set_defaults(func=cmd_graph)

    # lever export
    p_lever = subparsers.add_parser("lever", help="Export to lever-runner format")
    p_lever.add_argument("repo", help="Repo path")
    p_lever.add_argument("--db", help="Path to commands.db")
    p_lever.add_argument("--json", action="store_true", help="Output as JSON")
    p_lever.set_defaults(func=cmd_export_lever)

    # nail export
    p_nail = subparsers.add_parser("nail", help="Export to pincherOS .nail format")
    p_nail.add_argument("repo", help="Repo path")
    p_nail.add_argument("--dir", help="Export directory")
    p_nail.add_argument("--json", action="store_true", help="Output as JSON")
    p_nail.set_defaults(func=cmd_export_nail)

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Show hardware probe results")
    p_hw.set_defaults(func=cmd_hardware)

    # profiles
    p_profiles = subparsers.add_parser("profiles", help="List built-in profiles")
    p_profiles.set_defaults(func=cmd_profiles)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
