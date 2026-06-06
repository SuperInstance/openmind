"""Jupyter cell/line magic for openmind.

Usage in Jupyter:
    %load_ext openmind.jupyter

    # Analyze a repo inline
    %%openmind analyze ./my-project

    # Flex a chord
    %openmind flex spi_write

    # Search for functions
    %openmind recall gpio

    # Show muscle memory stats
    %openmind stats

    # Hardware probe
    %openmind probe
"""

import os
import json
from typing import Optional

try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython.display import display, HTML, JSON as IPythonJSON
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    # Stub for when IPython isn't available
    def magics_class(cls): return cls
    class Magics: pass
    def line_magic(func): return func
    def cell_magic(func): return func


if HAS_IPYTHON:
    @magics_class
    class OpenMindMagics(Magics):
        """Jupyter magic commands for agent muscle memory."""

        _memory_cache = {}  # repo_path → MuscleMemory

        def _get_memory(self, source: str):
            """Get or build muscle memory for a source."""
            from openmind.induction.ingester import ingest, ingest_repo
            from openmind.muscle import MuscleMemory

            if source in self._memory_cache:
                return self._memory_cache[source]

            if os.path.isdir(source):
                result = ingest_repo(source)
            else:
                result = ingest(source)

            mm = MuscleMemory.build(result)
            self._memory_cache[source] = mm
            return mm

        @cell_magic
        def openmind(self, line, cell=""):
            """Main cell magic: %%openmind <command> [args]

            Commands:
                analyze <source>   — Full analysis with rich HTML output
                flex <intent>      — Flex a chord (needs prior analyze)
                recall <query>     — Search chords (needs prior analyze)
            """
            parts = line.strip().split(maxsplit=1)
            if not parts:
                print("Usage: %%openmind <command> [args]")
                return

            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "analyze":
                self._cmd_analyze(args)
            elif cmd == "flex":
                self._cmd_flex(args)
            elif cmd == "recall":
                self._cmd_recall(args)
            elif cmd == "stats":
                self._cmd_stats()
            elif cmd == "probe":
                self._cmd_probe()
            else:
                print(f"Unknown command: {cmd}")
                print("Available: analyze, flex, recall, stats, probe")

        @line_magic
        def openmind_line(self, line):
            """Line magic: %openmind <command> [args]"""
            return self.openmind(line, "")

        def _cmd_analyze(self, source: str):
            """Analyze a repo and show rich HTML output."""
            from openmind.muscle import MuscleMemory

            mm = self._get_memory(source.strip())
            stats = mm.stats()

            # Build HTML table
            icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}

            breakdown_rows = ""
            for decision in ("hardcode", "cached", "hybrid", "model"):
                count = stats["decision_breakdown"].get(decision, 0)
                pct = 100 * count / max(stats["total_chords"], 1)
                icon = icons.get(decision, "?")
                breakdown_rows += f"""
                <tr>
                    <td>{icon} {decision}</td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>"""

            html = f"""
            <div style="font-family: monospace; background: #1e1e2e; color: #cdd6f4;
                        padding: 16px; border-radius: 8px; margin: 8px 0;">
                <h3 style="color: #89b4fa; margin-top: 0;">🧠 Muscle Memory: {source}</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="color: #a6adc8;">Total chords:</td>
                        <td><b>{stats['total_chords']}</b></td></tr>
                    <tr><td style="color: #a6adc8;">Muscle memory:</td>
                        <td style="color: #a6e3a1;"><b>{stats['muscle_memory']}</b> (HARDCODE + CACHED)</td></tr>
                    <tr><td style="color: #a6adc8;">Needs thinking:</td>
                        <td style="color: #f38ba8;"><b>{stats['needs_thinking']}</b> (MODEL + HYBRID)</td></tr>
                    <tr><td style="color: #a6adc8;">Tested:</td>
                        <td>{stats['tested']} / {stats['total_chords']}</td></tr>
                </table>
                <h4 style="color: #89b4fa;">Decision Breakdown</h4>
                <table style="width: 80%; border-collapse: collapse;">
                    <tr style="color: #a6adc8;">
                        <th align="left">Strategy</th><th>Count</th><th>%</th></tr>
                    {breakdown_rows}
                </table>
            </div>
            """
            display(HTML(html))

            # Store in user namespace for programmatic access
            if self.shell:
                self.shell.user_ns["_openmind_memory"] = mm

        def _cmd_flex(self, intent: str):
            """Flex a chord."""
            mm = self.shell.user_ns.get("_openmind_memory") if self.shell else None
            if mm is None:
                print("Run %%openmind analyze <repo> first")
                return

            reflex = mm.flex(intent.strip())
            chord = reflex.chord

            icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}
            icon = icons.get(chord.decision, "?")

            html = f"""
            <div style="font-family: monospace; background: #1e1e2e; color: #cdd6f4;
                        padding: 12px; border-radius: 8px; margin: 8px 0;">
                <h4 style="color: #89b4fa; margin-top: 0;">{icon} {chord.name}</h4>
                <table>
                    <tr><td style="color: #a6adc8;">Module:</td><td>{chord.module}</td></tr>
                    <tr><td style="color: #a6adc8;">Strategy:</td><td><b>{reflex.exec_strategy}</b></td></tr>
                    <tr><td style="color: #a6adc8;">Decision:</td><td>{chord.decision}</td></tr>
                    <tr><td style="color: #a6adc8;">Confidence:</td><td>{reflex.confidence:.0%}</td></tr>
                    <tr><td style="color: #a6adc8;">Signature:</td><td><code>{chord.signature}</code></td></tr>
                    <tr><td style="color: #a6adc8;">Tests:</td>
                        <td>{'✅' if chord.has_tests else '❌'}</td></tr>
                </table>
            </div>
            """
            display(HTML(html))

        def _cmd_recall(self, query: str):
            """Search for matching chords."""
            mm = self.shell.user_ns.get("_openmind_memory") if self.shell else None
            if mm is None:
                print("Run %%openmind analyze <repo> first")
                return

            chords = mm.recall(query.strip(), top_k=10)
            if not chords:
                print(f"No chords matching '{query}'")
                return

            icons = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔀", "cached": "📦"}

            rows = ""
            for c in chords:
                icon = icons.get(c.decision, "?")
                rows += f"""
                <tr>
                    <td>{icon}</td>
                    <td><code>{c.name}</code></td>
                    <td style="color: #a6adc8;">{c.module}</td>
                    <td>{c.decision}</td>
                    <td>{'✅' if c.has_tests else '❌'}</td>
                </tr>"""

            html = f"""
            <div style="font-family: monospace; background: #1e1e2e; color: #cdd6f4;
                        padding: 12px; border-radius: 8px; margin: 8px 0;">
                <h4 style="color: #89b4fa; margin-top: 0;">🔍 Results for '{query}' ({len(chords)})</h4>
                <table style="width: 100%;">
                    <tr style="color: #a6adc8;">
                        <th></th><th align="left">Name</th><th align="left">Module</th>
                        <th>Decision</th><th>Tests</th></tr>
                    {rows}
                </table>
            </div>
            """
            display(HTML(html))

        def _cmd_stats(self):
            """Show muscle memory statistics."""
            mm = self.shell.user_ns.get("_openmind_memory") if self.shell else None
            if mm is None:
                print("Run %%openmind analyze <repo> first")
                return

            stats = mm.stats()
            print(f"  Total chords: {stats['total_chords']}")
            print(f"  Muscle memory: {stats['muscle_memory']}")
            print(f"  Needs thinking: {stats['needs_thinking']}")
            print(f"  Decision breakdown: {stats['decision_breakdown']}")

        def _cmd_probe(self):
            """Run hardware probe."""
            from openmind.induction.hardware import probe_hardware
            hw = probe_hardware()

            html = f"""
            <div style="font-family: monospace; background: #1e1e2e; color: #cdd6f4;
                        padding: 12px; border-radius: 8px; margin: 8px 0;">
                <h4 style="color: #89b4fa; margin-top: 0;">🔍 Hardware Probe</h4>
                <table>
                    <tr><td style="color: #a6adc8;">Device:</td><td>{hw.device_type}</td></tr>
                    <tr><td style="color: #a6adc8;">Arch:</td><td>{hw.arch}</td></tr>
                    <tr><td style="color: #a6adc8;">CPU cores:</td><td>{hw.cpu_cores}</td></tr>
                    <tr><td style="color: #a6adc8;">RAM:</td><td>{hw.ram_gb:.1f} GB</td></tr>
                    <tr><td style="color: #a6adc8;">GPU:</td>
                        <td>{'✅ ' + (hw.gpu_name or 'yes') if hw.gpu else '❌ none'}</td></tr>
                </table>
            </div>
            """
            display(HTML(html))


def load_ipython_extension(ipython):
    """Register the magic with IPython."""
    if HAS_IPYTHON:
        ipython.register_magics(OpenMindMagic)
    else:
        print("IPython not available. Install with: pip install openmind[jupyter]")
