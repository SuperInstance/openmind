"""Continuous iteration engine for a repo.

Like murmuring (our existing pattern), but for any repo:
1. PASS 1: Index everything, build initial vectors
2. PASS 2: Run tests, observe behaviors, build output vectors
3. PASS 3: Identify hot paths (frequently called, latency-sensitive)
4. PASS 4: Hardcode hot paths via lever-runner
5. PASS 5: Deploy and monitor, feed hardware readings back

The spreader runs continuously, improving its model of the repo over time.

Usage:
    from openmind.induction import Spreader

    spreader = Spreader("https://github.com/user/repo")
    spreader.start(passes=5)
    spreader.status()
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from openmind.induction.ingester import IngestResult, ingest
from openmind.induction.synchronizer import Decision, Synchronizer
from openmind.induction.vectors import VectorBuilder


class PassPhase(Enum):
    """Phases of the spreading process."""

    IDLE = "idle"
    INGESTING = "ingesting"
    VECTORIZING = "vectorizing"
    TESTING = "testing"
    ANALYZING = "analyzing"
    HARDCODING = "hardcoding"
    MONITORING = "monitoring"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class SpreadState:
    """Current state of the spreader."""

    repo_url: str
    phase: PassPhase = PassPhase.IDLE
    current_pass: int = 0
    total_passes: int = 5
    started_at: Optional[float] = None
    last_pass_at: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    hot_paths: list[str] = field(default_factory=list)
    decisions: dict[str, Decision] = field(default_factory=dict)
    test_results: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


class Spreader:
    """Continuous iteration engine for a repository.

    Runs multiple passes over an ingested repo, progressively improving
    the vector model and making synchronization decisions.
    """

    def __init__(
        self,
        repo_url: str,
        target_dir: Optional[str] = None,
        state_dir: Optional[str] = None,
    ):
        self.repo_url = repo_url
        self.target_dir = target_dir
        self.state_dir = state_dir or os.path.expanduser("~/.openmind/spreader")
        os.makedirs(self.state_dir, exist_ok=True)

        self.state = SpreadState(repo_url=repo_url)
        self._ingest_result: Optional[IngestResult] = None
        self._vector_builder: Optional[VectorBuilder] = None
        self._synchronizer = Synchronizer()

        # Try to load existing state
        self._load_state()

    def _state_path(self) -> str:
        """Get the state file path for this repo."""
        safe_name = self.repo_url.replace("/", "_").replace(":", "_")
        return os.path.join(self.state_dir, f"{safe_name}.json")

    def _load_state(self):
        """Load persisted state if available."""
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.state.phase = PassPhase(data.get("phase", "idle"))
                self.state.current_pass = data.get("current_pass", 0)
                self.state.total_passes = data.get("total_passes", 5)
                self.state.hot_paths = data.get("hot_paths", [])
                self.state.decisions = data.get("decisions", {})
                self.state.stats = data.get("stats", {})
            except Exception:
                pass

    def _save_state(self):
        """Persist current state."""
        data = {
            "repo_url": self.state.repo_url,
            "phase": self.state.phase.value,
            "current_pass": self.state.current_pass,
            "total_passes": self.state.total_passes,
            "hot_paths": self.state.hot_paths,
            "decisions": {k: v.value for k, v in self.state.decisions.items()},
            "stats": self.state.stats,
            "last_pass_at": self.state.last_pass_at,
        }
        with open(self._state_path(), "w") as f:
            json.dump(data, f, indent=2)

    def start(self, passes: int = 5):
        """Run the spreading process for N passes.

        Args:
            passes: Number of passes to run (default 5)
        """
        self.state.total_passes = passes
        self.state.started_at = time.time()
        self.state.phase = PassPhase.INGESTING
        self._save_state()

        try:
            for pass_num in range(1, passes + 1):
                self.state.current_pass = pass_num
                self._run_pass(pass_num)
                self.state.last_pass_at = time.time()
                self._save_state()

            self.state.phase = PassPhase.COMPLETE
        except Exception as e:
            self.state.phase = PassPhase.ERROR
            self.state.errors.append(str(e))
        finally:
            self._save_state()

    def _run_pass(self, pass_num: int):
        """Run a single pass of the spreading process."""
        if pass_num == 1:
            self._pass_1_ingest()
        elif pass_num == 2:
            self._pass_2_vectorize()
        elif pass_num == 3:
            self._pass_3_test()
        elif pass_num == 4:
            self._pass_4_analyze()
        elif pass_num >= 5:
            self._pass_5_monitor()

    def _pass_1_ingest(self):
        """PASS 1: Index everything, build initial vectors."""
        self.state.phase = PassPhase.INGESTING
        self._save_state()

        self._ingest_result = ingest(self.repo_url, self.target_dir)
        self.state.stats["ingest"] = {
            "functions": len(self._ingest_result.functions),
            "classes": len(self._ingest_result.classes),
            "test_files": len(self._ingest_result.test_files),
        }

    def _pass_2_vectorize(self):
        """PASS 2: Build vectors for all functions."""
        self.state.phase = PassPhase.VECTORIZING
        self._save_state()

        if not self._ingest_result:
            self._ingest_result = ingest(self.repo_url, self.target_dir)

        self._vector_builder = VectorBuilder()
        vectors = self._vector_builder.build_all(self._ingest_result)
        self.state.stats["vectors"] = len(vectors)

    def _pass_3_test(self):
        """PASS 3: Run tests and observe behaviors."""
        self.state.phase = PassPhase.TESTING
        self._save_state()

        if not self._ingest_result:
            self._ingest_result = ingest(self.repo_url, self.target_dir)

        results = {"passed": 0, "failed": 0, "errors": []}
        repo_path = self._ingest_result.local_path

        # Try running pytest if available
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--tb=short", "-q"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + result.stderr
            # Parse pytest output
            for line in output.split("\n"):
                if "passed" in line:
                    results["passed"] = int(
                        line.split("passed")[0].strip().split()[-1] or "0"
                    )
                if "failed" in line:
                    results["failed"] = int(
                        line.split("failed")[0].strip().split()[-1] or "0"
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["errors"].append("pytest not available or timed out")

        self.state.test_results = results
        self.state.stats["tests"] = results

    def _pass_4_analyze(self):
        """PASS 4: Identify hot paths and make synchronization decisions."""
        self.state.phase = PassPhase.ANALYZING
        self._save_state()

        if not self._ingest_result:
            self._ingest_result = ingest(self.repo_url, self.target_dir)

        # Identify hot paths: functions called by many others
        call_counts = {}
        for func in self._ingest_result.functions:
            for callee in func.calls:
                call_counts[callee] = call_counts.get(callee, 0) + 1

        # Top 20% of called functions are "hot"
        if call_counts:
            sorted_calls = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)
            threshold = max(1, len(sorted_calls) // 5)
            self.state.hot_paths = [name for name, _ in sorted_calls[:threshold]]

        # Make synchronizer decisions for each function
        for func in self._ingest_result.functions:
            qualified = f"{func.module}.{func.name}"
            is_hot = func.name in self.state.hot_paths

            decision = self._synchronizer.decide(
                application={
                    "latency_ms": 10 if is_hot else 1000,
                    "safety": 0.8 if func.has_tests else 0.3,
                    "creativity": 0.1 if is_hot else 0.7,
                    "accuracy": 0.9 if is_hot else 0.5,
                }
            )
            self.state.decisions[qualified] = decision.decision

        self.state.stats["hot_paths"] = len(self.state.hot_paths)
        self.state.stats["decisions"] = {
            d.value: sum(1 for v in self.state.decisions.values() if v == d)
            for d in Decision
        }

    def _pass_5_monitor(self):
        """PASS 5+: Monitor and refine."""
        self.state.phase = PassPhase.MONITORING
        self._save_state()

        # Re-ingest to catch any changes
        self._ingest_result = ingest(self.repo_url, self.target_dir)

        # Re-vectorize
        if not self._vector_builder:
            self._vector_builder = VectorBuilder()
        self._vector_builder.build_all(self._ingest_result)

        self.state.stats["monitor_pass"] = self.state.current_pass

    def status(self) -> dict:
        """Get the current status of the spreader."""
        return {
            "repo_url": self.state.repo_url,
            "phase": self.state.phase.value,
            "current_pass": self.state.current_pass,
            "total_passes": self.state.total_passes,
            "hot_paths": len(self.state.hot_paths),
            "decisions": {
                d.value: sum(1 for v in self.state.decisions.values() if v == d)
                for d in Decision
            },
            "stats": self.state.stats,
            "errors": self.state.errors,
            "last_pass_at": self.state.last_pass_at,
        }
