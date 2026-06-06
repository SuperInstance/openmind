"""Decide: hardcode, model, or hybrid?

Tripartite factors:
1. HARDWARE: GPU available? RAM? Battery? Edge device?
2. APPLICATION: Latency required? Accuracy needed? Safety critical?
3. USER: Manual control wanted? Creative output? Consistent behavior?

Decision matrix:
- High safety + low latency → HARDCODE (lever-runner)
- High creativity + flexible latency → MODEL (LLM inference)
- Medium safety + medium latency → HYBRID (cache + fallback)
- Edge device + low power → CACHED (pincherOS .nail file)

Two APIs are provided:

1. ``Synchronizer`` — scoring-based, dict-based API (legacy)
2. ``TripartiteSynchronizer`` — rule-based, dataclass API (preferred)
"""

import enum
import math
import platform
import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Shared
# ============================================================================

class Decision(enum.Enum):
    """The four execution strategies."""
    HARDCODE = "hardcode"  # Compiled/fast path via lever-runner
    MODEL = "model"        # LLM inference (flexible, creative)
    HYBRID = "hybrid"      # Cache + model fallback
    CACHED = "cached"      # Pre-computed, read-only (pincherOS .nail file)


# ============================================================================
# Legacy dataclasses (for Synchronizer)
# ============================================================================

@dataclass
class HardwareProfile:
    """Hardware context for decision making."""
    gpu: bool = False
    gpu_name: Optional[str] = None
    ram_gb: float = 8.0
    battery_pct: Optional[float] = None  # None = plugged in
    is_edge: bool = False
    cpu_count: int = 4
    platform_name: str = ""

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Auto-detect hardware profile."""
        gpu = False
        gpu_name = None
        if os.path.exists("/usr/bin/nvidia-smi"):
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    gpu = True
                    gpu_name = result.stdout.strip()
            except Exception:
                pass
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            gpu = True
            gpu_name = "Apple Silicon"
        ram_gb = 8.0
        try:
            import subprocess
            result = subprocess.run(
                ["free", "-g"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    ram_gb = float(parts[1])
        except Exception:
            pass
        machine = platform.machine().lower()
        is_edge = machine in ("aarch64", "armv7l") and not (platform.system() == "Darwin")
        return cls(
            gpu=gpu, gpu_name=gpu_name, ram_gb=ram_gb,
            cpu_count=os.cpu_count() or 4, is_edge=is_edge,
            platform_name=f"{platform.system()}/{platform.machine()}",
        )


@dataclass
class ApplicationProfile:
    """Application requirements (legacy)."""
    latency_ms: float = 1000.0
    safety: float = 0.5
    accuracy: float = 0.5
    creativity: float = 0.5
    throughput_rps: float = 1.0


@dataclass
class UserProfile:
    """User preferences (legacy)."""
    manual_control: bool = False
    creative_output: bool = False
    prefer_speed: bool = False
    prefer_quality: bool = False
    power_saving: bool = False


@dataclass
class SyncDecision:
    """Result of a Synchronizer decision."""
    decision: Decision
    confidence: float
    reasoning: str
    factors: dict = field(default_factory=dict)
    alternatives: list = field(default_factory=list)


# ============================================================================
# Tripartite dataclasses (for TripartiteSynchronizer)
# ============================================================================

@dataclass
class TriHardwareProfile:
    """Hardware profile for TripartiteSynchronizer."""
    compute_power: float = 0.5       # 0-1
    gpu_available: bool = False
    memory_gb: float = 8.0
    battery_level: Optional[float] = None  # 0-1, None if plugged
    device_type: str = "desktop"     # desktop/laptop/server/edge/mobile


@dataclass
class TriApplicationProfile:
    """Application profile for TripartiteSynchronizer."""
    latency_requirement_ms: float = 100.0
    accuracy_requirement: float = 0.8
    safety_critical: bool = False
    scale: int = 10
    deterministic: bool = False


@dataclass
class TriUserProfile:
    """User profile for TripartiteSynchronizer."""
    wants_manual_control: bool = False
    wants_creativity: float = 0.3
    wants_consistency: float = 0.5
    tolerance_for_error: float = 0.5
    preference_override: Optional[str] = None


# ============================================================================
# Legacy Synchronizer (scoring-based)
# ============================================================================

class Synchronizer:
    """Scoring-based tripartite decision engine (legacy)."""

    def __init__(self):
        self._hardware = HardwareProfile.detect()

    def decide(self, hardware=None, application=None, user=None):
        hw = self._hardware
        if hardware:
            for k, v in hardware.items():
                if hasattr(hw, k):
                    setattr(hw, k, v)
        app = ApplicationProfile(**(application or {}))
        usr = UserProfile(**(user or {}))
        scores = {
            Decision.HARDCODE: self._score_hardcode(hw, app, usr),
            Decision.MODEL: self._score_model(hw, app, usr),
            Decision.HYBRID: self._score_hybrid(hw, app, usr),
            Decision.CACHED: self._score_cached(hw, app, usr),
        }
        best = max(scores, key=scores.get)
        confidence = scores[best]
        alternatives = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [(d, s) for d, s in alternatives if d != best]
        reasoning = self._explain(best, hw, app, usr)
        return SyncDecision(
            decision=best, confidence=confidence, reasoning=reasoning,
            factors={"hardware": hw, "application": app, "user": usr},
            alternatives=alternatives,
        )

    def _score_hardcode(self, hw, app, usr):
        score = 0.5
        score += app.safety * 0.3
        if app.latency_ms < 100:
            score += 0.2
        score += app.accuracy * 0.2
        if app.throughput_rps > 10:
            score += 0.1
        score -= app.creativity * 0.2
        if usr.prefer_speed:
            score += 0.1
        if hw.is_edge:
            score += 0.1
        return min(max(score, 0), 1)

    def _score_model(self, hw, app, usr):
        score = 0.3
        score += app.creativity * 0.3
        if app.latency_ms > 500:
            score += 0.1
        if hw.gpu:
            score += 0.2
        score -= app.safety * 0.1
        if usr.creative_output:
            score += 0.2
        if usr.prefer_quality:
            score += 0.1
        if app.accuracy < 0.3:
            score += 0.1
        return min(max(score, 0), 1)

    def _score_hybrid(self, hw, app, usr):
        score = 0.4
        score += 0.2 * (1 - abs(app.safety - 0.5) * 2)
        if 100 < app.latency_ms < 500:
            score += 0.1
        score += 0.1 * (1 - abs(app.creativity - 0.5) * 2)
        if hw.ram_gb >= 8:
            score += 0.1
        if not hw.gpu and not hw.is_edge:
            score += 0.1
        return min(max(score, 0), 1)

    def _score_cached(self, hw, app, usr):
        score = 0.2
        if hw.is_edge:
            score += 0.3
        if hw.battery_pct is not None and hw.battery_pct < 20:
            score += 0.2
        if usr.power_saving:
            score += 0.2
        if app.throughput_rps > 100:
            score += 0.2
        if hw.ram_gb < 4:
            score += 0.1
        return min(max(score, 0), 1)

    def _explain(self, decision, hw, app, usr):
        reasons = {
            Decision.HARDCODE: "Safety-critical or latency-sensitive — compiled path.",
            Decision.MODEL: "Creative or flexible-latency — LLM inference.",
            Decision.HYBRID: "Balanced — cached responses with model fallback.",
            Decision.CACHED: "Resource-constrained — pre-computed responses.",
        }
        parts = [f"Decision: {decision.value}", reasons[decision]]
        parts.append(f"Hardware: GPU={hw.gpu}, RAM={hw.ram_gb}GB, Edge={hw.is_edge}")
        parts.append(f"App: latency≤{app.latency_ms}ms, safety={app.safety:.1f}")
        return " | ".join(parts)


# ============================================================================
# TripartiteSynchronizer (rule-based, preferred)
# ============================================================================

class TripartiteSynchronizer:
    """
    Rule-based decision engine for the induction layer.

    Takes TriHardwareProfile, TriApplicationProfile, TriUserProfile
    and produces a concrete Decision.

    Priority ordering:
        1. User override (absolute)
        2. Safety-critical → HARDCODE
        3. Deterministic required → HARDCODE
        4. Ultra-low latency (<10ms) → HARDCODE or CACHED
        5. High creativity (>0.7) → MODEL or HYBRID
        6. High consistency (>0.8) → HARDCODE or CACHED
        7. Manual control → HARDCODE
        8. High accuracy (>0.9) + high scale (>100) → HYBRID
        9. Edge + low battery (<30%) → CACHED
       10. Default → HYBRID
    """

    def __init__(self):
        self.history: list = []

    def decide(
        self,
        hw: TriHardwareProfile,
        app: TriApplicationProfile,
        user: TriUserProfile,
    ) -> Decision:
        # 1. User override (absolute)
        if user.preference_override == "hardcode":
            return Decision.HARDCODE
        elif user.preference_override == "model":
            return Decision.MODEL

        # 2. Safety-critical → always hardcode
        if app.safety_critical:
            return Decision.HARDCODE

        # 3. Deterministic required → hardcode
        if app.deterministic:
            return Decision.HARDCODE

        # 4. Ultra-low latency → hardcode or cached
        if app.latency_requirement_ms < 10:
            if hw.device_type == "edge":
                return Decision.CACHED
            return Decision.HARDCODE

        # 5. High creativity wanted → model or hybrid
        if user.wants_creativity > 0.7:
            if hw.gpu_available or hw.compute_power > 0.5:
                return Decision.MODEL
            return Decision.HYBRID

        # 6. High consistency wanted → hardcode or cached
        if user.wants_consistency > 0.8:
            if hw.device_type == "edge":
                return Decision.CACHED
            return Decision.HARDCODE

        # 7. Manual control → hardcode
        if user.wants_manual_control:
            return Decision.HARDCODE

        # 8. High accuracy + high scale → hybrid
        if app.accuracy_requirement > 0.9 and app.scale > 100:
            return Decision.HYBRID

        # 9. Edge device with low battery → cached
        if hw.device_type == "edge" and hw.battery_level is not None and hw.battery_level < 0.3:
            return Decision.CACHED

        # 10. Default: hybrid
        return Decision.HYBRID

    def decide_batch(
        self,
        paths: list[tuple[TriHardwareProfile, TriApplicationProfile, TriUserProfile]],
    ) -> list[Decision]:
        return [self.decide(hw, app, user) for hw, app, user in paths]

    def record(self, hw, app, user, decision, outcome=None):
        """Record a decision for future learning."""
        self.history.append({
            "decision": decision,
            "hardware": hw,
            "application": app,
            "user": user,
            "outcome": outcome,
        })

    def decide_and_record(self, hw, app, user):
        """Decide and automatically record."""
        d = self.decide(hw, app, user)
        self.record(hw, app, user, d)
        return d
