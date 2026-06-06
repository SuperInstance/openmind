"""Tests for openmind tripartite synchronizer."""

import pytest

from openmind import (
    Decision, Synchronizer, SyncDecision,
    TripartiteSynchronizer,
    TriHardwareProfile, TriApplicationProfile, TriUserProfile,
)


# ============================================================================
# Legacy Synchronizer Tests
# ============================================================================

def test_decision_enum():
    """Test Decision enum values."""
    assert Decision.HARDCODE.value == "hardcode"
    assert Decision.MODEL.value == "model"
    assert Decision.HYBRID.value == "hybrid"
    assert Decision.CACHED.value == "cached"


def test_legacy_synchronizer_basic():
    """Test basic legacy Synchronizer decision."""
    sync = Synchronizer()
    result = sync.decide()
    assert isinstance(result, SyncDecision)
    assert isinstance(result.decision, Decision)
    assert 0 <= result.confidence <= 1
    assert len(result.reasoning) > 0


def test_legacy_synchronizer_safety_critical():
    """Test safety-critical app gets HARDCODE."""
    sync = Synchronizer()
    result = sync.decide(application={"safety": 0.99, "latency_ms": 10, "creativity": 0.0})
    assert result.decision == Decision.HARDCODE


def test_legacy_synchronizer_creative():
    """Test creative app gets MODEL."""
    sync = Synchronizer()
    result = sync.decide(application={"creativity": 0.9, "safety": 0.1})
    # With GPU, this should be MODEL; without, could be either MODEL or HYBRID
    assert result.decision in (Decision.MODEL, Decision.HYBRID)


def test_legacy_synchronizer_with_hardware():
    """Test Synchronizer with hardware overrides."""
    sync = Synchronizer()
    result = sync.decide(hardware={"is_edge": True, "battery_pct": 10})
    assert isinstance(result, SyncDecision)


def test_sync_decision_fields():
    """Test SyncDecision dataclass."""
    sd = SyncDecision(
        decision=Decision.HYBRID,
        confidence=0.7,
        reasoning="test",
        factors={"a": 1},
        alternatives=[(Decision.MODEL, 0.5)],
    )
    assert sd.decision == Decision.HYBRID
    assert sd.confidence == 0.7


# ============================================================================
# TripartiteSynchronizer Tests
# ============================================================================

def test_tripartite_basic():
    """Test basic tripartite decision."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile()
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert isinstance(d, Decision)


def test_tripartite_safety_critical():
    """Test safety-critical → HARDCODE."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile(safety_critical=True)
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_deterministic():
    """Test deterministic → HARDCODE."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile(deterministic=True)
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_low_latency():
    """Test ultra-low latency → HARDCODE."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(device_type="desktop")
    app = TriApplicationProfile(latency_requirement_ms=5)
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_low_latency_edge():
    """Test ultra-low latency on edge → CACHED."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(device_type="edge")
    app = TriApplicationProfile(latency_requirement_ms=5)
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.CACHED


def test_tripartite_high_creativity():
    """Test high creativity → MODEL or HYBRID."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(gpu_available=True, compute_power=0.8)
    app = TriApplicationProfile()
    user = TriUserProfile(wants_creativity=0.8)
    d = sync.decide(hw, app, user)
    assert d == Decision.MODEL


def test_tripartite_high_creativity_no_gpu():
    """Test high creativity without GPU → HYBRID."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(gpu_available=False, compute_power=0.3)
    app = TriApplicationProfile()
    user = TriUserProfile(wants_creativity=0.8)
    d = sync.decide(hw, app, user)
    assert d == Decision.HYBRID


def test_tripartite_high_consistency():
    """Test high consistency → HARDCODE."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(device_type="desktop")
    app = TriApplicationProfile()
    user = TriUserProfile(wants_consistency=0.9)
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_high_consistency_edge():
    """Test high consistency on edge → CACHED."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(device_type="edge")
    app = TriApplicationProfile()
    user = TriUserProfile(wants_consistency=0.9)
    d = sync.decide(hw, app, user)
    assert d == Decision.CACHED


def test_tripartite_manual_control():
    """Test manual control → HARDCODE."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile()
    user = TriUserProfile(wants_manual_control=True)
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_high_accuracy_high_scale():
    """Test high accuracy + high scale → HYBRID."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile(accuracy_requirement=0.95, scale=200)
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.HYBRID


def test_tripartite_edge_low_battery():
    """Test edge + low battery → CACHED."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile(device_type="edge", battery_level=0.1)
    app = TriApplicationProfile()
    user = TriUserProfile()
    d = sync.decide(hw, app, user)
    assert d == Decision.CACHED


def test_tripartite_default_hybrid():
    """Test default decision is HYBRID."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile()  # defaults
    user = TriUserProfile()  # defaults
    d = sync.decide(hw, app, user)
    assert d == Decision.HYBRID


def test_tripartite_user_override_hardcode():
    """Test user override hardcode."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile(safety_critical=False)
    user = TriUserProfile(preference_override="hardcode")
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE


def test_tripartite_user_override_model():
    """Test user override model."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile(safety_critical=False)
    user = TriUserProfile(preference_override="model")
    d = sync.decide(hw, app, user)
    assert d == Decision.MODEL


def test_tripartite_batch():
    """Test batch decisions."""
    sync = TripartiteSynchronizer()
    paths = [
        (TriHardwareProfile(), TriApplicationProfile(safety_critical=True), TriUserProfile()),
        (TriHardwareProfile(), TriApplicationProfile(), TriUserProfile()),
        (TriHardwareProfile(device_type="edge"), TriApplicationProfile(latency_requirement_ms=5), TriUserProfile()),
    ]
    decisions = sync.decide_batch(paths)
    assert len(decisions) == 3
    assert decisions[0] == Decision.HARDCODE
    assert decisions[1] == Decision.HYBRID  # default
    assert decisions[2] == Decision.CACHED


def test_tripartite_record():
    """Test recording decisions."""
    sync = TripartiteSynchronizer()
    hw = TriHardwareProfile()
    app = TriApplicationProfile()
    user = TriUserProfile()
    d = sync.decide_and_record(hw, app, user)
    assert len(sync.history) == 1
    assert sync.history[0]["decision"] == d


# ============================================================================
# Profiles Tests
# ============================================================================

def test_builtin_profiles():
    """Test built-in profiles produce valid decisions."""
    from openmind.induction.profiles import (
        GAMING_PC, DEV_LAPTOP, RASPBERRY_PI, CAR_BRAKE_SYSTEM,
        NPC_BEHAVIOR, TERMINAL_COMMANDS,
    )

    sync = TripartiteSynchronizer()

    for name, (hw, app, user) in [
        ("Gaming PC", GAMING_PC),
        ("Dev Laptop", DEV_LAPTOP),
        ("Raspberry Pi", RASPBERRY_PI),
        ("Car Brakes", CAR_BRAKE_SYSTEM),
        ("NPC Behavior", NPC_BEHAVIOR),
        ("Terminal", TERMINAL_COMMANDS),
    ]:
        d = sync.decide(hw, app, user)
        assert isinstance(d, Decision), f"{name} returned invalid decision"


def test_car_brakes_hardcode():
    """Test car brake system is always HARDCODE."""
    from openmind.induction.profiles import CAR_BRAKE_SYSTEM
    sync = TripartiteSynchronizer()
    hw, app, user = CAR_BRAKE_SYSTEM
    d = sync.decide(hw, app, user)
    assert d == Decision.HARDCODE
