"""
Pre-built profiles for common scenarios using TripartiteSynchronizer dataclasses.

Each scenario is a (TriHardwareProfile, TriApplicationProfile, TriUserProfile) tuple.
"""

from openmind.induction.synchronizer import TriHardwareProfile, TriApplicationProfile, TriUserProfile

# ---------------------------------------------------------------------------
# Hardware Profiles
# ---------------------------------------------------------------------------

HW_GAMING_PC = TriHardwareProfile(
    compute_power=0.95,
    gpu_available=True,
    memory_gb=32,
    battery_level=None,  # desktop, always plugged
    device_type="desktop",
)

HW_DEV_LAPTOP = TriHardwareProfile(
    compute_power=0.6,
    gpu_available=False,
    memory_gb=16,
    battery_level=0.7,
    device_type="laptop",
)

HW_RASPBERRY_PI = TriHardwareProfile(
    compute_power=0.1,
    gpu_available=False,
    memory_gb=4,
    battery_level=0.55,
    device_type="edge",
)

HW_CAR_ECU = TriHardwareProfile(
    compute_power=0.25,
    gpu_available=False,
    memory_gb=2,
    battery_level=None,  # vehicle power
    device_type="edge",
)

HW_CLOUD_SERVER = TriHardwareProfile(
    compute_power=0.9,
    gpu_available=True,
    memory_gb=128,
    battery_level=None,
    device_type="server",
)

# ---------------------------------------------------------------------------
# Application Profiles
# ---------------------------------------------------------------------------

APP_GAME_RENDERING = TriApplicationProfile(
    latency_requirement_ms=16,  # ~60fps
    accuracy_requirement=0.85,
    safety_critical=False,
    scale=60,
    deterministic=False,
)

APP_DEV_TOOLING = TriApplicationProfile(
    latency_requirement_ms=200,
    accuracy_requirement=0.7,
    safety_critical=False,
    scale=1,
    deterministic=True,
)

APP_IOT_SENSOR = TriApplicationProfile(
    latency_requirement_ms=50,
    accuracy_requirement=0.9,
    safety_critical=False,
    scale=500,
    deterministic=True,
)

APP_CAR_BRAKES = TriApplicationProfile(
    latency_requirement_ms=2,
    accuracy_requirement=1.0,
    safety_critical=True,
    scale=1000,
    deterministic=True,
)

APP_NPC_BEHAVIOR = TriApplicationProfile(
    latency_requirement_ms=30,
    accuracy_requirement=0.6,
    safety_critical=False,
    scale=60,
    deterministic=False,
)

APP_TERMINAL = TriApplicationProfile(
    latency_requirement_ms=50,
    accuracy_requirement=0.95,
    safety_critical=False,
    scale=10,
    deterministic=True,
)

# ---------------------------------------------------------------------------
# User Profiles
# ---------------------------------------------------------------------------

USER_GAMER = TriUserProfile(
    wants_manual_control=False,
    wants_creativity=0.8,
    wants_consistency=0.3,
    tolerance_for_error=0.6,
)

USER_DEVELOPER = TriUserProfile(
    wants_manual_control=True,
    wants_creativity=0.3,
    wants_consistency=0.9,
    tolerance_for_error=0.2,
)

USER_IOT_OPERATOR = TriUserProfile(
    wants_manual_control=False,
    wants_creativity=0.1,
    wants_consistency=0.95,
    tolerance_for_error=0.1,
)

USER_DRIVER = TriUserProfile(
    wants_manual_control=False,
    wants_creativity=0.0,
    wants_consistency=1.0,
    tolerance_for_error=0.0,
)

USER_PLAYER = TriUserProfile(
    wants_manual_control=False,
    wants_creativity=0.75,
    wants_consistency=0.4,
    tolerance_for_error=0.7,
)

USER_TERMINAL_USER = TriUserProfile(
    wants_manual_control=True,
    wants_creativity=0.1,
    wants_consistency=0.95,
    tolerance_for_error=0.1,
)

# ---------------------------------------------------------------------------
# Composite Scenarios: (hw, app, user) tuples
# ---------------------------------------------------------------------------

GAMING_PC = (HW_GAMING_PC, APP_GAME_RENDERING, USER_GAMER)
DEV_LAPTOP = (HW_DEV_LAPTOP, APP_DEV_TOOLING, USER_DEVELOPER)
RASPBERRY_PI = (HW_RASPBERRY_PI, APP_IOT_SENSOR, USER_IOT_OPERATOR)
CAR_BRAKE_SYSTEM = (HW_CAR_ECU, APP_CAR_BRAKES, USER_DRIVER)
NPC_BEHAVIOR = (HW_CLOUD_SERVER, APP_NPC_BEHAVIOR, USER_PLAYER)
TERMINAL_COMMANDS = (HW_DEV_LAPTOP, APP_TERMINAL, USER_TERMINAL_USER)
