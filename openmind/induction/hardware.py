"""Auto-detect hardware capabilities for the tripartite synchronizer.

Detects:
- GPU: torch.cuda, nvidia-smi, or absent
- RAM: psutil.virtual_memory()
- CPU cores: os.cpu_count()
- Device type: heuristic (edge if RAM < 4GB, mobile if ARM, etc.)
- Battery: psutil.sensors_battery() or None
"""

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Hardware profile produced by the probe."""

    gpu: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_mb: Optional[int] = None
    ram_gb: float = 0.0
    cpu_cores: int = 1
    arch: str = ""
    device_type: str = "desktop"  # desktop | edge | mobile | server
    battery_pct: Optional[float] = None
    battery_plugged: Optional[bool] = None

    def to_sync_profile(self):
        """Convert to a :class:`HardwareProfile` for the legacy Synchronizer."""
        from openmind.induction.synchronizer import HardwareProfile

        return HardwareProfile(
            gpu=self.gpu,
            gpu_name=self.gpu_name,
            ram_gb=self.ram_gb,
            battery_pct=self.battery_pct,
            cpu_count=self.cpu_cores,
        )


def _detect_gpu_torch() -> Optional[str]:
    """Try detecting GPU via torch.cuda."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _detect_gpu_nvidia_smi() -> Optional[str]:
    """Try detecting GPU via nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def _detect_gpu_vram() -> Optional[int]:
    """Try detecting GPU VRAM in MiB via nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return None


def _detect_ram() -> float:
    """Return total RAM in GB."""
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    # Fallback: read /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _detect_battery():
    """Return (percent, plugged) or (None, None)."""
    try:
        import psutil  # type: ignore

        bat = psutil.sensors_battery()
        if bat is not None:
            return bat.percent, bat.power_plugged
    except Exception:
        pass
    return None, None


def probe_hardware() -> HardwareCapabilities:
    """Run all probes and return a :class:`HardwareCapabilities` profile.

    Safe to call anywhere — failures are caught and result in sensible
    defaults (no GPU, unknown RAM, etc.).
    """
    gpu_name = _detect_gpu_torch() or _detect_gpu_nvidia_smi()
    gpu_vram = _detect_gpu_vram() if gpu_name else None
    ram_gb = _detect_ram()
    cpu_cores = os.cpu_count() or 1
    arch = platform.machine().lower()
    bat_pct, bat_plugged = _detect_battery()

    # Heuristic device type
    is_arm = arch in ("aarch64", "armv7l", "armv6l", "arm")
    if is_arm and ram_gb < 8:
        device_type = "mobile"
    elif ram_gb < 4:
        device_type = "edge"
    elif cpu_cores >= 16 and ram_gb >= 32:
        device_type = "server"
    else:
        device_type = "desktop"

    cap = HardwareCapabilities(
        gpu=gpu_name is not None,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        arch=arch,
        device_type=device_type,
        battery_pct=bat_pct,
        battery_plugged=bat_plugged,
    )

    logger.info("Hardware probe: %s", cap)
    return cap
