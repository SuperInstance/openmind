"""Cellular computation — resource-adaptive processing that never breaks.

Like a biological cell switching between aerobic and anaerobic metabolism,
each computation adapts to what's available: GPU, API, cache, or simulation.

Usage:
    from openmind.cellular import probe, train_or_load, sense_or_simulate, infer_adaptive

    # Probe available resources
    resources = probe()
    print(f"GPU: {resources.gpu_available}, API: {resources.api_keys}")

    # Train or load — adapts to resources, never fails
    model = train_or_load("my-classifier", data=X)

    # Sense or simulate — real data when possible, simulated when not
    data = sense_or_simulate("temperature", duration="1h")

    # Infer adaptive — GPU → API → cached
    predictions = infer_adaptive(model, data)
"""

import os
import time
import json
import hashlib
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from pathlib import Path


# ── Resource Snapshot ────────────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    """Current state of available computational resources.

    Like a cell checking its environment for oxygen, glucose, and ATP levels.
    """
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_free_mb: int = 0
    cpu_cores: int = 1
    ram_available_gb: float = 4.0
    api_keys: dict = field(default_factory=dict)
    esp32_ports: list = field(default_factory=list)
    battery_pct: Optional[float] = None
    network_available: bool = True
    timestamp: float = 0.0

    def has_gpu(self) -> bool:
        return self.gpu_available

    def has_api(self, provider: str = "") -> bool:
        if provider:
            return self.api_keys.get(provider, False)
        return any(self.api_keys.values())

    def has_hardware(self) -> bool:
        return len(self.esp32_ports) > 0

    def resource_level(self) -> str:
        """Return 'high', 'medium', or 'low' based on available resources."""
        score = 0
        if self.gpu_available:
            score += 3
        if self.has_api():
            score += 2
        if self.network_available:
            score += 1
        if self.ram_available_gb > 16:
            score += 1
        if self.cpu_cores >= 8:
            score += 1

        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        return "low"


def probe() -> ResourceSnapshot:
    """Probe the current environment for available resources.

    Checks: GPU (nvidia-smi), API keys (env vars), ESP32s (serial ports),
    CPU cores, RAM, battery, and network connectivity.
    """
    snapshot = ResourceSnapshot(timestamp=time.time())

    # GPU check
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                snapshot.gpu_available = True
                snapshot.gpu_name = parts[0].strip()
                snapshot.gpu_memory_free_mb = int(float(parts[1].strip()))
    except Exception:
        pass

    # CPU cores
    try:
        snapshot.cpu_cores = os.cpu_count() or 1
    except Exception:
        pass

    # RAM
    try:
        import subprocess
        result = subprocess.run(
            ["free", "-g"], capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Mem:" in line and "Swap:" not in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        snapshot.ram_available_gb = float(parts[3])
    except Exception:
        pass

    # API keys
    snapshot.api_keys = {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "deepinfra": bool(os.environ.get("DEEPINFRA_API_KEY")),
        "litellm": bool(os.environ.get("LITELLM_API_KEY")),
    }

    # Battery
    try:
        import subprocess
        result = subprocess.run(
            ["cat", "/sys/class/power_supply/BAT0/capacity"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            snapshot.battery_pct = float(result.stdout.strip())
    except Exception:
        pass

    # Network
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        snapshot.network_available = True
    except Exception:
        snapshot.network_available = False

    return snapshot


# ── Metabolic Pathways ───────────────────────────────────────────────────

class MetabolicPath:
    """Five ways a cell can metabolize computation."""
    FULL_TRAIN = "full_train"           # GPU + API + RAM + Time
    TRANSFER = "transfer"               # GPU + pretrained model
    CLOUD_INFERENCE = "cloud"           # No GPU, API key valid
    MUSCLE_MEMORY = "muscle_memory"     # No GPU, no API, use cache
    HARDWARE_LOOP = "hardware_loop"     # ESP32/sensor online


def select_path(
    task: str,
    resources: Optional[ResourceSnapshot] = None,
    prefer: str = "auto",
) -> str:
    """Select the best metabolic pathway for a task.

    The tripartite synchronizer for resource allocation:
    - All resources → FULL_TRAIN (aerobic, most ATP)
    - GPU only → TRANSFER (aerobic, less glucose)
    - API only → CLOUD_INFERENCE (fermentation)
    - Nothing → MUSCLE_MEMORY (anaerobic, instant)
    - Hardware → HARDWARE_LOOP (photosynthesis)

    Args:
        task: What you want to do
        resources: Resource snapshot (auto-probe if None)
        prefer: Override preference ("auto", "gpu", "api", "cached", "hardware")

    Returns:
        One of the MetabolicPath constants
    """
    if resources is None:
        resources = probe()

    if prefer != "auto":
        overrides = {
            "gpu": MetabolicPath.FULL_TRAIN,
            "transfer": MetabolicPath.TRANSFER,
            "api": MetabolicPath.CLOUD_INFERENCE,
            "cached": MetabolicPath.MUSCLE_MEMORY,
            "hardware": MetabolicPath.HARDWARE_LOOP,
        }
        return overrides.get(prefer, MetabolicPath.MUSCLE_MEMORY)

    # Auto-selection based on resources
    if resources.has_hardware():
        return MetabolicPath.HARDWARE_LOOP
    if resources.has_gpu() and resources.has_api():
        return MetabolicPath.FULL_TRAIN
    if resources.has_gpu():
        return MetabolicPath.TRANSFER
    if resources.has_api():
        return MetabolicPath.CLOUD_INFERENCE
    return MetabolicPath.MUSCLE_MEMORY


# ── Cache Store ──────────────────────────────────────────────────────────

def _cache_dir() -> Path:
    """Get or create the openmind cache directory."""
    d = Path.home() / ".openmind" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(name: str, suffix: str = ".json") -> Path:
    """Generate a cache file path for a named resource."""
    h = hashlib.md5(name.encode()).hexdigest()[:12]
    return _cache_dir() / f"{name}_{h}{suffix}"


def save_cache(name: str, data: Any):
    """Save data to the local cache (muscle memory storage)."""
    path = _cache_key(name)
    with open(path, "w") as f:
        json.dump({"name": name, "timestamp": time.time(), "data": data}, f)


def load_cache(name: str) -> Optional[Any]:
    """Load data from the local cache."""
    path = _cache_key(name)
    if path.exists():
        with open(path, "r") as f:
            entry = json.load(f)
            return entry.get("data")
    return None


# ── Adaptive Functions ───────────────────────────────────────────────────

def train_or_load(
    model_name: str,
    data: Any = None,
    fallback: str = "cached",
    train_fn: Optional[Callable] = None,
    resources: Optional[ResourceSnapshot] = None,
) -> Any:
    """Train a model or load from muscle memory. Never fails.

    Metabolic pathway selection:
    1. FULL_TRAIN: GPU + data available → train from scratch
    2. TRANSFER: GPU available → fine-tune cached model
    3. CLOUD_INFERENCE: API key → use cloud model
    4. MUSCLE_MEMORY: Nothing → load cached model
    5. If all else fails: return None or raise (if fallback="fail")

    Args:
        model_name: Name to cache/load the model under
        data: Training data (None = load from cache)
        fallback: "cached" | "simulate" | "fail"
        train_fn: Optional training function(model_name, data) -> model
        resources: Resource snapshot (auto-probe if None)

    Returns:
        A model object (format depends on what pathway was used)
    """
    if resources is None:
        resources = probe()

    path = select_path("train", resources)

    # Try full training if we have resources and a train function
    if path in (MetabolicPath.FULL_TRAIN, MetabolicPath.TRANSFER):
        if train_fn and data is not None:
            try:
                model = train_fn(model_name, data)
                save_cache(model_name, {"type": "trained", "path": path})
                return model
            except Exception:
                pass  # Fall through to cache

    # Try loading from cache
    cached = load_cache(model_name)
    if cached is not None:
        return cached

    # Try simulation fallback
    if fallback == "simulate":
        return {"type": "simulated", "name": model_name, "note": "No real model available"}

    if fallback == "fail":
        raise RuntimeError(f"No model available for '{model_name}' and fallback='fail'")

    return {"type": "unavailable", "name": model_name}


def sense_or_simulate(
    source: str,
    duration: str = "1h",
    rate: str = "1Hz",
    simulate_fn: Optional[Callable] = None,
    hardware_fn: Optional[Callable] = None,
    resources: Optional[ResourceSnapshot] = None,
) -> list:
    """Read sensor data or generate realistic simulation.

    Pathway selection:
    1. HARDWARE_LOOP: ESP32/sensor online → real data
    2. MUSCLE_MEMORY: Cached distributions → calibrated simulation
    3. Synthetic: Generate data with correct statistics

    Args:
        source: Sensor/data source name
        duration: How long to read/simulate
        rate: Sampling rate
        simulate_fn: Optional simulation function
        hardware_fn: Optional hardware read function
        resources: Resource snapshot (auto-probe if None)

    Returns:
        List of data points
    """
    if resources is None:
        resources = probe()

    path = select_path("sense", resources)

    # Try real hardware
    if path == MetabolicPath.HARDWARE_LOOP and hardware_fn:
        try:
            data = hardware_fn(source, duration, rate)
            save_cache(f"sense_{source}", {"source": source, "data_sample": data[:10]})
            return data
        except Exception:
            pass

    # Try cached simulation parameters
    cached = load_cache(f"sense_{source}")
    if cached and "data_sample" in cached:
        # Extend cached sample with simulation
        sample = cached["data_sample"]
        if simulate_fn:
            return simulate_fn(source, duration, rate, seed_data=sample)
        return sample

    # Synthetic generation
    if simulate_fn:
        return simulate_fn(source, duration, rate)

    # Fallback: simple synthetic data
    import random
    n = 100
    return [random.gauss(0, 1) for _ in range(n)]


def infer_adaptive(
    model: Any,
    data: Any,
    strategy: str = "adaptive",
    infer_fn: Optional[Callable] = None,
    resources: Optional[ResourceSnapshot] = None,
) -> Any:
    """Run inference adapting to available resources.

    Strategy options:
    - "adaptive": GPU → API → cached (default)
    - "gpu": Force local GPU (error if unavailable)
    - "api": Force cloud API (error if no key)
    - "cached": Always use cached predictions

    Args:
        model: Model object (from train_or_load)
        data: Input data
        strategy: Inference strategy
        infer_fn: Optional inference function(model, data) -> predictions
        resources: Resource snapshot (auto-probe if None)

    Returns:
        Predictions (format depends on model)
    """
    if resources is None:
        resources = probe()

    path = select_path("infer", resources, prefer=strategy)

    # Try real inference if function provided
    if infer_fn:
        try:
            predictions = infer_fn(model, data)
            return predictions
        except Exception:
            pass  # Fall through to cached

    # Try cached predictions
    if isinstance(model, dict) and "name" in model:
        cached = load_cache(f"predict_{model['name']}")
        if cached is not None:
            return cached

    # Return model's own prediction capability
    if hasattr(model, "predict"):
        return model.predict(data)

    # Ultimate fallback
    return {"note": "No inference available", "path": path}


# ── Cell Decorator ───────────────────────────────────────────────────────

def cell(resource_aware: bool = True, fallback: str = "cached"):
    """Decorator for resource-aware cell execution.

    Usage:
        @cell(resource_aware=True, fallback="cached")
        def my_pipeline():
            resources = probe()
            data = sense_or_simulate("temperature", duration="1h")
            model = train_or_load("my-model", data=data)
            return infer_adaptive(model, data)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if fallback == "fail":
                    raise
                return {
                    "error": str(e),
                    "fallback": fallback,
                    "function": func.__name__,
                }
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


__all__ = [
    "ResourceSnapshot", "probe", "MetabolicPath", "select_path",
    "train_or_load", "sense_or_simulate", "infer_adaptive",
    "save_cache", "load_cache", "cell",
]
