"""OpenMind — Agent muscle memory + cellular computation.

Three ways to use:

1. Python API:
    import openmind
    result = openmind.ingest("./my-project")
    mm = openmind.MuscleMemory.build(result)
    mm.flex("do_something")

2. CLI:
    openmind ingest ./my-project
    openmind flex ./my-project "function_name"
    openmind recall ./my-project "search_term"

3. Jupyter:
    %load_ext openmind.jupyter
    %%openmind analyze ./my-project

4. Cellular (resource-adaptive):
    from openmind.cellular import probe, train_or_load, sense_or_simulate
    resources = probe()
    model = train_or_load("classifier", data=X)  # adapts to GPU/API/cache
"""

__version__ = "0.2.0"

# ── Core ingestion ──
from openmind.induction.ingester import ingest, ingest_repo, IngestResult, FunctionInfo, ClassInfo

# ── Vector building ──
from openmind.induction.vectors import VectorBuilder, DualVector

# ── Tripartite synchronizer ──
from openmind.induction.synchronizer import (
    Synchronizer, SyncDecision, Decision, HardwareProfile,
    TripartiteSynchronizer, TriHardwareProfile, TriApplicationProfile, TriUserProfile,
)

# ── Spreader ──
from openmind.induction.spreader import Spreader

# ── Exports ──
from openmind.induction.export_lever import export_lever_pack, export_lever_pack_batch
from openmind.induction.export_nail import export_nail, export_nail_batch

# ── Hardware probe ──
from openmind.induction.hardware import probe_hardware, HardwareCapabilities

# ── Muscle memory (the guitarist's hand) ──
from openmind.muscle import MuscleMemory, Chord, Reflex

# ── Flex API (one-shot convenience) ──
from openmind.flex import quick_flex, load_and_flex

# ── Cellular computation (resource-adaptive) ──
from openmind.cellular import (
    ResourceSnapshot, probe as probe_resources,
    MetabolicPath, select_path,
    train_or_load, sense_or_simulate, infer_adaptive,
    save_cache, load_cache, cell,
)

__all__ = [
    "__version__",
    # Ingestion
    "ingest", "ingest_repo", "IngestResult", "FunctionInfo", "ClassInfo",
    # Vectors
    "VectorBuilder", "DualVector",
    # Synchronizer
    "Synchronizer", "SyncDecision", "Decision", "HardwareProfile",
    "TripartiteSynchronizer", "TriHardwareProfile", "TriApplicationProfile", "TriUserProfile",
    # Spreader
    "Spreader",
    # Exports
    "export_lever_pack", "export_lever_pack_batch",
    "export_nail", "export_nail_batch",
    # Hardware
    "probe_hardware", "HardwareCapabilities",
    # Muscle memory
    "MuscleMemory", "Chord", "Reflex",
    # Flex
    "quick_flex", "load_and_flex",
    # Cellular
    "ResourceSnapshot", "probe_resources", "MetabolicPath", "select_path",
    "train_or_load", "sense_or_simulate", "infer_adaptive",
    "save_cache", "load_cache", "cell",
]
