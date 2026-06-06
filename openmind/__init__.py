"""OpenMind — Code induction engine.

Works standalone, as CLI, or in Jupyter.

Usage:
    import openmind

    # Ingest a repo
    result = openmind.ingest("https://github.com/user/repo")

    # Or a local path
    result = openmind.ingest_repo("./my-project")

    # Access the data
    print(f"Found {len(result.functions)} functions")
    print(f"Call graph has {len(result.call_graph)} nodes")

    # Build vectors
    builder = openmind.VectorBuilder()
    builder.build_all(result)
    matches = builder.search_input("handle authentication")

    # Tripartite decisions
    sync = openmind.TripartiteSynchronizer()
    hw = openmind.TriHardwareProfile()
    app = openmind.TriApplicationProfile()
    user = openmind.TriUserProfile()
    decision = sync.decide(hw, app, user)

    # Hardware probe
    hw_caps = openmind.probe_hardware()
"""

__version__ = "0.1.0"

# Core ingestion
from openmind.induction.ingester import ingest, ingest_repo, IngestResult, FunctionInfo, ClassInfo

# Vector building
from openmind.induction.vectors import VectorBuilder, DualVector

# Tripartite synchronizer
from openmind.induction.synchronizer import (
    Synchronizer,
    SyncDecision,
    Decision,
    HardwareProfile,
    TripartiteSynchronizer,
    TriHardwareProfile,
    TriApplicationProfile,
    TriUserProfile,
)

# Spreader
from openmind.induction.spreader import Spreader

# Exports
from openmind.induction.export_lever import export_lever_pack, export_lever_pack_batch
from openmind.induction.export_nail import export_nail, export_nail_batch

# Hardware
from openmind.induction.hardware import probe_hardware, HardwareCapabilities

__all__ = [
    # Version
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
]
