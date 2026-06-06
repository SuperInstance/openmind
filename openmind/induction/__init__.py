"""Induction engine for openmind.

Ingests GitHub repos and builds living, iterating vector models that enable
both induction (learning via vectors) and deduction (analytic reasoning).

Quick start:
    from openmind.induction import ingest, VectorBuilder
    result = ingest("https://github.com/user/repo")
    builder = VectorBuilder()
    builder.build_all(result)
"""

from openmind.induction.ingester import ingest, ingest_repo, IngestResult, FunctionInfo, ClassInfo
from openmind.induction.vectors import VectorBuilder, DualVector
from openmind.induction.synchronizer import (
    Synchronizer, SyncDecision, Decision, HardwareProfile,
    TripartiteSynchronizer, TriHardwareProfile, TriApplicationProfile, TriUserProfile,
)
from openmind.induction.spreader import Spreader
from openmind.induction.export_lever import export_lever_pack, export_lever_pack_batch
from openmind.induction.export_nail import export_nail, export_nail_batch
from openmind.induction.hardware import probe_hardware, HardwareCapabilities

__all__ = [
    "ingest", "ingest_repo", "IngestResult", "FunctionInfo", "ClassInfo",
    "VectorBuilder", "DualVector",
    "Synchronizer", "SyncDecision", "Decision", "HardwareProfile",
    "TripartiteSynchronizer", "TriHardwareProfile", "TriApplicationProfile", "TriUserProfile",
    "Spreader",
    "export_lever_pack", "export_lever_pack_batch",
    "export_nail", "export_nail_batch",
    "probe_hardware", "HardwareCapabilities",
]
