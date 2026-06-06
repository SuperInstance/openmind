"""Export cached decisions as pincherOS .nail files.

A .nail file is a JSON manifest describing a pre-computed function result
that pincherOS can serve from cache without invoking a model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openmind.induction.synchronizer import Decision


def _nail_header(function_name: str) -> Dict[str, Any]:
    return {
        "format": "pincherOS-nail",
        "version": 1,
        "function": function_name,
        "decision": Decision.CACHED.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def export_nail(
    function_name: str,
    cached_output: Any,
    description: str = "",
    export_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a single cached decision as a .nail file.

    Parameters
    ----------
    function_name : str
        Name of the cached function.
    cached_output : Any
        The pre-computed result (must be JSON-serializable).
    description : str
        Human-readable description.
    export_dir : str | None
        Directory to write the .nail file into.  If *None*, only
        the dict is returned (no file is written).

    Returns
    -------
    dict
        The nail manifest that was (or would be) written.
    """
    manifest: Dict[str, Any] = {
        **_nail_header(function_name),
        "description": description,
        "output": cached_output,
    }

    if export_dir is not None:
        out = Path(export_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        nail_path = out / f"{function_name}.nail"
        nail_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def export_nail_batch(
    entries: List[Dict[str, Any]],
    export_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Export multiple cached decisions in one shot.

    Parameters
    ----------
    entries : list[dict]
        Each dict must contain ``function_name`` and ``cached_output``.
        Optional keys: ``description``.
    export_dir : str | None
        Directory to write .nail files into.

    Returns
    -------
    list[dict]
        The exported manifests.
    """
    results: List[Dict[str, Any]] = []
    for entry in entries:
        manifest = export_nail(
            function_name=entry["function_name"],
            cached_output=entry["cached_output"],
            description=entry.get("description", ""),
            export_dir=export_dir,
        )
        results.append(manifest)
    return results
