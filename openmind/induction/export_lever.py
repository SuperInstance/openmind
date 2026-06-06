"""Export hardcoded decisions as lever-runner skill packs.

When the synchronizer decides a function should be HARDCODE,
we can export it as a lever-runner command:
  intent: "function_name with args"
  command: "python -m module.function arg1 arg2"
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from openmind.induction.synchronizer import Decision


def export_lever_pack(
    function_name: str,
    module_path: str,
    args: Optional[List[str]] = None,
    description: str = "",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a single hardcoded function as a lever-runner command entry.

    Parameters
    ----------
    function_name : str
        Name of the function to hardcode.
    module_path : str
        Dotted module path, e.g. ``"mymodule.core"``.
    args : list[str] | None
        Positional args forwarded to the command.
    description : str
        Human-readable intent description.
    db_path : str | None
        Path to lever-runner's ``commands.db``.  If provided the entry
        is inserted directly; otherwise only the dict is returned.

    Returns
    -------
    dict
        The lever-runner command record that was (or would be) inserted.
    """
    args = args or []
    command = " ".join(
        ["python", "-m", module_path, function_name] + args
    )

    record: Dict[str, Any] = {
        "intent": f"{function_name} with {' '.join(args) if args else 'no args'}",
        "command": command,
        "description": description,
        "decision": Decision.HARDCODE.value,
    }

    if db_path is not None:
        db = Path(db_path).expanduser()
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        conn.execute(
            """\
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                command TEXT NOT NULL,
                description TEXT DEFAULT '',
                decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO commands (intent, command, description, decision) "
            "VALUES (?, ?, ?, ?)",
            (record["intent"], record["command"], record["description"], record["decision"]),
        )
        conn.commit()
        conn.close()

    return record


def export_lever_pack_batch(
    decisions: List[Dict[str, Any]],
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Export multiple hardcoded decisions in one shot.

    Parameters
    ----------
    decisions : list[dict]
        Each dict must contain at least ``function_name`` and ``module_path``.
        Optional keys: ``args``, ``description``.
    db_path : str | None
        Path to lever-runner's ``commands.db``.

    Returns
    -------
    list[dict]
        The exported records.
    """
    results: List[Dict[str, Any]] = []
    for entry in decisions:
        rec = export_lever_pack(
            function_name=entry["function_name"],
            module_path=entry["module_path"],
            args=entry.get("args"),
            description=entry.get("description", ""),
            db_path=db_path,
        )
        results.append(rec)
    return results
