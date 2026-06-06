"""Ingest a GitHub repo into structured knowledge.

Steps:
1. Clone repo
2. Parse AST for all functions/classes
3. Extract signatures, docstrings, type hints
4. Identify test files and test coverage
5. Build initial embeddings for all code chunks
6. Store in a vector database (LanceDB for local)

Usage:
    from openmind import ingest
    result = ingest("https://github.com/user/repo")
    # result has: functions, classes, tests, embeddings, graph
"""

import ast
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import git
except ImportError:
    git = None

# Multi-language support (lazy-loaded)
HAS_TREE_SITTER = None  # None = not checked yet
_ts_parse_file = None
_detect_language = None


def _ensure_tree_sitter():
    global HAS_TREE_SITTER, _ts_parse_file, _detect_language
    if HAS_TREE_SITTER is None:
        try:
            from openmind.induction.parser import parse_file, detect_language
            _ts_parse_file = parse_file
            _detect_language = detect_language
            HAS_TREE_SITTER = True
        except (ImportError, Exception):
            HAS_TREE_SITTER = False
    return HAS_TREE_SITTER


@dataclass
class FunctionInfo:
    """Extracted information about a single function."""

    name: str
    module: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str]
    source_code: str
    decorators: list[str] = field(default_factory=list)
    arg_names: list[str] = field(default_factory=list)
    arg_types: dict[str, str] = field(default_factory=dict)
    return_type: Optional[str] = None
    calls: list[str] = field(default_factory=list)  # functions this calls
    called_by: list[str] = field(default_factory=list)  # filled in later
    has_tests: bool = False
    embedding: Optional[list[float]] = None


@dataclass
class ClassInfo:
    """Extracted information about a class."""

    name: str
    module: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    source_code: str
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None


@dataclass
class IngestResult:
    """Result of ingesting a repo."""

    repo_url: str
    local_path: str
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    test_files: list[str]
    file_structure: dict
    call_graph: dict[str, list[str]]  # caller -> [callees]
    stats: dict = field(default_factory=dict)


def _clone_repo(repo_url: str, target_dir: Optional[str] = None) -> str:
    """Clone a git repo to a local directory."""
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="openmind-ingest-")

    if os.path.exists(target_dir) and os.listdir(target_dir):
        # Already cloned
        return target_dir

    if git is not None:
        git.Repo.clone_from(repo_url, target_dir)
    else:
        # Fallback to CLI git
        os.system(f"git clone --depth 1 {repo_url} {target_dir}")

    return target_dir


def _is_test_file(file_path: str) -> bool:
    """Heuristic: is this a test file?"""
    name = os.path.basename(file_path)
    return name.startswith("test_") or name.endswith("_test.py") or "test" in name.lower()


def _parse_python_file(file_path: str, module_prefix: str) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse a single Python file and extract functions and classes."""
    functions = []
    classes = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception:
        return functions, classes

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex extraction for files with syntax issues
        return _regex_parse(source, file_path, module_prefix)

    source_lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func = _extract_function(node, source_lines, file_path, module_prefix)
            if func:
                functions.append(func)
        elif isinstance(node, ast.ClassDef):
            cls = _extract_class(node, source_lines, file_path, module_prefix)
            if cls:
                classes.append(cls)

    return functions, classes


def _extract_function(node: ast.FunctionDef, source_lines: list[str], file_path: str, module: str) -> Optional[FunctionInfo]:
    """Extract FunctionInfo from an AST FunctionDef node."""
    try:
        # Get source code span
        start = node.lineno
        end = node.end_lineno or start
        source_code = "\n".join(source_lines[start - 1 : end])

        # Signature
        args = []
        arg_types = {}
        for arg in node.args.args:
            arg_name = arg.arg
            args.append(arg_name)
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)

        # Return type
        return_type = ast.unparse(node.returns) if node.returns else None

        # Decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Calls within this function
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)

        # Docstring
        docstring = ast.get_docstring(node)

        # Build signature with type annotations
        sig_parts = []
        for a in args:
            if a in arg_types:
                sig_parts.append(f"{a}: {arg_types[a]}")
            else:
                sig_parts.append(a)
        signature = f"def {node.name}({', '.join(sig_parts)})"
        if return_type:
            signature += f" -> {return_type}"

        return FunctionInfo(
            name=node.name,
            module=module,
            file_path=file_path,
            line_start=start,
            line_end=end,
            signature=signature,
            docstring=docstring,
            source_code=source_code,
            decorators=decorators,
            arg_names=args,
            arg_types=arg_types,
            return_type=return_type,
            calls=list(set(calls)),
        )
    except Exception:
        return None


def _extract_class(node: ast.ClassDef, source_lines: list[str], file_path: str, module: str) -> Optional[ClassInfo]:
    """Extract ClassInfo from an AST ClassDef node."""
    try:
        start = node.lineno
        end = node.end_lineno or start
        source_code = "\n".join(source_lines[start - 1 : end])

        bases = [ast.unparse(b) for b in node.bases]
        methods = [
            n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        return ClassInfo(
            name=node.name,
            module=module,
            file_path=file_path,
            line_start=start,
            line_end=end,
            docstring=ast.get_docstring(node),
            source_code=source_code,
            bases=bases,
            methods=methods,
        )
    except Exception:
        return None


def _regex_parse(source: str, file_path: str, module_prefix: str) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Fallback regex parser for files with syntax errors."""
    functions = []
    classes = []

    # Match function definitions
    func_pattern = re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
    for m in func_pattern.finditer(source):
        indent = len(m.group(1))
        if indent == 0 or indent == 4:  # top-level or class methods
            name = m.group(2)
            args_str = m.group(3)
            sig = f"def {name}({args_str})"
            functions.append(
                FunctionInfo(
                    name=name,
                    module=module_prefix,
                    file_path=file_path,
                    line_start=source[: m.start()].count("\n") + 1,
                    line_end=source[: m.start()].count("\n") + 1,
                    signature=sig,
                    docstring=None,
                    source_code=m.group(0),
                )
            )

    return functions, classes


def _build_file_structure(root: str) -> dict:
    """Build a tree representation of the file structure."""
    result = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and common non-code dirs
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in ("__pycache__", "node_modules", ".git", "venv", ".venv")
        ]
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            rel = ""
        result[rel] = sorted(filenames)
    return result


def _build_call_graph(functions: list[FunctionInfo]) -> dict[str, list[str]]:
    """Build a call graph: function_name -> [called_function_names]."""
    graph = {}
    for func in functions:
        qualified = f"{func.module}.{func.name}"
        graph[qualified] = func.calls
    return graph


def _resolve_callers(functions: list[FunctionInfo], call_graph: dict[str, list[str]]):
    """Populate the called_by field for each function."""
    name_map = {}
    for func in functions:
        qualified = f"{func.module}.{func.name}"
        name_map[func.name] = qualified
        name_map[qualified] = qualified

    for caller, callees in call_graph.items():
        for callee_name in callees:
            callee_qualified = name_map.get(callee_name)
            if callee_qualified:
                # Find the callee FunctionInfo and add caller
                for func in functions:
                    if f"{func.module}.{func.name}" == callee_qualified:
                        if caller not in func.called_by:
                            func.called_by.append(caller)


def _mark_tested_functions(functions: list[FunctionInfo], test_files: list[str], repo_path: str):
    """Mark functions that are referenced in test files."""
    # Simple heuristic: search for function names in test files
    test_content = ""
    for tf in test_files:
        full_path = os.path.join(repo_path, tf) if not os.path.isabs(tf) else tf
        try:
            with open(full_path, "r", errors="replace") as f:
                test_content += f.read().lower() + "\n"
        except Exception:
            pass

    for func in functions:
        # Check if the function name appears in test content in a call-like context
        if re.search(rf"\b{re.escape(func.name)}\s*\(", test_content):
            func.has_tests = True


def _supported_extensions():
    """File extensions we can parse."""
    if _ensure_tree_sitter():
        return {".py", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".js", ".ts"}
    return {".py"}


def ingest(repo_url: str, target_dir: Optional[str] = None, cleanup: bool = False) -> IngestResult:
    """Ingest a GitHub repo into structured knowledge.

    Args:
        repo_url: URL to the GitHub repository
        target_dir: Where to clone (temp dir if None)
        cleanup: Whether to remove the cloned repo after ingestion

    Returns:
        IngestResult with all extracted data
    """
    # Clone
    local_path = _clone_repo(repo_url, target_dir)
    return _ingest_local(local_path, repo_url=repo_url, cleanup=cleanup)


def ingest_repo(local_path: str, cleanup: bool = False) -> IngestResult:
    """Ingest a local repo directory into structured knowledge.

    Args:
        local_path: Path to the local repository
        cleanup: Whether to remove the repo after ingestion

    Returns:
        IngestResult with all extracted data
    """
    return _ingest_local(local_path, repo_url=local_path, cleanup=cleanup)


def _ingest_local(local_path: str, repo_url: str, cleanup: bool = False) -> IngestResult:
    """Core ingestion logic for a local directory."""
    all_functions = []
    all_classes = []
    test_files = []
    supported = _supported_extensions()

    # Track language stats
    lang_stats = {}

    for dirpath, dirnames, filenames in os.walk(local_path):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in ("__pycache__", "node_modules", ".git", "venv", ".venv",
                          "target", "build", "cmake-build-*", "out")
        ]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue

            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, local_path)

            # Module prefix from file path
            module = rel_path.replace(os.sep, ".")
            for suffix in (".py", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".js", ".ts"):
                module = module.removesuffix(suffix)
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")

            # Track language
            lang = _detect_language(file_path) if _ensure_tree_sitter() else ("python" if ext == ".py" else None)
            if lang:
                lang_stats[lang] = lang_stats.get(lang, 0) + 1

            if ext == ".py" and _is_test_file(rel_path):
                test_files.append(rel_path)
            elif ext == ".rs" and ("test" in rel_path.lower() or filename.startswith("test_")):
                test_files.append(rel_path)

            # Parse based on language
            if _ensure_tree_sitter():
                funcs, cls_list = _ts_parse_file(file_path, module)
            elif ext == ".py":
                funcs, cls_list = _parse_python_file(file_path, module)
            else:
                continue

            all_functions.extend(funcs)
            all_classes.extend(cls_list)

    # Build call graph
    call_graph = _build_call_graph(all_functions)
    _resolve_callers(all_functions, call_graph)

    # Mark tested functions
    _mark_tested_functions(all_functions, test_files, local_path)

    # File structure
    file_structure = _build_file_structure(local_path)

    result = IngestResult(
        repo_url=repo_url,
        local_path=local_path,
        functions=all_functions,
        classes=all_classes,
        test_files=test_files,
        file_structure=file_structure,
        call_graph=call_graph,
        stats={
            "total_functions": len(all_functions),
            "total_classes": len(all_classes),
            "test_files": len(test_files),
            "tested_functions": sum(1 for f in all_functions if f.has_tests),
            "python_files": sum(
                1
                for files in file_structure.values()
                for f in files
                if f.endswith(".py")
            ),
            "function_count": len(all_functions),
            "class_count": len(all_classes),
            "languages": lang_stats,
        },
    )

    if cleanup:
        shutil.rmtree(local_path, ignore_errors=True)

    return result
