"""Multi-language AST parser using tree-sitter.

Supports:
- Python (.py)
- Rust (.rs)
- C (.c, .h)
- C++ (.cpp, .hpp, .cc)
- JavaScript (.js)
- TypeScript (.ts)

For each file, extracts:
- Functions (name, signature, docstring, line numbers)
- Classes/structs/enums (name, methods, fields)
- Imports/modules
- Call graph (function calls within each function)
"""

import os
import re
from pathlib import Path
from typing import Optional

from .ingester import FunctionInfo, ClassInfo

# Language detection by extension
LANG_MAP = {
    ".py": "python",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".js": "javascript",
    ".ts": "typescript",
}

# Lazy-loaded parsers
_parsers = {}


def _get_ts():
    """Lazy import tree-sitter."""
    import tree_sitter
    return tree_sitter


def _get_language(lang_name: str):
    """Get a tree-sitter Language object for the given language."""
    ts = _get_ts()

    if lang_name == "rust":
        import tree_sitter_rust as tsrust
        return ts.Language(tsrust.language())
    elif lang_name == "c":
        import tree_sitter_c as tsc
        return ts.Language(tsc.language())
    elif lang_name == "cpp":
        import tree_sitter_cpp as tscpp
        return ts.Language(tscpp.language())
    elif lang_name == "python":
        import tree_sitter_python as tspython
        return ts.Language(tspython.language())
    else:
        raise ValueError(f"Unsupported language: {lang_name}")


def _get_parser(lang_name: str):
    """Get or create a parser for the given language."""
    if lang_name not in _parsers:
        ts = _get_ts()
        lang = _get_language(lang_name)
        parser = ts.Parser(lang)
        _parsers[lang_name] = parser
    return _parsers[lang_name]


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANG_MAP.get(ext)


def parse_file(file_path: str, module_prefix: str) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse a single file and extract functions and classes.

    Uses tree-sitter for all supported languages.
    Falls back to regex for unsupported languages.
    """
    lang = detect_language(file_path)
    if lang is None:
        return [], []

    try:
        with open(file_path, "rb") as f:
            source_bytes = f.read()
    except Exception:
        return [], []

    try:
        parser = _get_parser(lang)
        tree = parser.parse(source_bytes)
    except Exception:
        # Fallback to regex
        return _regex_parse_multi(source_bytes.decode("utf-8", errors="replace"), file_path, module_prefix, lang)

    source = source_bytes.decode("utf-8", errors="replace")
    source_lines = source.splitlines()

    if lang == "rust":
        return _parse_rust(tree, source_lines, file_path, module_prefix)
    elif lang == "c":
        return _parse_c(tree, source_lines, file_path, module_prefix)
    elif lang == "cpp":
        return _parse_cpp(tree, source_lines, file_path, module_prefix)
    elif lang == "python":
        return _parse_python_ts(tree, source_lines, file_path, module_prefix)
    else:
        return _regex_parse_multi(source, file_path, module_prefix, lang)


def _node_text(node, source_lines: list[str]) -> str:
    """Get text content of a node."""
    start = node.start_point[0]
    end = node.end_point[0]
    return "\n".join(source_lines[start:end + 1])


def _node_line(node) -> int:
    """Get 1-indexed line number."""
    return node.start_point[0] + 1


def _node_end_line(node) -> int:
    """Get 1-indexed end line number."""
    return node.end_point[0] + 1


# ---- Rust Parser ----

def _parse_rust(tree, source_lines: list[str], file_path: str, module: str) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse Rust source using tree-sitter."""
    functions = []
    classes = []

    def visit(node, parent_name: Optional[str] = None):
        if node.type == "function_item":
            func = _extract_rust_function(node, source_lines, file_path, module, parent_name)
            if func:
                functions.append(func)

        elif node.type == "struct_item":
            cls = _extract_rust_struct(node, source_lines, file_path, module)
            if cls:
                classes.append(cls)
                # Visit children for impl methods
                parent_name = cls.name

        elif node.type == "enum_item":
            cls = _extract_rust_enum(node, source_lines, file_path, module)
            if cls:
                classes.append(cls)

        elif node.type == "impl_item":
            # Extract the type name being implemented
            impl_type = _get_impl_type(node)
            for child in node.children:
                if child.type == "function_item":
                    func = _extract_rust_function(child, source_lines, file_path, module, impl_type)
                    if func:
                        functions.append(func)
            # Still visit deeper children
            for child in node.children:
                if child.type != "function_item":
                    visit(child, parent_name)
            return

        for child in node.children:
            visit(child, parent_name)

    visit(tree.root_node)
    return functions, classes


def _get_impl_type(node) -> Optional[str]:
    """Get the type name from an impl block."""
    for child in node.children:
        if child.type == "type_identifier":
            return child.text.decode("utf-8")
        elif child.type == "generic_type":
            for gc in child.children:
                if gc.type == "type_identifier":
                    return gc.text.decode("utf-8")
    return None


def _extract_rust_function(node, source_lines, file_path, module, parent_name=None) -> Optional[FunctionInfo]:
    """Extract a Rust function definition."""
    try:
        name = None
        params = []
        return_type = None
        is_pub = False
        is_async = False

        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8")
            elif child.type == "parameters":
                params = _extract_rust_params(child)
            elif child.type == "type_identifier":
                return_type = child.text.decode("utf-8")
            elif child.type == "visibility_modifier":
                is_pub = True
            elif child.type == "async":
                is_async = True
            elif child.type in ("block", "expression"):
                # Return type might be before block
                pass

        # Look for return type after -> 
        full_text = node.text.decode("utf-8")
        ret_match = re.search(r'\)\s*(?:->\s*([^{]+))?', full_text)
        if ret_match and ret_match.group(1):
            return_type = ret_match.group(1).strip()

        if name is None:
            return None

        prefix = f"{parent_name}::" if parent_name else ""
        sig = f"fn {prefix}{name}({', '.join(p for p in params)})"
        if return_type:
            sig += f" -> {return_type}"

        # Extract calls from body
        calls = _extract_calls(node, source_lines)

        # Docstring: look for doc comments above
        docstring = _extract_rust_docstring(node, source_lines)

        source_code = _node_text(node, source_lines)

        return FunctionInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            signature=sig,
            docstring=docstring,
            source_code=source_code,
            arg_names=params,
            return_type=return_type,
            calls=calls,
        )
    except Exception:
        return None


def _extract_rust_params(params_node) -> list[str]:
    """Extract parameter names from Rust function parameters."""
    params = []
    for child in params_node.children:
        if child.type == "parameter":
            for pc in child.children:
                if pc.type == "identifier":
                    param_text = pc.text.decode("utf-8")
                    # Try to get type too
                    params.append(param_text)
                    break
                elif child.type == "self_parameter":
                    params.append(child.text.decode("utf-8").strip())
        elif child.type == "self_parameter":
            params.append(child.text.decode("utf-8").strip())
    return params


def _extract_rust_docstring(node, source_lines) -> Optional[str]:
    """Extract Rust doc comments (/// or //! style)."""
    line_idx = node.start_point[0] - 1
    doc_lines = []
    while line_idx >= 0:
        line = source_lines[line_idx].strip()
        if line.startswith("///"):
            doc_lines.insert(0, line[3:].strip())
        elif line.startswith("//!"):
            doc_lines.insert(0, line[3:].strip())
        elif line.startswith("//"):
            line_idx -= 1
            continue
        elif line == "":
            line_idx -= 1
            continue
        else:
            break
        line_idx -= 1
    return "\n".join(doc_lines) if doc_lines else None


def _extract_rust_struct(node, source_lines, file_path, module) -> Optional[ClassInfo]:
    """Extract a Rust struct definition."""
    try:
        name = None
        fields = []

        for child in node.children:
            if child.type == "type_identifier":
                name = child.text.decode("utf-8")
            elif child.type == "field_declaration_list":
                for fc in child.children:
                    if fc.type == "field_declaration":
                        field_name = None
                        for fcc in fc.children:
                            if fcc.type == "field_identifier":
                                field_name = fcc.text.decode("utf-8")
                        if field_name:
                            fields.append(field_name)

        if name is None:
            return None

        return ClassInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            docstring=_extract_rust_docstring(node, source_lines),
            source_code=_node_text(node, source_lines),
            methods=[],
            bases=fields,  # Use bases field for struct fields
        )
    except Exception:
        return None


def _extract_rust_enum(node, source_lines, file_path, module) -> Optional[ClassInfo]:
    """Extract a Rust enum definition."""
    try:
        name = None
        variants = []

        for child in node.children:
            if child.type == "type_identifier":
                name = child.text.decode("utf-8")
            elif child.type == "enum_variant_list":
                for vc in child.children:
                    if vc.type == "enum_variant":
                        for vcc in vc.children:
                            if vcc.type == "identifier":
                                variants.append(vcc.text.decode("utf-8"))
                                break

        if name is None:
            return None

        return ClassInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            docstring=_extract_rust_docstring(node, source_lines),
            source_code=_node_text(node, source_lines),
            methods=[],
            bases=variants,  # Use bases field for enum variants
        )
    except Exception:
        return None


# ---- C Parser ----

def _parse_c(tree, source_lines, file_path, module) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse C source using tree-sitter."""
    functions = []
    classes = []  # C has structs, we store them as ClassInfo

    def visit(node):
        if node.type == "function_definition":
            func = _extract_c_function(node, source_lines, file_path, module)
            if func:
                functions.append(func)
        elif node.type == "struct_specifier":
            cls = _extract_c_struct(node, source_lines, file_path, module)
            if cls:
                classes.append(cls)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return functions, classes


def _extract_c_function(node, source_lines, file_path, module) -> Optional[FunctionInfo]:
    """Extract a C function definition."""
    try:
        name = None
        return_type = None
        params = []

        for child in node.children:
            if child.type == "function_declarator":
                for dc in child.children:
                    if dc.type == "identifier":
                        name = dc.text.decode("utf-8")
                    elif dc.type == "pointer_declarator":
                        for pc in dc.children:
                            if pc.type == "identifier":
                                name = pc.text.decode("utf-8")
                    elif dc.type == "parameter_list":
                        params = _extract_c_params(dc)
            elif child.type in ("primitive_type", "type_identifier"):
                return_type = child.text.decode("utf-8")
            elif child.type == "pointer_declarator" and return_type is None:
                pass

        # Get return type from text
        full_text = node.text.decode("utf-8")
        ret_match = re.match(r'^(\w+[\s*]*)\s*\w+\s*\(', full_text)
        if ret_match:
            return_type = ret_match.group(1).strip()

        if name is None:
            return None

        sig = f"{return_type or 'void'} {name}({', '.join(params)})"
        calls = _extract_calls(node, source_lines)

        return FunctionInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            signature=sig,
            docstring=_extract_c_comment(node, source_lines),
            source_code=_node_text(node, source_lines),
            arg_names=params,
            return_type=return_type,
            calls=calls,
        )
    except Exception:
        return None


def _extract_c_params(params_node) -> list[str]:
    """Extract C parameter names."""
    params = []
    for child in params_node.children:
        if child.type == "parameter_declaration":
            text = child.text.decode("utf-8").strip()
            # Simplify: just use the whole text
            params.append(text)
    return params


def _extract_c_comment(node, source_lines) -> Optional[str]:
    """Extract C-style comment above a node."""
    line_idx = node.start_point[0] - 1
    doc_lines = []
    while line_idx >= 0:
        line = source_lines[line_idx].strip()
        if line.startswith("//"):
            doc_lines.insert(0, line[2:].strip())
            line_idx -= 1
        elif line.startswith("*/") or line.endswith("*/"):
            # Multi-line comment
            doc_lines.insert(0, line.strip("*/").strip())
            line_idx -= 1
            while line_idx >= 0:
                line = source_lines[line_idx].strip()
                if line.startswith("/*"):
                    doc_lines.insert(0, line.lstrip("/*").strip())
                    break
                else:
                    doc_lines.insert(0, line.strip())
                    line_idx -= 1
            break
        elif line == "":
            line_idx -= 1
        else:
            break
    return "\n".join(doc_lines) if doc_lines else None


def _extract_c_struct(node, source_lines, file_path, module) -> Optional[ClassInfo]:
    """Extract a C struct definition."""
    try:
        name = None
        fields = []

        for child in node.children:
            if child.type == "type_identifier":
                name = child.text.decode("utf-8")
            elif child.type == "field_declaration_list":
                for fc in child.children:
                    if fc.type == "field_declaration":
                        fields.append(fc.text.decode("utf-8").strip())

        if name is None:
            return None

        return ClassInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            docstring=_extract_c_comment(node, source_lines),
            source_code=_node_text(node, source_lines),
            methods=[],
            bases=fields,
        )
    except Exception:
        return None


# ---- C++ Parser ----

def _parse_cpp(tree, source_lines, file_path, module) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse C++ source using tree-sitter."""
    functions = []
    classes = []

    def visit(node, parent_class=None):
        if node.type == "function_definition":
            func = _extract_cpp_function(node, source_lines, file_path, module, parent_class)
            if func:
                functions.append(func)
        elif node.type == "class_specifier":
            cls_name = None
            body = None
            for child in node.children:
                if child.type == "type_identifier":
                    cls_name = child.text.decode("utf-8")
                elif child.type == "field_declaration_list":
                    body = child
            if cls_name and body:
                methods = []
                fields = []
                for fc in body.children:
                    if fc.type == "function_definition":
                        methods.append(fc.child_by_field_name("declarator").text.decode("utf-8") if fc.child_by_field_name("declarator") else "unknown")
                    elif fc.type == "declaration":
                        fields.append(fc.text.decode("utf-8").strip())
                    elif fc.type == "field_declaration":
                        fields.append(fc.text.decode("utf-8").strip())

                classes.append(ClassInfo(
                    name=cls_name,
                    module=module,
                    file_path=file_path,
                    line_start=_node_line(node),
                    line_end=_node_end_line(node),
                    docstring=_extract_c_comment(node, source_lines),
                    source_code=_node_text(node, source_lines),
                    methods=methods,
                    bases=fields,
                ))
                # Visit method bodies
                for fc in body.children:
                    visit(fc, parent_class=cls_name)
                return
        elif node.type == "struct_specifier":
            cls = _extract_c_struct(node, source_lines, file_path, module)
            if cls:
                classes.append(cls)

        for child in node.children:
            visit(child, parent_class)

    visit(tree.root_node)
    return functions, classes


def _extract_cpp_function(node, source_lines, file_path, module, parent_class=None) -> Optional[FunctionInfo]:
    """Extract a C++ function definition."""
    try:
        # Get declarator for name and params
        declarator = None
        return_type = None

        for child in node.children:
            if child.type in ("primitive_type", "type_identifier"):
                return_type = child.text.decode("utf-8")
            elif child.type in ("function_declarator", "reference_declarator", "pointer_declarator"):
                declarator = child

        if declarator is None:
            # Try to find function_declarator nested
            for child in node.children:
                for gc in child.children:
                    if gc.type == "function_declarator":
                        declarator = gc
                        break

        name = None
        params = []
        if declarator:
            name, params = _parse_cpp_declarator(declarator)

        if name is None:
            return None

        prefix = f"{parent_class}::" if parent_class else ""
        sig = f"{return_type or 'auto'} {prefix}{name}({', '.join(params)})"
        calls = _extract_calls(node, source_lines)

        return FunctionInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            signature=sig,
            docstring=_extract_c_comment(node, source_lines),
            source_code=_node_text(node, source_lines),
            arg_names=params,
            return_type=return_type,
            calls=calls,
        )
    except Exception:
        return None


def _parse_cpp_declarator(node) -> tuple[Optional[str], list[str]]:
    """Parse a C++ function declarator to get name and params."""
    name = None
    params = []
    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
        elif child.type == "qualified_identifier":
            for gc in child.children:
                if gc.type == "identifier":
                    name = gc.text.decode("utf-8")
        elif child.type == "field_identifier":
            name = child.text.decode("utf-8")
        elif child.type == "parameter_list":
            params = _extract_c_params(child)
        elif child.type in ("pointer_declarator", "reference_declarator"):
            n, p = _parse_cpp_declarator(child)
            if n:
                name = n
            params = p or params
    return name, params


# ---- Python (tree-sitter) ----

def _parse_python_ts(tree, source_lines, file_path, module) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse Python using tree-sitter as alternative to ast module."""
    functions = []
    classes = []

    def visit(node):
        if node.type == "function_definition":
            func = _extract_python_ts_function(node, source_lines, file_path, module)
            if func:
                functions.append(func)
        elif node.type == "class_definition":
            cls = _extract_python_ts_class(node, source_lines, file_path, module)
            if cls:
                classes.append(cls)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return functions, classes


def _extract_python_ts_function(node, source_lines, file_path, module) -> Optional[FunctionInfo]:
    try:
        name = None
        params = []
        return_type = None

        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8")
            elif child.type == "parameters":
                for pc in child.children:
                    if pc.type == "identifier":
                        params.append(pc.text.decode("utf-8"))
                    elif pc.type == "typed_parameter":
                        for tpc in pc.children:
                            if tpc.type == "identifier":
                                params.append(tpc.text.decode("utf-8"))
                                break
                    elif pc.type == "default_parameter":
                        for dpc in pc.children:
                            if dpc.type == "identifier":
                                params.append(dpc.text.decode("utf-8"))
                                break
            elif child.type == "type":
                return_type = child.text.decode("utf-8")

        if name is None:
            return None

        sig = f"def {name}({', '.join(params)})"
        if return_type:
            sig += f" -> {return_type}"

        # Docstring
        docstring = None
        body = node.child_by_field_name("body")
        if body:
            for bc in body.children:
                if bc.type == "expression_statement":
                    for ecc in bc.children:
                        if ecc.type == "string":
                            docstring = ecc.text.decode("utf-8").strip("\"'")

        calls = _extract_calls(node, source_lines)

        return FunctionInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            signature=sig,
            docstring=docstring,
            source_code=_node_text(node, source_lines),
            arg_names=params,
            return_type=return_type,
            calls=calls,
        )
    except Exception:
        return None


def _extract_python_ts_class(node, source_lines, file_path, module) -> Optional[ClassInfo]:
    try:
        name = None
        bases = []
        methods = []

        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8")
            elif child.type == "argument_list":
                for ac in child.children:
                    if ac.type in ("identifier", "attribute"):
                        bases.append(ac.text.decode("utf-8"))
            elif child.type == "block":
                for bc in child.children:
                    if bc.type == "function_definition":
                        for fc in bc.children:
                            if fc.type == "identifier":
                                methods.append(fc.text.decode("utf-8"))
                                break

        if name is None:
            return None

        return ClassInfo(
            name=name,
            module=module,
            file_path=file_path,
            line_start=_node_line(node),
            line_end=_node_end_line(node),
            docstring=None,
            source_code=_node_text(node, source_lines),
            bases=bases,
            methods=methods,
        )
    except Exception:
        return None


# ---- Call extraction (generic) ----

def _extract_calls(node, source_lines) -> list[str]:
    """Extract function call names from a node's tree."""
    calls = set()

    def visit_calls(n):
        # Check for call expressions across different languages
        if n.type in ("call_expression",):
            children = list(n.children)
            if children:
                callee = children[0]
                if callee.type == "identifier":
                    calls.add(callee.text.decode("utf-8"))
                elif callee.type == "field_expression" or callee.type == "attribute":
                    # e.g., foo.bar() - get the bar part
                    for cc in callee.children:
                        if cc.type in ("field_identifier", "identifier"):
                            calls.add(cc.text.decode("utf-8"))
                elif callee.type == "selector_expression":
                    for cc in callee.children:
                        if cc.type == "field_identifier":
                            calls.add(cc.text.decode("utf-8"))
                else:
                    calls.add(callee.text.decode("utf-8"))

        for child in n.children:
            visit_calls(child)

    visit_calls(node)
    return list(calls)


# ---- Regex fallback ----

def _regex_parse_multi(source: str, file_path: str, module: str, lang: str) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Regex-based fallback parser."""
    functions = []
    classes = []

    if lang in ("rust",):
        pattern = re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]', re.MULTILINE)
        for m in pattern.finditer(source):
            name = m.group(1)
            functions.append(FunctionInfo(
                name=name,
                module=module,
                file_path=file_path,
                line_start=source[:m.start()].count("\n") + 1,
                line_end=source[:m.start()].count("\n") + 1,
                signature=m.group(0).strip(),
                docstring=None,
                source_code=m.group(0),
            ))
    elif lang in ("c", "cpp"):
        pattern = re.compile(r'^\s*(?:(?:static|inline|virtual|extern|const)\s+)*(\w[\w\s*]*?)\s+(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE)
        for m in pattern.finditer(source):
            ret = m.group(1).strip()
            name = m.group(2).strip()
            params = m.group(3).strip()
            functions.append(FunctionInfo(
                name=name,
                module=module,
                file_path=file_path,
                line_start=source[:m.start()].count("\n") + 1,
                line_end=source[:m.start()].count("\n") + 1,
                signature=f"{ret} {name}({params})",
                docstring=None,
                source_code=m.group(0),
            ))

    return functions, classes
