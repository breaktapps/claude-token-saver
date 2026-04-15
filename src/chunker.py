"""AST-based code chunking for claude-token-saver.

Uses tree-sitter for Python, TypeScript/JavaScript, and Dart parsing.
Extracts semantic chunks with metadata: class outlines, methods, functions.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser

from .config import Config

logger = logging.getLogger("cts.chunker")

# Language extension mapping
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".dart": "dart",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}

# Default ignore patterns (always excluded)
DEFAULT_IGNORE = [
    "__pycache__",
    "node_modules",
    ".git",
    "*.pyc",
    "*.lock",
    "*.g.dart",
    ".dart_tool",
    "build",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "*.egg-info",
]

# Default max chunk lines
_DEFAULT_MAX_CHUNK_LINES = 150


def detect_language(file_path: str | Path) -> str | None:
    """Detect programming language from file extension.

    Returns language name or None if unsupported.
    """
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def scan_files(
    repo_path: Path,
    extra_ignore: list[str] | None = None,
) -> list[Path]:
    """Scan repository for supported source files.

    Applies .gitignore patterns, default ignores, and extra_ignore config.
    Returns list of absolute paths to supported source files.
    """
    repo_path = repo_path.resolve()
    ignore_patterns = list(DEFAULT_IGNORE)
    if extra_ignore:
        ignore_patterns.extend(extra_ignore)

    # Read .gitignore if exists
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        for line in gitignore_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.append(line.rstrip("/"))

    result: list[Path] = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if detect_language(path) is None:
            continue
        if _should_ignore(path, repo_path, ignore_patterns):
            continue
        result.append(path)

    return sorted(result)


def _should_ignore(path: Path, repo_root: Path, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    rel = str(path.relative_to(repo_root))
    parts = rel.split("/")

    for pattern in patterns:
        # Check against full relative path
        if fnmatch.fnmatch(rel, pattern):
            return True
        # Check against filename
        if fnmatch.fnmatch(path.name, pattern):
            return True
        # Check against each path component (for directory patterns)
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def parse(
    file_path: Path,
    language: str,
    *,
    max_chunk_lines: int = _DEFAULT_MAX_CHUNK_LINES,
) -> list[dict]:
    """Parse a source file into semantic chunks.

    Args:
        file_path: Path to the source file.
        language: Language name (python, typescript, dart, javascript).
        max_chunk_lines: Maximum lines per chunk before splitting.

    Returns:
        List of chunk dicts with: file_path, chunk_type, name, line_start,
        line_end, content, language, parent_name, calls_json, outline_json,
        imports_json.
    """
    source = file_path.read_text()
    if not source.strip():
        return []

    rel_path = str(file_path)
    logger.debug("Parsing %s (%s)", file_path, language)

    if language == "dart":
        return _parse_dart_tree_sitter(source, rel_path, max_chunk_lines)
    return _parse_tree_sitter(source, rel_path, language, max_chunk_lines)


# ──────────────────────────────────────────────────────────────
# tree-sitter parsing (Python, TypeScript, JavaScript)
# ──────────────────────────────────────────────────────────────

_PARSERS: dict[str, Parser] = {}


def _get_parser(language: str) -> Parser:
    """Get or create a tree-sitter parser for the given language."""
    if language in _PARSERS:
        return _PARSERS[language]

    lang_obj = _get_language(language)
    parser = Parser(lang_obj)
    _PARSERS[language] = parser
    return parser


def _get_language(language: str) -> Language:
    """Load tree-sitter Language for the given language name."""
    if language == "python":
        import tree_sitter_python as tsp
        return Language(tsp.language())
    elif language == "typescript":
        import tree_sitter_typescript as tst
        return Language(tst.language_typescript())
    elif language == "javascript":
        import tree_sitter_javascript as tsj
        return Language(tsj.language())
    elif language == "dart":
        try:
            import tree_sitter_dart as tsd
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-dart is not installed. "
                "Build and install from: https://github.com/UserNobody14/tree-sitter-dart"
            ) from exc
        return Language(tsd.language())
    else:
        raise ValueError(f"No tree-sitter grammar for language: {language}")


# Node type mappings per language
_CLASS_TYPES = {
    "python": ["class_definition"],
    "typescript": ["class_declaration"],
    "javascript": ["class_declaration"],
}

_FUNCTION_TYPES = {
    "python": ["function_definition"],
    "typescript": ["function_declaration"],
    "javascript": ["function_declaration"],
}

# Method types within class bodies (distinct from top-level functions)
_METHOD_TYPES = {
    "python": ["function_definition"],
    "typescript": ["method_definition"],
    "javascript": ["method_definition"],
}

_CALL_TYPES = {
    "python": ["call"],
    "typescript": ["call_expression"],
    "javascript": ["call_expression"],
}

_EXPORT_TYPES = {
    "typescript": ["export_statement"],
    "javascript": ["export_statement"],
}

_INTERFACE_TYPES = {
    "typescript": ["interface_declaration"],
}

# Import node types per language
_IMPORT_TYPES = {
    "python": ["import_statement", "import_from_statement"],
    "typescript": ["import_statement"],
    "javascript": ["import_statement"],
}


def _extract_imports(root: Node, language: str, source: str) -> list[str]:
    """Extract import strings from the root AST node."""
    import_types = _IMPORT_TYPES.get(language, [])
    imports: list[str] = []
    for child in root.children:
        if child.type in import_types:
            imports.append(child.text.decode().strip())
    return imports


def _parse_tree_sitter(
    source: str,
    file_path: str,
    language: str,
    max_chunk_lines: int,
) -> list[dict]:
    """Parse source using tree-sitter."""
    parser = _get_parser(language)
    tree = parser.parse(source.encode())
    root = tree.root_node
    lines = source.split("\n")

    # Extract imports from the source
    imports = _extract_imports(root, language, source)

    chunks: list[dict] = []
    class_types = _CLASS_TYPES.get(language, [])
    func_types = _FUNCTION_TYPES.get(language, [])
    export_types = _EXPORT_TYPES.get(language, [])
    interface_types = _INTERFACE_TYPES.get(language, [])

    for child in root.children:
        if child.type in class_types:
            chunks.extend(
                _extract_class_chunks(child, file_path, language, lines, max_chunk_lines, imports)
            )
        elif child.type in func_types:
            chunks.extend(
                _extract_function_chunk(child, file_path, language, lines, None, max_chunk_lines, imports)
            )
        elif child.type in export_types:
            # Unwrap export statement — handle export default anonymous (M8)
            default_name = _export_default_name(child, file_path)
            for inner in child.children:
                if inner.type in func_types:
                    # M8: use filename-based name if function is anonymous
                    name_override = default_name if _get_node_name(inner, language) == "<anonymous>" else None
                    chunks.extend(
                        _extract_function_chunk(
                            inner, file_path, language, lines, None, max_chunk_lines, imports,
                            name_override=name_override,
                        )
                    )
                elif inner.type in class_types:
                    chunks.extend(
                        _extract_class_chunks(inner, file_path, language, lines, max_chunk_lines, imports)
                    )
                elif inner.type in interface_types:
                    chunks.extend(
                        _extract_interface_chunk(inner, file_path, language, lines, imports)
                    )
        elif child.type in interface_types:
            chunks.extend(
                _extract_interface_chunk(child, file_path, language, lines, imports)
            )

    return chunks


def _export_default_name(export_node: Node, file_path: str) -> str:
    """Return a fallback name for 'export default' anonymous functions/classes.

    Uses the stem of the file name (e.g. 'index' from 'index.ts').
    """
    for child in export_node.children:
        if child.type == "identifier":
            return child.text.decode()
    return Path(file_path).stem


def _extract_class_chunks(
    node: Node,
    file_path: str,
    language: str,
    lines: list[str],
    max_chunk_lines: int,
    imports: list[str] | None = None,
) -> list[dict]:
    """Extract class outline + method chunks from a class node."""
    chunks = []
    class_name = _get_node_name(node, language)
    method_types = _METHOD_TYPES.get(language, _FUNCTION_TYPES.get(language, []))

    # Find the body node
    body = _find_body(node, language)
    methods = []
    member_signatures = []

    if body:
        for child in body.children:
            if child.type in method_types:
                method_name = _get_node_name(child, language)
                sig = _get_signature(child, lines)
                member_signatures.append(sig)
                methods.append(child)

    # Class outline chunk
    outline_content = f"class {class_name}:\n"
    for sig in member_signatures:
        outline_content += f"  {sig}\n"

    chunks.append({
        "file_path": file_path,
        "chunk_type": "class",
        "name": class_name,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "content": outline_content.strip(),
        "language": language,
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": json.dumps(member_signatures),
        "imports_json": json.dumps(imports or []),
    })

    # Method chunks
    for method_node in methods:
        chunks.extend(
            _extract_function_chunk(
                method_node, file_path, language, lines, class_name, max_chunk_lines, imports
            )
        )

    return chunks


def _extract_function_chunk(
    node: Node,
    file_path: str,
    language: str,
    lines: list[str],
    parent_name: str | None,
    max_chunk_lines: int,
    imports: list[str] | None = None,
    *,
    name_override: str | None = None,
) -> list[dict]:
    """Extract function/method chunk(s), splitting if too large."""
    name = name_override or _get_node_name(node, language)
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    content = _get_node_text(node, lines)
    calls = _extract_calls(node, language)
    chunk_type = "method" if parent_name else "function"
    num_lines = end_line - start_line + 1

    if num_lines <= max_chunk_lines:
        return [{
            "file_path": file_path,
            "chunk_type": chunk_type,
            "name": name,
            "line_start": start_line,
            "line_end": end_line,
            "content": content,
            "language": language,
            "parent_name": parent_name or "",
            "calls_json": json.dumps(calls),
            "outline_json": "[]",
            "imports_json": json.dumps(imports or []),
        }]

    # Split large method at first-level AST nodes
    return _split_large_chunk(
        node, file_path, language, lines, name, parent_name, chunk_type, calls, max_chunk_lines, imports
    )


def _split_large_chunk(
    node: Node,
    file_path: str,
    language: str,
    lines: list[str],
    name: str,
    parent_name: str | None,
    chunk_type: str,
    calls: list[str],
    max_chunk_lines: int,
    imports: list[str] | None = None,
) -> list[dict]:
    """Split a large function/method into sub-chunks at first-level AST nodes."""
    body = _find_body(node, language)
    if not body or not body.children:
        # Can't split -- return as one chunk
        content = _get_node_text(node, lines)
        return [{
            "file_path": file_path,
            "chunk_type": chunk_type,
            "name": name,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "content": content,
            "language": language,
            "parent_name": parent_name or "",
            "calls_json": json.dumps(calls),
            "outline_json": "[]",
            "imports_json": json.dumps(imports or []),
        }]

    # Get the signature (everything before the body)
    sig_lines = lines[node.start_point[0]: body.start_point[0]]
    sig_text = "\n".join(sig_lines).rstrip()

    # Group body children into sub-chunks
    chunks = []
    current_children: list[Node] = []
    current_line_count = 0
    part = 1

    for child in body.children:
        if not child.is_named:
            continue
        child_lines = child.end_point[0] - child.start_point[0] + 1
        if current_line_count + child_lines > max_chunk_lines and current_children:
            # Emit current group
            chunks.append(_make_sub_chunk(
                current_children, lines, file_path, chunk_type, name, part,
                parent_name, language, calls, imports
            ))
            part += 1
            current_children = []
            current_line_count = 0

        current_children.append(child)
        current_line_count += child_lines

    if current_children:
        chunks.append(_make_sub_chunk(
            current_children, lines, file_path, chunk_type, name, part,
            parent_name, language, calls, imports
        ))

    return chunks


def _make_sub_chunk(
    children: list[Node],
    lines: list[str],
    file_path: str,
    chunk_type: str,
    name: str,
    part: int,
    parent_name: str | None,
    language: str,
    calls: list[str],
    imports: list[str] | None = None,
) -> dict:
    """Create a sub-chunk dict from a group of AST nodes."""
    start_line = children[0].start_point[0] + 1
    end_line = children[-1].end_point[0] + 1
    content = "\n".join(lines[start_line - 1: end_line])

    return {
        "file_path": file_path,
        "chunk_type": chunk_type,
        "name": f"{name}__part{part}",
        "line_start": start_line,
        "line_end": end_line,
        "content": content,
        "language": language,
        "parent_name": parent_name or "",
        "calls_json": json.dumps(calls),
        "outline_json": "[]",
        "imports_json": json.dumps(imports or []),
    }


def _extract_interface_chunk(
    node: Node,
    file_path: str,
    language: str,
    lines: list[str],
    imports: list[str] | None = None,
) -> list[dict]:
    """Extract an interface as a class-like outline chunk."""
    name = _get_node_name(node, language)
    content = _get_node_text(node, lines)
    members = []
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.is_named:
                members.append(_get_node_text(child, lines).strip())

    return [{
        "file_path": file_path,
        "chunk_type": "class",
        "name": name,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "content": content,
        "language": language,
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": json.dumps(members),
        "imports_json": json.dumps(imports or []),
    }]


def _get_node_name(node: Node, language: str) -> str:
    """Extract the name of a class/function/method node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()
    # Fallback: first identifier or property_identifier child
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return child.text.decode()
    return "<anonymous>"


def _get_node_text(node: Node, lines: list[str]) -> str:
    """Get the source text for a node."""
    start = node.start_point[0]
    end = node.end_point[0]
    return "\n".join(lines[start: end + 1])


def _get_signature(node: Node, lines: list[str]) -> str:
    """Get the first line (signature) of a function/method node."""
    line = lines[node.start_point[0]].strip()
    return line


def _find_body(node: Node, language: str) -> Node | None:
    """Find the body node of a class or function."""
    body = node.child_by_field_name("body")
    if body:
        return body
    # TypeScript/JS class uses "class_body"
    for child in node.children:
        if child.type in ("block", "class_body", "statement_block"):
            return child
    return None


def _extract_calls(node: Node, language: str) -> list[str]:
    """Extract function call names from an AST node."""
    call_types = _CALL_TYPES.get(language, [])
    calls: list[str] = []
    _walk_calls(node, call_types, calls)
    return sorted(set(calls))


def _walk_calls(node: Node, call_types: list[str], calls: list[str]) -> None:
    """Recursively walk AST to find call expressions."""
    if node.type in call_types:
        name = _call_name(node)
        if name:
            calls.append(name)
    for child in node.children:
        _walk_calls(child, call_types, calls)


def _call_name(node: Node) -> str | None:
    """Extract the function name from a call node."""
    # Python: call -> function (identifier or attribute)
    func = node.child_by_field_name("function")
    if func:
        if func.type == "identifier":
            return func.text.decode()
        elif func.type == "attribute":
            attr = func.child_by_field_name("attribute")
            if attr:
                return attr.text.decode()
    # TypeScript/JS: call_expression -> function
    if node.children:
        first = node.children[0]
        if first.type == "identifier":
            return first.text.decode()
        elif first.type == "member_expression":
            prop = first.child_by_field_name("property")
            if prop:
                return prop.text.decode()
    return None


# ──────────────────────────────────────────────────────────────
# Dart tree-sitter parsing
# ──────────────────────────────────────────────────────────────

# Dart AST node types that define class-like containers
_DART_CLASS_LIKE = {"class_definition", "mixin_declaration", "extension_declaration"}

# Maps class-like node type to its body node type
_DART_BODY_TYPE = {
    "class_definition": "class_body",
    "mixin_declaration": "class_body",
    "extension_declaration": "extension_body",
}


def _dart_node_name(node: Node) -> str:
    """Extract name identifier from a Dart class/mixin/extension/function node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode()
    return "<anonymous>"


def _dart_method_name(sig_inner: Node) -> str:
    """Extract method name from a method_signature inner node.

    Handles: function_signature, getter_signature, setter_signature,
    factory_constructor_signature, constructor_signature.
    """
    sig_type = sig_inner.type

    if sig_type == "getter_signature":
        found_get = False
        for child in sig_inner.children:
            if child.type == "get":
                found_get = True
            elif found_get and child.type == "identifier":
                return child.text.decode()

    if sig_type == "setter_signature":
        found_set = False
        for child in sig_inner.children:
            if child.type == "set":
                found_set = True
            elif found_set and child.type == "identifier":
                return child.text.decode()

    if sig_type in ("factory_constructor_signature", "constructor_signature"):
        # Named constructor: ClassName.name() — return the second identifier
        identifiers = [c for c in sig_inner.children if c.type == "identifier"]
        if len(identifiers) >= 2:
            return identifiers[1].text.decode()
        if identifiers:
            return identifiers[0].text.decode()

    # function_signature: find the identifier (skip return type tokens)
    for child in sig_inner.children:
        if child.type == "identifier":
            return child.text.decode()

    return "<anonymous>"


def _dart_extract_imports(root: Node) -> list[str]:
    """Extract import strings from a Dart program root node."""
    imports: list[str] = []
    for child in root.children:
        if child.type == "import_or_export":
            imports.append(child.text.decode().strip())
    return imports


def _dart_extract_calls(node: Node) -> list[str]:
    """Extract function call names from a Dart AST node.

    In Dart's AST, direct calls appear as: identifier + selector(argument_part).
    We only record the direct call identifier (bare function name), not the receiver.
    """
    calls: set[str] = set()
    _dart_walk_calls(node, calls)
    return sorted(calls)


def _dart_walk_calls(node: Node, calls: set[str]) -> None:
    """Recursively walk a Dart AST collecting direct function call names."""
    children = node.children
    for i, child in enumerate(children):
        if child.type == "selector":
            has_arg_part = any(cc.type == "argument_part" for cc in child.children)
            if has_arg_part and i > 0:
                prev = children[i - 1]
                if prev.type == "identifier":
                    name = prev.text.decode()
                    if name and not name[0].isupper():
                        calls.add(name)
                elif prev.type == "unconditional_assignable_selector":
                    for pc in prev.children:
                        if pc.type == "identifier":
                            name = pc.text.decode()
                            if name and not name[0].isupper():
                                calls.add(name)
        _dart_walk_calls(child, calls)


def _dart_get_body(node: Node, node_type: str) -> Node | None:
    """Return the body node of a class/mixin/extension node."""
    body_type = _DART_BODY_TYPE.get(node_type, "class_body")
    for child in node.children:
        if child.type == body_type:
            return child
    return None


def _dart_collect_members(
    body: Node,
) -> list[tuple[str, Node, Node, Node]]:
    """Collect (name, chunk_start_node, sig_node, body_end_node) tuples.

    chunk_start_node: annotation node if present (for chunk line_start)
    sig_node: method_signature or declaration (for outline signature text)
    body_end_node: function_body or declaration (for chunk line_end)

    Pairs consecutive method_signature + function_body nodes.
    Also handles declaration nodes (constructors).
    Skips field declarations and punctuation.
    """
    members: list[tuple[str, Node, Node, Node]] = []
    children = [c for c in body.children if c.type not in ("{", "}", ";")]
    pending_annotation: Node | None = None

    i = 0
    while i < len(children):
        node = children[i]

        if node.type == "annotation":
            pending_annotation = node
            i += 1
            continue

        if node.type == "method_signature":
            if i + 1 < len(children) and children[i + 1].type == "function_body":
                sig_node = node
                body_node = children[i + 1]
                sig_inner = next((c for c in sig_node.children if c.is_named), sig_node)
                name = _dart_method_name(sig_inner)
                chunk_start = pending_annotation if pending_annotation else sig_node
                members.append((name, chunk_start, sig_node, body_node))
                i += 2
                pending_annotation = None
                continue

        if node.type == "declaration":
            sig_inner = next(
                (c for c in node.children if "constructor" in c.type),
                None,
            )
            if sig_inner is not None:
                name = _dart_method_name(sig_inner)
                chunk_start = pending_annotation if pending_annotation else node
                members.append((name, chunk_start, node, node))
            pending_annotation = None
            i += 1
            continue

        pending_annotation = None
        i += 1

    return members


def _dart_member_chunk(
    name: str,
    chunk_start_node: Node,
    body_node: Node,
    parent_name: str,
    file_path: str,
    lines: list[str],
    imports: list[str],
    max_chunk_lines: int,
) -> list[dict]:
    """Build method chunk(s) for a Dart class member."""
    start_line = chunk_start_node.start_point[0] + 1
    end_line = body_node.end_point[0] + 1
    content = "\n".join(lines[start_line - 1: end_line])
    calls = _dart_extract_calls(body_node)
    num_lines = end_line - start_line + 1

    if num_lines <= max_chunk_lines:
        return [{
            "file_path": file_path,
            "chunk_type": "method",
            "name": name,
            "line_start": start_line,
            "line_end": end_line,
            "content": content,
            "language": "dart",
            "parent_name": parent_name,
            "calls_json": json.dumps(calls),
            "outline_json": "[]",
            "imports_json": json.dumps(imports),
        }]

    named_children = [c for c in body_node.children if c.is_named]
    sub_chunks: list[dict] = []
    current: list[Node] = []
    current_lines = 0
    part = 1

    for child in named_children:
        child_line_count = child.end_point[0] - child.start_point[0] + 1
        if current_lines + child_line_count > max_chunk_lines and current:
            sub_chunks.append(_make_sub_chunk(
                current, lines, file_path, "method", name, part,
                parent_name, "dart", calls, imports,
            ))
            part += 1
            current = []
            current_lines = 0
        current.append(child)
        current_lines += child_line_count

    if current:
        sub_chunks.append(_make_sub_chunk(
            current, lines, file_path, "method", name, part,
            parent_name, "dart", calls, imports,
        ))

    return sub_chunks


def _dart_class_like_chunks(
    node: Node,
    file_path: str,
    lines: list[str],
    imports: list[str],
    max_chunk_lines: int,
) -> list[dict]:
    """Extract class outline + member chunks from a class/mixin/extension node."""
    chunks: list[dict] = []
    node_type = node.type
    class_name = _dart_node_name(node)

    body = _dart_get_body(node, node_type)
    members = _dart_collect_members(body) if body else []

    member_sigs: list[str] = []
    for _name, _chunk_start, sig_node, _body_node in members:
        # Use sig_node (method_signature) for the outline, not the annotation
        sig_line = lines[sig_node.start_point[0]].strip()
        member_sigs.append(sig_line)

    outline_content = f"class {class_name}:\n"
    for sig in member_sigs:
        outline_content += f"  {sig}\n"

    chunks.append({
        "file_path": file_path,
        "chunk_type": "class",
        "name": class_name,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "content": outline_content.strip(),
        "language": "dart",
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": json.dumps(member_sigs),
        "imports_json": json.dumps(imports),
    })

    for name, chunk_start, _sig_node, body_node in members:
        chunks.extend(_dart_member_chunk(
            name, chunk_start, body_node, class_name,
            file_path, lines, imports, max_chunk_lines,
        ))

    return chunks


def _parse_dart_tree_sitter(
    source: str,
    file_path: str,
    max_chunk_lines: int,
) -> list[dict]:
    """Parse Dart source using tree-sitter-dart AST."""
    parser = _get_parser("dart")
    tree = parser.parse(source.encode())
    root = tree.root_node
    lines = source.split("\n")

    imports = _dart_extract_imports(root)
    chunks: list[dict] = []

    top_children = root.children
    i = 0
    while i < len(top_children):
        child = top_children[i]

        if child.type in _DART_CLASS_LIKE:
            chunks.extend(_dart_class_like_chunks(
                child, file_path, lines, imports, max_chunk_lines,
            ))
            i += 1
            continue

        # Top-level functions: function_signature followed by function_body
        if child.type == "function_signature":
            func_name = _dart_node_name(child)
            if i + 1 < len(top_children) and top_children[i + 1].type == "function_body":
                body_node = top_children[i + 1]
                start_line = child.start_point[0] + 1
                end_line = body_node.end_point[0] + 1
                content = "\n".join(lines[start_line - 1: end_line])
                calls = _dart_extract_calls(body_node)
                num_lines = end_line - start_line + 1

                if num_lines <= max_chunk_lines:
                    chunks.append({
                        "file_path": file_path,
                        "chunk_type": "function",
                        "name": func_name,
                        "line_start": start_line,
                        "line_end": end_line,
                        "content": content,
                        "language": "dart",
                        "parent_name": "",
                        "calls_json": json.dumps(calls),
                        "outline_json": "[]",
                        "imports_json": json.dumps(imports),
                    })
                else:
                    named_children = [c for c in body_node.children if c.is_named]
                    sub_chunks: list[dict] = []
                    current: list[Node] = []
                    current_lines = 0
                    part = 1
                    for bc in named_children:
                        bc_line_count = bc.end_point[0] - bc.start_point[0] + 1
                        if current_lines + bc_line_count > max_chunk_lines and current:
                            sub_chunks.append(_make_sub_chunk(
                                current, lines, file_path, "function", func_name, part,
                                "", "dart", calls, imports,
                            ))
                            part += 1
                            current = []
                            current_lines = 0
                        current.append(bc)
                        current_lines += bc_line_count
                    if current:
                        sub_chunks.append(_make_sub_chunk(
                            current, lines, file_path, "function", func_name, part,
                            "", "dart", calls, imports,
                        ))
                    chunks.extend(sub_chunks)

                i += 2
                continue

        i += 1

    return chunks
