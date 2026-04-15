"""AST-based code chunking for claude-token-saver.

Uses tree-sitter for Python and TypeScript/JavaScript parsing.
Uses regex-based parsing for Dart (no pip-installable grammar).
Extracts semantic chunks with metadata: class outlines, methods, functions.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser

from .config import Config

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

    if language == "dart":
        return _parse_dart(source, rel_path, max_chunk_lines)
    else:
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
# Dart regex-based parsing (no pip grammar available)
# ──────────────────────────────────────────────────────────────

_DART_CLASS_RE = re.compile(
    r"^(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?(?:\s+with\s+[\w,\s]+)?\s*\{",
    re.MULTILINE,
)

_DART_METHOD_RE = re.compile(
    r"^\s+(?:(?:static|async|Future<\w+>|void|int|double|String|bool|dynamic|List<\w+>|Map<\w+,\s*\w+>|\w+)\s+)+(\w+)\s*\([^)]*\)\s*(?:async\s*)?\{",
    re.MULTILINE,
)

_DART_FUNC_RE = re.compile(
    r"^(?:(?:Future<\w+>|void|int|double|String|bool|dynamic|\w+)\s+)(\w+)\s*\([^)]*\)\s*(?:async\s*)?\{",
    re.MULTILINE,
)

_DART_CALL_RE = re.compile(r"(\w+)\s*\(")

_DART_IMPORT_RE = re.compile(r"^import\s+'([^']+)'", re.MULTILINE)


def _parse_dart(
    source: str,
    file_path: str,
    max_chunk_lines: int,
) -> list[dict]:
    """Parse Dart source using regex-based approach."""
    lines = source.split("\n")
    chunks: list[dict] = []

    # C2: Extract imports
    imports = [m.group(0).strip() for m in _DART_IMPORT_RE.finditer(source)]

    # Build a set of line ranges covered by classes to detect top-level functions
    class_ranges: list[tuple[int, int]] = []

    # Find classes
    for match in _DART_CLASS_RE.finditer(source):
        class_name = match.group(1)
        class_start = source[:match.start()].count("\n")
        # Find matching closing brace
        class_end = _find_closing_brace(source, match.end() - 1)
        class_end_line = source[:class_end + 1].count("\n")
        class_ranges.append((class_start, class_end_line))

        class_source = "\n".join(lines[class_start: class_end_line + 1])

        # Extract methods within class
        method_chunks = []
        member_sigs = []
        for m_match in _DART_METHOD_RE.finditer(class_source):
            method_name = m_match.group(1)
            m_start_line = class_start + class_source[:m_match.start()].count("\n")
            m_brace_pos = class_source.index("{", m_match.start())
            m_end_pos = _find_closing_brace(class_source, m_brace_pos)
            m_end_line = class_start + class_source[:m_end_pos + 1].count("\n")

            method_text = "\n".join(lines[m_start_line: m_end_line + 1])
            sig = lines[m_start_line].strip()
            member_sigs.append(sig)

            calls = _extract_dart_calls(method_text)

            method_chunks.append({
                "file_path": file_path,
                "chunk_type": "method",
                "name": method_name,
                "line_start": m_start_line + 1,
                "line_end": m_end_line + 1,
                "content": method_text,
                "language": "dart",
                "parent_name": class_name,
                "calls_json": json.dumps(calls),
                "outline_json": "[]",
                "imports_json": json.dumps(imports),
            })

        # Class outline
        outline = f"class {class_name}:\n"
        for sig in member_sigs:
            outline += f"  {sig}\n"

        chunks.append({
            "file_path": file_path,
            "chunk_type": "class",
            "name": class_name,
            "line_start": class_start + 1,
            "line_end": class_end_line + 1,
            "content": outline.strip(),
            "language": "dart",
            "parent_name": "",
            "calls_json": "[]",
            "outline_json": json.dumps(member_sigs),
            "imports_json": json.dumps(imports),
        })
        chunks.extend(method_chunks)

    # C5: Extract top-level functions (not inside any class)
    for match in _DART_FUNC_RE.finditer(source):
        func_name = match.group(1)
        func_start_line = source[:match.start()].count("\n")

        # Skip if this match is inside a known class range
        if any(cs <= func_start_line <= ce for cs, ce in class_ranges):
            continue

        brace_pos = source.index("{", match.start())
        func_end_pos = _find_closing_brace(source, brace_pos)
        func_end_line = source[:func_end_pos + 1].count("\n")

        func_text = "\n".join(lines[func_start_line: func_end_line + 1])
        calls = _extract_dart_calls(func_text)

        chunks.append({
            "file_path": file_path,
            "chunk_type": "function",
            "name": func_name,
            "line_start": func_start_line + 1,
            "line_end": func_end_line + 1,
            "content": func_text,
            "language": "dart",
            "parent_name": "",
            "calls_json": json.dumps(calls),
            "outline_json": "[]",
            "imports_json": json.dumps(imports),
        })

    return chunks


def _find_closing_brace(source: str, open_pos: int) -> int:
    """Find the matching closing brace for an opening brace.

    Skips braces inside single-line comments (//), block comments (/* */),
    and string literals (single and double quotes).
    """
    depth = 0
    i = open_pos
    n = len(source)
    while i < n:
        ch = source[i]
        # Single-line comment: skip to end of line
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue
        # Block comment: skip to */
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1 and not (source[i] == "*" and source[i + 1] == "/"):
                i += 1
            i += 2  # skip */
            continue
        # String literal (single or double quote, non-multiline)
        if ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2  # skip escaped char
                    continue
                if source[i] == quote:
                    break
                i += 1
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _extract_dart_calls(source: str) -> list[str]:
    """Extract function call names from Dart source using regex."""
    # Exclude common keywords
    keywords = {"if", "for", "while", "switch", "catch", "return", "class", "new", "await", "assert"}
    calls = set()
    for match in _DART_CALL_RE.finditer(source):
        name = match.group(1)
        if name not in keywords and not name[0].isupper():
            calls.add(name)
    return sorted(calls)
