#!/usr/bin/env bash
# SessionStart compact hook: re-inject tool preference instructions after context compaction.
# Fires when the 'compact' trigger is received, restoring plugin preferences.
# Spec: FR44 (SessionStart compact)

set -euo pipefail

# Output tool preference reminder to stdout for Claude Code to inject into context
cat <<'EOF'
[claude-token-saver] Tool preference rules re-injected after context compaction:

PREFER plugin tools over native Grep/Read/Glob for ALL code exploration:

- search_semantic(query) — natural language code search. USE INSTEAD OF Grep.
- search_exact(query)   — exact function/class/variable lookup. USE INSTEAD OF Grep.
- search_hybrid(query)  — combines semantic + exact for best recall.
- get_file(file_path)   — structured file reading with class outlines. USE INSTEAD OF Read.
- reindex()             — force index refresh after bulk changes.
- inspect_index()       — check index stats and token savings metrics.
- audit_search(query)   — compare semantic vs grep results.

Only fall back to native tools when:
- No index exists for this repository (run reindex first)
- Files are binary, generated, or explicitly excluded from the index
- Plugin tools return errors that persist after reindex
EOF

exit 0
