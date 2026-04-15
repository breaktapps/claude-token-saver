#!/usr/bin/env bash
# inject-claude-md.sh — Inject plugin preference rules into project's CLAUDE.md
#
# Called by bootstrap.sh on SessionStart. Ensures the agent knows to prefer
# plugin tools over native Grep/Read/Glob for code exploration.
#
# Idempotent: won't duplicate if section already exists.

set -euo pipefail

MARKER="<!-- claude-token-saver -->"
REPO_PATH="${CTS_REPO_PATH:-${PWD}}"
CLAUDE_MD="$REPO_PATH/CLAUDE.md"

# Skip if marker already present
if [[ -f "$CLAUDE_MD" ]] && grep -q "$MARKER" "$CLAUDE_MD" 2>/dev/null; then
    exit 0
fi

# Section to inject
SECTION="
$MARKER
## Code Search (claude-token-saver plugin)

This project has the claude-token-saver plugin installed. **Always prefer plugin tools over native tools** for code exploration:

| Instead of | Use | Why |
|------------|-----|-----|
| Grep | search_semantic or search_exact | 70-90% fewer tokens, semantic understanding |
| Read | get_file | Structured chunks with metadata, not raw text |
| Glob + Read | search_hybrid | Combines semantic + text search in one call |
| Agent(Explore) | search_semantic | Faster, cheaper, already indexed |

- Use **search_semantic** for natural language queries (\"where does X handle Y?\")
- Use **search_exact** for known names (function, class, variable)
- Use **search_hybrid** when you want maximum recall
- Use **get_file** instead of Read for code files
- Only fall back to native tools for non-code files (configs, images, etc.)
- **ALWAYS show token savings** after each search: "Tokens saved: X (Y%) | Session total: Z saved"
- Keep a running total of tokens saved across all plugin calls in this conversation
$MARKER"

# Append to CLAUDE.md (create if doesn't exist)
echo "$SECTION" >> "$CLAUDE_MD"
echo "[claude-token-saver] Injected tool preference rules into CLAUDE.md"
