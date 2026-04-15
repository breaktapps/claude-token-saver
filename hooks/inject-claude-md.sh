#!/usr/bin/env bash
# inject-claude-md.sh — Inject/update plugin preference rules in project's CLAUDE.md
#
# Called on SessionStart. Ensures the agent knows to prefer plugin tools.
# Uses start/end markers to replace the section on updates (not append).

set -euo pipefail

MARKER_START="<!-- claude-token-saver-start -->"
MARKER_END="<!-- claude-token-saver-end -->"
OLD_MARKER="<!-- claude-token-saver -->"
REPO_PATH="${CTS_REPO_PATH:-${PWD}}"
CLAUDE_MD="$REPO_PATH/CLAUDE.md"

# The section content (bump this when plugin updates change behavior)
SECTION_VERSION="0.1.3"
SECTION="$MARKER_START
## Code Search (claude-token-saver plugin v${SECTION_VERSION})

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
- **ALWAYS show token savings** after each search: \"Tokens saved: X (Y%) | Session total: Z saved\"
- Keep a running total of tokens saved across all plugin calls in this conversation
$MARKER_END"

# Clean up legacy markers from older versions (<!-- claude-token-saver -->)
if [[ -f "$CLAUDE_MD" ]] && grep -q "$OLD_MARKER" "$CLAUDE_MD" 2>/dev/null; then
    python3 -c "
import re
with open('$CLAUDE_MD', 'r') as f:
    content = f.read()
# Remove everything between paired old markers (inclusive)
pattern = re.compile(
    re.escape('$OLD_MARKER') + r'.*?' + re.escape('$OLD_MARKER'),
    re.DOTALL
)
content = pattern.sub('', content)
# Clean up any leftover blank lines from removal
content = re.sub(r'\n{3,}', '\n\n', content)
with open('$CLAUDE_MD', 'w') as f:
    f.write(content)
"
    echo "[claude-token-saver] Cleaned up legacy markers from CLAUDE.md"
fi

# If markers exist, check if content needs updating
if [[ -f "$CLAUDE_MD" ]] && grep -q "$MARKER_START" "$CLAUDE_MD" 2>/dev/null; then
    # Check if version already matches (no update needed)
    if grep -q "v${SECTION_VERSION}" "$CLAUDE_MD" 2>/dev/null; then
        exit 0
    fi

    # Replace existing section between markers
    # Use python for reliable multi-line sed replacement
    python3 -c "
import re, sys
with open('$CLAUDE_MD', 'r') as f:
    content = f.read()
pattern = re.compile(
    re.escape('$MARKER_START') + r'.*?' + re.escape('$MARKER_END'),
    re.DOTALL
)
new_content = pattern.sub('''$SECTION''', content)
with open('$CLAUDE_MD', 'w') as f:
    f.write(new_content)
"
    echo "[claude-token-saver] Updated CLAUDE.md section to v${SECTION_VERSION}"
else
    # First time — append
    echo "" >> "$CLAUDE_MD"
    echo "$SECTION" >> "$CLAUDE_MD"
    echo "[claude-token-saver] Injected tool preference rules into CLAUDE.md"
fi
