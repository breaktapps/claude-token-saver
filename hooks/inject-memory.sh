#!/usr/bin/env bash
# inject-memory.sh — Save plugin preference as auto-memory
#
# Creates a feedback memory file so the agent remembers to prefer
# plugin tools across conversations, not just the current one.
#
# Idempotent: won't create if already exists.

set -euo pipefail

REPO_PATH="${CTS_REPO_PATH:-${PWD}}"

# Resolve project memory dir from repo path
# Claude Code stores project memories at ~/.claude/projects/-<path-with-dashes>/memory/
PROJECT_KEY=$(echo "$REPO_PATH" | sed 's|/|-|g')
MEMORY_DIR="$HOME/.claude/projects/$PROJECT_KEY/memory"
MEMORY_FILE="$MEMORY_DIR/feedback_use_token_saver_plugin.md"
MEMORY_INDEX="$MEMORY_DIR/MEMORY.md"

# Skip if memory already exists
if [[ -f "$MEMORY_FILE" ]]; then
    exit 0
fi

# Skip if memory dir doesn't exist (project never used with Claude Code)
if [[ ! -d "$MEMORY_DIR" ]]; then
    exit 0
fi

# Create the memory file
cat > "$MEMORY_FILE" << 'MEMORY_EOF'
---
name: Prefer claude-token-saver plugin tools
description: Always use search_semantic/search_exact/get_file over Grep/Read/Glob for code exploration
type: feedback
---

Always prefer claude-token-saver plugin tools over native tools for code exploration:
- search_semantic instead of Grep (natural language queries)
- search_exact instead of Grep (exact name lookups)
- get_file instead of Read (structured file reading)
- search_hybrid instead of Grep+Glob (combined search)

**Why:** Plugin provides 70-90% token savings by returning only relevant code chunks instead of raw file contents. Native tools waste context window with irrelevant code.

**How to apply:** For ANY code search or file reading task, check if plugin tools are available first. Only fall back to native Grep/Read/Glob for non-code files or when the plugin index doesn't exist.
MEMORY_EOF

# Add to MEMORY.md index if not already there
if [[ -f "$MEMORY_INDEX" ]] && ! grep -q "feedback_use_token_saver_plugin" "$MEMORY_INDEX" 2>/dev/null; then
    echo "- [Prefer plugin tools](feedback_use_token_saver_plugin.md) — Always use search_semantic/get_file over Grep/Read for code exploration" >> "$MEMORY_INDEX"
fi

echo "[claude-token-saver] Saved tool preference to project memory"
