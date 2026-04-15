#!/usr/bin/env bash
# PreToolUse hook: nudge Claude to prefer claude-token-saver tools over native Grep/Read.
# NEVER blocks or denies the tool — only injects additionalContext via stdout JSON.
# Spec: FR43 (PreToolUse nudge)

set -euo pipefail

# Determine repo root: walk up from CWD looking for .git
REPO_ROOT=""
dir="$PWD"
while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/.git" ]]; then
        REPO_ROOT="$dir"
        break
    fi
    dir="$(dirname "$dir")"
done

# If no repo found, no nudge
if [[ -z "$REPO_ROOT" ]]; then
    exit 0
fi

# Check if a claude-token-saver index exists for this repo
# Index is stored at ~/.claude-token-saver/indices/<repo-hash>/index.lance
INDEX_BASE="$HOME/.claude-token-saver/indices"
REPO_HASH=$(echo -n "$REPO_ROOT" | sha256sum | cut -c1-16 2>/dev/null || echo -n "$REPO_ROOT" | shasum -a 256 | cut -c1-16)
INDEX_PATH="$INDEX_BASE/$REPO_HASH/index.lance"

# Fallback: also check local .cts/ directory (portable index)
LOCAL_INDEX="$REPO_ROOT/.cts/index.lance"

if [[ ! -d "$INDEX_PATH" && ! -d "$LOCAL_INDEX" ]]; then
    # No index — do not nudge. Native tools are the right choice.
    exit 0
fi

# Determine which tool triggered this hook
TOOL_NAME="${CLAUDE_TOOL_NAME:-}"

GREP_NUDGE="This project has a semantic code index (claude-token-saver). Use search_semantic(query) for natural language search or search_exact(query) for exact name lookup instead of Grep. 70-90% token savings."

READ_NUDGE="This file may be indexed by claude-token-saver. Use get_file(file_path) for structured reading with class outlines and method details instead of raw Read."

GLOB_NUDGE="This project has a semantic code index (claude-token-saver). Use search_exact(query) to find functions, classes, or variables by name instead of Glob. Returns enriched results with metadata."

AGENT_NUDGE="This project has a semantic code index (claude-token-saver). Use search_semantic(query) for code exploration instead of spawning an Explore agent. Faster, cheaper, already indexed."

case "$TOOL_NAME" in
    Grep)
        echo "$GREP_NUDGE"
        ;;
    Read)
        echo "$READ_NUDGE"
        ;;
    Glob)
        echo "$GLOB_NUDGE"
        ;;
    Agent)
        # Only nudge if it looks like an Explore agent
        TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"
        if echo "$TOOL_INPUT" | grep -qiE 'explore|search|find|where|how'; then
            echo "$AGENT_NUDGE"
        fi
        ;;
    Bash)
        # Only nudge if input looks like grep/cat/find
        TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"
        if echo "$TOOL_INPUT" | grep -qE '^\s*(grep|cat|find|rg|ag)\s'; then
            echo "$GREP_NUDGE"
        fi
        ;;
    *)
        ;;
esac

# Always exit 0 — never block the tool call
exit 0
