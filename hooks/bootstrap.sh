#!/usr/bin/env bash
# bootstrap.sh — Claude Token Saver Plugin Bootstrap
#
# SessionStart hook: detects uv, installs deps if needed.
# Called by Claude Code on session start.
#
# Environment variables (set by Claude Code plugin system):
#   CLAUDE_PLUGIN_ROOT  — where plugin is installed (read-only)
#   CLAUDE_PLUGIN_DATA  — persistent data dir per-user (writable)
#   CLAUDE_WEB          — "1" if running on claude.ai/code (web)

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PLUGIN_DATA="${CLAUDE_PLUGIN_DATA:-$HOME/.claude/plugins/claude-token-saver}"
PYPROJECT_SRC="$PLUGIN_ROOT/pyproject.toml"
PYPROJECT_DST="$PLUGIN_DATA/pyproject.toml"
VENV_DIR="$PLUGIN_DATA/.venv"
UV_INSTALL_SCRIPT="https://astral.sh/uv/install.sh"
UV_DOC_URL="https://docs.astral.sh/uv/getting-started/installation/"

# ── Web environment detection ─────────────────────────────────────────────────
if [[ "${CLAUDE_WEB:-}" == "1" ]]; then
    # Claude Code on the Web: Python and uv are pre-installed, nothing to do
    exit 0
fi

# ── uv detection ──────────────────────────────────────────────────────────────
UV_BIN=""

# 1. Check if uv is in PATH
if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
fi

# 2. Check known install paths
if [[ -z "$UV_BIN" ]]; then
    for candidate in \
        "$HOME/.local/bin/uv" \
        "/opt/homebrew/bin/uv" \
        "/usr/local/bin/uv" \
        "$HOME/.cargo/bin/uv"
    do
        if [[ -x "$candidate" ]]; then
            UV_BIN="$candidate"
            break
        fi
    done
fi

# ── uv auto-install ───────────────────────────────────────────────────────────
if [[ -z "$UV_BIN" ]]; then
    echo "[claude-token-saver] uv not found. Attempting automatic installation..."

    if command -v curl &>/dev/null; then
        if curl -LsSf "$UV_INSTALL_SCRIPT" | sh; then
            # After install, uv is usually at ~/.local/bin/uv
            if [[ -x "$HOME/.local/bin/uv" ]]; then
                UV_BIN="$HOME/.local/bin/uv"
                echo "[claude-token-saver] uv installed successfully at $UV_BIN"
            elif command -v uv &>/dev/null; then
                UV_BIN="$(command -v uv)"
                echo "[claude-token-saver] uv installed successfully"
            fi
        fi
    fi
fi

# ── uv install fallback ───────────────────────────────────────────────────────
if [[ -z "$UV_BIN" ]]; then
    PLATFORM="$(uname -s 2>/dev/null || echo "Unknown")"
    case "$PLATFORM" in
        Darwin)
            INSTALL_CMD="brew install uv"
            ;;
        Linux)
            INSTALL_CMD="curl -LsSf $UV_INSTALL_SCRIPT | sh"
            ;;
        MINGW*|CYGWIN*|MSYS*|Windows*)
            INSTALL_CMD="powershell -c \"irm $UV_INSTALL_SCRIPT | iex\""
            ;;
        *)
            INSTALL_CMD="curl -LsSf $UV_INSTALL_SCRIPT | sh"
            ;;
    esac

    echo ""
    echo "[claude-token-saver] ERROR: Could not install 'uv' automatically."
    echo ""
    echo "  Runtime required: uv"
    echo "  Install command:  $INSTALL_CMD"
    echo "  Documentation:    $UV_DOC_URL"
    echo ""
    echo "  After installing uv, restart your Claude Code session."
    echo ""
    exit 1
fi

# ── Dependency installation ───────────────────────────────────────────────────
mkdir -p "$PLUGIN_DATA"

# Compare pyproject.toml between plugin root and plugin data
needs_install=false

if [[ ! -f "$PYPROJECT_DST" ]]; then
    needs_install=true
elif ! diff -q "$PYPROJECT_SRC" "$PYPROJECT_DST" &>/dev/null; then
    needs_install=true
fi

if [[ "$needs_install" == "true" ]]; then
    echo "[claude-token-saver] Installing/updating dependencies..."

    # Copy pyproject.toml to plugin data dir
    cp "$PYPROJECT_SRC" "$PYPROJECT_DST"

    # Create/recreate venv with Python 3.11+
    "$UV_BIN" venv --python ">=3.11" "$VENV_DIR"

    # Install dependencies from the copied pyproject.toml
    "$UV_BIN" pip install --python "$VENV_DIR" -r "$PYPROJECT_DST"

    echo "[claude-token-saver] Dependencies installed successfully."
else
    # Dependencies already up to date, skip silently
    :
fi
