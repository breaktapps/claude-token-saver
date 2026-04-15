#!/usr/bin/env bash
# Git post-commit hook: triggers incremental reindex for files changed in the commit.
# Install: cp this file to .git/hooks/post-commit && chmod +x .git/hooks/post-commit
# On lock timeout, logs a warning but does not crash (exit 0 always).

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python / uv
if command -v uv &>/dev/null; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

# Get list of files changed in the last commit (relative paths)
# Falls back to empty string on initial commit (HEAD~1 doesn't exist yet)
if git rev-parse --verify HEAD~1 &>/dev/null; then
    CHANGED_FILES="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || true)"
else
    # First commit: index all tracked files
    CHANGED_FILES="$(git diff --name-only --cached 2>/dev/null || true)"
fi

if [ -z "${CHANGED_FILES}" ]; then
    exit 0
fi

# Run incremental reindex in background (async, non-blocking)
(
    cd "${PLUGIN_DIR}" 2>/dev/null || exit 0

    ${PYTHON_CMD} -c "
import asyncio
import sys
import os

sys.path.insert(0, os.path.join('${PLUGIN_DIR}', 'src'))

CHANGED_FILES = '''${CHANGED_FILES}'''

async def run():
    try:
        from src.config import Config
        from src.embeddings import create_embedding_provider
        from src.indexer import Indexer
        from src.storage import Storage
        from pathlib import Path

        config = Config.load()
        repo_path = Path.cwd()
        storage = Storage(config, repo_path)
        provider = create_embedding_provider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Incremental reindex (force=False respects lock and only processes changed files)
        await indexer.reindex(force=False)
    except Exception as e:
        # Graceful failure: log but do not crash
        print(f'[claude-token-saver] post-commit reindex warning: {e}', file=sys.stderr)

asyncio.run(run())
" 2>/dev/null &
) &

exit 0
