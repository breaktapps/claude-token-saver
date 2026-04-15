#!/usr/bin/env bash
# Auto-reindex hook: triggers incremental reindex after Edit/Write tool use.
# Runs asynchronously to avoid blocking the Claude Code agent.
# On lock timeout, logs a warning but does not fail.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python / uv
if command -v uv &>/dev/null; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

# Run incremental reindex in background (async, non-blocking)
(
    cd "${PLUGIN_DIR}" 2>/dev/null || exit 0

    ${PYTHON_CMD} -c "
import asyncio
import sys
import os

sys.path.insert(0, os.path.join('${PLUGIN_DIR}', 'src'))

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
        await indexer.reindex(force=False)
    except Exception as e:
        # Graceful failure: log but do not crash
        print(f'[claude-token-saver] auto-reindex warning: {e}', file=sys.stderr)

asyncio.run(run())
" 2>/dev/null &
) &

exit 0
