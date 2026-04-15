---
name: token-saver-mode
user-invocable: false
description: Automatically applied rules for preferring claude-token-saver MCP tools over native file exploration tools
---

# Token Saver Mode

This skill is automatically loaded and applied. It is not user-invocable.

## Tool Selection Rules

When exploring code in a repository that has been indexed by claude-token-saver, prefer the plugin tools over native exploration:

### ALWAYS use plugin tools first:
- **search_semantic** — use instead of Grep + Read cascading when looking for code by concept or behavior
- **search_exact** — use instead of Grep when you know the exact function/class/variable name
- **search_hybrid** — use when you need both semantic understanding and exact name matching
- **get_file** — use instead of Read when you need a file's content structured by chunks

### Only fall back to native tools when:
- The plugin index is not available (reindex first with `reindex`)
- You need raw file content not indexed (binary files, generated files)
- The plugin explicitly returns stale results and reindex fails

### Code exploration workflow:
1. Start with `search_semantic` or `search_hybrid` for any code exploration task
2. Use `get_file` for full file content when needed
3. Use `inspect_index` to verify index state
4. Only use Grep/Read/Bash for file system operations, not code exploration

### Token savings reporting:
- **ALWAYS** show the `tokens_saved` info to the user after each search/get_file call
- Format: "Tokens saved: X (Y% reduction)" — e.g. "Tokens saved: 4,831 (74.9% reduction)"
- This is the plugin's core value proposition — the user needs to see the savings to justify the plugin
- The `tokens_saved` object is in every tool response: `{without_plugin, with_plugin, saved, reduction_pct}`

### Token savings target:
- Each semantic search call saves ~70-90% tokens vs grep/cat cascade
- Aim to resolve exploration tasks in 1-3 plugin tool calls instead of 10-20 native calls
