# claude-token-saver

**Semantic search for your codebase. 70-90% fewer tokens on every exploration.**

![CI](https://github.com/breaktapps/claude-token-saver/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## The Problem

When Claude Code explores a codebase, it typically runs cascading grep/cat chains across dozens of files — reading everything to find the three relevant functions. Research from multiple independent sources, confirmed by Anthropic in March 2026, shows that **70-87% of tokens consumed by Claude Code are spent reading code the agent never uses.**

On a busy day with a large monorepo, a single architectural question can drain 40% of your session quota before the agent types a single line of code.

## The Solution

claude-token-saver is a Claude Code plugin that replaces grep/cat-based exploration with vector search. It indexes your repository using AST-based chunking (tree-sitter) and local embeddings (fastembed), then exposes 7 MCP tools that return only the relevant chunks — ranked by semantic similarity, in the first call.

**Before (without plugin):**

```
Agent reads 18 files, 300K tokens consumed
→ finds 3 relevant functions
→ "Where does the app handle network errors?"
```

**After (with plugin):**

```
search_semantic("network error handling")
→ returns 5 ranked chunks, 2.8K tokens consumed
→ footer: "11,100 tokens saved (92% reduction)"
```

One install command. No API key. No external infrastructure. Works offline.

---

## Installation

```bash
/plugin install claude-token-saver
```

Or, before the official marketplace listing, add the BreaktApps marketplace:

```bash
/plugin source add breaktapps/claude-plugins
/plugin install claude-token-saver
```

The plugin bootstraps its own runtime: it detects or installs `uv`, then installs Python 3.11+ and all dependencies automatically. You are productive in under 60 seconds from the first index.

---

## How It Works

```
Your question (natural language)
        |
        v
  [Embed query]  ← fastembed, local, no API call
        |
        v
  [Vector search]  ← LanceDB, local file, Apache Arrow
        |
        v
  [Ranked chunks]  ← AST-aware boundaries, never mid-block
        |
        v
  Agent context  ← only what's relevant + tokens_saved metric
```

**Indexing flow:** On first use, the plugin scans your repo, parses each file with tree-sitter (AST-aware chunking), generates embeddings locally, and stores everything in a LanceDB file on your machine. No code leaves your machine in the default configuration.

**Auto-reindex:** After every Edit or Write operation by Claude Code, a post_tool hook re-indexes only the modified files (incremental, hash-based). Your index stays current without manual intervention.

---

## 7 MCP Tools

| Tool | Description | USE INSTEAD OF |
|------|-------------|----------------|
| `search_semantic` | Vector search by natural language query, ranked by similarity | `Grep` + `Read` chains |
| `search_exact` | Full-text search in the index (function names, patterns) | `Grep` |
| `search_hybrid` | Combines semantic + full-text for maximum recall | `Grep` + `Read` chains |
| `get_file` | Structured file reading via index — returns chunks by type (class, method, function) | `Read` |
| `reindex` | Incremental re-index (changed files only) or forced full re-index | — |
| `audit_search` | Compare semantic vs grep for the same query — shows gaps in each approach | — |
| `inspect_index` | Index statistics: total files, chunks, languages, cumulative tokens saved | — |

Every tool returns a `tokens_saved` field with before/after metrics for that query.

---

## Configuration

Configuration file lives at `~/.claude-token-saver/config.yaml` (created automatically on first run).

### Embedding Modes

| Mode | Model | Size | Quality | Cost |
|------|-------|------|---------|------|
| `lite` (default) | `BAAI/bge-small-en-v1.5` | 33 MB | Good for most queries | Free |
| `quality` | `BAAI/bge-m3` | 568 MB | MTEB Code 80.76, optimized for code | Free |

To switch to quality mode, set one line in your config:

```yaml
embedding_mode: quality
```

The model downloads once and is cached locally.

### BYOK Providers (Growth)

For professional use cases requiring higher recall, configure an external embedding provider via environment variables:

```bash
# Voyage AI (recommended — MTEB Code 84.0)
export CLAUDE_TOKEN_SAVER_PROVIDER=voyage
export VOYAGE_API_KEY=your_key_here

# OpenAI
export CLAUDE_TOKEN_SAVER_PROVIDER=openai
export OPENAI_API_KEY=your_key_here

# Ollama (local, any model)
export CLAUDE_TOKEN_SAVER_PROVIDER=ollama
export OLLAMA_MODEL=nomic-embed-text
```

API keys are never written to config files — environment variables only.

### Key Config Options

```yaml
# ~/.claude-token-saver/config.yaml
embedding_mode: lite          # lite | quality
top_k: 5                      # default results per search
exclude_patterns:             # additional ignore patterns
  - "**/*.generated.dart"
  - "**/node_modules/**"
```

The plugin automatically respects your `.gitignore`.

---

## Examples

**Finding where a feature is implemented:**

```
search_semantic("payment retry logic after gateway timeout")
→ payments/retry_handler.py:handle_timeout (score 0.91)
→ payments/gateway.py:GatewayClient._retry (score 0.87)
→ config/payment_settings.py:RetryConfig (score 0.72)
→ tokens_saved: 8,400 (89% reduction)
```

**Looking up a specific function:**

```
search_exact("startVoiceStream")
→ audio/audio_service.dart:startVoiceStream (exact match)
→ audio/audio_service.dart:_initStreamBuffer (called by startVoiceStream)
→ tokens_saved: 3,200 (81% reduction)
```

**Structured file reading instead of raw cat:**

```
get_file("src/auth/middleware.py")
→ AuthMiddleware (class outline: 6 methods, no bodies)
→ AuthMiddleware.validate_token (full method, 23 lines)
→ AuthMiddleware.refresh_session (full method, 18 lines)
→ tokens_saved: 5,100 (76% reduction)
```

**Checking cumulative savings:**

```
inspect_index()
→ 487 files indexed, 3,241 chunks, 3 languages (Dart, Python, TypeScript)
→ Index size: 42 MB
→ Cumulative tokens saved this session: 47,000
→ All-time tokens saved: 1,240,000
```

---

## Performance

| Operation | Target |
|-----------|--------|
| Semantic search (`search_semantic`) | < 500ms including query embedding |
| Exact search (`search_exact`) | < 100ms |
| Initial index of 500 files | < 60 seconds (lite mode) |
| Incremental re-index (1-5 files) | < 3 seconds |
| Repeated queries (LRU cache) | < 50ms |
| Memory in idle | < 200 MB (model loaded) |

Index and embeddings are stored locally. No external service calls in default mode.

---

## Supported Languages

| Language | Phase |
|----------|-------|
| Dart | MVP |
| Python | MVP |
| TypeScript | MVP |
| JavaScript | Growth |
| Swift | Growth |
| Kotlin | Growth |
| SQL | Growth |

Language detection is automatic by file extension. Monorepos with multiple languages are supported natively.

---

## Development

**Requirements:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/breaktapps/claude-token-saver
cd claude-token-saver
uv sync
uv run pytest
```

**Run tests with coverage:**

```bash
uv run pytest --cov=src tests/
```

**Project structure:**

```
src/
├── server.py       # MCP server (FastMCP, 7 tools)
├── indexer.py      # scan, chunk, embed, store
├── chunker.py      # tree-sitter AST parsing
├── embeddings.py   # provider abstraction (local, Voyage, OpenAI, Ollama)
├── storage.py      # LanceDB + file lock
├── metrics.py      # tokens_saved counter
└── config.py       # configuration
tests/              # pytest, multi-language fixtures
hooks/              # post_tool auto-reindex
```

**Contributing:** Open an issue before submitting a PR for new features. Bug fixes and language grammar additions are welcome directly.

Issues: [github.com/breaktapps/claude-token-saver/issues](https://github.com/breaktapps/claude-token-saver/issues)

---

## License

MIT — see [LICENSE](LICENSE).
