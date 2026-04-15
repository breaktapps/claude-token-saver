"""
E2E tests for the MCP server via real subprocess + JSON-RPC 2.0 over stdio.

Tests the full protocol flow:
  initialize → tools/list → tools/call search_semantic → tools/call inspect_index

Zero mocks: uses real LanceDB, fastembed, tree-sitter.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLUGIN_DIR = Path(__file__).parent.parent.parent  # products/plugin/
FIXTURES = Path(__file__).parent.parent / "fixtures"

_id_counter = 0


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _rpc(method: str, params: dict | None = None) -> bytes:
    """Serialize a JSON-RPC 2.0 request as a newline-terminated bytes string."""
    msg: dict = {"jsonrpc": "2.0", "id": _next_id(), "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


def _read_response(proc: subprocess.Popen, timeout: float = 30.0) -> dict:
    """Read one complete JSON line from the subprocess stdout with a real timeout.

    Uses os.read + select so we never block past `timeout` seconds.
    """
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    buf = b""

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stderr_data = _drain_stderr(proc)
            raise TimeoutError(
                f"No complete JSON-RPC response within {timeout}s.\n"
                f"Buffer so far: {buf!r}\nstderr: {stderr_data}"
            )

        ready, _, _ = select.select([fd], [], [], min(remaining, 0.1))
        if not ready:
            # Check if process has died
            retcode = proc.poll()
            if retcode is not None:
                stderr_data = _drain_stderr(proc)
                raise RuntimeError(
                    f"Server exited with code {retcode}.\n"
                    f"Buffer: {buf!r}\nstderr: {stderr_data}"
                )
            continue

        try:
            chunk = os.read(fd, 4096)
        except OSError as exc:
            stderr_data = _drain_stderr(proc)
            raise RuntimeError(
                f"Error reading from server stdout: {exc}\nstderr: {stderr_data}"
            ) from exc

        if not chunk:
            stderr_data = _drain_stderr(proc)
            raise RuntimeError(
                f"Server closed stdout unexpectedly.\n"
                f"Buffer: {buf!r}\nstderr: {stderr_data}"
            )

        buf += chunk
        # A complete message ends with \n; parse the last complete line
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line:
                return json.loads(line)


def _drain_stderr(proc: subprocess.Popen) -> str:
    """Non-blocking drain of stderr for diagnostic output."""
    if proc.stderr is None:
        return ""
    try:
        fd = proc.stderr.fileno()
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            return os.read(fd, 65536).decode(errors="replace")
    except OSError:
        pass
    return ""


def _send_and_recv(
    proc: subprocess.Popen,
    method: str,
    params: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    """Write a JSON-RPC request and return the parsed response."""
    proc.stdin.write(_rpc(method, params))
    proc.stdin.flush()
    return _read_response(proc, timeout=timeout)


# ---------------------------------------------------------------------------
# Test repo factory
# ---------------------------------------------------------------------------

def _create_test_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repo for the server to index."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # marks repo root for _find_repo_root

    src = repo / "src"
    src.mkdir()

    shutil.copy(FIXTURES / "python" / "data_processor.py", src / "data_processor.py")
    shutil.copy(FIXTURES / "python" / "simple_function.py", src / "simple_function.py")

    (src / "calculator.py").write_text(
        textwrap.dedent("""\
            def add(a: float, b: float) -> float:
                \"\"\"Return the sum of a and b.\"\"\"
                return a + b


            def subtract(a: float, b: float) -> float:
                \"\"\"Return the difference of a and b.\"\"\"
                return a - b


            class Calculator:
                \"\"\"Simple calculator with memory.\"\"\"

                def __init__(self) -> None:
                    self.memory: float = 0.0

                def store(self, value: float) -> None:
                    \"\"\"Store value in memory.\"\"\"
                    self.memory = value

                def recall(self) -> float:
                    \"\"\"Return value from memory.\"\"\"
                    return self.memory
        """)
    )

    return repo


# ---------------------------------------------------------------------------
# Fixture: live MCP server subprocess
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory):
    """Launch the real MCP server as a subprocess; yield (proc, repo).

    Scope=module so boot happens once (fastembed model download is expensive).
    The server CWD is the test repo so _find_repo_root resolves correctly.
    PLUGIN_DATA_DIR is isolated to avoid polluting the developer's index.
    """
    tmp = tmp_path_factory.mktemp("e2e")
    repo = _create_test_repo(tmp)
    index_dir = tmp / "index"
    index_dir.mkdir()

    # Build env: inherit PATH/HOME so uv/python are discoverable
    env = dict(os.environ)
    env["PLUGIN_DATA_DIR"] = str(index_dir)
    # Ensure the plugin src package is importable when using a plain python fallback
    env["PYTHONPATH"] = str(PLUGIN_DIR)

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [
            uv_bin,
            "--directory", str(PLUGIN_DIR),
            "run", "python", "-m", "src.server",
        ]
    else:
        cmd = [sys.executable, "-m", "src.server"]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(repo),
        env=env,
    )

    # Brief pause to let the server start before we send the first request.
    # The server is ready as soon as it begins reading stdin, which happens
    # almost immediately — 1 s is more than enough.
    time.sleep(1)

    retcode = proc.poll()
    if retcode is not None:
        stderr_data = proc.stderr.read().decode(errors="replace")
        pytest.fail(
            f"MCP server process exited immediately (code={retcode}).\n"
            f"cmd: {cmd}\nstderr:\n{stderr_data}"
        )

    yield proc, repo

    # Graceful teardown
    try:
        proc.stdin.close()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests — ordered so that search_semantic runs before inspect_index
# (the module-scoped fixture shares state across tests)
# ---------------------------------------------------------------------------

class TestMcpE2E:
    """Full E2E test suite exercising real JSON-RPC 2.0 over stdio."""

    def test_01_initialize(self, mcp_server):
        """MCP initialize handshake must succeed and return server info."""
        proc, repo = mcp_server

        response = _send_and_recv(
            proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0.0.1"},
            },
            timeout=15,
        )

        assert "result" in response, f"Expected result, got: {response}"
        result = response["result"]
        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "claude-token-saver"

    def test_02_initialized_notification(self, mcp_server):
        """Send notifications/initialized (no response expected); server stays alive."""
        proc, repo = mcp_server

        # This is a JSON-RPC notification (no id) — no response is sent
        notification = (
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        ).encode()
        proc.stdin.write(notification)
        proc.stdin.flush()

        # Give server a moment to process the notification
        time.sleep(0.2)
        assert proc.poll() is None, "Server should still be running after initialized notification"

    def test_03_tools_list_returns_seven_tools(self, mcp_server):
        """tools/list must return exactly the 7 registered MCP tools."""
        proc, repo = mcp_server

        response = _send_and_recv(proc, "tools/list", timeout=15)

        assert "result" in response, f"Expected result, got: {response}"
        tools = response["result"]["tools"]
        tool_names = {t["name"] for t in tools}

        expected = {
            "search_semantic",
            "search_exact",
            "search_hybrid",
            "get_file",
            "reindex",
            "audit_search",
            "inspect_index",
        }
        assert expected == tool_names, f"Tool mismatch. Got: {tool_names}"

    def test_04_search_semantic_returns_results_and_tokens_saved(self, mcp_server):
        """search_semantic via tools/call must return results with tokens_saved.

        First call triggers auto-indexation (fastembed model + LanceDB write).
        Allow up to 120 s for model download on a cold CI environment.
        """
        proc, repo = mcp_server

        response = _send_and_recv(
            proc,
            "tools/call",
            {
                "name": "search_semantic",
                "arguments": {"query": "load data from JSON file", "top_k": 3},
            },
            timeout=120,
        )

        assert "result" in response, f"Expected result, got: {response}"
        content = response["result"]["content"]
        assert len(content) > 0

        payload = json.loads(content[0]["text"])

        assert "error" not in payload, f"Tool returned error: {payload}"

        assert "results" in payload, f"Missing 'results': {payload}"
        assert "tokens_saved" in payload, f"Missing 'tokens_saved': {payload}"
        assert "metadata" in payload, f"Missing 'metadata': {payload}"

        savings = payload["tokens_saved"]
        for key in ("without_plugin", "with_plugin", "saved", "reduction_pct"):
            assert key in savings, f"Missing savings key '{key}': {savings}"

        meta = payload["metadata"]
        assert "query_time_ms" in meta
        assert meta["index_status"] in ("ready", "just_indexed")

    def test_05_search_semantic_result_fields(self, mcp_server):
        """Each result from search_semantic must have the full metadata schema."""
        proc, repo = mcp_server

        response = _send_and_recv(
            proc,
            "tools/call",
            {
                "name": "search_semantic",
                "arguments": {"query": "calculate sum", "top_k": 5},
            },
            timeout=30,
        )

        assert "result" in response
        payload = json.loads(response["result"]["content"][0]["text"])
        assert "error" not in payload, f"Unexpected error: {payload}"

        required = ("file_path", "name", "chunk_type", "line_start", "line_end", "content", "score", "language", "stale")
        for r in payload.get("results", []):
            for field in required:
                assert field in r, f"Missing field '{field}' in result: {r}"

    def test_06_inspect_index_returns_stats(self, mcp_server):
        """inspect_index must return global stats after indexation from prior tests."""
        proc, repo = mcp_server

        response = _send_and_recv(
            proc,
            "tools/call",
            {"name": "inspect_index", "arguments": {}},
            timeout=30,
        )

        assert "result" in response, f"Expected result, got: {response}"
        payload = json.loads(response["result"]["content"][0]["text"])

        assert "error" not in payload, f"inspect_index returned error: {payload}"

        for key in ("total_files", "total_chunks", "languages", "index_size_bytes",
                    "tokens_saved_total", "total_queries", "stale_files_count", "stale_files"):
            assert key in payload, f"Missing key '{key}': {payload}"

        assert payload["total_files"] >= 1, "Expected at least 1 indexed file"
        assert payload["total_chunks"] >= 1, "Expected at least 1 indexed chunk"
        assert "python" in payload["languages"], (
            f"Expected 'python' in languages: {payload['languages']}"
        )

    def test_07_inspect_index_total_queries_nonzero(self, mcp_server):
        """total_queries must reflect prior search_semantic calls."""
        proc, repo = mcp_server

        response = _send_and_recv(
            proc,
            "tools/call",
            {"name": "inspect_index", "arguments": {}},
            timeout=15,
        )

        payload = json.loads(response["result"]["content"][0]["text"])
        assert "error" not in payload, f"Unexpected error: {payload}"
        assert payload["total_queries"] >= 1, (
            f"Expected total_queries >= 1, got {payload['total_queries']}"
        )
