"""
CLI helper to register the earnings-call MCP server with Claude Desktop
and Claude Code.

Commands:
    python cli/setup_mcp.py install   — register the MCP server
    python cli/setup_mcp.py status    — check server registration & dependencies
"""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage the earnings-call MCP server registration.")
console = Console()

# Paths
REPO_ROOT = Path(__file__).parent.parent.resolve()
SERVER_SCRIPT = REPO_ROOT / "mcp_server" / "server.py"


def _get_claude_desktop_config_path() -> Optional[Path]:
    """Return the platform-specific Claude Desktop config path."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    return None


def _get_claude_code_config_path() -> Path:
    """Return the Claude Code (~/.mcp/config.json) config path."""
    return Path.home() / ".mcp" / "config.json"


def _build_server_entry() -> dict[str, Any]:
    """Build the MCP server configuration entry."""
    python_executable = sys.executable
    return {
        "command": python_executable,
        "args": [str(SERVER_SCRIPT)],
        "env": {
            "PYTHONPATH": str(REPO_ROOT),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning:[/yellow] {path} contains invalid JSON — starting fresh.")
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


@app.command()
def install() -> None:
    """Register the earnings-call MCP server with Claude Desktop and Claude Code."""
    entry = _build_server_entry()
    registered_somewhere = False

    # ── Claude Desktop ────────────────────────────────────────────────────────
    desktop_path = _get_claude_desktop_config_path()
    if desktop_path:
        config = _read_json(desktop_path)
        config.setdefault("mcpServers", {})
        config["mcpServers"]["earnings-call-server"] = entry
        _write_json(desktop_path, config)
        console.print(f"[green]✓[/green] Claude Desktop config updated: {desktop_path}")
        registered_somewhere = True
    else:
        console.print("[yellow]![/yellow] Could not determine Claude Desktop config path for this OS.")

    # ── Claude Code (~/.mcp/config.json) ─────────────────────────────────────
    code_path = _get_claude_code_config_path()
    code_config = _read_json(code_path)
    code_config.setdefault("mcpServers", {})
    code_config["mcpServers"]["earnings-call-server"] = entry
    _write_json(code_path, code_config)
    console.print(f"[green]✓[/green] Claude Code config updated: {code_path}")
    registered_somewhere = True

    if registered_somewhere:
        console.print("\n[bold green]Done![/bold green]  The MCP server is registered.")
        console.print("\nServer details:")
        console.print(f"  Script:  {SERVER_SCRIPT}")
        console.print(f"  Python:  {sys.executable}")
        console.print("\nNext steps:")
        console.print("  1. Restart Claude Desktop (if using it)")
        console.print("  2. In Claude, look for the [hammer] tool icon")
        console.print('  3. Ask: "What did NVDA say about data center demand?"')


@app.command()
def status() -> None:
    """Check server registration, Qdrant connectivity, and data file availability."""
    table = Table(title="Earnings Call MCP Server — Status", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Result")
    table.add_column("Detail")

    # ── 1. Server script exists ───────────────────────────────────────────────
    script_ok = SERVER_SCRIPT.exists()
    table.add_row(
        "Server script",
        "[green]OK[/green]" if script_ok else "[red]MISSING[/red]",
        str(SERVER_SCRIPT),
    )

    # ── 2. Claude Desktop registration ───────────────────────────────────────
    desktop_path = _get_claude_desktop_config_path()
    if desktop_path:
        config = _read_json(desktop_path)
        registered = "earnings-call-server" in config.get("mcpServers", {})
        table.add_row(
            "Claude Desktop",
            "[green]Registered[/green]" if registered else "[yellow]Not registered[/yellow]",
            str(desktop_path),
        )
    else:
        table.add_row("Claude Desktop", "[dim]N/A[/dim]", "Unsupported OS")

    # ── 3. Claude Code registration ───────────────────────────────────────────
    code_path = _get_claude_code_config_path()
    code_config = _read_json(code_path)
    code_registered = "earnings-call-server" in code_config.get("mcpServers", {})
    table.add_row(
        "Claude Code",
        "[green]Registered[/green]" if code_registered else "[yellow]Not registered[/yellow]",
        str(code_path),
    )

    # ── 4. Qdrant reachable ───────────────────────────────────────────────────
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        import httpx

        resp = httpx.get(f"{qdrant_url}/healthz", timeout=3.0)
        qdrant_ok = resp.status_code == 200
        table.add_row(
            "Qdrant",
            "[green]Reachable[/green]" if qdrant_ok else "[red]Unreachable[/red]",
            qdrant_url,
        )
    except Exception as exc:
        table.add_row("Qdrant", "[red]Unreachable[/red]", f"{qdrant_url}  ({exc})")

    # ── 5. Data files ──────────────────────────────────────────────────────────
    data_checks = {
        "Audio files": (REPO_ROOT / "data" / "audio", "*.mp3"),
        "Transcripts": (REPO_ROOT / "data" / "transcripts", "*.json"),
        "Embedding cache": (REPO_ROOT / "data", "embedding_cache.json"),
        "AskNews cache": (REPO_ROOT / "data" / "asknews_cache", "*.json"),
    }
    for label, (directory, pattern) in data_checks.items():
        files = list(directory.glob(pattern)) if directory.exists() else []
        if label == "Embedding cache":
            has_data = bool(files)
            detail = str(files[0]) if files else str(directory / pattern)
        else:
            has_data = len(files) > 0
            detail = f"{len(files)} file(s) in {directory}"
        table.add_row(
            label,
            "[green]Present[/green]" if has_data else "[yellow]Empty[/yellow]",
            detail,
        )

    console.print(table)


if __name__ == "__main__":
    app()
