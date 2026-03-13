"""Phase 3: Launch the QuranRAG API server.

Usage:
    python scripts/04_serve.py                    # REST API on port 8000
    python scripts/04_serve.py --mcp-http         # MCP HTTP server on port 8001 (for Claude.ai)
    python scripts/04_serve.py --mcp-stdio        # MCP stdio mode (for Claude Desktop app)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_api_server(host: str, port: int) -> None:
    """Run the FastAPI REST API server."""
    import uvicorn

    from src.api.app import create_app

    app = create_app()
    uvicorn.run(app, host=host, port=port)


def run_mcp_stdio() -> None:
    """Run MCP server in stdio mode (for Claude Desktop app).

    Components are lazy-loaded on first tool call, so startup is instant.
    """
    from src.mcp.server import mcp_server

    mcp_server.run(transport="stdio")


def run_mcp_http(host: str, port: int) -> None:
    """Run MCP server with Streamable HTTP transport (for Claude.ai web)."""
    from src.mcp.server import mcp_server

    print(f"\nMCP server starting on http://{host}:{port}/mcp")
    print("Paste this URL in Claude.ai > Personnaliser > Connecteurs > Ajouter un connecteur")
    print(f"\n  URL: http://localhost:{port}/mcp\n")

    mcp_server.run(transport="streamable-http", host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="QuranRAG Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--mcp-http",
        action="store_true",
        help="Run MCP server with HTTP transport (for Claude.ai web)",
    )
    parser.add_argument(
        "--mcp-stdio",
        action="store_true",
        help="Run MCP server in stdio mode (for Claude Desktop app)",
    )
    args = parser.parse_args()

    if args.mcp_stdio:
        run_mcp_stdio()
    elif args.mcp_http:
        run_mcp_http(args.host, args.port or 8001)
    else:
        run_api_server(args.host, args.port)


if __name__ == "__main__":
    main()
