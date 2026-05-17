"""
MCP服务器启动入口

Usage:
    python -m src.mcp_server
"""

import sys
from src.mcp_server.server import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C)", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\nServer error: {e}", file=sys.stderr)
        sys.exit(1)
