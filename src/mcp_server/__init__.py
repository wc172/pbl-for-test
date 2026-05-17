"""
MCP服务器包

提供视频转录内容的MCP工具服务
"""

from src.mcp_server.server import TranscriptionMCPServer, get_server, main

__all__ = ["TranscriptionMCPServer", "get_server", "main"]
