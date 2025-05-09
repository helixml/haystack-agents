from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo

# Create an MCP tool that uses stdio transport
server_info = StdioServerInfo(command="uvx", args=["run", "mcp-server-time", "--", "--local-timezone=Europe/Berlin"])
tool = MCPTool(name="time_tool", server_info=server_info)

# Get the current time in New York
result = tool.invoke(timezone="America/New_York")

