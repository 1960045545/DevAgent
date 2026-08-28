from mcp.server import MCPServer
from core.tool_registry import ToolRegistry
import mineru

def build_mcp_server(registry: ToolRegistry) -> MCPServer:
    server = MCPServer("AgentDemo")
    for spec in registry.list_specs():
        handler = registry.get_handler(spec.name)
        if handler is None:
            continue
        server.add_tool(
            handler,
            name=spec.name,
            description=spec.description,
        )
    return server