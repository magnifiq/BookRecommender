from app.tools.base import ToolDefinition


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        tool = self._tools.get(name)

        if tool is None:
            raise KeyError(f"Unknown tool: '{name}'")

        return tool

    def get_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list:
        return [
            tool.to_openai_tool()
            for tool in self._tools.values()
        ]