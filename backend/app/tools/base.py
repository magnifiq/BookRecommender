from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable

from openai import pydantic_function_tool
from pydantic import BaseModel


ToolHandler = Callable[[BaseModel], Awaitable[Any]]


class PermissionPolicy(StrEnum):
    AUTO = "auto"
    REQUIRES_APPROVAL = "requires_approval"
    DENIED = "denied"


@dataclass
class ToolDefinition:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO

    def to_openai_tool(self):
        return pydantic_function_tool(
            self.args_schema,
            name=self.name,
            description=self.description,
        )
