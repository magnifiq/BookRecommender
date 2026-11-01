from pydantic import BaseModel, Field

from app.tools.base import PermissionPolicy, ToolDefinition


class ApprovalDemoArgs(BaseModel):
    note: str = Field(description="A short note to include in the demo result")


class DeniedDemoArgs(BaseModel):
    reason: str = Field(description="Why the denied demo tool was requested")


async def approval_demo_handler(args: ApprovalDemoArgs):
    return {"accepted_note": args.note}


async def denied_demo_handler(args: DeniedDemoArgs):
    return {"unexpected": args.reason}


approval_demo_tool = ToolDefinition(
    name="approval_demo",
    description="A harmless example tool that demonstrates human approval.",
    args_schema=ApprovalDemoArgs,
    handler=approval_demo_handler,
    permission_policy=PermissionPolicy.REQUIRES_APPROVAL,
)

denied_demo_tool = ToolDefinition(
    name="denied_demo",
    description="A harmless tool that is disabled by application policy.",
    args_schema=DeniedDemoArgs,
    handler=denied_demo_handler,
    permission_policy=PermissionPolicy.DENIED,
)
