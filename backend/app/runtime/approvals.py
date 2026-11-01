from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["pending", "approved", "rejected", "expired"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    resolved_at: datetime | None = None


class ApprovalNotFound(KeyError):
    pass


class ApprovalConflict(ValueError):
    pass


class ApprovalStore:
    """Small in-memory store; mutations are synchronous and atomic in the event loop."""

    def __init__(self, expiration_seconds: int = 900,
                 on_expire: Callable[[ApprovalRequest], None] | None = None):
        if expiration_seconds <= 0:
            raise ValueError("expiration_seconds must be greater than zero")
        self.expiration_seconds = expiration_seconds
        self.on_expire = on_expire
        self._requests: dict[str, ApprovalRequest] = {}

    def create(self, run_id: str, tool_call_id: str, tool_name: str,
               arguments: dict[str, Any]) -> ApprovalRequest:
        request = ApprovalRequest(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            expires_at=utc_now() + timedelta(seconds=self.expiration_seconds),
        )
        self._requests[request.approval_id] = request
        return request

    def get(self, approval_id: str) -> ApprovalRequest:
        request = self._requests.get(approval_id)
        if request is None:
            raise ApprovalNotFound(approval_id)
        self._expire_if_needed(request)
        return request

    def list_pending(self) -> list[ApprovalRequest]:
        for request in self._requests.values():
            self._expire_if_needed(request)
        return [r for r in self._requests.values() if r.status == "pending"]

    def resolve(self, approval_id: str, decision: Literal["approved", "rejected"]):
        request = self.get(approval_id)
        if request.status != "pending":
            raise ApprovalConflict(f"Approval is already {request.status}")
        request.status = decision
        request.resolved_at = utc_now()
        return request

    def _expire_if_needed(self, request: ApprovalRequest) -> None:
        if request.status == "pending" and utc_now() >= request.expires_at:
            request.status = "expired"
            request.resolved_at = utc_now()
            if self.on_expire:
                self.on_expire(request)
