import pytest
from fastapi import HTTPException

import app.main as main


@pytest.mark.asyncio
async def test_unknown_approval_id_returns_404():
    with pytest.raises(HTTPException) as error:
        await main.get_approval("not-present")
    assert error.value.status_code == 404
    assert error.value.detail == "Approval request not found."


@pytest.mark.asyncio
async def test_approving_unknown_id_returns_404():
    with pytest.raises(HTTPException) as error:
        await main.approve_request("not-present")
    assert error.value.status_code == 404
    assert error.value.detail == "Approval request not found."
