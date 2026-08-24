"""POST /recover/{account_id} — the full pipeline: eligibility/idempotency
check -> control group bypass -> bandit selects arm -> agent crew executes
-> action logged.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Union

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.crew import CrewExhaustedError, run_crew
from agents.llm_client import LLMCallError
from api.errors import APIError
from api.schemas import ControlGroupResponse, RecoverResponse
from attribution.control_group import build_control_action
from bandit.reward import window_closed
from bandit.thompson_sampling import select_arm
from db.models import Account, RecoveryAction
from db.session import get_db
from tactics.registry import get_tactic

logger = logging.getLogger("api.routes.recovery")

router = APIRouter(tags=["recovery"])


def _existing_open_action(db: Session, account_id: str, now: datetime) -> RecoveryAction | None:
    action = (
        db.execute(
            select(RecoveryAction)
            .where(RecoveryAction.account_id == account_id)
            .order_by(RecoveryAction.deployed_at.desc())
        )
        .scalars()
        .first()
    )
    if action is not None and not window_closed(action, now=now):
        return action
    return None


@router.post("/recover/{account_id}", response_model=Union[RecoverResponse, ControlGroupResponse])
def recover(account_id: str, db: Session = Depends(get_db)):
    account = db.get(Account, account_id)
    if account is None:
        raise APIError(404, "account_not_found", f"no account with id {account_id!r}")

    now = datetime.now(timezone.utc)

    # Idempotency: an existing action still inside its recovery window means
    # this account is already mid-cycle -- return it, no new bandit pull or
    # LLM calls.
    existing = _existing_open_action(db, account_id, now)
    if existing is not None:
        if account.is_control_group:
            return ControlGroupResponse()
        return RecoverResponse(
            tactic_selected=existing.tactic_name,
            message_drafted=existing.message_drafted,
            guardrail_approved=existing.guardrail_approved,
            action_logged_at=existing.deployed_at,
        )

    if account.is_control_group:
        action = build_control_action(account)
        db.add(action)
        db.commit()
        return ControlGroupResponse()

    selection = select_arm(db, account)
    tactic = get_tactic(selection.tactic_name)

    try:
        crew_result = run_crew(account, tactic)
    except CrewExhaustedError as exc:
        raise APIError(502, "agent_pipeline_exhausted", str(exc)) from exc
    except LLMCallError as exc:
        raise APIError(502, "llm_call_failed", str(exc)) from exc

    final_tactic = get_tactic(crew_result.tactic_name)
    tactic_cost = final_tactic.resolve_cost(account, crew_result.message.discount_offered)

    action = RecoveryAction(
        account_id=account.account_id,
        segment_key=selection.segment_key,
        tactic_name=crew_result.tactic_name,
        tactic_cost=tactic_cost,
        discount_offered=crew_result.message.discount_offered,
        sampled_values=selection.sampled_values,
        message_drafted=crew_result.message.model_dump(),
        guardrail_approved=crew_result.guardrail_approved,
        guardrail_reason=crew_result.guardrail_reason,
        fallback_reason=crew_result.fallback_reason,
        deployed_at=now,
    )
    db.add(action)
    db.commit()

    return RecoverResponse(
        tactic_selected=crew_result.tactic_name,
        message_drafted=crew_result.message.model_dump(),
        guardrail_approved=crew_result.guardrail_approved,
        action_logged_at=action.deployed_at,
    )
