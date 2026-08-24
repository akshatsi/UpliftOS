"""POST /outcomes/{account_id} — records a payment outcome, and finalizes
reward + bandit update immediately for control accounts (always) or
non-control accounts whose window has already closed (a late-reported
outcome). Otherwise the outcome is just recorded and the daily batch
updater will finalize it once the window closes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.errors import APIError
from api.schemas import OutcomeRequest, OutcomeResponse
from attribution.uplift import refresh_segment_baseline
from bandit.reward import apply_reward, window_closed
from bandit.thompson_sampling import update_state
from db.models import Account, Outcome, RecoveryAction
from db.session import get_db

router = APIRouter(tags=["outcomes"])


def _latest_action(db: Session, account_id: str) -> RecoveryAction | None:
    return (
        db.execute(
            select(RecoveryAction)
            .where(RecoveryAction.account_id == account_id)
            .order_by(RecoveryAction.deployed_at.desc())
        )
        .scalars()
        .first()
    )


@router.post("/outcomes/{account_id}", response_model=OutcomeResponse)
def record_outcome(account_id: str, body: OutcomeRequest, db: Session = Depends(get_db)) -> OutcomeResponse:
    account = db.get(Account, account_id)
    if account is None:
        raise APIError(404, "account_not_found", f"no account with id {account_id!r}")

    action = _latest_action(db, account_id)
    if action is None:
        raise APIError(
            409, "no_active_recovery_action", f"account {account_id!r} has no recovery action to attach an outcome to"
        )

    now = datetime.now(timezone.utc)

    outcome = action.outcome
    if outcome is None:
        outcome = Outcome(action_id=action.action_id, account_id=account_id, reported_at=now)
        db.add(outcome)

    # Once reward has been finalized (by this endpoint or the daily batch),
    # a re-report updates the record but never re-triggers reward/bandit
    # update -- that would double-count.
    already_finalized = outcome.processed_at is not None

    outcome.recovered = body.recovered
    outcome.amount_recovered = body.amount_recovered if body.recovered else 0.0
    outcome.recovered_at = body.recovered_at
    outcome.reported_at = now
    db.flush()

    reward_finalized = already_finalized
    if not already_finalized and (account.is_control_group or window_closed(action, now=now)):
        result = apply_reward(db, outcome, now=now)
        outcome.processed_at = now
        if not account.is_control_group:
            update_state(db, action.segment_key, action.tactic_name, result.reward)
        refresh_segment_baseline(db, action.segment_key, now=now)
        reward_finalized = True

    db.commit()

    return OutcomeResponse(
        account_id=account_id,
        outcome_id=outcome.outcome_id,
        recovered=outcome.recovered,
        amount_recovered=outcome.amount_recovered,
        reward=outcome.reward,
        reward_finalized=reward_finalized,
    )
