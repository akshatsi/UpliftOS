"""Reward function: reward = (amount_recovered × uplift_indicator) - tactic_cost.

Reward is only ever computed once the recovery window has closed — never
partial-window — and control-group accounts always get 0 (they exist to
calibrate the baseline, not to feed the bandit).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from attribution.uplift import get_uplift
from db.models import Outcome, RecoveryAction

RECOVERY_WINDOW_DAYS = int(os.getenv("RECOVERY_WINDOW_DAYS", "7"))


class RewardWindowNotClosedError(Exception):
    """Raised when reward is requested before the recovery window has closed."""


@dataclass
class RewardResult:
    reward: float
    uplift_indicator: float
    amount_recovered: float
    tactic_cost: float


def window_closed(action: RecoveryAction, *, now: datetime | None = None, window_days: int = RECOVERY_WINDOW_DAYS) -> bool:
    now = now or datetime.now(timezone.utc)
    return (now - action.deployed_at).days >= window_days


def compute_reward(db: Session, outcome: Outcome, *, now: datetime | None = None) -> RewardResult:
    """Pure reward calculation for one resolved outcome. Raises rather than
    returning a partial-window number if the recovery window is still open.
    """
    action = outcome.action
    account = action.account

    if account.is_control_group:
        return RewardResult(reward=0.0, uplift_indicator=0.0, amount_recovered=0.0, tactic_cost=0.0)

    if not window_closed(action, now=now):
        raise RewardWindowNotClosedError(f"recovery window for action {action.action_id} has not closed yet")

    amount_recovered = outcome.amount_recovered if outcome.recovered else 0.0
    uplift_indicator = get_uplift(db, action.segment_key, outcome.recovered)
    tactic_cost = action.tactic_cost

    reward = (amount_recovered * uplift_indicator) - tactic_cost
    reward = min(reward, amount_recovered)  # cannot reward more than what was actually recovered
    reward = round(reward, 2)

    return RewardResult(reward=reward, uplift_indicator=uplift_indicator, amount_recovered=amount_recovered, tactic_cost=tactic_cost)


def apply_reward(db: Session, outcome: Outcome, *, now: datetime | None = None) -> RewardResult:
    """Compute reward and write it (+ uplift_indicator) onto `outcome`.

    Does not set `processed_at` or commit — that's the batch updater's
    call, so a batch that crashes partway through can't look complete.
    """
    result = compute_reward(db, outcome, now=now)
    outcome.reward = result.reward
    outcome.uplift_indicator = result.uplift_indicator
    return result
