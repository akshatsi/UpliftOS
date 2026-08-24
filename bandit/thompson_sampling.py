"""Contextual Thompson Sampling bandit: per-segment alpha/beta state and
arm selection. State lives entirely in the `bandit_state` table — there is
no in-process cache, so a server restart loses nothing.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Account, BanditState
from tactics.registry import CONTROL_TACTIC_NAME, FALLBACK_TACTIC_NAME, Tactic, eligible_tactics, get_tactic

logger = logging.getLogger("bandit.thompson_sampling")

INITIAL_ALPHA = 1.0
INITIAL_BETA = 1.0

# Reward is an unbounded, possibly-negative rupee amount (recovery value minus
# tactic cost), but the classic Beta-Bernoulli update needs a value in [0, 1].
# We squash reward through a logistic centered on 0: reward=0 -> 0.5 (neutral),
# reward>0 -> pulls alpha up (arm looks better), reward<0 -> pulls beta up (arm
# looks worse). REWARD_SCALE sets how many rupees it takes to approach 0/1;
# at +-REWARD_SCALE the pseudo-probability is ~0.73/~0.27.
REWARD_SCALE = 500.0


@dataclass
class ArmSelection:
    account_id: str
    segment_key: str
    tactic_name: str
    is_control: bool
    sampled_values: dict[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    selected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _get_or_create_state(db: Session, segment_key: str, tactic_name: str) -> BanditState:
    state = db.execute(
        select(BanditState).where(BanditState.segment_key == segment_key, BanditState.tactic_name == tactic_name)
    ).scalar_one_or_none()
    if state is None:
        state = BanditState(segment_key=segment_key, tactic_name=tactic_name, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)
        db.add(state)
        db.flush()
    return state


def get_segment_states(db: Session, segment_key: str) -> list[BanditState]:
    return list(db.execute(select(BanditState).where(BanditState.segment_key == segment_key)).scalars())


def get_all_states(db: Session) -> list[BanditState]:
    return list(db.execute(select(BanditState)).scalars())


def _select_among_eligible(db: Session, segment_key: str, eligible: list[Tactic], rng: random.Random) -> tuple[str, dict[str, float]]:
    sampled: dict[str, float] = {}
    for tactic in eligible:
        state = _get_or_create_state(db, segment_key, tactic.name)
        sampled[tactic.name] = rng.betavariate(state.alpha, state.beta)
    selected_name = max(sampled, key=sampled.get)
    return selected_name, sampled


def select_arm(db: Session, account: Account, rng: random.Random | None = None) -> ArmSelection:
    """Select a recovery tactic for `account`.

    Control-group accounts always get `no_action` (arm 10) and never touch
    the bandit. Non-control accounts sample among eligible arms only; if
    that fails for any reason, fall back to `soft_nudge_email` (arm 4) —
    this function never raises and never returns a null tactic.
    """
    rng = rng or random.Random()
    segment_key = account.segment
    now = datetime.now(timezone.utc)

    if account.is_control_group:
        return ArmSelection(
            account_id=account.account_id,
            segment_key=segment_key,
            tactic_name=CONTROL_TACTIC_NAME,
            is_control=True,
            selected_at=now,
        )

    eligible = eligible_tactics(account)  # never includes no_action
    selected_name: str
    sampled: dict[str, float] = {}
    fallback_reason: str | None = None

    if not eligible:
        selected_name = FALLBACK_TACTIC_NAME
        fallback_reason = f"no eligible tactics for segment {segment_key!r}"
    else:
        try:
            selected_name, sampled = _select_among_eligible(db, segment_key, eligible, rng)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is the last line of defense
            selected_name = FALLBACK_TACTIC_NAME
            fallback_reason = f"bandit sampling error: {exc!r}"

    if selected_name == CONTROL_TACTIC_NAME:
        # Structurally unreachable (eligible_tactics() excludes it), but this
        # is the hard constraint the spec calls out explicitly — fail loudly
        # rather than silently letting a non-control account get no_action.
        raise RuntimeError(f"invariant violated: no_action selected for non-control account {account.account_id}")

    if fallback_reason is not None:
        logger.warning(
            "bandit_fallback account_id=%s segment=%s selected=%s reason=%s timestamp=%s",
            account.account_id, segment_key, selected_name, fallback_reason, now.isoformat(),
        )
    else:
        logger.info(
            "bandit_arm_selected account_id=%s segment=%s sampled=%s selected=%s timestamp=%s",
            account.account_id, segment_key, sampled, selected_name, now.isoformat(),
        )

    return ArmSelection(
        account_id=account.account_id,
        segment_key=segment_key,
        tactic_name=selected_name,
        is_control=False,
        sampled_values=sampled,
        fallback_reason=fallback_reason,
        selected_at=now,
    )


def _reward_to_pseudo_probability(reward: float, scale: float = REWARD_SCALE) -> float:
    return 1.0 / (1.0 + math.exp(-reward / scale))


def update_state(db: Session, segment_key: str, tactic_name: str, reward: float) -> BanditState:
    """Update alpha/beta for (segment, tactic) from an observed reward.

    Control-group accounts are excluded from bandit updates by callers
    (their reward is always 0 and they exist only to calibrate baselines);
    this function additionally refuses to update the control arm itself,
    since it should never accumulate learned state.
    """
    if tactic_name == CONTROL_TACTIC_NAME:
        raise ValueError("no_action is never updated — control accounts don't feed the bandit")

    state = _get_or_create_state(db, segment_key, tactic_name)
    p = _reward_to_pseudo_probability(reward)
    state.alpha += p
    state.beta += 1 - p
    db.flush()
    return state
