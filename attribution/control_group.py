"""Control-group assignment and routing.

This is the single source of truth for the 12% control-group split, so
that assignment probability isn't duplicated (and able to drift) between
the synthetic generator and any real ingestion path added later.
"""

from __future__ import annotations

import random

from db.models import Account, ActionStatus, RecoveryAction
from tactics.registry import CONTROL_TACTIC_NAME

CONTROL_GROUP_SHARE = 0.12


def assign_control_group(rng: random.Random) -> bool:
    """Pre-check step: call this before generating/reading any other
    account context. The result must never depend on decline_reason,
    amount_tier, account_type, or any other field — control assignment is
    random and happens before context is evaluated.
    """
    return rng.random() < CONTROL_GROUP_SHARE


def is_control(account: Account) -> bool:
    return account.is_control_group


def build_control_action(account: Account) -> RecoveryAction:
    """The recovery action for a control-group account: no_action, with no
    LLM agents and no bandit involved in producing it.
    """
    if not account.is_control_group:
        raise ValueError(f"account {account.account_id} is not in the control group")

    return RecoveryAction(
        account_id=account.account_id,
        segment_key=account.segment,
        tactic_name=CONTROL_TACTIC_NAME,
        tactic_cost=0.0,
        discount_offered=None,
        sampled_values={},
        message_drafted=None,
        guardrail_approved=None,
        guardrail_reason=None,
        status=ActionStatus.ACTIVE,
    )
