"""The 10 recovery tactics as structured objects.

Eligibility is a hard gate: `eligible_tactics()` excludes ineligible tactics
from the pool entirely, so a downstream bandit can only ever sample among
tactics that are actually allowed for that account's context — there is no
code path where e.g. `account_manager_outreach` is merely down-weighted for
a B2C account instead of excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from db.models import Account, AccountType, AmountTier, DeclineReason

DEFAULT_DISCOUNT_RATE = 0.10

CONTROL_TACTIC_NAME = "no_action"
FALLBACK_TACTIC_NAME = "soft_nudge_email"


class Channel(str, Enum):
    GATEWAY = "gateway"
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    HUMAN_ESCALATION = "human_escalation"
    NONE = "none"


CostFn = Callable[[Account, "float | None"], float]


def _fixed_cost(base_cost: float) -> CostFn:
    def _fn(account: Account, discount_offered: float | None = None) -> float:
        return base_cost

    return _fn


def _discount_email_cost(account: Account, discount_offered: float | None = None) -> float:
    if discount_offered is not None:
        return discount_offered
    return round(account.amount * DEFAULT_DISCOUNT_RATE, 2)


@dataclass(frozen=True)
class Tactic:
    arm_index: int
    name: str
    channel: Channel
    base_cost: float
    eligibility: Callable[[Account], bool]
    cost_fn: CostFn

    def is_eligible(self, account: Account) -> bool:
        return self.eligibility(account)

    def resolve_cost(self, account: Account, discount_offered: float | None = None) -> float:
        return self.cost_fn(account, discount_offered)


TACTICS: tuple[Tactic, ...] = (
    Tactic(1, "immediate_retry", Channel.GATEWAY, 0.0, lambda a: True, _fixed_cost(0.0)),
    Tactic(
        2,
        "payday_retry",
        Channel.GATEWAY,
        0.0,
        lambda a: a.account_type == AccountType.B2C and a.decline_reason == DeclineReason.INSUFFICIENT_FUNDS,
        _fixed_cost(0.0),
    ),
    Tactic(3, "alternate_gateway_retry", Channel.GATEWAY, 15.0, lambda a: True, _fixed_cost(15.0)),
    Tactic(4, "soft_nudge_email", Channel.EMAIL, 0.0, lambda a: True, _fixed_cost(0.0)),
    Tactic(5, "discount_email", Channel.EMAIL, 0.0, lambda a: True, _discount_email_cost),
    Tactic(6, "sms_reminder", Channel.SMS, 2.0, lambda a: a.account_type == AccountType.B2C, _fixed_cost(2.0)),
    Tactic(7, "whatsapp_nudge", Channel.WHATSAPP, 3.0, lambda a: a.account_type == AccountType.B2C, _fixed_cost(3.0)),
    Tactic(
        8,
        "payment_plan_offer",
        Channel.EMAIL,
        0.0,
        lambda a: a.amount_tier == AmountTier.HIGH,
        _fixed_cost(0.0),
    ),
    Tactic(
        9,
        "account_manager_outreach",
        Channel.HUMAN_ESCALATION,
        200.0,
        lambda a: a.account_type == AccountType.B2B,
        _fixed_cost(200.0),
    ),
    Tactic(10, CONTROL_TACTIC_NAME, Channel.NONE, 0.0, lambda a: True, _fixed_cost(0.0)),
)

_BY_NAME: dict[str, Tactic] = {t.name: t for t in TACTICS}
_BY_ARM_INDEX: dict[int, Tactic] = {t.arm_index: t for t in TACTICS}


def get_tactic(name: str) -> Tactic:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown tactic: {name!r}") from None


def get_tactic_by_arm_index(arm_index: int) -> Tactic:
    try:
        return _BY_ARM_INDEX[arm_index]
    except KeyError:
        raise KeyError(f"unknown arm index: {arm_index!r}") from None


def eligible_tactics(account: Account, *, include_control: bool = False) -> list[Tactic]:
    """Tactics allowed for this account's context.

    `include_control` defaults to False: normal (non-control) recovery flows
    should never even see `no_action` in their eligible pool. Control-group
    accounts bypass this function entirely and are assigned `no_action`
    directly by the bandit/routing layer.
    """
    return [
        t
        for t in TACTICS
        if t.is_eligible(account) and (include_control or t.name != CONTROL_TACTIC_NAME)
    ]
