"""Deterministic synthetic failed-payment account generator.

Every record is built as a validated `SyntheticAccount` (Pydantic) before
it is ever handed to the ORM, so contradictory or out-of-range data can't
reach the database.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from attribution.control_group import assign_control_group
from db.models import Account, AccountType, AmountTier, DeclineReason, Industry, amount_tier_for

DECLINE_REASON_WEIGHTS: dict[DeclineReason, float] = {
    DeclineReason.INSUFFICIENT_FUNDS: 0.40,
    DeclineReason.DO_NOT_HONOR: 0.20,
    DeclineReason.CARD_EXPIRED: 0.17,
    DeclineReason.TECHNICAL_ERROR: 0.13,
    DeclineReason.FRAUD_SUSPECTED: 0.10,
}

INDUSTRY_WEIGHTS: dict[Industry, float] = {
    Industry.SAAS: 0.30,
    Industry.RETAIL: 0.30,
    Industry.LOGISTICS: 0.25,
    Industry.FINANCE: 0.15,
}

B2B_SHARE = 0.60


class SyntheticAccount(BaseModel):
    account_id: str
    account_type: AccountType
    amount: float
    amount_tier: AmountTier
    decline_reason: DeclineReason
    payment_history_score: int = Field(ge=0, le=100)
    days_since_failure: int = Field(ge=0, le=30)
    prior_recovery_attempts: int = Field(ge=0, le=3)
    industry: Optional[Industry] = None
    is_control_group: bool

    @model_validator(mode="after")
    def _validate_invariants(self) -> "SyntheticAccount":
        if self.prior_recovery_attempts > 0 and self.days_since_failure == 0:
            raise ValueError("prior_recovery_attempts > 0 requires days_since_failure > 0")
        if self.account_type == AccountType.B2B and self.industry is None:
            raise ValueError("B2B accounts require an industry")
        if self.account_type == AccountType.B2C and self.industry is not None:
            raise ValueError("B2C accounts must not have an industry")
        if amount_tier_for(self.amount) != self.amount_tier:
            raise ValueError("amount_tier does not match amount")
        return self


def _sample_amount(rng: random.Random) -> float:
    roll = rng.random()
    if roll < 0.50:
        return round(rng.uniform(50, 499), 2)
    if roll < 0.85:
        return round(rng.uniform(500, 5000), 2)
    return round(rng.uniform(5001, 50000), 2)


def _weighted_choice(rng: random.Random, weights: dict) -> "object":
    keys = list(weights.keys())
    values = list(weights.values())
    return rng.choices(keys, weights=values, k=1)[0]


def _deterministic_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def generate_accounts(n: int, seed: int = 42) -> list[SyntheticAccount]:
    rng = random.Random(seed)
    accounts: list[SyntheticAccount] = []

    for _ in range(n):
        is_control_group = assign_control_group(rng)

        account_type = AccountType.B2B if rng.random() < B2B_SHARE else AccountType.B2C
        decline_reason = _weighted_choice(rng, DECLINE_REASON_WEIGHTS)
        amount = _sample_amount(rng)
        payment_history_score = rng.randint(0, 100)
        days_since_failure = rng.randint(0, 30)
        prior_recovery_attempts = 0 if days_since_failure == 0 else rng.randint(0, 3)
        industry = _weighted_choice(rng, INDUSTRY_WEIGHTS) if account_type == AccountType.B2B else None

        account = SyntheticAccount(
            account_id=_deterministic_uuid(rng),
            account_type=account_type,
            amount=amount,
            amount_tier=amount_tier_for(amount),
            decline_reason=decline_reason,
            payment_history_score=payment_history_score,
            days_since_failure=days_since_failure,
            prior_recovery_attempts=prior_recovery_attempts,
            industry=industry,
            is_control_group=is_control_group,
        )
        accounts.append(account)

    return accounts


def to_orm(account: SyntheticAccount) -> Account:
    return Account(
        account_id=account.account_id,
        account_type=account.account_type,
        amount=account.amount,
        amount_tier=account.amount_tier,
        decline_reason=account.decline_reason,
        payment_history_score=account.payment_history_score,
        days_since_failure=account.days_since_failure,
        prior_recovery_attempts=account.prior_recovery_attempts,
        industry=account.industry,
        is_control_group=account.is_control_group,
    )
