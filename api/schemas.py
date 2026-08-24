"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from db.models import AccountType, ActionStatus, AmountTier, DeclineReason, Industry


class AccountSummary(BaseModel):
    model_config = {"from_attributes": True}

    account_id: str
    account_type: AccountType
    amount: float
    amount_tier: AmountTier
    decline_reason: DeclineReason
    payment_history_score: int
    days_since_failure: int
    prior_recovery_attempts: int
    industry: Optional[Industry]
    is_control_group: bool
    created_at: datetime


class RecoveryActionSummary(BaseModel):
    model_config = {"from_attributes": True}

    action_id: str
    segment_key: str
    tactic_name: str
    tactic_cost: float
    discount_offered: Optional[float]
    guardrail_approved: Optional[bool]
    guardrail_reason: Optional[str]
    fallback_reason: Optional[str]
    status: ActionStatus
    deployed_at: datetime


class AccountDetail(AccountSummary):
    recovery_actions: list[RecoveryActionSummary] = []


class AccountListResponse(BaseModel):
    accounts: list[AccountSummary]
    total: int
    page: int
    page_size: int


class RecoverResponse(BaseModel):
    tactic_selected: str
    message_drafted: Optional[dict]
    guardrail_approved: Optional[bool]
    action_logged_at: datetime


class ControlGroupResponse(BaseModel):
    status: Literal["control_group"] = "control_group"
    action: Literal["no_action"] = "no_action"


class OutcomeRequest(BaseModel):
    recovered: bool
    amount_recovered: float = 0.0
    recovered_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _recovered_implies_amount_and_timestamp(self) -> "OutcomeRequest":
        if self.recovered and (self.amount_recovered <= 0 or self.recovered_at is None):
            raise ValueError("recovered=true requires a positive amount_recovered and a recovered_at timestamp")
        return self


class OutcomeResponse(BaseModel):
    account_id: str
    outcome_id: str
    recovered: bool
    amount_recovered: float
    reward: Optional[float]
    reward_finalized: bool


class BanditArmState(BaseModel):
    segment_key: str
    tactic_name: str
    alpha: float
    beta: float
    estimated_win_rate: float


class BanditStateResponse(BaseModel):
    arms: list[BanditArmState]


class SegmentBaselineSummary(BaseModel):
    model_config = {"from_attributes": True}

    segment_key: str
    control_count: int
    control_recovered_count: int
    baseline_recovery_rate: Optional[float]
    window_start: Optional[datetime]
    window_end: Optional[datetime]


class BaselinesResponse(BaseModel):
    baselines: list[SegmentBaselineSummary]


class ErrorResponse(BaseModel):
    error: str
    code: str
