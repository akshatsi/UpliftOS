"""SQLAlchemy ORM models for the AI Revenue Recovery System.

Schema is written against generic SQLAlchemy types (no SQLite- or
Postgres-specific constructs) so it is portable between the two without
changes.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    inspect as sa_inspect,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Always a timezone-aware UTC `datetime` on the Python side, on both
    backends. SQLite has no native tz-aware datetime type: SQLAlchemy's
    `DateTime(timezone=True)` stores the value fine but silently hands back
    a naive `datetime` on read, while Postgres hands back an aware one for
    the same column type — so code comparing a loaded timestamp against
    `datetime.now(timezone.utc)` works on Postgres and raises on SQLite.
    This reattaches UTC tzinfo on read so both backends behave identically.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime passed to a UTCDateTime column — attach tzinfo first")
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


def _enum_col(enum_cls: type[enum.Enum]) -> Enum:
    """Store an enum's `.value` (not its `.name`) as a plain VARCHAR + CHECK.

    `native_enum=False` avoids Postgres `CREATE TYPE` / `ALTER TYPE`
    migration pain and keeps SQLite and Postgres byte-for-byte identical.
    """
    return Enum(enum_cls, native_enum=False, validate_strings=True, values_callable=lambda obj: [e.value for e in obj])


class AccountType(str, enum.Enum):
    B2B = "B2B"
    B2C = "B2C"


class AmountTier(str, enum.Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class DeclineReason(str, enum.Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_EXPIRED = "card_expired"
    DO_NOT_HONOR = "do_not_honor"
    FRAUD_SUSPECTED = "fraud_suspected"
    TECHNICAL_ERROR = "technical_error"


class Industry(str, enum.Enum):
    SAAS = "saas"
    LOGISTICS = "logistics"
    RETAIL = "retail"
    FINANCE = "finance"


class ActionStatus(str, enum.Enum):
    ACTIVE = "active"
    RECOVERED = "recovered"
    EXPIRED = "expired"


def amount_tier_for(amount: float) -> AmountTier:
    if amount < 500:
        return AmountTier.LOW
    if amount <= 5000:
        return AmountTier.MID
    return AmountTier.HIGH


def segment_key(decline_reason: str, amount_tier: str, account_type: str) -> str:
    return f"{decline_reason}|{amount_tier}|{account_type}"


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    account_type: Mapped[AccountType] = mapped_column(_enum_col(AccountType), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    amount_tier: Mapped[AmountTier] = mapped_column(_enum_col(AmountTier), nullable=False)
    decline_reason: Mapped[DeclineReason] = mapped_column(_enum_col(DeclineReason), nullable=False)
    payment_history_score: Mapped[int] = mapped_column(Integer, nullable=False)
    days_since_failure: Mapped[int] = mapped_column(Integer, nullable=False)
    prior_recovery_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    industry: Mapped[Industry | None] = mapped_column(_enum_col(Industry), nullable=True)
    is_control_group: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, nullable=False)

    recovery_actions: Mapped[list["RecoveryAction"]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("payment_history_score >= 0 AND payment_history_score <= 100", name="ck_payment_history_score_range"),
        CheckConstraint("days_since_failure >= 0 AND days_since_failure <= 30", name="ck_days_since_failure_range"),
        CheckConstraint("prior_recovery_attempts >= 0 AND prior_recovery_attempts <= 3", name="ck_prior_recovery_attempts_range"),
        CheckConstraint(
            "NOT (prior_recovery_attempts > 0 AND days_since_failure = 0)",
            name="ck_no_prior_attempts_on_day_zero",
        ),
        CheckConstraint(
            "(account_type = 'B2B' AND industry IS NOT NULL) OR (account_type = 'B2C' AND industry IS NULL)",
            name="ck_industry_matches_account_type",
        ),
    )

    @validates("is_control_group")
    def _immutable_is_control_group(self, key: str, value: bool) -> bool:
        # Enforced here, not just at the routing layer: once an account row
        # is persisted, no code path may flip its control-group assignment.
        state = sa_inspect(self)
        if state.persistent and value != self.is_control_group:
            raise ValueError("is_control_group is immutable once persisted")
        return value

    @property
    def segment(self) -> str:
        return segment_key(self.decline_reason.value, self.amount_tier.value, self.account_type.value)


class RecoveryAction(Base):
    __tablename__ = "recovery_actions"

    action_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    account_id: Mapped[str] = mapped_column(String(36), ForeignKey("accounts.account_id"), nullable=False, index=True)
    segment_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tactic_name: Mapped[str] = mapped_column(String(64), nullable=False)
    tactic_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discount_offered: Mapped[float | None] = mapped_column(Float, nullable=True)
    sampled_values: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    message_drafted: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    guardrail_approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    guardrail_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ActionStatus] = mapped_column(_enum_col(ActionStatus), nullable=False, default=ActionStatus.ACTIVE)
    deployed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, nullable=False)

    account: Mapped["Account"] = relationship(back_populates="recovery_actions")
    outcome: Mapped["Outcome | None"] = relationship(
        back_populates="action", uselist=False, cascade="all, delete-orphan"
    )


class Outcome(Base):
    __tablename__ = "outcomes"

    outcome_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    action_id: Mapped[str] = mapped_column(String(36), ForeignKey("recovery_actions.action_id"), nullable=False, unique=True)
    account_id: Mapped[str] = mapped_column(String(36), ForeignKey("accounts.account_id"), nullable=False, index=True)
    recovered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    amount_recovered: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recovered_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    uplift_indicator: Mapped[float | None] = mapped_column(Float, nullable=True)
    reported_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    action: Mapped["RecoveryAction"] = relationship(back_populates="outcome")


class BanditState(Base):
    __tablename__ = "bandit_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    tactic_name: Mapped[str] = mapped_column(String(64), nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    beta: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("segment_key", "tactic_name", name="uq_bandit_state_segment_tactic"),)


class SegmentBaseline(Base):
    __tablename__ = "segment_baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    control_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    control_recovered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_recovery_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_start: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, onupdate=_utcnow, nullable=False)
