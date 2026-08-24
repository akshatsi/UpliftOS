"""Segment-level baseline recovery rates and per-outcome uplift.

`get_baseline`/`get_uplift` always compute live from `outcomes` /
`recovery_actions` / `accounts` — the reward pipeline needs the true
current rate, not a cache that might lag within the same batch run.
`SegmentBaseline` is a separate materialized cache, refreshed by
`refresh_segment_baseline`, that exists purely so `GET /attribution/baselines`
can be read cheaply without re-joining on every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Account, AccountType, AmountTier, DeclineReason, Outcome, RecoveryAction, SegmentBaseline
from db.models import segment_key as _segment_key

ROLLING_WINDOW_DAYS = 30
MIN_CONTROL_ACCOUNTS = 5
NEUTRAL_UPLIFT = 0.5

ALL_SEGMENT_KEYS: list[str] = [
    _segment_key(dr.value, at.value, act.value) for dr, at, act in product(DeclineReason, AmountTier, AccountType)
]


@dataclass
class BaselineResult:
    segment_key: str
    control_count: int
    control_recovered_count: int
    baseline_recovery_rate: float | None  # None until MIN_CONTROL_ACCOUNTS is reached
    is_reliable: bool
    window_start: datetime
    window_end: datetime


def compute_baseline(db: Session, segment_key: str, *, now: datetime | None = None) -> BaselineResult:
    now = now or datetime.now(timezone.utc)
    window_start = now - timedelta(days=ROLLING_WINDOW_DAYS)

    recovered_flags = list(
        db.execute(
            select(Outcome.recovered)
            .join(RecoveryAction, Outcome.action_id == RecoveryAction.action_id)
            .join(Account, Outcome.account_id == Account.account_id)
            .where(
                Account.is_control_group.is_(True),
                RecoveryAction.segment_key == segment_key,
                RecoveryAction.deployed_at >= window_start,
                RecoveryAction.deployed_at <= now,
            )
        ).scalars()
    )

    control_count = len(recovered_flags)
    control_recovered_count = sum(1 for r in recovered_flags if r)
    is_reliable = control_count >= MIN_CONTROL_ACCOUNTS
    baseline_rate = (control_recovered_count / control_count) if is_reliable else None

    return BaselineResult(
        segment_key=segment_key,
        control_count=control_count,
        control_recovered_count=control_recovered_count,
        baseline_recovery_rate=baseline_rate,
        is_reliable=is_reliable,
        window_start=window_start,
        window_end=now,
    )


def get_baseline(db: Session, segment_key: str) -> BaselineResult:
    return compute_baseline(db, segment_key)


def get_uplift(db: Session, segment_key: str, recovered: bool) -> float:
    """Per-outcome uplift_indicator: observed indicator minus the segment's
    control baseline, i.e. how much of this recovery is credited beyond
    what would have happened anyway. Falls back to a neutral 0.5 until the
    segment has enough control data to trust its baseline.
    """
    baseline = get_baseline(db, segment_key)
    if not baseline.is_reliable:
        return NEUTRAL_UPLIFT
    observed = 1.0 if recovered else 0.0
    return observed - baseline.baseline_recovery_rate


def refresh_segment_baseline(db: Session, segment_key: str, *, now: datetime | None = None) -> SegmentBaseline:
    result = compute_baseline(db, segment_key, now=now)
    row = db.execute(select(SegmentBaseline).where(SegmentBaseline.segment_key == segment_key)).scalar_one_or_none()
    if row is None:
        row = SegmentBaseline(segment_key=segment_key)
        db.add(row)

    row.control_count = result.control_count
    row.control_recovered_count = result.control_recovered_count
    row.baseline_recovery_rate = result.baseline_recovery_rate
    row.window_start = result.window_start
    row.window_end = result.window_end
    db.flush()
    return row


def refresh_all_segment_baselines(db: Session, *, now: datetime | None = None) -> list[SegmentBaseline]:
    return [refresh_segment_baseline(db, seg, now=now) for seg in ALL_SEGMENT_KEYS]


def get_cached_baselines(db: Session) -> list[SegmentBaseline]:
    return list(db.execute(select(SegmentBaseline)).scalars())
