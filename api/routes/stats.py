"""GET /stats/tactic-performance, /stats/funnel, /stats/reward-over-time,
/stats/attribution -- the dashboard's own aggregation endpoints. These
used to be pandas queries run directly inside the (now removed) Streamlit
script; they're real API endpoints now so the frontend (or any other
consumer) can read them over HTTP instead of querying the DB directly.

/stats/attribution is distinct from GET /attribution/baselines: that
endpoint returns only the cached control-group baseline (its documented,
existing contract, left unchanged); this one also joins in live treated-
group recovery rates and uplift, matching what the dashboard actually
needs to show and what the old Streamlit script computed client-side.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from api.schemas import (
    AttributionResponse,
    AttributionRow,
    FunnelResponse,
    FunnelStage,
    RewardOverTimeResponse,
    RewardPoint,
    TacticPerformanceResponse,
    TacticPerformanceRow,
)
from attribution.uplift import MIN_CONTROL_ACCOUNTS, get_cached_baselines
from db.models import Account, Outcome, RecoveryAction
from db.session import get_db
from tactics.registry import TACTICS

router = APIRouter(tags=["stats"])


@router.get("/stats/tactic-performance", response_model=TacticPerformanceResponse)
def tactic_performance(db: Session = Depends(get_db)) -> TacticPerformanceResponse:
    # Recovered is a real boolean column, not an integer -- averaging it
    # directly works on SQLite (booleans are integer-backed there) but not
    # on Postgres, so it's cast to float explicitly here for portability.
    recovered_as_float = case((Outcome.recovered.is_(True), 1.0), (Outcome.recovered.is_(False), 0.0))

    rows = db.execute(
        select(
            RecoveryAction.tactic_name,
            func.count().label("deployments"),
            func.avg(recovered_as_float).label("recovery_rate"),
            func.avg(Outcome.reward).label("avg_reward"),
            func.avg(RecoveryAction.tactic_cost).label("avg_cost"),
            func.coalesce(func.sum(Outcome.amount_recovered), 0.0).label("total_recovered"),
            func.coalesce(func.sum(RecoveryAction.tactic_cost), 0.0).label("total_cost"),
        )
        .select_from(RecoveryAction)
        .outerjoin(Outcome, Outcome.action_id == RecoveryAction.action_id)
        .group_by(RecoveryAction.tactic_name)
    ).all()

    by_name = {
        r.tactic_name: TacticPerformanceRow(
            tactic_name=r.tactic_name,
            deployments=r.deployments,
            recovery_rate=r.recovery_rate,
            avg_reward=r.avg_reward,
            avg_cost=r.avg_cost or 0.0,
            net_value=r.total_recovered - r.total_cost,
        )
        for r in rows
    }

    all_tactics = [
        by_name.get(t.name)
        or TacticPerformanceRow(tactic_name=t.name, deployments=0, recovery_rate=None, avg_reward=None, avg_cost=0.0, net_value=0.0)
        for t in TACTICS
    ]
    all_tactics.sort(key=lambda r: r.deployments, reverse=True)
    return TacticPerformanceResponse(tactics=all_tactics)


@router.get("/stats/funnel", response_model=FunnelResponse)
def funnel(db: Session = Depends(get_db)) -> FunnelResponse:
    accounts_failed = db.execute(select(func.count()).select_from(Account)).scalar_one()

    triggered_rows = db.execute(
        select(RecoveryAction.guardrail_approved)
        .join(Account, Account.account_id == RecoveryAction.account_id)
        .where(Account.is_control_group.is_(False))
    ).all()

    recovered_n = db.execute(
        select(func.count())
        .select_from(Outcome)
        .join(RecoveryAction, RecoveryAction.action_id == Outcome.action_id)
        .join(Account, Account.account_id == RecoveryAction.account_id)
        .where(Account.is_control_group.is_(False), Outcome.recovered.is_(True))
    ).scalar_one()

    approved_n = sum(1 for (approved,) in triggered_rows if approved is True)

    stages = [
        FunnelStage(stage="Accounts failed", count=accounts_failed),
        FunnelStage(stage="Recovery triggered", count=len(triggered_rows)),
        FunnelStage(stage="Guardrail approved (clean)", count=approved_n),
        FunnelStage(stage="Recovered within window", count=recovered_n),
    ]
    return FunnelResponse(stages=stages)


@router.get("/stats/reward-over-time", response_model=RewardOverTimeResponse)
def reward_over_time(db: Session = Depends(get_db)) -> RewardOverTimeResponse:
    rows = db.execute(
        select(Outcome.reported_at, RecoveryAction.tactic_name, Outcome.reward)
        .join(RecoveryAction, RecoveryAction.action_id == Outcome.action_id)
        .where(Outcome.reward.is_not(None))
    ).all()

    # Date-bucketed in Python, not SQL -- date-truncation functions aren't
    # portable between SQLite and Postgres.
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for reported_at, tactic_name, reward in rows:
        buckets[(reported_at.date().isoformat(), tactic_name)].append(reward)

    points = [
        RewardPoint(date=date_key, tactic_name=tactic_name, avg_reward=sum(vals) / len(vals))
        for (date_key, tactic_name), vals in sorted(buckets.items())
    ]
    return RewardOverTimeResponse(points=points)


@router.get("/stats/attribution", response_model=AttributionResponse)
def attribution(db: Session = Depends(get_db)) -> AttributionResponse:
    baseline_by_segment = {b.segment_key: b for b in get_cached_baselines(db)}

    treated_as_float = case((Outcome.recovered.is_(True), 1.0), (Outcome.recovered.is_(False), 0.0))
    treated_rows = db.execute(
        select(
            RecoveryAction.segment_key,
            func.count().label("treated_count"),
            func.avg(treated_as_float).label("treated_recovery_rate"),
        )
        .select_from(RecoveryAction)
        .join(Outcome, Outcome.action_id == RecoveryAction.action_id)
        .join(Account, Account.account_id == RecoveryAction.account_id)
        .where(Account.is_control_group.is_(False))
        .group_by(RecoveryAction.segment_key)
    ).all()
    treated_by_segment = {r.segment_key: r for r in treated_rows}

    rows = []
    for seg in sorted(set(baseline_by_segment) | set(treated_by_segment)):
        b = baseline_by_segment.get(seg)
        t = treated_by_segment.get(seg)
        control_count = b.control_count if b else 0
        control_rate = b.baseline_recovery_rate if b else None
        treated_count = t.treated_count if t else 0
        treated_rate = t.treated_recovery_rate if t else None
        uplift_pct = (treated_rate - control_rate) * 100 if (treated_rate is not None and control_rate is not None) else None
        confidence = "reliable" if control_count >= MIN_CONTROL_ACCOUNTS else f"low data (n={control_count})"
        rows.append(
            AttributionRow(
                segment_key=seg,
                control_count=control_count,
                control_recovery_rate=control_rate,
                treated_count=treated_count,
                treated_recovery_rate=treated_rate,
                uplift_pct=uplift_pct,
                confidence=confidence,
            )
        )
    return AttributionResponse(rows=rows)
