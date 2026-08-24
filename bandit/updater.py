"""Daily batch job: processes resolved, window-closed outcomes, updates
bandit state, and refreshes segment baselines.

Idempotent via `processed_at` — a re-run only ever picks up outcomes that
are still unprocessed. `run_daily_update` itself never commits; the whole
batch is written by its caller's transaction (`run_scheduled_update` uses
`session_scope()`), so a crash mid-batch rolls back everything rather than
leaving reward computed for some outcomes but not others.

Actions whose recovery window has closed with no outcome ever recorded are
NOT auto-expired into a synthetic `recovered=False` outcome here — this
module only processes outcomes that already exist; deciding what happens to
an account nobody ever reported an outcome for belongs to whatever calls
`POST /outcomes` (or a separate expiry sweep), not the batch updater.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.orm import Session

from attribution.uplift import refresh_segment_baseline
from bandit.reward import RECOVERY_WINDOW_DAYS, apply_reward
from bandit.thompson_sampling import update_state
from db.models import Outcome, RecoveryAction
from db.session import session_scope

logger = logging.getLogger("bandit.updater")

BANDIT_UPDATE_HOUR = int(os.getenv("BANDIT_UPDATE_HOUR", "2"))


@dataclass
class UpdateSummary:
    run_at: datetime
    outcomes_processed: int
    control_outcomes_processed: int
    segments_bandit_updated: int
    segments_baseline_refreshed: int
    reward_min: float | None
    reward_max: float | None
    reward_mean: float | None


def _processable_outcomes(db: Session, now: datetime) -> list[Outcome]:
    """Unprocessed outcomes whose recovery action's window has closed."""
    cutoff = now - timedelta(days=RECOVERY_WINDOW_DAYS)
    return list(
        db.execute(
            select(Outcome)
            .join(RecoveryAction, Outcome.action_id == RecoveryAction.action_id)
            .where(Outcome.processed_at.is_(None), RecoveryAction.deployed_at <= cutoff)
        ).scalars()
    )


def run_daily_update(db: Session, *, now: datetime | None = None) -> UpdateSummary:
    now = now or datetime.now(timezone.utc)
    outcomes = _processable_outcomes(db, now)

    control_count = 0
    bandit_segments: set[str] = set()
    touched_segments: set[str] = set()
    non_control_rewards: list[float] = []

    for outcome in outcomes:
        action = outcome.action
        account = action.account

        result = apply_reward(db, outcome, now=now)
        outcome.processed_at = now
        touched_segments.add(action.segment_key)

        if account.is_control_group:
            control_count += 1
        else:
            update_state(db, action.segment_key, action.tactic_name, result.reward)
            bandit_segments.add(action.segment_key)
            non_control_rewards.append(result.reward)

    for segment in touched_segments:
        refresh_segment_baseline(db, segment, now=now)

    db.flush()

    summary = UpdateSummary(
        run_at=now,
        outcomes_processed=len(outcomes),
        control_outcomes_processed=control_count,
        segments_bandit_updated=len(bandit_segments),
        segments_baseline_refreshed=len(touched_segments),
        reward_min=min(non_control_rewards) if non_control_rewards else None,
        reward_max=max(non_control_rewards) if non_control_rewards else None,
        reward_mean=(sum(non_control_rewards) / len(non_control_rewards)) if non_control_rewards else None,
    )
    logger.info(
        "bandit_daily_update run_at=%s outcomes_processed=%d control_outcomes=%d "
        "segments_bandit_updated=%d segments_baseline_refreshed=%d reward_min=%s reward_max=%s reward_mean=%s",
        now.isoformat(),
        summary.outcomes_processed,
        summary.control_outcomes_processed,
        summary.segments_bandit_updated,
        summary.segments_baseline_refreshed,
        summary.reward_min,
        summary.reward_max,
        summary.reward_mean,
    )
    return summary


def run_scheduled_update() -> UpdateSummary:
    """APScheduler entry point: owns its own transaction so the whole batch
    commits or rolls back atomically.
    """
    with session_scope() as db:
        return run_daily_update(db)


def start_scheduler(hour: int = BANDIT_UPDATE_HOUR) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=timezone.utc)
    scheduler.add_job(run_scheduled_update, CronTrigger(hour=hour, minute=0), id="daily_bandit_update", replace_existing=True)
    scheduler.start()
    return scheduler
