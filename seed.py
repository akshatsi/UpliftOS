"""Seed script: creates tables, generates synthetic accounts, and simulates
30 days of outcomes for 60% of them so the bandit has data to learn from.

    python seed.py

Not idempotent against an already-seeded DB — account IDs are deterministic
(seed=42), so a second run would try to re-insert the same 500 accounts and
hit the unique constraint. Delete the DB file first to reseed:

    rm revenue_recovery.db && python seed.py

`guardrail_approved` is a placeholder (True) since seeding doesn't run the
real drafting/guardrail agents; reward/uplift_indicator are left NULL since
those are the daily batch updater's job, never computed at seed time.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from attribution.control_group import build_control_action
from bandit.reward import RECOVERY_WINDOW_DAYS
from bandit.thompson_sampling import select_arm
from data.synthetic_generator import generate_accounts, to_orm
from db.models import Account, ActionStatus, DeclineReason, Outcome, RecoveryAction
from db.session import init_db, session_scope
from tactics.registry import get_tactic

load_dotenv()

SEED = 42
NUM_ACCOUNTS = 500
SIMULATION_SHARE = 0.60


def _recovery_probability(account: Account, tactic_name: str) -> float:
    p = 0.25
    p += (account.payment_history_score / 100 - 0.5) * 0.30
    p += {
        DeclineReason.FRAUD_SUSPECTED: -0.15,
        DeclineReason.TECHNICAL_ERROR: 0.10,
        DeclineReason.DO_NOT_HONOR: -0.05,
        DeclineReason.CARD_EXPIRED: 0.0,
        DeclineReason.INSUFFICIENT_FUNDS: 0.0,
    }[account.decline_reason]
    if tactic_name != "no_action":
        p += 0.12
    return max(0.02, min(0.95, p))


def _simulate_recovery_action(
    db: Session, rng: random.Random, account: Account, now: datetime
) -> tuple[RecoveryAction, Outcome | None]:
    deployed_at = now - timedelta(days=rng.randint(0, 30))
    days_elapsed = (now - deployed_at).days
    window_closed = days_elapsed >= RECOVERY_WINDOW_DAYS

    if account.is_control_group:
        action = build_control_action(account)
        tactic_name = action.tactic_name
    else:
        selection = select_arm(db, account, rng=rng)
        tactic = get_tactic(selection.tactic_name)
        tactic_cost = tactic.resolve_cost(account)
        action = RecoveryAction(
            account_id=account.account_id,
            segment_key=selection.segment_key,
            tactic_name=tactic.name,
            tactic_cost=tactic_cost,
            discount_offered=tactic_cost if tactic.name == "discount_email" else None,
            sampled_values=selection.sampled_values,
            guardrail_approved=True,  # agents/guardrail_agent.py doesn't exist yet
        )
        tactic_name = tactic.name

    action.deployed_at = deployed_at

    p_recovered = _recovery_probability(account, tactic_name)
    will_recover = rng.random() < p_recovered
    recovered_at = None
    if will_recover:
        offset_days = rng.randint(0, RECOVERY_WINDOW_DAYS - 1)
        candidate = deployed_at + timedelta(days=offset_days)
        if candidate <= now:
            recovered_at = candidate
        else:
            will_recover = False  # would recover later than "now" — not observed yet

    outcome = None
    if recovered_at is not None:
        action.status = ActionStatus.RECOVERED
        outcome = Outcome(
            recovered=True,
            amount_recovered=account.amount,
            recovered_at=recovered_at,
            reported_at=recovered_at,
        )
    elif window_closed:
        action.status = ActionStatus.EXPIRED
        outcome = Outcome(recovered=False, amount_recovered=0.0, recovered_at=None, reported_at=now)
    else:
        action.status = ActionStatus.ACTIVE  # window still open, outcome pending

    return action, outcome


def main() -> None:
    init_db()

    with session_scope() as db:
        existing = db.query(Account).count()
    if existing:
        print(
            f"Database already has {existing} accounts — seed.py isn't idempotent "
            "(account IDs are deterministic, so re-running would collide on the "
            "unique constraint). Delete the DB file first if you want to reseed:\n"
            "  rm revenue_recovery.db && python seed.py"
        )
        sys.exit(1)

    now = datetime.now(timezone.utc)
    rng = random.Random(SEED)

    synthetic_accounts = generate_accounts(NUM_ACCOUNTS, seed=SEED)
    accounts = [to_orm(a) for a in synthetic_accounts]

    sim_pool = list(accounts)
    rng.shuffle(sim_pool)
    sim_count = int(len(sim_pool) * SIMULATION_SHARE)
    accounts_to_simulate = {a.account_id for a in sim_pool[:sim_count]}

    actions: list[RecoveryAction] = []
    outcomes: list[Outcome] = []

    with session_scope() as db:
        db.add_all(accounts)
        db.flush()

        for account in accounts:
            if account.account_id not in accounts_to_simulate:
                continue
            action, outcome = _simulate_recovery_action(db, rng, account, now)
            db.add(action)
            db.flush()  # populate action.action_id for the outcome FK
            actions.append(action)
            if outcome is not None:
                outcome.action_id = action.action_id
                outcome.account_id = account.account_id
                db.add(outcome)
                outcomes.append(outcome)

    _print_summary(accounts, actions, outcomes)


def _print_summary(accounts: list[Account], actions: list[RecoveryAction], outcomes: list[Outcome]) -> None:
    control = [a for a in accounts if a.is_control_group]
    treated = [a for a in accounts if not a.is_control_group]

    print(f"Seeded {len(accounts)} accounts ({len(control)} control / {len(treated)} treated)")
    print(f"  account_type:   {dict(Counter(a.account_type.value for a in accounts))}")
    print(f"  amount_tier:    {dict(Counter(a.amount_tier.value for a in accounts))}")
    print(f"  decline_reason: {dict(Counter(a.decline_reason.value for a in accounts))}")

    print(f"\nSimulated {len(actions)} recovery actions, {len(outcomes)} outcomes recorded")
    print(f"  status:      {dict(Counter(a.status.value for a in actions))}")
    print(f"  tactic_name: {dict(Counter(a.tactic_name for a in actions))}")

    outcomes_by_action = {o.action_id: o for o in outcomes}
    accounts_by_id = {a.account_id: a for a in accounts}

    def recovery_rate(is_control: bool) -> str:
        relevant = [
            outcomes_by_action[a.action_id]
            for a in actions
            if a.action_id in outcomes_by_action and accounts_by_id[a.account_id].is_control_group == is_control
        ]
        if not relevant:
            return "n/a (no closed-window outcomes yet)"
        rate = sum(o.recovered for o in relevant) / len(relevant)
        return f"{rate:.1%} ({len(relevant)} resolved)"

    print(f"\nControl recovery rate: {recovery_rate(True)}")
    print(f"Treated recovery rate: {recovery_rate(False)}")


if __name__ == "__main__":
    main()
