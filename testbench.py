"""Testbench: exercises the full pipeline — control bypass, bandit arm
selection, the real drafting/guardrail agents (actual Groq API calls, not
mocked), guardrail-triggered fallback, persistence, outcome reporting, and
reward finalization — against a curated set of realistic account scenarios.

This is deliberately not part of the pytest suite: pytest proves correctness
against mocked LLM calls on constructed edge cases; this proves the system
behaves sensibly on realistic data with the real model actually talking.
Costs a small amount of real API usage and takes a minute or two to run.

Uses its own dedicated DB file (testbench.db, gitignored via the existing
*.db rule) — never revenue_recovery.db, so running this never disturbs
your seeded dev data.

    python testbench.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

import db.session as db_session_mod
from agents.crew import CrewExhaustedError, run_crew
from agents.llm_client import LLMCallError
from attribution.control_group import build_control_action
from attribution.uplift import refresh_segment_baseline
from bandit.reward import apply_reward, window_closed
from bandit.thompson_sampling import select_arm, update_state
from db.models import (
    Account,
    AccountType,
    AmountTier,
    Base,
    DeclineReason,
    Industry,
    Outcome,
    RecoveryAction,
    amount_tier_for,
)
from tactics.registry import get_tactic

TESTBENCH_DB_PATH = Path(__file__).resolve().parent / "testbench.db"

# Discovered by actually running this: firing scenarios back-to-back tripped
# Groq's on-demand tier TPM limit (8000/min) after ~4 scenarios, since each
# one is 2 real calls (draft + guardrail). This spacing keeps a full run
# comfortably under that budget instead of relying on the retry/backoff
# (which is for genuine transient hits, not a self-inflicted pacing problem).
SCENARIO_PACING_SECONDS = 8


def _setup_testbench_db():
    if TESTBENCH_DB_PATH.exists():
        TESTBENCH_DB_PATH.unlink()
    engine = create_engine(f"sqlite:///{TESTBENCH_DB_PATH}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    # Point the shared db.session module at this DB, so every module that
    # calls session_scope()/get_db() (attribution, bandit, agents) uses it
    # transparently, exactly like tests/conftest.py does for pytest.
    db_session_mod.engine = engine
    db_session_mod.SessionLocal = session_local
    return session_local


@dataclass
class Scenario:
    label: str
    account_type: AccountType
    amount: float
    decline_reason: DeclineReason
    payment_history_score: int
    days_since_failure: int
    prior_recovery_attempts: int = 0
    industry: Industry | None = None
    is_control_group: bool = False


SCENARIOS = [
    Scenario(
        "B2C subscriber, insufficient funds, low amount",
        AccountType.B2C, 299.0, DeclineReason.INSUFFICIENT_FUNDS,
        payment_history_score=72, days_since_failure=2, prior_recovery_attempts=1,
    ),
    Scenario(
        "B2C shopper, card expired, mid amount",
        AccountType.B2C, 2450.0, DeclineReason.CARD_EXPIRED,
        payment_history_score=55, days_since_failure=4, prior_recovery_attempts=0,
    ),
    Scenario(
        "B2B SaaS customer, bank declined (do_not_honor), mid amount",
        AccountType.B2B, 3200.0, DeclineReason.DO_NOT_HONOR,
        payment_history_score=60, days_since_failure=3, prior_recovery_attempts=1,
        industry=Industry.SAAS,
    ),
    Scenario(
        "B2B retailer, fraud flagged, high amount",
        AccountType.B2B, 15000.0, DeclineReason.FRAUD_SUSPECTED,
        payment_history_score=40, days_since_failure=6, prior_recovery_attempts=2,
        industry=Industry.RETAIL,
    ),
    Scenario(
        "B2B logistics, insufficient funds, just under the escalation floor (₹500)",
        AccountType.B2B, 450.0, DeclineReason.INSUFFICIENT_FUNDS,
        payment_history_score=65, days_since_failure=1, prior_recovery_attempts=0,
        industry=Industry.LOGISTICS,
    ),
    Scenario(
        "B2C customer, technical error, high amount",
        AccountType.B2C, 8000.0, DeclineReason.TECHNICAL_ERROR,
        payment_history_score=80, days_since_failure=1, prior_recovery_attempts=0,
    ),
    Scenario(
        "Control-group account (must bypass bandit + LLM entirely)",
        AccountType.B2C, 1200.0, DeclineReason.INSUFFICIENT_FUNDS,
        payment_history_score=58, days_since_failure=3, prior_recovery_attempts=1,
        is_control_group=True,
    ),
]

# Of the non-control scenarios above, which get a simulated outcome report,
# whether they "recovered", and whether that action's window is backdated
# closed (immediate reward) or left open (deferred to the daily batch).
OUTCOME_PLAN = {
    "B2C subscriber, insufficient funds, low amount": (True, True),  # (recovered, window_closed)
    "B2C shopper, card expired, mid amount": (False, True),
    "B2B SaaS customer, bank declined (do_not_honor), mid amount": (True, True),
    "B2B retailer, fraud flagged, high amount": (False, True),
    "B2B logistics, insufficient funds, just under the escalation floor (₹500)": (True, False),
    "B2C customer, technical error, high amount": (True, False),
}


def make_account(db, scenario: Scenario) -> Account:
    account = Account(
        account_type=scenario.account_type,
        amount=scenario.amount,
        amount_tier=amount_tier_for(scenario.amount),
        decline_reason=scenario.decline_reason,
        payment_history_score=scenario.payment_history_score,
        days_since_failure=scenario.days_since_failure,
        prior_recovery_attempts=scenario.prior_recovery_attempts,
        industry=scenario.industry,
        is_control_group=scenario.is_control_group,
    )
    db.add(account)
    db.commit()
    return account


def run_recovery_pipeline(db, account: Account, *, deployed_at: datetime | None = None) -> dict:
    """Mirrors api/routes/recovery.py's orchestration, calling the real
    underlying modules directly (no HTTP layer) -- that's already covered
    by tests/test_api.py; this is exercising bandit+crew+guardrail behavior
    against realistic data with the real model.
    """
    deployed_at = deployed_at or datetime.now(timezone.utc)

    if account.is_control_group:
        action = build_control_action(account)
        action.deployed_at = deployed_at
        db.add(action)
        db.commit()
        return {"status": "control_group", "tactic": "no_action", "action": action}

    selection = select_arm(db, account)
    tactic = get_tactic(selection.tactic_name)

    try:
        crew_result = run_crew(account, tactic)
    except (CrewExhaustedError, LLMCallError) as exc:
        return {"status": "error", "error": str(exc)}

    final_tactic = get_tactic(crew_result.tactic_name)
    tactic_cost = final_tactic.resolve_cost(account, crew_result.message.discount_offered)

    action = RecoveryAction(
        account_id=account.account_id,
        segment_key=selection.segment_key,
        tactic_name=crew_result.tactic_name,
        tactic_cost=tactic_cost,
        discount_offered=crew_result.message.discount_offered,
        sampled_values=selection.sampled_values,
        message_drafted=crew_result.message.model_dump(),
        guardrail_approved=crew_result.guardrail_approved,
        guardrail_reason=crew_result.guardrail_reason,
        fallback_reason=crew_result.fallback_reason,
        deployed_at=deployed_at,
    )
    db.add(action)
    db.commit()
    return {"status": "ok", "tactic": crew_result.tactic_name, "action": action, "crew_result": crew_result}


def report_outcome(db, action: RecoveryAction, *, recovered: bool, amount_recovered: float) -> tuple[Outcome, bool]:
    """Mirrors api/routes/outcomes.py's finalize-if-window-closed logic."""
    now = datetime.now(timezone.utc)
    outcome = Outcome(
        action_id=action.action_id,
        account_id=action.account_id,
        recovered=recovered,
        amount_recovered=amount_recovered if recovered else 0.0,
        recovered_at=now if recovered else None,
        reported_at=now,
    )
    db.add(outcome)
    db.flush()

    account = action.account
    finalized = False
    if account.is_control_group or window_closed(action, now=now):
        result = apply_reward(db, outcome, now=now)
        outcome.processed_at = now
        if not account.is_control_group:
            update_state(db, action.segment_key, action.tactic_name, result.reward)
        refresh_segment_baseline(db, action.segment_key, now=now)
        finalized = True
    db.commit()
    return outcome, finalized


def _print_result(scenario: Scenario, result: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"SCENARIO: {scenario.label}")
    print(f"  {scenario.account_type.value} | ₹{scenario.amount:,.2f} | {scenario.decline_reason.value} "
          f"| history_score={scenario.payment_history_score} | days_since_failure={scenario.days_since_failure}"
          + (f" | industry={scenario.industry.value}" if scenario.industry else ""))
    print("-" * 78)

    if result["status"] == "control_group":
        print("  -> CONTROL GROUP: bypassed bandit and LLM entirely. tactic=no_action")
        return
    if result["status"] == "error":
        print(f"  -> PIPELINE ERROR: {result['error']}")
        return

    action = result["action"]
    crew = result["crew_result"]
    print(f"  -> Bandit selected: {result['tactic']}"
          + (f"  (FALLBACK: {crew.fallback_reason})" if crew.fallback_reason else ""))
    print(f"  -> Guardrail approved cleanly: {crew.guardrail_approved}  (reason: {crew.guardrail_reason})")
    print(f"  -> Tactic cost: ₹{action.tactic_cost:,.2f}"
          + (f"  discount_offered: ₹{action.discount_offered:,.2f}" if action.discount_offered else ""))
    msg = crew.message
    print(f"  -> Channel: {msg.channel}  |  Tone: {msg.tone}"
          + (f"  |  Subject: {msg.subject!r}" if msg.subject else ""))
    body_preview = msg.message_body.replace("\n", " ")
    print(f"  -> Message: {body_preview[:160]}{'...' if len(body_preview) > 160 else ''}")


def _print_outcome(outcome: Outcome, finalized: bool) -> None:
    if finalized:
        print(f"  -> Outcome reported: recovered={outcome.recovered}, amount=₹{outcome.amount_recovered:,.2f} "
              f"| reward FINALIZED immediately: {outcome.reward:+.2f} (uplift_indicator={outcome.uplift_indicator})")
    else:
        print(f"  -> Outcome reported: recovered={outcome.recovered}, amount=₹{outcome.amount_recovered:,.2f} "
              f"| window still open -> reward DEFERRED to the daily batch updater")


def main() -> None:
    print(f"Testbench DB: {TESTBENCH_DB_PATH}")
    print("Running the real pipeline (real Groq calls) against curated realistic scenarios...\n")
    session_local = _setup_testbench_db()
    db = session_local()

    scenarios_by_label = {s.label: s for s in SCENARIOS}
    actions_by_label: dict[str, RecoveryAction] = {}

    for i, scenario in enumerate(SCENARIOS):
        if i > 0 and not scenario.is_control_group:
            time.sleep(SCENARIO_PACING_SECONDS)  # stay under Groq's per-minute token limit
        account = make_account(db, scenario)
        result = run_recovery_pipeline(db, account)
        _print_result(scenario, result)
        if result["status"] == "ok":
            actions_by_label[scenario.label] = result["action"]

    print(f"\n{'=' * 78}\nSIMULATING OUTCOMES\n{'=' * 78}")
    for label, (recovered, window_should_be_closed) in OUTCOME_PLAN.items():
        action = actions_by_label.get(label)
        if action is None:
            continue
        if window_should_be_closed:
            action.deployed_at = datetime.now(timezone.utc) - timedelta(days=10)
            db.add(action)
            db.commit()
        amount = scenarios_by_label[label].amount
        outcome, finalized = report_outcome(db, action, recovered=recovered, amount_recovered=amount)
        print(f"\n{label}")
        _print_outcome(outcome, finalized)

    print(f"\n{'=' * 78}\nREPEAT RECOVERY EPISODE\n{'=' * 78}")
    print("Same account, a second /recover call after the first action's window closed --")
    print("this is currently unlimited: nothing checks prior_recovery_attempts as a cap.\n")
    repeat_scenario = Scenario(
        "Repeat-episode account", AccountType.B2C, 1800.0, DeclineReason.INSUFFICIENT_FUNDS,
        payment_history_score=50, days_since_failure=2, prior_recovery_attempts=1,
    )
    account = make_account(db, repeat_scenario)
    first = run_recovery_pipeline(db, account, deployed_at=datetime.now(timezone.utc) - timedelta(days=10))
    if first["status"] != "ok":
        print(f"  Episode 1 FAILED: {first.get('error', first['status'])}")
    else:
        print(f"  Episode 1 tactic: {first['tactic']} (window now closed)")
        time.sleep(SCENARIO_PACING_SECONDS)
        second = run_recovery_pipeline(db, account)
        if second["status"] != "ok":
            print(f"  Episode 2 FAILED: {second.get('error', second['status'])}")
        else:
            print(f"  Episode 2 tactic: {second['tactic']} (new action created, same account -- no cap enforced)")

    db.close()
    print(f"\nDone. Inspect {TESTBENCH_DB_PATH} directly, or point the dashboard/API at it via DB_URL if you want to browse it.")


if __name__ == "__main__":
    main()
