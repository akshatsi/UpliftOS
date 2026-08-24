"""bandit/reward.py"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bandit.reward as reward_mod
from bandit.reward import RewardWindowNotClosedError, compute_reward
from db.models import Outcome, RecoveryAction


def _deploy_action(db_session, account, tactic_name, tactic_cost, days_ago):
    action = RecoveryAction(
        account_id=account.account_id,
        segment_key=account.segment,
        tactic_name=tactic_name,
        tactic_cost=tactic_cost,
        sampled_values={},
        deployed_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    db_session.add(action)
    db_session.flush()
    return action


def _record_outcome(db_session, action, recovered, amount_recovered):
    outcome = Outcome(
        action_id=action.action_id,
        account_id=action.account_id,
        recovered=recovered,
        amount_recovered=amount_recovered,
        recovered_at=datetime.now(timezone.utc) if recovered else None,
    )
    db_session.add(outcome)
    db_session.flush()
    return outcome


def test_window_not_closed_raises(make_account, db_session):
    account = make_account()
    action = _deploy_action(db_session, account, "soft_nudge_email", 0.0, days_ago=1)
    outcome = _record_outcome(db_session, action, recovered=True, amount_recovered=500.0)

    with pytest.raises(RewardWindowNotClosedError):
        compute_reward(db_session, outcome)


def test_negative_reward_when_cost_exceeds_value(make_account, db_session, monkeypatch):
    monkeypatch.setattr(reward_mod, "get_uplift", lambda db, seg, recovered: 0.1)
    account = make_account()
    action = _deploy_action(db_session, account, "account_manager_outreach", tactic_cost=200.0, days_ago=10)
    outcome = _record_outcome(db_session, action, recovered=True, amount_recovered=100.0)

    result = compute_reward(db_session, outcome)

    assert result.reward == -190.0  # (100 * 0.1) - 200


def test_reward_capped_at_amount_recovered(make_account, db_session, monkeypatch):
    monkeypatch.setattr(reward_mod, "get_uplift", lambda db, seg, recovered: 5.0)  # absurd, out-of-range uplift
    account = make_account()
    action = _deploy_action(db_session, account, "soft_nudge_email", tactic_cost=0.0, days_ago=10)
    outcome = _record_outcome(db_session, action, recovered=True, amount_recovered=300.0)

    result = compute_reward(db_session, outcome)

    assert result.reward == 300.0  # not 1500 -- capped at what was actually recovered


def test_control_group_returns_zero_reward(make_account, db_session):
    account = make_account(is_control_group=True)
    action = _deploy_action(db_session, account, "no_action", tactic_cost=0.0, days_ago=0)  # window still open
    outcome = _record_outcome(db_session, action, recovered=True, amount_recovered=500.0)

    result = compute_reward(db_session, outcome)

    assert result.reward == 0.0
    assert result.uplift_indicator == 0.0


def test_discount_cost_deducted_correctly(make_account, db_session, monkeypatch):
    monkeypatch.setattr(reward_mod, "get_uplift", lambda db, seg, recovered: 1.0)
    account = make_account(amount=1000.0)
    action = _deploy_action(db_session, account, "discount_email", tactic_cost=100.0, days_ago=10)
    outcome = _record_outcome(db_session, action, recovered=True, amount_recovered=1000.0)

    result = compute_reward(db_session, outcome)

    assert result.reward == 900.0  # (1000 * 1.0) - 100
    assert result.tactic_cost == 100.0
