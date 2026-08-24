"""bandit/thompson_sampling.py"""

from __future__ import annotations

import random

import pytest

import bandit.thompson_sampling as ts
from bandit.thompson_sampling import (
    CONTROL_TACTIC_NAME,
    FALLBACK_TACTIC_NAME,
    get_segment_states,
    select_arm,
    update_state,
)
from db.models import AccountType, DeclineReason
from tactics.registry import get_tactic


def test_arm_selection_respects_eligibility_constraints(make_account, db_session):
    b2c_account = make_account(account_type=AccountType.B2C, decline_reason=DeclineReason.DO_NOT_HONOR)
    for _ in range(20):
        selection = select_arm(db_session, b2c_account, rng=random.Random())
        assert selection.tactic_name != "account_manager_outreach"

    b2b_account = make_account(account_type=AccountType.B2B)
    for _ in range(20):
        selection = select_arm(db_session, b2b_account, rng=random.Random())
        assert selection.tactic_name not in {"sms_reminder", "whatsapp_nudge", "payday_retry"}


def test_control_group_always_gets_no_action(make_account, db_session):
    control_account = make_account(is_control_group=True)

    selection = select_arm(db_session, control_account)

    assert selection.tactic_name == CONTROL_TACTIC_NAME
    assert selection.is_control is True
    assert selection.sampled_values == {}
    assert get_segment_states(db_session, control_account.segment) == []  # bandit never touched


def test_never_selects_no_action_for_non_control(make_account, db_session, monkeypatch):
    account = make_account()
    monkeypatch.setattr(ts, "eligible_tactics", lambda account, include_control=False: [get_tactic("no_action")])

    with pytest.raises(RuntimeError):
        ts.select_arm(db_session, account)


def test_update_state_refuses_control_arm(db_session):
    with pytest.raises(ValueError):
        update_state(db_session, "some_segment", CONTROL_TACTIC_NAME, reward=100.0)


def test_state_persists_and_reloads(make_account, db_session):
    account = make_account(account_type=AccountType.B2C, decline_reason=DeclineReason.INSUFFICIENT_FUNDS)
    select_arm(db_session, account, rng=random.Random(1))
    db_session.commit()

    states_before = {s.tactic_name: (s.alpha, s.beta) for s in get_segment_states(db_session, account.segment)}
    assert states_before

    update_state(db_session, account.segment, "soft_nudge_email", reward=500.0)
    db_session.commit()

    # a fresh query stands in for "reload after restart" -- state must come
    # from the DB, not an in-process cache
    states_after = {s.tactic_name: (s.alpha, s.beta) for s in get_segment_states(db_session, account.segment)}
    alpha, beta = states_after["soft_nudge_email"]
    assert alpha > beta  # a large positive reward should pull alpha above beta


def test_fallback_triggers_when_all_arms_fail(make_account, db_session, monkeypatch):
    account = make_account()
    monkeypatch.setattr(ts, "eligible_tactics", lambda account, include_control=False: [])

    selection = ts.select_arm(db_session, account)

    assert selection.tactic_name == FALLBACK_TACTIC_NAME
    assert selection.fallback_reason is not None
