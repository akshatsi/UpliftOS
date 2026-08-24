"""attribution/control_group.py and attribution/uplift.py"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from attribution.control_group import CONTROL_GROUP_SHARE, assign_control_group, build_control_action
from attribution.uplift import MIN_CONTROL_ACCOUNTS, NEUTRAL_UPLIFT, get_baseline, get_uplift
from data.synthetic_generator import generate_accounts
from db.models import AccountType, AmountTier, DeclineReason, Outcome


def test_assign_control_group_is_pure_probability_independent_of_context():
    rng = random.Random(42)
    results = [assign_control_group(rng) for _ in range(10000)]
    rate = sum(results) / len(results)
    assert abs(rate - CONTROL_GROUP_SHARE) < 0.02


def test_control_group_assignment_is_pre_bandit_and_context_independent():
    # If assignment happened after/from context, the rate would skew by
    # account_type. Generated across a large batch, it shouldn't.
    accounts = generate_accounts(2000, seed=7)
    by_type = Counter((a.account_type, a.is_control_group) for a in accounts)

    for account_type in (AccountType.B2B, AccountType.B2C):
        total = sum(v for (t, _c), v in by_type.items() if t == account_type)
        control = by_type.get((account_type, True), 0)
        rate = control / total
        assert abs(rate - CONTROL_GROUP_SHARE) < 0.03, f"{account_type}: control rate {rate} far from {CONTROL_GROUP_SHARE}"


def test_build_control_action_requires_control_group(make_account):
    non_control = make_account(is_control_group=False)
    with pytest.raises(ValueError):
        build_control_action(non_control)


def test_baseline_neutral_with_zero_control_data(make_account, db_session):
    account = make_account(is_control_group=False)
    assert get_uplift(db_session, account.segment, recovered=True) == NEUTRAL_UPLIFT
    assert get_uplift(db_session, account.segment, recovered=False) == NEUTRAL_UPLIFT


SEGMENT_KWARGS = dict(account_type=AccountType.B2C, amount_tier=AmountTier.LOW, decline_reason=DeclineReason.CARD_EXPIRED)
SEGMENT_KEY = f"{SEGMENT_KWARGS['decline_reason'].value}|{SEGMENT_KWARGS['amount_tier'].value}|{SEGMENT_KWARGS['account_type'].value}"


def _add_resolved_control_outcome(db_session, make_account, *, recovered, **segment_kwargs):
    account = make_account(is_control_group=True, **segment_kwargs)
    action = build_control_action(account)
    action.deployed_at = datetime.now(timezone.utc) - timedelta(days=1)  # inside the 30-day rolling window
    db_session.add(action)
    db_session.flush()
    db_session.add(
        Outcome(
            action_id=action.action_id,
            account_id=account.account_id,
            recovered=recovered,
            amount_recovered=1000.0 if recovered else 0.0,
        )
    )
    db_session.flush()


def test_baseline_neutral_below_threshold_then_reliable_at_threshold(db_session, make_account):
    for _ in range(MIN_CONTROL_ACCOUNTS - 1):
        _add_resolved_control_outcome(db_session, make_account, recovered=True, **SEGMENT_KWARGS)
    db_session.commit()

    below = get_baseline(db_session, SEGMENT_KEY)
    assert below.is_reliable is False
    assert below.control_count == MIN_CONTROL_ACCOUNTS - 1
    assert get_uplift(db_session, SEGMENT_KEY, recovered=True) == NEUTRAL_UPLIFT

    _add_resolved_control_outcome(db_session, make_account, recovered=True, **SEGMENT_KWARGS)
    db_session.commit()

    at_threshold = get_baseline(db_session, SEGMENT_KEY)
    assert at_threshold.is_reliable is True
    assert at_threshold.control_count == MIN_CONTROL_ACCOUNTS


def test_uplift_calculation_correct_against_known_values(db_session, make_account):
    # 5 control accounts: 2 recovered, 3 did not -> baseline = 0.4
    for recovered in [True, True, False, False, False]:
        _add_resolved_control_outcome(db_session, make_account, recovered=recovered, **SEGMENT_KWARGS)
    db_session.commit()

    baseline = get_baseline(db_session, SEGMENT_KEY)
    assert baseline.is_reliable is True
    assert baseline.control_count == 5
    assert baseline.baseline_recovery_rate == pytest.approx(0.4)

    assert get_uplift(db_session, SEGMENT_KEY, recovered=True) == pytest.approx(0.6)  # 1.0 - 0.4
    assert get_uplift(db_session, SEGMENT_KEY, recovered=False) == pytest.approx(-0.4)  # 0.0 - 0.4
