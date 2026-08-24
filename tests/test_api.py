"""api/main.py, api/routes/*.py"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from agents.drafting_agent import DraftedMessage
from agents.guardrail_agent import GuardrailVerdict
from db.models import ActionStatus, BanditState, RecoveryAction
from tests.conftest import FakeChatCompletion


@pytest.fixture()
def client(db_session, mocker):
    import api.main as api_main

    # api.main did `from db.session import engine`, which bound its own
    # frozen reference at import time -- patching db.session.engine (which
    # the db_session fixture already did) doesn't retroactively update
    # that binding, so the lifespan's own connectivity check needs it too.
    mocker.patch.object(api_main, "engine", db_session.get_bind())
    mocker.patch.object(api_main, "start_scheduler", return_value=None)
    with TestClient(api_main.app) as c:
        yield c


def test_404_on_unknown_account(client):
    r = client.get("/accounts/does-not-exist")
    assert r.status_code == 404
    assert set(r.json()) == {"error", "code"}
    assert r.json()["code"] == "account_not_found"

    r2 = client.post("/recover/does-not-exist")
    assert r2.status_code == 404
    assert r2.json()["code"] == "account_not_found"


def test_recover_control_group_returns_correct_status(client, make_account, fake_groq_client):
    account = make_account(is_control_group=True)

    r = client.post(f"/recover/{account.account_id}")

    assert r.status_code == 200
    assert r.json() == {"status": "control_group", "action": "no_action"}
    assert fake_groq_client.chat.completions.create.call_count == 0


def test_recover_idempotent(client, make_account, fake_groq_client):
    account = make_account(is_control_group=False)
    fake_groq_client.chat.completions.create.side_effect = [
        FakeChatCompletion(DraftedMessage(message_body="Hi {{first_name}}", channel="gateway", tone="conversational")),
        FakeChatCompletion(GuardrailVerdict(approved=True, reason="clean")),
    ]

    r1 = client.post(f"/recover/{account.account_id}")
    assert r1.status_code == 200
    tactic = r1.json()["tactic_selected"]

    calls_before = fake_groq_client.chat.completions.create.call_count
    r2 = client.post(f"/recover/{account.account_id}")

    assert r2.status_code == 200
    assert r2.json()["tactic_selected"] == tactic
    assert fake_groq_client.chat.completions.create.call_count == calls_before  # no new LLM calls, no new bandit pull


def test_outcome_recording_triggers_bandit_update(client, db_session, make_account):
    account = make_account(is_control_group=False)
    action = RecoveryAction(
        account_id=account.account_id,
        segment_key=account.segment,
        tactic_name="soft_nudge_email",
        tactic_cost=0.0,
        sampled_values={},
        guardrail_approved=True,
        status=ActionStatus.ACTIVE,
        deployed_at=datetime.now(timezone.utc) - timedelta(days=10),  # window already closed
    )
    db_session.add(action)
    db_session.commit()

    before = db_session.query(BanditState).filter_by(segment_key=account.segment, tactic_name="soft_nudge_email").first()
    before_alpha = before.alpha if before else 1.0

    r = client.post(
        f"/outcomes/{account.account_id}",
        json={"recovered": True, "amount_recovered": 500.0, "recovered_at": datetime.now(timezone.utc).isoformat()},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["reward_finalized"] is True
    assert body["reward"] is not None

    db_session.expire_all()  # the request wrote through a *different* Session object
    after = db_session.query(BanditState).filter_by(segment_key=account.segment, tactic_name="soft_nudge_email").first()
    assert after is not None
    assert after.alpha > before_alpha
