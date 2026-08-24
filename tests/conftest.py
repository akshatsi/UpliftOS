"""Shared fixtures. Every test gets a fresh in-memory SQLite DB — never the
dev `revenue_recovery.db` file — and every Groq call is mocked; no test in
this suite ever makes a real API call or needs an API key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import db.session as db_session_mod
from db.models import Account, AccountType, AmountTier, Base, DeclineReason, Industry


@pytest.fixture()
def db_session(monkeypatch):
    """Fresh in-memory DB per test, wired in as db.session's engine so any
    code that calls session_scope()/get_db() transparently uses it —
    StaticPool because plain SQLite memory URLs give each connection its
    own separate DB, which would make writes from one session invisible
    to another.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    monkeypatch.setattr(db_session_mod, "engine", engine)
    monkeypatch.setattr(db_session_mod, "SessionLocal", TestSessionLocal)

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def make_account(db_session):
    """Factory for a valid, persisted Account with sensible defaults,
    overridable by kwarg. Auto-fills `industry` for B2B accounts so
    individual tests don't need to remember the check constraint.
    """

    def _make(**overrides) -> Account:
        defaults = dict(
            account_type=AccountType.B2C,
            amount=1000.0,
            amount_tier=AmountTier.MID,
            decline_reason=DeclineReason.INSUFFICIENT_FUNDS,
            payment_history_score=50,
            days_since_failure=3,
            prior_recovery_attempts=0,
            industry=None,
            is_control_group=False,
        )
        defaults.update(overrides)
        if defaults["account_type"] == AccountType.B2B and defaults.get("industry") is None:
            defaults["industry"] = Industry.SAAS
        account = Account(**defaults)
        db_session.add(account)
        db_session.commit()
        return account

    return _make


class FakeChatCompletion:
    """Mimics Groq's ChatCompletion: only `.choices[0].message.content` (a
    JSON string), which agents.llm_client._extract_parsed actually reads.
    Pass a Pydantic instance to serialize as the response, or None to
    simulate malformed/empty output.
    """

    def __init__(self, parsed_output):
        content = parsed_output.model_dump_json() if parsed_output is not None else None
        message = SimpleNamespace(content=content)
        self.choices = [SimpleNamespace(message=message)]


@pytest.fixture()
def fake_groq_client(mocker):
    """Replaces the module-level Groq client singleton. Configure
    `.chat.completions.create.side_effect` / `.return_value` per test.
    """
    client = mocker.MagicMock()
    mocker.patch("agents.llm_client._client", client)
    return client
