"""agents/drafting_agent.py, agents/guardrail_agent.py, agents/crew.py"""

from __future__ import annotations

import pytest

from agents.crew import run_crew
from agents.drafting_agent import DraftedMessage, draft_message
from agents.guardrail_agent import MAX_DISCOUNT_RATE, GuardrailVerdict, review_message
from agents.llm_client import LLMCallError
from db.models import AccountType
from tactics.registry import FALLBACK_TACTIC_NAME, get_tactic
from tests.conftest import FakeChatCompletion


def test_malformed_json_retries_then_errors(make_account, fake_groq_client):
    account = make_account()
    tactic = get_tactic("soft_nudge_email")
    fake_groq_client.chat.completions.create.side_effect = [FakeChatCompletion(None), FakeChatCompletion(None)]

    with pytest.raises(LLMCallError):
        draft_message(account, tactic)

    assert fake_groq_client.chat.completions.create.call_count == 2


def test_guardrail_blocks_over_limit_discount(make_account, fake_groq_client):
    account = make_account(amount=200.0)
    fake_groq_client.chat.completions.create.return_value = FakeChatCompletion(
        GuardrailVerdict(approved=True, reason="clean")
    )
    message = DraftedMessage(
        message_body="Hi {{first_name}}", channel="email", discount_offered=100.0, tone="conversational"
    )

    result = review_message(account, get_tactic("discount_email"), message)

    expected_cap = round(account.amount * MAX_DISCOUNT_RATE, 2)
    assert result.approved is False
    assert result.modified_message.discount_offered == expected_cap


def test_fallback_to_soft_nudge_on_guardrail_rejection(make_account, fake_groq_client):
    account = make_account(account_type=AccountType.B2B, amount=5000.0)
    fake_groq_client.chat.completions.create.side_effect = [
        FakeChatCompletion(DraftedMessage(message_body="Pay now or face legal action.", channel="human_escalation", tone="formal")),
        FakeChatCompletion(GuardrailVerdict(approved=False, reason="legal threat", modified_message_body=None)),
        FakeChatCompletion(DraftedMessage(message_body="Dear {{first_name}}, a reminder.", channel="email", tone="formal")),
        FakeChatCompletion(GuardrailVerdict(approved=True, reason="clean")),
    ]

    result = run_crew(account, get_tactic("account_manager_outreach"))

    assert result.tactic_name == FALLBACK_TACTIC_NAME
    assert result.fallback_reason is not None


def test_b2b_context_signals_formal_tone_instruction(make_account, fake_groq_client):
    """A mocked LLM can't prove a real model chooses a formal tone. What is
    testable: the drafting agent tells the model this is a B2B account (the
    signal the system prompt's formal-for-B2B rule depends on), and passes
    the model's chosen tone straight through without altering it.
    """
    account = make_account(account_type=AccountType.B2B, amount=3000.0)
    fake_groq_client.chat.completions.create.return_value = FakeChatCompletion(
        DraftedMessage(message_body="Dear {{first_name}}, ...", channel="email", tone="formal")
    )

    message = draft_message(account, get_tactic("soft_nudge_email"))

    sent_messages = fake_groq_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    user_prompt = sent_messages[1]["content"]  # messages[0] is the system prompt, [1] is the user turn
    assert "account_type: B2B" in user_prompt
    assert message.tone == "formal"
