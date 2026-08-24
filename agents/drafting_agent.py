"""Drafting agent: an LLM drafts the recovery message for a given account +
selected tactic. `output_format=DraftedMessage` constrains the response to
valid JSON at the API level; agents.llm_client owns the malformed-output
retry-then-raise handling on top of that.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from agents.llm_client import parse_structured
from db.models import Account
from tactics.registry import Tactic

SYSTEM_PROMPT = """You are a payment recovery message drafter for a fintech platform.
Draft exactly ONE outbound recovery message for a customer whose payment failed,
using exactly the tactic you're given.

Rules, no exceptions:
- Never mention or imply a competitor, alternative provider, or alternative platform.
- Never promise anything the tactic doesn't cover — e.g. a nudge-email tactic nudges,
  it does not offer a discount, a payment plan, or a human callback unless the tactic
  you were given is specifically a discount, payment-plan, or escalation tactic.
- Adapt tone to the account: formal and precise for B2B accounts, warm and
  conversational for B2C accounts.
- You do not know the customer's name. If you address them personally, use the
  literal placeholder token {{first_name}} — never invent a name.
- If (and only if) the tactic is a discount tactic, set discount_offered to a
  reasonable rupee amount (the platform default is 10% of the amount) — do not
  assume any particular cap, that is enforced elsewhere.
- Output only the drafted message fields — no prose, no explanation, no markdown.
"""


class DraftedMessage(BaseModel):
    message_body: str
    channel: str
    subject: Optional[str] = None
    discount_offered: Optional[float] = None
    tone: str


def _account_context(account: Account, tactic: Tactic) -> str:
    lines = [
        f"account_type: {account.account_type.value}",
        f"amount: {account.amount:.2f}",
        f"amount_tier: {account.amount_tier.value}",
        f"decline_reason: {account.decline_reason.value}",
        f"payment_history_score: {account.payment_history_score}",
        f"days_since_failure: {account.days_since_failure}",
        f"prior_recovery_attempts: {account.prior_recovery_attempts}",
    ]
    if account.industry is not None:
        lines.append(f"industry: {account.industry.value}")
    lines += [
        f"selected_tactic: {tactic.name}",
        f"channel: {tactic.channel.value}",
    ]
    return "Draft the recovery message for this account and tactic:\n" + "\n".join(lines)


def draft_message(account: Account, tactic: Tactic) -> DraftedMessage:
    return parse_structured(
        system=SYSTEM_PROMPT,
        user_content=_account_context(account, tactic),
        output_format=DraftedMessage,
        context=f"drafting_agent account_id={account.account_id} tactic={tactic.name}",
    )
