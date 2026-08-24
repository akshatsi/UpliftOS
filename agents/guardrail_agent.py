"""Guardrail agent: enforces policy rules on a drafted message, as the
second line of defense against a drafting-agent output that hallucinates
outside its tactic's scope.

The two objectively-checkable rules (max discount, escalation amount floor)
are plain Python — hardcoded, not LLM-generated, and there's no reason to
ask a model to do arithmetic it might get wrong. The two rules that need
language judgment (tone, PII) go through the LLM, with the rules themselves
still fixed in its system prompt rather than left for the model to invent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from agents.drafting_agent import DraftedMessage
from agents.llm_client import parse_structured
from db.models import Account
from tactics.registry import Tactic

logger = logging.getLogger("agents.guardrail_agent")

MAX_DISCOUNT_RATE = 0.20
MIN_ESCALATION_AMOUNT = 500.0

GUARDRAIL_SYSTEM_PROMPT = """You are a compliance reviewer for outbound payment-recovery
messages. You check exactly two things about the drafted message body:

1. Tone: it must not be threatening, coercive, or create legal liability — no legal
   threats, no language implying legal or collections action, no shaming or aggressive
   language.
2. PII: it must not contain personal information beyond the literal {{first_name}}
   placeholder token — no account numbers, card numbers, government IDs, addresses,
   phone numbers, full names, or email addresses.

If either rule is violated, set approved to false and give a specific reason. If you
can fix it by rewriting the message body (same channel, tactic, and intent), set
modified_message_body to the corrected text. If the violation can't be fixed by
rewriting — e.g. the message's entire premise is a threat — leave
modified_message_body unset.

If the message is fine, set approved to true, reason to a short confirmation, and
leave modified_message_body unset. Output only the verdict fields — no prose.
"""


class GuardrailVerdict(BaseModel):
    approved: bool
    reason: str
    modified_message_body: Optional[str] = None


@dataclass
class GuardrailResult:
    approved: bool
    reason: str
    modified_message: Optional[DraftedMessage] = None


def _check_discount(account: Account, message: DraftedMessage) -> Optional[str]:
    if message.discount_offered is None:
        return None
    max_allowed = round(account.amount * MAX_DISCOUNT_RATE, 2)
    if message.discount_offered > max_allowed:
        return f"discount_offered={message.discount_offered} exceeds the 20% cap ({max_allowed})"
    return None


def _check_escalation(account: Account, tactic: Tactic) -> Optional[str]:
    if tactic.name == "account_manager_outreach" and account.amount < MIN_ESCALATION_AMOUNT:
        return f"human escalation not allowed for amounts under {MIN_ESCALATION_AMOUNT} (amount={account.amount})"
    return None


def _llm_review(account: Account, message: DraftedMessage) -> GuardrailVerdict:
    prompt = (
        f"account_type: {account.account_type.value}\n"
        f"channel: {message.channel}\n"
        f"tone: {message.tone}\n"
        f"subject: {message.subject or ''}\n"
        f"message_body:\n{message.message_body}"
    )
    return parse_structured(
        system=GUARDRAIL_SYSTEM_PROMPT,
        user_content=prompt,
        output_format=GuardrailVerdict,
        context=f"guardrail_agent account_id={account.account_id}",
    )


def review_message(account: Account, tactic: Tactic, message: DraftedMessage) -> GuardrailResult:
    """approved=True only if the message needed zero changes. Any fix
    (discount cap or LLM tone/PII rewrite) comes back as approved=False
    with `modified_message` set — per spec, the caller uses the modified
    version rather than treating this as a hard rejection. approved=False
    with `modified_message=None` means no fix was possible.
    """
    escalation_violation = _check_escalation(account, tactic)
    if escalation_violation:
        logger.warning("guardrail_rejected_no_fix account_id=%s reason=%s", account.account_id, escalation_violation)
        return GuardrailResult(approved=False, reason=escalation_violation, modified_message=None)

    corrected = message
    reasons: list[str] = []

    discount_violation = _check_discount(account, message)
    if discount_violation:
        max_allowed = round(account.amount * MAX_DISCOUNT_RATE, 2)
        corrected = corrected.model_copy(update={"discount_offered": max_allowed})
        reasons.append(discount_violation)

    verdict = _llm_review(account, corrected)
    if not verdict.approved:
        if verdict.modified_message_body:
            corrected = corrected.model_copy(update={"message_body": verdict.modified_message_body})
            reasons.append(verdict.reason)
        else:
            reasons.append(verdict.reason)
            logger.warning("guardrail_rejected_no_fix account_id=%s reason=%s", account.account_id, "; ".join(reasons))
            return GuardrailResult(approved=False, reason="; ".join(reasons), modified_message=None)

    if reasons:
        logger.warning("guardrail_modified account_id=%s reason=%s", account.account_id, "; ".join(reasons))
        return GuardrailResult(approved=False, reason="; ".join(reasons), modified_message=corrected)

    return GuardrailResult(approved=True, reason="approved without changes", modified_message=None)
