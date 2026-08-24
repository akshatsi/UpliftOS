"""Orchestrates the two-agent flow: draft, then guardrail-review.

Scope is deliberately narrow — this runs the crew for one account and one
*already-selected* tactic. Control-group bypass and bandit arm selection
happen one layer up (the future `POST /recover` route), matching the
pipeline order in the spec: eligibility check -> control group bypass ->
bandit selects arm -> agent crew executes -> action logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.drafting_agent import DraftedMessage, draft_message
from agents.guardrail_agent import review_message
from db.models import Account
from tactics.registry import FALLBACK_TACTIC_NAME, Tactic, get_tactic

logger = logging.getLogger("agents.crew")


class CrewExhaustedError(Exception):
    """Raised when even the fallback tactic's draft gets rejected with no
    fix available — the last line of defense failed too, so this surfaces
    as a hard error rather than a silent no-op.
    """


@dataclass
class CrewResult:
    tactic_name: str
    message: DraftedMessage
    guardrail_approved: bool
    guardrail_reason: str
    fallback_reason: Optional[str] = None


def run_crew(account: Account, tactic: Tactic, *, _is_fallback: bool = False) -> CrewResult:
    message = draft_message(account, tactic)
    verdict = review_message(account, tactic, message)

    if verdict.approved:
        return CrewResult(
            tactic_name=tactic.name,
            message=message,
            guardrail_approved=True,
            guardrail_reason=verdict.reason,
        )

    if verdict.modified_message is not None:
        return CrewResult(
            tactic_name=tactic.name,
            message=verdict.modified_message,
            guardrail_approved=False,
            guardrail_reason=verdict.reason,
        )

    # No fix possible for this tactic's draft.
    if _is_fallback:
        raise CrewExhaustedError(
            f"guardrail rejected the fallback tactic {FALLBACK_TACTIC_NAME!r} too, with no fix, "
            f"for account {account.account_id}: {verdict.reason}"
        )

    fallback_reason = (
        f"guardrail rejected {tactic.name!r} with no fix ({verdict.reason}); "
        f"falling back to {FALLBACK_TACTIC_NAME!r}"
    )
    logger.warning("crew_fallback account_id=%s reason=%s", account.account_id, fallback_reason)
    fallback_tactic = get_tactic(FALLBACK_TACTIC_NAME)
    result = run_crew(account, fallback_tactic, _is_fallback=True)
    result.fallback_reason = fallback_reason
    return result
