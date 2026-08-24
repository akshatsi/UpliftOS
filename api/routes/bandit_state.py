"""GET /bandit/state — current alpha/beta weights per segment per arm.

Named `bandit_state.py` rather than `bandit.py` purely to avoid a route
module shadowing the top-level `bandit` package it imports from.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import BanditArmState, BanditStateResponse
from bandit.thompson_sampling import get_all_states
from db.session import get_db

router = APIRouter(tags=["bandit"])


@router.get("/bandit/state", response_model=BanditStateResponse)
def bandit_state(db: Session = Depends(get_db)) -> BanditStateResponse:
    states = get_all_states(db)
    arms = [
        BanditArmState(
            segment_key=s.segment_key,
            tactic_name=s.tactic_name,
            alpha=s.alpha,
            beta=s.beta,
            estimated_win_rate=s.alpha / (s.alpha + s.beta),
        )
        for s in states
    ]
    return BanditStateResponse(arms=arms)
