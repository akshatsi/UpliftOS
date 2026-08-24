"""GET /attribution/baselines — current baseline recovery rates per segment,
read from the materialized cache table (see attribution.uplift).

Named `attribution_baselines.py` rather than `attribution.py` purely to
avoid a route module shadowing the top-level `attribution` package it
imports from.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import BaselinesResponse, SegmentBaselineSummary
from attribution.uplift import get_cached_baselines
from db.session import get_db

router = APIRouter(tags=["attribution"])


@router.get("/attribution/baselines", response_model=BaselinesResponse)
def attribution_baselines(db: Session = Depends(get_db)) -> BaselinesResponse:
    baselines = get_cached_baselines(db)
    return BaselinesResponse(baselines=[SegmentBaselineSummary.model_validate(b) for b in baselines])
