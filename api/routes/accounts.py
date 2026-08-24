"""GET /accounts, GET /accounts/{id}"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.errors import APIError
from api.schemas import AccountDetail, AccountListResponse, AccountSummary
from db.models import Account, Outcome, RecoveryAction
from db.session import get_db

router = APIRouter(tags=["accounts"])


@router.get("/accounts", response_model=AccountListResponse)
def list_accounts(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    segment_key: Optional[str] = None,
    tactic_name: Optional[str] = None,
    recovered: Optional[bool] = None,
    db: Session = Depends(get_db),
) -> AccountListResponse:
    query = select(Account)
    count_query = select(func.count(func.distinct(Account.account_id)))

    needs_action_join = segment_key is not None or tactic_name is not None or recovered is not None
    if needs_action_join:
        query = query.join(RecoveryAction, RecoveryAction.account_id == Account.account_id)
        count_query = count_query.join(RecoveryAction, RecoveryAction.account_id == Account.account_id)
    if recovered is not None:
        query = query.join(Outcome, Outcome.action_id == RecoveryAction.action_id)
        count_query = count_query.join(Outcome, Outcome.action_id == RecoveryAction.action_id)

    if segment_key is not None:
        query = query.where(RecoveryAction.segment_key == segment_key)
        count_query = count_query.where(RecoveryAction.segment_key == segment_key)
    if tactic_name is not None:
        query = query.where(RecoveryAction.tactic_name == tactic_name)
        count_query = count_query.where(RecoveryAction.tactic_name == tactic_name)
    if recovered is not None:
        query = query.where(Outcome.recovered == recovered)
        count_query = count_query.where(Outcome.recovered == recovered)

    total = db.execute(count_query).scalar_one()
    query = query.distinct().order_by(Account.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    accounts = list(db.execute(query).scalars())

    return AccountListResponse(
        accounts=[AccountSummary.model_validate(a) for a in accounts],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/accounts/{account_id}", response_model=AccountDetail)
def get_account(account_id: str, db: Session = Depends(get_db)) -> AccountDetail:
    account = db.get(Account, account_id)
    if account is None:
        raise APIError(404, "account_not_found", f"no account with id {account_id!r}")
    return AccountDetail.model_validate(account)
