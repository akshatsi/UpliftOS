"""Read-only Streamlit dashboard: tactic performance, bandit arm weights,
attribution/uplift, recovery funnel, and reward-over-time. Every function
here only reads — there is no button or action that writes to the database.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# `streamlit run dashboard/app.py` executes this file directly, which puts
# dashboard/ (not the project root) on sys.path -- without this, every
# absolute import below (attribution, bandit, db, tactics) fails with
# ModuleNotFoundError regardless of the cwd streamlit was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import select

from attribution.uplift import MIN_CONTROL_ACCOUNTS, get_cached_baselines
from bandit.thompson_sampling import get_all_states
from db.models import Account, Outcome, RecoveryAction
from db.session import session_scope
from tactics.registry import TACTICS

st.set_page_config(page_title="Revenue Recovery Dashboard", layout="wide")

CACHE_TTL_SECONDS = 30


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_tactic_performance() -> pd.DataFrame:
    with session_scope() as db:
        rows = db.execute(
            select(
                RecoveryAction.tactic_name,
                RecoveryAction.tactic_cost,
                Outcome.recovered,
                Outcome.amount_recovered,
                Outcome.reward,
            ).outerjoin(Outcome, Outcome.action_id == RecoveryAction.action_id)
        ).all()

    all_tactic_names = [t.name for t in TACTICS]
    empty_row = {"deployments": 0, "recovery_rate": None, "avg_reward": None, "avg_cost": 0.0, "net_value": 0.0}

    if not rows:
        return pd.DataFrame([{"tactic_name": n, **empty_row} for n in all_tactic_names])

    df = pd.DataFrame(rows, columns=["tactic_name", "tactic_cost", "recovered", "amount_recovered", "reward"])
    df["recovered"] = df["recovered"].astype("boolean")  # nullable bool: True/False/<NA>, mean() skips <NA>

    grouped = df.groupby("tactic_name")
    perf = grouped.agg(
        deployments=("tactic_name", "size"),
        recovery_rate=("recovered", "mean"),
        avg_reward=("reward", "mean"),
        avg_cost=("tactic_cost", "mean"),
        total_recovered=("amount_recovered", lambda s: s.fillna(0).sum()),
        total_cost=("tactic_cost", "sum"),
    ).reset_index()
    perf["net_value"] = perf["total_recovered"] - perf["total_cost"]
    perf = perf.drop(columns=["total_recovered", "total_cost"])

    missing = set(all_tactic_names) - set(perf["tactic_name"])
    if missing:
        filler = pd.DataFrame([{"tactic_name": n, **empty_row} for n in missing])
        perf = pd.concat([perf, filler], ignore_index=True)

    return perf.sort_values("deployments", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_bandit_state() -> pd.DataFrame:
    with session_scope() as db:
        states = get_all_states(db)
        rows = [
            {
                "segment_key": s.segment_key,
                "tactic_name": s.tactic_name,
                "alpha": s.alpha,
                "beta": s.beta,
                "estimated_win_rate": s.alpha / (s.alpha + s.beta),
            }
            for s in states
        ]
    return pd.DataFrame(rows, columns=["segment_key", "tactic_name", "alpha", "beta", "estimated_win_rate"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_attribution() -> pd.DataFrame:
    with session_scope() as db:
        baselines = get_cached_baselines(db)
        baseline_rows = [
            {"segment_key": b.segment_key, "control_count": b.control_count, "control_recovery_rate": b.baseline_recovery_rate}
            for b in baselines
        ]
        treated_rows = db.execute(
            select(RecoveryAction.segment_key, Outcome.recovered)
            .join(Outcome, Outcome.action_id == RecoveryAction.action_id)
            .join(Account, Account.account_id == RecoveryAction.account_id)
            .where(Account.is_control_group.is_(False))
        ).all()

    baseline_df = pd.DataFrame(baseline_rows, columns=["segment_key", "control_count", "control_recovery_rate"])
    treated_df = pd.DataFrame(treated_rows, columns=["segment_key", "recovered"])

    if treated_df.empty:
        treated_summary = pd.DataFrame(columns=["segment_key", "treated_count", "treated_recovery_rate"])
    else:
        treated_summary = (
            treated_df.groupby("segment_key")["recovered"]
            .agg(treated_count="size", treated_recovery_rate="mean")
            .reset_index()
        )

    merged = baseline_df.merge(treated_summary, on="segment_key", how="outer")
    merged["control_count"] = merged["control_count"].fillna(0).astype(int)
    merged["treated_count"] = merged["treated_count"].fillna(0).astype(int)
    merged["uplift_pct"] = (merged["treated_recovery_rate"] - merged["control_recovery_rate"]) * 100
    merged["confidence"] = merged["control_count"].apply(
        lambda n: "reliable" if n >= MIN_CONTROL_ACCOUNTS else f"low data (n={n})"
    )
    return merged.sort_values("segment_key").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_funnel() -> pd.DataFrame:
    with session_scope() as db:
        accounts_failed_n = db.execute(select(Account.account_id)).scalars().all()
        triggered_rows = db.execute(
            select(RecoveryAction.account_id, RecoveryAction.guardrail_approved)
            .join(Account, Account.account_id == RecoveryAction.account_id)
            .where(Account.is_control_group.is_(False))
        ).all()
        recovered_rows = db.execute(
            select(Outcome.account_id)
            .join(RecoveryAction, RecoveryAction.action_id == Outcome.action_id)
            .join(Account, Account.account_id == RecoveryAction.account_id)
            .where(Account.is_control_group.is_(False), Outcome.recovered.is_(True))
        ).all()

    approved_n = sum(1 for _, approved in triggered_rows if approved is True)

    return pd.DataFrame(
        {
            "stage": ["Accounts failed", "Recovery triggered", "Guardrail approved (clean)", "Recovered within window"],
            "count": [len(accounts_failed_n), len(triggered_rows), approved_n, len(recovered_rows)],
        }
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_reward_over_time() -> pd.DataFrame:
    with session_scope() as db:
        rows = db.execute(
            select(Outcome.reported_at, RecoveryAction.tactic_name, Outcome.reward)
            .join(RecoveryAction, RecoveryAction.action_id == Outcome.action_id)
            .where(Outcome.reward.is_not(None))
        ).all()

    df = pd.DataFrame(rows, columns=["reported_at", "tactic_name", "reward"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["reported_at"]).dt.date
    return df.groupby(["date", "tactic_name"])["reward"].mean().reset_index()


def render() -> None:
    st.title("AI Revenue Recovery — Dashboard")
    st.caption(
        f"Read-only — nothing here triggers a recovery action. "
        f"Data refreshes every {CACHE_TTL_SECONDS}s. "
        f"Last loaded: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    st.header("Tactic performance")
    st.caption("avg_reward is the bandit's own uplift-adjusted signal (per unit); net_value is raw recovered-minus-cost dollars (aggregate).")
    perf_df = load_tactic_performance()
    st.dataframe(
        perf_df.style.format(
            {"recovery_rate": "{:.1%}", "avg_reward": "₹{:.2f}", "avg_cost": "₹{:.2f}", "net_value": "₹{:.2f}"},
            na_rep="—",
        ),
        width='stretch',
        hide_index=True,
    )

    st.header("Bandit arm weights")
    st.caption("Estimated win rate = alpha / (alpha + beta) per segment x tactic. Blank cells haven't been sampled yet.")
    bandit_df = load_bandit_state()
    if bandit_df.empty:
        st.info("No bandit state yet — run some recoveries first.")
    else:
        heatmap = (
            alt.Chart(bandit_df)
            .mark_rect()
            .encode(
                x=alt.X("tactic_name:N", title="Tactic"),
                y=alt.Y("segment_key:N", title="Segment"),
                color=alt.Color("estimated_win_rate:Q", title="Est. win rate", scale=alt.Scale(scheme="greens", domain=[0, 1])),
                tooltip=[
                    "segment_key",
                    "tactic_name",
                    alt.Tooltip("estimated_win_rate:Q", format=".1%"),
                    alt.Tooltip("alpha:Q", format=".2f"),
                    alt.Tooltip("beta:Q", format=".2f"),
                ],
            )
            .properties(height=max(300, 20 * bandit_df["segment_key"].nunique()))
        )
        st.altair_chart(heatmap, width='stretch')

    st.header("Attribution: control vs. treated")
    attribution_df = load_attribution()
    if attribution_df.empty:
        st.info("No attribution data yet — the daily batch updater (or an early-reported outcome) populates this.")
    else:
        st.dataframe(
            attribution_df.style.format(
                {"control_recovery_rate": "{:.1%}", "treated_recovery_rate": "{:.1%}", "uplift_pct": "{:+.1f}pp"},
                na_rep="—",
            ),
            width='stretch',
            hide_index=True,
        )
        low_data = attribution_df[attribution_df["confidence"] != "reliable"]
        if not low_data.empty:
            st.warning(
                f"{len(low_data)} segment(s) have fewer than {MIN_CONTROL_ACCOUNTS} control accounts — "
                "their baseline (and uplift) isn't reliable yet."
            )

    st.header("Recovery funnel")
    funnel_df = load_funnel()
    # st.bar_chart sorts categorical axes alphabetically by default, which
    # scrambles a funnel's stage order -- an explicit Altair chart with
    # sort=<stage list> keeps it in the actual failed -> triggered ->
    # approved -> recovered sequence.
    funnel_chart = (
        alt.Chart(funnel_df)
        .mark_bar()
        .encode(
            x=alt.X("stage:N", title=None, sort=funnel_df["stage"].tolist()),
            y=alt.Y("count:Q", title="Accounts"),
            tooltip=["stage", "count"],
        )
    )
    st.altair_chart(funnel_chart, width='stretch')
    st.dataframe(funnel_df, width='stretch', hide_index=True)

    st.header("Reward over time")
    reward_df = load_reward_over_time()
    if reward_df.empty:
        st.info("No finalized rewards yet — rewards appear once a recovery window closes and gets processed.")
    else:
        pivoted = reward_df.pivot(index="date", columns="tactic_name", values="reward")
        st.line_chart(pivoted)


render()
