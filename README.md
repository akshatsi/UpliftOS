# AI Revenue Recovery System

A backend-heavy Python system that works failed payments back to a successful
charge. A contextual Thompson Sampling bandit picks which of 10 recovery
tactics (immediate retry, discount email, SMS, human escalation, ...) to try
for each account; a two-agent LLM crew drafts the customer-facing message and
checks it against hardcoded policy rules before anything goes out; and a
causal attribution engine — a held-out 12% control group that never gets a
real tactic — measures whether a tactic actually *caused* the recovery rather
than just correlating with it. Everything runs on synthetic data (there's no
real payment gateway); a FastAPI layer exposes the pipeline, a daily batch job
closes the reward loop, and a read-only Streamlit dashboard makes the whole
thing observable.

## Architecture

```
  data/synthetic_generator.py  (seed.py)
        |
        |  500 synthetic failed-payment accounts, seed=42 (deterministic)
        v
  api/  (FastAPI)
        |  POST /recover/{id}, POST /outcomes/{id}
        |  GET  /accounts, /bandit/state, /attribution/baselines
        |
        |  control group?  --yes-->  no_action, logged, done (no bandit, no LLM)
        |  no
        v
  bandit/thompson_sampling.py
        |  select arm: Thompson Sampling, per segment
        |  (decline_reason x amount_tier x account_type) x eligible tactic
        v
  agents/crew.py
        |  drafting_agent -> guardrail_agent  ---->  Groq API (openai/gpt-oss-20b)
        |  (falls back to soft_nudge_email if guardrail rejects with no fix)
        v
  db/
        |  accounts . recovery_actions . outcomes . bandit_state . segment_baselines
        |  (SQLite in dev, same schema on Postgres --
        |   every module above reads/writes here; nothing is held in memory)
        ^
        |  outcomes with processed_at IS NULL and a closed recovery window
        |
  bandit/updater.py  (APScheduler, daily @ 2am)
        |  reward.py:            reward = amount_recovered * uplift_indicator - tactic_cost
        |  attribution/uplift.py: refreshes the control-vs-treated baseline per segment
        |  thompson_sampling.py:  alpha/beta update, persisted back to db/
        v
  dashboard/app.py  (Streamlit, read-only, reloads db/ every 30s)
        tactic performance . bandit arm heatmap . attribution . recovery funnel . reward over time
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` to a real key — the drafting and guardrail
agents need it for any non-control-group recovery. `GROQ_MODEL` defaults to
`openai/gpt-oss-20b`, one of only two Groq-hosted models that support strict
JSON-schema structured outputs (the other is `openai/gpt-oss-120b`) — swap it
only for another strict-mode model, or expect the malformed-output retry path
to fire a lot more often. Everything else in `.env` has a working default
(SQLite, a 7-day recovery window, a 2am daily batch run).

```bash
python seed.py                                    # creates tables, seeds 500 accounts + 30 days of simulated outcomes
uvicorn api.main:app --reload --port 8000          # API, from the project root
streamlit run dashboard/app.py                     # dashboard, in a second terminal
```

Open `http://localhost:8000/docs` for interactive API docs, and
`http://localhost:8501` for the dashboard.

## Simulating a full recovery cycle end to end

Pick an account, trigger recovery, then report how it turned out:

```bash
# 1. Find an account to work with
curl -s "http://localhost:8000/accounts?page_size=5" | python3 -m json.tool

# 2. Trigger the pipeline for one of those account_ids
curl -s -X POST "http://localhost:8000/recover/<account_id>" | python3 -m json.tool

# 3. Report the outcome once you know whether (and how much) the customer paid
curl -s -X POST "http://localhost:8000/outcomes/<account_id>" \
  -H "Content-Type: application/json" \
  -d '{"recovered": true, "amount_recovered": 250.0, "recovered_at": "2026-08-24T12:00:00Z"}' \
  | python3 -m json.tool
```

Step 2 branches on `is_control_group`:

- **Control-group account** (~12% of accounts) — bypasses the bandit and the
  LLM entirely and returns immediately:
  ```json
  {"status": "control_group", "action": "no_action"}
  ```
  This path needs no API key and always works, which makes it the fastest way
  to sanity-check the API without touching Groq.
- **Everyone else** — the bandit picks a tactic, the drafting agent writes the
  message, the guardrail agent reviews it, and the response looks like (real
  output, `whatsapp_nudge` selected for a low-amount B2C account):
  ```json
  {
    "tactic_selected": "whatsapp_nudge",
    "message_drafted": {
      "message_body": "Hey {{first_name}}, we noticed your recent payment of ₹248.91 didn't go through. Could you please try again? Just click the link below to complete the payment:\n\n[Retry Payment]\n\nIf you have any questions, we're here to help!",
      "channel": "whatsapp",
      "subject": null,
      "discount_offered": null,
      "tone": "warm"
    },
    "guardrail_approved": true,
    "action_logged_at": "2026-08-24T10:27:50.255555Z"
  }
  ```
  This needs a real `GROQ_API_KEY`. Without one, the call still returns a safe
  structured error (`{"error": "internal server error", "code": "internal_error"}`)
  rather than a stack trace — the underlying auth failure is logged
  server-side, not shown to the caller.

Step 3's reward/bandit-update behavior depends on timing: if the account's
7-day recovery window has already closed (a late-reported outcome, or a
control account, which isn't window-gated at all), the reward is computed and
the bandit updated immediately — `reward_finalized: true` in the response. If
the window is still open, the outcome is recorded but reward computation is
deferred to the next daily batch run (`reward_finalized: false`) — the spec is
explicit that reward is never computed on a partial window.

To watch the bandit actually learn, run several cycles across different
accounts, wait for (or manually trigger) the daily batch, then check:

```bash
curl -s http://localhost:8000/bandit/state | python3 -m json.tool
curl -s http://localhost:8000/attribution/baselines | python3 -m json.tool
```

or just look at the dashboard's bandit-arm heatmap and attribution panel.

## Running tests

```bash
pytest
```

No API key needed — every Groq call in the suite is mocked
(`pytest-mock`), and every test runs against a fresh in-memory database, never
the dev `revenue_recovery.db` file.

## Known limitations

- **Synthetic data only.** `data/synthetic_generator.py` produces
  statistically-plausible accounts and `seed.py` simulates outcomes with a
  simple probability model — there's no real payment gateway anywhere in this
  system, and none of the recovery rates reflect real-world behavior.
- **The bandit needs volume to converge.** With 30 segments and up to 9
  eligible arms each, most segment/tactic pairs stay near their `alpha=1,
  beta=1` prior for a long time on a few hundred accounts. Read the bandit
  heatmap's estimated win rates as low-confidence until real volume
  accumulates — the attribution panel's `confidence` column (reliable vs. low
  data) is there specifically to flag this per segment.
- **Unreported outcomes are invisible to the bandit.** The daily batch updater
  (`bandit/updater.py`) only processes outcomes that already exist in the
  database — an account whose recovery window closes with nobody ever having
  called `POST /outcomes` doesn't get auto-expired into a `recovered=false`
  record. In this synthetic system nothing else creates that record either, so
  such accounts silently never contribute a learning signal. A production
  system would need a payment-gateway webhook (or an explicit expiry sweep) to
  close that gap.
- **No migration tooling.** The schema is written to be portable to Postgres
  without changes (generic column types, no SQLite-specific constructs), but
  there's no Alembic setup — `Base.metadata.create_all()` only creates tables
  that don't exist yet, it doesn't handle schema changes to existing ones.

## Notable design decisions

A few places in the spec were ambiguous or (in one case) self-contradictory;
here's how they were resolved, in case the reasoning matters later:

- **Structured output on Groq.** Unlike Anthropic's SDK, Groq's Python client
  has no `.parse()` helper — `agents/llm_client.py` builds a
  `response_format={"type": "json_schema", ...}` request by hand, then does
  `json.loads()` + `output_format.model_validate()` itself. Groq's strict mode
  also requires every property in `required` (optional fields become
  present-but-nullable, never absent) and `additionalProperties: false`,
  which Pydantic's own `model_json_schema()` doesn't produce by default — see
  `_strict_schema()`. Verified against the real API, not just the docs.
- **No customer name field.** The account schema has no name/PII field at
  all, but the guardrail policy allows "PII... beyond first name" in message
  bodies. The drafting agent is instructed to use a literal `{{first_name}}`
  merge-tag placeholder instead of inventing a name — a real send pipeline
  would fill that in at send time.
- **`uplift_indicator`.** The spec gives two descriptions that don't quite
  match: "1 if recovered and not in control baseline" alongside a formula,
  `P(recovered|tactic,segment) - baseline_recovery_rate(segment)`. Implemented
  as the observed per-outcome indicator (1 or 0) minus the segment's control
  baseline — `1 - baseline` when recovered, `0 - baseline` when not (the
  latter is moot for reward purposes since `amount_recovered` is 0 either
  way). See `attribution/uplift.py`.
- **`POST /outcomes` vs. the daily updater.** The spec describes the outcomes
  endpoint as triggering reward computation directly, and separately says
  reward can never be computed on a partial window. Reconciled by finalizing
  reward immediately only when the window has already closed (or the account
  is a control), and deferring to the next batch run otherwise — see the
  "Simulating a full recovery cycle" section above.
- **Tactic performance's `avg_reward` vs. `net_value`.** These are
  intentionally different metrics on the dashboard: `avg_reward` is the
  bandit's own per-unit, uplift-adjusted signal; `net_value` is the raw
  aggregate `amount_recovered - tactic_cost`, with no causal adjustment. The
  dashboard captions this inline.
