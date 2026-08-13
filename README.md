# Clinical Trial Matching Agent API

A FastAPI wrapper around the LangGraph agent from the capstone, for a
demo/portfolio deployment -- not a HIPAA-compliant production build. See
"Before this touches real patients" at the bottom for what that would
actually require.

## What's different from the Colab notebook

- **Real code, not notebook cells.** The graph, nodes, and DB layer live in
  `app/agent.py` and `app/db.py`, ported directly from the validated
  Section 14 real-API version (7/8, 88% on the hand-labeled test set --
  see the capstone writeup).
- **Postgres-ready.** `app/db.py` uses SQLAlchemy instead of raw `sqlite3`,
  so the exact same code runs against a local SQLite file (no setup) or a
  real Postgres database (set `DATABASE_URL`), selected automatically.
- **Explicit human commit point.** `POST /match` creates ranked trials in a
  pending review. A named reviewer must call `/reviews/{id}/decision` before
  the recommendation is approved or rejected. Separate `/feedback` data can
  still improve confidence-weighted lessons after review.

## Endpoints

| Method | Path        | Purpose                                            |
|--------|-------------|-----------------------------------------------------|
| GET    | `/health`   | Public liveness check                                |
| POST   | `/match`    | Create rankings and a pending human review           |
| GET    | `/reviews/{id}` | Read the pending or decided review              |
| POST   | `/reviews/{id}/decision` | Approve/reject once, with rationale     |
| POST   | `/feedback` | Log trial-level physician feedback                   |
| GET    | `/lessons`  | Read current confidence-weighted memory              |
| GET    | `/audit-events` | Read PHI-minimized operational audit events      |

Full request/response schemas are auto-documented at `/docs` once running
(FastAPI's built-in Swagger UI).

All routes except `/health` require `X-API-Key`; supply `X-Actor-ID` so
review and audit records identify the synthetic/demo reviewer. A `/match`
response is always pending until the one-time decision endpoint records a
human approval or rejection. A second decision receives HTTP 409.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

`app/main.py` calls `load_dotenv()` before anything else, so `.env` loads
automatically -- no manual `export` step needed.

No `DATABASE_URL` needed for local dev -- it falls back to a SQLite file
(`agent_memory.db`) created automatically on first run.

Verify it actually works before deploying anywhere:

```bash
python test_smoke.py
```

This exercises the full pipeline -- the LangGraph control flow, the
age-gate, the DB read/write, and the request/response schemas -- with the
two genuinely external calls (live ClinicalTrials.gov search, live Claude
Match call) mocked out, so it runs without network access or API credits.
It's not a substitute for hitting the real endpoints with a real key at
least once, just a fast way to catch a broken wire before you do.

For the repeatable offline regression suite used in continuous integration:

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m evals.regression_gate
```

The tests mock both external services, exercise the API contract, verify that
the deterministic age gate prevents unnecessary model calls, and check that
reviewer feedback persists and aggregates correctly.

The regression gate fails closed if verdict accuracy, output parsing, or
rationale faithfulness falls below its declared threshold. GitHub Actions runs
that gate before the optional Render deployment hook, so a failed evaluation
blocks deployment.

## MCP tools

Run the stdio MCP server with:

```bash
python -m app.mcp_server
```

It publishes two deliberately small, reusable tools:

- `search_recruiting_trials` queries ClinicalTrials.gov using a condition.
- `screen_trial_age` applies the deterministic age rule without an LLM.

The MCP server does not accept free-text patient summaries and cannot approve
a recommendation. Those operations remain behind the authenticated HTTP API,
keeping the tool boundary safer and the human commit point explicit.

## Portfolio roadmap

The current API is the foundation of a larger forward-deployed engineering
portfolio project. [`docs/PORTFOLIO_ROADMAP.md`](docs/PORTFOLIO_ROADMAP.md)
defines the user, product boundary, success criteria, implementation
milestones, and the evidence each milestone should produce.

## Evals

`evals/` holds offline evaluation scripts plus the CI regression gate; none
are invoked inside the live API request path.

**Match accuracy:** a faithfulness check (does the Match rationale actually
stick to the real trial text?) and an independent judge, meta-evaluated
against the eight hand-labeled cases before being trusted to score larger
batches of live trials. `baseline_results.json` records the validated
capstone run, and `regression_gate.py` turns its accuracy, parsing, and
faithfulness metrics into deployment-blocking thresholds. See
`evals/README.md` and `EVAL_SKETCH.md` for commands, design rationale, and
real results, including the `max_tokens` truncation bug the evaluations
exposed and helped fix in `real_match_trial()`.

**Retrieval recall:** `evals/retrieval_eval.py` measures a different failure
mode neither of the above can see -- a trial the patient genuinely qualifies
for that search never surfaces at all, so it's never even scored. Run with
`python evals/retrieval_eval.py`. This is what found and confirmed two real,
live bugs: `SEARCH_RESULT_CAP` (10 -> 100; ground-truth trials were ranking
82nd-83rd), and a discarded `Condition` API field the disease-relevance
filter needed. It also drove `classify_disease_relevance` +
`AMBIGUOUS_RELEVANCE_CAP` in `app/agent.py` (uncapped for confidently-
relevant trials, capped for ambiguous ones -- a real 611-trial competing AML
pool makes an unbounded "keep everything ambiguous" filter too expensive),
and `build_search_query`, which sharpens the search term for patients whose
summary names a validated point-mutation biomarker (currently just NPM1 --
see that function's docstring in `app/agent.py` for exactly what's
validated vs. not). `evals/find_new_ground_truth.py` and
`evals/test_query_specificity.py` are the one-off diagnostics that found and
validated that fix; not part of the ongoing eval suite, kept for reference.

**Cost:** `measure_cost.py` runs the real `search_node` +
`validate_hard_criteria_node` (not a hardcoded stand-in pull) and reports
real measured token cost per session, including a relevant-vs-ambiguous
breakdown of what's actually driving it. Run with `python measure_cost.py`.

**Python 3.13 note:** `requirements.txt` pins `anthropic<0.100` and
`langgraph<0.4`. Both packages' newest release trains (as of mid-2026)
use a very new TypedDict feature (`extra_items`, from PEP 728) that isn't
compatible yet with Python 3.13's `typing` module, and crash on import
with `TypeError: _TypedDictMeta.__new__() got an unexpected keyword
argument 'extra_items'`. If a future `pip install` without these pins
hits that error again, it means one of these packages (or a new
transitive dependency) shipped the same pattern -- pin it down the same
way and it'll resolve to an older, compatible version automatically.

## Deploying (demo/portfolio path)

**1. Database -- Neon (free Postgres, no credit card).**
Sign up at neon.tech, create a project, copy the connection string it
gives you (starts with `postgresql://`).

**2. Hosting -- Render (free web service tier).**
- Push this directory to a GitHub repo.
- On render.com: New → Web Service → connect the repo.
- Render auto-detects the `Dockerfile`, or if you'd rather skip Docker,
  set Build Command to `pip install -r requirements.txt` and Start
  Command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Add `ANTHROPIC_API_KEY`, `AUTH_API_KEY`, `ALLOWED_ORIGINS`, and
  `DATABASE_URL` (the Neon connection string from step 1).
- Deploy. Render gives you a public URL like
  `https://your-app.onrender.com` -- `/health` should return `{"status":
  "ok"}`, and `/docs` gives you a UI to try `/match` without writing any
  client code.

Railway and Fly.io both work the same way (connect repo, set the same two
env vars, deploy) if you'd rather use one of those instead of Render.

**3. Tracing -- LangSmith (optional but worth it).**
Sign up at smith.langchain.com, get an API key, set `LANGCHAIN_TRACING_V2=true`,
`LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` as additional env vars on
whatever host you picked. Every `/match` call then shows up in the
LangSmith UI as a full trace of the graph run -- which node ran, how long
each step took, and the exact prompt sent to Claude in the Match step.
Useful for debugging, and it's the same "observability layer" the
capstone's Recommendations section calls for.

## Before this touches real patients

This is a demo deployment, not a compliant one. The capstone's Security
section already says this system handles PHI and falls under HIPAA --
concretely, before any real patient data touches this:

- **A BAA has to be in place** before calling Anthropic's API with real
  patient summaries. Either accept Anthropic's BAA directly (requires an
  Enterprise-tier account and a Primary Owner's sign-off), or route the
  Match call through AWS Bedrock instead, where AWS's existing BAA covers
  the whole path since Bedrock runs inside your own VPC. Anthropic's own
  BAA explicitly does not extend to using their API through a third-party
  cloud reseller, so these are two genuinely different compliance stories,
  not interchangeable options.
- **Portfolio authentication is not clinical authentication.** The demo uses
  a constant-time API-key check and actor IDs, but a real deployment needs
  OAuth/OIDC, role-based authorization, key rotation, and managed secrets.
- **Audit events intentionally exclude patient-summary text.** Production
  retention, access, export, and tamper-evidence policies still require formal
  design and review.
- **Free-tier hosting has cold starts and no uptime guarantee** -- fine
  for a portfolio piece, not fine for something a physician is relying on
  mid-workflow.
