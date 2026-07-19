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
- **Two-call design.** The notebook's `run_once()` took physician feedback
  as an input, because it was simulating an already-known decision for
  demo purposes. A real API can't do that -- a physician reviews results
  *after* getting them back. So this splits into `POST /match` (returns
  ranked trials) and `POST /feedback` (submitted afterward, once a
  physician has actually decided). See the docstring at the top of
  `app/main.py` for the full reasoning.

## Endpoints

| Method | Path        | Purpose                                            |
|--------|-------------|-----------------------------------------------------|
| GET    | `/health`   | Liveness check                                       |
| POST   | `/match`    | Run search → validate → match, return ranked trials |
| POST   | `/feedback` | Log a physician's accept/reject decision            |
| GET    | `/lessons`  | Read current confidence-weighted memory             |

Full request/response schemas are auto-documented at `/docs` once running
(FastAPI's built-in Swagger UI).

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
- Add environment variables: `ANTHROPIC_API_KEY` and `DATABASE_URL` (the
  Neon connection string from step 1).
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
- **CORS is wide open** (`allow_origins=["*"]`) for demo convenience --
  restrict this to your actual frontend's origin before it's public with
  real data flowing through it.
- **No auth on any endpoint.** Anyone with the URL can call `/match` or
  read `/lessons`. A real deployment needs at least an API key check, more
  realistically OAuth tied to a clinician's identity.
- **Free-tier hosting has cold starts and no uptime guarantee** -- fine
  for a portfolio piece, not fine for something a physician is relying on
  mid-workflow.
