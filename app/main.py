"""
FastAPI wrapper around the LangGraph clinical trial matching agent.

Two-call design, on purpose:
  1. POST /match     -- runs search -> validate -> match, returns ranked
                         trials for a physician to review. No feedback yet.
  2. POST /feedback   -- called AFTER a physician reviews the /match result
                         and accepts/rejects a specific trial. Logs the
                         decision and returns the updated confidence-weighted
                         lessons.

This mirrors how the real workflow happens (review, then decide) rather
than the notebook demo's single-call run_once(), which only worked because
it was simulating an already-known decision at invocation time.

Run locally:
    uvicorn app.main:app --reload

Every response includes a mandatory disclaimer -- this system is not
positioned as a final eligibility determination anywhere in this API. See
the capstone writeup's Safety section for why that's a product decision,
not just UI copy.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Must run before importing agent/db -- db.py reads DATABASE_URL from the
# environment at import time (module-level), and agent.py reads
# ANTHROPIC_API_KEY per-request but should still see a fully-loaded
# environment either way. Loading .env here means `uvicorn app.main:app`
# just works without needing `export $(cat .env | xargs)` first.
load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import agent, db  # noqa: E402

DISCLAIMER = (
    "This is not a final eligibility determination. All results require "
    "physician review before a patient acts on them."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="Clinical Trial Matching Agent API",
    description="Demo/portfolio deployment of the capstone clinical trial matching agent.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    diagnosis_code: str = Field(..., description="Search term, e.g. a condition name like 'leukemia'")
    age: int = Field(..., ge=0, le=130)
    patient_summary: str = Field(
        ..., description="Free-text patient summary handed to the Match stage's LLM call"
    )


class RankingOut(BaseModel):
    nct_id: str
    verdict: str
    rationale: str


class MatchResponse(BaseModel):
    status: str
    search_attempts: int
    final_query: str
    rejected_hard_criteria: list[str]
    rankings: list[RankingOut]
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    diagnosis_code: str
    nct_id: str
    decision: str = Field(..., pattern="^(accept|reject)$")
    reason: str = ""


class LessonOut(BaseModel):
    pattern: str
    sample_size: int
    weight: float


class FeedbackResponse(BaseModel):
    logged: bool
    lessons: list[LessonOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    try:
        result = agent.run_match(req.diagnosis_code, req.age, req.patient_summary)
    except Exception as exc:  # noqa: BLE001 -- surface as a clean 502, don't leak internals
        raise HTTPException(status_code=502, detail=f"agent run failed: {exc}") from exc

    return MatchResponse(
        status=result["status"],
        search_attempts=result["search_attempts"],
        final_query=result["search_query"],
        rejected_hard_criteria=result["rejected_hard_criteria"],
        rankings=[RankingOut(**r) for r in result["rankings"]],
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    lessons = agent.submit_feedback(req.diagnosis_code, req.nct_id, req.decision, req.reason)
    return FeedbackResponse(logged=True, lessons=[LessonOut(**l) for l in lessons])


@app.get("/lessons", response_model=list[LessonOut])
def lessons(min_sample_size: int = 1) -> list[LessonOut]:
    return [LessonOut(**l) for l in db.get_confidence_weighted_lessons(min_sample_size)]
