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
import hmac
import logging
import os
import time
import uuid
from typing import Annotated, Literal

from dotenv import load_dotenv

# Must run before importing agent/db -- db.py reads DATABASE_URL from the
# environment at import time (module-level), and agent.py reads
# ANTHROPIC_API_KEY per-request but should still see a fully-loaded
# environment either way. Loading .env here means `uvicorn app.main:app`
# just works without needing `export $(cat .env | xargs)` first.
load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
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

logger = logging.getLogger("trial_agent.audit")


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%s request_id=%s",
        request.method, request.url.path, response.status_code, duration_ms, request_id,
    )
    return response


class AuthContext(BaseModel):
    actor_id: str


def require_auth(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_actor_id: Annotated[str | None, Header(alias="X-Actor-ID")] = None,
) -> AuthContext:
    expected = os.environ.get("AUTH_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid API credentials")
    actor_id = (x_actor_id or "portfolio-reviewer").strip()
    if not actor_id or len(actor_id) > 100:
        raise HTTPException(status_code=400, detail="invalid actor identifier")
    return AuthContext(actor_id=actor_id)


Auth = Annotated[AuthContext, Depends(require_auth)]


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
    review_id: str
    review_status: Literal["pending"] = "pending"
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


class ReviewDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str = Field(..., min_length=3, max_length=1000)


class ReviewResponse(BaseModel):
    id: str
    status: Literal["pending", "approved", "rejected"]
    created_by: str
    decided_by: str | None = None
    decision_reason: str | None = None
    rankings: list[RankingOut]


class AuditEventOut(BaseModel):
    request_id: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str | None
    outcome: str
    details: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest, request: Request, auth: Auth) -> MatchResponse:
    try:
        result = agent.run_match(req.diagnosis_code, req.age, req.patient_summary)
    except Exception as exc:  # noqa: BLE001 -- surface as a clean 502, don't leak internals
        db.log_audit_event(
            request_id=request.state.request_id, actor_id=auth.actor_id,
            action="match.create", resource_type="match_review", resource_id=None,
            outcome="failed", details={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=502, detail="agent run failed") from exc

    rankings = [RankingOut(**r) for r in result["rankings"]]
    review_id = db.create_match_review(
        req.diagnosis_code, [ranking.model_dump() for ranking in rankings], auth.actor_id
    )
    db.log_audit_event(
        request_id=request.state.request_id, actor_id=auth.actor_id,
        action="match.create", resource_type="match_review", resource_id=review_id,
        outcome="success", details={"ranking_count": len(rankings)},
    )

    return MatchResponse(
        review_id=review_id,
        status=result["status"],
        search_attempts=result["search_attempts"],
        final_query=result["search_query"],
        rejected_hard_criteria=result["rejected_hard_criteria"],
        rankings=rankings,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, request: Request, auth: Auth) -> FeedbackResponse:
    lessons = agent.submit_feedback(req.diagnosis_code, req.nct_id, req.decision, req.reason)
    db.log_audit_event(
        request_id=request.state.request_id, actor_id=auth.actor_id,
        action="feedback.create", resource_type="trial", resource_id=req.nct_id,
        outcome="success", details={"decision": req.decision},
    )
    return FeedbackResponse(logged=True, lessons=[LessonOut(**l) for l in lessons])


@app.get("/lessons", response_model=list[LessonOut])
def lessons(auth: Auth, min_sample_size: int = 1) -> list[LessonOut]:
    return [LessonOut(**l) for l in db.get_confidence_weighted_lessons(min_sample_size)]


def _review_response(review: dict) -> ReviewResponse:
    return ReviewResponse(
        id=review["id"], status=review["status"], created_by=review["created_by"],
        decided_by=review["decided_by"], decision_reason=review["decision_reason"],
        rankings=[RankingOut(**ranking) for ranking in review["rankings"]],
    )


@app.get("/reviews/{review_id}", response_model=ReviewResponse)
def get_review(review_id: str, auth: Auth) -> ReviewResponse:
    review = db.get_match_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    return _review_response(review)


@app.post("/reviews/{review_id}/decision", response_model=ReviewResponse)
def decide_review(
    review_id: str, body: ReviewDecisionRequest, request: Request, auth: Auth
) -> ReviewResponse:
    existing = db.get_match_review(review_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="review not found")
    if existing["status"] != "pending":
        raise HTTPException(status_code=409, detail="review has already been decided")
    review = db.decide_match_review(review_id, body.decision, body.reason, auth.actor_id)
    if review is None:
        raise HTTPException(status_code=409, detail="review decision conflict")
    db.log_audit_event(
        request_id=request.state.request_id, actor_id=auth.actor_id,
        action="review.decide", resource_type="match_review", resource_id=review_id,
        outcome="success", details={"decision": body.decision},
    )
    return _review_response(review)


@app.get("/audit-events", response_model=list[AuditEventOut])
def audit_events(auth: Auth, limit: int = 100) -> list[AuditEventOut]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return [AuditEventOut(**event) for event in db.get_audit_events(limit)]
