"""
Database layer -- ported from the notebook's sqlite3-only version to
SQLAlchemy Core so the exact same query logic works against either a local
SQLite file (dev) or Postgres (production), selected purely by DATABASE_URL.

Local dev:   no DATABASE_URL set -> falls back to sqlite:///./agent_memory.db
Production:  DATABASE_URL=postgresql://... (e.g. from Neon/Render/RDS)

The two retrieval patterns are unchanged from the notebook:
  - get_patient_history(): point lookup by diagnosis code
  - get_confidence_weighted_lessons(): GROUP BY aggregation, weight scales
    with independent sample size rather than being hardcoded
"""
from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# `or` (not just `.get(..., default)`) is deliberate here: .env.example
# ships with `DATABASE_URL=` (blank, meant to be optional), and once
# load_dotenv() reads that, os.environ["DATABASE_URL"] becomes an empty
# string rather than staying unset -- `.get()` alone would return that
# empty string instead of falling back, and create_engine("") fails to
# parse. `or` falls back on both "unset" and "empty string".
DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///./agent_memory.db"

# Render/Heroku-style URLs sometimes use the old "postgres://" scheme, which
# SQLAlchemy's psycopg2 dialect no longer accepts -- normalize it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


class Lesson(TypedDict):
    pattern: str
    sample_size: int
    weight: float


def init_db() -> None:
    """Create application, human-review, and audit tables at startup."""
    is_sqlite = _engine.dialect.name == "sqlite"
    id_col = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "SERIAL PRIMARY KEY"
    created_default = "CURRENT_TIMESTAMP" if is_sqlite else "NOW()"
    with _engine.begin() as con:
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS physician_feedback (
                id {id_col},
                patient_diagnosis_code TEXT NOT NULL,
                nct_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason_pattern TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT {created_default}
            )
        """))
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS match_reviews (
                id TEXT PRIMARY KEY,
                patient_diagnosis_code TEXT NOT NULL,
                rankings_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT NOT NULL,
                decided_by TEXT,
                decision_reason TEXT,
                created_at TIMESTAMP DEFAULT {created_default},
                decided_at TIMESTAMP
            )
        """))
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS audit_events (
                id {id_col},
                request_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                outcome TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT {created_default}
            )
        """))


def create_match_review(
    diagnosis_code: str, rankings: list[dict], created_by: str
) -> str:
    """Persist a reviewable recommendation without storing patient free text."""
    review_id = str(uuid.uuid4())
    with _engine.begin() as con:
        con.execute(
            text(
                "INSERT INTO match_reviews "
                "(id, patient_diagnosis_code, rankings_json, status, created_by) "
                "VALUES (:id, :diagnosis_code, :rankings_json, 'pending', :created_by)"
            ),
            {
                "id": review_id,
                "diagnosis_code": diagnosis_code,
                "rankings_json": json.dumps(rankings),
                "created_by": created_by,
            },
        )
    return review_id


def get_match_review(review_id: str) -> dict | None:
    with _engine.connect() as con:
        row = con.execute(
            text(
                "SELECT id, patient_diagnosis_code, rankings_json, status, "
                "created_by, decided_by, decision_reason, created_at, decided_at "
                "FROM match_reviews WHERE id = :id"
            ),
            {"id": review_id},
        ).mappings().first()
    if row is None:
        return None
    result = dict(row)
    result["rankings"] = json.loads(result.pop("rankings_json"))
    return result


def decide_match_review(
    review_id: str, decision: str, reason: str, decided_by: str
) -> dict | None:
    """Atomically decide a pending review; a decided review cannot be overwritten."""
    decided_at = datetime.now(timezone.utc)
    with _engine.begin() as con:
        result = con.execute(
            text(
                "UPDATE match_reviews SET status = :decision, decided_by = :decided_by, "
                "decision_reason = :reason, decided_at = :decided_at "
                "WHERE id = :id AND status = 'pending'"
            ),
            {
                "id": review_id,
                "decision": decision,
                "decided_by": decided_by,
                "reason": reason,
                "decided_at": decided_at,
            },
        )
        if result.rowcount == 0:
            return None
    return get_match_review(review_id)


def log_audit_event(
    *, request_id: str, actor_id: str, action: str, resource_type: str,
    resource_id: str | None, outcome: str, details: dict | None = None,
) -> None:
    """Write a minimal audit event. Callers must never pass PHI in details."""
    with _engine.begin() as con:
        con.execute(
            text(
                "INSERT INTO audit_events "
                "(request_id, actor_id, action, resource_type, resource_id, outcome, details_json) "
                "VALUES (:request_id, :actor_id, :action, :resource_type, :resource_id, :outcome, :details_json)"
            ),
            {
                "request_id": request_id,
                "actor_id": actor_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "outcome": outcome,
                "details_json": json.dumps(details or {}, sort_keys=True),
            },
        )


def get_audit_events(limit: int = 100) -> list[dict]:
    with _engine.connect() as con:
        rows = con.execute(
            text(
                "SELECT request_id, actor_id, action, resource_type, resource_id, "
                "outcome, details_json, created_at FROM audit_events "
                "ORDER BY created_at DESC, id DESC LIMIT :limit"
            ),
            {"limit": limit},
        ).mappings().all()
    events = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event.pop("details_json"))
        events.append(event)
    return events


def log_feedback(diagnosis_code: str, nct_id: str, decision: str, reason_pattern: str) -> None:
    with _engine.begin() as con:
        con.execute(
            text(
                "INSERT INTO physician_feedback "
                "(patient_diagnosis_code, nct_id, decision, reason_pattern) "
                "VALUES (:diagnosis_code, :nct_id, :decision, :reason_pattern)"
            ),
            {
                "diagnosis_code": diagnosis_code,
                "nct_id": nct_id,
                "decision": decision,
                "reason_pattern": reason_pattern,
            },
        )


def get_patient_history(diagnosis_code: str) -> list[dict]:
    """Pattern 1: point lookup -- every past decision for this diagnosis code."""
    with _engine.connect() as con:
        rows = con.execute(
            text(
                "SELECT nct_id, decision, reason_pattern, created_at "
                "FROM physician_feedback WHERE patient_diagnosis_code = :diagnosis_code "
                "ORDER BY created_at DESC"
            ),
            {"diagnosis_code": diagnosis_code},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_confidence_weighted_lessons(min_sample_size: int = 1) -> list[Lesson]:
    """Pattern 2: GROUP BY aggregation -- weight scales with how many
    independent physicians hit the same pattern, not a hardcoded value."""
    with _engine.connect() as con:
        rows = con.execute(
            text(
                "SELECT reason_pattern, COUNT(*) as n "
                "FROM physician_feedback "
                "WHERE decision = 'reject' "
                "GROUP BY reason_pattern "
                "HAVING COUNT(*) >= :min_sample_size"
            ),
            {"min_sample_size": min_sample_size},
        ).all()

    lessons: list[Lesson] = []
    for reason_pattern, n in rows:
        weight = min(n / 5, 1.0)
        lessons.append({"pattern": reason_pattern, "sample_size": n, "weight": round(weight, 2)})
    return lessons
