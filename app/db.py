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
    """Create the feedback table if it doesn't exist. Call once at startup."""
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
