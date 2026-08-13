"""API regression tests that run without network access or API credentials."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# db.py builds its SQLAlchemy engine at import time, so select an isolated
# test database before importing the application.
TEST_DB = Path("/tmp/trial_agent_api_pytest.db")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder")
os.environ["AUTH_API_KEY"] = "test-api-key"

from app import agent, db  # noqa: E402
from app.main import app  # noqa: E402


FAKE_TRIAL = {
    "nct_id": "NCT05551234",
    "title": "Phase II Study of Agent-X in Relapsed Disease",
    "eligibility_text": "Adults 18-75 with confirmed diagnosis; adequate organ function.",
    "min_age": 18,
    "max_age": 75,
    "location": "Boston, MA",
}

AUTH_HEADERS = {"X-API-Key": "test-api-key", "X-Actor-ID": "dr-test"}


@pytest.fixture(autouse=True)
def clean_database():
    """Keep tests independent while reusing the application's engine."""
    db.init_db()
    with db._engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM physician_feedback")
        connection.exec_driver_sql("DELETE FROM match_reviews")
        connection.exec_driver_sql("DELETE FROM audit_events")
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def fake_search(search_query: str, max_results: int = 10):
    return [FAKE_TRIAL] if search_query == "leukemia" else []


def fake_match(patient_summary: str, trial: dict):
    return {
        "verdict": "Likely eligible",
        "rationale": "Patient meets the stated criteria (mocked).",
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_match_returns_ranked_trial_and_safety_disclaimer(client, monkeypatch):
    monkeypatch.setattr(agent, "real_search_clinicaltrials_gov", fake_search)
    monkeypatch.setattr(agent, "real_match_trial", fake_match)

    response = client.post(
        "/match",
        headers=AUTH_HEADERS,
        json={
            "diagnosis_code": "leukemia",
            "age": 45,
            "patient_summary": (
                "45F, confirmed diagnosis, ECOG 1, normal organ function."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["review_status"] == "pending"
    assert body["review_id"]
    assert body["rankings"][0]["nct_id"] == "NCT05551234"
    assert body["rankings"][0]["verdict"] == "Likely eligible"
    assert "physician review" in body["disclaimer"].lower()


def test_age_gate_rejects_trial_without_calling_model(client, monkeypatch):
    monkeypatch.setattr(agent, "real_search_clinicaltrials_gov", fake_search)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called after deterministic rejection")

    monkeypatch.setattr(agent, "real_match_trial", fail_if_called)
    response = client.post(
        "/match",
        headers=AUTH_HEADERS,
        json={
            "diagnosis_code": "leukemia",
            "age": 10,
            "patient_summary": "10M, pediatric patient.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_eligible_trials"
    assert body["rankings"] == []
    assert "age out of range" in body["rejected_hard_criteria"][0]


def test_feedback_is_persisted_and_aggregated(client):
    response = client.post(
        "/feedback",
        headers=AUTH_HEADERS,
        json={
            "diagnosis_code": "leukemia",
            "nct_id": "NCT05551234",
            "decision": "reject",
            "reason": "Travel distance is too far for the patient.",
        },
    )

    assert response.status_code == 200
    lesson = response.json()["lessons"][0]
    assert lesson["pattern"] == "rejects trials requiring travel > 50mi"
    assert lesson["sample_size"] == 1
    assert lesson["weight"] == 0.2

    lessons_response = client.get("/lessons", headers=AUTH_HEADERS)
    assert lessons_response.status_code == 200
    assert lessons_response.json() == [lesson]


def test_request_validation_rejects_impossible_age(client):
    response = client.post(
        "/match",
        headers=AUTH_HEADERS,
        json={"diagnosis_code": "leukemia", "age": 999, "patient_summary": "x"},
    )
    assert response.status_code == 422
    assert "age" in str(response.json())


def test_feedback_rejects_unknown_decision(client):
    response = client.post(
        "/feedback",
        headers=AUTH_HEADERS,
        json={
            "diagnosis_code": "leukemia",
            "nct_id": "NCT05551234",
            "decision": "maybe",
            "reason": "",
        },
    )
    assert response.status_code == 422


def test_protected_endpoint_requires_api_key(client):
    response = client.post(
        "/match",
        json={"diagnosis_code": "leukemia", "age": 45, "patient_summary": "summary"},
    )
    assert response.status_code == 401


def test_human_review_is_required_and_decision_is_one_time(client, monkeypatch):
    monkeypatch.setattr(agent, "real_search_clinicaltrials_gov", fake_search)
    monkeypatch.setattr(agent, "real_match_trial", fake_match)
    match_response = client.post(
        "/match", headers=AUTH_HEADERS,
        json={
            "diagnosis_code": "leukemia", "age": 45,
            "patient_summary": "45F, confirmed diagnosis, normal organ function.",
        },
    )
    review_id = match_response.json()["review_id"]

    pending = client.get(f"/reviews/{review_id}", headers=AUTH_HEADERS)
    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"

    decision = client.post(
        f"/reviews/{review_id}/decision", headers=AUTH_HEADERS,
        json={"decision": "approved", "reason": "Physician verified relevant criteria."},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
    assert decision.json()["decided_by"] == "dr-test"

    duplicate = client.post(
        f"/reviews/{review_id}/decision", headers=AUTH_HEADERS,
        json={"decision": "rejected", "reason": "Attempted overwrite."},
    )
    assert duplicate.status_code == 409


def test_audit_log_excludes_patient_summary(client, monkeypatch):
    monkeypatch.setattr(agent, "real_search_clinicaltrials_gov", fake_search)
    monkeypatch.setattr(agent, "real_match_trial", fake_match)
    sensitive_text = "UNIQUE-PATIENT-SUMMARY-TEXT"
    client.post(
        "/match", headers=AUTH_HEADERS,
        json={"diagnosis_code": "leukemia", "age": 45, "patient_summary": sensitive_text},
    )
    response = client.get("/audit-events", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert sensitive_text not in response.text
    assert response.json()[0]["actor_id"] == "dr-test"
