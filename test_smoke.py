"""
Smoke test, not part of the deployed app. Exercises the full FastAPI
pipeline in-process via TestClient. The two genuinely external calls
(live ClinicalTrials.gov search, live Anthropic Match call) are mocked --
this sandbox can't reach either (clinicaltrials.gov is proxy-blocked here,
and no API key is available) -- but every other code path, including the
LangGraph control flow, the age-gate, the DB layer, and the request/response
schemas, runs for real.

Run: DATABASE_URL=sqlite:////tmp/trial_agent_api/smoke.db python test_smoke.py
"""
import os
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/trial_agent_api/smoke.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "smoke-test-placeholder")

# SQLite will not create a missing parent directory -- it raises a bare
# "unable to open database file" instead. On any fresh machine (Colab, CI,
# a teammate's laptop) /tmp/trial_agent_api does not exist yet, so create it
# before the engine is constructed.
os.makedirs("/tmp/trial_agent_api", exist_ok=True)

if os.path.exists("/tmp/trial_agent_api/smoke.db"):
    os.remove("/tmp/trial_agent_api/smoke.db")

from fastapi.testclient import TestClient  # noqa: E402

from app import agent  # noqa: E402
from app.main import app  # noqa: E402

FAKE_TRIAL = {
    "nct_id": "NCT05551234",
    "title": "Phase II Study of Agent-X in Relapsed Disease",
    "eligibility_text": "Adults 18-75 with confirmed diagnosis; adequate organ function.",
    "min_age": 18,
    "max_age": 75,
    "location": "Boston, MA",
}


def fake_search(search_query: str, max_results: int = 10):
    return [FAKE_TRIAL] if search_query == "leukemia" else []


def fake_match(patient_summary: str, trial: dict) -> dict:
    return {"verdict": "Likely eligible", "rationale": "meets stated criteria (mocked)"}


with TestClient(app) as client:  # context manager -- triggers lifespan startup/shutdown
    print("=== /health ===")
    r = client.get("/health")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("\n=== /match (mocked search + match) ===")
    with patch.object(agent, "real_search_clinicaltrials_gov", fake_search), \
         patch.object(agent, "real_match_trial", fake_match):
        r = client.post("/match", json={
            "diagnosis_code": "leukemia",
            "age": 45,
            "patient_summary": "45F, confirmed diagnosis, ECOG 1, normal organ function.",
        })
    print(r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["rankings"][0]["nct_id"] == "NCT05551234"
    assert body["rankings"][0]["verdict"] == "Likely eligible"
    assert "physician review" in body["disclaimer"].lower()

    print("\n=== /match age-gate rejection (mocked search, no match needed) ===")
    with patch.object(agent, "real_search_clinicaltrials_gov", fake_search):
        r = client.post("/match", json={
            "diagnosis_code": "leukemia",
            "age": 10,  # below the fake trial's min_age of 18
            "patient_summary": "10M, pediatric.",
        })
    print(r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_eligible_trials"
    assert body["rankings"] == []
    assert "age out of range" in body["rejected_hard_criteria"][0]

    print("\n=== /feedback (real DB, no mocking needed) ===")
    r = client.post("/feedback", json={
        "diagnosis_code": "leukemia",
        "nct_id": "NCT05551234",
        "decision": "reject",
        "reason": "travel distance too far for the patient",
    })
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["lessons"][0]["pattern"] == "rejects trials requiring travel > 50mi"
    assert r.json()["lessons"][0]["sample_size"] == 1

    print("\n=== /lessons (real DB read) ===")
    r = client.get("/lessons")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert len(r.json()) == 1

    print("\n=== /match input validation (age out of allowed range) ===")
    r = client.post("/match", json={"diagnosis_code": "leukemia", "age": 999, "patient_summary": "x"})
    print(r.status_code)
    assert r.status_code == 422  # pydantic validation, never reaches the agent

print("\nALL SMOKE TESTS PASSED")
