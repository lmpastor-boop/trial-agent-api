"""
The LangGraph agent itself -- ported from the Colab notebook's real-API
Section 14 (real_search_clinicaltrials_gov + real_match_trial), with one
architectural change from the notebook: the graph now ends at `match`
instead of continuing on into `reflect`.

Why: the notebook's run_once() passed physician_feedback in at invocation
time, because it was simulating a decision that had already happened, for
demo purposes. In a real API, a physician reviews the *returned* rankings
and submits feedback in a separate, later request -- there's no feedback to
react to yet when /match is called. So `reflect`'s logic is exposed here as
a plain function (submit_feedback) that the API's separate /feedback
endpoint calls directly, not as a graph node reached by an edge.

Everything else -- the age-only hard-criteria gate, the broaden-and-retry
loop, the confidence-weighted memory read -- is unchanged from the
validated notebook version.
"""
from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

import requests
from anthropic import Anthropic
from langgraph.graph import END, StateGraph
from langsmith.wrappers import wrap_anthropic

from . import db

MAX_SEARCH_ATTEMPTS = 2

SUBTYPE_TO_PARENT_CATEGORY: dict[str, str] = {
    # Extend this as real narrow subtype codes are observed in production.
    # In a mature build this is a real ICD-10 hierarchy lookup, not a
    # hardcoded map -- see the capstone writeup for why.
}

MATCH_VERDICTS = ["Likely eligible", "Possibly eligible (needs more info)", "Likely not eligible"]

MATCH_SYSTEM_PROMPT = """You are helping a physician triage which clinical trials a patient might qualify for.
You will be given a patient summary and one trial's real eligibility criteria (copied verbatim from ClinicalTrials.gov).
Read the FULL inclusion and exclusion criteria carefully, not just age/sex.
Respond with ONLY a JSON object: {"verdict": one of ["Likely eligible", "Possibly eligible (needs more info)", "Likely not eligible"], "rationale": a short explanation citing the SPECIFIC criterion that drove your verdict}.
Use "Possibly eligible (needs more info)" whenever a genuinely relevant criterion (biomarker status, treatment history detail, lab value) is not stated in the patient summary -- do not guess.
This is not medical advice and must not be treated as a final eligibility determination."""


class TrialCandidate(TypedDict):
    nct_id: str
    title: str
    eligibility_text: str
    min_age: int
    max_age: int
    location: str


class Ranking(TypedDict):
    nct_id: str
    verdict: str
    rationale: str


class AgentState(TypedDict):
    patient_diagnosis_code: str
    patient_age: int
    patient_summary: str  # free text handed to the Match LLM call

    candidates: list[TrialCandidate]
    validated_candidates: list[TrialCandidate]
    rankings: list[Ranking]
    lessons: Annotated[list[db.Lesson], lambda a, b: a + b]

    rejected_hard_criteria: list[str]
    status: str

    search_query: str
    search_attempts: int


# ---------------------------------------------------------------------------
# Search node -- real ClinicalTrials.gov v2 API call
# ---------------------------------------------------------------------------
def _parse_age(age_str: str | None) -> int | None:
    if not age_str or "Year" not in age_str:
        return None
    return int(age_str.split()[0])


def real_search_clinicaltrials_gov(search_query: str, max_results: int = 10) -> list[TrialCandidate]:
    resp = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={
            "query.cond": search_query,
            "filter.overallStatus": "RECRUITING",
            "pageSize": max_results,
            "fields": "NCTId,BriefTitle,EligibilityCriteria,MinimumAge,MaximumAge,Sex,Condition,OverallStatus",
        },
        timeout=15,
    )
    resp.raise_for_status()
    studies = resp.json().get("studies", [])

    candidates: list[TrialCandidate] = []
    for s in studies:
        proto = s["protocolSection"]
        elig = proto.get("eligibilityModule", {})
        candidates.append({
            "nct_id": proto["identificationModule"]["nctId"],
            "title": proto["identificationModule"]["briefTitle"],
            "eligibility_text": elig.get("eligibilityCriteria", ""),
            "min_age": _parse_age(elig.get("minimumAge")) or 0,
            "max_age": _parse_age(elig.get("maximumAge")) or 130,
            "location": "see contactsLocationsModule (fetch separately if needed)",
        })
    return candidates


def search_node(state: AgentState) -> dict:
    candidates = real_search_clinicaltrials_gov(state["search_query"])
    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Hard-criteria gate -- deterministic, age-only (see capstone writeup:
# diagnosis relevance is handled upstream by the search API's own
# query.cond matching, not a local exact-match check)
# ---------------------------------------------------------------------------
def validate_hard_criteria_node(state: AgentState) -> dict:
    validated, rejected = [], []
    for trial in state["candidates"]:
        age_ok = trial["min_age"] <= state["patient_age"] <= trial["max_age"]
        if age_ok:
            validated.append(trial)
        else:
            rejected.append(f"{trial['nct_id']}: age out of range ({trial['min_age']}-{trial['max_age']})")
    status = "ok" if validated else "no_eligible_trials"
    return {"validated_candidates": validated, "rejected_hard_criteria": rejected, "status": status}


def route_after_validation(state: AgentState) -> str:
    if state["status"] == "ok":
        return "proceed"
    if state["search_attempts"] < MAX_SEARCH_ATTEMPTS:
        return "broaden"
    return "give_up"


def broaden_query_node(state: AgentState) -> dict:
    attempts = state["search_attempts"] + 1
    current_query = state["search_query"]
    if attempts == 1:
        broadened = SUBTYPE_TO_PARENT_CATEGORY.get(current_query, current_query)
    else:
        broadened = current_query
    return {"search_query": broadened, "search_attempts": attempts}


# ---------------------------------------------------------------------------
# Memory read
# ---------------------------------------------------------------------------
def retrieve_memory_node(state: AgentState) -> dict:
    return {"lessons": db.get_confidence_weighted_lessons(min_sample_size=1)}


# ---------------------------------------------------------------------------
# Match node -- real Claude call, validated at 7/8 (88%) against a
# hand-labeled test set (see capstone writeup, Accuracy section)
# ---------------------------------------------------------------------------
def real_match_trial(patient_summary: str, trial: TrialCandidate) -> dict:
    # wrap_anthropic is a no-op unless LANGCHAIN_TRACING_V2=true is set --
    # when tracing is on, it captures this call as a proper LLM span (exact
    # prompt, response, token counts, latency) in LangSmith, instead of just
    # the node's input/output state that LangGraph traces automatically.
    client = wrap_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        # 500, not 300: found via evals/scale_eval.py against 30 live trials --
        # at 300, some rationales were getting cut off mid-string, producing
        # invalid JSON that silently fell back to a generic "needs more info"
        # default instead of the model's real (often more decisive) verdict.
        max_tokens=500,
        system=MATCH_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"PATIENT SUMMARY:\n{patient_summary}\n\nTRIAL {trial['nct_id']} ELIGIBILITY CRITERIA:\n{trial['eligibility_text']}",
        }],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"verdict": "Possibly eligible (needs more info)", "rationale": f"[unparsed LLM output] {raw}"}
    return parsed


def match_node(state: AgentState) -> dict:
    rankings: list[Ranking] = []
    for trial in state["validated_candidates"]:
        result = real_match_trial(state["patient_summary"], trial)
        rankings.append({
            "nct_id": trial["nct_id"],
            "verdict": result.get("verdict", "Possibly eligible (needs more info)"),
            "rationale": result.get("rationale", ""),
        })
    return {"rankings": rankings}


# ---------------------------------------------------------------------------
# Feedback -- called directly by the API's /feedback endpoint, NOT a graph
# node. See module docstring for why.
# ---------------------------------------------------------------------------
def normalize_reason(free_text_reason: str) -> str:
    text_lower = free_text_reason.lower()
    if "travel" in text_lower or "distance" in text_lower or "location" in text_lower:
        return "rejects trials requiring travel > 50mi"
    if "organ function" in text_lower or "eligibility" in text_lower:
        return "disputes ambiguous eligibility judgment"
    return "uncategorized"


def submit_feedback(diagnosis_code: str, nct_id: str, decision: str, reason: str) -> list[db.Lesson]:
    reason_pattern = normalize_reason(reason)
    db.log_feedback(diagnosis_code, nct_id, decision, reason_pattern)
    return db.get_confidence_weighted_lessons(min_sample_size=1)


# ---------------------------------------------------------------------------
# Graph assembly -- ends at match; reflect is not a graph node here.
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("validate_hard_criteria", validate_hard_criteria_node)
    graph.add_node("broaden_query", broaden_query_node)
    graph.add_node("retrieve_memory", retrieve_memory_node)
    graph.add_node("match", match_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "validate_hard_criteria")
    graph.add_conditional_edges(
        "validate_hard_criteria",
        route_after_validation,
        {"proceed": "retrieve_memory", "broaden": "broaden_query", "give_up": END},
    )
    graph.add_edge("broaden_query", "search")
    graph.add_edge("retrieve_memory", "match")
    graph.add_edge("match", END)

    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_match(patient_diagnosis_code: str, patient_age: int, patient_summary: str) -> dict:
    app = get_app()
    initial_state: AgentState = {
        "patient_diagnosis_code": patient_diagnosis_code,
        "patient_age": patient_age,
        "patient_summary": patient_summary,
        "candidates": [],
        "validated_candidates": [],
        "rankings": [],
        "lessons": [],
        "rejected_hard_criteria": [],
        "status": "",
        "search_query": patient_diagnosis_code,
        "search_attempts": 0,
    }
    return app.invoke(initial_state)
