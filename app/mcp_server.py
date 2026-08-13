"""MCP tools for composable, independently testable trial operations.

Run over stdio:
    python -m app.mcp_server

The tools intentionally stop before LLM matching or approval. Patient free text
and consequential review decisions stay behind the authenticated HTTP API.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .agent import real_search_clinicaltrials_gov

mcp = FastMCP("clinical-trial-tools")


@mcp.tool()
def search_recruiting_trials(condition: str, max_results: int = 10) -> list[dict]:
    """Find recruiting trials from ClinicalTrials.gov for a condition.

    This tool accepts a condition, not a patient record. Do not include names,
    dates of birth, medical-record numbers, or patient-summary text.
    """
    condition = condition.strip()
    if not condition or len(condition) > 200:
        raise ValueError("condition must contain 1-200 characters")
    if max_results < 1 or max_results > 25:
        raise ValueError("max_results must be between 1 and 25")
    return list(real_search_clinicaltrials_gov(condition, max_results))


@mcp.tool()
def screen_trial_age(
    nct_id: str, patient_age: int, minimum_age: int = 0, maximum_age: int = 130
) -> dict:
    """Apply the deterministic age gate to one trial without invoking an LLM."""
    if patient_age < 0 or patient_age > 130:
        raise ValueError("patient_age must be between 0 and 130")
    if minimum_age < 0 or maximum_age > 130 or minimum_age > maximum_age:
        raise ValueError("invalid trial age range")
    eligible = minimum_age <= patient_age <= maximum_age
    return {
        "nct_id": nct_id,
        "passes_age_gate": eligible,
        "reason": (
            "age within trial range"
            if eligible
            else f"age out of range ({minimum_age}-{maximum_age})"
        ),
        "requires_physician_review": True,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
