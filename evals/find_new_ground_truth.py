"""
One-off diagnostic, NOT part of the permanent eval suite: checks candidate
AML trials (found via general web search, not yet verified against the live
API) for (a) still RECRUITING, (b) real eligibility criteria + age range,
(c) rank within SEARCH_RESULT_CAP for the two query terms already used
elsewhere in this eval suite ("leukemia", "acute myeloid leukemia").

Why this exists: the two ground-truth trials retrieval_eval.py has been using
(NCT05886049, NCT05101551) are still RECRUITING on ClinicalTrials.gov, but
have drifted beyond rank 200 for both query terms as of this run -- a live
ranking-volatility problem, not a fixture data-entry problem (see
retrieval_eval.py's docstring for the full investigation). This script finds
their replacements.

Candidates below were sourced from a general web search for actively-
recruiting, newly-diagnosed adult AML trials (Aug 2026) -- NOT yet confirmed
against the live API. That's what this script does.

Once this confirms which candidates are both RECRUITING and findable within
cap, their real eligibility text gets copied into a new, SEPARATE fixtures
block. The original graded TEST_CASES/REAL_TRIALS in fixtures.py are not
touched by this script or by anything downstream of it.

Run from the trial_agent_api directory:
    python evals/find_new_ground_truth.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import SEARCH_RESULT_CAP, real_search_clinicaltrials_gov  # noqa: E402

# Sourced via web search (Aug 2026), not yet verified against the live API --
# that's what this script does. One entry per candidate, with a short note on
# why it was picked (mostly: large/multi-site Phase 3 trials are more likely
# to rank well against a broad "leukemia" query than a small single-site one).
CANDIDATES = [
    "NCT06852222",  # cAMeLot-2: Phase 3, bleximenib+venetoclax+azacitidine, newly dx AML w/ KMT2A or NPM1, not candidate for intensive chemo
    "NCT06802523",  # venetoclax + ASTX-727 + targeted radiotherapy, newly dx AML
    "NCT06386302",  # chidamide + venetoclax + azacitidine, newly dx AML
    "NCT04023526",  # cusatuzumab + azacitidine, newly dx AML not candidate for intensive chemo
    "NCT03164057",  # epigenetic priming, newly dx AML
]

QUERIES = ["leukemia", "acute myeloid leukemia"]
PULL = 200  # pull past SEARCH_RESULT_CAP so a near-miss is still visible


def check_trial(nct_id: str) -> dict:
    """Direct lookup -- real status, age range, and full eligibility text,
    straight from ClinicalTrials.gov, not the search index."""
    resp = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", timeout=15)
    if resp.status_code == 404:
        return {"nct_id": nct_id, "status": "NOT_FOUND_ON_CTGOV"}
    resp.raise_for_status()
    proto = resp.json()["protocolSection"]
    elig = proto.get("eligibilityModule", {})
    return {
        "nct_id": nct_id,
        "status": proto["statusModule"]["overallStatus"],
        "title": proto["identificationModule"]["briefTitle"],
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "eligibility_text": elig.get("eligibilityCriteria", ""),
    }


def check_rank(nct_id: str, query: str) -> int | None:
    """Same real search function search_node calls in production -- rank
    here is the exact rank a real session's search would produce."""
    pulled = real_search_clinicaltrials_gov(query, max_results=PULL)
    ids = [t["nct_id"] for t in pulled]
    return ids.index(nct_id) + 1 if nct_id in ids else None


def main() -> None:
    print(f"SEARCH_RESULT_CAP = {SEARCH_RESULT_CAP}  |  pulling top {PULL} per query\n")
    print("=" * 70)
    for nct_id in CANDIDATES:
        info = check_trial(nct_id)
        print(f"\n{nct_id}: status={info.get('status')}  age={info.get('min_age')}-{info.get('max_age')}")
        print(f"  {info.get('title', '')[:100]}")
        print(f"  conditions: {info.get('conditions')}")

        if info.get("status") != "RECRUITING":
            print("  SKIP -- not RECRUITING, not a usable ground-truth candidate")
            continue

        any_within_cap = False
        for q in QUERIES:
            rank = check_rank(nct_id, q)
            if rank is None:
                print(f"  query {q!r}: NOT FOUND in top {PULL}")
            else:
                within = rank <= SEARCH_RESULT_CAP
                any_within_cap = any_within_cap or within
                tag = "WITHIN CAP -- usable" if within else "beyond cap"
                print(f"  query {q!r}: rank {rank} of {PULL}  ({tag})")

        if any_within_cap:
            print(f"  >> CANDIDATE CONFIRMED for at least one query term. Full eligibility_text:")
            print(f"  {info['eligibility_text'][:500]}...")

    print("\n" + "=" * 70)
    print("Paste this full output back and the confirmed candidates (RECRUITING +")
    print("within-cap for at least one query) will become the new retrieval_eval.py")
    print("ground truth, in a separate fixtures block from the original TEST_CASES.")


if __name__ == "__main__":
    main()
