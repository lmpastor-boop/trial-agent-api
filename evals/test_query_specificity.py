"""
One-off diagnostic, NOT part of the permanent eval suite: tests whether a
MORE SPECIFIC query term (e.g. "NPM1 mutated leukemia" instead of a bare
"leukemia") gets a known-real trial to rank meaningfully higher in
ClinicalTrials.gov's own relevance ranking.

Why this exists: find_new_ground_truth.py already showed that 0 of 5 real,
currently-RECRUITING, prominent AML trials (including a large multi-site
Phase 3) rank within SEARCH_RESULT_CAP=100 for "leukemia" or "acute myeloid
leukemia" -- competing against a 611-trial pool of currently-recruiting AML
trials. That's not a fixture problem, it's a search-term problem: production
(app/main.py's /match endpoint) passes whatever free-text `diagnosis_code`
the caller supplies straight into `query.cond` with zero refinement. If a
NARROWER term (built from the patient's actual subtype/mutation, which a
real caller often has) ranks these trials dramatically higher, that's a real,
implementable fix -- sharpen the query, not just widen the pull.

Only tests the 2 candidates from find_new_ground_truth.py that are actually
RECRUITING: NCT06852222 (targets KMT2A rearrangements or NPM1 mutations --
a strong subtype signal to test) and NCT06386302 (no biomarker in its title;
included as a control to see whether specificity helps even without an
obvious subtype hook, and its full eligibility text is printed here so a
better term can be picked if the ones tried don't move the needle).

Run from the trial_agent_api directory:
    python evals/test_query_specificity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import SEARCH_RESULT_CAP, real_search_clinicaltrials_gov  # noqa: E402

PULL = 300  # past SEARCH_RESULT_CAP, so "found but beyond cap" is distinguishable from "not found"

# Each trial paired with a battery of query terms to test, from generic
# (already known to fail) to progressively more specific, built from what
# that trial's own eligibility text says it targets.
TRIALS = {
    "NCT06852222": {
        "note": "cAMeLot-2 (bleximenib+ven+aza), targets KMT2A rearrangements or NPM1 mutations",
        "queries": [
            "leukemia",
            "acute myeloid leukemia",
            "NPM1 mutated leukemia",
            "NPM1 mutation acute myeloid leukemia",
            "KMT2A rearranged leukemia",
            "KMT2A rearrangement acute myeloid leukemia",
        ],
    },
    "NCT06386302": {
        "note": "chidamide+ven+aza, no biomarker named in title -- control case",
        "queries": [
            "leukemia",
            "acute myeloid leukemia",
            "newly diagnosed acute myeloid leukemia",
        ],
    },
}


def fetch_full(nct_id: str) -> dict:
    resp = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", timeout=15)
    resp.raise_for_status()
    proto = resp.json()["protocolSection"]
    elig = proto.get("eligibilityModule", {})
    return {
        "status": proto["statusModule"]["overallStatus"],
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "eligibility_text": elig.get("eligibilityCriteria", ""),
    }


def rank_for_query(nct_id: str, query: str) -> int | None:
    pulled = real_search_clinicaltrials_gov(query, max_results=PULL)
    ids = [t["nct_id"] for t in pulled]
    return ids.index(nct_id) + 1 if nct_id in ids else None


def main() -> None:
    print(f"SEARCH_RESULT_CAP = {SEARCH_RESULT_CAP}  |  pulling top {PULL} per query\n")
    print("=" * 70)

    for nct_id, spec in TRIALS.items():
        info = fetch_full(nct_id)
        print(f"\n{nct_id}: {spec['note']}")
        print(f"  status: {info['status']}  |  conditions: {info['conditions']}")
        print(f"  full eligibility text (for picking better query terms if needed):")
        print(f"  {info['eligibility_text'][:800]}")
        print()

        best_rank = None
        best_query = None
        for q in spec["queries"]:
            rank = rank_for_query(nct_id, q)
            if rank is None:
                print(f"  {q!r:45s} -> NOT FOUND in top {PULL}")
            else:
                within = "WITHIN CAP" if rank <= SEARCH_RESULT_CAP else "beyond cap"
                print(f"  {q!r:45s} -> rank {rank:4d} of {PULL}  ({within})")
                if best_rank is None or rank < best_rank:
                    best_rank, best_query = rank, q

        if best_rank is not None:
            print(f"  >> BEST: {best_query!r} at rank {best_rank}")
            if best_rank <= SEARCH_RESULT_CAP:
                print(f"     Sharper query term brings it within cap. Specificity fix works for this trial.")
            else:
                print(f"     Sharper query term still beyond cap ({SEARCH_RESULT_CAP}), even if better than generic.")
        else:
            print(f"  >> None of the tested terms found this trial in top {PULL}.")

    print("\n" + "=" * 70)
    print("Paste this full output back. If specific terms consistently rank these trials")
    print("far higher than 'leukemia'/'acute myeloid leukemia', that validates sharpening")
    print("the search query (using patient subtype/mutation info, not just diagnosis name)")
    print("as the real fix -- not just raising SEARCH_RESULT_CAP.")


if __name__ == "__main__":
    main()
