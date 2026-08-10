"""
Measures the one failure mode nothing else in this eval suite can see: the
silent false negative, where the pipeline never surfaces a trial the patient
actually qualifies for. meta_eval.py and scale_eval.py only ever score
trials that already made it into the candidate pool -- neither can detect
one that never got there.

Ground truth here is the "Likely eligible" pairs already validated in
fixtures.TEST_CASES: if a patient is genuinely eligible for a trial, the
pipeline should end up with that trial in validated_candidates, using the
same query term it would actually issue.

For each ground-truth-positive (patient, trial) pair, this checks THREE
things, in order, because each has a different fix:

  1. FINDABLE AT ALL -- does the trial appear anywhere in a large search
     pull? If not, the search TERM/phrasing is the problem.

  2. FINDABLE WITHIN THE OPERATIONAL CAP -- does it rank within
     app.agent.SEARCH_RESULT_CAP, the actual number search_node requests?
     Imported directly from app.agent, not hardcoded here, so this can't
     silently drift out of sync the way the original 10-result cap did the
     day this file was written and the day it was fixed to 100.

  3. SURVIVES THE RELEVANCE FILTER -- given the trial is findable within the
     cap, does app.agent._is_disease_relevant keep it once
     validate_hard_criteria_node runs? A trial can pass step 2 and still be
     wrongly dropped by an overzealous filter -- this is what would catch
     that regression.

Before concluding a trial was "missed," this script first confirms via a
direct NCT-ID lookup that the trial is still RECRUITING. TEST_CASES was
frozen 2026-07-16; a trial that has since closed would show up as "not
found" for a reason that has nothing to do with retrieval quality.

Run from the trial_agent_api directory:
    python evals/retrieval_eval.py                    # default query "leukemia"
    python evals/retrieval_eval.py --query "AML"       # test an alternate phrasing

Only tests the 2 clean "Likely eligible" ground-truth pairs by default, plus
the 1 "Possibly eligible (needs more info)" pair as a labeled weaker signal
(a trial that's a plausible-but-unconfirmed match should still surface for
physician review, even if Match itself should hedge on it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import SEARCH_RESULT_CAP, _is_disease_relevant, real_search_clinicaltrials_gov  # noqa: E402
from evals.fixtures import TEST_CASES  # noqa: E402

# Pull a bit past the real cap so a trial that's *just* outside it is still
# visible in the report, instead of looking identical to "not found at all."
UNCAPPED_PULL = max(SEARCH_RESULT_CAP * 2, 100)

# Strong ground truth: the model should find these and Match should say yes.
STRONG_POSITIVES = [c for c in TEST_CASES if c["ground_truth"] == "Likely eligible"]
# Weak ground truth: plausible-but-unconfirmed. Should still surface for
# review; whether Match hedges on it is a separate, already-tested question.
WEAK_POSITIVES = [c for c in TEST_CASES if c["ground_truth"] == "Possibly eligible (needs more info)"]


def check_still_recruiting(nct_id: str) -> str:
    """Direct lookup, bypassing search entirely -- confirms whether a 'not
    found in search' result is a real retrieval failure or just a trial
    that has closed since TEST_CASES was frozen."""
    resp = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", timeout=15)
    if resp.status_code == 404:
        return "NOT_FOUND_ON_CTGOV"
    resp.raise_for_status()
    status = resp.json()["protocolSection"]["statusModule"]["overallStatus"]
    return status


def check_retrievability(nct_id: str, patient_summary: str, query: str) -> dict:
    pulled = real_search_clinicaltrials_gov(query, max_results=UNCAPPED_PULL)
    pulled_ids = [t["nct_id"] for t in pulled]
    findable_at_all = nct_id in pulled_ids
    rank = pulled_ids.index(nct_id) + 1 if findable_at_all else None
    findable_within_cap = findable_at_all and rank <= SEARCH_RESULT_CAP

    survives_filter = None
    if findable_within_cap:
        trial = pulled[pulled_ids.index(nct_id)]
        survives_filter = _is_disease_relevant(patient_summary, trial.get("conditions", []))

    return {
        "findable_at_all": findable_at_all,
        "rank": rank,
        "findable_within_cap": findable_within_cap,
        "survives_filter": survives_filter,
        "n_pulled": len(pulled),
    }


def run_group(cases: list[dict], query: str, label: str) -> list[dict]:
    print(f"\n--- {label} ({len(cases)} case{'s' if len(cases) != 1 else ''}) ---")
    rows = []
    for case in cases:
        nct_id = case["trial"]
        status = check_still_recruiting(nct_id)
        if status != "RECRUITING":
            print(f"{nct_id}: SKIPPED -- status is '{status}', not RECRUITING "
                  f"(fixture may be stale, not a retrieval failure)")
            rows.append({"nct_id": nct_id, "skipped": True, "status": status})
            continue

        result = check_retrievability(nct_id, case["patient"], query)
        rows.append({"nct_id": nct_id, "skipped": False, "status": status, **result})

        if result["findable_within_cap"] and result["survives_filter"]:
            verdict = f"OK -- rank {result['rank']} of {result['n_pulled']}, within cap, passes relevance filter"
        elif result["findable_within_cap"] and not result["survives_filter"]:
            verdict = (f"FILTER FAILURE -- rank {result['rank']} of {result['n_pulled']}, within cap, but "
                       f"_is_disease_relevant excluded it (filter regression -- check INCOMPATIBLE_SUBTYPES)")
        elif result["findable_at_all"]:
            verdict = (f"CAP FAILURE -- rank {result['rank']} of {result['n_pulled']}, "
                       f"beyond SEARCH_RESULT_CAP ({SEARCH_RESULT_CAP})")
        else:
            verdict = f"QUERY FAILURE -- not found in {result['n_pulled']} pulled results at all"
        print(f"{nct_id}: {verdict}")

    return rows


def main(query: str) -> None:
    print(f"Query term: {query!r}  |  SEARCH_RESULT_CAP: {SEARCH_RESULT_CAP}  |  pulled for this eval: {UNCAPPED_PULL}")

    strong_rows = run_group(STRONG_POSITIVES, query, "STRONG ground truth (Likely eligible)")
    weak_rows = run_group(WEAK_POSITIVES, query, "WEAK ground truth (Possibly eligible -- should still surface)")

    all_scored = [r for r in strong_rows + weak_rows if not r["skipped"]]
    if not all_scored:
        print("\nEvery ground-truth case was skipped (no longer recruiting). "
              "Refresh fixtures.py with currently-recruiting trials to re-run this eval.")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"retrieval_eval_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_at": stamp,
            "query": query,
            "search_result_cap": SEARCH_RESULT_CAP,
            "n_pulled": UNCAPPED_PULL,
            "strong_positive_results": strong_rows,
            "weak_positive_results": weak_rows,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

    strong_scored = [r for r in strong_rows if not r["skipped"]]
    n = len(strong_scored)
    findable_at_all = sum(r["findable_at_all"] for r in strong_scored)
    findable_within_cap = sum(r["findable_within_cap"] for r in strong_scored)
    fully_ok = sum(bool(r["findable_within_cap"] and r["survives_filter"]) for r in strong_scored)

    print()
    print("=" * 64)
    print(f"STRONG ground truth: {n} case(s) scored (others skipped, see above)")
    print(f"  Findable at all:            {findable_at_all}/{n}")
    print(f"  Findable within cap:        {findable_within_cap}/{n}")
    print(f"  Within cap AND passes filter (end-to-end OK): {fully_ok}/{n}")

    if n and findable_at_all < n:
        print("  >> Some ground-truth-eligible trials cannot be found by this query term at")
        print("     all. That points at the search TERM/phrasing.")
    if n and findable_at_all > findable_within_cap:
        print(f"  >> Some trials are findable but rank beyond SEARCH_RESULT_CAP ({SEARCH_RESULT_CAP}).")
        print("     That points at raising the cap further or paginating.")
    if n and findable_within_cap > fully_ok:
        print("  >> Some trials rank within the cap but got excluded by the relevance filter.")
        print("     That's a filter regression, not a search problem -- check INCOMPATIBLE_SUBTYPES")
        print("     in app/agent.py for an overly broad deny term.")
    if n and fully_ok == n:
        print("  All strong ground-truth trials made it all the way through: retrievable,")
        print("  within the cap, and correctly kept by the relevance filter.")
        print(f"  This does NOT mean the false-negative risk is zero -- it means these {n}")
        print("  specific known cases aren't currently being missed. The risk from trials")
        print("  outside this small ground-truth set remains unmeasured.")

    print("\n(This eval only covers the trials in fixtures.py. A single query term and a")
    print(" small ground-truth set cannot rule out retrieval gaps for other diseases, other")
    print(" phrasings, or trials outside this fixture set -- treat this as a floor, not a")
    print(" full retrieval audit.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="leukemia",
                         help="search term to test (default matches the rest of the eval suite)")
    args = parser.parse_args()
    main(args.query)
