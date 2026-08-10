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
     pull? If not, the search TERM/phrasing is the problem. (Tested and
     ruled out for the two frozen fixtures here: their live ClinicalTrials.gov
     records contain the exact query phrase in `conditions`, and there are
     611 other currently-recruiting AML trials competing for rank -- this is
     a volume problem, not a wording problem.)

  2. FINDABLE WITHIN THE OPERATIONAL CAP -- does it rank within
     app.agent.SEARCH_RESULT_CAP, the actual number search_node requests?
     Imported directly from app.agent, not hardcoded here, so this can't
     silently drift out of sync the way the original 10-result cap did.

  3. SURVIVES IN PRACTICE -- given the trial is within the cap, does it
     actually end up in validated_candidates once the REAL
     validate_hard_criteria_node runs over the REAL first-SEARCH_RESULT_CAP
     slate (not just this one trial in isolation)? This matters because
     "ambiguous" trials now compete for a shared AMBIGUOUS_RELEVANCE_CAP --
     a trial's own classification can be fine and it can still lose out to
     other ambiguous trials ahead of it. This is the check that would catch
     that trade-off actually costing a ground-truth case.

Before concluding a trial was "missed," this script first confirms via a
direct NCT-ID lookup that the trial is still RECRUITING. TEST_CASES was
frozen 2026-07-16; a trial that has since closed would show up as "not
found" for a reason that has nothing to do with retrieval quality.

Run from the trial_agent_api directory:
    python evals/retrieval_eval.py                    # default query "leukemia"
    python evals/retrieval_eval.py --query "AML"       # test an alternate phrasing

Only tests the 2 clean "Likely eligible" ground-truth pairs by default, plus
the 1 "Possibly eligible (needs more info)" pair as a labeled weaker signal
-- note that pair is a good stress test for the ambiguous cap specifically,
since its trial is a broad "hematologic malignancies" basket study unlikely
to contain an exact subtype match in its conditions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import (  # noqa: E402
    AMBIGUOUS_RELEVANCE_CAP,
    SEARCH_RESULT_CAP,
    classify_disease_relevance,
    real_search_clinicaltrials_gov,
    validate_hard_criteria_node,
)
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


def _extract_patient_age(patient_text: str) -> int:
    """fixtures.TEST_CASES patient strings all follow 'Name, NNx.' (e.g.
    '45F', '16M'). Age isn't the thing being tested here -- validate_hard_
    criteria_node needs *some* age to run, and every TEST_CASES patient's
    real age is already known to pass their assigned trial's range by
    construction (they're hand-labeled). Fall back to 45 (the sample patient
    used elsewhere in this eval suite) if the pattern isn't found."""
    m = re.search(r"(\d+)\s*[MF]\b", patient_text)
    return int(m.group(1)) if m else 45


def check_retrievability(nct_id: str, patient_summary: str, query: str) -> dict:
    pulled = real_search_clinicaltrials_gov(query, max_results=UNCAPPED_PULL)
    pulled_ids = [t["nct_id"] for t in pulled]
    findable_at_all = nct_id in pulled_ids
    rank = pulled_ids.index(nct_id) + 1 if findable_at_all else None
    findable_within_cap = findable_at_all and rank <= SEARCH_RESULT_CAP

    own_classification = None
    survives_in_practice = None
    if findable_within_cap:
        trial = pulled[pulled_ids.index(nct_id)]
        own_classification = classify_disease_relevance(patient_summary, trial.get("conditions", []))

        # Run the REAL node over exactly what search_node would actually
        # produce -- the first SEARCH_RESULT_CAP raw results -- not just this
        # trial in isolation, so AMBIGUOUS_RELEVANCE_CAP competition among
        # OTHER ambiguous trials in the same pull is captured too.
        state = {
            "search_query": query,
            "patient_age": _extract_patient_age(patient_summary),
            "patient_summary": patient_summary,
            "candidates": pulled[:SEARCH_RESULT_CAP],
        }
        result = validate_hard_criteria_node(state)
        survives_in_practice = nct_id in [t["nct_id"] for t in result["validated_candidates"]]

    return {
        "findable_at_all": findable_at_all,
        "rank": rank,
        "findable_within_cap": findable_within_cap,
        "own_classification": own_classification,
        "survives_in_practice": survives_in_practice,
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

        if result["findable_within_cap"] and result["survives_in_practice"]:
            verdict = (f"OK -- rank {result['rank']} of {result['n_pulled']}, within cap, "
                       f"classified {result['own_classification']!r}, survives in practice")
        elif result["findable_within_cap"] and result["own_classification"] == "irrelevant":
            verdict = (f"FILTER FAILURE -- rank {result['rank']} of {result['n_pulled']}, within cap, but "
                       f"classified irrelevant (filter regression -- check INCOMPATIBLE_SUBTYPES)")
        elif result["findable_within_cap"]:
            verdict = (f"AMBIGUOUS-CAP FAILURE -- rank {result['rank']} of {result['n_pulled']}, within cap, "
                       f"classified {result['own_classification']!r}, but lost out to "
                       f"AMBIGUOUS_RELEVANCE_CAP ({AMBIGUOUS_RELEVANCE_CAP}) competition")
        elif result["findable_at_all"]:
            verdict = (f"CAP FAILURE -- rank {result['rank']} of {result['n_pulled']}, "
                       f"beyond SEARCH_RESULT_CAP ({SEARCH_RESULT_CAP})")
        else:
            verdict = f"QUERY FAILURE -- not found in {result['n_pulled']} pulled results at all"
        print(f"{nct_id}: {verdict}")

    return rows


def main(query: str) -> None:
    print(f"Query term: {query!r}  |  SEARCH_RESULT_CAP: {SEARCH_RESULT_CAP}  |  "
          f"AMBIGUOUS_RELEVANCE_CAP: {AMBIGUOUS_RELEVANCE_CAP}  |  pulled for this eval: {UNCAPPED_PULL}")

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
            "ambiguous_relevance_cap": AMBIGUOUS_RELEVANCE_CAP,
            "n_pulled": UNCAPPED_PULL,
            "strong_positive_results": strong_rows,
            "weak_positive_results": weak_rows,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

    strong_scored = [r for r in strong_rows if not r["skipped"]]
    n = len(strong_scored)
    findable_at_all = sum(r["findable_at_all"] for r in strong_scored)
    findable_within_cap = sum(r["findable_within_cap"] for r in strong_scored)
    fully_ok = sum(bool(r["findable_within_cap"] and r["survives_in_practice"]) for r in strong_scored)

    print()
    print("=" * 64)
    print(f"STRONG ground truth: {n} case(s) scored (others skipped, see above)")
    print(f"  Findable at all:                        {findable_at_all}/{n}")
    print(f"  Findable within cap:                    {findable_within_cap}/{n}")
    print(f"  Within cap AND survives in practice:    {fully_ok}/{n}")

    if n and findable_at_all < n:
        print("  >> Some ground-truth-eligible trials cannot be found by this query term at")
        print("     all. Before assuming it's phrasing, check directly whether ClinicalTrials.gov's")
        print("     own record for that trial even contains the query text in its conditions --")
        print("     it may just be outranked by a large pool of equally-relevant trials.")
    if n and findable_at_all > findable_within_cap:
        print(f"  >> Some trials are findable but rank beyond SEARCH_RESULT_CAP ({SEARCH_RESULT_CAP}).")
    if n and findable_within_cap > fully_ok:
        print("  >> Some trials rank within the cap but didn't survive in practice. Check each")
        print("     row's own_classification above: 'irrelevant' means a filter regression;")
        print("     'ambiguous' means it lost a seat to AMBIGUOUS_RELEVANCE_CAP competition --")
        print("     a real, disclosed trade-off for cost control, not a bug.")
    if n and fully_ok == n:
        print("  All strong ground-truth trials made it all the way through in practice.")
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
