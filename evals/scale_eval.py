"""
Scale the accuracy_judge (once meta_eval.py shows it's trustworthy) to real,
live trials pulled fresh from ClinicalTrials.gov -- no hand-labeling required.

Run from the trial_agent_api directory:
    python evals/scale_eval.py               # 20 live trials (default)
    python evals/scale_eval.py --n 50         # 50 live trials

Only run this AFTER meta_eval.py shows strong judge/human agreement (>=87.5%,
i.e. 7/8 or better) -- otherwise you're scaling an untrustworthy judge.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import real_match_trial, real_search_clinicaltrials_gov  # noqa: E402
from evals.evaluators import accuracy_judge  # noqa: E402

# Same patient used throughout the FastAPI /match testing, for continuity.
SAMPLE_PATIENT_SUMMARY = (
    "45-year-old female, newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype, "
    "ECOG 1, normal organ function, no prior treatment except hydroxyurea."
)
SAMPLE_PATIENT_AGE = 45
SEARCH_QUERY = "leukemia"


def main(n: int) -> None:
    candidates = real_search_clinicaltrials_gov(SEARCH_QUERY, max_results=n)

    # Same age-only hard gate as validate_hard_criteria_node in app/agent.py --
    # only judge trials the patient would actually reach the Match step for.
    eligible_candidates = [t for t in candidates if t["min_age"] <= SAMPLE_PATIENT_AGE <= t["max_age"]]
    print(f"Pulled {len(candidates)} live trials, {len(eligible_candidates)} passed the age gate.\n")

    results = []
    for trial in eligible_candidates:
        match_result = real_match_trial(SAMPLE_PATIENT_SUMMARY, trial)
        judged = accuracy_judge(
            SAMPLE_PATIENT_SUMMARY, trial["eligibility_text"], match_result["verdict"], match_result["rationale"]
        )
        results.append({"nct_id": trial["nct_id"], **match_result, "judge_label": judged["label"], "judge_explanation": judged["explanation"]})
        print(f"{trial['nct_id']}: {match_result['verdict']}  |  judge: {judged['label']}")
        print(f"   rationale: {match_result['rationale'][:160]}")
        print(f"   judge:     {judged['explanation'][:160]}")

    if not results:
        print("No trials passed the age gate -- nothing to score.")
        return

    correct = sum(r["judge_label"] == "correct" for r in results)
    from collections import Counter
    verdict_counts = Counter(r["verdict"] for r in results)

    print()
    print("=" * 60)
    print(f"Judge-scored accuracy across {len(results)} live trials: {correct}/{len(results)} ({correct/len(results):.0%})")
    print(f"Verdict distribution: {dict(verdict_counts)}")
    majority_share = max(verdict_counts.values()) / len(results)
    if majority_share >= 0.75:
        print(f"WARNING: {majority_share:.0%} of results share one verdict. A high accuracy number here is")
        print("mostly measuring the easy majority-class calls, not the harder distinguishing cases your")
        print("8-case hand-labeled set targets. Don't cite this figure as equivalent to that one.")
    print("(Cross-check a handful of these by hand before citing this number --")
    print(" the judge is only as good as its meta-eval agreement score.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of live trials to pull and score")
    args = parser.parse_args()
    main(args.n)
