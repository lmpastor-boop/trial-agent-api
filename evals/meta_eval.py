"""
Meta-eval: run the independent accuracy_judge and faithfulness_eval against the
same 8 hand-labeled cases the capstone's 7/8 (88%) result came from, and check
whether the judge agrees with the ground truth you already established by hand.

Run from the trial_agent_api directory:
    python evals/meta_eval.py

Needs ANTHROPIC_API_KEY set (reads it from .env in the repo root automatically).
Makes ~24 real Anthropic calls total (8 cases x [1 match + 1 judge + 1 faithfulness]),
so this costs real (small) money and takes a minute or two. See evals/README.md
for a cost estimate.
"""
from __future__ import annotations

import os
import sys

# Make `app` importable regardless of the working directory this is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.agent import real_match_trial  # noqa: E402
from evals.evaluators import accuracy_judge, faithfulness_eval  # noqa: E402
from evals.fixtures import REAL_TRIALS, TEST_CASES  # noqa: E402


def main() -> None:
    rows = []
    for i, case in enumerate(TEST_CASES, 1):
        trial = REAL_TRIALS[case["trial"]]
        match_result = real_match_trial(case["patient"], trial)

        your_call = "correct" if match_result["verdict"] == case["ground_truth"] else "incorrect"
        judge = accuracy_judge(case["patient"], trial["eligibility_text"], match_result["verdict"], match_result["rationale"])
        faith = faithfulness_eval(trial["eligibility_text"], match_result["verdict"], match_result["rationale"])

        rows.append({
            "case": i,
            "trial": case["trial"],
            "ground_truth": case["ground_truth"],
            "llm_verdict": match_result["verdict"],
            "your_call": your_call,
            "judge_call": judge["label"],
            "faithfulness": faith["label"],
        })

        print(f"--- Case {i}: {case['trial']} ---")
        print("Patient:            ", case["patient"][:90], "...")
        print("Ground truth:       ", case["ground_truth"])
        print("LLM verdict:        ", match_result["verdict"], "  <-- MATCH" if your_call == "correct" else "  <-- MISMATCH")
        print("Judge call:         ", judge["label"], f"({judge['explanation'][:120]})")
        print("Faithfulness:       ", faith["label"], f"({faith['explanation'][:120]})")
        print()

    # --- Summary ---
    n = len(rows)
    your_accuracy = sum(r["your_call"] == "correct" for r in rows) / n
    agreement = sum(r["your_call"] == r["judge_call"] for r in rows) / n
    faithful_rate = sum(r["faithfulness"] == "faithful" for r in rows) / n

    # Precision/recall of the judge on the "incorrect" (failure) class -- the
    # class we most care about catching, per the workshop's framing.
    human_fail = {r["case"] for r in rows if r["your_call"] == "incorrect"}
    judge_fail = {r["case"] for r in rows if r["judge_call"] == "incorrect"}
    tp = len(human_fail & judge_fail)
    precision = tp / len(judge_fail) if judge_fail else float("nan")
    recall = tp / len(human_fail) if human_fail else float("nan")

    print("=" * 60)
    print(f"Your (ground-truth) accuracy:  {sum(r['your_call']=='correct' for r in rows)}/{n} ({your_accuracy:.0%})")
    print(f"Judge agreement with you:      {sum(r['your_call']==r['judge_call'] for r in rows)}/{n} ({agreement:.0%})")
    print(f"Faithfulness pass rate:        {sum(r['faithfulness']=='faithful' for r in rows)}/{n} ({faithful_rate:.0%})")
    print(f"Judge precision on failures:   {precision:.0%}" if judge_fail else "Judge precision on failures:   n/a (judge found no failures)")
    print(f"Judge recall on failures:      {recall:.0%}" if human_fail else "Judge recall on failures:      n/a (no real failures in this set)")
    print()
    if agreement >= 0.875:  # 7/8 or better
        print("Judge agreement is strong enough to trust for scaling -- see evals/scale_eval.py.")
    else:
        print("Judge disagreement is notable. Read the mismatched cases' judge explanations above --")
        print("per the workshop's point, this often reveals ambiguity in the rubric, not a judge failure.")


if __name__ == "__main__":
    main()
