"""
Error analysis for the clinical trial matching agent.

Two-pass workflow, modeled on the workshop's Step 4 but adapted to a
classification task with a heavily imbalanced label set:

    python error_analysis.py worksheet evals/runs/scale_eval_<stamp>.json
        -> writes labels.csv, one row per scored case, with blank label columns

    (you fill in the two label columns by hand, reading each case)

    python error_analysis.py report labels.csv
        -> prints the frequency tables

WHY TWO LABEL COLUMNS, NOT ONE
The instinct is to label only failures. Don't. With ~2 failures in 34 cases,
a failure-only table has no statistical content at all -- it's the same two
anecdotes in a box.

The useful finding is the DIFFICULTY distribution: what fraction of cases were
easy majority-class calls versus genuinely hard distinctions. That's what turns
"26/26 correct" into "24 of 26 were trivial disease-type mismatches; of the 3
that required a real eligibility judgment, Match got 3." The second number is
smaller, less impressive, and far more honest -- and it's the direct answer to
the false-negative risk the Safety section names as the top failure mode.

So: axis 1 is how hard the case was, axis 2 is what happened.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Axis 1: DIFFICULTY -- what kind of judgment did this case actually require?
# ---------------------------------------------------------------------------
DIFFICULTY = {
    "disease_mismatch": (
        "Trial is for a different disease entirely (CLL/ALL/CML/lymphoma vs AML). "
        "The majority class. Correct here means almost nothing."
    ),
    "structured_field": (
        "Decidable from a structured field alone (age, sex). A deterministic "
        "check would have caught it -- no LLM reasoning required."
    ),
    "disease_status": (
        "Right disease, wrong stage/timing: newly diagnosed vs relapsed vs "
        "in remission after induction. Requires reading the criteria."
    ),
    "freetext_exclusion": (
        "An exclusion buried in prose that no structured field captures "
        "(APL/FAB M3, Down syndrome, secondary AML from prior MDS)."
    ),
    "biomarker": (
        "Turns on biomarker status -- required, excluded, or unstated in the "
        "patient summary (NPM1, FLT3-ITD, CD70)."
    ),
    "numeric_threshold": (
        "A numeric threshold stated mid-paragraph, possibly against vague "
        "patient phrasing (ECOG/Lansky/Karnofsky >= N vs 'adequate')."
    ),
    "ambiguous": (
        "Genuinely underdetermined -- multi-cohort eligibility, or the patient "
        "summary lacks a fact the criteria require. Ground truth is arguable."
    ),
}

# ---------------------------------------------------------------------------
# Axis 2: OUTCOME -- what happened, and if it went wrong, why?
# ---------------------------------------------------------------------------
OUTCOME = {
    "correct": "Match's verdict was right.",
    "wrong_too_permissive": (
        "FALSE POSITIVE. Said eligible/possible when the criteria exclude the "
        "patient. Wastes physician time; caught by review."
    ),
    "wrong_too_restrictive": (
        "FALSE NEGATIVE -- the high-severity class. Said not eligible when the "
        "patient qualifies. Silent: physician review cannot catch what was "
        "never surfaced."
    ),
    "vague_hedge": (
        "Retreated to 'Possibly eligible (needs more info)' when the criteria "
        "actually did determine an answer. Not wrong, but useless -- and this "
        "is the exact value the truncation bug defaulted to."
    ),
    "unfaithful_rationale": (
        "Verdict right, but the rationale cites a criterion not in the trial "
        "text. Right answer, wrong reasoning -- only the faithfulness eval "
        "catches this."
    ),
    "ground_truth_wrong": (
        "On review, the hand label was the thing that was wrong. Count these "
        "honestly; they were 2 of your 8 meta-eval disagreements."
    ),
}


def cmd_worksheet(path: str) -> None:
    """Flatten a scale_eval run into a CSV with blank label columns."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        rows.append({
            "nct_id": r.get("nct_id", ""),
            "verdict": r.get("verdict", ""),
            "judge_label": r.get("judge_label", ""),
            # scale_eval also records a faithfulness label per trial. Pull it
            # through: it is the only signal that separates "right answer,
            # invented reasoning" from a genuinely correct case, which is the
            # unfaithful_rationale outcome below.
            "faithfulness": r.get("faithfulness", ""),
            "rationale": (r.get("rationale", "") or "").replace("\n", " ")[:300],
            "judge_explanation": (r.get("judge_explanation", "") or "").replace("\n", " ")[:300],
            "faithfulness_explanation": (r.get("faithfulness_explanation", "") or "").replace("\n", " ")[:200],
            "difficulty": "",   # <- you fill this in
            "outcome": "",      # <- and this
            "notes": "",
        })

    with open("labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote labels.csv with {len(rows)} rows.\n")
    print("Now open it and fill the 'difficulty' and 'outcome' columns.")
    print("Add your 8 hand-labeled cases as extra rows -- they're your hardest")
    print("cases and excluding them biases the difficulty table toward easy.\n")
    print("DIFFICULTY values:")
    for k, v in DIFFICULTY.items():
        print(f"  {k:<20} {v}")
    print("\n  Tip: any row where faithfulness is not 'faithful' is a candidate")
    print("  for unfaithful_rationale -- read those first, they are the cheapest wins.")
    print("\nOUTCOME values:")
    for k, v in OUTCOME.items():
        print(f"  {k:<24} {v}")
    print("\nIf you hesitate on a case, that hesitation is data -- put it in")
    print("'notes'. Ambiguity you can name is a finding; ambiguity you smooth")
    print("over is a bug in your rubric.")


def _bar(n: int, total: int, width: int = 28) -> str:
    return "#" * max(1, round(n / total * width)) if n else ""


def cmd_report(path: str) -> None:
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("difficulty")]

    if not rows:
        print("No labeled rows found. Fill in the 'difficulty' column first.")
        return

    total = len(rows)
    diff = Counter(r["difficulty"] for r in rows)
    outc = Counter(r["outcome"] for r in rows)

    print(f"\nERROR ANALYSIS -- {total} scored cases")
    print("=" * 64)

    print("\nDIFFICULTY DISTRIBUTION (what the eval set actually tested)")
    print("-" * 64)
    for k, n in diff.most_common():
        print(f"  {k:<20} {n:>3}  {n/total:>5.0%}  {_bar(n, total)}")

    trivial = diff.get("disease_mismatch", 0) + diff.get("structured_field", 0)
    hard = total - trivial
    print(f"\n  Trivial (disease mismatch + structured field): {trivial}/{total} ({trivial/total:.0%})")
    print(f"  Required real eligibility reasoning:          {hard}/{total} ({hard/total:.0%})")
    if total and trivial / total >= 0.6:
        print("\n  >> Majority of this eval set is easy. Cite accuracy on the")
        print("     'hard' subset separately; the headline number is inflated.")

    print("\nOUTCOME DISTRIBUTION")
    print("-" * 64)
    for k, n in outc.most_common():
        print(f"  {k:<24} {n:>3}  {n/total:>5.0%}  {_bar(n, total)}")

    # Accuracy restricted to cases that required real reasoning.
    hard_rows = [r for r in rows if r["difficulty"] not in ("disease_mismatch", "structured_field")]
    if hard_rows:
        hard_correct = sum(r["outcome"] == "correct" for r in hard_rows)
        print(f"\nACCURACY ON HARD CASES ONLY: {hard_correct}/{len(hard_rows)} "
              f"({hard_correct/len(hard_rows):.0%})")
        print("  ^ this is the number to put in the report, alongside the")
        print("    overall figure and the n.")

    # Severity-weighted view: false negatives are the ones that matter.
    fn = outc.get("wrong_too_restrictive", 0)
    fp = outc.get("wrong_too_permissive", 0)
    print("\nSEVERITY VIEW (frequency x severity = priority)")
    print("-" * 64)
    print(f"  False negatives (silent, high severity): {fn}")
    print(f"  False positives (caught by review):      {fp}")
    print(f"  Vague hedges (low severity, erodes trust): {outc.get('vague_hedge', 0)}")
    if fn == 0:
        print("\n  Zero observed false negatives -- but note this eval CANNOT")
        print("  measure the real false-negative risk, which is trials the")
        print("  search never returned. Every case here is a trial that was")
        print("  retrieved. State that limitation explicitly.")

    by_diff_fail = Counter(
        r["difficulty"] for r in rows if r["outcome"] not in ("correct", "ground_truth_wrong", "")
    )
    if by_diff_fail:
        print("\nFAILURES BY DIFFICULTY (where to focus next)")
        print("-" * 64)
        for k, n in by_diff_fail.most_common():
            denom = diff[k]
            print(f"  {k:<20} {n}/{denom} failed ({n/denom:.0%})")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("worksheet", "report"):
        print(__doc__)
        sys.exit(1)
    (cmd_worksheet if sys.argv[1] == "worksheet" else cmd_report)(sys.argv[2])
