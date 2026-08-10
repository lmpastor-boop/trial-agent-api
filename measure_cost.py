"""
Measure real token usage instead of estimating it.

Runs the ACTUAL pipeline nodes -- search_node then validate_hard_criteria_node
-- and reports real input/output tokens for a Match call on every candidate
that node would really pass through to Match. Replaces the guessed
25,000 / 4,000 figures in the cost table with measured ones.

    python measure_cost.py

Previously this pulled its own fixed n trials directly via
real_search_clinicaltrials_gov, bypassing search_node's real
SEARCH_RESULT_CAP and validate_hard_criteria_node's age + disease-relevance
filter entirely -- so it was measuring "cost of Matching n arbitrary
trials," not the real per-session cost. Reusing the actual node functions
means this can't drift out of sync with production the way that could.

Why per-trial matters: the Matching stage makes one model call per trial
that clears the hard-criteria gate, so session cost is
(tokens per trial) x (trials that survive the gate). Since raising
SEARCH_RESULT_CAP raises how many trials can survive that gate, this number
should be re-measured any time that constant changes -- it is not stable
across the retrieval-recall fix the way the old n=10 assumption implied.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anthropic import Anthropic  # noqa: E402

from app.agent import (  # noqa: E402
    AMBIGUOUS_RELEVANCE_CAP,
    MATCH_SYSTEM_PROMPT,
    SEARCH_RESULT_CAP,
    classify_disease_relevance,
    search_node,
    validate_hard_criteria_node,
)

# Claude Sonnet pricing, USD per million tokens.
IN_PER_M = 3.00
OUT_PER_M = 15.00
CACHED_IN_PER_M = 0.30  # cache reads bill at 10% of base input

SAMPLE_PATIENT = (
    "Maria, 45F. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. "
    "ECOG 1. Normal organ function. No prior transplant."
)
SAMPLE_PATIENT_AGE = 45
SAMPLE_QUERY = "leukemia"


def measure_one(client: Anthropic, patient_summary: str, trial: dict) -> dict:
    """One Match call, returning the real token counts the API reports."""
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        temperature=0,
        system=MATCH_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"PATIENT:\n{patient_summary}\n\n"
                f"TRIAL {trial['nct_id']} -- {trial['title']}\n"
                f"ELIGIBILITY CRITERIA:\n{trial['eligibility_text']}"
            ),
        }],
    )
    return {
        "nct_id": trial["nct_id"],
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }


def get_real_validated_candidates(query: str, patient_age: int, patient_summary: str) -> list[dict]:
    """Runs the ACTUAL search_node and validate_hard_criteria_node in
    sequence, exactly as the compiled graph would, without needing the full
    LangGraph/DB machinery. Whatever comes out of this is exactly what a
    real session would send to Match -- not an approximation of it."""
    state = {
        "search_query": query,
        "patient_age": patient_age,
        "patient_summary": patient_summary,
        "candidates": [],
    }
    state.update(search_node(state))
    state.update(validate_hard_criteria_node(state))
    return state["validated_candidates"], state["rejected_hard_criteria"]


def main(query: str, patient_summary: str, patient_age: int, limit: int | None) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set.")

    client = Anthropic()
    print(f"Pulling up to SEARCH_RESULT_CAP={SEARCH_RESULT_CAP} raw candidates for {query!r}, "
          f"then applying the real hard-criteria gate...")
    validated, rejected = get_real_validated_candidates(query, patient_age, patient_summary)
    print(f"{len(validated)} trial(s) survived (age + disease-relevance); "
          f"{len(rejected)} rejected by the hard-criteria gate.\n")

    # Breakdown: of the survivors, how many are confidently "relevant"
    # (never capped, kept no matter how many there are) versus "ambiguous"
    # (capped at AMBIGUOUS_RELEVANCE_CAP)? This answers whether a high
    # session cost is the ambiguous-cap default being too loose, or just
    # reflects a genuinely large population of confidently-matching trials
    # -- those need different fixes, so don't guess which one it is.
    relevant_n = sum(
        1 for t in validated if classify_disease_relevance(patient_summary, t.get("conditions", [])) == "relevant"
    )
    ambiguous_n = len(validated) - relevant_n
    print("Relevance breakdown of survivors:")
    print(f"  relevant (uncapped, confident subtype match):  {relevant_n}")
    print(f"  ambiguous (capped at {AMBIGUOUS_RELEVANCE_CAP}):                    {ambiguous_n}")
    if relevant_n > ambiguous_n:
        print(f"  >> Most of the cost below comes from confidently-relevant trials, not the")
        print(f"     ambiguous cap. Lowering AMBIGUOUS_RELEVANCE_CAP further won't move this much --")
        print(f"     this is a real population of matching trials, not filter leakage.\n")
    else:
        print(f"  >> The ambiguous bucket is the bigger share. AMBIGUOUS_RELEVANCE_CAP is the lever")
        print(f"     that actually controls this cost.\n")

    if not validated:
        sys.exit("No trials survived the hard-criteria gate -- nothing to measure.")

    to_measure = validated[:limit] if limit else validated
    if limit and limit < len(validated):
        print(f"Measuring the first {limit} of {len(validated)} real validated candidates "
              f"(--limit set; spend control, not a full session).\n")
    else:
        print(f"Measuring all {len(to_measure)} real validated candidates "
              f"-- this IS what a real session would cost.\n")

    rows = []
    for t in to_measure:
        try:
            row = measure_one(client, patient_summary, t)
        except Exception as e:
            print(f"  {t['nct_id']}: failed ({e})")
            continue
        rows.append(row)
        cost = row["input_tokens"] / 1e6 * IN_PER_M + row["output_tokens"] / 1e6 * OUT_PER_M
        print(f"  {row['nct_id']}: {row['input_tokens']:>6} in, "
              f"{row['output_tokens']:>4} out  = ${cost:.4f}")

    if not rows:
        sys.exit("No successful calls.")

    tot_in = sum(r["input_tokens"] for r in rows)
    tot_out = sum(r["output_tokens"] for r in rows)
    avg_in = tot_in / len(rows)
    avg_out = tot_out / len(rows)
    measured_cost = tot_in / 1e6 * IN_PER_M + tot_out / 1e6 * OUT_PER_M

    # If --limit truncated the real candidate count, the measured total isn't
    # the real session cost -- extrapolate using the per-trial average so the
    # headline number still answers "what would the real session cost."
    session_cost = measured_cost * (len(validated) / len(rows)) if rows else 0.0

    print("\n" + "=" * 58)
    print(f"PER TRIAL           {avg_in:>8.0f} input   {avg_out:>6.0f} output")
    print(f"MEASURED THIS RUN   {tot_in:>8} input   {tot_out:>6} output   "
          f"({len(rows)} of {len(validated)} real candidates)")
    print(f"MEASURED COST       ${measured_cost:.4f}")
    if len(rows) < len(validated):
        print(f"EXTRAPOLATED SESSION COST (all {len(validated)} real candidates): ${session_cost:.4f}")
    else:
        print(f"REAL SESSION COST   ${session_cost:.4f}")
    print("=" * 58)

    # The system prompt and patient summary repeat on every call in a session,
    # so caching them is the lever that actually moves the bill.
    sys_tokens = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        system=MATCH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": patient_summary}],
    ).input_tokens
    repeated = sys_tokens * (len(validated) - 1)
    saved = repeated / 1e6 * (IN_PER_M - CACHED_IN_PER_M)
    print(f"\nRepeated prefix (system + patient) is ~{sys_tokens} tokens, resent "
          f"{len(validated) - 1} times across the real {len(validated)}-trial session.")
    if sys_tokens < 1024:
        print(f"Below Sonnet's 1,024-token minimum cacheable prompt length -- caching this")
        print(f"prefix would do nothing, regardless of the theoretical ${saved:.4f} above.")
    else:
        print(f"Caching it would save about ${saved:.4f} per session "
              f"({saved / session_cost * 100:.0f}% of session cost).")

    print("\nMonthly projections at 2 sessions/user/month:")
    for users, label in ((1, "1 user (pilot, 8 sessions)"),
                         (1_000, "1,000 users"),
                         (1_000_000, "1,000,000 users")):
        sessions = 8 if users == 1 else users * 2
        print(f"  {label:<30} {sessions:>10,} sessions   "
              f"${sessions * session_cost:>12,.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, default=SAMPLE_QUERY,
                    help="search term to test (default matches the rest of the eval suite)")
    p.add_argument("--patient", type=str, default=SAMPLE_PATIENT,
                    help="patient summary text")
    p.add_argument("--patient-age", type=int, default=SAMPLE_PATIENT_AGE)
    p.add_argument("--limit", type=int, default=None,
                    help="measure only the first N real validated candidates, for spend "
                         "control during testing; omit to measure the full real session")
    args = p.parse_args()
    main(args.query, args.patient, args.patient_age, args.limit)
