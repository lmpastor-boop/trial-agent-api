"""
Measure real token usage instead of estimating it.

Runs the live pipeline once and reports actual input/output tokens from the
Anthropic response, per trial and per session. Replaces the guessed
25,000 / 4,000 figures in the cost table with measured ones.

    python measure_cost.py --n 10

Why per-trial matters: the Matching stage makes one model call per trial that
clears the age filter, so session cost is (tokens per trial) x (trials scored).
The single biggest cost lever is the search result cap, not the model.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anthropic import Anthropic  # noqa: E402

from app.agent import (  # noqa: E402
    MATCH_SYSTEM_PROMPT,
    real_search_clinicaltrials_gov,
)

# Claude Sonnet pricing, USD per million tokens.
IN_PER_M = 3.00
OUT_PER_M = 15.00
CACHED_IN_PER_M = 0.30  # cache reads bill at 10% of base input

SAMPLE_PATIENT = (
    "Maria, 45F. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. "
    "ECOG 1. Normal organ function. No prior transplant."
)


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


def main(n: int) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set.")

    client = Anthropic()
    trials = real_search_clinicaltrials_gov("leukemia", max_results=n)
    if not trials:
        sys.exit("Search returned nothing.")

    print(f"Measuring {len(trials)} Match calls...\n")
    rows = []
    for t in trials:
        try:
            row = measure_one(client, SAMPLE_PATIENT, t)
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
    session_cost = tot_in / 1e6 * IN_PER_M + tot_out / 1e6 * OUT_PER_M

    print("\n" + "=" * 58)
    print(f"PER TRIAL      {avg_in:>8.0f} input   {avg_out:>6.0f} output")
    print(f"PER SESSION    {tot_in:>8} input   {tot_out:>6} output   "
          f"({len(rows)} trials scored)")
    print(f"SESSION COST   ${session_cost:.4f}")
    print("=" * 58)

    # The system prompt and patient summary repeat on every call in a session,
    # so caching them is the lever that actually moves the bill.
    sys_tokens = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        system=MATCH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": SAMPLE_PATIENT}],
    ).input_tokens
    repeated = sys_tokens * (len(rows) - 1)
    saved = repeated / 1e6 * (IN_PER_M - CACHED_IN_PER_M)
    print(f"\nRepeated prefix (system + patient) is ~{sys_tokens} tokens, resent "
          f"{len(rows) - 1} times.")
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
    p.add_argument("--n", type=int, default=10,
                   help="trials to pull and score (default 10, the pipeline's own cap)")
    main(p.parse_args().n)
