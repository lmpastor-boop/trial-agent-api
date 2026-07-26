# Offline evals: Faithfulness + Meta-evaluation

Implements the two pieces from `EVAL_SKETCH.md` (Faithfulness eval + independent
accuracy judge / meta-evaluation), scoped as offline scripts only — nothing here
is wired into the live FastAPI app or `/match` endpoint.

Verified end-to-end with mocked Anthropic/search calls before being handed to
you, so the plumbing (imports, JSON parsing including the markdown-fence case,
aggregation math) is confirmed working. What hasn't been run yet is the *real*
version with your actual `ANTHROPIC_API_KEY` — that's the step below.

## Files

- `evaluators.py` — `faithfulness_eval()` and `accuracy_judge()`, each a separate
  real Anthropic call (not reusing Match's own reasoning).
- `fixtures.py` — the exact 8 hand-labeled `TEST_CASES` and 3 `REAL_TRIALS` from
  the capstone notebook, copied verbatim (same ground truth as your 7/8 result).
- `meta_eval.py` — run this first. Checks whether the judge agrees with your
  own hand labels.
- `scale_eval.py` — run this only after `meta_eval.py` shows strong agreement.
  Scores real, freshly-pulled live trials without any hand-labeling.

## How to run

From the `trial_agent_api` directory, with your `.env` already set up (same one
the FastAPI app uses):

```bash
python evals/meta_eval.py
```

Read the output. Each case prints your ground-truth call, the LLM's verdict,
the judge's independent call, and the faithfulness check. At the bottom:
judge agreement, faithfulness pass rate, and precision/recall on failures.

If agreement is 7/8 (87.5%) or better, move on to:

```bash
python evals/scale_eval.py --n 30
```

This pulls 30 live trials, runs Match + the judge on each one that passes the
age gate, and reports an accuracy number across however many real trials that
turns out to be (usually fewer than 30, since not all pass the age gate).

## Cost and time (rough estimates)

Based on the real token/cost numbers from your LangSmith trace of a single
`/match` call (11.3K tokens, $0.0472 for 8 Match calls — about $0.006/trial):

- `meta_eval.py`: 8 cases x 3 calls each (match + judge + faithfulness) = 24
  calls. Judge/faithfulness calls are smaller (max_tokens=300) than Match.
  Rough total: under $0.15, well under 2 minutes.
- `scale_eval.py --n 30`: roughly 15-25 eligible trials after the age gate x 2
  calls each (match + judge). Rough total: $0.10-0.25, a few minutes.

Trivial either way relative to your existing Anthropic usage on this project.

## What to do with the results

If `meta_eval.py` shows strong agreement and `scale_eval.py` gives you an
accuracy number across 20-30+ real trials, that's a legitimate upgrade to the
"n=8, single non-clinician author" limitation already in your capstone
writeup — worth a follow-up sentence or two if you want to mention it,
though you already decided the current docx is submission-ready as-is.

If `meta_eval.py` shows weak agreement, don't scale yet — read the specific
cases where the judge disagreed with you. That disagreement is itself useful
signal about where "Likely eligible" vs. "Possibly eligible (needs more info)"
is genuinely ambiguous, independent of whether you build anything further.
