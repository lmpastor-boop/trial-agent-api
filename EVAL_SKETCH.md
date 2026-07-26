# Eval Sketch: Faithfulness + Meta-Evaluation for the Clinical Trial Matching Agent

Status: design sketch, not implemented. Scoped to the two pieces of the Arize workshop
that add real value on top of what's already built (LangSmith tracing, 8-case hand-labeled
accuracy test), without duplicating tracing infrastructure you already have.

No new platform account needed. This uses plain Anthropic calls in the same style as
`real_match_trial()` — or, optionally, the open-source `arize-phoenix-evals` pip package
for dataframe scaffolding, without signing up for hosted Arize AX. Recommendation below is
to start DIY, since the whole eval fits in ~2 small functions.

---

## Why these two pieces specifically

Your current validation (capstone Accuracy section) is a single hand-labeled snapshot:
you personally read 8 cases and scored the Match agent 7/8 against your own judgment. That's
honest, but it has two real limits your writeup already names: it can't scale (labeling is
manual), and it doesn't automatically re-run when you change a prompt or model.

**Faithfulness** catches a different failure mode than accuracy does: a verdict can be
directionally *correct* while the rationale still cites something not actually in the
trial's eligibility text (a subtle hallucination). Your ground-truth accuracy check never
looks at this.

**Meta-evaluation** solves the scaling problem: instead of hand-labeling more cases
yourself, build an independent LLM judge, check that it agrees with your existing 8
hand labels, and — once trusted — let it grade dozens of new live trials you'll never
manually read.

---

## Piece 1: Faithfulness eval

Checks whether `real_match_trial()`'s rationale is actually grounded in the trial's real
`eligibility_text`, not invented or drifted.

```python
# app/evals.py (new file)

from anthropic import Anthropic
import os, json

FAITHFULNESS_PROMPT = """You are auditing an AI's clinical trial eligibility rationale for accuracy.

TRIAL ELIGIBILITY CRITERIA (ground truth source):
{eligibility_text}

AI'S VERDICT: {verdict}
AI'S RATIONALE: {rationale}

Does the rationale's specific factual claims about the trial's criteria actually appear in
the eligibility criteria above? Flag any claim that is not supported by the source text,
even if the final verdict seems reasonable.

Respond with ONLY JSON: {{"label": "faithful" or "unfaithful", "explanation": "..."}}"""


def faithfulness_eval(eligibility_text: str, verdict: str, rationale: str) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": FAITHFULNESS_PROMPT.format(
                eligibility_text=eligibility_text, verdict=verdict, rationale=rationale
            ),
        }],
    )
    raw = msg.content[0].text.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)
```

Run this once per ranking returned by `match_node` — either live (call it right after
`real_match_trial()` inside `match_node`, log the result), or offline against a batch of
saved `/match` responses. Offline is cheaper to iterate on; wire it into the live node only
once the rubric is stable.

---

## Piece 2: Independent accuracy judge + meta-evaluation

The key design choice: the judge must be a **separate call** from `real_match_trial()`,
not the same call grading itself. Otherwise you're just checking that the model agrees
with itself.

```python
# app/evals.py (continued)

JUDGE_PROMPT = """You are an independent auditor checking an AI's clinical trial eligibility verdict.

PATIENT SUMMARY:
{patient_summary}

TRIAL ELIGIBILITY CRITERIA:
{eligibility_text}

AI'S VERDICT: {verdict}
AI'S RATIONALE: {rationale}

Independently re-read the criteria and the patient summary. Is the AI's verdict correct?

Respond with ONLY JSON: {{"label": "correct" or "incorrect", "explanation": "..."}}"""


def accuracy_judge(patient_summary: str, eligibility_text: str, verdict: str, rationale: str) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                patient_summary=patient_summary, eligibility_text=eligibility_text,
                verdict=verdict, rationale=rationale,
            ),
        }],
    )
    raw = msg.content[0].text.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)
```

### Meta-eval step: does the judge agree with you?

Run the judge against your existing 8 hand-labeled `TEST_CASES` from the notebook (you
already know the right answer for each). Compare judge's `correct`/`incorrect` call against
whether the Match verdict actually matched your `ground_truth`.

```python
# meta_eval.py (sketch — run offline, not part of the API)

agreements = 0
for case in TEST_CASES:
    match_result = real_match_trial(case["patient_summary"], case["trial"])
    your_call = "correct" if match_result["verdict"] == case["ground_truth"] else "incorrect"
    judge_call = accuracy_judge(
        case["patient_summary"], case["trial"]["eligibility_text"],
        match_result["verdict"], match_result["rationale"],
    )["label"]
    agreements += (your_call == judge_call)
    print(case["patient_summary"][:40], "| you:", your_call, "| judge:", judge_call)

print(f"Judge agrees with your hand labels on {agreements}/{len(TEST_CASES)} cases")
```

If agreement is high (e.g. 7-8/8, matching or close to your own accuracy number), the judge
is trustworthy enough to scale. If it disagrees, read *why* — the workshop's point that
judge disagreements usually expose ambiguity in your own rubric applies directly here too
(e.g. is "Possibly eligible" being scored as correct or incorrect when ground truth is
"Likely eligible"? That's a labeling ambiguity, not necessarily a judge failure).

### Scaling once the judge is trusted

```python
# scale_eval.py (sketch)

live_trials = real_search_clinicaltrials_gov("leukemia", max_results=50)
results = []
for trial in live_trials:
    match_result = real_match_trial(SAMPLE_PATIENT_SUMMARY, trial)
    judged = accuracy_judge(
        SAMPLE_PATIENT_SUMMARY, trial["eligibility_text"],
        match_result["verdict"], match_result["rationale"],
    )
    results.append({"nct_id": trial["nct_id"], **match_result, **judged})

accuracy = sum(r["label"] == "correct" for r in results) / len(results)
print(f"Judge-scored accuracy across {len(results)} live trials: {accuracy:.0%}")
```

This gets you an accuracy number over 50+ real trials instead of 8, without you personally
reading 50 eligibility criteria documents — directly answering the "should we scale up
testing" question from earlier, and giving your limitations section (small sample, single
author) a real upgrade path.

---

## What I would NOT do

Skip the workshop's Step 9 (auto-rewrite the agent's prompts from judge feedback, no human
review) for `MATCH_SYSTEM_PROMPT` specifically. That prompt does clinical-eligibility
reasoning; letting an LLM silently rewrite it based on another LLM's opinion removes the
human review step that makes your current results trustworthy. Use judge explanations as
input *to you* deciding whether to revise the prompt, not as an automatic rewrite loop.

## Suggested order if you build this

1. Faithfulness eval, run offline against a handful of saved `/match` responses — cheapest,
   fastest signal.
2. Meta-eval against your existing 8 `TEST_CASES` — no new data needed, just the judge.
3. Only if judge agreement is strong: scale to live trials for a bigger accuracy number.
4. Optionally log faithfulness/judge results as LangSmith feedback (same
   `client.create_feedback()` pattern you'd use for physician feedback) so they show up
   next to the existing traces, instead of standing up a second platform.
