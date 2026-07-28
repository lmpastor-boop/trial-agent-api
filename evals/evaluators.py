"""
Two independent LLM-as-judge evaluators, run OFFLINE against outputs from
app.agent.real_match_trial(). Neither is wired into the live API (see
EVAL_SKETCH.md for why: this is scoped as offline scripts for now).

Both deliberately make a SEPARATE Anthropic call from the Match step itself --
an evaluator that reused real_match_trial()'s own reasoning wouldn't tell you
anything (it would just agree with itself).
"""
from __future__ import annotations

import json
import os

from anthropic import Anthropic

FAITHFULNESS_PROMPT = """You are auditing an AI's clinical trial eligibility rationale for accuracy.

TRIAL ELIGIBILITY CRITERIA (ground truth source):
{eligibility_text}

AI'S VERDICT: {verdict}
AI'S RATIONALE: {rationale}

Does the rationale's specific factual claims about the trial's criteria actually appear in
the eligibility criteria above? Flag any claim that is not supported by the source text,
even if the final verdict seems reasonable.

Respond with ONLY JSON: {{"label": "faithful" or "unfaithful", "explanation": "..."}}"""

JUDGE_PROMPT = """You are an independent auditor checking an AI's clinical trial eligibility verdict.

PATIENT SUMMARY:
{patient_summary}

TRIAL ELIGIBILITY CRITERIA:
{eligibility_text}

AI'S VERDICT: {verdict}
AI'S RATIONALE: {rationale}

Independently re-read the criteria and the patient summary. Is the AI's verdict correct?

Respond with ONLY JSON: {{"label": "correct" or "incorrect", "explanation": "..."}}"""


def _call_json(prompt: str) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        # 500, not 300: at 300 some explanations were getting cut off mid-string,
        # producing invalid JSON that fell back to "unparsed" -- inflating the
        # apparent failure rate with token-budget noise rather than real signal.
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    # Same markdown-fence-stripping bug we already fixed once in app.agent --
    # Claude wraps JSON in ```json ... ``` even when told not to.
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"label": "unparsed", "explanation": f"[unparsed judge output] {raw}"}


def faithfulness_eval(eligibility_text: str, verdict: str, rationale: str) -> dict:
    """Does the rationale's claims actually appear in the trial's real eligibility text?"""
    return _call_json(
        FAITHFULNESS_PROMPT.format(eligibility_text=eligibility_text, verdict=verdict, rationale=rationale)
    )


def accuracy_judge(patient_summary: str, eligibility_text: str, verdict: str, rationale: str) -> dict:
    """Independent second opinion on whether the verdict is correct."""
    return _call_json(
        JUDGE_PROMPT.format(
            patient_summary=patient_summary,
            eligibility_text=eligibility_text,
            verdict=verdict,
            rationale=rationale,
        )
    )
