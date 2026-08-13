"""Fail a build when recorded candidate behavior falls below safety thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "accuracy": 0.85,
    "parse_success": 1.0,
    "faithfulness": 0.95,
}


def score_results(payload: dict) -> dict[str, float]:
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("results contain no cases")
    required = {"case_id", "expected", "predicted", "parsed", "faithful"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"case missing fields: {sorted(missing)}")
    total = len(cases)
    return {
        "accuracy": sum(c["predicted"] == c["expected"] for c in cases) / total,
        "parse_success": sum(bool(c["parsed"]) for c in cases) / total,
        "faithfulness": sum(bool(c["faithful"]) for c in cases) / total,
    }


def check_gate(payload: dict, thresholds: dict[str, float] | None = None) -> tuple[dict, list[str]]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metrics = score_results(payload)
    failures = [
        f"{name}={metrics[name]:.3f} below threshold={minimum:.3f}"
        for name, minimum in thresholds.items()
        if metrics[name] < minimum
    ]
    return metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path(__file__).with_name("baseline_results.json"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    metrics, failures = check_gate(payload)
    report = {
        "passed": not failures,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "thresholds": DEFAULT_THRESHOLDS,
        "failures": failures,
        "results_file": str(args.results),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
