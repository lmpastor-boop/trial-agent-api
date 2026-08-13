from evals.regression_gate import check_gate, score_results


def case(expected="Likely eligible", predicted="Likely eligible", parsed=True, faithful=True):
    return {
        "case_id": 1, "expected": expected, "predicted": predicted,
        "parsed": parsed, "faithful": faithful,
    }


def test_gate_passes_healthy_results():
    metrics, failures = check_gate({"cases": [case()]})
    assert metrics == {"accuracy": 1.0, "parse_success": 1.0, "faithfulness": 1.0}
    assert failures == []


def test_gate_blocks_accuracy_regression():
    bad = case(predicted="Likely not eligible")
    _, failures = check_gate({"cases": [bad]})
    assert any("accuracy" in failure for failure in failures)


def test_gate_blocks_unparsed_output():
    _, failures = check_gate({"cases": [case(parsed=False)]})
    assert any("parse_success" in failure for failure in failures)


def test_score_rejects_empty_result_set():
    try:
        score_results({"cases": []})
    except ValueError as exc:
        assert "no cases" in str(exc)
    else:
        raise AssertionError("empty result set should fail closed")
