import pytest

from app import mcp_server


def test_age_screen_tool_is_deterministic_and_requires_review():
    result = mcp_server.screen_trial_age("NCT-DEMO", 45, 18, 75)
    assert result["passes_age_gate"] is True
    assert result["requires_physician_review"] is True


def test_age_screen_tool_rejects_invalid_age():
    with pytest.raises(ValueError, match="patient_age"):
        mcp_server.screen_trial_age("NCT-DEMO", 999, 18, 75)


def test_search_tool_passes_bounded_input_to_registry(monkeypatch):
    seen = {}

    def fake_search(condition, max_results):
        seen.update(condition=condition, max_results=max_results)
        return []

    monkeypatch.setattr(mcp_server, "real_search_clinicaltrials_gov", fake_search)
    assert mcp_server.search_recruiting_trials(" leukemia ", 5) == []
    assert seen == {"condition": "leukemia", "max_results": 5}
