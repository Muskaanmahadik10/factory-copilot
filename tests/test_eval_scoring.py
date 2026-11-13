"""The evaluation metrics check what their names say (audit findings FC-06 to FC-08, FC-12)."""

import pytest

from evaluation.run import action_matches, run_status, sop_id


def test_action_keyword_match_rejects_negation_and_escalation():
    assert action_matches("schedule_bearing_replacement", "bearing_wear")
    assert action_matches("perform_axis_calibration", "calibration_drift")
    assert not action_matches("do_not_replace_bearing", "bearing_wear")
    assert not action_matches("no bearing work needed", "bearing_wear")
    assert not action_matches("escalate_bearing_issue", "bearing_wear")
    assert not action_matches("replace_coolant_filter", "bearing_wear")
    assert not action_matches("", "bearing_wear")


def test_sop_id_must_match_exactly():
    assert sop_id("SOP-C-201 §3 Diagnosis") == "SOP-C-201"
    assert sop_id("SOP-C-2010") == ""  # a longer, nonexistent id no longer passes as a prefix match
    assert sop_id("SOP-C-201FAKE") == ""  # nor does an alphabetic suffix
    assert sop_id("see manual") == ""


def test_run_status_separates_clean_recovered_and_failed():
    assert run_status({"error": ""}) == "clean"
    assert run_status({"error": "HTTPError: 400", "fallback": False}) == "recovered"
    assert run_status({"error": "budget", "fallback": True}) == "failed"


def test_real_provider_without_key_fails_instead_of_becoming_the_mock(monkeypatch):
    from factory_copilot.config import settings
    from factory_copilot.llm.clients import LLMConfigError, make_llm
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.groq.com/openai/v1")
    with pytest.raises(LLMConfigError):
        make_llm("cheap")
    monkeypatch.setattr(settings, "llm_base_url", "http://localhost:11434/v1")  # local servers may run without a key
    assert make_llm("cheap").name == settings.cheap_model
