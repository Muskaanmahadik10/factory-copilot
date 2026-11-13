import pytest

from factory_copilot.agent import Agent, ModelRouter
from factory_copilot.data.schemas import AnomalyEvent, IncidentReport
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.llm import MockLLM
from factory_copilot.tools import build_default_registry


@pytest.fixture(scope="module")
def registry():
    return build_default_registry(ApprovalGate())


def test_read_tools_return_structured_results(registry):
    s = registry.call("query_sensors", {"unit_id": "U002", "last_n_cycles": 30})
    assert "most_deviated" in s and len(s["sensors"]) == 12
    h = registry.call("get_maintenance_history", {"unit_id": "U002"})
    assert h["tickets"] and {"ticket_id", "technician_note"} <= set(h["tickets"][0])
    r = registry.call("search_sops", {"query": "coolant pressure falling", "k": 2})
    assert r["hits"][0]["citation"].startswith("SOP-C-201")
    rul = registry.call("estimate_rul", {"unit_id": "U002"})
    assert 0 <= rul["rul_cycles"] <= 125 and rul["policy_ref"] == "SOP-G-004"


def test_invalid_args_are_rejected_not_executed(registry):
    out = registry.call("query_sensors", {"unit_id": "not-a-unit"})
    assert out["error"] == "invalid arguments"
    assert registry.call("no_such_tool", {})["error"].startswith("unknown tool")


def test_write_tool_goes_to_approval_gate(registry):
    out = registry.call("open_ticket", {"unit_id": "U003", "root_cause": "coolant_leak", "recommended_action": "stop_and_inspect_coolant_circuit", "evidence_summary": "x"})
    assert out["status"] == "pending_approval"
    assert registry.gate.get(out["action_id"]).status == "pending"


def test_injection_in_ticket_notes_is_redacted(registry):
    # Planted payloads live on units U001, U008, U015 (see data.generate); pull a big history and check all are redacted
    h = registry.call("get_maintenance_history", {"unit_id": "U001", "include_similar_units": True, "limit": 20})
    notes = " ".join(t["technician_note"] for t in h["tickets"])
    assert "Ignore all previous instructions" not in notes
    assert "[REDACTED" in notes
    assert registry.detector.findings


def test_agent_end_to_end_with_mock(registry):
    agent = Agent(registry, router=ModelRouter(cheap=MockLLM("mock-cheap"), strong=MockLLM("mock-strong")), max_steps=8)
    ev = AnomalyEvent(unit_id="U002", cycle=200, score=0.9, top_sensors=["coolant_pressure_bar", "coolant_flow_l_min", "coolant_temp_c"])
    res = agent.investigate(ev)
    assert isinstance(res.report, IncidentReport)
    assert res.report.root_cause == "coolant_leak"
    assert res.report.sop_reference == "SOP-C-201"
    assert res.report.requires_approval is True
    assert any(e.source == "search_sops" and e.citation.startswith("SOP-C-201") for e in res.report.evidence)
    tools_called = [t.name for t in res.trace if t.kind == "tool"]
    assert tools_called[0] == "query_sensors" and "search_sops" in tools_called
    assert "open_ticket" not in tools_called  # agent recommends, never executes
    assert res.steps <= 8 and res.error == ""
    assert res.citation_check["ok"] and "SOP-C-201" in res.citation_check["cited_sop_ids"] and not res.fallback
