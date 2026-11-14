"""Regression tests for the independent audit of e218e00 (14 Nov 2026), findings F1-F7."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from factory_copilot.agent import Agent, ModelRouter
from factory_copilot.citations import check_report_citations
from factory_copilot.data.schemas import AnomalyEvent
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.guardrails.core import AlreadyDecided
from factory_copilot.tools import build_default_registry
from tests.test_reaudit_fixes import REPORT, ScriptedLLM

EVENT = AnomalyEvent(unit_id="U002", cycle=100, score=1.0, top_sensors=["coolant_pressure_bar"])


# ---------- F1: citation provenance comes from returned records only ----------
def test_searching_for_an_invented_id_does_not_make_it_retrieved(tmp_path):
    registry = build_default_registry(ApprovalGate())

    def script(step, prompt):
        if step == "write_report":
            return "submit_report", REPORT | {"sop_reference": "SOP-Z-999"}
        return ("finish", {}) if "tool=search_sops" in prompt else ("search_sops", {"query": "SOP-Z-999 coolant procedure"})

    res = Agent(registry, router=ModelRouter(cheap=ScriptedLLM(script), strong=ScriptedLLM(script)), trace_dir=tmp_path).investigate(EVENT)
    search = next(t.output for t in res.trace if t.kind == "tool" and t.name == "search_sops")
    assert "SOP-Z-999" in json.dumps(search)  # the query is echoed in the tool output...
    assert res.citation_check["ok"] is False and "SOP-Z-999" in res.citation_check["unretrieved_citations"]  # ...but is not evidence


def test_an_invented_id_in_the_summary_alone_is_flagged():
    cc = check_report_citations({"summary": "Recommendation based on SOP-Z-999.", "sop_reference": "", "evidence": []}, [])
    assert cc["ok"] is False and cc["unretrieved_citations"] == ["SOP-Z-999"]


def test_cross_references_and_errors_inside_outputs_are_not_retrieved_records():
    outputs = [("search_sops", {"query": "x", "hits": [{"doc_id": "SOP-C-202", "text": "A leak shows low pressure on both sides, see SOP-C-201."}]}),
               ("get_maintenance_history", {"error": "no ticket T9999"})]
    cc = check_report_citations({"sop_reference": "SOP-C-201", "evidence": [{"citation": "T9999"}]}, outputs)
    assert cc["retrieved_sop_ids"] == ["SOP-C-202"] and cc["unretrieved_citations"] == ["SOP-C-201", "T9999"]


# ---------- F2: a pre-onset alarm is a false alarm, not a known failure ----------
def test_pre_onset_alarm_is_scored_as_a_false_alarm():
    from evaluation.run import action_matches, root_cause_correct

    cases = json.loads((__import__("factory_copilot.config", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR / "eval_cases.json").read_text(encoding="utf-8"))
    for c in cases:
        if c["event"]["cycle"] < c["onset_cycle"]:
            assert c["phase"] == "pre_onset_false_alarm" and c["truth"]["root_cause"] == "no_fault_yet" and c["truth"]["eventual_failure_mode"]
        else:
            assert c["phase"] == "post_onset" and c["truth"]["root_cause"] != "no_fault_yet"
    assert root_cause_correct("unknown", "no_fault_yet") and not root_cause_correct("coolant_leak", "no_fault_yet")
    assert action_matches("monitor_and_escalate_per_SOP-G-001", "no_fault_yet")
    assert not action_matches("stop_and_inspect_coolant_circuit", "no_fault_yet")


# ---------- F3: rescoring refuses when the truth changed under a saved run ----------
def test_rescore_refuses_changed_truth(tmp_path, monkeypatch):
    import evaluation.run as run

    case = {"case_id": "C01", "unit_id": "U001", "held_out": True, "injection_present": False, "truth": {"root_cause": "bearing_wear", "sop": "SOP-M-101", "action": "x"}}
    saved = {"results": [{"system": "rules", "p50_latency_s": 0.0}], "per_case": {"C01": {"unit_id": "U001", "truth": case["truth"],
             "rules": {"root_cause": "bearing_wear", "sop": "SOP-M-101", "action": "schedule_bearing_replacement", "cost": 0.0, "steps": 1, "error": ""}}}}
    (tmp_path / "eval_results_mock.json").write_text(json.dumps(saved), encoding="utf-8")
    (tmp_path / "eval_cases.json").write_text(json.dumps([case | {"truth": case["truth"] | {"root_cause": "sensor_fault"}}]), encoding="utf-8")
    monkeypatch.setattr(run, "ARTIFACTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        run.rescore("mock")
    assert json.loads((tmp_path / "eval_results_mock.json").read_text(encoding="utf-8")) == saved  # not overwritten
    (tmp_path / "eval_cases.json").write_text(json.dumps([case]), encoding="utf-8")
    assert run.rescore("mock")[0]["root_cause_acc"] == 1.0


# ---------- F5: an investigation cannot dispatch a tool it was not offered ----------
def test_unoffered_write_tool_is_refused_without_creating_a_pending_order(tmp_path):
    gate = ApprovalGate()
    registry = build_default_registry(gate)
    ticket = {"unit_id": "U002", "root_cause": "coolant_leak", "recommended_action": "x", "evidence_summary": "x"}

    def script(step, prompt):
        if step == "write_report":
            return "submit_report", REPORT
        return ("finish", {}) if "tool=open_ticket" in prompt else ("open_ticket", ticket)

    res = Agent(registry, router=ModelRouter(cheap=ScriptedLLM(script), strong=ScriptedLLM(script)), trace_dir=tmp_path).investigate(EVENT)
    out = next(t.output for t in res.trace if t.kind == "tool" and t.name == "open_ticket")
    assert "not available" in out["error"] and gate.pending() == []


# ---------- F7: approval ids are unique and a decision is final ----------
def test_approval_ids_are_unique_under_threads_and_decisions_are_final():
    gate = ApprovalGate()
    ids: list[str] = []
    lock = threading.Lock()

    def worker():
        for _ in range(300):
            a = gate.request("U001", "open_ticket", {}).action_id
            with lock:
                ids.append(a)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ids) == len(set(ids)) == 1200 and len(gate.pending()) == 1200
    gate.decide(ids[0], True, "lead")
    with pytest.raises(AlreadyDecided):
        gate.decide(ids[0], False, "someone_else")
    assert gate.get(ids[0]).status == "approved" and gate.get(ids[0]).decided_by == "lead"


def test_api_returns_409_for_a_second_decision():
    from factory_copilot import api

    client = TestClient(api.app)
    aid = api.state().gate.request("U001", "open_ticket", {}).action_id
    assert client.post(f"/approvals/{aid}", json={"approve": True, "by": "a"}).status_code == 200
    assert client.post(f"/approvals/{aid}", json={"approve": False, "by": "b"}).status_code == 409
