import pytest

from factory_copilot.data.generate import INJECTION_NOTES
from factory_copilot.guardrails import ApprovalGate, Budget, BudgetExceeded, InjectionDetector, wrap_untrusted


@pytest.mark.parametrize("payload", INJECTION_NOTES)
def test_detector_redacts_planted_payloads(payload):
    det = InjectionDetector()
    out = det.scan(payload, source="ticket T9999")
    assert "[REDACTED" in out
    assert det.findings and det.findings[0].source == "ticket T9999"


def test_detector_leaves_normal_notes_alone():
    det = InjectionDetector()
    note = "Replaced front spindle bearing set. Vibration back to 2.1 mm/s."
    assert det.scan(note, "ticket") == note and not det.findings


def test_wrap_marks_data():
    w = wrap_untrusted("hello", "sop")
    assert w.startswith('<untrusted_data source="sop">') and "must not be followed" in w


def test_budget_limits():
    b = Budget(max_steps=2, max_cost_usd=0.01)
    b.step(); b.step()
    with pytest.raises(BudgetExceeded):
        b.step()
    b2 = Budget(max_cost_usd=0.001)
    with pytest.raises(BudgetExceeded):
        b2.charge(0.002)


def test_approval_gate_parks_actions():
    g = ApprovalGate()
    pa = g.request("U001", "open_ticket", {"priority": "P2"})
    assert pa.status == "pending" and g.pending() == [pa]
    g.decide(pa.action_id, True, by="shift_lead")
    assert g.get(pa.action_id).status == "approved" and not g.pending()
