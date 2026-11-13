from fastapi.testclient import TestClient

from factory_copilot import api


def test_api_investigate_and_approve_flow(monkeypatch):
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    inc = client.get("/incidents").json()
    assert inc and {"unit_id", "cycle", "top_sensors"} <= set(inc[0])
    r = client.post("/investigate", json={"unit_id": inc[0]["unit_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["requires_approval"] is True
    aid = body["pending_action_id"]
    assert any(p["action_id"] == aid for p in client.get("/approvals").json())
    d = client.post(f"/approvals/{aid}", json={"approve": True, "by": "tester"}).json()
    assert d["status"] == "approved" and d["decided_by"] == "tester"
    assert client.get(f"/incidents/{body['incident_id']}").status_code == 200
    assert client.post("/investigate", json={"unit_id": "U999"}).status_code == 404


def test_manual_investigation_of_a_unit_without_detected_anomaly(monkeypatch):
    """The no-event branch builds a valid event from engineered features (it used to pass a raw CSV row)."""
    client = TestClient(api.app)
    st = api.state()
    monkeypatch.setattr(st.detector, "detect", lambda df, **k: [])
    r = client.post("/investigate", json={"unit_id": "U002", "cycle": 120})
    assert r.status_code == 200
    assert r.json()["report"]["requires_approval"] is True
    assert client.get("/health").json()["models"]["cheap"].startswith("mock")
