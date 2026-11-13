"""FastAPI service: detect anomalies, investigate one, approve/reject the resulting work order, read metrics.

Run: uvicorn factory_copilot.api:app --reload
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from factory_copilot import __version__
from factory_copilot.agent import Agent
from factory_copilot.analytics.features import make_features
from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR, settings
from factory_copilot.data.schemas import AnomalyEvent
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.tools import build_default_registry

app = FastAPI(title="Factory Copilot", version=__version__)


class _State:
    def __init__(self):
        self.gate = ApprovalGate()
        self.registry = build_default_registry(self.gate)
        self.agent = Agent(self.registry)
        self.results: dict[str, dict] = {}
        with open(ARTIFACTS_DIR / "models.pkl", "rb") as f:
            self.detector = pickle.load(f)["anomaly"]
        self.sensors = pd.read_csv(DATA_DIR / "generated" / "sensors.csv")


@lru_cache(maxsize=1)
def state() -> _State:
    return _State()


class InvestigateIn(BaseModel):
    unit_id: str
    cycle: int | None = None


class DecisionIn(BaseModel):
    approve: bool
    by: str = "shift_lead"


@app.get("/health")
def health():
    st = state()
    # The *effective* models, so a run that fell back to the mock is visible as such.
    return {"status": "ok", "version": __version__, "llm_provider": settings.llm_provider, "models": {"cheap": st.agent.router.cheap.name, "strong": st.agent.router.strong.name}}


@app.get("/incidents")
def incidents():
    """Anomaly events across all units (what the dashboard lists)."""
    st = state()
    return [e.model_dump() for e in st.detector.detect(st.sensors)]


@app.post("/investigate")
def investigate(body: InvestigateIn):
    st = state()
    g = st.sensors[st.sensors["unit_id"] == body.unit_id]
    if g.empty:
        raise HTTPException(404, f"unknown unit {body.unit_id}")
    events = st.detector.detect(g)
    if events:
        ev = events[0]
    else:
        # No detected anomaly: investigate manually at the requested cycle (default: the latest one). Contributions need
        # the engineered rolling features, not a raw CSV row.
        feats = make_features(g)
        if body.cycle:
            feats = feats[feats["cycle"] <= body.cycle]
        if feats.empty:
            raise HTTPException(400, f"no data for {body.unit_id} at or before cycle {body.cycle}")
        last = feats.iloc[-1]
        ev = AnomalyEvent(unit_id=body.unit_id, cycle=int(last["cycle"]), score=0.0, top_sensors=[s for s, _ in st.detector.contributions(last)[:3]], message="manual investigation (no anomaly detected)")
    if body.cycle:
        ev.cycle = body.cycle
    res = st.agent.investigate(ev)
    # The agent only recommends; the API turns the recommendation into a pending work order for a human.
    rep = res.report
    pa = st.gate.request(rep.unit_id, "open_ticket", {"root_cause": rep.root_cause, "recommended_action": rep.recommended_action, "sop_reference": rep.sop_reference, "priority": "P2", "evidence_summary": rep.summary})
    out = res.to_dict() | {"pending_action_id": pa.action_id}
    st.results[res.incident_id] = out
    return out


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    st = state()
    if incident_id not in st.results:
        raise HTTPException(404, "unknown incident")
    return st.results[incident_id]


@app.get("/approvals")
def approvals():
    return [p.__dict__ for p in state().gate.pending()]


@app.post("/approvals/{action_id}")
def decide(action_id: str, body: DecisionIn):
    st = state()
    if not st.gate.get(action_id):
        raise HTTPException(404, "unknown action")
    pa = st.gate.decide(action_id, body.approve, body.by)
    return pa.__dict__


@app.get("/metrics")
def metrics():
    out = {}
    for name in ("analytics_metrics.json", "eval_results_mock.json", "eval_results_env.json"):
        p = ARTIFACTS_DIR / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out
