"""FastAPI service: detect anomalies, investigate one, approve/reject the resulting work order, read metrics.

Run: uvicorn factory_copilot.api:app --reload
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from factory_copilot import __version__
from factory_copilot.agent import Agent
from factory_copilot.analytics.features import make_features
from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR, settings
from factory_copilot.data.schemas import AnomalyEvent
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.guardrails.core import AlreadyDecided
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
    cycle: int | None = Field(default=None, description="Investigate as of this cycle, using only data up to it. Omit to use the unit's first detected anomaly (or its latest cycle if there is none).")


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


def _event_at_last_cycle(detector, g: pd.DataFrame, message: str) -> AnomalyEvent:
    """An event describing the last cycle of `g`, computed from `g` only (features are causal, so later rows of the
    unit cannot influence it). Contributions need the engineered rolling features, not a raw CSV row."""
    last = make_features(g).iloc[-1]
    score = float(detector.score(g).iloc[-1])
    return AnomalyEvent(unit_id=str(last["unit_id"]), cycle=int(last["cycle"]), score=score, top_sensors=[s for s, _ in detector.contributions(last)[:3]], message=message)


@app.post("/investigate")
def investigate(body: InvestigateIn):
    st = state()
    g = st.sensors[st.sensors["unit_id"] == body.unit_id].sort_values("cycle")
    if g.empty:
        raise HTTPException(404, f"unknown unit {body.unit_id}")
    if body.cycle is not None:  # `is not None`: cycle 0 is a value to validate, not "omitted"
        lo, hi = int(g["cycle"].min()), int(g["cycle"].max())
        if not lo <= body.cycle <= hi:
            raise HTTPException(400, f"cycle {body.cycle} outside {body.unit_id}'s history ({lo}-{hi})")
        # Historical investigation: cut the history at the requested cycle BEFORE detection, so the event's score,
        # sensor ranking and message come from data that existed then, never from a later alert.
        g = g[g["cycle"] <= body.cycle]
        alerts = st.detector.detect(g)
        seen = f"first alert up to then at cycle {alerts[0].cycle} (score {alerts[0].score:.3f})" if alerts else "no alert detected up to then"
        ev = _event_at_last_cycle(st.detector, g, f"investigation requested at cycle {body.cycle}; {seen}")
    else:
        events = st.detector.detect(g)
        ev = events[0] if events else _event_at_last_cycle(st.detector, g, "manual investigation at the latest cycle (no anomaly detected)")
    res = st.agent.investigate(ev)
    # The agent only recommends; the API turns the recommendation into a pending work order for a human.
    rep = res.report
    pa = st.gate.request(rep.unit_id, "open_ticket", {"root_cause": rep.root_cause, "recommended_action": rep.recommended_action, "sop_reference": rep.sop_reference, "priority": "P2", "evidence_summary": rep.summary})
    out = res.to_dict() | {"event": ev.model_dump(), "pending_action_id": pa.action_id}
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
    try:
        pa = st.gate.decide(action_id, body.approve, body.by)
    except AlreadyDecided as e:
        raise HTTPException(409, str(e)) from e
    return pa.__dict__


@app.get("/metrics")
def metrics():
    out = {}
    for name in ("analytics_metrics.json", "eval_results_mock.json", "eval_results_env.json"):
        p = ARTIFACTS_DIR / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out
