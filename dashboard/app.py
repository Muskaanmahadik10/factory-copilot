"""Streamlit dashboard: pick a unit, see the sensors and the anomaly flag, run the copilot, read the report and trace,
approve or reject the work order.

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere: `streamlit run dashboard/app.py`

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factory_copilot.agent import Agent
from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR, settings
from factory_copilot.data.schemas import SENSOR_NAMES, AnomalyEvent
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.tools import build_default_registry

st.set_page_config(page_title="Factory Copilot", layout="wide")


@st.cache_resource
def load():
    gate = ApprovalGate()
    registry = build_default_registry(gate)
    agent = Agent(registry)
    with open(ARTIFACTS_DIR / "models.pkl", "rb") as f:
        det = pickle.load(f)["anomaly"]
    sensors = pd.read_csv(DATA_DIR / "generated" / "sensors.csv")
    return gate, registry, agent, det, sensors


gate, registry, agent, det, sensors = load()
if "results" not in st.session_state:
    st.session_state.results = {}

st.title("Factory Copilot")
st.caption(f"Evidence-backed anomaly investigation · models: **{agent.router.cheap.name} / {agent.router.strong.name}** · synthetic data · "
           "the copilot sees only data up to the anomaly cycle · work orders are local approval records")

left, right = st.columns([1, 2])
with left:
    events = det.detect(sensors)
    ev_by_unit = {e.unit_id: e for e in events}
    unit = st.selectbox("Unit", sorted(sensors["unit_id"].unique()), format_func=lambda u: f"{u}  {'⚠ anomaly' if u in ev_by_unit else ''}")
    g = sensors[sensors["unit_id"] == unit].sort_values("cycle")
    ev = ev_by_unit.get(unit)
    if ev:
        st.warning(f"Anomaly at cycle {ev.cycle} (score {ev.score:.2f}). Top sensors: {', '.join(ev.top_sensors)}")
    else:
        st.success("No anomaly detected on this unit.")
    chosen = st.multiselect("Sensors to plot", SENSOR_NAMES, default=(ev.top_sensors if ev else SENSOR_NAMES[:3]))
    run = st.button("Investigate with the copilot", type="primary", disabled=ev is None)

with right:
    fig = go.Figure()
    for s in chosen:
        fig.add_trace(go.Scatter(x=g["cycle"], y=g[s], name=s, mode="lines"))
    if ev:
        fig.add_vline(x=ev.cycle, line_dash="dash", annotation_text="anomaly detected · copilot's data cutoff")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

if run and ev:
    with st.spinner("Investigating..."):
        res = agent.investigate(ev)
        pa = gate.request(unit, "open_ticket", {"root_cause": res.report.root_cause, "action": res.report.recommended_action, "sop": res.report.sop_reference})
        st.session_state.results[unit] = (res, pa.action_id)

if unit in st.session_state.results:
    res, action_id = st.session_state.results[unit]
    rep = res.report
    st.subheader(f"Incident report · {res.incident_id}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Root cause", rep.root_cause.replace("_", " "))
    c2.metric("Model self-assessment", f"{rep.confidence:.0%}", help="The model's own confidence. Not calibrated: it is not the probability that the diagnosis is right.")
    c3.metric("Cost", f"${res.cost_usd:.4f}")
    c4.metric("Steps / latency", f"{res.steps} / {res.latency_s:.1f}s")
    st.markdown(f"**Recommended action:** `{rep.recommended_action}` · **SOP:** {rep.sop_reference}")
    st.write(rep.summary)
    st.markdown("**Evidence**")
    st.table(pd.DataFrame([e.model_dump() for e in rep.evidence]))
    unretrieved = res.citation_check.get("unretrieved_citations", [])
    if unretrieved:
        st.warning(f"Citation check: {', '.join(unretrieved)} cited but never retrieved in this investigation. Treat as unsupported.")
    elif res.citation_check.get("ok"):
        st.caption("Citation check: every cited SOP/ticket id was retrieved in this investigation (this does not check that the passage supports the conclusion).")
    if res.injection_findings:
        st.error(f"Guardrail: {len(res.injection_findings)} instruction-like passage(s) found in tool data and redacted. See trace.")
    if res.error:
        st.info(f"Note: {res.error}")

    pa = gate.get(action_id)
    st.markdown(f"**Work order {action_id}:** status `{pa.status}`")
    a, b = st.columns(2)
    if pa.status == "pending":
        if a.button("Approve work order", key=f"ok{action_id}"):
            gate.decide(action_id, True, "dashboard_user"); st.rerun()
        if b.button("Reject", key=f"no{action_id}"):
            gate.decide(action_id, False, "dashboard_user"); st.rerun()

    with st.expander("Agent trace (every LLM call and tool call, with tokens and cost)"):
        for i, t in enumerate(res.trace, 1):
            st.markdown(f"**{i}. {t.kind} · {t.name}** {('· ' + t.tier) if t.tier else ''} · {t.latency_s:.2f}s · ${t.cost_usd:.5f}")
            st.code(json.dumps(t.output, indent=1, ensure_ascii=False)[:1500] if isinstance(t.output, dict) else str(t.output)[:1500])

with st.sidebar:
    st.header("Model metrics")
    p = ARTIFACTS_DIR / "analytics_metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        st.json({"anomaly": m["anomaly"], "rul": m["rul"]})
    for name in ("eval_results_env.json", "eval_results_mock.json"):
        q = ARTIFACTS_DIR / name
        if q.exists():
            e = json.loads(q.read_text(encoding="utf-8"))
            st.subheader(f"Evaluation ({e['provider']})")
            cols = [c for c in ("system", "cases", "root_cause_acc", "sop_id_match", "action_keyword_match", "clean_runs", "mean_cost_usd") if c in e["results"][0]]
            st.dataframe(pd.DataFrame(e["results"])[cols], hide_index=True)
    st.subheader("Pending approvals")
    st.write(len(gate.pending()))
