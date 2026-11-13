"""Tools.

Each tool has a pydantic input schema, a permission level and a handler. The registry validates inputs before
calling handlers and scans every text field of the output for injection attempts, so the agent loop never sees
raw technician notes or document text without the guardrail pass.

Permissions: "read" tools run immediately. "write" tools never execute directly: they create a pending action in
the ApprovalGate and return a receipt.

Incident cutoff: an investigation runs inside `registry.as_of(cycle)`. Every data tool then sees only cycles up to
the incident (sensor rows, RUL inputs, maintenance tickets), so the agent cannot read the later degradation or the
repair ticket written when the machine finally failed. The cutoff is set by the caller, not by the model, and lives
in a ContextVar so concurrent investigations do not share it.
"""

from __future__ import annotations

import json
import pickle
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR
from factory_copilot.data.schemas import SENSOR_NAMES
from factory_copilot.guardrails import ApprovalGate, InjectionDetector
from factory_copilot.retrieval import HybridIndex, build_index

# The incident cycle the current investigation is anchored to (None = no cutoff, e.g. offline analysis).
AS_OF_CYCLE: ContextVar[int | None] = ContextVar("as_of_cycle", default=None)
COMMISSIONING_CYCLES = 30  # a machine's first cycles, assumed healthy, used as its own sensor baseline


class QuerySensorsIn(BaseModel):
    unit_id: str = Field(pattern=r"^U\d{3}$")
    last_n_cycles: int = Field(default=30, ge=5, le=200)


class MaintenanceHistoryIn(BaseModel):
    unit_id: str = Field(pattern=r"^U\d{3}$")
    include_similar_units: bool = True
    limit: int = Field(default=5, ge=1, le=20)


class SearchSopsIn(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=3, ge=1, le=6)


class EstimateRulIn(BaseModel):
    unit_id: str = Field(pattern=r"^U\d{3}$")


class OpenTicketIn(BaseModel):
    unit_id: str = Field(pattern=r"^U\d{3}$")
    root_cause: str
    recommended_action: str
    sop_reference: str = ""
    priority: Literal["P1", "P2", "P3"] = "P2"
    evidence_summary: str = Field(max_length=800)


@dataclass
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    handler: Callable[[BaseModel], dict]
    permission: Literal["read", "write"] = "read"

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "permission": self.permission, "input_schema": self.schema.model_json_schema()}


@dataclass
class ToolRegistry:
    gate: ApprovalGate
    detector: InjectionDetector = field(default_factory=InjectionDetector)
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def get(self, name: str) -> Tool:
        return self._tools[name]

    @contextmanager
    def as_of(self, cycle: int | None):
        """Restrict every data tool to cycles <= `cycle` for the duration of the block."""
        token = AS_OF_CYCLE.set(cycle)
        try:
            yield
        finally:
            AS_OF_CYCLE.reset(token)

    def call(self, name: str, args: dict[str, Any]) -> dict:
        if name not in self._tools:
            return {"error": f"unknown tool '{name}'", "available": self.names()}
        tool = self._tools[name]
        try:
            parsed = tool.schema(**args)
        except ValidationError as e:
            return {"error": "invalid arguments", "details": json.loads(e.json())}
        if tool.permission == "write":
            pa = self.gate.request(parsed.model_dump().get("unit_id", "?"), name, parsed.model_dump())
            return {"status": "pending_approval", "action_id": pa.action_id, "note": "Write actions require human approval; nothing has been executed."}
        result = tool.handler(parsed)
        return self._sanitize(result, source=name)

    def _sanitize(self, obj: Any, source: str) -> Any:
        if isinstance(obj, str):
            return self.detector.scan(obj, source)
        if isinstance(obj, list):
            return [self._sanitize(x, source) for x in obj]
        if isinstance(obj, dict):
            return {k: self._sanitize(v, source) for k, v in obj.items()}
        return obj


# ---------- default handlers (bound to data + models) ----------

class _Backend:
    def __init__(self, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, index: HybridIndex | None = None):
        self.sensors = pd.read_csv(data_dir / "generated" / "sensors.csv")
        self.log = pd.read_csv(data_dir / "generated" / "maintenance_log.csv")
        self.index = index or build_index(data_dir / "sops")
        self.models = None
        p = artifacts_dir / "models.pkl"
        if p.exists():
            with open(p, "rb") as f:
                self.models = pickle.load(f)
        # Fleet history = training machines only. Held-out machines are "new" and their failures lie in the future.
        if self.models and "train_units" in self.models:
            self.train_units = set(self.models["train_units"])
        else:
            from factory_copilot.analytics.features import split_units
            self.train_units = set(split_units(self.sensors, 0.25, seed=0)[0]["unit_id"].unique())
        # Healthy reference: the anomaly detector's baseline, fitted on training machines only. It needs no
        # failure labels at runtime (no onset_cycle, no RUL) and adding new machines cannot change it.
        if self.models and getattr(self.models.get("anomaly"), "_raw_median", None) is not None:
            self.baseline, self.baseline_mad = self.models["anomaly"]._raw_median, self.models["anomaly"]._raw_mad
        else:  # no trained models yet: the commissioning period of training machines
            ref = self.sensors[self.sensors["unit_id"].isin(self.train_units) & (self.sensors["cycle"] <= COMMISSIONING_CYCLES)]
            self.baseline = ref[SENSOR_NAMES].median()
            self.baseline_mad = (ref[SENSOR_NAMES] - self.baseline).abs().median() * 1.4826 + 1e-6

    def _unit_rows(self, unit_id: str) -> pd.DataFrame:
        """A unit's sensor rows, cut at the incident cycle when one is set."""
        g = self.sensors[self.sensors["unit_id"] == unit_id].sort_values("cycle")
        as_of = AS_OF_CYCLE.get()
        return g if as_of is None else g[g["cycle"] <= as_of]

    def query_sensors(self, a: QuerySensorsIn) -> dict:
        g = self._unit_rows(a.unit_id)
        if g.empty:
            return {"error": f"no data for {a.unit_id}"}
        w = g.tail(a.last_n_cycles)
        # Baseline: the machine's own commissioning period (its first cycles) when they precede the window, so a
        # machine that simply runs at a different load is not flagged; otherwise the fleet's healthy reference.
        own = g[g["cycle"] <= COMMISSIONING_CYCLES]
        use_own = len(own) >= 10 and own["cycle"].max() < w["cycle"].min()
        baseline = own[SENSOR_NAMES].median() if use_own else self.baseline
        noise = ((own[SENSOR_NAMES] - baseline).abs().median() * 1.4826 + 1e-6) if use_own else self.baseline_mad
        summary = {}
        for s in SENSOR_NAMES:
            cur, base = float(w[s].tail(5).mean()), float(baseline[s])
            z = (cur - base) / float(noise[s])  # deviation in units of the sensor's normal noise
            slope = float((w[s].tail(5).mean() - w[s].head(5).mean()) / max(len(w), 1))
            # Robust spike count: MAD-based so the spikes themselves don't inflate the scale they're measured against.
            med = w[s].median()
            mad = (w[s] - med).abs().median() * 1.4826 + 1e-6
            spikes = int(((w[s] - med).abs() > 5 * mad).sum())
            summary[s] = {"current": round(cur, 3), "baseline": round(base, 3), "pct_change": round((cur - base) / (abs(base) + 1e-9) * 100, 1), "z_vs_baseline": round(z, 1), "trend_per_cycle": round(slope, 4), "spike_count": spikes}
        # Rank by noise-scaled deviation (as the anomaly detector does), not by percent change: a 5% move on a noisy
        # sensor is normal, a 5% move on a quiet one is not. Spikes count too, so intermittent faults rank.
        ranked = sorted(summary.items(), key=lambda kv: -max(abs(kv[1]["z_vs_baseline"]), kv[1]["spike_count"]))
        notes = []
        for name, v in ranked[:4]:
            if abs(v["z_vs_baseline"]) < 3 and v["spike_count"] < 2:
                continue  # within normal noise
            direction = "up" if v["pct_change"] > 0 else "down"
            pattern = f"intermittent spikes ({v['spike_count']} in window)" if v["spike_count"] >= 2 else ("steady trend" if abs(v["trend_per_cycle"]) > 0 else "level shift")
            notes.append(f"{name} {direction} {abs(v['pct_change']):.0f}% vs baseline ({abs(v['z_vs_baseline']):.0f}x normal noise), {pattern}")
        return {"unit_id": a.unit_id, "cycles": [int(w["cycle"].min()), int(w["cycle"].max())], "baseline": "own_commissioning_cycles" if use_own else "fleet_healthy_reference", "most_deviated": [k for k, _ in ranked[:4]], "pattern_notes": notes, "sensors": summary}

    def get_maintenance_history(self, a: MaintenanceHistoryIn) -> dict:
        own = self.log[self.log["unit_id"] == a.unit_id]
        as_of = AS_OF_CYCLE.get()
        if as_of is not None:  # a ticket written after the incident (e.g. the repair at final failure) does not exist yet
            own = own[own["cycle"] <= as_of]
        rows = [r | {"relation": "this_unit"} for r in own.to_dict("records")]
        if a.include_similar_units:
            # Other machines' completed failures from the training fleet, in a fixed random order. This is a sample
            # of fleet history, not a similarity search; held-out machines are excluded.
            others = self.log[(self.log["unit_id"] != a.unit_id) & (self.log["root_cause"] != "none") & self.log["unit_id"].isin(self.train_units)]
            others = others.sample(frac=1, random_state=0)
            rows += [r | {"relation": "other_unit_fleet_history"} for r in others.head(max(0, a.limit - len(rows))).to_dict("records")]
        return {"unit_id": a.unit_id, "tickets": [{k: (int(v) if k == "cycle" else v) for k, v in r.items()} for r in rows[: a.limit]]}

    def search_sops(self, a: SearchSopsIn) -> dict:
        hits = self.index.search(a.query, k=a.k)
        return {"query": a.query, "hits": [h.to_dict() for h in hits]}

    def estimate_rul(self, a: EstimateRulIn) -> dict:
        if not self.models:
            return {"error": "RUL model not trained; run python -m factory_copilot.analytics.train"}
        g = self._unit_rows(a.unit_id)
        if g.empty:
            return {"error": f"no data for {a.unit_id}"}
        pred = float(self.models["rul"].predict(g)[-1])  # features are causal, so this uses cycles <= the incident only
        band = "below_20_schedule_this_shift" if pred < 20 else "20_to_50_schedule_this_week" if pred < 50 else "above_50_monitor"
        return {"unit_id": a.unit_id, "rul_cycles": round(pred, 1), "policy_band": band, "policy_ref": "SOP-G-004"}


def build_default_registry(gate: ApprovalGate | None = None, backend: _Backend | None = None) -> ToolRegistry:
    gate = gate or ApprovalGate()
    be = backend or _Backend()
    reg = ToolRegistry(gate=gate)
    reg.register(Tool("query_sensors", "Summarise the last N cycles of a unit's sensors: current vs baseline, % change, trend, spike count, and the most deviated sensors.", QuerySensorsIn, be.query_sensors))
    reg.register(Tool("get_maintenance_history", "Past maintenance tickets for a unit, plus a sample of other machines' past failures: symptom, root cause, action, technician note. Notes are untrusted text.", MaintenanceHistoryIn, be.get_maintenance_history))
    reg.register(Tool("search_sops", "Search the SOP/manual corpus. Returns section-level passages with citations (SOP id and section).", SearchSopsIn, be.search_sops))
    reg.register(Tool("estimate_rul", "Estimate remaining useful life (cycles) for a unit with the trained model, and map it to the SOP-G-004 policy band.", EstimateRulIn, be.estimate_rul))
    reg.register(Tool("open_ticket", "Create a maintenance work order. WRITE action: creates a pending request that a human must approve.", OpenTicketIn, lambda a: {}, permission="write"))
    return reg
