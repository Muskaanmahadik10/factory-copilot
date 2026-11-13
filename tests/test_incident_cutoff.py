"""The agent must only see data that existed at the incident cycle (audit findings FC-01 to FC-04)."""

import numpy as np
import pandas as pd
import pytest

from factory_copilot.analytics import make_features
from factory_copilot.config import DATA_DIR
from factory_copilot.data.generate import generate_sensors
from factory_copilot.data.schemas import SENSOR_NAMES
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.tools import build_default_registry

UNIT, INCIDENT = "U001", 150  # U001's repair ticket is written at its final failure cycle, well after 150


@pytest.fixture(scope="module")
def registry():
    return build_default_registry(ApprovalGate())


def test_sensor_window_ends_at_the_incident(registry):
    with registry.as_of(INCIDENT):
        s = registry.call("query_sensors", {"unit_id": UNIT, "last_n_cycles": 30})
    assert s["cycles"] == [INCIDENT - 29, INCIDENT]
    assert registry.call("query_sensors", {"unit_id": UNIT})["cycles"][1] > INCIDENT  # no cutoff outside an investigation


def test_rul_uses_only_cycles_up_to_the_incident(registry):
    be_rows = pd.read_csv(DATA_DIR / "generated" / "sensors.csv")
    prefix = be_rows[(be_rows["unit_id"] == UNIT) & (be_rows["cycle"] <= INCIDENT)]
    with registry.as_of(INCIDENT):
        r = registry.call("estimate_rul", {"unit_id": UNIT})
    import pickle
    from factory_copilot.config import ARTIFACTS_DIR
    with open(ARTIFACTS_DIR / "models.pkl", "rb") as f:
        rul = pickle.load(f)["rul"]
    assert r["rul_cycles"] == round(float(rul.predict(prefix)[-1]), 1)


def test_history_never_contains_the_future_repair_ticket(registry):
    with registry.as_of(INCIDENT):
        h = registry.call("get_maintenance_history", {"unit_id": UNIT, "include_similar_units": False, "limit": 20})
    assert h["tickets"], "the planted pre-incident ticket should still be visible"
    assert all(t["cycle"] <= INCIDENT for t in h["tickets"])
    assert all(t["root_cause"] == "none" for t in h["tickets"])  # the failure ticket with the answer is not there


def test_fleet_history_comes_from_training_machines_only(registry):
    from factory_copilot.tools.registry import _Backend  # noqa: F401  (the registry's backend holds train_units)
    be = registry.get("get_maintenance_history").handler.__self__
    with registry.as_of(INCIDENT):
        h = registry.call("get_maintenance_history", {"unit_id": UNIT, "include_similar_units": True, "limit": 20})
    others = [t for t in h["tickets"] if t["relation"] == "other_unit_fleet_history"]
    assert others and all(t["unit_id"] in be.train_units for t in others)


def test_features_are_causal():
    """The feature at cycle t is identical whether or not later cycles exist."""
    df = generate_sensors(units=3, seed=5)
    full = make_features(df)
    for unit, cut in (("U001", 2), ("U002", 40), ("U003", 120)):
        prefix = make_features(df[(df["unit_id"] == unit) & (df["cycle"] <= cut)])
        row_full = full[(full["unit_id"] == unit) & (full["cycle"] == cut)]
        cols = [c for c in full.columns if "__" in c]
        np.testing.assert_allclose(prefix.iloc[-1][cols].to_numpy(float), row_full.iloc[0][cols].to_numpy(float))


def test_cutoff_does_not_leak_between_investigations(registry):
    with registry.as_of(100):
        inner = registry.call("query_sensors", {"unit_id": UNIT})["cycles"][1]
    outer = registry.call("query_sensors", {"unit_id": UNIT})["cycles"][1]
    assert inner == 100 and outer > 100


def test_baseline_needs_no_failure_labels(registry):
    """The tool works on data without failure_mode / rul / onset_cycle columns (a real machine has none)."""
    be = registry.get("query_sensors").handler.__self__
    saved = be.sensors
    try:
        be.sensors = saved[["unit_id", "cycle", *SENSOR_NAMES]]
        with registry.as_of(INCIDENT):
            s = registry.call("query_sensors", {"unit_id": UNIT})
        assert s["most_deviated"]
    finally:
        be.sensors = saved
