import numpy as np

from factory_copilot.analytics import AnomalyDetector, DriftMonitor, RULModel, make_features, split_units
from factory_copilot.data.generate import generate_sensors


def _data():
    return generate_sensors(units=20, seed=11)


def test_split_is_by_unit_and_keeps_order():
    df = _data()
    train, test = split_units(df, 0.25, seed=1)
    assert not set(train["unit_id"]) & set(test["unit_id"])
    for _, g in test.groupby("unit_id"):
        assert (np.diff(g["cycle"].to_numpy()) == 1).all()


def test_features_are_within_unit():
    df = _data()
    f = make_features(df)
    assert f["spindle_vibration_mm_s__mean"].notna().all()
    assert len(f) == len(df)


def test_anomaly_detector_finds_degradation_with_lead_time():
    df = _data()
    train, test = split_units(df, 0.25, seed=2)
    det = AnomalyDetector().fit(train)
    events = det.detect(test)
    assert len(events) >= test["unit_id"].nunique() * 0.6
    for ev in events:
        g = test[test["unit_id"] == ev.unit_id]
        onset = g["onset_cycle"].iloc[0]
        assert ev.cycle >= onset * 0.5  # not firing in the clearly healthy phase
        assert ev.cycle < g["cycle"].max()  # detected before failure
        assert len(ev.top_sensors) == 3


def test_rul_model_beats_naive_baseline():
    df = _data()
    train, test = split_units(df, 0.25, seed=3)
    rul = RULModel().fit(train)
    f = make_features(test)
    pred = rul.predict(test)
    m = RULModel.metrics(f["rul"].to_numpy(), pred)
    naive = RULModel.metrics(f["rul"].to_numpy(), np.full(len(f), 60.0))
    assert m["rmse"] < naive["rmse"]


def test_drift_monitor_flags_degraded_window():
    df = _data()
    healthy = df[df["cycle"] < df["onset_cycle"] * 0.5]
    degraded = df[df["cycle"] > df["onset_cycle"]]
    psi = DriftMonitor(healthy).psi(degraded)
    assert max(psi.values()) > 0.25
    same = DriftMonitor(healthy).psi(healthy)
    assert max(same.values()) < 0.05
