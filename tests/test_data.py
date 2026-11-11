import pandas as pd

from factory_copilot.data.generate import INJECTION_NOTES, generate_maintenance_log, generate_sensors, write_dataset
from factory_copilot.data.schemas import FAILURE_SIGNATURES, SENSOR_NAMES, FailureMode
from factory_copilot.data.sops import SOPS, write_sops


def test_sensor_generation_shape_and_labels():
    df = generate_sensors(units=10, seed=1)
    assert set(SENSOR_NAMES) <= set(df.columns)
    assert df["unit_id"].nunique() == 10
    assert set(df["failure_mode"].unique()) == {m.value for m in FailureMode}
    assert (df["rul"] >= 0).all()


def test_signature_moves_the_right_sensors():
    df = generate_sensors(units=10, seed=3)
    for mode, sig in FAILURE_SIGNATURES.items():
        g = df[df["failure_mode"] == mode.value]
        onset = g["onset_cycle"].iloc[0]
        early = g[g["cycle"] < onset * 0.8]
        late = g[g["cycle"] > g["cycle"].max() - 15]
        for sensor, (direction, _) in sig.items():
            delta = late[sensor].mean() - early[sensor].mean()
            assert delta * direction > 0, f"{mode} should move {sensor} in direction {direction}"


def test_maintenance_log_has_injection_cases():
    df = generate_sensors(units=5, seed=1)
    log = generate_maintenance_log(df, seed=1)
    assert len(log) == 5 + len(INJECTION_NOTES)
    assert set(INJECTION_NOTES) <= set(log["technician_note"])


def test_write_dataset_and_sops(tmp_path):
    meta = write_dataset(tmp_path / "gen", units=5, seed=2)
    assert (tmp_path / "gen" / "sensors.csv").exists()
    assert meta["units"] == 5
    n = write_sops(tmp_path / "sops")
    assert n == len(SOPS) == 12
    ids = {s["id"] for s in SOPS}
    assert len(ids) == 12
