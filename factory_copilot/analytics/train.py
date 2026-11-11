"""Train analytics models and write metrics: python -m factory_copilot.analytics.train"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from factory_copilot.analytics.features import make_features, split_units
from factory_copilot.analytics.models import AnomalyDetector, DriftMonitor, RULModel, save_json
from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR


def anomaly_metrics(det: AnomalyDetector, test: pd.DataFrame) -> dict:
    """Precision/recall on cycles: positive = degraded (cycle >= onset). Plus detection lead time per unit."""
    feats = make_features(test)
    scores = det.score(test).to_numpy()
    y = (feats["cycle"] >= feats["onset_cycle"]).to_numpy()
    pred = scores > det.threshold
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    events = det.detect(test)
    leads = []
    for ev in events:
        g = test[test["unit_id"] == ev.unit_id]
        leads.append(int(g["cycle"].max() - ev.cycle))
    return {
        "cycle_precision": precision,
        "cycle_recall": recall,
        "units_detected": len(events),
        "units_total": int(test["unit_id"].nunique()),
        "median_cycles_before_failure_at_detection": float(np.median(leads)) if leads else None,
        "threshold": det.threshold,
    }


def main() -> dict:
    df = pd.read_csv(DATA_DIR / "generated" / "sensors.csv")
    train, test = split_units(df, test_frac=0.25, seed=0)
    det = AnomalyDetector().fit(train)
    rul = RULModel().fit(train)
    pred = rul.predict(test)
    feats_test = make_features(test)
    metrics = {
        "split": {"train_units": int(train["unit_id"].nunique()), "test_units": int(test["unit_id"].nunique()), "method": "held-out units, time order preserved"},
        "anomaly": anomaly_metrics(det, test),
        "rul": RULModel.metrics(feats_test["rul"].to_numpy(), pred),
    }
    # Drift: reference = first 30% of train cycles per unit vs last 30% of test cycles (degraded) -> expect drift
    ref = train[train["cycle"] < train["onset_cycle"] * 0.5]
    cur = test[test["cycle"] > test["onset_cycle"]]
    psi = DriftMonitor(ref).psi(cur)
    metrics["drift_psi_degraded_vs_healthy"] = {k: round(v, 3) for k, v in sorted(psi.items(), key=lambda kv: -kv[1])}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "models.pkl", "wb") as f:
        pickle.dump({"anomaly": det, "rul": rul}, f)
    save_json(metrics, ARTIFACTS_DIR / "analytics_metrics.json")
    return metrics


if __name__ == "__main__":
    m = main()
    print({"anomaly": m["anomaly"], "rul": m["rul"]})
