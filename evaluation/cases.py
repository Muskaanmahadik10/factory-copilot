"""Build the evaluation set from the generated data: one case per held-out unit at its detection cycle,
plus injection cases where the unit's history contains a planted attack.

Ground truth per case: failure mode, SOP id, action (all from the generator), and whether an injection is present.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

from factory_copilot.analytics.features import split_units
from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR
from factory_copilot.data.generate import INJECTION_NOTES
from factory_copilot.data.schemas import ACTION_FOR_MODE, SOP_FOR_MODE, FailureMode


def build_cases(n_per_mode: int = 6) -> list[dict]:
    df = pd.read_csv(DATA_DIR / "generated" / "sensors.csv")
    log = pd.read_csv(DATA_DIR / "generated" / "maintenance_log.csv")
    with open(ARTIFACTS_DIR / "models.pkl", "rb") as f:
        det = pickle.load(f)["anomaly"]
    _, test = split_units(df, 0.25, seed=0)
    # Use test units first (truly unseen by the analytics models), then fill from train units to reach n_per_mode.
    ordered = pd.concat([test, df[~df["unit_id"].isin(test["unit_id"])]])
    injected_units = set(log[log["technician_note"].isin(INJECTION_NOTES)]["unit_id"])
    cases, counts = [], {m.value: 0 for m in FailureMode}
    for unit_id, g in ordered.groupby("unit_id", sort=False):
        mode = g["failure_mode"].iloc[0]
        if counts[mode] >= n_per_mode:
            continue
        events = det.detect(g)
        if not events:
            continue
        ev = events[0]
        counts[mode] += 1
        cases.append({
            "case_id": f"C{len(cases) + 1:02d}", "unit_id": unit_id, "held_out": unit_id in set(test["unit_id"]),
            "event": ev.model_dump(),
            "truth": {"root_cause": mode, "sop": SOP_FOR_MODE[FailureMode(mode)], "action": ACTION_FOR_MODE[FailureMode(mode)]},
            "injection_present": unit_id in injected_units,
        })
    return cases


if __name__ == "__main__":
    cases = build_cases()
    out = ARTIFACTS_DIR / "eval_cases.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(len(cases), "cases;", sum(c["injection_present"] for c in cases), "with injection;", sum(c["held_out"] for c in cases), "on held-out units")
