"""Feature engineering and leakage-safe splitting.

Features are per-unit rolling statistics so a model sees trends, not just levels. The split is by unit and
by time: all cycles of a held-out unit are unseen, which mirrors deployment (a new machine, future cycles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factory_copilot.data.schemas import SENSOR_NAMES

WINDOW = 10


def make_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add rolling mean, rolling std and rolling slope per sensor, computed within each unit (no cross-unit leakage)."""
    out = df.sort_values(["unit_id", "cycle"]).copy()
    g = out.groupby("unit_id", sort=False)
    for s in SENSOR_NAMES:
        r = g[s].rolling(window, min_periods=3)
        out[f"{s}__mean"] = r.mean().reset_index(level=0, drop=True)
        out[f"{s}__std"] = r.std().reset_index(level=0, drop=True)
        out[f"{s}__slope"] = (out[s] - g[s].shift(window)).fillna(0.0) / window
    return out.bfill().ffill()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in SENSOR_NAMES or "__" in c]


def split_units(df: pd.DataFrame, test_frac: float = 0.25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole units (machines). Within a unit, order is kept, so test cycles are never interleaved with train."""
    units = np.array(sorted(df["unit_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = max(1, int(len(units) * test_frac))
    test_units = set(units[:n_test])
    test = df[df["unit_id"].isin(test_units)]
    train = df[~df["unit_id"].isin(test_units)]
    return train, test
