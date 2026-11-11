"""Anomaly detector, RUL model and drift monitor.

AnomalyDetector: IsolationForest trained on *healthy* cycles only (cycles before the degradation onset in training
units), so anything that looks unlike healthy operation scores high. Sensor contributions are robust z-scores
against the healthy baseline, which gives the agent a ranked list of suspicious sensors.

RULModel: gradient boosting on the rolling features, trained with RUL clipped at 125 cycles (standard C-MAPSS practice:
early-life RUL is not predictable and clipping stops the model wasting capacity on it).

DriftMonitor: Population Stability Index per feature between a reference window and a current window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

from factory_copilot.analytics.features import feature_columns, make_features
from factory_copilot.data.schemas import SENSOR_NAMES, AnomalyEvent

RUL_CLIP = 125
WARMUP = 12  # cycles; rolling window is 10


@dataclass
class AnomalyDetector:
    contamination: float = 0.02
    threshold: float = field(default=0.0, init=False)
    _model: IsolationForest | None = field(default=None, init=False, repr=False)
    _median: pd.Series | None = field(default=None, init=False, repr=False)
    _mad: pd.Series | None = field(default=None, init=False, repr=False)
    _cols: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, train: pd.DataFrame) -> "AnomalyDetector":
        feats = make_features(train)
        feats = feats[feats["cycle"] > WARMUP]  # rolling features are unstable in the first cycles of a unit
        healthy = feats[feats["cycle"] < feats["onset_cycle"] * 0.9] if "onset_cycle" in feats else feats
        self._cols = feature_columns(feats)
        self._model = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=0)
        self._model.fit(healthy[self._cols].to_numpy())
        # Robust baseline on *rolling means* (trend), so a slow drift registers even when single readings look normal.
        self._mean_cols = [f"{s}__mean" for s in SENSOR_NAMES]
        self._median = healthy[self._mean_cols].median()
        self._mad = (healthy[self._mean_cols] - self._median).abs().median() * 1.4826 + 1e-6
        self._raw_median = healthy[SENSOR_NAMES].median()
        self._raw_mad = (healthy[SENSOR_NAMES] - self._raw_median).abs().median() * 1.4826 + 1e-6
        healthy_scores = self._score_feats(healthy)
        # Threshold: 99.5th percentile of healthy scores -> ~0.5% false alarms per cycle on healthy data
        self.threshold = float(np.percentile(healthy_scores, 99.5))
        return self

    def _score_feats(self, feats: pd.DataFrame) -> np.ndarray:
        """Ensemble: max robust z-score of rolling means (drift) + IsolationForest (multivariate oddness)."""
        z = ((feats[self._mean_cols] - self._median) / self._mad).abs().max(axis=1).to_numpy()
        iso = -self._model.decision_function(feats[self._cols].to_numpy())
        return z / 6.0 + iso

    def score(self, df: pd.DataFrame) -> pd.Series:
        feats = make_features(df)
        return pd.Series(self._score_feats(feats), index=feats.index)

    def contributions(self, row: pd.Series) -> list[tuple[str, float]]:
        """Per-sensor robust z of the rolling mean, ranked. Spiky faults also get credit via raw-value z."""
        z_trend = ((row[self._mean_cols] - self._median) / self._mad).abs()
        z_trend.index = SENSOR_NAMES
        z_raw = ((row[SENSOR_NAMES] - self._raw_median) / self._raw_mad).abs()
        z = pd.concat([z_trend, z_raw], axis=1).max(axis=1).sort_values(ascending=False)
        return [(str(k), float(v)) for k, v in z.items()]

    def detect(self, df: pd.DataFrame, min_consecutive: int = 3) -> list[AnomalyEvent]:
        """Emit one event per unit at the first cycle where the score stays above threshold for min_consecutive cycles."""
        feats = make_features(df)
        feats["score"] = self._score_feats(feats)
        events: list[AnomalyEvent] = []
        for unit_id, g in feats.groupby("unit_id", sort=False):
            g = g[g["cycle"] > WARMUP]
            above = (g["score"] > self.threshold).to_numpy()
            run = 0
            for i, flag in enumerate(above):
                run = run + 1 if flag else 0
                if run >= min_consecutive:
                    row = g.iloc[i]
                    top = [s for s, _ in self.contributions(row)[:3]]
                    events.append(
                        AnomalyEvent(
                            unit_id=str(unit_id),
                            cycle=int(row["cycle"]),
                            score=float(row["score"]),
                            top_sensors=top,
                            message=f"{unit_id}: anomaly score {row['score']:.3f} > {self.threshold:.3f}; top sensors {', '.join(top)}",
                        )
                    )
                    break
        return events


@dataclass
class RULModel:
    _model: GradientBoostingRegressor | None = field(default=None, init=False, repr=False)
    _cols: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, train: pd.DataFrame) -> "RULModel":
        feats = make_features(train)
        self._cols = feature_columns(feats)
        y = feats["rul"].clip(upper=RUL_CLIP)
        self._model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        self._model.fit(feats[self._cols].to_numpy(), y.to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        feats = make_features(df)
        return np.clip(self._model.predict(feats[self._cols].to_numpy()), 0, RUL_CLIP)

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.clip(y_true, 0, RUL_CLIP)
        d = y_pred - y_true
        rmse = float(np.sqrt(np.mean(d**2)))
        # C-MAPSS scoring function: late predictions (d > 0) are penalised more than early ones.
        score = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
        return {"rmse": rmse, "cmapss_score": score, "n": int(len(d))}


class DriftMonitor:
    """PSI per feature. PSI < 0.1 stable, 0.1–0.25 moderate drift, > 0.25 significant drift."""

    def __init__(self, reference: pd.DataFrame, bins: int = 10):
        self.bins = bins
        self.edges = {s: np.quantile(reference[s], np.linspace(0, 1, bins + 1)) for s in SENSOR_NAMES}
        self.ref_dist = {s: self._dist(reference[s], self.edges[s]) for s in SENSOR_NAMES}

    def _dist(self, x: pd.Series, edges: np.ndarray) -> np.ndarray:
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(x, bins=edges)
        return (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))

    def psi(self, current: pd.DataFrame) -> dict[str, float]:
        out = {}
        for s in SENSOR_NAMES:
            cur = self._dist(current[s], self.edges[s])
            ref = self.ref_dist[s]
            out[s] = float(np.sum((cur - ref) * np.log(cur / ref)))
        return out


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
