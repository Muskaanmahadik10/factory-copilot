"""Analytics layer: anomaly detection, remaining-useful-life estimation, drift monitoring."""

from factory_copilot.analytics.features import make_features, split_units  # noqa: F401
from factory_copilot.analytics.models import AnomalyDetector, DriftMonitor, RULModel  # noqa: F401
