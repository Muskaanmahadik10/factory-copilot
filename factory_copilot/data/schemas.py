"""Pydantic schemas shared across the data, analytics, tools and agent layers."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FailureMode(str, Enum):
    BEARING_WEAR = "bearing_wear"
    COOLANT_LEAK = "coolant_leak"
    SENSOR_FAULT = "sensor_fault"
    FILTER_CLOG = "filter_clog"
    CALIBRATION_DRIFT = "calibration_drift"


SENSOR_NAMES: list[str] = [
    "spindle_vibration_mm_s",
    "spindle_temp_c",
    "motor_current_a",
    "coolant_temp_c",
    "coolant_flow_l_min",
    "coolant_pressure_bar",
    "inlet_pressure_bar",
    "outlet_pressure_bar",
    "filter_dp_bar",
    "position_error_um",
    "load_pct",
    "rpm",
]

# Which sensors move for each failure mode, with direction (+1 up, -1 down) and relative magnitude.
FAILURE_SIGNATURES: dict[FailureMode, dict[str, tuple[int, float]]] = {
    FailureMode.BEARING_WEAR: {"spindle_vibration_mm_s": (1, 3.0), "spindle_temp_c": (1, 1.2), "motor_current_a": (1, 0.6)},
    FailureMode.COOLANT_LEAK: {"coolant_pressure_bar": (-1, 1.5), "coolant_flow_l_min": (-1, 1.5), "coolant_temp_c": (1, 1.0)},
    FailureMode.SENSOR_FAULT: {"position_error_um": (1, 4.0)},  # spiky, not smooth; handled in generator
    FailureMode.FILTER_CLOG: {"filter_dp_bar": (1, 2.5), "outlet_pressure_bar": (-1, 1.0), "coolant_flow_l_min": (-1, 0.8)},
    FailureMode.CALIBRATION_DRIFT: {"position_error_um": (1, 1.2), "spindle_temp_c": (1, 0.3)},
}

SOP_FOR_MODE: dict[FailureMode, str] = {
    FailureMode.BEARING_WEAR: "SOP-M-101",
    FailureMode.COOLANT_LEAK: "SOP-C-201",
    FailureMode.SENSOR_FAULT: "SOP-E-301",
    FailureMode.FILTER_CLOG: "SOP-C-202",
    FailureMode.CALIBRATION_DRIFT: "SOP-E-302",
}

ACTION_FOR_MODE: dict[FailureMode, str] = {
    FailureMode.BEARING_WEAR: "schedule_bearing_replacement",
    FailureMode.COOLANT_LEAK: "stop_and_inspect_coolant_circuit",
    FailureMode.SENSOR_FAULT: "replace_position_sensor",
    FailureMode.FILTER_CLOG: "replace_coolant_filter",
    FailureMode.CALIBRATION_DRIFT: "run_axis_calibration",
}


class SensorReading(BaseModel):
    unit_id: str
    cycle: int
    values: dict[str, float]


class AnomalyEvent(BaseModel):
    unit_id: str
    cycle: int
    score: float = Field(description="Anomaly score; higher is more anomalous")
    top_sensors: list[str] = Field(description="Sensors contributing most to the anomaly, most important first")
    message: str = ""


class MaintenanceTicket(BaseModel):
    ticket_id: str
    unit_id: str
    cycle: int
    symptom: str
    root_cause: str
    action: str
    technician_note: str


class Evidence(BaseModel):
    source: Literal["query_sensors", "get_maintenance_history", "search_sops", "estimate_rul", "analysis"]
    summary: str = Field(max_length=400)
    citation: str = Field(default="", description="SOP id + section, ticket id, or sensor window")


class IncidentReport(BaseModel):
    unit_id: str
    root_cause: str = Field(description="One of the known failure modes or 'unknown'")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(min_length=1, max_length=8)
    sop_reference: str = Field(description="SOP id that supports the recommended action, or ''")
    recommended_action: str
    requires_approval: bool = True
    summary: str = Field(max_length=800)
