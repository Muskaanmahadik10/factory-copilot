"""Guardrails: untrusted-text wrapping, injection detection, approval gate, step/cost limits."""

from factory_copilot.guardrails.core import (  # noqa: F401
    ApprovalGate,
    Budget,
    BudgetExceeded,
    InjectionDetector,
    wrap_untrusted,
)
