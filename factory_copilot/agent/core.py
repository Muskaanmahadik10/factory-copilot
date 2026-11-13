"""The agent loop.

plan/act/observe with a cheap model choosing tools, a strong model writing the final report, pydantic validation of
everything that crosses a boundary, and a trace that records prompts, tool calls, tokens and cost per step.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from factory_copilot.agent.prompts import CHOOSE_STEP, SYSTEM, WRITE_REPORT
from factory_copilot.config import ARTIFACTS_DIR, settings
from factory_copilot.data.schemas import AnomalyEvent, IncidentReport
from factory_copilot.guardrails import Budget, BudgetExceeded, wrap_untrusted
from factory_copilot.llm import LLM, Message, make_llm
from factory_copilot.tools import ToolRegistry

UNTRUSTED_TOOLS = {"get_maintenance_history", "search_sops"}


@dataclass
class ModelRouter:
    """Routes each step to a model tier. Policy is explicit and logged so cost decisions are auditable."""

    cheap: LLM
    strong: LLM

    def for_step(self, step_type: str, retry: bool = False) -> tuple[str, LLM]:
        if step_type == "write_report" or retry:
            return "strong", self.strong
        return "cheap", self.cheap

    @classmethod
    def default(cls) -> "ModelRouter":
        return cls(cheap=make_llm("cheap"), strong=make_llm("strong"))


@dataclass
class TraceStep:
    kind: str  # llm | tool
    name: str
    tier: str = ""
    input: dict | str = ""
    output: dict | str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


@dataclass
class AgentResult:
    incident_id: str
    report: IncidentReport | None
    trace: list[TraceStep]
    cost_usd: float
    latency_s: float
    steps: int
    error: str = ""
    injection_findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id, "report": self.report.model_dump() if self.report else None,
            "cost_usd": round(self.cost_usd, 6), "latency_s": round(self.latency_s, 3), "steps": self.steps, "error": self.error,
            "injection_findings": self.injection_findings,
            "trace": [t.__dict__ for t in self.trace],
        }


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class Agent:
    def __init__(self, registry: ToolRegistry, router: ModelRouter | None = None, max_steps: int | None = None, max_cost_usd: float | None = None, trace_dir: Path | None = None):
        self.registry = registry
        self.router = router or ModelRouter.default()
        self.max_steps = max_steps or settings.max_steps
        self.max_cost_usd = max_cost_usd or settings.max_cost_usd
        self.trace_dir = trace_dir or ARTIFACTS_DIR / "traces"

    # ---- helpers ----
    def _native_tools(self) -> list[dict]:
        """Read tools as OpenAI function schemas, plus a 'finish' pseudo-tool. Write tools are deliberately not offered."""
        out = []
        for s in self.registry.specs():
            if s["permission"] != "read":
                continue
            schema = {k: v for k, v in s["input_schema"].items() if k in ("type", "properties", "required")}
            out.append({"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": schema}})
        out.append({"type": "function", "function": {"name": "finish", "description": "Stop investigating and write the report. Call when you have sensor evidence and an SOP citation.", "parameters": {"type": "object", "properties": {}}}})
        return out

    def _report_tool(self) -> list[dict]:
        """The report is submitted through a native tool whose parameters are the IncidentReport schema: structured output
        enforced by the provider, no JSON-mode quirks."""
        schema = IncidentReport.model_json_schema()
        props = {k: v for k, v in schema["properties"].items() if k not in ("unit_id", "requires_approval")}
        # Inline the Evidence definition so providers that don't resolve $defs still get a complete schema.
        ev = schema["$defs"]["Evidence"]
        props["evidence"] = {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "properties": ev["properties"], "required": ev["required"]}}
        return [{"type": "function", "function": {"name": "submit_report", "description": "Submit the final incident report.", "parameters": {"type": "object", "properties": props, "required": [k for k in props]}}}]

    def _llm(self, step_type: str, prompt: str, budget: Budget, trace: list[TraceStep], retry: bool = False) -> dict:
        tier, llm = self.router.for_step(step_type, retry=retry)
        tools = self._native_tools() if step_type == "choose_next_step" else self._report_tool()
        resp = llm.complete([Message("system", SYSTEM), Message("user", prompt)], json_mode=True, tools=tools, max_tokens=3000 if step_type == "write_report" else 1200)
        budget.charge(resp.cost_usd)
        trace.append(TraceStep("llm", step_type, tier=f"{tier}:{llm.name}", input=prompt[-1500:], output=(json.dumps(resp.tool_calls) if resp.tool_calls else resp.text)[:2000], tokens_in=resp.input_tokens, tokens_out=resp.output_tokens, cost_usd=resp.cost_usd, latency_s=resp.latency_s))
        if resp.tool_calls:  # native tool call: take the first one as this step's decision (or the submitted report)
            tc = resp.tool_calls[0]
            if step_type == "write_report":
                return tc["args"] if tc["name"] == "submit_report" else _extract_json(resp.text)
            return {"thought": resp.text, "tool": tc["name"], "args": tc["args"]}
        if step_type == "choose_next_step" and not resp.text.strip():
            return {"tool": "finish", "args": {}}
        return _extract_json(resp.text)

    def _history(self, trace: list[TraceStep]) -> str:
        lines = []
        n = 0
        for t in trace:
            if t.kind != "tool":
                continue
            n += 1
            out = json.dumps(t.output, ensure_ascii=False)[:2500]
            if t.name in UNTRUSTED_TOOLS:
                out = wrap_untrusted(out, t.name)
            lines.append(f"[step {n}] tool={t.name} args={json.dumps(t.input)}\nresult={out}")
        return "\n".join(lines) or "(none yet)"

    # ---- main entry ----
    def investigate(self, event: AnomalyEvent) -> AgentResult:
        """Investigate an event using only data available at its cycle (the registry's incident cutoff)."""
        with self.registry.as_of(event.cycle):
            return self._investigate(event)

    def _investigate(self, event: AnomalyEvent) -> AgentResult:
        incident_id = f"INC-{uuid.uuid4().hex[:8]}"
        budget = Budget(max_steps=self.max_steps, max_cost_usd=self.max_cost_usd)
        trace: list[TraceStep] = []
        t0 = time.perf_counter()
        incident = json.dumps(event.model_dump())
        # Only read tools are listed, matching the native tool schemas: offering open_ticket in text but not as a
        # callable tool made one real model try to call it (provider HTTP 400 on case C05 of the first real run).
        tools = json.dumps([{"name": s["name"], "description": s["description"], "permission": s["permission"], "args": list(s["input_schema"].get("properties", {}))} for s in self.registry.specs() if s["permission"] == "read"])
        report, error = None, ""
        try:
            while True:
                budget.step()
                decision = self._llm("choose_next_step", CHOOSE_STEP.format(incident=incident, tools=tools, history=self._history(trace), max_steps=self.max_steps), budget, trace)
                tool, args = decision.get("tool", "finish"), decision.get("args", {}) or {}
                if tool == "finish":
                    break
                ts = time.perf_counter()
                out = self.registry.call(tool, args)
                trace.append(TraceStep("tool", tool, input=args, output=out, latency_s=time.perf_counter() - ts))
                if budget.steps >= self.max_steps:
                    break
        except BudgetExceeded as e:
            error = str(e)
        except Exception as e:  # provider/network/JSON errors: record, then still try to write a report from what we have
            error = f"{type(e).__name__}: {e}"

        # Report, with one strong-model retry on validation failure, then a safe fallback.
        for attempt in range(2):
            try:
                raw = self._llm("write_report", WRITE_REPORT.format(incident=incident, history=self._history(trace)), budget, trace, retry=attempt == 1)
                raw["unit_id"] = event.unit_id
                raw["requires_approval"] = True
                report = IncidentReport(**raw)
                break
            except (ValidationError, ValueError, KeyError, BudgetExceeded) as e:
                error = error or f"report validation failed: {type(e).__name__}"
                continue
            except Exception as e:
                error = error or f"{type(e).__name__}: {e}"
                break
        if report is None:
            report = IncidentReport(unit_id=event.unit_id, root_cause="unknown", confidence=0.1,
                                    evidence=[{"source": "analysis", "summary": (f"Automatic investigation incomplete ({(error or 'no valid report')[:120]}). Anomaly on sensors: {', '.join(event.top_sensors)}.")[:400], "citation": f"cycle {event.cycle}"}],
                                    sop_reference="SOP-G-001", recommended_action="escalate_to_maintenance_engineer", requires_approval=True,
                                    summary="The copilot could not produce a validated report; escalating per SOP-G-001 with the raw anomaly evidence attached.")
        result = AgentResult(incident_id=incident_id, report=report, trace=trace, cost_usd=budget.cost_usd, latency_s=time.perf_counter() - t0, steps=budget.steps, error=error,
                             injection_findings=[f.__dict__ for f in self.registry.detector.findings])
        self.registry.detector.findings.clear()
        self._persist(result)
        return result

    def _persist(self, result: AgentResult) -> None:
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.trace_dir / f"{result.incident_id}.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
