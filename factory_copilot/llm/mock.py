"""MockLLM: a deterministic stand-in that follows the agent's protocol using simple rules over the prompt.

It is not pretending to be smart. Its purpose is (1) tests and CI without API keys, (2) a reproducible baseline in
the evaluation table, (3) exercising the tool loop, guardrails and schemas end to end. It reads the same prompts the
real model gets, so a prompt-format bug fails here first.
"""

from __future__ import annotations

import json
import re

from factory_copilot.llm.base import LLM, LLMResponse, Message

MODE_HINTS = {
    "bearing_wear": ["vibration", "spindle_temp", "bearing"],
    "coolant_leak": ["coolant_pressure", "coolant_flow", "leak"],
    "sensor_fault": ["spike", "intermittent", "reader head", "sensor fault"],
    "filter_clog": ["filter_dp", "filter", "clog"],
    "calibration_drift": ["drift", "calibration", "slowly"],
}
SOP = {"bearing_wear": "SOP-M-101", "coolant_leak": "SOP-C-201", "sensor_fault": "SOP-E-301", "filter_clog": "SOP-C-202", "calibration_drift": "SOP-E-302"}
ACTION = {"bearing_wear": "schedule_bearing_replacement", "coolant_leak": "stop_and_inspect_coolant_circuit", "sensor_fault": "replace_position_sensor", "filter_clog": "replace_coolant_filter", "calibration_drift": "run_axis_calibration"}


class MockLLM(LLM):
    def __init__(self, name: str = "mock"):
        self.name = name
        self.calls = 0

    def _guess_mode(self, text: str) -> tuple[str, float]:
        """Prefer the structured sensor summary (what a careful model would read first); fall back to text hints."""
        m = re.search(r'"most_deviated":\s*\[([^\]]*)\]', text)
        if m:
            top = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
            spikes = re.search(r'"position_error_um":\s*\{[^}]*"spike_count":\s*(\d+)', text)
            n_spikes = int(spikes.group(1)) if spikes else 0
            sig = {"bearing_wear": {"spindle_vibration_mm_s", "spindle_temp_c"}, "coolant_leak": {"coolant_pressure_bar", "coolant_flow_l_min"},
                   "filter_clog": {"filter_dp_bar", "outlet_pressure_bar"}, "position": {"position_error_um"}}
            best, best_cov = "unknown", 0.0
            for mode, s in sig.items():
                cov = len(set(top[:3]) & s) / len(s) + (0.01 if top and top[0] in s else 0)
                if cov > best_cov:
                    best, best_cov = mode, cov
            if best == "position":
                best = "sensor_fault" if n_spikes >= 2 else "calibration_drift"
            if best != "unknown":
                return best, round(min(0.95, 0.55 + 0.4 * best_cov), 2)
        t = text.lower()
        scores = {m: sum(t.count(h) for h in hints) for m, hints in MODE_HINTS.items()}
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        return (best, min(0.9, 0.4 + scores[best] / total * 0.5)) if scores[best] else ("unknown", 0.2)

    def complete(self, messages: list[Message], *, max_tokens=800, temperature=0.0, json_mode=False, tools=None) -> LLMResponse:
        self.calls += 1
        prompt = "\n".join(m.content for m in messages)
        if "TASK: choose_next_step" in prompt:
            text = self._choose_step(prompt)
        elif "TASK: write_report" in prompt:
            text = self._write_report(prompt)
        else:
            text = json.dumps({"answer": "ok"})
        return LLMResponse(text=text, model=self.name, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4, cost_usd=0.0, latency_s=0.0)

    def _choose_step(self, prompt: str) -> str:
        done = set(re.findall(r"\[step \d+\] tool=(\w+)", prompt))
        order = ["query_sensors", "estimate_rul", "get_maintenance_history", "search_sops"]
        unit = (re.search(r"unit_id[\"']?: ?[\"']?(U\d{3})", prompt) or re.search(r"\b(U\d{3})\b", prompt))
        unit_id = unit.group(1) if unit else "U001"
        for tool in order:
            if tool in done:
                continue
            if tool == "search_sops":
                mode, _ = self._guess_mode(prompt)
                q = {"bearing_wear": "spindle vibration rising spindle temperature high bearing", "coolant_leak": "coolant pressure falling coolant flow low coolant temperature rising", "sensor_fault": "position error spikes intermittent sensor", "filter_clog": "filter differential pressure high outlet pressure falling", "calibration_drift": "position error increasing slowly calibration drift"}.get(mode, "alarm response escalation")
                return json.dumps({"thought": f"Symptoms match {mode}; find the SOP.", "tool": "search_sops", "args": {"query": q, "k": 3}})
            args = {"unit_id": unit_id}
            if tool == "query_sensors":
                args["last_n_cycles"] = 30
            return json.dumps({"thought": f"Need {tool} evidence.", "tool": tool, "args": args})
        return json.dumps({"thought": "Enough evidence collected.", "tool": "finish", "args": {}})

    def _write_report(self, prompt: str) -> str:
        mode, conf = self._guess_mode(prompt)
        unit = re.search(r"\b(U\d{3})\b", prompt)
        cites = re.findall(r"(SOP-[A-Z]-\d{3} §[^\"\n]+?)(?=\"|\n|,)", prompt)
        sop_ref = SOP.get(mode, "")
        sop_cite = next((c for c in cites if c.startswith(sop_ref)), sop_ref)
        tickets = re.findall(r"\bT\d{4}\b", prompt)
        evidence = [{"source": "query_sensors", "summary": f"Sensor trend consistent with {mode.replace('_', ' ')}.", "citation": "last 30 cycles"}]
        if sop_ref:
            evidence.append({"source": "search_sops", "summary": f"{sop_ref} symptoms and diagnosis match the observed sensors.", "citation": sop_cite})
        if tickets:
            evidence.append({"source": "get_maintenance_history", "summary": "Similar past ticket on this or a sibling unit.", "citation": tickets[0]})
        rul = re.search(r"rul_cycles[\"']?: ?([\d.]+)", prompt)
        if rul:
            evidence.append({"source": "estimate_rul", "summary": f"Estimated remaining useful life {float(rul.group(1)):.0f} cycles.", "citation": "RUL model"})
        report = {
            "unit_id": unit.group(1) if unit else "U001",
            "root_cause": mode, "confidence": round(conf, 2), "evidence": evidence,
            "sop_reference": sop_ref, "recommended_action": ACTION.get(mode, "escalate_to_maintenance_engineer"),
            "requires_approval": True,
            "summary": f"Evidence points to {mode.replace('_', ' ')}. Recommended: {ACTION.get(mode, 'escalate')} per {sop_ref or 'SOP-G-001'}. Human approval required before any work order.",
        }
        return json.dumps(report)
