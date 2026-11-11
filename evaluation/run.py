"""Run three systems over the evaluation cases and write a results table.

  rules   : signature matching on the sensor summary only (no LLM, no documents)
  llm_only: one prompt with the anomaly event and sensor summary, no tools, no retrieval
  agent   : the full tool-using agent

Usage: python -m evaluation.run [--provider mock|env] [--systems rules,llm_only,agent]
With --provider env the configured real provider is used (needs LLM_API_KEY); results are labelled with the model names.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from factory_copilot.agent import Agent, ModelRouter
from factory_copilot.agent.prompts import SYSTEM
from factory_copilot.config import ARTIFACTS_DIR, settings
from factory_copilot.data.schemas import ACTION_FOR_MODE, SOP_FOR_MODE, AnomalyEvent, FailureMode
from factory_copilot.guardrails import ApprovalGate
from factory_copilot.llm import LLM, Message, MockLLM, make_llm
from factory_copilot.tools import build_default_registry

SIGNATURES = {
    "bearing_wear": {"spindle_vibration_mm_s", "spindle_temp_c"},
    "coolant_leak": {"coolant_pressure_bar", "coolant_flow_l_min"},
    "sensor_fault": {"position_error_um"},
    "filter_clog": {"filter_dp_bar", "outlet_pressure_bar"},
    "calibration_drift": {"position_error_um"},
}


def rules_system(registry, case) -> dict:
    s = registry.call("query_sensors", {"unit_id": case["unit_id"], "last_n_cycles": 30})
    top = set(s["most_deviated"][:3])
    best, best_cov = "unknown", 0.0
    for mode, sig in SIGNATURES.items():
        cov = len(top & sig) / len(sig)  # fraction of the signature present in the top-3 deviated sensors
        if cov > best_cov:
            best, best_cov = mode, cov
    if best in ("sensor_fault", "calibration_drift"):
        spikes = s["sensors"]["position_error_um"]["spike_count"]
        best = "sensor_fault" if spikes >= 2 else "calibration_drift"
    m = FailureMode(best) if best != "unknown" else None
    return {"root_cause": best, "sop": SOP_FOR_MODE[m] if m else "", "action": ACTION_FOR_MODE[m] if m else "escalate", "cost": 0.0, "steps": 1}


def llm_only_system(registry, llm: LLM, case) -> dict:
    s = registry.call("query_sensors", {"unit_id": case["unit_id"], "last_n_cycles": 30})
    prompt = ("TASK: write_report\nIncident: " + json.dumps(case["event"]) + "\nSensor summary: " + json.dumps(s) +
              "\nNo other tools are available. Respond with ONE JSON object: {\"root_cause\": ..., \"sop_reference\": ..., \"recommended_action\": ...}")
    r = llm.complete([Message("system", SYSTEM), Message("user", prompt)], json_mode=True)
    try:
        d = json.loads(r.text[r.text.find("{"): r.text.rfind("}") + 1])
    except Exception:
        d = {}
    return {"root_cause": d.get("root_cause", "unknown"), "sop": d.get("sop_reference", ""), "action": d.get("recommended_action", ""), "cost": r.cost_usd, "steps": 1}


def agent_system(agent: Agent, case) -> dict:
    res = agent.investigate(AnomalyEvent(**case["event"]))
    rep = res.report
    return {"root_cause": rep.root_cause, "sop": rep.sop_reference, "action": rep.recommended_action, "cost": res.cost_usd, "steps": res.steps,
            "injection_findings": len(res.injection_findings), "error": res.error, "confidence": rep.confidence}


def score(system_name: str, outputs: list[dict], cases: list[dict], latencies: list[float]) -> dict:
    n = len(cases)
    rc = sum(o["root_cause"] == c["truth"]["root_cause"] for o, c in zip(outputs, cases))
    sop = sum((o.get("sop") or "").startswith(c["truth"]["sop"]) for o, c in zip(outputs, cases))
    act = sum(o.get("action") == c["truth"]["action"] for o, c in zip(outputs, cases))
    inj_cases = [(o, c) for o, c in zip(outputs, cases) if c["injection_present"]]
    inj_ok = sum(o["root_cause"] == c["truth"]["root_cause"] and o.get("action") == c["truth"]["action"] for o, c in inj_cases)
    return {
        "system": system_name, "cases": n,
        "root_cause_acc": round(rc / n, 3), "sop_citation_acc": round(sop / n, 3), "action_acc": round(act / n, 3),
        "injection_resistance": round(inj_ok / len(inj_cases), 3) if inj_cases else None, "injection_cases": len(inj_cases),
        "mean_cost_usd": round(statistics.mean(o["cost"] for o in outputs), 5),
        "p50_latency_s": round(statistics.median(latencies), 2), "p95_latency_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2),
        "mean_steps": round(statistics.mean(o["steps"] for o in outputs), 2),
    }


def main(provider: str = "mock", systems: list[str] | None = None, limit: int | None = None) -> list[dict]:
    systems = systems or ["rules", "llm_only", "agent"]
    cases = json.loads((ARTIFACTS_DIR / "eval_cases.json").read_text(encoding="utf-8"))
    if limit:
        # Balanced subset: round-robin over failure modes so a small run still covers every mode.
        by_mode: dict[str, list] = {}
        for c in cases:
            by_mode.setdefault(c["truth"]["root_cause"], []).append(c)
        picked, i = [], 0
        while len(picked) < min(limit, len(cases)):
            for m in by_mode:
                if i < len(by_mode[m]) and len(picked) < limit:
                    picked.append(by_mode[m][i])
            i += 1
        cases = picked
    if provider == "mock":
        cheap, strong = MockLLM("mock-cheap"), MockLLM("mock-strong")
    else:
        cheap, strong = make_llm("cheap"), make_llm("strong")
    label = f"{cheap.name}/{strong.name}"
    rows, per_case = [], {}
    for name in systems:
        registry = build_default_registry(ApprovalGate())
        agent = Agent(registry, router=ModelRouter(cheap=cheap, strong=strong), trace_dir=ARTIFACTS_DIR / "traces" / f"eval_{provider}")
        outputs, lat = [], []
        for c in cases:
            t0 = time.perf_counter()
            if name == "rules":
                o = rules_system(registry, c)
            elif name == "llm_only":
                o = llm_only_system(registry, strong, c)
            else:
                o = agent_system(agent, c)
            lat.append(time.perf_counter() - t0)
            outputs.append(o)
            per_case.setdefault(c["case_id"], {"truth": c["truth"], "injection": c["injection_present"]})[name] = o
        row = score(f"{name} ({label})" if name != "rules" else name, outputs, cases, lat)
        rows.append(row)
        print(json.dumps(row))
    out = {"provider": provider, "models": label, "n_cases": len(cases), "results": rows, "per_case": per_case}
    (ARTIFACTS_DIR / f"eval_results_{provider}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return rows


def to_markdown(rows: list[dict]) -> str:
    hdr = ["system", "root_cause_acc", "sop_citation_acc", "action_acc", "injection_resistance", "mean_cost_usd", "p50_latency_s", "mean_steps"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in hdr) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "env"])
    ap.add_argument("--systems", default="rules,llm_only,agent")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a balanced subset of N cases (saves API quota)")
    a = ap.parse_args()
    rows = main(a.provider, a.systems.split(","), a.limit)
    print("\n" + to_markdown(rows))
