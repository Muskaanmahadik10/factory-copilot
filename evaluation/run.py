"""Run three systems over the SAME evaluation cases and write a results table.

  rules   : signature matching on the sensor summary only (no LLM, no documents)
  llm_only: one prompt with the anomaly event and sensor summary, no tools, no retrieval
  agent   : the full tool-using agent

Every system sees data only up to the incident cycle (the registry's cutoff), and every system is scored on the
same case IDs, so the rows of the table are directly comparable.

Usage: python -m evaluation.run [--provider mock|env] [--systems rules,llm_only,agent] [--limit N] [--held-out-only]
With --provider env the configured real provider is used (needs LLM_API_KEY); results are labelled with the model names.

What each metric checks (and does not):
  root_cause_acc        exact match of the failure-mode label
  sop_id_match          the SOP id (e.g. SOP-C-201) parsed from the answer equals the expected one; it does not check
                        that the cited section supports the recommendation
  action_keyword_match  the action names the failed component/procedure and is not negated or a bare escalation; a
                        keyword check, not an expert review of the procedure
  clean / recovered / failed runs   whether the run finished without an error, finished with a usable report after an
                        error, or fell back to the escalation report
  injection_cases_root_cause_and_action_match   on cases with a planted injection note: root cause correct AND action
                        keyword match. It does not measure redaction, payload exposure or prohibited behaviour.
  citations_not_retrieved   agent reports citing an SOP/ticket id that its own investigation never retrieved
  confidence (per case) the model's self-assessment, not a calibrated probability of being right

Provenance: `code_commit` is `git describe --always --dirty`; when the tree is dirty `code_diff_sha256` hashes the
uncommitted diff, and `inputs_sha256` hashes the case file, so a result can be tied to the exact source and inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import time

from factory_copilot.agent import Agent, ModelRouter
from factory_copilot.agent.prompts import SYSTEM
from factory_copilot.citations import first_sop_id
from factory_copilot.config import ARTIFACTS_DIR, ROOT
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


def _sensors_at_incident(registry, case) -> dict:
    with registry.as_of(case["event"]["cycle"]):
        return registry.call("query_sensors", {"unit_id": case["unit_id"], "last_n_cycles": 30})


def rules_system(registry, case) -> dict:
    s = _sensors_at_incident(registry, case)
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
    return {"root_cause": best, "sop": SOP_FOR_MODE[m] if m else "", "action": ACTION_FOR_MODE[m] if m else "escalate", "cost": 0.0, "steps": 1, "error": ""}


def llm_only_system(registry, llm: LLM, case) -> dict:
    s = _sensors_at_incident(registry, case)
    prompt = ("TASK: write_report\nIncident: " + json.dumps(case["event"]) + "\nSensor summary: " + json.dumps(s) +
              "\nNo other tools are available. Respond with ONE JSON object: {\"root_cause\": ..., \"sop_reference\": ..., \"recommended_action\": ...}")
    try:
        r = llm.complete([Message("system", SYSTEM), Message("user", prompt)], json_mode=True)
    except Exception as e:  # provider error: an honest failed run, not a crash of the whole evaluation
        return {"root_cause": "unknown", "sop": "", "action": "", "cost": 0.0, "steps": 1, "error": f"{type(e).__name__}: {str(e)[:160]}", "fallback": True}
    try:
        d = json.loads(r.text[r.text.find("{"): r.text.rfind("}") + 1])
        err = ""
    except Exception:
        d, err = {}, "unparseable model output"
    return {"root_cause": d.get("root_cause", "unknown"), "sop": d.get("sop_reference", ""), "action": d.get("recommended_action", ""), "cost": r.cost_usd, "steps": 1,
            "error": err, "fallback": not d}


def agent_system(agent: Agent, case) -> dict:
    res = agent.investigate(AnomalyEvent(**case["event"]))
    rep = res.report
    # incident_id links this row to its full trace (report, evidence, tool outputs) in the trace directory.
    return {"incident_id": res.incident_id, "root_cause": rep.root_cause, "sop": rep.sop_reference, "action": rep.recommended_action, "cost": res.cost_usd, "steps": res.steps,
            "injection_findings": len(res.injection_findings), "error": res.error, "fallback": res.fallback, "confidence": rep.confidence,
            "citations_not_retrieved": res.citation_check.get("unretrieved_citations", [])}


ACTION_KEYWORDS = {
    "bearing_wear": ["bearing"],
    "coolant_leak": ["leak", "coolant circuit", "hose", "seal"],
    "sensor_fault": ["sensor", "reader", "encoder", "scale"],
    "filter_clog": ["filter"],
    "calibration_drift": ["calibrat", "backlash"],
}
_NEGATION = re.compile(r"\b(no|not|never|don't|dont|avoid|skip|without)\b")


def action_matches(action: str, truth_mode: str) -> bool:
    """Keyword check: the action names the failed component/procedure, is not negated and is not a bare escalation.
    Wording varies ('perform_axis_calibration' vs 'run_axis_calibration'), so this is deliberately lenient about
    phrasing; it does not judge sequencing, urgency or whether the procedure is right."""
    a = (action or "").lower().replace("_", " ").replace("-", " ")
    if not a or "escalate" in a or _NEGATION.search(a):
        return False
    return any(k in a for k in ACTION_KEYWORDS.get(truth_mode, []))


def sop_id(text: str) -> str:
    """The first exact SOP id in the text (token-bounded: 'SOP-C-201FAKE' and 'SOP-C-2010' are not SOP-C-201)."""
    return first_sop_id(text)


def run_status(o: dict) -> str:
    if o.get("fallback"):
        return "failed"
    return "recovered" if o.get("error") else "clean"


def score(system_name: str, outputs: list[dict], cases: list[dict], latencies: list[float]) -> dict:
    n = len(cases)
    pairs = list(zip(outputs, cases))
    held = [(o, c) for o, c in pairs if c.get("held_out")]
    rc = lambda ps: sum(o["root_cause"] == c["truth"]["root_cause"] for o, c in ps)
    statuses = [run_status(o) for o in outputs]
    row = {
        "system": system_name, "cases": n, "held_out_cases": len(held),
        "root_cause_acc": round(rc(pairs) / n, 3),
        "root_cause_acc_held_out": round(rc(held) / len(held), 3) if held else None,
        "sop_id_match": round(sum(sop_id(o.get("sop")) == c["truth"]["sop"] for o, c in pairs) / n, 3),
        "action_keyword_match": round(sum(action_matches(o.get("action"), c["truth"]["root_cause"]) for o, c in pairs) / n, 3),
        "clean_runs": statuses.count("clean"), "recovered_runs": statuses.count("recovered"), "failed_runs": statuses.count("failed"),
        "mean_cost_usd": round(statistics.mean(o["cost"] for o in outputs), 5),
        "p50_latency_s": round(statistics.median(latencies), 2), "p95_latency_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2),
        "mean_steps": round(statistics.mean(o["steps"] for o in outputs), 2),
    }
    # Only meaningful for a system that reads the (untrusted) maintenance notes. Named for what it checks: diagnosis and
    # action keyword on injection-labelled cases, not whether the attack text was redacted or obeyed.
    inj = [(o, c) for o, c in pairs if c["injection_present"]]
    if system_name.startswith("agent") and inj:
        row["injection_cases"] = len(inj)
        row["injection_cases_root_cause_and_action_match"] = round(sum(o["root_cause"] == c["truth"]["root_cause"] and action_matches(o.get("action"), c["truth"]["root_cause"]) for o, c in inj) / len(inj), 3)
    if any("citations_not_retrieved" in o for o in outputs):
        row["citations_not_retrieved"] = sum(bool(o.get("citations_not_retrieved")) for o in outputs)
    return row


def _select(cases: list[dict], limit: int | None, held_out_only: bool) -> list[dict]:
    if held_out_only:
        cases = [c for c in cases if c["held_out"]]
    if not limit:
        return cases
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
    return picked


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout


def _provenance() -> dict:
    """Which source produced this run. `git describe --dirty` marks uncommitted changes and the diff hash pins them, so
    a run from a modified tree can no longer be mistaken for a run of the last commit."""
    try:
        describe = _git("describe", "--always", "--dirty", "--abbrev=7").strip()
        untracked = [f for f in _git("ls-files", "--others", "--exclude-standard").splitlines() if f.endswith(".py")]
        out = {"code_commit": describe, "code_dirty": describe.endswith("-dirty") or bool(untracked)}
        if out["code_dirty"]:
            out["code_diff_sha256"] = hashlib.sha256(_git("diff", "HEAD", "--binary").encode("utf-8")).hexdigest()
            out["untracked_python_files"] = untracked
        return out
    except Exception:
        return {"code_commit": "", "code_dirty": None}


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(provider: str = "mock", systems: list[str] | None = None, limit: int | None = None, held_out_only: bool = False) -> list[dict]:
    systems = systems or ["rules", "llm_only", "agent"]
    cases = _select(json.loads((ARTIFACTS_DIR / "eval_cases.json").read_text(encoding="utf-8")), limit, held_out_only)
    if provider == "mock":
        cheap, strong = MockLLM("mock-cheap"), MockLLM("mock-strong")
    else:
        cheap, strong = make_llm("cheap"), make_llm("strong")
        if isinstance(cheap, MockLLM) or isinstance(strong, MockLLM):
            raise SystemExit("--provider env resolved to the mock model; refusing to label mock output as a real-model run")
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
            o["latency_s"] = round(time.perf_counter() - t0, 2)
            lat.append(o["latency_s"])
            outputs.append(o)
            per_case.setdefault(c["case_id"], {"unit_id": c["unit_id"], "held_out": c["held_out"], "truth": c["truth"], "injection": c["injection_present"]})[name] = o
        row = score(f"{name} ({label})" if name != "rules" else name, outputs, cases, lat)
        rows.append(row)
        print(json.dumps(row))
    out = {"provider": provider, "models": label, "n_cases": len(cases), "case_ids": [c["case_id"] for c in cases],
           "selection": {"limit": limit, "held_out_only": held_out_only}, "data_cutoff": "incident cycle",
           **_provenance(), "inputs_sha256": {"eval_cases.json": _sha256(ARTIFACTS_DIR / "eval_cases.json")},
           "trace_dir": f"artifacts/traces/eval_{provider}", "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "results": rows, "per_case": per_case}
    (ARTIFACTS_DIR / f"eval_results_{provider}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return rows


def to_markdown(rows: list[dict]) -> str:
    hdr = ["system", "cases", "root_cause_acc", "sop_id_match", "action_keyword_match", "clean_runs", "recovered_runs", "failed_runs", "mean_cost_usd", "p50_latency_s", "mean_steps"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in hdr) + " |")
    return "\n".join(lines)


def rescore(provider: str) -> list[dict]:
    """Recompute the results table from saved per-case outputs (no model calls)."""
    path = ARTIFACTS_DIR / f"eval_results_{provider}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    cases_by_id = {c["case_id"]: c for c in json.loads((ARTIFACTS_DIR / "eval_cases.json").read_text(encoding="utf-8"))}
    rows = []
    for old in saved["results"]:
        name = old["system"].split(" (")[0]
        ids = [cid for cid, v in saved["per_case"].items() if name in v]
        outputs = [saved["per_case"][cid][name] for cid in ids]
        cases = [cases_by_id[cid] for cid in ids]
        row = score(old["system"], outputs, cases, [o.get("latency_s", old["p50_latency_s"]) for o in outputs])
        rows.append(row)
    saved["results"] = rows
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "env"])
    ap.add_argument("--systems", default="rules,llm_only,agent")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a balanced subset of N cases (saves API quota)")
    ap.add_argument("--held-out-only", action="store_true", help="only cases on machines the analytics models never saw")
    ap.add_argument("--rescore", action="store_true", help="recompute the table from saved outputs without calling models")
    a = ap.parse_args()
    rows = rescore(a.provider) if a.rescore else main(a.provider, a.systems.split(","), a.limit, a.held_out_only)
    print("\n" + to_markdown(rows))
