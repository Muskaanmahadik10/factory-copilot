# Factory Copilot

**An AI agent that investigates production-line anomalies the way a good maintenance engineer would: pull the sensor history, check the RUL model, read the maintenance log, find the right SOP, then write an evidence-backed report. A human approves every action.**

Built as a portfolio project for AI-engineer roles in manufacturing. Everything runs on a laptop with no API key (deterministic mock mode); with a key it uses real models through any OpenAI-compatible or Anthropic endpoint.

![CI](https://github.com/Muskaanmahadik10/factory-copilot/actions/workflows/ci.yml/badge.svg) [![M8ven Score](https://m8ven.ai/badge/mcp/muskaanmahadik10-factory-copilot-tls60g?v=0794f1b750d7ebcd499e5ec963341099)](https://m8ven.ai/mcp/muskaanmahadik10-factory-copilot-tls60g?s=readme) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

## What it does

```
sensor data (CSV replay) ──► anomaly detector ──► event ──► agent ──► incident report ──► human approval ──► local work-order record
                             + RUL model                     │ tools: query_sensors · estimate_rul ·
                             + drift monitor                 │        get_maintenance_history · search_sops (lexical RAG)
                                                             │ investigation tools see data only up to the incident cycle
                                                             │ router: cheap model for steps, strong model for the report
                                                             │ guardrails: untrusted-data wrapping, injection redaction,
                                                             │            least-privilege tools, step/cost budget
```

A shift lead sees "Line 3 is drifting" and gets a report that says *what* (root cause, with the model's own uncalibrated confidence), *why* (sensor evidence and the SOP section that describes these symptoms), *what to do* (the SOP's action), and *what it cost* (tokens and dollars per incident), with the work order parked for their approval. With the real models below a report takes about 75 seconds.

**Information boundary.** An investigation sees only what existed at the anomaly cycle: the sensor window and RUL estimate stop there, maintenance tickets written later (including the repair ticket at final failure) are invisible, and the sensor baseline comes from the machine's own first 30 cycles or the training fleet, never from held-out machines. An earlier version did not enforce this; see [Results](#results).

![Dashboard: unit U001 investigated live with gpt-oss models on Groq](docs/img/dashboard.jpg)

*Screenshot from the earlier version (before the incident cutoff), so its evidence list includes ticket T1000, written at this unit's final failure. The current version cannot see that ticket. The guardrail redacting a planted "ignore all previous instructions" note works the same way in both.*

## Results

### Analytics (held-out machines, time order preserved; `artifacts/analytics_metrics.json`)

| Metric | Value |
|---|---|
| Units with degradation detected before failure | **15 / 15** |
| Cycle-level precision / recall on degraded cycles | 0.97 / 0.77 |
| Median warning before failure | **80 cycles** |
| RUL model RMSE (clipped at 125, C-MAPSS convention; 3,391 cycle-level predictions from 15 machines, so not independent samples) | 20.0 cycles |
| Drift monitor (PSI, degraded vs healthy) | position error 0.40 (significant), coolant flow 0.19 (moderate) |

### Agent: real models, same 10 cases for every system

Ten incidents on held-out machines (2 per failure mode), each investigated at its first anomaly cycle with data up to that cycle only. Models: gpt-oss-20b (tool steps) and gpt-oss-120b (report) on Groq. `artifacts/eval_results_env.json`, 14 Nov 2026.

Provenance: this file (and `eval_results_mock.json` before 14 Nov) records `code_commit: b221be0`, but it was produced from the **uncommitted repair state on top of `b221be0`**, which was committed afterwards as `8912458`. The runner recorded HEAD only. It now records `git describe --dirty`, a hash of any uncommitted diff and a hash of the case file. The saved file was left as generated, not relabelled. The full report, tool outputs and sensor window behind each of the 10 agent rows are in [`artifacts/live_run_2026-11-13/`](artifacts/live_run_2026-11-13/).

| System | Root cause | SOP id match | Action keyword match | Clean / recovered / failed runs | Cost / incident | p50 latency | Steps |
|---|---|---|---|---|---|---|---|
| Rules (signature matching on the sensor summary, no LLM) | 6/10 | 6/10 | 6/10 | 10 / 0 / 0 | $0 | 0.01 s | 1 |
| LLM only (gpt-oss-120b, sensor summary, no tools) | 4/10 | **0/10** | 7/10 | 10 / 0 / 0 | $0.0003 | 1.8 s | 1 |
| **Agent (tools + SOP retrieval)** | **7/10** | **7/10** | 7/10 | 10 / 0 / 0 | $0.0019 | 75 s | 4.8 |

How to read this:
- **The agent's measured advantage over rules is small: 7 vs 6 correct on 10 cases, which is not a meaningful difference.** They also fail differently. Rules mistook both filter clogs for coolant leaks; the agent got both right. The agent called one coolant leak a filter clog. Both systems called both sensor faults "calibration drift": at the first anomaly a faulty sensor has produced only one or two spikes, which looks like slow drift.
- **What the agent adds is checkable citations, not accuracy.** The LLM without tools named a cause but never a real SOP (0/10; it invented ids like `SOP-FT-001`). In all 10 agent reports, every SOP and ticket id mentioned anywhere in the report was a record that investigation's tools had returned, and each trace shows the sensor window used ([`artifacts/live_run_2026-11-13/`](artifacts/live_run_2026-11-13/)). That shows the ids exist in the evidence, not that the passages support the conclusion: the wrong C05 diagnosis also cited a retrieved SOP. Reports now carry this check (`citation_check`), and an id the investigation never retrieved is flagged. The check counts only returned records (search hit ids, ticket ids, the RUL policy reference). An earlier version also counted any id appearing in a tool's output, so a model could have made an invented id pass by searching for it; an independent audit found that, and it is fixed. Whether any of this saves an engineer time is not measured here.
- **Confidence is the model's self-assessment, not a probability.** All three wrong agent answers were stated at 80–85%, higher than any correct one (60–78%).
- **Early diagnosis is genuinely hard.** These cases are deliberately taken at the first anomaly, often dozens of cycles before the pattern is obvious.
- No real-model injection test yet: the three planted attacks are on training machines, and this run used held-out machines only.

### Offline checks (mock model, all 30 cases, `artifacts/eval_results_mock.json`)

The mock model is a rule-based stand-in used for tests and CI. Its scores check that the pipeline works, not that a model reasons well: rules 16/30, mock agent 18/30. One of the 30 cases (C23) alarms at cycle 41, before its unit's degradation starts at cycle 130; it is now scored as a false alarm (right answer: cause unknown, monitor or escalate), not as the coolant leak that ends the unit's life later. All three systems got it wrong either way, so the totals did not change. Of the 3 cases with a planted injection note, the mock agent diagnosed 2 correctly (`injection_cases_root_cause_and_action_match`, formerly mislabelled `injection_resistance`), and in all 3 the note was redacted before the model saw it. Its single miss is a diagnosis error, not the attack working.

### What changed from the first published results

The first version (commit `b221be0`, originals kept in [`artifacts/superseded_b221be0/`](artifacts/superseded_b221be0/)) reported agent 10/10 vs rules 28/30. An audit found that comparison was not valid:

1. The tools read data from after the incident, including the repair ticket written at final failure, which states the root cause.
2. The healthy baseline used every machine, including held-out ones.
3. Rules and agent were scored on different case sets. On the agent's 10 cases, rules also scored 10/10.
4. A run with a provider error still counted as fully correct.

Fixes: an incident-time cutoff in every tool, causal features, a training-only baseline, one case set for all systems, metrics named for what they check, and clean/recovered/failed runs counted separately.

While fixing the baseline I also changed how `query_sensors` ranks sensors (each machine's own early cycles as baseline, deviation scaled by sensor noise). The change is general, but I made it after inspecting results on these cases, so these numbers are not from a fully untouched test set.

### What each metric checks

- **Root cause:** exact match of the failure-mode label.
- **SOP id match:** the SOP id parsed from the answer equals the expected one. It does not check that the cited section supports the recommendation.
- **Action keyword match:** the action names the failed component or procedure, is not negated and is not just "escalate". It is a keyword check, not an expert review.
- **SOP ids** are parsed with token boundaries: `SOP-C-201FAKE` and `SOP-C-2010` do not count as `SOP-C-201`.

## Why this design

| Decision | Reason |
|---|---|
| **Synthetic data with labelled failure modes** (C-MAPSS-style run-to-failure, 60 units, 12 sensors) | Real datasets have degradation but no ground-truth failure mode, SOP or action, so an agent can't be scored honestly. The generator keeps the C-MAPSS structure and adds five failure modes whose sensor signatures match the SOP corpus. |
| **Anomaly ensemble: robust z-score of rolling means + IsolationForest**, trained on healthy cycles only | Drift shows in trends, not single readings; IsolationForest alone missed half the units. Threshold = 99.5th percentile of healthy scores. |
| **Split by machine, time order kept, incident cutoff** | A new machine and future cycles are what deployment looks like. Random k-fold would leak, and so would letting an investigation read data from after the alarm. |
| **Hybrid lexical retrieval (BM25 + TF-IDF) over SOP sections, with hand-written reranking boosts** | Section-level chunks give precise citations ("SOP-C-201 §2 Diagnosis"). Lexical search runs anywhere; no embeddings or learned reranker yet, and the `Index` interface accepts an embedding backend. |
| **Native tool calling, report submitted as a tool** | Structured arguments instead of free-text JSON parsing, which removed the JSON-mode failures seen in the first real run. Every report is pydantic-validated before anyone sees it; that checks structure, not facts. Cited SOP/ticket ids are also checked against what the investigation retrieved. |
| **Model router: cheap model for tool choice, strong model for the report** | Tool choice is short and structured; synthesis with citations is where quality matters. Every completed model call is in the per-incident trace with tier, tokens and cost, including the call that pushes a run over budget. |
| **Guardrails as code, not prompt text** | Tool permissions live in the registry (`open_ticket` is a write tool and always goes to the approval gate). Ticket notes and documents are wrapped as `<untrusted_data>` and scanned; instruction-like text is redacted and recorded per investigation. Only the offered read tools are dispatched during an investigation; a write tool named anyway is refused, and a human decision on a work order is final (a second decision is rejected). Cost and time budgets are checked before each model call starts; the call that crosses the cost limit is logged, then no further call is made (report included) and a local escalation report is returned, so a run can overshoot by at most one call. The time budget is checked between calls, not enforced inside one: a single call with its network retries can run past it. |
| **Mock LLM in CI** | Tests and the baseline evaluation never need a key, so a prompt-format bug fails in CI, not in a demo. |

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
make all            # data → train → tests → mock evaluation  (≈1 minute)
make dashboard      # Streamlit UI
make api            # FastAPI at http://localhost:8000/docs
```

Real models: copy `.env.example` to `.env`, set `LLM_PROVIDER=openai_compatible` (Groq, OpenAI, Ollama, vLLM) or `anthropic`, add the key, then `make eval-real` (10 balanced held-out cases, to respect free-tier limits). A real provider with no key now fails with an error instead of quietly using the mock.

Historical investigation: `POST /investigate {"unit_id": "U001", "cycle": 208}` investigates as of that cycle. Detection, the anomaly event (score, top sensors, message) and every tool use only rows up to it; the response includes the event. Out-of-range cycles return 400.

MCP: `make mcp` exposes six tools over stdio (the five agent tools plus `list_pending_approvals`) for Claude Desktop / Claude Code (config snippet in `factory_copilot/mcp_server.py`). These are latest-state tools with no incident cutoff: they return a unit's newest data, including a repair ticket written at final failure. The incident-time boundary applies to `Agent.investigate()` and the API's `/investigate`, not to MCP. Every tool declares all four MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) so a host can warn before a call: five tools are read-only, and `open_ticket` only queues a request for human approval (tested in `tests/test_mcp_annotations.py`). The server is listed on the [M8ven MCP Trust Index](https://m8ven.ai/mcp/muskaanmahadik10/factory-copilot), whose static analysis reported no credential exfiltration, sensitive file access or obfuscation.

## Repository map

```
factory_copilot/
  data/        schemas.py (failure modes, signatures, report schema) · generate.py · sops.py (12 SOPs) · build.py
  analytics/   features.py (rolling features, unit split) · models.py (AnomalyDetector, RULModel, DriftMonitor) · train.py
  retrieval/   index.py (chunker, HybridIndex, reranker, citations)
  tools/       registry.py (schemas, permissions, sanitised outputs, 5 tools)
  guardrails/  core.py (wrap_untrusted, InjectionDetector, Budget, ApprovalGate)
  llm/         base.py · clients.py (OpenAI-compatible, Anthropic, retry/backoff) · mock.py
  agent/       prompts.py · core.py (loop, ModelRouter, trace, fallback report)
  citations.py exact SOP/ticket id parsing; checks a report cites only ids its investigation retrieved
  api.py       FastAPI: /incidents /investigate /approvals /metrics
  mcp_server.py
dashboard/app.py    Streamlit
evaluation/         cases.py (30 incidents with ground truth) · run.py (rules / llm_only / agent, results table)
tests/              63 tests (data, analytics, incident cutoff, retrieval, guardrails, tools, agent, scoring, API, audit regressions)
docs/DESIGN.md      the design document this was built from
```

## What I'd do next

- Embedding retrieval + reranker model behind the same `Index` interface, evaluated against the lexical baseline.
- Cross-incident memory (what was approved last time on this unit) with the same untrusted-data treatment.
- Real connectors (OPC-UA / historian) behind `query_sensors`; the tool schema would not change.
- Separate sensor faults from calibration drift earlier (spike timing, not just counts): both systems miss this at first detection.
- Single-pass RAG vs the iterative agent on identical frozen evidence, to see whether the extra loop earns its cost.
- A missing-sensor policy: a NaN reading currently makes the RUL model raise; real feeds need validation or causal imputation.
- Replace the pre-cutoff dashboard screenshot with a current live report.
- A real-model injection test on held-out machines, and a larger frozen case set that was not used during development.
- Measure what matters to a shift lead: time to a verified diagnosis with and without the copilot, against the rules workflow.
- A proper LLM-as-judge for report *quality* (evidence sufficiency), not just root-cause/SOP/action accuracy.

## License

MIT, see [LICENSE](LICENSE).

## Author

Muskaan Mahadik · M.S. Intelligent Manufacturing Technology (Data Science track), Taiwan Tech · thesis on quantization and machine unlearning in LLMs.
