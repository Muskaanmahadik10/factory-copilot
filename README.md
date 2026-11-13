# Factory Copilot

**An AI agent that investigates production-line anomalies the way a good maintenance engineer would: pull the sensor history, check the RUL model, read the maintenance log, find the right SOP, then write an evidence-backed report. A human approves every action.**

Built as a portfolio project for AI-engineer roles in manufacturing. Everything runs on a laptop with no API key (deterministic mock mode); with a key it uses real models through any OpenAI-compatible or Anthropic endpoint.

![CI](https://github.com/Muskaanmahadik10/factory-copilot/actions/workflows/ci.yml/badge.svg)

## What it does

```
sensor data (CSV replay) ──► anomaly detector ──► event ──► agent ──► incident report ──► human approval ──► local work-order record
                             + RUL model                     │ tools: query_sensors · estimate_rul ·
                             + drift monitor                 │        get_maintenance_history · search_sops (lexical RAG)
                                                             │ every tool sees data only up to the incident cycle
                                                             │ router: cheap model for steps, strong model for the report
                                                             │ guardrails: untrusted-data wrapping, injection redaction,
                                                             │            least-privilege tools, step/cost budget
```

A shift lead sees "Line 3 is drifting" and gets a report that says *what* (root cause with confidence), *why* (sensor evidence and the SOP section that describes these symptoms), *what to do* (the SOP's action), and *what it cost* (tokens and dollars per incident), with the work order parked for their approval. With the real models below a report takes about 75 seconds.

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
| Drift monitor (PSI, degraded vs healthy) | position error 0.40, coolant flow 0.19 (significant) |

### Agent: real models, same 10 cases for every system

Ten incidents on held-out machines (2 per failure mode), each investigated at its first anomaly cycle with data up to that cycle only. Models: gpt-oss-20b (tool steps) and gpt-oss-120b (report) on Groq. `artifacts/eval_results_env.json`, 14 Nov 2026.

| System | Root cause | SOP id match | Action keyword match | Clean / recovered / failed runs | Cost / incident | p50 latency | Steps |
|---|---|---|---|---|---|---|---|
| Rules (signature matching on the sensor summary, no LLM) | 6/10 | 6/10 | 6/10 | 10 / 0 / 0 | $0 | 0.01 s | 1 |
| LLM only (gpt-oss-120b, sensor summary, no tools) | 4/10 | **0/10** | 7/10 | 10 / 0 / 0 | $0.0003 | 1.8 s | 1 |
| **Agent (tools + SOP retrieval)** | **7/10** | **7/10** | 7/10 | 10 / 0 / 0 | $0.0019 | 75 s | 4.8 |

How to read this:
- **The agent's measured advantage over rules is small: 7 vs 6 correct on 10 cases, which is not a meaningful difference.** They also fail differently. Rules mistook both filter clogs for coolant leaks; the agent got both right. The agent called one coolant leak a filter clog. Both systems called both sensor faults "calibration drift": at the first anomaly a faulty sensor has produced only one or two spikes, which looks like slow drift.
- **What the agent adds is traceability, not accuracy.** The LLM without tools named a cause but never a real SOP (0/10; it invented ids like `SOP-FT-001`). Every agent report cites a real SOP section and the sensor window it used, so a reviewer can check it. Whether that saves an engineer time is not measured here.
- **Early diagnosis is genuinely hard.** These cases are deliberately taken at the first anomaly, often dozens of cycles before the pattern is obvious.
- No real-model injection test yet: the three planted attacks are on training machines, and this run used held-out machines only.

### Offline checks (mock model, all 30 cases, `artifacts/eval_results_mock.json`)

The mock model is a rule-based stand-in used for tests and CI. Its scores check that the pipeline works, not that a model reasons well: rules 16/30, mock agent 18/30. Of the 3 cases with a planted injection note, the mock agent diagnosed 2 correctly, and in all 3 the note was redacted before the model saw it. Its single miss is a diagnosis error, not the attack working.

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

## Why this design

| Decision | Reason |
|---|---|
| **Synthetic data with labelled failure modes** (C-MAPSS-style run-to-failure, 60 units, 12 sensors) | Real datasets have degradation but no ground-truth failure mode, SOP or action, so an agent can't be scored honestly. The generator keeps the C-MAPSS structure and adds five failure modes whose sensor signatures match the SOP corpus. |
| **Anomaly ensemble: robust z-score of rolling means + IsolationForest**, trained on healthy cycles only | Drift shows in trends, not single readings; IsolationForest alone missed half the units. Threshold = 99.5th percentile of healthy scores. |
| **Split by machine, time order kept, incident cutoff** | A new machine and future cycles are what deployment looks like. Random k-fold would leak, and so would letting an investigation read data from after the alarm. |
| **Hybrid lexical retrieval (BM25 + TF-IDF) over SOP sections, with hand-written reranking boosts** | Section-level chunks give precise citations ("SOP-C-201 §2 Diagnosis"). Lexical search runs anywhere; no embeddings or learned reranker yet, and the `Index` interface accepts an embedding backend. |
| **Native tool calling, report submitted as a tool** | Provider-enforced structured output: no JSON-mode truncation, every report validated by pydantic before anyone sees it. |
| **Model router: cheap model for tool choice, strong model for the report** | Tool choice is short and structured; synthesis with citations is where quality matters. Every call is logged with tokens and cost. |
| **Guardrails as code, not prompt text** | Tool permissions live in the registry (`open_ticket` is a write tool and always goes to the approval gate). Ticket notes and documents are wrapped as `<untrusted_data>` and scanned; instruction-like text is redacted and logged. Step and cost budgets are checked after each call, so they bound a run approximately rather than as a hard cap. |
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

MCP: `make mcp` exposes the same five tools over stdio for Claude Desktop / Claude Code (config snippet in `factory_copilot/mcp_server.py`).

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
  api.py       FastAPI: /incidents /investigate /approvals /metrics
  mcp_server.py
dashboard/app.py    Streamlit
evaluation/         cases.py (30 incidents with ground truth) · run.py (rules / llm_only / agent, results table)
tests/              43 tests (data, analytics, incident cutoff, retrieval, guardrails, tools, agent, scoring, API)
docs/DESIGN.md      the design document this was built from
```

## What I'd do next

- Embedding retrieval + reranker model behind the same `Index` interface, evaluated against the lexical baseline.
- Cross-incident memory (what was approved last time on this unit) with the same untrusted-data treatment.
- Real connectors (OPC-UA / historian) behind `query_sensors`; the tool schema would not change.
- Separate sensor faults from calibration drift earlier (spike timing, not just counts): both systems miss this at first detection.
- A real-model injection test on held-out machines, and a larger frozen case set that was not used during development.
- Measure what matters to a shift lead: time to a verified diagnosis with and without the copilot, against the rules workflow.
- A proper LLM-as-judge for report *quality* (evidence sufficiency), not just root-cause/SOP/action accuracy.

## Author

Muskaan Mahadik · M.S. Intelligent Manufacturing Technology (Data Science track), Taiwan Tech · thesis on quantization and machine unlearning in LLMs.
