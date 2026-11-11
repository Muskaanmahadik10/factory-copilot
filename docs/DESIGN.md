# Factory Copilot: design document

An AI agent that watches production-line sensor data, detects anomalies, investigates them by calling tools (sensor history, maintenance logs, SOP documents, a remaining-useful-life model), and writes an evidence-backed incident report with a recommended action. A human approves any action before it is executed. Every LLM call is routed by cost, logged, and evaluated against scripted incidents with known root causes.

Owner: Muskaan Mahadik · Status: v1 plan, Nov 2026

---

## 1. Problem

Production lines generate continuous sensor streams. When a machine drifts, the people who notice (operators, shift leads) are not the people who know why (maintenance engineers, process owners), and the knowledge they need is scattered across the historian, the maintenance ticket system, and PDF manuals. Investigating a single alarm takes 20–60 minutes of a skilled engineer's time, and most alarms are repeats.

**Goal:** cut time-to-diagnosis for routine anomalies from tens of minutes to under a minute, without letting an AI take unsupervised action on the line.

## 2. Users

| User | Needs |
|---|---|
| Shift lead | "Something is wrong on Line 3. What is it and what do I do now?" |
| Maintenance engineer | Evidence: which sensors, since when, what the manual says, what happened last time. |
| AI/platform owner | Cost per incident, model usage, audit trail, a way to measure whether the agent is right. |

## 3. Scope

**In v1**
- Batch and streaming-style ingestion of multivariate sensor data (public dataset, see §5).
- Anomaly detection + remaining-useful-life (RUL) estimation with time-ordered validation and drift monitoring.
- A tool-using agent (plan → act → observe loop) with five tools, exposed both in-process and as an MCP server.
- Retrieval over SOP/manual documents (hybrid lexical retrieval: BM25 + TF-IDF cosine, with a reranking step).
- Cost-aware model routing: cheap model for classification/extraction steps, strong model for the final report. Every call logged with tokens and cost.
- Guardrails: least-privilege tools, approval gate before `open_ticket`, prompt-injection defence for untrusted text (maintenance notes, documents), schema-validated tool I/O.
- Evaluation harness: 30 scripted incidents with ground truth; scores root-cause accuracy, SOP citation accuracy, action correctness, cost, latency.
- Streamlit dashboard: live chart, anomaly flags, agent trace, report, approval button.
- Tests, CI, Docker, README with results.

**Out of v1:** real PLC/OPC-UA connectors, multi-line scheduling, fine-tuned models, embeddings/vector DB (optional upgrade; interface is ready for it).

## 4. Architecture

```
                     ┌──────────────────────────────────────────┐
  sensor stream ───► │ analytics: anomaly detector + RUL model  │──► anomaly event
  (C-MAPSS / AI4I)   └──────────────────────────────────────────┘        │
                                                                         ▼
  ┌───────────────────────────────── agent runtime ─────────────────────────────────┐
  │  planner/executor loop (ReAct)      model router (cheap ↔ strong)               │
  │  tools (least privilege):                                                        │
  │    query_sensors · get_maintenance_history · search_sops · estimate_rul          │
  │    open_ticket (needs human approval)                                            │
  │  guardrails: tool schemas · untrusted-text wrapping · injection detector · limits│
  │  trace + cost log (JSONL)                                                        │
  └──────────────────────────────────────────────────────────────────────────────────┘
          ▲                              │                                 ▲
     MCP server                      incident report                 evaluation harness
   (same tools, for                (markdown + JSON)                  (30 scripted cases)
    Claude Desktop/Code)                 │
                                         ▼
                               Streamlit dashboard + FastAPI
```

### Components

| Module | Responsibility |
|---|---|
| `factory_copilot/data/` | Dataset download/synthesis, maintenance-log and SOP corpus generation, schemas |
| `factory_copilot/analytics/` | `AnomalyDetector` (IsolationForest on rolling features), `RULModel` (gradient boosting on degradation features), `DriftMonitor` (PSI on feature distributions), time-ordered split |
| `factory_copilot/retrieval/` | `Chunker`, `HybridIndex` (BM25 + TF-IDF), `Reranker` (cross-feature score fusion), `search()` returning citations |
| `factory_copilot/tools/` | Tool registry; each tool = pydantic input/output schema + handler + permission level |
| `factory_copilot/agent/` | `Agent` loop, `ModelRouter`, prompts, trace logger, `IncidentReport` schema |
| `factory_copilot/guardrails/` | `wrap_untrusted()`, injection detector, approval gate, step/cost limits |
| `factory_copilot/llm/` | Provider clients (OpenAI-compatible: Groq/OpenAI/local; Anthropic) + `MockLLM` for tests and offline eval |
| `factory_copilot/mcp_server.py` | Exposes the tool registry over MCP (stdio) |
| `factory_copilot/api.py` | FastAPI: `/incidents`, `/investigate`, `/approve`, `/metrics` |
| `dashboard/app.py` | Streamlit UI |
| `evaluation/` | Scripted incidents, runner, scorer, results table |

## 5. Data

- **Sensor data:** NASA C-MAPSS FD001 (turbofan run-to-failure; 100 training engines, 21 sensors). If the download is unavailable, a synthetic generator with the same schema and injected failure modes is used, and the README says so.
- **Failure modes (synthetic layer):** `bearing_wear`, `coolant_leak`, `sensor_fault`, `filter_clog`, `calibration_drift`. Each has a signature (which sensors move, how fast), a matching SOP, and matching history entries, so ground truth exists for evaluation.
- **Maintenance log:** generated table (`ticket_id, unit, timestamp, symptom, root_cause, action, technician_note`). A few technician notes deliberately contain prompt-injection text to test guardrails.
- **SOP corpus:** 12 markdown/PDF documents (one per failure mode plus general procedures), written for this project.

## 6. Agent design

**Loop:** `plan` → for each step: choose tool (cheap model) → run tool → observe → until `finish` or step limit (8). Then `write_report` (strong model) with a strict JSON schema: `root_cause`, `confidence`, `evidence[]` (each with source tool + citation), `recommended_action`, `sop_reference`, `requires_approval`.

**Routing policy** (`ModelRouter`):
| Step | Model tier | Why |
|---|---|---|
| tool selection, extraction | cheap | short, structured |
| final report | strong | synthesis, cited reasoning |
| retry after validation failure | strong | correctness over cost |

**Memory:** the trace of the current incident only (no cross-incident memory in v1); long tool outputs are summarised before entering context.

## 7. Guardrails

1. **Least privilege:** each tool declares `permission: read | write`. `open_ticket` is `write` and always routes through the approval gate.
2. **Untrusted text:** maintenance notes and document chunks are wrapped in `<untrusted_data>` tags with an instruction that they are data, not commands; an injection detector flags phrases like "ignore previous instructions" and strips them from what the model sees, logging the event.
3. **Validation:** every tool input/output and the final report are pydantic-validated; a failed report triggers one strong-model retry, then a safe fallback report with `confidence: low`.
4. **Limits:** max 8 steps, max cost per incident (configurable, default $0.05), timeout.
5. **Audit:** full trace (prompts, tool calls, outputs, costs) persisted per incident.

## 8. Evaluation

- **Cases:** 30 scripted incidents (6 per failure mode) with ground truth: root cause, correct SOP id, correct action, and 5 cases with injected prompt attacks.
- **Metrics:** root-cause accuracy, SOP-citation accuracy, action correctness, injection resistance (did the attack change the action?), mean cost, p50/p95 latency, mean steps.
- **Baselines:** (a) rule-based (signature matching only), (b) single-prompt LLM with no tools, (c) full agent. The README shows the three side by side.
- **Analytics metrics:** anomaly precision/recall on held-out engines; RUL RMSE and the standard C-MAPSS score; drift PSI over time.

## 9. Milestones (vertical slices)

| # | Slice | Done when |
|---|---|---|
| 1 | Data + schemas | `make data` builds sensor, log, SOP files; tests pass |
| 2 | Analytics | anomaly + RUL trained with time-ordered split; metrics JSON written |
| 3 | Retrieval | `search_sops("coolant temperature rising")` returns the coolant SOP first |
| 4 | Tools + MCP | all 5 tools callable in-process and via MCP inspector |
| 5 | Agent + router | one scripted incident produces a valid report with MockLLM and with a real provider |
| 6 | Guardrails | injection case is logged and neutralised; approval gate blocks `open_ticket` |
| 7 | Evaluation | 30 cases run; results table generated |
| 8 | API + dashboard | investigate → approve flow works end to end |
| 9 | Packaging | tests, CI, Docker, README with GIF and results |

## 10. Non-goals and honesty notes

- The LLM reasoning quality depends on the provider; offline runs use `MockLLM` (deterministic, rule-backed) so tests and CI never need a key. The README reports results for both and labels which is which.
- Lexical retrieval is used by default to keep the project runnable on any laptop; the `Index` interface allows an embedding backend.
