# Live run of 13 Nov 2026: full agent reports behind `eval_results_env.json`

The 10 agent rows in `../eval_results_env.json` (gpt-oss-20b / gpt-oss-120b on Groq, run_at 2026-11-13T16:22:39Z)
summarise these investigations. Each file in `traces/` is the complete record the agent persisted: the report, every
tool call with its full output (sensor window, RUL estimate, maintenance tickets, retrieved SOP passages), and every
model call with tokens and cost.

`index.json` links each case to its trace. Traces were saved by that run under the gitignored `artifacts/traces/`
and matched afterwards by unit, recorded cost and root cause; all 10 match one-to-one. The files are unmodified.

Provenance: the run came from the uncommitted repair state on top of `b221be0`, committed afterwards as `8912458`
(the result file says `b221be0` because the runner then recorded HEAD only).

`citation_check` in the index was computed offline afterwards with `factory_copilot.citations.check_report_citations`.
In all 10 reports every cited SOP and ticket id appears in that investigation's tool outputs. That shows the ids
were retrieved, **not** that the passages support the conclusion: C05 (a coolant leak diagnosed as a filter clog)
cites a retrieved SOP-C-202 just as confidently. Semantic review of the procedures has not been done.

Limits: model prompts are stored as their last 1,500 characters and model outputs truncated to 2,000.

Two to look at: C13 (`INC-b03e1df2`, correct coolant leak) and C05 (`INC-dd6fef54`, wrong).
