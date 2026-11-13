# Superseded results (commit b221be0, 12 Nov 2026)

These are the original evaluation files, kept unchanged for the record. Do not quote them as current results.

An audit on 13 Nov 2026 found that, in this version, the agent's tools could read data from after the incident:
the sensor window and RUL estimate used the machine's whole run, and the maintenance history included the ticket
written at the machine's final failure, which states the root cause and repair. The healthy baseline was also
computed from all machines, including held-out ones. The README also compared rules on 30 cases with the agent
on 10; on the same 10 cases the rules baseline scored 10/10 too, and case C05 counted as correct despite a
provider error during the run.

Current results, with an incident-time cutoff and matched cases, are in `../eval_results_*.json` and the README.
