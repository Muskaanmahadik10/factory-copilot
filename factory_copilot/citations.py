"""Citation IDs: exact parsing, and a check that a report only cites what its own investigation retrieved.

The check is about *existence in the retrieved evidence*, not about support: an SOP id that was returned by
`search_sops` during this investigation passes even if the passage does not justify the conclusion. Whether a cited
passage supports the recommendation still needs a human (or a separate, measured review).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Token boundaries on both sides: "SOP-C-201 §2" matches, "SOP-C-2010", "SOP-C-201FAKE", "XSOP-C-201" do not.
SOP_ID_RE = re.compile(r"(?<![A-Za-z0-9_-])SOP-[A-Z]-\d{3}(?![A-Za-z0-9_-])")
TICKET_ID_RE = re.compile(r"(?<![A-Za-z0-9_-])T\d{4}(?![A-Za-z0-9_-])")


def sop_ids(text: str | None) -> list[str]:
    return SOP_ID_RE.findall(text or "")


def first_sop_id(text: str | None) -> str:
    ids = sop_ids(text)
    return ids[0] if ids else ""


def ticket_ids(text: str | None) -> list[str]:
    return TICKET_ID_RE.findall(text or "")


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def retrieved_ids(tool_outputs: Iterable[tuple[str, Any]]) -> tuple[set[str], set[str]]:
    """SOP and ticket ids that one investigation actually retrieved, from typed record fields only:
    `search_sops` hits[].doc_id, `get_maintenance_history` tickets[].ticket_id, and `estimate_rul` policy_ref.
    Free text is not used: a search query, an error message, or another SOP's cross-reference ("see SOP-C-201")
    is not a retrieved record. Otherwise a model could make an invented id pass by searching for it."""
    sops, tickets = set(), set()
    for name, out in tool_outputs:
        if not isinstance(out, dict):
            continue
        if name == "search_sops":
            for h in out.get("hits") or []:
                sops.update(sop_ids(str(h.get("doc_id", ""))))
        elif name == "get_maintenance_history":
            for t in out.get("tickets") or []:
                tickets.update(ticket_ids(str(t.get("ticket_id", ""))))
        elif name == "estimate_rul":
            sops.update(sop_ids(str(out.get("policy_ref", ""))))
    return sops, tickets


def check_report_citations(report: dict, tool_outputs: Iterable[tuple[str, Any]]) -> dict:
    """Compare the ids a report mentions anywhere (summary, cause, action, evidence, SOP reference) with the ids its
    investigation actually retrieved.

    Returns the cited ids and any that were *not* in the retrieved records (invented, or from outside this
    investigation). `ok` is True only when every mentioned id was retrieved.
    """
    got_sops, got_tickets = retrieved_ids(tool_outputs)
    texts = list(_strings(report))
    cited_sops = sorted({i for t in texts for i in sop_ids(t)})
    cited_tickets = sorted({i for t in texts for i in ticket_ids(t)})
    unsupported = sorted(set(cited_sops) - got_sops) + sorted(set(cited_tickets) - got_tickets)
    return {"cited_sop_ids": cited_sops, "cited_ticket_ids": cited_tickets, "retrieved_sop_ids": sorted(got_sops),
            "unretrieved_citations": unsupported, "ok": not unsupported,
            "checks": "cited id appears in this investigation's tool outputs; does not check that the passage supports the conclusion"}
