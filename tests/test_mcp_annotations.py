"""Every MCP tool declares all four tool annotations with the values that match what the handler does."""

import asyncio

from factory_copilot.mcp_server import mcp

HINTS = ("read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint")


def _tools():
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_every_tool_declares_all_four_hints():
    tools = _tools()
    assert len(tools) == 6
    for name, tool in tools.items():
        assert tool.annotations is not None, name
        for hint in HINTS:
            assert isinstance(getattr(tool.annotations, hint), bool), f"{name} is missing {hint}"


def test_hint_values_match_behaviour():
    tools = _tools()
    for name in ("query_sensors", "get_maintenance_history", "search_sops", "estimate_rul", "list_pending_approvals"):
        a = tools[name].annotations
        assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (True, False, True, False), name
    a = tools["open_ticket"].annotations
    # Queues a new PENDING request each call; nothing is executed or overwritten.
    assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (False, False, False, False)
