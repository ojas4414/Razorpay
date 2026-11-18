"""The LLM glue, with the model mocked so every path is deterministic.

Unit tests replace both the model and the MCP call. The last test goes through
the real MCP server over stdio with a scripted model, so the transport and the
tool-result unpacking are exercised without needing Ollama.
"""
import asyncio
import json

import pytest

from recovery.audit import AuditLog
from recovery.llm_diagnosis import FALLBACK, diagnose_session, run, to_ollama_tools

TOOLS = [{"type": "function", "function": {"name": "explain_decision_tool"}}]
FOUND = {"found": True, "session_id": "t0", "quit_driver": "price",
         "why_flagged": {"trigger": "price_shock"}}


def tool_call(name="explain_decision_tool", session_id="t0"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": {"session_id": session_id}}}]}


def answer(**overrides):
    body = {"cause_of_abandonment": "price", "recommended_intervention": "discount",
            "confidence": 0.8, "rationale": "price_shock at cart"}
    body.update(overrides)
    return {"role": "assistant", "content": json.dumps(body)}


class ScriptedChat:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tools=None, fmt=None):
        self.calls.append({"messages": list(messages), "tools": tools, "fmt": fmt})
        return self.replies.pop(0)


class FakeTool:
    def __init__(self, output=FOUND):
        self.output = output
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        return self.output


def diagnose(chat, tool=None):
    return asyncio.run(diagnose_session("t0", chat, tool or FakeTool(), TOOLS))


def test_valid_output_on_first_try():
    chat, tool = ScriptedChat(tool_call(), answer()), FakeTool()
    out = diagnose(chat, tool)
    assert out["status"] == "ok" and out["attempts"] == 1 and out["error"] is None
    assert out["diagnosis"]["cause_of_abandonment"] == "price"
    assert tool.calls == [("explain_decision_tool", {"session_id": "t0"})]
    assert chat.calls[0]["tools"] == TOOLS and chat.calls[1]["fmt"] is not None


def test_tool_output_is_sent_back_to_the_model():
    chat = ScriptedChat(tool_call(), answer())
    diagnose(chat)
    tool_msg = chat.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"]) == FOUND


@pytest.mark.parametrize("bad", [
    {"role": "assistant", "content": "the cause is price"},
    answer(confidence=1.7),
    answer(cause_of_abandonment="weather"),
    answer(cause_of_abandonment="intent", recommended_intervention="discount"),
])
def test_invalid_then_valid_retries_once(bad):
    chat = ScriptedChat(tool_call(), bad, answer())
    out = diagnose(chat)
    assert out["status"] == "retried" and out["attempts"] == 2
    feedback = chat.calls[2]["messages"][-1]
    assert feedback["role"] == "user" and "failed validation" in feedback["content"]


def test_invalid_twice_falls_back_safely():
    chat = ScriptedChat(tool_call(), answer(confidence=2), answer(confidence=3))
    out = diagnose(chat)
    assert out["status"] == "fallback" and out["attempts"] == 2
    assert out["diagnosis"] == FALLBACK.model_dump()
    assert out["diagnosis"]["recommended_intervention"] == "none"
    assert "confidence" in out["error"]
    assert len(chat.calls) == 3


def test_no_tool_call_falls_back_without_asking_for_json():
    chat = ScriptedChat(answer())
    out = diagnose(chat)
    assert out["status"] == "fallback" and out["error"] == "model did not call an allowed tool"
    assert len(chat.calls) == 1


def test_tool_outside_allowlist_is_not_executed():
    chat, tool = ScriptedChat(tool_call(name="run_counterfactual_tool")), FakeTool()
    out = diagnose(chat, tool)
    assert tool.calls == []
    assert out["status"] == "fallback"


def test_unknown_session_falls_back_without_guessing():
    chat = ScriptedChat(tool_call())
    out = diagnose(chat, FakeTool({"found": False, "session_id": "t0"}))
    assert out["status"] == "fallback" and "not found" in out["error"]
    assert len(chat.calls) == 1


def test_string_arguments_are_parsed():
    call = tool_call()
    call["tool_calls"][0]["function"]["arguments"] = '{"session_id": "t0"}'
    tool = FakeTool()
    diagnose(ScriptedChat(call, answer()), tool)
    assert tool.calls == [("explain_decision_tool", {"session_id": "t0"})]


def test_only_allowed_mcp_tools_are_offered():
    class T:
        def __init__(self, name):
            self.name, self.description, self.inputSchema = name, "d", {"type": "object"}
    names = [t["function"]["name"] for t in to_ollama_tools([T("explain_decision_tool"), T("run_counterfactual_tool")])]
    assert names == ["explain_decision_tool"]


def test_end_to_end_through_real_mcp_server(tmp_path):
    pytest.importorskip("mcp")
    from test_mcp_server import make_session

    db = str(tmp_path / "audit.db")
    audit = AuditLog(db)
    audit.record_many([make_session("t0", "treatment", treated=True)], "ab-test-001", seed=42)
    audit.close()

    chat = ScriptedChat(tool_call(), answer())
    out = asyncio.run(run("t0", db, chat))
    assert out["status"] == "ok"
    tool_output = json.loads(chat.calls[1]["messages"][-1]["content"])
    assert tool_output["found"] is True and tool_output["session_id"] == "t0"
    assert tool_output["why_flagged"]["trigger"] == "price_shock"
    assert [t["function"]["name"] for t in chat.calls[0]["tools"]] == ["explain_decision_tool"]
