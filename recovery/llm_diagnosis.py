"""An LLM reads one recovery decision through the MCP server and returns a
validated diagnosis.

The causes and interventions are the repo's own vocabulary: core.friction.Driver
and recovery.playbook.Kind. Anything outside them fails validation.
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ALLOWED_TOOLS = {"explain_decision_tool"}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = """You diagnose why one checkout session was abandoned and what to do about it.
Always call explain_decision_tool with the session id first, and base your answer only on what it returns.

cause_of_abandonment, one of:
- price: the basket cost more than the shopper would pay
- hesitation: the shopper stalled in one stage
- churn: the shopper bounced back and forth between stages
- intent: the shopper never meant to buy
- unknown: the evidence does not say

recommended_intervention, one of: discount, reminder, simplify, none.
A shopper with cause 'intent' cannot be recovered, so the intervention must be 'none'.
confidence is between 0 and 1. rationale is one or two sentences citing the evidence."""

Cause = Literal["price", "hesitation", "churn", "intent", "unknown"]
Intervention = Literal["discount", "reminder", "simplify", "none"]


class SessionDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_of_abandonment: Cause
    recommended_intervention: Intervention
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def no_offer_without_intent(self):
        if self.cause_of_abandonment == "intent" and self.recommended_intervention != "none":
            raise ValueError("a shopper who never intended to buy cannot be recovered; "
                             "recommended_intervention must be 'none'")
        return self


FALLBACK = SessionDiagnosis(
    cause_of_abandonment="unknown",
    recommended_intervention="none",
    confidence=0.0,
    rationale="fallback: no validated model output, so no action is recommended",
)

SCHEMA = SessionDiagnosis.model_json_schema()


def _result(diagnosis, status, attempts, error, transcript):
    return {"diagnosis": diagnosis.model_dump(), "status": status,
            "attempts": attempts, "error": error, "transcript": transcript}


def _short(err):
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in err.errors())


async def diagnose_session(session_id, chat, call_tool, tools, retries=1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Diagnose checkout session {session_id}."},
    ]
    reply = chat(messages, tools=tools)
    messages.append(reply)

    evidence = None
    for call in reply.get("tool_calls") or []:
        name = call["function"]["name"]
        args = call["function"].get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        if name in ALLOWED_TOOLS:
            output = await call_tool(name, args)
            evidence = output
        else:
            output = {"error": f"tool {name} is not available"}
        messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output)})

    if evidence is None:
        return _result(FALLBACK, "fallback", 0, "model did not call an allowed tool", messages)
    if not evidence.get("found"):
        return _result(FALLBACK, "fallback", 0, "session not found in the audit trail", messages)

    error = None
    for attempt in range(1, retries + 2):
        reply = chat(messages, fmt=SCHEMA)
        messages.append(reply)
        try:
            diagnosis = SessionDiagnosis.model_validate_json(reply.get("content") or "")
            return _result(diagnosis, "ok" if attempt == 1 else "retried", attempt, None, messages)
        except ValidationError as e:
            error = _short(e)
            messages.append({"role": "user", "content":
                             f"That answer failed validation: {error}. "
                             "Reply again with only the corrected JSON object."})
    return _result(FALLBACK, "fallback", retries + 1, error, messages)


def ollama_chat(model=DEFAULT_MODEL, url=OLLAMA_URL, timeout=120):
    def chat(messages, tools=None, fmt=None):
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": 0}}
        if tools:
            body["tools"] = tools
        if fmt:
            body["format"] = fmt
        r = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        message = r.json()["message"]
        return {k: message[k] for k in ("role", "content", "tool_calls") if message.get(k)}
    return chat


def to_ollama_tools(mcp_tools):
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.inputSchema}}
            for t in mcp_tools if t.name in ALLOWED_TOOLS]


async def run(session_id, db_path, chat):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "recovery.mcp_server"], cwd=root,
        env={**os.environ, "FF_AUDIT_DB": os.path.abspath(db_path)})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_ollama_tools((await session.list_tools()).tools)

            async def call_tool(name, args):
                result = await session.call_tool(name, args)
                if result.isError:
                    return {"error": result.content[0].text if result.content else "tool error"}
                return result.structuredContent or json.loads(result.content[0].text)

            return await diagnose_session(session_id, chat, call_tool, tools)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_id")
    p.add_argument("--db", default=os.environ.get("FF_AUDIT_DB", "recovery_audit.db"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--transcript", action="store_true")
    args = p.parse_args()
    out = asyncio.run(run(args.session_id, args.db, ollama_chat(args.model)))
    if not args.transcript:
        out.pop("transcript")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
