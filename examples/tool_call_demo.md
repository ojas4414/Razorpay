# Tool-call demo: llama3.2:3b diagnoses one session through the MCP server

One real run, 2026-11-19, on this laptop (Ollama `llama3.2:3b`, temperature 0, running locally; nothing sent to an external API).
Everything below is copied programmatically from the run's raw JSON output; nothing was edited.

## How to reproduce

```
# 1. build an audit trail: the repo's own A/B simulation at its minimum size (2,000 sessions; took 18 s)
python -c "from recovery.mcp_server import run_counterfactual; run_counterfactual(sessions=2000, db_path='demo_audit.db')"
# 2. ask the model about one session (took 1 min 42 s, including loading the model)
python -m recovery.llm_diagnosis s0000042 --db demo_audit.db --model llama3.2:3b --transcript
```

Session `s0000042` was chosen by a fixed rule, not by looking at results: the first abandoned
treatment-arm row in the audit table (`ORDER BY row_id`).

## Result

```json
{
  "diagnosis": {
    "cause_of_abandonment": "hesitation",
    "recommended_intervention": "reminder",
    "confidence": 0.9,
    "rationale": "The shopper stalled in the browse stage for 4.202 seconds, indicating hesitation. Applying a reminder at this stage may help re-engage the shopper."
  },
  "status": "ok",
  "attempts": 1,
  "error": null
}
```

- `status: ok` and `attempts: 1`: the first answer passed Pydantic validation; the retry and fallback paths were not needed on this run (they are covered by `tests/test_llm_diagnosis.py`).
- The rationale's figures (stage `browse`, dwell `4.202`) match the tool output below.
- **What this does and does not show.** It shows the plumbing working end to end: the model chose to call the MCP tool, the real MCP server answered over stdio, and the final JSON validated. It is **not** evidence of diagnostic skill: the tool output already contains the playbook's `diagnosis` (`hesitation`) and the simulator's ground-truth `quit_driver` (`hesitation`), so the model could copy the answer. `confidence` is the model's own number, not a calibrated probability. One run, one session.

## Raw input

### Tools offered to the model (from the MCP server's `list_tools`, filtered to the read-only allowlist)

```json
[
  {
    "type": "function",
    "function": {
      "name": "explain_decision_tool",
      "description": "Explain why one shopping session was flagged as at risk, what offer it received, and how it ended.",
      "parameters": {
        "properties": {
          "session_id": {
            "title": "Session Id",
            "type": "string"
          },
          "experiment_id": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "title": "Experiment Id"
          }
        },
        "required": [
          "session_id"
        ],
        "title": "explain_decision_toolArguments",
        "type": "object"
      }
    }
  }
]
```

### Messages sent on the first call

```json
[
  {
    "role": "system",
    "content": "You diagnose why one checkout session was abandoned and what to do about it.\nAlways call explain_decision_tool with the session id first, and base your answer only on what it returns.\n\ncause_of_abandonment, one of:\n- price: the basket cost more than the shopper would pay\n- hesitation: the shopper stalled in one stage\n- churn: the shopper bounced back and forth between stages\n- intent: the shopper never meant to buy\n- unknown: the evidence does not say\n\nrecommended_intervention, one of: discount, reminder, simplify, none.\nA shopper with cause 'intent' cannot be recovered, so the intervention must be 'none'.\nconfidence is between 0 and 1. rationale is one or two sentences citing the evidence."
  },
  {
    "role": "user",
    "content": "Diagnose checkout session s0000042."
  }
]
```

### JSON schema sent as Ollama `format` on the final call

```json
{
  "additionalProperties": false,
  "properties": {
    "cause_of_abandonment": {
      "enum": [
        "price",
        "hesitation",
        "churn",
        "intent",
        "unknown"
      ],
      "title": "Cause Of Abandonment",
      "type": "string"
    },
    "recommended_intervention": {
      "enum": [
        "discount",
        "reminder",
        "simplify",
        "none"
      ],
      "title": "Recommended Intervention",
      "type": "string"
    },
    "confidence": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Confidence",
      "type": "number"
    },
    "rationale": {
      "maxLength": 400,
      "minLength": 1,
      "title": "Rationale",
      "type": "string"
    }
  },
  "required": [
    "cause_of_abandonment",
    "recommended_intervention",
    "confidence",
    "rationale"
  ],
  "title": "SessionDiagnosis",
  "type": "object"
}
```

## Raw output: full transcript after the system and user messages

```json
[
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_yhzos87i",
        "function": {
          "index": 0,
          "name": "explain_decision_tool",
          "arguments": {
            "session_id": "s0000042"
          }
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_name": "explain_decision_tool",
    "content": "{\"found\": true, \"session_id\": \"s0000042\", \"experiment_id\": \"ab-n2000-d0.15-l0.5-s42\", \"persona\": \"power_buyer\", \"arm\": \"treatment\", \"flagged_at_sim\": 8.909028150564284, \"why_flagged\": {\"trigger\": \"revisit\", \"stage\": \"browse\", \"evidence\": {\"actions\": 6, \"cart_size\": 1, \"cart_value\": 102.0, \"dwell\": 4.202, \"elapsed\": 8.364, \"gap_zscore\": 1.739, \"persona\": \"power_buyer\", \"price_shock\": false, \"revisits\": 1}}, \"offer\": \"15% discount, applied at t=9.4s\", \"diagnosis\": \"hesitation\", \"attempts\": [], \"escalated\": false, \"dispatched_at_sim\": 9.409028150564284, \"applied_at_sim\": 9.409028150564284, \"ended_at_sim\": 20.983576180466518, \"idempotency_key\": \"s0000042:discount\", \"outcome\": \"abandoned\", \"reason_code\": \"quit\", \"quit_driver\": \"hesitation\", \"was_recoverable\": false, \"cart_value\": 102.0, \"discount_amount\": 15.3, \"realized_value\": 0.0, \"detector_version\": \"stall+hesitation+price_shock/v2\"}"
  },
  {
    "role": "assistant",
    "content": "{\n  \"cause_of_abandonment\": \"hesitation\",\n  \"recommended_intervention\": \"reminder\",\n  \"confidence\": 0.9,\n  \"rationale\": \"The shopper stalled in the browse stage for 4.202 seconds, indicating hesitation. Applying a reminder at this stage may help re-engage the shopper.\"\n}"
  }
]
```

