# Token Factory capability probe — 2026-10-19

Command: `uv run python scripts/probe_tf.py` (prompt: one billing ticket; tools: lookup_order). Raw output:

```
{"tag": "lightning/plain", "ms": 3483, "content": "Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - User provides a ticket description: \"I was charged twice for invoice 991.\"\n   - User asks to \"Class", "think_tag": false, "reasoning_content": false, "tool_calls": [], "usage": [33, 400], "finish": "length"}
{"tag": "lightning/json_schema", "ms": 1780, "content": "", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 400], "finish": "length"}
{"tag": "lightning/json_object", "ms": 1545, "content": "{\"label\": \"billing\"}", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [47, 349], "finish": "stop"}
{"tag": "lightning/tools", "ms": 550, "content": "", "think_tag": false, "reasoning_content": false, "tool_calls": ["lookup_order{\"order_id\":\"A-1042\"}"], "usage": [406, 31], "finish": "tool_calls"}
{"tag": "lightning/think_off", "ms": 698, "content": "**Classification: Billing Error / Duplicate Charge**\n\n**Reasoning:**\nThe user is reporting a specific financial discrepancy (being charged twice for the same in", "think_tag": false, "reasoning_content": false, "tool_calls": [], "usage": [33, 60], "finish": "stop"}
{"tag": "lightning/think_on", "ms": 2081, "content": "**Classification:** `Billing Issue - Duplicate Charge`\n\nThis ticket falls under billing/payment errors, specifically a duplicate charge for invoice #991. Typica", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 464], "finish": "stop"}
{"tag": "lightning/effort_low", "ms": 1904, "content": "Classification: **Billing/Payment \u2013 Duplicate Charge**", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 450], "finish": "stop"}
{"tag": "super/plain", "ms": 1540, "content": "\n\n**Classification:**\u202fBilling \u2013 Duplicate/Over\u2011Charge (the customer reports being charged twice for invoice\u202f991).", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 148], "finish": "stop"}
{"tag": "super/json_schema", "ms": 1615, "content": "{\"label\": \"billing\"}", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 144], "finish": "stop"}
{"tag": "super/json_object", "ms": 982, "content": "{\"label\": \"billing\"}", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [47, 54], "finish": "stop"}
{"tag": "super/tools", "ms": 1216, "content": "", "think_tag": false, "reasoning_content": true, "tool_calls": ["lookup_order{\"order_id\": \"A-1042\"}"], "usage": [359, 49], "finish": "tool_calls"}
{"tag": "super/think_off", "ms": 2751, "content": "The ticket \"I was charged twice for invoice 991.\" is a clear report of a **duplicate charge** on a specific invoice.\n\n### Classification:\n- **Category**: Billin", "think_tag": false, "reasoning_content": false, "tool_calls": [], "usage": [33, 381], "finish": "stop"}
{"tag": "super/think_on", "ms": 1878, "content": "\n\n**Classification:** Billing\u202f\u2013\u202fDuplicate Charge (the customer reports being charged twice for invoice\u202f991).", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 161], "finish": "stop"}
{"tag": "super/effort_low", "ms": 898, "content": "\nCategory: Billing \u2013 Duplicate Charge (Invoice\u202f991)", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [40, 39], "finish": "stop"}
{"tag": "ultra/plain", "ms": 1658, "content": "**Classification: Billing Issue \u2192 Duplicate Charge**\n\n**Reasoning:** The customer explicitly states they were charged two times for a single invoice (Invoice #9", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 170], "finish": "stop"}
{"tag": "ultra/json_schema", "ms": 1216, "content": "{\n  \"label\": \"billing\"\n}", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 120], "finish": "stop"}
{"tag": "ultra/json_object", "ms": 1260, "content": "{\"label\": \"billing\"}", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [47, 42], "finish": "stop"}
{"tag": "ultra/tools", "ms": 922, "content": "", "think_tag": false, "reasoning_content": true, "tool_calls": ["lookup_order{\"order_id\": \"A-1042\"}"], "usage": [312, 80], "finish": "tool_calls"}
{"tag": "ultra/think_off", "ms": 730, "content": "**Category: Billing / Duplicate Charge**", "think_tag": false, "reasoning_content": false, "tool_calls": [], "usage": [33, 10], "finish": "stop"}
{"tag": "ultra/think_on", "ms": 1368, "content": "**Classification:**\n*   **Category:** Billing & Payments\n*   **Sub-category:** Duplicate Charge / Overbilling\n*   **Sentiment:** Negative / Urgent\n*   **Priorit", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 221], "finish": "stop"}
{"tag": "ultra/effort_low", "ms": 1353, "content": "**Classification:**\n*   **Category:** Billing & Payments\n*   **Sub-category:** Duplicate Charge / Overbilling\n*   **Sentiment:** Negative / Urgent\n*   **Priorit", "think_tag": false, "reasoning_content": true, "tool_calls": [], "usage": [33, 221], "finish": "stop"}
```

Findings:
- Model ids confirmed via GET /v1/models: nvidia/Nemotron-3_5-Lightning, nvidia/nemotron-3-super-120b-a12b, nvidia/Nemotron-3-Ultra-550b-a55b (also nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B, nvidia/Nemotron-3.5-Super-VL).
- `response_format` json_schema and json_object work on all three tiers; tool calling works on all three.
- Reasoning is ON by default. `extra_body={"chat_template_kwargs":{"enable_thinking": false}}` turns it off (no reasoning_content, short output). Reasoning tokens count toward max_tokens.
- Lightning + json_schema + max_tokens=400 with thinking on: content empty, finish=length (reasoning consumed the budget).
- Lightning plain call with thinking on: reasoning prose leaked into `content` ("Here is a thinking process: ...") instead of `reasoning_content`.
- `reasoning_effort="low"` did not shorten Ultra output (same as thinking on).
