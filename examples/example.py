#!/usr/bin/env python3
"""Minimal DLS-2026 endpoint example using the OpenAI Python SDK.

Install:  pip install openai
Run:      python example.py
Override: DLS_2026_API_KEY=your-key python example.py

The endpoint is OpenAI-compatible, so you can point any OpenAI client at it.
It's a reasoning model: chain-of-thought comes back in a separate
`reasoning_content` field and `content` holds the clean final answer. Give a
generous max_tokens (>=600) so reasoning doesn't eat the whole budget.
"""
import os

from openai import OpenAI

HOST = os.environ.get("DLS_2026_HOST", "209.182.219.81")
PORT = os.environ.get("DLS_2026_PORT", "4010")
KEY = os.environ.get("DLS_2026_API_KEY", "DLS-2026")

client = OpenAI(
    base_url=f"http://{HOST}:{PORT}/v1",
    api_key=KEY,
)

resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Explain a hash table in two sentences."}],
    max_tokens=600,
)

msg = resp.choices[0].message
print("reasoning:", getattr(msg, "reasoning_content", None))  # chain-of-thought (separate field)
print("answer:", msg.content)                                 # final answer
print("usage:", resp.usage)
