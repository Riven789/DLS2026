#!/usr/bin/env python3
"""Add/update a Pi provider from DLS_QWEN_* environment variables.
Creates a timestamped backup and never prints the key.
"""
import json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
key = os.environ.get("DLS_QWEN_API_KEY")
base = os.environ.get("DLS_QWEN_BASE_URL", "http://209.182.219.81:4010/v1")
model = os.environ.get("DLS_QWEN_MODEL", "qwen3.8-27b")
if not key:
    sys.exit("DLS_QWEN_API_KEY is not set")
p = Path.home()/".pi"/"agent"/"models.json"
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists():
    shutil.copy2(p, p.with_suffix(f".json.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"))
    data = json.loads(p.read_text())
else:
    data = {}
providers = data.setdefault("providers", {})
providers["dls-qwen"] = {
    "name": "CMSC 35200 Chicago-6 Qwen3.8-27B",
    "api": "openai-completions",
    "authHeader": True,
    "apiKey": key,
    "baseUrl": base,
    "models": [{"id": model, "name": model, "input": ["text"],
                "contextWindow": 262144, "maxTokens": 32768, "reasoning": True}],
}
p.write_text(json.dumps(data, indent=2) + "\n")
print(f"Configured provider dls-qwen in {p}; key not displayed.")
