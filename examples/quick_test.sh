#!/usr/bin/env bash
# quick_test.sh — smoke test the DLS-2026 endpoint with plain curl (no pi needed).
#
# Usage:
#   ./quick_test.sh
#   DLS_2026_API_KEY=your-key ./quick_test.sh
set -euo pipefail

HOST="${DLS_2026_HOST:-209.182.219.81}"
PORT="${DLS_2026_PORT:-4010}"
KEY="${DLS_2026_API_KEY:-DLS-2026}"
BASE="http://${HOST}:${PORT}"

echo "== liveliness =="
curl -s -m 10 "${BASE}/health/liveliness"; echo

echo "== models =="
curl -s -m 10 "${BASE}/v1/models" -H "Authorization: Bearer ${KEY}"; echo

echo "== chat completion =="
curl -s -m 120 "${BASE}/v1/chat/completions" \
  -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -d '{
     "model": "qwen3.8-27b",
     "messages": [{"role": "user", "content": "Explain a hash table in two sentences."}],
     "max_tokens": 600
  }'; echo
