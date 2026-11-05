# DLS-2026

Everything needed to talk to the **DLS-2026 class inference endpoint** — an
OpenAI-compatible chat endpoint serving **Qwen3.8-27B** with a full 262K context
window, reasoning, and tool calling. Load-balanced across 8× A100 GPUs
(LiteLLM aggregator → 8× SGLang). Reachable over the **public internet** — no
VPN, no Tailscale, no special network required.

> ⚠️ The endpoint is plain **HTTP** and is secured only by a shared bearer key.
> Treat the key as a **shared class secret** — do not post it publicly, and do
> not make this repo public.

## What's here

| Path | What it is |
|------|-----------|
| [`piago-dls-2026/`](piago-dls-2026/) | The `piago-dls-2026` launcher for the [pi](https://pi.dev) coding agent, plus its installer and `models.json`. |
| [`examples/`](examples/) | Standalone curl / Python examples that hit the raw endpoint (no pi required). |
| [`Lecture-topics/`](Lecture-topics/) | Deep-dive topic kits (Transformer, Pretraining & the MAI case study, Scaling Laws, Post-training, Sparsity & Precision, LLM Runtime) — slides, notes, reading lists, and papers. |
| [`Lab1/`](Lab1/) | Lab 1: coding-agent environments and controlled agent comparison. |
| [`Lab2/`](Lab2/) | Lab 2: experiments on microgpt with a coding agent. |
| [`Lab3/`](Lab3/) | Lab 3: sparse and reduced-precision MLP/CNN/GNN experiments with Pi. |

## Quick start (pi coding agent)

The `piago-dls-2026` wrapper launches the pi coding agent pre-wired to the DLS
endpoint. It's modeled on the internal `piago` launcher but points at the
**public IP** and defaults to the class model.

```bash
git clone git@github.com:rick-stevens-ai/DLS-2026.git
cd DLS-2026/piago-dls-2026
./install.sh
```

Then:

```bash
piago-dls-2026                                   # interactive TUI
piago-dls-2026 "explain a hash table"            # interactive w/ initial prompt
piago-dls-2026 -p "explain this function"        # non-interactive (print & exit)
```

The installer:
1. Installs the `pi` CLI if it's missing (via `npm install -g @earendil-works/pi-coding-agent`).
2. Copies `piago-dls-2026` into `~/.local/bin`.
3. Merges the `dls-2026` provider into `~/.pi/agent/models.json` (non-destructive — it won't touch your other providers).

### Requirements
- **Node.js 22+** and **npm** (for the `pi` CLI). Get it from <https://nodejs.org> or `nvm`.
- **bash**, **curl**, and internet access.

## Connection details

| Field | Value |
|-------|-------|
| Base URL | `http://209.182.219.81:4010/v1` |
| Model name | `qwen3.8-27b` (exposed to pi as `dls-2026/qwen3.8-27b`) |
| API key | `DLS-2026` |
| Protocol | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) |
| Auth header | `Authorization: Bearer DLS-2026` |
| Health check | `curl http://209.182.219.81:4010/health/liveliness` |

### Environment overrides (wrapper)

| Variable | Default | Meaning |
|----------|---------|---------|
| `DLS_2026_API_KEY` | `DLS-2026` | Bearer key for the endpoint |
| `DLS_2026_HOST` | `209.182.219.81` | Endpoint host (public IP) |
| `DLS_2026_PORT` | `4010` | Endpoint port |
| `PIAGO_DLS_NO_PROBE` | `0` | Set to `1` to skip the pre-flight health check |

## Capabilities

- **Context window:** 262,144 tokens
- **Reasoning:** enabled — chain-of-thought is returned in a separate
  `reasoning_content` field, so `content` holds only the clean final answer.
- **Tool calling:** standard OpenAI `tools` / `tool_calls`.
- **Streaming:** supported (`"stream": true`).

## Two things to know

1. **It's a reasoning model.** It spends tokens *thinking* before answering.
   Give a generous `max_tokens` (≥600 for anything non-trivial) — otherwise the
   whole budget can be consumed by reasoning and `content` comes back empty with
   `finish_reason: "length"`.
2. **Plain HTTP, shared key.** The endpoint is HTTP (not HTTPS) and secured only
   by the bearer key. Don't post the key publicly.

## Performance

- ~26 tokens/sec single-stream per request.
- Scales near-linearly with concurrency: ~205 tok/s at 8 parallel requests,
  ~717 tok/s at 32 parallel (8 GPU instances, round-robin).

## Talk to the endpoint directly (no pi)

See [`examples/`](examples/):
- [`examples/quick_test.sh`](examples/quick_test.sh) — curl smoke test.
- [`examples/example.py`](examples/example.py) — OpenAI Python SDK.

## Support

Contact **Rick Stevens** for access issues or key rotation.
