#!/usr/bin/env bash
# install.sh — install the `piago-dls-2026` wrapper so students can run the
# pi coding agent against the DLS-2026 class inference endpoint.
#
# What it does:
#   1. Checks for (and helps install) the `pi` CLI.
#   2. Installs `piago-dls-2026` into ~/.local/bin (on PATH for most shells).
#   3. Merges the dls-2026 provider into ~/.pi/agent/models.json.
#
# Usage:
#   ./install.sh                 # install
#   PREFIX=~/bin ./install.sh    # install the launcher somewhere else
#
# The wrapper itself is self-contained: it also (re)merges the provider config
# every time it runs, so this script mainly puts the launcher on your PATH.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/bin}"

echo "==> Installing piago-dls-2026"

# ---------------------------------------------------------------------------
# 1. Check for pi.
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$HOME/.local/node_modules/.bin:${PATH}"
if ! command -v pi >/dev/null 2>&1; then
  echo "    pi CLI not found."
  if command -v npm >/dev/null 2>&1; then
    echo "    Installing pi via npm (npm install -g @earendil-works/pi-coding-agent)..."
    npm install -g @earendil-works/pi-coding-agent
  else
    echo "    ERROR: npm not found. Install Node.js 22+ (https://nodejs.org) then run:" >&2
    echo "      npm install -g @earendil-works/pi-coding-agent" >&2
    echo "    and re-run ./install.sh" >&2
    exit 1
  fi
else
  echo "    Found pi: $(command -v pi)"
fi

# ---------------------------------------------------------------------------
# 2. Install the launcher.
# ---------------------------------------------------------------------------
mkdir -p "$PREFIX"
install -m 0755 "$HERE/piago-dls-2026" "$PREFIX/piago-dls-2026"
echo "    Installed launcher: $PREFIX/piago-dls-2026"

# ---------------------------------------------------------------------------
# 3. Merge provider config into ~/.pi/agent/models.json.
# ---------------------------------------------------------------------------
AGENT_DIR="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
MODELS_JSON="${AGENT_DIR}/models.json"
mkdir -p "$AGENT_DIR"

if command -v node >/dev/null 2>&1; then
  SRC_JSON="$HERE/models.json" node - "$MODELS_JSON" <<'NODE'
const fs = require("fs");
const path = process.argv[2];
const src = JSON.parse(fs.readFileSync(process.env.SRC_JSON, "utf8"));
let cfg = { providers: {} };
try {
  const raw = fs.readFileSync(path, "utf8").trim();
  if (raw) cfg = JSON.parse(raw);
} catch (_) {}
if (!cfg || typeof cfg !== "object") cfg = {};
if (!cfg.providers || typeof cfg.providers !== "object") cfg.providers = {};
cfg.providers["dls-2026"] = src.providers["dls-2026"];
fs.writeFileSync(path, JSON.stringify(cfg, null, 2) + "\n");
console.log("    Merged dls-2026 provider into " + path);
NODE
else
  if [ ! -f "$MODELS_JSON" ]; then
    cp "$HERE/models.json" "$MODELS_JSON"
    echo "    Wrote $MODELS_JSON"
  else
    echo "    node not found; not modifying existing $MODELS_JSON."
    echo "    Merge the dls-2026 provider from $HERE/models.json manually if needed."
  fi
fi

# ---------------------------------------------------------------------------
# PATH hint.
# ---------------------------------------------------------------------------
case ":$PATH:" in
  *":$PREFIX:"*) : ;;
  *)
    echo ""
    echo "    NOTE: $PREFIX is not on your PATH. Add this to your shell profile:"
    echo "      export PATH=\"$PREFIX:\$PATH\""
    ;;
esac

echo ""
echo "==> Done. Try it:"
echo "      piago-dls-2026 -p \"Explain a hash table in two sentences.\""
echo ""
echo "    The default class key (DLS-2026) is baked in. To override:"
echo "      DLS_2026_API_KEY=your-key piago-dls-2026"
