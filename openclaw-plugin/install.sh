#!/bin/bash
# antaris-flame plugin installer
# Copies the plugin into OpenClaw's extensions directory and enables it.
# Run this during DMG first-launch setup.

set -e

PLUGIN_ID="antaris-flame"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_DIR="${OPENCLAW_DIR:-$HOME/.openclaw}"
EXT_DIR="$OPENCLAW_DIR/extensions/$PLUGIN_ID"

echo "[antaris-flame] Installing plugin..."

# Copy plugin files
mkdir -p "$EXT_DIR"
cp "$SCRIPT_DIR/index.js" "$EXT_DIR/"
cp "$SCRIPT_DIR/openclaw.plugin.json" "$EXT_DIR/"

echo "[antaris-flame] Plugin installed to $EXT_DIR"

# Patch openclaw.json — add to plugins.allow (pin trust/provenance)
CONFIG_FILE="$OPENCLAW_DIR/openclaw.json"
if [ -f "$CONFIG_FILE" ] && command -v node &>/dev/null; then
  node -e "
const fs = require('fs');
try {
  const cfg = JSON.parse(fs.readFileSync('$CONFIG_FILE', 'utf8'));
  if (!cfg.plugins) cfg.plugins = {};
  if (!Array.isArray(cfg.plugins.allow)) cfg.plugins.allow = [];
  if (!cfg.plugins.allow.includes('$PLUGIN_ID')) {
    cfg.plugins.allow.push('$PLUGIN_ID');
    fs.writeFileSync('$CONFIG_FILE', JSON.stringify(cfg, null, 2) + '\n');
    console.log('[$PLUGIN_ID] Added to plugins.allow');
  } else {
    console.log('[$PLUGIN_ID] Already in plugins.allow');
  }
} catch(e) { console.error('[$PLUGIN_ID] Could not patch openclaw.json:', e.message); }
"
fi

# Enable via openclaw CLI (if available)
if command -v openclaw &>/dev/null; then
  openclaw plugins enable "$PLUGIN_ID" 2>/dev/null || true
  echo "[antaris-flame] Plugin enabled. Restart OpenClaw to activate."
else
  echo "[antaris-flame] OpenClaw CLI not found. Enable plugin manually or via Gateway UI."
fi
