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

# Enable via openclaw CLI (if available)
if command -v openclaw &>/dev/null; then
  openclaw plugins enable "$PLUGIN_ID" 2>/dev/null || true
  echo "[antaris-flame] Plugin enabled. Restart OpenClaw to activate."
else
  echo "[antaris-flame] OpenClaw CLI not found. Enable plugin manually or via Gateway UI."
fi
