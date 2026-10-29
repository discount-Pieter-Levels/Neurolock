#!/bin/bash
# Copy this QML app into Logos Basecamp (dev build) as "neurolock_demo".
set -e
cd "$(dirname "$0")"
P=~/.local/share/Logos/LogosBasecampDev/plugins/neurolock_demo
rm -rf "$P"; mkdir -p "$P"
cp -r Main.qml nl_data.js sha256.js img icon.png manifest.json "$P/"
cp manifest.json "$P/metadata.json"
printf linux-amd64-dev > "$P/variant"
echo "installed -> $P (restart Basecamp)"
