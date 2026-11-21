#!/bin/bash
# Extracts tutk_client.py from the pinned owlet-bridge-native image and applies
# patches/tutk_client.diff for the legacy 20-byte ThroughTek connect struct and talk-back fix.
set -euo pipefail

IMAGE="ghcr.io/btoth525/owlet-bridge-native@sha256:e127dbaa12fa65b98bc81b8b162cb1719691251d29aacaafd2764c22994553ee"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

id=$(docker create "$IMAGE")
trap 'docker rm "$id" >/dev/null 2>&1 || true' EXIT

docker cp "$id:/app/tutk_client.py" "$ROOT/patches/tutk_client.orig.py"
cp "$ROOT/patches/tutk_client.orig.py" "$ROOT/patches/tutk_client.py"
(cd "$ROOT/patches" && patch -p1 < tutk_client.diff)
echo "Wrote patched $ROOT/patches/tutk_client.py"
