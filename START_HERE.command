#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
APP_BUNDLE_NAME="Podcast Radar.app"
APP_EXECUTABLE_NAME="PodcastRadar"

latest_app="$(
  find "$ROOT_DIR/release" -maxdepth 2 -type d -name "$APP_BUNDLE_NAME" -print0 2>/dev/null \
    | while IFS= read -r -d '' candidate; do
        if [[ -x "$candidate/Contents/MacOS/$APP_EXECUTABLE_NAME" ]]; then
          stat -f "%m %N" "$candidate"
        fi
      done \
    | sort -nr \
    | head -n 1 \
    | sed 's/^[0-9][0-9]* //'
)"

if [[ -z "${latest_app:-}" ]]; then
  echo "No packaged app found. Starting from source instead."
  exec "$ROOT_DIR/run_macos.command"
fi

pkill -f "$ROOT_DIR/release/.*/Podcast Radar.app/Contents/MacOS/PodcastRadar" >/dev/null 2>&1 || true
sleep 1

echo "Opening latest app:"
echo "$latest_app"
open "$latest_app"
