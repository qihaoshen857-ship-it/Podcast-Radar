#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"

export PATH="$RUNTIME_DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export TK_SILENCE_DEPRECATION=1

if [[ ! -x "$RUNTIME_DIR/bin/python" ]]; then
  echo "The local runtime is not initialized."
  echo "Run setup_macos.command first, or use run_macos.command."
  exit 1
fi

missing_tools=0
for tool in ffmpeg ffprobe deno; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "OK: $tool -> $(command -v "$tool")"
  else
    echo "MISSING: $tool"
    missing_tools=1
  fi
done

if [[ "$missing_tools" -ne 0 ]]; then
  echo
  echo "Full functionality requires ffmpeg, ffprobe, and deno."
  echo "Run setup_macos.command to rebuild the local runtime."
  exit 1
fi

"$RUNTIME_DIR/bin/python" "$ROOT_DIR/main.py"
