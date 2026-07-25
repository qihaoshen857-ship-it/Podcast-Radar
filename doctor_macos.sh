#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
TOOLS_DIR="$ROOT_DIR/.tools"

export PATH="$RUNTIME_DIR/bin:$TOOLS_DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

check_command() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "OK   $name: $(command -v "$name")"
  else
    echo "MISS $name"
  fi
}

echo "Project: $ROOT_DIR"
echo

for candidate in "$RUNTIME_DIR/bin/python" python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
    "$candidate" - <<'PY'
import sys
try:
    import tkinter
    tk_status = "tkinter ok"
except Exception as exc:
    tk_status = f"tkinter missing: {exc}"
full = "full-ok" if sys.version_info >= (3, 10) else "compat-only"
print(f"PY   {sys.executable}: {sys.version.split()[0]} ({tk_status}, {full})")
PY
  fi
done

echo
if [[ -x "$RUNTIME_DIR/bin/python" ]]; then
  "$RUNTIME_DIR/bin/python" - <<'PY'
import importlib
import sys

print(f"RUNTIME {sys.executable}: {sys.version.split()[0]}")
for module in ["yt_dlp", "dashscope", "librosa", "soundfile", "silero_vad", "imageio_ffmpeg"]:
    try:
        importlib.import_module(module)
        print(f"OK   python module: {module}")
    except Exception as exc:
        print(f"MISS python module: {module} ({exc})")
PY
else
  echo "MISS .runtime/bin/python"
fi

echo
check_command micromamba
check_command brew
check_command ffmpeg
check_command ffprobe
check_command deno

echo
for app in "Google Chrome" "Microsoft Edge" "Firefox"; do
  if [[ -d "/Applications/$app.app" || -d "$HOME/Applications/$app.app" ]]; then
    echo "OK   browser: $app"
  else
    echo "MISS browser: $app"
  fi
done
