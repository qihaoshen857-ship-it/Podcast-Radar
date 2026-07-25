#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
TOOLS_DIR="$ROOT_DIR/.tools"
RUNTIME_DIR="$ROOT_DIR/.runtime"
MAMBA_BIN="$TOOLS_DIR/bin/micromamba"

export PATH="$RUNTIME_DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS only."
  exit 1
fi

has_full_runtime() {
  [[ -x "$RUNTIME_DIR/bin/python" ]] || return 1
  "$RUNTIME_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info < (3, 10):
    raise SystemExit(1)
import tkinter
PY
  [[ -x "$RUNTIME_DIR/bin/ffmpeg" ]] || return 1
  [[ -x "$RUNTIME_DIR/bin/ffprobe" ]] || return 1
  [[ -x "$RUNTIME_DIR/bin/deno" ]] || return 1
}

install_micromamba() {
  if [[ -x "$MAMBA_BIN" ]]; then
    return 0
  fi

  echo "Installing local micromamba runtime manager..."
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  mkdir -p "$TOOLS_DIR"
  curl -L --fail --retry 3 \
    "https://micro.mamba.pm/api/micromamba/osx-arm64/latest" \
    -o "$tmp_dir/micromamba.tar.bz2"
  tar -xjf "$tmp_dir/micromamba.tar.bz2" -C "$tmp_dir"
  mkdir -p "$TOOLS_DIR/bin"
  cp "$tmp_dir/bin/micromamba" "$MAMBA_BIN"
  chmod +x "$MAMBA_BIN"
  rm -rf "$tmp_dir"
}

if ! has_full_runtime; then
  install_micromamba
  echo "Creating local full runtime with Python 3.13, Tk, ffmpeg, and Deno..."
  "$MAMBA_BIN" create -y -p "$RUNTIME_DIR" -c conda-forge \
    python=3.13 tk ffmpeg deno pip
fi

export PATH="$RUNTIME_DIR/bin:$PATH"

"$RUNTIME_DIR/bin/python" - <<'PY'
import sys
import tkinter
print(f"Using local Python {sys.version.split()[0]} at {sys.executable}")
PY

for tool in ffmpeg ffprobe deno; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required runtime tool: $tool"
    exit 1
  fi
  echo "OK: $tool -> $(command -v "$tool")"
done

"$RUNTIME_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$RUNTIME_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

"$RUNTIME_DIR/bin/python" - <<'PY'
modules = [
    "tkinter",
    "requests",
    "yt_dlp",
    "dashscope",
    "dotenv",
    "librosa",
    "soundfile",
    "silero_vad",
    "numpy",
    "websocket",
    "imageio_ffmpeg",
]
for module in modules:
    __import__(module)
print("OK: Python dependencies are importable.")
PY

echo
echo "Setup completed with full functionality."
echo "Run launch_macos.command to start the app, or run_macos.command for setup+launch."
