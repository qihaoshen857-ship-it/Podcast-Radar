#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"

if [[ ! -x "$RUNTIME_DIR/bin/python" ]]; then
  echo "Local runtime is missing. Run setup_macos.command first."
  exit 1
fi

export PATH="$RUNTIME_DIR/bin:$PATH"
"$RUNTIME_DIR/bin/python" "$ROOT_DIR/system_check.py"
