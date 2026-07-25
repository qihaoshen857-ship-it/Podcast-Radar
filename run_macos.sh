#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ".runtime/bin/python" ]]; then
  ./setup_macos.sh
fi

./launch_macos.sh
