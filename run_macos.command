#!/usr/bin/env bash
cd "$(dirname "$0")"
./run_macos.sh
status=$?
echo
if [[ $status -ne 0 ]]; then
  echo "App exited with code $status."
  echo
  read -r -p "Press Return to close this window..."
fi
exit "$status"
