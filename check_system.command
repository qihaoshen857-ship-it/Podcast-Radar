#!/usr/bin/env bash
cd "$(dirname "$0")"
./check_system.sh
status=$?
echo
if [[ $status -eq 0 ]]; then
  echo "System check passed."
else
  echo "System check failed with exit code $status."
fi
echo
read -r -p "Press Return to close this window..."
exit "$status"
