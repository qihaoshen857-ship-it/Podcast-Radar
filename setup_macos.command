#!/usr/bin/env bash
cd "$(dirname "$0")"
./setup_macos.sh
status=$?
echo
if [[ $status -eq 0 ]]; then
  echo "Setup finished successfully."
else
  echo "Setup failed with exit code $status."
fi
echo
read -r -p "Press Return to close this window..."
exit "$status"
