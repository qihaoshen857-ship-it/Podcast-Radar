#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
RELEASE_DIR="$ROOT_DIR/release"
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_PATH="$RELEASE_DIR/podcast-transcriber-mac_$STAMP.zip"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$RELEASE_DIR"
mkdir -p "$TMP_DIR/podcast-transcriber-mac"

rsync -a "$ROOT_DIR/" "$TMP_DIR/podcast-transcriber-mac/" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".runtime/" \
  --exclude ".env" \
  --exclude "PRIVATE_DASHSCOPE_API_KEY.md" \
  --exclude "settings.json" \
  --exclude ".DS_Store" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/" \
  --exclude "*.bat" \
  --exclude "*.ps1" \
  --exclude ".tools/" \
  --exclude "downloads/" \
  --exclude "downloads_smoke/" \
  --exclude "release/" \
  --exclude "auto_edge_cookies.txt" \
  --exclude "auto_edge_cookies_test.txt" \
  --exclude "auto_cdp_cookies.txt" \
  --exclude "edge_profile_test/"

(cd "$TMP_DIR" && zip -qr "$ZIP_PATH" "podcast-transcriber-mac")

echo "Created: $ZIP_PATH"
