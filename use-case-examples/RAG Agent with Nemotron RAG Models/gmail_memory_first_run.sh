#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models"
PYTHON="/home/cgandycl/jupyterlab/.venv/bin/python"
CONFIG_DIR="${GMAIL_SYNC_DIR:-$HOME/.config/gmail_sync}"
CREDENTIALS_FILE="$CONFIG_DIR/credentials.json"
SECOND_BRAIN_URL="${SECOND_BRAIN_URL:-http://127.0.0.1:8011}"
LABEL="${1:-INBOX}"

mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"

if [[ ! -s "$CREDENTIALS_FILE" ]]; then
  cat <<EOF
Missing Gmail OAuth credentials:
  $CREDENTIALS_FILE

Create a Google Cloud Desktop OAuth client for the Gmail API, download its JSON,
and save it exactly as:
  $CREDENTIALS_FILE

Then rerun:
  $0 $LABEL
EOF
  exit 2
fi

curl -fsS "$SECOND_BRAIN_URL/health" >/dev/null

cd "$ROOT"
SECOND_BRAIN_URL="$SECOND_BRAIN_URL" GMAIL_SYNC_TARGET=memory "$PYTHON" gmail_sync.py --label "$LABEL"

systemctl --user daemon-reload
systemctl --user enable --now gmail-memory-sync.service
systemctl --user status --no-pager --lines=8 gmail-memory-sync.service || true
