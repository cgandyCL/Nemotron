"""
Syncs Gmail messages into Mnemosyne.

Default behavior sends messages into the Memory Wrapper via POST /memory/capture
so you can review and approve them before they enter RAG. Use --direct-rag if
you intentionally want the old direct /ingest/text behavior.

First run: opens browser for OAuth2 consent and caches the token.
Subsequent runs: uses cached token; only fetches messages newer than last sync.

Usage:
    python3 gmail_sync.py                         # one-shot sync, INBOX
    python3 gmail_sync.py --watch                 # poll every 5 minutes
    python3 gmail_sync.py --label INBOX --label Sent  # specific labels
    python3 gmail_sync.py --full                  # re-index everything

Setup (one-time):
    1. Go to https://console.cloud.google.com/
    2. Create a project → Enable "Gmail API"
    3. OAuth consent screen → External → add your Gmail as test user
    4. Credentials → Create OAuth client ID → Desktop app
    5. Download JSON → save as ~/.config/gmail_sync/credentials.json
    6. Run this script; a browser window will open for consent
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gmail-sync] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCOPES           = ["https://www.googleapis.com/auth/gmail.readonly"]
CONFIG_DIR       = Path(os.getenv("GMAIL_SYNC_DIR", Path.home() / ".config" / "gmail_sync"))
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE       = CONFIG_DIR / "token.json"
STATE_FILE       = CONFIG_DIR / "state.json"
API_BASE         = os.getenv("SECOND_BRAIN_URL", "http://127.0.0.1:8011")
SYNC_TARGET      = os.getenv("GMAIL_SYNC_TARGET", "memory").strip().lower()
MEMORY_PROCESS_NOW = os.getenv("GMAIL_MEMORY_PROCESS_NOW", "1").lower() not in {"0", "false", "no"}
POLL_INTERVAL    = int(os.getenv("GMAIL_POLL_SECONDS", "300"))  # 5 min
MAX_PER_LABEL    = int(os.getenv("GMAIL_MAX_MESSAGES", "500"))  # per sync run


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_credentials() -> Credentials:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CREDENTIALS_FILE.exists():
        logger.error(f"credentials.json not found at {CREDENTIALS_FILE}")
        logger.error("See the setup instructions at the top of this file.")
        sys.exit(1)

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return creds


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Email parsing ─────────────────────────────────────────────────────────────

def _decode_part(payload: dict) -> str:
    data = payload.get("body", {}).get("data", "")
    if not data:
        return ""
    return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")


def _extract_body(payload: dict) -> str:
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        return _decode_part(payload)

    if mime == "text/html":
        html = _decode_part(payload)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    if mime.startswith("multipart/"):
        parts = payload.get("parts", [])
        # prefer plain text
        for part in parts:
            if part.get("mimeType") == "text/plain":
                text = _decode_part(part)
                if text.strip():
                    return text
        # fall back to HTML
        for part in parts:
            if part.get("mimeType") == "text/html":
                html = _decode_part(part)
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text.strip():
                    return text
        # recurse into nested multiparts
        for part in parts:
            text = _extract_body(part)
            if text.strip():
                return text

    return ""


def message_to_text(msg: dict) -> tuple[str, dict]:
    """Return (full_text, header_dict) for a Gmail API message object."""
    payload  = msg.get("payload", {})
    headers  = {h["name"]: h["value"] for h in payload.get("headers", [])}
    subject  = headers.get("Subject", "(no subject)")
    sender   = headers.get("From", "")
    to       = headers.get("To", "")
    date     = headers.get("Date", "")
    msg_id   = headers.get("Message-ID", msg.get("id", ""))

    body = _extract_body(payload).strip()

    lines = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {to}",
        f"Date: {date}",
        "",
        body,
    ]
    return "\n".join(lines), {"subject": subject, "from": sender, "date": date, "message_id": msg_id}


# ── Sync logic ────────────────────────────────────────────────────────────────

def capture_memory(text: str, source: str, tags: list[str], headers: dict) -> bool:
    subject = headers.get("subject") or source
    sender = headers.get("from") or ""
    try:
        resp = requests.post(
            f"{API_BASE}/memory/capture",
            json={
                "content": text,
                "source": source,
                "title": subject,
                "source_kind": "email",
                "device": "gmail-sync",
                "project": "email",
                "tags": tags,
                "namespace": "memory",
                "process_now": MEMORY_PROCESS_NOW,
                "policy": {
                    "add_to_rag": False,
                    "summarize": True,
                    "extract_tasks": True,
                    "create_note": True,
                    "memory_level": "raw",
                    "sensitivity": "private",
                    "allowed_use": ["summarize", "answer", "extract_tasks"],
                    "require_review": True,
                    "include_raw_text": True,
                },
            },
            timeout=90,
        )
        resp.raise_for_status()
        item_id = resp.json().get("id", "")
        logger.info(f"  captured {source} → memory item {item_id} from {sender}")
        return True
    except Exception as e:
        logger.warning(f"  failed to capture {source}: {e}")
        return False


def ingest_direct(text: str, source: str, tags: list[str]) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/ingest/text",
            json={"text": text, "source": source, "tags": tags},
            timeout=60,
        )
        resp.raise_for_status()
        chunks = resp.json().get("chunks_indexed", 0)
        logger.info(f"  indexed {source} → {chunks} chunks")
        return True
    except Exception as e:
        logger.warning(f"  failed to ingest {source}: {e}")
        return False


def ingest(text: str, source: str, tags: list[str], headers: dict, target: str) -> bool:
    if target == "direct-rag":
        return ingest_direct(text, source, tags)
    return capture_memory(text, source, tags, headers)


def sync_label(service, label: str, state: dict, full: bool = False, target: str = "memory") -> int:
    label_key = f"label:{label}"
    history_id = None if full else state.get(label_key, {}).get("history_id")

    kwargs: dict = {
        "userId": "me",
        "labelIds": [label],
        "maxResults": 100,
    }

    # Use history API for incremental sync when we have a prior historyId
    message_ids: list[str] = []

    if history_id and not full:
        try:
            history_resp = service.users().history().list(
                userId="me",
                startHistoryId=history_id,
                labelId=label,
                historyTypes=["messageAdded"],
                maxResults=500,
            ).execute()
            for record in history_resp.get("history", []):
                for added in record.get("messagesAdded", []):
                    message_ids.append(added["message"]["id"])
            logger.info(f"{label}: {len(message_ids)} new messages since last sync")
        except HttpError as e:
            if e.resp.status == 404:
                # historyId expired — fall back to full label list
                logger.info(f"{label}: history expired, falling back to full list")
                history_id = None
                message_ids = []
            else:
                raise

    if not message_ids and not history_id:
        # Full label list
        page_token = None
        while len(message_ids) < MAX_PER_LABEL:
            resp = service.users().messages().list(
                userId="me", labelIds=[label], maxResults=100,
                pageToken=page_token,
            ).execute()
            for m in resp.get("messages", []):
                message_ids.append(m["id"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        logger.info(f"{label}: fetching {len(message_ids)} messages")

    if not message_ids:
        return 0

    # Filter already-ingested ids
    ingested_ids: set = set(state.get(label_key, {}).get("ingested_ids", []))
    to_fetch = [mid for mid in message_ids if mid not in ingested_ids]
    logger.info(f"{label}: {len(to_fetch)} to ingest ({len(message_ids)-len(to_fetch)} already done)")

    count = 0
    new_history_id = state.get(label_key, {}).get("history_id")

    for mid in to_fetch:
        try:
            full_msg = service.users().messages().get(
                userId="me", id=mid, format="full",
            ).execute()
        except Exception as e:
            logger.warning(f"  failed to fetch {mid}: {e}")
            continue

        text, headers = message_to_text(full_msg)
        if not text.strip():
            ingested_ids.add(mid)
            continue

        # Track the highest historyId seen
        msg_hist = full_msg.get("historyId", "0")
        if new_history_id is None or int(msg_hist) > int(new_history_id):
            new_history_id = msg_hist

        date_str = headers.get("date", "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        source = f"gmail/{label}/{mid} ({date_str})"
        tags = ["gmail", label.lower().replace(" ", "-")]

        if ingest(text, source, tags, headers, target):
            count += 1

        ingested_ids.add(mid)
        # Save progress every 10 messages
        if count % 10 == 0:
            state[label_key] = {
                "history_id": new_history_id,
                "ingested_ids": list(ingested_ids),
            }
            save_state(state)

    state[label_key] = {
        "history_id": new_history_id,
        "ingested_ids": list(ingested_ids),
    }
    save_state(state)
    return count


def sync_all(service, labels: list[str], state: dict, full: bool = False, target: str = "memory") -> int:
    total = 0
    for label in labels:
        try:
            total += sync_label(service, label, state, full=full, target=target)
        except Exception as e:
            logger.error(f"Error syncing label {label}: {e}")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync Gmail into second brain")
    parser.add_argument("--label", action="append", default=[], dest="labels",
                        help="Gmail label to sync (default: INBOX). Repeatable.")
    parser.add_argument("--watch", action="store_true",
                        help=f"Poll every {POLL_INTERVAL}s instead of one-shot")
    parser.add_argument("--full", action="store_true",
                        help="Re-ingest all messages, ignoring sync state")
    parser.add_argument("--direct-rag", action="store_true",
                        help="Send mail directly to /ingest/text instead of the Memory Wrapper")
    parser.add_argument("--memory", action="store_true",
                        help="Send mail to /memory/capture for review before RAG (default)")
    args = parser.parse_args()

    labels = args.labels or ["INBOX"]
    target = "direct-rag" if args.direct_rag or SYNC_TARGET == "direct-rag" else "memory"

    logger.info(f"Second brain: {API_BASE}")
    logger.info(f"Labels: {labels}")
    logger.info(f"Target: {target}")

    # Wait for second brain to be ready
    for _ in range(30):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.json().get("agent"):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        logger.error("Second brain not reachable — exiting")
        sys.exit(1)

    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)

    profile = service.users().getProfile(userId="me").execute()
    logger.info(f"Authenticated as: {profile.get('emailAddress')}")

    state = load_state()

    if args.watch:
        logger.info(f"Watch mode: syncing every {POLL_INTERVAL}s")
        while True:
            n = sync_all(service, labels, state, full=args.full, target=target)
            if n:
                logger.info(f"Synced {n} new messages")
            else:
                logger.info("No new messages")
            args.full = False  # only full on first pass
            time.sleep(POLL_INTERVAL)
    else:
        n = sync_all(service, labels, state, full=args.full, target=target)
        logger.info(f"Done — {n} messages ingested")


if __name__ == "__main__":
    main()
