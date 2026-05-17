"""
Watches ~/.clifflabs-chat/chats/*.json and syncs new/changed chat sessions
into the second brain via POST /ingest/text.

Tracks which chats have been indexed (and at what size) in a state file so
repeat runs only re-send changed chats.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [clifflabs-chat-sync] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8011"
CHAT_DIR = Path(os.getenv("CLIFFLABS_CHAT_DIR", Path.home() / ".clifflabs-chat" / "chats"))
STATE_FILE = Path(
    os.getenv(
        "STATE_FILE",
        Path.home() / ".clifflabs-chat" / "second_brain_sync.json",
    )
)
MIN_BYTES = 100
DEBOUNCE = 10


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _format_ts(ts: object) -> str | None:
    if not isinstance(ts, (int, float)):
        return None
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return None


def extract_text(chat_path: Path) -> str:
    """Convert a chat JSON file into a readable transcript."""
    try:
        data = json.loads(chat_path.read_text(errors="replace"))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", chat_path.name, exc)
        return ""

    title = str(data.get("title") or chat_path.stem).strip()
    model = str(data.get("model") or "").strip()
    system = str(data.get("system") or "").strip()
    messages = data.get("messages") or []

    lines: list[str] = [f"Title: {title}"]
    if model:
        lines.append(f"Model: {model}")
    if system:
        lines.append(f"System: {system}")
    lines.append("")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        stamp = _format_ts(msg.get("ts"))
        prefix = {"user": "User", "assistant": "Assistant", "system": "System"}[role]
        if stamp:
            lines.append(f"{prefix} [{stamp}]: {text}")
        else:
            lines.append(f"{prefix}: {text}")
        lines.append("")

    return "\n".join(lines).strip()


def session_label(chat_path: Path) -> str:
    """Generate a human-readable source label."""
    try:
        data = json.loads(chat_path.read_text(errors="replace"))
    except Exception:
        data = {}
    title = str(data.get("title") or chat_path.stem).strip() or chat_path.stem
    updated = data.get("updated")
    if isinstance(updated, (int, float)):
        dt = datetime.fromtimestamp(float(updated), tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(chat_path.stat().st_mtime, tz=timezone.utc)
    return f"clifflabs-chat/{title} ({chat_path.stem[:8]}, {dt.strftime('%Y-%m-%d')})"


def sync_chat(chat_path: Path, state: dict) -> bool:
    """Sync a single chat file. Returns True if indexed."""
    path_key = str(chat_path)
    current_size = chat_path.stat().st_size
    if current_size < MIN_BYTES:
        return False

    last_size = state.get(path_key, {}).get("size", 0)
    if current_size == last_size:
        return False

    text = extract_text(chat_path)
    if not text.strip():
        return False

    source = session_label(chat_path)
    try:
        resp = requests.post(
            f"{API_BASE}/ingest/text",
            json={
                "text": text,
                "source": source,
                "tags": ["clifflabs-chat", "chat-transcript"],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        state[path_key] = {
            "size": current_size,
            "chunks": data.get("chunks_indexed", 0),
            "source": source,
        }
        save_state(state)
        logger.info("Synced %s -> %s chunks", source, data.get("chunks_indexed", 0))
        return True
    except Exception as exc:
        logger.warning("Failed to sync %s: %s", source, exc)
        return False


def sync_all(*, quiet_when_unchanged: bool = False) -> None:
    if not CHAT_DIR.exists():
        logger.warning("Chat directory does not exist: %s", CHAT_DIR)
        return

    state = load_state()
    total = 0
    for chat_path in CHAT_DIR.rglob("*.json"):
        if sync_chat(chat_path, state):
            total += 1
    if total:
        logger.info("Initial sync: %s chats indexed", total)
    elif not quiet_when_unchanged:
        logger.info("Initial sync: all chats up to date")


def watch_polling() -> None:
    logger.info("watchdog not available - using polling mode")
    try:
        while True:
            sync_all(quiet_when_unchanged=True)
            time.sleep(DEBOUNCE)
    except KeyboardInterrupt:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Clifflabs chats into second brain")
    parser.add_argument("--once", action="store_true", help="Run one sync pass and exit")
    args = parser.parse_args()

    logger.info("Starting - watching %s", CHAT_DIR)
    logger.info("Second brain: %s", API_BASE)

    for _ in range(30):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=3)
            if resp.json().get("agent"):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        logger.error("Second brain not reachable - exiting")
        sys.exit(1)

    sync_all()

    if args.once:
        return

    if not CHAT_DIR.exists():
        return

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        watch_polling()
        return

    class ChatHandler:
        def __init__(self) -> None:
            self._pending: dict[str, float] = {}

        def on_modified(self, event) -> None:
            if not event.is_directory and event.src_path.endswith(".json"):
                self._pending[event.src_path] = time.time()

        on_created = on_modified

        def flush_pending(self) -> None:
            now = time.time()
            ready = [p for p, t in self._pending.items() if now - t >= DEBOUNCE]
            if not ready:
                return
            state = load_state()
            for path_str in ready:
                del self._pending[path_str]
                chat_path = Path(path_str)
                if chat_path.exists():
                    sync_chat(chat_path, state)

    class WatchChatHandler(FileSystemEventHandler, ChatHandler):
        pass

    handler = WatchChatHandler()
    observer = Observer()
    observer.schedule(handler, str(CHAT_DIR), recursive=True)
    observer.start()
    logger.info("Watching for new chat sessions...")

    try:
        while True:
            handler.flush_pending()
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
