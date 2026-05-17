"""
Watches ~/.claude/projects/**/*.jsonl and syncs new/changed Claude Code
conversation sessions into the second brain via POST /ingest/text.

Tracks which sessions have been indexed (and at what size) in a state file
so only new messages are re-sent on the next sync.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [claude-sync] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_BASE   = "http://127.0.0.1:8011"
CLAUDE_DIR = Path(os.getenv("CLAUDE_DIR", Path.home() / ".claude" / "projects"))
STATE_FILE = Path(os.getenv("STATE_FILE", Path.home() / ".claude" / "second_brain_sync.json"))
MIN_BYTES  = 500   # ignore tiny/empty sessions
DEBOUNCE   = 10    # seconds to wait after last write before syncing


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def extract_text(jsonl_path: Path) -> str:
    """Convert a Claude JSONL session into a readable human/assistant transcript."""
    turns = []
    try:
        for line in jsonl_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", "").strip())
                        elif block.get("type") == "tool_result":
                            for sub in block.get("content", []):
                                if isinstance(sub, dict) and sub.get("type") == "text":
                                    t = sub.get("text", "").strip()
                                    if t:
                                        parts.append(f"[tool output]: {t[:500]}")
                text = "\n".join(p for p in parts if p)
            elif isinstance(content, str):
                text = content.strip()
            else:
                continue

            if text:
                turns.append(f"{'User' if role == 'user' else 'Claude'}: {text}")
    except Exception as e:
        logger.warning(f"Failed to parse {jsonl_path.name}: {e}")
        return ""

    return "\n\n".join(turns)


def session_label(jsonl_path: Path) -> str:
    """Generate a human-readable source label."""
    project = jsonl_path.parent.name.lstrip("-").replace("-", "/", 1)
    session_id = jsonl_path.stem[:8]
    mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime, tz=timezone.utc)
    return f"claude/{project}/{session_id} ({mtime.strftime('%Y-%m-%d')})"


def sync_session(jsonl_path: Path, state: dict) -> bool:
    """Sync a single session. Returns True if indexed."""
    path_key = str(jsonl_path)
    current_size = jsonl_path.stat().st_size

    if current_size < MIN_BYTES:
        return False

    last_size = state.get(path_key, {}).get("size", 0)
    if current_size == last_size:
        return False  # nothing new

    text = extract_text(jsonl_path)
    if not text.strip():
        return False

    source = session_label(jsonl_path)
    try:
        resp = requests.post(
            f"{API_BASE}/ingest/text",
            json={"text": text, "source": source, "tags": ["claude-chat"]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        state[path_key] = {"size": current_size, "chunks": data.get("chunks_indexed", 0), "source": source}
        save_state(state)
        logger.info(f"Synced {source} → {data.get('chunks_indexed')} chunks")
        return True
    except Exception as e:
        logger.warning(f"Failed to sync {source}: {e}")
        return False


def sync_all():
    state = load_state()
    total = 0
    for jsonl in CLAUDE_DIR.rglob("*.jsonl"):
        if sync_session(jsonl, state):
            total += 1
    if total:
        logger.info(f"Initial sync: {total} sessions indexed")
    else:
        logger.info("Initial sync: all sessions up to date")


class JSONLHandler(FileSystemEventHandler):
    def __init__(self):
        self._pending: dict[str, float] = {}

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            self._pending[event.src_path] = time.time()

    on_created = on_modified

    def flush_pending(self):
        now = time.time()
        ready = [p for p, t in self._pending.items() if now - t >= DEBOUNCE]
        if not ready:
            return
        state = load_state()
        for path_str in ready:
            del self._pending[path_str]
            p = Path(path_str)
            if p.exists():
                sync_session(p, state)


def main():
    logger.info(f"Starting — watching {CLAUDE_DIR}")
    logger.info(f"Second brain: {API_BASE}")

    # Wait for the API to be ready
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

    sync_all()

    handler = JSONLHandler()
    observer = Observer()
    observer.schedule(handler, str(CLAUDE_DIR), recursive=True)
    observer.start()
    logger.info("Watching for new Claude sessions...")

    try:
        while True:
            handler.flush_pending()
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
