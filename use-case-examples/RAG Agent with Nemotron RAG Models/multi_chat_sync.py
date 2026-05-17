"""
Sync Claude Code, Codex, and Cursor chat transcripts into the NeMo RAG brain.

The script is intentionally source-focused:
- Claude Code: ~/.claude/projects/**/*.jsonl
- Codex: ~/.codex/sessions/**/*.jsonl plus history.jsonl
- Cursor: ~/.config/Cursor/User/globalStorage/state.vscdb composer records

It also scans the same paths under ~/mnt/node3 when mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

import requests


API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8011").rstrip("/")
HOME = Path.home()
STATE_FILE = Path(os.getenv("CHAT_SYNC_STATE", HOME / ".codex" / "memories" / "nemo_chat_sync_state.json"))
MIN_CHARS = int(os.getenv("CHAT_SYNC_MIN_CHARS", "80"))
MAX_PART_CHARS = int(os.getenv("CHAT_SYNC_MAX_PART_CHARS", "350000"))
MAX_TOOL_CHARS = int(os.getenv("CHAT_SYNC_MAX_TOOL_CHARS", "1500"))
INGEST_RETRIES = int(os.getenv("CHAT_SYNC_INGEST_RETRIES", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [chat-sync] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    provider: str
    source_id: str
    title: str
    text: str
    path: Optional[Path] = None
    mtime: float = 0.0
    size: int = 0


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def host_for(path: Path) -> str:
    value = str(path)
    if "/mnt/node3/" in value:
        return "node3"
    return "node1"


def utc_date_from_path(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def stable_hash(value: str, chars: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:chars]


def text_from_any(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [text_from_any(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            return text_from_any(value.get("text"))
        if block_type == "tool_result":
            text = text_from_any(value.get("content"))
            return f"[tool output]\n{text[:MAX_TOOL_CHARS]}" if text else ""
        if block_type in {"tool_use", "function_call"}:
            name = value.get("name") or value.get("tool_name") or value.get("call_id") or "tool"
            args = value.get("input") or value.get("arguments") or {}
            return f"[tool call] {name}: {json.dumps(args, ensure_ascii=False)[:MAX_TOOL_CHARS]}"
        for key in ("text", "content", "message", "output", "value"):
            if key in value:
                return text_from_any(value.get(key))
    return ""


def split_text(text: str, max_chars: int = MAX_PART_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    blocks = re.split(r"\n(?=## |### |User:|Claude:|Codex:|Cursor:|Assistant:)", text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        if not block:
            continue
        if current and current_len + len(block) + 2 > max_chars:
            parts.append("\n".join(current).strip())
            current = []
            current_len = 0
        if len(block) > max_chars:
            for offset in range(0, len(block), max_chars):
                if current:
                    parts.append("\n".join(current).strip())
                    current = []
                    current_len = 0
                parts.append(block[offset:offset + max_chars].strip())
            continue
        current.append(block)
        current_len += len(block) + 1
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]


def transcript_fingerprint(item: Transcript) -> str:
    if item.path:
        try:
            stat = item.path.stat()
            return f"{stat.st_size}:{int(stat.st_mtime)}:{stable_hash(str(item.path))}"
        except OSError:
            pass
    return stable_hash(item.text, chars=24)


def source_for_part(item: Transcript, part_index: int, total_parts: int) -> str:
    base = item.source_id
    if total_parts <= 1:
        return base
    return f"{base} part {part_index:03d} of {total_parts:03d}"


def claude_project_label(jsonl_path: Path) -> str:
    parts = list(jsonl_path.parts)
    try:
        idx = parts.index("projects")
        rel = Path(*parts[idx + 1:])
    except ValueError:
        rel = jsonl_path.name
    rel_text = str(rel).replace("\\", "/")
    rel_text = rel_text.replace("-home-cgandycl", "home-cgandycl")
    if rel_text.endswith(".jsonl"):
        rel_text = rel_text[:-6]
    return rel_text


def parse_claude_jsonl(path: Path) -> Optional[Transcript]:
    turns: list[str] = []
    session_id = path.stem
    cwd = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("Claude read failed %s: %s", path, exc)
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        cwd = entry.get("cwd") or cwd
        session_id = entry.get("sessionId") or session_id
        msg = entry.get("message") or {}
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = text_from_any(msg.get("content"))
        if not text:
            continue
        label = "User" if role == "user" else "Claude"
        turns.append(f"{label}: {text}")
    if not turns:
        return None
    host = host_for(path)
    date = utc_date_from_path(path)
    project = claude_project_label(path)
    header = [f"# Claude Code Session {session_id[:8]}", f"Host: {host}", f"Date: {date}"]
    if cwd:
        header.append(f"CWD: {cwd}")
    text = "\n".join(header) + "\n\n" + "\n\n".join(turns)
    return Transcript(
        provider="claude",
        source_id=f"claude-code/{host}/{project} ({date})",
        title=f"Claude {project}",
        text=text,
        path=path,
        mtime=path.stat().st_mtime,
        size=path.stat().st_size,
    )


def parse_codex_jsonl(path: Path) -> Optional[Transcript]:
    turns: list[str] = []
    session_id = path.stem
    cwd = ""
    started = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("Codex read failed %s: %s", path, exc)
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = entry.get("payload") or {}
        if entry.get("type") == "session_meta":
            session_id = payload.get("id") or session_id
            cwd = payload.get("cwd") or cwd
            started = payload.get("timestamp") or started
            continue
        if entry.get("type") == "turn_context":
            cwd = payload.get("cwd") or cwd
            continue
        if entry.get("type") != "response_item":
            continue
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = text_from_any(payload.get("content"))
            if text:
                label = "User" if role == "user" else "Codex"
                turns.append(f"{label}: {text}")
        elif kind in {"function_call", "tool_call"}:
            name = payload.get("name") or payload.get("tool_name") or payload.get("call_id") or "tool"
            args = payload.get("arguments") or payload.get("input") or ""
            turns.append(f"[Codex tool call] {name}: {text_from_any(args)[:MAX_TOOL_CHARS]}")
        elif kind in {"function_call_output", "tool_result"}:
            output = text_from_any(payload.get("output") or payload.get("content"))
            if output:
                turns.append(f"[Codex tool output]\n{output[:MAX_TOOL_CHARS]}")
    if not turns:
        return None
    host = host_for(path)
    date = started[:10] if started else utc_date_from_path(path)
    try:
        rel = path.relative_to(path.parents[4])
    except Exception:
        rel = path.name
    header = [f"# Codex Session {session_id[:8]}", f"Host: {host}", f"Date: {date}"]
    if cwd:
        header.append(f"CWD: {cwd}")
    text = "\n".join(header) + "\n\n" + "\n\n".join(turns)
    return Transcript(
        provider="codex",
        source_id=f"codex/{host}/{rel} ({date})",
        title=f"Codex {session_id[:8]}",
        text=text,
        path=path,
        mtime=path.stat().st_mtime,
        size=path.stat().st_size,
    )


def sqlite_rows(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            return list(con.execute(sql, params))
        finally:
            con.close()
    except sqlite3.Error as exc:
        logger.warning("Cursor DB read failed %s: %s", db_path, exc)
        return []


def cursor_headers(db_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = sqlite_rows(db_path, "select value from ItemTable where key = 'composer.composerHeaders'")
    if not rows:
        return out
    try:
        payload = json.loads(rows[0][0])
    except Exception:
        return out
    for item in payload.get("allComposers") or []:
        composer_id = item.get("composerId")
        if composer_id:
            out[composer_id] = item
    return out


def cursor_bubble_map(db_path: Path) -> dict[str, dict[str, Any]]:
    bubbles: dict[str, dict[str, Any]] = {}
    rows = sqlite_rows(db_path, "select key, value from cursorDiskKV where key like 'bubbleId:%'")
    for key, value in rows:
        try:
            payload = json.loads(value)
        except Exception:
            continue
        key_parts = str(key).split(":")
        if len(key_parts) >= 3:
            map_key = f"{key_parts[1]}:{key_parts[2]}"
            bubbles[map_key] = payload
        bubble_id = payload.get("bubbleId")
        if bubble_id:
            bubbles[str(bubble_id)] = payload
    return bubbles


def workspace_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    uri = value.get("uri") or {}
    path = uri.get("fsPath") or uri.get("path") or value.get("id") or ""
    return Path(str(path)).name if path else ""


def parse_cursor_db(db_path: Path) -> list[Transcript]:
    headers = cursor_headers(db_path)
    if not headers:
        return []
    bubbles = cursor_bubble_map(db_path)
    rows = sqlite_rows(db_path, "select key, value from cursorDiskKV where key like 'composerData:%'")
    transcripts: list[Transcript] = []
    seen: set[str] = set()
    for key, value in rows:
        try:
            composer = json.loads(value)
        except Exception:
            continue
        composer_id = composer.get("composerId") or str(key).split(":", 1)[-1]
        if not composer_id or composer_id in seen:
            continue
        seen.add(composer_id)
        header = headers.get(composer_id, {})
        title = composer.get("name") or header.get("name") or header.get("subtitle") or f"Cursor composer {composer_id[:8]}"
        created_ms = composer.get("createdAt") or header.get("createdAt") or composer.get("lastUpdatedAt") or header.get("lastUpdatedAt")
        date = ""
        if isinstance(created_ms, (int, float)) and created_ms:
            date = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            date = utc_date_from_path(db_path)
        host = host_for(db_path)
        ws = workspace_name(composer.get("workspaceIdentifier") or header.get("workspaceIdentifier")) or "global"
        turns: list[str] = []
        for item in composer.get("fullConversationHeadersOnly") or []:
            bubble_id = item.get("bubbleId")
            if not bubble_id:
                continue
            bubble = bubbles.get(f"{composer_id}:{bubble_id}") or bubbles.get(str(bubble_id))
            if not bubble:
                continue
            text = text_from_any(bubble.get("text"))
            if not text:
                continue
            bubble_type = bubble.get("type") or item.get("type")
            label = "User" if bubble_type == 1 else "Cursor"
            turns.append(f"{label}: {text}")
        if not turns:
            continue
        transcript = "\n".join([
            f"# Cursor Chat {title}",
            f"Host: {host}",
            f"Workspace: {ws}",
            f"Date: {date}",
            f"Composer ID: {composer_id}",
            "",
            "\n\n".join(turns),
        ])
        transcripts.append(Transcript(
            provider="cursor",
            source_id=f"cursor/{host}/{ws}/{composer_id[:8]} - {title} ({date})",
            title=f"Cursor {title}",
            text=transcript,
            path=db_path,
            mtime=db_path.stat().st_mtime,
            size=len(value),
        ))
    return transcripts


def candidate_roots() -> list[Path]:
    roots = [HOME]
    node3 = HOME / "mnt" / "node3"
    if node3.exists():
        roots.append(node3)
    return roots


def discover_transcripts() -> list[Transcript]:
    items: list[Transcript] = []
    for root in candidate_roots():
        claude_dir = root / ".claude" / "projects"
        if claude_dir.exists():
            for path in sorted(claude_dir.rglob("*.jsonl")):
                item = parse_claude_jsonl(path)
                if item and len(item.text) >= MIN_CHARS:
                    items.append(item)

        codex_dir = root / ".codex" / "sessions"
        if codex_dir.exists():
            for path in sorted(codex_dir.rglob("*.jsonl")):
                item = parse_codex_jsonl(path)
                if item and len(item.text) >= MIN_CHARS:
                    items.append(item)
        codex_history = root / ".codex" / "history.jsonl"
        if codex_history.exists():
            item = parse_codex_jsonl(codex_history)
            if item and len(item.text) >= MIN_CHARS:
                item.source_id = f"codex/{host_for(codex_history)}/history ({utc_date_from_path(codex_history)})"
                items.append(item)

        for cursor_base in (root / ".config" / "Cursor", root / ".config" / "cursor"):
            db_path = cursor_base / "User" / "globalStorage" / "state.vscdb"
            if db_path.exists():
                items.extend(parse_cursor_db(db_path))

    dedup: dict[str, Transcript] = {}
    for item in items:
        dedup[item.source_id] = item
    return sorted(dedup.values(), key=lambda item: (item.provider, item.source_id))


def api_delete_source(source: str) -> None:
    url = f"{API_BASE}/sources/{quote(source, safe='')}"
    try:
        requests.delete(url, timeout=30)
    except Exception:
        pass


def api_ingest(source: str, text: str, provider: str) -> int:
    last_error: Optional[Exception] = None
    for attempt in range(1, INGEST_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE}/ingest/text",
                json={
                    "text": text,
                    "source": source,
                    "tags": [provider, f"{provider}-chat", "chat", "auto-sync"],
                    "namespace": "ai-chats",
                    "semantic": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return int(resp.json().get("chunks_indexed", 0) or 0)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == INGEST_RETRIES:
                break
            time.sleep(min(30, attempt * 3))
    raise RuntimeError(f"ingest failed after {INGEST_RETRIES} attempts: {last_error}") from last_error


def sync_transcripts(force: bool = False, provider_filter: Optional[str] = None) -> dict[str, Any]:
    state = load_state()
    discovered = discover_transcripts()
    synced = 0
    skipped = 0
    chunks = 0
    errors: list[str] = []
    provider_counts: dict[str, int] = {}

    for item in discovered:
        provider_counts[item.provider] = provider_counts.get(item.provider, 0) + 1
        if provider_filter and item.provider != provider_filter:
            continue
        key = f"{item.provider}:{item.source_id}"
        fingerprint = transcript_fingerprint(item)
        prior = state.get(key, {})
        if not force and prior.get("fingerprint") == fingerprint and prior.get("sources"):
            skipped += 1
            continue
        old_sources = list(prior.get("sources") or [])
        for source in old_sources:
            api_delete_source(source)

        parts = split_text(item.text)
        new_sources: list[str] = []
        try:
            for idx, part in enumerate(parts, start=1):
                source = source_for_part(item, idx, len(parts))
                api_delete_source(source)
                chunks += api_ingest(source, part, item.provider)
                new_sources.append(source)
            state[key] = {
                "fingerprint": fingerprint,
                "provider": item.provider,
                "title": item.title,
                "sources": new_sources,
                "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            save_state(state)
            synced += 1
            logger.info("Synced %s: %s part(s)", item.source_id, len(new_sources))
        except Exception as exc:
            errors.append(f"{item.source_id}: {exc}")
            logger.warning("Failed %s: %s", item.source_id, exc)

    summary = {
        "discovered": len(discovered),
        "provider_counts": provider_counts,
        "synced": synced,
        "skipped": skipped,
        "chunks_indexed": chunks,
        "errors": errors[:25],
    }
    logger.info("Summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def wait_for_api() -> None:
    for _ in range(30):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=5)
            if resp.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"RAG API not reachable at {API_BASE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Claude, Codex, and Cursor chats into NeMo RAG.")
    parser.add_argument("--force", action="store_true", help="Reindex even if the local sync state says unchanged.")
    parser.add_argument("--provider", choices=["claude", "codex", "cursor"], help="Only sync one provider.")
    args = parser.parse_args()
    wait_for_api()
    summary = sync_transcripts(force=args.force, provider_filter=args.provider)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
