#!/usr/bin/env python3
"""nv-ingest extraction helper for rag_api.

Reads PDF bytes from stdin, submits to nv-ingest (text + tables + charts),
emits JSON to stdout: list of {page, text} per page.

Designed to run in the nv-ingest .venv (which has the client lib);
rag_api.py shells out to this via subprocess to keep its own venv clean.

Args:
  argv[1] -- broker host (default: node3)
  argv[2] -- broker port (default: 7671)
  argv[3] -- extension hint (default: .pdf)
"""
from __future__ import annotations

import json
import sys
import tempfile

NV_INGEST_HOST = sys.argv[1] if len(sys.argv) > 1 else "node3"
NV_INGEST_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 7671
EXT_HINT = sys.argv[3] if len(sys.argv) > 3 else ".pdf"


def main() -> int:
    try:
        from nv_ingest_client.client import NvIngestClient, Ingestor
        from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
    except ImportError as e:
        json.dump({"error": f"nv_ingest_client not installed: {e}"}, sys.stdout)
        return 2

    data = sys.stdin.buffer.read()
    if not data:
        json.dump({"error": "no_input"}, sys.stdout)
        return 1

    with tempfile.NamedTemporaryFile(suffix=EXT_HINT, delete=False) as f:
        f.write(data)
        path = f.name

    try:
        client = NvIngestClient(
            message_client_allocator=SimpleClient,
            message_client_hostname=NV_INGEST_HOST,
            message_client_port=NV_INGEST_PORT,
        )
        ingestor = (
            Ingestor(documents=[path], client=client)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=False,
                text_depth="page",
            )
        )
        results = ingestor.ingest(show_progress=False)
    except Exception as e:
        json.dump({"error": f"ingest_failed: {e!r}"}, sys.stdout)
        return 1

    if not results:
        json.dump({"error": "no_results"}, sys.stdout)
        return 1

    # Group primitives by page; nv-ingest emits text/table/chart per element.
    pages: dict[int, list[dict]] = {}
    for primitive in results[0]:
        if not isinstance(primitive, dict):
            continue
        meta = primitive.get("metadata") or {}
        cm = meta.get("content_metadata") or {}
        page_idx = cm.get("page_number")
        if page_idx is None:
            page_idx = 0
        content_type = (
            cm.get("type")
            or primitive.get("document_type")
            or "text"
        )
        content = meta.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            # Tables / charts often store structured representation under sub-fields
            table_meta = meta.get("table_metadata") or {}
            chart_meta = meta.get("chart_metadata") or {}
            content = (
                table_meta.get("table_content")
                or chart_meta.get("caption")
                or chart_meta.get("table_content")
                or ""
            )
        if not isinstance(content, str) or not content.strip():
            continue
        pages.setdefault(int(page_idx), []).append(
            {"type": str(content_type), "text": content.strip()}
        )

    # One combined text per page with light type prefixes for tables/charts.
    out = []
    for page_num in sorted(pages.keys()):
        parts: list[str] = []
        for chunk in pages[page_num]:
            t = chunk["type"]
            text = chunk["text"]
            if t in ("text", "structured"):
                parts.append(text)
            elif t == "table":
                parts.append("[Table]\n" + text)
            elif t in ("chart", "image"):
                parts.append("[Figure]\n" + text)
            else:
                parts.append(text)
        if parts:
            # nv-ingest pages are 0-based; surface as 1-based for human-friendly metadata.
            out.append({"page": page_num + 1, "text": "\n\n".join(parts)})

    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
