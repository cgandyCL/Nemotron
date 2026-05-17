import email
import getpass
import hashlib
import io
import ipaddress
import mailbox
import os
import re
import logging
import secrets
import socket
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse

import frontmatter as fm

import json
import yaml
import tempfile
import subprocess
import requests
import whisper
import pytesseract
import docx
from PIL import Image
from bs4 import BeautifulSoup
from pypdf import PdfReader
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

from pymilvus import MilvusClient, DataType
from langchain_core.vectorstores import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VLLM_LLM_BASE_URL  = os.getenv("VLLM_LLM_BASE_URL",  "http://localhost:8003/v1")
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://127.0.0.1:8012/v1")
LLM_MODEL      = os.getenv("LLM_MODEL",      "nvidia/Qwen3-14B-FP8")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "nvidia/llama-3.2-nv-embedqa-1b-v2")
EMBED_DIM       = int(os.getenv("EMBED_DIM",  "2048"))
RERANK_MODEL    = os.getenv("RERANK_MODEL",    "nvidia/llama-3.2-nv-rerankqa-1b-v2")
RERANK_BASE_URL   = os.getenv("RERANK_BASE_URL",   "http://node3:8005/v1/ranking")
MILVUS_URL        = os.getenv("MILVUS_URL",        "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "second_brain_nvidia")
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 240
RAG_API_TOKEN = os.getenv("RAG_API_TOKEN", "").strip()
RAG_ALLOW_REMOTE_NO_AUTH = os.getenv("RAG_ALLOW_REMOTE_NO_AUTH", "0").lower() in {"1", "true", "yes"}
RAG_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("RAG_CORS_ORIGINS", "http://127.0.0.1:8011,http://localhost:8011").split(",")
    if origin.strip()
]
MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("RAG_MAX_TEXT_CHARS", str(2_000_000)))
MAX_URL_BYTES = int(os.getenv("RAG_MAX_URL_BYTES", str(5 * 1024 * 1024)))
MAX_URL_REDIRECTS = int(os.getenv("RAG_MAX_URL_REDIRECTS", "3"))
MAX_BATCH_FILES = int(os.getenv("RAG_MAX_BATCH_FILES", "100"))
MAX_FOLDER_FILES = int(os.getenv("RAG_MAX_FOLDER_FILES", "1000"))
MAX_FOLDER_FILE_BYTES = int(os.getenv("RAG_MAX_FOLDER_FILE_BYTES", str(50 * 1024 * 1024)))
MAX_ZIP_MEMBERS = int(os.getenv("RAG_MAX_ZIP_MEMBERS", "2000"))
MAX_ZIP_UNCOMPRESSED_BYTES = int(os.getenv("RAG_MAX_ZIP_UNCOMPRESSED_BYTES", str(250 * 1024 * 1024)))
DEFAULT_FS_ROOTS = ",".join(
    [
        str(Path.home()),
        "/mnt",
        "/media",
        str(Path("/run/media") / getpass.getuser()),
        "/srv",
        "/tmp",
    ]
)
RAG_ALLOWED_FS_ROOTS = [
    Path(item).expanduser().resolve()
    for item in os.getenv("RAG_ALLOWED_FS_ROOTS", DEFAULT_FS_ROOTS).split(",")
    if item.strip()
]

# nv-ingest integration for PDFs (text + table cells + chart captions).
# When enabled and reachable, PDF ingestion routes through nv-ingest instead
# of PyPDF, so charts/tables become searchable. Falls back to PyPDF on any
# failure so the working path stays intact.
NV_INGEST_ENABLED = os.getenv('NV_INGEST_ENABLED', '1') not in ('0', 'false', 'False')
NV_INGEST_HOST = os.getenv('NV_INGEST_HOST', 'node3')
NV_INGEST_PORT = os.getenv('NV_INGEST_PORT', '7671')
NV_INGEST_HELPER_PY = os.getenv(
    'NV_INGEST_HELPER_PY',
    '/home/cgandycl/nv-ingest/.venv/bin/python3'
)
NV_INGEST_HELPER_SCRIPT = os.getenv(
    'NV_INGEST_HELPER_SCRIPT',
    str(Path(__file__).resolve().parent / '_nv_ingest_helper.py')
)
NV_INGEST_TIMEOUT_S = int(os.getenv('NV_INGEST_TIMEOUT_S', '300'))
NVIDIA_RECOMMENDED_EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
NVIDIA_RECOMMENDED_EMBED_DIM = 2048
SEMANTIC_TAG_LIMIT = 8
ENTITY_LIMIT = 12
LINK_GRAPH_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/link_graph.json")
SOURCE_CATALOG_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/source_catalog.json")
ACTIVITY_LOG_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/activity_log.json")
SEMANTIC_GRAPH_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/semantic_graph.json")
GRAPH_LAYOUT_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/graph_layout.json")
GRAPH_VIEWS_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/graph_views.json")
AUTHORING_STATE_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/authoring_state.json")
QUERY_TRACE_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/query_trace.json")
SECOND_BRAIN_STATE_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/second_brain_state.json")
MEMORY_WRAPPER_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/memory_wrapper.json")
HOUSEKEEPING_STATE_PATH = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/housekeeping_agent.json")
FRONTEND_DIR = Path("/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models/frontend")
FRONTEND_INDEX_PATH = FRONTEND_DIR / "index.html"
FRONTEND_ADMIN_PATH = FRONTEND_DIR / "admin.html"
FRONTEND_MEMORY_PATH = FRONTEND_DIR / "memory.html"
FRONTEND_CAPTURE_PATH = FRONTEND_DIR / "capture.html"
TRASH_SOURCE_TERMS = [
    term.strip().lower()
    for term in os.getenv(
        "RAG_TRASH_SOURCE_TERMS",
        "/.trash/,/trash/,/recycle/,/__pycache__/,/node_modules/,/.git/,/.cache/,/.pytest_cache/,/.mypy_cache/,/.ruff_cache/,/.venv/,/venv/,/tmp/,/temp/,speedtest,speed-test,benchmark probe,test probe,end-to-end speed probe",
    ).split(",")
    if term.strip()
]
TRASH_TAGS = {
    tag.strip().lower()
    for tag in os.getenv("RAG_TRASH_TAGS", "trash,junk,test,probe,speedtest,benchmark,cache,temp").split(",")
    if tag.strip()
}
TRASH_MAX_TEXT_CHARS = int(os.getenv("RAG_TRASH_MAX_TEXT_CHARS", str(750_000)))
RAW_CONTEXT_NAMESPACES = {
    item.strip()
    for item in os.getenv("RAG_RAW_CONTEXT_NAMESPACES", "ai-chats,execdesk-chat").split(",")
    if item.strip()
}
RAW_CONTEXT_NAMESPACE_PREFIXES = tuple(
    item.strip()
    for item in os.getenv("RAG_RAW_CONTEXT_NAMESPACE_PREFIXES", "execdesk-chat:").split(",")
    if item.strip()
)
RAW_CONTEXT_SOURCE_PREFIXES = tuple(
    item.strip()
    for item in os.getenv("RAG_RAW_CONTEXT_SOURCE_PREFIXES", "claude-code/,codex/,cursor/,execdesk://chat/").split(",")
    if item.strip()
)
NOISE_NAMESPACES = {
    item.strip()
    for item in os.getenv("RAG_NOISE_NAMESPACES", "web").split(",")
    if item.strip()
}
RETRIEVAL_BLOCK_STATUSES = {
    item.strip()
    for item in os.getenv("RAG_RETRIEVAL_BLOCK_STATUSES", "archived,junk,merged,discarded").split(",")
    if item.strip()
}
RETRIEVAL_BLOCK_TRUST = {
    item.strip()
    for item in os.getenv("RAG_RETRIEVAL_BLOCK_TRUST", "discard").split(",")
    if item.strip()
}
RETRIEVAL_LOW_TRUST = {"questionable", "use_with_caution", "stale"}
MIN_DEFAULT_QUALITY_SCORE = int(os.getenv("RAG_MIN_DEFAULT_QUALITY_SCORE", "45"))
DEFAULT_RAW_CHAT_CONTEXT_LIMIT = int(os.getenv("RAG_DEFAULT_RAW_CHAT_CONTEXT_LIMIT", "1"))
GRAPH_BRAIN_NAMESPACES = {
    item.strip()
    for item in os.getenv("RAG_GRAPH_BRAIN_NAMESPACES", "notes,documents,memory,code,host-node3,execdesk-deliverables").split(",")
    if item.strip()
}
DEEP_LINK_LIMIT = 6
DEEP_LINK_MIN_SCORE = 4
SEMANTIC_EDGE_LIMIT = 2
SEMANTIC_EDGE_MIN_SCORE = 2
SEMANTIC_RETRIEVAL_MAX_SOURCES = 250
SEMANTIC_EMBED_MAX_SOURCES = 2500
QUERY_TRACE_LIMIT = 60
GRAPH_CACHE_TTL_SECONDS = 20
QUERY_CACHE_TTL_SECONDS = 120
HOUSEKEEPING_ENABLED = os.getenv("RAG_HOUSEKEEPING_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
HOUSEKEEPING_INTERVAL_SECONDS = max(60, int(os.getenv("RAG_HOUSEKEEPING_INTERVAL_SECONDS", "300")))
HOUSEKEEPING_HEARTBEAT_SECONDS = max(10, int(os.getenv("RAG_HOUSEKEEPING_HEARTBEAT_SECONDS", "30")))
HOUSEKEEPING_MEMORY_BATCH = max(1, int(os.getenv("RAG_HOUSEKEEPING_MEMORY_BATCH", "5")))
EXECDESK_ROOT = Path(os.getenv("EXECDESK_ROOT", "/home/cgandycl/mnt/node3-aihome/execdesk")).expanduser()
HYBRID_CHAT_TIMEOUT_SECONDS = 18
HYBRID_RETRIEVE_K = 4
HYBRID_BRAIN_LIMIT = 3
HYBRID_NEIGHBOR_LIMIT = 2
HYBRID_CHUNK_CHARS = 900
HYBRID_ARTICLE_CHARS = 1200
HYBRID_NEIGHBOR_ARTICLE_CHARS = 700
FAST_QUERY_SECTIONS = 4
FAST_QUERY_SENTENCE_CHARS = 260
LLM_CHAT_TIMEOUT_SECONDS = int(os.getenv("RAG_LLM_TIMEOUT_SECONDS", "180"))
CURATION_STATUSES = {"inbox", "active", "trusted", "stale", "archived", "unreviewed", "kept", "merged", "needs_action", "junk", "discarded"}
TRUST_LEVELS = {"unreviewed", "trusted", "questionable", "use_with_caution", "stale", "discard"}
MEMORY_LEVELS = {"raw", "temporary", "project", "long-term", "sensitive"}
MEMORY_STATUSES = {"captured", "processed", "indexed", "reviewed", "archived", "blocked"}
MEMORY_SENSITIVITY = {"public", "private", "financial", "medical", "legal", "credentials", "sensitive"}
_catalog_cache_lock = threading.Lock()
_memory_lock = threading.Lock()
_source_records_cache: Dict[str, Dict[str, Any]] = {}
_graph_snapshot_cache: Dict[str, Any] = {"entries": {}}
_query_cache_lock = threading.Lock()
_query_answer_cache: Dict[str, Dict[str, Any]] = {}
_housekeeping_lock = threading.Lock()
_housekeeping_stop = threading.Event()
_housekeeping_thread: Optional[threading.Thread] = None
_housekeeping_run_lock = threading.Lock()
_reanalyze_job_lock = threading.Lock()
_reanalyze_job: Dict[str, Any] = {"running": False, "completed": 0, "total": 0, "errors": [], "started_at": None, "finished_at": None}
_semantic_build_lock = threading.Lock()
_semantic_build_running = False


NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _normalize_namespace(namespace: Optional[str]) -> Optional[str]:
    if namespace is None:
        return None
    value = str(namespace).strip()
    if not value:
        return None
    if not NAMESPACE_RE.fullmatch(value):
        raise HTTPException(400, "namespace must be 1-128 chars: letters, numbers, underscore, dash, dot, or colon")
    return value


def _is_raw_context_namespace(namespace: Optional[str]) -> bool:
    value = str(namespace or "")
    return value in RAW_CONTEXT_NAMESPACES or any(value.startswith(prefix) for prefix in RAW_CONTEXT_NAMESPACE_PREFIXES)


def _is_raw_context_record(namespace: Optional[str], source: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
    if _is_raw_context_namespace(namespace):
        return True
    source_value = str(source or "")
    if any(source_value.startswith(prefix) for prefix in RAW_CONTEXT_SOURCE_PREFIXES):
        return True
    return "chat" in {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}


def _graph_profile_allows(record: Dict[str, Any], profile: str) -> bool:
    if profile != "brain":
        return True
    namespace = str(record.get("namespace") or "")
    status = _normalize_curation_status(record.get("curation_status"))
    trust = _normalize_trust_level(record.get("trust_level"))
    return (
        namespace in GRAPH_BRAIN_NAMESPACES
        or bool(record.get("pinned"))
        or status in {"active", "trusted", "kept"}
        or trust == "trusted"
    )


def _milvus_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _milvus_eq(field: str, value: str) -> str:
    return f"{field} == {_milvus_string(value)}"


def _milvus_in(field: str, values: List[str]) -> str:
    return f"{field} in [{', '.join(_milvus_string(value) for value in values)}]"


def _is_loopback_client(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip() or request.query_params.get("token", "").strip()


def _authorized_remote_request(request: Request) -> bool:
    if RAG_ALLOW_REMOTE_NO_AUTH:
        return True
    if not RAG_API_TOKEN:
        return False
    token = _request_token(request)
    return bool(token) and secrets.compare_digest(token, RAG_API_TOKEN)


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_allowed_fs_path(path: Path) -> None:
    if any(_path_under(path, root) for root in RAG_ALLOWED_FS_ROOTS):
        return
    roots = ", ".join(str(root) for root in RAG_ALLOWED_FS_ROOTS)
    raise HTTPException(403, f"path is outside allowed roots: {roots}")


def _is_blocked_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        [
            ip.is_loopback,
            ip.is_private,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        ]
    )


def _validate_fetch_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(400, "URL scheme must be http or https")
    if not parsed.hostname:
        raise HTTPException(400, "URL host is required")
    if parsed.username or parsed.password:
        raise HTTPException(400, "URL credentials are not allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HTTPException(400, f"URL host could not be resolved: {exc}") from exc
    for info in infos:
        address = info[4][0]
        if _is_blocked_address(address):
            raise HTTPException(400, "URL host resolves to a private or reserved address")
    return value


async def _read_upload_limited(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"upload is too large; limit is {max_bytes} bytes")
    return data


# ── Reranker ──────────────────────────────────────────────────────────────────

class NIMReranker(BaseDocumentCompressor):
    def __init__(self, base_url: str, model_name: str, top_k: int = 3):
        super().__init__()
        self._base_url  = base_url
        self._model_name = model_name
        self._top_k     = top_k

    def compress_documents(self, documents, query, callbacks=None):
        if not documents:
            return documents
        payload = {
            "model": self._model_name,
            "query": {"text": query},
            "passages": [{"text": d.page_content} for d in documents],
        }
        resp = requests.post(self._base_url, json=payload, timeout=30)
        resp.raise_for_status()
        ranked = sorted(resp.json()["rankings"], key=lambda r: r["logit"], reverse=True)
        return [documents[r["index"]] for r in ranked[:self._top_k]]


def _supports_nvidia_dual_embedding_mode(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(token in name for token in ("embedqa", "e5"))


class SpecAwareEmbeddings(Embeddings):
    """
    Use dedicated query/passage embedding modes for NVIDIA QA-style embedding NIMs.

    NVIDIA's embedding NIM docs require `query` mode for online lookup embeddings and
    `passage` mode for indexed corpus embeddings. The OpenAI-compatible API supports
    this via `-query` and `-passage` model suffixes.
    """

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        base_kwargs = {
            "openai_api_base": base_url,
            "openai_api_key": "dummy",
            "check_embedding_ctx_length": False,
        }
        if _supports_nvidia_dual_embedding_mode(model_name):
            self._document_embeddings = None
            self._query_embeddings = None
            self.mode = "dual"
        else:
            shared = OpenAIEmbeddings(model=model_name, **base_kwargs)
            self._document_embeddings = shared
            self._query_embeddings = shared
            self.mode = "single"

    def _embed_nvidia(self, texts: List[str], input_type: str) -> List[List[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model_name, "input": texts, "input_type": input_type},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "dual":
            return self._embed_nvidia(texts, "document")
        return self._document_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.mode == "dual":
            return self._embed_nvidia([text], "query")[0]
        return self._query_embeddings.embed_query(text)


# ── Milvus VectorStore (MilvusClient-based, no deprecated ORM) ────────────────

class MilvusVectorStore(VectorStore):
    """Minimal LangChain VectorStore backed by MilvusClient + GPU_CAGRA index."""

    def __init__(self, client: MilvusClient, collection: str, embed_fn, dim: int):
        self._client            = client
        self._collection        = collection
        self._embed_fn          = embed_fn
        self._dim               = dim
        self._current_namespace = None   # set per-request by /query

    def add_texts(self, texts, metadatas=None, **kwargs):
        import uuid
        vecs = self._embed_fn.embed_documents(list(texts))
        rows = []
        ids  = []
        for i, (text, vec) in enumerate(zip(texts, vecs)):
            meta = (metadatas or [{}] * len(texts))[i]
            rid  = str(uuid.uuid4()).replace("-", "")[:64]
            ids.append(rid)
            row = {
                "id":          rid,
                "vector":      vec,
                "text":        text[:65500],
                "source":      str(meta.get("source", "unknown"))[:2048],
                "doc_type":    str(meta.get("type", "text"))[:64],
                "ingested_at": str(meta.get("ingested_at", ""))[:32],
                "namespace":   str(meta.get("namespace", "inbox"))[:128],
            }
            for key, value in meta.items():
                if key in row or key in {"type"}:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
                elif isinstance(value, list):
                    row[key] = value[:32]
                elif isinstance(value, dict):
                    row[key] = value
                else:
                    row[key] = str(value)
            rows.append(row)
        self._client.insert(self._collection, rows)
        return ids

    def similarity_search(self, query: str, k: int = 4, namespace: Optional[str] = None, source_filter: Optional[str] = None, **kwargs) -> List[Document]:
        vec = self._embed_fn.embed_query(query)
        ns   = _normalize_namespace(namespace or self._current_namespace)
        filters = []
        if ns:
            filters.append(_milvus_eq("namespace", ns))
        if source_filter:
            filters.append(_milvus_eq("source", source_filter))
        filt = " and ".join(filters)
        results = self._client.search(
            collection_name=self._collection,
            data=[vec],
            limit=k,
            filter=filt,
            output_fields=["text", "source", "doc_type", "ingested_at", "namespace"],
            search_params={"metric_type": "COSINE", "params": {"itopk_size": 128}},
        )
        docs = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append(Document(
                page_content=entity.get("text", ""),
                metadata={
                    "source":      entity.get("source", ""),
                    "type":        entity.get("doc_type", ""),
                    "ingested_at": entity.get("ingested_at", ""),
                    "namespace":   entity.get("namespace", "inbox"),
                },
            ))
        return docs

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError

    def _get_relevant_documents(self, query, **kwargs):
        return self.similarity_search(query, **kwargs)


# ── Globals ───────────────────────────────────────────────────────────────────

agent    = None
vectordb: Optional[MilvusVectorStore] = None
embeddings = None
mclient: Optional[MilvusClient] = None
reranker_inst: Optional[NIMReranker] = None
vectordb_lock = threading.Lock()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_collection(client: MilvusClient, name: str, dim: int) -> bool:
    if client.has_collection(name):
        return False
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",          DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector",      DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text",        DataType.VARCHAR, max_length=65535)
    schema.add_field("source",      DataType.VARCHAR, max_length=2048)
    schema.add_field("doc_type",    DataType.VARCHAR, max_length=64)
    schema.add_field("ingested_at", DataType.VARCHAR, max_length=32)
    schema.add_field("namespace",   DataType.VARCHAR, max_length=128)
    client.create_collection(collection_name=name, schema=schema, metric_type="COSINE")
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vector",
        index_name="cagra_idx",
        index_type="GPU_CAGRA",
        metric_type="COSINE",
        params={"intermediate_graph_degree": 64, "graph_degree": 32, "build_algo": "IVF_PQ"},
    )
    client.create_index(name, idx)
    return True


def _index_size() -> int:
    if mclient is None:
        return 0
    try:
        return mclient.get_collection_stats(MILVUS_COLLECTION).get("row_count", 0)
    except Exception:
        return 0


def _invalidate_catalog_cache() -> None:
    with _catalog_cache_lock:
        _source_records_cache.clear()
        _graph_snapshot_cache["entries"] = {}


def _catalog_sort_key(record: Dict[str, Any]) -> str:
    return str(record.get("ingested_at") or "")


def _load_source_catalog_records() -> List[dict]:
    if not SOURCE_CATALOG_PATH.exists():
        return []
    try:
        payload = json.loads(SOURCE_CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("records", []) if isinstance(payload, dict) else []
    clean: List[dict] = []
    for item in records:
        if not isinstance(item, dict) or not item.get("source"):
            continue
        clean.append({
            "source": item.get("source"),
            "type": item.get("type", "unknown"),
            "ingested_at": item.get("ingested_at", ""),
            "namespace": item.get("namespace", "inbox"),
            "chunk_count": int(item.get("chunk_count", 0) or 0),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "article": item.get("article"),
            "tags": list(item.get("tags") or []),
            "entities": list(item.get("entities") or []),
            "smart_collection": item.get("smart_collection"),
            "quality_score": item.get("quality_score"),
            "quality_flags": list(item.get("quality_flags") or []),
            "retrieval_tier": item.get("retrieval_tier"),
        })
    return sorted(clean, key=_catalog_sort_key, reverse=True)


def _save_source_catalog_records(records: List[dict]) -> None:
    try:
        content = json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "records": sorted(records, key=_catalog_sort_key, reverse=True),
            },
            ensure_ascii=True,
        )
        tmp = SOURCE_CATALOG_PATH.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(SOURCE_CATALOG_PATH)
    except Exception as exc:
        logger.warning(f"failed to save source catalog: {exc}")


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json_file(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"failed to save {path.name}: {exc}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return {}


def _default_memory_policy() -> Dict[str, Any]:
    return {
        "add_to_rag": False,
        "summarize": True,
        "extract_tasks": True,
        "create_note": True,
        "memory_level": "raw",
        "sensitivity": "private",
        "allowed_use": ["summarize", "answer", "extract_tasks"],
        "expires_at": None,
        "require_review": True,
        "include_raw_text": True,
    }


def _normalize_memory_policy(policy: Optional[Any] = None, **overrides: Any) -> Dict[str, Any]:
    merged = _default_memory_policy()
    merged.update({k: v for k, v in _model_to_dict(policy).items() if v is not None})
    merged.update({k: v for k, v in overrides.items() if v is not None})
    level = str(merged.get("memory_level") or "raw").strip().lower()
    sensitivity = str(merged.get("sensitivity") or "private").strip().lower()
    merged["memory_level"] = level if level in MEMORY_LEVELS else "raw"
    merged["sensitivity"] = sensitivity if sensitivity in MEMORY_SENSITIVITY else "private"
    allowed = merged.get("allowed_use") or []
    if isinstance(allowed, str):
        allowed = [allowed]
    merged["allowed_use"] = [str(item).strip() for item in allowed if str(item).strip()][:12]
    if not merged["allowed_use"]:
        merged["allowed_use"] = ["summarize", "answer", "extract_tasks"]
    for key in ("add_to_rag", "summarize", "extract_tasks", "create_note", "require_review", "include_raw_text"):
        merged[key] = bool(merged.get(key))
    return merged


def _default_memory_state() -> Dict[str, Any]:
    return {
        "updated_at": _now_iso(),
        "rules": {
            "default_review_required": True,
            "auto_index_trusted": False,
            "trusted_domains": [],
            "blocked_domains": [],
        },
        "items": [],
    }


def _load_memory_state() -> Dict[str, Any]:
    payload = _load_json_file(MEMORY_WRAPPER_PATH, _default_memory_state())
    if not isinstance(payload, dict):
        payload = _default_memory_state()
    payload.setdefault("rules", _default_memory_state()["rules"])
    payload.setdefault("items", [])
    clean_items = []
    for raw in payload.get("items") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        item = dict(raw)
        item["policy"] = _normalize_memory_policy(item.get("policy"))
        item["status"] = item.get("status") if item.get("status") in MEMORY_STATUSES else "captured"
        item.setdefault("tags", [])
        item.setdefault("tasks", [])
        item.setdefault("questions", [])
        item.setdefault("errors", [])
        item.setdefault("blocked", [])
        clean_items.append(item)
    payload["items"] = clean_items
    return payload


def _save_memory_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    content = json.dumps(state, ensure_ascii=True, indent=2)
    with _memory_lock:
        tmp = MEMORY_WRAPPER_PATH.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(MEMORY_WRAPPER_PATH)


def _memory_item(state: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for item in state.get("items", []):
        if item.get("id") == item_id:
            return item
    return None


def _memory_source_kind(source: str, content: str, explicit: Optional[str] = None) -> str:
    if explicit:
        clean = str(explicit).strip().lower()
        if clean:
            return clean[:32]
    value = (source or "").lower()
    if re.search(r"(youtube\.com|youtu\.be|vimeo\.com|\.mp4($|\?)|\.mov($|\?)|\.webm($|\?))", value):
        return "video"
    if value.startswith(("http://", "https://")):
        return "url"
    if content.strip():
        return "text"
    return "capture"


def _memory_title_from_content(content: str, fallback: str = "Captured memory") -> str:
    for line in (content or "").splitlines():
        clean = line.strip(" #\t")
        if clean:
            return clean[:120]
    return fallback


def _memory_source_label(item: Dict[str, Any], requested: Optional[str] = None) -> str:
    if requested and requested.strip():
        return requested.strip()[:2048]
    title = item.get("title") or item.get("summary") or item.get("source") or item.get("id")
    slug = _safe_slug(str(title), "memory")
    level = item.get("policy", {}).get("memory_level", "raw")
    return f"memory/{level}/{slug}-{str(item.get('id', ''))[:8]}"


def _clean_caption_text(text: str) -> str:
    lines: List[str] = []
    seen = set()
    for line in (text or "").splitlines():
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if not clean or clean == "WEBVTT" or clean.startswith(("Kind:", "Language:")):
            continue
        if re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->", clean):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        lines.append(clean)
    return "\n".join(lines).strip()


def _extract_video_transcript(url: str) -> str:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise ValueError("yt-dlp is not installed, so video URL transcript extraction is unavailable") from exc

    options = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title") or url
    description = info.get("description") or ""
    tracks = info.get("subtitles") or info.get("automatic_captions") or {}
    preferred = tracks.get("en") or tracks.get("en-US") or next(iter(tracks.values()), [])
    transcript = ""
    if preferred:
        track = next((item for item in preferred if item.get("ext") in {"vtt", "json3", "srv3"}), None) or preferred[0]
        if track and track.get("url"):
            resp = requests.get(track["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.text
            if track.get("ext") == "json3":
                try:
                    payload = json.loads(raw)
                    parts = []
                    for event in payload.get("events", []):
                        for seg in event.get("segs", []) or []:
                            value = seg.get("utf8", "").strip()
                            if value:
                                parts.append(value)
                    transcript = " ".join(parts)
                except Exception:
                    transcript = raw
            else:
                transcript = _clean_caption_text(raw)
    if transcript.strip():
        return f"Title: {title}\nURL: {url}\n\nTranscript:\n{transcript.strip()}"
    if description.strip():
        return f"Title: {title}\nURL: {url}\n\nDescription:\n{description.strip()}"
    raise ValueError("No transcript or description was available for this video URL")


def _extract_memory_text(item: Dict[str, Any]) -> str:
    content = (item.get("content") or "").strip()
    if content:
        return content
    source = (item.get("source") or "").strip()
    kind = item.get("source_kind") or _memory_source_kind(source, content)
    if kind == "video" and source:
        return _extract_video_transcript(source)
    if source.startswith(("http://", "https://")):
        return _extract_url(source)
    return content


def _fallback_memory_analysis(text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    compact = " ".join((text or "").split())
    tags = _merge_tags(item.get("tags", []), [item.get("source_kind", "capture")])
    task_lines = []
    for line in (text or "").splitlines():
        clean = line.strip(" -*\t")
        if re.search(r"\b(todo|follow up|action|next step|need to|should|must)\b", clean, re.IGNORECASE):
            task_lines.append(clean[:220])
        if len(task_lines) >= 8:
            break
    return {
        "title": item.get("title") or _memory_title_from_content(text, "Captured memory"),
        "summary": compact[:420],
        "tags": tags,
        "tasks": task_lines,
        "questions": [],
        "entities": _extract_entities_fallback(text),
        "suggested_memory_level": item.get("policy", {}).get("memory_level", "raw"),
        "suggested_sensitivity": item.get("policy", {}).get("sensitivity", "private"),
    }


def _analyze_memory_text(text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    excerpt = (text or "")[:9000]
    if not excerpt.strip():
        return _fallback_memory_analysis(text, item)
    system_prompt = (
        "You are the policy layer for a personal second-brain RAG system. "
        "Return strict JSON with keys: title, summary, tags, tasks, questions, entities, "
        "suggested_memory_level, suggested_sensitivity. Keep it concise and operational."
    )
    user_payload = {
        "source": item.get("source"),
        "source_kind": item.get("source_kind"),
        "project": item.get("project"),
        "current_policy": item.get("policy"),
        "content": excerpt,
    }
    raw = _chat_completion(system_prompt, json.dumps(user_payload, ensure_ascii=True), temperature=0.1, max_tokens=700, timeout=60)
    parsed = _coerce_json_object(raw)
    if not parsed:
        return _fallback_memory_analysis(text, item)
    fallback = _fallback_memory_analysis(text, item)
    tags = parsed.get("tags") if isinstance(parsed.get("tags"), list) else []
    tasks = parsed.get("tasks") if isinstance(parsed.get("tasks"), list) else []
    questions = parsed.get("questions") if isinstance(parsed.get("questions"), list) else []
    entities = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
    level = str(parsed.get("suggested_memory_level") or fallback["suggested_memory_level"]).lower()
    sensitivity = str(parsed.get("suggested_sensitivity") or fallback["suggested_sensitivity"]).lower()
    return {
        "title": str(parsed.get("title") or fallback["title"])[:160],
        "summary": " ".join(str(parsed.get("summary") or fallback["summary"]).split())[:600],
        "tags": _merge_tags(item.get("tags", []), [str(t) for t in tags], [item.get("source_kind", "capture")]),
        "tasks": [str(task).strip()[:260] for task in tasks if str(task).strip()][:12],
        "questions": [str(q).strip()[:260] for q in questions if str(q).strip()][:12],
        "entities": [str(entity).strip()[:120] for entity in entities if str(entity).strip()][:ENTITY_LIMIT],
        "suggested_memory_level": level if level in MEMORY_LEVELS else fallback["suggested_memory_level"],
        "suggested_sensitivity": sensitivity if sensitivity in MEMORY_SENSITIVITY else fallback["suggested_sensitivity"],
    }


def _process_memory_item(item: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    if item.get("processed_at") and not force:
        return item
    errors = list(item.get("errors") or [])
    try:
        text = _extract_memory_text(item)
    except Exception as exc:
        text = item.get("content") or ""
        errors.append(str(exc))
    analysis = _analyze_memory_text(text, item)
    policy = _normalize_memory_policy(item.get("policy"))
    if policy.get("memory_level") == "raw" and analysis.get("suggested_memory_level") in MEMORY_LEVELS:
        policy["memory_level"] = analysis["suggested_memory_level"]
    if policy.get("sensitivity") == "private" and analysis.get("suggested_sensitivity") in MEMORY_SENSITIVITY:
        policy["sensitivity"] = analysis["suggested_sensitivity"]
    item.update({
        "title": item.get("title") or analysis.get("title"),
        "summary": analysis.get("summary", ""),
        "tags": _merge_tags(item.get("tags", []), analysis.get("tags", [])),
        "tasks": analysis.get("tasks", []),
        "questions": analysis.get("questions", []),
        "entities": analysis.get("entities", []),
        "extracted_text": text,
        "policy": policy,
        "status": "processed" if item.get("status") == "captured" else item.get("status", "processed"),
        "processed_at": _now_iso(),
        "updated_at": _now_iso(),
        "errors": errors[-12:],
    })
    return item


def _memory_document_text(item: Dict[str, Any]) -> str:
    policy = _normalize_memory_policy(item.get("policy"))
    parts = [
        f"Title: {item.get('title') or 'Untitled memory'}",
        f"Source: {item.get('source') or item.get('source_kind') or 'capture'}",
        f"Memory level: {policy.get('memory_level')}",
        f"Sensitivity: {policy.get('sensitivity')}",
    ]
    if item.get("project"):
        parts.append(f"Project: {item.get('project')}")
    if item.get("summary"):
        parts.append(f"Summary:\n{item.get('summary')}")
    if item.get("tasks"):
        parts.append("Action items:\n" + "\n".join(f"- {task}" for task in item.get("tasks", [])))
    if item.get("questions"):
        parts.append("Open questions:\n" + "\n".join(f"- {question}" for question in item.get("questions", [])))
    if policy.get("include_raw_text", True):
        raw = item.get("extracted_text") or item.get("content") or ""
        if raw.strip():
            parts.append(f"Source text:\n{raw.strip()}")
    return "\n\n".join(parts).strip()


def _delete_indexed_source(source: str) -> None:
    if not source or mclient is None:
        return
    mclient.delete(collection_name=MILVUS_COLLECTION, filter=_milvus_eq("source", source))
    _remove_from_link_graph(source)
    _remove_from_activity_log(source)
    _remove_from_authoring_state(source)
    _remove_sources_from_metadata({source})
    _invalidate_catalog_cache()


def _approve_memory_item(item: Dict[str, Any], req: Optional[Any] = None) -> Dict[str, Any]:
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    request_data = _model_to_dict(req)
    policy = _normalize_memory_policy(item.get("policy"), **_model_to_dict(request_data.get("policy")))
    if request_data.get("memory_level"):
        policy = _normalize_memory_policy(policy, memory_level=request_data.get("memory_level"))
    if request_data.get("sensitivity"):
        policy = _normalize_memory_policy(policy, sensitivity=request_data.get("sensitivity"))
    policy["add_to_rag"] = True
    item["policy"] = policy
    if not item.get("processed_at") or not (item.get("extracted_text") or item.get("summary")):
        _process_memory_item(item, force=False)

    source_label = _memory_source_label(item, request_data.get("source_label"))
    if item.get("rag_source"):
        _delete_indexed_source(item["rag_source"])

    namespace = _normalize_namespace(request_data.get("namespace") or item.get("namespace") or "memory") or "memory"
    tags = _merge_tags(item.get("tags", []), ["memory", policy.get("memory_level", "raw"), item.get("source_kind", "capture")])
    body = _memory_document_text(item)
    if not body.strip():
        raise HTTPException(400, "memory item has no content to index")
    metadata = {
        "tags": tags,
        "memory_item_id": item.get("id"),
        "memory_level": policy.get("memory_level"),
        "sensitivity": policy.get("sensitivity"),
        "source_kind": item.get("source_kind"),
        "original_source": item.get("source"),
        "project": item.get("project"),
        "review_status": "reviewed" if request_data.get("reviewed", True) else "indexed",
        "allowed_use": policy.get("allowed_use", []),
    }
    docs = _apply_semantic_metadata(
        [_make_doc(body, source_label, "memory", metadata, namespace=namespace)],
        source_label,
        "memory",
        namespace,
        tags=tags,
    )
    n = _add_chunks(docs)
    _save_authoring_for_source(
        source_label,
        aliases=[item.get("title") or source_label],
        manual_links=[],
        pinned=False,
        template_name=None,
        curation_status="kept",
        trust_level="trusted" if policy.get("sensitivity") in {"public", "private"} else "use_with_caution",
        review_note=f"Approved through memory wrapper item {item.get('id')}",
    )
    item.update({
        "status": "reviewed" if request_data.get("reviewed", True) else "indexed",
        "namespace": namespace,
        "rag_source": source_label,
        "chunks_indexed": n,
        "indexed_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    return item


def _approved_memory_sources(levels: List[str], include_raw: bool, include_sensitive: bool) -> Dict[str, Dict[str, Any]]:
    allowed_levels = {str(level).strip().lower() for level in levels if str(level).strip()}
    if include_raw:
        allowed_levels.add("raw")
    if not allowed_levels:
        allowed_levels = {"temporary", "project", "long-term"}
    source_map: Dict[str, Dict[str, Any]] = {}
    state = _load_memory_state()
    for item in state.get("items", []):
        source = item.get("rag_source")
        if not source or item.get("status") not in {"indexed", "reviewed"}:
            continue
        policy = _normalize_memory_policy(item.get("policy"))
        if policy.get("memory_level") not in allowed_levels:
            continue
        if not include_sensitive and policy.get("sensitivity") not in {"public", "private"}:
            continue
        source_map[source] = item
    return source_map


def _load_activity_log() -> Dict[str, Any]:
    if not ACTIVITY_LOG_PATH.exists():
        return {"sources": {}}
    try:
        payload = json.loads(ACTIVITY_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": {}}
    if not isinstance(payload, dict):
        return {"sources": {}}
    payload.setdefault("sources", {})
    return payload


def _save_activity_log(log: Dict[str, Any]) -> None:
    _save_json_file(ACTIVITY_LOG_PATH, log)


def _load_authoring_state() -> Dict[str, Any]:
    payload = _load_json_file(AUTHORING_STATE_PATH, {"sources": {}})
    if not isinstance(payload, dict):
        return {"sources": {}}
    payload.setdefault("sources", {})
    return payload


def _save_authoring_state(payload: Dict[str, Any]) -> None:
    _save_json_file(AUTHORING_STATE_PATH, payload)


def _normalize_curation_status(value: Optional[str]) -> str:
    status = str(value or "inbox").strip().lower()
    return status if status in CURATION_STATUSES else "inbox"


def _normalize_trust_level(value: Optional[str]) -> str:
    trust = str(value or "unreviewed").strip().lower()
    return trust if trust in TRUST_LEVELS else "unreviewed"


def _authoring_for_source(source: str) -> Dict[str, Any]:
    entry = _load_authoring_state().get("sources", {}).get(source, {})
    aliases = [str(item).strip() for item in (entry.get("aliases") or []) if str(item).strip()]
    manual_links = [str(item).strip() for item in (entry.get("manual_links") or []) if str(item).strip()]
    return {
        "aliases": aliases[:24],
        "manual_links": manual_links[:24],
        "pinned": bool(entry.get("pinned", False)),
        "template_name": str(entry.get("template_name") or "").strip() or None,
        "curation_status": _normalize_curation_status(entry.get("curation_status")),
        "trust_level": _normalize_trust_level(entry.get("trust_level")),
        "review_note": str(entry.get("review_note") or "").strip()[:1000],
    }


def _save_authoring_for_source(
    source: str,
    aliases: List[str],
    manual_links: List[str],
    pinned: bool,
    template_name: Optional[str],
    curation_status: Optional[str] = None,
    trust_level: Optional[str] = None,
    review_note: Optional[str] = None,
) -> None:
    payload = _load_authoring_state()
    sources = payload.setdefault("sources", {})
    sources[source] = {
        "aliases": aliases[:24],
        "manual_links": manual_links[:24],
        "pinned": bool(pinned),
        "template_name": template_name or None,
        "curation_status": _normalize_curation_status(curation_status),
        "trust_level": _normalize_trust_level(trust_level),
        "review_note": str(review_note or "").strip()[:1000],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_authoring_state(payload)


def _remove_from_authoring_state(source: str) -> None:
    payload = _load_authoring_state()
    sources = payload.setdefault("sources", {})
    if source in sources:
        del sources[source]
        _save_authoring_state(payload)


def _rename_authoring_state(source: str, new_source: str) -> None:
    if source == new_source:
        return
    payload = _load_authoring_state()
    sources = payload.setdefault("sources", {})
    if source in sources:
        sources[new_source] = sources.pop(source)
        _save_authoring_state(payload)


def _load_query_trace() -> Dict[str, Any]:
    payload = _load_json_file(QUERY_TRACE_PATH, {"events": []})
    if not isinstance(payload, dict):
        return {"events": []}
    payload.setdefault("events", [])
    return payload


def _save_query_trace(payload: Dict[str, Any]) -> None:
    _save_json_file(QUERY_TRACE_PATH, payload)


CURATION_STATUSES = {"inbox", "active", "trusted", "stale", "archived", "unreviewed", "kept", "merged", "needs_action", "junk", "discarded"}
TRUST_LEVELS = {"unreviewed", "trusted", "questionable", "use_with_caution", "stale", "discard"}
CONNECTOR_SERVICES = {
    "rag_api": "nemo-rag-api.service",
    "clifflabs_chat": "clifflabs-chat-sync.service",
    "execdesk_bridge": "execdesk-bridge.service",
    "execdesk_vite": "execdesk-vite.service",
    "execdesk_worker": "execdesk-worker.service",
}


def _load_second_brain_state() -> Dict[str, Any]:
    payload = _load_json_file(SECOND_BRAIN_STATE_PATH, {"sources": {}, "actions": {}})
    if not isinstance(payload, dict):
        payload = {"sources": {}, "actions": {}}
    payload.setdefault("sources", {})
    payload.setdefault("actions", {})
    return payload


def _save_second_brain_state(payload: Dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_json_file(SECOND_BRAIN_STATE_PATH, payload)


def _source_brain_state(source: str) -> Dict[str, Any]:
    state = _load_second_brain_state().get("sources", {}).get(source, {})
    return {
        "curation_status": state.get("curation_status", "unreviewed"),
        "trust_level": state.get("trust_level", "unreviewed"),
        "review_note": state.get("review_note", ""),
        "action_status": state.get("action_status", "open"),
        "reviewed_at": state.get("reviewed_at"),
    }


def _record_with_brain_state(record: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(record)
    merged.update(_source_brain_state(record.get("source", "")))
    return merged


def _set_source_brain_state(source: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_second_brain_state()
    source_map = state.setdefault("sources", {})
    current = dict(source_map.get(source, {}))
    current.update(updates)
    current["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_map[source] = current
    _save_second_brain_state(state)
    return _source_brain_state(source)


def _token_set(*values: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "into", "your", "you", "are", "was", "were", "has", "have"}
    tokens: set[str] = set()
    for value in values:
        for token in re.split(r"[^a-z0-9]+", str(value or "").lower()):
            if len(token) >= 3 and token not in stop:
                tokens.add(token)
    return tokens


def _source_hygiene_reason(source: str, namespace: Optional[str] = None, tags: Optional[List[str]] = None, text: str = "") -> Optional[str]:
    haystack = f"/{str(source or '').replace(chr(92), '/').lower().strip('/')}/ {str(namespace or '').lower()} {str(text or '')[:500].lower()}"
    for term in TRASH_SOURCE_TERMS:
        if term and term in haystack:
            return f"blocked source term: {term}"
    tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
    blocked_tags = sorted(tag_set & TRASH_TAGS)
    if blocked_tags:
        return f"blocked tag: {blocked_tags[0]}"
    if text and len(text) > TRASH_MAX_TEXT_CHARS and any(token in haystack for token in ("auto-capture", "chat transcript", "tool output", "session")):
        return f"oversized auto-capture: {len(text)} chars"
    return None


def _is_trash_source(source: str, namespace: Optional[str] = None, tags: Optional[List[str]] = None, text: str = "") -> bool:
    return _source_hygiene_reason(source, namespace, tags, text) is not None


def _skip_trash_ingest(source: str, namespace: Optional[str], tags: Optional[List[str]], text: str = "") -> Optional[Any]:
    reason = _source_hygiene_reason(source, namespace, tags, text)
    if not reason:
        return None
    logger.info("Skipped trash ingest source=%s namespace=%s reason=%s", source, namespace, reason)
    return IngestResponse(source=source, chunks_indexed=0, index_size=_index_size(), namespace=namespace)


def _visible_second_brain_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible = []
    for record in records:
        curation = record.get("curation_status", "unreviewed")
        trust = record.get("trust_level", "unreviewed")
        if curation in {"archived", "discarded", "merged"} or trust == "discard":
            continue
        if _is_trash_source(record.get("source", ""), record.get("namespace"), record.get("tags"), " ".join(str(record.get(key) or "") for key in ("title", "summary"))):
            continue
        visible.append(record)
    return visible


def _dedupe_candidates(records: List[Dict[str, Any]], limit: int = 25) -> List[Dict[str, Any]]:
    enriched = []
    for record in records:
        tokens = _token_set(record.get("source", ""), record.get("title", ""), record.get("summary", ""))
        if tokens:
            enriched.append((record, tokens))
    candidates: List[Dict[str, Any]] = []
    for idx, (left, left_tokens) in enumerate(enriched):
        for right, right_tokens in enriched[idx + 1:]:
            union = left_tokens | right_tokens
            if not union:
                continue
            score = len(left_tokens & right_tokens) / len(union)
            same_title = (left.get("title") or "").strip().lower() == (right.get("title") or "").strip().lower()
            same_collection = left.get("smart_collection") and left.get("smart_collection") == right.get("smart_collection")
            if score >= 0.42 or (same_title and score >= 0.2) or (same_collection and score >= 0.5):
                candidates.append({
                    "score": round(score, 3),
                    "left": _record_with_brain_state(left),
                    "right": _record_with_brain_state(right),
                    "reason": "same title" if same_title else "similar metadata",
                })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def _action_candidates(records: List[Dict[str, Any]], limit: int = 50) -> List[Dict[str, Any]]:
    patterns = re.compile(r"\b(todo|to do|follow up|follow-up|action|fix|ship|send|call|review|decide|blocked|next step|needs?)\b", re.I)
    actions: List[Dict[str, Any]] = []
    for record in records:
        text = " ".join(str(record.get(key) or "") for key in ("title", "summary", "article"))
        if not patterns.search(text):
            continue
        source = record.get("source", "")
        digest = hashlib.sha1(f"{source}:{text[:240]}".encode("utf-8", errors="ignore")).hexdigest()[:12]
        actions.append({
            "id": digest,
            "source": source,
            "title": record.get("title") or Path(source).name,
            "summary": (record.get("summary") or text)[:260],
            "namespace": record.get("namespace"),
            "status": _source_brain_state(source).get("action_status", "open"),
        })
    return actions[:limit]


def _service_state(service_name: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=2,
        )
        active = result.stdout.strip() or result.stderr.strip() or "unknown"
    except Exception as exc:
        active = f"unknown: {exc}"
    return {"service": service_name, "state": active}


def _future_iso(seconds: int) -> str:
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat(timespec="seconds")


def _default_housekeeping_state() -> Dict[str, Any]:
    return {
        "name": "Mnemosyne Housekeeping Agent",
        "enabled": HOUSEKEEPING_ENABLED,
        "status": "disabled" if not HOUSEKEEPING_ENABLED else "starting",
        "started_at": None,
        "heartbeat_at": None,
        "last_run_at": None,
        "last_finished_at": None,
        "next_run_at": None,
        "last_duration_ms": 0,
        "run_count": 0,
        "error_count": 0,
        "last_error": None,
        "last_reason": None,
        "last_actions": [],
        "findings": [],
        "stats": {},
        "connectors": {},
        "execdesk": {},
        "policy": {
            "interval_seconds": HOUSEKEEPING_INTERVAL_SECONDS,
            "heartbeat_seconds": HOUSEKEEPING_HEARTBEAT_SECONDS,
            "memory_batch": HOUSEKEEPING_MEMORY_BATCH,
            "auto_delete": False,
            "auto_approve_rag": False,
            "execdesk_root": str(EXECDESK_ROOT),
        },
    }


def _load_housekeeping_state() -> Dict[str, Any]:
    payload = _load_json_file(HOUSEKEEPING_STATE_PATH, _default_housekeeping_state())
    if not isinstance(payload, dict):
        payload = _default_housekeeping_state()
    default = _default_housekeeping_state()
    for key, value in default.items():
        payload.setdefault(key, value)
    payload["enabled"] = HOUSEKEEPING_ENABLED
    payload["policy"] = {**default["policy"], **dict(payload.get("policy") or {})}
    return payload


def _save_housekeeping_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    with _housekeeping_lock:
        tmp = HOUSEKEEPING_STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(HOUSEKEEPING_STATE_PATH)


def _housekeeping_public_state() -> Dict[str, Any]:
    state = _load_housekeeping_state()
    thread_alive = bool(_housekeeping_thread and _housekeeping_thread.is_alive())
    state["thread_alive"] = thread_alive
    if not HOUSEKEEPING_ENABLED:
        state["status"] = "disabled"
    elif not thread_alive and state.get("status") != "running":
        state["status"] = "stopped"
    return state


def _housekeeping_update(**updates: Any) -> Dict[str, Any]:
    state = _load_housekeeping_state()
    state.update(updates)
    _save_housekeeping_state(state)
    return state


def _execdesk_housekeeping_status() -> Dict[str, Any]:
    root = EXECDESK_ROOT
    deliverables = root / "deliverables"
    state_db = deliverables / "state.db"
    status: Dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "deliverables_exists": deliverables.exists(),
        "state_db_exists": state_db.exists(),
        "state_db_bytes": state_db.stat().st_size if state_db.exists() else 0,
        "services": {
            name: _service_state(service)
            for name, service in CONNECTOR_SERVICES.items()
            if name.startswith("execdesk_")
        },
        "learning": {
            "rag_base_url": "http://node1:8011",
            "chat_namespace_prefix": "execdesk-chat",
            "task_namespace_prefix": "execdesk-task",
            "deliverable_namespace": "execdesk-deliverables",
            "policy": "subagent chats and task outputs should recall from global RAG and write agent memory through the ExecDesk bridge",
        },
    }
    try:
        package_path = root / "package.json"
        if package_path.exists():
            package = json.loads(package_path.read_text(encoding="utf-8"))
            status["package"] = {"name": package.get("name"), "version": package.get("version")}
    except Exception as exc:
        status["package_error"] = str(exc)[:240]
    return status


def _housekeeping_memory_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {status: 0 for status in sorted(MEMORY_STATUSES)}
    for item in items:
        status = str(item.get("status") or "captured")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _housekeeping_process_memory(actions: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = _load_memory_state()
    items = list(state.get("items") or [])
    processed = 0
    errors: List[Dict[str, str]] = []
    for item in items:
        if processed >= HOUSEKEEPING_MEMORY_BATCH:
            break
        policy = _normalize_memory_policy(item.get("policy"))
        if item.get("status") != "captured":
            continue
        if item.get("summary") and item.get("tasks"):
            continue
        if not policy.get("summarize") and not policy.get("extract_tasks"):
            continue
        try:
            _process_memory_item(item, force=False)
            processed += 1
            actions.append({
                "kind": "memory_processed",
                "item_id": item.get("id"),
                "title": item.get("title") or item.get("source"),
            })
        except Exception as exc:
            errors.append({"item_id": str(item.get("id")), "error": str(exc)[:240]})
    if processed:
        _save_memory_state(state)
    counts = _housekeeping_memory_counts(items)
    captured_backlog = counts.get("captured", 0)
    if captured_backlog:
        findings.append({
            "kind": "memory_review_backlog",
            "severity": "info",
            "message": f"{captured_backlog} captured memory item(s) are waiting for review.",
        })
    for err in errors[:5]:
        findings.append({
            "kind": "memory_process_error",
            "severity": "warning",
            "message": f"Memory item {err['item_id']} failed processing: {err['error']}",
        })
    return {"counts": counts, "processed_this_run": processed, "errors": errors[:10]}


def _housekeeping_scan_sources(actions: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_records = [_record_with_brain_state(item) for item in _list_source_records()]
    records = _visible_second_brain_records(all_records)
    status_counts: Dict[str, int] = {}
    namespace_counts: Dict[str, int] = {}
    for record in records:
        status = record.get("curation_status", "unreviewed")
        status_counts[status] = status_counts.get(status, 0) + 1
        namespace = record.get("namespace") or "inbox"
        namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
    dedupe = _dedupe_candidates(records, limit=10)
    actions_found = _action_candidates(records, limit=25)
    noise = []
    for record in records[:250]:
        try:
            review = _noise_review_for_record(record)
        except Exception:
            continue
        if review.get("action") and review.get("action") != "none":
            noise.append(review)
        if len(noise) >= 10:
            break
    if dedupe:
        findings.append({
            "kind": "duplicate_candidates",
            "severity": "info",
            "message": f"{len(dedupe)} likely duplicate source pair(s) need review.",
        })
    if noise:
        findings.append({
            "kind": "noise_candidates",
            "severity": "info",
            "message": f"{len(noise)} noisy or low-signal source(s) should be curated.",
        })
    if actions_found:
        findings.append({
            "kind": "action_candidates",
            "severity": "info",
            "message": f"{len(actions_found)} action-like source(s) are open.",
        })
    actions.append({
        "kind": "source_catalog_scanned",
        "source_count": len(records),
        "hidden_count": len(all_records) - len(records),
    })
    return {
        "source_count": len(records),
        "hidden_trash_or_discarded": len(all_records) - len(records),
        "status_counts": status_counts,
        "namespace_counts": namespace_counts,
        "duplicate_candidates": len(dedupe),
        "open_actions": sum(1 for item in actions_found if item.get("status") != "done"),
        "noise_candidates": len(noise),
    }


def _housekeeping_run_once(reason: str = "manual") -> Dict[str, Any]:
    if not HOUSEKEEPING_ENABLED:
        return _housekeeping_update(status="disabled", last_reason=reason)
    if not _housekeeping_run_lock.acquire(blocking=False):
        state = _housekeeping_public_state()
        state["accepted"] = False
        state["message"] = "housekeeping agent is already running"
        return state
    started = time.monotonic()
    actions: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    run_started_at = _now_iso()
    try:
        state = _load_housekeeping_state()
        state.update({
            "status": "running",
            "heartbeat_at": run_started_at,
            "last_run_at": run_started_at,
            "last_reason": reason,
            "last_error": None,
        })
        _save_housekeeping_state(state)
        memory_stats = _housekeeping_process_memory(actions, findings)
        source_stats = _housekeeping_scan_sources(actions, findings)
        connectors = {name: _service_state(service) for name, service in CONNECTOR_SERVICES.items()}
        execdesk = _execdesk_housekeeping_status()
        stats = {
            "index_size": _index_size(),
            "memory": memory_stats,
            "sources": source_stats,
        }
        unknown = [name for name, item in connectors.items() if str(item.get("state", "")).startswith("unknown")]
        inactive_execdesk = [
            name for name, item in execdesk.get("services", {}).items()
            if item.get("state") != "active"
        ]
        if unknown:
            findings.append({
                "kind": "connector_state",
                "severity": "warning",
                "message": f"Connector status could not be confirmed: {', '.join(unknown[:5])}.",
            })
        if inactive_execdesk:
            findings.append({
                "kind": "execdesk_services",
                "severity": "info",
                "message": f"ExecDesk service units are not active: {', '.join(inactive_execdesk[:5])}.",
            })
        finished = _now_iso()
        state = _load_housekeeping_state()
        state.update({
            "status": "idle",
            "heartbeat_at": finished,
            "last_finished_at": finished,
            "next_run_at": _future_iso(HOUSEKEEPING_INTERVAL_SECONDS),
            "last_duration_ms": int((time.monotonic() - started) * 1000),
            "run_count": int(state.get("run_count", 0) or 0) + 1,
            "last_actions": actions[-25:],
            "findings": findings[-25:],
            "stats": stats,
            "connectors": connectors,
            "execdesk": execdesk,
            "last_error": None,
        })
        _save_housekeeping_state(state)
        return _housekeeping_public_state()
    except Exception as exc:
        logger.exception("housekeeping agent run failed")
        state = _load_housekeeping_state()
        state.update({
            "status": "error",
            "heartbeat_at": _now_iso(),
            "last_finished_at": _now_iso(),
            "last_duration_ms": int((time.monotonic() - started) * 1000),
            "error_count": int(state.get("error_count", 0) or 0) + 1,
            "last_error": str(exc)[:1000],
            "findings": findings[-25:] + [{
                "kind": "housekeeping_error",
                "severity": "error",
                "message": str(exc)[:500],
            }],
        })
        _save_housekeeping_state(state)
        return _housekeeping_public_state()
    finally:
        _housekeeping_run_lock.release()


def _housekeeping_loop() -> None:
    _housekeeping_update(
        status="idle",
        started_at=_now_iso(),
        heartbeat_at=_now_iso(),
        next_run_at=_future_iso(5),
    )
    next_run = time.monotonic() + 5
    while not _housekeeping_stop.is_set():
        now_mono = time.monotonic()
        if now_mono >= next_run:
            _housekeeping_run_once(reason="scheduled")
            next_run = time.monotonic() + HOUSEKEEPING_INTERVAL_SECONDS
        else:
            state = _load_housekeeping_state()
            if state.get("status") != "running":
                state.update({
                    "status": "idle",
                    "heartbeat_at": _now_iso(),
                    "next_run_at": _future_iso(max(1, int(next_run - now_mono))),
                })
                _save_housekeeping_state(state)
        wait_for = min(HOUSEKEEPING_HEARTBEAT_SECONDS, max(1, int(next_run - time.monotonic())))
        _housekeeping_stop.wait(wait_for)
    _housekeeping_update(status="stopped", heartbeat_at=_now_iso(), next_run_at=None)


def _start_housekeeping_agent() -> None:
    global _housekeeping_thread
    if not HOUSEKEEPING_ENABLED:
        _housekeeping_update(status="disabled", heartbeat_at=_now_iso(), next_run_at=None)
        return
    if _housekeeping_thread and _housekeeping_thread.is_alive():
        return
    _housekeeping_stop.clear()
    _housekeeping_thread = threading.Thread(target=_housekeeping_loop, name="mnemosyne-housekeeper", daemon=True)
    _housekeeping_thread.start()


def _stop_housekeeping_agent() -> None:
    if _housekeeping_thread and _housekeeping_thread.is_alive():
        _housekeeping_stop.set()
        _housekeeping_thread.join(timeout=5)


def _append_query_trace(event_type: str, query: str, sources: List[str], sections: Optional[List[Dict[str, str]]] = None, namespace: Optional[str] = None) -> None:
    payload = _load_query_trace()
    events = list(payload.get("events") or [])
    section_items = []
    for section in list(sections or [])[:4]:
        section_items.append({
            "kind": section.get("kind"),
            "source": section.get("source"),
            "title": section.get("title"),
            "excerpt": _first_sentence(section.get("text", ""), max_chars=220),
        })
    events.insert(0, {
        "type": event_type,
        "query": query[:400],
        "namespace": namespace,
        "sources": list(dict.fromkeys(source for source in sources if source))[:8],
        "sections": section_items,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    payload["events"] = events[:QUERY_TRACE_LIMIT]
    _save_query_trace(payload)


def _record_activity(sources: List[str], event: str) -> None:
    if not sources:
        return
    log = _load_activity_log()
    source_map = log.setdefault("sources", {})
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for source in sources:
        if not source:
            continue
        entry = source_map.setdefault(source, {})
        counter_key = f"{event}_count"
        entry[counter_key] = int(entry.get(counter_key, 0) or 0) + 1
        entry["last_activity_at"] = timestamp
        entry[f"last_{event}_at"] = timestamp
    _save_activity_log(log)
    _invalidate_catalog_cache()


def _activity_for_source(source: str) -> Dict[str, Any]:
    entry = _load_activity_log().get("sources", {}).get(source, {})
    return {
        "query_count": int(entry.get("query_count", 0) or 0),
        "retrieve_count": int(entry.get("retrieve_count", 0) or 0),
        "synthesis_count": int(entry.get("synthesis_count", 0) or 0),
        "ingest_count": int(entry.get("ingest_count", 0) or 0),
        "last_activity_at": entry.get("last_activity_at"),
    }


def _remove_from_activity_log(source: str) -> None:
    log = _load_activity_log()
    sources = log.setdefault("sources", {})
    if source in sources:
        del sources[source]
        _save_activity_log(log)


def _set_activity_entry(source: str, entry: Dict[str, Any]) -> None:
    if not source:
        return
    log = _load_activity_log()
    sources = log.setdefault("sources", {})
    if entry:
        sources[source] = dict(entry)
    else:
        sources.pop(source, None)
    _save_activity_log(log)


def _rename_activity_log(source: str, new_source: str) -> None:
    if source == new_source:
        return
    log = _load_activity_log()
    sources = log.setdefault("sources", {})
    if source in sources:
        sources[new_source] = sources.pop(source)
        _save_activity_log(log)


def _strip_think_traces(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _semantic_edge_score(record: Dict[str, Any], other: Dict[str, Any]) -> int:
    if record["source"] == other["source"]:
        return 0
    score = 0
    if record.get("smart_collection") and record.get("smart_collection") == other.get("smart_collection"):
        score += 2
    if record.get("namespace") and record.get("namespace") == other.get("namespace"):
        score += 1
    tags_a = set(record.get("tags") or [])
    tags_b = set(other.get("tags") or [])
    entities_a = set(record.get("entities") or [])
    entities_b = set(other.get("entities") or [])
    score += min(3, len(tags_a & tags_b))
    score += min(3, len(entities_a & entities_b))
    if record.get("type") and record.get("type") == other.get("type"):
        score += 1
    return score


def _semantic_signature(record: Dict[str, Any]) -> str:
    payload = {
        "source": record.get("source"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "tags": list(record.get("tags") or []),
        "entities": list(record.get("entities") or []),
        "smart_collection": record.get("smart_collection"),
        "namespace": record.get("namespace"),
        "ingested_at": record.get("ingested_at"),
        "type": record.get("type"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _semantic_text(record: Dict[str, Any]) -> str:
    parts = [
        record.get("title") or Path(record.get("source", "")).stem,
        record.get("summary") or "",
        " ".join(record.get("tags") or []),
        " ".join(record.get("entities") or []),
        record.get("smart_collection") or "",
        record.get("namespace") or "",
        record.get("type") or "",
    ]
    text = "\n".join(part.strip() for part in parts if part and str(part).strip())
    return text[:4000] or (record.get("title") or record.get("source") or "")


def _heuristic_semantic_neighbor_map(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> Dict[str, List[str]]:
    neighbor_map: Dict[str, List[str]] = {}
    for record in records:
        blocked = set(graph_nodes.get(record["source"], {}).get("outbound_links", []))
        blocked.add(record["source"])
        candidates = []
        for other in records:
            if other["source"] in blocked:
                continue
            score = _semantic_edge_score(record, other)
            if score >= SEMANTIC_EDGE_MIN_SCORE:
                candidates.append((score, other["source"]))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        neighbor_map[record["source"]] = [target for _score, target in candidates[:SEMANTIC_EDGE_LIMIT]]
    return neighbor_map


def _load_semantic_graph_cache() -> Dict[str, Any]:
    payload = _load_json_file(SEMANTIC_GRAPH_PATH, {"signatures": {}, "neighbors": {}})
    if not isinstance(payload, dict):
        return {"signatures": {}, "neighbors": {}}
    payload.setdefault("signatures", {})
    payload.setdefault("neighbors", {})
    return payload


def _save_semantic_graph_cache(signatures: Dict[str, str], neighbors: Dict[str, List[str]]) -> None:
    _save_json_file(
        SEMANTIC_GRAPH_PATH,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signatures": signatures,
            "neighbors": neighbors,
        },
    )


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_fallback_map(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        source: [{"target": target, "weight": 0.1} for target in targets]
        for source, targets in _heuristic_semantic_neighbor_map(records, graph_nodes).items()
    }


def _build_semantic_graph_cache(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> None:
    global _semantic_build_running
    fallback = _semantic_fallback_map(records, graph_nodes)
    signatures = {record["source"]: _semantic_signature(record) for record in records}
    vectors: Dict[str, List[float]] = {}
    try:
        batch_size = 24
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [_semantic_text(record) for record in batch]
            embeds = embeddings.embed_documents(texts)
            for record, vector in zip(batch, embeds):
                vectors[record["source"]] = vector
    except Exception as exc:
        logger.warning(f"source-level semantic embedding fallback: {exc}")
        _save_semantic_graph_cache(signatures, fallback)
        with _semantic_build_lock:
            _semantic_build_running = False
        return
    neighbors: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        source = record["source"]
        blocked = set(graph_nodes.get(source, {}).get("outbound_links", []))
        blocked.add(source)
        src_vec = vectors.get(source)
        if not src_vec:
            neighbors[source] = list(fallback.get(source, []))
            continue
        candidates = []
        for other in records:
            target = other["source"]
            if target in blocked:
                continue
            other_vec = vectors.get(target)
            if not other_vec:
                continue
            score = _cosine_similarity(src_vec, other_vec)
            if score <= 0:
                continue
            adjusted = score + (_semantic_edge_score(record, other) * 0.03)
            candidates.append((adjusted, target))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        picked = [{"target": target, "weight": round(score, 4)} for score, target in candidates[:SEMANTIC_EDGE_LIMIT]]
        neighbors[source] = picked or list(fallback.get(source, []))
    _save_semantic_graph_cache(signatures, neighbors)
    with _semantic_build_lock:
        _semantic_build_running = False


def _queue_semantic_graph_rebuild(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> None:
    global _semantic_build_running
    with _semantic_build_lock:
        if _semantic_build_running:
            return
        _semantic_build_running = True
    thread = threading.Thread(target=_build_semantic_graph_cache, args=(records, graph_nodes), daemon=True)
    thread.start()


def _source_embedding_neighbor_map(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    fallback = _semantic_fallback_map(records, graph_nodes)
    if embeddings is None or len(records) > SEMANTIC_EMBED_MAX_SOURCES:
        return fallback
    signatures = {record["source"]: _semantic_signature(record) for record in records}
    cache = _load_semantic_graph_cache()
    if cache.get("signatures") == signatures and cache.get("neighbors"):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for key, value in cache.get("neighbors", {}).items():
            items = []
            for item in value or []:
                if isinstance(item, dict) and item.get("target"):
                    items.append({"target": item["target"], "weight": float(item.get("weight", 0.0) or 0.0)})
                elif isinstance(item, str):
                    items.append({"target": item, "weight": 0.1})
            out[key] = items
        if out:
            return out
    _queue_semantic_graph_rebuild(records, graph_nodes)
    return fallback


def _retrieval_semantic_neighbor_map(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> Dict[str, List[str]]:
    weighted = _source_embedding_neighbor_map(records, graph_nodes)
    return {source: [item["target"] for item in items[:SEMANTIC_EDGE_LIMIT]] for source, items in weighted.items()}


def _semantic_edges(records: List[Dict[str, Any]], graph_nodes: Dict[str, Any]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    by_source = {record["source"]: record for record in records}
    neighbor_map = _source_embedding_neighbor_map(records, graph_nodes)
    for source, targets in neighbor_map.items():
        for item in targets[:SEMANTIC_EDGE_LIMIT]:
            target = item["target"]
            if target in by_source:
                edges.append({"source": source, "target": target, "kind": "semantic", "weight": item.get("weight", 0.0)})
    return edges


def _semantic_links_for_source(source: str, records: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    records = records or _list_source_records()
    graph_nodes = _load_link_graph().get("nodes", {})
    return list(_source_embedding_neighbor_map(records, graph_nodes).get(source, []))


def _update_source_catalog_from_chunks(chunks: List[Document]) -> None:
    if not chunks:
        return
    record_map = {item["source"]: dict(item) for item in _load_source_catalog_records()}
    counts: Dict[str, int] = {}
    meta_by_source: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        source = meta.get("source")
        if not source:
            continue
        counts[source] = counts.get(source, 0) + 1
        meta_by_source.setdefault(source, meta)
    for source, count in counts.items():
        meta = meta_by_source[source]
        record = record_map.get(source, {
            "source": source,
            "type": meta.get("type", "unknown"),
            "ingested_at": meta.get("ingested_at", ""),
            "namespace": meta.get("namespace", "inbox"),
            "chunk_count": 0,
            "title": meta.get("title"),
            "summary": meta.get("summary"),
            "article": meta.get("article"),
            "tags": list(meta.get("tags") or []),
            "entities": list(meta.get("entities") or []),
            "smart_collection": meta.get("smart_collection"),
            "quality_score": meta.get("quality_score"),
            "quality_flags": list(meta.get("quality_flags") or []),
            "retrieval_tier": meta.get("retrieval_tier"),
        })
        record["type"] = meta.get("type", record.get("type", "unknown"))
        record["namespace"] = meta.get("namespace", record.get("namespace", "inbox"))
        record["ingested_at"] = meta.get("ingested_at", record.get("ingested_at", ""))
        record["title"] = meta.get("title", record.get("title"))
        record["summary"] = meta.get("summary", record.get("summary"))
        record["article"] = meta.get("article", record.get("article"))
        record["tags"] = list(meta.get("tags") or record.get("tags") or [])
        record["entities"] = list(meta.get("entities") or record.get("entities") or [])
        record["smart_collection"] = meta.get("smart_collection", record.get("smart_collection"))
        record["quality_score"] = meta.get("quality_score", record.get("quality_score"))
        record["quality_flags"] = list(meta.get("quality_flags") or record.get("quality_flags") or [])
        record["retrieval_tier"] = meta.get("retrieval_tier", record.get("retrieval_tier"))
        record["chunk_count"] = int(record.get("chunk_count", 0) or 0) + count
        record_map[source] = record
    _save_source_catalog_records(list(record_map.values()))


def _add_chunks(docs: List[Document], pre_chunked: bool = False) -> int:
    clean_docs = []
    for doc in docs:
        meta = doc.metadata or {}
        reason = _source_hygiene_reason(
            str(meta.get("source", "")),
            meta.get("namespace"),
            list(meta.get("tags") or []),
            doc.page_content or "",
        )
        if reason:
            logger.info("Skipped trash document source=%s reason=%s", meta.get("source", ""), reason)
            continue
        clean_docs.append(doc)
    chunks = clean_docs if pre_chunked else splitter.split_documents(clean_docs)
    kept_chunks: List[Document] = []
    skipped_sources: Dict[str, str] = {}
    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        source = meta.get("source", "")
        reason = _junk_source_reason(source, meta.get("type") or meta.get("doc_type") or "")
        if reason:
            skipped_sources[source] = reason
            continue
        kept_chunks.append(chunk)
    if skipped_sources:
        preview = ", ".join(f"{src} ({reason})" for src, reason in list(skipped_sources.items())[:5])
        logger.info("Skipped %s junk source(s) during indexing: %s", len(skipped_sources), preview)
    chunks = kept_chunks
    if not chunks:
        return 0
    with vectordb_lock:
        vectordb.add_documents(chunks)
    _update_source_catalog_from_chunks(chunks)
    _record_activity(list(dict.fromkeys(chunk.metadata.get("source", "") for chunk in chunks if chunk.metadata.get("source"))), "ingest")
    _invalidate_catalog_cache()
    return len(chunks)


def _make_doc(text: str, source: str, doc_type: str, extra: Dict = {}, namespace: Optional[str] = None) -> Document:
    namespace = namespace or _auto_namespace(doc_type, source)
    link_targets = list(extra.get("link_targets") or _extract_explicit_links(text))
    doc = Document(
        page_content=text,
        metadata={
            "source":    source,
            "type":      doc_type,
            "namespace": namespace,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "link_targets": link_targets,
            **extra,
        },
    )
    _apply_quality_metadata(doc, source, doc_type, namespace, tags=doc.metadata.get("tags", []))
    return doc


def _safe_slug(value: str, fallback: str = "general") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug[:64] or fallback


def _source_aliases(source: str, title: str = "") -> List[str]:
    aliases: List[str] = []
    candidates = [
        source,
        Path(source).name,
        Path(source).stem,
        title,
        title.replace(" ", "-") if title else "",
        title.replace(" ", "_") if title else "",
    ]
    candidates.extend(_authoring_for_source(source).get("aliases", []))
    for candidate in candidates:
        slug = _safe_slug(candidate, "")
        if slug and slug not in aliases:
            aliases.append(slug)
    return aliases


def _tokenize_reference(value: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", (value or "").lower()) if len(token) >= 3]


def _extract_explicit_links(text: str) -> List[str]:
    links: List[str] = []
    for match in re.findall(r"\[\[([^\]]+)\]\]", text):
        target = match.split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in links:
            links.append(target)
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        value = target.strip()
        if value.startswith(("http://", "https://", "mailto:")):
            continue
        value = value.split("#", 1)[0].strip()
        if value and value not in links:
            links.append(value)
        elif label.strip() and label.strip() not in links:
            links.append(label.strip())
    return links[:32]


def _load_link_graph() -> Dict[str, Any]:
    try:
        return json.loads(LINK_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"nodes": {}}


def _save_link_graph(graph: Dict[str, Any]) -> None:
    LINK_GRAPH_PATH.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_link_target(raw_target: str, graph: Dict[str, Any]) -> Optional[str]:
    target_slug = _safe_slug(raw_target, "")
    if not target_slug:
        return None
    for source, node in graph.get("nodes", {}).items():
        aliases = node.get("aliases", [])
        if target_slug in aliases:
            return source
    return None


def _record_aliases(record: Dict[str, Any]) -> List[str]:
    base = _source_aliases(record.get("source", ""), record.get("title", "") or "")
    for alias in _authoring_for_source(record.get("source", "")).get("aliases", []):
        slug = _safe_slug(alias, "")
        if slug and slug not in base:
            base.append(slug)
    return base


def _source_reference_text(record: Dict[str, Any]) -> str:
    parts = [
        record.get("source", ""),
        record.get("title", "") or "",
        record.get("summary", "") or "",
        " ".join(record.get("tags") or []),
        " ".join(record.get("entities") or []),
        record.get("smart_collection", "") or "",
        record.get("namespace", "") or "",
    ]
    return " ".join(part for part in parts if part)


def _score_reconstructed_link(source_record: Dict[str, Any], target_record: Dict[str, Any], raw_unresolved: List[str]) -> int:
    if source_record["source"] == target_record["source"]:
        return 0
    score = 0
    source_tags = set(source_record.get("tags") or [])
    target_tags = set(target_record.get("tags") or [])
    source_entities = {str(item).lower() for item in (source_record.get("entities") or [])}
    target_entities = {str(item).lower() for item in (target_record.get("entities") or [])}
    tag_overlap = source_tags & target_tags
    entity_overlap = source_entities & target_entities
    if source_record.get("smart_collection") and source_record.get("smart_collection") == target_record.get("smart_collection"):
        score += 2
    if source_record.get("namespace") and source_record.get("namespace") == target_record.get("namespace"):
        score += 1
    if len(tag_overlap) >= 2:
        score += 2
    elif len(tag_overlap) == 1:
        score += 1
    if len(entity_overlap) >= 2:
        score += 3
    elif len(entity_overlap) == 1:
        score += 1

    source_tokens = set(_tokenize_reference(_source_reference_text(source_record)))
    target_tokens = set(_tokenize_reference(_source_reference_text(target_record)))
    title_overlap = source_tokens & target_tokens
    if len(title_overlap) >= 3:
        score += 2
    elif len(title_overlap) >= 2:
        score += 1

    target_aliases = set(_record_aliases(target_record))
    for raw in raw_unresolved:
        raw_slug = _safe_slug(raw, "")
        raw_tokens = set(_tokenize_reference(raw))
        if raw_slug and raw_slug in target_aliases:
            score += 4
            continue
        if raw_tokens and len(raw_tokens & target_tokens) >= 2:
            score += 3
    return score


def _reconstruct_deep_links(records: List[Dict[str, Any]], graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = graph.setdefault("nodes", {})
    source_map = {record["source"]: record for record in records}
    for record in records:
        source = record["source"]
        node = nodes.setdefault(source, {})
        node["title"] = record.get("title") or source
        node["aliases"] = _record_aliases(record)
        node.setdefault("raw_links", [])
        node.setdefault("manual_links", [])
        explicit = []
        unresolved = []
        for raw in (list(node.get("raw_links", [])[:32]) + list(node.get("manual_links", [])[:24])):
            target = _resolve_link_target(raw, graph)
            if target and target != source and target not in explicit:
                explicit.append(target)
            elif raw not in unresolved:
                unresolved.append(raw)
        node["explicit_links"] = explicit
        node["unresolved_links"] = unresolved
        node["inferred_links"] = []

    for record in records:
        source = record["source"]
        node = nodes[source]
        candidates = []
        for target_record in records:
            if target_record["source"] == source or target_record["source"] in node["explicit_links"]:
                continue
            score = _score_reconstructed_link(record, target_record, node.get("unresolved_links", []))
            if score >= DEEP_LINK_MIN_SCORE:
                candidates.append((score, target_record["source"]))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        node["inferred_links"] = [target for _score, target in candidates[:DEEP_LINK_LIMIT]]
        combined = []
        for target in node["explicit_links"] + node["inferred_links"]:
            if target != source and target not in combined:
                combined.append(target)
        node["outbound_links"] = combined[:DEEP_LINK_LIMIT + len(node["explicit_links"])]
    _save_link_graph(graph)
    return nodes


def _upsert_link_graph(source: str, title: str, raw_links: List[str], manual_links: Optional[List[str]] = None, aliases: Optional[List[str]] = None) -> Dict[str, Any]:
    graph = _load_link_graph()
    nodes = graph.setdefault("nodes", {})
    node = nodes.setdefault(source, {})
    manual_links = [str(item).strip() for item in (manual_links or _authoring_for_source(source).get("manual_links", [])) if str(item).strip()]
    alias_items = [str(item).strip() for item in (aliases or _authoring_for_source(source).get("aliases", [])) if str(item).strip()]
    node["title"] = title or source
    node["aliases"] = _source_aliases(source, title) + [slug for slug in (_safe_slug(item, "") for item in alias_items) if slug]
    node["aliases"] = list(dict.fromkeys(node["aliases"]))[:32]
    node["raw_links"] = raw_links[:32]
    node["manual_links"] = manual_links[:24]
    resolved: List[str] = []
    unresolved: List[str] = []
    for raw in (raw_links[:32] + manual_links[:24]):
        target = _resolve_link_target(raw, graph)
        if target and target != source and target not in resolved:
            resolved.append(target)
        elif raw not in unresolved:
            unresolved.append(raw)
    node["explicit_links"] = list(resolved)
    node["inferred_links"] = [item for item in node.get("inferred_links", []) if item not in resolved and item != source]
    node["outbound_links"] = resolved + [item for item in node.get("inferred_links", []) if item not in resolved]
    node["unresolved_links"] = unresolved
    for other_source, other_node in nodes.items():
        if other_source == source:
            continue
        aliases = other_node.get("aliases", [])
        if any(_safe_slug(raw, "") in aliases for raw in (raw_links + manual_links)) and other_source not in node["explicit_links"]:
            node["explicit_links"].append(other_source)
        other_resolved = []
        for raw in (list(other_node.get("raw_links", [])) + list(other_node.get("manual_links", []))):
            target = _resolve_link_target(raw, graph)
            if target and target != other_source and target not in other_resolved:
                other_resolved.append(target)
        other_node["explicit_links"] = list(other_resolved)
        other_node["inferred_links"] = [item for item in other_node.get("inferred_links", []) if item not in other_resolved and item != other_source]
        other_node["outbound_links"] = other_resolved + [item for item in other_node.get("inferred_links", []) if item not in other_resolved]
        other_node["unresolved_links"] = [raw for raw in (list(other_node.get("raw_links", [])) + list(other_node.get("manual_links", []))) if not _resolve_link_target(raw, graph)]
    node["outbound_links"] = list(dict.fromkeys(node["explicit_links"] + node.get("inferred_links", [])))
    _save_link_graph(graph)
    return node


def _remove_from_link_graph(source: str) -> None:
    graph = _load_link_graph()
    nodes = graph.get("nodes", {})
    if source in nodes:
        del nodes[source]
    for node in nodes.values():
        node["outbound_links"] = [item for item in node.get("outbound_links", []) if item != source]
        node["explicit_links"] = [item for item in node.get("explicit_links", []) if item != source]
        node["inferred_links"] = [item for item in node.get("inferred_links", []) if item != source]
    _save_link_graph(graph)


def _graph_links_for(source: str, graph: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    graph = graph or _load_link_graph()
    nodes = graph.get("nodes", {})
    node = nodes.get(source, {})
    backlinks: List[str] = []
    for other_source, other_node in nodes.items():
        if other_source == source:
            continue
        if source in other_node.get("outbound_links", []) and other_source not in backlinks:
            backlinks.append(other_source)
    return {
        "explicit_links": list(node.get("explicit_links", [])),
        "inferred_links": list(node.get("inferred_links", [])),
        "outbound_links": list(node.get("outbound_links", [])),
        "unresolved_links": list(node.get("unresolved_links", [])),
        "manual_links": list(node.get("manual_links", [])),
        "backlinks": backlinks,
    }


def _graph_snapshot(
    include_semantic: bool = True,
    detailed: bool = True,
    namespace: Optional[str] = None,
    profile: str = "curated",
    include_noise: bool = False,
    include_raw_chats: bool = False,
) -> Dict[str, Any]:
    now = time.monotonic()
    namespace = _normalize_namespace(namespace)
    profile = (profile or "curated").strip().lower()
    if profile not in {"brain", "curated"}:
        profile = "curated"
    cache_key = (
        f"semantic={int(include_semantic)}:detailed={int(detailed)}:"
        f"namespace={namespace or ''}:profile={profile}:noise={int(include_noise)}:raw={int(include_raw_chats)}"
    )
    with _catalog_cache_lock:
        entries = _graph_snapshot_cache.setdefault("entries", {})
        entry = entries.get(cache_key, {})
        if entry.get("value") is not None and entry.get("expires_at", 0.0) > now:
            return entry["value"]
    explicit_scope = bool(namespace)
    all_records = _list_source_records(namespace=namespace)
    records = [
        item for item in all_records
        if _retrieval_primary(
            item,
            explicit_scope=explicit_scope,
            include_noise=include_noise,
            include_raw_chats=include_raw_chats,
        )
        and _graph_profile_allows(item, profile)
    ]
    record_map = {item["source"]: item for item in records}
    graph = _load_link_graph()
    graph_nodes = graph.get("nodes", {})
    activity_map = _load_activity_log().get("sources", {})
    backlinks_by_source: Dict[str, List[str]] = {source: [] for source in graph_nodes}
    for source, node in graph_nodes.items():
        for target in node.get("outbound_links", []):
            if target in backlinks_by_source and source not in backlinks_by_source[target]:
                backlinks_by_source[target].append(source)
    nodes = []
    for item in records:
        node = graph_nodes.get(item["source"], {})
        explicit_links = list(node.get("explicit_links", []))
        inferred_links = list(node.get("inferred_links", []))
        outbound_links = list(node.get("outbound_links", []))
        backlinks = backlinks_by_source.get(item["source"], [])
        activity = activity_map.get(item["source"], {})
        authoring = _authoring_for_source(item["source"])
        summary = (item.get("summary") or "")[:(220 if detailed else 96)]
        aliases = list(authoring.get("aliases") or [])
        article_excerpt = (item.get("article") or "")[:(420 if detailed else 0)]
        search_text = ""
        if detailed:
            search_text = " ".join(
                str(part).strip().lower()
                for part in [
                    item["source"],
                    item.get("title") or "",
                    summary,
                    (item.get("article") or "")[:1200],
                    item.get("smart_collection") or "",
                    item.get("namespace") or "",
                    item.get("type") or "",
                    " ".join(item.get("tags") or []),
                    " ".join(item.get("entities") or []),
                    " ".join(aliases),
                    authoring.get("curation_status") or "",
                    authoring.get("trust_level") or "",
                    authoring.get("review_note") or "",
                ]
                if part
            )
        nodes.append({
            "source": item["source"],
            "title": item.get("title"),
            "summary": summary,
            "type": item.get("type"),
            "namespace": item.get("namespace"),
            "smart_collection": item.get("smart_collection"),
            "chunk_count": item.get("chunk_count", 0),
            "explicit_count": len(explicit_links),
            "inferred_count": len(inferred_links),
            "outbound_count": len(outbound_links),
            "backlink_count": len(backlinks),
            "query_count": int(activity.get("query_count", 0) or 0),
            "retrieve_count": int(activity.get("retrieve_count", 0) or 0),
            "synthesis_count": int(activity.get("synthesis_count", 0) or 0),
            "ingest_count": int(activity.get("ingest_count", 0) or 0),
            "last_activity_at": activity.get("last_activity_at"),
            "pinned": authoring.get("pinned", False),
            "aliases": aliases if detailed else [],
            "alias_count": len(aliases),
            "curation_status": authoring.get("curation_status", "inbox"),
            "trust_level": authoring.get("trust_level", "unreviewed"),
            "quality_score": item.get("quality_score"),
            "quality_flags": list(item.get("quality_flags") or []),
            "retrieval_tier": item.get("retrieval_tier"),
        })
        if article_excerpt:
            nodes[-1]["article_excerpt"] = article_excerpt
        if search_text:
            nodes[-1]["search_text"] = search_text
    edges = []
    seen = set()
    for source, node in graph_nodes.items():
        if source not in record_map:
            continue
        for target in node.get("explicit_links", []):
            if target not in record_map:
                continue
            key = (source, target, "explicit")
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": source, "target": target, "kind": "explicit"})
        for target in node.get("inferred_links", []):
            if target not in record_map:
                continue
            key = (source, target, "inferred")
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": source, "target": target, "kind": "inferred"})
    if include_semantic:
        for edge in _semantic_edges(records, graph_nodes):
            key = (edge["source"], edge["target"], edge["kind"])
            if key in seen or edge["source"] not in record_map or edge["target"] not in record_map:
                continue
            seen.add(key)
            edges.append(edge)
    snapshot = {"nodes": nodes, "edges": edges}
    with _catalog_cache_lock:
        entries = _graph_snapshot_cache.setdefault("entries", {})
        entries[cache_key] = {
            "value": snapshot,
            "expires_at": time.monotonic() + GRAPH_CACHE_TTL_SECONDS,
        }
    return snapshot


def _fallback_title(source: str, doc_type: str, text: str) -> str:
    if source and not source.startswith("http"):
        stem = Path(source).stem or Path(source).name
        if stem:
            return stem.replace("_", " ").replace("-", " ")[:120]
    if source.startswith("http"):
        return source[:120]
    first = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
    return (first or doc_type.title())[:120]


def _merge_tags(*tag_sets: List[str]) -> List[str]:
    merged: List[str] = []
    for tags in tag_sets:
        for tag in tags or []:
            clean = re.sub(r"\s+", "-", str(tag).strip().lower())
            clean = re.sub(r"[^a-z0-9_./-]", "", clean)
            if clean and clean not in merged:
                merged.append(clean)
    return merged[:SEMANTIC_TAG_LIMIT]


def _extract_entities_fallback(text: str) -> List[str]:
    candidates = re.findall(r"\b([A-Z][a-zA-Z0-9_.-]+(?:\s+[A-Z][a-zA-Z0-9_.-]+){0,3})\b", text[:5000])
    entities: List[str] = []
    for item in candidates:
        value = item.strip()
        if len(value) < 3 or value.lower() in {"the", "this", "that"}:
            continue
        if value not in entities:
            entities.append(value)
        if len(entities) >= ENTITY_LIMIT:
            break
    return entities


def _infer_smart_collection(source: str, doc_type: str, namespace: str, tags: List[str], entities: List[str]) -> str:
    if source.startswith("claude/") or doc_type == "claude-chat":
        return "assistant-history"
    if source.startswith("http") or doc_type in {"url", "html"}:
        return "web-research"
    if doc_type in {"pdf", "docx"}:
        return "reference-docs"
    if doc_type in {"email", "pst", "eml", "mbox"}:
        return "communications"
    if doc_type in {"obsidian", "markdown", "note"}:
        return "notes"
    if doc_type == "code":
        return "code-context"
    for entity in entities:
        if any(ch.isdigit() for ch in entity) or len(entity.split()) >= 2:
            return _safe_slug(entity, namespace or "general")
    for tag in tags:
        if tag not in {doc_type, namespace, "page"}:
            return _safe_slug(tag, namespace or "general")
    return namespace or "general"


def _semantic_metadata(text: str, source: str, doc_type: str, namespace: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    seed_tags = list(tags or [])
    fallback_title = _fallback_title(source, doc_type, text)
    excerpt = text[:6000]
    prompt = (
        "You organize personal knowledge like an AI memory product.\n"
        "Return strict JSON with keys: title, summary, tags, entities, smart_collection.\n"
        "Rules:\n"
        "- title: concise and specific, max 10 words\n"
        "- summary: 1 sentence, max 220 chars\n"
        f"- tags: array of {SEMANTIC_TAG_LIMIT} or fewer lowercase topical tags\n"
        f"- entities: array of {ENTITY_LIMIT} or fewer important people, orgs, products, systems, projects, tickets, or places\n"
        "- smart_collection: short slug-like project/topic bucket\n"
        "- infer from content, do not mention JSON or formatting\n"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({
                "source": source,
                "doc_type": doc_type,
                "namespace": namespace,
                "existing_tags": seed_tags,
                "content": excerpt,
            })},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(f"{VLLM_LLM_BASE_URL}/chat/completions", json=payload, timeout=45)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        logger.warning(f"semantic organizer fallback for {source}: {exc}")
        merged_tags = _merge_tags(seed_tags, [doc_type, namespace])
        entities = _extract_entities_fallback(text)
        return {
            "title": fallback_title,
            "summary": excerpt[:220].replace("\n", " ").strip(),
            "tags": merged_tags,
            "entities": entities,
            "smart_collection": _infer_smart_collection(source, doc_type, namespace, merged_tags, entities),
        }

    model_tags = parsed.get("tags") if isinstance(parsed.get("tags"), list) else []
    model_entities = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
    merged_tags = _merge_tags(seed_tags, model_tags, [doc_type, namespace])
    entities = []
    for item in model_entities:
        clean = str(item).strip()
        if clean and clean not in entities:
            entities.append(clean)
        if len(entities) >= ENTITY_LIMIT:
            break
    summary = " ".join(str(parsed.get("summary", "")).split())[:220]
    title = " ".join(str(parsed.get("title", "")).split())[:120] or fallback_title
    smart_collection = _safe_slug(parsed.get("smart_collection", ""), "")
    if not smart_collection:
        smart_collection = _infer_smart_collection(source, doc_type, namespace, merged_tags, entities)
    return {
        "title": title,
        "summary": summary or excerpt[:220].replace("\n", " ").strip(),
        "tags": merged_tags,
        "entities": entities,
        "smart_collection": smart_collection,
    }


def _knowledge_article(text: str, source: str, title: str, summary: str) -> str:
    excerpt = (text or "")[:14000]
    if not excerpt.strip():
        return ""
    system_prompt = (
        "You write clean internal knowledge-base articles for a second-brain system.\n"
        "Return plain markdown only.\n"
        "Use short sections with concrete details.\n"
        "Do not invent facts not present in the source material.\n"
    )
    user_prompt = (
        "Create a full knowledge-base article from this source.\n"
        "Structure it as:\n"
        "# Title\n"
        "## Overview\n"
        "## Key Details\n"
        "## Important Entities\n"
        "## Actions or Risks\n"
        "## Search Keywords\n\n"
        f"Source: {source}\n"
        f"Current title: {title}\n"
        f"Current summary: {summary}\n\n"
        f"Content:\n{excerpt}"
    )
    try:
        return _chat_completion(system_prompt, user_prompt, temperature=0.1, max_tokens=1200)
    except Exception as exc:
        logger.warning(f"knowledge article fallback for {source}: {exc}")
        fallback = [f"# {title}", "", "## Overview", summary or _strip_think_traces(excerpt[:320]).replace("\n", " ")]
        entities = _extract_entities_fallback(excerpt)
        if entities:
            fallback.extend(["", "## Important Entities", ", ".join(entities[:ENTITY_LIMIT])])
        fallback.extend(["", "## Key Details", _strip_think_traces(excerpt[:2400])])
        return "\n".join(fallback).strip()


def _apply_semantic_metadata(docs: List[Document], source: str, doc_type: str, namespace: str, tags: Optional[List[str]] = None) -> List[Document]:
    if not docs:
        return docs
    combined = "\n\n".join(doc.page_content[:2500] for doc in docs[:4])
    semantic = _semantic_metadata(combined, source, doc_type, namespace, tags=tags)
    article = _knowledge_article(combined, source, semantic["title"], semantic["summary"])
    raw_links = []
    for doc in docs:
        for link in doc.metadata.get("link_targets", []) or []:
            if link not in raw_links:
                raw_links.append(link)
    link_node = _upsert_link_graph(source, semantic["title"], raw_links)
    for idx, doc in enumerate(docs, start=1):
        doc.metadata.setdefault("source", source)
        doc.metadata.setdefault("type", doc_type)
        doc.metadata.setdefault("namespace", namespace)
        doc.metadata["title"] = semantic["title"]
        doc.metadata["summary"] = semantic["summary"]
        doc.metadata["article"] = article
        doc.metadata["tags"] = _merge_tags(doc.metadata.get("tags", []), semantic.get("tags", []))
        doc.metadata["entities"] = semantic.get("entities", [])
        doc.metadata["smart_collection"] = semantic.get("smart_collection", namespace)
        doc.metadata["explicit_links"] = list(link_node.get("explicit_links", []))
        doc.metadata["inferred_links"] = list(link_node.get("inferred_links", []))
        doc.metadata["outbound_links"] = list(link_node.get("outbound_links", []))
        doc.metadata["unresolved_links"] = list(link_node.get("unresolved_links", []))
        if "page" in doc.metadata:
            doc.metadata["page_label"] = f"Page {doc.metadata['page']}"
        doc.metadata["chunk_role"] = "source-root" if idx == 1 else "supporting-chunk"
    return docs


SUPPORTED_EXTS = {
    ".pdf", ".docx", ".html", ".htm",
    ".json", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".webm",
    ".txt", ".md", ".rst", ".csv", ".log",
    ".pst", ".eml", ".mbox",
}


JUNK_SOURCE_DIRS = {
    ".cache", ".eggs", ".git", ".hg", ".ipynb_checkpoints", ".mypy_cache",
    ".next", ".nox", ".nuxt", ".pytest_cache", ".ruff_cache", ".svn",
    ".svelte-kit", ".tox", ".turbo", ".venv", "__macosx", "__pycache__",
    "bower_components", "build", "coverage", "dist", "env", "htmlcov",
    "node_modules", "site-packages", "target", "vendor", "venv",
}
JUNK_SOURCE_FILE_NAMES = {
    "cargo.lock", "composer.lock", "gemfile.lock", "package-lock.json",
    "pipfile.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "yarn.lock",
}
JUNK_SOURCE_EXTS = {
    ".bak", ".cache", ".class", ".coverage", ".egg-info", ".lock", ".log",
    ".map", ".min.js", ".o", ".obj", ".orig", ".pid", ".pyc", ".pyo",
    ".so", ".sqlite-shm", ".sqlite-wal", ".tmp",
}
JUNK_SOURCE_SUFFIXES = (
    ".dist-info",
    ".egg-info",
    ".log.",
)


def _source_path_parts(source: str) -> List[str]:
    normalized = str(source or "").replace("\\", "/").strip()
    if not normalized:
        return []
    parsed = urlparse(normalized)
    if parsed.scheme in {"http", "https"}:
        normalized = parsed.path
    return [part.lower() for part in normalized.split("/") if part]


def _junk_source_reason(source: str, doc_type: str = "") -> Optional[str]:
    parts = _source_path_parts(source)
    if not parts:
        return None
    lowered = str(source or "").lower()
    normalized = f"/{str(source or '').replace(chr(92), '/').lower().strip('/')}/"
    for term in TRASH_SOURCE_TERMS:
        if term and term in normalized:
            return f"trash source term: {term}"
    for part in parts:
        if part in JUNK_SOURCE_DIRS:
            return f"junk path segment: {part}"
        if any(part.endswith(suffix) for suffix in JUNK_SOURCE_SUFFIXES):
            return f"junk metadata directory: {part}"
    name = parts[-1]
    if name in JUNK_SOURCE_FILE_NAMES:
        return f"junk lockfile: {name}"
    suffixes = Path(name).suffixes
    suffix = "".join(suffixes[-2:]) if len(suffixes) >= 2 and suffixes[-2] in {".min", ".sqlite"} else Path(name).suffix.lower()
    if suffix in JUNK_SOURCE_EXTS:
        return f"junk file extension: {suffix}"
    if doc_type == "text" and (name.endswith(".log") or ".log." in lowered):
        return "junk log file"
    return None


def _line_repetition_flags(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 6]
    if len(lines) < 40:
        return []
    counts: Dict[str, int] = {}
    for line in lines[:5000]:
        counts[line] = counts.get(line, 0) + 1
    unique_ratio = len(counts) / max(1, len(lines))
    max_repeat = max(counts.values()) if counts else 0
    flags: List[str] = []
    if unique_ratio < 0.28:
        flags.append("repetitive")
    if max_repeat >= 12:
        flags.append("line-spam")
    return flags


def _source_quality_profile(text: str, source: str, doc_type: str, namespace: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    clean = _strip_think_traces(text or "")
    lowered = clean[:250000].lower()
    tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
    flags: List[str] = []
    score = 100

    junk_reason = _junk_source_reason(source, doc_type)
    if junk_reason:
        flags.append("junk-source")
        score -= 100
    if tag_set & TRASH_TAGS:
        flags.append("trash-tag")
        score -= 70
    if _is_raw_context_record(namespace, source, list(tag_set)):
        flags.append("raw-chat")
        score -= 18
    if namespace in NOISE_NAMESPACES:
        flags.append("noisy-namespace")
        score -= 30
    if len(clean) < 200:
        flags.append("thin")
        score -= 25
    if len(clean) > TRASH_MAX_TEXT_CHARS:
        flags.append("oversized")
        score -= 18

    for flag in _line_repetition_flags(clean):
        if flag not in flags:
            flags.append(flag)
        score -= 28 if flag == "repetitive" else 22

    tool_markers = sum(lowered.count(marker) for marker in ("[tool output]", "[codex tool output]", "tool_result", "tool output", "stdout", "stderr"))
    if tool_markers >= 12:
        flags.append("tool-output-heavy")
        score -= 24
    elif tool_markers >= 4:
        flags.append("tool-output")
        score -= 10

    error_markers = sum(lowered.count(marker) for marker in (
        "traceback ",
        "connection refused",
        "networkerror",
        "internalservererror",
        "you've hit your limit",
        "you have hit your limit",
        "404 not found",
        "failed after",
    ))
    if error_markers >= 4:
        flags.append("error-heavy")
        score -= 20
    elif error_markers:
        flags.append("has-errors")
        score -= 8

    source_lower = str(source or "").lower()
    if any(term in source_lower for term in ("/downloads/", "/tmp/", "/temp/", "untitled", "scratch")):
        flags.append("loose-source")
        score -= 8

    score = max(0, min(100, score))
    if score <= 30 or "junk-source" in flags or "trash-tag" in flags:
        tier = "noise"
    elif score < 55:
        tier = "low"
    elif score < 78:
        tier = "medium"
    else:
        tier = "high"
    return {
        "quality_score": score,
        "quality_flags": list(dict.fromkeys(flags))[:16],
        "retrieval_tier": tier,
    }


def _apply_quality_metadata(doc: Document, source: str, doc_type: str, namespace: str, tags: Optional[List[str]] = None) -> None:
    profile = _source_quality_profile(doc.page_content, source, doc_type, namespace, tags=tags)
    doc.metadata["quality_score"] = profile["quality_score"]
    doc.metadata["quality_flags"] = profile["quality_flags"]
    doc.metadata["retrieval_tier"] = profile["retrieval_tier"]
    doc.metadata["tags"] = _merge_tags(
        doc.metadata.get("tags", []),
        [f"quality-{profile['retrieval_tier']}"],
        [f"flag-{flag}" for flag in profile["quality_flags"][:6]],
    )


def _dispatch_file(data: bytes, filename: str) -> tuple[str, str]:
    """Return (text, doc_type) for any supported file. Raises ValueError if unsupported/empty."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data), "pdf"
    elif ext == ".docx":
        return _extract_docx(data), "docx"
    elif ext in (".html", ".htm"):
        return _extract_html(data), "html"
    elif ext == ".json":
        return _extract_structured(data, ext), "json"
    elif ext in (".yaml", ".yml"):
        return _extract_structured(data, ext), "yaml"
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        return _extract_image(data), "image"
    elif ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".webm"):
        return _extract_audio(data, ext), "audio"
    elif ext in (".txt", ".md", ".rst", ".csv", ".log", ""):
        return data.decode("utf-8", errors="replace"), "text"
    elif ext == ".pst":
        return _extract_pst(data), "pst"
    elif ext == ".eml":
        return _extract_eml(data), "eml"
    elif ext == ".mbox":
        return _extract_mbox(data), "mbox"
    else:
        raise ValueError(f"unsupported extension: {ext}")


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()


def _extract_pdf_pages(data: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _extract_pdf_pages_via_nvingest(data: bytes) -> Optional[List[str]]:
    """Run the PDF through nv-ingest (text + tables + chart captions).

    Returns one combined text per page, or None on any failure so callers
    can fall back to PyPDF.
    """
    logger.info('nv-ingest: extract attempt (NV_INGEST_ENABLED=%s, host=%s:%s)', NV_INGEST_ENABLED, NV_INGEST_HOST, NV_INGEST_PORT)
    if not NV_INGEST_ENABLED:
        logger.info('nv-ingest: disabled by config')
        return None
    try:
        from nv_ingest_client.client import NvIngestClient, Ingestor
        from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
    except ImportError as e:
        logger.warning('nv_ingest_client not available (%s); falling back to PyPDF', e)
        return None

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            client = NvIngestClient(
                message_client_allocator=SimpleClient,
                message_client_hostname=NV_INGEST_HOST,
                message_client_port=int(NV_INGEST_PORT),
            )
            ingestor = (
                Ingestor(documents=[tmp.name], client=client)
                .extract(
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_images=False,
                    text_depth='page',
                )
            )
            results = ingestor.ingest(show_progress=False)
        except Exception as e:
            logger.warning('nv-ingest extraction failed: %r; falling back to PyPDF', e)
            return None

    if not results or not isinstance(results, list):
        return None

    pages = {}
    for primitive in results[0]:
        if not isinstance(primitive, dict):
            continue
        meta = primitive.get('metadata') or {}
        cm = meta.get('content_metadata') or {}
        page_idx = cm.get('page_number')
        if page_idx is None:
            page_idx = 0
        content_type = cm.get('type') or primitive.get('document_type') or 'text'
        content = meta.get('content') or ''
        if not isinstance(content, str) or not content.strip():
            tm = meta.get('table_metadata') or {}
            cmk = meta.get('chart_metadata') or {}
            content = (
                tm.get('table_content')
                or cmk.get('caption')
                or cmk.get('table_content')
                or ''
            )
        if not isinstance(content, str) or not content.strip():
            continue
        pages.setdefault(int(page_idx), []).append((str(content_type), content.strip()))

    if not pages:
        return None
    out = []
    for page_num in sorted(pages.keys()):
        parts = []
        for ctype, text in pages[page_num]:
            if ctype in ('text', 'structured'):
                parts.append(text)
            elif ctype == 'table':
                parts.append('[Table]\n' + text)
            elif ctype in ('chart', 'image'):
                parts.append('[Figure]\n' + text)
            else:
                parts.append(text)
        if parts:
            out.append('\n\n'.join(parts))
    return out or None


def _parse_obsidian_note(path: str, content: str) -> Document:
    """Parse a single Obsidian markdown note into a LangChain Document."""
    try:
        post = fm.loads(content)
        meta = dict(post.metadata)
        body = post.content
    except Exception:
        meta = {}
        body = content

    # Normalise frontmatter tags (string or list)
    tags = meta.get("tags", meta.get("tag", []))
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.replace(",", " ").split()]

    # Collect inline #tags from body
    inline_tags = re.findall(r"(?<!\w)#([A-Za-z0-9_/-]+)", body)
    all_tags = list(dict.fromkeys(tags + inline_tags))  # dedup, preserve order
    explicit_links = _extract_explicit_links(body)

    # Convert wikilinks [[target|alias]] → alias or target
    body = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)

    # Strip remaining markdown syntax that adds no semantic value
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)   # images
    body = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", body)  # links → text

    title = meta.get("title") or Path(path).stem.replace("-", " ").replace("_", " ")
    full_text = f"# {title}\n\n{body.strip()}" if body.strip() else f"# {title}"

    return Document(
        page_content=full_text,
        metadata={
            "source": path,
            "type": "obsidian",
            "title": title,
            "tags": all_tags,
            "link_targets": explicit_links,
            "aliases": meta.get("aliases", []),
            "namespace": "notes",
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def _extract_docx(data: bytes) -> str:
    doc = docx.Document(io.BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(c.text.strip() for c in row.cells if c.text.strip()))
    return "\n\n".join(parts)


def _extract_image(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img).strip()


def _extract_audio(data: bytes, suffix: str) -> str:
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        result = model.transcribe(tmp)
        return result["text"].strip()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _extract_structured(data: bytes, ext: str) -> str:
    text = data.decode("utf-8", errors="replace")
    try:
        if ext == ".json":
            obj = json.loads(text)
            return json.dumps(obj, indent=2)
        elif ext in (".yaml", ".yml"):
            obj = yaml.safe_load(text)
            return json.dumps(obj, indent=2)
    except Exception:
        pass
    return text


def _email_to_text(msg: email.message.Message) -> str:
    """Extract subject, sender, date, and body from a single email message."""
    parts = []
    subject = msg.get("Subject", "")
    sender  = msg.get("From", "")
    date    = msg.get("Date", "")
    to      = msg.get("To", "")
    if subject:
        parts.append(f"Subject: {subject}")
    if sender:
        parts.append(f"From: {sender}")
    if to:
        parts.append(f"To: {to}")
    if date:
        parts.append(f"Date: {date}")
    parts.append("")  # blank line before body

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                charset = part.get_content_charset() or "utf-8"
                try:
                    parts.append(part.get_payload(decode=True).decode(charset, errors="replace"))
                except Exception:
                    pass
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            parts.append(msg.get_payload(decode=True).decode(charset, errors="replace"))
        except Exception:
            pass

    return "\n".join(parts).strip()


def _extract_eml(data: bytes) -> str:
    msg = email.message_from_bytes(data)
    return _email_to_text(msg)


def _extract_mbox(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mbox", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        mbox = mailbox.mbox(tmp)
        texts = [_email_to_text(msg) for msg in mbox]
        return "\n\n---\n\n".join(t for t in texts if t)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _extract_pst(data: bytes) -> str:
    """Convert PST → mbox via readpst, then parse all emails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pst_path = Path(tmpdir) / "archive.pst"
        pst_path.write_bytes(data)
        result = subprocess.run(
            ["readpst", "-o", tmpdir, "-r", "-M", str(pst_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"readpst stderr: {result.stderr[:500]}")

        all_texts: list[str] = []
        # readpst -M writes individual .eml files; -r writes mbox per folder
        for p in Path(tmpdir).rglob("*"):
            if p.suffix == ".eml" and p.is_file():
                try:
                    msg = email.message_from_bytes(p.read_bytes())
                    t = _email_to_text(msg)
                    if t:
                        all_texts.append(t)
                except Exception:
                    pass
            elif p.suffix == "" and p.is_file():
                # readpst may write mbox files without extension
                try:
                    mbox = mailbox.mbox(str(p))
                    for msg in mbox:
                        t = _email_to_text(msg)
                        if t:
                            all_texts.append(t)
                except Exception:
                    pass

    if not all_texts:
        raise ValueError("PST contained no readable email text")
    return "\n\n---\n\n".join(all_texts)


def _detect_source_type(text: str) -> tuple[str, str]:
    """
    Inspect raw text and return (doc_type, source_label).
    Called when no explicit source is provided to /ingest/text.
    """
    # Sample the first 3000 chars for pattern matching
    sample = text[:3000]
    lines  = [l.strip() for l in sample.splitlines() if l.strip()]

    # ── Claude Code / Claude chat ─────────────────────────────────────────────
    # claude_sync.py writes "User: …\n\nClaude: …" transcripts
    claude_lines = sum(1 for l in lines if re.match(r'^(User|Claude):', l))
    if claude_lines >= 2:
        return "claude-chat", "claude-chat"

    # ── ChatGPT / OpenAI ──────────────────────────────────────────────────────
    chatgpt_lines = sum(1 for l in lines if re.match(r'^(You|ChatGPT|GPT-\d|Assistant|Human):', l))
    if chatgpt_lines >= 2:
        return "chatgpt", "chatgpt"

    # ── GitHub Copilot (markdown Q&A blocks) ─────────────────────────────────
    if re.search(r'github\.com/copilot|copilot chat|#+ Copilot', sample, re.IGNORECASE):
        return "copilot", "copilot"

    # ── Codex / code-heavy content ────────────────────────────────────────────
    code_fences = len(re.findall(r'```', sample))
    code_lines  = sum(1 for l in lines if l.startswith(('def ', 'class ', 'import ', 'from ', 'const ', 'function ', 'export ')))
    if code_lines >= 4 or code_fences >= 4:
        return "code", "code"

    # ── Slack export (JSON-ish with user/text keys) ───────────────────────────
    if re.search(r'"user_profile"\s*:', sample) or re.search(r'"username"\s*:.*"text"\s*:', sample, re.DOTALL):
        return "slack", "slack"

    # ── Email-style text ──────────────────────────────────────────────────────
    email_headers = sum(1 for l in lines if re.match(r'^(From|To|Subject|Date|Cc):', l))
    if email_headers >= 2:
        return "email", "email"

    # ── Markdown document ─────────────────────────────────────────────────────
    md_headings = sum(1 for l in lines if l.startswith('#'))
    if md_headings >= 2:
        return "markdown", "note"

    return "text", "paste"


def _extract_html(data: bytes) -> str:
    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_url(url: str) -> str:
    current = _validate_fetch_url(url)
    for _ in range(MAX_URL_REDIRECTS + 1):
        resp = requests.get(
            current,
            timeout=(5, 15),
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=False,
            stream=True,
        )
        try:
            if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                current = _validate_fetch_url(urljoin(current, resp.headers["location"]))
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if content_type and not any(token in content_type for token in ("text/", "html", "xml", "json")):
                raise HTTPException(415, "URL response is not a supported text document")
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_URL_BYTES:
                    raise HTTPException(413, f"URL response is too large; limit is {MAX_URL_BYTES} bytes")
                chunks.append(chunk)
            encoding = resp.encoding or "utf-8"
            text = b"".join(chunks).decode(encoding, errors="replace")
            break
        finally:
            resp.close()
    else:
        raise HTTPException(400, "URL redirected too many times")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _build_documents_from_file(data: bytes, filename: str, namespace: str) -> tuple[List[Document], bool, str]:
    """
    Return documents, whether the documents are already page-sized, and the detected type.

    PDFs are emitted as page-scoped documents so retrieval can stay close to NVIDIA's
    recommended page-first chunking strategy for paginated files.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        docs: List[Document] = []
        pages = _extract_pdf_pages_via_nvingest(data) or _extract_pdf_pages(data)
        for idx, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            docs.append(_make_doc(
                page_text,
                filename,
                "pdf",
                extra={"page": idx, "tags": ["pdf", "page"]},
                namespace=namespace,
            ))
        if not docs:
            raise ValueError("No text extracted from PDF")
        return docs, True, "pdf"

    text, doc_type = _dispatch_file(data, filename)
    if not text.strip():
        raise ValueError("No text extracted from file")
    return [_make_doc(text, filename, doc_type, namespace=namespace)], False, doc_type


def _iter_rows(filt: str, output_fields: List[str], batch_size: int = 1000):
    """Stream every matching row via Milvus' query_iterator.

    Plain `query(..., limit, offset)` is capped at `limit + offset <= 16384`,
    so paginating with offset blows up past ~16K chunks. The iterator API has
    no such cap.
    """
    if mclient is None:
        return
    it = mclient.query_iterator(
        collection_name=MILVUS_COLLECTION,
        filter=filt,
        output_fields=output_fields,
        batch_size=batch_size,
    )
    try:
        while True:
            rows = it.next()
            if not rows:
                return
            for row in rows:
                yield row
    finally:
        try:
            it.close()
        except Exception:
            pass


def _list_source_records(namespace: Optional[str] = None) -> List[dict]:
    if mclient is None:
        return []
    cache_key = namespace or "__all__"
    now = time.monotonic()
    with _catalog_cache_lock:
        cached = _source_records_cache.get(cache_key)
        if cached and cached.get("expires_at", 0.0) > now:
            return [dict(item) for item in cached.get("value", [])]
    activity_sources = _load_activity_log().get("sources", {})
    catalog_records = _load_source_catalog_records()
    if catalog_records:
        records = []
        for item in catalog_records:
            if namespace and item.get("namespace") != namespace:
                continue
            record = dict(item)
            activity = activity_sources.get(record["source"], {})
            authoring = _authoring_for_source(record["source"])
            record["query_count"] = int(activity.get("query_count", 0) or 0)
            record["retrieve_count"] = int(activity.get("retrieve_count", 0) or 0)
            record["synthesis_count"] = int(activity.get("synthesis_count", 0) or 0)
            record["ingest_count"] = int(activity.get("ingest_count", 0) or 0)
            record["last_activity_at"] = activity.get("last_activity_at")
            record["pinned"] = authoring.get("pinned", False)
            record["aliases"] = authoring.get("aliases", [])
            record["curation_status"] = authoring.get("curation_status", "inbox")
            record["trust_level"] = authoring.get("trust_level", "unreviewed")
            record["review_note"] = authoring.get("review_note", "")
            records.append(record)
        with _catalog_cache_lock:
            _source_records_cache[cache_key] = {
                "value": [dict(item) for item in records],
                "expires_at": time.monotonic() + GRAPH_CACHE_TTL_SECONDS,
            }
        return records
    seen: Dict[str, dict] = {}
    namespace = _normalize_namespace(namespace)
    filt = _milvus_eq("namespace", namespace) if namespace else ""
    for row in _iter_rows(
        filt,
        ["source", "doc_type", "ingested_at", "namespace", "title", "summary", "tags", "entities", "smart_collection", "quality_score", "quality_flags", "retrieval_tier"],
    ):
        src = row.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source":      src,
                "type":        row.get("doc_type", "unknown"),
                "ingested_at": row.get("ingested_at", ""),
                "namespace":   row.get("namespace", "inbox"),
                "chunk_count": 0,
                "title":       row.get("title"),
                "summary":     row.get("summary"),
                "tags":        list(row.get("tags") or []),
                "entities":    list(row.get("entities") or []),
                "smart_collection": row.get("smart_collection"),
                "quality_score": row.get("quality_score"),
                "quality_flags": list(row.get("quality_flags") or []),
                "retrieval_tier": row.get("retrieval_tier"),
            }
        seen[src]["chunk_count"] += 1
    records = sorted(seen.values(), key=lambda x: x["ingested_at"], reverse=True)
    for record in records:
        activity = activity_sources.get(record["source"], {})
        authoring = _authoring_for_source(record["source"])
        record["query_count"] = int(activity.get("query_count", 0) or 0)
        record["retrieve_count"] = int(activity.get("retrieve_count", 0) or 0)
        record["synthesis_count"] = int(activity.get("synthesis_count", 0) or 0)
        record["ingest_count"] = int(activity.get("ingest_count", 0) or 0)
        record["last_activity_at"] = activity.get("last_activity_at")
        record["pinned"] = authoring.get("pinned", False)
        record["aliases"] = authoring.get("aliases", [])
        record["curation_status"] = authoring.get("curation_status", "inbox")
        record["trust_level"] = authoring.get("trust_level", "unreviewed")
        record["review_note"] = authoring.get("review_note", "")
    if namespace is None:
        _save_source_catalog_records([{k: v for k, v in item.items() if k not in {"query_count", "retrieve_count", "synthesis_count", "ingest_count", "last_activity_at"}} for item in records])
    with _catalog_cache_lock:
        _source_records_cache[cache_key] = {
            "value": [dict(item) for item in records],
            "expires_at": time.monotonic() + GRAPH_CACHE_TTL_SECONDS,
        }
    return records


def _related_sources(source: str, summary: str, namespace: Optional[str], limit: int = 5) -> List[str]:
    if not summary or vectordb is None:
        return []
    try:
        docs = vectordb.similarity_search(summary, k=limit + 5, namespace=namespace)
    except Exception as exc:
        logger.warning(f"related lookup failed for {source}: {exc}")
        return []
    related: List[str] = []
    for doc in docs:
        candidate = doc.metadata.get("source", "")
        if not candidate or candidate == source or candidate in related:
            continue
        related.append(candidate)
        if len(related) >= limit:
            break
    return related


def _source_rows(source: str, output_fields: Optional[List[str]] = None) -> List[dict]:
    if mclient is None:
        return []
    fields = output_fields or ["text", "source", "doc_type", "ingested_at", "namespace", "title", "summary", "article", "tags", "entities", "smart_collection", "page", "page_label", "chunk_role"]
    rows = list(_iter_rows(_milvus_eq("source", source), fields, batch_size=256))
    rows.sort(key=lambda row: (row.get("page") or 0, row.get("page_label") or "", row.get("chunk_role") or "", row.get("ingested_at") or ""))
    return rows


def _load_graph_layout() -> Dict[str, Any]:
    payload = _load_json_file(GRAPH_LAYOUT_PATH, {"positions": {}})
    if not isinstance(payload, dict):
        return {"positions": {}}
    payload.setdefault("positions", {})
    return payload


def _save_graph_layout(positions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    clean: Dict[str, Dict[str, float]] = {}
    for source, item in positions.items():
        try:
            x = float(item.get("x", 0))
            y = float(item.get("y", 0))
        except Exception:
            continue
        clean[source] = {"x": round(x, 2), "y": round(y, 2)}
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "positions": clean,
    }
    _save_json_file(GRAPH_LAYOUT_PATH, payload)
    return payload


def _load_graph_views() -> List[Dict[str, Any]]:
    payload = _load_json_file(GRAPH_VIEWS_PATH, {"views": []})
    views = payload.get("views", []) if isinstance(payload, dict) else []
    return [item for item in views if isinstance(item, dict) and item.get("name")]


def _save_graph_views(views: List[Dict[str, Any]]) -> None:
    _save_json_file(
        GRAPH_VIEWS_PATH,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "views": views,
        },
    )


def _source_record(source: str) -> Optional[dict]:
    records = [item for item in _list_source_records() if item["source"] == source]
    return records[0] if records else None


def _source_text(source: str, max_chars: int = 20000) -> Dict[str, Any]:
    rows = _source_rows(source)
    if not rows:
        raise HTTPException(404, f"Source not found: {source}")
    record = _source_record(source) or {}
    parts: List[str] = []
    total = 0
    for row in rows:
        chunk = row.get("text", "")
        prefix = row.get("page_label") or ""
        text = (f"{prefix}\n{chunk}" if prefix else chunk).strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        text = text[:remaining]
        parts.append(text)
        total += len(text)
    base = rows[0]
    return {
        "source": source,
        "title": record.get("title") or base.get("title") or _fallback_title(source, base.get("doc_type", "text"), parts[0] if parts else ""),
        "summary": record.get("summary") or base.get("summary") or "",
        "article": record.get("article") or base.get("article") or "",
        "type": base.get("doc_type", "text"),
        "namespace": record.get("namespace") or base.get("namespace", "inbox"),
        "tags": list(record.get("tags") or base.get("tags") or []),
        "entities": list(record.get("entities") or base.get("entities") or []),
        "smart_collection": record.get("smart_collection") or base.get("smart_collection"),
        "text": "\n\n".join(parts),
    }


def _save_source_catalog_record(record: Dict[str, Any]) -> None:
    record_map = {item["source"]: dict(item) for item in _load_source_catalog_records()}
    record_map[record["source"]] = {
        **record_map.get(record["source"], {}),
        **record,
    }
    _save_source_catalog_records(list(record_map.values()))


def _reanalyze_source(source: str) -> Dict[str, Any]:
    section = _source_text(source, max_chars=24000)
    semantic = _semantic_metadata(
        section.get("text", ""),
        source,
        section.get("type", "text"),
        section.get("namespace", "inbox"),
        tags=section.get("tags") or [],
    )
    article = _knowledge_article(section.get("text", ""), source, semantic["title"], semantic["summary"])
    record = _source_record(source) or {}
    updated = {
        "source": source,
        "type": section.get("type", record.get("type", "text")),
        "ingested_at": record.get("ingested_at", ""),
        "namespace": section.get("namespace", record.get("namespace", "inbox")),
        "chunk_count": int(record.get("chunk_count", 0) or len(_source_rows(source, ["source"])) or 0),
        "title": semantic["title"],
        "summary": semantic["summary"],
        "article": article,
        "tags": semantic.get("tags", []),
        "entities": semantic.get("entities", []),
        "smart_collection": semantic.get("smart_collection"),
    }
    _save_source_catalog_record(updated)
    raw_links = _extract_explicit_links(section.get("text", ""))
    _upsert_link_graph(source, updated["title"], raw_links)
    return updated


def _run_reanalysis_job(sources: List[str]) -> None:
    with _reanalyze_job_lock:
        _reanalyze_job.update({
            "running": True,
            "completed": 0,
            "total": len(sources),
            "errors": [],
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
        })
    for source in sources:
        try:
            _reanalyze_source(source)
        except Exception as exc:
            logger.warning(f"reanalysis failed for {source}: {exc}")
            with _reanalyze_job_lock:
                _reanalyze_job["errors"].append(f"{source}: {exc}")
        finally:
            with _reanalyze_job_lock:
                _reanalyze_job["completed"] = int(_reanalyze_job.get("completed", 0) or 0) + 1
    _save_json_file(SEMANTIC_GRAPH_PATH, {"signatures": {}, "neighbors": {}})
    _invalidate_catalog_cache()
    try:
        rebuild_links()
    finally:
        with _reanalyze_job_lock:
            _reanalyze_job["running"] = False
            _reanalyze_job["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _collection_source_names(smart_collection: str, max_sources: int = 12) -> List[str]:
    names: List[str] = []
    for item in _list_source_records():
        if item.get("smart_collection") == smart_collection and item["source"] not in names:
            names.append(item["source"])
        if len(names) >= max_sources:
            break
    return names


def _llm_fallback_message(reason: str) -> str:
    return (
        "The LLM backend did not complete the generation request. "
        f"{reason} Retrieval and source browsing are still available; retry when the model endpoint is responsive."
    )


def _chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 700, timeout: int = LLM_CHAT_TIMEOUT_SECONDS) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(f"{VLLM_LLM_BASE_URL}/chat/completions", json=payload, timeout=(10, timeout))
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("LLM backend timed out after %ss: %s", timeout, VLLM_LLM_BASE_URL)
        return _llm_fallback_message(f"The model endpoint timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as exc:
        logger.warning("LLM backend request failed: %s", exc)
        return _llm_fallback_message("The model endpoint returned a transport error.")
    message = response.json()["choices"][0]["message"]
    content = (message.get("content") or message.get("reasoning") or "").strip()
    content = _strip_think_traces(content)
    return content.strip()


def _coerce_json_object(text: str) -> Dict[str, Any]:
    raw = _strip_think_traces(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def _compact_text(value: Optional[str], limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _uiux_agent_context(selected_source: Optional[str] = None) -> Dict[str, Any]:
    records = _list_source_records()
    analytics = retrieval_analytics()
    collections = list_smart_collections()
    detail = None
    if selected_source:
        try:
            detail_obj = source_detail(selected_source)
            detail = detail_obj.model_dump() if hasattr(detail_obj, "model_dump") else detail_obj.dict()
        except Exception:
            detail = None
    return {
        "health": health(),
        "config": config(),
        "source_count": len(records),
        "top_namespaces": [
            {
                "namespace": item.get("namespace"),
                "source_count": item.get("source_count"),
                "chunk_count": item.get("chunk_count"),
            }
            for item in list_namespaces()[:8]
        ],
        "top_collections": [
            {
                "smart_collection": item.get("smart_collection"),
                "source_count": item.get("source_count"),
                "chunk_count": item.get("chunk_count"),
            }
            for item in collections[:8]
        ],
        "recent_queries": [
            {
                "query": item.get("query"),
                "sources": item.get("sources", [])[:4],
            }
            for item in analytics.get("recent_queries", [])[:5]
        ],
        "top_notes": [
            {
                "title": item.get("title") or item.get("source", ""),
                "source": item.get("source", ""),
                "query_count": item.get("query_count", 0),
                "retrieve_count": item.get("retrieve_count", 0),
                "synthesis_count": item.get("synthesis_count", 0),
            }
            for item in analytics.get("top_notes", [])[:6]
        ],
        "selected_note": {
            "source": detail.get("source"),
            "title": detail.get("title"),
            "namespace": detail.get("namespace"),
            "summary": _compact_text(detail.get("summary"), 400),
            "article": _compact_text(detail.get("article"), 600),
            "tags": detail.get("tags", [])[:8],
            "entities": detail.get("entities", [])[:10],
            "semantic_links": detail.get("semantic_links", [])[:5],
            "backlinks": detail.get("backlinks", [])[:8],
        } if detail else None,
        "current_ui": {
            "surface": "graph-first second-brain and RAG workspace",
            "left_rail_sections": ["Explore", "Explorer"],
            "center_sections": ["Graph View", "toolbar", "canvas"],
            "right_rail_sections": ["Selected Note", "Grounded Chat", "Synthesis", "Admin", "Create Note"],
            "major_actions": [
                "search notes",
                "filter by collection or namespace",
                "ask grounded chat questions",
                "run syntheses",
                "create or update notes",
                "ingest text, files, URLs, folders",
                "admin rebuild and reanalyze actions",
            ],
        },
    }


def _run_uiux_agent(req: "UIUXAgentRequest") -> "UIUXAgentResponse":
    ctx = _uiux_agent_context(req.selected_source)
    system_prompt = (
        "You are a senior frontend UI/UX agent for a graph-based RAG and second-brain app.\n"
        "Your job is to diagnose clarity, hierarchy, navigation, density, task flow, readability, and performance perception.\n"
        "Favor specific interface fixes over generic advice.\n"
        "The app is an operator tool, not a marketing site.\n"
        "Return strict JSON with keys: summary, top_issues, quick_wins, structural_changes, implementation_notes, prompt.\n"
        "Each list should contain concise strings. Do not include markdown fences."
    )
    user_prompt = json.dumps(
        {
            "goal": req.goal or "Fix the app so it feels fast, clear, dense, and easy to use for real knowledge work.",
            "pain_points": req.pain_points or "",
            "focus": req.focus or "overall",
            "current_path": req.current_path or "/",
            "viewport": req.viewport or "",
            "context": ctx,
        },
        ensure_ascii=True,
    )
    raw = _chat_completion(system_prompt, user_prompt, temperature=0.15, max_tokens=1100, timeout=min(60, LLM_CHAT_TIMEOUT_SECONDS))
    parsed = _coerce_json_object(raw)
    summary = str(parsed.get("summary") or "").strip() or _compact_text(raw, 700)
    top_issues = [str(item).strip() for item in (parsed.get("top_issues") or []) if str(item).strip()][:8]
    quick_wins = [str(item).strip() for item in (parsed.get("quick_wins") or []) if str(item).strip()][:8]
    structural_changes = [str(item).strip() for item in (parsed.get("structural_changes") or []) if str(item).strip()][:8]
    implementation_notes = [str(item).strip() for item in (parsed.get("implementation_notes") or []) if str(item).strip()][:10]
    prompt = str(parsed.get("prompt") or "").strip()
    if not prompt:
        prompt = (
            "Redesign Mnemosyne's graph workspace for faster daily use. Reduce visual noise, preserve dense knowledge access, "
            "prioritize selection, search, retrieval, note editing, and synthesis, and keep the graph central."
        )
    if not top_issues:
        top_issues = ["The current screen tries to support graph exploration, authoring, ingest, chat, and admin in one persistent rail, which dilutes task hierarchy."]
    if not quick_wins:
        quick_wins = ["Make one primary action per panel, collapse secondary actions, and tighten copy so the graph remains the dominant surface."]
    return UIUXAgentResponse(
        summary=summary,
        top_issues=top_issues,
        quick_wins=quick_wins,
        structural_changes=structural_changes,
        implementation_notes=implementation_notes,
        prompt=prompt,
    )


def _synthesize(title: str, sections: List[Dict[str, str]], instruction: str, mode: str, smart_collection: Optional[str] = None) -> "SynthesisResponse":
    context_parts = []
    for section in sections:
        header = f"Source: {section['source']}"
        if section.get("title"):
            header += f"\nTitle: {section['title']}"
        if section.get("summary"):
            header += f"\nExisting summary: {section['summary']}"
        context_parts.append(f"{header}\n\n{section['text']}")
    system_prompt = (
        "You are synthesizing a user's second-brain content.\n"
        "Be concise, specific, and grounded in the provided sources.\n"
        "Surface concrete overlaps, differences, action items, and risks where relevant.\n"
        "If the materials are incomplete, say that directly.\n"
    )
    mode_instructions = {
        "source": "Produce a concise briefing of this source.",
        "collection": "Produce a collection-level synthesis that highlights patterns, themes, risks, and action items across the sources.",
        "compare": "Compare the sources directly. Call out agreements, differences, unique details, and likely gaps.",
        "changes": "Summarize what changed from the earlier source to the later source. Separate additions, removals, and altered details.",
    }
    prompt = (
        f"Task: {mode_instructions.get(mode, 'Synthesize the material.')}\n"
        f"User instruction: {instruction or 'Use a brief operational style.'}\n\n"
        + "\n\n---\n\n".join(context_parts)
    )
    output = _chat_completion(system_prompt, prompt)
    return SynthesisResponse(
        title=title,
        output=output,
        sources=[section["source"] for section in sections],
        smart_collection=smart_collection,
    )


def _query_tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(token) >= 2]


def _history_signature(history: Optional[List[dict]]) -> str:
    items = []
    for msg in list(history or [])[-6:]:
        items.append({
            "role": str(msg.get("role", "")),
            "content": str(msg.get("content", ""))[:500],
        })
    return json.dumps(items, sort_keys=True, ensure_ascii=True)


def _query_cache_key(query: str, namespace: Optional[str], history: Optional[List[dict]], mode: str, include_noise: bool = False, include_raw_chats: bool = False) -> str:
    return json.dumps(
        {
            "query": query.strip(),
            "namespace": namespace or "",
            "history": _history_signature(history),
            "mode": mode,
            "include_noise": include_noise,
            "include_raw_chats": include_raw_chats,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _get_cached_query_answer(query: str, namespace: Optional[str], history: Optional[List[dict]], mode: str, include_noise: bool = False, include_raw_chats: bool = False) -> Optional[tuple[str, List[str]]]:
    key = _query_cache_key(query, namespace, history, mode, include_noise=include_noise, include_raw_chats=include_raw_chats)
    now = time.monotonic()
    with _query_cache_lock:
        cached = _query_answer_cache.get(key)
        if cached and cached.get("expires_at", 0.0) > now:
            return cached.get("answer", ""), list(cached.get("sources", []))
    return None


def _set_cached_query_answer(query: str, namespace: Optional[str], history: Optional[List[dict]], answer: str, sources: List[str], mode: str, include_noise: bool = False, include_raw_chats: bool = False) -> None:
    key = _query_cache_key(query, namespace, history, mode, include_noise=include_noise, include_raw_chats=include_raw_chats)
    with _query_cache_lock:
        _query_answer_cache[key] = {
            "answer": answer,
            "sources": list(sources),
            "expires_at": time.monotonic() + QUERY_CACHE_TTL_SECONDS,
        }


def _first_sentence(text: str, max_chars: int = FAST_QUERY_SENTENCE_CHARS) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", clean)
    snippet = parts[0] if parts else clean
    return snippet[:max_chars].rstrip()


def _fast_answer_from_sections(query: str, sections: List[Dict[str, str]], sources: List[str]) -> str:
    if not sections:
        return "I could not find enough relevant material in the brain or retrieval index to answer that yet.\n\nSources: none"
    evidence = [section for section in sections if section.get("kind") == "retrieved_evidence"][:2]
    memory = [section for section in sections if section.get("kind") == "brain_memory"][:2]
    linked = [section for section in sections if section.get("kind") == "linked_context"][:2]
    lines: List[str] = []
    if evidence:
        lead = _first_sentence(evidence[0].get("text", "")) or _first_sentence(memory[0].get("text", "")) if memory else ""
        if lead:
            lines.append(lead)
    elif memory:
        lead = _first_sentence(memory[0].get("text", ""))
        if lead:
            lines.append(lead)
    if not lines:
        lines.append(f"I found relevant context for: {query}")
    bullets: List[str] = []
    for section in (evidence + memory + linked)[:FAST_QUERY_SECTIONS]:
        title = section.get("title") or section.get("source") or "source"
        snippet = _first_sentence(section.get("text", ""))
        if snippet:
            bullets.append(f"- {title}: {snippet}")
    if bullets:
        lines.append("")
        lines.extend(bullets)
    if linked:
        related = ", ".join(section.get("title") or section.get("source") or "" for section in linked if section.get("source"))
        if related:
            lines.append("")
            lines.append(f"Connected context: {related}")
    lines.append("")
    lines.append("Sources: " + (", ".join(sources[:6]) if sources else "none"))
    return "\n".join(lines).strip()


def _record_quality_score(record: Dict[str, Any]) -> int:
    try:
        value = record.get("quality_score")
        return 70 if value is None else int(value)
    except Exception:
        return 70


def _retrieval_hard_blocked(record: Dict[str, Any], include_noise: bool = False) -> bool:
    status = _normalize_curation_status(record.get("curation_status"))
    trust = _normalize_trust_level(record.get("trust_level"))
    flags = {str(flag) for flag in (record.get("quality_flags") or [])}
    tier = str(record.get("retrieval_tier") or "").lower()
    if include_noise:
        return status in {"archived", "junk", "discarded"} or trust == "discard"
    if status in RETRIEVAL_BLOCK_STATUSES or trust in RETRIEVAL_BLOCK_TRUST:
        return True
    if "junk-source" in flags or "trash-tag" in flags or tier == "noise":
        return True
    return False


def _retrieval_primary(record: Dict[str, Any], explicit_scope: bool, include_noise: bool = False, include_raw_chats: bool = False) -> bool:
    if _retrieval_hard_blocked(record, include_noise=include_noise):
        return False
    if include_noise or explicit_scope:
        return True
    namespace = str(record.get("namespace") or "")
    if namespace in NOISE_NAMESPACES:
        return False
    if _is_raw_context_record(namespace, record.get("source"), record.get("tags")) and not include_raw_chats:
        return False
    if _record_quality_score(record) < MIN_DEFAULT_QUALITY_SCORE and not record.get("pinned"):
        return False
    return True


def _retrieval_weight(record: Dict[str, Any], explicit_scope: bool, include_raw_chats: bool = False) -> int:
    namespace = str(record.get("namespace") or "")
    status = _normalize_curation_status(record.get("curation_status"))
    trust = _normalize_trust_level(record.get("trust_level"))
    tier = str(record.get("retrieval_tier") or "").lower()
    weight = max(-20, min(20, (_record_quality_score(record) - 60) // 4))
    if record.get("pinned"):
        weight += 30
    if status in {"active", "trusted", "kept"}:
        weight += 18
    if trust == "trusted":
        weight += 18
    if namespace in {"notes", "memory", "documents", "code"}:
        weight += 8
    if namespace in NOISE_NAMESPACES and not explicit_scope:
        weight -= 35
    if _is_raw_context_record(namespace, record.get("source"), record.get("tags")) and not explicit_scope:
        weight -= 8 if include_raw_chats else 28
    if trust in RETRIEVAL_LOW_TRUST:
        weight -= 10
    if status in {"inbox", "unreviewed", "needs_action"}:
        weight -= 5
    if tier == "low":
        weight -= 14
    return weight


def _record_from_doc(doc: Document, records_by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source = doc.metadata.get("source", "")
    record = records_by_source.get(source)
    if record:
        return record
    return {
        "source": source,
        "namespace": doc.metadata.get("namespace"),
        "quality_score": doc.metadata.get("quality_score"),
        "quality_flags": doc.metadata.get("quality_flags") or [],
        "retrieval_tier": doc.metadata.get("retrieval_tier"),
        "curation_status": "inbox",
        "trust_level": "unreviewed",
    }


def _select_retrieval_docs(
    docs: List[Document],
    records_by_source: Dict[str, Dict[str, Any]],
    limit: int,
    explicit_scope: bool,
    include_noise: bool = False,
    include_raw_chats: bool = False,
) -> List[Document]:
    primary: List[Document] = []
    fallback: List[Document] = []
    for doc in docs:
        if not doc.metadata.get("source"):
            continue
        record = _record_from_doc(doc, records_by_source)
        if _retrieval_hard_blocked(record, include_noise=include_noise):
            continue
        if _retrieval_primary(record, explicit_scope, include_noise=include_noise, include_raw_chats=include_raw_chats):
            primary.append(doc)
        else:
            fallback.append(doc)

    selected: List[Document] = []
    seen_sources: set[str] = set()
    raw_chat_sources = 0
    for doc in primary + fallback:
        source = doc.metadata.get("source", "")
        record = _record_from_doc(doc, records_by_source)
        namespace = str(record.get("namespace") or doc.metadata.get("namespace") or "")
        if _is_raw_context_record(namespace, source, record.get("tags")) and not explicit_scope and not include_raw_chats:
            if raw_chat_sources >= DEFAULT_RAW_CHAT_CONTEXT_LIMIT:
                continue
            raw_chat_sources += 1
        if source in seen_sources and len(selected) >= limit:
            continue
        selected.append(doc)
        seen_sources.add(source)
        if len(selected) >= limit:
            break
    return selected


def _brain_search_records(query: str, namespace: Optional[str], limit: int = 6, include_noise: bool = False, include_raw_chats: bool = False) -> List[Dict[str, Any]]:
    tokens = _query_tokens(query)
    if not tokens:
        return []
    explicit_scope = bool(namespace)
    ranked_primary = []
    ranked_fallback = []
    for record in _list_source_records(namespace=namespace):
        if _retrieval_hard_blocked(record, include_noise=include_noise):
            continue
        title = str(record.get("title") or "").lower()
        source = str(record.get("source") or "").lower()
        summary = str(record.get("summary") or "").lower()
        article = str(record.get("article") or "").lower()
        collection = str(record.get("smart_collection") or "").lower()
        namespace_name = str(record.get("namespace") or "").lower()
        tags = " ".join(record.get("tags") or []).lower()
        entities = " ".join(record.get("entities") or []).lower()
        score = 0
        for token in tokens:
            if title == token:
                score += 18
            if token in title:
                score += 10
            if token in source:
                score += 8
            if token in summary:
                score += 6
            if token in article:
                score += 4
            if token in collection:
                score += 5
            if token in namespace_name:
                score += 3
            if token in tags:
                score += 5
            if token in entities:
                score += 6
        if score > 0:
            score += _retrieval_weight(record, explicit_scope, include_raw_chats=include_raw_chats)
            if _retrieval_primary(record, explicit_scope, include_noise=include_noise, include_raw_chats=include_raw_chats):
                ranked_primary.append((score, record))
            else:
                ranked_fallback.append((score, record))
    ranked_primary.sort(key=lambda item: (-item[0], item[1].get("title") or item[1]["source"]))
    ranked_fallback.sort(key=lambda item: (-item[0], item[1].get("title") or item[1]["source"]))
    ranked = ranked_primary + ranked_fallback
    return [record for _score, record in ranked[:limit]]


def _query_context_sections(query: str, namespace: Optional[str], include_noise: bool = False, include_raw_chats: bool = False) -> tuple[List[Dict[str, str]], List[str]]:
    sections: List[Dict[str, str]] = []
    sources: List[str] = []
    records = _list_source_records(namespace=namespace)
    records_by_source = {record["source"]: record for record in records}
    explicit_scope = bool(namespace)

    # 1. Retrieved evidence chunks
    docs = []
    if vectordb is not None:
        try:
            docs = vectordb.similarity_search(query, k=HYBRID_RETRIEVE_K * 8, namespace=namespace)
            if docs and reranker_inst:
                docs = reranker_inst.compress_documents(docs, query)
            docs = _select_retrieval_docs(
                docs,
                records_by_source,
                HYBRID_RETRIEVE_K,
                explicit_scope=explicit_scope,
                include_noise=include_noise,
                include_raw_chats=include_raw_chats,
            )
        except Exception as exc:
            logger.warning(f"hybrid retrieval failed for query '{query}': {exc}")
            docs = []
    chunk_groups: Dict[str, List[str]] = {}
    for doc in docs[:HYBRID_RETRIEVE_K]:
        source = doc.metadata.get("source", "")
        if not source:
            continue
        chunk_groups.setdefault(source, []).append(doc.page_content[:HYBRID_CHUNK_CHARS])
        if source not in sources:
            sources.append(source)
    for source, chunks in chunk_groups.items():
        source_record = records_by_source.get(source) or _source_record(source) or {}
        sections.append({
            "kind": "retrieved_evidence",
            "source": source,
            "title": source_record.get("title") or source,
            "text": "\n\n".join(chunks[:2]),
        })

    # 2. Brain-level matching records
    brain_records = _brain_search_records(query, namespace=namespace, limit=HYBRID_BRAIN_LIMIT, include_noise=include_noise, include_raw_chats=include_raw_chats)
    for record in brain_records:
        source = record["source"]
        if source not in sources:
            sources.append(source)
        links = _graph_links_for(source)
        semantic_links = _semantic_links_for_source(source, records=records)
        sections.append({
            "kind": "brain_memory",
            "source": source,
            "title": record.get("title") or source,
            "text": "\n".join([
                f"Summary: {record.get('summary') or ''}",
                f"Collection: {record.get('smart_collection') or ''}",
                f"Tags: {', '.join(record.get('tags') or [])}",
                f"Entities: {', '.join(record.get('entities') or [])}",
                f"Backlinks: {', '.join(links.get('backlinks', [])[:3])}",
                f"Links: {', '.join(links.get('outbound_links', [])[:3])}",
                f"Semantic links: {', '.join(item.get('target', '') for item in semantic_links[:3])}",
                "",
                (record.get("article") or "")[:HYBRID_ARTICLE_CHARS],
            ]).strip(),
        })

    # 3. Expand local neighborhood from the strongest matched notes
    neighbor_sources: List[str] = []
    for record in brain_records[:3]:
        links = _graph_links_for(record["source"])
        semantic_links = _semantic_links_for_source(record["source"], records=records)
        for candidate in (links.get("backlinks", []) + links.get("outbound_links", []) + links.get("inferred_links", []) + [item.get("target", "") for item in semantic_links]):
            if candidate not in neighbor_sources and candidate not in sources:
                neighbor_sources.append(candidate)
            if len(neighbor_sources) >= HYBRID_NEIGHBOR_LIMIT:
                break
        if len(neighbor_sources) >= HYBRID_NEIGHBOR_LIMIT:
            break
    for source in neighbor_sources[:HYBRID_NEIGHBOR_LIMIT]:
        record = records_by_source.get(source) or _source_record(source)
        if not record:
            continue
        if not _retrieval_primary(record, explicit_scope, include_noise=include_noise, include_raw_chats=include_raw_chats):
            continue
        sources.append(source)
        sections.append({
            "kind": "linked_context",
            "source": source,
            "title": record.get("title") or source,
            "text": "\n".join([
                f"Summary: {record.get('summary') or ''}",
                f"Collection: {record.get('smart_collection') or ''}",
                (record.get("article") or "")[:HYBRID_NEIGHBOR_ARTICLE_CHARS],
            ]).strip(),
        })
    return sections, list(dict.fromkeys(sources))


def _deep_answer_with_brain_context(query: str, history: Optional[List[dict]], namespace: Optional[str], include_noise: bool = False, include_raw_chats: bool = False) -> tuple[str, List[str], List[Dict[str, str]]]:
    cached = _get_cached_query_answer(query, namespace, history, "deep", include_noise=include_noise, include_raw_chats=include_raw_chats)
    if cached is not None:
        return cached[0], cached[1], []
    sections, sources = _query_context_sections(query, namespace=namespace, include_noise=include_noise, include_raw_chats=include_raw_chats)
    history_lines = []
    for msg in list(history or [])[-8:]:
        role = str(msg.get("role", "user")).title()
        content = str(msg.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    context_blocks = []
    for section in sections:
        context_blocks.append(
            f"[{section['kind']}] Source: {section['source']}\nTitle: {section.get('title') or section['source']}\n\n{section['text']}"
        )
    prompt = (
        "You are Mnemosyne, a hybrid second-brain and RAG assistant.\n"
        "Use retrieved evidence and memory context together.\n"
        "Prefer grounded facts from retrieved evidence for factual claims.\n"
        "Use linked notes, summaries, and articles to add context, synthesis, and connections.\n"
        "If the evidence is insufficient, say so clearly.\n"
        "End with a short Sources line naming the sources you relied on.\n\n"
        f"Conversation history:\n{chr(10).join(history_lines) if history_lines else '(none)'}\n\n"
        f"User query: {query}\n"
        f"Namespace: {namespace or 'all'}\n\n"
        f"Context:\n{chr(10).join(context_blocks) if context_blocks else '(no context found)'}"
    )
    try:
        answer = _chat_completion(
            "You answer from both knowledge retrieval and second-brain context. Be concise, explicit, and source-grounded.",
            prompt,
            temperature=0.2,
            max_tokens=700,
            timeout=HYBRID_CHAT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(f"hybrid query model fallback for '{query}': {exc}")
        bullets = []
        for section in sections[:4]:
            snippet = " ".join(section["text"].split())[:220]
            bullets.append(f"- {section.get('title') or section['source']}: {snippet}")
        answer = (
            "I could not complete the full model synthesis quickly, so here is a direct brain-and-RAG summary from the retrieved context.\n\n"
            + "\n".join(bullets if bullets else ["- No relevant context was found."])
            + "\n\nSources: "
            + ", ".join(sources[:6])
        ).strip()
    _set_cached_query_answer(query, namespace, history, answer, sources, "deep", include_noise=include_noise, include_raw_chats=include_raw_chats)
    return answer, sources, sections


def _answer_with_brain_context(query: str, history: Optional[List[dict]], namespace: Optional[str], include_noise: bool = False, include_raw_chats: bool = False) -> tuple[str, List[str], List[Dict[str, str]]]:
    cached = _get_cached_query_answer(query, namespace, history, "fast", include_noise=include_noise, include_raw_chats=include_raw_chats)
    if cached is not None:
        return cached[0], cached[1], []
    sections, sources = _query_context_sections(query, namespace=namespace, include_noise=include_noise, include_raw_chats=include_raw_chats)
    answer = _fast_answer_from_sections(query, sections, sources)
    _set_cached_query_answer(query, namespace, history, answer, sources, "fast", include_noise=include_noise, include_raw_chats=include_raw_chats)
    return answer, sources, sections


def _extract_workspace_blocks(text: str, marker: str) -> List[str]:
    blocks: List[str] = []
    if not text:
        return blocks
    pattern = re.compile(rf"\[{re.escape(marker)}\]\s*(.*?)(?=\n\[[A-Za-z]+\]\s|\Z)", re.DOTALL)
    for match in pattern.finditer(text):
        block = match.group(1).strip()
        if block:
            blocks.append(block[:6000])
    return blocks


def _source_workspace(source: str) -> Dict[str, Any]:
    section = _source_text(source, max_chars=120000)
    rows = _source_rows(source)
    pages: List[Dict[str, Any]] = []
    page_map: Dict[Any, Dict[str, Any]] = {}
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    if rows:
        for idx, row in enumerate(rows, start=1):
            page_no = row.get("page")
            label = row.get("page_label") or (f"Page {page_no}" if page_no else f"Section {idx}")
            key = page_no if page_no is not None else f"section-{idx}"
            entry = page_map.setdefault(key, {"page": page_no or idx, "label": label, "text_parts": []})
            chunk_text = (row.get("text") or "").strip()
            if chunk_text:
                entry["text_parts"].append(chunk_text)
                for block in _extract_workspace_blocks(chunk_text, "Table"):
                    tables.append({"page": entry["page"], "label": label, "text": block})
                for block in _extract_workspace_blocks(chunk_text, "Figure"):
                    figures.append({"page": entry["page"], "label": label, "text": block})
        for item in page_map.values():
            page_text = "\n\n".join(item.pop("text_parts", [])).strip()
            item["text"] = page_text[:12000]
            pages.append(item)
    if not pages and section.get("text"):
        pages.append({"page": 1, "label": "Document", "text": section["text"][:12000]})
        for block in _extract_workspace_blocks(section["text"], "Table"):
            tables.append({"page": 1, "label": "Document", "text": block})
        for block in _extract_workspace_blocks(section["text"], "Figure"):
            figures.append({"page": 1, "label": "Document", "text": block})
    return {
        "source": source,
        "title": section.get("title"),
        "type": section.get("type"),
        "namespace": section.get("namespace"),
        "summary": section.get("summary"),
        "article": section.get("article"),
        "smart_collection": section.get("smart_collection"),
        "tags": list(section.get("tags") or []),
        "entities": list(section.get("entities") or []),
        "page_count": len(pages),
        "pages": pages,
        "tables": tables[:24],
        "figures": figures[:24],
    }


def _answer_document_query(source: str, query: str, deep: bool = False) -> tuple[str, List[str]]:
    workspace = _source_workspace(source)
    source_text = _source_text(source, max_chars=30000)
    sections = [{
        "kind": "document_workspace",
        "source": source,
        "title": workspace.get("title") or source,
        "text": "\n".join([
            f"Summary: {workspace.get('summary') or ''}",
            f"Collection: {workspace.get('smart_collection') or ''}",
            f"Tags: {', '.join(workspace.get('tags') or [])}",
            f"Entities: {', '.join(workspace.get('entities') or [])}",
            "",
            (workspace.get("article") or "")[:2000],
            "",
            (source_text.get("text") or "")[:12000],
        ]).strip(),
    }]
    sources = [source]
    if not deep:
        return _fast_answer_from_sections(query, sections, sources), sources
    prompt = (
        "You are answering about a single document inside a second-brain system.\n"
        "Stay grounded in the supplied document workspace only.\n"
        "Use page cues, tables, and figures when relevant.\n"
        "If the document does not contain the answer, say so clearly.\n\n"
        f"Question: {query}\n\n"
        f"Document workspace:\n{sections[0]['text']}"
    )
    answer = _chat_completion(
        "You answer questions about one document using only the provided workspace context.",
        prompt,
        temperature=0.1,
        max_tokens=700,
        timeout=HYBRID_CHAT_TIMEOUT_SECONDS,
    )
    return answer + f"\n\nSources: {source}", sources


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, vectordb, embeddings, mclient, reranker_inst
    logger.info("Starting Mnemosyne...")

    embeddings = SpecAwareEmbeddings(
        base_url=VLLM_EMBED_BASE_URL,
        model_name=EMBED_MODEL,
    )

    reranker_inst = NIMReranker(base_url=RERANK_BASE_URL, model_name=RERANK_MODEL, top_k=5)
    logger.info(f"Reranker: {RERANK_MODEL} @ {RERANK_BASE_URL}")

    mclient = MilvusClient(uri=MILVUS_URL)
    created = _ensure_collection(mclient, MILVUS_COLLECTION, EMBED_DIM)
    mclient.load_collection(MILVUS_COLLECTION)
    logger.info(f"Milvus/cuVS collection={MILVUS_COLLECTION} ({'new' if created else 'existing'}), size={_index_size()}")

    vectordb = MilvusVectorStore(
        client=mclient,
        collection=MILVUS_COLLECTION,
        embed_fn=embeddings,
        dim=EMBED_DIM,
    )

    retriever = ContextualCompressionRetriever(
        base_retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8}),
        base_compressor=reranker_inst,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=VLLM_LLM_BASE_URL,
        openai_api_key="dummy",
        temperature=0.6,
        max_tokens=4096,
    )

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="knowledge_base",
        description="Search your Mnemosyne knowledge base. Always use this before answering factual questions.",
        document_prompt=PromptTemplate.from_template("Source: {source}\n\n{page_content}"),
    )

    agent = create_react_agent(
        model=llm,
        tools=[retriever_tool],
        prompt=(
            "You are a grounded NVIDIA-style RAG agent with access to the user's knowledge base. "
            "Always search the knowledge base before answering factual questions. "
            "Use only retrieved evidence when making factual claims. "
            "If the retrieved context is insufficient, say so directly instead of guessing. "
            "Cite the source names you used. "
            "Be concise and direct."
        ),
    )

    _start_housekeeping_agent()
    logger.info("Mnemosyne ready.")
    try:
        yield
    finally:
        _stop_housekeeping_agent()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mnemosyne API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=RAG_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_local_or_token(request: Request, call_next):
    if request.method == "OPTIONS" or _is_loopback_client(request) or _authorized_remote_request(request):
        return await call_next(request)
    detail = "remote access requires RAG_API_TOKEN" if RAG_API_TOKEN else "remote access disabled; bind locally or set RAG_API_TOKEN"
    return JSONResponse({"detail": detail}, status_code=401)


# ── Schemas ───────────────────────────────────────────────────────────────────

CONTENT_NAMESPACES = ["claude", "web", "documents", "email", "notes", "images", "audio", "code", "ai-chats", "data", "slack", "memory", "inbox"]


def _auto_namespace(doc_type: str, source: str) -> str:
    """Derive namespace from doc_type and source — no manual filing needed."""
    if doc_type in ("claude-chat",) or source.startswith("claude/"):
        return "claude"
    if doc_type in ("chatgpt", "copilot"):
        return "ai-chats"
    if doc_type in ("url", "html") or (source.startswith("http") and doc_type not in ("pdf", "docx")):
        return "web"
    if doc_type in ("email", "pst", "eml", "mbox"):
        return "email"
    if doc_type in ("obsidian", "markdown", "note"):
        return "notes"
    if doc_type in ("pdf", "docx"):
        return "documents"
    if doc_type == "image":
        return "images"
    if doc_type == "audio":
        return "audio"
    if doc_type == "code":
        return "code"
    if doc_type in ("json", "yaml", "data"):
        return "data"
    if doc_type == "slack":
        return "slack"
    return "inbox"

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None
    namespace: Optional[str] = None   # None = search all namespaces
    mode: str = "fast"
    include_noise: bool = False
    include_raw_chats: bool = False

class QueryResponse(BaseModel):
    answer: str
    model: str
    sources: List[str] = []
    namespace: Optional[str] = None
    mode: str = "fast"

class IngestTextRequest(BaseModel):
    text: str
    source: Optional[str] = None
    tags: Optional[List[str]] = []
    namespace: Optional[str] = None
    semantic: bool = True

class IngestURLRequest(BaseModel):
    url: str
    tags: Optional[List[str]] = []
    namespace: Optional[str] = None

class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
    index_size: int
    namespace: Optional[str] = None


class UpdateSourceRequest(BaseModel):
    source: str
    text: str
    title: Optional[str] = None
    tags: Optional[List[str]] = []
    namespace: Optional[str] = None
    new_source: Optional[str] = None
    aliases: Optional[List[str]] = None
    manual_links: Optional[List[str]] = None
    pinned: Optional[bool] = None
    template_name: Optional[str] = None
    curation_status: Optional[str] = None
    trust_level: Optional[str] = None
    review_note: Optional[str] = None


class SourceCurationRequest(BaseModel):
    source: str
    aliases: Optional[List[str]] = None
    manual_links: Optional[List[str]] = None
    pinned: Optional[bool] = None
    template_name: Optional[str] = None
    curation_status: Optional[str] = None
    trust_level: Optional[str] = None
    review_note: Optional[str] = None

class SourceItem(BaseModel):
    source: str
    type: str
    ingested_at: str
    chunk_count: int
    namespace: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    entities: List[str] = []
    smart_collection: Optional[str] = None
    quality_score: Optional[int] = None
    quality_flags: List[str] = []
    retrieval_tier: Optional[str] = None
    query_count: int = 0
    retrieve_count: int = 0
    synthesis_count: int = 0
    ingest_count: int = 0
    last_activity_at: Optional[str] = None
    pinned: bool = False
    aliases: List[str] = []
    curation_status: str = "inbox"
    trust_level: str = "unreviewed"
    review_note: Optional[str] = None

class ObsidianIngestResponse(BaseModel):
    notes_found: int
    notes_indexed: int
    chunks_indexed: int
    skipped: int
    index_size: int

class FolderImportRequest(BaseModel):
    path: str
    recursive: bool = True
    label: Optional[str] = None
    namespace: Optional[str] = None

class FolderImportResponse(BaseModel):
    path: str
    files_found: int
    files_indexed: int
    chunks_indexed: int
    skipped: int
    errors: List[str]
    index_size: int


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    source_filter: Optional[str] = None
    namespace: Optional[str] = None
    include_noise: bool = False
    include_raw_chats: bool = False

class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: Optional[float] = None


class SourceSynthesisRequest(BaseModel):
    source: str
    instruction: Optional[str] = None


class CollectionSynthesisRequest(BaseModel):
    smart_collection: str
    instruction: Optional[str] = None
    max_sources: int = 12


class CompareSourcesRequest(BaseModel):
    sources: List[str]
    instruction: Optional[str] = None


class ChangeSummaryRequest(BaseModel):
    earlier_source: str
    later_source: str
    instruction: Optional[str] = None


class SynthesisResponse(BaseModel):
    title: str
    output: str
    sources: List[str]
    smart_collection: Optional[str] = None


class SourceChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]


class WorkspacePage(BaseModel):
    page: int
    label: str
    text: str


class WorkspaceBlock(BaseModel):
    page: int
    label: str
    text: str


class SourceWorkspace(BaseModel):
    source: str
    title: Optional[str] = None
    type: Optional[str] = None
    namespace: Optional[str] = None
    summary: Optional[str] = None
    article: Optional[str] = None
    smart_collection: Optional[str] = None
    tags: List[str] = []
    entities: List[str] = []
    page_count: int = 0
    pages: List[WorkspacePage] = []
    tables: List[WorkspaceBlock] = []
    figures: List[WorkspaceBlock] = []


class DocumentQueryRequest(BaseModel):
    source: str
    query: str
    deep: bool = False


class SourceDetail(BaseModel):
    source: str
    type: str
    ingested_at: str
    chunk_count: int
    title: Optional[str] = None
    summary: Optional[str] = None
    article: Optional[str] = None
    tags: List[str] = []
    entities: List[str] = []
    smart_collection: Optional[str] = None
    quality_score: Optional[int] = None
    quality_flags: List[str] = []
    retrieval_tier: Optional[str] = None
    related_sources: List[str] = []
    explicit_links: List[str] = []
    inferred_links: List[str] = []
    outbound_links: List[str] = []
    unresolved_links: List[str] = []
    backlinks: List[str] = []
    semantic_links: List[Dict[str, Any]] = []
    manual_links: List[str] = []
    query_count: int = 0
    retrieve_count: int = 0
    synthesis_count: int = 0
    ingest_count: int = 0
    last_activity_at: Optional[str] = None
    pinned: bool = False
    aliases: List[str] = []
    template_name: Optional[str] = None
    curation_status: str = "inbox"
    trust_level: str = "unreviewed"
    review_note: Optional[str] = None
    full_text: Optional[str] = None
    chunks: List[SourceChunk]


class GraphLayoutRequest(BaseModel):
    positions: Dict[str, Dict[str, float]]


class GraphViewRequest(BaseModel):
    name: str
    state: Dict[str, Any]


class MemoryPolicy(BaseModel):
    add_to_rag: bool = False
    summarize: bool = True
    extract_tasks: bool = True
    create_note: bool = True
    memory_level: str = "raw"
    sensitivity: str = "private"
    allowed_use: List[str] = Field(default_factory=lambda: ["summarize", "answer", "extract_tasks"])
    expires_at: Optional[str] = None
    require_review: bool = True
    include_raw_text: bool = True


class MemoryCaptureRequest(BaseModel):
    content: Optional[str] = ""
    url: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    source_kind: Optional[str] = None
    device: Optional[str] = None
    project: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    namespace: Optional[str] = "memory"
    policy: Optional[MemoryPolicy] = None
    process_now: bool = False
    approve_now: bool = False


class MemoryUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    project: Optional[str] = None
    tags: Optional[List[str]] = None
    namespace: Optional[str] = None
    policy: Optional[MemoryPolicy] = None


class MemoryProcessRequest(BaseModel):
    force: bool = False


class MemoryApproveRequest(BaseModel):
    policy: Optional[MemoryPolicy] = None
    namespace: Optional[str] = None
    memory_level: Optional[str] = None
    sensitivity: Optional[str] = None
    source_label: Optional[str] = None
    reviewed: bool = True


class MemoryQueryRequest(BaseModel):
    query: str
    levels: List[str] = Field(default_factory=lambda: ["temporary", "project", "long-term"])
    include_raw: bool = False
    include_sensitive: bool = False
    namespace: Optional[str] = None
    k: int = 6


class ReanalyzeRequest(BaseModel):
    namespace: Optional[str] = None
    max_sources: int = 0


class PurgeJunkRequest(BaseModel):
    dry_run: bool = False
    include_web: bool = True


class NoiseCurationRequest(BaseModel):
    dry_run: bool = True
    namespace: Optional[str] = None
    limit: int = 0


class UIUXAgentRequest(BaseModel):
    goal: Optional[str] = None
    pain_points: Optional[str] = None
    focus: str = "overall"
    selected_source: Optional[str] = None
    current_path: Optional[str] = None
    viewport: Optional[str] = None


class UIUXAgentResponse(BaseModel):
    summary: str
    top_issues: List[str] = []
    quick_wins: List[str] = []
    structural_changes: List[str] = []
    implementation_notes: List[str] = []
    prompt: str


# ── Routes ────────────────────────────────────────────────────────────────────

def _dependency_probe(name: str, kind: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 4.0) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        if kind == "post":
            response = requests.post(url, json=payload or {}, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        latency_ms = int((time.monotonic() - started) * 1000)
        ok = 200 <= response.status_code < 300
        detail = response.text[:240] if not ok else "ok"
        return {"name": name, "ok": ok, "status_code": response.status_code, "latency_ms": latency_ms, "detail": detail}
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        return {"name": name, "ok": False, "status_code": None, "latency_ms": latency_ms, "detail": str(exc)[:240]}


class CurateSourceRequest(BaseModel):
    source: str
    curation_status: Optional[str] = None
    trust_level: Optional[str] = None
    review_note: Optional[str] = None
    action_status: Optional[str] = None


class ActionStatusRequest(BaseModel):
    action_id: str
    source: str
    status: str = "done"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": agent is not None,
        "housekeeping": _housekeeping_public_state().get("status"),
        "store": "milvus+cuvs",
        "collection": MILVUS_COLLECTION,
        "index_size": _index_size(),
    }


@app.get("/health/deep")
def deep_health():
    base = health()
    dependencies = [
        {
            "name": "milvus",
            "ok": mclient is not None,
            "status_code": None,
            "latency_ms": 0,
            "detail": f"{MILVUS_COLLECTION}: {_index_size()} rows" if mclient is not None else "client not initialized",
        },
        _dependency_probe("llm", "get", f"{VLLM_LLM_BASE_URL.rstrip('/')}/models"),
        _dependency_probe("embedder", "get", f"{VLLM_EMBED_BASE_URL.rstrip('/')}/models"),
        _dependency_probe(
            "reranker",
            "post",
            RERANK_BASE_URL,
            payload={
                "model": RERANK_MODEL,
                "query": {"text": "health check"},
                "passages": [{"text": "health check"}],
            },
        ),
    ]
    base["dependencies"] = dependencies
    base["status"] = "ok" if all(item.get("ok") for item in dependencies) and base.get("agent") else "degraded"
    return base


@app.get("/config")
def config():
    embed_mode = "query-passage" if _supports_nvidia_dual_embedding_mode(EMBED_MODEL) else "single"
    return {
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm_base_url": VLLM_LLM_BASE_URL,
        "embed_base_url": VLLM_EMBED_BASE_URL,
        "rerank_base_url": RERANK_BASE_URL,
        "milvus_url": MILVUS_URL,
        "collection": MILVUS_COLLECTION,
        "embed_dim": EMBED_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_mode": embed_mode,
        "nvidia_recommended_embed_model": NVIDIA_RECOMMENDED_EMBED_MODEL,
        "nvidia_recommended_embed_dim": NVIDIA_RECOMMENDED_EMBED_DIM,
        "spec_alignment": {
            "milvus_cuvs": True,
            "reranker_nim": RERANK_MODEL.startswith("nvidia/"),
            "embed_nim_recommended_model": EMBED_MODEL == NVIDIA_RECOMMENDED_EMBED_MODEL,
            "embed_query_passage_mode": embed_mode == "query-passage",
            "page_chunking_for_pdfs": True,
        },
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(req.namespace)
    try:
        mode = (req.mode or "fast").lower()
        if mode == "deep":
            answer, sources, sections = _deep_answer_with_brain_context(
                req.query,
                req.history,
                namespace,
                include_noise=req.include_noise,
                include_raw_chats=req.include_raw_chats,
            )
        else:
            answer, sources, sections = _answer_with_brain_context(
                req.query,
                req.history,
                namespace,
                include_noise=req.include_noise,
                include_raw_chats=req.include_raw_chats,
            )
    except Exception as e:
        raise HTTPException(500, str(e))
    _record_activity(sources, "query")
    _append_query_trace("query", req.query, sources, sections=sections, namespace=namespace)
    return QueryResponse(answer=answer, model=LLM_MODEL, sources=sources, namespace=namespace, mode=mode)


@app.post("/query/deep", response_model=QueryResponse)
def query_deep(req: QueryRequest):
    req.mode = "deep"
    return query(req)


@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    return query(req)


@app.get("/memory/items")
def memory_items(status: Optional[str] = None, level: Optional[str] = None, q: Optional[str] = None):
    state = _load_memory_state()
    items = list(state.get("items", []))
    if status:
        items = [item for item in items if item.get("status") == status]
    if level:
        items = [item for item in items if item.get("policy", {}).get("memory_level") == level]
    if q:
        needle = q.strip().lower()
        items = [
            item for item in items
            if needle in " ".join([
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                str(item.get("source") or ""),
                " ".join(item.get("tags") or []),
                str(item.get("project") or ""),
            ]).lower()
        ]
    return {
        "updated_at": state.get("updated_at"),
        "rules": state.get("rules", {}),
        "items": sorted(items, key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True),
    }


@app.get("/memory/items/{item_id}")
def memory_get_item(item_id: str):
    state = _load_memory_state()
    item = _memory_item(state, item_id)
    if not item:
        raise HTTPException(404, f"Memory item not found: {item_id}")
    return item


@app.post("/memory/capture")
def memory_capture(req: MemoryCaptureRequest):
    source = (req.source or req.url or "").strip()
    content = (req.content or "").strip()
    if not source and not content:
        raise HTTPException(400, "content or url is required")
    namespace = _normalize_namespace(req.namespace) or "memory"
    policy = _normalize_memory_policy(req.policy)
    if req.approve_now:
        policy["add_to_rag"] = True
        policy["require_review"] = False
    now = _now_iso()
    item = {
        "id": uuid.uuid4().hex[:16],
        "status": "captured",
        "created_at": now,
        "updated_at": now,
        "source": source,
        "source_kind": _memory_source_kind(source, content, req.source_kind),
        "device": req.device or "web",
        "title": req.title or _memory_title_from_content(content, source or "Captured memory"),
        "content": content,
        "project": req.project,
        "namespace": namespace,
        "tags": _merge_tags(req.tags or []),
        "summary": "",
        "tasks": [],
        "questions": [],
        "entities": [],
        "errors": [],
        "blocked": [],
        "policy": policy,
    }
    state = _load_memory_state()
    state.setdefault("items", []).append(item)
    if req.process_now or req.approve_now or policy.get("summarize") and not policy.get("require_review"):
        _process_memory_item(item, force=False)
    if req.approve_now:
        _approve_memory_item(item, {"reviewed": True, "namespace": namespace})
    _save_memory_state(state)
    return item


@app.patch("/memory/items/{item_id}")
def memory_update_item(item_id: str, req: MemoryUpdateRequest):
    state = _load_memory_state()
    item = _memory_item(state, item_id)
    if not item:
        raise HTTPException(404, f"Memory item not found: {item_id}")
    updates = _model_to_dict(req)
    if "status" in updates and updates["status"] is not None:
        if updates["status"] not in MEMORY_STATUSES:
            raise HTTPException(400, f"Invalid status: {updates['status']}")
        item["status"] = updates["status"]
    for key in ("title", "content", "project"):
        if key in updates and updates[key] is not None:
            item[key] = updates[key]
    if "namespace" in updates and updates["namespace"] is not None:
        item["namespace"] = _normalize_namespace(updates["namespace"]) or "memory"
    if updates.get("tags") is not None:
        item["tags"] = _merge_tags(updates["tags"])
    if updates.get("policy") is not None:
        item["policy"] = _normalize_memory_policy(item.get("policy"), **_model_to_dict(updates["policy"]))
    item["updated_at"] = _now_iso()
    _save_memory_state(state)
    return item


@app.post("/memory/items/{item_id}/process")
def memory_process_item(item_id: str, req: MemoryProcessRequest = MemoryProcessRequest()):
    state = _load_memory_state()
    item = _memory_item(state, item_id)
    if not item:
        raise HTTPException(404, f"Memory item not found: {item_id}")
    _process_memory_item(item, force=req.force)
    _save_memory_state(state)
    return item


@app.post("/memory/items/{item_id}/approve")
def memory_approve_item(item_id: str, req: MemoryApproveRequest = MemoryApproveRequest()):
    state = _load_memory_state()
    item = _memory_item(state, item_id)
    if not item:
        raise HTTPException(404, f"Memory item not found: {item_id}")
    _approve_memory_item(item, req)
    _save_memory_state(state)
    return item


@app.delete("/memory/items/{item_id}")
def memory_delete_item(item_id: str, delete_rag: bool = False):
    state = _load_memory_state()
    item = _memory_item(state, item_id)
    if not item:
        raise HTTPException(404, f"Memory item not found: {item_id}")
    if delete_rag and item.get("rag_source"):
        _delete_indexed_source(item["rag_source"])
    state["items"] = [candidate for candidate in state.get("items", []) if candidate.get("id") != item_id]
    _save_memory_state(state)
    return {"deleted": item_id, "deleted_rag": bool(delete_rag and item.get("rag_source"))}


@app.get("/memory/rules")
def memory_rules():
    state = _load_memory_state()
    return state.get("rules", {})


@app.post("/memory/rules")
def memory_update_rules(rules: Dict[str, Any]):
    state = _load_memory_state()
    current = dict(state.get("rules") or {})
    for key in ("default_review_required", "auto_index_trusted"):
        if key in rules:
            current[key] = bool(rules[key])
    for key in ("trusted_domains", "blocked_domains"):
        if key in rules and isinstance(rules[key], list):
            current[key] = [str(item).strip().lower() for item in rules[key] if str(item).strip()][:100]
    state["rules"] = current
    _save_memory_state(state)
    return current


@app.post("/memory/query")
def memory_query(req: MemoryQueryRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    namespace = _normalize_namespace(req.namespace)
    allowed = _approved_memory_sources(req.levels, req.include_raw, req.include_sensitive)
    if not allowed:
        return {
            "answer": "No approved memory matched the selected policy levels.",
            "sources": [],
            "blocked": [],
            "levels": req.levels,
        }
    search_k = min(max(req.k * 6, req.k), 40)
    try:
        docs = vectordb.similarity_search(req.query, k=search_k, namespace=namespace)
        if reranker_inst is not None and docs:
            docs = reranker_inst.compress_documents(docs, req.query)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    approved_docs = []
    blocked = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source in allowed:
            approved_docs.append(doc)
        elif source:
            blocked.append(source)
        if len(approved_docs) >= max(1, min(req.k, 12)):
            break
    if not approved_docs:
        return {
            "answer": "Retrieval found no chunks allowed by the selected memory policy.",
            "sources": [],
            "blocked": list(dict.fromkeys(blocked))[:12],
            "levels": req.levels,
        }
    context_parts = []
    sources = []
    for doc in approved_docs:
        source = doc.metadata.get("source", "")
        memory = allowed.get(source, {})
        if source not in sources:
            sources.append(source)
        context_parts.append(
            f"Source: {source}\n"
            f"Title: {memory.get('title') or source}\n"
            f"Level: {memory.get('policy', {}).get('memory_level')}\n"
            f"Sensitivity: {memory.get('policy', {}).get('sensitivity')}\n\n"
            f"{doc.page_content}"
        )
    system_prompt = (
        "You answer using only the approved memory wrapper context provided. "
        "Do not use blocked or unapproved memory. If the context is insufficient, say so directly. "
        "Cite source names used."
    )
    prompt = f"Question: {req.query}\n\nApproved context:\n\n" + "\n\n---\n\n".join(context_parts)
    answer = _chat_completion(system_prompt, prompt, max_tokens=900)
    _record_activity(sources, "query")
    _append_query_trace("memory-query", req.query, sources, sections=[{"source": src, "text": ""} for src in sources], namespace=namespace)
    return {
        "answer": answer,
        "sources": sources,
        "blocked": list(dict.fromkeys(blocked))[:12],
        "levels": req.levels,
    }


@app.get("/memory/share", response_class=HTMLResponse)
def memory_share(url: str = "", title: str = "", text: str = "", device: str = "share"):
    capture = MemoryCaptureRequest(
        url=url or None,
        content=text or "",
        title=title or None,
        device=device,
        source_kind=None,
        process_now=False,
    )
    item = memory_capture(capture)
    safe_title = (item.get("title") or item.get("source") or item.get("id")).replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(
        "<!doctype html><html><head><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Captured</title><style>body{font-family:system-ui;margin:24px;line-height:1.4}"
        "a{color:#0b6b5c}</style></head><body>"
        f"<h1>Captured</h1><p>{safe_title}</p><p><a href=\"/memory\">Open memory wrapper</a></p>"
        "</body></html>"
    )


@app.post("/agent/uiux", response_model=UIUXAgentResponse)
def uiux_agent(req: UIUXAgentRequest):
    try:
        return _run_uiux_agent(req)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text is too large; limit is {MAX_TEXT_CHARS} characters")
    detected_type, detected_source = _detect_source_type(req.text)
    source   = req.source or detected_source
    doc_type = detected_type
    junk_reason = _junk_source_reason(source, doc_type)
    ns       = _normalize_namespace(req.namespace) or _auto_namespace(doc_type, source)
    tags     = list(req.tags or [])
    if junk_reason:
        logger.info("Skipped junk ingest source=%s reason=%s", source, junk_reason)
        return IngestResponse(source=source, chunks_indexed=0, index_size=_index_size(), namespace=ns)
    if doc_type not in tags:
        tags.append(doc_type)
    skipped = _skip_trash_ingest(source, ns, tags, req.text)
    if skipped:
        return skipped
    extra = {"tags": tags}
    if not req.semantic:
        clean = _strip_think_traces(req.text)
        extra.update({
            "title": _fallback_title(source, doc_type, clean),
            "summary": clean[:360].replace("\n", " ").strip(),
            "smart_collection": "ai-chats" if ns == "ai-chats" else ns,
        })
    docs = [_make_doc(req.text, source, doc_type, extra, namespace=ns)]
    if req.semantic:
        docs = _apply_semantic_metadata(docs, source, doc_type, ns, tags=tags)
    n = _add_chunks(docs)
    return IngestResponse(source=source, chunks_indexed=n, index_size=_index_size(), namespace=ns)


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url(req: IngestURLRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    try:
        text = _extract_url(req.url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch URL: {e}")
    if not text.strip():
        raise HTTPException(400, "No text extracted from URL")
    ns = _normalize_namespace(req.namespace) or _auto_namespace("url", req.url)
    tags = list(req.tags or [])
    skipped = _skip_trash_ingest(req.url, ns, tags, text)
    if skipped:
        return skipped
    docs = _apply_semantic_metadata([_make_doc(text, req.url, "url", {"tags": tags}, namespace=ns)], req.url, "url", ns, tags=tags)
    n = _add_chunks(docs)
    logger.info(f"/ingest/url: {req.url} ns={ns} → {n} chunks")
    return IngestResponse(source=req.url, chunks_indexed=n, index_size=_index_size(), namespace=ns)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), namespace: Optional[str] = None):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(namespace)
    data = await _read_upload_limited(file)
    filename = file.filename or "upload"
    junk_reason = _junk_source_reason(filename)
    if junk_reason:
        raise HTTPException(400, f"Refusing to index junk source: {junk_reason}")

    try:
        docs, page_scoped, doc_type = _build_documents_from_file(data, filename, namespace)
    except ValueError as e:
        message = str(e)
        status = 415 if "unsupported extension" in message else 400
        raise HTTPException(status, message)
    except Exception as e:
        raise HTTPException(400, f"Parse failed: {e}")

    resolved_ns = docs[0].metadata.get("namespace", "inbox") if docs else "inbox"
    docs = _apply_semantic_metadata(docs, filename, doc_type, resolved_ns, tags=docs[0].metadata.get("tags", []) if docs else [])
    n = _add_chunks(docs, pre_chunked=page_scoped)
    logger.info(f"/ingest/file: {filename} ({doc_type}) ns={resolved_ns} → {n} chunks")
    return IngestResponse(source=filename, chunks_indexed=n, index_size=_index_size(), namespace=resolved_ns)


@app.post("/ingest/folder", response_model=FolderImportResponse)
def ingest_folder(req: FolderImportRequest):
    """Import all supported files from a server-side path (local or mounted)."""
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(req.namespace)
    folder = Path(req.path).expanduser().resolve()
    _require_allowed_fs_path(folder)
    if not folder.exists():
        raise HTTPException(404, f"Path not found: {folder}")
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {folder}")

    glob = "**/*" if req.recursive else "*"
    all_files = []
    for p in folder.glob(glob):
        if (
            p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(folder).parts)
            and p.suffix.lower() in SUPPORTED_EXTS
            and not _junk_source_reason(str(p))
            and not _source_hygiene_reason(str(p), namespace)
        ):
            all_files.append(p)
            if len(all_files) > MAX_FOLDER_FILES:
                raise HTTPException(413, f"folder import exceeds {MAX_FOLDER_FILES} supported files")

    docs, errors = [], []
    files_indexed = 0
    total_chunks = 0
    for fpath in all_files:
        try:
            if fpath.stat().st_size > MAX_FOLDER_FILE_BYTES:
                errors.append(f"{fpath.name}: file too large")
                continue
            data = fpath.read_bytes()
            source = req.label + "/" + str(fpath.relative_to(folder)) if req.label else str(fpath)
            file_docs, page_scoped, _doc_type = _build_documents_from_file(data, fpath.name, namespace)
            for doc in file_docs:
                doc.metadata["source"] = source
            resolved_ns = file_docs[0].metadata.get("namespace", "inbox") if file_docs else "inbox"
            file_docs = _apply_semantic_metadata(file_docs, source, _doc_type, resolved_ns, tags=file_docs[0].metadata.get("tags", []) if file_docs else [])
            total_chunks += _add_chunks(file_docs, pre_chunked=page_scoped)
            docs.extend(file_docs)
            files_indexed += 1
        except Exception as e:
            errors.append(f"{fpath.name}: {e}")

    logger.info(f"/ingest/folder: {folder} files={files_indexed} chunks={total_chunks} errors={len(errors)}")
    return FolderImportResponse(
        path=str(folder),
        files_found=len(all_files),
        files_indexed=files_indexed,
        chunks_indexed=total_chunks,
        skipped=len(all_files) - files_indexed - len(errors),
        errors=errors[:20],
        index_size=_index_size(),
    )


@app.post("/ingest/files", response_model=FolderImportResponse)
async def ingest_files(files: List[UploadFile] = File(...), namespace: Optional[str] = None):
    """Upload multiple files at once (browser folder select)."""
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(namespace)
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"too many files; limit is {MAX_BATCH_FILES}")
    docs, errors = [], []
    files_indexed = 0
    total_chunks = 0
    for upload in files:
        filename = upload.filename or "upload"
        if Path(filename).suffix.lower() not in SUPPORTED_EXTS:
            errors.append(f"{filename}: unsupported type")
            continue
        junk_reason = _junk_source_reason(filename)
        if junk_reason:
            errors.append(f"{filename}: skipped junk source ({junk_reason})")
            continue
        hygiene_reason = _source_hygiene_reason(filename, namespace)
        if hygiene_reason:
            errors.append(f"{filename}: skipped trash source ({hygiene_reason})")
            continue
        try:
            data = await _read_upload_limited(upload)
            file_docs, page_scoped, _doc_type = _build_documents_from_file(data, filename, namespace)
            resolved_ns = file_docs[0].metadata.get("namespace", "inbox") if file_docs else "inbox"
            file_docs = _apply_semantic_metadata(file_docs, filename, _doc_type, resolved_ns, tags=file_docs[0].metadata.get("tags", []) if file_docs else [])
            total_chunks += _add_chunks(file_docs, pre_chunked=page_scoped)
            docs.extend(file_docs)
            files_indexed += 1
        except Exception as e:
            errors.append(f"{filename}: {e}")

    logger.info(f"/ingest/files: uploaded={len(files)} indexed={files_indexed} chunks={total_chunks}")
    return FolderImportResponse(
        path="browser-upload",
        files_found=len(files),
        files_indexed=files_indexed,
        chunks_indexed=total_chunks,
        skipped=len(files) - files_indexed - len(errors),
        errors=errors[:20],
        index_size=_index_size(),
    )


@app.get("/fs/ls")
def fs_ls(path: str = "/home/cgandycl"):
    """List directories at a path for the folder picker."""
    p = Path(path).expanduser().resolve()
    _require_allowed_fs_path(p)
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"Not found: {p}")
    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
                "ext": child.suffix.lower() if child.is_file() else None,
            })
    except PermissionError:
        pass
    return {"path": str(p), "parent": str(p.parent), "entries": entries}


@app.get("/fs/roots")
def fs_roots():
    """Return common local roots and mounted locations for the folder picker."""
    include_prefixes = tuple(str(root) for root in RAG_ALLOWED_FS_ROOTS)
    network_fs_types = {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "sshfs",
        "fuse.sshfs",
        "fuse.rclone",
        "davfs2",
    }
    candidates = list(RAG_ALLOWED_FS_ROOTS)
    mounts: List[Path] = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_path = Path(parts[1].replace("\\040", " "))
                fs_type = parts[2]
                mount_str = str(mount_path)
                allowed_prefix = any(mount_str == prefix or mount_str.startswith(prefix + "/") for prefix in include_prefixes)
                if allowed_prefix and fs_type in network_fs_types and mount_path.exists() and mount_path.is_dir():
                    mounts.append(mount_path)
    except OSError:
        pass

    seen = set()
    roots = []
    for path in candidates + mounts:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        path_str = str(resolved)
        if path_str in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(path_str)
        roots.append({
            "name": resolved.name or path_str,
            "path": path_str,
        })
    return {"roots": roots}


@app.post("/ingest/obsidian", response_model=ObsidianIngestResponse)
async def ingest_obsidian(file: UploadFile = File(...), namespace: Optional[str] = None):
    """Accept a .zip of an Obsidian vault (or any folder of .md files)."""
    if vectordb is None:
        raise HTTPException(503, "Index not ready")

    namespace = _normalize_namespace(namespace)
    data = await _read_upload_limited(file)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise HTTPException(415, "Upload must be a .zip file")

    docs: List[Document] = []
    skipped = 0

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        md_infos = [
            info for info in zf.infolist()
            if (info.filename.endswith(".md") or Path(info.filename).suffix.lower() in IMAGE_EXTS)
            and not Path(info.filename).name.startswith(".")
            and not _junk_source_reason(info.filename)
            and not _source_hygiene_reason(info.filename, namespace)
            and not info.is_dir()
        ]
        if len(md_infos) > MAX_ZIP_MEMBERS:
            raise HTTPException(413, f"zip has too many supported entries; limit is {MAX_ZIP_MEMBERS}")
        if sum(info.file_size for info in md_infos) > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise HTTPException(413, f"zip expands beyond {MAX_ZIP_UNCOMPRESSED_BYTES} bytes")
        for info in md_infos:
            name = info.filename
            try:
                raw_bytes = zf.read(info)
                ext = Path(name).suffix.lower()
                if ext in IMAGE_EXTS:
                    text = _extract_image(raw_bytes)
                    if not text.strip():
                        skipped += 1
                        continue
                    doc = _make_doc(text, name, "image", namespace=namespace)
                else:
                    raw = raw_bytes.decode("utf-8", errors="replace")
                    if not raw.strip():
                        skipped += 1
                        continue
                    doc = _parse_obsidian_note(name, raw)
                    doc.metadata["namespace"] = namespace or doc.metadata.get("namespace", "notes")
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
                skipped += 1

    if not docs:
        raise HTTPException(400, "No valid markdown notes found in zip")

    docs = [_apply_semantic_metadata([doc], doc.metadata.get("source", ""), doc.metadata.get("type", "obsidian"), doc.metadata.get("namespace", "notes"), tags=doc.metadata.get("tags", []))[0] for doc in docs]
    total_chunks = _add_chunks(docs)
    logger.info(f"/ingest/obsidian: notes={len(docs)} chunks={total_chunks} skipped={skipped}")
    return ObsidianIngestResponse(
        notes_found=len(md_infos),
        notes_indexed=len(docs),
        chunks_indexed=total_chunks,
        skipped=skipped,
        index_size=_index_size(),
    )


@app.get("/namespaces")
def list_namespaces():
    """Return each namespace with chunk count and source count."""
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    seen: Dict[str, dict] = {}
    for row in _iter_rows("", ["source", "namespace"]):
        ns  = row.get("namespace", "inbox")
        src = row.get("source", "")
        if ns not in seen:
            seen[ns] = {"namespace": ns, "chunk_count": 0, "sources": set()}
        seen[ns]["chunk_count"] += 1
        seen[ns]["sources"].add(src)
    result = [
        {"namespace": v["namespace"], "chunk_count": v["chunk_count"], "source_count": len(v["sources"])}
        for v in sorted(seen.values(), key=lambda x: x["chunk_count"], reverse=True)
    ]
    # Ensure all content namespaces appear even if empty
    existing = {r["namespace"] for r in result}
    for ns in CONTENT_NAMESPACES:
        if ns not in existing:
            result.append({"namespace": ns, "chunk_count": 0, "source_count": 0})
    return result


@app.get("/collections/smart")
def list_smart_collections():
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    seen: Dict[str, dict] = {}
    for row in _list_source_records():
        bucket = row.get("smart_collection") or "general"
        src = row.get("source", "")
        if bucket not in seen:
            seen[bucket] = {"smart_collection": bucket, "chunk_count": 0, "sources": set(), "examples": []}
        seen[bucket]["chunk_count"] += int(row.get("chunk_count", 0) or 0)
        if src:
            seen[bucket]["sources"].add(src)
        title = row.get("title")
        if title and title not in seen[bucket]["examples"] and len(seen[bucket]["examples"]) < 3:
            seen[bucket]["examples"].append(title)
    return sorted(
        [
            {
                "smart_collection": item["smart_collection"],
                "chunk_count": item["chunk_count"],
                "source_count": len(item["sources"]),
                "examples": item["examples"],
            }
            for item in seen.values()
        ],
        key=lambda item: item["chunk_count"],
        reverse=True,
    )


@app.get("/graph")
def graph(
    lite: bool = False,
    semantic: Optional[bool] = None,
    namespace: Optional[str] = None,
    profile: str = "curated",
    include_noise: bool = False,
    include_raw_chats: bool = False,
):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    include_semantic = semantic if semantic is not None else not lite
    detailed = not lite
    return _graph_snapshot(
        include_semantic=include_semantic,
        detailed=detailed,
        namespace=namespace,
        profile=profile,
        include_noise=include_noise,
        include_raw_chats=include_raw_chats,
    )


@app.get("/graph/layout")
def get_graph_layout():
    return _load_graph_layout()


@app.post("/graph/layout")
def save_graph_layout(req: GraphLayoutRequest):
    return _save_graph_layout(req.positions)


@app.get("/graph/views")
def list_graph_views():
    return {"views": _load_graph_views()}


@app.get("/analytics/retrieval")
def retrieval_analytics():
    records = _list_source_records()
    trace = _load_query_trace().get("events", [])
    hot = sorted(records, key=lambda item: (item.get("retrieve_count", 0) + item.get("query_count", 0) + item.get("synthesis_count", 0), item.get("last_activity_at") or ""), reverse=True)[:8]
    cold = [item for item in records if not any(int(item.get(key, 0) or 0) for key in ("query_count", "retrieve_count", "synthesis_count"))][:8]
    recent = trace[:8]
    return {
        "recent_queries": recent,
        "top_notes": [
            {
                "source": item["source"],
                "title": item.get("title"),
                "query_count": int(item.get("query_count", 0) or 0),
                "retrieve_count": int(item.get("retrieve_count", 0) or 0),
                "synthesis_count": int(item.get("synthesis_count", 0) or 0),
                "last_activity_at": item.get("last_activity_at"),
            }
            for item in hot
        ],
        "cold_notes": [
            {
                "source": item["source"],
                "title": item.get("title"),
                "namespace": item.get("namespace"),
            }
            for item in cold
        ],
    }


@app.post("/graph/views")
def save_graph_view(req: GraphViewRequest):
    views = [item for item in _load_graph_views() if item.get("name") != req.name]
    views.append(
        {
            "name": req.name[:80],
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state": req.state,
        }
    )
    views.sort(key=lambda item: item.get("name", "").lower())
    _save_graph_views(views)
    return {"saved": req.name[:80], "views": views}


@app.delete("/graph/views/{name}")
def delete_graph_view(name: str):
    views = [item for item in _load_graph_views() if item.get("name") != name]
    _save_graph_views(views)
    return {"deleted": name, "views": views}


@app.get("/second-brain/dashboard")
def second_brain_dashboard():
    all_records = [_record_with_brain_state(item) for item in _list_source_records()]
    records = _visible_second_brain_records(all_records)
    status_counts: Dict[str, int] = {status: 0 for status in sorted(CURATION_STATUSES)}
    trust_counts: Dict[str, int] = {level: 0 for level in sorted(TRUST_LEVELS)}
    namespace_counts: Dict[str, int] = {}
    for record in records:
        status_counts[record.get("curation_status", "unreviewed")] = status_counts.get(record.get("curation_status", "unreviewed"), 0) + 1
        trust_counts[record.get("trust_level", "unreviewed")] = trust_counts.get(record.get("trust_level", "unreviewed"), 0) + 1
        namespace = record.get("namespace") or "inbox"
        namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
    actions = _action_candidates(records, limit=200)
    return {
        "source_count": len(records),
        "hidden_trash_or_discarded": len(all_records) - len(records),
        "status_counts": status_counts,
        "trust_counts": trust_counts,
        "namespace_counts": namespace_counts,
        "inbox_count": sum(1 for item in records if item.get("curation_status") in {"unreviewed", "needs_action"}),
        "duplicate_candidates": len(_dedupe_candidates(records, limit=100)),
        "open_actions": sum(1 for item in actions if item.get("status") != "done"),
        "connectors": {name: _service_state(service) for name, service in CONNECTOR_SERVICES.items()},
        "housekeeping": _housekeeping_public_state(),
    }


@app.get("/housekeeping/status")
@app.get("/second-brain/housekeeping")
def housekeeping_status():
    return _housekeeping_public_state()


@app.post("/housekeeping/run")
@app.post("/second-brain/housekeeping/run")
def housekeeping_run():
    return _housekeeping_run_once(reason="manual")


@app.get("/second-brain/review")
def second_brain_review(status: Optional[str] = None, namespace: Optional[str] = None, limit: int = 50):
    namespace = _normalize_namespace(namespace)
    records = _visible_second_brain_records([_record_with_brain_state(item) for item in _list_source_records(namespace=namespace)])
    if status:
        records = [item for item in records if item.get("curation_status", "unreviewed") == status]
    records.sort(key=lambda item: (item.get("curation_status") != "needs_action", item.get("last_activity_at") or item.get("ingested_at") or ""), reverse=True)
    return {"items": records[:max(1, min(limit, 200))]}


@app.post("/second-brain/sources/curate")
def curate_source(req: CurateSourceRequest):
    if not _source_record(req.source):
        raise HTTPException(404, f"Source not found: {req.source}")
    updates: Dict[str, Any] = {}
    if req.curation_status is not None:
        if req.curation_status not in CURATION_STATUSES:
            raise HTTPException(400, f"curation_status must be one of: {', '.join(sorted(CURATION_STATUSES))}")
        updates["curation_status"] = req.curation_status
    if req.trust_level is not None:
        if req.trust_level not in TRUST_LEVELS:
            raise HTTPException(400, f"trust_level must be one of: {', '.join(sorted(TRUST_LEVELS))}")
        updates["trust_level"] = req.trust_level
    if req.review_note is not None:
        updates["review_note"] = req.review_note[:1000]
    if req.action_status is not None:
        updates["action_status"] = req.action_status[:64]
    state = _set_source_brain_state(req.source, updates)
    return {"source": req.source, **state}


@app.get("/second-brain/dedupe")
def second_brain_dedupe(limit: int = 25):
    records = _visible_second_brain_records([_record_with_brain_state(item) for item in _list_source_records()])
    return {"candidates": _dedupe_candidates(records, limit=max(1, min(limit, 100)))}


@app.get("/second-brain/actions")
def second_brain_actions(limit: int = 50):
    records = _visible_second_brain_records([_record_with_brain_state(item) for item in _list_source_records()])
    return {"items": _action_candidates(records, limit=max(1, min(limit, 200)))}


@app.post("/second-brain/actions/status")
def second_brain_action_status(req: ActionStatusRequest):
    if not _source_record(req.source):
        raise HTTPException(404, f"Source not found: {req.source}")
    state = _set_source_brain_state(req.source, {"action_status": req.status[:64]})
    return {"action_id": req.action_id, "source": req.source, **state}


@app.get("/second-brain/connectors")
def second_brain_connectors():
    return {"connectors": {name: _service_state(service) for name, service in CONNECTOR_SERVICES.items()}}


@app.get("/second-brain/export")
def second_brain_export():
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": MILVUS_COLLECTION,
        "sources": _load_json_file(SOURCE_CATALOG_PATH, {"records": []}),
        "second_brain": _load_second_brain_state(),
        "activity": _load_activity_log(),
        "authoring": _load_authoring_state(),
        "link_graph": _load_link_graph(),
        "query_trace": _load_query_trace(),
        "graph_layout": _load_graph_layout(),
        "note": "Metadata export only; vector rows are stored in Milvus.",
    }


@app.get("/sources", response_model=List[SourceItem])
def list_sources(namespace: Optional[str] = None):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    return _list_source_records(namespace=namespace)


@app.get("/sources/{source:path}", response_model=SourceDetail)
def source_detail(source: str):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    rows = mclient.query(
        collection_name=MILVUS_COLLECTION,
        filter=_milvus_eq("source", source),
        output_fields=["text", "source", "doc_type", "ingested_at", "title", "summary", "tags", "entities", "smart_collection", "page", "page_label", "chunk_role", "outbound_links", "unresolved_links"],
        limit=200,
    )
    if not rows:
        raise HTTPException(404, f"Source not found: {source}")
    base = rows[0]
    graph_links = _graph_links_for(source)
    records = _list_source_records()
    record = next((item for item in records if item["source"] == source), {})
    activity = _activity_for_source(source)
    authoring = _authoring_for_source(source)
    return SourceDetail(
        source=source,
        type=record.get("type") or base.get("doc_type", "unknown"),
        ingested_at=record.get("ingested_at") or base.get("ingested_at", ""),
        chunk_count=len(rows),
        title=record.get("title") or base.get("title"),
        summary=record.get("summary") or base.get("summary"),
        article=record.get("article") or base.get("article"),
        tags=list(record.get("tags") or base.get("tags") or []),
        entities=list(record.get("entities") or base.get("entities") or []),
        smart_collection=record.get("smart_collection") or base.get("smart_collection"),
        quality_score=record.get("quality_score") or base.get("quality_score"),
        quality_flags=list(record.get("quality_flags") or base.get("quality_flags") or []),
        retrieval_tier=record.get("retrieval_tier") or base.get("retrieval_tier"),
        related_sources=_related_sources(source, record.get("summary") or base.get("summary", ""), None),
        explicit_links=graph_links["explicit_links"],
        inferred_links=graph_links["inferred_links"],
        outbound_links=graph_links["outbound_links"] or list(base.get("outbound_links") or []),
        unresolved_links=graph_links["unresolved_links"] or list(base.get("unresolved_links") or []),
        backlinks=graph_links["backlinks"],
        manual_links=graph_links.get("manual_links", []),
        semantic_links=_semantic_links_for_source(source, records=records),
        query_count=activity["query_count"],
        retrieve_count=activity["retrieve_count"],
        synthesis_count=activity["synthesis_count"],
        ingest_count=activity["ingest_count"],
        last_activity_at=activity["last_activity_at"],
        pinned=authoring.get("pinned", False),
        aliases=authoring.get("aliases", []),
        template_name=authoring.get("template_name"),
        curation_status=authoring.get("curation_status", "inbox"),
        trust_level=authoring.get("trust_level", "unreviewed"),
        review_note=authoring.get("review_note", ""),
        full_text=_source_text(source, max_chars=120000).get("text", ""),
        chunks=[
            SourceChunk(
                text=r.get("text", ""),
                metadata={
                    "source": r.get("source"),
                    "type": r.get("doc_type"),
                    "page": r.get("page"),
                    "page_label": r.get("page_label"),
                    "chunk_role": r.get("chunk_role"),
                },
            )
            for r in rows
        ],
    )


@app.get("/sources/{source:path}/workspace", response_model=SourceWorkspace)
def source_workspace(source: str):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    if not _source_record(source) and not _source_rows(source, ["source"]):
        raise HTTPException(404, f"Source not found: {source}")
    return SourceWorkspace(**_source_workspace(source))


@app.post("/query/document", response_model=QueryResponse)
def query_document(req: DocumentQueryRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    try:
        answer, sources = _answer_document_query(req.source, req.query, deep=req.deep)
    except Exception as e:
        raise HTTPException(500, str(e))
    _record_activity(sources, "query")
    _append_query_trace("document-query", req.query, sources, sections=[{"kind": "document", "source": req.source, "title": req.source, "text": answer}], namespace=_source_record(req.source).get("namespace") if _source_record(req.source) else None)
    return QueryResponse(
        answer=answer,
        model=LLM_MODEL,
        sources=sources,
        namespace=_source_record(req.source).get("namespace") if _source_record(req.source) else None,
        mode="deep" if req.deep else "fast",
    )


@app.post("/ai/reanalyze")
def reanalyze_sources(req: ReanalyzeRequest):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    with _reanalyze_job_lock:
        if _reanalyze_job.get("running"):
            return dict(_reanalyze_job)
    candidates = _list_source_records(namespace=_normalize_namespace(req.namespace))
    if req.max_sources and req.max_sources > 0:
        candidates = candidates[:max(1, req.max_sources)]
    worker = threading.Thread(target=_run_reanalysis_job, args=([item["source"] for item in candidates],), daemon=True)
    worker.start()
    with _reanalyze_job_lock:
        return dict(_reanalyze_job)


@app.get("/ai/reanalyze/status")
def reanalyze_status():
    with _reanalyze_job_lock:
        return dict(_reanalyze_job)


def _junk_purge_candidates(include_web: bool = True) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for item in _list_source_records():
        source = str(item.get("source") or "")
        if not source or source in seen:
            continue
        reason = None
        if include_web and item.get("namespace") == "web":
            reason = "web namespace"
        if not reason:
            reason = _junk_source_reason(source, item.get("type") or "")
        if not reason:
            reason = _source_hygiene_reason(source, item.get("namespace"), item.get("tags"), " ".join(str(item.get(key) or "") for key in ("title", "summary")))
        if not reason:
            continue
        seen.add(source)
        candidates.append({
            "source": source,
            "namespace": item.get("namespace"),
            "type": item.get("type"),
            "chunk_count": int(item.get("chunk_count", 0) or 0),
            "reason": reason,
        })
    return candidates


def _remove_sources_from_metadata(sources: set[str]) -> None:
    if not sources:
        return

    graph = _load_link_graph()
    nodes = graph.setdefault("nodes", {})
    for source in sources:
        nodes.pop(source, None)
    for node in nodes.values():
        for key in ("outbound_links", "explicit_links", "inferred_links", "unresolved_links"):
            node[key] = [item for item in node.get(key, []) if item not in sources]
    _save_link_graph(graph)

    activity = _load_activity_log()
    activity_sources = activity.setdefault("sources", {})
    for source in sources:
        activity_sources.pop(source, None)
    _save_activity_log(activity)

    authoring = _load_authoring_state()
    authoring_sources = authoring.setdefault("sources", {})
    for source in sources:
        authoring_sources.pop(source, None)
    _save_authoring_state(authoring)

    records = [item for item in _load_source_catalog_records() if item.get("source") not in sources]
    _save_source_catalog_records(records)


def _noise_review_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    source = str(record.get("source") or "")
    namespace = str(record.get("namespace") or "")
    doc_type = str(record.get("type") or "")
    tags = list(record.get("tags") or [])
    sample_text = "\n\n".join(str(record.get(key) or "") for key in ("title", "summary", "article"))
    profile = _source_quality_profile(sample_text, source, doc_type, namespace, tags=tags)
    reason = _junk_source_reason(source, doc_type) or _source_hygiene_reason(source, namespace, tags, sample_text)
    action = "none"
    updates: Dict[str, str] = {}
    if reason:
        action = "junk"
        updates = {
            "curation_status": "junk",
            "trust_level": "discard",
            "review_note": f"Auto-curated as junk: {reason}",
        }
    elif namespace in NOISE_NAMESPACES:
        action = "archive-noisy-namespace"
        updates = {
            "curation_status": "archived",
            "trust_level": "use_with_caution",
            "review_note": "Auto-archived noisy namespace; still searchable with include_noise.",
        }
    elif _is_raw_context_record(namespace, source, tags):
        action = "raw-chat-limited"
        updates = {
            "curation_status": "inbox",
            "trust_level": "use_with_caution",
            "review_note": "Raw chat kept searchable but excluded from default all-brain retrieval unless explicitly requested or used as fallback.",
        }
    elif profile["retrieval_tier"] == "noise":
        action = "quality-noise"
        updates = {
            "curation_status": "needs_action",
            "trust_level": "questionable",
            "review_note": f"Auto-flagged quality noise: {', '.join(profile['quality_flags']) or 'low quality score'}",
        }
    elif profile["retrieval_tier"] == "low":
        action = "quality-low"
        updates = {
            "curation_status": "needs_action",
            "trust_level": "use_with_caution",
            "review_note": f"Auto-flagged low-signal source: {', '.join(profile['quality_flags']) or 'low quality score'}",
        }
    return {"source": source, "action": action, **profile, **updates}


@app.post("/admin/curate-noise")
def curate_noise(req: NoiseCurationRequest):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(req.namespace)
    records = _list_source_records(namespace=namespace)
    if req.limit and req.limit > 0:
        records = records[:max(1, req.limit)]

    reviews = [_noise_review_for_record(record) for record in records]
    action_counts: Dict[str, int] = {}
    for review in reviews:
        action = review.get("action", "none")
        action_counts[action] = action_counts.get(action, 0) + 1

    if not req.dry_run:
        catalog = {item["source"]: dict(item) for item in _load_source_catalog_records()}
        for review in reviews:
            source = review["source"]
            if source in catalog:
                catalog[source]["quality_score"] = review.get("quality_score")
                catalog[source]["quality_flags"] = list(review.get("quality_flags") or [])
                catalog[source]["retrieval_tier"] = review.get("retrieval_tier")
                catalog[source]["tags"] = _merge_tags(
                    list(catalog[source].get("tags") or []),
                    [f"quality-{review.get('retrieval_tier') or 'unknown'}"],
                    [f"flag-{flag}" for flag in list(review.get("quality_flags") or [])[:6]],
                )
            if review.get("action") != "none":
                prior = _authoring_for_source(source)
                _save_authoring_for_source(
                    source,
                    prior.get("aliases", []),
                    prior.get("manual_links", []),
                    prior.get("pinned", False),
                    prior.get("template_name"),
                    curation_status=review.get("curation_status"),
                    trust_level=review.get("trust_level"),
                    review_note=review.get("review_note"),
                )
                _set_source_brain_state(
                    source,
                    {
                        "curation_status": review.get("curation_status"),
                        "trust_level": review.get("trust_level"),
                        "review_note": review.get("review_note"),
                    },
                )
        _save_source_catalog_records(list(catalog.values()))
        _invalidate_catalog_cache()

    return {
        "dry_run": req.dry_run,
        "sources_reviewed": len(reviews),
        "action_counts": action_counts,
        "sample": reviews[:25],
    }


@app.post("/admin/purge-junk")
def purge_junk(req: PurgeJunkRequest):
    if mclient is None:
        raise HTTPException(503, "Index not ready")

    candidates = _junk_purge_candidates(include_web=req.include_web)
    reason_counts: Dict[str, int] = {}
    chunk_count = 0
    for item in candidates:
        reason = item["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        chunk_count += int(item.get("chunk_count", 0) or 0)

    if req.dry_run:
        return {
            "dry_run": True,
            "candidate_sources": len(candidates),
            "candidate_chunks": chunk_count,
            "reason_counts": reason_counts,
            "sample": candidates[:25],
            "index_size": _index_size(),
        }

    deleted_sources: set[str] = set()
    errors: List[Dict[str, str]] = []
    for item in candidates:
        source = item["source"]
        try:
            mclient.delete(collection_name=MILVUS_COLLECTION, filter=_milvus_eq("source", source))
            deleted_sources.add(source)
        except Exception as exc:
            errors.append({"source": source, "error": str(exc)})

    _remove_sources_from_metadata(deleted_sources)
    _invalidate_catalog_cache()
    logger.warning("Purged %s junk source(s), %s failed", len(deleted_sources), len(errors))
    return {
        "dry_run": False,
        "deleted_sources": len(deleted_sources),
        "failed_sources": len(errors),
        "candidate_chunks": chunk_count,
        "reason_counts": reason_counts,
        "errors": errors[:20],
        "index_size": _index_size(),
    }


@app.delete("/sources/{source:path}")
def delete_source(source: str):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    mclient.delete(
        collection_name=MILVUS_COLLECTION,
        filter=_milvus_eq("source", source),
    )
    _remove_from_link_graph(source)
    _remove_from_activity_log(source)
    _remove_from_authoring_state(source)
    records = [item for item in _load_source_catalog_records() if item.get("source") != source]
    _save_source_catalog_records(records)
    _invalidate_catalog_cache()
    logger.info(f"Deleted source: {source}")
    return {"deleted": source, "index_size": _index_size()}


@app.post("/sources/curate")
def curate_source(req: SourceCurationRequest):
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    current = _source_record(req.source)
    if not current and not _source_rows(req.source, ["source"]):
        raise HTTPException(404, f"Source not found: {req.source}")
    prior = _authoring_for_source(req.source)
    aliases_source = prior.get("aliases") if req.aliases is None else req.aliases
    manual_source = prior.get("manual_links") if req.manual_links is None else req.manual_links
    aliases = [str(item).strip() for item in (aliases_source or []) if str(item).strip()]
    manual_links = [str(item).strip() for item in (manual_source or []) if str(item).strip()]
    pinned = prior.get("pinned", False) if req.pinned is None else bool(req.pinned)
    template_name = prior.get("template_name") if req.template_name is None else req.template_name
    curation_status = prior.get("curation_status", "inbox") if req.curation_status is None else req.curation_status
    trust_level = prior.get("trust_level", "unreviewed") if req.trust_level is None else req.trust_level
    review_note = prior.get("review_note", "") if req.review_note is None else req.review_note
    _save_authoring_for_source(
        req.source,
        aliases,
        manual_links,
        pinned,
        template_name,
        curation_status=curation_status,
        trust_level=trust_level,
        review_note=review_note,
    )
    text = _source_text(req.source, max_chars=120000).get("text", "")
    title = (current or {}).get("title") or req.source
    _upsert_link_graph(req.source, title, _extract_explicit_links(text), manual_links=manual_links, aliases=aliases)
    _invalidate_catalog_cache()
    return {
        "source": req.source,
        "pinned": pinned,
        "aliases": aliases,
        "manual_links": manual_links,
        "template_name": template_name,
        "curation_status": _normalize_curation_status(curation_status),
        "trust_level": _normalize_trust_level(trust_level),
        "review_note": str(review_note or "").strip()[:1000],
    }


@app.post("/sources/update", response_model=IngestResponse)
def update_source(req: UpdateSourceRequest):
    if vectordb is None or mclient is None:
        raise HTTPException(503, "Index not ready")
    current = _source_record(req.source)
    if not current:
        raise HTTPException(404, f"Source not found: {req.source}")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text is too large; limit is {MAX_TEXT_CHARS} characters")
    namespace = _normalize_namespace(req.namespace) or _normalize_namespace(current.get("namespace")) or "notes"
    doc_type = current.get("type") or "markdown"
    target_source = (req.new_source or req.source).strip() or req.source
    junk_reason = _junk_source_reason(target_source, doc_type)
    if junk_reason:
        raise HTTPException(400, f"Refusing to index junk source: {junk_reason}")
    tags = list(req.tags or current.get("tags") or [])
    prior_activity = _load_activity_log().get("sources", {}).get(req.source, {})
    prior_authoring = _authoring_for_source(req.source)
    aliases_source = prior_authoring.get("aliases") if req.aliases is None else req.aliases
    manual_source = prior_authoring.get("manual_links") if req.manual_links is None else req.manual_links
    aliases = [str(item).strip() for item in (aliases_source or []) if str(item).strip()]
    manual_links = [str(item).strip() for item in (manual_source or []) if str(item).strip()]
    pinned = prior_authoring.get("pinned", False) if req.pinned is None else bool(req.pinned)
    template_name = prior_authoring.get("template_name") if req.template_name is None else req.template_name
    curation_status = prior_authoring.get("curation_status", "inbox") if req.curation_status is None else req.curation_status
    trust_level = prior_authoring.get("trust_level", "unreviewed") if req.trust_level is None else req.trust_level
    review_note = prior_authoring.get("review_note", "") if req.review_note is None else req.review_note
    if doc_type not in tags:
        tags.append(doc_type)
    mclient.delete(collection_name=MILVUS_COLLECTION, filter=_milvus_eq("source", req.source))
    _remove_from_link_graph(req.source)
    _remove_from_activity_log(req.source)
    records = [item for item in _load_source_catalog_records() if item.get("source") != req.source]
    _save_source_catalog_records(records)
    if target_source != req.source:
        _rename_authoring_state(req.source, target_source)
    _save_authoring_for_source(
        target_source,
        aliases,
        manual_links,
        pinned,
        template_name,
        curation_status=curation_status,
        trust_level=trust_level,
        review_note=review_note,
    )
    docs = _apply_semantic_metadata(
        [_make_doc(text, target_source, doc_type, {"tags": tags, "link_targets": _extract_explicit_links(text) + manual_links}, namespace=namespace)],
        target_source,
        doc_type,
        namespace,
        tags=tags,
    )
    if req.title:
        docs[0].metadata["title"] = req.title[:200]
    chunk_count = _add_chunks(docs)
    if prior_activity:
        _set_activity_entry(target_source, prior_activity)
    _invalidate_catalog_cache()
    return IngestResponse(source=target_source, chunks_indexed=chunk_count, index_size=_index_size(), namespace=namespace)


@app.post("/links/rebuild")
def rebuild_links():
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    graph = {"nodes": {}}
    _save_link_graph(graph)
    rebuilt = 0
    records: List[Dict[str, Any]] = []
    for item in _list_source_records():
        rows = mclient.query(
            collection_name=MILVUS_COLLECTION,
            filter=_milvus_eq("source", item["source"]),
            output_fields=["text", "title", "summary", "tags", "entities", "smart_collection", "namespace", "doc_type"],
            limit=1,
        )
        if not rows:
            continue
        row = rows[0]
        raw_links = _extract_explicit_links(row.get("text", ""))
        title = row.get("title") or item.get("title") or item["source"]
        _upsert_link_graph(item["source"], title, raw_links)
        records.append({
            **item,
            "title": title,
            "summary": row.get("summary") or item.get("summary") or "",
            "tags": list(row.get("tags") or item.get("tags") or []),
            "entities": list(row.get("entities") or item.get("entities") or []),
            "smart_collection": row.get("smart_collection") or item.get("smart_collection"),
            "namespace": row.get("namespace") or item.get("namespace"),
            "type": row.get("doc_type") or item.get("type"),
        })
        rebuilt += 1
    graph = _load_link_graph()
    nodes = _reconstruct_deep_links(records, graph)
    _invalidate_catalog_cache()
    explicit_edges = sum(len(node.get("explicit_links", [])) for node in nodes.values())
    inferred_edges = sum(len(node.get("inferred_links", [])) for node in nodes.values())
    return {
        "rebuilt_sources": rebuilt,
        "explicit_edges": explicit_edges,
        "inferred_edges": inferred_edges,
    }


@app.delete("/namespaces/{namespace}")
def delete_namespace(namespace: str):
    """Delete all chunks in a namespace (e.g. move to archives first)."""
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(namespace)
    if not namespace:
        raise HTTPException(400, "namespace is required")
    mclient.delete(collection_name=MILVUS_COLLECTION, filter=_milvus_eq("namespace", namespace))
    records = [item for item in _load_source_catalog_records() if item.get("namespace") != namespace]
    for item in [item for item in _load_source_catalog_records() if item.get("namespace") == namespace]:
        _remove_from_authoring_state(item.get("source", ""))
    _save_source_catalog_records(records)
    _invalidate_catalog_cache()
    logger.info(f"Deleted namespace: {namespace}")
    return {"deleted_namespace": namespace, "index_size": _index_size()}


class MoveRequest(BaseModel):
    source: str
    namespace: str

@app.post("/sources/move")
def move_source(req: MoveRequest):
    """Reassign all chunks of a source to a different namespace."""
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    namespace = _normalize_namespace(req.namespace)
    if not namespace:
        raise HTTPException(400, "namespace is required")
    source_filter = _milvus_eq("source", req.source)
    # Milvus doesn't support partial update; fetch → delete → reinsert
    rows = list(
        _iter_rows(
            source_filter,
            ["id", "text", "source", "doc_type", "ingested_at", "title", "summary", "tags", "entities", "smart_collection", "page", "page_label", "chunk_role"],
            batch_size=512,
        )
    )
    if not rows:
        raise HTTPException(404, f"Source not found: {req.source}")
    ids = [r["id"] for r in rows]
    # Fetch vectors separately
    vec_rows = mclient.get(collection_name=MILVUS_COLLECTION, ids=ids, output_fields=["id", "vector"])
    id_to_vec = {r["id"]: r["vector"] for r in vec_rows}
    missing_vectors = [rid for rid in ids if rid not in id_to_vec]
    if missing_vectors:
        raise HTTPException(500, f"missing vectors for {len(missing_vectors)} chunks; move aborted")
    new_rows = [{
        "id":          r["id"],
        "vector":      id_to_vec[r["id"]],
        "text":        r.get("text", "")[:65500],
        "source":      r.get("source", "")[:2048],
        "doc_type":    r.get("doc_type", "text")[:64],
        "ingested_at": r.get("ingested_at", "")[:32],
        "namespace":   namespace,
        "title":       r.get("title"),
        "summary":     r.get("summary"),
        "tags":        r.get("tags") or [],
        "entities":    r.get("entities") or [],
        "smart_collection": r.get("smart_collection") or namespace,
        "page":        r.get("page"),
        "page_label":  r.get("page_label"),
        "chunk_role":  r.get("chunk_role"),
    } for r in rows]
    mclient.delete(collection_name=MILVUS_COLLECTION, filter=source_filter)
    mclient.insert(MILVUS_COLLECTION, new_rows)
    record_map = {item["source"]: dict(item) for item in _load_source_catalog_records()}
    if req.source in record_map:
        record_map[req.source]["namespace"] = namespace
        _save_source_catalog_records(list(record_map.values()))
    _invalidate_catalog_cache()
    logger.info(f"Moved {req.source} → namespace {namespace} ({len(rows)} chunks)")
    return {"source": req.source, "namespace": namespace, "chunks_moved": len(rows)}


_RAG_ADMIN_DISABLED = os.getenv("RAG_ADMIN_DISABLED", "0").lower() in {"1","true","yes"}

@app.delete("/admin/reset")
def reset_collection():
    if _RAG_ADMIN_DISABLED:
        logger.warning("/admin/reset blocked: RAG_ADMIN_DISABLED=true")
        raise HTTPException(403, "admin/reset disabled by RAG_ADMIN_DISABLED env")
    if mclient is None:
        raise HTTPException(503, "Index not ready")
    mclient.delete(collection_name=MILVUS_COLLECTION, filter="id != ''")
    _save_source_catalog_records([])
    _save_activity_log({"sources": {}})
    _save_authoring_state({"sources": {}})
    _save_query_trace({"events": []})
    _save_json_file(SEMANTIC_GRAPH_PATH, {"signatures": {}, "neighbors": {}})
    _invalidate_catalog_cache()
    logger.warning("Deleted all indexed sources")
    return {"deleted": "all", "index_size": _index_size(), "sources": 0}


@app.post("/retrieve", response_model=List[RetrievedChunk])
def retrieve(req: RetrieveRequest):
    if vectordb is None:
        raise HTTPException(503, "Index not ready")
    k = max(1, min(req.k, 20))
    namespace = _normalize_namespace(req.namespace)
    records = _list_source_records(namespace=namespace)
    records_by_source = {record["source"]: record for record in records}
    docs = vectordb.similarity_search(req.query, k=k * 8, namespace=namespace, source_filter=req.source_filter)
    if docs and reranker_inst:
        docs = reranker_inst.compress_documents(docs, req.query)
    docs = _select_retrieval_docs(
        docs,
        records_by_source,
        k,
        explicit_scope=bool(namespace or req.source_filter),
        include_noise=req.include_noise,
        include_raw_chats=req.include_raw_chats,
    )
    _record_activity([d.metadata.get("source", "") for d in docs[:k]], "retrieve")
    _append_query_trace(
        "retrieve",
        req.query,
        [d.metadata.get("source", "") for d in docs[:k]],
        sections=[{"kind": "retrieved_chunk", "source": d.metadata.get("source", ""), "title": d.metadata.get("source", ""), "text": d.page_content} for d in docs[:k]],
        namespace=namespace,
    )
    return [
        RetrievedChunk(
            text=d.page_content,
            source=d.metadata.get("source", ""),
        )
        for d in docs[:k]
    ]


@app.post("/synthesis/source", response_model=SynthesisResponse)
def synthesize_source(req: SourceSynthesisRequest):
    section = _source_text(req.source)
    _record_activity([req.source], "synthesis")
    return _synthesize(
        title=f"Briefing: {section['title']}",
        sections=[section],
        instruction=req.instruction or "Summarize the source, key takeaways, action items, and risks.",
        mode="source",
        smart_collection=section.get("smart_collection"),
    )


@app.post("/synthesis/collection", response_model=SynthesisResponse)
def synthesize_collection(req: CollectionSynthesisRequest):
    names = _collection_source_names(req.smart_collection, max_sources=max(1, min(req.max_sources, 20)))
    if not names:
        raise HTTPException(404, f"Smart collection not found or empty: {req.smart_collection}")
    _record_activity(names, "synthesis")
    sections = [_source_text(name, max_chars=12000) for name in names]
    return _synthesize(
        title=f"Collection Synthesis: {req.smart_collection}",
        sections=sections,
        instruction=req.instruction or "Summarize the major themes, recurring patterns, active work, and notable gaps.",
        mode="collection",
        smart_collection=req.smart_collection,
    )


@app.post("/synthesis/compare", response_model=SynthesisResponse)
def compare_sources(req: CompareSourcesRequest):
    sources = list(dict.fromkeys(req.sources))
    if len(sources) < 2:
        raise HTTPException(400, "Provide at least two sources to compare")
    _record_activity(sources, "synthesis")
    sections = [_source_text(source, max_chars=10000) for source in sources[:8]]
    return _synthesize(
        title="Source Comparison",
        sections=sections,
        instruction=req.instruction or "Compare these sources. Highlight overlap, differences, and unique material.",
        mode="compare",
    )


@app.post("/synthesis/changes", response_model=SynthesisResponse)
def summarize_changes(req: ChangeSummaryRequest):
    earlier = _source_text(req.earlier_source, max_chars=10000)
    later = _source_text(req.later_source, max_chars=10000)
    _record_activity([req.earlier_source, req.later_source], "synthesis")
    return _synthesize(
        title="Change Summary",
        sections=[earlier, later],
        instruction=req.instruction or "Describe what changed from the earlier source to the later source.",
        mode="changes",
    )


# ── Built-in UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    if not FRONTEND_INDEX_PATH.exists():
        raise HTTPException(500, f"Missing frontend file: {FRONTEND_INDEX_PATH}")
    return FileResponse(FRONTEND_INDEX_PATH, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/admin", response_class=HTMLResponse)
def admin_ui():
    if not FRONTEND_ADMIN_PATH.exists():
        raise HTTPException(500, f"Missing frontend file: {FRONTEND_ADMIN_PATH}")
    return FileResponse(FRONTEND_ADMIN_PATH, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/memory", response_class=HTMLResponse)
def memory_ui():
    if not FRONTEND_MEMORY_PATH.exists():
        raise HTTPException(500, f"Missing frontend file: {FRONTEND_MEMORY_PATH}")
    return FileResponse(
        FRONTEND_MEMORY_PATH,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/capture", response_class=HTMLResponse)
@app.get("/c", response_class=HTMLResponse)
def capture_ui():
    if not FRONTEND_CAPTURE_PATH.exists():
        raise HTTPException(500, f"Missing frontend file: {FRONTEND_CAPTURE_PATH}")
    return FileResponse(
        FRONTEND_CAPTURE_PATH,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# Frontend is served from external static HTML files in ./frontend
