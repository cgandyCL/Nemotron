#!/usr/bin/env bash
set -euo pipefail

cd "/home/cgandycl/Nemotron/use-case-examples/RAG Agent with Nemotron RAG Models"

export VLLM_LLM_BASE_URL="${VLLM_LLM_BASE_URL:-http://192.168.100.11:8000/v1}"
export VLLM_EMBED_BASE_URL="http://127.0.0.1:8012/v1"
export LLM_MODEL="${LLM_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
export EMBED_MODEL="nvidia/llama-3.2-nv-embedqa-1b-v2"
export EMBED_DIM="2048"
export MILVUS_URL="http://localhost:19530"
export MILVUS_COLLECTION="second_brain_nvidia"

PORT="${RAG_PORT:-8011}"
HOST="${RAG_HOST:-127.0.0.1}"

exec /home/cgandycl/jupyterlab/.venv/bin/uvicorn rag_api:app --host "${HOST}" --port "${PORT}"
