import uuid
from typing import Dict, List

import requests
from pymilvus import DataType, MilvusClient


SRC_COLLECTION = "second_brain"
DST_COLLECTION = "second_brain_nvidia"
MILVUS_URL = "http://localhost:19530"
EMBED_BASE_URL = "http://127.0.0.1:8012/v1"
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
EMBED_DIM = 2048
BATCH_SIZE = 64


def ensure_collection(client: MilvusClient, name: str, dim: int) -> None:
    if client.has_collection(name):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=2048)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
    schema.add_field("ingested_at", DataType.VARCHAR, max_length=32)
    schema.add_field("namespace", DataType.VARCHAR, max_length=128)
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
    client.load_collection(name)


def chunked(items: List[Dict], size: int):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def recreate_collection(client: MilvusClient, name: str, dim: int) -> None:
    if client.has_collection(name):
        client.drop_collection(name)
    ensure_collection(client, name, dim)


def main() -> None:
    client = MilvusClient(uri=MILVUS_URL)
    recreate_collection(client, DST_COLLECTION, EMBED_DIM)

    total = 0
    iterator = client.query_iterator(
        collection_name=SRC_COLLECTION,
        batch_size=200,
        output_fields=["text", "source", "doc_type", "ingested_at", "namespace"],
    )
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break

            for batch in chunked(rows, BATCH_SIZE):
                texts = [row.get("text", "") for row in batch]
                response = requests.post(
                    f"{EMBED_BASE_URL}/embeddings",
                    json={"model": EMBED_MODEL, "input": texts, "input_type": "document"},
                    timeout=120,
                )
                response.raise_for_status()
                vectors = [item["embedding"] for item in response.json().get("data", [])]
                payload = []
                for row, vector in zip(batch, vectors):
                    payload.append({
                        "id": str(uuid.uuid4()).replace("-", "")[:64],
                        "vector": vector,
                        "text": row.get("text", "")[:65500],
                        "source": row.get("source", "unknown")[:2048],
                        "doc_type": row.get("doc_type", "text")[:64],
                        "ingested_at": row.get("ingested_at", "")[:32],
                        "namespace": row.get("namespace", "inbox")[:128],
                    })
                client.insert(DST_COLLECTION, payload)
                total += len(payload)
                print(f"migrated={total}", flush=True)
    finally:
        iterator.close()

    stats = client.get_collection_stats(DST_COLLECTION)
    print({"dst_collection": DST_COLLECTION, "rows": stats.get("row_count", total)}, flush=True)


if __name__ == "__main__":
    main()
