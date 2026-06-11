"""Mirror the shared workshop collection into a local (writable) Qdrant.

Reads every point — payloads AND both named vectors — from the cloud
cluster using the read-only workshop key, and upserts them into a local
Qdrant (docker run -p 6333:6333 qdrant/qdrant). The local collection gets
the same schema plus an extra `quarter` keyword index so our MCP tools can
filter on it.

Usage:
    python scripts/migrate_to_local.py

Also importable: run_migration() powers the web app's "Clone" button.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from qdrant_client import QdrantClient, models

COLLECTION = os.getenv("COLLECTION_NAME", "earnings_calls")
BATCH = 32

PAYLOAD_INDEXES = (
    ("ticker", models.PayloadSchemaType.KEYWORD),
    ("speaker", models.PayloadSchemaType.KEYWORD),
    ("quarter", models.PayloadSchemaType.KEYWORD),  # extra vs. cloud
    ("year", models.PayloadSchemaType.INTEGER),
    ("date", models.PayloadSchemaType.DATETIME),
)


def run_migration(
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Clone the cloud collection into the local Qdrant. Returns counts."""
    cloud_url = os.environ["CLOUD_QDRANT_URL"]
    cloud_key = os.environ["CLOUD_QDRANT_API_KEY"]
    local_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    cloud = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=60)
    local = QdrantClient(url=local_url, timeout=60)

    total = cloud.get_collection(COLLECTION).points_count or 0

    if local.collection_exists(COLLECTION):
        local.delete_collection(COLLECTION)

    local.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "text": models.VectorParams(size=3072, distance=models.Distance.COSINE),
            "audio": models.VectorParams(size=3072, distance=models.Distance.COSINE),
        },
        hnsw_config=models.HnswConfigDiff(m=16, payload_m=16),
    )
    for field, schema in PAYLOAD_INDEXES:
        local.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

    copied = 0
    offset = None
    while True:
        points, offset = cloud.scroll(
            collection_name=COLLECTION,
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        local.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(id=p.id, vector=cast(Any, p.vector or {}), payload=p.payload)
                for p in points
            ],
            wait=True,
        )
        copied += len(points)
        if progress:
            progress(copied, total)
        if offset is None:
            break

    local_count = local.get_collection(COLLECTION).points_count or 0
    return {"cloud_points": total, "local_points": local_count}


if __name__ == "__main__":
    print(f"Cloning collection {COLLECTION!r} from cloud to local...")
    stats = run_migration(progress=lambda c, t: print(f"\r  copied {c}/{t}", end="", flush=True))
    print(f"\nDone. Local collection has {stats['local_points']} points.")
    assert stats["local_points"] == stats["cloud_points"], stats
