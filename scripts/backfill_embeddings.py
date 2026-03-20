import os
import time
from typing import Any

from supabase import create_client

from rag.config import load_global_settings
from rag.embeddings import generate_embedding


def main() -> None:
    settings = load_global_settings()
    if not settings.embedding_dimension:
        raise SystemExit("EMBEDDING_DIMENSION must be set for backfill")

    supabase = create_client(settings.supabase_url, settings.supabase_service_key)

    # Demo default: backfill the knowledge base table used by the worker.
    table = os.getenv("EMBEDDINGS_TABLE", "knowledge_base")
    content_col = os.getenv("EMBEDDINGS_CONTENT_COLUMN", "content")
    embedding_col = os.getenv("EMBEDDINGS_EMBEDDING_COLUMN", "embedding")

    batch_size = int(os.getenv("EMBEDDINGS_BACKFILL_BATCH_SIZE", "50"))
    sleep_s = float(os.getenv("EMBEDDINGS_SLEEP_SECONDS", "0.0"))

    # Fetch rows without embeddings. We only pull what we need.
    rows_resp = (
        supabase.table(table)
        .select("id," + content_col)
        .is_(embedding_col, None)
        .limit(batch_size)
        .execute()
    )
    rows: list[dict[str, Any]] = rows_resp.data or []

    if not rows:
        print("No rows found with NULL embeddings. Nothing to do.")
        return

    print(f"Backfilling {len(rows)} rows into `{table}` (embedding NULL).")

    for i, row in enumerate(rows, start=1):
        doc_id = row["id"]
        text = row.get(content_col) or ""
        vec = generate_embedding(text, settings)

        supabase.table(table).update({embedding_col: vec}).eq("id", doc_id).execute()

        print(f"[{i}/{len(rows)}] updated id={doc_id}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    print("Done. If you have more NULL rows, re-run the script (it backfills in batches).")


if __name__ == "__main__":
    main()

