import logging
import time
from typing import Any

from supabase import Client, create_client

from models.request import BackfillEmbeddingsRequest, BackfillEmbeddingsResponse
from rag.config import GlobalSettings, load_global_settings, load_project_config
from rag.embeddings import generate_embedding

logger = logging.getLogger(__name__)


def backfill_embeddings(
    req: BackfillEmbeddingsRequest,
    *,
    settings: GlobalSettings | None = None,
    supabase: Client | None = None,
) -> BackfillEmbeddingsResponse:
    settings = settings or load_global_settings()
    project = load_project_config(req.project_id)

    if supabase is None:
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)

    table = req.table or project.docs_table or "knowledge_base"
    content_column = req.content_column or project.docs_content_column or "content"
    embedding_column = req.embedding_column or project.docs_embedding_column or "embedding"

    started = time.perf_counter()
    updated = 0
    candidates_checked = 0

    while updated < req.limit:
        batch_limit = min(req.batch_size, req.limit - updated)

        # Fetch only rows that need embeddings.
        rows_resp = (
            supabase.table(table)
            .select(f"id,{content_column}")
            .is_(embedding_column, None)
            .limit(batch_limit)
            .execute()
        )
        rows: list[dict[str, Any]] = rows_resp.data or []
        if not rows:
            break

        candidates_checked += len(rows)

        for row in rows:
            if updated >= req.limit:
                break
            doc_id = row["id"]
            text = row.get(content_column) or ""
            vec = generate_embedding(text, settings)
            if not req.dry_run:
                supabase.table(table).update({embedding_column: vec}).eq("id", doc_id).execute()
            updated += 1

        if req.sleep_seconds > 0:
            time.sleep(req.sleep_seconds)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return BackfillEmbeddingsResponse(
        updated=updated,
        candidates_checked=candidates_checked,
        table=table,
        content_column=content_column,
        embedding_column=embedding_column,
        dry_run=req.dry_run,
        processing_time_ms=elapsed_ms,
    )

