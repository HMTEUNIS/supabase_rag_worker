import logging
from typing import Any

from supabase import Client

from rag.config import ProjectConfig

logger = logging.getLogger(__name__)


def query_similar_docs(
    supabase: Client,
    project: ProjectConfig,
    embedding: list[float],
    match_count: int,
    docs_filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Calls project-configured PostgREST RPC. Params must match your SQL function signature.
    Convention (see sql/example_match_documents.sql):
      p_query_embedding, p_match_count, p_category, p_tags
    Extra keys in docs_filters are passed through as p_<key> if your RPC accepts them.
    """
    params: dict[str, Any] = {
        "p_query_embedding": embedding,
        "p_match_count": match_count,
        "p_category": docs_filters.get("category"),
        "p_tags": docs_filters.get("tags"),
    }

    for k, v in docs_filters.items():
        if k in ("category", "tags"):
            continue
        pk = k if k.startswith("p_") else f"p_{k}"
        params[pk] = v

    try:
        res = supabase.rpc(project.vector_rpc, params).execute()
    except Exception as e:  # noqa: BLE001
        logger.exception("RPC %s failed: %s", project.vector_rpc, e)
        raise

    data = getattr(res, "data", None)
    if data is None:
        return []
    if not isinstance(data, list):
        return [data] if isinstance(data, dict) else []
    return list(data)
