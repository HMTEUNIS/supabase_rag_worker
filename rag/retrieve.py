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
      query_embedding, match_count, match_threshold, filter_category, filter_tags
    Extra keys in docs_filters are passed through only if your RPC accepts them.
    """
    params: dict[str, Any] = {
        "query_embedding": embedding,
        "match_count": match_count,
    }

    # PostgREST matches functions by parameter names/types. Your RPC signature includes
    # `match_threshold`, so always send it (defaulting to 0.0 if unset).
    params["match_threshold"] = project.match_threshold if project.match_threshold is not None else 0.0

    # Optional filters: only include when present so it can match RPC defaults.
    category = docs_filters.get("category")
    tags = docs_filters.get("tags")
    if category is not None:
        params["filter_category"] = category
    if tags is not None:
        params["filter_tags"] = tags

    # Pass through any additional filter keys verbatim. If the RPC doesn't accept them,
    # PostgREST will throw an error with a helpful signature hint.
    for k, v in docs_filters.items():
        if k in ("category", "tags"):
            continue
        if k not in params:
            params[k] = v

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
