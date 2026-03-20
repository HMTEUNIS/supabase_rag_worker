import logging
import time
from typing import Any

from supabase import Client, create_client

from models.request import InterpretRequest, InterpretResponse
from rag.config import GlobalSettings, ProjectConfig, load_global_settings, load_project_config
from rag.embeddings import generate_embedding
from rag.llm import call_llm_with_fallback
from rag.prompts import build_prompt
from rag.retrieve import query_similar_docs

logger = logging.getLogger(__name__)


def _docs_used_ids(rows: list[dict[str, Any]]) -> list[Any]:
    out: list[Any] = []
    for r in rows:
        if "id" in r and r["id"] is not None:
            out.append(r["id"])
    return out


def _confidence_from_rows(rows: list[dict[str, Any]]) -> float | None:
    sims = [r["similarity"] for r in rows if r.get("similarity") is not None]
    if not sims:
        return None
    try:
        return float(max(sims))
    except (TypeError, ValueError):
        return None


def _log_llm_io(*, settings: GlobalSettings, system: str, user: str, assistant: str, model: str) -> None:
    if settings.log_pii:
        logger.info(
            "LLM model=%s system=%r user=%r assistant=%r",
            model,
            system[:5000],
            user[:5000],
            assistant[:5000],
        )
    else:
        logger.info(
            "LLM model=%s lens system=%s user=%s assistant=%s",
            model,
            len(system),
            len(user),
            len(assistant),
        )


def _insert_failed_run(
    supabase: Client,
    project: ProjectConfig,
    payload: dict[str, Any],
    error: str,
    stage: str,
) -> None:
    if not project.failed_runs_table:
        return
    try:
        supabase.table(project.failed_runs_table).insert(
            {
                "task": payload.get("task"),
                "request_payload": payload,
                "error": error[:8000],
                "stage": stage,
            }
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record failure row: %s", e)


def process_interpretation(
    req: InterpretRequest,
    *,
    settings: GlobalSettings | None = None,
    supabase: Client | None = None,
) -> InterpretResponse:
    settings = settings or load_global_settings()
    project = load_project_config(req.project_id)

    if supabase is None:
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)

    started = time.perf_counter()
    persist_payload: dict[str, Any] = {
        "task": req.task,
        "project_id": req.project_id,
        "data": {"text": req.data.text, "metadata": req.data.metadata},
        "docs_filters": req.docs_filters,
    }

    try:
        emb = generate_embedding(req.data.text, settings)
        rows = query_similar_docs(
            supabase,
            project,
            emb,
            req.match_count,
            req.docs_filters,
        )
        instructions = (req.instructions or project.default_instructions or "").strip()
        system, user = build_prompt(
            query_text=req.data.text,
            similar_docs=rows,
            instructions=instructions,
            metadata=req.data.metadata,
            task=req.task,
        )
        interpretation, model_used = call_llm_with_fallback(system, user, settings)
        _log_llm_io(settings=settings, system=system, user=user, assistant=interpretation, model=model_used)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        docs_used = _docs_used_ids(rows)
        conf = _confidence_from_rows(rows)

        external_blob = req.data.model_dump()
        external_blob.pop("text", None)
        external_ref: dict[str, Any] = external_blob
        row = {
            "task": req.task,
            "external_ref": external_ref,
            "interpretation": interpretation,
            "metadata": req.data.metadata,
            "docs_used": docs_used,
            "confidence": conf,
            "model": model_used,
            "processing_time_ms": elapsed_ms,
        }
        if project.interpretations_table:
            try:
                supabase.table(project.interpretations_table).insert(row).execute()
            except Exception as e:  # noqa: BLE001
                logger.warning("Interpretation persistence failed: %s", e)
                _insert_failed_run(
                    supabase,
                    project,
                    {**persist_payload, "row": row},
                    str(e),
                    "persist",
                )

        return InterpretResponse(
            interpretation=interpretation,
            confidence=conf,
            docs_used=docs_used,
            processing_time_ms=elapsed_ms,
            external_ref=external_ref,
            model=model_used,
        )

    except Exception as e:  # noqa: BLE001
        try:
            _insert_failed_run(
                supabase,
                project,
                persist_payload,
                str(e),
                "pipeline",
            )
        except Exception as log_exc:  # noqa: BLE001
            logger.warning("Could not record pipeline failure: %s", log_exc)
        raise
