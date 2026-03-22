import logging
import os
import secrets
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from supabase import create_client

from models.request import (
    InterpretRequest,
    InterpretResponse,
    RunInterpretRequest,
    BackfillEmbeddingsRequest,
    BackfillEmbeddingsResponse,
)
from rag.config import load_global_settings, load_project_config
from rag.run_issue_group import build_run_interpret_request
from rag.service import process_interpretation
from rag.ops_backfill_embeddings import backfill_embeddings

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supabase RAG Worker",
    description="Config-driven RAG over Supabase pgvector + pluggable LLM providers.",
    version="0.1.0",
)


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def require_auth(project_id: str, authorization: str | None) -> None:
    settings = load_global_settings()
    project = load_project_config(project_id)
    token = _bearer_token(authorization)

    if settings.worker_api_key:
        if not token or not secrets.compare_digest(token, settings.worker_api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return

    if project.worker_api_key:
        if not token or not secrets.compare_digest(token, project.worker_api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/rag/interpret", response_model=InterpretResponse)
async def interpret(
    body: InterpretRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> InterpretResponse:
    require_auth(body.project_id, authorization)
    try:
        # `process_interpretation` is synchronous (Supabase + OpenAI clients),
        # so run it in a worker thread to avoid blocking the event loop.
        return await run_in_threadpool(process_interpretation, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("interpret failed")
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.post("/api/rag/run-interpret", response_model=InterpretResponse)
async def run_interpret(
    body: RunInterpretRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> InterpretResponse:
    require_auth(body.project_id, authorization)

    def _job() -> InterpretResponse:
        settings = load_global_settings()
        project = load_project_config(body.project_id)
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        inner = build_run_interpret_request(
            project_id=body.project_id,
            issue_group_id=body.issue_group_id,
            task=body.task,
            instructions=body.instructions,
            docs_filters=body.docs_filters,
            match_count=body.match_count,
            supabase=supabase,
            project=project,
        )
        return process_interpretation(inner, settings=settings, supabase=supabase)

    try:
        return await run_in_threadpool(_job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("run_interpret failed")
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.post("/api/ops/backfill-embeddings", response_model=BackfillEmbeddingsResponse)
async def ops_backfill_embeddings(
    body: BackfillEmbeddingsRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> BackfillEmbeddingsResponse:
    require_auth(body.project_id, authorization)
    try:
        return await run_in_threadpool(backfill_embeddings, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("backfill failed")
        raise HTTPException(status_code=500, detail="Internal error") from e