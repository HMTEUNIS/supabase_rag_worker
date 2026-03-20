import logging
import os
import secrets
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from models.request import InterpretRequest, InterpretResponse
from rag.config import load_global_settings, load_project_config
from rag.service import process_interpretation

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