from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InterpretData(BaseModel):
    """Domain-agnostic payload; stable ids (e.g. ticket_group_id) may sit beside `metadata`."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="Primary text to interpret (embedded + sent to LLM).")
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterpretRequest(BaseModel):
    project_id: str = Field(..., description="Logical tenant id; normalized to env prefix (e.g. greatrx → GREATRX).")
    task: str = Field(..., description="Task name for logging, persistence, and future routing.")
    data: InterpretData
    instructions: str | None = Field(
        default=None,
        description="Optional system-style override; falls back to {PREFIX}_DEFAULT_INSTRUCTIONS.",
    )
    docs_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Forwarded to the vector RPC (e.g. category, tags). Shape must match your SQL function.",
    )
    match_count: int = Field(default=5, ge=1, le=50, description="Max documents to retrieve.")


class InterpretResponse(BaseModel):
    interpretation: str
    confidence: float | None = None
    docs_used: list[Any] = Field(default_factory=list)
    processing_time_ms: int
    external_ref: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


class BackfillEmbeddingsRequest(BaseModel):
    """
    Admin operation: populate pgvector embeddings for rows where embedding IS NULL.

    This is domain-agnostic: table + column names can come from tenant env config or request overrides.
    """

    project_id: str
    limit: int = Field(default=100, ge=1, le=10000, description="Max rows to backfill.")
    batch_size: int = Field(default=50, ge=1, le=500, description="Rows per Supabase batch.")
    sleep_seconds: float = Field(default=0.0, ge=0.0, le=10.0, description="Optional throttle between batches.")
    dry_run: bool = Field(default=False, description="If true, computes embeddings but does not persist them.")

    # Optional overrides for non-standard schemas.
    table: str | None = None
    content_column: str | None = None
    embedding_column: str | None = None


class BackfillEmbeddingsResponse(BaseModel):
    updated: int
    candidates_checked: int
    table: str
    content_column: str
    embedding_column: str
    dry_run: bool
    processing_time_ms: int
