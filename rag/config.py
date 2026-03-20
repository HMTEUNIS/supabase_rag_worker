import os
import re
from dataclasses import dataclass

_PREFIX_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def normalize_project_prefix(project_id: str) -> str:
    s = project_id.strip()
    if not s or not _PREFIX_RE.match(s):
        raise ValueError("project_id must be a non-empty alphanumeric/underscore identifier")
    return s.upper()


@dataclass(frozen=True)
class ProjectConfig:
    prefix: str
    vector_rpc: str
    match_threshold: float | None
    interpretations_table: str | None
    failed_runs_table: str | None
    default_instructions: str
    docs_table: str | None
    docs_content_column: str | None
    docs_embedding_column: str | None
    worker_api_key: str | None


def _prefixed(prefix: str, key: str) -> str:
    return os.getenv(f"{prefix}_{key}", "").strip()


def load_project_config(project_id: str) -> ProjectConfig:
    prefix = normalize_project_prefix(project_id)
    vector_rpc = _prefixed(prefix, "VECTOR_RPC")
    if not vector_rpc:
        raise ValueError(
            f"Missing {prefix}_VECTOR_RPC: Postgres RPC name for similarity search "
            "(see sql/example_match_documents.sql)."
        )

    interp = _prefixed(prefix, "INTERPRETATIONS_TABLE") or None
    failed = _prefixed(prefix, "FAILED_RUNS_TABLE") or None
    default_instructions = _prefixed(prefix, "DEFAULT_INSTRUCTIONS")
    match_threshold_raw = _prefixed(prefix, "MATCH_THRESHOLD")
    match_threshold: float | None
    if match_threshold_raw:
        match_threshold = float(match_threshold_raw)
    else:
        match_threshold = None

    docs_table = _prefixed(prefix, "DOCS_TABLE") or None
    docs_content = _prefixed(prefix, "DOCS_CONTENT_COLUMN") or None
    docs_embedding = _prefixed(prefix, "DOCS_EMBEDDING_COLUMN") or None

    worker_key = _prefixed(prefix, "WORKER_API_KEY") or None

    return ProjectConfig(
        prefix=prefix,
        vector_rpc=vector_rpc,
        match_threshold=match_threshold,
        interpretations_table=interp,
        failed_runs_table=failed,
        default_instructions=default_instructions,
        docs_table=docs_table,
        docs_content_column=docs_content,
        docs_embedding_column=docs_embedding,
        worker_api_key=worker_key,
    )


@dataclass(frozen=True)
class GlobalSettings:
    supabase_url: str
    supabase_service_key: str
    worker_api_key: str | None
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None
    llm_provider: str
    llm_model: str
    llm_temperature: float
    gemini_api_key: str | None
    gemini_llm_model: str
    gemini_llm_temperature: float
    openai_api_key: str | None
    deepseek_api_key: str | None
    deepseek_base_url: str
    openai_llm_model: str
    deepseek_llm_model: str
    log_pii: bool


def load_global_settings() -> GlobalSettings:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

    dim_raw = os.getenv("EMBEDDING_DIMENSION", "").strip()
    embedding_dimension: int | None
    if dim_raw:
        embedding_dimension = int(dim_raw)
    else:
        embedding_dimension = None

    temp_raw = os.getenv("LLM_TEMPERATURE", "0.2").strip()
    try:
        llm_temperature = float(temp_raw)
    except ValueError:
        llm_temperature = 0.2

    return GlobalSettings(
        supabase_url=url,
        supabase_service_key=key,
        worker_api_key=os.getenv("WORKER_API_KEY", "").strip() or None,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        embedding_dimension=embedding_dimension,
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview").strip(),
        llm_temperature=llm_temperature,
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        gemini_llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite-preview").strip(),
        gemini_llm_temperature=float(
            os.getenv("GEMINI_LLM_TEMPERATURE", "").strip()
            or os.getenv("LLM_TEMPERATURE", "0.2").strip()
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip() or None,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/"),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini").strip(),
        deepseek_llm_model=os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat").strip(),
        log_pii=os.getenv("LOG_PII", "").strip().lower() in ("1", "true", "yes"),
    )
