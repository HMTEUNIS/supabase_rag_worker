import logging

from openai import OpenAI

from rag.config import GlobalSettings

logger = logging.getLogger(__name__)


def generate_embedding(text: str, settings: GlobalSettings) -> list[float]:
    if settings.embedding_provider != "openai":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
            "Use 'openai' or extend rag/embeddings.py."
        )
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=text)
    vec = list(resp.data[0].embedding)

    if settings.embedding_dimension is not None and len(vec) != settings.embedding_dimension:
        raise ValueError(
            f"Embedding length {len(vec)} does not match EMBEDDING_DIMENSION={settings.embedding_dimension}"
        )

    if not settings.log_pii:
        logger.info("Embedding generated dim=%s model=%s", len(vec), settings.embedding_model)
    else:
        logger.debug("Embedding for text=%r dim=%s", text[:200], len(vec))

    return vec
