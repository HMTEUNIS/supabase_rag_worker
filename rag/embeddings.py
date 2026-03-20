import logging

from openai import OpenAI

from rag.config import GlobalSettings
from rag.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


def generate_embedding(text: str, settings: GlobalSettings) -> list[float]:
    if settings.embedding_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        if settings.embedding_dimension is None:
            raise ValueError("EMBEDDING_DIMENSION must be set when using Gemini embeddings")

        client = GeminiClient(
            settings.gemini_api_key,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
        vec = client.generate_embedding(text)
    elif settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(model=settings.embedding_model, input=text)
        vec = list(resp.data[0].embedding)
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}")

    if settings.embedding_dimension is not None and len(vec) != settings.embedding_dimension:
        raise ValueError(
            f"Embedding length {len(vec)} does not match EMBEDDING_DIMENSION={settings.embedding_dimension}"
        )

    if not settings.log_pii:
        logger.info("Embedding generated dim=%s model=%s", len(vec), settings.embedding_model)
    else:
        logger.debug("Embedding for text=%r dim=%s", text[:200], len(vec))

    return vec
