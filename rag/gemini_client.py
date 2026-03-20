import logging
from typing import Any, List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around google-genai for embeddings + chat."""

    def __init__(
        self,
        api_key: str,
        *,
        embedding_model: str,
        embedding_dimension: int | None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

    def generate_embedding(self, text: str) -> List[float]:
        """
        Returns a vector suitable for pgvector.

        If `embedding_dimension` is set, we ask Gemini to output that dimensionality
        (Gemini supports truncation via `output_dimensionality`) so it matches pgvector.
        """
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.embedding_dimension,
        )

        result: Any = self._client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=config,
        )

        embeddings = getattr(result, "embeddings", None)
        if not embeddings:
            raise ValueError("Gemini embedding response did not include embeddings")

        embedding_obj = embeddings[0]
        values = getattr(embedding_obj, "values", None)
        if values is None:
            raise ValueError("Gemini embedding response did not include embedding values")

        vec = list(values)
        if self.embedding_dimension is not None and len(vec) != self.embedding_dimension:
            # Defensive guard in case the provider ignores output_dimensionality.
            vec = vec[: self.embedding_dimension]
        return vec

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float,
    ) -> str:
        config = types.GenerateContentConfig(temperature=temperature)
        response: Any = self._client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=config,
        )
        return (getattr(response, "text", None) or "").strip()

