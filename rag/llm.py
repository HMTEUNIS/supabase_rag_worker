import logging
from typing import Protocol

from openai import OpenAI

from rag.config import GlobalSettings
from rag.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> tuple[str, str]: ...


class _OpenAICompat:
    def __init__(self, *, api_key: str, base_url: str | None, model: str, temperature: float) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> tuple[str, str]:
        r = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (r.choices[0].message.content or "").strip()
        return text, self._model


def call_llm_with_fallback(system: str, user: str, settings: GlobalSettings) -> tuple[str, str]:
    """
    Returns (assistant_text, model_used).
    Order: LLM_PROVIDER primary, then OpenAI if configured and different.
    """
    errors: list[str] = []

    def try_gemini() -> tuple[str, str] | None:
        if not settings.gemini_api_key:
            return None
        try:
            client = GeminiClient(
                settings.gemini_api_key,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
            )
            text = client.generate_chat(
                system,
                user,
                model=settings.gemini_llm_model,
                temperature=settings.gemini_llm_temperature,
            )
            return text, f"gemini:{settings.gemini_llm_model}"
        except Exception as e:  # noqa: BLE001
            errors.append(f"gemini: {e}")
            logger.warning("Gemini call failed: %s", e)
            return None

    def try_deepseek() -> tuple[str, str] | None:
        if not settings.deepseek_api_key:
            return None
        try:
            base = settings.deepseek_base_url or "https://api.deepseek.com"
            ds_model = settings.llm_model if settings.llm_provider == "deepseek" else settings.deepseek_llm_model
            client = _OpenAICompat(
                api_key=settings.deepseek_api_key,
                base_url=f"{base}/v1",
                model=ds_model,
                temperature=settings.llm_temperature,
            )
            return client.complete(system, user)
        except Exception as e:  # noqa: BLE001
            errors.append(f"deepseek: {e}")
            logger.warning("DeepSeek call failed: %s", e)
            return None

    def try_openai() -> tuple[str, str] | None:
        if not settings.openai_api_key:
            return None
        try:
            model = settings.llm_model if settings.llm_provider == "openai" else settings.openai_llm_model
            client = _OpenAICompat(
                api_key=settings.openai_api_key,
                base_url=None,
                model=model,
                temperature=settings.llm_temperature,
            )
            return client.complete(system, user)
        except Exception as e:  # noqa: BLE001
            errors.append(f"openai: {e}")
            logger.warning("OpenAI call failed: %s", e)
            return None

    primary = settings.llm_provider
    if primary == "gemini":
        out = try_gemini()
        if out:
            return out
        out = try_openai()
        if out:
            return out
        out = try_deepseek()
        if out:
            return out
    elif primary == "deepseek":
        out = try_deepseek()
        if out:
            return out
        out = try_openai()
        if out:
            return out
    elif primary == "openai":
        out = try_openai()
        if out:
            return out
        out = try_deepseek()
        if out:
            return out
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r}")

    raise RuntimeError("All LLM providers failed: " + "; ".join(errors) if errors else "no providers configured")
