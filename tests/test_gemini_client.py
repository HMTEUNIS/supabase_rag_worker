from unittest.mock import MagicMock

import pytest


def test_embedding_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Import inside the test so we can safely monkeypatch before any network calls.
    import rag.gemini_client as gemini_client

    class FakeEmbedObj:
        values = list(range(3072))

    class FakeResult:
        embeddings = [FakeEmbedObj()]

    class FakeModels:
        def embed_content(self, *, model: str, contents: str, config: object) -> FakeResult:
            # The worker requests retrieval_document embeddings + truncation.
            assert getattr(config, "task_type", None) == "RETRIEVAL_DOCUMENT"
            assert getattr(config, "output_dimensionality", None) == 1536
            return FakeResult()

    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(gemini_client.genai, "Client", lambda api_key: FakeClient())

    client = gemini_client.GeminiClient(
        "fake-key",
        embedding_model="gemini-embedding-001",
        embedding_dimension=1536,
    )
    vec = client.generate_embedding("hello world")

    assert len(vec) == 1536
    assert vec[0] == 0
    assert vec[-1] == 1535

