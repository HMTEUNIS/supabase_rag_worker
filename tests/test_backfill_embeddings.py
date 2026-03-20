from unittest.mock import MagicMock

import pytest

from models.request import BackfillEmbeddingsRequest
from rag.config import load_global_settings
from rag.ops_backfill_embeddings import backfill_embeddings


def test_backfill_embeddings_updates_null_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.is_.return_value.limit.return_value.execute.return_value.data = [
        {"id": 1, "content": "hello"},
        {"id": 2, "content": "world"},
    ]
    mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

    monkeypatch.setattr("rag.ops_backfill_embeddings.generate_embedding", lambda text, st: [0.0] * 1536)

    # minimal settings require supabase keys; values are irrelevant since we pass supabase mock
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("GREATRX_VECTOR_RPC", "match_documents")

    req = BackfillEmbeddingsRequest(project_id="greatrx", limit=2, batch_size=50)
    settings = load_global_settings()

    # Need docs mapping for table/columns; rely on defaults in code ("knowledge_base","content","embedding").
    out = backfill_embeddings(req, settings=settings, supabase=mock_sb)

    assert out.updated == 2
    assert out.candidates_checked == 2
    assert out.embedding_column == "embedding"

