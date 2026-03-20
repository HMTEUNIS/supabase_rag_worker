from unittest.mock import MagicMock

import pytest

from models.request import InterpretRequest
from rag.config import load_global_settings, load_project_config, normalize_project_prefix
from rag.prompts import build_prompt
from rag.service import process_interpretation


@pytest.fixture
def env_greatrx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("GREATRX_VECTOR_RPC", "match_documents")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")


def test_normalize_prefix() -> None:
    assert normalize_project_prefix("greatrx") == "GREATRX"
    assert normalize_project_prefix("DemoApp") == "DEMOAPP"


def test_load_project_config(env_greatrx: None) -> None:
    cfg = load_project_config("greatrx")
    assert cfg.prefix == "GREATRX"
    assert cfg.vector_rpc == "match_documents"


def test_load_project_config_missing_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://x")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    monkeypatch.delenv("GREATRX_VECTOR_RPC", raising=False)
    with pytest.raises(ValueError, match="VECTOR_RPC"):
        load_project_config("greatrx")


def test_build_prompt() -> None:
    system, user = build_prompt(
        query_text="shortage text",
        similar_docs=[{"id": 9, "content": "playbook snippet", "similarity": 0.91}],
        instructions="Focus on actions.",
        metadata={"region": "west"},
        task="interpret_ticket_group",
    )
    assert "actions" in system
    assert "playbook snippet" in user
    assert "shortage text" in user
    assert "interpret_ticket_group" in user


def test_process_interpretation_mocked(env_greatrx: None, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_sb = MagicMock()
    mock_sb.rpc.return_value.execute.return_value.data = [
        {"id": 42, "content": "retrieved doc", "similarity": 0.87},
    ]
    mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock()

    monkeypatch.setattr("rag.service.generate_embedding", lambda text, st: [0.0] * 1536)
    monkeypatch.setattr(
        "rag.service.call_llm_with_fallback",
        lambda system, user, st: ("Final interpretation.", "gpt-mock"),
    )

    req = InterpretRequest(
        project_id="greatrx",
        task="interpret_ticket_group",
        data={
            "text": "Tickets about inventory",
            "metadata": {"pharmacy_ids": [1, 2]},
            "ticket_group_id": 123,
        },
        docs_filters={"category": "pharmacy_playbook", "tags": ["shortage"]},
    )
    settings = load_global_settings()
    out = process_interpretation(req, settings=settings, supabase=mock_sb)

    assert out.interpretation == "Final interpretation."
    assert out.docs_used == [42]
    assert out.confidence == pytest.approx(0.87)
    assert out.external_ref.get("ticket_group_id") == 123
    assert out.model == "gpt-mock"

    mock_sb.rpc.assert_called_once()
    call_kw = mock_sb.rpc.call_args[0][1]
    assert call_kw["p_category"] == "pharmacy_playbook"
    assert call_kw["p_tags"] == ["shortage"]
