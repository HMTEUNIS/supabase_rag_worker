from unittest.mock import MagicMock

import pytest

from models.request import RunInterpretRequest
from rag.config import load_project_config, load_global_settings
from rag.run_issue_group import build_run_interpret_request


def test_build_run_interpret_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    monkeypatch.setenv("GREATRX_VECTOR_RPC", "match_documents")

    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {"body": "First comment"},
        {"body": "Second comment"},
    ]

    project = load_project_config("greatrx")
    req = build_run_interpret_request(
        project_id="greatrx",
        issue_group_id=101,
        task="interpret_issue_group",
        instructions=None,
        docs_filters={},
        match_count=5,
        supabase=mock_sb,
        project=project,
    )

    assert "First comment" in req.data.text
    assert "Second comment" in req.data.text
    assert req.data.metadata.get("issue_group_id") == 101
    assert req.data.model_dump().get("issue_group_id") == 101

    mock_sb.table.assert_called_with("issue_group_comments")
