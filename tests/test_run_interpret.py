from unittest.mock import MagicMock

import pytest

from rag.config import load_project_config
from rag.run_issue_group import build_run_interpret_request


def test_build_run_interpret_request_default_comments_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    monkeypatch.setenv("GREATRX_VECTOR_RPC", "match_documents")

    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {"id": 1, "body": "First comment"},
        {"id": 2, "body": "Second comment"},
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
    mock_sb.table.return_value.select.assert_called_with("id,body")


def test_build_run_interpret_request_tickets_multicolumn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    monkeypatch.setenv("GREATRX_VECTOR_RPC", "match_documents")
    monkeypatch.setenv("GREATRX_ISSUE_GROUP_SOURCE_TABLE", "tickets")
    monkeypatch.setenv("GREATRX_ISSUE_GROUP_TEXT_COLUMNS", "title,description")

    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {"id": 10, "title": "Refill stuck", "description": "Patient reports …"},
        {"id": 11, "title": "Duplicate charge", "description": "Billed twice"},
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

    assert "Refill stuck" in req.data.text
    assert "title:" in req.data.text.lower() or "Title:" in req.data.text
    mock_sb.table.assert_called_with("tickets")
    mock_sb.table.return_value.select.assert_called_with("id,title,description")
