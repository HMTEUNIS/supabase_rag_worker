import logging
from typing import Any

from supabase import Client

from models.request import InterpretData, InterpretRequest
from rag.config import ProjectConfig

logger = logging.getLogger(__name__)


def _table_name(project: ProjectConfig) -> str:
    return project.issue_group_comments_table or "issue_group_comments"


def _id_col(project: ProjectConfig) -> str:
    return project.issue_group_id_column or "issue_group_id"


def _body_col(project: ProjectConfig) -> str:
    return project.comment_body_column or "body"


def fetch_comments_text(supabase: Client, project: ProjectConfig, issue_group_id: int) -> str:
    table = _table_name(project)
    id_col = _id_col(project)
    body_col = _body_col(project)

    q = (
        supabase.table(table)
        .select(body_col)
        .eq(id_col, issue_group_id)
        .order("id", desc=False)
    )
    res = q.execute()
    rows: list[dict[str, Any]] = res.data or []
    parts: list[str] = []
    for i, row in enumerate(rows, start=1):
        body = (row.get(body_col) or "").strip()
        if body:
            parts.append(f"Comment {i}:\n{body}")
    return "\n\n".join(parts)


def build_run_interpret_request(
    *,
    project_id: str,
    issue_group_id: int,
    task: str,
    instructions: str | None,
    docs_filters: dict[str, Any],
    match_count: int,
    supabase: Client,
    project: ProjectConfig,
) -> InterpretRequest:
    text = fetch_comments_text(supabase, project, issue_group_id)
    if not text.strip():
        raise ValueError(
            f"No comments found for issue_group_id={issue_group_id} in table {_table_name(project)}"
        )

    data = InterpretData(
        text=text,
        metadata={"issue_group_id": issue_group_id},
        issue_group_id=issue_group_id,
    )
    return InterpretRequest(
        project_id=project_id,
        task=task,
        data=data,
        instructions=instructions,
        docs_filters=docs_filters,
        match_count=match_count,
    )
