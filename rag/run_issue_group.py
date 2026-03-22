import logging
from typing import Any

from supabase import Client

from models.request import InterpretData, InterpretRequest
from rag.config import ProjectConfig

logger = logging.getLogger(__name__)


def _table_name(project: ProjectConfig) -> str:
    if project.issue_group_source_table:
        return project.issue_group_source_table
    return project.issue_group_comments_table or "issue_group_comments"


def _id_col(project: ProjectConfig) -> str:
    return project.issue_group_id_column or "issue_group_id"


def _order_col(project: ProjectConfig) -> str:
    return project.issue_group_order_column or "id"


def _body_col(project: ProjectConfig) -> str:
    return project.comment_body_column or "body"


def _select_columns(project: ProjectConfig) -> list[str]:
    order = _order_col(project)
    if project.issue_group_text_columns:
        cols: list[str] = []
        seen: set[str] = set()
        for c in (order, *project.issue_group_text_columns):
            if c not in seen:
                seen.add(c)
                cols.append(c)
        return cols
    body = _body_col(project)
    if order == body:
        return [order]
    return [order, body]


def _row_text_block(project: ProjectConfig, row: dict[str, Any]) -> str:
    if project.issue_group_text_columns:
        lines: list[str] = []
        for col in project.issue_group_text_columns:
            raw = row.get(col)
            if raw is None:
                continue
            s = str(raw).strip()
            if s:
                lines.append(f"{col}: {s}")
        return "\n".join(lines)
    body_col = _body_col(project)
    return (row.get(body_col) or "").strip()


def fetch_issue_group_context(supabase: Client, project: ProjectConfig, issue_group_id: int) -> str:
    table = _table_name(project)
    id_col = _id_col(project)
    order_col = _order_col(project)
    cols = _select_columns(project)

    q = (
        supabase.table(table)
        .select(",".join(cols))
        .eq(id_col, issue_group_id)
        .order(order_col, desc=False)
    )
    res = q.execute()
    rows: list[dict[str, Any]] = res.data or []
    parts: list[str] = []
    for i, row in enumerate(rows, start=1):
        block = _row_text_block(project, row)
        if block:
            parts.append(f"Record {i}:\n{block}")
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
    text = fetch_issue_group_context(supabase, project, issue_group_id)
    if not text.strip():
        raise ValueError(
            f"No rows with usable text for issue_group_id={issue_group_id} in table {_table_name(project)}"
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
