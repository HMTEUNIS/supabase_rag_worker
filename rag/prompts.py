import json
from typing import Any


def build_prompt(
    *,
    query_text: str,
    similar_docs: list[dict[str, Any]],
    instructions: str,
    metadata: dict[str, Any],
    task: str,
) -> tuple[str, str]:
    """
    Returns (system_message, user_message) for chat completion.
    Keeps retrieval context in the user message so providers behave consistently.
    """
    system = (
        "You are a retrieval-augmented assistant. Use ONLY the provided context snippets "
        "when they are relevant; if context is insufficient, say so clearly. "
        "Follow the caller's instructions precisely."
    )
    if instructions.strip():
        system = f"{system}\n\nCaller instructions:\n{instructions.strip()}"

    blocks: list[str] = []
    for i, row in enumerate(similar_docs, start=1):
        cid = row.get("id")
        content = row.get("content") or row.get("body") or row.get("text") or ""
        sim = row.get("similarity")
        head = f"--- Context {i}"
        if cid is not None:
            head += f" (id={cid})"
        if sim is not None:
            head += f" (similarity={sim})"
        head += " ---"
        blocks.append(f"{head}\n{content}")

    context = "\n\n".join(blocks) if blocks else "(No matching documents were retrieved.)"

    user_payload = {
        "task": task,
        "query_text": query_text,
        "metadata": metadata,
        "context": context,
    }
    user = (
        "Use the JSON below. Answer in plain language for the end user.\n\n"
        f"```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"
    )
    return system, user
