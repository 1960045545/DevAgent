from __future__ import annotations

from collections.abc import Sequence

from rag_server.schemas import SearchHit


def build_context(
    hits: Sequence[SearchHit],
    *,
    include_sources: bool = True,
) -> str:
    """Format retrieved evidence for a grounded model prompt."""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        source = hit.source_uri or hit.doc_id
        header = f"[证据 {index}] {hit.title or hit.doc_id}"
        if include_sources:
            header += f" | 来源: {source}"
        blocks.append(f"{header}\n{hit.content}")
    return "\n\n".join(blocks)


def build_grounded_prompt(
    query: str,
    hits: Sequence[SearchHit],
) -> str:
    """Build a conservative prompt that asks the model to cite evidence."""
    context = build_context(hits)
    if not context:
        context = "没有检索到可用的知识库证据。"

    return (
        "请仅依据下面的知识库证据回答问题。\n"
        "如果证据不足，请明确说明“知识库中没有足够信息”，"
        "不要补充未经证实的事实。\n"
        "回答中请使用 [证据 N] 标注依据。\n\n"
        f"知识库证据：\n{context}\n\n"
        f"用户问题：\n{query}"
    )
