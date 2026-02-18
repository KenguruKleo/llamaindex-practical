from __future__ import annotations

from typing import Iterable, List, Set

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool

from ..models import CandidateProfile
from .common import display_name, display_profession


def _render_retrieval_response(response, profiles_by_id: dict[str, CandidateProfile]) -> str:
    answer_text = str(getattr(response, "response", response))
    source_nodes = getattr(response, "source_nodes", []) or []

    seen_ids: Set[str] = set()
    id_lines: List[str] = []
    for source in source_nodes:
        node = getattr(source, "node", None)
        metadata = getattr(node, "metadata", {}) if node else {}
        candidate_id = metadata.get("candidate_id")
        if not candidate_id or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        profile = profiles_by_id.get(candidate_id)
        if profile:
            id_lines.append(
                f"- [{candidate_id}](?candidate={candidate_id}): "
                f"{display_name(profile.name)} / {display_profession(profile.profession)}"
            )
        else:
            id_lines.append(
                f"- [{candidate_id}](?candidate={candidate_id}): profile details unavailable"
            )

    if id_lines:
        answer_text += "\n\nMatched candidate identifiers:\n" + "\n".join(id_lines)
    else:
        answer_text += "\n\nNo candidate identifiers found for this query."
    return answer_text


def build_candidate_rag_retriever_tool(
    index: VectorStoreIndex,
    profiles: Iterable[CandidateProfile],
) -> FunctionTool:
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    profiles_by_id = {profile.id: profile for profile in profiles}

    def _retrieve_with_ids(question: str) -> str:
        response = query_engine.query(question)
        return _render_retrieval_response(response, profiles_by_id)

    return FunctionTool.from_defaults(
        fn=_retrieve_with_ids,
        name="candidate_rag_retriever",
        description=(
            "Primary RAG (Retrieval-Augmented Generation) tool for resume-grounded answers. "
            "Use it for candidate search, skill matching, experience comparisons, and any "
            "question that must be grounded in CV data. It returns concise evidence-backed "
            "content and matched candidate identifiers."
        ),
    )


def fallback_rag_retrieval_answer(
    index: VectorStoreIndex,
    question: str,
    profiles: Iterable[CandidateProfile],
) -> str:
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    response = query_engine.query(question)
    profiles_by_id = {profile.id: profile for profile in profiles}
    return _render_retrieval_response(response, profiles_by_id)
