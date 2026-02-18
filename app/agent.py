from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Set

import chromadb
from workflows.errors import WorkflowRuntimeError

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore

from .data_pipeline import (
    CHROMA_COLLECTION,
    CandidateProfile,
    create_embedding_model,
    create_llm,
    load_profiles,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
MISSING_TEXT_VALUES = {"", "not provided", "none", "null", "n/a", "na", "unknown"}
NAME_FALLBACK = "Candidate name not provided"
PROFESSION_FALLBACK = "Profession not provided"

LANGUAGE_INSTRUCTION = (
    "Please respond using the same language as the user's latest request. "
    "If the request mixes languages, follow the English language."
)
OUTPUT_INSTRUCTION = (
    "Always include all matched candidate identifiers you relied on in your final answer. "
    "List them clearly using the relative link format `?candidate=<id>` so they can be clicked, "
    "text inside link should be just candidate id. "
    "If one candidate is clearly the most relevant, you may call profile_lookup for additional detail. "
    "Provide structured information about the candidate."
    "Use bullets or new lines to separate different fields."
    "For other candidates provide at least their ID, name and profession."
)


def _display_name(raw_name: Optional[str]) -> str:
    value = (raw_name or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return NAME_FALLBACK
    return value


def _display_profession(raw_profession: Optional[str]) -> str:
    value = (raw_profession or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return PROFESSION_FALLBACK
    return value


@lru_cache(maxsize=1)
def _load_profiles() -> List[CandidateProfile]:
    return load_profiles(STORAGE_DIR)


@lru_cache(maxsize=1)
def _load_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=create_embedding_model(),
    )


def _render_retrieval_response(response, profiles: dict[str, CandidateProfile]) -> str:
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
        profile = profiles.get(candidate_id)
        if profile:
            id_lines.append(
                f"- [{candidate_id}](?candidate={candidate_id}): "
                f"{_display_name(profile.name)} / {_display_profession(profile.profession)}"
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


def _fallback_retrieval_answer(question: str) -> str:
    index = _load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    response = query_engine.query(question)
    profiles = {profile.id: profile for profile in _load_profiles()}
    return _render_retrieval_response(response, profiles)


def _build_retrieval_tool(index: VectorStoreIndex) -> FunctionTool:
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )

    profiles = {profile.id: profile for profile in _load_profiles()}

    def _retrieve_with_ids(question: str) -> str:
        response = query_engine.query(question)
        return _render_retrieval_response(response, profiles)

    return FunctionTool.from_defaults(
        fn=_retrieve_with_ids,
        name="candidate_retriever",
        description=(
            "Use this tool to answer questions about candidates, their experience, and "
            "professional highlights extracted from resumes. It always returns the matched "
            "candidate identifiers—make sure you surface them in the final answer."
        ),
    )


def _format_profile(profile: CandidateProfile) -> str:
    link = f"[{profile.id}](?candidate={profile.id})"
    parts = [
        f"ID: {link}",
        f"Name: {_display_name(profile.name)}",
        f"Profession: {_display_profession(profile.profession)}",
    ]
    if profile.years_experience:
        parts.append(f"Experience: {profile.years_experience} years")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.summary:
        parts.append(f"Summary: {profile.summary}")
    parts.append(f"Source: {profile.source_file}")
    return "\n".join(parts)


def _profile_lookup_tool(profiles: Iterable[CandidateProfile]) -> FunctionTool:
    name_to_profile = {profile.name.lower(): profile for profile in profiles}
    id_to_profile = {profile.id: profile for profile in profiles}

    def _lookup(identifier: str) -> str:
        if not identifier:
            return "Please specify a candidate name or identifier."
        key = identifier.strip().lower()
        profile = name_to_profile.get(key) or id_to_profile.get(key)
        if not profile:
            return (
                "Candidate not found. Try using the candidate's exact name or their ID "
                "from the directory."
            )
        return _format_profile(profile)

    return FunctionTool.from_defaults(
        fn=_lookup,
        name="profile_lookup",
        description=(
            "Look up a candidate by name or identifier to retrieve their profile "
            "information such as profession, skills, and summary."
            "Provide structured information about the candidate."
            "Use bullets or new lines to separate different fields."
        ),
    )


@lru_cache(maxsize=1)
def get_agent() -> ReActAgent:
    index = _load_index()
    profiles = _load_profiles()
    tools = [
        _build_retrieval_tool(index),
        _profile_lookup_tool(profiles),
        # _skill_search_tool(profiles),
    ]
    llm = create_llm()
    return ReActAgent(tools=tools, llm=llm, verbose=False, streaming=False)


def _run_agent_sync(
    agent: ReActAgent,
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
) -> AgentOutput:
    async def _arun() -> AgentOutput:
        handler = agent.run(user_msg=message, chat_history=chat_history)
        return await handler

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_arun())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    else:  # pragma: no cover - only triggered if a loop already runs
        future = asyncio.run_coroutine_threadsafe(_arun(), loop)
        return future.result()


def chat_with_agent(
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
) -> AgentOutput:
    agent = get_agent()
    user_message = message.strip()
    instructions = f"{LANGUAGE_INSTRUCTION}\n{OUTPUT_INSTRUCTION}"
    if user_message:
        prompt = f"{instructions}\n\nUser request:\n{user_message}"
    else:
        prompt = instructions
    try:
        return _run_agent_sync(agent, prompt, chat_history)
    except WorkflowRuntimeError as exc:
        if "Max iterations" not in str(exc):
            raise
        fallback_content = _fallback_retrieval_answer(user_message or message)
        fallback_content += (
            "\n\n_Note: used fallback retrieval because the agent exceeded its iteration limit._"
        )
        return AgentOutput(
            response=ChatMessage(role=MessageRole.ASSISTANT, content=fallback_content),
            current_agent_name="fallback_retriever",
            raw={"error": str(exc), "fallback": True},
        )


__all__ = ["chat_with_agent", "get_agent"]
