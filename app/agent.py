from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import chromadb
from workflows.errors import WorkflowRuntimeError

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore

from .data_pipeline import (
    CHROMA_COLLECTION,
)
from .llm_provider import create_embedding_model, create_llm
from .models import CandidateProfile
from .profile_store import load_profiles
from .tools import (
    build_candidate_directory_tool,
    build_candidate_rag_retriever_tool,
    build_general_knowledge_tool,
    build_profile_lookup_tool,
    fallback_rag_retrieval_answer,
    get_wikipedia_search_tool,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"

LANGUAGE_INSTRUCTION = (
    "Please respond using the same language as the user's latest request. "
    "If the request mixes languages, follow the English language."
)
OUTPUT_INSTRUCTION = (
    "Always include all matched candidate identifiers you relied on in your final answer. "
    "List them clearly using the relative link format `?candidate=<id>` so they can be clicked, "
    "text inside link should be just candidate id. "
    "Tool routing policy: ALWAYS use candidate_rag_retriever first for all resume-based "
    "questions (skills, experience, ranking, comparisons, role fit, profile evidence). "
    "Use profile_lookup only when you already have a candidate ID/name or user asks for detailed profile. "
    "Use candidate_directory only for quick metadata listing/filtering or as fallback when RAG returns no evidence. "
    "For non-resume questions, use general_knowledge or the available wikipedia tool. "
    "Provide structured information about the candidate. "
    "Use bullets or new lines to separate different fields. "
    "For other candidates provide at least their ID, name and profession. "
    "If no relevant resume evidence is found, state that clearly."
)


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


@lru_cache(maxsize=1)
def get_agent() -> ReActAgent:
    index = _load_index()
    profiles = _load_profiles()
    llm = create_llm()

    tools = [
        build_candidate_rag_retriever_tool(index, profiles),
        build_profile_lookup_tool(profiles),
        build_candidate_directory_tool(profiles),
        build_general_knowledge_tool(llm),
    ]

    wikipedia_tool = get_wikipedia_search_tool()
    if wikipedia_tool is not None:
        tools.append(wikipedia_tool)

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
        fallback_content = fallback_rag_retrieval_answer(
            _load_index(),
            user_message or message,
            _load_profiles(),
        )
        fallback_content += (
            "\n\n_Note: used fallback retrieval because the agent exceeded its iteration limit._"
        )
        return AgentOutput(
            response=ChatMessage(role=MessageRole.ASSISTANT, content=fallback_content),
            current_agent_name="fallback_retriever",
            raw={"error": str(exc), "fallback": True},
        )


__all__ = ["chat_with_agent", "get_agent"]
