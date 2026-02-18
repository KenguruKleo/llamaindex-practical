from __future__ import annotations

import os

from dotenv import load_dotenv
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", OPENAI_EMBED_MODEL
)
AZURE_OPENAI_LLM_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_LLM_DEPLOYMENT", OPENAI_LLM_MODEL
)


def _get_provider() -> str:
    provider = LLM_PROVIDER.strip().lower()
    if provider not in {"openai", "azure"}:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Expected one of: 'openai', 'azure'."
        )
    return provider


def create_embedding_model() -> BaseEmbedding:
    provider = _get_provider()
    if provider == "openai":
        return OpenAIEmbedding(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

    try:
        from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Azure provider selected, but azure embedding integration is missing. "
            "Install 'llama-index-embeddings-azure-openai'."
        ) from exc

    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure."
        )

    deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    api_key = AZURE_OPENAI_API_KEY or OPENAI_API_KEY
    return AzureOpenAIEmbedding(
        model=deployment,
        deployment_name=deployment,
        api_key=api_key,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def create_llm() -> LLM:
    provider = _get_provider()
    if provider == "openai":
        return OpenAI(model=OPENAI_LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)

    try:
        from llama_index.llms.azure_openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Azure provider selected, but azure LLM integration is missing. "
            "Install 'llama-index-llms-azure-openai'."
        ) from exc

    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure."
        )

    deployment = AZURE_OPENAI_LLM_DEPLOYMENT
    api_key = AZURE_OPENAI_API_KEY or OPENAI_API_KEY
    return AzureOpenAI(
        model=deployment,
        engine=deployment,
        api_key=api_key,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.0,
    )

