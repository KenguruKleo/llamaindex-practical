"""Candidate profiling application powered by LlamaIndex."""

from pathlib import Path
from typing import List

from .data_pipeline import (
    CHROMA_COLLECTION,
    CandidateIndexer,
)
from .llm_provider import create_embedding_model, create_llm
from .models import CandidateProfile
from .profile_store import index_exists, load_profiles

__all__ = [
    "CandidateIndexer",
    "CandidateProfile",
    "CHROMA_COLLECTION",
    "create_embedding_model",
    "create_llm",
    "index_exists",
    "load_profiles",
]


def prepare_candidates(data_dir: Path, storage_dir: Path) -> List[CandidateProfile]:
    indexer = CandidateIndexer(data_dir=data_dir, storage_dir=storage_dir)
    return indexer.run()
