from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import CandidateProfile

PROFILES_FILENAME = "candidates.json"


def _profiles_path(storage_dir: Path) -> Path:
    return storage_dir / PROFILES_FILENAME


def save_profiles(storage_dir: Path, profiles: List[CandidateProfile]) -> None:
    payload = [profile.to_json() for profile in profiles]
    output_path = _profiles_path(storage_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_profiles(storage_dir: Path) -> List[CandidateProfile]:
    data_path = _profiles_path(storage_dir)
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    profiles: List[CandidateProfile] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        profiles.append(CandidateProfile.from_json(item))
    return profiles


def index_exists(storage_dir: Path) -> bool:
    profiles_path = _profiles_path(storage_dir)
    chroma_dir = storage_dir / "chroma"

    if not profiles_path.exists():
        return False
    if not chroma_dir.exists():
        return False

    try:
        has_chroma_data = any(chroma_dir.iterdir())
    except OSError:
        return False

    if has_chroma_data:
        return True

    # Empty profile list is a valid indexed state when there are no resumes yet.
    return len(load_profiles(storage_dir)) == 0

