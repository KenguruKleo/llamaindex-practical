from __future__ import annotations

from typing import Optional

from ..models import CandidateProfile

MISSING_TEXT_VALUES = {"", "not provided", "none", "null", "n/a", "na", "unknown"}
NAME_FALLBACK = "Candidate name not provided"
PROFESSION_FALLBACK = "Profession not provided"


def display_name(raw_name: Optional[str]) -> str:
    value = (raw_name or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return NAME_FALLBACK
    return value


def display_profession(raw_profession: Optional[str]) -> str:
    value = (raw_profession or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return PROFESSION_FALLBACK
    return value


def format_profile(profile: CandidateProfile) -> str:
    link = f"[{profile.id}](?candidate={profile.id})"
    parts = [
        f"ID: {link}",
        f"Name: {display_name(profile.name)}",
        f"Profession: {display_profession(profile.profession)}",
    ]
    if profile.years_experience:
        parts.append(f"Experience: {profile.years_experience} years")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.summary:
        parts.append(f"Summary: {profile.summary}")
    parts.append(f"Source: {profile.source_file}")
    return "\n".join(parts)
