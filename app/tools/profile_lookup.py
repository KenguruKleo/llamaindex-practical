from __future__ import annotations

from typing import Iterable

from llama_index.core.tools import FunctionTool

from ..models import CandidateProfile
from .common import format_profile


def build_profile_lookup_tool(profiles: Iterable[CandidateProfile]) -> FunctionTool:
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
        return format_profile(profile)

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
