from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set

from llama_index.core.tools import FunctionTool

from ..models import CandidateProfile
from .common import display_name, display_profession

def _tokenize(text: str, *, stop_words: Set[str]) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and len(token) > 2 and token not in stop_words
    ]


def build_candidate_directory_tool(profiles: Iterable[CandidateProfile]) -> FunctionTool:
    profile_list = list(profiles)
    doc_count = max(1, len(profile_list))
    token_doc_frequency: Counter[str] = Counter()

    for profile in profile_list:
        searchable = " ".join(
            [
                profile.id,
                display_name(profile.name),
                display_profession(profile.profession),
                profile.summary,
                ", ".join(profile.skills),
            ]
        ).lower()
        token_doc_frequency.update(set(_tokenize(searchable, stop_words=set())))

    # Dynamic corpus-derived stop words to avoid hardcoded language lists.
    # Tokens appearing in most profiles add noise for matching quality.
    dynamic_stop_words: Set[str] = {
        token
        for token, freq in token_doc_frequency.items()
        if (freq / doc_count) >= 0.6
    }

    def _search_candidates(keyword: str = "", limit: int = 10) -> str:
        query = keyword.strip().lower()
        try:
            max_rows = max(1, min(int(limit), 30))
        except (TypeError, ValueError):
            max_rows = 10

        query_tokens = _tokenize(query.replace("/", " "), stop_words=dynamic_stop_words)
        min_overlap = 1 if len(query_tokens) <= 2 else 2
        ranked_matches: List[tuple[int, CandidateProfile]] = []
        for profile in profile_list:
            searchable = " ".join(
                [
                    profile.id,
                    display_name(profile.name),
                    display_profession(profile.profession),
                    profile.summary,
                    ", ".join(profile.skills),
                ]
            ).lower()
            if not query:
                ranked_matches.append((0, profile))
                continue

            phrase_match = query in searchable
            profile_tokens = set(_tokenize(searchable, stop_words=dynamic_stop_words))
            overlap = sum(1 for token in query_tokens if token in profile_tokens)
            if phrase_match or overlap >= min_overlap:
                score = overlap + (100 if phrase_match else 0)
                ranked_matches.append((score, profile))

        ranked_matches.sort(key=lambda item: item[0], reverse=True)
        matches = [profile for _, profile in ranked_matches]

        if not matches:
            if query:
                return f"No candidates matched keyword '{keyword}'."
            return "No candidates are currently available."

        lines = ["Candidates:"]
        for profile in matches[:max_rows]:
            lines.append(
                f"- [{profile.id}](?candidate={profile.id}) | "
                f"{display_name(profile.name)} | {display_profession(profile.profession)}"
            )
        return "\n".join(lines)

    return FunctionTool.from_defaults(
        fn=_search_candidates,
        name="candidate_directory",
        description=(
            "List candidates from the directory with keyword filtering by profession, "
            "skills, summary, candidate ID, or name. Supports phrase and token-based matching. "
            "Best for lightweight directory listing/filtering, not deep resume-semantic retrieval."
        ),
    )
