from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class CandidateProfile:
    id: str
    name: str
    profession: str
    years_experience: Optional[str]
    summary: str
    skills: List[str]
    source_file: str

    def to_json(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "CandidateProfile":
        raw_skills = payload.get("skills", [])
        skills = raw_skills if isinstance(raw_skills, list) else []
        years_experience = payload.get("years_experience")
        years_text = (
            str(years_experience)
            if isinstance(years_experience, (str, int, float))
            else None
        )
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            profession=str(payload.get("profession", "Unknown")),
            years_experience=years_text,
            summary=str(payload.get("summary", "")),
            skills=[str(skill) for skill in skills],
            source_file=str(payload.get("source_file", "")),
        )
