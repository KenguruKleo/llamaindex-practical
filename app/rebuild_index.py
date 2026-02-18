from __future__ import annotations

from pathlib import Path

from . import prepare_candidates
from .paths import DATA_DIR, STORAGE_DIR
from .profile_store import clear_index_artifacts


def rebuild_index(data_dir: Path, storage_dir: Path) -> int:
    clear_index_artifacts(storage_dir)
    profiles = prepare_candidates(data_dir=data_dir, storage_dir=storage_dir)
    return len(profiles)


def main() -> None:
    print(f"Removing index artifacts in {STORAGE_DIR}...")
    indexed_count = rebuild_index(data_dir=DATA_DIR, storage_dir=STORAGE_DIR)
    print(f"Rebuilt index for {indexed_count} candidate(s).")


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
