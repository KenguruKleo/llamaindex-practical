from __future__ import annotations

import importlib


def get_wikipedia_search_tool():
    try:
        module = importlib.import_module("llama_index.tools.wikipedia")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return None

    tool_spec_cls = getattr(module, "WikipediaToolSpec", None)
    if tool_spec_cls is None:
        return None

    try:
        tools = tool_spec_cls().to_tool_list()
    except Exception:  # pragma: no cover - defensive
        return None

    if not tools:
        return None

    for tool in tools:
        metadata = getattr(tool, "metadata", None)
        name = str(getattr(metadata, "name", "")).lower()
        if "search" in name:
            return tool
    return tools[0]
