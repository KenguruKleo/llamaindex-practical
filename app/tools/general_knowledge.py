from __future__ import annotations

from llama_index.core.tools import FunctionTool


def build_general_knowledge_tool(llm) -> FunctionTool:
    def _answer_general_question(question: str) -> str:
        query = question.strip()
        if not query:
            return "Please provide a question."
        try:
            result = llm.complete(query)
        except Exception as exc:
            return f"General knowledge request failed: {exc}"
        text = getattr(result, "text", None)
        return str(text if text else result).strip()

    return FunctionTool.from_defaults(
        fn=_answer_general_question,
        name="general_knowledge",
        description="Answer general questions not directly related to resume data.",
    )
