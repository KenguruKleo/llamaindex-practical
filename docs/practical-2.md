# LlamaIndex + Agents Practical - 2

## Assignment scope
Extend the resume system with a ReAct agent and multiple tools, including retrieval from vector storage, to answer candidate-related and broader questions.

## ReAct agent implementation

- Agent module: `app/agent.py`
- Agent type: `ReActAgent`
- LLM source: `create_llm()` from `app/llm_provider.py` (OpenAI or Azure configurable)
- Candidate metadata source: `load_profiles()` from `app/profile_store.py`

## Tools implemented

Each tool is isolated in its own module under `app/tools/`.

### 1. Retrieval Tool (resume-focused)
- Tool name: `candidate_rag_retriever`
- Backed by `VectorStoreIndex` over ChromaDB.
- Returns concise answer + matched candidate IDs with `?candidate=<id>` links.
- File: `app/tools/candidate_rag_retriever.py`

### 2. Profile Lookup Tool
- Tool name: `profile_lookup`
- Retrieves full structured profile by candidate ID or exact name.
- File: `app/tools/profile_lookup.py`

### 3. Candidate Directory Tool
- Tool name: `candidate_directory`
- Lists candidates with optional keyword filtering over ID, profession, skills, and summary.
- File: `app/tools/candidate_directory.py`

### 4. General Knowledge Tool
- Tool name: `general_knowledge`
- Uses configured LLM directly for questions outside resume data.
- File: `app/tools/general_knowledge.py`

### 5. Pre-built Integration Tool (LlamaIndex community)
- Tool source: `WikipediaToolSpec` (from `llama-index-tools-wikipedia`).
- Agent auto-registers Wikipedia search tool when dependency is available.
- File: `app/tools/wikipedia_search.py`

## Conversation behavior

### Chat history
- Streamlit stores message history in `st.session_state.messages`.
- History is converted to `List[ChatMessage]` and passed to the agent.
- Enables multi-turn follow-up queries.

### Fallback behavior
- If ReAct reaches max iterations, app catches `WorkflowRuntimeError`.
- Fallback retrieval answer is returned instead of crashing the app.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/web.py
```

## Provider configuration (OpenAI / Azure)

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

### Azure OpenAI

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_LLM_DEPLOYMENT=<chat-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
```
