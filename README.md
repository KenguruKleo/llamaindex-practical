# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, generates embeddings with LlamaIndex, stores vectors in ChromaDB, and serves a Streamlit web app for candidate browsing and agent chat.

The app supports both providers:
- OpenAI
- Azure OpenAI

## Assessment docs

- [Practical 1 implementation notes](docs/practical-1.md) - CV ingestion, chunking, embeddings, vector DB, candidate metadata extraction, and web UI mapping.
- [Practical 2 implementation notes](docs/practical-2.md) - ReAct agent architecture, tools, chat history behavior, and fallback handling.

## Requirements

- Python 3.12+
- Virtual environment (recommended)
- API credentials for one provider:
  - OpenAI (`OPENAI_API_KEY`)
  - Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, deployments)
- PDF resumes placed under `data/` (current dataset size: 25 CV files)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment configuration

Create `.env` from sample first:

```bash
cp .env.sample .env
```

Important:
- Set `LLM_PROVIDER` explicitly to `openai` or `azure`.
- Fill credentials for the selected provider.

### Option A: OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

### Option B: Azure OpenAI

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_LLM_DEPLOYMENT=<chat-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
```

## Preparing the data

The web app auto-indexes only if index artifacts are missing or incomplete.

Run indexing manually:

```bash
python -m app.data_pipeline
```

Run a full clean rebuild (deletes `storage/chroma/` and `storage/candidates.json`, then re-indexes):

```bash
python -m app.rebuild_index
```

Generated artifacts:
- `storage/candidates.json` - candidate metadata and summaries
- `storage/chroma/` - persistent ChromaDB vector store (`candidates` collection)

## Running the web application

```bash
streamlit run app/web.py
```

Default UI behavior:
- `Agent Chat` tab opens first
- `Clear chat / New chat` resets session chat history
- `Candidate Directory` shows candidate cards and detailed profile pages
- Candidate details include summary, skills, source file, and PDF preview/download
- Opening links like `?candidate=<id>` switches default focus to profile view in `Candidate Directory`

## Agent Chat test prompts

Use these prompts in `Agent Chat` to validate behavior with current data:

- `Show full profile for candidate 12472574.`
- `Find accountant candidates and rank top 3 by years of experience (focus on 10554236, 10674770, 11163645, 11759079).`
- `Compare chef candidates 19831366, 20817322, and 21101152 for an Executive Chef role.`
- `Find candidates with digital transformation/strategy background (17432318, 17562754, 18488289, 10515955).`
- `Які кандидати найкраще підходять на роль QA Team Lead? Розглянь 12472574 і суміжних кандидатів.`
- `Покажи 2 найрелевантніших кандидатів для hospitality/food service (11835339, 12334650, 16248476).`
- `Who wrote the book Clean Code?` (general knowledge tool)

Tool routing in agent:
- `candidate_rag_retriever` is the primary RAG tool for resume-grounded answers.
- `profile_lookup` is used for detailed output when candidate ID/name is known.
- `candidate_directory` is used for lightweight metadata list/filter or fallback when RAG has no evidence.
- `general_knowledge` and optional Wikipedia tool are used for non-resume questions.

### Chat history test (multi-turn)

Run in the same session:

1. `Find 3 best candidates for a finance/accounting manager role and explain briefly.`
2. `Take the first candidate from your previous answer and provide a detailed profile.`
3. `Now compare that candidate with 11759079 and give a final recommendation.`

Expected behavior:
- candidate IDs appear as clickable `?candidate=<id>` links
- best-match candidate has structured profile fields
- follow-up prompts use prior chat context
- if ReAct reaches iteration limit, fallback retrieval response is returned instead of app crash
- placeholder values are rendered as user-friendly text (`Candidate name not provided`, `Profession not provided`)

## Project layout

```text
app/
  __init__.py        # shared exports + prepare_candidates helper
  agent.py           # ReAct agent orchestration and fallback behavior
  data_pipeline.py   # indexing pipeline (ingestion/chunking + Chroma writes)
  llm_provider.py    # OpenAI/Azure provider selection + LLM/embedding factories
  models.py          # domain models (CandidateProfile)
  paths.py           # centralized project paths (data/storage/chroma)
  profile_store.py   # candidate metadata read/write + index existence checks
  rebuild_index.py   # full clean rebuild command (clear + reindex)
  web.py             # Streamlit UI
  tools/
    __init__.py
    common.py                # shared formatting helpers for tool outputs
    candidate_rag_retriever.py  # vector retrieval (RAG) tool + fallback renderer
    profile_lookup.py        # profile lookup by id/name
    candidate_directory.py   # list/filter candidates from metadata
    general_knowledge.py     # non-resume Q&A via configured LLM
    wikipedia_search.py      # optional pre-built Wikipedia tool

docs/
  practical-1.md     # Practical 1 mapping to implementation
  practical-2.md     # Practical 2 mapping to implementation
```
