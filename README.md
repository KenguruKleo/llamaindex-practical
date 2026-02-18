# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, embeds the passages with OpenAI embeddings, and writes them to a persistent ChromaDB vector store via LlamaIndex. Candidate metadata (name, profession, skills, summary) is produced by querying each Chroma-backed index with an OpenAI chat model. A Streamlit UI provides an `Agent Chat` tab (default view) plus a candidate directory with detailed profile pages.

## Requirements

- Python 3.12+
- A virtual environment (recommended)
- An OpenAI API key with access to the embedding and chat models you intend to use
- PDF resumes placed under `data/`

Create a virtual environment, activate it, and install dependencies (adjust the activation command for your shell/OS):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# optional: echo "OPENAI_API_KEY=sk-..." > .env
```

Reactivate the environment in new shells with `source .venv/bin/activate` (or the corresponding Windows command) before running any project commands.

## Preparing the data

The web app will automatically run indexing only if the index is missing or incomplete. To trigger indexing manually:

```bash
python -m app.data_pipeline
```

Generated artifacts are stored under `storage/`:

- `candidates.json` – metadata, summaries, and skills
- `chroma/` – persistent ChromaDB storage (single `candidates` collection containing every resume chunk)

If you add or update resumes, rebuild the index:

```bash
REBUILD_INDEX=1 python -m app.data_pipeline
```

## Running the web application

Serve the Streamlit UI:

```bash
streamlit run app/web.py
```

Streamlit prints the local URL in the terminal (typically http://localhost:8501).

- `Agent Chat` opens first by default.
- Use `Clear chat / New chat` to reset chat history in the current session.
- `Candidate Directory` shows all indexed candidates with high-level metadata and summaries.
- Clicking `View full profile` opens a dedicated profile view that exposes full summary, skills, and resume preview/download.

### Agent Chat test prompts

Use these prompts in the `Agent Chat` tab to validate retrieval against the current dataset in `data/` (25 resumes):

- `Show full profile for candidate 12472574.`
- `Find accountant candidates and rank top 3 by years of experience (focus on 10554236, 10674770, 11163645, 11759079).`
- `Compare chef candidates 19831366, 20817322, and 21101152 for an Executive Chef role.`
- `Find candidates with digital transformation/strategy background (17432318, 17562754, 18488289, 10515955).`
- `Які кандидати найкраще підходять на роль QA Team Lead? Розглянь 12472574 і суміжних кандидатів.`
- `Покажи 2 найрелевантніших кандидатів для hospitality/food service (11835339, 12334650, 16248476).`

### Chat history test (multi-turn)

Run these messages in order in the same chat session:

1. `Find 3 best candidates for a finance/accounting manager role and explain briefly.`
2. `Take the first candidate from your previous answer and provide a detailed profile.`
3. `Now compare that candidate with 11759079 and give a final recommendation.`

If chat history is working, messages 2 and 3 should be interpreted using context from earlier turns (without repeating the first candidate ID manually).

Expected behavior:
- The answer includes matched candidate IDs as clickable links in format `?candidate=<id>`.
- For the most relevant candidate, the assistant returns detailed profile fields (name, profession, skills, summary).
- If ReAct reaches iteration limit, the app should return a fallback retrieval answer instead of crashing.
- Placeholder values are rendered as user-friendly text (`Candidate name not provided`, `Profession not provided`).

## Project layout

```
app/
  __init__.py        # prepare_candidates helper
  agent.py           # ReAct agent, tools, and fallback behavior
  data_pipeline.py   # ingestion, chunking, embeddings, Chroma persistence
  web.py             # Streamlit entry point
```

## Notes

- Set `OPENAI_EMBED_MODEL` to switch to a different embedding model (defaults to `text-embedding-3-small`).
- Set `OPENAI_LLM_MODEL` to the chat model that extracts metadata and summaries (defaults to `gpt-4o-mini`).
- Make sure `OPENAI_API_KEY` is available in the environment (or `.env`) before running the pipeline or web app.
- ChromaDB stores vectors locally under `storage/chroma/`; remove this folder if you need a clean slate or to clear the shared `candidates` collection.
- Legacy Flask templates have been removed; the Streamlit app renders the UI directly in `app/web.py`.
