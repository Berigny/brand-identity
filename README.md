# Brand Identity Agent (VS Code Chat)

## Setup
- Ensure `.env.local` in repo root with:
  - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  - RAG_ENDPOINT and OPENROUTER_API_KEY (or RAG_API_KEY)

## Run (Local, no Microsoft login)
- Ensure Python 3.10+ is installed.
- Create a virtualenv and install deps: `make install`
- Start the FastAPI server (optional for local UI): `make api` (serves on `http://localhost:8000`)
- Start the Streamlit UI: `make start` (opens on `http://localhost:8501`)

Notes
- Microsoft 365/Copilot integration under `BrandID/` is optional and requires Microsoft login; ignore those `make ms365-*` targets for local use.
- Some features (Agent tab) require LLM credentials. Set either Azure OpenAI variables (`AZURE_OPENAI_*`) or OpenRouter (`OPENROUTER_API_KEY`) in `.env.local`.
