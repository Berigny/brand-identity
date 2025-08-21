# Brand Identity Agent (VS Code Chat)

## Setup
- Ensure `.env.local` in repo root with:
  - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  - RAG_ENDPOINT and OPENROUTER_API_KEY (or RAG_API_KEY)

## Run
- Open this folder in VS Code
- `npm install`
- `F5` to launch “Extension Development Host”
- Open **Chat** (Copilot/Chat), choose **Brand Identity Agent**
  - `/queryBrandRule icon monochrome constraints`
  - `/ragSearch How should we name sale icons? topK: 5`
