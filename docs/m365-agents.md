Microsoft 365 Agent (CLI-only) – BrandID
========================================

This repo includes Make targets to scaffold, wire, and preview a Microsoft 365 agent using the Microsoft 365 Agents Toolkit CLI (`atk`). You can run everything from the root of this repo.

Prerequisites
- Node 18/20/22 and npm
- ngrok installed and signed in
- Your BrandID API running locally (default `http://localhost:8000`)
- An API Key registration in the Microsoft Dev Portal (keep the Registration ID handy)
- An existing Entra app named “BrandID-Agent” in your tenant

Environment
- Add these to `.env.local` or export in your shell before Make targets:
  - `NGROK_URL` e.g. `https://<id>.ngrok-free.app`
  - `API_KEY_REGISTRATION_ID` from the Dev Portal (not the secret)

Quickstart (API key path)
1) Start your API and ngrok
- In one terminal: `make api` (starts FastAPI on 8000)
- In another: `make tunnel` (shows your `https://<id>.ngrok-free.app`)

2) Run the agent flow (API key)
- `make atk-quickstart NGROK_URL=https://<id>.ngrok-free.app API_KEY_REGISTRATION_ID=<registration_id>`
  - Scaffolds `brandid-agent/` (if missing)
  - Writes `plugins/brandid/openapi.yaml` from a template
  - Adds the BrandID action
  - Validates, packages, and opens preview

Interactive steps
- During `make atk-quickstart` you may be prompted to:
  - Sign in: `atk auth login` (select your BrandIDdev directory)
  - Bind Entra app: later run `make atk-entra-update` and choose “Use existing” → “BrandID-Agent”

OAuth (client credentials) option
- If your API is protected by Entra ID, render the OAuth variant:
  - `make atk-new`
  - `make atk-openapi-oauth NGROK_URL=https://<id>.ngrok-free.app TENANT_ID=<tenant_guid> APP_ID_URI="api://<app-guid>/AuthenticatedBrandID" OAUTH_REGISTRATION_ID=<vault_reg_id>`
  - `make atk-add-action && make atk-validate && make atk-package && make atk-preview`
- Notes:
  - `APP_ID_URI` must match your API’s Application ID URI + scope (e.g., `api://9159...5f91/AuthenticatedBrandID`).
  - `OAUTH_REGISTRATION_ID` refers to the OAuth registration you created in the Dev Portal (holds client id/secret). No secrets are stored in this repo.

No-auth smoke test (when Dev Portal registration isn’t ready)
- Use this to validate the agent → action → API round trip without secrets:
  - `make atk-new`
  - `make atk-openapi-noauth NGROK_URL=https://<id>.ngrok-free.app`
  - `make atk-add-action && make atk-validate && make atk-package && make atk-preview`
- Ensure your API temporarily allows GET `/rules` without auth (200 OK). Do not ship this to production.
- Once your API Key registration is created, re-run `make atk-openapi NGROK_URL=... API_KEY_REGISTRATION_ID=...` and `make atk-preview`.

Individual commands (if you prefer step-by-step)
- Install CLI: `make atk-install-cli`
- Check env: `make atk-doctor`
- Scaffold project: `make atk-new`
- Render OpenAPI: `make atk-openapi NGROK_URL=... API_KEY_REGISTRATION_ID=...`
- Fallback (copy example): `make atk-openapi-copy-example` (uses `agents/openapi.brandid.api-key.yaml`)
- Add action: `make atk-add-action`
- Bind Entra app: `make atk-entra-update`
- Validate/package/preview: `make atk-validate && make atk-package && make atk-preview`
- Install to Teams: `make atk-install`

Test prompt in preview
- “Find the brand rules for ‘header logo spacing’ using the BrandID tool.”

Notes and tips
- Update the OpenAPI `servers[0].url` anytime ngrok changes by rerunning `make atk-openapi ...` and then `make atk-preview`.
- Keep your API root `/` returning 200 (GET/HEAD) for Dev Portal validation checks.
- Real endpoints should require the API key; the platform injects it via `ApiKeyPluginVault` at runtime.
