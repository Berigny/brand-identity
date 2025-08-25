SHELL := /bin/bash

VENV_DIR := .venv
PYTHON3 ?= python3
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

PROMPT ?= Add a new tertiary emotional token with high prime, test for drift, and refine for consilience.

.PHONY: install start agent stop restart check clean logs reset \
        ms365-env ms365-openapi ms365-provision ms365-preview tunnel ms365-setup \
        ms365-install-cli ms365-publish ms365-login ms365-doctor \
        atk-install-cli atk-doctor atk-new atk-openapi atk-openapi-noauth atk-openapi-oauth atk-openapi-copy-example atk-add-action \
        atk-entra-update atk-validate atk-package atk-preview atk-install \
        atk-quickstart help

$(VENV_DIR):
	$(PYTHON3) -m venv $(VENV_DIR)

install: $(VENV_DIR)
	@echo "Upgrading pip and installing requirements..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nInstall complete. Ensure .env.local contains:"
	@echo " - Neo4j: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE"
	@echo " - Azure OpenAI: OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION, OPENAI_DEPLOYMENT_NAME"
	@echo " - (Optional) OpenRouter: OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL"

PORT ?= 8000

api: $(VENV_DIR)
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

# Start the Streamlit UI in the background
start: $(VENV_DIR)
	@echo "Starting Streamlit UI in the background..."
	@nohup $(PYTHON) -m streamlit run streamlit_app.py > streamlit.log 2>&1 &
	@echo "Streamlit UI started. Check streamlit.log for output."

# Run the RAG agent once with a default prompt
agent: $(VENV_DIR)
	$(PYTHON) rag_agent.py "$(PROMPT)"

# Stop Streamlit and any running agent process
stop:
	@echo "Stopping processes..."
	# Gracefully kill Streamlit and agent processes using pkill
	-pkill -f "streamlit run streamlit_app.py" >/dev/null 2>&1 || true
	-pkill -f "[p]ython.*rag_agent.py" >/dev/null 2>&1 || true
	# On macOS, to free up a port, use lsof. This is a more forceful stop.
	# Useful if a process is stuck and pkill doesn't work.
	-lsof -ti :8501 | xargs kill -9 2>/dev/null || true
	-lsof -ti :8000 | xargs kill -9 2>/dev/null || true
	@echo "Stopped Streamlit, agent, and FastAPI (port 8501 & 8000)."

# Restart Streamlit
restart: stop start

# View logs for the agent and streamlit app
logs:
	@echo "Tailing logs for streamlit..."
	@tail -f streamlit.log

# Initialize Neo4j DB with constraints and seed data
db-setup: $(VENV_DIR)
	$(PYTHON) scripts/setup_neo4j.py

# Reset by stopping all processes and running checks
reset: stop check

# Quick syntax check for main modules
check: $(VENV_DIR)
	$(PYTHON) -m py_compile rag_agent.py app/utils/flow_simulator.py streamlit_app.py || true

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f streamlit.log
	@echo "Clean complete."

# ---- Microsoft 365 Agent helpers ----

# Write the BRANDID secret for the declarative agent
ms365-env:
	@if [ -z "$(BRANDID)" ]; then \
		echo "BRANDID is not set. Add it to .env.local or export before running."; \
		exit 1; \
	fi
	@mkdir -p BrandID/env
	@echo "Writing BrandID/env/.env.dev.user with BRANDID..."
	@printf "BRANDID=%s\n" "$(BRANDID)" > BrandID/env/.env.dev.user
	@echo "Done."

# Update OpenAPI server URL to current tunnel
# Usage: make ms365-openapi NGROK_URL=https://<id>.ngrok-free.app
ms365-openapi:
	@if [ -z "$(NGROK_URL)" ]; then \
		echo "NGROK_URL is required. Usage: make ms365-openapi NGROK_URL=https://<id>.ngrok-free.app"; \
		exit 1; \
	fi
	@FILE=BrandID/appPackage/apiSpecificationFile/openapi.yaml; \
	if [ "`uname`" = "Darwin" ]; then \
		sed -E -i '' "s|^[[:space:]]*-[[:space:]]url:.*|  - url: $(NGROK_URL)|" $$FILE; \
	else \
		sed -E -i "s|^[[:space:]]*-[[:space:]]url:.*|  - url: $(NGROK_URL)|" $$FILE; \
	fi; \
	@echo "Updated OpenAPI servers[0].url -> $(NGROK_URL)"

# Provision the agent app via Teams Toolkit CLI
ms365-provision:
	@cd BrandID && teamsapp provision

# Preview in Copilot (Edge by default)
ms365-preview:
	@cd BrandID && teamsapp preview --m365-host copilot --browser edge

# Start a tunnel to the API port
tunnel:
	@echo "Starting ngrok tunnel on port $(PORT)..."
	@ngrok http $(PORT)

# One-shot: write env, update OpenAPI, and provision
# Usage: make ms365-setup NGROK_URL=https://<id>.ngrok-free.app
ms365-setup: ms365-doctor ms365-env ms365-openapi ms365-provision

# Install Microsoft 365 Teams Toolkit CLI (global)
ms365-install-cli:
	@if command -v teamsapp >/dev/null 2>&1; then \
		echo "Teams Toolkit CLI already installed: $$(teamsapp --version)"; \
	else \
		echo "Installing Teams Toolkit CLI globally via npm..."; \
		npm i -g @microsoft/teamsapp-cli; \
		echo "Installed: $$(teamsapp --version)"; \
	fi

# Publish the app to your tenant's admin center
ms365-publish:
	@cd BrandID && teamsapp publish

# Show handy Microsoft 365 targets and usage
help:
	@echo "Microsoft 365 Agent targets:" 
	@printf "  %-28s %s\n" "ms365-install-cli" "Install Teams Toolkit CLI globally"
	@printf "  %-28s %s\n" "ms365-login" "Sign in to Microsoft 365"
	@printf "  %-28s %s\n" "ms365-env" "Write BrandID/env/.env.dev.user (uses BRANDID)"
	@printf "  %-28s %s\n" "ms365-openapi NGROK_URL=..." "Update OpenAPI servers[0].url"
	@printf "  %-28s %s\n" "ms365-provision" "Provision the agent app"
	@printf "  %-28s %s\n" "ms365-preview" "Open Copilot preview (Edge)"
	@printf "  %-28s %s\n" "ms365-publish" "Publish app to tenant admin center"
	@printf "  %-28s %s\n" "ms365-setup NGROK_URL=..." "Env + OpenAPI + Provision"
	@printf "  %-28s %s\n" "ms365-doctor" "Check prerequisites and configuration"
	@printf "  %-28s %s\n" "tunnel" "Start ngrok http $(PORT) (default 8000)"
	@printf "  %-28s %s\n" "api" "Run FastAPI locally on $(PORT)"
	@echo ""
	@echo "Agents Toolkit (atk) targets:"
	@printf "  %-28s %s\n" "atk-install-cli" "Install M365 Agents Toolkit CLI"
	@printf "  %-28s %s\n" "atk-doctor" "Check atk prerequisites"
	@printf "  %-28s %s\n" "atk-new" "Scaffold declarative agent (brandid-agent)"
	@printf "  %-28s %s\n" "atk-openapi NGROK_URL=... API_KEY_REGISTRATION_ID=..." "Render plugin OpenAPI"
	@printf "  %-28s %s\n" "atk-openapi-noauth NGROK_URL=..." "Render plugin OpenAPI (no auth)"
	@printf "  %-28s %s\n" "atk-openapi-copy-example" "Copy pre-rendered example spec into project"
	@printf "  %-28s %s\n" "atk-add-action" "Add BrandID action from OpenAPI"
	@printf "  %-28s %s\n" "atk-entra-update" "Bind to existing Entra app"
	@printf "  %-28s %s\n" "atk-validate" "Validate project"
	@printf "  %-28s %s\n" "atk-package" "Package app (zip)"
	@printf "  %-28s %s\n" "atk-preview" "Upload and open preview"
	@printf "  %-28s %s\n" "atk-install" "Install packaged app to Teams"
	@printf "  %-28s %s\n" "atk-quickstart NGROK_URL=... API_KEY_REGISTRATION_ID=..." "Run scaffold→wire→preview"
	@printf "  %-28s %s\n" "kernel-validate" "Validate brand_kernel.json primes vs policy"

# Quick test of Azure OpenAI wiring
.PHONY: test-azure
test-azure: $(VENV_DIR)
	$(PYTHON) test_azure.py

# Quick test of OpenRouter wiring
.PHONY: test-openrouter
test-openrouter: $(VENV_DIR)
	$(PYTHON) test_openai.py

# Test LLM factory (proxy/local/openrouter auto)
.PHONY: test-factory
test-factory: $(VENV_DIR)
	$(PYTHON) test_factory.py

# Test reranker
.PHONY: test-rerank
test-rerank: $(VENV_DIR)
	$(PYTHON) test_rerank.py

# LiteLLM proxy helpers
.PHONY: proxy-up proxy-down proxy-logs
proxy-up:
	docker compose up -d
proxy-down:
	docker compose down
proxy-logs:
	docker compose logs -f litellm

.PHONY: kernel-validate
kernel-validate: $(VENV_DIR)
	$(PYTHON) scripts/validate_brand_kernel.py brand_kernel.json

# ---- Microsoft 365 Agents Toolkit (atk) helpers ----

ATK ?= atk
AGENT_DIR ?= brandid-agent
PLUGIN_DIR := $(AGENT_DIR)/plugins/brandid
PLUGIN_SPEC := $(PLUGIN_DIR)/openapi.yaml
PLUGIN_TEMPLATE := agents/openapi.yaml.tmpl
PLUGIN_TEMPLATE_OAUTH := agents/openapi.oauth.yaml.tmpl

# Install the Agents Toolkit CLI globally
atk-install-cli:
	@if command -v $(ATK) >/dev/null 2>&1; then \
		echo "Agents Toolkit CLI already installed: $$($(ATK) -v 2>/dev/null)"; \
	else \
		echo "Installing Agents Toolkit CLI globally via npm..."; \
		npm i -g @microsoft/m365agentstoolkit-cli; \
		echo "Installed: $$($(ATK) -v 2>/dev/null)"; \
	fi

# Quick prerequisite and configuration checks for atk
atk-doctor:
	@echo "Checking atk + environment..."
	@echo "- atk version:" && ($(ATK) -v || echo "[MISSING] install via 'make atk-install-cli'")
	@echo "- atk doctor:" && ($(ATK) doctor || true)
	@echo "- Node version:" && node -v || echo "Node not found"
	@if command -v ngrok >/dev/null 2>&1; then \
		echo "- ngrok: present"; \
	else \
		echo "- ngrok: [MISSING] install from https://ngrok.com/"; \
	fi

# Scaffold a minimal declarative agent project
atk-new:
	@if [ -d "$(AGENT_DIR)" ]; then \
		echo "Directory '$(AGENT_DIR)' already exists; skipping scaffold."; \
	else \
		$(ATK) new -n $(AGENT_DIR) -c declarative-agent -l typescript; \
	fi

# Render the BrandID plugin OpenAPI from template
# Usage: make atk-openapi NGROK_URL=https://<id>.ngrok-free.app API_KEY_REGISTRATION_ID=abc123
atk-openapi:
	@if [ -z "$(NGROK_URL)" ]; then \
		echo "NGROK_URL is required. Usage: make atk-openapi NGROK_URL=https://<id>.ngrok-free.app API_KEY_REGISTRATION_ID=..."; \
		exit 1; \
	fi
	@if [ -z "$(API_KEY_REGISTRATION_ID)" ]; then \
		echo "API_KEY_REGISTRATION_ID is required. Find it in the Dev Portal registration."; \
		exit 1; \
	fi
	@mkdir -p $(PLUGIN_DIR)
	@echo "Writing $(PLUGIN_SPEC) from $(PLUGIN_TEMPLATE) ..."
	@sed \
		-e "s|{{NGROK_URL}}|$(NGROK_URL)|g" \
		-e "s|{{API_KEY_REGISTRATION_ID}}|$(API_KEY_REGISTRATION_ID)|g" \
		$(PLUGIN_TEMPLATE) > $(PLUGIN_SPEC)
	@echo "OpenAPI written to $(PLUGIN_SPEC)"

# Render a no-auth variant for smoke testing
# Usage: make atk-openapi-noauth NGROK_URL=https://<id>.ngrok-free.app
atk-openapi-noauth:
	@if [ -z "$(NGROK_URL)" ]; then \
		echo "NGROK_URL is required. Usage: make atk-openapi-noauth NGROK_URL=https://<id>.ngrok-free.app"; \
		exit 1; \
	fi
	@mkdir -p $(PLUGIN_DIR)
	@echo "Writing $(PLUGIN_SPEC) from agents/openapi.noauth.yaml.tmpl ..."
	@sed \
		-e "s|{{NGROK_URL}}|$(NGROK_URL)|g" \
		agents/openapi.noauth.yaml.tmpl > $(PLUGIN_SPEC)
	@echo "No-auth OpenAPI written to $(PLUGIN_SPEC)"

# Copy the pre-rendered example OpenAPI into the scaffolded project
atk-openapi-copy-example:
	@if [ ! -d "$(AGENT_DIR)" ]; then \
		echo "'$(AGENT_DIR)' does not exist. Run 'make atk-new' first."; \
		exit 1; \
	fi
	@mkdir -p $(PLUGIN_DIR)
	@cp -f agents/openapi.brandid.api-key.yaml $(PLUGIN_SPEC)
	@echo "Copied agents/openapi.brandid.api-key.yaml -> $(PLUGIN_SPEC)"

# Render the OAuth (client credentials) variant of the plugin OpenAPI
# Usage: make atk-openapi-oauth NGROK_URL=https://<id>.ngrok-free.app TENANT_ID=<guid> APP_ID_URI="api://.../scope" OAUTH_REGISTRATION_ID=<vault_reg_id>
atk-openapi-oauth:
	@if [ -z "$(NGROK_URL)" ]; then \
		echo "NGROK_URL is required."; exit 1; \
	fi
	@if [ -z "$(TENANT_ID)" ]; then \
		echo "TENANT_ID is required (AAD tenant GUID)."; exit 1; \
	fi
	@if [ -z "$(APP_ID_URI)" ]; then \
		echo "APP_ID_URI is required (e.g., api://<app-guid>/scope)."; exit 1; \
	fi
	@if [ -z "$(OAUTH_REGISTRATION_ID)" ]; then \
		echo "OAUTH_REGISTRATION_ID is required (Dev Portal OAuth registration id)."; exit 1; \
	fi
	@mkdir -p $(PLUGIN_DIR)
	@echo "Writing $(PLUGIN_SPEC) from $(PLUGIN_TEMPLATE_OAUTH) ..."
	@sed \
		-e "s|{{NGROK_URL}}|$(NGROK_URL)|g" \
		-e "s|{{TENANT_ID}}|$(TENANT_ID)|g" \
		-e "s|{{APP_ID_URI}}|$(APP_ID_URI)|g" \
		-e "s|{{OAUTH_REGISTRATION_ID}}|$(OAUTH_REGISTRATION_ID)|g" \
		$(PLUGIN_TEMPLATE_OAUTH) > $(PLUGIN_SPEC)
	@echo "OAuth OpenAPI written to $(PLUGIN_SPEC)"

# Add the action to the agent project from the OpenAPI spec
atk-add-action:
	@if [ ! -f "$(PLUGIN_SPEC)" ]; then \
		echo "Missing $(PLUGIN_SPEC). Run 'make atk-openapi NGROK_URL=... API_KEY_REGISTRATION_ID=...' first."; \
		exit 1; \
	fi
	@cd $(AGENT_DIR) && $(ATK) add action --name brandidRules --openapi ./plugins/brandid/openapi.yaml

# Bind to existing Entra app (interactive to select BrandID-Agent)
atk-entra-update:
	@cd $(AGENT_DIR) && $(ATK) entra-app update

# Validate, package, preview, and install
atk-validate:
	@cd $(AGENT_DIR) && $(ATK) validate

atk-package:
	@cd $(AGENT_DIR) && $(ATK) package

atk-preview:
	@cd $(AGENT_DIR) && $(ATK) preview

atk-install:
	@cd $(AGENT_DIR) && $(ATK) install --file-path appPackage/build/appPackage.dev.zip

# One-shot quickstart (requires NGROK_URL and API_KEY_REGISTRATION_ID)
atk-quickstart: atk-install-cli atk-doctor atk-new atk-openapi atk-add-action atk-validate atk-package atk-preview

# Login to Microsoft 365 for Teams Toolkit
ms365-login:
	@cd BrandID && teamsapp account login m365

# Quick prerequisite and configuration checks
ms365-doctor:
	@echo "Checking prerequisites and config..."
	@echo "- Node version:" && node -v || echo "Node not found"
	@NODE_MAJOR=$$(node -v 2>/dev/null | sed 's/v\([0-9]*\).*/\1/'); \
	if [ -n "$$NODE_MAJOR" ] && [ $$NODE_MAJOR -ge 18 ] && [ $$NODE_MAJOR -le 22 ]; then \
		echo "  [OK] Node $$NODE_MAJOR is supported (18/20/22)"; \
	else \
		echo "  [WARN] Use Node 18/20/22 (found $$NODE_MAJOR)"; \
	fi
	@if command -v teamsapp >/dev/null 2>&1; then \
		echo "- Teams Toolkit CLI: $$(teamsapp --version)"; \
	else \
		echo "- Teams Toolkit CLI: [MISSING] install via 'make ms365-install-cli'"; \
	fi
	@if command -v ngrok >/dev/null 2>&1; then \
		echo "- ngrok: present"; \
	else \
		echo "- ngrok: [MISSING] install from https://ngrok.com/"; \
	fi
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) -c "import fastapi, uvicorn; print('- Python deps: FastAPI/Uvicorn OK')" 2>/dev/null || echo "- Python deps: [MISSING] run 'make install'"; \
	else \
		echo "- Virtualenv: not created. Run 'make install'"; \
	fi
	@if [ -n "$(BRANDID)" ]; then \
		echo "- BRANDID: set"; \
	else \
		echo "- BRANDID: [MISSING] add to .env.local or export before 'make ms365-env'"; \
	fi
	@URL=$$(grep -E "^[[:space:]]*- url:" BrandID/appPackage/apiSpecificationFile/openapi.yaml | head -1 | awk '{print $$3}'); \
	if [ -n "$$URL" ]; then \
		echo "- OpenAPI servers[0].url: $$URL"; \
	else \
		echo "- OpenAPI url: [UNKNOWN]"
	fi
	@echo "- M365 account status:" && (cd BrandID && teamsapp account show m365 2>/dev/null || echo "  Not signed in")