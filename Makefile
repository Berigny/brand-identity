SHELL := /bin/bash

VENV_DIR := .venv
PYTHON3 ?= python3
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Load environment variables from .env.local if present
ifneq (,$(wildcard .env.local))
	include .env.local
	export
endif

PROMPT ?= Add a new tertiary emotional token with high prime, test for drift, and refine for consilience.

.PHONY: install start agent stop restart check clean logs reset

$(VENV_DIR):
	$(PYTHON3) -m venv $(VENV_DIR)

install: $(VENV_DIR)
	@echo "Upgrading pip and installing requirements..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nInstall complete. Ensure .env.local contains NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, and OpenRouter keys."

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
	@echo "Stopped Streamlit (port 8501) and agent processes if they were running."

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
