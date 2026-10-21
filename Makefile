# ─── Hindi RAG Service ─────────────────────────────────────────────────────
.DEFAULT_GOAL := help
PY ?= python
VENV := .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PYTHON := $(BIN)/python
PIP := $(PYTHON) -m pip

.PHONY: help install dev lint format typecheck test cov run ingest evaluate evaluate-offline docker-build docker-up clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install runtime dependencies
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install .

dev: ## Install with development + evaluation extras
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint: ## Ruff
	$(PYTHON) -m ruff check src tests scripts

format: ## Ruff autofix + format
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

typecheck: ## mypy
	$(PYTHON) -m mypy src

test: ## Unit tests
	$(PYTHON) -m pytest -q

cov: ## Tests with coverage
	$(PYTHON) -m pytest --cov=hindi_rag --cov-report=term-missing

run: ## Start the API
	$(PYTHON) -m hindi_rag.cli serve

ingest: ## Build/refresh the vector index
	$(PYTHON) -m hindi_rag.cli ingest

evaluate: ## Full evaluation (needs LLM credentials)
	$(PYTHON) -m hindi_rag.cli evaluate

evaluate-offline: ## Retrieval-only evaluation (no API calls)
	$(PYTHON) -m hindi_rag.cli evaluate --retrieval-only

docker-build: ## Build the image
	docker build -t hindi-rag-whatsapp:latest .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
