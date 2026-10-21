# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-21

### Added
- `src/` package layout with a FastAPI app factory and dependency injection.
- Typed, validated configuration (`pydantic-settings`) with fail-fast credential checks.
- Structured JSON logging with request correlation ids and latency fields.
- Twilio webhook signature validation (on by default).
- Per-client rate limiting middleware with `Retry-After`.
- Retries with exponential backoff + jitter and explicit timeouts for LLM calls.
- Refusal detection surfaced as `grounded: false`; no LLM call when retrieval is empty.
- `/health`, `/ready`, `/v1/chat`, `/whatsapp/webhook`; `/chat` kept as an alias.
- Verified evaluation set (26 questions) and retrieval metrics (hit@k, recall@k, MRR).
- Retrieval-only evaluation mode for offline/CI runs; optional RAGAS judge metrics.
- 47 unit tests (config, schemas, chunking, generation, API, middleware, WhatsApp).
- Multi-stage non-root Dockerfile, docker-compose, CI workflow, Makefile.
- Documentation: README, CONTRIBUTING, SECURITY.

### Fixed
- `python-multipart` was missing, so `Form(...)` routes failed at import time.
- `langchain-ollama` was missing while Ollama was the default backend.
- Groq model default pointed at a decommissioned model (`llama3-8b-8192`).
- Embedding model name disagreed between config and docs.
- Vector store used LangChain's private `_collection` attribute, which breaks across releases.
- Unhandled provider errors leaked a 500 with a stack trace to WhatsApp users.

### Changed
- Retrieval is MMR-based with a configurable candidate pool.
- Answer pipeline is explicit (retrieve → prompt → invoke) rather than an opaque chain.

## [1.0.0] - 2025-04-26

- Initial Hindi RAG WhatsApp bot: Chroma + FastEmbed + Groq/Ollama + Twilio.
