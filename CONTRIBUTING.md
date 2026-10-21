# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
make test
```

## Before opening a PR

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest
```

All three must pass; CI runs the same commands.

## Conventions

- **Typed public boundaries.** New settings go in `config.py` with validation; new API shapes go in `schemas.py`.
- **No private attributes of third-party libraries.** Wrap them (`retrieval.VectorStore` exists precisely for this).
- **Failure modes first.** Network calls need retries and timeouts; user-facing errors must not leak stack traces.
- **Tests must not require network, an LLM key, or a vector database.** Inject fakes (see `tests/conftest.py`).
- **Hindi handling is not incidental.** If you touch chunking or prompting, keep `।`-aware splitting and the refusal contract.

## Adding evaluation questions

Use `scripts/build_eval_set.py`: it keeps a question only if the ground-truth string is literally present in the cited corpus file, so the benchmark cannot drift into fiction.
