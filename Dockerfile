# syntax=docker/dockerfile:1

# ─── build stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# Pre-download the embedding model so container cold starts stay fast.
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ─── runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /home/builder /home/appuser
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser data/corpus ./data/corpus
COPY --chown=appuser:appuser data/eval ./data/eval

RUN mkdir -p /app/data/chroma && chown -R appuser:appuser /app /home/appuser

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "hindi_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
