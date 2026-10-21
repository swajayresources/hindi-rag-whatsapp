"""Operator CLI: serve | ingest | evaluate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, get_settings
from .evaluation import evaluate_service, load_cases, render_report, write_report
from .generation import get_rag_service
from .ingestion import ingest_corpus
from .logging_config import configure_logging
from .retrieval import get_vector_store


def _settings(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    if getattr(args, "corpus_dir", None):
        settings.corpus_dir = Path(args.corpus_dir)
    if getattr(args, "rebuild", False):
        pass
    return settings


def ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    configure_logging(settings.log_level, settings.log_json)
    store = get_vector_store(settings)
    report = ingest_corpus(
        settings.corpus_dir,
        store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        rebuild=args.rebuild,
    )
    print(
        json.dumps(
            {
                "documents": report.documents,
                "chunks": report.chunks,
                "skipped_existing": report.skipped_existing,
                "collection": settings.collection_name,
            },
            ensure_ascii=False,
        )
    )
    return 0


def evaluate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    configure_logging(settings.log_level, False)
    cases = load_cases(args.eval_set)
    service = get_rag_service(settings)
    report = evaluate_service(
        service,
        cases,
        k=settings.retrieval_k,
        include_ragas=args.ragas,
        include_answers=not args.retrieval_only,
    )
    print(render_report(report))
    if args.output:
        path = write_report(report, args.output)
        print(f"report written to {path}")
    return 0


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "hindi_rag.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hindi-rag", description="Hindi RAG service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=serve)

    p_ingest = sub.add_parser("ingest", help="index the corpus")
    p_ingest.add_argument("--corpus-dir", default=None)
    p_ingest.add_argument("--rebuild", action="store_true")
    p_ingest.set_defaults(func=ingest)

    p_eval = sub.add_parser("evaluate", help="run retrieval/answer evaluation")
    p_eval.add_argument("--eval-set", default="./data/eval/questions.jsonl")
    p_eval.add_argument("--output", default="./data/eval/latest_report.json")
    p_eval.add_argument("--ragas", action="store_true", help="also run RAGAS judge metrics")
    p_eval.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip LLM answers (offline, no API calls)",
    )
    p_eval.set_defaults(func=evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
