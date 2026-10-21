"""Evaluation: retrieval quality + answer correctness.

Two layers, deliberately separated:

1. **Retrieval metrics** (no LLM judge, deterministic, cheap):
   hit@k, recall@k, MRR over the verified question set.
2. **Answer metrics**: grounded-answer accuracy (expected answer substring
   present AND not a refusal) and optional RAGAS scores when installed.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = Path("./data/eval/questions.jsonl")


@dataclass
class EvalCase:
    question: str
    ground_truth: str
    source: str


@dataclass
class CaseResult:
    question: str
    expected: str
    answer: str
    grounded: bool
    answer_correct: bool
    retrieved_sources: list[str]
    hit: bool
    reciprocal_rank: float
    latency_ms: float


@dataclass
class EvalReport:
    cases: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    k: int
    answers_evaluated: bool = True
    answer_accuracy: float | None = None
    refusal_rate: float | None = None
    mean_latency_ms: float | None = None
    results: list[CaseResult] = field(default_factory=list)
    ragas: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [asdict(r) for r in self.results]
        return data


def load_cases(path: str | Path = DEFAULT_EVAL_PATH) -> list[EvalCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evaluation set not found: {p}")
    cases: list[EvalCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                question=row["question"],
                ground_truth=row["ground_truth"],
                source=row.get("source", ""),
            )
        )
    if not cases:
        raise ValueError(f"empty evaluation set: {p}")
    return cases


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def answer_matches(answer: str, expected: str) -> bool:
    """Substring match on normalised text (Hindi-safe, no tokenisation needed)."""
    return _normalise(expected) in _normalise(answer)


def evaluate_service(
    service: Any,
    cases: Sequence[EvalCase],
    k: int,
    *,
    include_ragas: bool = False,
    include_answers: bool = True,
) -> EvalReport:
    """Evaluate retrieval (always) and answer quality (optionally).

    ``include_answers=False`` runs fully offline — useful in CI and for
    retrieval ablations without spending LLM calls.
    """
    results: list[CaseResult] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        docs = service.retrieve(case.question)
        retrieve_ms = (time.perf_counter() - started) * 1000
        sources = [str(d.metadata.get("source", "unknown")) for d in docs]
        rank = next((i + 1 for i, s in enumerate(sources) if s == case.source), 0)

        answer_text = ""
        grounded = False
        correct = False
        latency = retrieve_ms

        if include_answers:
            answer = service.answer(case.question, include_sources=False)
            answer_text = answer.text
            grounded = answer.grounded
            correct = answer_matches(answer.text, case.ground_truth) and answer.grounded
            latency = answer.latency_ms
        latencies.append(latency)

        results.append(
            CaseResult(
                question=case.question,
                expected=case.ground_truth,
                answer=answer_text,
                grounded=grounded,
                answer_correct=correct,
                retrieved_sources=sources,
                hit=rank > 0,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                latency_ms=round(latency, 1),
            )
        )

    n = len(results)
    report = EvalReport(
        cases=n,
        hit_at_k=sum(r.hit for r in results) / n,
        recall_at_k=sum(r.hit for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        k=k,
        answers_evaluated=include_answers,
        answer_accuracy=(sum(r.answer_correct for r in results) / n) if include_answers else None,
        refusal_rate=(sum(not r.grounded for r in results) / n) if include_answers else None,
        mean_latency_ms=round(statistics.fmean(latencies), 1) if latencies else None,
        results=results,
    )

    if include_ragas:
        report.ragas = _ragas_scores(service, results)

    return report


def _ragas_scores(service: Any, results: Iterable[CaseResult]) -> dict[str, float] | None:
    """Optional RAGAS pass. Returns None when the extra is not installed."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        logger.warning("RAGAS not installed - skipping judge-based metrics (pip install '.[eval]')")
        return None

    rows = []
    for r in results:
        docs = service.retrieve(r.question)
        rows.append(
            {
                "question": r.question,
                "answer": r.answer,
                "contexts": [d.page_content for d in docs],
                "ground_truth": r.expected,
            }
        )
    result = evaluate(
        Dataset.from_list(rows), metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return {
        key: float(value)
        for key, value in dict(result).items()
        if isinstance(value, (int, float))
    }


def render_report(report: EvalReport) -> str:
    lines = [
        "",
        "=" * 62,
        f"  Hindi RAG evaluation  (n={report.cases}, k={report.k})",
        "=" * 62,
        f"  hit@{report.k}            {report.hit_at_k:.3f}",
        f"  recall@{report.k}         {report.recall_at_k:.3f}",
        f"  MRR                  {report.mrr:.3f}",
    ]
    if report.answers_evaluated:
        lines += [
            f"  answer accuracy      {report.answer_accuracy:.3f}",
            f"  refusal rate         {report.refusal_rate:.3f}",
            f"  mean latency         {report.mean_latency_ms:.0f} ms",
        ]
    else:
        lines.append("  (answers not evaluated - retrieval-only mode)")
    if report.ragas:
        lines.append("  " + "-" * 58)
        for key, value in report.ragas.items():
            lines.append(f"  {key:<20} {value:.3f}")
    lines.append("=" * 62)
    return "\n".join(lines)


def write_report(report: EvalReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p
