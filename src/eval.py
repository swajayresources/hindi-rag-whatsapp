"""
RAG evaluation using RAGAS framework.
Measures: faithfulness, answer_relevancy, context_recall, context_precision.
Run: python scripts/evaluate.py
"""
from typing import List, Dict
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from .generation import build_rag_chain
from .retrieval import get_retriever


EVAL_QUESTIONS = [
    {
        "question": "भारत की राजधानी क्या है?",
        "ground_truth": "भारत की राजधानी नई दिल्ली है।",
    },
    {
        "question": "हिंदी भाषा किन राज्यों में बोली जाती है?",
        "ground_truth": "हिंदी उत्तर प्रदेश, बिहार, राजस्थान, मध्य प्रदेश, हरियाणा सहित कई राज्यों में बोली जाती है।",
    },
]


def run_evaluation(questions: List[Dict] = None) -> Dict:
    questions = questions or EVAL_QUESTIONS
    retriever = get_retriever()
    chain = build_rag_chain()

    rows = []
    for item in questions:
        q = item["question"]
        retrieved = retriever.invoke(q)
        ans = chain.invoke(q)
        rows.append({
            "question": q,
            "answer": ans,
            "contexts": [d.page_content for d in retrieved],
            "ground_truth": item.get("ground_truth", ""),
        })

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result
