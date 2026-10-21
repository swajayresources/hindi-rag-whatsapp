"""Build a verified Hindi QA evaluation set from the corpus.

Every candidate question is kept ONLY if its ground-truth answer string
actually occurs in the cited corpus file. This prevents fabricated eval
data — the failure mode that makes RAG benchmarks meaningless.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "eval" / "questions.jsonl"

# (question, expected_answer_substring, source_file_substring)
CANDIDATES = [
    ("भारत की राजधानी क्या है?", "नई दिल्ली", "भारत"),
    ("भारत का राष्ट्रीय पक्षी कौन सा है?", "मोर", "भारत"),
    ("भारत का राष्ट्रीय पशु कौन सा है?", "बाघ", "भारत"),
    ("भारत को स्वतंत्रता कब मिली?", "1947", "भारत"),
    ("भारत की जनसंख्या लगभग कितनी है?", "121", "भारत"),
    ("भारत में कितने राज्य हैं?", "28", "भारत"),
    ("दिल्ली किस नदी के किनारे बसी है?", "यमुना", "दिल्ली"),
    ("दिल्ली किस देश की राजधानी है?", "भारत", "दिल्ली"),
    ("उत्तर प्रदेश की राजधानी क्या है?", "लखनऊ", "उत्तर"),
    ("उत्तर प्रदेश किस देश में स्थित है?", "भारत", "उत्तर"),
    ("क्रिकेट की एक टीम में कितने खिलाड़ी होते हैं?", "11", "क्रिकेट"),
    ("क्रिकेट में कितने विकेट होते हैं?", "10", "क्रिकेट"),
    ("क्रिकेट का सबसे बड़ा अंतरराष्ट्रीय टूर्नामेंट कौन सा है?", "विश्व कप", "क्रिकेट"),
    ("आयुर्वेद किस प्रकार की चिकित्सा पद्धति है?", "चिकित्सा", "आयुर्वेद"),
    ("आयुर्वेद में कितने दोष माने जाते हैं?", "तीन", "आयुर्वेद"),
    ("बौद्ध धर्म की स्थापना किसने की?", "गौतम बुद्ध", "बौद्ध"),
    ("बौद्ध धर्म में चार आर्य सत्य किसने बताए?", "बुद्ध", "बौद्ध"),
    ("बॉलीवुड किस शहर में स्थित है?", "मुंबई", "बॉलीवुड"),
    ("बॉलीवुड किस भाषा की फिल्मों का उद्योग है?", "हिंदी", "बॉलीवुड"),
    ("भारतीय संगीत के दो प्रमुख रूप कौन से हैं?", "हिंदुस्तानी", "संगीत"),
    ("भारतीय शास्त्रीय संगीत में कितने स्वर होते हैं?", "सात", "संगीत"),
    ("भारतीय इतिहास में मुगल साम्राज्य की स्थापना किसने की?", "बाबर", "इतिहास"),
    ("भारत में मुगल साम्राज्य कब स्थापित हुआ?", "1526", "इतिहास"),
    ("भारत का पहला प्रधानमंत्री कौन था?", "नेहरू", "इतिहास"),
    ("भारत में सिंधु घाटी सभ्यता कहाँ विकसित हुई?", "सिंधु", "इतिहास"),
    ("ताजमहल कहाँ स्थित है?", "आगरा", ""),
    ("हिंदी भारत की कितनी प्रतिशत जनसंख्या बोलती है?", "प्रतिशत", "भारत"),
    ("भारत का साक्षरता दर कितना है?", "74", "भारत"),
    ("भारत का सबसे बड़ा राज्य क्षेत्रफल में कौन सा है?", "राजस्थान", "भारत"),
    ("क्रिकेट का जन्म किस देश में हुआ?", "इंग्लैंड", "क्रिकेट"),
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main() -> None:
    files = {p.name: p.read_text(encoding="utf-8", errors="ignore") for p in CORPUS.glob("*.txt")}
    print(f"corpus files: {len(files)}")

    kept, dropped = [], []
    for question, answer, source_hint in CANDIDATES:
        matches = [
            (name, text) for name, text in files.items()
            if (source_hint in name if source_hint else True)
            and answer in normalise(text)
        ]
        if not matches:
            dropped.append((question, answer, source_hint))
            continue
        name, text = matches[0]
        flat = normalise(text)
        pos = flat.find(answer)
        snippet = flat[max(0, pos - 120): pos + 180].strip()
        kept.append({
            "question": question,
            "ground_truth": answer,
            "source": name,
            "evidence_snippet": snippet,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in kept) + "\n",
        encoding="utf-8",
    )
    print(f"kept={len(kept)} dropped={len(dropped)} -> {OUT}")
    for q, a, s in dropped:
        print(f"  DROPPED: {q} | expected={a} | source~{s}")


if __name__ == "__main__":
    main()
