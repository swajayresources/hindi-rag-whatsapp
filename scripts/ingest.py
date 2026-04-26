"""
One-time data ingestion script.
Downloads Hindi Wikipedia articles, saves to data/corpus/, then embeds into ChromaDB.

Usage:
    python scripts/ingest.py                          # default: 50 Hindi Wikipedia articles
    python scripts/ingest.py --articles 200           # more articles
    python scripts/ingest.py --dir ./data/my_corpus   # custom local directory
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import load_and_chunk
from src.retrieval import ingest_chunks


def download_hindi_wikipedia(output_dir: str, n_articles: int = 50):
    try:
        import wikipedia
    except ImportError:
        print("Installing wikipedia library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "wikipedia-api"], check=True)
        import wikipedia

    wikipedia.set_lang("hi")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Broad Hindi topics to cover diverse knowledge
    seed_topics = [
        "भारत", "हिंदी भाषा", "महात्मा गांधी", "भारतीय संविधान",
        "दिल्ली", "मुंबई", "योग", "आयुर्वेद", "भारतीय संगीत",
        "क्रिकेट", "बॉलीवुड", "राजस्थान", "उत्तर प्रदेश",
        "भारतीय इतिहास", "मुगल साम्राज्य", "भारतीय स्वतंत्रता संग्राम",
        "रामायण", "महाभारत", "हिंदू धर्म", "बौद्ध धर्म",
        "भारतीय रेलवे", "भारतीय अर्थव्यवस्था", "कृषि",
        "भारतीय सिनेमा", "शास्त्रीय नृत्य",
    ]

    saved = 0
    for topic in seed_topics:
        if saved >= n_articles:
            break
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            fname = out / f"{topic.replace(' ', '_')}.txt"
            fname.write_text(page.content, encoding="utf-8")
            print(f"  Saved: {topic} ({len(page.content)} chars)")
            saved += 1
        except Exception as e:
            print(f"  Skip {topic}: {e}")

    print(f"\nDownloaded {saved} articles to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./data/corpus", help="Corpus directory")
    parser.add_argument("--articles", type=int, default=50)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just re-embed existing files")
    args = parser.parse_args()

    if not args.skip_download:
        print(f"Downloading {args.articles} Hindi Wikipedia articles...")
        download_hindi_wikipedia(args.dir, args.articles)

    print("\nChunking and embedding...")
    chunks = load_and_chunk(args.dir)
    ingest_chunks(chunks)
    print("\nDone! ChromaDB ready. Run the server with: uvicorn src.main:app --reload")


if __name__ == "__main__":
    main()
