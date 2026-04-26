"""
Run RAGAS evaluation and print a score table.
Usage: python scripts/evaluate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval import run_evaluation

if __name__ == "__main__":
    print("Running RAGAS evaluation...\n")
    results = run_evaluation()
    print("\n=== Evaluation Results ===")
    for metric, score in results.items():
        bar = "█" * int(score * 20)
        print(f"{metric:<25} {score:.3f}  {bar}")
