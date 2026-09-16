"""Interactive CLI for the support agent pipeline: `python -m src.pipeline`"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PIPELINE_PATH = ROOT / "data" / "processed" / "pipeline.pkl"


def main() -> None:
    if not PIPELINE_PATH.exists():
        print("No fitted pipeline found. Run:\n"
              "  python scripts/download_data.py\n"
              "  python scripts/build_threads.py\n"
              "  python scripts/preprocess_data.py\n"
              "  python scripts/build_index.py\n"
              "then retry.")
        sys.exit(1)

    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    print("AI Support Agent -- interactive mode. Type a customer message (or 'quit').\n")
    while True:
        try:
            msg = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg or msg.lower() in {"quit", "exit"}:
            break

        result = pipeline.process(msg)
        print(f"\nIntent:\n{result.intent}\n")
        print(f"Confidence:\n{result.intent_confidence:.2f}\n")
        print(f"Decision:\n{result.escalation_decision}\n")
        print(f"Reason:\n{result.escalation_reason}\n")
        print(f"Draft response:\n{result.reply}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
