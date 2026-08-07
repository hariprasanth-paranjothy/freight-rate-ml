"""End-to-end: train best model, write predictions, run Spotter scorer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(module: str) -> None:
    print(f"\n=== Running {module} ===", flush=True)
    subprocess.check_call([sys.executable, "-m", module], cwd=ROOT)


def main() -> None:
    run("src.train")
    run("src.predict")
    cmd = [
        sys.executable,
        "score.py",
        "--predictions",
        "validation_predictions.csv",
        "--december-predictions",
        "december_chart_predictions.csv",
        "--output-dir",
        "scorer_results",
    ]
    print("\n=== Running score.py ===", flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
