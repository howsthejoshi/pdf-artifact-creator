"""Build analysis panel, facts, model, and anomaly list from data/raw."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics import run as run_analytics  # noqa: E402
from src.anomaly import run as run_anomaly  # noqa: E402
from src.ingestion import build_analysis_set  # noqa: E402
from src.model import train  # noqa: E402
from src.docs import write_pdfs  # noqa: E402


def main() -> None:
    panel = build_analysis_set()
    facts = run_analytics(panel)
    metrics = train(panel)
    from src.paths import PROCESSED_DIR
    import pandas as pd

    panel = pd.read_parquet(PROCESSED_DIR / "analysis_panel.parquet")
    flagged = run_anomaly(panel)
    write_pdfs(facts, metrics, len(flagged))
    print("accounts", len(panel), "default_rate", round(float(panel["target_default"].mean()), 4))
    print("metrics", metrics["logistic"], metrics["lightgbm"])
    print("anomalies", len(flagged))


if __name__ == "__main__":
    main()
