from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

for _p in (PROCESSED_DIR, MODEL_DIR, DOCS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
