"""
pipeline.py
-----------
Orchestrate scraping + prediction extraction and append to expert_predictions.csv.

Usage:
    python -m world_cup_2026.expert_consensus.pipeline
    python -m world_cup_2026.expert_consensus.pipeline --urls-file path/to/urls.txt
    python -m world_cup_2026.expert_consensus.pipeline --output path/to/out.csv
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import typer
from loguru import logger

from world_cup_2026.config import DATA_DIR
from world_cup_2026.expert_consensus.extractor import extract_predictions
from world_cup_2026.expert_consensus.scraper import PaywallError, scrape_url

app = typer.Typer(add_completion=False)

_DEFAULT_URLS = DATA_DIR / "expert_opinions" / "urls.txt"
_DEFAULT_OUTPUT = DATA_DIR / "expert_opinions" / "processed" / "expert_predictions.csv"

_CSV_COLUMNS = [
    "team", "prediction_type", "confidence", "sentiment", "quote",
    "source", "date", "url", "extracted_at",
]


def _load_processed_urls(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["url"])
    return set(df["url"].dropna().unique())


@app.command()
def main(
    urls_file: Path = typer.Option(_DEFAULT_URLS, "--urls-file", help="Text file of URLs to process."),
    output: Path = typer.Option(_DEFAULT_OUTPUT, "--output", help="Output CSV path."),
) -> None:
    if not urls_file.exists():
        logger.error(f"URLs file not found: {urls_file}")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    processed_urls = _load_processed_urls(output)
    logger.info(f"Already processed: {len(processed_urls)} URLs")

    raw_lines = urls_file.read_text(encoding="utf-8").splitlines()
    urls = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]
    logger.info(f"Total URLs in file: {len(urls)}")

    new_rows: list[dict] = []

    for url in urls:
        if url in processed_urls:
            logger.debug(f"Skipping already-processed: {url}")
            continue

        try:
            article = scrape_url(url)
        except PaywallError as e:
            logger.warning(str(e))
            continue
        except requests.RequestException as e:
            logger.warning(f"Network error for {url}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected scrape error for {url}: {e}")
            continue

        predictions = extract_predictions(article)
        if not predictions:
            logger.info(f"No predictions extracted from {url}")
            continue

        extracted_at = datetime.now(timezone.utc).isoformat()
        for pred in predictions:
            pred["extracted_at"] = extracted_at
            new_rows.append({col: pred.get(col) for col in _CSV_COLUMNS})

    if not new_rows:
        logger.info("No new predictions to save.")
        return

    df_new = pd.DataFrame(new_rows, columns=_CSV_COLUMNS)
    write_header = not output.exists() or output.stat().st_size == 0
    df_new.to_csv(output, mode="a", header=write_header, index=False)
    logger.success(f"Saved {len(df_new)} new predictions → {output}")


if __name__ == "__main__":
    app()
