"""
scraper.py
----------
Fetch and parse article text from URLs for the expert consensus pipeline.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

from world_cup_2026.config import DATA_DIR

RAW_DIR = DATA_DIR / "expert_opinions" / "raw"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_MIN_WORDS = 200


class PaywallError(Exception):
    """Raised when extracted text is too short — likely paywalled or empty."""


def scrape_url(url: str) -> dict:
    """Fetch article text from a URL and save raw text to disk.

    Returns:
        dict with keys: url, source_domain, title, date, text, scraped_at

    Raises:
        PaywallError: if extracted text has fewer than 200 words.
        requests.RequestException: on network or HTTP errors.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    scraped_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"Scraping: {url}")
    response = requests.get(url, headers=_HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    date = _extract_date(soup)

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    word_count = len(text.split())
    if word_count < _MIN_WORDS:
        raise PaywallError(
            f"Text too short ({word_count} words) — likely paywalled: {url}"
        )

    parsed = urlparse(url)
    source_domain = parsed.netloc.removeprefix("www.")

    safe_domain = re.sub(r"[^\w.-]", "_", source_domain)
    raw_path = RAW_DIR / f"{safe_domain}_{date}.txt"
    raw_path.write_text(f"{title}\n\n{text}", encoding="utf-8")
    logger.info(f"Saved raw text → {raw_path}")

    return {
        "url": url,
        "source_domain": source_domain,
        "title": title,
        "date": date,
        "text": text,
        "scraped_at": scraped_at,
    }


def _extract_date(soup: BeautifulSoup) -> str:
    """Extract publication date from meta tags; falls back to today (UTC)."""
    for attr, prop in [
        ("property", "article:published_time"),
        ("name", "date"),
        ("itemprop", "datePublished"),
    ]:
        tag = soup.find("meta", attrs={attr: prop})
        if tag and tag.get("content"):
            try:
                return datetime.fromisoformat(tag["content"][:10]).date().isoformat()
            except ValueError:
                pass
    return datetime.now(timezone.utc).date().isoformat()
