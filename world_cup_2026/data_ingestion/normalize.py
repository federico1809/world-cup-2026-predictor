"""
normalize.py
------------
Master dictionary for normalizing team names across all data sources.

The same national team appears with different names depending on the source:
    - Kaggle results: "South Korea"
    - FIFA Rankings:  "Korea Republic"
    - StatsBomb:      "South Korea"
    - Transfermarkt:  "Südkorea"

All names are normalized to a single FIFA official canonical name.

Usage:
    from world_cup_2026.data_ingestion.normalize import normalize_team_name
    df["team"] = df["team"].map(normalize_team_name)
"""

from loguru import logger

# ---------------------------------------------------------------------------
# Master alias dictionary
# key   = any known alias (lowercase for matching)
# value = FIFA canonical name (official English)
# ---------------------------------------------------------------------------

_TEAM_ALIASES: dict[str, str] = {
    # --- Asia ---
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "korea, republic of": "Korea Republic",
    "korea rep.": "Korea Republic",
    "north korea": "Korea DPR",
    "korea dpr": "Korea DPR",
    "korea, dem. people's rep.": "Korea DPR",
    "iran": "IR Iran",
    "ir iran": "IR Iran",
    "islamic republic of iran": "IR Iran",
    "china": "China PR",
    "china pr": "China PR",
    "china, pr": "China PR",
    "chinese taipei": "Chinese Taipei",
    "taiwan": "Chinese Taipei",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "syria": "Syria",
    "kyrgyzstan": "Kyrgyz Republic",
    "kyrgyz republic": "Kyrgyz Republic",

    # --- Europe ---
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "the czech republic": "Czechia",
    "slovak republic": "Slovakia",
    "slovakia": "Slovakia",
    "russia": "Russia",
    "russian federation": "Russia",
    "turkey": "Türkiye",
    "türkiye": "Türkiye",
    "turkiye": "Türkiye",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "republic of north macedonia": "North Macedonia",
    "bosnia and herzegovina": "Bosnia & Herzegovina",
    "bosnia & herzegovina": "Bosnia & Herzegovina",
    "bosnia-herzegovina": "Bosnia & Herzegovina",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "germany": "Germany",
    "west germany": "Germany",

    # --- Americas ---
    "usa": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "us": "United States",
    "trinidad & tobago": "Trinidad and Tobago",
    "trinidad and tobago": "Trinidad and Tobago",
    "saint kitts and nevis": "St. Kitts and Nevis",
    "st. kitts and nevis": "St. Kitts and Nevis",
    "saint vincent and the grenadines": "St. Vincent and the Grenadines",
    "st. vincent and the grenadines": "St. Vincent and the Grenadines",
    "antigua and barbuda": "Antigua and Barbuda",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",

    # --- Africa ---
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "cape verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "democratic republic of the congo": "DR Congo",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "republic of the congo": "Congo",
    "congo": "Congo",
    "tanzania": "Tanzania",
    "united republic of tanzania": "Tanzania",
    "equatorial guinea": "Equatorial Guinea",
    "swaziland": "Eswatini",
    "eswatini": "Eswatini",

    # --- Oceania ---
    "new zealand": "New Zealand",
    "aotearoa new zealand": "New Zealand",
    "tahiti": "Tahiti",
    "papua new guinea": "Papua New Guinea",
}

# Build a reverse lookup: canonical → canonical (for names already correct)
_CANONICAL_NAMES = set(_TEAM_ALIASES.values())


def normalize_team_name(name: str) -> str:
    """Normalize a team name to its FIFA canonical form.

    Args:
        name: Raw team name string from any data source.

    Returns:
        FIFA canonical team name. If not found in the dictionary,
        returns the original name with a warning logged.
    """
    if not isinstance(name, str):
        return name

    key = name.strip().lower()
    normalized = _TEAM_ALIASES.get(key)

    if normalized:
        return normalized

    # If it's already a canonical name (exact match, case-insensitive)
    for canonical in _CANONICAL_NAMES:
        if key == canonical.lower():
            return canonical

    # Not found — return as-is but log for manual review
    logger.warning(f"Team name not in alias dictionary: '{name}' — returning as-is.")
    return name


def normalize_dataframe_teams(
    df,
    columns: list[str],
) -> object:
    """Apply normalize_team_name to one or more columns of a DataFrame.

    Args:
        df: pandas DataFrame containing team name columns.
        columns: List of column names to normalize.

    Returns:
        DataFrame with normalized team name columns (copy, not in-place).
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(normalize_team_name)
            logger.info(f"Normalized column: '{col}'")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame — skipping.")
    return df