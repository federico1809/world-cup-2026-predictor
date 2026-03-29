"""
form.py
-------
Recent form features for international football teams.

Computes rolling performance metrics for each team over configurable
match windows (default: 5, 10, 20 matches), strictly before each
match date to avoid data leakage.

Features computed per window:
    - win/draw/loss rates
    - goals scored/conceded averages
    - goal difference average
    - clean sheet rate
    - failed to score rate
    - form points (3/1/0 weighted average)
    - weighted form (exponential decay — most recent matches weighted more)

Usage:
    from world_cup_2026.features.form import FormCalculator
    calc = FormCalculator(df_results_norm, windows=[5, 10, 20])
    df_with_form = calc.compute_form_features(df_results_norm)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOWS: list[int] = [5, 10, 20]
FORM_DECAY_LAMBDA: float = 0.1   # decay per match (not per month)
OFFICIAL_TOURNAMENTS: set[str] = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Africa Cup of Nations",
    "Africa Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "FIFA Confederations Cup",
}


# ---------------------------------------------------------------------------
# FormCalculator
# ---------------------------------------------------------------------------

class FormCalculator:
    """Compute recent form features for all teams over all matches.

    Args:
        df: Normalized match results DataFrame (output of normalize pipeline).
        windows: List of match window sizes to compute form over.
        decay_lambda: Exponential decay rate per match for weighted form.
        official_only_windows: If True, also compute form on official
                               matches only (excludes friendlies).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        windows: list[int] = DEFAULT_WINDOWS,
        decay_lambda: float = FORM_DECAY_LAMBDA,
        official_only_windows: bool = True,
    ) -> None:
        self.df = df.copy().sort_values("date").reset_index(drop=True)
        self.windows = windows
        self.decay_lambda = decay_lambda
        self.official_only = official_only_windows
        logger.info(
            f"FormCalculator initialized — {len(self.df):,} matches, "
            f"windows={windows}, decay_lambda={decay_lambda}"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_team_match_history(self) -> dict[str, list[dict]]:
        """Build per-team chronological match history.

        Returns:
            Dict mapping team_name → list of match dicts sorted by date.
            Each dict contains: date, goals_for, goals_against,
            result (W/D/L), points, is_official.
        """
        history: dict[str, list[dict]] = {}

        for _, row in self.df.iterrows():
            home, away = row["home_team"], row["away_team"]
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            date = row["date"]
            is_official = row["tournament"] in OFFICIAL_TOURNAMENTS

            # Home team entry
            if hs > as_:
                h_result, h_pts, a_result, a_pts = "W", 3, "L", 0
            elif hs < as_:
                h_result, h_pts, a_result, a_pts = "L", 0, "W", 3
            else:
                h_result, h_pts, a_result, a_pts = "D", 1, "D", 1

            for team, gf, ga, result, pts in [
                (home, hs, as_, h_result, h_pts),
                (away, as_, hs, a_result, a_pts),
            ]:
                if team not in history:
                    history[team] = []
                history[team].append({
                    "date": date,
                    "goals_for": gf,
                    "goals_against": ga,
                    "result": result,
                    "points": pts,
                    "is_official": is_official,
                    "clean_sheet": int(ga == 0),
                    "failed_to_score": int(gf == 0),
                })

        # Sort each team's history by date
        for team in history:
            history[team].sort(key=lambda x: x["date"])

        logger.debug(f"Built match history for {len(history)} teams.")
        return history

    def _compute_window_stats(
        self,
        matches: list[dict],
        window: int,
    ) -> dict:
        """Compute form stats over the last N matches.

        Args:
            matches: List of match dicts (already filtered to before match date).
            window: Number of most recent matches to use.

        Returns:
            Dict of form features for this window.
        """
        recent = matches[-window:] if len(matches) >= window else matches
        n = len(recent)

        if n == 0:
            return self._neutral_window_stats(window)

        goals_for     = [m["goals_for"]        for m in recent]
        goals_against = [m["goals_against"]     for m in recent]
        points        = [m["points"]            for m in recent]
        results       = [m["result"]            for m in recent]
        clean_sheets  = [m["clean_sheet"]       for m in recent]
        failed        = [m["failed_to_score"]   for m in recent]

        # Exponential decay weights — most recent match has highest weight
        weights = np.array([
            np.exp(-self.decay_lambda * (n - 1 - i))
            for i in range(n)
        ])
        weights_norm = weights / weights.sum()

        win_rate   = sum(r == "W" for r in results) / n
        draw_rate  = sum(r == "D" for r in results) / n
        loss_rate  = sum(r == "L" for r in results) / n

        return {
            f"form_{window}_matches_played":    n,
            f"form_{window}_win_rate":          win_rate,
            f"form_{window}_draw_rate":         draw_rate,
            f"form_{window}_loss_rate":         loss_rate,
            f"form_{window}_goals_scored_avg":  float(np.mean(goals_for)),
            f"form_{window}_goals_conceded_avg":float(np.mean(goals_against)),
            f"form_{window}_goal_diff_avg":     float(np.mean(
                [gf - ga for gf, ga in zip(goals_for, goals_against)]
            )),
            f"form_{window}_clean_sheet_rate":  float(np.mean(clean_sheets)),
            f"form_{window}_failed_score_rate": float(np.mean(failed)),
            f"form_{window}_points_avg":        float(np.mean(points)),
            f"form_{window}_weighted_points":   float(np.dot(points, weights_norm)),
        }

    def _neutral_window_stats(self, window: int) -> dict:
        """Return neutral (no history) form stats for a window."""
        return {
            f"form_{window}_matches_played":    0,
            f"form_{window}_win_rate":          0.333,
            f"form_{window}_draw_rate":         0.333,
            f"form_{window}_loss_rate":         0.333,
            f"form_{window}_goals_scored_avg":  1.0,
            f"form_{window}_goals_conceded_avg":1.0,
            f"form_{window}_goal_diff_avg":     0.0,
            f"form_{window}_clean_sheet_rate":  0.2,
            f"form_{window}_failed_score_rate": 0.2,
            f"form_{window}_points_avg":        1.0,
            f"form_{window}_weighted_points":   1.0,
        }

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------

    def compute_form_features(
        self,
        df: pd.DataFrame,
        official_only: bool = False,
    ) -> pd.DataFrame:
        """Add form features to a match DataFrame.

        For each match row, computes form features for both home and
        away teams based on their history strictly before that match.

        Args:
            df: Normalized match DataFrame sorted by date.
            official_only: If True, compute form only from official matches.

        Returns:
            Copy of df with form feature columns added for both teams.
        """
        suffix = "_off" if official_only else ""
        df_out = df.copy().sort_values("date").reset_index(drop=True)

        # Filter source matches if official only
        source_df = self.df.copy()
        if official_only:
            source_df = source_df[
                source_df["tournament"].isin(OFFICIAL_TOURNAMENTS)
            ].copy()

        history = self._build_team_match_history_from(source_df)

        logger.info(
            f"Computing form features ({'official only' if official_only else 'all matches'}) "
            f"for {len(df_out):,} rows..."
        )

        # For each match, get index in team history at that date
        # We use a pointer approach for O(n) instead of O(n²)
        team_pointers: dict[str, int] = {}

        # Pre-sort and index history by team
        home_form_rows = []
        away_form_rows = []

        for _, row in df_out.iterrows():
            match_date = row["date"]
            home = row["home_team"]
            away = row["away_team"]

            # Get history up to (not including) this match
            home_hist = [
                m for m in history.get(home, [])
                if m["date"] < match_date
            ]
            away_hist = [
                m for m in history.get(away, [])
                if m["date"] < match_date
            ]

            home_row, away_row = {}, {}
            for w in self.windows:
                home_stats = self._compute_window_stats(home_hist, w)
                away_stats = self._compute_window_stats(away_hist, w)

                # Add home/away prefix
                home_row.update({
                    f"home_{k}{suffix}": v
                    for k, v in home_stats.items()
                })
                away_row.update({
                    f"away_{k}{suffix}": v
                    for k, v in away_stats.items()
                })

            home_form_rows.append(home_row)
            away_form_rows.append(away_row)

        # Combine into DataFrame
        df_home_form = pd.DataFrame(home_form_rows)
        df_away_form = pd.DataFrame(away_form_rows)

        df_out = pd.concat(
            [df_out.reset_index(drop=True),
             df_home_form.reset_index(drop=True),
             df_away_form.reset_index(drop=True)],
            axis=1,
        )

        logger.success(
            f"Form features computed. Added {len(df_home_form.columns) + len(df_away_form.columns)} columns."
        )
        return df_out

    def _build_team_match_history_from(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[dict]]:
        """Build team history from a specific DataFrame (all or official only)."""
        history: dict[str, list[dict]] = {}

        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            date = row["date"]
            is_official = row["tournament"] in OFFICIAL_TOURNAMENTS

            if hs > as_:
                h_result, h_pts, a_result, a_pts = "W", 3, "L", 0
            elif hs < as_:
                h_result, h_pts, a_result, a_pts = "L", 0, "W", 3
            else:
                h_result, h_pts, a_result, a_pts = "D", 1, "D", 1

            for team, gf, ga, result, pts in [
                (home, hs, as_, h_result, h_pts),
                (away, as_, hs, a_result, a_pts),
            ]:
                if team not in history:
                    history[team] = []
                history[team].append({
                    "date": date,
                    "goals_for": gf,
                    "goals_against": ga,
                    "result": result,
                    "points": pts,
                    "is_official": is_official,
                    "clean_sheet": int(ga == 0),
                    "failed_to_score": int(gf == 0),
                })

        for team in history:
            history[team].sort(key=lambda x: x["date"])

        return history

    def get_team_current_form(
        self,
        team: str,
        as_of: pd.Timestamp,
        window: int = 10,
        official_only: bool = False,
    ) -> dict:
        """Get current form snapshot for a single team.

        Args:
            team: Team name (normalized).
            as_of: Reference date — only matches before this date.
            window: Number of recent matches.
            official_only: Use only official matches.

        Returns:
            Form stats dict for this team.
        """
        source = self.df
        if official_only:
            source = source[source["tournament"].isin(OFFICIAL_TOURNAMENTS)]

        history = self._build_team_match_history_from(source)
        team_hist = [
            m for m in history.get(team, [])
            if m["date"] < as_of
        ]
        return self._compute_window_stats(team_hist, window)