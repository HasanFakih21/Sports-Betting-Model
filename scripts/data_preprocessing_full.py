"""
data_preprocessing_full.py

Reads raw season CSV files from data/raw/, merges and cleans them, resolves team
name mismatches via a canonical mapping, merges FIFA team ratings and Understat
xG stats, then writes the cleaned combined dataset to
data/processed/combined_dataset_clean.csv.

Usage (run from the project root):
    python scripts/data_preprocessing_full.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = _PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Team-name canonicalisation map (raw name → canonical name)
# ---------------------------------------------------------------------------
TEAM_NAME_MAPPING: dict[str, str] = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "brighton": "brighton & hove albion",
    "nott'm forest": "nottingham forest",
    "leicester": "leicester city",
    "leeds": "leeds united",
    "wolves": "wolverhampton wanderers",
    "bournemouth": "afc bournemouth",
    "ipswich": "ipswich town",
    "west ham": "west ham united",
    "norwich": "norwich city",
    "luton": "luton town",
    "fulham": "fulham fc",
    "newcastle": "newcastle united",
}

# Columns to retain from the raw season match files
_MATCH_KEEP_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA"]

# Final column order for the processed output
_FINAL_COLS = [
    "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA",
    "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield", "HomeRating_Defence",
    "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
    "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield", "AwayRating_Defence",
    "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
    "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L",
    "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS",
    "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
    "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L",
    "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS",
    "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def merge_season_files() -> pd.DataFrame:
    """Load and concatenate all available season CSV files from data/raw/."""
    season_files = ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv", "2024-2025.csv"]
    df_list = []
    for filename in season_files:
        filepath = RAW_DATA_PATH / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found in {RAW_DATA_PATH}, skipping.")
            continue
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Season"] = filepath.stem  # e.g. "2021-2022"
        cols_to_use = [c for c in _MATCH_KEEP_COLS if c in df.columns]
        df_list.append(df[cols_to_use + ["Season"]])
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def load_team_ratings() -> pd.DataFrame:
    """Load FIFA team ratings from data/raw/all_team_ratings.csv."""
    filepath = RAW_DATA_PATH / "all_team_ratings.csv"
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    if "Season" in df.columns:
        df["Season"] = df["Season"].str.replace("/", "-", regex=False)
    keep = ["Season", "Team", "Overall", "Attack", "Midfield", "Defence", "Players", "StartingXI_AvgAge"]
    available = [c for c in keep if c in df.columns]
    return df[available]


def load_understat_stats() -> pd.DataFrame:
    """Load Understat xG stats from data/raw/understat_team_stats.csv.

    The xG/xGA/xPTS columns may contain values like ``"63.67+3.67"``; only the
    base figure before the sign is kept.
    """
    filepath = RAW_DATA_PATH / "understat_team_stats.csv"
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    if "Season" in df.columns:
        df["Season"] = df["Season"].str.replace("/", "-", regex=False)
    keep = ["Season", "Team", "M", "W", "D", "L", "G", "GA", "PTS", "xG", "xGA", "xPTS"]
    available = [c for c in keep if c in df.columns]
    df = df[available]
    for col in ["xG", "xGA", "xPTS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.extract(r"^([0-9.]+)", expand=False),
                errors="coerce",
            )
    return df


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def normalise_team_names(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Lowercase and strip whitespace from team-name columns (in-place copy)."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def apply_team_mapping(df: pd.DataFrame, mapping: dict[str, str], column: str) -> pd.DataFrame:
    """Replace team names in *column* using *mapping*."""
    df = df.copy()
    df[column] = df[column].replace(mapping)
    return df


def merge_team_ratings(
    match_df: pd.DataFrame, ratings_df: pd.DataFrame, team_side: str, prefix: str
) -> pd.DataFrame:
    """Left-join *ratings_df* onto *match_df* for the given team column."""
    subset = ratings_df.copy().rename(columns={"Team": team_side})
    rename_map = {
        "Overall": f"{prefix}_Overall",
        "Attack": f"{prefix}_Attack",
        "Midfield": f"{prefix}_Midfield",
        "Defence": f"{prefix}_Defence",
        "Players": f"{prefix}_Players",
        "StartingXI_AvgAge": f"{prefix}_StartingXI_AvgAge",
    }
    subset = subset.rename(columns={k: v for k, v in rename_map.items() if k in subset.columns})
    return match_df.merge(subset, on=["Season", team_side], how="left")


def merge_understat(
    match_df: pd.DataFrame, understat_df: pd.DataFrame, team_side: str, prefix: str
) -> pd.DataFrame:
    """Left-join *understat_df* onto *match_df* for the given team column."""
    subset = understat_df.copy().rename(columns={"Team": team_side})
    rename_map = {
        col: f"{prefix}_{col}"
        for col in subset.columns
        if col not in ("Season", team_side)
    }
    subset = subset.rename(columns=rename_map)
    return match_df.merge(subset, on=["Season", team_side], how="left")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    match_df = merge_season_files()
    if match_df.empty:
        print("No match data loaded. Aborting.")
        return

    ratings_df = load_team_ratings()
    understat_df = load_understat_stats()

    # Normalise team names to lowercase canonical forms
    match_df = normalise_team_names(match_df, ["HomeTeam", "AwayTeam"])
    if not ratings_df.empty:
        ratings_df = normalise_team_names(ratings_df, ["Team"])
    if not understat_df.empty:
        understat_df = normalise_team_names(understat_df, ["Team"])

    # Apply canonical team-name mapping to all tables
    for col in ("HomeTeam", "AwayTeam"):
        match_df = apply_team_mapping(match_df, TEAM_NAME_MAPPING, col)
    if not ratings_df.empty:
        ratings_df = apply_team_mapping(ratings_df, TEAM_NAME_MAPPING, "Team")
    if not understat_df.empty:
        understat_df = apply_team_mapping(understat_df, TEAM_NAME_MAPPING, "Team")

    # Merge FIFA ratings
    combined = match_df
    if not ratings_df.empty:
        combined = merge_team_ratings(combined, ratings_df, "HomeTeam", "HomeRating")
        combined = merge_team_ratings(combined, ratings_df, "AwayTeam", "AwayRating")

    # Merge Understat xG stats
    if not understat_df.empty:
        combined = merge_understat(combined, understat_df, "HomeTeam", "HomeUnderstat")
        combined = merge_understat(combined, understat_df, "AwayTeam", "AwayUnderstat")

    # Keep only the expected final columns (skip any that are absent)
    available_final_cols = [c for c in _FINAL_COLS if c in combined.columns]
    final_df = combined[available_final_cols].copy()

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / "combined_dataset_clean.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Clean combined dataset saved to {output_path}  ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
