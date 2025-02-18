"""
data_preprocessing.py

This script reads raw season CSV files from data/raw/, merges and cleans them,
resolves any team name mismatches (using a mapping), merges team ratings and Understat stats,
and finally writes the cleaned combined dataset to data/processed/combined_dataset_clean.csv.
"""

import os
import pandas as pd

# Define paths for raw and processed data
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def load_season_csv(filename: str) -> pd.DataFrame:
    """
    Load a season CSV file from data/raw/ and do minimal processing.
    """
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    # Convert Date column to datetime (if present)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    # Use the filename (without extension) as Season (e.g. "2021-2022")
    season = os.path.splitext(filename)[0]
    df["Season"] = season
    return df

def merge_season_files() -> pd.DataFrame:
    """
    Load and merge multiple season CSV files.
    """
    # List of season files to load
    season_files = ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv", "2024-2025.csv"]
    keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA"]
    df_list = []
    for file in season_files:
        filepath = os.path.join(RAW_DATA_PATH, file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df.columns = [col.strip() for col in df.columns]
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            season = os.path.splitext(file)[0]
            df["Season"] = season
            # Keep only columns that are available plus Season
            cols_to_use = [c for c in keep_cols if c in df.columns]
            df = df[cols_to_use + ["Season"]]
            df_list.append(df)
        else:
            print(f"File {file} not found in {RAW_DATA_PATH}.")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def load_team_ratings() -> pd.DataFrame:
    """
    Load team ratings CSV from data/raw/ and convert Season format.
    """
    ratings_file = os.path.join(RAW_DATA_PATH, "all_team_ratings.csv")
    if os.path.exists(ratings_file):
        df = pd.read_csv(ratings_file)
        df.columns = [col.strip() for col in df.columns]
        if "Season" in df.columns:
            df["Season"] = df["Season"].str.replace("/", "-")
        keep = ["Season", "Team", "Overall", "Attack", "Midfield", "Defence", "Players", "StartingXI_AvgAge"]
        return df[keep]
    else:
        print(f"File {ratings_file} not found.")
        return pd.DataFrame()

def load_understat_stats() -> pd.DataFrame:
    """
    Load Understat stats CSV from data/raw/ and convert Season format; also process xG, xGA, xPTS.
    """
    understat_file = os.path.join(RAW_DATA_PATH, "understat_team_stats.csv")
    if os.path.exists(understat_file):
        df = pd.read_csv(understat_file)
        df.columns = [col.strip() for col in df.columns]
        if "Season" in df.columns:
            df["Season"] = df["Season"].str.replace("/", "-")
        keep = ["Season", "Team", "M", "W", "D", "L", "G", "GA", "PTS", "xG", "xGA", "xPTS"]
        df = df[keep]
        for col in ["xG", "xGA", "xPTS"]:
            df[col] = df[col].astype(str).str.split(r'[+-]').str[0]
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        print(f"File {understat_file} not found.")
        return pd.DataFrame()

def preprocess_team_names(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Lowercase and strip whitespace from team names in specified columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df

def apply_mapping(df: pd.DataFrame, mapping: dict, column: str) -> pd.DataFrame:
    """
    Apply a mapping dictionary to a specific column.
    """
    df[column] = df[column].replace(mapping)
    return df

def merge_team_data(match_df: pd.DataFrame, team_df: pd.DataFrame, team_side: str, prefix: str) -> pd.DataFrame:
    """
    Merge team ratings data with the match data for a given team side (HomeTeam or AwayTeam).
    """
    subset = team_df.copy()
    subset.rename(columns={'Team': team_side}, inplace=True)
    rename_map = {
        "Overall": f"{prefix}_Overall",
        "Attack": f"{prefix}_Attack",
        "Midfield": f"{prefix}_Midfield",
        "Defence": f"{prefix}_Defence",
        "Players": f"{prefix}_Players",
        "StartingXI_AvgAge": f"{prefix}_StartingXI_AvgAge"
    }
    subset.rename(columns=rename_map, inplace=True)
    merged = match_df.merge(subset, on=["Season", team_side], how="left")
    return merged

def merge_understat_data(match_df: pd.DataFrame, understat_df: pd.DataFrame, team_side: str, prefix: str) -> pd.DataFrame:
    """
    Merge Understat stats with the match data for a given team side.
    """
    subset = understat_df.copy()
    subset.rename(columns={'Team': team_side}, inplace=True)
    rename_map = {}
    for col in subset.columns:
        if col not in ["Season", team_side]:
            rename_map[col] = f"{prefix}_{col}"
    subset.rename(columns=rename_map, inplace=True)
    merged = match_df.merge(subset, on=["Season", team_side], how="left")
    return merged

def main():
    # Load season match data
    match_df = merge_season_files()
    ratings_df = load_team_ratings()
    understat_df = load_understat_stats()
    
    if match_df.empty:
        print("No match data loaded.")
        return

    # Preprocess team names
    match_df = preprocess_team_names(match_df, ["HomeTeam", "AwayTeam"])
    ratings_df = preprocess_team_names(ratings_df, ["Team"])
    understat_df = preprocess_team_names(understat_df, ["Team"])
    
    # Mapping dictionary to standardize team names
    mapping = {
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
        "newcastle": "newcastle united"
    }
    match_df = apply_mapping(match_df, mapping, "HomeTeam")
    match_df = apply_mapping(match_df, mapping, "AwayTeam")
    ratings_df = apply_mapping(ratings_df, mapping, "Team")
    understat_df = apply_mapping(understat_df, mapping, "Team")
    
    # Merge team ratings
    combined = merge_team_data(match_df, ratings_df, "HomeTeam", "HomeRating")
    combined = merge_team_data(combined, ratings_df, "AwayTeam", "AwayRating")
    
    # Merge Understat stats
    combined = merge_understat_data(combined, understat_df, "HomeTeam", "HomeUnderstat")
    combined = merge_understat_data(combined, understat_df, "AwayTeam", "AwayUnderstat")
    
    # Select final columns to keep
    final_cols = [
        "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA",
        "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield", "HomeRating_Defence", "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
        "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield", "AwayRating_Defence", "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
        "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L", "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS", "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
        "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L", "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS", "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS"
    ]
    final_df = combined[final_cols]
    
    # Ensure the processed folder exists
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, "combined_dataset_clean.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Clean combined dataset saved to {output_path}")

if __name__ == "__main__":
    main()
