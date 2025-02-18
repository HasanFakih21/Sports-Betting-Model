import os
import pandas as pd

def load_match_data():
    # Load season match files and keep only essential columns
    files = ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv", "2024-2025.csv"]
    keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA"]
    df_list = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = [col.strip() for col in df.columns]
            # Convert Date to datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            # Use file name as Season (e.g. "2021-2022")
            season = file.split(".")[0]
            df["Season"] = season
            # Keep only the needed columns (if available)
            cols_to_use = [c for c in keep_cols if c in df.columns]
            df = df[cols_to_use + ["Season"]]
            df_list.append(df)
        else:
            print(f"File {file} not found.")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_team_ratings():
    # Load team ratings and convert Season format from "YYYY/YYYY" to "YYYY-YYYY"
    ratings_file = "all_team_ratings.csv"
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

def load_understat_stats():
    # Load understat team stats and convert Season format; also convert xG, xGA, xPTS to numeric
    understat_file = "understat_team_stats.csv"
    if os.path.exists(understat_file):
        df = pd.read_csv(understat_file)
        df.columns = [col.strip() for col in df.columns]
        if "Season" in df.columns:
            df["Season"] = df["Season"].str.replace("/", "-")
        keep = ["Season", "Team", "M", "W", "D", "L", "G", "GA", "PTS", "xG", "xGA", "xPTS"]
        df = df[keep]
        # For columns xG, xGA, xPTS, extract the first numeric part (before any '+' or '-')
        for col in ["xG", "xGA", "xPTS"]:
            df[col] = df[col].astype(str).str.split(r'[+-]').str[0]
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        print(f"File {understat_file} not found.")
        return pd.DataFrame()

def preprocess_team_names(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df

def apply_mapping(df, mapping, column):
    df[column] = df[column].replace(mapping)
    return df

def merge_team_data(match_df, team_df, team_side, prefix):
    # Merge team ratings for a given side (HomeTeam or AwayTeam)
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

def merge_understat_data(match_df, understat_df, team_side, prefix):
    # Merge understat stats for a given side
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
    # Load datasets
    match_df = load_match_data()
    ratings_df = load_team_ratings()
    understat_df = load_understat_stats()
    
    if match_df.empty:
        print("No match data loaded.")
        return

    # Preprocess team names in all datasets
    match_df = preprocess_team_names(match_df, ["HomeTeam", "AwayTeam"])
    ratings_df = preprocess_team_names(ratings_df, ["Team"])
    understat_df = preprocess_team_names(understat_df, ["Team"])

    # Standardize team names using a mapping dictionary
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

    # Merge team ratings for home and away teams
    combined = merge_team_data(match_df, ratings_df, "HomeTeam", "HomeRating")
    combined = merge_team_data(combined, ratings_df, "AwayTeam", "AwayRating")
    
    # Merge understat data for home and away teams
    combined = merge_understat_data(combined, understat_df, "HomeTeam", "HomeUnderstat")
    combined = merge_understat_data(combined, understat_df, "AwayTeam", "AwayUnderstat")
    
    # Select final columns needed for modeling
    final_cols = [
        "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA",
        "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield", "HomeRating_Defence", "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
        "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield", "AwayRating_Defence", "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
        "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L", "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS", "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
        "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L", "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS", "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS"
    ]
    final_df = combined[final_cols]
    final_df.to_csv("combined_dataset_clean.csv", index=False)
    print("Clean combined dataset saved to combined_dataset_clean.csv")

if __name__ == "__main__":
    main()
