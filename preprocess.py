#!/usr/bin/env python3
"""
NBA Game Outcome Prediction – Data Preprocessing Pipeline

Loads three CSVs (games, team stats, advanced team stats), filters to regular
season games from 2000 onward, merges home/away team stats per game, computes
10-game rolling averages shifted by 1 to prevent data leakage, engineers
AST/TOV ratio, eFG% proxy, net scoring margin, and home-minus-away difference
features for all rolling stats, and adds schedule features (rest days,
back-to-back flag, win streak).

Produces 56 input features plus the two target columns (home_win, point_diff).

Usage
-----
    python preprocess.py \\
        --games      data/games.csv \\
        --team_stats data/team_stats.csv \\
        --adv_stats  data/advanced_team_stats.csv \\
        --output     data/processed.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

ROLLING_WINDOW = 10

# Stats computed from the traditional game-log CSV
TRADITIONAL_ROLL = [
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "pts",
    "plus_minus",
]

# Stats computed from the advanced game-log CSV
ADVANCED_ROLL = ["off_rtg", "def_rtg", "ts_pct", "pace"]

# All stats for which rolling averages are computed (14 total)
ROLL_STATS = TRADITIONAL_ROLL + ADVANCED_ROLL


# ── Column-name helpers ────────────────────────────────────────────────────────

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip, and replace special characters in column names."""
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def _resolve_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """Rename columns using {canonical: [candidate, ...]} mapping."""
    for canonical, candidates in aliases.items():
        if canonical not in df.columns:
            for cand in candidates:
                if cand in df.columns:
                    df = df.rename(columns={cand: canonical})
                    break
    return df


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(
    games_path: str,
    team_stats_path: str,
    adv_stats_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and normalise the three source CSVs."""
    games = pd.read_csv(games_path)
    team_stats = pd.read_csv(team_stats_path)
    adv_stats = pd.read_csv(adv_stats_path)

    games = _normalize_cols(games)
    team_stats = _normalize_cols(team_stats)
    adv_stats = _normalize_cols(adv_stats)

    games = _resolve_aliases(
        games,
        {
            "game_id": ["gameid", "id"],
            "game_date": ["date", "game_dt", "matchup_date"],
            "season": ["season_id", "season_year"],
            "season_type": ["game_type", "type", "seasontype"],
            "home_team_id": ["team_id_home"],
            "away_team_id": ["team_id_away"],
            "home_score": ["pts_home", "home_pts", "home_team_score", "homescore"],
            "away_score": ["pts_away", "away_pts", "away_team_score", "awayscore"],
        },
    )
    team_stats = _resolve_aliases(
        team_stats,
        {
            "game_id": ["gameid", "id"],
            "team_id": ["teamid"],
            "fg_pct": ["fgpct", "fgm_pct", "fgpercent"],
            "fg3_pct": ["fg3pct", "fg_3pct", "3ppct", "fg3percent"],
            "ft_pct": ["ftpct", "ftpercent"],
            "plus_minus": ["plusminus", "pm"],
        },
    )
    adv_stats = _resolve_aliases(
        adv_stats,
        {
            "game_id": ["gameid", "id"],
            "team_id": ["teamid"],
            "off_rtg": ["e_off_rating", "off_rating", "ortg", "offrtg"],
            "def_rtg": ["e_def_rating", "def_rating", "drtg", "defrtg"],
            "ts_pct": ["ts", "trueshootingpct", "true_shooting_pct"],
            "efg_pct": ["efg", "efgpct", "effective_fg_pct"],
            "pace": ["pace_per40", "poss"],
        },
    )

    games["game_date"] = pd.to_datetime(games["game_date"])
    return games, team_stats, adv_stats


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_regular_season(games: pd.DataFrame) -> pd.DataFrame:
    """Keep regular season games played in the 2000 season or later."""
    if "season_type" in games.columns:
        games = games[
            games["season_type"].str.contains("Regular", case=False, na=False)
        ].copy()

    # Derive the start year of each season
    if "season" in games.columns:
        games["season_year"] = (
            games["season"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
            .astype(int)
        )
    else:
        games["season_year"] = games["game_date"].dt.year

    games = games[games["season_year"] >= 2000].copy()
    return games


# ── Per-team game log ──────────────────────────────────────────────────────────

def _build_team_game_log(
    games: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one row per (team × game) enriched with stats.

    Columns added: game_date, opp_team_id, team_score, opp_score,
                   is_home (bool), win (int 0/1).
    """
    home = games[
        ["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opp_team_id",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    ).copy()
    home["is_home"] = True

    away = games[
        ["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opp_team_id",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    ).copy()
    away["is_home"] = False

    game_info = pd.concat([home, away], ignore_index=True)
    game_info["win"] = (game_info["team_score"] > game_info["opp_score"]).astype(int)

    team_log = game_info.merge(stats, on=["game_id", "team_id"], how="inner")
    team_log = team_log.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return team_log


# ── Rolling averages ───────────────────────────────────────────────────────────

def compute_rolling_averages(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute a ROLLING_WINDOW-game rolling mean shifted by 1
    for every stat in ROLL_STATS that exists in the DataFrame.

    The shift(1) ensures no information from the current game leaks into the
    feature used to predict that game's outcome.
    """
    available = [s for s in ROLL_STATS if s in team_log.columns]
    for stat in available:
        team_log[f"roll_{stat}"] = team_log.groupby("team_id")[stat].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean().shift(1)
        )
    return team_log


# ── Schedule features ──────────────────────────────────────────────────────────

def _win_streak_series(wins: pd.Series) -> pd.Series:
    """
    Return the number of consecutive wins *before* each game.
    The streak resets to 0 after a loss.
    """
    streak: list[int] = []
    current = 0
    for w in wins:
        streak.append(current)
        current = current + 1 if w == 1 else 0
    return pd.Series(streak, index=wins.index)


def compute_schedule_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days, b2b flag, and win_streak columns to the team game log."""
    team_log = team_log.sort_values(["team_id", "game_date"]).copy()

    # Days since last game (NaN for each team's first game → filled with 7)
    team_log["rest_days"] = (
        team_log.groupby("team_id")["game_date"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(7)
    )

    # Back-to-back: played a game the previous calendar day
    team_log["b2b"] = (team_log["rest_days"] <= 1).astype(int)

    # Win streak entering each game
    team_log["win_streak"] = team_log.groupby("team_id")["win"].transform(
        _win_streak_series
    )

    return team_log


# ── Merge rolling features back into the games DataFrame ──────────────────────

def merge_rolling_into_games(
    games: pd.DataFrame,
    team_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot the per-team rolling/schedule features into wide per-game format,
    creating home_<feature> and away_<feature> columns.
    """
    roll_cols = [c for c in team_log.columns if c.startswith("roll_")]
    sched_cols = ["rest_days", "b2b", "win_streak"]
    keep_cols = ["game_id", "team_id"] + roll_cols + sched_cols

    home_feat = (
        team_log[team_log["is_home"]][keep_cols]
        .rename(
            columns={
                "team_id": "home_team_id",
                **{c: f"home_{c}" for c in roll_cols + sched_cols},
            }
        )
    )
    away_feat = (
        team_log[~team_log["is_home"]][keep_cols]
        .rename(
            columns={
                "team_id": "away_team_id",
                **{c: f"away_{c}" for c in roll_cols + sched_cols},
            }
        )
    )

    df = games.merge(home_feat, on=["game_id", "home_team_id"], how="inner")
    df = df.merge(away_feat, on=["game_id", "away_team_id"], how="inner")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features:
      • AST/TOV ratio (home, away)
      • eFG% proxy  (home, away)   = roll_fg_pct + 0.5 × roll_fg3_pct
      • Net scoring margin          = home_roll_pts − away_roll_pts
      • Home-minus-away differences for all rolling stats and composite features
    """
    eps = 1e-6

    # AST/TOV ratio
    if "home_roll_ast" in df.columns and "home_roll_tov" in df.columns:
        df["home_ast_tov_ratio"] = df["home_roll_ast"] / (df["home_roll_tov"] + eps)
        df["away_ast_tov_ratio"] = df["away_roll_ast"] / (df["away_roll_tov"] + eps)

    # eFG% proxy – the true formula is (FGM + 0.5×FG3M) / FGA, which requires
    # raw attempt counts not available in rolling-percentage form.  The proxy
    # FG% + 0.5×FG3% approximates this by treating FG3% as a stand-in for the
    # three-point bonus component; it correlates strongly with true eFG%.
    if "home_roll_fg_pct" in df.columns and "home_roll_fg3_pct" in df.columns:
        df["home_efg_proxy"] = df["home_roll_fg_pct"] + 0.5 * df["home_roll_fg3_pct"]
        df["away_efg_proxy"] = df["away_roll_fg_pct"] + 0.5 * df["away_roll_fg3_pct"]

    # Net scoring margin from each side's perspective (matchup context)
    if "home_roll_pts" in df.columns and "away_roll_pts" in df.columns:
        df["home_net_margin"] = df["home_roll_pts"] - df["away_roll_pts"]
        df["away_net_margin"] = df["away_roll_pts"] - df["home_roll_pts"]

    # Home-minus-away differences for all 14 rolling stats
    stat_names = {c[len("home_roll_"):] for c in df.columns if c.startswith("home_roll_")}
    for stat in sorted(stat_names):
        h, a = f"home_roll_{stat}", f"away_roll_{stat}"
        if h in df.columns and a in df.columns:
            df[f"diff_{stat}"] = df[h] - df[a]

    # Differences for composite features
    if "home_ast_tov_ratio" in df.columns:
        df["diff_ast_tov_ratio"] = df["home_ast_tov_ratio"] - df["away_ast_tov_ratio"]
    if "home_efg_proxy" in df.columns:
        df["diff_efg_proxy"] = df["home_efg_proxy"] - df["away_efg_proxy"]

    return df


# ── Target variables ───────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive home_win (binary) and point_diff (continuous) from the raw scores.
    The raw score columns are retained here so callers can verify correctness;
    train.py explicitly excludes them from the feature matrix.
    """
    if "home_score" in df.columns and "away_score" in df.columns:
        df["point_diff"] = df["home_score"] - df["away_score"]
        df["home_win"] = (df["point_diff"] > 0).astype(int)
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

def preprocess(
    games_path: str,
    team_stats_path: str,
    adv_stats_path: str,
    output_path: str,
) -> pd.DataFrame:
    print("Loading CSVs …")
    games, team_stats, adv_stats = load_data(games_path, team_stats_path, adv_stats_path)

    print("Filtering to regular season 2000+ …")
    games = filter_regular_season(games)
    print(f"  {len(games):,} games retained")

    print("Merging traditional + advanced stats …")
    stats = team_stats.merge(adv_stats, on=["game_id", "team_id"], how="left")

    print("Building per-team game log …")
    team_log = _build_team_game_log(games, stats)

    print(f"Computing {ROLLING_WINDOW}-game rolling averages (shift=1) …")
    team_log = compute_rolling_averages(team_log)

    print("Computing schedule features …")
    team_log = compute_schedule_features(team_log)

    print("Merging features into per-game rows …")
    df = merge_rolling_into_games(games, team_log)

    print("Engineering composite features and difference features …")
    df = engineer_features(df)

    print("Adding target variables …")
    df = add_targets(df)

    # Count the usable input features (not scores, not identifiers, not targets)
    skip = {
        "home_win", "point_diff",
        "home_score", "away_score",
        "game_id", "game_date", "season", "season_year", "season_type",
        "home_team_id", "away_team_id",
    }
    feature_count = sum(1 for c in df.columns if c not in skip)
    print(f"  {df.shape[0]:,} rows | {feature_count} input features | {df.shape[1]} total columns")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA preprocessing pipeline")
    parser.add_argument("--games",      default="data/games.csv",
                        help="Path to games CSV")
    parser.add_argument("--team_stats", default="data/team_stats.csv",
                        help="Path to traditional team-stats CSV")
    parser.add_argument("--adv_stats",  default="data/advanced_team_stats.csv",
                        help="Path to advanced team-stats CSV")
    parser.add_argument("--output",     default="data/processed.csv",
                        help="Output path for processed CSV")
    args = parser.parse_args()

    preprocess(args.games, args.team_stats, args.adv_stats, args.output)
