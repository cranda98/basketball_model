"""
CSE 482 - Sports Analytics Project
NBA Game Outcome Prediction - Data Preprocessing Pipeline
"""

import os

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE

# Resolve paths relative to this script so it runs from any working directory
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, 'data')

# 1. LOAD RAW DATA
games = pd.read_csv(os.path.join(_DATA_DIR, 'cse482_project_db_Games.csv'))
stats = pd.read_csv(os.path.join(_DATA_DIR, 'cse482_project_db_TeamStatistics.csv'))
adv   = pd.read_csv(os.path.join(_DATA_DIR, 'cse482_project_db_TeamStatisticsAdvanced.csv'))

for df in [games, stats, adv]:
    df['gameId'] = df['gameId'].astype(str)

games['gameDateTimeEst'] = pd.to_datetime(games['gameDateTimeEst'])

print(f"  Raw games:    {len(games):,}")
print(f"  Raw stats:    {len(stats):,}")
print(f"  Raw advanced: {len(adv):,}")

# 2. FILTER TO REGULAR SEASON 2000+
# gameId first digit encodes game type:
#   1=Preseason, 2=Regular Season, 3=All-Star,
#   4=Playoffs, 5=Play-In, 6=NBA Cup
# make sure our dates are aligned with nba season dates
games['type_code']   = games['gameId'].str[0]
games['season_year'] = np.where(
    games['gameDateTimeEst'].dt.month >= 10,
    games['gameDateTimeEst'].dt.year,
    games['gameDateTimeEst'].dt.year - 1
)

reg = games[
    (games['type_code'] == '2') &
    (games['season_year'] >= 2015)
].copy().sort_values('gameDateTimeEst').reset_index(drop=True)
before = len(reg)
reg = reg[
    (reg['homeScore'] > 0) &
    (reg['awayScore'] > 0) &
    (reg['homeScore'] != reg['awayScore'])
].copy()
print(f"Dropped {before - len(reg)} games with missing/invalid scores")

print(f"\nRegular season games (2015+): {len(reg):,}")
print(f"  Date range:   {reg['gameDateTimeEst'].min().date()} → {reg['gameDateTimeEst'].max().date()}")
print(f"  Home win rate: {(reg['homeScore'] > reg['awayScore']).mean():.3%}")

# 3. TARGET VARIABLES
reg['home_win'] = (reg['homeScore'] > reg['awayScore']).astype(int)
reg['point_diff'] = reg['homeScore'] - reg['awayScore']

# 4. REST DAYS & BACK-TO-BACK
print("\nComputing rest days and back-to-back flags...")

home_sched = reg[['gameId','gameDateTimeEst','hometeamId']].rename(columns={'hometeamId':'teamId'})
away_sched = reg[['gameId','gameDateTimeEst','awayteamId']].rename(columns={'awayteamId':'teamId'})
team_sched = pd.concat([home_sched, away_sched]).sort_values(['teamId','gameDateTimeEst']).reset_index(drop=True)

# Extract calendar date only (fixes the rounding-down bug)
team_sched['game_date'] = team_sched['gameDateTimeEst'].dt.normalize()
team_sched['prev_date'] = team_sched.groupby('teamId')['game_date'].shift(1)
team_sched['rest_days'] = (team_sched['game_date'] - team_sched['prev_date']).dt.days

REST_CAP = 14
team_sched['rest_days'] = team_sched['rest_days'].clip(upper=REST_CAP)

# Fill NaN (first game in dataset per team) with median after capping
median_rest = team_sched['rest_days'].median()
team_sched['rest_days'] = team_sched['rest_days'].fillna(median_rest)

# Back-to-back: played on consecutive calendar days (rest = 1)
# Note: rest == 0 should not occur after the date-fix above
team_sched['is_b2b'] = (team_sched['rest_days'] == 1).astype(int)

rest_lookup = team_sched[['gameId','teamId','rest_days','is_b2b']]

# Join home and away rest separately
reg = reg.merge(
    rest_lookup.rename(columns={'teamId':'hometeamId','rest_days':'home_rest_days','is_b2b':'home_is_b2b'}),
    on=['gameId','hometeamId'], how='left'
)
reg = reg.merge(
    rest_lookup.rename(columns={'teamId':'awayteamId','rest_days':'away_rest_days','is_b2b':'away_is_b2b'}),
    on=['gameId','awayteamId'], how='left'
)

reg['rest_advantage'] = reg['home_rest_days'] - reg['away_rest_days']

# Sanity check
assert team_sched['rest_days'].min() >= 1.0 or team_sched['rest_days'].isna().any(), \
    "rest_days should not contain 0 after date-fix"
print(f"  rest_days range: {team_sched['rest_days'].min():.0f} – {team_sched['rest_days'].max():.0f} (capped at {REST_CAP})")
print(f"  Back-to-back rate: {team_sched['is_b2b'].mean():.1%}")
print(f"  Avg rest days: {team_sched['rest_days'].mean():.2f}")

# 5. BOX SCORE STATS PER TEAM PER GAME
BOX_COLS = [
    'assists', 'blocks', 'steals',
    'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
    'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
    'turnovers'
]

reg_ids   = set(reg['gameId'])
reg_stats = stats[stats['gameId'].isin(reg_ids)][['gameId','teamId'] + BOX_COLS].copy()

# Include season_year in lookup so win streak can use it
game_lookup = reg.set_index('gameId')[
    ['hometeamId','awayteamId','homeScore','awayScore','gameDateTimeEst','season_year']
]
reg_stats = reg_stats.join(game_lookup, on='gameId')

reg_stats['role'] = np.where(reg_stats['teamId'] == reg_stats['hometeamId'], 'home', 'away')
reg_stats['team_score'] = np.where(reg_stats['role'] == 'home', reg_stats['homeScore'], reg_stats['awayScore'])
reg_stats['opp_score']  = np.where(reg_stats['role'] == 'home', reg_stats['awayScore'], reg_stats['homeScore'])
reg_stats['win'] = (reg_stats['team_score'] > reg_stats['opp_score']).astype(int)
reg_stats['reboundsTotal'] = reg_stats['reboundsDefensive'] + reg_stats['reboundsOffensive']

# Sanity check: every row should be home or away, never neither
assert ((reg_stats['teamId'] == reg_stats['hometeamId']) |
        (reg_stats['teamId'] == reg_stats['awayteamId'])).all(), \
    "Some teamId rows matched neither home nor away!"

# 6. ENGINEER PER-GAME FEATURES
# Assist-to-Turnover Ratio
reg_stats['ast_to_ratio'] = reg_stats['assists'] / reg_stats['turnovers'].replace(0, np.nan)
reg_stats['ast_to_ratio'] = reg_stats['ast_to_ratio'].fillna(reg_stats['ast_to_ratio'].median())
# Fill null FT% (games where a team had zero free throw attempts)
reg_stats['freeThrowsPercentage'] = reg_stats['freeThrowsPercentage'].fillna(
    reg_stats['freeThrowsPercentage'].median()
)

# eFG% proxy (no raw shot counts available, FG% + weighted 3P% approximates it)
reg_stats['efg_proxy'] = reg_stats['fieldGoalsPercentage'] + 0.15 * reg_stats['threePointersPercentage']

# Net margin per game
reg_stats['net_margin'] = reg_stats['team_score'] - reg_stats['opp_score']

ENGINEERED_COLS = ['ast_to_ratio', 'efg_proxy', 'net_margin']

# Build team_game must include season_year for win streak reset
team_game = reg_stats[
    ['gameId','gameDateTimeEst','season_year','teamId','role','win','team_score','opp_score']
    + BOX_COLS + ENGINEERED_COLS
].copy().sort_values(['teamId','gameDateTimeEst']).reset_index(drop=True)

print(f"\nTeam-game rows: {len(team_game):,} (expected {len(reg)*2:,})")
print(f"Null rate in box cols: {team_game[BOX_COLS].isnull().mean().max():.4f}")

# 7. WIN STREAK
print("Computing win streaks (with season reset)...")

def compute_streak_for_team(sub):
    """
    For each game, count consecutive wins the team had going into it.
    Resets at season boundaries and on any loss.
    sub must be sorted by gameDateTimeEst ascending.
    """
    wins    = sub['win'].values
    seasons = sub['season_year'].values
    streaks = np.zeros(len(wins), dtype=int)
    current = 0
    for i in range(len(wins)):
        # Reset at start of new season
        if i > 0 and seasons[i] != seasons[i - 1]:
            current = 0
        # Record streak going INTO this game (before outcome is known)
        streaks[i] = current
        # Update streak based on this game's result
        current = current + 1 if wins[i] == 1 else 0
    return pd.Series(streaks, index=sub.index)

results = []
for tid, grp in team_game.groupby('teamId'):
    results.append(compute_streak_for_team(grp))
team_game['win_streak'] = pd.concat(results)

# Sanity check against known records
# Warriors 2015-16 started 24-0 → max streak that season should be 24
warriors_id = reg[reg['hometeamName'] == 'Warriors']['hometeamId'].iloc[0]
w15 = team_game[(team_game['teamId'] == warriors_id) & (team_game['season_year'] == 2015)]
print(f"  Warriors 2015-16 max streak: {w15['win_streak'].max()} (expect ~24)")

spurs_id = reg[reg['hometeamName'] == 'Spurs']['hometeamId'].iloc[0]
sp = team_game[team_game['teamId'] == spurs_id]
print(f"  Spurs all-time max streak:   {sp['win_streak'].max()}")

print(f"  Overall max streak: {team_game['win_streak'].max()} (NBA record since 2000 is 27)")

# 8. ROLLING AVERAGES (last N games, no leakage)
# shift(1) ensures we only use information from BEFORE the current game.
# min_periods=3 allows early-season rows with some history.
# NOTE: rolling window intentionally crosses season boundaries —
# a team's last 10 games going into game 1 of a new season includes
# games from the previous season, which is valid recent form.
N = 10
ROLL_COLS = ['win','team_score','opp_score','net_margin'] + BOX_COLS + ENGINEERED_COLS

print(f"Computing {N}-game rolling averages...")
roll_feats = {}
for col in ROLL_COLS:
    roll_feats[f'roll_{col}'] = (
        team_game.groupby('teamId')[col]
        .transform(lambda x: x.shift(1).rolling(N, min_periods=3).mean())
    )

rolling_df = team_game[['gameId','teamId','role','win_streak']].copy()
for k, v in roll_feats.items():
    rolling_df[k] = v

before = len(rolling_df)
rolling_df = rolling_df.dropna(subset=['roll_win'])
print(f"  Dropped {before - len(rolling_df):,} rows (insufficient history)")

# 9. PIVOT TO ONE ROW PER GAME
home_feats = rolling_df[rolling_df['role'] == 'home'].drop('role', axis=1).copy()
home_feats.columns = ['gameId','teamId','home_win_streak'] + \
                     [f'home_{c}' for c in home_feats.columns[3:]]

away_feats = rolling_df[rolling_df['role'] == 'away'].drop('role', axis=1).copy()
away_feats.columns = ['gameId','teamId','away_win_streak'] + \
                     [f'away_{c}' for c in away_feats.columns[3:]]

final = reg[[
    'gameId','gameDateTimeEst','hometeamName','awayteamName',
    'hometeamId','awayteamId','homeScore','awayScore',
    'home_win','point_diff',
    'home_rest_days','away_rest_days','home_is_b2b','away_is_b2b','rest_advantage'
]].copy()

final = final.merge(home_feats.drop('teamId', axis=1), on='gameId', how='inner')
final = final.merge(away_feats.drop('teamId', axis=1), on='gameId', how='inner')

assert final.isnull().sum().sum() == 0, "Unexpected nulls in final dataset!"
print(f"\nFinal shape: {final.shape} | Nulls: {final.isnull().sum().sum()} ✓")
print(f"Home win rate: {final['home_win'].mean():.3%}")

# 10. DIFFERENCE FEATURES (home − away)
diff_base = [c.replace('home_roll_','') for c in final.columns if c.startswith('home_roll_')]
for col in diff_base:
    final[f'diff_{col}'] = final[f'home_roll_{col}'] - final[f'away_roll_{col}']

final['diff_win_streak'] = final['home_win_streak'] - final['away_win_streak']

print(f"After diff features: {final.shape}")

# 10b. ROLLING HOME COURT TREND (3-season rolling win rate at home / on road)
print("Computing home court trend features...")

# Compute per-team per-season home win rate
home_seasonal = (
    reg.groupby(['hometeamId', 'season_year'])['home_win']
    .mean()
    .reset_index()
    .rename(columns={'hometeamId': 'teamId', 'home_win': 'home_win_rate_season'})
    .sort_values(['teamId', 'season_year'])
)

# Compute per-team per-season away win rate
reg['away_win'] = 1 - reg['home_win']
away_seasonal = (
    reg.groupby(['awayteamId', 'season_year'])['away_win']
    .mean()
    .reset_index()
    .rename(columns={'awayteamId': 'teamId', 'away_win': 'away_win_rate_season'})
    .sort_values(['teamId', 'season_year'])
)

# Rolling 3-season average (using only prior seasons via shift(1))
home_seasonal['home_court_trend'] = (
    home_seasonal.groupby('teamId')['home_win_rate_season']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
away_seasonal['away_court_trend'] = (
    away_seasonal.groupby('teamId')['away_win_rate_season']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Attach season_year to final for merging
final['season_year'] = np.where(
    final['gameDateTimeEst'].dt.month >= 10,
    final['gameDateTimeEst'].dt.year,
    final['gameDateTimeEst'].dt.year - 1
)

final = final.merge(
    home_seasonal[['teamId', 'season_year', 'home_court_trend']].rename(columns={'teamId': 'hometeamId'}),
    on=['hometeamId', 'season_year'], how='left'
)
final = final.merge(
    away_seasonal[['teamId', 'season_year', 'away_court_trend']].rename(columns={'teamId': 'awayteamId'}),
    on=['awayteamId', 'season_year'], how='left'
)

# Fill NaN for teams in their first season (no prior data) with median
median_hct = final['home_court_trend'].median()
median_act = final['away_court_trend'].median()
final['home_court_trend'] = final['home_court_trend'].fillna(median_hct)
final['away_court_trend'] = final['away_court_trend'].fillna(median_act)

final['diff_court_trend'] = final['home_court_trend'] - final['away_court_trend']

print(f"After court trend features: {final.shape}")

# 11. TRAIN / TEST SPLIT
final['season_year'] = np.where(
    final['gameDateTimeEst'].dt.month >= 10,
    final['gameDateTimeEst'].dt.year,
    final['gameDateTimeEst'].dt.year - 1
)

train = final[final['season_year'] < 2025].copy()
test  = final[final['season_year'] >= 2025].copy()

print(f"\nTrain: {len(train):,} ({train['gameDateTimeEst'].min().date()} → {train['gameDateTimeEst'].max().date()})")
print(f"Test:  {len(test):,}  ({test['gameDateTimeEst'].min().date()} → {test['gameDateTimeEst'].max().date()})")
print(f"Train home win rate: {train['home_win'].mean():.3%}")
print(f"Test  home win rate: {test['home_win'].mean():.3%}")

# 11b. FEATURE SELECTION
# Explicitly drop highly correlated redundant features
explicit_drop = [
    'home_roll_efg_proxy', 'away_roll_efg_proxy', 'diff_efg_proxy',
    'home_roll_win',       'away_roll_win',        'diff_win',
]
explicit_drop = [c for c in explicit_drop if c in final.columns]
final.drop(columns=explicit_drop, inplace=True)
train.drop(columns=explicit_drop, inplace=True)
test.drop(columns=explicit_drop, inplace=True)
print(f"\nExplicitly dropped {len(explicit_drop)} features: {explicit_drop}")

# Auto-drop features with |correlation| < 0.05 against home_win (computed on train)
meta_cols = {'gameId','gameDateTimeEst','hometeamName','awayteamName',
             'hometeamId','awayteamId','homeScore','awayScore','season_year',
             'home_win','point_diff'}
feature_candidates = [c for c in train.columns if c not in meta_cols]
corr_series = train[feature_candidates].corrwith(train['home_win']).abs()
low_corr_cols = corr_series[corr_series < 0.05].index.tolist()
if low_corr_cols:
    final.drop(columns=low_corr_cols, inplace=True)
    train.drop(columns=low_corr_cols, inplace=True)
    test.drop(columns=low_corr_cols, inplace=True)
print(f"Auto-dropped {len(low_corr_cols)} low-correlation features (|corr| < 0.05):")
for c in low_corr_cols:
    print(f"  {c}  (|corr|={corr_series[c]:.4f})")

# 11c. RECURSIVE FEATURE ELIMINATION (RFE) via GradientBoostingClassifier
# Keep the top 15–20 features by importance score.
RFE_N_FEATURES   = 17   # target number of features to keep (within 15–20)
RFE_N_ESTIMATORS = 100

meta_cols_set = {
    'gameId', 'gameDateTimeEst', 'hometeamName', 'awayteamName',
    'hometeamId', 'awayteamId', 'homeScore', 'awayScore', 'season_year',
    'home_win', 'point_diff'
}

rfe_candidates = [c for c in train.columns if c not in meta_cols_set]

X_train_rfe = train[rfe_candidates].values
y_train_rfe = train['home_win'].values

gbc = GradientBoostingClassifier(n_estimators=RFE_N_ESTIMATORS, random_state=42)
selector = RFE(estimator=gbc, n_features_to_select=RFE_N_FEATURES, step=1)
selector.fit(X_train_rfe, y_train_rfe)

rfe_mask = selector.support_
kept_features   = [c for c, s in zip(rfe_candidates, rfe_mask) if s]
dropped_features = [c for c, s in zip(rfe_candidates, rfe_mask) if not s]

print(f"\nRFE: kept {len(kept_features)} features, dropped {len(dropped_features)} features")
print("  Kept:")
for c in kept_features:
    print(f"    {c}")
print("  Dropped by RFE:")
for c in dropped_features:
    print(f"    {c}")

keep_cols = [c for c in meta_cols_set if c in final.columns] + kept_features
# Preserve original column order
keep_cols_ordered = [c for c in final.columns if c in set(keep_cols)]

final = final[keep_cols_ordered]
train = train[keep_cols_ordered]
test  = test[keep_cols_ordered]

# 12. SAVE
final.to_csv(os.path.join(_DATA_DIR, 'nba_processed_full.csv'), index=False)
train.to_csv(os.path.join(_DATA_DIR, 'nba_train.csv'), index=False)
test.to_csv(os.path.join(_DATA_DIR, 'nba_test.csv'), index=False)

print("\nSaved: nba_processed_full.csv | nba_train.csv | nba_test.csv")

# 13. FINAL FEATURE LIST
feat_cols = [c for c in final.columns if c not in meta_cols_set]
print(f"\nTotal features for modeling: {len(feat_cols)}")
for c in feat_cols:
    print(f"  {c}")
