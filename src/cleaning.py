"""
CSE 482 - Sports Analytics Project
NBA Game Outcome Prediction - Data Preprocessing Pipeline
"""

import os

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

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

ADV_COLS = ['defRating', 'eOffRating', 'efgPct', 'netRating', 'pace', 'tsPct']

reg_ids   = set(reg['gameId'])
reg_stats = stats[stats['gameId'].isin(reg_ids)][['gameId','teamId'] + BOX_COLS].copy()

# 5b. JOIN ADVANCED STATS
print("Joining advanced stats...")
adv_reg = adv[adv['gameId'].isin(reg_ids)][['gameId', 'teamId'] + ADV_COLS].copy()
reg_stats = reg_stats.merge(adv_reg, on=['gameId', 'teamId'], how='left')
for col in ADV_COLS:
    reg_stats[col] = reg_stats[col].fillna(reg_stats[col].median())
print(f"  Advanced stats joined: {len(adv_reg):,} rows matched")

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
    + BOX_COLS + ENGINEERED_COLS + ADV_COLS
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

# 7b. ELO RATINGS (no leakage — use Elo going INTO each game)
print("Computing Elo ratings...")
ELO_K = 20
elo_dict = {}  # teamId → current Elo
elo_records = []

for _, game in reg.iterrows():
    home_id = game['hometeamId']
    away_id = game['awayteamId']

    home_elo = elo_dict.get(home_id, 1500)
    away_elo = elo_dict.get(away_id, 1500)

    elo_records.append({
        'gameId': game['gameId'],
        'home_elo': home_elo,
        'away_elo': away_elo,
    })

    # Expected scores
    exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    exp_away = 1 - exp_home

    # Actual outcome
    actual_home = 1 if game['homeScore'] > game['awayScore'] else 0
    actual_away = 1 - actual_home

    # Update Elo
    elo_dict[home_id] = home_elo + ELO_K * (actual_home - exp_home)
    elo_dict[away_id] = away_elo + ELO_K * (actual_away - exp_away)

elo_df = pd.DataFrame(elo_records)
elo_df['diff_elo'] = elo_df['home_elo'] - elo_df['away_elo']
print(f"  Elo range: {elo_df['home_elo'].min():.1f} – {elo_df['home_elo'].max():.1f}")

# 7c. SEASON STAGE (games_into_season counts games played BEFORE this one)
print("Computing season stage features...")

team_game_ss = team_game.sort_values(['teamId', 'season_year', 'gameDateTimeEst']).copy()
team_game_ss['games_into_season'] = team_game_ss.groupby(['teamId', 'season_year']).cumcount()

def _season_stage_bucket(n):
    if n < 25:
        return 0  # early
    elif n < 57:
        return 1  # mid
    else:
        return 2  # late

team_game_ss['season_stage'] = team_game_ss['games_into_season'].apply(_season_stage_bucket)

# Pivot to home/away for merging into final later
season_stage_home = team_game_ss[team_game_ss['role'] == 'home'][
    ['gameId', 'games_into_season', 'season_stage']
].rename(columns={'games_into_season': 'home_games_into_season', 'season_stage': 'home_season_stage'})
season_stage_away = team_game_ss[team_game_ss['role'] == 'away'][
    ['gameId', 'games_into_season', 'season_stage']
].rename(columns={'games_into_season': 'away_games_into_season', 'season_stage': 'away_season_stage'})

print(f"  Season stage distribution: early={(_season_stage_bucket(0) == 0)}, mid, late")

# 8. ROLLING AVERAGES (multi-window sweep, no leakage)
# shift(1) ensures we only use information from BEFORE the current game.
# min_periods=3 allows early-season rows with some history.
# NOTE: rolling window intentionally crosses season boundaries —
# a team's last 10 games going into game 1 of a new season includes
# games from the previous season, which is valid recent form.
ROLL_WINDOWS = [5, 10, 15]
ROLL_COLS = ['win','team_score','opp_score','net_margin'] + BOX_COLS + ENGINEERED_COLS + ADV_COLS

print(f"Computing rolling averages for windows {ROLL_WINDOWS}...")
roll_feats = {}
for n in ROLL_WINDOWS:
    for col in ROLL_COLS:
        roll_feats[f'roll{n}_{col}'] = (
            team_game.groupby('teamId')[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=3).mean())
        )

rolling_df = team_game[['gameId','teamId','role','win_streak']].copy()
for k, v in roll_feats.items():
    rolling_df[k] = v

before = len(rolling_df)
rolling_df = rolling_df.dropna(subset=['roll5_win'])
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

# 10. DIFFERENCE FEATURES (home − away) for each rolling window
for n in ROLL_WINDOWS:
    prefix = f'home_roll{n}_'
    base_cols = [c.replace(prefix, '') for c in final.columns if c.startswith(prefix)]
    for col in base_cols:
        final[f'diff_roll{n}_{col}'] = final[f'home_roll{n}_{col}'] - final[f'away_roll{n}_{col}']

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

# 10c. HEAD-TO-HEAD WIN RATE (rolling 3 seasons, no leakage)
print("Computing head-to-head win rate features...")

# Build a lookup of (hometeamId, awayteamId, season_year) → home win rate
h2h_seasonal = (
    reg.groupby(['hometeamId', 'awayteamId', 'season_year'])['home_win']
    .mean()
    .reset_index()
    .rename(columns={'home_win': 'h2h_win_rate_season'})
    .sort_values(['hometeamId', 'awayteamId', 'season_year'])
)

# Rolling 3-season average using only prior seasons (shift(1))
h2h_seasonal['h2h_home_win_rate'] = (
    h2h_seasonal.groupby(['hometeamId', 'awayteamId'])['h2h_win_rate_season']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

final = final.merge(
    h2h_seasonal[['hometeamId', 'awayteamId', 'season_year', 'h2h_home_win_rate']],
    on=['hometeamId', 'awayteamId', 'season_year'], how='left'
)
# Fill NaN (first meeting or first season) with overall home win rate
median_h2h = final['h2h_home_win_rate'].median()
final['h2h_home_win_rate'] = final['h2h_home_win_rate'].fillna(median_h2h)

print(f"  h2h_home_win_rate range: {final['h2h_home_win_rate'].min():.3f} – {final['h2h_home_win_rate'].max():.3f}")

# 10d. OPPONENT STRENGTH (rolling average of opponents' win rates, last 10 games)
print("Computing opponent strength features...")

# For each team-game, get the opponent's rolling win rate
# First compute each team's rolling win rate (already in team_game as 'win')
team_game_sorted = team_game.sort_values(['teamId', 'gameDateTimeEst']).copy()
team_game_sorted['team_roll_win_rate'] = (
    team_game_sorted.groupby('teamId')['win']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Derive opponent team ID using game_lookup (which has hometeamId/awayteamId)
opp_lookup = game_lookup[['hometeamId', 'awayteamId']].reset_index()
# For each team_game row, the opponent is the other team in the same game
team_game_sorted = team_game_sorted.merge(opp_lookup, on='gameId', how='left')
team_game_sorted['oppTeamId'] = np.where(
    team_game_sorted['role'] == 'home',
    team_game_sorted['awayteamId'],
    team_game_sorted['hometeamId']
)
team_game_sorted.drop(columns=['hometeamId', 'awayteamId'], inplace=True)

# For each game, look up the opponent's rolling win rate
opp_wr_lookup = team_game_sorted[['gameId', 'teamId', 'team_roll_win_rate']].copy()
opp_wr_lookup = opp_wr_lookup.rename(columns={'teamId': 'oppTeamId', 'team_roll_win_rate': 'opp_roll_win_rate'})

team_game_sorted = team_game_sorted.merge(opp_wr_lookup, on=['gameId', 'oppTeamId'], how='left')

# Now compute rolling average of opponents' win rates (last 10 games) for opponent strength
team_game_sorted = team_game_sorted.sort_values(['teamId', 'gameDateTimeEst']).reset_index(drop=True)
team_game_sorted['opp_strength'] = (
    team_game_sorted.groupby('teamId')['opp_roll_win_rate']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Pivot to home/away
home_opp_str = team_game_sorted[team_game_sorted['role'] == 'home'][['gameId', 'opp_strength']].rename(
    columns={'opp_strength': 'home_opp_strength'}
)
away_opp_str = team_game_sorted[team_game_sorted['role'] == 'away'][['gameId', 'opp_strength']].rename(
    columns={'opp_strength': 'away_opp_strength'}
)

final = final.merge(home_opp_str, on='gameId', how='left')
final = final.merge(away_opp_str, on='gameId', how='left')

# Fill NaN with median
final['home_opp_strength'] = final['home_opp_strength'].fillna(final['home_opp_strength'].median())
final['away_opp_strength'] = final['away_opp_strength'].fillna(final['away_opp_strength'].median())
final['diff_opp_strength'] = final['home_opp_strength'] - final['away_opp_strength']

print(f"  home_opp_strength range: {final['home_opp_strength'].min():.3f} – {final['home_opp_strength'].max():.3f}")

# 10e. HOME-SPECIFIC WIN STREAK (consecutive home wins / road wins)
print("Computing home-specific win streak features...")

def compute_venue_streak(sub):
    """
    Count consecutive wins at the same venue going into each game.
    sub must be sorted by gameDateTimeEst ascending and all same role (home or away).
    Resets at season boundaries and on any loss.
    """
    wins = sub['win'].values
    seasons = sub['season_year'].values
    streaks = np.zeros(len(wins), dtype=int)
    current = 0
    for i in range(len(wins)):
        if i > 0 and seasons[i] != seasons[i - 1]:
            current = 0
        streaks[i] = current
        current = current + 1 if wins[i] == 1 else 0
    return pd.Series(streaks, index=sub.index)

# Split team_game by role and compute venue-specific streaks
home_games = team_game[team_game['role'] == 'home'].sort_values(['teamId', 'gameDateTimeEst']).copy()
away_games = team_game[team_game['role'] == 'away'].sort_values(['teamId', 'gameDateTimeEst']).copy()

home_venue_streaks = []
for tid, grp in home_games.groupby('teamId'):
    home_venue_streaks.append(compute_venue_streak(grp))
home_games['home_venue_win_streak'] = pd.concat(home_venue_streaks)

away_venue_streaks = []
for tid, grp in away_games.groupby('teamId'):
    away_venue_streaks.append(compute_venue_streak(grp))
away_games['away_venue_win_streak'] = pd.concat(away_venue_streaks)

# Merge into final
final = final.merge(
    home_games[['gameId', 'home_venue_win_streak']], on='gameId', how='left'
)
final = final.merge(
    away_games[['gameId', 'away_venue_win_streak']], on='gameId', how='left'
)

final['home_venue_win_streak'] = final['home_venue_win_streak'].fillna(0).astype(int)
final['away_venue_win_streak'] = final['away_venue_win_streak'].fillna(0).astype(int)
final['diff_venue_win_streak'] = final['home_venue_win_streak'] - final['away_venue_win_streak']

print(f"  home_venue_win_streak max: {final['home_venue_win_streak'].max()}")
print(f"  away_venue_win_streak max: {final['away_venue_win_streak'].max()}")

print(f"After new engineered features: {final.shape}")

# 10f. MERGE ELO RATINGS
print("Merging Elo ratings into final...")
final = final.merge(elo_df, on='gameId', how='left')
final['home_elo'] = final['home_elo'].fillna(1500)
final['away_elo'] = final['away_elo'].fillna(1500)
final['diff_elo'] = final['diff_elo'].fillna(0)
print(f"  Elo features added: home_elo, away_elo, diff_elo")

# 10g. MERGE SEASON STAGE
print("Merging season stage features into final...")
final = final.merge(season_stage_home, on='gameId', how='left')
final = final.merge(season_stage_away, on='gameId', how='left')
final['home_games_into_season'] = final['home_games_into_season'].fillna(0).astype(int)
final['away_games_into_season'] = final['away_games_into_season'].fillna(0).astype(int)
final['home_season_stage'] = final['home_season_stage'].fillna(0).astype(int)
final['away_season_stage'] = final['away_season_stage'].fillna(0).astype(int)
print(f"  Season stage features added")

print(f"After all engineered features: {final.shape}")

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

# 11b. FEATURE SELECTION VIA LDA (Linear Discriminant Analysis)
meta_cols_set = {
    'gameId', 'gameDateTimeEst', 'hometeamName', 'awayteamName',
    'hometeamId', 'awayteamId', 'homeScore', 'awayScore', 'season_year',
    'home_win', 'point_diff'
}

feature_candidates = [c for c in train.columns if c not in meta_cols_set]

X_train_lda = train[feature_candidates].values
y_train_lda = train['home_win'].values

# Scale features for LDA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lda)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train_lda)

# Rank features by absolute LDA coefficient weights, normalized to percentages
abs_weights = np.abs(lda.coef_[0])
weight_pct = abs_weights / abs_weights.sum() * 100

lda_df = pd.DataFrame({
    'feature': feature_candidates,
    'weight_pct': weight_pct,
}).sort_values('weight_pct', ascending=False)

# Threshold sweep: show how many features each threshold keeps
print("\n--- LDA Threshold Sweep ---")
for threshold in [0.005, 0.01, 0.015, 0.02]:
    threshold_pct = threshold * 100
    n_kept = (lda_df['weight_pct'] >= threshold_pct).sum()
    print(f"  Threshold {threshold} ({threshold_pct:.1f}%): keeps {n_kept} / {len(feature_candidates)} features")

# Apply 1% threshold for final selection
LDA_THRESHOLD_PCT = 1.0
kept_mask = lda_df['weight_pct'] >= LDA_THRESHOLD_PCT
kept_lda = lda_df[kept_mask]
dropped_lda = lda_df[~kept_mask]

print(f"\nLDA feature selection (threshold={LDA_THRESHOLD_PCT}%): kept {len(kept_lda)} features, dropped {len(dropped_lda)} features")
print("  Kept features and weights:")
for _, row in kept_lda.iterrows():
    print(f"    {row['feature']:40s} {row['weight_pct']:.2f}%")
print("  Dropped features (< 1% weight):")
for _, row in dropped_lda.iterrows():
    print(f"    {row['feature']:40s} {row['weight_pct']:.2f}%")

kept_features = kept_lda['feature'].tolist()

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
