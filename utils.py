import pandas as pd

def load_data(matches_path, deliveries_path):
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries

def preprocess_matches(matches):
    # Example preprocessing steps
    matches['date'] = pd.to_datetime(matches['date'])
    matches.dropna(subset=['winner'], inplace=True)
    return matches

def get_team_list(matches):
    teams = pd.unique(matches['team1'].append(matches['team2']))
    return sorted(teams)

def get_season_wise_data(matches):
    season_summary = matches.groupby('season')['winner'].value_counts().unstack().fillna(0)
    return season_summary
