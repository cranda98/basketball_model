# NBA Game Outcome Prediction

A machine learning project that predicts NBA game outcomes using historical team 
statistics. Two tasks are modeled: predicting whether the home team wins 
(classification) and predicting the point differential (regression).

## Data

Data sourced from Kaggle: [Historical NBA Data & Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)

Three tables are used:
- Games — game dates, home/away team IDs, final scores
- Team Statistics — box score stats per team per game
- Team Statistics Advanced — efficiency metrics per team per game

## Models

- Classification: Logistic Regression and Gradient Boosting (predicts home win)
- Regression: Ridge Regression and Gradient Boosting (predicts point differential)

## Pipeline

1. `src/cleaning.py` — loads raw data, engineers features, outputs train/test CSVs
2. `src/classification.py` — trains and evaluates classification models
3. `src/regression.py` — trains and evaluates regression models

## Future Work

- Run feature selection before modeling using recursive feature elimination to 
  identify the most useful predictors and reduce noise
- Source a more complete advanced stats dataset with consistent coverage of netRtg, 
  pace, and efgPct across all seasons
- Train on recent seasons only to better reflect modern NBA conditions given the 
  declining home court advantage trend
- Add a rolling home court trend feature per team rather than treating home court 
  as a static yes or no signal

## Responsible AI Use

Claude was used to help draft and scaffold parts of this project. All 
generated code has been reviewed, tested, and revised before being treated as 
correct. AI assistance was used to accelerate development but not to replace my 
own understanding or judgment of the resulting workflow.
