#!/usr/bin/env python3
"""
NBA Game Outcome Prediction – Model Training and Evaluation

Loads the preprocessed game data produced by preprocess.py, splits into
training (seasons 2000-2024) and test (2025), and trains/evaluates:

  Classification (target: home_win)
  ──────────────────────────────────
  • Logistic Regression  (with StandardScaler)
  • Gradient Boosting    (max_depth=4, lr=0.05, n_estimators=200, subsample=0.8)
  Metrics: accuracy, ROC-AUC, F1, Brier score, confusion matrix

  Regression (target: point_diff)
  ────────────────────────────────
  • Ridge Regression  (with StandardScaler)
  • Gradient Boosting (same hyperparameters)
  Metrics: MAE, RMSE, R², % predictions within 5 points

Also produces:
  • Feature importance plots for both GBM models
  • Recursive Feature Elimination (RFECV) to identify the most useful
    features from the 56-feature set

Usage
-----
    python train.py --data data/processed.csv --output_dir results/
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for any environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Hyperparameters ────────────────────────────────────────────────────────────

GB_PARAMS: dict = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    random_state=42,
)

# ── Feature-column selection ───────────────────────────────────────────────────

# Columns that must NEVER appear in the feature matrix
_FORBIDDEN: set[str] = {
    # Raw scores – would constitute target leakage
    "home_score", "away_score", "homeScore", "awayScore",
    "pts_home", "pts_away", "home_pts", "away_pts",
    # Target variables
    "home_win", "point_diff",
    # Non-predictive identifiers / metadata
    "game_id", "game_date", "season", "season_year", "season_type",
    "home_team_id", "away_team_id",
}

# Substring patterns that disqualify a column (case-insensitive)
_FORBIDDEN_PATTERNS: list[str] = ["score", "wl", "matchup", "team_name", "team_abbrev"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of input feature columns."""
    cols = []
    for c in df.columns:
        if c in _FORBIDDEN:
            continue
        cl = c.lower()
        if any(p in cl for p in _FORBIDDEN_PATTERNS):
            continue
        cols.append(c)
    return cols


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _pct_within_5(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 5))


# ── Classifier training ────────────────────────────────────────────────────────

def evaluate_classifier(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":    name,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_prob),
        "f1":       f1_score(y_test, y_pred),
        "brier":    brier_score_loss(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)

    header = f"  {name}  –  Classification (home_win)"
    print(f"\n{'='*60}\n{header}\n{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Brier     : {metrics['brier']:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Away Win", "Home Win"]).plot(ax=ax)
    ax.set_title(f"{name}")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{slug}_confusion_matrix.png"), dpi=150)
    plt.close()

    return metrics


# ── Regressor training ─────────────────────────────────────────────────────────

def evaluate_regressor(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model":    name,
        "mae":      mean_absolute_error(y_test, y_pred),
        "rmse":     _rmse(y_test, y_pred),
        "r2":       r2_score(y_test, y_pred),
        "within_5": _pct_within_5(y_test, y_pred),
    }

    header = f"  {name}  –  Regression (point_diff)"
    print(f"\n{'='*60}\n{header}\n{'='*60}")
    print(f"  MAE          : {metrics['mae']:.4f}")
    print(f"  RMSE         : {metrics['rmse']:.4f}")
    print(f"  R²           : {metrics['r2']:.4f}")
    print(f"  Within 5 pts : {metrics['within_5']:.4f}  ({metrics['within_5']*100:.1f}%)")

    return metrics


# ── Feature importance plots ───────────────────────────────────────────────────

def plot_feature_importance(
    model_name: str,
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str,
    top_n: int = 20,
) -> None:
    n = min(top_n, len(feature_names))
    idx = np.argsort(importances)[::-1][:n]

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    ax.barh(range(n), importances[idx][::-1], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} – Top {n} Feature Importances")
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{slug}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved → {path}")


# ── Recursive Feature Elimination ─────────────────────────────────────────────

def run_rfecv_classification(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    output_dir: str,
) -> list[str]:
    """RFECV using Logistic Regression (fast, coefficients available)."""
    print("\n--- RFECV · Classification (Logistic Regression) ---")
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=5,
        n_jobs=-1,
    )
    rfecv.fit(X_train_scaled, y_train)

    selected = [feature_names[i] for i, s in enumerate(rfecv.support_) if s]
    print(f"  Optimal number of features : {rfecv.n_features_}")
    print(f"  Selected features ({len(selected)}):")
    for f in selected:
        print(f"    · {f}")

    _plot_rfe_curve(
        rfecv.cv_results_["mean_test_score"],
        "Classification RFECV (ROC-AUC)",
        "ROC-AUC",
        os.path.join(output_dir, "rfe_classification.png"),
    )
    return selected


def run_rfecv_regression(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    output_dir: str,
) -> list[str]:
    """RFECV using Ridge Regression (fast, coefficients available)."""
    print("\n--- RFECV · Regression (Ridge) ---")
    estimator = Ridge()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="neg_mean_absolute_error",
        min_features_to_select=5,
        n_jobs=-1,
    )
    rfecv.fit(X_train_scaled, y_train)

    selected = [feature_names[i] for i, s in enumerate(rfecv.support_) if s]
    print(f"  Optimal number of features : {rfecv.n_features_}")
    print(f"  Selected features ({len(selected)}):")
    for f in selected:
        print(f"    · {f}")

    _plot_rfe_curve(
        rfecv.cv_results_["mean_test_score"],
        "Regression RFECV (neg-MAE)",
        "Neg. MAE",
        os.path.join(output_dir, "rfe_regression.png"),
    )
    return selected


def _plot_rfe_curve(
    scores: np.ndarray,
    title: str,
    ylabel: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", ms=3)
    ax.axvline(int(np.argmax(scores)) + 1, color="red", linestyle="--", label="optimal")
    ax.set_xlabel("Number of Features Selected")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  RFE curve saved → {path}")


# ── Main training function ─────────────────────────────────────────────────────

def train(data_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading preprocessed data from {data_path} …")
    df = pd.read_csv(data_path, parse_dates=["game_date"])

    # Verify required target columns exist
    for col in ("home_win", "point_diff", "season_year"):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in {data_path}. "
                "Run preprocess.py first."
            )

    df = df.dropna(subset=["home_win", "point_diff"])

    # ── Train / test split ─────────────────────────────────────────────────────
    train_df = df[df["season_year"] <= 2024].copy()
    test_df  = df[df["season_year"] == 2025].copy()

    print(f"  Train : {len(train_df):,} games  (seasons 2000–2024)")
    print(f"  Test  : {len(test_df):,} games  (season 2025)")

    feature_cols = get_feature_cols(df)
    print(f"  Features before dropna : {len(feature_cols)}")

    required_cols = feature_cols + ["home_win", "point_diff"]
    train_df = train_df[required_cols].dropna()
    test_df  = test_df[required_cols].dropna()
    print(f"  Train after dropna : {len(train_df):,}")
    print(f"  Test  after dropna : {len(test_df):,}")
    print(f"  Final feature count : {len(feature_cols)}")

    X_train = train_df[feature_cols].to_numpy()
    y_train_clf = train_df["home_win"].to_numpy().astype(int)
    y_train_reg = train_df["point_diff"].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test_clf = test_df["home_win"].to_numpy().astype(int)
    y_test_reg = test_df["point_diff"].to_numpy()

    # Pre-scaled arrays for linear models and RFECV
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Classification ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n\n{sep}\n  CLASSIFICATION MODELS\n{sep}")

    clf_results = []

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf_results.append(
        evaluate_classifier(
            "Logistic Regression",
            lr_pipeline,
            X_train, y_train_clf,
            X_test,  y_test_clf,
            output_dir,
        )
    )

    gbc = GradientBoostingClassifier(**GB_PARAMS)
    clf_results.append(
        evaluate_classifier(
            "Gradient Boosting Classifier",
            gbc,
            X_train, y_train_clf,
            X_test,  y_test_clf,
            output_dir,
        )
    )

    # Feature importance for GBC (model already fitted above)
    plot_feature_importance(
        "Gradient Boosting Classifier",
        feature_cols,
        gbc.feature_importances_,
        output_dir,
    )

    pd.DataFrame(clf_results).to_csv(
        os.path.join(output_dir, "classification_results.csv"), index=False
    )

    # ── Regression ─────────────────────────────────────────────────────────────
    print(f"\n\n{sep}\n  REGRESSION MODELS\n{sep}")

    reg_results = []

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])
    reg_results.append(
        evaluate_regressor(
            "Ridge Regression",
            ridge_pipeline,
            X_train, y_train_reg,
            X_test,  y_test_reg,
            output_dir,
        )
    )

    gbr = GradientBoostingRegressor(**GB_PARAMS)
    reg_results.append(
        evaluate_regressor(
            "Gradient Boosting Regressor",
            gbr,
            X_train, y_train_reg,
            X_test,  y_test_reg,
            output_dir,
        )
    )

    # Feature importance for GBR (model already fitted above)
    plot_feature_importance(
        "Gradient Boosting Regressor",
        feature_cols,
        gbr.feature_importances_,
        output_dir,
    )

    pd.DataFrame(reg_results).to_csv(
        os.path.join(output_dir, "regression_results.csv"), index=False
    )

    # ── Recursive Feature Elimination ─────────────────────────────────────────
    print(f"\n\n{sep}\n  RECURSIVE FEATURE ELIMINATION (RFECV)\n{sep}")

    selected_clf = run_rfecv_classification(
        X_train_scaled, y_train_clf, feature_cols, output_dir
    )
    selected_reg = run_rfecv_regression(
        X_train_scaled, y_train_reg, feature_cols, output_dir
    )

    rfe_path = os.path.join(output_dir, "rfe_selected_features.txt")
    with open(rfe_path, "w") as fh:
        fh.write("Classification RFECV – Selected Features\n")
        fh.write("─" * 40 + "\n")
        for feat in selected_clf:
            fh.write(f"  {feat}\n")
        fh.write("\nRegression RFECV – Selected Features\n")
        fh.write("─" * 40 + "\n")
        for feat in selected_reg:
            fh.write(f"  {feat}\n")

    print(f"\n  RFE results saved → {rfe_path}")
    print(f"\n✓  All results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA model training pipeline")
    parser.add_argument("--data",       default="data/processed.csv",
                        help="Path to preprocessed CSV (output of preprocess.py)")
    parser.add_argument("--output_dir", default="results/",
                        help="Directory for plots and metric CSVs")
    args = parser.parse_args()

    train(args.data, args.output_dir)
