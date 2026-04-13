"""
NBA Game Outcome Prediction Pipeline
=====================================
Tasks:
  1. Classification – predict whether the home team wins (home_win)
  2. Regression     – predict the point differential (point_diff)

Models:
  Classification : Logistic Regression, Gradient Boosting
  Regression     : Ridge Regression, Gradient Boosting

Feature constraints:
  Classification : never use homeScore, awayScore, or point_diff as features
  Regression     : never use homeScore, awayScore, or home_win as features
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saved plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATA_DIR, "nba_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "nba_test.csv")
FULL_CSV = os.path.join(DATA_DIR, "nba_processed_full.csv")

PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Gradient Boosting hyper-parameters (shared by both tasks)
GB_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    random_state=42,
)

# Logistic Regression hyper-parameters
LR_PARAMS = dict(max_iter=1000, random_state=42)

# Ridge Regression hyper-parameters
RIDGE_PARAMS = dict(alpha=1.0)

# Target columns and columns that must never be used as features
CLF_TARGET = "home_win"
REG_TARGET = "point_diff"

CLF_EXCLUDE = {"homeScore", "awayScore", "point_diff"}
REG_EXCLUDE = {"homeScore", "awayScore", "home_win"}

# Number of top features to select with RFE
RFE_N_FEATURES = 20


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_data():
    """Load train, test, and full processed CSVs."""
    print("Loading data …")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    full = pd.read_csv(FULL_CSV)
    print(f"  train : {train.shape}")
    print(f"  test  : {test.shape}")
    print(f"  full  : {full.shape}")
    return train, test, full


def get_feature_columns(df, target, exclude_cols):
    """
    Return numeric feature column names by dropping the target,
    any excluded columns, and any non-numeric columns.
    """
    drop = {target} | exclude_cols
    cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def prepare_xy(train_df, test_df, target, exclude_cols):
    """
    Return (X_train, y_train, X_test, y_test, feature_names)
    after aligning columns and dropping rows with NaN targets.
    """
    feature_cols = get_feature_columns(train_df, target, exclude_cols)

    train_clean = train_df.dropna(subset=[target])
    test_clean = test_df.dropna(subset=[target])

    X_train = train_clean[feature_cols].fillna(train_clean[feature_cols].median())
    y_train = train_clean[target]
    train_medians = train_clean[feature_cols].median()
    X_test = test_clean[feature_cols].fillna(train_medians)
    y_test = test_clean[target]

    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_feature_importance(importances, feature_names, title, filename, top_n=20):
    """Bar chart of top-N feature importances."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved: {path}")


def plot_logistic_coefficients(coef, feature_names, title, filename, top_n=20):
    """Bar chart of absolute logistic regression coefficients."""
    df = pd.DataFrame({"feature": feature_names, "coef": np.abs(coef)})
    df = df.sort_values("coef", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="coef", y="feature", ax=ax, palette="magma")
    ax.set_title(title)
    ax.set_xlabel("|Coefficient|")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved: {path}")


def plot_ridge_coefficients(coef, feature_names, title, filename, top_n=20):
    """Bar chart of absolute Ridge regression coefficients."""
    plot_logistic_coefficients(coef, feature_names, title, filename, top_n)


def plot_rfe_ranking(ranking, feature_names, selected_mask, title, filename, top_n=20):
    """Bar chart showing RFE-selected features ranked by support."""
    df = pd.DataFrame({"feature": feature_names, "ranking": ranking, "selected": selected_mask})
    df_sel = df[df["selected"]].sort_values("ranking")
    df_show = df_sel.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_show, x="ranking", y="feature", ax=ax, palette="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("RFE Ranking (lower = more important)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved: {path}")


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------


def run_classification(train_df, test_df):
    print("\n" + "=" * 60)
    print("CLASSIFICATION TASK  –  predicting home_win")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = prepare_xy(
        train_df, test_df, CLF_TARGET, CLF_EXCLUDE
    )
    print(f"  features : {len(feature_names)}")
    print(f"  train    : {X_train.shape[0]} games")
    print(f"  test     : {X_test.shape[0]} games")

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------------
    print("\n[1/4] Logistic Regression …")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    results["Logistic Regression"] = {"Accuracy": acc_lr, "ROC-AUC": auc_lr}

    print(f"  Accuracy : {acc_lr:.4f}")
    print(f"  ROC-AUC  : {auc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["Away Win", "Home Win"]))

    plot_logistic_coefficients(
        lr.coef_[0], feature_names,
        "Logistic Regression – Feature Coefficients (Classification)",
        "clf_logistic_coefficients.png",
    )

    # ------------------------------------------------------------------
    # 2. Gradient Boosting Classifier
    # ------------------------------------------------------------------
    print("[2/4] Gradient Boosting Classifier …")
    gbc = GradientBoostingClassifier(**GB_PARAMS)
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)
    y_prob_gbc = gbc.predict_proba(X_test)[:, 1]

    acc_gbc = accuracy_score(y_test, y_pred_gbc)
    auc_gbc = roc_auc_score(y_test, y_prob_gbc)
    results["Gradient Boosting"] = {"Accuracy": acc_gbc, "ROC-AUC": auc_gbc}

    print(f"  Accuracy : {acc_gbc:.4f}")
    print(f"  ROC-AUC  : {auc_gbc:.4f}")
    print(classification_report(y_test, y_pred_gbc, target_names=["Away Win", "Home Win"]))

    plot_feature_importance(
        gbc.feature_importances_, feature_names,
        "Gradient Boosting – Feature Importances (Classification)",
        "clf_gb_feature_importance.png",
    )

    # ------------------------------------------------------------------
    # 3. RFE with Logistic Regression
    # ------------------------------------------------------------------
    print(f"[3/4] RFE (Logistic Regression, selecting {RFE_N_FEATURES} features) …")
    rfe_lr = RFE(
        LogisticRegression(**LR_PARAMS),
        n_features_to_select=RFE_N_FEATURES,
        step=5,
    )
    rfe_lr.fit(X_train_scaled, y_train)
    selected_features_lr = np.array(feature_names)[rfe_lr.support_]
    print(f"  Selected features: {list(selected_features_lr)}")

    plot_rfe_ranking(
        rfe_lr.ranking_, feature_names, rfe_lr.support_,
        f"RFE (Logistic Regression) – Top {RFE_N_FEATURES} Features (Classification)",
        "clf_rfe_logistic.png",
    )

    # ------------------------------------------------------------------
    # 4. RFE with Gradient Boosting
    # ------------------------------------------------------------------
    print(f"[4/4] RFE (Gradient Boosting, selecting {RFE_N_FEATURES} features) …")
    rfe_gb = RFE(
        GradientBoostingClassifier(**GB_PARAMS),
        n_features_to_select=RFE_N_FEATURES,
        step=5,
    )
    rfe_gb.fit(X_train, y_train)
    selected_features_gb = np.array(feature_names)[rfe_gb.support_]
    print(f"  Selected features: {list(selected_features_gb)}")

    plot_rfe_ranking(
        rfe_gb.ranking_, feature_names, rfe_gb.support_,
        f"RFE (Gradient Boosting) – Top {RFE_N_FEATURES} Features (Classification)",
        "clf_rfe_gb.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Classification Results Summary ---")
    for model, metrics in results.items():
        print(f"  {model:25s}  Accuracy={metrics['Accuracy']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Regression pipeline
# ---------------------------------------------------------------------------


def run_regression(train_df, test_df):
    print("\n" + "=" * 60)
    print("REGRESSION TASK  –  predicting point_diff")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = prepare_xy(
        train_df, test_df, REG_TARGET, REG_EXCLUDE
    )
    print(f"  features : {len(feature_names)}")
    print(f"  train    : {X_train.shape[0]} games")
    print(f"  test     : {X_test.shape[0]} games")

    # Scale for Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ------------------------------------------------------------------
    # 1. Ridge Regression
    # ------------------------------------------------------------------
    print("\n[1/4] Ridge Regression …")
    ridge = Ridge(**RIDGE_PARAMS)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    rmse_ridge = mean_squared_error(y_test, y_pred_ridge) ** 0.5
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    results["Ridge Regression"] = {"RMSE": rmse_ridge, "MAE": mae_ridge, "R2": r2_ridge}

    print(f"  RMSE : {rmse_ridge:.4f}")
    print(f"  MAE  : {mae_ridge:.4f}")
    print(f"  R²   : {r2_ridge:.4f}")

    plot_ridge_coefficients(
        ridge.coef_, feature_names,
        "Ridge Regression – Feature Coefficients (Regression)",
        "reg_ridge_coefficients.png",
    )

    # ------------------------------------------------------------------
    # 2. Gradient Boosting Regressor
    # ------------------------------------------------------------------
    print("[2/4] Gradient Boosting Regressor …")
    gbr = GradientBoostingRegressor(**GB_PARAMS)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)

    rmse_gbr = mean_squared_error(y_test, y_pred_gbr) ** 0.5
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    results["Gradient Boosting"] = {"RMSE": rmse_gbr, "MAE": mae_gbr, "R2": r2_gbr}

    print(f"  RMSE : {rmse_gbr:.4f}")
    print(f"  MAE  : {mae_gbr:.4f}")
    print(f"  R²   : {r2_gbr:.4f}")

    plot_feature_importance(
        gbr.feature_importances_, feature_names,
        "Gradient Boosting – Feature Importances (Regression)",
        "reg_gb_feature_importance.png",
    )

    # ------------------------------------------------------------------
    # 3. RFE with Ridge Regression
    # ------------------------------------------------------------------
    print(f"[3/4] RFE (Ridge, selecting {RFE_N_FEATURES} features) …")
    rfe_ridge = RFE(
        Ridge(**RIDGE_PARAMS),
        n_features_to_select=RFE_N_FEATURES,
        step=5,
    )
    rfe_ridge.fit(X_train_scaled, y_train)
    selected_features_ridge = np.array(feature_names)[rfe_ridge.support_]
    print(f"  Selected features: {list(selected_features_ridge)}")

    plot_rfe_ranking(
        rfe_ridge.ranking_, feature_names, rfe_ridge.support_,
        f"RFE (Ridge) – Top {RFE_N_FEATURES} Features (Regression)",
        "reg_rfe_ridge.png",
    )

    # ------------------------------------------------------------------
    # 4. RFE with Gradient Boosting
    # ------------------------------------------------------------------
    print(f"[4/4] RFE (Gradient Boosting, selecting {RFE_N_FEATURES} features) …")
    rfe_gb = RFE(
        GradientBoostingRegressor(**GB_PARAMS),
        n_features_to_select=RFE_N_FEATURES,
        step=5,
    )
    rfe_gb.fit(X_train, y_train)
    selected_features_gb = np.array(feature_names)[rfe_gb.support_]
    print(f"  Selected features: {list(selected_features_gb)}")

    plot_rfe_ranking(
        rfe_gb.ranking_, feature_names, rfe_gb.support_,
        f"RFE (Gradient Boosting) – Top {RFE_N_FEATURES} Features (Regression)",
        "reg_rfe_gb.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Regression Results Summary ---")
    for model, metrics in results.items():
        print(
            f"  {model:25s}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    train_df, test_df, full_df = load_data()

    print(f"\nFull dataset info: {full_df.shape[0]} rows, {full_df.shape[1]} columns")
    print(f"Columns: {list(full_df.columns)}")

    clf_results = run_classification(train_df, test_df)
    reg_results = run_regression(train_df, test_df)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("\nClassification (home_win):")
    for model, m in clf_results.items():
        print(f"  {model:25s}  Accuracy={m['Accuracy']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}")
    print("\nRegression (point_diff):")
    for model, m in reg_results.items():
        print(f"  {model:25s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
