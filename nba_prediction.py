"""
NBA Game Outcome Prediction Pipeline
=====================================
Tasks:
  1. Classification – predict whether the home team wins (home_win)
  2. Regression     – predict the point differential (point_diff)

Models:
  Classification : Logistic Regression, Gradient Boosting (tuned + calibrated),
                   PCA + Logistic Regression
  Regression     : Ridge Regression, Gradient Boosting (tuned)

Evaluation:
  TimeSeriesSplit (5 folds) for cross-validation across season boundaries,
  plus a held-out test set for final evaluation.

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

from scipy.stats import uniform, randint
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "nba_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "nba_test.csv")
FULL_CSV = os.path.join(DATA_DIR, "nba_processed_full.csv")

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Logistic Regression hyper-parameters
LR_PARAMS = dict(max_iter=1000, random_state=42)

# Ridge Regression hyper-parameters
RIDGE_PARAMS = dict(alpha=1.0)

# Hyperparameter search space for Gradient Boosting (shared by clf and reg)
# n_estimators: 100–500 covers light to heavy ensembles
# max_depth: 2–6 prevents overfitting while allowing interaction effects
# learning_rate: 0.01–0.2 ranges from very conservative to moderate
# subsample: 0.6–1.0 (uniform(0.6, 0.4) → U[0.6, 1.0]) for stochastic boosting
GB_SEARCH_SPACE = dict(
    n_estimators=randint(100, 501),
    max_depth=randint(2, 7),
    learning_rate=uniform(0.01, 0.19),
    subsample=uniform(0.6, 0.4),
)

# Target columns and columns that must never be used as features
CLF_TARGET = "home_win"
REG_TARGET = "point_diff"

CLF_EXCLUDE = {"homeScore", "awayScore", "point_diff"}
REG_EXCLUDE = {"homeScore", "awayScore", "home_win"}


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


def prepare_full_xy(full_df, target, exclude_cols):
    """
    Return (X, y, feature_names) from the full dataset for TimeSeriesSplit CV.
    """
    feature_cols = get_feature_columns(full_df, target, exclude_cols)
    clean = full_df.dropna(subset=[target])
    X = clean[feature_cols].fillna(clean[feature_cols].median())
    y = clean[target]
    return X, y, feature_cols


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


def plot_calibration_curve(y_true, y_prob, title, filename, n_bins=10):
    """Plot and save a calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated GB")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved: {path}")


# ---------------------------------------------------------------------------
# TimeSeriesSplit cross-validation helpers
# ---------------------------------------------------------------------------


def ts_cv_classification(X, y, model, n_splits=5):
    """Run TimeSeriesSplit CV for a classifier and return mean accuracy/AUC."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs = [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        accs.append(accuracy_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_prob))
        print(f"    Fold {fold}: Accuracy={accs[-1]:.4f}  AUC={aucs[-1]:.4f}")
    return np.mean(accs), np.mean(aucs)


def ts_cv_regression(X, y, model, n_splits=5):
    """Run TimeSeriesSplit CV for a regressor and return mean RMSE/MAE/R²."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmses.append(mean_squared_error(y_val, y_pred) ** 0.5)
        maes.append(mean_absolute_error(y_val, y_pred))
        r2s.append(r2_score(y_val, y_pred))
        print(f"    Fold {fold}: RMSE={rmses[-1]:.4f}  MAE={maes[-1]:.4f}  R²={r2s[-1]:.4f}")
    return np.mean(rmses), np.mean(maes), np.mean(r2s)


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------


def run_classification(train_df, test_df, full_df):
    print("\n" + "=" * 60)
    print("CLASSIFICATION TASK  –  predicting home_win")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = prepare_xy(
        train_df, test_df, CLF_TARGET, CLF_EXCLUDE
    )
    X_full, y_full, _ = prepare_full_xy(full_df, CLF_TARGET, CLF_EXCLUDE)

    print(f"  features : {len(feature_names)}")
    print(f"  train    : {X_train.shape[0]} games")
    print(f"  test     : {X_test.shape[0]} games")
    print(f"  full     : {X_full.shape[0]} games (for CV)")

    # Scale for Logistic Regression and PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression (with TimeSeriesSplit CV)
    # ------------------------------------------------------------------
    print("\n[1/4] Logistic Regression …")
    print("  TimeSeriesSplit CV (5 folds):")
    scaler_full = StandardScaler()
    X_full_scaled = pd.DataFrame(
        scaler_full.fit_transform(X_full), index=X_full.index, columns=X_full.columns
    )
    lr_cv = LogisticRegression(**LR_PARAMS)
    cv_acc_lr, cv_auc_lr = ts_cv_classification(X_full_scaled, y_full, lr_cv)
    print(f"  CV Mean Accuracy: {cv_acc_lr:.4f}  CV Mean AUC: {cv_auc_lr:.4f}")

    # Final evaluation on held-out test set
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    results["Logistic Regression"] = {
        "Accuracy": acc_lr, "ROC-AUC": auc_lr,
        "CV_Accuracy": cv_acc_lr, "CV_AUC": cv_auc_lr,
    }

    print(f"  Test Accuracy : {acc_lr:.4f}")
    print(f"  Test ROC-AUC  : {auc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["Away Win", "Home Win"]))

    plot_logistic_coefficients(
        lr.coef_[0], feature_names,
        "Logistic Regression – Feature Coefficients (Classification)",
        "clf_logistic_coefficients.png",
    )

    # ------------------------------------------------------------------
    # 2. Gradient Boosting Classifier (tuned + calibrated)
    # ------------------------------------------------------------------
    print("[2/4] Gradient Boosting Classifier (hyperparameter tuning) …")
    tscv = TimeSeriesSplit(n_splits=5)
    gbc_base = GradientBoostingClassifier(random_state=42)
    gbc_search = RandomizedSearchCV(
        gbc_base, GB_SEARCH_SPACE, n_iter=20, cv=tscv, scoring="accuracy",
        random_state=42, n_jobs=-1, verbose=0,
    )
    gbc_search.fit(X_train, y_train)
    best_gbc_params = gbc_search.best_params_
    print(f"  Best params: {best_gbc_params}")
    print(f"  Best CV accuracy: {gbc_search.best_score_:.4f}")

    # TimeSeriesSplit CV with best params on full data
    print("  TimeSeriesSplit CV (5 folds) on full data:")
    gbc_best = GradientBoostingClassifier(random_state=42, **best_gbc_params)
    cv_acc_gbc, cv_auc_gbc = ts_cv_classification(X_full, y_full, gbc_best)
    print(f"  CV Mean Accuracy: {cv_acc_gbc:.4f}  CV Mean AUC: {cv_auc_gbc:.4f}")

    # Final model on train, evaluate on test
    gbc_final = GradientBoostingClassifier(random_state=42, **best_gbc_params)
    gbc_final.fit(X_train, y_train)
    y_pred_gbc = gbc_final.predict(X_test)
    y_prob_gbc = gbc_final.predict_proba(X_test)[:, 1]

    acc_gbc = accuracy_score(y_test, y_pred_gbc)
    auc_gbc = roc_auc_score(y_test, y_prob_gbc)
    results["Gradient Boosting"] = {
        "Accuracy": acc_gbc, "ROC-AUC": auc_gbc,
        "CV_Accuracy": cv_acc_gbc, "CV_AUC": cv_auc_gbc,
    }

    print(f"  Test Accuracy : {acc_gbc:.4f}")
    print(f"  Test ROC-AUC  : {auc_gbc:.4f}")
    print(classification_report(y_test, y_pred_gbc, target_names=["Away Win", "Home Win"]))

    plot_feature_importance(
        gbc_final.feature_importances_, feature_names,
        "Gradient Boosting – Feature Importances (Classification)",
        "clf_gb_feature_importance.png",
    )

    # Calibrated classifier
    print("  Calibrating classifier …")
    cal_gbc = CalibratedClassifierCV(gbc_final, cv=5, method="sigmoid")
    cal_gbc.fit(X_train, y_train)
    y_prob_cal = cal_gbc.predict_proba(X_test)[:, 1]
    y_pred_cal = cal_gbc.predict(X_test)

    acc_cal = accuracy_score(y_test, y_pred_cal)
    auc_cal = roc_auc_score(y_test, y_prob_cal)
    results["Calibrated GB"] = {"Accuracy": acc_cal, "ROC-AUC": auc_cal}
    print(f"  Calibrated Test Accuracy: {acc_cal:.4f}  ROC-AUC: {auc_cal:.4f}")

    plot_calibration_curve(
        y_test, y_prob_cal,
        "Calibrated Gradient Boosting – Calibration Curve",
        "clf_calibration_curve.png",
    )

    # ------------------------------------------------------------------
    # 3. PCA + Logistic Regression (with TimeSeriesSplit CV)
    # ------------------------------------------------------------------
    print("[3/4] PCA + Logistic Regression …")
    # Keep enough components to explain 95% of variance
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"  PCA components: {pca.n_components_} (explaining {pca.explained_variance_ratio_.sum():.2%} variance)")

    lr_pca = LogisticRegression(**LR_PARAMS)
    lr_pca.fit(X_train_pca, y_train)
    y_pred_pca = lr_pca.predict(X_test_pca)
    y_prob_pca = lr_pca.predict_proba(X_test_pca)[:, 1]

    acc_pca = accuracy_score(y_test, y_pred_pca)
    auc_pca = roc_auc_score(y_test, y_prob_pca)
    results["PCA + Logistic Regression"] = {"Accuracy": acc_pca, "ROC-AUC": auc_pca}

    print(f"  Accuracy : {acc_pca:.4f}")
    print(f"  ROC-AUC  : {auc_pca:.4f}")
    print(classification_report(y_test, y_pred_pca, target_names=["Away Win", "Home Win"]))

    # Plot PCA explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
           alpha=0.6, label="Individual")
    ax.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label="Cumulative")
    ax.set_title("PCA – Explained Variance Ratio (Classification)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "clf_pca_variance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved: {path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Classification Results Summary ---")
    for model, metrics in results.items():
        line = f"  {model:30s}  Accuracy={metrics['Accuracy']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}"
        if "CV_Accuracy" in metrics:
            line += f"  CV-Acc={metrics['CV_Accuracy']:.4f}  CV-AUC={metrics['CV_AUC']:.4f}"
        print(line)

    return results


# ---------------------------------------------------------------------------
# Regression pipeline
# ---------------------------------------------------------------------------


def run_regression(train_df, test_df, full_df):
    print("\n" + "=" * 60)
    print("REGRESSION TASK  –  predicting point_diff")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = prepare_xy(
        train_df, test_df, REG_TARGET, REG_EXCLUDE
    )
    X_full, y_full, _ = prepare_full_xy(full_df, REG_TARGET, REG_EXCLUDE)

    print(f"  features : {len(feature_names)}")
    print(f"  train    : {X_train.shape[0]} games")
    print(f"  test     : {X_test.shape[0]} games")
    print(f"  full     : {X_full.shape[0]} games (for CV)")

    # Scale for Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ------------------------------------------------------------------
    # 1. Ridge Regression (with TimeSeriesSplit CV)
    # ------------------------------------------------------------------
    print("\n[1/2] Ridge Regression …")
    print("  TimeSeriesSplit CV (5 folds):")
    scaler_full = StandardScaler()
    X_full_scaled = pd.DataFrame(
        scaler_full.fit_transform(X_full), index=X_full.index, columns=X_full.columns
    )
    ridge_cv = Ridge(**RIDGE_PARAMS)
    cv_rmse_ridge, cv_mae_ridge, cv_r2_ridge = ts_cv_regression(X_full_scaled, y_full, ridge_cv)
    print(f"  CV Mean RMSE: {cv_rmse_ridge:.4f}  MAE: {cv_mae_ridge:.4f}  R²: {cv_r2_ridge:.4f}")

    ridge = Ridge(**RIDGE_PARAMS)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    rmse_ridge = mean_squared_error(y_test, y_pred_ridge) ** 0.5
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    results["Ridge Regression"] = {
        "RMSE": rmse_ridge, "MAE": mae_ridge, "R2": r2_ridge,
        "CV_RMSE": cv_rmse_ridge, "CV_MAE": cv_mae_ridge, "CV_R2": cv_r2_ridge,
    }

    print(f"  Test RMSE : {rmse_ridge:.4f}")
    print(f"  Test MAE  : {mae_ridge:.4f}")
    print(f"  Test R²   : {r2_ridge:.4f}")

    plot_ridge_coefficients(
        ridge.coef_, feature_names,
        "Ridge Regression – Feature Coefficients (Regression)",
        "reg_ridge_coefficients.png",
    )

    # ------------------------------------------------------------------
    # 2. Gradient Boosting Regressor (tuned with RandomizedSearchCV)
    # ------------------------------------------------------------------
    print("[2/2] Gradient Boosting Regressor (hyperparameter tuning) …")
    tscv = TimeSeriesSplit(n_splits=5)
    gbr_base = GradientBoostingRegressor(random_state=42)
    gbr_search = RandomizedSearchCV(
        gbr_base, GB_SEARCH_SPACE, n_iter=20, cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0,
    )
    gbr_search.fit(X_train, y_train)
    best_gbr_params = gbr_search.best_params_
    print(f"  Best params: {best_gbr_params}")
    print(f"  Best CV neg-MSE: {gbr_search.best_score_:.4f}")

    # TimeSeriesSplit CV with best params on full data
    print("  TimeSeriesSplit CV (5 folds) on full data:")
    gbr_best = GradientBoostingRegressor(random_state=42, **best_gbr_params)
    cv_rmse_gbr, cv_mae_gbr, cv_r2_gbr = ts_cv_regression(X_full, y_full, gbr_best)
    print(f"  CV Mean RMSE: {cv_rmse_gbr:.4f}  MAE: {cv_mae_gbr:.4f}  R²: {cv_r2_gbr:.4f}")

    # Final model on train, evaluate on test
    gbr_final = GradientBoostingRegressor(random_state=42, **best_gbr_params)
    gbr_final.fit(X_train, y_train)
    y_pred_gbr = gbr_final.predict(X_test)

    rmse_gbr = mean_squared_error(y_test, y_pred_gbr) ** 0.5
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    results["Gradient Boosting"] = {
        "RMSE": rmse_gbr, "MAE": mae_gbr, "R2": r2_gbr,
        "CV_RMSE": cv_rmse_gbr, "CV_MAE": cv_mae_gbr, "CV_R2": cv_r2_gbr,
    }

    print(f"  Test RMSE : {rmse_gbr:.4f}")
    print(f"  Test MAE  : {mae_gbr:.4f}")
    print(f"  Test R²   : {r2_gbr:.4f}")

    plot_feature_importance(
        gbr_final.feature_importances_, feature_names,
        "Gradient Boosting – Feature Importances (Regression)",
        "reg_gb_feature_importance.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Regression Results Summary ---")
    for model, metrics in results.items():
        line = f"  {model:25s}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}"
        if "CV_RMSE" in metrics:
            line += f"  CV-RMSE={metrics['CV_RMSE']:.4f}"
        print(line)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    train_df, test_df, full_df = load_data()

    print(f"\nFull dataset info: {full_df.shape[0]} rows, {full_df.shape[1]} columns")
    print(f"Columns: {list(full_df.columns)}")

    clf_results = run_classification(train_df, test_df, full_df)
    reg_results = run_regression(train_df, test_df, full_df)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("\nClassification (home_win):")
    for model, m in clf_results.items():
        line = f"  {model:30s}  Accuracy={m['Accuracy']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}"
        if "CV_Accuracy" in m:
            line += f"  CV-Acc={m['CV_Accuracy']:.4f}"
        print(line)
    print("\nRegression (point_diff):")
    for model, m in reg_results.items():
        line = f"  {model:25s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}"
        if "CV_RMSE" in m:
            line += f"  CV-RMSE={m['CV_RMSE']:.4f}"
        print(line)
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
