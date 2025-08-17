# ==============================#
#  BIDIRECTIONAL STACKED ENSEMBLE
#  - Classifiers ↔ Regressor feedback loop
#  - Stall risk detection
#  - No hyperparameter tuning (fast defaults)
# ==============================#

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, classification_report

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "test-combined.csv"
MODEL_DIR = "./bidirectional_ensemble_v1/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_OOF = 3
THRESHOLDS = [1, 2, 3]  # 1R, 2R, 3R

# Set threading
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------
# Base Parameters (Unbalanced)
# -----------------------------
DEFAULT_MC_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "verbosity": 1,
    "random_state": RANDOM_STATE,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "class_weight": None,
    "n_jobs": os.cpu_count() or 1,
}

DEFAULT_REG_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "verbosity": 1,
    "random_state": RANDOM_STATE,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "n_jobs": os.cpu_count() or 1,
}

DEFAULT_META_PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "boosting_type": "gbdt",
    "verbosity": 1,
    "random_state": RANDOM_STATE,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "class_weight": None,
    "n_jobs": os.cpu_count() or 1,
}

# -----------------------------
# Core Feature Engineering
# -----------------------------
def calculate_confidence_gap(probs: np.ndarray) -> np.ndarray:
    """max(P(win),P(loss)) - P(neutral)"""
    return np.max(probs[:, [0, 2]], axis=1) - probs[:, 1]

def calculate_stall_risk(prob_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """Average P(0) across all thresholds"""
    return np.mean([probs[:, 1] for probs in prob_dict.values()], axis=0)

def compute_reg_error_stats(y_true: pd.Series, y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Pre-compute median/MAD of absolute errors per RR bin"""
    bins = [0, 1, 2, 3, np.inf]
    errors = np.abs(y_true - y_pred)
    bin_assignments = pd.cut(y_pred, bins=bins, labels=False)
    return {
        bin_id: {
            "median": np.median(errors[bin_assignments == bin_id]),
            "mad": np.median(np.abs(errors[bin_assignments == bin_id] - 
                            np.median(errors[bin_assignments == bin_id])))
        }
        for bin_id in range(len(bins)-1)
    }

def get_binned_error(preds: np.ndarray, error_stats: dict) -> np.ndarray:
    """Map predictions to precomputed error stats"""
    bins = [0, 1, 2, 3, np.inf]
    bin_ids = pd.cut(preds, bins=bins, labels=False)
    return np.array([error_stats.get(bin_id, {"median": 1.0})["median"] for bin_id in bin_ids])

# -----------------------------
# Data Preparation
# -----------------------------
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    
    # Ensure categorical
    if 'pair' in df.columns:
        df['pair'] = df['pair'].astype('category')
    
    # Cast numerics to float32
    exclude_cols = {'pair', 'rr_label'}.union(
        {c for c in df.columns if re.match(r"^y_(\d+(\.\d+)?)R$", c)}
    )
    num_cols = [c for c in df.columns if c not in exclude_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype('float32')
    
    # Map thresholds to columns
    thr_to_col = {}
    for t in THRESHOLDS:
        col = min(
            [c for c in df.columns if re.match(r"^y_(\d+(\.\d+)?)R$", c)],
            key=lambda x: abs(float(re.match(r"^y_(\d+(\.\d+)?)R$", x).group(1)) - t)
        )
        thr_to_col[t] = col
    
    # Filter rows with all targets
    mask = np.ones(len(df), dtype=bool)
    for c in thr_to_col.values():
        mask &= df[c].notna()
    df = df.loc[mask].reset_index(drop=True)
    
    # Prepare targets
    rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
    y_meta = pd.cut(rr, bins=[-np.inf, 1, 2, 3, np.inf], labels=False, right=False).astype('int32')
    y_thr = {t: df[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
    
    # Features
    drop_cols = {'rr_label'}.union(thr_to_col.values())
    features = [c for c in df.columns if c not in drop_cols]
    
    return df[features], y_thr, rr, y_meta

# -----------------------------
# Model Training (Bidirectional)
# -----------------------------
def train_bidirectional_ensemble():
    X, y_thr, rr, y_meta = load_and_prepare_data()
    
    # Split data
    X_train, X_test, rr_train, rr_test, y_meta_train, y_meta_test = train_test_split(
        X, rr, y_meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta
    )
    y_thr_train = {t: y_thr[t].iloc[X_train.index] for t in THRESHOLDS}
    y_thr_test = {t: y_thr[t].iloc[X_test.index] for t in THRESHOLDS}
    
    # --- Phase 1: Initial Regressor ---
    print("\n=== Training Initial Regressor ===")
    reg_initial = LGBMRegressor(**DEFAULT_REG_PARAMS)
    reg_initial.fit(X_train, rr_train)
    initial_pred = reg_initial.predict(X_train)
    
    # --- Phase 2: Classifiers with Regressor Context ---
    print("\n=== Training Classifiers ===")
    classifiers = {}
    oof_probs = {}
    
    for t in THRESHOLDS:
        print(f"\nTraining {t}R classifier...")
        X_aug = X_train.assign(reg_pred=initial_pred)
        y_enc = y_thr_train[t].map({-1: 0, 0: 1, 1: 2}).values
        
        model = LGBMClassifier(**DEFAULT_MC_PARAMS)
        oof_probs[t], models = oof_multiclass(model, X_aug, y_enc)
        classifiers[t] = models[0]  # Use first fold model
    
    # --- Phase 3: Final Regressor with Classifier Features ---
    print("\n=== Training Final Regressor ===")
    X_reg_train = build_regressor_features(X_train, oof_probs)
    reg_final = LGBMRegressor(**DEFAULT_REG_PARAMS)
    reg_final.fit(X_reg_train, rr_train)
    final_pred = reg_final.predict(X_reg_train)
    
    # --- Phase 4: Meta Model ---
    print("\n=== Training Meta Model ===")
    error_stats = compute_reg_error_stats(rr_train, final_pred)
    X_meta_train = build_meta_features(X_train, oof_probs, final_pred, error_stats)
    meta_model = LGBMClassifier(**DEFAULT_META_PARAMS)
    meta_model.fit(X_meta_train, y_meta_train)
    
    # --- Evaluation ---
    evaluate_models(X_test, y_thr_test, rr_test, y_meta_test, 
                   classifiers, reg_initial, reg_final, meta_model)
    
    # --- Save Artifacts ---
    save_artifacts(classifiers, reg_initial, reg_final, meta_model, 
                  X.columns.tolist(), error_stats)

# -----------------------------
# Feature Builders
# -----------------------------
def build_regressor_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
    Xf = X.copy()
    for t in THRESHOLDS:
        probs = prob_dict[t]
        Xf[f'clf_{t}R_pneg1'] = probs[:, 0]
        Xf[f'clf_{t}R_p0'] = probs[:, 1]
        Xf[f'clf_{t}R_p1'] = probs[:, 2]
        Xf[f'clf_{t}R_conf_gap'] = calculate_confidence_gap(probs)
    return Xf

def build_meta_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray],
                       reg_pred: np.ndarray, error_stats: dict) -> pd.DataFrame:
    Xf = X.copy()
    
    # Classifier features
    for t in THRESHOLDS:
        probs = prob_dict[t]
        Xf[f'clf_{t}R_pneg1'] = probs[:, 0]
        Xf[f'clf_{t}R_p0'] = probs[:, 1]
        Xf[f'clf_{t}R_p1'] = probs[:, 2]
        Xf[f'clf_{t}R_ev'] = -probs[:, 0] + probs[:, 2]
        Xf[f'clf_{t}R_conf_gap'] = calculate_confidence_gap(probs)
    
    # Regressor features
    Xf['reg_pred'] = reg_pred
    Xf['reg_error'] = get_binned_error(reg_pred, error_stats)
    
    # Combined features
    Xf['stall_risk'] = calculate_stall_risk(prob_dict)
    
    return Xf

# -----------------------------
# Evaluation & Saving
# -----------------------------
def evaluate_models(X_test, y_thr_test, rr_test, y_meta_test, 
                   classifiers, reg_initial, reg_final, meta_model):
    """Full evaluation pipeline"""
    # Initial regressor
    initial_pred = reg_initial.predict(X_test)
    print(f"\nInitial Regressor MAE: {mean_absolute_error(rr_test, initial_pred):.4f}")
    
    # Classifiers
    X_test_aug = X_test.assign(reg_pred=initial_pred)
    for t in THRESHOLDS:
        probs = classifiers[t].predict_proba(X_test_aug)
        y_pred = np.argmax(probs, axis=1)
        y_true = y_thr_test[t].map({-1: 0, 0: 1, 1: 2}).values
        print(f"\n{t}R Classifier Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=["loss(-1)", "neutral(0)", "win(+1)"]))
    
    # Final regressor
    prob_dict_test = {t: classifiers[t].predict_proba(X_test_aug) for t in THRESHOLDS}
    X_reg_test = build_regressor_features(X_test, prob_dict_test)
    final_pred = reg_final.predict(X_reg_test)
    print(f"\nFinal Regressor MAE: {mean_absolute_error(rr_test, final_pred):.4f}")
    
    # Meta model
    error_stats = compute_reg_error_stats(rr_test, final_pred)
    X_meta_test = build_meta_features(X_test, prob_dict_test, final_pred, error_stats)
    meta_pred = meta_model.predict(X_meta_test)
    print("\nMeta Model Report:")
    print(classification_report(y_meta_test, meta_pred,
                              target_names=["Reject(0)", "Target1R(1)", "Target2R(2)", "Target3R(3)"]))
    
    # Regressor adjustment analysis
    adjustments = np.abs(initial_pred - final_pred)
    print(f"\nRegressor Adjustments:")
    print(f"Median: {np.median(adjustments):.2f}R | Max: {np.max(adjustments):.2f}R")

def save_artifacts(classifiers, reg_initial, reg_final, meta_model, 
                 feature_names, error_stats):
    """Save all models and metadata"""
    # Save models
    for t in THRESHOLDS:
        classifiers[t].booster_.save_model(f"{MODEL_DIR}classifier_{t}R.txt")
    joblib.dump(reg_initial, f"{MODEL_DIR}regressor_initial.pkl")
    reg_final.booster_.save_model(f"{MODEL_DIR}regressor_final.txt")
    meta_model.booster_.save_model(f"{MODEL_DIR}meta_model.txt")
    
    # Save error stats
    joblib.dump(error_stats, f"{MODEL_DIR}reg_error_stats.pkl")
    
    # Save metadata
    metadata = {
        "features": feature_names,
        "thresholds": THRESHOLDS,
        "model_params": {
            "classifiers": DEFAULT_MC_PARAMS,
            "regressor": DEFAULT_REG_PARAMS,
            "meta": DEFAULT_META_PARAMS
        },
        "notes": "Bidirectional ensemble with regressor-classifier feedback (no tuning)"
    }
    with open(f"{MODEL_DIR}metadata.json", "w") as f:
        json.dump(metadata, f)

# -----------------------------
# OOF Helpers
# -----------------------------
def oof_multiclass(model, X, y, n_splits=3):
    """Out-of-fold predictions for classifiers"""
    oof = np.zeros((len(X), 3))
    models = []
    for tr_idx, val_idx in StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                         random_state=RANDOM_STATE).split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])
        models.append(m)
    return oof, models

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    train_bidirectional_ensemble()
    print(f"\n✅ All models saved to {MODEL_DIR}")