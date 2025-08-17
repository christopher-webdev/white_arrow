# ==============================#
#  BIDIRECTIONAL STACKED ENSEMBLE
#  - Classifiers ↔ Regressor feedback loop
#  - Stall risk detection
#  - Light Optuna tuning
# ==============================#

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import optuna
from typing import Dict, List, Tuple, Union
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "test-combined.csv"
MODEL_DIR = "./bidirectional_ensemble_v1/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_OOF = 3
THRESHOLDS = [1, 2, 3]  # 1R, 2R, 3R
OPTUNA_TRIALS = 10

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
    "verbosity": -1,
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
    "class_weight": None,  # Critical for preserving edge
    "n_jobs": os.cpu_count() or 1,
}

DEFAULT_REG_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "verbosity": -1,
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
    "verbosity": -1,
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

def compute_reg_error_stats(y_true: pd.Series, y_pred: np.ndarray, 
                          bins: List[float] = [0, 1, 2, 3, np.inf]) -> Dict[int, Dict[str, float]]:
    """Pre-compute median/MAD of absolute errors per RR bin"""
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

def get_binned_error(preds: np.ndarray, error_stats: dict, 
                   bins: List[float] = [0, 1, 2, 3, np.inf]) -> np.ndarray:
    """Map predictions to precomputed error stats"""
    bin_ids = pd.cut(preds, bins=bins, labels=False)
    return np.array([error_stats.get(bin_id, {"median": 1.0})["median"] for bin_id in bin_ids])

# -----------------------------
# Data Preparation
# -----------------------------
def load_and_prepare_data() -> Tuple[pd.DataFrame, Dict[int, pd.Series], pd.Series, pd.Series]:
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
    thr_to_col = find_y_columns_for_thresholds(df, THRESHOLDS)
    
    # Filter rows with all targets
    mask = np.ones(len(df), dtype=bool)
    for c in [thr_to_col[t] for t in THRESHOLDS]:
        mask &= df[c].notna()
    df = df.loc[mask].reset_index(drop=True)
    
    # Prepare targets
    rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
    y_meta = pd.cut(rr, bins=[-np.inf, 1, 2, 3, np.inf], labels=False, right=False).astype('int32')
    y_thr = {t: df[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
    
    # Features
    drop_cols = {'rr_label'}.union({thr_to_col[t] for t in THRESHOLDS})
    features = [c for c in df.columns if c not in drop_cols]
    
    return df[features], y_thr, rr, y_meta

def find_y_columns_for_thresholds(df: pd.DataFrame, thr_list: List[int]) -> Dict[int, str]:
    pattern = re.compile(r"^y_(\d+(?:\.\d+)?)R$")
    candidates = [(c, float(m.group(1))) for c in df.columns if (m := pattern.match(c))]
    if not candidates:
        raise ValueError("No y_*R columns found")
    return {t: min(candidates, key=lambda x: abs(x[1] - t))[0] for t in thr_list}

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
        
        # Light Optuna tuning
        if OPTUNA_TRIALS > 0:
            print(f"Tuning {t}R classifier ({OPTUNA_TRIALS} trials)...")
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 32, 128),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                }
                model = LGBMClassifier(**{**DEFAULT_MC_PARAMS, **params})
                return cross_val_score(model, X_aug, y_enc, cv=3, scoring='f1_macro').mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
        else:
            best_params = {}
        
        # Train with OOF
        model = LGBMClassifier(**{**DEFAULT_MC_PARAMS, **best_params})
        oof_probs[t], models = oof_multiclass(model, X_aug, y_enc)
        classifiers[t] = models[0]  # Use first fold model for simplicity
    
    # --- Phase 3: Final Regressor with Classifier Features ---
    print("\n=== Training Final Regressor ===")
    X_reg_train = build_regressor_features(X_train, oof_probs)
    
    # Light Optuna tuning
    if OPTUNA_TRIALS > 0:
        print(f"Tuning final regressor ({OPTUNA_TRIALS} trials)...")
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 32, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
            }
            model = LGBMRegressor(**{**DEFAULT_REG_PARAMS, **params})
            return -cross_val_score(model, X_reg_train, rr_train, cv=3, 
                                   scoring='neg_mean_absolute_error').mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
    else:
        best_params = {}
    
    reg_final = LGBMRegressor(**{**DEFAULT_REG_PARAMS, **best_params})
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
def build_regressor_features(X: pd.DataFrame, 
                           prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
    """Build features for final regressor"""
    Xf = X.copy()
    for t in THRESHOLDS:
        probs = prob_dict[t]
        Xf[f'clf_{t}R_pneg1'] = probs[:, 0]
        Xf[f'clf_{t}R_p0'] = probs[:, 1]
        Xf[f'clf_{t}R_p1'] = probs[:, 2]
        Xf[f'clf_{t}R_conf_gap'] = calculate_confidence_gap(probs)
    return Xf

def build_meta_features(X: pd.DataFrame, 
                      prob_dict: Dict[int, np.ndarray],
                      reg_pred: np.ndarray,
                      error_stats: dict) -> pd.DataFrame:
    """Build features for meta model"""
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
        "notes": "Bidirectional ensemble with regressor-classifier feedback"
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
# # ==============================#
# #  STACKED 3×MULTICLASS (±1/0/1)
# #  + REGRESSOR (rr_label)
# #  + META (0,1,2,3)
# #  (No Optuna; fast defaults + early stopping + threading + float32)
# # ==============================#

# import os
# import re
# import json
# import joblib
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple

# from lightgbm import LGBMClassifier, LGBMRegressor
# from lightgbm import early_stopping, log_evaluation  # <-- added

# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import (
#     classification_report, confusion_matrix, roc_auc_score,
#     f1_score, accuracy_score,
#     mean_absolute_error, mean_squared_error, r2_score
# )
# from sklearn.base import clone

# # ---------------------------------
# # Ensure multi-threading not capped
# # ---------------------------------
# if os.getenv("OMP_NUM_THREADS") in (None, "", "1"):
#     os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
# # (OpenBLAS/MKL caps to avoid oversub)
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH = "test-combined.csv"       # <- your labeled dataset
# MODEL_DIR = "./tmodel_artifacts_3xmc_meta_v1/"
# os.makedirs(MODEL_DIR, exist_ok=True)

# RANDOM_STATE = 42
# N_SPLITS_OOF  = 3
# N_SPLITS_CV   = 3

# THRESHOLDS = [1, 2, 3]   # target R:R thresholds

# # -----------------------------
# # Fast default LightGBM params
# # -----------------------------
# # Multiclass base (3 classes: {-1,0,1} -> {0,1,2})
# DEFAULT_MC_PARAMS = {
#     "objective": "multiclass",
#     "num_class": 3,
#     "boosting_type": "gbdt",
#     "verbosity": -1,
#     "random_state": RANDOM_STATE,
#     "n_estimators": 500,          # used for final fits (OOF will set a higher cap + early stop)
#     "learning_rate": 0.05,
#     "num_leaves": 64,
#     "max_depth": -1,
#     "min_child_samples": 30,
#     "feature_fraction": 0.9,
#     "bagging_fraction": 0.9,
#     "bagging_freq": 1,
#     "lambda_l1": 0.0,
#     "lambda_l2": 0.0,
#     "class_weight": "balanced",
#     "n_jobs": os.cpu_count() or 1,       # keep for compatibility
#     "num_threads": os.cpu_count() or 1,  # <-- added explicit threads
# }

# # Regressor (rr_label)
# DEFAULT_REG_PARAMS = {
#     "objective": "regression",
#     "boosting_type": "gbdt",
#     "verbosity": -1,
#     "random_state": RANDOM_STATE,
#     "n_estimators": 600,                # used for final fits (OOF uses higher cap + early stop)
#     "learning_rate": 0.05,
#     "num_leaves": 64,
#     "max_depth": -1,
#     "min_child_samples": 30,
#     "feature_fraction": 0.9,
#     "bagging_fraction": 0.9,
#     "bagging_freq": 1,
#     "lambda_l1": 0.0,
#     "lambda_l2": 0.0,
#     "n_jobs": os.cpu_count() or 1,
#     "num_threads": os.cpu_count() or 1,  # <-- added
# }

# # Meta (num_class set dynamically)
# DEFAULT_META_BASE = {
#     "objective": "multiclass",
#     "boosting_type": "gbdt",
#     "verbosity": -1,
#     "random_state": RANDOM_STATE,
#     "n_estimators": 500,                # used for final fits (train meta with early stop in final stage)
#     "learning_rate": 0.05,
#     "num_leaves": 64,
#     "max_depth": -1,
#     "min_child_samples": 30,
#     "feature_fraction": 0.9,
#     "bagging_fraction": 0.9,
#     "bagging_freq": 1,
#     "lambda_l1": 0.0,
#     "lambda_l2": 0.0,
#     "class_weight": "balanced",
#     "n_jobs": os.cpu_count() or 1,
#     "num_threads": os.cpu_count() or 1,  # <-- added
# }

# # -----------------------------
# # Helpers
# # -----------------------------
# def rmse(y_true, y_pred):
#     return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

# def save_text(path: str, text: str):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(text)

# def ensure_category(df: pd.DataFrame, col: str):
#     if col in df.columns:
#         df[col] = df[col].astype("category")

# def find_y_columns_for_thresholds(df: pd.DataFrame, thr_list: List[int]) -> Dict[int, str]:
#     pattern = re.compile(r"^y_(\d+(?:\.\d+)?)R$")
#     candidates = []
#     for c in df.columns:
#         m = pattern.match(c)
#         if m is not None:
#             v = float(m.group(1))
#             candidates.append((c, v))
#     if not candidates:
#         raise KeyError("No y_*R columns found. Expected columns like y_1R, y_2R, y_3R or y_1.2R, etc.")
#     mapping = {}
#     for t in thr_list:
#         best = min(candidates, key=lambda cv: abs(cv[1] - t))
#         mapping[t] = best[0]
#     return mapping

# def tri_class_remap(y: pd.Series) -> Tuple[np.ndarray, Dict[int,int], Dict[int,int]]:
#     fwd = {-1: 0, 0: 1, 1: 2}
#     inv = {v: k for k, v in fwd.items()}
#     y_enc = y.map(fwd).astype("int32").values
#     return y_enc, fwd, inv

# def inv_freq_weights_multi(y_enc: np.ndarray) -> Dict[int, float]:
#     vals, counts = np.unique(y_enc, return_counts=True)
#     N, K = len(y_enc), len(vals)
#     w = {}
#     for v, c in zip(vals, counts):
#         w[int(v)] = float(N / (K * max(c, 1)))
#     return w

# def dump_importance(model, cols, path, title):
#     if hasattr(model, "feature_importances_"):
#         imp = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}) \
#               .sort_values("Importance", ascending=False)
#         imp.to_csv(path, index=False)
#         print(f"{title} top10:\n", imp.head(10), "\n")
#         return imp
#     else:
#         print(f"⚠️ No feature_importances_ for {title}")
#         return pd.DataFrame()

# def report_multiclass(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, label_names: List[str]) -> str:
#     rep = classification_report(y_true_enc, y_pred_enc, target_names=label_names, digits=4, zero_division=0)
#     cf  = confusion_matrix(y_true_enc, y_pred_enc)
#     return rep + "\nConfusion Matrix:\n" + str(cf)

# # -----------------------------
# # OOF helpers  (with EARLY STOP)
# # -----------------------------
# def oof_multiclass(base_estimator, X, y_enc, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
#     """
#     Returns OOF probabilities (N,3) and trained fold models.
#     Uses a high n_estimators cap with early stopping on each fold.
#     """
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros((len(X), 3), dtype=float)
#     models = []
#     for tr, va in skf.split(X, y_enc):
#         m = clone(base_estimator)
#         m.set_params(n_estimators=2000)  # high cap; will early stop
#         m.fit(
#             X.iloc[tr], y_enc[tr],
#             eval_set=[(X.iloc[va], y_enc[va])],
#             categorical_feature=['pair'],
#             callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
#         )
#         oof[va, :] = m.predict_proba(X.iloc[va])
#         models.append(m)
#     return oof, models

# def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in kf.split(X, y):
#         m = clone(base_estimator)
#         m.set_params(n_estimators=2000)  # high cap; will early stop
#         m.fit(
#             X.iloc[tr], y.iloc[tr],
#             eval_set=[(X.iloc[va], y.iloc[va])],
#             categorical_feature=['pair'],
#             callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
#         )
#         oof[va] = m.predict(X.iloc[va])
#         models.append(m)
#     return oof, models

# # -----------------------------
# # Feature builders (stacking)
# # -----------------------------
# def add_base_probs_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
#     Xf = X.copy()
#     for t, probs in prob_dict.items():
#         p_neg1 = probs[:, 0]; p_0 = probs[:, 1]; p_1 = probs[:, 2]
#         Xf[f'clf_{t}R_pneg1'] = p_neg1
#         Xf[f'clf_{t}R_p0']    = p_0
#         Xf[f'clf_{t}R_p1']    = p_1
#         Xf[f'clf_{t}R_ev']    = (-1.0 * p_neg1) + (0.0 * p_0) + (1.0 * p_1)
#     return Xf

# def build_meta_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray], reg_vec: np.ndarray) -> pd.DataFrame:
#     Xf = add_base_probs_features(X, prob_dict)
#     Xf['reg_pred'] = reg_vec
#     return Xf

# # -----------------------------
# # Load & prepare
# # -----------------------------
# df = pd.read_csv(DATA_PATH)

# # Ensure pair is categorical
# ensure_category(df, 'pair')

# # ----- Cast numerics to float32, keep only 'pair' as category -----
# exclude_cols = {'pair', 'rr_label'}
# exclude_cols.update([c for c in df.columns if re.match(r"^y_(\d+(\.\d+)?)R$", c)])
# num_cols = [c for c in df.columns if c not in exclude_cols]
# # Only cast columns that are numeric-like
# df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype('float32')
# # ------------------------------------------------------------------

# # Map thresholds -> available y_*R columns
# thr_to_col = find_y_columns_for_thresholds(df, THRESHOLDS)

# # Keep only rows where all needed y_*R columns exist (drop NaNs across all three)
# needed_cols = [thr_to_col[t] for t in THRESHOLDS]
# mask_all = np.ones(len(df), dtype=bool)
# for c in needed_cols:
#     mask_all &= df[c].notna().values
# df = df.loc[mask_all].reset_index(drop=True)

# # Targets
# rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)

# # Meta 4-class: 0:<1R, 1:[1,2), 2:[2,3), 3:>=3
# bins_meta = [-np.inf, 1, 2, 3, np.inf]
# y_meta4 = pd.cut(rr, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # Build feature list: exclude targets & rr_label & any y_*R columns
# drop_cols = set(['rr_label'])
# for c in df.columns:
#     if re.match(r"^y_(\d+(\.\d+)?)R$", c):
#         drop_cols.add(c)
# features = [c for c in df.columns if c not in drop_cols]

# # Split (stratify by meta action)
# X_train, X_test, y_meta_train, y_meta_test = train_test_split(
#     df[features], y_meta4, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta4
# )

# # Also keep aligned full rr_label and per-threshold labels for train/test
# rr_train = rr.iloc[X_train.index]
# rr_test  = rr.iloc[X_test.index]

# y_thr_train = {t: df.loc[X_train.index, thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# y_thr_test  = {t: df.loc[X_test.index,  thr_to_col[t]].astype('int32') for t in THRESHOLDS}

# # -----------------------------
# # 1) Base tri-class classifiers (with OOF early stopping)
# # -----------------------------
# best_params_base = {}
# oof_probs_train = {}    # {t: (N_train,3)}
# models_base = {}        # {t: [fold models]}
# label_names_triclass = ["loss(-1)", "neutral(0)", "win(+1)"]

# for t in THRESHOLDS:
#     print(f"\n>>> Training base multiclass for {t}R using column '{thr_to_col[t]}' (early stopping OOF)")
#     y_enc_train, fwd_map, inv_map = tri_class_remap(y_thr_train[t])

#     params_t = DEFAULT_MC_PARAMS.copy()
#     best_params_base[t] = params_t

#     base = LGBMClassifier(**params_t)
#     oof_probs, models = oof_multiclass(base, X_train, y_enc_train)
#     oof_probs_train[t] = oof_probs
#     models_base[t] = models

#     # Holdout evaluation with fold-ensemble
#     p_test = np.mean([m.predict_proba(X_test) for m in models], axis=0)
#     y_test_enc, _, _ = tri_class_remap(y_thr_test[t])
#     y_pred_enc = np.argmax(p_test, axis=1)

#     rep = report_multiclass(y_test_enc, y_pred_enc, label_names_triclass)
#     print(f"\n=== BASE {t}R (multiclass -1/0/1) ===\n{rep}")
#     save_text(f"{MODEL_DIR}eval_base_{t}R.txt", rep)

# # -----------------------------
# # 2) Regressor on rr_label using OOF probs (with early stopping)
# # -----------------------------
# X_train_reg = add_base_probs_features(X_train, oof_probs_train)

# best_params_reg = DEFAULT_REG_PARAMS.copy()
# reg_base = LGBMRegressor(**best_params_reg)

# oof_reg_train, reg_models = oof_regressor(reg_base, X_train_reg, rr_train)

# # Holdout eval (build test-side features using fold models)
# test_probs = {t: np.mean([m.predict_proba(X_test) for m in models_base[t]], axis=0)
#               for t in THRESHOLDS}
# X_test_reg  = add_base_probs_features(X_test, test_probs)

# regressor   = LGBMRegressor(**best_params_reg).fit(
#     X_train_reg, rr_train,
#     categorical_feature=['pair']
# )
# reg_pred    = regressor.predict(X_test_reg)

# reg_report  = (
#     f"MAE:  {mean_absolute_error(rr_test, reg_pred):.6f}\n"
#     f"RMSE: {rmse(rr_test, reg_pred):.6f}\n"
#     f"R2:   {r2_score(rr_test, reg_pred):.6f}\n"
# )
# print("\n=== REGRESSOR (rr_label) ===")
# print(reg_report)
# save_text(f"{MODEL_DIR}eval_regressor.txt", reg_report)

# # -----------------------------
# # 3) Meta model (0,1,2,3) using base probs + reg
# # -----------------------------
# X_train_meta = build_meta_features(X_train, oof_probs_train, oof_reg_train)

# meta_num_class = int(len(np.unique(y_meta_train)))
# best_params_meta = DEFAULT_META_BASE.copy()
# best_params_meta["num_class"] = meta_num_class

# meta_model = LGBMClassifier(**best_params_meta).fit(
#     X_train_meta, y_meta_train,
#     categorical_feature=['pair']
# )

# X_test_meta  = build_meta_features(X_test, test_probs, reg_pred)
# meta_pred    = meta_model.predict(X_test_meta)

# meta_names = ["Reject(0)", "Target1R(1)", "Target2R(2)", "Target3R(3)"][:meta_num_class]
# meta_rep = classification_report(y_meta_test, meta_pred, target_names=meta_names, digits=4, zero_division=0)
# meta_cf  = confusion_matrix(y_meta_test, meta_pred)
# meta_txt = meta_rep + "\nConfusion Matrix:\n" + str(meta_cf)
# print("\n=== META (0/1/2/3) ===")
# print(meta_txt)
# save_text(f"{MODEL_DIR}eval_meta.txt", meta_txt)

# # -----------------------------
# # 4) Train final single models on ALL data (with early stopping)
# # -----------------------------
# print("\nTraining final models on ALL data...")

# df_full = df.copy()
# X_full  = df_full[features]
# ensure_category(X_full, 'pair')

# # final targets
# y_full_thr = {t: df_full[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# y_full_thr_enc = {t: tri_class_remap(y_full_thr[t])[0] for t in THRESHOLDS}
# rr_full     = pd.to_numeric(df_full['rr_label'], errors='coerce').fillna(-1.0)
# y_meta_full = pd.cut(rr_full, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # simple 90/10 split for early-stopping eval in final fits
# idx = np.arange(len(X_full))
# np.random.seed(RANDOM_STATE); np.random.shuffle(idx)
# cut = int(0.9 * len(idx))
# tr_idx, va_idx = idx[:cut], idx[cut:]

# # Final base models (early stopping)
# final_base = {}
# for t in THRESHOLDS:
#     print(f"Fitting final base multiclass {t}R with early stopping ...")
#     m = LGBMClassifier(**best_params_base[t])
#     m.set_params(n_estimators=2000)  # high cap; early stop
#     m.fit(
#         X_full.iloc[tr_idx], y_full_thr_enc[t][tr_idx],
#         eval_set=[(X_full.iloc[va_idx], y_full_thr_enc[t][va_idx])],
#         categorical_feature=['pair'],
#         callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
#     )
#     final_base[t] = m

# # Final regressor features/preds
# p_full = {t: final_base[t].predict_proba(X_full) for t in THRESHOLDS}
# X_full_reg = add_base_probs_features(X_full, p_full)

# print("Fitting final regressor with early stopping ...")
# reg_final  = LGBMRegressor(**best_params_reg)
# reg_final.set_params(n_estimators=2000)
# reg_final.fit(
#     X_full_reg.iloc[tr_idx], rr_full.iloc[tr_idx],
#     eval_set=[(X_full_reg.iloc[va_idx], rr_full.iloc[va_idx])],
#     categorical_feature=['pair'],
#     callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
# )

# # Final meta features/model
# reg_full_pred = reg_final.predict(X_full_reg)
# X_full_meta   = build_meta_features(X_full, p_full, reg_full_pred)

# meta_num_class_full = int(len(np.unique(y_meta_full)))
# meta_params_full = best_params_meta.copy()
# meta_params_full["num_class"] = meta_num_class_full

# print("Fitting final meta with early stopping ...")
# meta_final    = LGBMClassifier(**meta_params_full)
# meta_final.set_params(n_estimators=2000)
# meta_final.fit(
#     X_full_meta.iloc[tr_idx], y_meta_full[tr_idx],
#     eval_set=[(X_full_meta.iloc[va_idx], y_meta_full[va_idx])],
#     categorical_feature=['pair'],
#     callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
# )

# # -----------------------------
# # 5) Save artifacts & metadata
# # -----------------------------
# for t, model in final_base.items():
#     model.booster_.save_model(f"{MODEL_DIR}classifier_{t}R.txt")
# reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
# meta_final.booster_.save_model(f"{MODEL_DIR}meta_model.txt")

# metadata = {
#     "features": features,
#     "thresholds": THRESHOLDS,
#     "thr_to_col": thr_to_col,
#     "label_names_triclass": ["loss(-1)","neutral(0)","win(+1)"],
#     "meta_names": ["Reject(0)","Target1R(1)","Target2R(2)","Target3R(3)"],
#     "best_params": {
#         **{f"mc_{t}R": DEFAULT_MC_PARAMS for t in THRESHOLDS},
#         "reg":  DEFAULT_REG_PARAMS,
#         "meta": {k: v for k, v in DEFAULT_META_BASE.items() if k != "num_class"} | {"num_class": meta_num_class_full},
#     },
#     "notes": "Added early stopping for OOF and final fits; forced num_threads; cast numerics to float32; kept 'pair' as category."
# }
# joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

# # -----------------------------
# # 6) Dump feature importances
# # -----------------------------
# for t, model in final_base.items():
#     dump_importance(model, features, f"{MODEL_DIR}feature_importance_base_{t}R.csv", f"Base {t}R")

# reg_feature_names = X_full_reg.columns.tolist()
# dump_importance(reg_final, reg_feature_names, f"{MODEL_DIR}feature_importance_regressor.csv", "Regressor")

# meta_feature_names = X_full_meta.columns.tolist()
# dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}feature_importance_meta.csv", "Meta")

# print("\n✅ All models trained, evaluated, and saved.")
# print(f"Artifacts written to: {os.path.abspath(MODEL_DIR)}")
# print("Threads -> OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"), "| num_threads:", os.cpu_count())

# # # ==============================#
# # #  STACKED 3×MULTICLASS (±1/0/1)
# # #  + REGRESSOR (rr_label)
# # #  + META (0,1,2,3)
# # #  (No Optuna; fast defaults)
# # # ==============================#

# # import os
# # import re
# # import json
# # import joblib
# # import numpy as np
# # import pandas as pd
# # from typing import Dict, List, Tuple

# # from lightgbm import LGBMClassifier, LGBMRegressor

# # from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# # from sklearn.metrics import (
# #     classification_report, confusion_matrix, roc_auc_score,
# #     f1_score, accuracy_score,
# #     mean_absolute_error, mean_squared_error, r2_score
# # )
# # from sklearn.base import clone

# # # -----------------------------
# # # Config
# # # -----------------------------
# # DATA_PATH = "test-combined.csv"       # <- your labeled dataset
# # MODEL_DIR = "./tmodel_artifacts_3xmc_meta_v1/"
# # os.makedirs(MODEL_DIR, exist_ok=True)

# # RANDOM_STATE = 42
# # # Removed N_TRIALS_* (no Optuna)
# # N_SPLITS_OOF  = 3
# # N_SPLITS_CV   = 3

# # THRESHOLDS = [1, 2, 3]   # target R:R thresholds

# # # -----------------------------
# # # Fast default LightGBM params (no tuning)
# # # -----------------------------
# # # Multiclass base (3 classes: {-1,0,1} -> {0,1,2})
# # DEFAULT_MC_PARAMS = {
# #     "objective": "multiclass",
# #     "num_class": 3,
# #     "boosting_type": "gbdt",
# #     "verbosity": -1,
# #     "random_state": RANDOM_STATE,
# #     "n_estimators": 500,          # good speed/accuracy compromise
# #     "learning_rate": 0.05,
# #     "num_leaves": 64,
# #     "max_depth": -1,              # let leaves control complexity
# #     "min_child_samples": 30,
# #     "feature_fraction": 0.9,
# #     "bagging_fraction": 0.9,
# #     "bagging_freq": 1,
# #     "lambda_l1": 0.0,
# #     "lambda_l2": 0.0,
# #     "class_weight": "balanced",   # handles label skew
# #     "n_jobs": -1
# # }

# # # Regressor (rr_label)
# # DEFAULT_REG_PARAMS = {
# #     "objective": "regression",
# #     "boosting_type": "gbdt",
# #     "verbosity": -1,
# #     "random_state": RANDOM_STATE,
# #     "n_estimators": 600,
# #     "learning_rate": 0.05,
# #     "num_leaves": 64,
# #     "max_depth": -1,
# #     "min_child_samples": 30,
# #     "feature_fraction": 0.9,
# #     "bagging_fraction": 0.9,
# #     "bagging_freq": 1,
# #     "lambda_l1": 0.0,
# #     "lambda_l2": 0.0,
# #     "n_jobs": -1
# # }

# # # Meta (4-class by default; we’ll set num_class dynamically)
# # DEFAULT_META_BASE = {
# #     "objective": "multiclass",
# #     "boosting_type": "gbdt",
# #     "verbosity": -1,
# #     "random_state": RANDOM_STATE,
# #     "n_estimators": 500,
# #     "learning_rate": 0.05,
# #     "num_leaves": 64,
# #     "max_depth": -1,
# #     "min_child_samples": 30,
# #     "feature_fraction": 0.9,
# #     "bagging_fraction": 0.9,
# #     "bagging_freq": 1,
# #     "lambda_l1": 0.0,
# #     "lambda_l2": 0.0,
# #     "class_weight": "balanced",
# #     "n_jobs": -1
# # }

# # # -----------------------------
# # # Helpers
# # # -----------------------------
# # def rmse(y_true, y_pred):
# #     return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

# # def save_text(path: str, text: str):
# #     with open(path, "w", encoding="utf-8") as f:
# #         f.write(text)

# # def ensure_category(df: pd.DataFrame, col: str):
# #     if col in df.columns:
# #         df[col] = df[col].astype("category")

# # def find_y_columns_for_thresholds(df: pd.DataFrame, thr_list: List[int]) -> Dict[int, str]:
# #     """
# #     Map each threshold to an existing tri-class y column in df.
# #     Accepts columns like y_1R, y_1.2R, y_2.2R, y_3.2R, etc. Picks the closest (by float) per threshold.
# #     """
# #     pattern = re.compile(r"^y_(\d+(?:\.\d+)?)R$")
# #     candidates = []
# #     for c in df.columns:
# #         m = pattern.match(c)
# #         if m is not None:
# #             v = float(m.group(1))
# #             candidates.append((c, v))
# #     if not candidates:
# #         raise KeyError("No y_*R columns found. Expected columns like y_1R, y_2R, y_3R or y_1.2R, etc.")

# #     mapping = {}
# #     for t in thr_list:
# #         # find closest by |v - t|
# #         best = min(candidates, key=lambda cv: abs(cv[1] - t))
# #         mapping[t] = best[0]
# #     return mapping

# # def tri_class_remap(y: pd.Series) -> Tuple[np.ndarray, Dict[int,int], Dict[int,int]]:
# #     """
# #     LightGBM multiclass expects labels {0..K-1}.
# #     We map {-1,0,1} -> {0,1,2}.
# #     Returns encoded y, forward map, inverse map.
# #     """
# #     fwd = {-1: 0, 0: 1, 1: 2}
# #     inv = {v: k for k, v in fwd.items()}
# #     y_enc = y.map(fwd).astype("int32").values
# #     return y_enc, fwd, inv

# # def inv_freq_weights_multi(y_enc: np.ndarray) -> Dict[int, float]:
# #     vals, counts = np.unique(y_enc, return_counts=True)
# #     N, K = len(y_enc), len(vals)
# #     w = {}
# #     for v, c in zip(vals, counts):
# #         w[int(v)] = float(N / (K * max(c, 1)))
# #     return w

# # def dump_importance(model, cols, path, title):
# #     if hasattr(model, "feature_importances_"):
# #         imp = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}) \
# #               .sort_values("Importance", ascending=False)
# #         imp.to_csv(path, index=False)
# #         print(f"{title} top10:\n", imp.head(10), "\n")
# #         return imp
# #     else:
# #         print(f"⚠️ No feature_importances_ for {title}")
# #         return pd.DataFrame()

# # def report_multiclass(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, label_names: List[str]) -> str:
# #     rep = classification_report(y_true_enc, y_pred_enc, target_names=label_names, digits=4, zero_division=0)
# #     cf  = confusion_matrix(y_true_enc, y_pred_enc)
# #     return rep + "\nConfusion Matrix:\n" + str(cf)

# # # -----------------------------
# # # CV scorers (kept for structure; not used for tuning)
# # # -----------------------------
# # def cv_score_multiclass(params, X, y_enc, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     scores = []
# #     for tr, va in skf.split(X, y_enc):
# #         m = LGBMClassifier(**params)
# #         m.fit(X.iloc[tr], y_enc[tr], categorical_feature=['pair'])
# #         pred = m.predict(X.iloc[va])
# #         scores.append(f1_score(y_enc[va], pred, average="weighted"))
# #     return float(np.mean(scores))  # maximize

# # def cv_score_regressor(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
# #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     scores = []
# #     for tr, va in kf.split(X, y):
# #         m = LGBMRegressor(**params)
# #         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
# #         pred = m.predict(X.iloc[va])
# #         scores.append(rmse(y.iloc[va], pred))
# #     return float(np.mean(scores))  # lower is better

# # # -----------------------------
# # # OOF helpers
# # # -----------------------------
# # def oof_multiclass(base_estimator, X, y_enc, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
# #     """
# #     Returns OOF probabilities (N,3) and trained fold models.
# #     """
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     oof = np.zeros((len(X), 3), dtype=float)
# #     models = []
# #     for tr, va in skf.split(X, y_enc):
# #         m = clone(base_estimator)
# #         m.fit(X.iloc[tr], y_enc[tr], categorical_feature=['pair'])
# #         oof[va, :] = m.predict_proba(X.iloc[va])
# #         models.append(m)
# #     return oof, models

# # def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
# #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     oof = np.zeros(len(X), dtype=float)
# #     models = []
# #     for tr, va in kf.split(X, y):
# #         m = clone(base_estimator)
# #         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
# #         oof[va] = m.predict(X.iloc[va])
# #         models.append(m)
# #     return oof, models

# # # -----------------------------
# # # Feature builders (stacking)
# # # -----------------------------
# # def add_base_probs_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
# #     """
# #     prob_dict: {threshold -> (N,3) probs in class order [p(-1), p(0), p(1)]}
# #     Adds features:
# #       clf_T_pneg1, clf_T_p0, clf_T_p1, clf_T_ev  (where ev = -1*pneg1 + 0*p0 + 1*p1)
# #     """
# #     Xf = X.copy()
# #     for t, probs in prob_dict.items():
# #         p_neg1 = probs[:, 0]
# #         p_0    = probs[:, 1]
# #         p_1    = probs[:, 2]
# #         Xf[f'clf_{t}R_pneg1'] = p_neg1
# #         Xf[f'clf_{t}R_p0']    = p_0
# #         Xf[f'clf_{t}R_p1']    = p_1
# #         Xf[f'clf_{t}R_ev']    = (-1.0 * p_neg1) + (0.0 * p_0) + (1.0 * p_1)
# #     return Xf

# # def build_meta_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray], reg_vec: np.ndarray) -> pd.DataFrame:
# #     Xf = add_base_probs_features(X, prob_dict)
# #     Xf['reg_pred'] = reg_vec
# #     return Xf

# # # -----------------------------
# # # Load & prepare
# # # -----------------------------
# # df = pd.read_csv(DATA_PATH)

# # # Ensure pair is categorical
# # ensure_category(df, 'pair')

# # # Map thresholds -> available y_*R columns
# # thr_to_col = find_y_columns_for_thresholds(df, THRESHOLDS)

# # # Keep only rows where all needed y_*R columns exist (drop NaNs across all three)
# # needed_cols = [thr_to_col[t] for t in THRESHOLDS]
# # mask_all = np.ones(len(df), dtype=bool)
# # for c in needed_cols:
# #     mask_all &= df[c].notna().values
# # df = df.loc[mask_all].reset_index(drop=True)

# # # Targets
# # rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)

# # # Meta 4-class: 0:<1R, 1:[1,2), 2:[2,3), 3:>=3
# # bins_meta = [-np.inf, 1, 2, 3, np.inf]
# # y_meta4 = pd.cut(rr, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # # Build feature list: exclude targets & rr_label & any y_*R columns
# # drop_cols = set(['rr_label'])
# # for c in df.columns:
# #     if re.match(r"^y_(\d+(\.\d+)?)R$", c):
# #         drop_cols.add(c)
# # features = [c for c in df.columns if c not in drop_cols]

# # # Split (stratify by meta action)
# # X_train, X_test, y_meta_train, y_meta_test = train_test_split(
# #     df[features], y_meta4, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta4
# # )

# # # Also keep aligned full rr_label and per-threshold labels for train/test
# # rr_train = rr.iloc[X_train.index]
# # rr_test  = rr.iloc[X_test.index]

# # y_thr_train = {t: df.loc[X_train.index, thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# # y_thr_test  = {t: df.loc[X_test.index,  thr_to_col[t]].astype('int32') for t in THRESHOLDS}

# # # -----------------------------
# # # 1) Base tri-class classifiers (fast defaults)
# # # -----------------------------
# # best_params_base = {}
# # oof_probs_train = {}    # {t: (N_train,3)}
# # models_base = {}        # {t: [fold models]}
# # label_names_triclass = ["loss(-1)", "neutral(0)", "win(+1)"]

# # for t in THRESHOLDS:
# #     print(f"\n>>> Training base multiclass for {t}R using column '{thr_to_col[t]}' (no tuning)")
# #     y_enc_train, fwd_map, inv_map = tri_class_remap(y_thr_train[t])

# #     # Use defaults (copy to avoid accidental mutation)
# #     params_t = DEFAULT_MC_PARAMS.copy()
# #     best_params_base[t] = params_t

# #     base = LGBMClassifier(**params_t)
# #     oof_probs, models = oof_multiclass(base, X_train, y_enc_train)
# #     oof_probs_train[t] = oof_probs
# #     models_base[t] = models

# #     # Holdout evaluation with fold-ensemble
# #     p_test = np.mean([m.predict_proba(X_test) for m in models], axis=0)
# #     y_test_enc, _, _ = tri_class_remap(y_thr_test[t])
# #     y_pred_enc = np.argmax(p_test, axis=1)

# #     rep = report_multiclass(y_test_enc, y_pred_enc, label_names_triclass)
# #     print(f"\n=== BASE {t}R (multiclass -1/0/1) ===\n{rep}")
# #     save_text(f"{MODEL_DIR}eval_base_{t}R.txt", rep)

# # # -----------------------------
# # # 2) Regressor on rr_label using OOF probs (fast defaults)
# # # -----------------------------
# # X_train_reg = add_base_probs_features(X_train, oof_probs_train)

# # best_params_reg = DEFAULT_REG_PARAMS.copy()
# # reg_base = LGBMRegressor(**best_params_reg)

# # oof_reg_train, reg_models = oof_regressor(reg_base, X_train_reg, rr_train)

# # # Holdout eval (build test-side features using fold models)
# # test_probs = {t: np.mean([m.predict_proba(X_test) for m in models_base[t]], axis=0)
# #               for t in THRESHOLDS}
# # X_test_reg  = add_base_probs_features(X_test, test_probs)

# # regressor   = LGBMRegressor(**best_params_reg).fit(X_train_reg, rr_train, categorical_feature=['pair'])
# # reg_pred    = regressor.predict(X_test_reg)

# # reg_report  = (
# #     f"MAE:  {mean_absolute_error(rr_test, reg_pred):.6f}\n"
# #     f"RMSE: {rmse(rr_test, reg_pred):.6f}\n"
# #     f"R2:   {r2_score(rr_test, reg_pred):.6f}\n"
# # )
# # print("\n=== REGRESSOR (rr_label) ===")
# # print(reg_report)
# # save_text(f"{MODEL_DIR}eval_regressor.txt", reg_report)

# # # -----------------------------
# # # 3) Meta model (0,1,2,3) using base probs + reg (fast defaults)
# # # -----------------------------
# # X_train_meta = build_meta_features(X_train, oof_probs_train, oof_reg_train)

# # # set num_class based on training labels
# # meta_num_class = int(len(np.unique(y_meta_train)))
# # best_params_meta = DEFAULT_META_BASE.copy()
# # best_params_meta["num_class"] = meta_num_class

# # meta_model = LGBMClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, categorical_feature=['pair'])

# # X_test_meta  = build_meta_features(X_test, test_probs, reg_pred)
# # meta_pred    = meta_model.predict(X_test_meta)

# # meta_names = ["Reject(0)", "Target1R(1)", "Target2R(2)", "Target3R(3)"][:meta_num_class]
# # meta_rep = classification_report(y_meta_test, meta_pred, target_names=meta_names, digits=4, zero_division=0)
# # meta_cf  = confusion_matrix(y_meta_test, meta_pred)
# # meta_txt = meta_rep + "\nConfusion Matrix:\n" + str(meta_cf)
# # print("\n=== META (0/1/2/3) ===")
# # print(meta_txt)
# # save_text(f"{MODEL_DIR}eval_meta.txt", meta_txt)

# # # -----------------------------
# # # 4) Train final single models on ALL data
# # # -----------------------------
# # print("\nTraining final models on ALL data...")

# # # restrict full DF to rows used earlier (have all labels)
# # df_full = df.copy()
# # X_full  = df_full[features]
# # ensure_category(X_full, 'pair')

# # # full tri-class labels per threshold encoded
# # y_full_thr = {t: df_full[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# # y_full_thr_enc = {t: tri_class_remap(y_full_thr[t])[0] for t in THRESHOLDS}

# # # full rr / meta targets
# # rr_full     = pd.to_numeric(df_full['rr_label'], errors='coerce').fillna(-1.0)
# # y_meta_full = pd.cut(rr_full, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # # Final base models
# # final_base = {}
# # for t in THRESHOLDS:
# #     print(f"Fitting final base multiclass {t}R ...")
# #     m = LGBMClassifier(**best_params_base[t]).fit(X_full, y_full_thr_enc[t], categorical_feature=['pair'])
# #     final_base[t] = m

# # # Final regressor features/preds
# # p_full = {t: final_base[t].predict_proba(X_full) for t in THRESHOLDS}
# # X_full_reg = add_base_probs_features(X_full, p_full)
# # reg_final  = LGBMRegressor(**best_params_reg).fit(X_full_reg, rr_full, categorical_feature=['pair'])

# # # Final meta features/model
# # reg_full_pred = reg_final.predict(X_full_reg)
# # X_full_meta   = build_meta_features(X_full, p_full, reg_full_pred)

# # meta_num_class_full = int(len(np.unique(y_meta_full)))
# # meta_params_full = best_params_meta.copy()
# # meta_params_full["num_class"] = meta_num_class_full

# # meta_final    = LGBMClassifier(**meta_params_full).fit(X_full_meta, y_meta_full, categorical_feature=['pair'])

# # # -----------------------------
# # # 5) Save artifacts & metadata
# # # -----------------------------
# # for t, model in final_base.items():
# #     model.booster_.save_model(f"{MODEL_DIR}classifier_{t}R.txt")
# # reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
# # meta_final.booster_.save_model(f"{MODEL_DIR}meta_model.txt")

# # metadata = {
# #     "features": features,
# #     "thresholds": THRESHOLDS,
# #     "thr_to_col": thr_to_col,
# #     "label_names_triclass": ["loss(-1)","neutral(0)","win(+1)"],
# #     "meta_names": ["Reject(0)","Target1R(1)","Target2R(2)","Target3R(3)"],
# #     "best_params": {
# #         **{f"mc_{t}R": best_params_base[t] for t in THRESHOLDS},
# #         "reg":  best_params_reg,
# #         "meta": best_params_meta
# #     },
# #     "notes": "Tri-class base classifiers use mapping {-1,0,1}->{0,1,2} internally for LightGBM. No Optuna; default fast params."
# # }
# # joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

# # -----------------------------
# # 6) Dump feature importances
# # -----------------------------
# # Base models (full-data)
# # for t, model in final_base.items():
# #     dump_importance(model, features, f"{MODEL_DIR}feature_importance_base_{t}R.csv", f"Base {t}R")

# # # Regressor
# # reg_feature_names = X_full_reg.columns.tolist()
# # dump_importance(reg_final, reg_feature_names, f"{MODEL_DIR}feature_importance_regressor.csv", "Regressor")

# # # Meta
# # meta_feature_names = X_full_meta.columns.tolist()
# # dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}feature_importance_meta.csv", "Meta")

# # print("\n✅ All models trained, evaluated, and saved.")
# # print(f"Artifacts written to: {os.path.abspath(MODEL_DIR)}")

# # # ==============================#

# # #  STACKED 3×MULTICLASS (±1/0/1)
# # #  + REGRESSOR (rr_label)
# # #  + META (0,1,2,3)
# # # ==============================#

# # import os
# # import re
# # import json
# # import joblib
# # import numpy as np
# # import pandas as pd
# # import optuna
# # from typing import Dict, List, Tuple

# # from lightgbm import LGBMClassifier, LGBMRegressor

# # from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# # from sklearn.metrics import (
# #     classification_report, confusion_matrix, roc_auc_score,
# #     f1_score, accuracy_score,
# #     mean_absolute_error, mean_squared_error, r2_score
# # )
# # from sklearn.base import clone

# # # -----------------------------
# # # Config
# # # -----------------------------
# # DATA_PATH = "test-combined.csv"       # <- your labeled dataset
# # MODEL_DIR = "./tmodel_artifacts_3xmc_meta_v1/"
# # os.makedirs(MODEL_DIR, exist_ok=True)

# # RANDOM_STATE = 42
# # N_TRIALS_BASE = 50
# # N_TRIALS_REG  = 50
# # N_TRIALS_META = 50
# # N_SPLITS_OOF  = 3
# # N_SPLITS_CV   = 3

# # THRESHOLDS = [1, 2, 3]   # target R:R thresholds

# # # -----------------------------
# # # Helpers
# # # -----------------------------
# # def rmse(y_true, y_pred):
# #     return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

# # def save_text(path: str, text: str):
# #     with open(path, "w", encoding="utf-8") as f:
# #         f.write(text)

# # def ensure_category(df: pd.DataFrame, col: str):
# #     if col in df.columns:
# #         df[col] = df[col].astype("category")

# # def find_y_columns_for_thresholds(df: pd.DataFrame, thr_list: List[int]) -> Dict[int, str]:
# #     """
# #     Map each threshold to an existing tri-class y column in df.
# #     Accepts columns like y_1R, y_1.2R, y_2.2R, y_3.2R, etc. Picks the closest (by float) per threshold.
# #     """
# #     pattern = re.compile(r"^y_(\d+(?:\.\d+)?)R$")
# #     candidates = []
# #     for c in df.columns:
# #         m = pattern.match(c)
# #         if m is not None:
# #             v = float(m.group(1))
# #             candidates.append((c, v))
# #     if not candidates:
# #         raise KeyError("No y_*R columns found. Expected columns like y_1R, y_2R, y_3R or y_1.2R, etc.")

# #     mapping = {}
# #     for t in thr_list:
# #         # find closest by |v - t|
# #         best = min(candidates, key=lambda cv: abs(cv[1] - t))
# #         mapping[t] = best[0]
# #     return mapping

# # def tri_class_remap(y: pd.Series) -> Tuple[np.ndarray, Dict[int,int], Dict[int,int]]:
# #     """
# #     LightGBM multiclass expects labels {0..K-1}.
# #     We map {-1,0,1} -> {0,1,2}.
# #     Returns encoded y, forward map, inverse map.
# #     """
# #     fwd = {-1: 0, 0: 1, 1: 2}
# #     inv = {v: k for k, v in fwd.items()}
# #     y_enc = y.map(fwd).astype("int32").values
# #     return y_enc, fwd, inv

# # def inv_freq_weights_multi(y_enc: np.ndarray) -> Dict[int, float]:
# #     vals, counts = np.unique(y_enc, return_counts=True)
# #     N, K = len(y_enc), len(vals)
# #     w = {}
# #     for v, c in zip(vals, counts):
# #         w[int(v)] = float(N / (K * max(c, 1)))
# #     return w

# # def dump_importance(model, cols, path, title):
# #     if hasattr(model, "feature_importances_"):
# #         imp = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}) \
# #               .sort_values("Importance", ascending=False)
# #         imp.to_csv(path, index=False)
# #         print(f"{title} top10:\n", imp.head(10), "\n")
# #         return imp
# #     else:
# #         print(f"⚠️ No feature_importances_ for {title}")
# #         return pd.DataFrame()

# # def report_multiclass(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, label_names: List[str]) -> str:
# #     rep = classification_report(y_true_enc, y_pred_enc, target_names=label_names, digits=4, zero_division=0)
# #     cf  = confusion_matrix(y_true_enc, y_pred_enc)
# #     return rep + "\nConfusion Matrix:\n" + str(cf)

# # # -----------------------------
# # # CV scorers
# # # -----------------------------
# # def cv_score_multiclass(params, X, y_enc, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     scores = []
# #     for tr, va in skf.split(X, y_enc):
# #         m = LGBMClassifier(**params)
# #         m.fit(X.iloc[tr], y_enc[tr], categorical_feature=['pair'])
# #         pred = m.predict(X.iloc[va])
# #         scores.append(f1_score(y_enc[va], pred, average="weighted"))
# #     return float(np.mean(scores))  # maximize

# # def cv_score_regressor(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
# #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     scores = []
# #     for tr, va in kf.split(X, y):
# #         m = LGBMRegressor(**params)
# #         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
# #         pred = m.predict(X.iloc[va])
# #         scores.append(rmse(y.iloc[va], pred))
# #     return float(np.mean(scores))  # lower is better

# # # -----------------------------
# # # Optuna tuning
# # # -----------------------------
# # def tune_multiclass_params(X, y_enc, name="mc"):
# #     weight_grid = [
# #         None,
# #         "balanced",
# #         inv_freq_weights_multi(y_enc),
# #         {0:1.0, 1:1.2, 2:1.5},
# #         {0:1.5, 1:1.0, 2:1.5}
# #     ]
# #     def objective(trial):
# #         params = {
# #             "objective": "multiclass",
# #             "num_class": 3,
# #             "boosting_type": "gbdt",
# #             "verbosity": -1,
# #             "random_state": RANDOM_STATE,
# #             "n_estimators": trial.suggest_int("n_estimators", 200, 900),
# #             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
# #             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
# #             "max_depth": trial.suggest_int("max_depth", 3, 12),
# #             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
# #             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #             "class_weight": trial.suggest_categorical("class_weight", weight_grid),
# #         }
# #         return cv_score_multiclass(params, X, y_enc)
# #     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
# #     study.optimize(objective, n_trials=N_TRIALS_BASE)
# #     return study.best_params

# # def tune_regressor_params(X, y, name="reg"):
# #     def objective(trial):
# #         params = {
# #             "objective": "regression",
# #             "boosting_type": "gbdt",
# #             "verbosity": -1,
# #             "random_state": RANDOM_STATE,
# #             "n_estimators": trial.suggest_int("n_estimators", 200, 900),
# #             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
# #             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
# #             "max_depth": trial.suggest_int("max_depth", 3, 12),
# #             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
# #             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #         }
# #         return cv_score_regressor(params, X, y)
# #     study = optuna.create_study(direction="minimize", study_name=f"tune_{name}")
# #     study.optimize(objective, n_trials=N_TRIALS_REG)
# #     return study.best_params

# # def tune_meta_params(X, y_enc, name="meta"):
# #     weight_grid = [
# #         None,
# #         "balanced",
# #         inv_freq_weights_multi(y_enc),
# #         {0:1.0, 1:1.2, 2:1.4, 3:1.6}
# #     ]
# #     # num_class determined by unique(y_enc)
# #     num_class = int(len(np.unique(y_enc)))
# #     def objective(trial):
# #         params = {
# #             "objective": "multiclass",
# #             "num_class": num_class,
# #             "boosting_type": "gbdt",
# #             "verbosity": -1,
# #             "random_state": RANDOM_STATE,
# #             "n_estimators": trial.suggest_int("n_estimators", 200, 900),
# #             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
# #             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
# #             "max_depth": trial.suggest_int("max_depth", 3, 12),
# #             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
# #             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #             "class_weight": trial.suggest_categorical("class_weight", weight_grid),
# #         }
# #         skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
# #         f1s = []
# #         for tr, va in skf.split(X, y_enc):
# #             m = LGBMClassifier(**params)
# #             m.fit(X.iloc[tr], y_enc[tr], categorical_feature=['pair'])
# #             pred = m.predict(X.iloc[va])
# #             f1s.append(f1_score(y_enc[va], pred, average="weighted"))
# #         return float(np.mean(f1s))
# #     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
# #     study.optimize(objective, n_trials=N_TRIALS_META)
# #     return study.best_params

# # # -----------------------------
# # # OOF helpers
# # # -----------------------------
# # def oof_multiclass(base_estimator, X, y_enc, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
# #     """
# #     Returns OOF probabilities (N,3) and trained fold models.
# #     """
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     oof = np.zeros((len(X), 3), dtype=float)
# #     models = []
# #     for tr, va in skf.split(X, y_enc):
# #         m = clone(base_estimator)
# #         m.fit(X.iloc[tr], y_enc[tr], categorical_feature=['pair'])
# #         oof[va, :] = m.predict_proba(X.iloc[va])
# #         models.append(m)
# #     return oof, models

# # def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
# #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# #     oof = np.zeros(len(X), dtype=float)
# #     models = []
# #     for tr, va in kf.split(X, y):
# #         m = clone(base_estimator)
# #         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
# #         oof[va] = m.predict(X.iloc[va])
# #         models.append(m)
# #     return oof, models

# # # -----------------------------
# # # Feature builders (stacking)
# # # -----------------------------
# # def add_base_probs_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
# #     """
# #     prob_dict: {threshold -> (N,3) probs in class order [p(-1), p(0), p(1)]}
# #     Adds features:
# #       clf_T_pneg1, clf_T_p0, clf_T_p1, clf_T_ev  (where ev = -1*pneg1 + 0*p0 + 1*p1)
# #     """
# #     Xf = X.copy()
# #     for t, probs in prob_dict.items():
# #         p_neg1 = probs[:, 0]
# #         p_0    = probs[:, 1]
# #         p_1    = probs[:, 2]
# #         Xf[f'clf_{t}R_pneg1'] = p_neg1
# #         Xf[f'clf_{t}R_p0']    = p_0
# #         Xf[f'clf_{t}R_p1']    = p_1
# #         Xf[f'clf_{t}R_ev']    = (-1.0 * p_neg1) + (0.0 * p_0) + (1.0 * p_1)
# #     return Xf

# # def build_meta_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray], reg_vec: np.ndarray) -> pd.DataFrame:
# #     Xf = add_base_probs_features(X, prob_dict)
# #     Xf['reg_pred'] = reg_vec
# #     return Xf

# # # -----------------------------
# # # Load & prepare
# # # -----------------------------
# # df = pd.read_csv(DATA_PATH)

# # # Ensure pair is categorical
# # ensure_category(df, 'pair')

# # # Map thresholds -> available y_*R columns
# # thr_to_col = find_y_columns_for_thresholds(df, THRESHOLDS)

# # # Keep only rows where all needed y_*R columns exist (drop NaNs across all three)
# # needed_cols = [thr_to_col[t] for t in THRESHOLDS]
# # mask_all = np.ones(len(df), dtype=bool)
# # for c in needed_cols:
# #     mask_all &= df[c].notna().values
# # df = df.loc[mask_all].reset_index(drop=True)

# # # Targets
# # rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)

# # # Meta 4-class: 0:<1R, 1:[1,2), 2:[2,3), 3:>=3
# # bins_meta = [-np.inf, 1, 2, 3, np.inf]
# # y_meta4 = pd.cut(rr, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # # Build feature list: exclude targets & rr_label & any y_*R columns
# # drop_cols = set(['rr_label'])
# # for c in df.columns:
# #     if re.match(r"^y_(\d+(\.\d+)?)R$", c):
# #         drop_cols.add(c)
# # features = [c for c in df.columns if c not in drop_cols]

# # # Split (stratify by meta action)
# # X_train, X_test, y_meta_train, y_meta_test = train_test_split(
# #     df[features], y_meta4, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta4
# # )

# # # Also keep aligned full rr_label and per-threshold labels for train/test
# # rr_train = rr.iloc[X_train.index]
# # rr_test  = rr.iloc[X_test.index]

# # y_thr_train = {t: df.loc[X_train.index, thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# # y_thr_test  = {t: df.loc[X_test.index,  thr_to_col[t]].astype('int32') for t in THRESHOLDS}

# # # -----------------------------
# # # 1) Tune + OOF for each base tri-class classifier
# # # -----------------------------
# # best_params_base = {}
# # oof_probs_train = {}    # {t: (N_train,3)}
# # models_base = {}        # {t: [fold models]}
# # label_names_triclass = ["loss(-1)", "neutral(0)", "win(+1)"]

# # for t in THRESHOLDS:
# #     print(f"\n>>> Tuning base multiclass for {t}R using column '{thr_to_col[t]}'")
# #     y_enc_train, fwd_map, inv_map = tri_class_remap(y_thr_train[t])
# #     params_t = tune_multiclass_params(X_train, y_enc_train, name=f"mc_{t}R")
# #     best_params_base[t] = params_t

# #     base = LGBMClassifier(**params_t)
# #     oof_probs, models = oof_multiclass(base, X_train, y_enc_train)
# #     oof_probs_train[t] = oof_probs
# #     models_base[t] = models

# #     # Holdout evaluation
# #     # build test probs by fold-ensemble
# #     p_test = np.mean([m.predict_proba(X_test) for m in models], axis=0)
# #     y_test_enc, _, _ = tri_class_remap(y_thr_test[t])
# #     y_pred_enc = np.argmax(p_test, axis=1)

# #     rep = report_multiclass(y_test_enc, y_pred_enc, label_names_triclass)
# #     print(f"\n=== BASE {t}R (multiclass -1/0/1) ===\n{rep}")
# #     save_text(f"{MODEL_DIR}eval_base_{t}R.txt", rep)

# # # -----------------------------
# # # 2) Regressor on rr_label using OOF probs
# # # -----------------------------
# # X_train_reg = add_base_probs_features(X_train, oof_probs_train)
# # best_params_reg = tune_regressor_params(X_train_reg, rr_train, name="reg_rr")
# # reg_base = LGBMRegressor(**best_params_reg)

# # oof_reg_train, reg_models = oof_regressor(reg_base, X_train_reg, rr_train)

# # # Holdout eval (build test-side features using fold models)
# # test_probs = {t: np.mean([m.predict_proba(X_test) for m in models_base[t]], axis=0)
# #               for t in THRESHOLDS}
# # X_test_reg  = add_base_probs_features(X_test, test_probs)
# # regressor   = LGBMRegressor(**best_params_reg).fit(X_train_reg, rr_train, categorical_feature=['pair'])
# # reg_pred    = regressor.predict(X_test_reg)

# # reg_report  = (
# #     f"MAE:  {mean_absolute_error(rr_test, reg_pred):.6f}\n"
# #     f"RMSE: {rmse(rr_test, reg_pred):.6f}\n"
# #     f"R2:   {r2_score(rr_test, reg_pred):.6f}\n"
# # )
# # print("\n=== REGRESSOR (rr_label) ===")
# # print(reg_report)
# # save_text(f"{MODEL_DIR}eval_regressor.txt", reg_report)

# # # -----------------------------
# # # 3) Meta model (0,1,2,3) using base probs + reg
# # # -----------------------------
# # X_train_meta = build_meta_features(X_train, oof_probs_train, oof_reg_train)
# # best_params_meta = tune_meta_params(X_train_meta, y_meta_train, name="meta_4class")
# # meta_model = LGBMClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, categorical_feature=['pair'])

# # X_test_meta  = build_meta_features(X_test, test_probs, reg_pred)
# # meta_pred    = meta_model.predict(X_test_meta)

# # meta_names = ["Reject(0)", "Target1R(1)", "Target2R(2)", "Target3R(3)"][:int(len(np.unique(y_meta_train)))]
# # meta_rep = classification_report(y_meta_test, meta_pred, target_names=meta_names, digits=4, zero_division=0)
# # meta_cf  = confusion_matrix(y_meta_test, meta_pred)
# # meta_txt = meta_rep + "\nConfusion Matrix:\n" + str(meta_cf)
# # print("\n=== META (0/1/2/3) ===")
# # print(meta_txt)
# # save_text(f"{MODEL_DIR}eval_meta.txt", meta_txt)

# # # -----------------------------
# # # 4) Train final single models on ALL data
# # # -----------------------------
# # print("\nTraining final models on ALL data...")

# # # restrict full DF to rows used earlier (have all labels)
# # df_full = df.copy()
# # X_full  = df_full[features]
# # ensure_category(X_full, 'pair')

# # # full tri-class labels per threshold encoded
# # y_full_thr = {t: df_full[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
# # y_full_thr_enc = {t: tri_class_remap(y_full_thr[t])[0] for t in THRESHOLDS}

# # # full rr / meta targets
# # rr_full     = pd.to_numeric(df_full['rr_label'], errors='coerce').fillna(-1.0)
# # y_meta_full = pd.cut(rr_full, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values

# # # Final base models
# # final_base = {}
# # for t in THRESHOLDS:
# #     print(f"Fitting final base multiclass {t}R ...")
# #     m = LGBMClassifier(**best_params_base[t]).fit(X_full, y_full_thr_enc[t], categorical_feature=['pair'])
# #     final_base[t] = m

# # # Final regressor features/preds
# # p_full = {t: final_base[t].predict_proba(X_full) for t in THRESHOLDS}
# # X_full_reg = add_base_probs_features(X_full, p_full)
# # reg_final  = LGBMRegressor(**best_params_reg).fit(X_full_reg, rr_full, categorical_feature=['pair'])

# # # Final meta features/model
# # reg_full_pred = reg_final.predict(X_full_reg)
# # X_full_meta   = build_meta_features(X_full, p_full, reg_full_pred)
# # meta_final    = LGBMClassifier(**best_params_meta).fit(X_full_meta, y_meta_full, categorical_feature=['pair'])

# # # -----------------------------
# # # 5) Save artifacts & metadata
# # # -----------------------------
# # for t, model in final_base.items():
# #     model.booster_.save_model(f"{MODEL_DIR}classifier_{t}R.txt")
# # reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
# # meta_final.booster_.save_model(f"{MODEL_DIR}meta_model.txt")

# # metadata = {
# #     "features": features,
# #     "thresholds": THRESHOLDS,
# #     "thr_to_col": thr_to_col,
# #     "label_names_triclass": ["loss(-1)","neutral(0)","win(+1)"],
# #     "meta_names": ["Reject(0)","Target1R(1)","Target2R(2)","Target3R(3)"],
# #     "best_params": {
# #         **{f"mc_{t}R": best_params_base[t] for t in THRESHOLDS},
# #         "reg":  best_params_reg,
# #         "meta": best_params_meta
# #     },
# #     "notes": "Tri-class base classifiers use mapping {-1,0,1}->{0,1,2} internally for LightGBM."
# # }
# # joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

# # # -----------------------------
# # # 6) Dump feature importances
# # # -----------------------------
# # # Base models (full-data)
# # for t, model in final_base.items():
# #     dump_importance(model, features, f"{MODEL_DIR}feature_importance_base_{t}R.csv", f"Base {t}R")

# # # Regressor
# # reg_feature_names = X_full_reg.columns.tolist()
# # dump_importance(reg_final, reg_feature_names, f"{MODEL_DIR}feature_importance_regressor.csv", "Regressor")

# # # Meta
# # meta_feature_names = X_full_meta.columns.tolist()
# # dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}feature_importance_meta.csv", "Meta")

# # print("\n✅ All models trained, evaluated, and saved.")
# # print(f"Artifacts written to: {os.path.abspath(MODEL_DIR)}")
# # # 