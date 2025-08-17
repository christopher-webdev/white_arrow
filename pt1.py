# ==============================#
#  STACKED ENSEMBLE WITH REGRESSION META
#  - 3 Classifiers (1R, 2R, 3R)
#  - 1 Regressor (RR prediction)
#  - 1 Meta Regressor (final RR prediction)
# ==============================#

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.base import clone

# ---------------------------------
# Config
# ---------------------------------
DATA_PATH = "test-combined.csv"
MODEL_DIR = "./stacked_ensemble_regression_meta/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_OOF = 3
THRESHOLDS = [1, 2, 3]

# Set threading
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------
# Model Parameters
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
    "class_weight": None,
    "n_jobs": -1,
    "force_col_wise": True
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
    "n_jobs": -1,
    "force_col_wise": True
}

# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def ensure_category(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = df[col].astype("category")

def find_y_columns_for_thresholds(df: pd.DataFrame, thr_list: List[int]) -> Dict[int, str]:
    pattern = re.compile(r"^y_(\d+(?:\.\d+)?)R$")
    candidates = []
    for c in df.columns:
        m = pattern.match(c)
        if m is not None:
            v = float(m.group(1))
            candidates.append((c, v))
    if not candidates:
        raise KeyError("No y_*R columns found. Expected columns like y_1R, y_2R, y_3R or y_1.2R, etc.")
    mapping = {}
    for t in thr_list:
        best = min(candidates, key=lambda cv: abs(cv[1] - t))
        mapping[t] = best[0]
    return mapping

def tri_class_remap(y: pd.Series) -> Tuple[np.ndarray, Dict[int,int], Dict[int,int]]:
    fwd = {-1: 0, 0: 1, 1: 2}
    inv = {v: k for k, v in fwd.items()}
    y_enc = y.map(fwd).astype("int32").values
    return y_enc, fwd, inv

def report_multiclass(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, label_names: List[str]) -> str:
    rep = classification_report(y_true_enc, y_pred_enc, target_names=label_names, digits=4, zero_division=0)
    cf  = confusion_matrix(y_true_enc, y_pred_enc)
    return rep + "\nConfusion Matrix:\n" + str(cf)

# -----------------------------
# OOF Helpers
# -----------------------------
def oof_multiclass(base_estimator, X, y_enc, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
    print(f"Generating OOF predictions for classifier...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), 3), dtype=float)
    models = []
    for tr, va in skf.split(X, y_enc):
        m = clone(base_estimator)
        m.set_params(n_estimators=2000)
        m.fit(
            X.iloc[tr], y_enc[tr],
            eval_set=[(X.iloc[va], y_enc[va])],
            categorical_feature=['pair'],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )
        oof[va, :] = m.predict_proba(X.iloc[va])
        models.append(m)
    return oof, models

def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE) -> Tuple[np.ndarray, List]:
    print(f"Generating OOF predictions for regressor...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    models = []
    for tr, va in kf.split(X, y):
        m = clone(base_estimator)
        m.set_params(n_estimators=2000)
        m.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            categorical_feature=['pair'],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
    return oof, models

# -----------------------------
# Feature Builders
# -----------------------------
def add_base_probs_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray]) -> pd.DataFrame:
    Xf = X.copy()
    for t, probs in prob_dict.items():
        p_neg1 = probs[:, 0]; p_0 = probs[:, 1]; p_1 = probs[:, 2]
        Xf[f'clf_{t}R_pneg1'] = p_neg1
        Xf[f'clf_{t}R_p0']    = p_0
        Xf[f'clf_{t}R_p1']    = p_1
        Xf[f'clf_{t}R_ev']    = (-1.0 * p_neg1) + (0.0 * p_0) + (1.0 * p_1)
    return Xf

def build_meta_features(X: pd.DataFrame, prob_dict: Dict[int, np.ndarray], reg_vec: np.ndarray) -> pd.DataFrame:
    Xf = add_base_probs_features(X, prob_dict)
    Xf['reg_pred'] = reg_vec
    
    # Add original features (excluding pair if it's categorical)
    original_features = [col for col in X.columns if col != 'pair']
    for col in original_features:
        Xf[col] = X[col]
        
    return Xf

# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    ensure_category(df, 'pair')
    
    exclude_cols = {'pair', 'rr_label'}
    exclude_cols.update([c for c in df.columns if re.match(r"^y_(\d+(\.\d+)?)R$", c)])
    num_cols = [c for c in df.columns if c not in exclude_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype('float32')
    
    thr_to_col = find_y_columns_for_thresholds(df, THRESHOLDS)
    needed_cols = [thr_to_col[t] for t in THRESHOLDS]
    mask_all = np.ones(len(df), dtype=bool)
    for c in needed_cols:
        mask_all &= df[c].notna().values
    df = df.loc[mask_all].reset_index(drop=True)
    
    rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
    bins_meta = [-np.inf, 1, 2, 3, np.inf]
    y_meta4 = pd.cut(rr, bins=bins_meta, labels=False, right=False, include_lowest=True).astype('int32').values
    
    drop_cols = set(['rr_label'])
    for c in df.columns:
        if re.match(r"^y_(\d+(\.\d+)?)R$", c):
            drop_cols.add(c)
    features = [c for c in df.columns if c not in drop_cols]
    
    X_train, X_test, y_meta_train, y_meta_test = train_test_split(
        df[features], y_meta4, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta4
    )
    rr_train = rr.iloc[X_train.index]
    rr_test  = rr.iloc[X_test.index]
    y_thr_train = {t: df.loc[X_train.index, thr_to_col[t]].astype('int32') for t in THRESHOLDS}
    y_thr_test  = {t: df.loc[X_test.index, thr_to_col[t]].astype('int32') for t in THRESHOLDS}

    # -----------------------------
    # 1) Train Base Classifiers with OOF
    # -----------------------------
    print("\n=== Training Base Classifiers ===")
    best_params_base = {}
    oof_probs_train = {}
    models_base = {}
    label_names_triclass = ["loss(-1)", "neutral(0)", "win(+1)"]

    for t in THRESHOLDS:
        print(f"\n>>> Training base multiclass for {t}R using column '{thr_to_col[t]}'")
        y_enc_train, fwd_map, inv_map = tri_class_remap(y_thr_train[t])
        base = LGBMClassifier(**DEFAULT_MC_PARAMS)
        oof_probs, models = oof_multiclass(base, X_train, y_enc_train)
        oof_probs_train[t] = oof_probs
        models_base[t] = models

        p_test = np.mean([m.predict_proba(X_test) for m in models], axis=0)
        y_test_enc, _, _ = tri_class_remap(y_thr_test[t])
        y_pred_enc = np.argmax(p_test, axis=1)
        rep = report_multiclass(y_test_enc, y_pred_enc, label_names_triclass)
        print(f"\n=== BASE {t}R (multiclass -1/0/1) ===\n{rep}")
        save_text(f"{MODEL_DIR}eval_base_{t}R.txt", rep)

    # -----------------------------
    # 2) Train Regressor with OOF
    # -----------------------------
    print("\n=== Training Regressor ===")
    X_train_reg = add_base_probs_features(X_train, oof_probs_train)
    reg_base = LGBMRegressor(**DEFAULT_REG_PARAMS)
    oof_reg_train, reg_models = oof_regressor(reg_base, X_train_reg, rr_train)

    test_probs = {t: np.mean([m.predict_proba(X_test) for m in models_base[t]], axis=0)
                for t in THRESHOLDS}
    X_test_reg = add_base_probs_features(X_test, test_probs)
    reg_pred = np.mean([m.predict(X_test_reg) for m in reg_models], axis=0)

    reg_report = (
        f"MAE:  {mean_absolute_error(rr_test, reg_pred):.6f}\n"
        f"RMSE: {rmse(rr_test, reg_pred):.6f}\n"
        f"R2:   {r2_score(rr_test, reg_pred):.6f}\n"
    )
    print("\n=== REGRESSOR (rr_label) ===")
    print(reg_report)
    save_text(f"{MODEL_DIR}eval_regressor.txt", reg_report)

    # -----------------------------
    # 3) Train Meta Regressor
    # -----------------------------
    print("\n=== Training Meta Regressor ===")
    X_train_meta = build_meta_features(X_train, oof_probs_train, oof_reg_train)
    meta_model = LGBMRegressor(**DEFAULT_REG_PARAMS)
    meta_model.fit(X_train_meta, rr_train)

    X_test_meta = build_meta_features(X_test, test_probs, reg_pred)
    meta_pred = meta_model.predict(X_test_meta)
    
    meta_report = (
        f"\n=== META REGRESSOR ===\n"
        f"MAE:  {mean_absolute_error(rr_test, meta_pred):.6f}\n"
        f"RMSE: {rmse(rr_test, meta_pred):.6f}\n"
        f"R2:   {r2_score(rr_test, meta_pred):.6f}\n"
    )
    print(meta_report)
    save_text(f"{MODEL_DIR}eval_meta_regressor.txt", meta_report)

    # -----------------------------
    # 4) Retrain All Models on Full Data
    # -----------------------------
    print("\n=== Retraining on Full Data ===")
    df_full = df.copy()
    X_full = df_full[features]
    ensure_category(X_full, 'pair')

    # Final targets
    y_full_thr = {t: df_full[thr_to_col[t]].astype('int32') for t in THRESHOLDS}
    y_full_thr_enc = {t: tri_class_remap(y_full_thr[t])[0] for t in THRESHOLDS}
    rr_full = pd.to_numeric(df_full['rr_label'], errors='coerce').fillna(-1.0)

    # 90/10 split for early stopping
    idx = np.arange(len(X_full))
    np.random.seed(RANDOM_STATE); np.random.shuffle(idx)
    cut = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    # Retrain base classifiers
    print("Retraining base classifiers...")
    final_base = {}
    for t in THRESHOLDS:
        print(f"Retraining {t}R classifier...")
        m = LGBMClassifier(**DEFAULT_MC_PARAMS)
        m.set_params(n_estimators=2000)
        m.fit(
            X_full.iloc[tr_idx], y_full_thr_enc[t][tr_idx],
            eval_set=[(X_full.iloc[va_idx], y_full_thr_enc[t][va_idx])],
            categorical_feature=['pair'],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )
        final_base[t] = m

    # Retrain regressor
    print("Retraining regressor...")
    p_full = {t: final_base[t].predict_proba(X_full) for t in THRESHOLDS}
    X_full_reg = add_base_probs_features(X_full, p_full)
    reg_final = LGBMRegressor(**DEFAULT_REG_PARAMS)
    reg_final.set_params(n_estimators=2000)
    reg_final.fit(
        X_full_reg.iloc[tr_idx], rr_full.iloc[tr_idx],
        eval_set=[(X_full_reg.iloc[va_idx], rr_full.iloc[va_idx])],
        categorical_feature=['pair'],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
    )

    # Retrain meta regressor
    print("Retraining meta regressor...")
    reg_full_pred = reg_final.predict(X_full_reg)
    X_full_meta = build_meta_features(X_full, p_full, reg_full_pred)
    meta_final = LGBMRegressor(**DEFAULT_REG_PARAMS)
    meta_final.set_params(n_estimators=2000)
    meta_final.fit(
        X_full_meta.iloc[tr_idx], rr_full.iloc[tr_idx],
        eval_set=[(X_full_meta.iloc[va_idx], rr_full.iloc[va_idx])],
        categorical_feature=['pair'],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
    )
        # -----------------------------
    # Collect & Save Evaluation Stats
    # -----------------------------
    print("\nCollecting evaluation metrics...")

    eval_records = []

    # Base classifier reports (already printed + saved as txt)
    for t in THRESHOLDS:
        p_test = np.mean([m.predict_proba(X_test) for m in models_base[t]], axis=0)
        y_test_enc, _, _ = tri_class_remap(y_thr_test[t])
        y_pred_enc = np.argmax(p_test, axis=1)

        # Overall accuracy
        acc = (y_pred_enc == y_test_enc).mean()
        eval_records.append({
            "model": f"BaseClassifier_{t}R",
            "metric": "accuracy",
            "class": "all",
            "value": acc
        })

        # Confusion matrix
        cm = confusion_matrix(y_test_enc, y_pred_enc).tolist()
        eval_records.append({
            "model": f"BaseClassifier_{t}R",
            "metric": "confusion_matrix",
            "class": "all",
            "value": str(cm)
        })

        # Per-class precision/recall/F1
        cls_report = classification_report(
            y_test_enc,
            y_pred_enc,
            target_names=["-1", "0", "1"],
            output_dict=True
        )
        for cls in ["-1", "0", "1"]:
            eval_records.append({
                "model": f"BaseClassifier_{t}R",
                "metric": "precision",
                "class": cls,
                "value": cls_report[cls]["precision"]
            })
            eval_records.append({
                "model": f"BaseClassifier_{t}R",
                "metric": "recall",
                "class": cls,
                "value": cls_report[cls]["recall"]
            })
            eval_records.append({
                "model": f"BaseClassifier_{t}R",
                "metric": "f1",
                "class": cls,
                "value": cls_report[cls]["f1-score"]
            })

    # Regressor metrics
    eval_records.append({
        "model": "Regressor",
        "metric": "MAE",
        "class": "all",
        "value": mean_absolute_error(rr_test, reg_pred)
    })
    eval_records.append({
        "model": "Regressor",
        "metric": "RMSE",
        "class": "all",
        "value": rmse(rr_test, reg_pred)
    })
    eval_records.append({
        "model": "Regressor",
        "metric": "R2",
        "class": "all",
        "value": r2_score(rr_test, reg_pred)
    })

    # Meta regressor metrics
    eval_records.append({
        "model": "MetaRegressor",
        "metric": "MAE",
        "class": "all",
        "value": mean_absolute_error(rr_test, meta_pred)
    })
    eval_records.append({
        "model": "MetaRegressor",
        "metric": "RMSE",
        "class": "all",
        "value": rmse(rr_test, meta_pred)
    })
    eval_records.append({
        "model": "MetaRegressor",
        "metric": "R2",
        "class": "all",
        "value": r2_score(rr_test, meta_pred)
    })

    # Save CSV
    eval_df = pd.DataFrame(eval_records)
    eval_csv_path = os.path.join(MODEL_DIR, "evaluation_summary.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    print("\n=== Evaluation Summary ===")
    print(eval_df)
    print(f"\n✅ Evaluation summary saved to: {eval_csv_path}")
    
    
        # -----------------------------
    # Save Feature Importances
    # -----------------------------
    print("\nExtracting feature importances...")

    fi_dir = os.path.join(MODEL_DIR, "feature_importances")
    os.makedirs(fi_dir, exist_ok=True)

    def save_feature_importances(model, name):
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(fi_dir, f"{name}_feature_importances.csv"), index=False)

    # Base classifiers
    for t in THRESHOLDS:
        for i, m in enumerate(models_base[t]):
            save_feature_importances(m, f"BaseClassifier_{t}R_fold{i}")

    # Regressor
    for i, m in enumerate(models_reg):
        save_feature_importances(m, f"Regressor_fold{i}")

    # Meta regressor
    save_feature_importances(meta_reg, "MetaRegressor")

    print(f"✅ Feature importances saved to {fi_dir}")
    
    # -----------------------------
    # Save Artifacts
    # -----------------------------
    print("\nSaving models...")
    for t, model in final_base.items():
        model.booster_.save_model(f"{MODEL_DIR}classifier_{t}R.txt")
    reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
    meta_final.booster_.save_model(f"{MODEL_DIR}meta_regressor.txt")

    metadata = {
        "features": features,
        "thresholds": THRESHOLDS,
        "thr_to_col": thr_to_col,
        "label_names_triclass": ["loss(-1)","neutral(0)","win(+1)"],
        "best_params": {
            **{f"mc_{t}R": DEFAULT_MC_PARAMS for t in THRESHOLDS},
            "reg": DEFAULT_REG_PARAMS,
        },
        "notes": "Updated with regression meta model and full retraining"
    }
    joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

    print("\n✅ Training complete. Models saved to:", os.path.abspath(MODEL_DIR))

if __name__ == "__main__":
    main()
