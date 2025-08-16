# ===========================
# train_stack_part2.py
# ===========================
# Part 2 of the stacking pipeline:
# - Loads meta features from Part 1's artifacts dir
# - Trains a multiclass meta model (lgbm/xgb)
# - OOF on meta-train, final fit, test predictions
# - Threshold optimization for profit (from OOF), apply on test
# - Saves PNGs (confusion, PR), CSVs (preds), JSON (metrics/params)
#
# pip install lightgbm xgboost optuna scikit-learn pandas numpy matplotlib

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, log_loss,
                             f1_score, accuracy_score, roc_auc_score, precision_recall_curve)
from sklearn.impute import SimpleImputer

# -----------------------------
# Defaults (can override via CLI)
# -----------------------------
DEFAULT_META_TARGET = "y_2.2R"
DEFAULT_BASE_MODEL_LIB = "lgbm"     # this is the base dir suffix used by Part 1
DEFAULT_META_MODEL_LIB = "lgbm"     # choose lgbm or xgb for meta
DEFAULT_BASE_DIR = None             # if None, will infer: artifacts_stack_base_<meta_target>_<base_model_lib>
RANDOM_STATE = 42
N_SPLITS = 5
MAX_EARLY_STOP = 200
N_TRIALS = 60

# -----------------------------
# Utils / plotting
# -----------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_rr_from_target(target_name: str, default_rr=1.0) -> float:
    m = re.search(r"y_([0-9]+(?:\.[0-9]+)?)R", target_name)
    if m:
        return float(m.group(1))
    return float(default_rr)

def plot_and_save_confusion(cm, classes, title, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:d}", ha="center", va="center")
    ax.set_ylabel('True'); ax.set_xlabel('Pred')
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)

def plot_and_save_pr_curve(y_true_012, proba, positive_class_index, title, outpath):
    y_true_bin = (y_true_012 == positive_class_index).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_bin, proba[:, positive_class_index])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(recall, precision)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)

def summarize_metrics(y_true_012, y_pred_012, proba, class_names):
    res = {}
    res["accuracy"] = float(accuracy_score(y_true_012, y_pred_012))
    res["macro_f1"] = float(f1_score(y_true_012, y_pred_012, average="macro"))
    # multiclass logloss
    res["logloss"] = float(log_loss(y_true_012, proba, labels=np.arange(len(class_names))))
    # AUC for "win" vs rest
    try:
        y_true_win = (y_true_012 == 2).astype(int)
        res["roc_auc_win_ovr"] = float(roc_auc_score(y_true_win, proba[:, 2]))
    except Exception:
        res["roc_auc_win_ovr"] = None
    res["classification_report"] = classification_report(y_true_012, y_pred_012, target_names=class_names)
    return res

def optimize_thresholds_for_profit(y_true_012, proba, rr_win=1.0):
    """
    Optimize (th_win, th_loss) on OOF.
    Take trade if: p_win >= th_win AND p_loss <= th_loss
    Profit per trade: +rr_win if true==win, -1 if true==loss, 0 if neutral
    """
    p_loss = proba[:, 0]
    p_win  = proba[:, 2]
    best = {"th_win": 0.5, "th_loss": 0.2, "expR_per_trade": -999.0, "take_rate": 0.0}
    for th_win in np.linspace(0.40, 0.90, 26):
        for th_loss in np.linspace(0.00, 0.50, 26):
            take = (p_win >= th_win) & (p_loss <= th_loss)
            if not np.any(take):
                continue
            y_sel = y_true_012[take]
            wins = np.sum(y_sel == 2)
            losses = np.sum(y_sel == 0)
            total = len(y_sel)
            expR = (wins * rr_win - losses * 1.0) / total
            if expR > best["expR_per_trade"]:
                best = {
                    "th_win": float(th_win),
                    "th_loss": float(th_loss),
                    "expR_per_trade": float(expR),
                    "take_rate": float(total / len(y_true_012))
                }
    return best

# -----------------------------
# Tuning (Optuna) for meta
# -----------------------------
def tune_lightgbm_meta(X, y, Xv, yv, n_trials=N_TRIALS):
    import lightgbm as lgb
    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(
            X, y,
            eval_set=[(Xv, yv)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=MAX_EARLY_STOP, verbose=False)]
        )
        preds = model.predict_proba(Xv)
        return log_loss(yv, preds, labels=[0,1,2])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_xgboost_meta(X, y, Xv, yv, n_trials=N_TRIALS):
    import xgboost as xgb
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = xgb.XGBClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, nthread=-1)
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False, early_stopping_rounds=MAX_EARLY_STOP)
        preds = model.predict_proba(Xv)
        return log_loss(yv, preds, labels=[0,1,2])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def fit_meta(model_lib, params, X, y, Xv=None, yv=None):
    if model_lib == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(
            X, y,
            eval_set=[(Xv, yv)] if Xv is not None else None,
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=MAX_EARLY_STOP, verbose=False)] if Xv is not None else None
        )
        return model
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, nthread=-1)
        model.fit(
            X, y,
            eval_set=[(Xv, yv)] if Xv is not None else None,
            verbose=False,
            early_stopping_rounds=MAX_EARLY_STOP if Xv is not None else None
        )
        return model

def plot_and_save_feature_importance(model, feature_names, outpath, topn=30):
    try:
        importances = model.feature_importances_
    except Exception:
        return
    idx = np.argsort(importances)[::-1][:topn]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(range(len(names)), vals[::-1])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1])
    ax.set_title("Meta Model: Top Feature Importances")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main(base_dir, meta_target, base_model_lib, meta_model_lib):
    if base_dir is None:
        base_dir = f"artifacts_stack_base_{meta_target}_{base_model_lib}"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base artifacts dir not found: {base_dir}")

    # Load meta features & targets
    meta_train = pd.read_csv(os.path.join(base_dir, "meta_train_features.csv"))
    meta_test  = pd.read_csv(os.path.join(base_dir, "meta_test_features.csv"))
    y_train_meta = pd.read_csv(os.path.join(base_dir, "y_train_meta.csv"))["y_train_meta"].astype(int).values
    y_test_meta  = pd.read_csv(os.path.join(base_dir, "y_test_meta.csv"))["y_test_meta"].astype(int).values

    # Impute
    imputer = SimpleImputer(strategy="median")
    meta_train[:] = imputer.fit_transform(meta_train)
    meta_test[:]  = imputer.transform(meta_test)

    # Train/valid split for tuning (from meta_train)
    Xtr, Xva, ytr, yva = train_test_split(meta_train, y_train_meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_meta)

    # Tune
    if meta_model_lib == "lgbm":
        best_params = tune_lightgbm_meta(Xtr, ytr, Xva, yva, n_trials=N_TRIALS)
    else:
        best_params = tune_xgboost_meta(Xtr, ytr, Xva, yva, n_trials=N_TRIALS)

    # OOF on meta_train
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros((len(meta_train), 3), dtype=float)
    models = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(meta_train, y_train_meta), 1):
        X_tr, y_tr = meta_train.iloc[idx_tr], y_train_meta[idx_tr]
        X_va, y_va = meta_train.iloc[idx_va], y_train_meta[idx_va]
        model = fit_meta(meta_model_lib, best_params, X_tr, y_tr, X_va, y_va)
        oof_pred[idx_va] = model.predict_proba(X_va)
        models.append(model)
        print(f"[meta] fold {fold} done")

    # Final fit on all meta_train (use small val split for early stopping)
    final_model = fit_meta(meta_model_lib, best_params, Xtr, ytr, Xva, yva)
    test_proba = final_model.predict_proba(meta_test)
    test_pred = np.argmax(test_proba, axis=1)

    # Metrics
    class_names = ["loss","neutral","win"]
    oof_pred_cls = np.argmax(oof_pred, axis=1)
    oof_metrics = summarize_metrics(y_train_meta, oof_pred_cls, oof_pred, class_names)
    test_metrics = summarize_metrics(y_test_meta, test_pred, test_proba, class_names)

    # Optimize thresholds on OOF, apply on test (profit)
    rr_win = parse_rr_from_target(meta_target, default_rr=1.0)
    th = optimize_thresholds_for_profit(y_train_meta, oof_pred, rr_win=rr_win)

    p_loss_test = test_proba[:, 0]
    p_win_test  = test_proba[:, 2]
    take_test = (p_win_test >= th["th_win"]) & (p_loss_test <= th["th_loss"])
    y_sel = y_test_meta[take_test]
    wins = int(np.sum(y_sel == 2))
    losses = int(np.sum(y_sel == 0))
    neutrals = int(np.sum(y_sel == 1))
    total = int(len(y_sel))
    expR = (wins * rr_win - losses * 1.0) / total if total else 0.0
    take_rate = total / len(y_test_meta) if len(y_test_meta) else 0.0

    # Artifacts dir for meta
    meta_dir = f"artifacts_stack_meta_{meta_target}_{meta_model_lib}"
    ensure_dir(meta_dir)

    # Save params & metrics
    with open(os.path.join(meta_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(meta_dir, "metrics.json"), "w") as f:
        json.dump({
            "oof": oof_metrics,
            "test": test_metrics,
            "rr_win": rr_win,
            "thresholds": th,
            "test_trading_summary": {
                "selected_trades": total,
                "take_rate": take_rate,
                "wins": wins, "losses": losses, "neutrals": neutrals,
                "expR_per_trade": expR
            }
        }, f, indent=2)

    # Save predictions
    pd.DataFrame(oof_pred, columns=[f"proba_{c}" for c in class_names])\
      .assign(y_true=y_train_meta)\
      .to_csv(os.path.join(meta_dir, "oof_predictions.csv"), index=False)

    pd.DataFrame(test_proba, columns=[f"proba_{c}" for c in class_names])\
      .assign(y_true=y_test_meta, take=take_test.astype(int))\
      .to_csv(os.path.join(meta_dir, "test_predictions.csv"), index=False)

    # Plots
    cm_oof = confusion_matrix(y_train_meta, oof_pred_cls, labels=[0,1,2])
    plot_and_save_confusion(cm_oof, class_names, f"OOF Confusion (meta {meta_target})", os.path.join(meta_dir, "cm_oof.png"))
    cm_test = confusion_matrix(y_test_meta, test_pred, labels=[0,1,2])
    plot_and_save_confusion(cm_test, class_names, f"Test Confusion (meta {meta_target})", os.path.join(meta_dir, "cm_test.png"))

    plot_and_save_pr_curve(y_train_meta, oof_pred, positive_class_index=2,
                           title=f"OOF PR (win) - meta {meta_target}",
                           outpath=os.path.join(meta_dir, "pr_win_oof.png"))
    plot_and_save_pr_curve(y_test_meta, test_proba, positive_class_index=2,
                           title=f"Test PR (win) - meta {meta_target}",
                           outpath=os.path.join(meta_dir, "pr_win_test.png"))

    # Feature importances (if supported)
    try:
        plot_and_save_feature_importance(final_model, meta_train.columns.tolist(),
                                         os.path.join(meta_dir, "feature_importance.png"), topn=40)
    except Exception:
        pass

    # Console summary
    print("\n=== OOF Metrics (meta) ===")
    print(json.dumps(oof_metrics, indent=2))
    print("\n=== Test Metrics (meta) ===")
    print(json.dumps(test_metrics, indent=2))
    print("\n=== Thresholds (from OOF) ===")
    print(th)
    print("\n=== Test Trading Summary ===")
    print(f"Selected trades: {total}  (take_rate={take_rate:.2%})")
    print(f"Wins: {wins}  Losses: {losses}  Neutrals: {neutrals}")
    print(f"rr_win (from meta_target): {rr_win}")
    print(f"Expected R per selected trade: {expR:.3f}")
    print(f"\nArtifacts saved to: {meta_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, type=str,
                        help="Path to Part-1 artifacts dir. If empty, inferred from meta_target & base_model_lib.")
    parser.add_argument("--meta_target", default=DEFAULT_META_TARGET, type=str,
                        help="Which y_*R the meta is learning (3-class)")
    parser.add_argument("--base_model_lib", default=DEFAULT_BASE_MODEL_LIB, type=str,
                        help="The base learner lib used in Part 1 dir name: lgbm or xgb")
    parser.add_argument("--meta_model_lib", default=DEFAULT_META_MODEL_LIB, type=str,
                        help="Which lib to use for the meta model: lgbm or xgb")
    args = parser.parse_args()
    main(args.base_dir, args.meta_target, args.base_model_lib, args.meta_model_lib)


# If Part 1 used LightGBM and meta_target y_2.2R (default)
# python train_stack_part2.py --meta_target y_2.2R --base_model_lib lgbm --meta_model_lib lgbm

# # If Part 1 used XGBoost
# python train_stack_part2.py --meta_target y_2.2R --base_model_lib xgb --meta_model_lib lgbm

# # Or specify the base dir explicitly
# python train_stack_part2.py --base_dir artifacts_stack_base_y_2.2R_lgbm --meta_model_lib xgb
