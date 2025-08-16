# ===========================
# stack_part2.py
# ===========================
# Trains a multiclass META model:
# Input = [all base probs + rr_pred + FULL features]
# Output = final class + probabilities
# Saves: best params, OOF/Test metrics, confusion matrices, PR curves, thresholded profit sim.

import os, re, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, log_loss,
                             f1_score, accuracy_score, roc_auc_score, precision_recall_curve)
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
N_SPLITS = 5
MAX_EARLY_STOP = 200
N_TRIALS = 60

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def parse_rr_from_target(name, default_rr=1.0):
    import re
    m = re.search(r"y_([0-9]+(?:\.[0-9]+)?)R", name)
    return float(m.group(1)) if m else float(default_rr)

def plot_cm(cm, classes, title, outpath):
    fig = plt.figure(); ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest'); ax.set_title(title); plt.colorbar(im)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,f"{cm[i,j]:d}",ha="center",va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Pred"); fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight'); plt.close(fig)

def plot_pr(y_true_012, proba, pos_idx, title, outpath):
    from sklearn.metrics import precision_recall_curve
    yb = (y_true_012==pos_idx).astype(int)
    P,R,_ = precision_recall_curve(yb, proba[:,pos_idx])
    fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(R,P)
    ax.set_title(title); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    fig.savefig(outpath, dpi=140, bbox_inches='tight'); plt.close(fig)

def summarize(y_true, y_pred, proba):
    names = ["loss","neutral","win"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(log_loss(y_true, proba, labels=[0,1,2])),
        "roc_auc_win_ovr": float(roc_auc_score((y_true==2).astype(int), proba[:,2])),
        "report": classification_report(y_true, y_pred, target_names=names)
    }

def optimize_thresholds(y_true, proba, rr_win):
    p_loss, p_win = proba[:,0], proba[:,2]
    best = {"th_win":0.5,"th_loss":0.2,"expR_per_trade":-999.0,"take_rate":0.0}
    for thw in np.linspace(0.40,0.90,26):
        for thl in np.linspace(0.00,0.50,26):
            take = (p_win>=thw) & (p_loss<=thl)
            if not np.any(take): continue
            sel = y_true[take]
            wins = int(np.sum(sel==2)); losses = int(np.sum(sel==0)); total = len(sel)
            expR = (wins*rr_win - losses*1.0)/total
            if expR > best["expR_per_trade"]:
                best = {"th_win":float(thw),"th_loss":float(thl),
                        "expR_per_trade":float(expR),
                        "take_rate":float(total/len(y_true))}
    return best

# ------ tuning / fit for meta
def tune_lgbm_meta(X, y, Xv, yv, n_trials):
    import lightgbm as lgb
    def obj(trial):
        params = {
            "objective":"multiclass","num_class":3,"boosting_type":"gbdt","verbosity":-1,
            "learning_rate": trial.suggest_float("learning_rate",0.02,0.2,log=True),
            "num_leaves":   trial.suggest_int("num_leaves",16,256,log=True),
            "max_depth":    trial.suggest_int("max_depth",4,12),
            "min_child_samples": trial.suggest_int("min_child_samples",10,200),
            "subsample": trial.suggest_float("subsample",0.6,1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
            "reg_lambda": trial.suggest_float("reg_lambda",1e-3,50,log=True),
            "reg_alpha":  trial.suggest_float("reg_alpha",1e-3,10,log=True),
        }
        m = lgb.LGBMClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X,y, eval_set=[(Xv,yv)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(stopping_rounds=MAX_EARLY_STOP, verbose=False)])
        p = m.predict_proba(Xv)
        return log_loss(yv, p, labels=[0,1,2])
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_xgb_meta(X, y, Xv, yv, n_trials):
    import xgboost as xgb
    def obj(trial):
        params = {
            "objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
            "tree_method":"hist","max_bin": trial.suggest_int("max_bin",128,512),
            "learning_rate": trial.suggest_float("learning_rate",0.02,0.2,log=True),
            "max_depth": trial.suggest_int("max_depth",3,10),
            "min_child_weight": trial.suggest_float("min_child_weight",1e-3,10,log=True),
            "gamma": trial.suggest_float("gamma",0.0,5.0),
            "subsample": trial.suggest_float("subsample",0.6,1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
            "reg_lambda": trial.suggest_float("reg_lambda",1e-3,50,log=True),
            "reg_alpha":  trial.suggest_float("reg_alpha",1e-3,10,log=True),
        }
        m = xgb.XGBClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, nthread=-1)
        m.fit(X,y, eval_set=[(Xv,yv)], verbose=False, early_stopping_rounds=MAX_EARLY_STOP)
        p = m.predict_proba(Xv)
        return log_loss(yv, p, labels=[0,1,2])
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def fit_meta(lib, params, X, y, Xv=None, yv=None):
    if lib=="lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X,y, eval_set=[(Xv,yv)] if Xv is not None else None, eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(stopping_rounds=MAX_EARLY_STOP, verbose=False)] if Xv is not None else None)
        return m
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(**params, n_estimators=5000, random_state=RANDOM_STATE, nthread=-1)
        m.fit(X,y, eval_set=[(Xv,yv)] if Xv is not None else None, verbose=False,
              early_stopping_rounds=MAX_EARLY_STOP if Xv is not None else None)
        return m

def plot_feat_imp(model, names, out, topn=40):
    try: imp = model.feature_importances_
    except: return
    idx = np.argsort(imp)[::-1][:topn]
    vals = imp[idx]; labs = [names[i] for i in idx]
    fig = plt.figure(); ax = fig.add_subplot(111); ax.barh(range(len(labs)), vals[::-1])
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs[::-1])
    ax.set_title("Meta Feature Importance"); fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)

# -----------------------------
def main(base_dir, meta_model_lib, meta_target):
    if base_dir is None:
        base_dir = f"artifacts_stack_baseFULL_{meta_target}_lgbm"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(base_dir)

    Xtr = pd.read_csv(os.path.join(base_dir,"meta_train_features.csv"))
    Xte = pd.read_csv(os.path.join(base_dir,"meta_test_features.csv"))
    ytr = pd.read_csv(os.path.join(base_dir,"y_train_meta.csv"))["y_train_meta"].astype(int).values
    yte = pd.read_csv(os.path.join(base_dir,"y_test_meta.csv"))["y_test_meta"].astype(int).values

    # Impute any residual NaNs
    imp = SimpleImputer(strategy="median")
    Xtr[:] = imp.fit_transform(Xtr); Xte[:] = imp.transform(Xte)

    # tune on a small split of Xtr
    X1,X2,Y1,Y2 = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=RANDOM_STATE)
    best = tune_lgbm_meta(X1,Y1,X2,Y2,N_TRIALS) if meta_model_lib=="lgbm" else tune_xgb_meta(X1,Y1,X2,Y2,N_TRIALS)

    # OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(Xtr),3))
    for k,(i_tr,i_va) in enumerate(skf.split(Xtr,ytr),1):
        m = fit_meta(meta_model_lib, best, Xtr.iloc[i_tr], ytr[i_tr], Xtr.iloc[i_va], ytr[i_va])
        oof[i_va] = m.predict_proba(Xtr.iloc[i_va])
        print(f"[meta] fold {k} done")

    # final model and test predictions
    final = fit_meta(meta_model_lib, best, X1, Y1, X2, Y2)
    te_proba = final.predict_proba(Xte)
    te_pred = te_proba.argmax(1)

    # Metrics
    oof_cls = oof.argmax(1)
    names = ["loss","neutral","win"]
    oof_metrics = {
        "accuracy": float(accuracy_score(ytr, oof_cls)),
        "macro_f1": float(f1_score(ytr, oof_cls, average="macro")),
        "logloss": float(log_loss(ytr, oof, labels=[0,1,2])),
        "roc_auc_win_ovr": float(roc_auc_score((ytr==2).astype(int), oof[:,2])),
        "report": classification_report(ytr, oof_cls, target_names=names)
    }
    te_metrics = {
        "accuracy": float(accuracy_score(yte, te_pred)),
        "macro_f1": float(f1_score(yte, te_pred, average="macro")),
        "logloss": float(log_loss(yte, te_proba, labels=[0,1,2])),
        "roc_auc_win_ovr": float(roc_auc_score((yte==2).astype(int), te_proba[:,2])),
        "report": classification_report(yte, te_pred, target_names=names)
    }

    # Threshold optimization on OOF → apply to test for profit
    rr_win = parse_rr_from_target(meta_target, default_rr=1.0)
    th = optimize_thresholds(ytr, oof, rr_win)
    take = (te_proba[:,2] >= th["th_win"]) & (te_proba[:,0] <= th["th_loss"])
    sel = yte[take]
    wins = int(np.sum(sel==2)); losses = int(np.sum(sel==0)); neutrals = int(np.sum(sel==1))
    total = len(sel); expR = (wins*rr_win - losses*1.0)/total if total else 0.0
    take_rate = total/len(yte) if len(yte) else 0.0

    # Save artifacts
    meta_dir = f"artifacts_stack_META_{meta_target}_{meta_model_lib}"
    ensure_dir(meta_dir)
    with open(os.path.join(meta_dir,"best_params.json"),"w") as f: json.dump(best,f,indent=2)
    with open(os.path.join(meta_dir,"metrics.json"),"w") as f:
        json.dump({
            "oof":oof_metrics,"test":te_metrics,"rr_win":rr_win,"thresholds":th,
            "test_trading_summary":{"selected":total,"take_rate":take_rate,"wins":wins,"losses":losses,
                                    "neutrals":neutrals,"expR_per_trade":expR}
        }, f, indent=2)

    pd.DataFrame(oof, columns=[f"proba_{c}" for c in names]).assign(y_true=ytr)\
      .to_csv(os.path.join(meta_dir,"oof_predictions.csv"), index=False)
    pd.DataFrame(te_proba, columns=[f"proba_{c}" for c in names]).assign(y_true=yte, take=take.astype(int))\
      .to_csv(os.path.join(meta_dir,"test_predictions.csv"), index=False)

    cm_oof = confusion_matrix(ytr, oof_cls, labels=[0,1,2])
    plot_cm(cm_oof, names, f"Meta OOF Confusion ({meta_target})", os.path.join(meta_dir,"cm_oof.png"))
    cm_test = confusion_matrix(yte, te_pred, labels=[0,1,2])
    plot_cm(cm_test, names, f"Meta Test Confusion ({meta_target})", os.path.join(meta_dir,"cm_test.png"))
    plot_pr(ytr, oof, 2, f"OOF PR (win) - {meta_target}", os.path.join(meta_dir,"pr_win_oof.png"))
    plot_pr(yte, te_proba, 2, f"Test PR (win) - {meta_target}", os.path.join(meta_dir,"pr_win_test.png"))

    # feature importance
    try: plot_feat_imp(final, Xtr.columns.tolist(), os.path.join(meta_dir,"feature_importance.png"), topn=40)
    except: pass

    print("\n=== OOF (meta) ==="); print(json.dumps(oof_metrics, indent=2))
    print("\n=== Test (meta) ==="); print(json.dumps(te_metrics, indent=2))
    print("\n=== Thresholds ===", th)
    print("Selected:", total, "Take rate:", f"{take_rate:.2%}", "Wins:", wins, "Losses:", losses, "Neutrals:", neutrals)
    print("Expected R/trade:", f"{expR:.3f}")
    print("Artifacts →", meta_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=None,
                    help="Dir from Part 1. If None, defaults to artifacts_stack_baseFULL_<meta_target>_lgbm")
    ap.add_argument("--meta_model", type=str, default="lgbm", choices=["lgbm","xgb"])
    ap.add_argument("--meta_target", type=str, default="y_2.2R")
    args = ap.parse_args()
    main(args.base_dir, args.meta_model, args.meta_target)

    # If Part 1 used LightGBM and meta_target y_2.2R:
# python stack_part2.py --meta_target y_2.2R --meta_model lgbm

# # Explicit base dir (if you changed defaults)
# python stack_part2.py --base_dir artifacts_stack_baseFULL_y_2.2R_lgbm --meta_model xgb --meta_target y_2.2R

