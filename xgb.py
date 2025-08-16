# # ==============================#
# #   XGB PIPELINE (2 BIN + REG + META 0/1/2) + OPTUNA TUNING
# #   Binaries:  y_ge_1R, y_ge_2R
# #   Regressor: rr_label
# #   Meta:      0:<1R, 1:1..2R, 2:>=2R
# #   + Holdout Win-Rate Simulation (PNG plots)
# #   + Optuna tuning for all stages
# #   + Feature Importances CSV
# # ==============================#

# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")  # safe for CLI/headless
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import (
#     accuracy_score, f1_score, roc_auc_score,
#     mean_absolute_error, mean_squared_error, r2_score,
#     classification_report, confusion_matrix
# )

# from xgboost import XGBClassifier, XGBRegressor
# import optuna

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH      = "test-combined.csv"       # your combined labeled dataset
# MODEL_DIR      = "./xgb_artifacts_meta3_opt/"
# os.makedirs(MODEL_DIR, exist_ok=True)

# RANDOM_STATE   = 42
# N_SPLITS_OOF   = 3

# # Optuna trials per stage
# N_TRIALS_BIN   = 15
# N_TRIALS_REG   = 15
# N_TRIALS_META  = 15

# # thresholds we care about
# RR_THRESHOLDS  = [1, 2]  # ONLY 1R and 2R

# # -----------------------------
# # Helpers
# # -----------------------------
# def rmse(y_true, y_pred):
#     return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

# def best_threshold(y_true, p, metric='f1'):
#     grid = np.linspace(0.05, 0.95, 19)
#     best, best_t = -1.0, 0.5
#     for t in grid:
#         yhat = (p >= t).astype(int)
#         s = f1_score(y_true, yhat) if metric == 'f1' else accuracy_score(y_true, yhat)
#         if s > best:
#             best, best_t = s, t
#     return float(best_t)

# def inv_freq_weights_multi(y):
#     counts = pd.Series(y).value_counts()
#     N, K = len(y), counts.shape[0]
#     w = {int(c): float(N / (K * counts.get(c, 1))) for c in counts.index}
#     return w

# def dump_importance_xgb(model, cols, path, title):
#     try:
#         imp = pd.DataFrame({
#             "Feature": cols,
#             "Importance": model.feature_importances_
#         }).sort_values("Importance", ascending=False)
#         imp.to_csv(path, index=False)
#         print(f"{title} top10:\n", imp.head(10), "\n")
#         return imp
#     except Exception as e:
#         print(f"⚠️ No feature_importances_ for {title}: {e}")
#         return pd.DataFrame()

# # -----------------------------
# # Labels
# # -----------------------------
# def generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS):
#     rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
#     thr_list = sorted(set(thr_list))
#     n = len(df)

#     print(rr.describe([.5, .75, .90, .95, .99]))
#     for t in [1, 2, 3, 4, 5]:
#         share = (rr >= t).mean()
#         print(f"P(rr ≥ {t}) = {share:.3f}")

#     if 'pair' in df.columns:
#         print("\nShare ≥ last threshold by pair (top 10):")
#         print(
#             df.groupby('pair')['rr_label']
#               .apply(lambda s: (pd.to_numeric(s, errors='coerce') >= thr_list[-1]).mean())
#               .sort_values(ascending=False)
#               .head(10)
#         )

#     print("\n=== Binary targets (y_ge_{T}R) ===")
#     for thr in thr_list:
#         col = f'y_ge_{thr}R'
#         df[col] = (rr >= thr).astype('int8')
#         vc = df[col].value_counts().reindex([0, 1], fill_value=0)
#         neg, pos = int(vc[0]), int(vc[1])
#         print(f"{col}: pos={pos} ({pos/n:.2%}) | neg={neg} ({neg/n:.2%})")

#     # meta3: 0:<1R, 1:[1,2), 2:>=2R
#     bins = [-np.inf, 1, 2, np.inf]
#     df['y_meta3'] = pd.cut(rr, bins=bins, labels=False, right=False, include_lowest=True).astype('int8')

#     names = ["<1R", "1..2R", ">=2R"]
#     print("\n=== Meta3 target (y_meta3) ===")
#     vc_meta = df['y_meta3'].value_counts().sort_index()
#     for k in range(3):
#         cnt = int(vc_meta.get(k, 0))
#         print(f"class {k} ({names[k]:>5}): {cnt} ({cnt/n:.2%})")
#     print(f"TOTAL rows: {n}")
#     return df

# # -----------------------------
# # OOF (no leakage)
# # -----------------------------
# def oof_binary_xgb(X, y, params, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in skf.split(X, y):
#         neg = (y.iloc[tr] == 0).sum()
#         pos = (y.iloc[tr] == 1).sum()
#         spw = max(1.0, neg / max(1, pos))
#         params_fold = {**params, "scale_pos_weight": spw}
#         m = XGBClassifier(**params_fold)
#         m.fit(X.iloc[tr], y.iloc[tr])
#         oof[va] = m.predict_proba(X.iloc[va])[:, 1]
#         models.append(m)
#     return oof, models

# def oof_regressor_xgb(X, y, params, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in kf.split(X, y):
#         m = XGBRegressor(**params)
#         m.fit(X.iloc[tr], y.iloc[tr])
#         oof[va] = m.predict(X.iloc[va])
#         models.append(m)
#     return oof, models

# # -----------------------------
# # Tuning (Optuna)
# # -----------------------------
# def tune_binary_params(X, y, study_name):
#     def objective(trial):
#         params = dict(
#             n_estimators = trial.suggest_int("n_estimators", 200, 900),
#             learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
#             max_depth    = trial.suggest_int("max_depth", 3, 8),
#             subsample    = trial.suggest_float("subsample", 0.6, 1.0),
#             colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#             reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
#             objective    = "binary:logistic",
#             eval_metric  = "auc",
#             random_state = RANDOM_STATE,
#             n_jobs       = -1
#         )
#         # CV AUC
#         skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
#         aucs = []
#         for tr, va in skf.split(X, y):
#             y_tr = y.iloc[tr]
#             neg = (y_tr == 0).sum()
#             pos = (y_tr == 1).sum()
#             spw = max(1.0, neg / max(1, pos))
#             params_fold = {**params, "scale_pos_weight": spw}
#             m = XGBClassifier(**params_fold)
#             m.fit(X.iloc[tr], y.iloc[tr])
#             p = m.predict_proba(X.iloc[va])[:, 1]
#             aucs.append(roc_auc_score(y.iloc[va], p))
#         return float(np.mean(aucs))
#     study = optuna.create_study(direction="maximize", study_name=study_name)
#     study.optimize(objective, n_trials=N_TRIALS_BIN)
#     return study.best_params

# def tune_regressor_params(X, y, study_name):
#     def objective(trial):
#         params = dict(
#             n_estimators = trial.suggest_int("n_estimators", 300, 1000),
#             learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
#             max_depth    = trial.suggest_int("max_depth", 4, 10),
#             subsample    = trial.suggest_float("subsample", 0.6, 1.0),
#             colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#             reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
#             objective    = "reg:squarederror",
#             random_state = RANDOM_STATE,
#             n_jobs       = -1
#         )
#         # CV RMSE (lower is better)
#         kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
#         rmses = []
#         for tr, va in kf.split(X, y):
#             m = XGBRegressor(**params)
#             m.fit(X.iloc[tr], y.iloc[tr])
#             pred = m.predict(X.iloc[va])
#             rmses.append(rmse(y.iloc[va].values, pred))
#         return float(np.mean(rmses))
#     study = optuna.create_study(direction="minimize", study_name=study_name)
#     study.optimize(objective, n_trials=N_TRIALS_REG)
#     return study.best_params

# def tune_meta_params(X, y, study_name):
#     # 3 classes, we’ll optimize weighted-F1 via 3-fold CV
#     def objective(trial):
#         params = dict(
#             n_estimators = trial.suggest_int("n_estimators", 200, 900),
#             learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
#             max_depth    = trial.suggest_int("max_depth", 4, 10),
#             subsample    = trial.suggest_float("subsample", 0.6, 1.0),
#             colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#             reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
#             objective    = "multi:softprob",
#             eval_metric  = "mlogloss",
#             num_class    = 3,
#             random_state = RANDOM_STATE,
#             n_jobs       = -1
#         )
#         skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
#         f1s = []
#         for tr, va in skf.split(X, y):
#             # inverse-freq weights to avoid class 1 collapse
#             w_map = inv_freq_weights_multi(y.iloc[tr])
#             sw = y.iloc[tr].map(w_map).astype(float).values
#             m = XGBClassifier(**params)
#             m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sw)
#             pred = m.predict(X.iloc[va])
#             f1s.append(f1_score(y.iloc[va], pred, average='weighted'))
#         return float(np.mean(f1s))
#     study = optuna.create_study(direction="maximize", study_name=study_name)
#     study.optimize(objective, n_trials=N_TRIALS_META)
#     return study.best_params

# # -----------------------------
# # Stacking feature builders
# # -----------------------------
# def create_reg_features(X, oof_probs=None, models_dict=None):
#     Xf = X.copy()
#     if oof_probs is not None:
#         for name in oof_probs:
#             Xf[f'clf_{name}_prob'] = oof_probs[name]
#     elif models_dict is not None:
#         for name, model_list in models_dict.items():
#             Xf[f'clf_{name}_prob'] = np.mean([m.predict_proba(X)[:, 1] for m in model_list], axis=0)
#     return Xf

# def create_meta3_features(X, oof_probs=None, reg_vec=None, models_dict=None, reg_models=None):
#     Xf = create_reg_features(X, oof_probs=oof_probs, models_dict=models_dict)
#     if reg_vec is None and reg_models is not None:
#         reg_vec = np.mean([m.predict(create_reg_features(X, models_dict=models_dict)) for m in reg_models], axis=0)
#     if reg_vec is not None:
#         Xf['reg_pred'] = reg_vec
#     return Xf

# # -----------------------------
# # PNG-saving evaluator
# # -----------------------------
# def evaluate_selection(name, mask, rr, rr_target=None, plot=False, save_dir="selection_plots_xgb"):
#     os.makedirs(save_dir, exist_ok=True)
#     mask = np.asarray(mask, dtype=bool)
#     rr   = np.asarray(pd.to_numeric(rr, errors="coerce"))
#     L = min(len(mask), len(rr))
#     mask, rr = mask[:L], rr[:L]
#     ok = ~np.isnan(rr)
#     mask, rr = mask & ok, rr[ok]

#     n = int(mask.sum())
#     if n == 0:
#         print(f"\n[{name}] No trades selected.")
#         return

#     rr_sel = rr[mask]
#     win_rate = float((rr_sel >= (rr_target if rr_target is not None else 0)).mean()) if rr_target is not None else np.nan
#     avg_rr = float(rr_sel.mean())
#     med_rr = float(np.median(rr_sel))

#     print(f"\n[{name}]")
#     print(f"Trades taken: {n}")
#     if rr_target is not None:
#         print(f"Win rate (≥ {rr_target}R): {win_rate:.3f}")
#     print(f"Average RR: {avg_rr:.3f}")
#     print(f"Median RR:  {med_rr:.3f}")

#     if plot:
#         plt.figure()
#         plt.hist(rr_sel, bins=40)
#         plt.title(name)
#         plt.xlabel("RR")
#         plt.ylabel("Count")
#         plt.tight_layout()
#         safe_name = name.replace(" ", "_").replace(":", "").replace("/", "-").replace("\\", "-")
#         path = os.path.join(save_dir, f"{safe_name}.png")
#         plt.savefig(path, dpi=300)
#         print(f"📁 Saved plot to: {path}")
#         plt.close()

# # -----------------------------
# # Load & prepare data
# # -----------------------------
# df = pd.read_csv(DATA_PATH)
# # df = df[df['label'] != 0].copy()              # keep only win/loss rows

# # ensure 'pair' is categorical → one-hot only for 'pair'
# if 'pair' in df.columns:
#     df['pair'] = df['pair'].astype('category')

# df = generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS)

# # Build feature list (exclude targets + raw labels)
# drop_cols = ['label', 'rr_label', 'y_meta3'] + [f'y_ge_{i}R' for i in RR_THRESHOLDS]
# features = [c for c in df.columns if c not in drop_cols]

# # One-hot only 'pair' (others are numeric already)
# if 'pair' in features:
#     features_no_pair = [c for c in features if c != 'pair']
#     X_all = pd.concat([df[features_no_pair], pd.get_dummies(df[['pair']], drop_first=True)], axis=1)
# else:
#     X_all = df[features].copy()

# # Targets
# y_reg_all  = pd.to_numeric(df['rr_label'], errors='coerce')
# y_meta_all = df['y_meta3'].astype('int8')
# y_bin_all  = {f'1_{i}': df[f'y_ge_{i}R'].astype('int8') for i in RR_THRESHOLDS}

# # Train/test split stratified by meta3
# X_train, X_test, y_reg_train, y_reg_test, y_meta_train, y_meta_test = train_test_split(
#     X_all, y_reg_all, y_meta_all, test_size=0.2, stratify=y_meta_all, random_state=RANDOM_STATE
# )

# # -----------------------------
# # 1) TUNE & OOF: Binary models (≥1R and ≥2R)
# # -----------------------------
# print("\n>>> Tuning Binary 1_1 (≥1R)")
# best_params_1 = tune_binary_params(X_train, y_bin_all['1_1'].loc[X_train.index], "xgb_bin_1_1")
# print("Best params 1_1:", best_params_1)

# print("\n>>> Tuning Binary 1_2 (≥2R)")
# best_params_2 = tune_binary_params(X_train, y_bin_all['1_2'].loc[X_train.index], "xgb_bin_1_2")
# print("Best params 1_2:", best_params_2)

# # OOF using best params
# oof_probs = {}
# models_dict = {}

# oof_1, models_1 = oof_binary_xgb(X_train, y_bin_all['1_1'].loc[X_train.index], best_params_1)
# oof_probs['1_1'] = oof_1
# models_dict['1_1'] = models_1

# oof_2, models_2 = oof_binary_xgb(X_train, y_bin_all['1_2'].loc[X_train.index], best_params_2)
# oof_probs['1_2'] = oof_2
# models_dict['1_2'] = models_2

# # -----------------------------
# # 2) TUNE & OOF: Regressor on stacked (bin OOF)
# # -----------------------------
# X_train_reg = create_reg_features(X_train, oof_probs=oof_probs)

# print("\n>>> Tuning Regressor (rr_label)")
# best_params_reg = tune_regressor_params(X_train_reg, y_reg_train, "xgb_reg_rr")
# print("Best params reg:", best_params_reg)

# oof_reg, reg_models = oof_regressor_xgb(X_train_reg, y_reg_train, best_params_reg)

# # -----------------------------
# # 3) TUNE & Train: Meta3 on stacked (bin OOF + reg OOF)
# # -----------------------------
# X_train_meta = create_meta3_features(X_train, oof_probs=oof_probs, reg_vec=oof_reg)

# print("\n>>> Tuning Meta3 (0,1,2)")
# best_params_meta = tune_meta_params(X_train_meta, y_meta_train, "xgb_meta3")
# print("Best params meta:", best_params_meta)

# # Train meta with inverse-freq weights
# w_map = inv_freq_weights_multi(y_meta_train)
# meta_w = y_meta_train.map(w_map).astype(float).values

# meta_clf = XGBClassifier(**best_params_meta)
# meta_clf.fit(X_train_meta, y_meta_train, sample_weight=meta_w)

# # -----------------------------
# # Holdout: evaluate (UNCHANGED)
# # -----------------------------
# print("\n=== BINARY CLASSIFIERS (Holdout) ===")
# p_test = {}

# def eval_binary(name, model_list, y_true, X):
#     p = np.mean([m.predict_proba(X)[:, 1] for m in model_list], axis=0)
#     t = best_threshold(y_true, p, metric='f1')
#     yhat = (p >= t).astype(int)
#     print(f"\n-- {name} --")
#     print("ROC AUC:", roc_auc_score(y_true, p))
#     print("Optimal threshold:", round(t, 3))
#     print("Accuracy:", accuracy_score(y_true, yhat))
#     print("F1:", f1_score(y_true, yhat))
#     print(confusion_matrix(y_true, yhat))
#     print(classification_report(y_true, yhat, target_names=['<thr','>=thr'], digits=4))
#     return p, t

# # Binary 1R
# p1, t1 = eval_binary("1_1 (y_ge_1R)", models_dict['1_1'], y_bin_all['1_1'].loc[X_test.index], X_test)
# p_test['1_1'] = p1

# # Binary 2R
# p2, t2 = eval_binary("1_2 (y_ge_2R)", models_dict['1_2'], y_bin_all['1_2'].loc[X_test.index], X_test)
# p_test['1_2'] = p2

# # Regressor
# X_test_reg  = create_reg_features(X_test, models_dict=models_dict)
# reg_holdout = np.mean([m.predict(X_test_reg) for m in reg_models], axis=0)
# print("\n=== REGRESSOR (rr_label) ===")
# print("MAE:", mean_absolute_error(y_reg_test, reg_holdout))
# print("RMSE:", rmse(y_reg_test, reg_holdout))
# print("R2:", r2_score(y_reg_test, reg_holdout))

# # Meta3
# X_test_meta = create_meta3_features(X_test, models_dict=models_dict, reg_models=reg_models)
# meta_pred = meta_clf.predict(X_test_meta)
# print("\n=== META3 (0=<1R, 1=1..2R, 2=>=2R) ===")
# target_names = [f"class_{c}" for c in sorted(np.unique(y_meta_test))]
# print(classification_report(y_meta_test, meta_pred, target_names=target_names, digits=4))
# print(confusion_matrix(y_meta_test, meta_pred))

# # -----------------------------
# # Holdout WIN-RATE SIMULATION (PNG saved)
# # -----------------------------
# rr_hold = pd.to_numeric(y_reg_test, errors='coerce').values

# mask_A = (p_test['1_1'] >= t1)
# evaluate_selection("A) Binary 1R filter (prob >= t1)", mask_A, rr_hold, rr_target=1, plot=True)

# mask_B = (p_test['1_2'] >= t2)
# evaluate_selection("B) Binary 2R filter (prob >= t2)", mask_B, rr_hold, rr_target=2, plot=True)

# mask_C = (p_test['1_1'] >= t1) & (p_test['1_2'] >= t2)
# evaluate_selection("C) Both binaries (≥1R & ≥2R)", mask_C, rr_hold, rr_target=2, plot=True)

# mask_D = (meta_pred == 2)
# evaluate_selection("D) Meta3 predicts >=2R", mask_D, rr_hold, rr_target=2, plot=True)

# q = 0.75
# thr_reg = np.quantile(reg_holdout, q)
# mask_E = (reg_holdout >= thr_reg)
# evaluate_selection(f"E) Regressor top 25% (thr={thr_reg:.3f})", mask_E, rr_hold, rr_target=2, plot=True)

# # -----------------------------
# # Final training on ALL data (single models) + save
# # -----------------------------
# print("\nTraining final XGB models on ALL data...")

# # Refit binaries on all data with tuned params
# # 1R
# neg = (y_bin_all['1_1'] == 0).sum()
# pos = (y_bin_all['1_1'] == 1).sum()
# spw_1 = max(1.0, neg / max(1, pos))
# clf1_all = XGBClassifier(**{**best_params_1, "scale_pos_weight": spw_1}).fit(X_all, y_bin_all['1_1'])

# # 2R
# neg = (y_bin_all['1_2'] == 0).sum()
# pos = (y_bin_all['1_2'] == 1).sum()
# spw_2 = max(1.0, neg / max(1, pos))
# clf2_all = XGBClassifier(**{**best_params_2, "scale_pos_weight": spw_2}).fit(X_all, y_bin_all['1_2'])

# # Regressor on all data (stack with proba from full models)
# p1_full = clf1_all.predict_proba(X_all)[:, 1]
# p2_full = clf2_all.predict_proba(X_all)[:, 1]
# X_all_reg = X_all.copy()
# X_all_reg['clf_1_1_prob'] = p1_full
# X_all_reg['clf_1_2_prob'] = p2_full

# reg_all = XGBRegressor(**best_params_reg).fit(X_all_reg, y_reg_all)

# # Meta on all data (stacked)
# reg_all_pred = reg_all.predict(X_all_reg)
# X_all_meta = X_all_reg.copy()
# X_all_meta['reg_pred'] = reg_all_pred

# w_map_all = inv_freq_weights_multi(y_meta_all)
# meta_w_all = y_meta_all.map(w_map_all).astype(float).values

# meta_all = XGBClassifier(**best_params_meta).fit(X_all_meta, y_meta_all, sample_weight=meta_w_all)

# # Save models (xgboost native)
# clf1_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_1.json"))
# clf2_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_2.json"))
# reg_all.save_model(os.path.join(MODEL_DIR, "xgb_regressor.json"))
# meta_all.save_model(os.path.join(MODEL_DIR, "xgb_meta3.json"))

# # Save metadata json
# meta_info = {
#     "features_base": list(X_all.columns),
#     "features_reg": list(X_all_reg.columns),
#     "features_meta": list(X_all_meta.columns),
#     "rr_thresholds": RR_THRESHOLDS,
#     "meta3_classes": ["<1R", "1..2R", ">=2R"],
#     "notes": "XGB two-binary + reg + meta3 with Optuna. Pair one-hot encoded.",
#     "best_params": {
#         "bin_1_1": best_params_1,
#         "bin_1_2": best_params_2,
#         "reg": best_params_reg,
#         "meta": best_params_meta
#     }
# }
# with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
#     json.dump(meta_info, f, indent=2)

# # -----------------------------
# # Feature Importances (CSV)
# # -----------------------------
# dump_importance_xgb(clf1_all, list(X_all.columns), os.path.join(MODEL_DIR, "feat_importance_classifier_1_1.csv"), "Classifier 1_1")
# dump_importance_xgb(clf2_all, list(X_all.columns), os.path.join(MODEL_DIR, "feat_importance_classifier_1_2.csv"), "Classifier 1_2")
# dump_importance_xgb(reg_all,  list(X_all_reg.columns), os.path.join(MODEL_DIR, "feat_importance_regressor.csv"),         "Regressor")
# dump_importance_xgb(meta_all, list(X_all_meta.columns), os.path.join(MODEL_DIR, "feat_importance_meta3.csv"),            "Meta3")

# print("\n✅ XGBoost models tuned, evaluated, and saved.")
# ==============================#
#   XGB PIPELINE (2 BIN + REG + META 0/1/2) + OPTUNA TUNING
#   Binaries:  y_ge_1R, y_ge_2R
#   Regressor: rr_label
#   Meta:      0:<1R, 1:1..2R, 2:>=2R
#   + Holdout Win-Rate Simulation (PNG plots)
#   + Optuna tuning for all stages
#   + Feature Importances CSV
#   (NO OOF, NO EARLY STOPPING)
# ==============================#

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for CLI/headless
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

from xgboost import XGBClassifier, XGBRegressor
import optuna

# -----------------------------
# Config
# -----------------------------
DATA_PATH      = "test-combined.csv"       # your combined labeled dataset
MODEL_DIR      = "./xgb_artifacts_meta3_opt/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE   = 42

# Optuna trials per stage
N_TRIALS_BIN   = 15
N_TRIALS_REG   = 15
N_TRIALS_META  = 15

# thresholds we care about
RR_THRESHOLDS  = [1, 2]  # ONLY 1R and 2R

# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

def best_threshold(y_true, p, metric='f1'):
    grid = np.linspace(0.05, 0.95, 19)
    best, best_t = -1.0, 0.5
    for t in grid:
        yhat = (p >= t).astype(int)
        s = f1_score(y_true, yhat) if metric == 'f1' else accuracy_score(y_true, yhat)
        if s > best:
            best, best_t = s, t
    return float(best_t)

def inv_freq_weights_multi(y):
    counts = pd.Series(y).value_counts()
    N, K = len(y), counts.shape[0]
    w = {int(c): float(N / (K * counts.get(c, 1))) for c in counts.index}
    return w

def dump_importance_xgb(model, cols, path, title):
    try:
        imp = pd.DataFrame({
            "Feature": cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        imp.to_csv(path, index=False)
        print(f"{title} top10:\n", imp.head(10), "\n")
        return imp
    except Exception as e:
        print(f"⚠️ No feature_importances_ for {title}: {e}")
        return pd.DataFrame()

# -----------------------------
# Labels
# -----------------------------
def generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS):
    rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
    thr_list = sorted(set(thr_list))
    n = len(df)

    print(rr.describe([.5, .75, .90, .95, .99]))
    for t in [1, 2, 3, 4, 5]:
        share = (rr >= t).mean()
        print(f"P(rr ≥ {t}) = {share:.3f}")

    if 'pair' in df.columns:
        print("\nShare ≥ last threshold by pair (top 10):")
        print(
            df.groupby('pair')['rr_label']
              .apply(lambda s: (pd.to_numeric(s, errors='coerce') >= thr_list[-1]).mean())
              .sort_values(ascending=False)
              .head(10)
        )

    print("\n=== Binary targets (y_ge_{T}R) ===")
    for thr in thr_list:
        col = f'y_ge_{thr}R'
        df[col] = (rr >= thr).astype('int8')
        vc = df[col].value_counts().reindex([0, 1], fill_value=0)
        neg, pos = int(vc[0]), int(vc[1])
        print(f"{col}: pos={pos} ({pos/n:.2%}) | neg={neg} ({neg/n:.2%})")

    # meta3: 0:<1R, 1:[1,2), 2:>=2R
    bins = [-np.inf, 1, 2, np.inf]
    df['y_meta3'] = pd.cut(rr, bins=bins, labels=False, right=False, include_lowest=True).astype('int8')

    names = ["<1R", "1..2R", ">=2R"]
    print("\n=== Meta3 target (y_meta3) ===")
    vc_meta = df['y_meta3'].value_counts().sort_index()
    for k in range(3):
        cnt = int(vc_meta.get(k, 0))
        print(f"class {k} ({names[k]:>5}): {cnt} ({cnt/n:.2%})")
    print(f"TOTAL rows: {n}")
    return df

# -----------------------------
# Tuning (Optuna)
# -----------------------------
def tune_binary_params(X, y, study_name):
    def objective(trial):
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 200, 900),
            learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth    = trial.suggest_int("max_depth", 3, 8),
            subsample    = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            objective    = "binary:logistic",
            eval_metric  = "auc",
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        # CV AUC
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr, va in skf.split(X, y):
            y_tr = y.iloc[tr]
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            spw = max(1.0, neg / max(1, pos))
            params_fold = {**params, "scale_pos_weight": spw}
            m = XGBClassifier(**params_fold)
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], p))
        return float(np.mean(aucs))
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=N_TRIALS_BIN, show_progress_bar=True)
    print(f"[{study_name}] Best AUC: {study.best_value:.5f}")
    print(f"[{study_name}] Best params: {study.best_params}")
    return study.best_params

def tune_regressor_params(X, y, study_name):
    def objective(trial):
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 300, 1000),
            learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth    = trial.suggest_int("max_depth", 4, 10),
            subsample    = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            objective    = "reg:squarederror",
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        # CV RMSE (lower is better)
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        rmses = []
        for tr, va in kf.split(X, y):
            m = XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[va])
            rmses.append(rmse(y.iloc[va].values, pred))
        return float(np.mean(rmses))
    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=N_TRIALS_REG, show_progress_bar=True)
    print(f"[{study_name}] Best RMSE: {study.best_value:.5f}")
    print(f"[{study_name}] Best params: {study.best_params}")
    return study.best_params

def tune_meta_params(X, y, study_name):
    # 3 classes, we’ll optimize weighted-F1 via 3-fold CV
    def objective(trial):
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 200, 900),
            learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth    = trial.suggest_int("max_depth", 4, 10),
            subsample    = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda   = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha    = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            objective    = "multi:softprob",
            eval_metric  = "mlogloss",
            num_class    = 3,
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        f1s = []
        for tr, va in skf.split(X, y):
            # inverse-freq weights to avoid class 1 collapse
            w_map = inv_freq_weights_multi(y.iloc[tr])
            sw = y.iloc[tr].map(w_map).astype(float).values
            m = XGBClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sw)
            pred = m.predict(X.iloc[va])
            f1s.append(f1_score(y.iloc[va], pred, average='weighted'))
        return float(np.mean(f1s))
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=N_TRIALS_META, show_progress_bar=True)
    print(f"[{study_name}] Best weighted-F1: {study.best_value:.5f}")
    print(f"[{study_name}] Best params: {study.best_params}")
    return study.best_params

# -----------------------------
# Stacking feature builders (NO OOF)
# -----------------------------
def create_reg_features(X, models_dict=None):
    Xf = X.copy()
    if models_dict is not None:
        for name, model in models_dict.items():
            Xf[f'clf_{name}_prob'] = model.predict_proba(X)[:, 1]
    return Xf

def create_meta3_features(X, models_dict=None, reg_model=None):
    Xf = create_reg_features(X, models_dict=models_dict)
    if reg_model is not None:
        Xf['reg_pred'] = reg_model.predict(Xf)
    return Xf

# -----------------------------
# PNG-saving evaluator
# -----------------------------
def evaluate_selection(name, mask, rr, rr_target=None, plot=False, save_dir="selection_plots_xgb"):
    os.makedirs(save_dir, exist_ok=True)
    mask = np.asarray(mask, dtype=bool)
    rr   = np.asarray(pd.to_numeric(rr, errors="coerce"))
    L = min(len(mask), len(rr))
    mask, rr = mask[:L], rr[:L]
    ok = ~np.isnan(rr)
    mask, rr = mask & ok, rr[ok]

    n = int(mask.sum())
    if n == 0:
        print(f"\n[{name}] No trades selected.")
        return

    rr_sel = rr[mask]
    win_rate = float((rr_sel >= (rr_target if rr_target is not None else 0)).mean()) if rr_target is not None else np.nan
    avg_rr = float(rr_sel.mean())
    med_rr = float(np.median(rr_sel))

    print(f"\n[{name}]")
    print(f"Trades taken: {n}")
    if rr_target is not None:
        print(f"Win rate (≥ {rr_target}R): {win_rate:.3f}")
    print(f"Average RR: {avg_rr:.3f}")
    print(f"Median RR:  {med_rr:.3f}")

    if plot:
        plt.figure()
        plt.hist(rr_sel, bins=40)
        plt.title(name)
        plt.xlabel("RR")
        plt.ylabel("Count")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace(":", "").replace("/", "-").replace("\\", "-")
        path = os.path.join(save_dir, f"{safe_name}.png")
        plt.savefig(path, dpi=300)
        print(f"📁 Saved plot to: {path}")
        plt.close()

# -----------------------------
# Load & prepare data
# -----------------------------
df = pd.read_csv(DATA_PATH)
# df = df[df['label'] != 0].copy()              # keep only win/loss rows

# ensure 'pair' is categorical → one-hot only for 'pair'
if 'pair' in df.columns:
    df['pair'] = df['pair'].astype('category')

df = generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS)

# Build feature list (exclude targets + raw labels)
drop_cols = ['label', 'rr_label', 'y_meta3'] + [f'y_ge_{i}R' for i in RR_THRESHOLDS]
features = [c for c in df.columns if c not in drop_cols]

# One-hot only 'pair' (others are numeric already)
if 'pair' in features:
    features_no_pair = [c for c in features if c != 'pair']
    X_all = pd.concat([df[features_no_pair], pd.get_dummies(df[['pair']], drop_first=True)], axis=1)
else:
    X_all = df[features].copy()

# Targets
y_reg_all  = pd.to_numeric(df['rr_label'], errors='coerce')
y_meta_all = df['y_meta3'].astype('int8')
y_bin_all  = {f'1_{i}': df[f'y_ge_{i}R'].astype('int8') for i in RR_THRESHOLDS}

# Train/test split stratified by meta3
X_train, X_test, y_reg_train, y_reg_test, y_meta_train, y_meta_test = train_test_split(
    X_all, y_reg_all, y_meta_all, test_size=0.2, stratify=y_meta_all, random_state=RANDOM_STATE
)

# -----------------------------
# 1) TUNE & Train: Binary models (≥1R and ≥2R)  [NO OOF]
# -----------------------------
print("\n>>> Tuning Binary 1_1 (≥1R)")
best_params_1 = tune_binary_params(X_train, y_bin_all['1_1'].loc[X_train.index], "xgb_bin_1_1")
print("Best params 1_1:", best_params_1)

print("\n>>> Tuning Binary 1_2 (≥2R)")
best_params_2 = tune_binary_params(X_train, y_bin_all['1_2'].loc[X_train.index], "xgb_bin_1_2")
print("Best params 1_2:", best_params_2)

# Train single binary models on training split
neg = (y_bin_all['1_1'].loc[X_train.index] == 0).sum()
pos = (y_bin_all['1_1'].loc[X_train.index] == 1).sum()
spw_1_tr = max(1.0, neg / max(1, pos))
clf1_tr = XGBClassifier(**{**best_params_1, "scale_pos_weight": spw_1_tr}).fit(X_train, y_bin_all['1_1'].loc[X_train.index])

neg = (y_bin_all['1_2'].loc[X_train.index] == 0).sum()
pos = (y_bin_all['1_2'].loc[X_train.index] == 1).sum()
spw_2_tr = max(1.0, neg / max(1, pos))
clf2_tr = XGBClassifier(**{**best_params_2, "scale_pos_weight": spw_2_tr}).fit(X_train, y_bin_all['1_2'].loc[X_train.index])

# -----------------------------
# 2) TUNE & Train: Regressor on stacked (bin TRAIN preds) [NO OOF]
# -----------------------------
models_dict_tr = {"1_1": clf1_tr, "1_2": clf2_tr}
X_train_reg = create_reg_features(X_train, models_dict=models_dict_tr)

print("\n>>> Tuning Regressor (rr_label)")
best_params_reg = tune_regressor_params(X_train_reg, y_reg_train, "xgb_reg_rr")
print("Best params reg:", best_params_reg)

reg_tr = XGBRegressor(**best_params_reg).fit(X_train_reg, y_reg_train)

# -----------------------------
# 3) TUNE & Train: Meta3 on stacked (bin TRAIN preds + reg TRAIN pred) [NO OOF]
# -----------------------------
X_train_meta = create_meta3_features(X_train, models_dict=models_dict_tr, reg_model=reg_tr)

print("\n>>> Tuning Meta3 (0,1,2)")
best_params_meta = tune_meta_params(X_train_meta, y_meta_train, "xgb_meta3")
print("Best params meta:", best_params_meta)

# Train meta with inverse-freq weights
w_map = inv_freq_weights_multi(y_meta_train)
meta_w = y_meta_train.map(w_map).astype(float).values
meta_tr = XGBClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, sample_weight=meta_w)

# -----------------------------
# Holdout: evaluate
# -----------------------------
print("\n=== BINARY CLASSIFIERS (Holdout) ===")
def eval_binary(name, model, y_true, X):
    p = model.predict_proba(X)[:, 1]
    t = best_threshold(y_true, p, metric='f1')
    yhat = (p >= t).astype(int)
    print(f"\n-- {name} --")
    print("ROC AUC:", roc_auc_score(y_true, p))
    print("Optimal threshold:", round(t, 3))
    print("Accuracy:", accuracy_score(y_true, yhat))
    print("F1:", f1_score(y_true, yhat))
    print(confusion_matrix(y_true, yhat))
    print(classification_report(y_true, yhat, target_names=['<thr','>=thr'], digits=4))
    return p, t

p_test = {}
p1, t1 = eval_binary("1_1 (y_ge_1R)", clf1_tr, y_bin_all['1_1'].loc[X_test.index], X_test)
p_test['1_1'] = p1
p2, t2 = eval_binary("1_2 (y_ge_2R)", clf2_tr, y_bin_all['1_2'].loc[X_test.index], X_test)
p_test['1_2'] = p2

# Regressor
X_test_reg  = create_reg_features(X_test, models_dict=models_dict_tr)
reg_holdout = reg_tr.predict(X_test_reg)
print("\n=== REGRESSOR (rr_label) ===")
print("MAE:", mean_absolute_error(y_reg_test, reg_holdout))
print("RMSE:", rmse(y_reg_test, reg_holdout))
print("R2:", r2_score(y_reg_test, reg_holdout))

# Meta3
X_test_meta = create_meta3_features(X_test, models_dict=models_dict_tr, reg_model=reg_tr)
meta_pred = meta_tr.predict(X_test_meta)
print("\n=== META3 (0=<1R, 1=1..2R, 2=>=2R) ===")
target_names = [f"class_{c}" for c in sorted(np.unique(y_meta_test))]
print(classification_report(y_meta_test, meta_pred, target_names=target_names, digits=4))
print(confusion_matrix(y_meta_test, meta_pred))

# -----------------------------
# Holdout WIN-RATE SIMULATION (PNG saved)
# -----------------------------
rr_hold = pd.to_numeric(y_reg_test, errors='coerce').values

mask_A = (p_test['1_1'] >= t1)
evaluate_selection("A) Binary 1R filter (prob >= t1)", mask_A, rr_hold, rr_target=1, plot=True)

mask_B = (p_test['1_2'] >= t2)
evaluate_selection("B) Binary 2R filter (prob >= t2)", mask_B, rr_hold, rr_target=2, plot=True)

mask_C = (p_test['1_1'] >= t1) & (p_test['1_2'] >= t2)
evaluate_selection("C) Both binaries (≥1R & ≥2R)", mask_C, rr_hold, rr_target=2, plot=True)

mask_D = (meta_pred == 2)
evaluate_selection("D) Meta3 predicts >=2R", mask_D, rr_hold, rr_target=2, plot=True)

q = 0.75
thr_reg = np.quantile(reg_holdout, q)
mask_E = (reg_holdout >= thr_reg)
evaluate_selection(f"E) Regressor top 25% (thr={thr_reg:.3f})", mask_E, rr_hold, rr_target=2, plot=True)

# -----------------------------
# Final training on ALL data (single models) + save
# -----------------------------
print("\nTraining final XGB models on ALL data...")

# Refit binaries on all data with tuned params
neg = (y_bin_all['1_1'] == 0).sum()
pos = (y_bin_all['1_1'] == 1).sum()
spw_1 = max(1.0, neg / max(1, pos))
clf1_all = XGBClassifier(**{**best_params_1, "scale_pos_weight": spw_1}).fit(X_all, y_bin_all['1_1'])

neg = (y_bin_all['1_2'] == 0).sum()
pos = (y_bin_all['1_2'] == 1).sum()
spw_2 = max(1.0, neg / max(1, pos))
clf2_all = XGBClassifier(**{**best_params_2, "scale_pos_weight": spw_2}).fit(X_all, y_bin_all['1_2'])

# Regressor on all data (stack with proba from full models)
p1_full = clf1_all.predict_proba(X_all)[:, 1]
p2_full = clf2_all.predict_proba(X_all)[:, 1]
X_all_reg = X_all.copy()
X_all_reg['clf_1_1_prob'] = p1_full
X_all_reg['clf_1_2_prob'] = p2_full

reg_all = XGBRegressor(**best_params_reg).fit(X_all_reg, y_reg_all)

# Meta on all data (stacked)
reg_all_pred = reg_all.predict(X_all_reg)
X_all_meta = X_all_reg.copy()
X_all_meta['reg_pred'] = reg_all_pred

w_map_all = inv_freq_weights_multi(y_meta_all)
meta_w_all = y_meta_all.map(w_map_all).astype(float).values

meta_all = XGBClassifier(**best_params_meta).fit(X_all_meta, y_meta_all, sample_weight=meta_w_all)

# Save models (xgboost native)
clf1_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_1.json"))
clf2_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_2.json"))
reg_all.save_model(os.path.join(MODEL_DIR, "xgb_regressor.json"))
meta_all.save_model(os.path.join(MODEL_DIR, "xgb_meta3.json"))

# Save metadata json
meta_info = {
    "features_base": list(X_all.columns),
    "features_reg": list(X_all_reg.columns),
    "features_meta": list(X_all_meta.columns),
    "rr_thresholds": RR_THRESHOLDS,
    "meta3_classes": ["<1R", "1..2R", ">=2R"],
    "notes": "XGB two-binary + reg + meta3 with Optuna. Pair one-hot encoded. NO OOF / NO EARLY STOPPING.",
    "best_params": {
        "bin_1_1": best_params_1,
        "bin_1_2": best_params_2,
        "reg": best_params_reg,
        "meta": best_params_meta
    }
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(meta_info, f, indent=2)

# -----------------------------
# Feature Importances (CSV)
# -----------------------------
dump_importance_xgb(clf1_all, list(X_all.columns), os.path.join(MODEL_DIR, "feat_importance_classifier_1_1.csv"), "Classifier 1_1")
dump_importance_xgb(clf2_all, list(X_all.columns), os.path.join(MODEL_DIR, "feat_importance_classifier_1_2.csv"), "Classifier 1_2")
dump_importance_xgb(reg_all,  list(X_all_reg.columns), os.path.join(MODEL_DIR, "feat_importance_regressor.csv"),         "Regressor")
dump_importance_xgb(meta_all, list(X_all_meta.columns), os.path.join(MODEL_DIR, "feat_importance_meta3.csv"),            "Meta3")

print("\n✅ XGBoost models tuned, evaluated, and saved.")
