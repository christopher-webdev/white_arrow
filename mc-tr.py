# ==============================#
#   XGB PIPELINE (2 BIN + REG + META 0/1/2)
#   Binaries:  y_ge_1R, y_ge_2R
#   Regressor: rr_label
#   Meta:      0:<1R, 1:1..2R, 2:>=2R
#   + Holdout Win-Rate Simulation (PNG plots)
# ==============================#

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import (
#     accuracy_score, f1_score, roc_auc_score,
#     mean_absolute_error, mean_squared_error, r2_score,
#     classification_report, confusion_matrix
# )

# from xgboost import XGBClassifier, XGBRegressor

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH      = "test-combined.csv"       # your combined labeled dataset
# MODEL_DIR      = "./xgb_artifacts_meta3/"
# os.makedirs(MODEL_DIR, exist_ok=True)

# RANDOM_STATE   = 42
# N_SPLITS_OOF   = 3

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
# def oof_binary_xgb(X, y, params=None, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     if params is None:
#         params = dict(
#             n_estimators=400, learning_rate=0.05,
#             max_depth=5, subsample=0.8, colsample_bytree=0.8,
#             reg_lambda=1.0, reg_alpha=0.0,
#             objective="binary:logistic",
#             eval_metric="auc", random_state=seed, n_jobs=-1
#         )
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in skf.split(X, y):
#         # class balance -> scale_pos_weight
#         neg = (y.iloc[tr] == 0).sum()
#         pos = (y.iloc[tr] == 1).sum()
#         spw = max(1.0, neg / max(1, pos))
#         params_fold = {**params, "scale_pos_weight": spw}
#         m = XGBClassifier(**params_fold)
#         m.fit(X.iloc[tr], y.iloc[tr])
#         oof[va] = m.predict_proba(X.iloc[va])[:, 1]
#         models.append(m)
#     return oof, models

# def oof_regressor_xgb(X, y, params=None, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     if params is None:
#         params = dict(
#             n_estimators=600, learning_rate=0.05,
#             max_depth=6, subsample=0.8, colsample_bytree=0.8,
#             reg_lambda=1.0, reg_alpha=0.0,
#             objective="reg:squarederror",
#             random_state=seed, n_jobs=-1
#         )
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
# # Meta3 features from stack
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
#     import os
#     os.makedirs(save_dir, exist_ok=True)

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
#         safe_name = name.replace(" ", "_").replace(":", "").replace("/", "-")
#         path = os.path.join(save_dir, f"{safe_name}.png")
#         plt.savefig(path, dpi=300)
#         print(f"📁 Saved plot to: {path}")
#         plt.close()

# # -----------------------------
# # Load & prepare data
# # -----------------------------
# df = pd.read_csv(DATA_PATH)
# df = df[df['label'] != 0].copy()              # keep only win/loss rows

# # ensure 'pair' is categorical → one-hot only for 'pair' to keep it simple for XGB
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
# # Train: 2 Binary models (OOF) + Regressor (OOF) + Meta
# # -----------------------------
# oof_probs = {}
# models_dict = {}

# # Binary 1R
# oof_1, models_1 = oof_binary_xgb(X_train, y_bin_all['1_1'].loc[X_train.index])
# oof_probs['1_1'] = oof_1
# models_dict['1_1'] = models_1

# # Binary 2R
# oof_2, models_2 = oof_binary_xgb(X_train, y_bin_all['1_2'].loc[X_train.index])
# oof_probs['1_2'] = oof_2
# models_dict['1_2'] = models_2

# # Regressor OOF (stack on bin OOF)
# X_train_reg = create_reg_features(X_train, oof_probs=oof_probs)
# oof_reg, reg_models = oof_regressor_xgb(X_train_reg, y_reg_train)

# # Meta3 training on stacked features
# X_train_meta = create_meta3_features(X_train, oof_probs=oof_probs, reg_vec=oof_reg)

# # meta sample weights (inverse-freq)
# w_map = inv_freq_weights_multi(y_meta_train)
# meta_w = y_meta_train.map(w_map).astype(float).values

# meta_clf = XGBClassifier(
#     n_estimators=500, learning_rate=0.05, max_depth=6,
#     subsample=0.8, colsample_bytree=0.8,
#     reg_lambda=1.0, reg_alpha=0.0,
#     objective="multi:softprob", eval_metric="mlogloss",
#     num_class=3, random_state=RANDOM_STATE, n_jobs=-1
# )
# meta_clf.fit(X_train_meta, y_meta_train, sample_weight=meta_w)

# # -----------------------------
# # Holdout: evaluate
# # -----------------------------
# print("\n=== BINARY CLASSIFIERS (Holdout) ===")
# p_test = {}

# # Eval helper for binary
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

# # Refit binaries on all data
# # 1R
# neg = (y_bin_all['1_1'] == 0).sum()
# pos = (y_bin_all['1_1'] == 1).sum()
# spw_1 = max(1.0, neg / max(1, pos))
# clf1_all = XGBClassifier(
#     n_estimators=400, learning_rate=0.05, max_depth=5,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#     objective="binary:logistic", eval_metric="auc",
#     random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=spw_1
# ).fit(X_all, y_bin_all['1_1'])

# # 2R
# neg = (y_bin_all['1_2'] == 0).sum()
# pos = (y_bin_all['1_2'] == 1).sum()
# spw_2 = max(1.0, neg / max(1, pos))
# clf2_all = XGBClassifier(
#     n_estimators=400, learning_rate=0.05, max_depth=5,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#     objective="binary:logistic", eval_metric="auc",
#     random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=spw_2
# ).fit(X_all, y_bin_all['1_2'])

# # Regressor on all data (stack with proba from full models)
# p1_full = clf1_all.predict_proba(X_all)[:, 1]
# p2_full = clf2_all.predict_proba(X_all)[:, 1]
# X_all_reg = X_all.copy()
# X_all_reg['clf_1_1_prob'] = p1_full
# X_all_reg['clf_1_2_prob'] = p2_full

# reg_all = XGBRegressor(
#     n_estimators=600, learning_rate=0.05, max_depth=6,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#     objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
# ).fit(X_all_reg, y_reg_all)

# # Meta on all data (stacked)
# reg_all_pred = reg_all.predict(X_all_reg)
# X_all_meta = X_all_reg.copy()
# X_all_meta['reg_pred'] = reg_all_pred

# w_map_all = inv_freq_weights_multi(y_meta_all)
# meta_w_all = y_meta_all.map(w_map_all).astype(float).values

# meta_all = XGBClassifier(
#     n_estimators=500, learning_rate=0.05, max_depth=6,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#     objective="multi:softprob", eval_metric="mlogloss",
#     num_class=3, random_state=RANDOM_STATE, n_jobs=-1
# ).fit(X_all_meta, y_meta_all, sample_weight=meta_w_all)

# # Save models (xgboost native)
# clf1_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_1.json"))
# clf2_all.save_model(os.path.join(MODEL_DIR, "xgb_classifier_1_2.json"))
# reg_all.save_model(os.path.join(MODEL_DIR, "xgb_regressor.json"))
# meta_all.save_model(os.path.join(MODEL_DIR, "xgb_meta3.json"))

# # Save a small metadata json
# import json
# meta_info = {
#     "features": list(X_all.columns),
#     "rr_thresholds": RR_THRESHOLDS,
#     "meta3_classes": ["<1R", "1..2R", ">=2R"],
#     "notes": "XGB two-binary + reg + meta3. Pair one-hot encoded."
# }
# with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
#     json.dump(meta_info, f, indent=2)

# print("\n✅ XGBoost models trained and saved.")


# # # ==============================#
# # #  STACKED RR TRAINING PIPELINE #
# # #    (1:1..1:5 + Meta 0..5)     #
# # # ==============================#

# # import os
# # import joblib
# # import numpy as np# ==============================#
# #   RR PIPELINE (2 BIN + REG + META 0/1/2)
# #   + HOLDOUT WIN-RATE SIMULATION
# #   Binaries:  y_ge_1R, y_ge_2R
# #   Regressor: rr_label
# #   Meta:      0:<1R, 1:1..2R, 2:>=2R
# # ==============================#

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import optuna
# from lightgbm import LGBMClassifier, LGBMRegressor

# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import (
#     mean_absolute_error, mean_squared_error, r2_score,
#     accuracy_score, f1_score, roc_auc_score,
#     confusion_matrix, classification_report
# )
# from sklearn.base import clone

# import matplotlib.pyplot as plt  # for histograms

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH = "test-combined.csv"   # your combined labeled dataset
# MODEL_DIR = "./tmodel_artifacts_r2_meta3_v1/"
# os.makedirs(MODEL_DIR, exist_ok=True)

# RANDOM_STATE   = 42
# N_TRIALS_BIN   = 5
# N_TRIALS_REG   = 10
# N_TRIALS_META  = 5
# N_SPLITS_OOF   = 3
# N_SPLITS_CV    = 3

# # thresholds we care about
# RR_THRESHOLDS = [1, 2]  # ONLY 1R and 2R

# # -----------------------------
# # Label builder
# # -----------------------------
# def generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS):
#     """
#     Builds:
#       - y_ge_{1,2}R binaries
#       - y_meta3 in {0,1,2}: 0:<1R, 1:[1,2)R, 2:>=2R
#       - prints diagnostics
#     """
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

#     # binaries
#     print("\n=== Binary targets (y_ge_{T}R) ===")
#     for thr in thr_list:
#         col = f'y_ge_{thr}R'
#         df[col] = (rr >= thr).astype('int8')
#         vc = df[col].value_counts().reindex([0,1], fill_value=0)
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

# # Map classifier names to actual target columns
# CLASSIFIER_TARGETS = {f'1_{i}': f'y_ge_{i}R' for i in RR_THRESHOLDS}   # {'1_1': 'y_ge_1R', '1_2': 'y_ge_2R'}

# # -----------------------------
# # Utilities
# # -----------------------------
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def inv_freq_weights_binary(y):
#     counts = pd.Series(y).value_counts()
#     N, K = len(y), 2
#     return {int(c): float(N / (K * counts.get(c, 1))) for c in [0, 1]}

# def inv_freq_weights_multi(y):
#     counts = pd.Series(y).value_counts()
#     N, K = len(y), counts.shape[0]
#     return {int(c): float(N / (K * counts.get(c, 1))) for c in counts.index}

# def best_threshold(y_true, p, metric='f1'):
#     grid = np.linspace(0.05, 0.95, 19)
#     best, best_t = -1.0, 0.5
#     for t in grid:
#         yhat = (p >= t).astype(int)
#         if metric == 'f1':
#             s = f1_score(y_true, yhat)
#         else:
#             s = accuracy_score(y_true, yhat)
#         if s > best:
#             best, best_t = s, t
#     return float(best_t)

# # -----------------------------
# # CV scorers
# # -----------------------------
# def cv_score_classifier(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     scores = []
#     for tr, va in skf.split(X, y):
#         m = LGBMClassifier(**params)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         prob = m.predict_proba(X.iloc[va])[:, 1]
#         scores.append(roc_auc_score(y.iloc[va], prob))
#     return float(np.mean(scores))

# def cv_score_regressor(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     scores = []
#     for tr, va in kf.split(X, y):
#         m = LGBMRegressor(**params)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         pred = m.predict(X.iloc[va])
#         scores.append(rmse(y.iloc[va], pred))
#     return float(np.mean(scores))  # lower is better

# # -----------------------------
# # Optuna tuning
# # -----------------------------
# def tune_classifier_params(X, y, weight_grid=None, name="clf"):
#     if weight_grid is None:
#         weight_grid = [
#             None,
#             "balanced",
#             inv_freq_weights_binary(y),
#             {0: 1.0, 1: 2.0},
#             {0: 1.0, 1: 3.0}
#         ]
#     def objective(trial):
#         params = {
#             "objective": "binary",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": trial.suggest_categorical("class_weight", weight_grid),
#         }
#         return cv_score_classifier(params, X, y)
#     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_BIN)
#     return study.best_params

# def tune_regressor_params(X, y, name="reg"):
#     def objective(trial):
#         params = {
#             "objective": "regression",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         }
#         return cv_score_regressor(params, X, y)  # lower is better
#     study = optuna.create_study(direction="minimize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_REG)
#     return study.best_params

# def tune_meta3_params(X, y, name="meta3"):
#     weight_grid = [
#         None,
#         "balanced",
#         inv_freq_weights_multi(y)
#     ]
#     def objective(trial):
#         params = {
#             "objective": "multiclass",
#             "num_class": int(len(np.unique(y))),  # should be 3
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": trial.suggest_categorical("class_weight", weight_grid)
#         }
#         skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
#         # Macro-F1 to not ignore any class
#         f1s = []
#         for tr, va in skf.split(X, y):
#             m = LGBMClassifier(**params)
#             m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#             pred = m.predict(X.iloc[va])
#             f1s.append(f1_score(y.iloc[va], pred, average='macro'))
#         return float(np.mean(f1s))
#     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_META)
#     return study.best_params

# # -----------------------------
# # OOF helpers (no leakage)
# # -----------------------------
# def oof_binary(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in skf.split(X, y):
#         m = clone(base_estimator)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         oof[va] = m.predict_proba(X.iloc[va])[:, 1]
#         models.append(m)
#     return oof, models

# def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in kf.split(X, y):
#         m = clone(base_estimator)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         oof[va] = m.predict(X.iloc[va])
#         models.append(m)
#     return oof, models

# # -----------------------------
# # Feature builders (stacking)
# # -----------------------------
# def create_reg_features(X, oof_probs=None, models_dict=None):
#     Xf = X.copy()
#     if oof_probs is not None:
#         for name in CLASSIFIER_TARGETS.keys():
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
# # Load & prepare data
# # -----------------------------
# df = pd.read_csv(DATA_PATH)
# # df = df[df['label'] != 0].copy()   # keep only win/loss rows
# df['pair'] = df['pair'].astype('category')
# df = generate_rr_labels_and_meta3(df, thr_list=RR_THRESHOLDS)

# # Build feature list (exclude targets + raw labels)
# drop_cols = ['label', 'rr_label', 'y_meta3'] + [f'y_ge_{i}R' for i in RR_THRESHOLDS]
# features = [c for c in df.columns if c not in drop_cols]

# # Split (stratify on y_meta3)
# y_bin_cols = [f'y_ge_{i}R' for i in RR_THRESHOLDS]
# X_train, X_test, y_reg_train, y_reg_test, y_bin_train, y_bin_test, y_meta_train, y_meta_test = train_test_split(
#     df[features],
#     df['rr_label'],
#     df[y_bin_cols],
#     df['y_meta3'],
#     test_size=0.2,
#     stratify=df['y_meta3'],
#     random_state=RANDOM_STATE
# )

# # -----------------------------
# # 1) Tune + OOF both binary classifiers (≥1R, ≥2R)
# # -----------------------------
# oof_probs = {}
# models_dict = {}
# best_params_dict = {}
# opt_thresholds = {}

# for name, col in CLASSIFIER_TARGETS.items():
#     print(f"\n>>> Tuning {name} ({col})")
#     best_params = tune_classifier_params(X_train, y_bin_train[col], name=f"clf_{name}")
#     clf_base = LGBMClassifier(**best_params)
#     oof, models = oof_binary(clf_base, X_train, y_bin_train[col])
#     oof_probs[name] = oof
#     models_dict[name] = models
#     best_params_dict[name] = best_params

# # -----------------------------
# # 2) Tune regressor on OOF probs (no leakage)
# # -----------------------------
# X_train_reg = create_reg_features(X_train, oof_probs=oof_probs)
# best_params_reg = tune_regressor_params(X_train_reg, y_reg_train, name="reg")
# reg_base = LGBMRegressor(**best_params_reg)
# oof_reg, reg_models = oof_regressor(reg_base, X_train_reg, y_reg_train)

# # -----------------------------
# # 3) Tune meta3 on OOF features
# # -----------------------------
# X_train_meta = create_meta3_features(X_train, oof_probs=oof_probs, reg_vec=oof_reg)
# best_params_meta = tune_meta3_params(X_train_meta, y_meta_train, name="meta3")
# meta_model = LGBMClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, categorical_feature=['pair'])

# # -----------------------------
# # 4) Holdout Evaluation
# # -----------------------------
# print("\n=== BINARY CLASSIFIERS (Holdout) ===")
# p_test_dict = {}
# for name, models in models_dict.items():
#     col = CLASSIFIER_TARGETS[name]
#     p_test = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
#     t_opt = best_threshold(y_bin_test[col], p_test, metric='f1')
#     y_pred = (p_test >= t_opt).astype(int)
#     print(f"\n-- {name} ({col}) --")
#     print("ROC AUC:", roc_auc_score(y_bin_test[col], p_test))
#     print("Optimal threshold:", round(t_opt, 3))
#     print("Accuracy:", accuracy_score(y_bin_test[col], y_pred))
#     print("F1:", f1_score(y_bin_test[col], y_pred))
#     print(confusion_matrix(y_bin_test[col], y_pred))
#     print(classification_report(y_bin_test[col], y_pred, target_names=['<thr','>=thr'], digits=4))
#     opt_thresholds[name] = float(t_opt)
#     p_test_dict[name] = p_test

# # Regressor
# X_test_reg = create_reg_features(X_test, models_dict=models_dict)
# regressor = LGBMRegressor(**best_params_reg).fit(X_train_reg, y_reg_train, categorical_feature=['pair'])
# reg_pred = regressor.predict(X_test_reg)
# print("\n=== REGRESSOR (rr_label) ===")
# print("MAE:", mean_absolute_error(y_reg_test, reg_pred))
# print("RMSE:", rmse(y_reg_test, reg_pred))
# print("R2:", r2_score(y_reg_test, reg_pred))

# # Meta3
# X_test_meta = create_meta3_features(X_test, models_dict=models_dict, reg_models=reg_models)
# meta_pred = meta_model.predict(X_test_meta)
# print("\n=== META3 (0=<1R, 1=1..2R, 2=>=2R) ===")
# target_names = [f"class_{c}" for c in sorted(np.unique(y_meta_test))]
# print(classification_report(y_meta_test, meta_pred, target_names=target_names, digits=4))
# print(confusion_matrix(y_meta_test, meta_pred))

# # -----------------------------
# # 4b) Holdout WIN-RATE SIMULATION
# # -----------------------------
# def evaluate_selection(name, mask, rr, rr_target=None, plot=False, save_dir="selection_plots"):
#     import os
#     os.makedirs(save_dir, exist_ok=True)

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

#         # save to PNG
#         safe_name = name.replace(" ", "_").replace(":", "").replace("/", "-")
#         save_path = os.path.join(save_dir, f"{safe_name}.png")
#         plt.savefig(save_path, dpi=300)
#         print(f"📁 Saved plot to: {save_path}")

#         plt.close()


# # Scenario A: Use ≥1R binary at its optimal threshold
# t1 = opt_thresholds.get('1_1', 0.5)
# mask_A = (p_test_dict['1_1'] >= t1)
# evaluate_selection("A) Binary 1R filter (prob >= t1)", mask_A, rr_hold, rr_target=1, plot=False)

# # Scenario B: Use ≥2R binary at its optimal threshold
# t2 = opt_thresholds.get('1_2', 0.5)
# mask_B = (p_test_dict['1_2'] >= t2)
# evaluate_selection("B) Binary 2R filter (prob >= t2)", mask_B, rr_hold, rr_target=2, plot=False)

# # Scenario C: Combine both (must pass 1R and 2R thresholds)
# mask_C = (p_test_dict['1_1'] >= t1) & (p_test_dict['1_2'] >= t2)
# evaluate_selection("C) Both binaries (≥1R & ≥2R)", mask_C, rr_hold, rr_target=2, plot=False)

# # Scenario D: Meta3 class==2 (>=2R)
# mask_D = (meta_pred == 2)
# evaluate_selection("D) Meta3 predicts >=2R", mask_D, rr_hold, rr_target=2, plot=False)

# # Scenario E: Top-N by regressor prediction (e.g., top 25% of scores)
# q = 0.75
# thr_reg = np.quantile(reg_pred, q)
# mask_E = (reg_pred >= thr_reg)
# evaluate_selection(f"E) Regressor top 25% (thr={thr_reg:.3f})", mask_E, rr_hold, rr_target=2, plot=False)

# # -----------------------------
# # 5) Final training on ALL data (single models)
# # -----------------------------
# print("\nTraining final models on ALL data...")

# df_full = df.copy()
# X_full = df_full[features]
# y_full_reg  = df_full['rr_label']
# y_full_meta = df_full['y_meta3']
# y_full_bins = {name: df_full[CLASSIFIER_TARGETS[name]] for name in CLASSIFIER_TARGETS.keys()}

# # Final base models
# final_clf_models = {}
# for name, params in best_params_dict.items():
#     print(f"Fitting final base classifier {name} ...")
#     m = LGBMClassifier(**params).fit(X_full, y_full_bins[name], categorical_feature=['pair'])
#     final_clf_models[name] = m

# # Final regressor features/preds
# p_full_bins = {name: final_clf_models[name].predict_proba(X_full)[:, 1] for name in CLASSIFIER_TARGETS.keys()}
# X_full_reg = create_reg_features(X_full, oof_probs=p_full_bins)
# reg_final = LGBMRegressor(**best_params_reg).fit(X_full_reg, y_full_reg, categorical_feature=['pair'])

# # Final meta3 features/preds
# reg_full_pred = reg_final.predict(X_full_reg)
# X_full_meta = create_meta3_features(X_full, oof_probs=p_full_bins, reg_vec=reg_full_pred)
# meta_final = LGBMClassifier(**best_params_meta).fit(X_full_meta, y_full_meta, categorical_feature=['pair'])

# # -----------------------------
# # 6) Save artifacts
# # -----------------------------
# for name, model in final_clf_models.items():
#     model.booster_.save_model(f"{MODEL_DIR}classifier_{name}.txt")

# reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
# meta_final.booster_.save_model(f"{MODEL_DIR}meta3_model.txt")

# pair_categories = df_full['pair'].cat.categories.tolist()
# metadata = {
#     "features": features,
#     "rr_thresholds": RR_THRESHOLDS,          # [1, 2]
#     "classifier_names": list(CLASSIFIER_TARGETS.keys()),  # ['1_1','1_2']
#     "best_params": {
#         **{f"clf_{name}": best_params_dict[name] for name in CLASSIFIER_TARGETS.keys()},
#         "reg": best_params_reg,
#         "meta3": best_params_meta
#     },
#     "opt_thresholds": opt_thresholds,        # store holdout-optimal thresholds
#     "pair_categories": pair_categories,
#     "meta3_classes": ["<1R", "1..2R", ">=2R"]
# }
# joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

# print("\n✅ All models trained and saved successfully!")

# # -----------------------------
# # 7) Feature importances (optional CSVs)
# # -----------------------------
# def dump_importance(model, cols, path, title):
#     if hasattr(model, "feature_importances_"):
#         imp = pd.DataFrame({
#             "Feature": cols,
#             "Importance": model.feature_importances_
#         }).sort_values("Importance", ascending=False)
#         imp.to_csv(path, index=False)
#         print(f"{title} top10:\n", imp.head(10), "\n")
#         return imp
#     else:
#         print(f"⚠️ No feature_importances_ for {title}")
#         return pd.DataFrame()

# # dump
# for name, model in final_clf_models.items():
#     dump_importance(model, features, f"{MODEL_DIR}classifier_feature_importance_{name}.csv", f"Classifier {name}")

# reg_feature_names  = X_full_reg.columns.tolist()
# meta_feature_names = X_full_meta.columns.tolist()
# dump_importance(reg_final,  reg_feature_names,  f"{MODEL_DIR}regression_feature_importance.csv", "Regressor")
# dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}meta3_feature_importance.csv", "Meta3")

# import pandas as pd
# import optuna
# from lightgbm import LGBMClassifier, LGBMRegressor

# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import (
#     mean_absolute_error, mean_squared_error, r2_score,
#     accuracy_score, f1_score, roc_auc_score,
#     confusion_matrix, classification_report
# )
# from sklearn.base import clone

# # -----------------------------
# # Config
# # -----------------------------
# DATA_PATH = "test-combined.csv"   # your combined labeled dataset
# MODEL_DIR = "./tmodel_artifacts_rmax5_v2/"
# os.makedirs(MODEL_DIR, exist_ok=True)

# RANDOM_STATE   = 42
# N_TRIALS_BIN   = 10
# N_TRIALS_REG   = 15
# N_TRIALS_META  = 10
# N_SPLITS_OOF   = 3
# N_SPLITS_CV    = 3

# # thresholds to support
# RR_THRESHOLDS = [1, 2, 3, 4, 5]  # 1R..5R

# # -----------------------------
# # Label builder
# # -----------------------------
# def generate_rr_classification_labels(df, thr_list=RR_THRESHOLDS):
#     """
#     For each threshold T in thr_list, creates y_ge_TR (binary).
#     Also creates exclusive multiclass y_meta in {0..K} where:
#       0: <thr_list[0]R
#       1: [thr_list[0]R, thr_list[1]R)
#       ...
#       K: >= thr_list[-1]R
#     Prints summary counts for quick sanity checks.
#     """
#     # --- prep ---
#     rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
#     thr_list = sorted(set(thr_list))  # ensure sorted & unique
#     n = len(df)

#     # --- tail/summary diagnostics (keep if useful) ---
#     print(df['rr_label'].describe([.5, .75, .90, .95, .99]))
#     for t in [1, 2, 3, 4, 5, 6, 8, 10]:
#         share = (df['rr_label'] >= t).mean()
#         print(f"P(rr ≥ {t}) = {share:.3f}")
#     # by pair share of >= last threshold
#     if 'pair' in df.columns:
#         print("\nShare >= last threshold by pair (top 10):")
#         print(
#             df.groupby('pair')['rr_label']
#               .apply(lambda s: (s >= thr_list[-1]).mean())
#               .sort_values(ascending=False)
#               .head(10)
#         )

#     # --- multi-label binaries y_ge_{T}R ---
#     print("\n=== Binary targets (y_ge_{T}R) ===")
#     for thr in thr_list:
#         col = f'y_ge_{thr}R'
#         df[col] = (rr >= thr).astype('int8')
#         vc = df[col].value_counts().reindex([0, 1], fill_value=0)
#         neg, pos = int(vc[0]), int(vc[1])
#         print(f"{col}: pos={pos} ({pos/n:.2%}) | neg={neg} ({neg/n:.2%})")

#     # --- exclusive multiclass y_meta ---
#     # left-closed / right-open bins: [-inf, t1), [t1, t2), ..., [tK, inf)
#     bins = [-np.inf] + list(thr_list) + [np.inf]
#     df['y_meta'] = pd.cut(rr, bins=bins, labels=False, right=False, include_lowest=True).astype('int8')

#     # human-friendly names
#     names = [f"<{thr_list[0]}R"]
#     for a, b in zip(thr_list[:-1], thr_list[1:]):
#         names.append(f"{a}..{b}R")
#     names.append(f">={thr_list[-1]}R")

#     print("\n=== Multiclass target (y_meta) ===")
#     vc_meta = df['y_meta'].value_counts().sort_index()
#     # ensure we print all classes 0..K even if some are empty
#     for k in range(len(bins) - 1):
#         cnt = int(vc_meta.get(k, 0))
#         print(f"class {k:>1} ({names[k]:>6}): {cnt} ({cnt/n:.2%})")
#     print(f"TOTAL rows: {n}")

#     return df



# # Map classifier names to actual target columns
# CLASSIFIER_TARGETS = {f'1_{i}': f'y_ge_{i}R' for i in RR_THRESHOLDS}

# # -----------------------------
# # Utilities
# # -----------------------------
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def inv_freq_weights_binary(y):
#     counts = pd.Series(y).value_counts()
#     N, K = len(y), 2
#     return {int(c): float(N / (K * counts.get(c, 1))) for c in [0, 1]}

# def inv_freq_weights_multi(y):
#     counts = pd.Series(y).value_counts()
#     N, K = len(y), counts.shape[0]
#     return {int(c): float(N / (K * counts.get(c, 1))) for c in counts.index}

# # -----------------------------
# # Manual CV scorers (pass categorical_feature=['pair'])
# # -----------------------------
# def cv_score_classifier(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     scores = []
#     for tr, va in skf.split(X, y):
#         m = LGBMClassifier(**params)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         prob = m.predict_proba(X.iloc[va])[:, 1]
#         scores.append(roc_auc_score(y.iloc[va], prob))
#     return float(np.mean(scores))

# def cv_score_regressor(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     scores = []
#     for tr, va in kf.split(X, y):
#         m = LGBMRegressor(**params)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         pred = m.predict(X.iloc[va])
#         scores.append(rmse(y.iloc[va], pred))
#     return float(np.mean(scores))  # lower is better

# # -----------------------------
# # Optuna tuning
# # -----------------------------
# def tune_classifier_params(X, y, weight_grid=None, name="clf"):
#     if weight_grid is None:
#         weight_grid = [
#             None,
#             "balanced",
#             inv_freq_weights_binary(y),
#             {0: 1.0, 1: 2.0},
#             {0: 1.0, 1: 3.0}
#         ]
#     def objective(trial):
#         params = {
#             "objective": "binary",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": trial.suggest_categorical("class_weight", weight_grid),
#         }
#         return cv_score_classifier(params, X, y)
#     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_BIN)
#     return study.best_params

# def tune_regressor_params(X, y, name="reg"):
#     def objective(trial):
#         params = {
#             "objective": "regression",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         }
#         return cv_score_regressor(params, X, y)  # lower is better
#     study = optuna.create_study(direction="minimize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_REG)
#     return study.best_params

# def tune_meta_params(X, y, name="meta"):
#     weight_grid = [
#         None,
#         "balanced",
#         inv_freq_weights_multi(y)
#     ]
#     def objective(trial):
#         params = {
#             "objective": "multiclass",
#             "num_class": int(len(np.unique(y))),  # should be 6 for 0..5
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "random_state": RANDOM_STATE,
#             "n_estimators": trial.suggest_int("n_estimators", 150, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 128),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": trial.suggest_categorical("class_weight", weight_grid)
#         }
#         skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
#         f1s = []
#         for tr, va in skf.split(X, y):
#             m = LGBMClassifier(**params)
#             m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#             pred = m.predict(X.iloc[va])
#             f1s.append(f1_score(y.iloc[va], pred, average='weighted'))
#         return float(np.mean(f1s))
#     study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
#     study.optimize(objective, n_trials=N_TRIALS_META)
#     return study.best_params

# # -----------------------------
# # OOF helpers (no leakage)
# # -----------------------------
# def oof_binary(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in skf.split(X, y):
#         m = clone(base_estimator)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         oof[va] = m.predict_proba(X.iloc[va])[:, 1]
#         models.append(m)
#     return oof, models

# def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X), dtype=float)
#     models = []
#     for tr, va in kf.split(X, y):
#         m = clone(base_estimator)
#         m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
#         oof[va] = m.predict(X.iloc[va])
#         models.append(m)
#     return oof, models

# # -----------------------------
# # Feature builders (stacking)
# # -----------------------------
# def create_reg_features(X, oof_probs=None, models_dict=None):
#     Xf = X.copy()
#     if oof_probs is not None:
#         for name in CLASSIFIER_TARGETS.keys():
#             Xf[f'clf_{name}_prob'] = oof_probs[name]
#     elif models_dict is not None:
#         for name, model_list in models_dict.items():
#             Xf[f'clf_{name}_prob'] = np.mean([m.predict_proba(X)[:, 1] for m in model_list], axis=0)
#     return Xf

# def create_meta_features(X, oof_probs=None, reg_vec=None, models_dict=None, reg_models=None):
#     Xf = create_reg_features(X, oof_probs=oof_probs, models_dict=models_dict)
#     if reg_vec is None and reg_models is not None:
#         reg_vec = np.mean([m.predict(create_reg_features(X, models_dict=models_dict)) for m in reg_models], axis=0)
#     if reg_vec is not None:
#         Xf['reg_pred'] = reg_vec
#     return Xf

# # -----------------------------
# # Load & prepare data
# # -----------------------------
# df = pd.read_csv(DATA_PATH)
# df = df[df['label'] != 0].copy()       # keep only win/loss
# df['pair'] = df['pair'].astype('category')
# df = generate_rr_classification_labels(df, thr_list=RR_THRESHOLDS)

# # Build feature list (exclude targets + raw label)
# drop_cols = ['label', 'rr_label', 'y_meta'] + [f'y_ge_{i}R' for i in RR_THRESHOLDS]
# features = [c for c in df.columns if c not in drop_cols]

# # Split (stratify on y_meta 0..K)
# y_bin_cols = [f'y_ge_{i}R' for i in RR_THRESHOLDS]
# X_train, X_test, y_reg_train, y_reg_test, y_bin_train, y_bin_test, y_meta_train, y_meta_test = train_test_split(
#     df[features],
#     df['rr_label'],
#     df[y_bin_cols],
#     df['y_meta'],
#     test_size=0.2,
#     stratify=df['y_meta'],
#     random_state=RANDOM_STATE
# )

# # -----------------------------
# # 1) Tune + OOF all base classifiers (1R..5R)
# # -----------------------------
# oof_probs = {}
# models_dict = {}
# best_params_dict = {}

# for name, col in CLASSIFIER_TARGETS.items():
#     print(f"\n>>> Tuning {name} ({col})")
#     best_params = tune_classifier_params(X_train, y_bin_train[col], name=f"clf_{name}")
#     clf_base = LGBMClassifier(**best_params)
#     oof, models = oof_binary(clf_base, X_train, y_bin_train[col])
#     oof_probs[name] = oof
#     models_dict[name] = models
#     best_params_dict[name] = best_params

# # -----------------------------
# # 2) Tune regressor on train using OOF probs (no leakage)
# # -----------------------------
# X_train_reg = create_reg_features(X_train, oof_probs=oof_probs)
# best_params_reg = tune_regressor_params(X_train_reg, y_reg_train, name="reg")
# reg_base = LGBMRegressor(**best_params_reg)
# oof_reg, reg_models = oof_regressor(reg_base, X_train_reg, y_reg_train)

# # -----------------------------
# # 3) Tune meta on OOF features (0..5)
# # -----------------------------
# X_train_meta = create_meta_features(X_train, oof_probs=oof_probs, reg_vec=oof_reg)
# best_params_meta = tune_meta_params(X_train_meta, y_meta_train, name="meta")
# meta_model = LGBMClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, categorical_feature=['pair'])

# # -----------------------------
# # 4) Holdout Evaluation
# # -----------------------------
# print("\n=== BINARY CLASSIFIERS (Holdout) ===")
# for name, models in models_dict.items():
#     col = CLASSIFIER_TARGETS[name]
#     p_test = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
#     y_pred = (p_test >= 0.5).astype(int)
#     print(f"\n-- {name} ({col}) --")
#     print("ROC AUC:", roc_auc_score(y_bin_test[col], p_test))
#     print("Accuracy:", accuracy_score(y_bin_test[col], y_pred))
#     print("F1:", f1_score(y_bin_test[col], y_pred))
#     print(confusion_matrix(y_bin_test[col], y_pred))
#     print(classification_report(y_bin_test[col], y_pred, target_names=['<thr','>=thr'], digits=4))

# # Regressor
# X_test_reg = create_reg_features(X_test, models_dict=models_dict)
# regressor = LGBMRegressor(**best_params_reg).fit(X_train_reg, y_reg_train, categorical_feature=['pair'])
# reg_pred = regressor.predict(X_test_reg)
# print("\n=== REGRESSOR (rr_label) ===")
# print("MAE:", mean_absolute_error(y_reg_test, reg_pred))
# print("RMSE:", rmse(y_reg_test, reg_pred))
# print("R2:", r2_score(y_reg_test, reg_pred))

# # Meta
# X_test_meta = create_meta_features(X_test, models_dict=models_dict, reg_models=reg_models)
# meta_pred = meta_model.predict(X_test_meta)
# print("\n=== META (0=<1R, 1=1..2R, 2=2..3R, 3=3..4R, 4=4..5R, 5=>=5R) ===")
# target_names = [f"class_{c}" for c in sorted(np.unique(y_meta_test))]
# print(classification_report(y_meta_test, meta_pred, target_names=target_names, digits=4))
# print(confusion_matrix(y_meta_test, meta_pred))

# # -----------------------------
# # 5) Final training on ALL data (single models)
# # -----------------------------
# print("\nTraining final models on ALL data...")

# df_full = df.copy()
# X_full = df_full[features]
# y_full_reg  = df_full['rr_label']
# y_full_meta = df_full['y_meta']
# y_full_bins = {name: df_full[CLASSIFIER_TARGETS[name]] for name in CLASSIFIER_TARGETS.keys()}

# # Final base models
# final_clf_models = {}
# for name, params in best_params_dict.items():
#     print(f"Fitting final base classifier {name} ...")
#     m = LGBMClassifier(**params).fit(X_full, y_full_bins[name], categorical_feature=['pair'])
#     final_clf_models[name] = m

# # Final regressor features/preds
# p_full_bins = {name: final_clf_models[name].predict_proba(X_full)[:, 1] for name in CLASSIFIER_TARGETS.keys()}
# X_full_reg = create_reg_features(X_full, oof_probs=p_full_bins)
# reg_final = LGBMRegressor(**best_params_reg).fit(X_full_reg, y_full_reg, categorical_feature=['pair'])

# # Final meta features/preds
# reg_full_pred = reg_final.predict(X_full_reg)
# X_full_meta = create_meta_features(X_full, oof_probs=p_full_bins, reg_vec=reg_full_pred)
# meta_final = LGBMClassifier(**best_params_meta).fit(X_full_meta, y_full_meta, categorical_feature=['pair'])

# # -----------------------------
# # 6) Save artifacts
# # -----------------------------
# for name, model in final_clf_models.items():
#     model.booster_.save_model(f"{MODEL_DIR}classifier_{name}.txt")

# reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
# meta_final.booster_.save_model(f"{MODEL_DIR}meta_model.txt")

# pair_categories = df_full['pair'].cat.categories.tolist()
# metadata = {
#     "features": features,
#     "rr_thresholds": RR_THRESHOLDS,
#     "classifier_names": list(CLASSIFIER_TARGETS.keys()),  # ['1_1',...'1_5']
#     "best_params": {
#         **{f"clf_{name}": best_params_dict[name] for name in CLASSIFIER_TARGETS.keys()},
#         "reg": best_params_reg,
#         "meta": best_params_meta
#     },
#     "pair_categories": pair_categories,
#     "live_meta_min_class": 3  # take trades with meta >= 3
# }
# joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")

# print("\n✅ All models trained and saved successfully!")

# # -----------------------------
# # 7) Feature importances (optional CSVs)
# # -----------------------------
# def dump_importance(model, cols, path, title):
#     if hasattr(model, "feature_importances_"):
#         imp = pd.DataFrame({
#             "Feature": cols,
#             "Importance": model.feature_importances_
#         }).sort_values("Importance", ascending=False)
#         imp.to_csv(path, index=False)
#         print(f"{title} top10:\n", imp.head(10), "\n")
#         return imp
#     else:
#         print(f"⚠️ No feature_importances_ for {title}")
#         return pd.DataFrame()



# reg_feature_names  = X_full_reg.columns.tolist()
# meta_feature_names = X_full_meta.columns.tolist()

# for name, model in final_clf_models.items():
#     dump_importance(model, features, f"{MODEL_DIR}classifier_feature_importance_{name}.csv", f"Classifier {name}")

# dump_importance(reg_final,  reg_feature_names,  f"{MODEL_DIR}regression_feature_importance.csv", "Regressor")
# dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}meta_feature_importance.csv", "Meta")

# ==============================#
#  STACKED RR TRAINING PIPELINE #
#  (1:1, 1:2, Regressor, Meta)  #
# ==============================#

import os
import joblib
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.base import clone

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "test-combined.csv"
MODEL_DIR = "./tmodel_artifacts_combinedx/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS_BIN = 50
N_TRIALS_REG = 50
N_TRIALS_META = 50
N_SPLITS_OOF = 3
N_SPLITS_CV = 3

# -----------------------------
# Labels
# -----------------------------
def generate_rr_classification_labels(df, thr1=1.0, thr2=2.0):
    """
    Creates:
      - y_ge_1R, y_ge_2R (binary, overlapping: multi-label style)
      - y_meta (exclusive multiclass: 0=<1R, 1=1–<2R, 2=≥2R)
    """
    rr = pd.to_numeric(df['rr_label'], errors='coerce').fillna(-1.0)
    df['y_ge_1R'] = (rr >= thr1).astype('int8')
    df['y_ge_2R'] = (rr >= thr2).astype('int8')
    df['y_meta']  = np.where(rr >= thr2, 2,
                      np.where(rr >= thr1, 1, 0)).astype('int8')
    return df

TARGET_COLUMNS = {
    'binary_rr1': 'y_ge_1R',
    'binary_rr2': 'y_ge_2R',
    'regression': 'rr_label',
    'meta': 'y_meta'
}
CLASSIFIER_NAMES = {'1_1': 'binary_rr1', '1_2': 'binary_rr2'}

# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def inv_freq_weights_binary(y):
    counts = pd.Series(y).value_counts()
    N, K = len(y), 2
    return {int(c): float(N / (K * counts.get(c, 1))) for c in [0, 1]}

def inv_freq_weights_multi(y):
    counts = pd.Series(y).value_counts()
    N, K = len(y), counts.shape[0]
    return {int(c): float(N / (K * counts.get(c, 1))) for c in counts.index}

# -----------------------------
# Manual CV scorers (so we can pass categorical_feature=['pair'])
# -----------------------------
def cv_score_classifier(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        m = LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
        prob = m.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], prob))
    return float(np.mean(scores))

def cv_score_regressor(params, X, y, n_splits=N_SPLITS_CV, seed=RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X, y):
        m = LGBMRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
        pred = m.predict(X.iloc[va])
        scores.append(rmse(y.iloc[va], pred))
    return float(np.mean(scores))  # lower is better

# -----------------------------
# Optuna tuning
# -----------------------------
def tune_classifier_params(X, y, weight_grid=None, name="clf"):
    if weight_grid is None:
        weight_grid = [
            None,
            "balanced",
            inv_freq_weights_binary(y),
            {0: 1.0, 1: 2.0},
            {0: 1.0, 1: 3.0}
        ]
    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", weight_grid),
        }
        return cv_score_classifier(params, X, y)
    study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
    study.optimize(objective, n_trials=N_TRIALS_BIN)
    return study.best_params

def tune_regressor_params(X, y, name="reg"):
    def objective(trial):
        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        return cv_score_regressor(params, X, y)  # lower is better
    study = optuna.create_study(direction="minimize", study_name=f"tune_{name}")
    study.optimize(objective, n_trials=N_TRIALS_REG)
    return study.best_params

def tune_meta_params(X, y, name="meta"):
    weight_grid = [
        None,
        "balanced",
        inv_freq_weights_multi(y),
        {0: 1.0, 1: 1.5, 2: 1.8} if set(pd.Series(y).unique()) == {0,1,2} else None
    ]
    weight_grid = [w for w in weight_grid if w is not None]

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": int(len(np.unique(y))),
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", weight_grid)
        }
        # weighted F1 across classes
        skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
        f1s = []
        for tr, va in skf.split(X, y):
            m = LGBMClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
            pred = m.predict(X.iloc[va])
            f1s.append(f1_score(y.iloc[va], pred, average='weighted'))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", study_name=f"tune_{name}")
    study.optimize(objective, n_trials=N_TRIALS_META)
    return study.best_params

# -----------------------------
# OOF helpers (no leakage)
# -----------------------------
def oof_binary(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    models = []
    for tr, va in skf.split(X, y):
        m = clone(base_estimator)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    return oof, models

def oof_regressor(base_estimator, X, y, n_splits=N_SPLITS_OOF, seed=RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    models = []
    for tr, va in kf.split(X, y):
        m = clone(base_estimator)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=['pair'])
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
    return oof, models

# -----------------------------
# Feature builders (stacking)
# -----------------------------
def create_reg_features(X, p11=None, p12=None, models_11=None, models_12=None):
    Xf = X.copy()
    if p11 is None:
        p11 = np.mean([m.predict_proba(X)[:, 1] for m in models_11], axis=0)
    if p12 is None:
        p12 = np.mean([m.predict_proba(X)[:, 1] for m in models_12], axis=0)
    Xf['clf_1_1_prob'] = p11
    Xf['clf_1_2_prob'] = p12
    return Xf

def create_meta_features(X, p11=None, p12=None, reg_vec=None,
                         models_11=None, models_12=None, reg_models=None):
    Xf = X.copy()
    if p11 is None:
        p11 = np.mean([m.predict_proba(X)[:, 1] for m in models_11], axis=0)
    if p12 is None:
        p12 = np.mean([m.predict_proba(X)[:, 1] for m in models_12], axis=0)
    Xf['clf_1_1_prob'] = p11
    Xf['clf_1_2_prob'] = p12

    if reg_vec is None:
        reg_vec = np.mean([m.predict(create_reg_features(X, p11, p12, models_11, models_12))
                           for m in reg_models], axis=0)
    Xf['reg_pred'] = reg_vec
    return Xf

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df[df['label'] != 0].copy()              # keep only win/loss
df['pair'] = df['pair'].astype('category')
df = generate_rr_classification_labels(df, thr1=1.0, thr2=2.0)

# Feature list
drop_cols = ['label',
             TARGET_COLUMNS['regression'],
             TARGET_COLUMNS['binary_rr1'],
             TARGET_COLUMNS['binary_rr2'],
             TARGET_COLUMNS['meta']]
features = [c for c in df.columns if c not in drop_cols]

# Split
X_train, X_test, y_reg_train, y_reg_test, y_bin_train, y_bin_test, y_meta_train, y_meta_test = train_test_split(
    df[features],
    df[TARGET_COLUMNS['regression']],
    df[[TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2']]],
    df[TARGET_COLUMNS['meta']],
    test_size=0.2,
    stratify=df[TARGET_COLUMNS['meta']],
    random_state=RANDOM_STATE
)

# -----------------------------
# 1) Tune base classifiers on train
# -----------------------------
best_params_11 = tune_classifier_params(X_train, y_bin_train[TARGET_COLUMNS['binary_rr1']], name="clf_1_1")
best_params_12 = tune_classifier_params(X_train, y_bin_train[TARGET_COLUMNS['binary_rr2']], name="clf_1_2")

clf11_base = LGBMClassifier(**best_params_11)
clf12_base = LGBMClassifier(**best_params_12)

# -----------------------------
# 2) OOF for base (train) and fold models for test
# -----------------------------
oof_11, models_11 = oof_binary(clf11_base, X_train, y_bin_train[TARGET_COLUMNS['binary_rr1']])
oof_12, models_12 = oof_binary(clf12_base, X_train, y_bin_train[TARGET_COLUMNS['binary_rr2']])

# -----------------------------
# 3) Tune regressor on train using OOF probs (no leakage)
# -----------------------------
X_train_reg = create_reg_features(X_train, p11=oof_11, p12=oof_12)
best_params_reg = tune_regressor_params(X_train_reg, y_reg_train, name="reg")
reg_base = LGBMRegressor(**best_params_reg)

# OOF reg for meta training
oof_reg, reg_models = oof_regressor(reg_base, X_train_reg, y_reg_train)

# -----------------------------
# 4) Tune meta on OOF features
# -----------------------------
X_train_meta = create_meta_features(X_train, p11=oof_11, p12=oof_12, reg_vec=oof_reg)
best_params_meta = tune_meta_params(X_train_meta, y_meta_train, name="meta")
meta_model = LGBMClassifier(**best_params_meta).fit(X_train_meta, y_meta_train, categorical_feature=['pair'])

# -----------------------------
# 5) Holdout Evaluation
# -----------------------------
# Base binaries (use fold ensemble on test)
p11_test = np.mean([m.predict_proba(X_test)[:, 1] for m in models_11], axis=0)
p12_test = np.mean([m.predict_proba(X_test)[:, 1] for m in models_12], axis=0)
y11_pred = (p11_test >= 0.5).astype(int)
y12_pred = (p12_test >= 0.5).astype(int)

print("\n=== BINARY 1:1 ===")
print("ROC AUC:", roc_auc_score(y_bin_test[TARGET_COLUMNS['binary_rr1']], p11_test))
print("Accuracy:", accuracy_score(y_bin_test[TARGET_COLUMNS['binary_rr1']], y11_pred))
print("F1:", f1_score(y_bin_test[TARGET_COLUMNS['binary_rr1']], y11_pred))
print(confusion_matrix(y_bin_test[TARGET_COLUMNS['binary_rr1']], y11_pred))
print(classification_report(y_bin_test[TARGET_COLUMNS['binary_rr1']], y11_pred, target_names=['<1R', '>=1R']))

print("\n=== BINARY 1:2 ===")
print("ROC AUC:", roc_auc_score(y_bin_test[TARGET_COLUMNS['binary_rr2']], p12_test))
print("Accuracy:", accuracy_score(y_bin_test[TARGET_COLUMNS['binary_rr2']], y12_pred))
print("F1:", f1_score(y_bin_test[TARGET_COLUMNS['binary_rr2']], y12_pred))
print(confusion_matrix(y_bin_test[TARGET_COLUMNS['binary_rr2']], y12_pred))
print(classification_report(y_bin_test[TARGET_COLUMNS['binary_rr2']], y12_pred, target_names=['<2R', '>=2R']))

# Regressor (use fold ensemble on test features)
X_test_reg = create_reg_features(X_test, models_11=models_11, models_12=models_12)
regressor = LGBMRegressor(**best_params_reg).fit(X_train_reg, y_reg_train, categorical_feature=['pair'])
reg_pred = regressor.predict(X_test_reg)

print("\n=== REGRESSOR (rr_label) ===")
print("MAE:", mean_absolute_error(y_reg_test, reg_pred))
print("RMSE:", rmse(y_reg_test, reg_pred))
print("R2:", r2_score(y_reg_test, reg_pred))

# Meta (use fold ensembles for base + reg)
X_test_meta = create_meta_features(X_test, models_11=models_11, models_12=models_12, reg_models=reg_models)
meta_pred = meta_model.predict(X_test_meta)

print("\n=== META (0=Reject,1=1:1,2=1:2) ===")
target_names = ['Reject', '1:1', '1:2'][:len(np.unique(y_meta_test))]
print(classification_report(y_meta_test, meta_pred, target_names=target_names, digits=4))
print(confusion_matrix(y_meta_test, meta_pred))

# -----------------------------
# 6) Final training on ALL data (single models)
# -----------------------------
print("\nTraining final models on ALL data...")

# Rebuild labels (ensure consistency)
df_full = df.copy()
X_full = df_full[features]
y_full_bin11 = df_full[TARGET_COLUMNS['binary_rr1']]
y_full_bin12 = df_full[TARGET_COLUMNS['binary_rr2']]
y_full_reg   = df_full[TARGET_COLUMNS['regression']]
y_full_meta  = df_full[TARGET_COLUMNS['meta']]

# Final base models
clf11_final = LGBMClassifier(**best_params_11).fit(X_full, y_full_bin11, categorical_feature=['pair'])
clf12_final = LGBMClassifier(**best_params_12).fit(X_full, y_full_bin12, categorical_feature=['pair'])

# Final regressor features/preds
p11_full = clf11_final.predict_proba(X_full)[:, 1]
p12_full = clf12_final.predict_proba(X_full)[:, 1]
X_full_reg = create_reg_features(X_full, p11=p11_full, p12=p12_full)

reg_final = LGBMRegressor(**best_params_reg).fit(X_full_reg, y_full_reg, categorical_feature=['pair'])

# Final meta features/preds
reg_full_pred = reg_final.predict(X_full_reg)
X_full_meta = create_meta_features(X_full, p11=p11_full, p12=p12_full, reg_vec=reg_full_pred)
meta_final = LGBMClassifier(**best_params_meta).fit(X_full_meta, y_full_meta, categorical_feature=['pair'])

# -----------------------------
# 7) Save artifacts
# -----------------------------
clf11_final.booster_.save_model(f"{MODEL_DIR}classifier_1_1.txt")
clf12_final.booster_.save_model(f"{MODEL_DIR}classifier_1_2.txt")
reg_final.booster_.save_model(f"{MODEL_DIR}regressor.txt")
meta_final.booster_.save_model(f"{MODEL_DIR}meta_model.txt")

pair_categories = df_full['pair'].cat.categories.tolist()
metadata = {
    "features": features,
    "target_columns": TARGET_COLUMNS,
    "best_params": {
        "clf_1_1": best_params_11,
        "clf_1_2": best_params_12,
        "reg": best_params_reg,
        "meta": best_params_meta
    },
    # NEW: store category ordering used in training
    "pair_categories": pair_categories,
}
joblib.dump(metadata, f"{MODEL_DIR}model_metadata.pkl")


print("\n✅ All models trained and saved successfully!")

# -----------------------------
# 8) Feature importances (optional CSVs)
# -----------------------------
def dump_importance(model, cols, path, title):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}) \
                .sort_values("Importance", ascending=False)
        imp.to_csv(path, index=False)
        print(f"{title} top10:\n", imp.head(10), "\n")
        return imp
    else:
        print(f"⚠️ No feature_importances_ for {title}")
        return pd.DataFrame()

# Names for the final (full-data) stacks
reg_feature_names  = X_full_reg.columns.tolist()
meta_feature_names = X_full_meta.columns.tolist()

dump_importance(clf11_final, features, f"{MODEL_DIR}classifier_feature_importance_1_1.csv", "Classifier 1:1")
dump_importance(clf12_final, features, f"{MODEL_DIR}classifier_feature_importance_1_2.csv", "Classifier 1:2")
dump_importance(reg_final,  reg_feature_names,  f"{MODEL_DIR}regression_feature_importance.csv", "Regressor")
dump_importance(meta_final, meta_feature_names, f"{MODEL_DIR}meta_feature_importance.csv", "Meta")


# import pandas as pd
# import numpy as np
# import optuna
# import os
# import joblib
# from lightgbm import LGBMClassifier, LGBMRegressor
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
#                             accuracy_score, f1_score, roc_auc_score, confusion_matrix,
#                             classification_report, make_scorer)
# from sklearn.preprocessing import LabelEncoder

# # --- Constants ---
# TARGET_COLUMNS = {
#     'binary_rr1': 'label_rr_1.0',
#     'binary_rr2': 'label_rr_2.0', 
#     'regression': 'rr_label',
#     'meta': 'meta_target'
# }

# CLASSIFIER_NAMES = {
#     '1_1': 'binary_rr1',
#     '1_2': 'binary_rr2',
# }

# # --- 1. Load and Prepare Data ---
# df = pd.read_csv("test-combined.csv")

# # Filter and prepare data
# df = df[df['label'] != 0].copy()  # Remove neutral trades
# df['pair'] = df['pair'].astype('category')

# # Define features and targets
# features = [col for col in df.columns if col not in [
#     'label', TARGET_COLUMNS['regression'], 
#     TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2'], 
# ]]

# # Create multi-class meta target
# conditions = [
#     df[TARGET_COLUMNS['regression']] >= 2.0,
#     df[TARGET_COLUMNS['regression']] >= 1.0
# ]
# choices = [2, 1]
# df[TARGET_COLUMNS['meta']] = np.select(conditions, choices, default=0)

# # Train-test split
# X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_meta_train, y_meta_test = train_test_split(
#     df[features], 
#     df[TARGET_COLUMNS['regression']], 
#     df[[TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2']]], 
#     df[TARGET_COLUMNS['meta']],
#     test_size=0.2, 
#     stratify=df[TARGET_COLUMNS['meta']], 
#     random_state=42
# )

# # --- 2. Define Evaluation Metrics ---
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))
# rmse_scorer = make_scorer(rmse, greater_is_better=False)

# # --- 3. Tune Base Classifiers ---
# def tune_classifier(X, y, name):
#     def objective(trial):
#         class_weight_option = trial.suggest_categorical("class_weight", [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}])
#         params = {
#             "objective": "binary",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#             "max_depth": trial.suggest_int("max_depth", 3, 10),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": class_weight_option,
#         }
#         model = LGBMClassifier(**params)
#         return cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=25)

#     best_params = study.best_params
#     best_params.update({
#         "objective": "binary",
#         "verbosity": -1,
#         "random_state": 42
#     })
    
#     model = LGBMClassifier(**best_params)
#     model.fit(X, y, categorical_feature=['pair'])
#     return model

# # Train classifiers
# classifiers = {
#     '1_1': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr1']], "1:1"),
#     '1_2': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr2']], "1:2")
# }

# # --- 4. Tune Regression Model ---
# def create_reg_features(X):
#     return pd.DataFrame({
#         **{col: X[col] for col in features},
#         'clf_1_1_prob': classifiers['1_1'].predict_proba(X)[:, 1],
#         'clf_1_2_prob': classifiers['1_2'].predict_proba(X)[:, 1]
#     })

# def tune_regressor(X, y):
#     def objective(trial):
#         params = {
#             "objective": "regression",
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#             "max_depth": trial.suggest_int("max_depth", 3, 10),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         }
#         model = LGBMRegressor(**params)
#         return np.abs(cross_val_score(model, X, y, cv=3, scoring=rmse_scorer).mean())
    
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=25)
    
#     best_params = study.best_params
#     best_params.update({
#         "objective": "regression",
#         "verbosity": -1,
#         "random_state": 42
#     })
    
#     model = LGBMRegressor(**best_params)
#     model.fit(X, y)
#     return model

# X_train_reg = create_reg_features(X_train)
# regressor = tune_regressor(X_train_reg, y_reg_train)

# # --- 5. Tune Meta Classifier ---
# def create_meta_features(X):
#     reg_input = create_reg_features(X)
#     return pd.DataFrame({
#         **{col: X[col] for col in features},
#         'clf_1_1_prob': classifiers['1_1'].predict_proba(X)[:, 1],
#         'clf_1_2_prob': classifiers['1_2'].predict_proba(X)[:, 1],
#         'reg_pred': regressor.predict(reg_input)
#     })

# def tune_meta(X, y):
#     unique_classes = np.unique(y)
#     num_classes = len(unique_classes)
    
#     def objective(trial):
#         weight_options = [
#             None,
#             "balanced",
#             {c: 1 for c in unique_classes},
#             {c: (i+1) for i, c in enumerate(unique_classes)},
#             {0: 1, 1: 2, 2:1.5 } if set(unique_classes) == {0, 1, 2} else None
#         ]
#         weight_options = [wo for wo in weight_options if wo is not None]
        
#         class_weight = trial.suggest_categorical("class_weight", weight_options)
        
#         params = {
#             "objective": "multiclass",
#             "num_class": num_classes,
#             "boosting_type": "gbdt",
#             "verbosity": -1,
#             "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#             "max_depth": trial.suggest_int("max_depth", 3, 10),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#             "class_weight": class_weight
#         }
        
#         model = LGBMClassifier(**params)
#         return cross_val_score(model, X, y, cv=3, scoring='f1_weighted').mean()

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=25)

#     best_params = study.best_params
#     best_params.update({
#         "objective": "multiclass",
#         "num_class": num_classes,
#         "verbosity": -1,
#         "random_state": 42
#     })

#     model = LGBMClassifier(**best_params)
#     model.fit(X, y)
#     return model

# X_train_meta = create_meta_features(X_train)
# meta_model = tune_meta(X_train_meta, y_meta_train)

# # --- 6. Enhanced Evaluation Function ---
# def evaluate_models():
#     print("\n=== COMPREHENSIVE MODEL EVALUATION ===\n")
    
#     # 1. Evaluate Binary Classifiers
#     for model_name, target_name in CLASSIFIER_NAMES.items():
#         target_col = TARGET_COLUMNS[target_name]
#         if target_col not in y_clf_test.columns:
#             print(f"⚠️ Target column {target_col} not found in test data")
#             continue
            
#         y_true = y_clf_test[target_col]
#         y_pred = classifiers[model_name].predict(X_test)
#         y_prob = classifiers[model_name].predict_proba(X_test)[:, 1]
        
#         print(f"\n🔹 Binary Classifier {model_name.replace('_',':')} RR Evaluation:")
#         print("----------------------------------------")
#         print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")
#         print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
#         print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
#         print("\nConfusion Matrix:")
#         print(confusion_matrix(y_true, y_pred))
#         print("\nClassification Report:")
#         print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
#     # 2. Evaluate Regression Model
#     X_test_reg = create_reg_features(X_test)
#     reg_pred = regressor.predict(X_test_reg)
    
#     print("\n🔹 Regression Model Evaluation:")
#     print("----------------------------------------")
#     print(f"MAE: {mean_absolute_error(y_reg_test, reg_pred):.4f}")
#     print(f"RMSE: {rmse(y_reg_test, reg_pred):.4f}")
#     print(f"R²: {r2_score(y_reg_test, reg_pred):.4f}")
    
#     # 3. Evaluate Meta Model
#     X_test_meta = create_meta_features(X_test)
#     meta_pred = meta_model.predict(X_test_meta)
    
#     # Get actual class labels present in the test data
#     unique_classes = np.unique(y_meta_test)
#     target_names = ['Reject', '1:1', '1:2'][:len(unique_classes)]
    
#     print("\n🔹 Meta Model Evaluation:")
#     print("----------------------------------------")
#     print(classification_report(y_meta_test, meta_pred, 
#                               target_names=target_names,
#                               digits=4))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_meta_test, meta_pred))

# # Run evaluation
# evaluate_models()

# # --- 7. Final Training & Saving ---
# print("\nTraining final models on all data...")

# # 1. Retrain classifiers on full data
# for model_name, target_name in CLASSIFIER_NAMES.items():
#     target_col = TARGET_COLUMNS[target_name]
#     classifiers[model_name].fit(
#         df[features],
#         df[target_col],
#         categorical_feature=['pair']
#     )

# # 2. Create feature sets for final training
# X_full_reg = create_reg_features(df[features])
# X_full_meta = create_meta_features(df[features])

# # 3. Retrain regression and meta models
# regressor.fit(X_full_reg, df[TARGET_COLUMNS['regression']])
# meta_model.fit(X_full_meta, df[TARGET_COLUMNS['meta']])

# # 4. Save models
# model_dir = "./tmodel_artifacts_buy/"
# os.makedirs(model_dir, exist_ok=True)

# for name, model in classifiers.items():
#     model.booster_.save_model(f"{model_dir}classifier_{name}.txt")

# regressor.booster_.save_model(f"{model_dir}regressor.txt")
# meta_model.booster_.save_model(f"{model_dir}meta_model.txt")

# # 5. Save metadata
# metadata = {
#     'features': features,
#     'target_columns': TARGET_COLUMNS,
#     'classifier_names': CLASSIFIER_NAMES
# }
# joblib.dump(metadata, f"{model_dir}model_metadata.pkl")

# print("\nAll models trained and saved successfully!")
# print("\n📊 Feature Importances:\n")

# def get_feature_importance(model, feature_names, title):
#     if hasattr(model, 'feature_importances_'):
#         importance = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': model.feature_importances_
#         }).sort_values('Importance', ascending=False)
#         print(f"\n{title}:")
#         print(importance.head(10))
#         return importance
#     else:
#         print(f"⚠️ Model does not support feature_importances_: {title}")
#         return pd.DataFrame()
# # Correct feature names
# reg_feature_names = X_full_reg.columns.tolist()
# meta_feature_names = X_full_meta.columns.tolist()

# # 1. Regression model importance
# reg_importance = get_feature_importance(regressor, reg_feature_names, "Regression Model Feature Importance")
# reg_importance.to_csv(f"{model_dir}regression_feature_importance.csv", index=False)

# # 2. Classifier models importance
# for clf_name, clf_model in classifiers.items():
#     title = f"Classifier ({clf_name}) Feature Importance"
#     clf_importance = get_feature_importance(clf_model, features, title)
#     clf_importance.to_csv(f"{model_dir}classifier_feature_importance_{clf_name}.csv", index=False)

# # 3. Meta model importance
# meta_importance = get_feature_importance(meta_model, meta_feature_names, "Meta Model Feature Importance")
# meta_importance.to_csv(f"{model_dir}meta_feature_importance.csv", index=False)
