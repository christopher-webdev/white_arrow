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
MODEL_DIR = "./tmodel_artifacts_combined/"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS_BIN = 5
N_TRIALS_REG = 5
N_TRIALS_META = 5
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

metadata = {
    "features": features,
    "target_columns": TARGET_COLUMNS,
    "best_params": {
        "clf_1_1": best_params_11,
        "clf_1_2": best_params_12,
        "reg": best_params_reg,
        "meta": best_params_meta
    }
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
# df = pd.read_csv("testbuy.csv")

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
