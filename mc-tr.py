

import pandas as pd
import numpy as np
import optuna
import os
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                            accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            classification_report, make_scorer)
from sklearn.preprocessing import LabelEncoder

# --- Constants ---
TARGET_COLUMNS = {
    'binary_rr1': 'label_rr_1.0',
    'binary_rr2': 'label_rr_2.0', 
    'regression': 'rr_label',
    'meta': 'meta_target'
}

CLASSIFIER_NAMES = {
    '1_1': 'binary_rr1',
    '1_2': 'binary_rr2',
}

# --- 1. Load and Prepare Data ---
df = pd.read_csv("testsell.csv")

# Filter and prepare data
df = df[df['label'] != 0].copy()  # Remove neutral trades
df['pair'] = df['pair'].astype('category')

# Define features and targets
features = [col for col in df.columns if col not in [
    'label', TARGET_COLUMNS['regression'], 
    TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2'], 
]]

# Create multi-class meta target
conditions = [
    df[TARGET_COLUMNS['regression']] >= 2.0,
    df[TARGET_COLUMNS['regression']] >= 1.1
]
choices = [2, 1]
df[TARGET_COLUMNS['meta']] = np.select(conditions, choices, default=0)

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_meta_train, y_meta_test = train_test_split(
    df[features], 
    df[TARGET_COLUMNS['regression']], 
    df[[TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2']]], 
    df[TARGET_COLUMNS['meta']],
    test_size=0.2, 
    stratify=df[TARGET_COLUMNS['meta']], 
    random_state=42
)

# --- 2. Define Evaluation Metrics ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# --- 3. Tune Base Classifiers ---
def tune_classifier(X, y, name):
    def objective(trial):
        class_weight_option = trial.suggest_categorical("class_weight", [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}])
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "class_weight": class_weight_option,
        }
        model = LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "verbosity": -1,
        "random_state": 42
    })
    
    model = LGBMClassifier(**best_params)
    model.fit(X, y, categorical_feature=['pair'])
    return model

# Train classifiers
classifiers = {
    '1_1': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr1']], "1:1"),
    '1_2': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr2']], "1:2")
}

# --- 4. Tune Regression Model ---
def create_reg_features(X):
    return pd.DataFrame({
        **{col: X[col] for col in features},
        'clf_1_1_prob': classifiers['1_1'].predict_proba(X)[:, 1],
        'clf_1_2_prob': classifiers['1_2'].predict_proba(X)[:, 1]
    })

def tune_regressor(X, y):
    def objective(trial):
        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        model = LGBMRegressor(**params)
        return np.abs(cross_val_score(model, X, y, cv=3, scoring=rmse_scorer).mean())
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    best_params.update({
        "objective": "regression",
        "verbosity": -1,
        "random_state": 42
    })
    
    model = LGBMRegressor(**best_params)
    model.fit(X, y)
    return model

X_train_reg = create_reg_features(X_train)
regressor = tune_regressor(X_train_reg, y_reg_train)

# --- 5. Tune Meta Classifier ---
def create_meta_features(X):
    reg_input = create_reg_features(X)
    return pd.DataFrame({
        **{col: X[col] for col in features},
        'clf_1_1_prob': classifiers['1_1'].predict_proba(X)[:, 1],
        'clf_1_2_prob': classifiers['1_2'].predict_proba(X)[:, 1],
        'reg_pred': regressor.predict(reg_input)
    })

def tune_meta(X, y):
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    def objective(trial):
        weight_options = [
            None,
            "balanced",
            {c: 1 for c in unique_classes},
            {c: (i+1) for i, c in enumerate(unique_classes)},
            {0: 1, 1: 1.5, 2: 2} if set(unique_classes) == {0, 1, 2} else None
        ]
        weight_options = [wo for wo in weight_options if wo is not None]
        
        class_weight = trial.suggest_categorical("class_weight", weight_options)
        
        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "class_weight": class_weight
        }
        
        model = LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring='f1_weighted').mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_params.update({
        "objective": "multiclass",
        "num_class": num_classes,
        "verbosity": -1,
        "random_state": 42
    })

    model = LGBMClassifier(**best_params)
    model.fit(X, y)
    return model

X_train_meta = create_meta_features(X_train)
meta_model = tune_meta(X_train_meta, y_meta_train)

# --- 6. Enhanced Evaluation Function ---
def evaluate_models():
    print("\n=== COMPREHENSIVE MODEL EVALUATION ===\n")
    
    # 1. Evaluate Binary Classifiers
    for model_name, target_name in CLASSIFIER_NAMES.items():
        target_col = TARGET_COLUMNS[target_name]
        if target_col not in y_clf_test.columns:
            print(f"⚠️ Target column {target_col} not found in test data")
            continue
            
        y_true = y_clf_test[target_col]
        y_pred = classifiers[model_name].predict(X_test)
        y_prob = classifiers[model_name].predict_proba(X_test)[:, 1]
        
        print(f"\n🔹 Binary Classifier {model_name.replace('_',':')} RR Evaluation:")
        print("----------------------------------------")
        print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    # 2. Evaluate Regression Model
    X_test_reg = create_reg_features(X_test)
    reg_pred = regressor.predict(X_test_reg)
    
    print("\n🔹 Regression Model Evaluation:")
    print("----------------------------------------")
    print(f"MAE: {mean_absolute_error(y_reg_test, reg_pred):.4f}")
    print(f"RMSE: {rmse(y_reg_test, reg_pred):.4f}")
    print(f"R²: {r2_score(y_reg_test, reg_pred):.4f}")
    
    # 3. Evaluate Meta Model
    X_test_meta = create_meta_features(X_test)
    meta_pred = meta_model.predict(X_test_meta)
    
    # Get actual class labels present in the test data
    unique_classes = np.unique(y_meta_test)
    target_names = ['Reject', '1:1', '1:2'][:len(unique_classes)]
    
    print("\n🔹 Meta Model Evaluation:")
    print("----------------------------------------")
    print(classification_report(y_meta_test, meta_pred, 
                              target_names=target_names,
                              digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_meta_test, meta_pred))

# Run evaluation
evaluate_models()

# --- 7. Final Training & Saving ---
print("\nTraining final models on all data...")

# 1. Retrain classifiers on full data
for model_name, target_name in CLASSIFIER_NAMES.items():
    target_col = TARGET_COLUMNS[target_name]
    classifiers[model_name].fit(
        df[features],
        df[target_col],
        categorical_feature=['pair']
    )

# 2. Create feature sets for final training
X_full_reg = create_reg_features(df[features])
X_full_meta = create_meta_features(df[features])

# 3. Retrain regression and meta models
regressor.fit(X_full_reg, df[TARGET_COLUMNS['regression']])
meta_model.fit(X_full_meta, df[TARGET_COLUMNS['meta']])

# 4. Save models
model_dir = "./tmodel_artifacts_sell/"
os.makedirs(model_dir, exist_ok=True)

for name, model in classifiers.items():
    model.booster_.save_model(f"{model_dir}classifier_{name}.txt")

regressor.booster_.save_model(f"{model_dir}regressor.txt")
meta_model.booster_.save_model(f"{model_dir}meta_model.txt")

# 5. Save metadata
metadata = {
    'features': features,
    'target_columns': TARGET_COLUMNS,
    'classifier_names': CLASSIFIER_NAMES
}
joblib.dump(metadata, f"{model_dir}model_metadata.pkl")

print("\nAll models trained and saved successfully!")
print("\n📊 Feature Importances:\n")

def get_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n{title}:")
        print(importance.head(10))
        return importance
    else:
        print(f"⚠️ Model does not support feature_importances_: {title}")
        return pd.DataFrame()
# Correct feature names
reg_feature_names = X_full_reg.columns.tolist()
meta_feature_names = X_full_meta.columns.tolist()

# 1. Regression model importance
reg_importance = get_feature_importance(regressor, reg_feature_names, "Regression Model Feature Importance")
reg_importance.to_csv(f"{model_dir}regression_feature_importance.csv", index=False)

# 2. Classifier models importance
for clf_name, clf_model in classifiers.items():
    title = f"Classifier ({clf_name}) Feature Importance"
    clf_importance = get_feature_importance(clf_model, features, title)
    clf_importance.to_csv(f"{model_dir}classifier_feature_importance_{clf_name}.csv", index=False)

# 3. Meta model importance
meta_importance = get_feature_importance(meta_model, meta_feature_names, "Meta Model Feature Importance")
meta_importance.to_csv(f"{model_dir}meta_feature_importance.csv", index=False)
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
#                              accuracy_score, f1_score, roc_auc_score, confusion_matrix,
#                              make_scorer)
# from sklearn.ensemble import StackingClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
# import optuna
# import joblib
# from sklearn.base import BaseEstimator, TransformerMixin

# # Load and combine data
# df1 = pd.read_csv("binary_1:1_CAD.csv")
# # df = pd.concat([df1, df2], ignore_index=True)

# # Preprocessing
# df = df.dropna(subset=['label', 'rr_label'])
# df = df[df['label'] != 0].copy()
# df['binary_label'] = df['label'].map({-1: 0, 1: 1})
# df['pair'] = df['pair'].astype('category')

# features = [c for c in df.columns if c not in ['label', 'binary_label', 'rr_label']]
# X = df[features]
# y_reg = df['rr_label']  # Regression target
# y_clf = df['binary_label']  # Classification target

# # Train-test split (stratified by classification label)
# X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
#     X, y_reg, y_clf, stratify=y_clf, test_size=0.2, random_state=42)

# # Define scorers
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))
# rmse_scorer = make_scorer(rmse, greater_is_better=False)

# # 1. Regression Model Tuning
# def objective_lgb_reg(trial):
#     params = {
#         "objective": "regression",
#         "boosting_type": "gbdt",
#         "verbosity": -1,
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#     }
#     model = LGBMRegressor(**params)
#     score = np.abs(cross_val_score(model, X_train, y_reg_train, cv=3, scoring=rmse_scorer).mean())
#     return score

# reg_study = optuna.create_study(direction="minimize")
# reg_study.optimize(objective_lgb_reg, n_trials=50)

# # 2. Classifier Model Tuning
# def objective_lgb_clf(trial):
#     params = {
#         "objective": "binary",
#         "boosting_type": "gbdt",
#         "verbosity": -1,
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         "is_unbalance": True,
#     }
#     model = LGBMClassifier(**params)
#     score = cross_val_score(model, X_train, y_clf_train, cv=3, scoring='roc_auc').mean()
#     return score

# clf_study = optuna.create_study(direction="maximize")
# clf_study.optimize(objective_lgb_clf, n_trials=50)

# # Train base models with best params
# best_reg_params = reg_study.best_params
# best_reg_params.update({
#     "objective": "regression",
#     "verbosity": -1,
#     "random_state": 42
# })
# reg_model = LGBMRegressor(**best_reg_params)
# reg_model.fit(X_train, y_reg_train, categorical_feature=['pair'])

# best_clf_params = clf_study.best_params
# best_clf_params.update({
#     "objective": "binary",
#     "verbosity": -1,
#     "random_state": 42
# })
# clf_model = LGBMClassifier(**best_clf_params)
# clf_model.fit(X_train, y_clf_train, categorical_feature=['pair'])

# # Create meta-features
# def create_meta_features(model_reg, model_clf, X):
#     meta_features = X.copy()
#     meta_features['reg_pred'] = model_reg.predict(X)
#     meta_features['clf_prob'] = model_clf.predict_proba(X)[:, 1]
#     return meta_features

# X_train_meta = create_meta_features(reg_model, clf_model, X_train)
# X_test_meta = create_meta_features(reg_model, clf_model, X_test)

# # 3. Meta Classifier Tuning
# def objective_meta_clf(trial):
#     params = {
#         "objective": "binary",
#         "boosting_type": "gbdt",
#         "verbosity": -1,
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         "is_unbalance": True,
#     }
#     model = LGBMClassifier(**params)
#     score = cross_val_score(model, X_train_meta, y_clf_train, cv=3, scoring='roc_auc').mean()
#     return score

# meta_study = optuna.create_study(direction="maximize")
# meta_study.optimize(objective_meta_clf, n_trials=50)

# # Train final meta classifier
# best_meta_params = meta_study.best_params
# best_meta_params.update({
#     "objective": "binary",
#     "verbosity": -1,
#     "random_state": 42
# })
# meta_model = LGBMClassifier(**best_meta_params)
# meta_model.fit(X_train_meta, y_clf_train)

# # Evaluate all models
# def evaluate_models():
#     # Regression evaluation
#     reg_pred = reg_model.predict(X_test)
#     print("\nRegression Evaluation:")
#     print(f"MAE: {mean_absolute_error(y_reg_test, reg_pred):.4f}")
#     print(f"RMSE: {rmse(y_reg_test, reg_pred):.4f}")
#     print(f"R²: {r2_score(y_reg_test, reg_pred):.4f}")
    
#     # Base classifier evaluation
#     clf_pred = clf_model.predict(X_test)
#     clf_prob = clf_model.predict_proba(X_test)[:, 1]
#     print("\nBase Classifier Evaluation:")
#     print(f"Accuracy: {accuracy_score(y_clf_test, clf_pred):.4f}")
#     print(f"F1 Score: {f1_score(y_clf_test, clf_pred):.4f}")
#     print(f"ROC AUC: {roc_auc_score(y_clf_test, clf_prob):.4f}")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_clf_test, clf_pred))
    
#     # Meta classifier evaluation
#     meta_pred = meta_model.predict(X_test_meta)
#     meta_prob = meta_model.predict_proba(X_test_meta)[:, 1]
#     print("\nMeta Classifier Evaluation:")
#     print(f"Accuracy: {accuracy_score(y_clf_test, meta_pred):.4f}")
#     print(f"F1 Score: {f1_score(y_clf_test, meta_pred):.4f}")
#     print(f"ROC AUC: {roc_auc_score(y_clf_test, meta_prob):.4f}")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_clf_test, meta_pred))

# evaluate_models()

# # Final training on all data
# print("\nTraining final models on all data...")
# # Create meta features for full dataset
# X_meta = create_meta_features(reg_model, clf_model, X)

# # Retrain base models on full data
# reg_model.fit(X, y_reg, categorical_feature=['pair'])
# clf_model.fit(X, y_clf, categorical_feature=['pair'])

# # Retrain meta model on full data
# meta_model.fit(X_meta, y_clf)

# # Save models
# print("Saving models...")
# reg_model.booster_.save_model("regression_model_sell.txt")
# clf_model.booster_.save_model("classifier_model_sell.txt")
# meta_model.booster_.save_model("meta_classifier_model_sell.txt")
# joblib.dump({'features': features}, 'model_features_sell.pkl')

# print("\nFeature Importances:")
# # Get feature importances for all models
# def get_feature_importance(model, feature_names, title):
#     importance = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
#     print(f"\n{title}:")
#     print(importance.head(10))
#     return importance

# reg_importance = get_feature_importance(reg_model, features, "Regression Model Feature Importance")
# clf_importance = get_feature_importance(clf_model, features, "Classifier Model Feature Importance")
# meta_features = features + ['reg_pred', 'clf_prob']
# meta_importance = get_feature_importance(meta_model, meta_features, "Meta Model Feature Importance")

# # Save importances
# reg_importance.to_csv("regression_feature_importance_sell.csv", index=False)
# clf_importance.to_csv("classifier_feature_importance_sell.csv", index=False)
# meta_importance.to_csv("meta_feature_importance_sell.csv", index=False)

# print("\nAll models trained and saved successfully!")


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
#     # 'binary_rr3': 'label_rr_3.0',
#     'regression': 'rr_label',
#     'meta': 'meta_target'
# }

# CLASSIFIER_NAMES = {
#     '1_1': 'binary_rr1',
#     '1_2': 'binary_rr2',
#     # '1_3': 'binary_rr3'
# }

# # --- 1. Load and Prepare Data ---
# df = pd.read_csv("test.csv")

# # Filter and prepare data
# df = df[df['label'] != 0].copy()  # Remove neutral trades
# df['pair'] = df['pair'].astype('category')

# # Define features and targets
# features = [col for col in df.columns if col not in [
#     'label', TARGET_COLUMNS['regression'], 
#     TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2'], TARGET_COLUMNS['binary_rr3']
# ]]

# # Create multi-class meta target
# conditions = [
#     df[TARGET_COLUMNS['regression']] >= 3.0,
#     df[TARGET_COLUMNS['regression']] >= 2.0,
#     df[TARGET_COLUMNS['regression']] >= 1.0
# ]
# choices = [3, 2, 1]
# df[TARGET_COLUMNS['meta']] = np.select(conditions, choices, default=0)

# # Train-test split
# X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_meta_train, y_meta_test = train_test_split(
#     df[features], 
#     df[TARGET_COLUMNS['regression']], 
#     df[[TARGET_COLUMNS['binary_rr1'], TARGET_COLUMNS['binary_rr2'], TARGET_COLUMNS['binary_rr3']]], 
#     df[TARGET_COLUMNS['meta']],
#     test_size=0.2, 
#     stratify=df[TARGET_COLUMNS['meta']], 
#     random_state=42
# )

# # --- 2. Define Evaluation Metrics ---
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))
# rmse_scorer = make_scorer(rmse, greater_is_better=False)


# # --- 3. Tune Base Classifiers with class weight tuning ---
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
#     model.fit(
#         X, y,
#         categorical_feature=['pair'],
#         eval_set=[(X, y)])
#     return model

# # Train all three classifiers
# classifiers = {
#     '1_1': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr1']], "1:1"),
#     '1_2': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr2']], "1:2"),
#     '1_3': tune_classifier(X_train, y_clf_train[TARGET_COLUMNS['binary_rr3']], "1:3")
# }

# # --- 4. Tune Regression Model ---
# def create_reg_features(X):
#     return pd.DataFrame({
#         **{col: X[col] for col in features},
#         'clf_1_1_prob': classifiers['1_1'].predict_proba(X)[:, 1],
#         'clf_1_2_prob': classifiers['1_2'].predict_proba(X)[:, 1],
#         'clf_1_3_prob': classifiers['1_3'].predict_proba(X)[:, 1]
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
#         'clf_1_3_prob': classifiers['1_3'].predict_proba(X)[:, 1],
#         'reg_pred': regressor.predict(reg_input)
#     })

# # --- 5. Tune Meta Classifier with class_weight tuning ---
# def tune_meta(X, y):
#     def objective(trial):
#         weight_option = trial.suggest_categorical("class_weight", [
#             None,
#             {0: 1, 1: 1.5, 2: 3, 3: 4},
#             {0: 1, 1: 2, 2: 5, 3: 6},
#             {0: 1, 1: 2, 2: 8, 3: 9}
#         ])
#         params = {
#             "objective": "multiclass",
#             "num_class": 4,
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
#             "class_weight": weight_option
#         }
#         model = LGBMClassifier(**params)
#         return cross_val_score(model, X, y, cv=3, scoring='f1_weighted').mean()

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=25)

#     best_params = study.best_params
#     best_params.update({
#         "objective": "multiclass",
#         "num_class": 4,
#         "verbosity": -1,
#         "random_state": 42
#     })

#     model = LGBMClassifier(**best_params)
#     model.fit(X, y, eval_set=[(X, y)])
    
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
    
#     print("\n🔹 Meta Model Evaluation:")
#     print("----------------------------------------")
#     print(classification_report(y_meta_test, meta_pred, 
#                               target_names=['Reject', '1:1', '1:2', '1:3'],
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
# model_dir = "./model_artifacts_buy/"
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

# print("\nFeature Importances:")
# # Get feature importances for all models
# def get_feature_importance(model, feature_names, title):
#     importance = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
#     print(f"\n{title}:")
#     print(importance.head(10))
#     return importance

# reg_importance = get_feature_importance(regressor, features, "Regression Model Feature Importance")
# clf_importance = get_feature_importance(model, features, "Classifier Model Feature Importance")
# meta_features = features + ['reg_pred', 'clf_prob']
# meta_importance = get_feature_importance(meta_model, meta_features, "Meta Model Feature Importance")

# # Save importances
# reg_importance.to_csv("regression_feature_importance_sell.csv", index=False)
# clf_importance.to_csv("classifier_feature_importance_sell.csv", index=False)
# meta_importance.to_csv("meta_feature_importance_sell.csv", index=False)

# print("\nAll models trained and saved successfully!")
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# # from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
# #                              accuracy_score, f1_score, roc_auc_score, confusion_matrix,
# #                              make_scorer)
# # from sklearn.ensemble import StackingClassifier
# # from lightgbm import LGBMRegressor, LGBMClassifier
# # import optuna
# # import joblib
# # from sklearn.base import BaseEstimator, TransformerMixin

# # # Load and combine data
# # df1 = pd.read_csv("binary_1:1_CAD.csv")
# # # df = pd.concat([df1, df2], ignore_index=True)

# # # Preprocessing
# # df = df.dropna(subset=['label', 'rr_label'])
# # df = df[df['label'] != 0].copy()
# # df['binary_label'] = df['label'].map({-1: 0, 1: 1})
# # df['pair'] = df['pair'].astype('category')

# # features = [c for c in df.columns if c not in ['label', 'binary_label', 'rr_label']]
# # X = df[features]
# # y_reg = df['rr_label']  # Regression target
# # y_clf = df['binary_label']  # Classification target

# # # Train-test split (stratified by classification label)
# # X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
# #     X, y_reg, y_clf, stratify=y_clf, test_size=0.2, random_state=42)

# # # Define scorers
# # def rmse(y_true, y_pred):
# #     return np.sqrt(mean_squared_error(y_true, y_pred))
# # rmse_scorer = make_scorer(rmse, greater_is_better=False)

# # # 1. Regression Model Tuning
# # def objective_lgb_reg(trial):
# #     params = {
# #         "objective": "regression",
# #         "boosting_type": "gbdt",
# #         "verbosity": -1,
# #         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
# #         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
# #         "max_depth": trial.suggest_int("max_depth", 3, 10),
# #         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
# #         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #     }
# #     model = LGBMRegressor(**params)
# #     score = np.abs(cross_val_score(model, X_train, y_reg_train, cv=3, scoring=rmse_scorer).mean())
# #     return score

# # reg_study = optuna.create_study(direction="minimize")
# # reg_study.optimize(objective_lgb_reg, n_trials=50)

# # # 2. Classifier Model Tuning
# # def objective_lgb_clf(trial):
# #     params = {
# #         "objective": "binary",
# #         "boosting_type": "gbdt",
# #         "verbosity": -1,
# #         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
# #         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
# #         "max_depth": trial.suggest_int("max_depth", 3, 10),
# #         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
# #         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #         "is_unbalance": True,
# #     }
# #     model = LGBMClassifier(**params)
# #     score = cross_val_score(model, X_train, y_clf_train, cv=3, scoring='roc_auc').mean()
# #     return score

# # clf_study = optuna.create_study(direction="maximize")
# # clf_study.optimize(objective_lgb_clf, n_trials=50)

# # # Train base models with best params
# # best_reg_params = reg_study.best_params
# # best_reg_params.update({
# #     "objective": "regression",
# #     "verbosity": -1,
# #     "random_state": 42
# # })
# # reg_model = LGBMRegressor(**best_reg_params)
# # reg_model.fit(X_train, y_reg_train, categorical_feature=['pair'])

# # best_clf_params = clf_study.best_params
# # best_clf_params.update({
# #     "objective": "binary",
# #     "verbosity": -1,
# #     "random_state": 42
# # })
# # clf_model = LGBMClassifier(**best_clf_params)
# # clf_model.fit(X_train, y_clf_train, categorical_feature=['pair'])

# # # Create meta-features
# # def create_meta_features(model_reg, model_clf, X):
# #     meta_features = X.copy()
# #     meta_features['reg_pred'] = model_reg.predict(X)
# #     meta_features['clf_prob'] = model_clf.predict_proba(X)[:, 1]
# #     return meta_features

# # X_train_meta = create_meta_features(reg_model, clf_model, X_train)
# # X_test_meta = create_meta_features(reg_model, clf_model, X_test)

# # # 3. Meta Classifier Tuning
# # def objective_meta_clf(trial):
# #     params = {
# #         "objective": "binary",
# #         "boosting_type": "gbdt",
# #         "verbosity": -1,
# #         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
# #         "num_leaves": trial.suggest_int("num_leaves", 16, 60),
# #         "max_depth": trial.suggest_int("max_depth", 3, 10),
# #         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
# #         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
# #         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
# #         "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
# #         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #         "is_unbalance": True,
# #     }
# #     model = LGBMClassifier(**params)
# #     score = cross_val_score(model, X_train_meta, y_clf_train, cv=3, scoring='roc_auc').mean()
# #     return score

# # meta_study = optuna.create_study(direction="maximize")
# # meta_study.optimize(objective_meta_clf, n_trials=50)

# # # Train final meta classifier
# # best_meta_params = meta_study.best_params
# # best_meta_params.update({
# #     "objective": "binary",
# #     "verbosity": -1,
# #     "random_state": 42
# # })
# # meta_model = LGBMClassifier(**best_meta_params)
# # meta_model.fit(X_train_meta, y_clf_train)

# # # Evaluate all models
# # def evaluate_models():
# #     # Regression evaluation
# #     reg_pred = reg_model.predict(X_test)
# #     print("\nRegression Evaluation:")
# #     print(f"MAE: {mean_absolute_error(y_reg_test, reg_pred):.4f}")
# #     print(f"RMSE: {rmse(y_reg_test, reg_pred):.4f}")
# #     print(f"R²: {r2_score(y_reg_test, reg_pred):.4f}")
    
# #     # Base classifier evaluation
# #     clf_pred = clf_model.predict(X_test)
# #     clf_prob = clf_model.predict_proba(X_test)[:, 1]
# #     print("\nBase Classifier Evaluation:")
# #     print(f"Accuracy: {accuracy_score(y_clf_test, clf_pred):.4f}")
# #     print(f"F1 Score: {f1_score(y_clf_test, clf_pred):.4f}")
# #     print(f"ROC AUC: {roc_auc_score(y_clf_test, clf_prob):.4f}")
# #     print("Confusion Matrix:")
# #     print(confusion_matrix(y_clf_test, clf_pred))
    
# #     # Meta classifier evaluation
# #     meta_pred = meta_model.predict(X_test_meta)
# #     meta_prob = meta_model.predict_proba(X_test_meta)[:, 1]
# #     print("\nMeta Classifier Evaluation:")
# #     print(f"Accuracy: {accuracy_score(y_clf_test, meta_pred):.4f}")
# #     print(f"F1 Score: {f1_score(y_clf_test, meta_pred):.4f}")
# #     print(f"ROC AUC: {roc_auc_score(y_clf_test, meta_prob):.4f}")
# #     print("Confusion Matrix:")
# #     print(confusion_matrix(y_clf_test, meta_pred))

# # evaluate_models()

# # # Final training on all data
# # print("\nTraining final models on all data...")
# # # Create meta features for full dataset
# # X_meta = create_meta_features(reg_model, clf_model, X)

# # # Retrain base models on full data
# # reg_model.fit(X, y_reg, categorical_feature=['pair'])
# # clf_model.fit(X, y_clf, categorical_feature=['pair'])

# # # Retrain meta model on full data
# # meta_model.fit(X_meta, y_clf)

# # # Save models
# # print("Saving models...")
# # reg_model.booster_.save_model("regression_model_sell.txt")
# # clf_model.booster_.save_model("classifier_model_sell.txt")
# # meta_model.booster_.save_model("meta_classifier_model_sell.txt")
# # joblib.dump({'features': features}, 'model_features_sell.pkl')

# # print("\nFeature Importances:")
# # # Get feature importances for all models
# # def get_feature_importance(model, feature_names, title):
# #     importance = pd.DataFrame({
# #         'Feature': feature_names,
# #         'Importance': model.feature_importances_
# #     }).sort_values('Importance', ascending=False)
# #     print(f"\n{title}:")
# #     print(importance.head(10))
# #     return importance

# # reg_importance = get_feature_importance(reg_model, features, "Regression Model Feature Importance")
# # clf_importance = get_feature_importance(clf_model, features, "Classifier Model Feature Importance")
# # meta_features = features + ['reg_pred', 'clf_prob']
# # meta_importance = get_feature_importance(meta_model, meta_features, "Meta Model Feature Importance")

# # # Save importances
# # reg_importance.to_csv("regression_feature_importance_sell.csv", index=False)
# # clf_importance.to_csv("classifier_feature_importance_sell.csv", index=False)
# # meta_importance.to_csv("meta_feature_importance_sell.csv", index=False)

# # print("\nAll models trained and saved successfully!")