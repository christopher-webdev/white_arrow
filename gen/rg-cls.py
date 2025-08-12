import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings("ignore")

# === Load and combine labeled CSVs ===
csv_files = ["./csv/cad-sell.csv", "./csv/xau-sell.csv", "./csv/jpy-sell.csv", "./csv/gbp-sell.csv"]
dfs = []

for file in csv_files:
    print(f"📥 Reading {file}")
    df_tmp = pd.read_csv(file)
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df.dropna(inplace=True)
print(f"\n✅ Combined dataset shape: {df.shape}")

# === Drop invalid class labels ===
df = df[df["rr_class"].isin([0, 1])]
df["pair"] = df["pair"].astype("category")

# === Prepare features and labels ===
X = df.drop(columns=["rr_label", "rr_class", "time_to_outcome"])
y = df["rr_class"].astype(int)

# === Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Compute class weights for imbalance ===
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive
print(f"⚖️ scale_pos_weight = {scale_pos_weight:.3f}")

# === Optuna objective for classifier
def objective_clf(trial):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "metric": "binary_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 60),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "categorical_feature": ["pair"],
        # "is_unbalance": True,
        "scale_pos_weight": scale_pos_weight,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
    return scores.mean()

print("\n🔧 Tuning LightGBM Classifier...")
study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=50)

best_params_clf = {
    **study_clf.best_params,
    "objective": "binary",
    "verbosity": -1,
    "categorical_feature": ["pair"],
    # "is_unbalance": True,
    "scale_pos_weight": scale_pos_weight
}

# === Train final model
print("\n📈 Training best model on train set...")
clf = lgb.LGBMClassifier(**best_params_clf)
clf.fit(X_train, y_train)
clf.booster_.save_model("lgb_classifier_model_sell.txt")

# === Predict probabilities for threshold tuning
y_prob = clf.predict_proba(X_test)[:, 1]

# === Find optimal threshold for max F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🎯 Best Threshold for F1: {best_threshold:.3f} (F1={best_f1:.3f})")

# === Apply optimal threshold
y_pred_opt = (y_prob >= best_threshold).astype(int)
acc = accuracy_score(y_test, y_pred_opt)
cm = confusion_matrix(y_test, y_pred_opt)
report = classification_report(y_test, y_pred_opt, target_names=["SL", "TP"])

print(f"\n✅ Accuracy on test set (optimized threshold): {acc:.4f}")
print("📊 Classification Report (Optimized Threshold):\n", report)

# === Save confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["SL", "TP"], yticklabels=["SL", "TP"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Optimized Threshold)")
plt.tight_layout()
plt.savefig("confusion_matrix_optimized.png")
plt.close()
print("🖼️ Confusion matrix saved as 'confusion_matrix_optimized.png'")

# === ROC Curve & AUC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()
print("🖼️ ROC Curve saved as 'roc_curve.png'")

# === Precision-Recall Curve & F1 Threshold
plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, label='PR Curve')
plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label=f'Best F1 Thr={best_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.close()
print("🖼️ Precision-Recall Curve saved as 'precision_recall_curve.png'")

# === Final training on full data
print("\n📦 Training on full dataset...")
final_clf = lgb.LGBMClassifier(**best_params_clf)
final_clf.fit(X, y)
final_clf.booster_.save_model("lgb_classifier_model_full_sell.txt")
print("✅ Full model saved as 'lgb_classifier_model_full.txt'")

# === Feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("lgb_classifier_feature_importance.csv", index=False)
print("📊 Feature importances saved to 'lgb_classifier_feature_importance.csv'")

# === Top 20 features
importance_df.head(20).to_csv("lgb_classifier_feature_importance_top20.csv", index=False)

# === Correlation Matrix of Numerical Features
numeric_X = X.select_dtypes(include=[np.number])
correlation_matrix = numeric_X.corr()
correlation_matrix.to_csv("feature_correlation_matrix_classifier.csv")
print("📉 Correlation matrix saved as 'feature_correlation_matrix_classifier.csv'")

# === Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("feature_correlation_heatmap_classifier.png", dpi=300)
plt.close()
print("🖼️ Heatmap saved as 'feature_correlation_heatmap_classifier.png'")
