import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import os

# === CONFIG ===
model_path = "./model_artifacts_buy/classifier_1_2.txt"
output_dir = "./model_analysis_output/"
X_sample_path = "./labeled_combined_dataset_buy.csv"  # Replace with a small sample of your training data
top_n_features = 20

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# === LOAD MODEL ===
model = lgb.Booster(model_file=model_path)

# === FEATURE IMPORTANCE ===
features = model.feature_name()
importances_gain = model.feature_importance(importance_type="gain")
importances_split = model.feature_importance(importance_type="split")

df_importance = pd.DataFrame({
    "feature": features,
    "importance_gain": importances_gain,
    "importance_split": importances_split
}).sort_values(by="importance_gain", ascending=False)

# === SAVE IMPORTANCES CSV ===
df_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

# === PLOT FEATURE IMPORTANCE (Gain) ===
plt.figure(figsize=(10, 8))
sns.barplot(
    data=df_importance.head(top_n_features),
    x="importance_gain",
    y="feature",
    palette="viridis"
)
plt.title(f"Top {top_n_features} Feature Importances (Gain)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_gain.png"))
plt.close()

# === PLOT FEATURE IMPORTANCE (Split) ===
plt.figure(figsize=(10, 8))
sns.barplot(
    data=df_importance.head(top_n_features),
    x="importance_split",
    y="feature",
    palette="magma"
)
plt.title(f"Top {top_n_features} Feature Importances (Split)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_split.png"))
plt.close()

# === LOAD SAMPLE DATA FOR SHAP ===
X = pd.read_csv(X_sample_path)
X = X[features]  # Ensure correct order

# === SHAP EXPLAINER ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === SHAP SUMMARY PLOT ===
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
plt.close()

# === SHAP VALUES AS DF ===
shap_df = pd.DataFrame(shap_values, columns=features)
shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)

# === CORRELATION BETWEEN FEATURES AND SHAP VALUES ===
correlation_df = pd.DataFrame({
    "feature": features,
    "correlation": [X[feat].corr(shap_df[feat]) for feat in features]
}).sort_values(by="correlation", key=abs, ascending=False)

correlation_df.to_csv(os.path.join(output_dir, "feature_shap_correlation.csv"), index=False)

# === PLOT SHAP-FEATURE CORRELATION ===
plt.figure(figsize=(10, 8))
sns.barplot(data=correlation_df.head(top_n_features), x="correlation", y="feature", palette="coolwarm")
plt.title(f"Top {top_n_features} Feature ↔ SHAP Correlation")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_feature_correlation.png"))
plt.close()

print("✅ Analysis complete. Outputs saved to:", output_dir)
