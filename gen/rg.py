import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import optuna
import warnings
warnings.filterwarnings("ignore")

# === Load and combine multiple labeled pair CSVs ===
csv_files = ["ggtx.csv"]  # "./csv/xau-sell.csv", "./csv/jpy-sell.csv", "./csv/gbp-sell.csv" Add more if needed

dfs = []
for file in csv_files:
    print(f"📥 Reading {file}")
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.dropna(inplace=True)
# df = df[(df["rr_label"] >= 1.1) & (df["rr_label"] <= 3.0)]
print(f"\n✅ Combined dataset shape: {df.shape}")

# ───────── Make 'pair' categorical ─────────
df['pair'] = df['pair'].astype('category')

# ───────── Prepare X/y ─────────
X = df.drop(columns=["rr_label", "rr_class", "time_to_outcome"])
y = df["rr_label"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === RMSE scorer ===
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# === Optuna: LightGBM objective ===
def objective_lgb(trial):
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 60),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "categorical_feature": ["pair"],
    }
    model = lgb.LGBMRegressor(**params)
    score = np.abs(cross_val_score(
        model, X_train, y_train,
        cv=3, scoring=rmse_scorer
    ).mean())
    return score

print("\n🔧 Tuning LightGBM with Optuna...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=1000)

# Save best params
best_params_lgb = {
    **study_lgb.best_params,
    "objective": "regression",
    "verbosity": -1,
    "categorical_feature": ["pair"],
}

# === Train LightGBM ===
print("\n📈 Training final LightGBM model...")
lgb_model = lgb.LGBMRegressor(**best_params_lgb)
lgb_model.fit(X_train, y_train)
lgb_model.booster_.save_model("./model/test2.txt")
print("💾 Saved LightGBM model")

# === Evaluate on Test Set ===
y_pred = lgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = rmse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'Model':<10} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
print(f"{'LightGBM':<10} | {mae:<8.4f} | {rmse_val:<8.4f} | {r2:<8.4f}")

# === Final Training on Full Data ===
final_model = lgb.LGBMRegressor(**best_params_lgb)
final_model.fit(X, y)
final_model.booster_.save_model("./model/test23.txt")
print("\n✅ Final LightGBM model trained on full dataset and saved.")

# === Feature Importance ===
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("lgb_feature_importance.csv", index=False)
print("📊 Feature importances saved to 'lgb_feature_importance.csv'")

# === Save Top 20 Features (Optional)
top_20_df = importance_df.head(20)
top_20_df.to_csv("lgb_feature_importance_top20.csv", index=False)

# === Correlation Matrix of Input Features (Numerics Only)
numeric_X = X.select_dtypes(include=[np.number])
correlation_matrix = numeric_X.corr()
correlation_matrix.to_csv("feature_correlation_matrix.csv")
print("📉 Feature correlation matrix saved to 'feature_correlation_matrix.csv'")

# === Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png", dpi=300)
plt.close()
print("🖼️ Correlation heatmap saved as 'feature_correlation_heatmap.png'")


#-------------------------------------------------------------------
# # === Optuna: XGBoost objective ===
# def objective_xgb(trial):
#     params = {
#         "objective": "reg:squarederror",
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
#         "n_estimators": 1000
#     }
#     model = xgb.XGBRegressor(**params)
#     score = np.abs(cross_val_score(model, X_train, y_train, cv=5, scoring=rmse_scorer).mean())
#     return score

# print("\n🔧 Tuning XGBoost...")
# study_xgb = optuna.create_study(direction="minimize")
# study_xgb.optimize(objective_xgb, n_trials=1)

# #Safe copy to avoid mutating Optuna dict
# best_params_xgb = {
#     **study_xgb.best_params,
#     "objective": "reg:squarederror",
#     "n_estimators": 1000,
#     "early_stopping_rounds": 50  # Moved here from fit()
# }

# # === Train XGBoost ===
# xgb_model = xgb.XGBRegressor(**best_params_xgb)
# xgb_model.fit(
#     X_train, 
#     y_train,
#     eval_set=[(X_test, y_test)],
#     verbose=False
# )
# # xgb_model.save_model("xgb_model.json")
# print("💾 Saved XGBoost model to 'xgb_model.json'")

# # === Evaluate Both Models ===
# def evaluate_model(name, model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse_val = rmse(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f"\n📊 {name} Evaluation:")
#     print(f"MAE:  {mae:.4f}")
#     print(f"RMSE: {rmse_val:.4f}")
#     print(f"R²:   {r2:.4f}")
#     return mae, rmse_val, r2

# mae_lgb, rmse_lgb, r2_lgb = evaluate_model("LightGBM", lgb_model, X_test, y_test)
# mae_xgb, rmse_xgb, r2_xgb = evaluate_model("XGBoost", xgb_model, X_test, y_test)

# # === Save Feature Importances ===
# fi_lgb = pd.DataFrame({
#     "feature": X.columns,
#     "gain": lgb_model.booster_.feature_importance(importance_type="gain")
# }).sort_values(by="gain", ascending=False)

# fi_xgb = pd.DataFrame({
#     "feature": X.columns,
#     "gain": xgb_model.feature_importances_
# }).sort_values(by="gain", ascending=False)

# fi_lgb.to_csv("feature_importance_lgb.csv", index=False)
# fi_xgb.to_csv("feature_importance_xgb.csv", index=False)
# print("\n📁 Feature importances saved to:")
# print("   - feature_importance_lgb.csv")
# print("   - feature_importance_xgb.csv")

# # === Compare Summary ===
# print("\n📈 Model Comparison Summary:")
# print(f"{'Model':<10} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
# print("-" * 38)
# print(f"{'LightGBM':<10} | {mae_lgb:.4f} | {rmse_lgb:.4f} | {r2_lgb:.4f}")
# print(f"{'XGBoost':<10} | {mae_xgb:.4f} | {rmse_xgb:.4f} | {r2_xgb:.4f}")
