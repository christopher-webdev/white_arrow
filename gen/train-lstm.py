import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Load and Filter Data ===
df = pd.read_csv("./csv/gbp.csv")
df = df[df["rr_label"] >= 0].copy()

# === 2. Preprocess for LightGBM ===
drop_cols = ["rr_label", "rr_class", "time_to_outcome"]
features = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
target = df["rr_label"].values

X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(features, target, test_size=0.2, random_state=42)

# === 3. Train LightGBM Regressor ===
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7)
lgb_model.fit(X_train_lgb, y_train_lgb)
lgb_preds = lgb_model.predict(X_val_lgb)

lgb_rmse = mean_squared_error(y_val_lgb, lgb_preds, squared=False)
lgb_r2 = r2_score(y_val_lgb, lgb_preds)

print("🔶 LightGBM Results:")
print(f"   📉 RMSE: {lgb_rmse:.4f}")
print(f"   📈 R² Score: {lgb_r2:.4f}")

# === 4. Preprocess for LSTM ===
def get_lstm_data(df, target_col="rr_label", window_size=48):
    drop_cols = ["rr_label", "rr_class", "time_to_outcome"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(window_size, len(X_scaled)):
        X_seq.append(X_scaled[i - window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_lstm, y_lstm = get_lstm_data(df)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# === 5. Train LSTM Regressor ===
model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', metrics=['mae'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm),
          epochs=20, batch_size=64, callbacks=[early_stop], verbose=0)

lstm_preds = model.predict(X_val_lstm).flatten()
lstm_rmse = mean_squared_error(y_val_lstm, lstm_preds, squared=False)
lstm_r2 = r2_score(y_val_lstm, lstm_preds)

print("\n🔷 LSTM Results:")
print(f"   📉 RMSE: {lstm_rmse:.4f}")
print(f"   📈 R² Score: {lstm_r2:.4f}")
