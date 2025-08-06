
# import pandas as pd
# import numpy as np
# from ta.volatility import BollingerBands
# from ta.trend import SMAIndicator
# from ta.momentum import RSIIndicator
# from tqdm import tqdm

# # === Load Data ===
# df = pd.read_csv("xx.csv")  # Replace with your CSV
# df = df.rename(columns=lambda x: x.strip().capitalize())
# df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
# df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
# df['hour'] = df['Time'].dt.hour
# df['valid_hour'] = df['hour'].between(2, 21)
# df.reset_index(drop=True, inplace=True)

# # === Indicator Calculations ===
# df['sma10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
# df['sma9'] = SMAIndicator(df['Close'], window=9).sma_indicator()
# df['sma20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
# df['sma50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
# df['vol_ma12'] = df['Volume'].rolling(12).mean()
# df['rsi'] = RSIIndicator(df['Close'], window=9).rsi()

# bb = BollingerBands(close=df['Close'], window=10, window_dev=2)
# df['bb_lower'] = bb.bollinger_lband()
# df['bb_mid'] = bb.bollinger_mavg()
# df['bb_upper'] = bb.bollinger_hband()
# df['bb_width'] = df['bb_upper'] - df['bb_lower']
# df['bb_percent'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])


# df['returns'] = df['Close'].pct_change()
# df['rolling_std'] = df['returns'].rolling(5).std()
# df['hourly_volatility'] = df.groupby('hour')['returns'].transform(lambda x: x.rolling(5).std())
# df['recent_high'] = df['High'].rolling(20).max()
# df['recent_low'] = df['Low'].rolling(20).min()

# # === Trade Logic & Feature Collection ===
# results = []

# for i in tqdm(range(15, len(df) - 40)):
#     row = df.iloc[i]
#     prev = df.iloc[i - 1]
#     valid_hour = row['valid_hour']

#     # Entry conditions
#     sm9_cross = prev['Close'] < prev['sma9'] and row['Close'] > row['sma9'] 
#     vol_condition = row['Volume'] > row['vol_ma12']
#     bb_condition = row['Close'] > row['bb_mid']
#     bullish = row['Close'] > row['Open']

#     if sm9_cross and valid_hour:
#         entry_price = row['High']
#         next_high = df.iloc[i + 1]['High']
#         if next_high < entry_price:
#             continue

#         stop_loss = row['bb_lower']
#         if pd.isna(stop_loss) or stop_loss >= entry_price or stop_loss >= row['Low']:
#             continue

#         sl_distance = entry_price - stop_loss
#         tp2 = entry_price + 2 * sl_distance
#         tp3 = entry_price + 3 * sl_distance

#         future = df.iloc[i + 1:i + 41]
#         hit_tp2 = (future['High'] >= tp2).any()
#         hit_tp3 = (future['High'] >= tp3).any()
#         hit_sl = (future['Low'] <= stop_loss).any()

#         # Label outcome
#         result = "SL"
#         if hit_tp2 and hit_tp3:
#             result = "TP3"
#         elif hit_tp2:
#             result = "TP2"
#         elif not hit_sl:
#             result = "NoHit"

#         # === Feature Snapshot ===
#         snapshot = {
#             "Index": i,
#             "Result": result,
#             "RSI": row['rsi'],
#             "PGI": (row['Close'] - row['Open']) / (row['High'] - row['Low']) if (row['High'] - row['Low']) > 0 else 0,
#             "VolSpike": row['Volume'] / row['vol_ma12'] if row['vol_ma12'] else 0,
#             "BBWidth": row['bb_width'],
#             "BB_Percent": row['bb_percent'],
#             "Close_SMA20_Diff": row['Close'] - row['sma20'],
#             "Close_SMA50_Diff": row['Close'] - row['sma50'],
#             "Volatility": row['rolling_std'],
#             'Hourly_Volatility': row['hourly_volatility'],
#             "Dist_Recent_High": (row['recent_high'] - row['Close']) / row['Close'] if row['Close'] != 0 else 0,
#             "Dist_Recent_Low": (row['Close'] - row['recent_low']) / row['Close'] if row['Close'] != 0 else 0,

#         }
#         results.append(snapshot)

# # === Save to CSV ===
# snapshot_df = pd.DataFrame(results)
# snapshot_df.to_csv("trade_snapshots_xx.csv", index=False)
# print("✅ Trade snapshot saved to 'trade_snapshots.csv'")


#-------------------------------------grok

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import logging

# # === Setup Logging ===
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # === Load and Prepare Data ===
# try:
#     df = pd.read_csv("trade_snapshots_xx.csv")
# except FileNotFoundError:
#     logging.error("File 'trade_snapshots_xx.csv' not found")
#     raise

# # === Handle Missing Values ===
# print("\n=== Missing Values Before Handling ===")
# print(df.isna().sum())

# # Drop rows where 'Result' is missing
# df = df.dropna(subset=['Result'])

# # Replace infinities with NaN and impute numeric columns with median
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# print("\n=== Missing Values After Handling ===")
# print(df.isna().sum())

# # === Check Class Distribution ===
# print("\n=== Class Distribution ===")
# print(df['Result'].value_counts(normalize=True))

# # === Encode Results ===
# label_map = {"SL": 0, "NoHit": 1, "TP2": 2, "TP3": 3}
# df['Label'] = df['Result'].map(label_map)
# df['Binary_Label'] = df['Result'].apply(lambda x: 1 if x in ['TP2', 'TP3'] else 0)

# # === Feature Selection ===
# features = df.drop(columns=["Index", "Result", "Label", "Binary_Label"])

# # Check for highly correlated features
# print("\n=== Highly Correlated Feature Pairs (>|0.8|) ===")
# feature_corr = features.corr()
# high_corr = feature_corr.where(np.triu(np.abs(feature_corr) > 0.8, k=1)).stack()
# print(high_corr)

# # 1. Linear Correlation (Binary Label for TP2/TP3 focus)
# corr_matrix = features.corrwith(df['Binary_Label'])
# print("\n=== Top 5 Correlations with Binary Label (TP2/TP3) ===")
# print(corr_matrix.abs().sort_values(ascending=False).head(5).round(3))

# # 2. Mutual Information (Binary Label)
# X = features.replace([np.inf, -np.inf], np.nan).fillna(features.median())
# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
# mi_scores = mutual_info_classif(X, df['Binary_Label'], discrete_features=False, random_state=42)
# mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# print("\n=== Top 5 Mutual Information Scores (Binary Label) ===")
# print(mi_series.head(5).round(3))

# # === Visualizations ===
# # 1. Correlation Heatmap
# plt.figure(figsize=(12, 8))
# top_corr_features = corr_matrix.abs().sort_values(ascending=False).head(5).index
# corr_data = df[top_corr_features.tolist() + ['Binary_Label']].corr()
# sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
# plt.title("Correlation Heatmap with Binary Label (TP2/TP3)")
# plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)
# plt.close()

# # 2. Feature Distribution by Result
# plt.figure(figsize=(14, 6))
# for i, feature in enumerate(top_corr_features[:3], 1):
#     plt.subplot(1, 3, i)
#     plot_data = np.log1p(df[feature]) if df[feature].max() > 100 else df[feature]
#     sns.boxplot(x='Result', y=plot_data, data=df, order=["SL", "NoHit", "TP2", "TP3"])
#     plt.title(f"{feature} Distribution (Log)" if df[feature].max() > 100 else f"{feature} Distribution")
#     plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('feature_distributions.png', bbox_inches='tight', dpi=300)
# plt.close()

# # 3. Mutual Information Bar Plot
# plt.figure(figsize=(10, 6))
# mi_series.head(10).plot(kind='bar', color='teal')
# plt.axhline(y=0.1, color='red', linestyle='--', label='MI Threshold (0.1)')
# plt.title("Top 10 Features by Mutual Information (Binary Label)")
# plt.ylabel("MI Score")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.savefig('mutual_info.png', bbox_inches='tight', dpi=300)
# plt.close()

# print("Plots saved as PNG files in current directory")

#--------------------------------end grok

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_selection import mutual_info_classif
# import numpy as np

# # === Load and Prepare Data ===
# df = pd.read_csv("trade_snapshots_xx.csv")

# # === Handle Missing Values ===
# print("\n=== Missing Values Before Handling ===")
# print(df.isna().sum())

# # Simple imputation for numeric columns
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# # === Encode Results ===
# label_map = {"SL": 0, "NoHit": 1, "TP2": 2, "TP3": 3}  # Changed to ordinal encoding
# df['Label'] = df['Result'].map(label_map)

# # === Feature Selection ===
# features = df.drop(columns=["Index", "Result", "Label"])

# # 1. Linear Correlation
# corr_matrix = features.corrwith(df['Label'])
# print("\n=== Top 5 Correlations with Label ===")
# print(corr_matrix.abs().sort_values(ascending=False).head(5).round(3))

# # 2. Mutual Information (non-linear)
# X = features
# y = df['Label']

# # Ensure no NaN/infinity values remain
# X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
# mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# print("\n=== Top 5 Mutual Information Scores ===")
# print(mi_series.head(5).round(3))

# # === Visualizations ===

# # 1. Correlation Heatmap
# plt.figure(figsize=(12, 8))
# top_corr_features = corr_matrix.abs().sort_values(ascending=False).head(5).index
# sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm')
# plt.title("Top Correlated Features Heatmap")
# plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)  # Save instead of show
# plt.close()

# # 2. Feature Distribution by Result
# plt.figure(figsize=(14, 6))
# for i, feature in enumerate(top_corr_features[:3], 1):
#     plt.subplot(1, 3, i)
#     sns.boxplot(x='Result', y=feature, data=df, order=["SL", "NoHit", "TP2", "TP3"])
#     plt.title(f"{feature} Distribution")
#     plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('feature_distributions.png', bbox_inches='tight', dpi=300)
# plt.close()

# # 3. Mutual Information Bar Plot
# plt.figure(figsize=(10, 6))
# mi_series.head(10).plot(kind='bar', color='teal')
# plt.title("Top 10 Features by Mutual Information")
# plt.ylabel("MI Score")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('mutual_info.png', bbox_inches='tight', dpi=300)
# plt.close()

# print("Plots saved as PNG files in current directory")

#------------------------------------------------------

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from tqdm import tqdm

# === Load Data ===
df = pd.read_csv("jjt.csv")  # Replace with your OHLCV file
df = df.rename(columns=lambda x: x.strip().capitalize())
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
df['hour'] = df['Time'].dt.hour
df['valid_hour'] = df['hour'].between(2, 21)
df.reset_index(drop=True, inplace=True)

# === Indicators ===
df['sma10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
df['sma9']  = SMAIndicator(df['Close'], window=9).sma_indicator()
df['sma20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
df['vol_ma12']  = df['Volume'].rolling(12).mean()
df['rsi']      = RSIIndicator(df['Close'], window=9).rsi()
# === RSI Slope and Volatility ===
df['rsi_slope'] = np.gradient(df['rsi'])
df['rsi_slope_std'] = df['rsi_slope'].rolling(5).std()

bb = BollingerBands(close=df['Close'], window=10, window_dev=2)
df['bb_lower'] = bb.bollinger_lband()
df['bb_mid']   = bb.bollinger_mavg()
df['bb_upper'] = bb.bollinger_hband()
df['bb_width'] = df['bb_upper'] - df['bb_lower']

df['returns']     = df['Close'].pct_change()
df['rolling_std'] = df['returns'].rolling(5).std()

# === Compute thresholds ===
bb_quantile_thresh    = df['bb_width'].quantile(0.15)
vol_quantile_thresh   = df['rolling_std'].quantile(0.15)


print(f"Quantile thresholds: BBWidth ≤ {bb_quantile_thresh:.4f}, Volatility ≤ {vol_quantile_thresh:.6f}")
# === RSI Filter Thresholds ===
rsi_slope_thresh = df['rsi_slope'].quantile(0.9)       # only keep rows where slope is decently strong
rsi_slope_std_thresh = df['rsi_slope_std'].quantile(0.2)  # avoid unstable RSI
print(f"RSI Slope ≥ {rsi_slope_thresh:.4f}, RSI Slope Std ≤ {rsi_slope_std_thresh:.4f}")

# def passes_filter(row):
#     return (
#         row['bb_width']    <= bb_quantile_thresh  and
       
#         row['rolling_std'] <= vol_quantile_thresh
#     )
def passes_filter(row):
    return (
        row['bb_width']       <= bb_quantile_thresh and
        row['rolling_std']    <= vol_quantile_thresh 
        # row['rsi_slope']      >= rsi_slope_thresh 
        # row['rsi_slope_std']  <= rsi_slope_std_thresh
    )


pair_code = "jpy"
spread_limits = {
        "xau": 03.00, "jpy": 0.070,
        "gbp": 0.00080, 
        "cad": 0.00060
 }


# === Strategy Logic ===
rr_ratio      = 2
filtered_in   = 0
trades        = 0
wins          = 0
losses        = 0
gross_profit  = 0.0
gross_loss    = 0.0

log = []

for i in tqdm(range(20, len(df) - 48)):
    row  = df.iloc[i]
    prev = df.iloc[i - 1]

    # entry conditions
    if pair_code not in spread_limits:
        raise KeyError(f"No spread limit defined for pair '{pair_code}'")
    if not row['valid_hour']:
        continue
    sm9_cross = (prev['Close'] < prev['sma9']) and (row['Close'] > row['sma9'])
    if not sm9_cross:
        continue

    # combined quantile + static BBWidth filter
    if not passes_filter(row):
        continue
    filtered_in += 1

    entry_price = row['High']
    next_high   = df.iloc[i + 1]['High']
    if next_high < entry_price:
        continue

    stop_loss = row['bb_lower']
    if pd.isna(stop_loss) or stop_loss >= entry_price:
        continue

    sl_distance = entry_price - stop_loss
    tp_price    = entry_price + rr_ratio * sl_distance

    max_spread = spread_limits[pair_code]
    if sl_distance < max_spread:
        continue


    future = df.iloc[i + 1 : i + 48]
    hit_tp = (future['High'] >= tp_price).any()
    hit_sl = (future['Low']  <= stop_loss).any()

    trade_result = "None"
    rr           = -1

    if hit_tp and hit_sl:
        tp_idx = future['High'][future['High'] >= tp_price].index[0]
        sl_idx = future['Low'][future['Low'] <= stop_loss].index[0]
        if tp_idx < sl_idx:
            wins         += 1
            gross_profit += rr_ratio
            rr            = rr_ratio
            trade_result = "TP"
        else:
            losses       += 1
            gross_loss   += 1
            trade_result = "SL"
    elif hit_tp:
        wins         += 1
        gross_profit += rr_ratio
        rr            = rr_ratio
        trade_result = "TP"
    else:
        losses      += 1
        gross_loss  += 1
        trade_result = "SL" if hit_sl else "NoHit"

    trades += 1
    log.append({
        "Index":      i,
        "Entry":      entry_price,
        "StopLoss":   stop_loss,
        "TakeProfit": tp_price,
        "RR":         rr,
        "Result":     trade_result,
        "Time":       row['Time']
    })

# === Metrics ===
if trades > 0:
    win_rate      = wins / trades * 100
    avg_rr        = (gross_profit - gross_loss) / trades
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    print("\n=== Filtered Strategy Performance ===")
    print(f"Total Trades Triggered by Strategy: {trades}")
    print(f"Filtered‑In Trades: {filtered_in}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor (PF): {profit_factor:.2f}")
    print(f"Average RR: {avg_rr:.2f}")
else:
    print("No trades passed the filter.")

# === Export Trade Log ===
log_df = pd.DataFrame(log)
log_df.to_csv("filtered_trade_log.csv", index=False)
print("\n✅ Trade log saved to 'filtered_trade_log.csv'")




















# import pandas as pd
# import numpy as np
# from ta.volatility import BollingerBands
# from ta.trend import SMAIndicator
# from ta.momentum import RSIIndicator
# from tqdm import tqdm

# # === Load Data ===
# df = pd.read_csv("xxt.csv")  # Replace with your OHLCV file
# df = df.rename(columns=lambda x: x.strip().capitalize())
# df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
# df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
# df['hour'] = df['Time'].dt.hour
# df['valid_hour'] = df['hour'].between(2, 21)

# df.reset_index(drop=True, inplace=True)

# # === Indicators ===
# df['sma10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
# df['sma9'] = SMAIndicator(df['Close'], window=9).sma_indicator()
# df['sma20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
# df['vol_ma12'] = df['Volume'].rolling(12).mean()
# df['rsi'] = RSIIndicator(df['Close'], window=9).rsi()

# bb = BollingerBands(close=df['Close'], window=10, window_dev=2)
# df['bb_lower'] = bb.bollinger_lband()
# df['bb_mid'] = bb.bollinger_mavg()
# df['bb_upper'] = bb.bollinger_hband()
# df['bb_width'] = df['bb_upper'] - df['bb_lower']

# df['returns'] = df['Close'].pct_change()
# df['rolling_std'] = df['returns'].rolling(5).std()
# df['returns'] = df['Close'].pct_change()



# # === Live Filter Based on Stats of TP2/TP3 Winners ===
# # Define filter thresholds using percentiles from your data
# bbwidth_thresh = df['bb_width'].quantile(0.15)
# volatility_thresh = df['rolling_std'].quantile(0.15)

# def passes_filter(row):
#     try:
#         return (
#             row['bb_width'] <= bbwidth_thresh and

#             row['rolling_std'] <= volatility_thresh
#         )
#     except:
#         return False
# # === Strategy Logic ===
# rr_ratio = 2
# filtered_in = 0
# trades = 0
# wins = 0
# losses = 0
# gross_profit = 0
# gross_loss = 0

# log = []

# for i in tqdm(range(20, len(df) - 40)):
#     row = df.iloc[i]
#     prev = df.iloc[i - 1]
#     valid_hour = row['valid_hour']

#     sm9_cross = prev['Close'] < prev['sma9'] and row['Close'] > row['sma9']
  
#     if sm9_cross and valid_hour:#and vol_condition and bb_condition  and bullish
#         if not passes_filter(row):
#             continue
#         filtered_in += 1

#         entry_price = row['High']
#         next_high = df.iloc[i + 1]['High']
#         if next_high < entry_price:
#             continue

#         stop_loss = row['bb_lower']
#         if pd.isna(stop_loss) or stop_loss >= entry_price:
#             continue

#         sl_distance = entry_price - stop_loss
#         tp_price = entry_price + rr_ratio * sl_distance

#         future = df.iloc[i + 1:i + 41]
#         hit_tp = (future['High'] >= tp_price).any()
#         hit_sl = (future['Low'] <= stop_loss).any()

#         trade_result = "None"
#         rr = -1

#         if hit_tp and hit_sl:
#             tp_index = future['High'][future['High'] >= tp_price].index[0]
#             sl_index = future['Low'][future['Low'] <= stop_loss].index[0]
#             if tp_index < sl_index:
#                 wins += 1
#                 gross_profit += rr_ratio
#                 rr = rr_ratio
#                 trade_result = "TP"
#             else:
#                 losses += 1
#                 gross_loss += 1
#                 trade_result = "SL"
#         elif hit_tp:
#             wins += 1
#             gross_profit += rr_ratio
#             rr = rr_ratio
#             trade_result = "TP"
#         elif hit_sl:
#             losses += 1
#             gross_loss += 1
#             trade_result = "SL"
#         else:
#             losses += 1
#             gross_loss += 1
#             trade_result = "NoHit"

#         trades += 1
#         log.append({
#             "Index": i,
#             "Entry": entry_price,
#             "StopLoss": stop_loss,
#             "TakeProfit": tp_price,
#             "RR": rr,
#             "Result": trade_result,
#             "Time": row['Time']
#         })

# # === Metrics ===
# if trades > 0:
#     win_rate = wins / trades * 100
#     avg_rr = (gross_profit - gross_loss) / trades
#     profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float('inf')

#     print("\n=== Filtered Strategy Performance ===")
#     print(f"Total Trades Triggered by Strategy: {trades}")
#     print(f"Filtered-In Trades: {filtered_in}")
#     print(f"Wins: {wins} | Losses: {losses}")
#     print(f"Win Rate: {win_rate:.2f}%")
#     print(f"Profit Factor (PF): {profit_factor:.2f}")
#     print(f"Average RR: {avg_rr:.2f}")
# else:
#     print("No trades passed the filter.")

# # === Export Trade Log ===
# log_df = pd.DataFrame(log)
# log_df.to_csv("filtered_trade_log.csv", index=False)
# print("\n✅ Trade log saved to 'filtered_trade_log.csv'")
