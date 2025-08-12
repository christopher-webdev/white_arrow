import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# === CONFIG ===
DATA_PATH = '3model-buy.csv'
LOOKAHEAD = 60
RR_TARGET = 1.0
RR_MARGIN = 0.0
EPS = 1e-9

THRESHOLDS = {
    "gbp": {"clf": 0.6, "meta": 0.5},
    "xau": {"clf": 0.7, "meta": 0.5},
    "jpy": {"clf": 0.55, "meta": 0.5},
    "cad": {"clf": 0.55, "meta": 0.5},
    "_default_": {"clf": 0.55, "meta": 0.6},
}

spread_limits_low = {"xau":3.00, "jpy": 0.050, "gbp": 0.00050, "cad": 0.00050}
spread_limits_high = {"xau": 9.00, "jpy": 0.350, "gbp": 0.00350, "cad": 0.00350}

# === Load Data ===
df = pd.read_csv(DATA_PATH)
df['pair'] = pd.Categorical(df['pair'], categories=['xau', 'gbp', 'jpy', 'cad'])

# === Load Models ===
def load_model(path):
    with open(path, 'r') as f:
        return lgb.Booster(model_str=f.read())

reg = load_model('./model2/lgb_regression_buy_v3.txt')
clf = load_model('./model2/lgb_classifier_buy_v2.txt')
meta = load_model('./model2/lgb_meta_buy_v1.txt')

# reg = load_model('./model2/lgb_regression_sell_v3.txt')
# clf = load_model('./model2/lgb_classifier_sell_v2.txt')
# meta = load_model('./model2/lgb_meta_sell_v1.txt')

# === Feature Setup ===
features = [c for c in df.columns if c not in ["rr_label", "rr_class", "time_to_outcome"]]

df['reg_pred'] = reg.predict(df[features])
df['clf_prob'] = clf.predict(df[features])

results = []
filtered_in = 0

for i in range(3, len(df) - LOOKAHEAD):
    row = df.iloc[i]
    pair_code = row['pair']
    entry_price = row['entry_price']
    sl_price = row['stop_loss_price']
    vh = row['valid_hour']

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        continue

    thresholds = THRESHOLDS.get(pair_code, THRESHOLDS['_default_'])
    clf_threshold = thresholds['clf']
    meta_threshold = thresholds['meta']

    # Classifier filter
    if row['clf_prob'] < clf_threshold:
        continue

    # Meta-model input
    meta_input = row[features].copy()
    meta_input = pd.DataFrame([meta_input])  # Ensure 2D format

    # Handle categorical
    if 'pair' in meta_input.columns:
        meta_input['pair'] = pd.Categorical(meta_input['pair'], categories=['xau', 'gbp', 'jpy', 'cad'])

    # Add additional features for meta-model
    meta_input['reg_pred'] = row['reg_pred']
    meta_input['clf_prob'] = row['clf_prob']

    # Align with model input
    expected_cols = meta.feature_name()
    meta_input = meta_input.reindex(columns=expected_cols, fill_value=0)

    meta_pred = meta.predict(meta_input)[0]

    if meta_pred < meta_threshold:
        continue

    if not vh or sl_price >= entry_price or sl_price > df['Low'].iloc[i]:
        continue

    sl_dist = entry_price - sl_price
    if sl_dist < spread_limits_low[pair_code] or sl_dist > spread_limits_high[pair_code]:
        continue

    if row['reg_pred'] < RR_MARGIN:
        continue

    tp_price = entry_price + RR_TARGET * sl_dist
    rr_achieved = 0
    outcome = 'breakeven'
    time_to_outcome = LOOKAHEAD

    for j in range(1, LOOKAHEAD + 1):
        future = df.iloc[i + j]
        if future['Low'] <= sl_price:
            outcome = 'loss'
            rr_achieved = -1
            time_to_outcome = j
            break
        if future['High'] >= tp_price:
            outcome = 'win'
            rr_achieved = RR_TARGET
            time_to_outcome = j
            break

    if outcome == 'breakeven':
        final_close = df['Close'].iloc[i + LOOKAHEAD]
        if final_close > entry_price:
            outcome = 'breakeven_win'
            rr_achieved = 0.0
        else:
            outcome = 'breakeven_loss'
            rr_achieved = -0.5

    results.append({
        'index': i,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'reg_pred': row['reg_pred'],
        'actual_rr': rr_achieved,
        'outcome': outcome,
        'time_to_outcome': time_to_outcome,
        'clf_prob': row['clf_prob'],
        'meta_conf': meta_pred,
        'pair': pair_code
    })



# # # # === Sell Backtest Loop ===
# results = []
# filtered_in_sell = 0

# for i in range(3, len(df) - LOOKAHEAD):
#     row = df.iloc[i]
#     pair_code = row['pair']
#     entry_price = row['entry_price']
#     sl_price = row['stop_loss_price']
#     vh = row['valid_hour']

#     if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
#         continue

#     thresholds = THRESHOLDS.get(pair_code, THRESHOLDS['_default_'])
#     clf_threshold = thresholds['clf']
#     meta_threshold = thresholds['meta']

#     # Classifier filter
#     if row['clf_prob'] < clf_threshold:
#         continue

#     # Meta-model input
#     meta_input = row[features].copy()
#     meta_input = pd.DataFrame([meta_input])  # Ensure 2D format

#     # Handle categorical
#     if 'pair' in meta_input.columns:
#         meta_input['pair'] = pd.Categorical(meta_input['pair'], categories=['xau', 'gbp', 'jpy', 'cad'])

#     # Add additional features for meta-model
#     meta_input['reg_pred'] = row['reg_pred']
#     meta_input['clf_prob'] = row['clf_prob']

#     # Align with model input
#     expected_cols = meta.feature_name()
#     meta_input = meta_input.reindex(columns=expected_cols, fill_value=0)

#     meta_pred = meta.predict(meta_input)[0]

#     if meta_pred < meta_threshold:
#         continue

#     # === SELL-SPECIFIC CHECKS (INVERTED FROM BUY) ===
#     if not vh or sl_price <= entry_price or sl_price < df['High'].iloc[i]:
#         continue

#     sl_dist = sl_price - entry_price  # Distance is positive for sells
#     if sl_dist < spread_limits_low[pair_code] or sl_dist > spread_limits_high[pair_code]:
#         continue

#     if row['reg_pred'] < RR_MARGIN:  # Same RR check as buy
#         continue

#     tp_price = entry_price - RR_TARGET * sl_dist  # TP below entry for sells
#     rr_achieved = 0
#     outcome = 'breakeven'
#     time_to_outcome = LOOKAHEAD

#     # === INVERTED PRICE CHECKS FOR SELLS ===
#     for j in range(1, LOOKAHEAD + 1):
#         future = df.iloc[i + j]
#         if future['High'] >= sl_price:  # SL hit when price rises
#             outcome = 'loss'
#             rr_achieved = -1
#             time_to_outcome = j
#             break
#         if future['Low'] <= tp_price:  # TP hit when price falls
#             outcome = 'win'
#             rr_achieved = RR_TARGET
#             time_to_outcome = j
#             break

#     if outcome == 'breakeven':
#         final_close = df['Close'].iloc[i + LOOKAHEAD]
#         if final_close < entry_price:  # Price ended below entry
#             outcome = 'breakeven_win'
#             rr_achieved = 0.0
#         else:
#             outcome = 'breakeven_loss'
#             rr_achieved = -0.5

#     results.append({
#         'index': i,
#         'entry_price': entry_price,
#         'sl_price': sl_price,
#         'tp_price': tp_price,
#         'reg_pred': row['reg_pred'],
#         'actual_rr': rr_achieved,
#         'outcome': outcome,
#         'time_to_outcome': time_to_outcome,
#         'clf_prob': row['clf_prob'],
#         'meta_conf': meta_pred,
#         'pair': pair_code,
#         'side': 'sell'  # Added to distinguish from buys
#     })


print(f"✅ Backtest complete. Trades run: {len(results)}")
pd.DataFrame(results).to_csv("backtest_results.csv", index=False)



# === Save & Analyze Results ===
bt = pd.DataFrame(results)
bt.to_csv('bt_results.csv', index=False)

n_total = len(bt)
n_win = len(bt[bt['actual_rr'] >= RR_TARGET])
n_loss = len(bt[bt['actual_rr'] == -1])
n_be = n_total - n_win - n_loss

r_multiple = bt['actual_rr'].mean()
win_rate = n_win / (n_win + n_loss + EPS)
loss_rate = n_loss / (n_win + n_loss + EPS)
pf = bt[bt['actual_rr'] > 0]['actual_rr'].sum() / (-bt[bt['actual_rr'] < 0]['actual_rr'].sum() + EPS)
std = bt['actual_rr'].std()
sharpe = r_multiple / (std + EPS)
expectancy = (win_rate * RR_TARGET) + (loss_rate * -1)

# === Print Stats ===
print("📊 BACKTEST STATS")
print(f"Total Trades:     {n_total}")
print(f"Wins:             {n_win}")
print(f"Losses:           {n_loss}")
print(f"Breakevens:       {n_be}")
print(f"Win Rate:         {win_rate:.2%}")
print(f"R-Multiple:       {r_multiple:.3f}")
print(f"Profit Factor:    {pf:.3f}")
print(f"Expectancy:       {expectancy:.3f}")
print(f"Sharpe Ratio:     {sharpe:.3f}")



#Load your data (assuming it's already in a DataFrame named df)
df = pd.read_csv("bt_results.csv")

# Replace actual_rr of -1.0 with -1 and 2.0 with 2 for simplicity
df['pnl'] = df['actual_rr'].replace(-1.0, -1).replace(2.0, 2)

# Calculate equity curve (starting from 100)
df['equity'] = 100 + df['pnl'].cumsum()

# Calculate running maximum of equity
df['peak_equity'] = df['equity'].cummax()

# Calculate drawdown as percentage
df['drawdown_pct'] = (df['equity'] - df['peak_equity']) / df['peak_equity'] * 100

# Calculate maximum drawdown
max_drawdown = df['drawdown_pct'].min()

# Show final result
print("Maximum Drawdown (%):", round(max_drawdown, 2))

import matplotlib.pyplot as plt
from io import StringIO

# === Load your CSV Data ===
df = pd.read_csv("bt_results.csv")
df.columns = df.columns.str.strip()


# === Simulate Compounded Balance ===
initial_balance = 100.0
balance = [initial_balance]

for rr in df['actual_rr']:
    current = balance[-1]
    if rr == -1.0:
        new_balance = current - current * 0.02  # 1% loss
    else:
        new_balance = current + current * 0.04  # 4% gain
    balance.append(new_balance)

df['balance'] = balance[1:]
df['peak'] = df['balance'].cummax()
df['drawdown_pct'] = (df['balance'] - df['peak']) / df['peak'] * 100
max_drawdown = df['drawdown_pct'].min()

#=== Print Result Table ===
print(df[['index', 'balance', 'peak', 'drawdown_pct']])

# #=== Plot the Equity Curve ===
# balance_arr = df['balance'].to_numpy(dtype=np.float64)
# peak_arr = df['peak'].to_numpy(dtype=np.float64)
# drawdown_where = balance_arr < peak_arr

# plt.figure(figsize=(4, 3))
# plt.plot(balance_arr, label='Equity (Compounded)', linewidth=2)
# plt.plot(peak_arr, label='Equity Peak', linestyle='--')
# plt.fill_between(range(len(balance_arr)), balance_arr, peak_arr, where=drawdown_where, color='red', alpha=0.3, label='Drawdown')
# plt.title(f"Equity Curve with Drawdown (Max Drawdown: {max_drawdown:.2f}%)")
# plt.xlabel("Trade Number")
# plt.ylabel("Equity ($)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
