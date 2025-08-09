import numpy as np
import requests
import MetaTrader5 as mt5
import pandas as pd
import ta
import time
import json
import threading
import datetime
import os
import warnings
import logging
import sys
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import timezone
import lightgbm as lgb
import psutil
from multiprocessing import cpu_count


# Fix UnicodeEncodeError on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# === Configure Logging ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("trade_management.log", encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)



risk_percent = 2  # Risk percentage for lot size calculation

trading_enabled = True 

last_summary_sent_time = 0  # global or somewhere accessible
SUMMARY_SEND_INTERVAL = 5 * 60  # 5 minutes in seconds
last_trade_summary_time = 0
TRADE_SUMMARY_INTERVAL = 5 * 60  # 5 minutes in seconds

# Telegram Bot Details
TELEGRAM_BOT_TOKEN = "8156645817:AAH_KHYsM_9OZ6Q7Uj55cJsyA6gZKybCp1s"
CHAT_ID = "7050439107"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"



# MT5 Login Details)
login = 6320145
password = "c!*f$B4WYoyLW?!"
server = "Bybit-Demo"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# #T5 Login Details
# login = 6320176
# password = "chosenONE1986@"
# server = "Bybit-Live"
# MT5_PATH = r"C:\Program Files\MetaTrader 52\terminal64.exe"

# === Load models  ===
import lightgbm as lgb


def load_model(model_path, model_name=""):
    """Model loader with validation"""
    try:
        with open(model_path, 'r') as f:
            model = lgb.Booster(model_str=f.read())
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid LightGBM model")
        print(f"✅ Loaded {model_name}: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_name} ({model_path}): {type(e).__name__} - {str(e)}")
        return None

# ====== Model Registry ======
MODELS = {
    # BUY side
    "clf_buy_1.1": "./model_artifacts_buy/classifier_1_1.txt",
    "clf_buy_1.2": "./model_artifacts_buy/classifier_1_2.txt",
    # "clf_buy_1.3": "./model_artifacts_buy_1_1/classifier_1_3.txt",
    "reg_buy": "./model_artifacts_buy/regressor.txt",
    "meta_buy": "./model_artifacts_buy/meta_model.txt",

    # SELL side
    "clf_sell_1.1": "./model_artifacts_sell/classifier_1_1.txt",
    "clf_sell_1.2": "./model_artifacts_sell/classifier_1_2.txt",
    # "clf_sell_1.3": "./model_artifacts_sell_1_1/classifier_1_3.txt",
    "reg_sell": "./model_artifacts_sell/regressor.txt",
    "meta_sell": "./model_artifacts_sell/meta_model.txt"
}

# ====== Load All ======
loaded_models = {}
for name, path in MODELS.items():
    loaded_models[name] = load_model(path, model_name=name)

# ====== Verify ======
missing = [name for name, model in loaded_models.items() if model is None]
if missing:
    raise RuntimeError("❌ Missing models:\n" + "\n".join(f"- {name}: {MODELS[name]}" for name in missing))
else:
    print("🔥 All models loaded successfully:")
    print("\n".join(f"- {name:14s} → {path}" for name, path in MODELS.items()))


feature_list = [
"Open", "High", "Low", "Close", "Volume", "hour_sin", "hour_cos", "valid_hour", "SMA_9", "SMA_9_pct_distance",
"SMA_9_slope_diff5", "SMA_9_slope_diff", "SMA_9_slope_grad", "SMA_9_direction", "SMA_9_cross", "SMA_9_percentile", "SMA_9_up_ratio_5", "SMA_20", "SMA_20_pct_distance", "SMA_20_slope_diff5",
"SMA_20_slope_diff", "SMA_20_slope_grad", "SMA_20_direction", "SMA_20_cross", "SMA_20_percentile", "SMA_20_up_ratio_5", "SMA_50", "SMA_50_pct_distance", "SMA_50_slope_diff5", "SMA_50_slope_diff",
"SMA_50_slope_grad", "SMA_50_direction", "SMA_50_cross", "SMA_50_percentile", "SMA_50_up_ratio_5", "SMA_100", "SMA_100_pct_distance", "SMA_100_slope_diff5", "SMA_100_slope_diff", "SMA_100_slope_grad",
"SMA_100_direction", "SMA_100_cross", "SMA_100_percentile", "SMA_100_up_ratio_5", "SMA_200", "SMA_200_pct_distance", "SMA_200_slope_diff5", "SMA_200_slope_diff", "SMA_200_slope_grad", "SMA_200_direction",
"SMA_200_cross", "SMA_200_percentile", "SMA_200_up_ratio_5", "RSI", "rsi_change", "rsi_slope", "rsi_slope_std_5", "rsi_slope_std_10", "rsi_rolling_mean", "rsi_std",
"rsi_zscore", "rsi_above_65_duration", "rsi_below_45_duration", "CCI", "cci_change", "cci_slope", "cci_slope_std_5", "cci_slope_std_10", "cci_rolling_mean", "cci_std",
"cci_zscore", "cci_above_100_duration", "cci_below_minus100_duration", "cci_to_100", "MACD_Line", "MACD_Signal", "MACD_Histogram", "macd_line_pct_change", "macd_hist_pct_change", "macd_line_zscore",
"macd_hist_zscore", "macd_cross_signal", "macd_hist_direction", "macd_hist_slope", "macd_hist_slope_std_10", "macd_hist_slope_std_5", "MACD_Line_Slope", "MACD_Line_Slope_std_10", "MACD_Line_Slope_std_5", "adx",
"adx_slope", "adx_slope_std_10", "adx_slope_std_5", "OBV", "obv_change", "obv_pct_change", "obv_slope", "pain_ratio", "gain_ratio", "feat_c1_lt_pre_s9",
"feat_c0_gt_s9", "feat_o1_gt_c1", "feat_o0_lt_c0", "feat_c1_lt_pre_s20", "feat_c0_gt_s20", "feat_cross_above_s9", "feat_cross_above_s20", "feat_cross_below_s9", "feat_cross_below_s20", "gk",
"pn", "MACD_Histogram1", "MACD_Signal1", "MACD_Line1", "MACD_Histogram2", "MACD_Signal2", "MACD_Line2", "MACD_Histogram3", "MACD_Signal3", "MACD_Line3",
"MACD_Histogram4", "MACD_Signal4", "MACD_Line4", "OC_ratio", "OC_ratio_1", "OC_ratio_2", "OC_ratio_3", "OC_ratio_4", "Close_Change_1", "RSI_Change_1",
"Close_Change_2", "RSI_Change_2", "Close_Change_3", "RSI_Change_3", "Close_Change_4", "RSI_Change_4", "volume_z", "volume_slope", "RSI_Slope_15min", "MACD_Slope_15min",
"RSI_Mean_15min", "MACD_Mean_15min", "RSI_Std_15min", "MACD_Std_15min", "RSI_Slope_1hr", "MACD_Slope_1hr", "RSI_Mean_1hr", "MACD_Mean_1hr", "RSI_Std_1hr", "MACD_Std_1hr",
"RSI_Slope_4hr", "MACD_Slope_4hr", "RSI_Mean_4hr", "MACD_Mean_4hr", "RSI_Std_4hr", "MACD_Std_4hr", "RSI_Slope_1day", "MACD_Slope_1day", "RSI_Mean_1day", "MACD_Mean_1day",
"RSI_Std_1day", "MACD_Std_1day", "BB_Lower", "BB_Upper", "BB_Mid", "BB_Width", "BB_Percent", "BB_Close_Dist_Mid", "BB_Close_Dist_Lower", "BB_Close_Dist_Upper",
"BB_Mid_Slope", "BB_Is_Squeeze", "BB_Expansion", "BB_Width_shift_1", "BB_Pct_Width_shift_1", "BB_Close_Dist_Mid_shift_1", "candle_body", "upper_wick", "lower_wick", "candle_range",
"momentum_unbalance", "wick_dominance", "range_spike", "price_surge", "PGI_alt", "BB_Width_shift_2", "BB_Pct_Width_shift_2", "BB_Close_Dist_Mid_shift_2", "BB_Width_shift_3", "BB_Pct_Width_shift_3",
"BB_Close_Dist_Mid_shift_3", "BB_Width_shift_4", "BB_Pct_Width_shift_4", "BB_Close_Dist_Mid_shift_4", "log_returns", "volatility_5", "skewness", "kurtosis", "volatility_20", "skewness_20",
"kurtosis_20", "volatility_10", "skewness_10", "kurtosis_10", "price_slope", "price_slope_std_20", "price_slope_std_1", "drawdown", "max_drawdown", "velocity",
"xtopher", "acceleration", "smoothed_velocity_5", "smoothed_acceleration_5", "cum_log_returns", "returns", "rolling_std", "god_oscillator", "range_5", "range_10",
"bullish_engulf", "bearish_engulf", "pin_bar", "inside_bar", "outside_bar", "cluster_10", "range_5_lag1", "range_5_lag2", "range_5_lag3", "range_5_lag4",
"range_5_lag5", "range_10_lag1", "range_10_lag2", "range_10_lag3", "range_10_lag4", "range_10_lag5", "bullish_engulf_lag1", "bullish_engulf_lag2", "bullish_engulf_lag3", "bullish_engulf_lag4",
"bullish_engulf_lag5", "bearish_engulf_lag1", "bearish_engulf_lag2", "bearish_engulf_lag3", "bearish_engulf_lag4", "bearish_engulf_lag5", "pin_bar_lag1", "pin_bar_lag2", "pin_bar_lag3", "pin_bar_lag4",
"pin_bar_lag5", "inside_bar_lag1", "inside_bar_lag2", "inside_bar_lag3", "inside_bar_lag4", "inside_bar_lag5", "outside_bar_lag1", "outside_bar_lag2", "outside_bar_lag3", "outside_bar_lag4",
"outside_bar_lag5", "cluster_10_lag1", "cluster_10_lag2", "cluster_10_lag3", "cluster_10_lag4", "cluster_10_lag5", "SMA_9_lt_SMA_20", "BoS_Up", "BoS_Down", "Double_Top",
"Double_Bottom", "Candles_Since_BB_Upper", "Candles_Since_BB_Lower", "entry_price", "stop_loss_price", "stop_loss_distance", "sl_ratio_to_entry", "side", "volatility", "pair",
]

BROKER_TIME_OFFSET_MINUTES = -120


# Function to send a Telegram message
def send_telegram_message(message):
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(TELEGRAM_URL, json=payload)

# Function to initialize MT5 
def initialize_mt5(login, password, server):
    """Initialize MT5 and login with provided credentials."""
    if not mt5.initialize():
        error_message = "❌ MT5 Initialization Failed"
        print(error_message)
        send_telegram_message(error_message)
        return False

    authorized = mt5.login(login, password=password, server=server)
    
    if not authorized:
        error_message = f"❌ MT5 Login Failed: {mt5.last_error()}"
        print(error_message)
        send_telegram_message(error_message)
        return False
    
    success_message = "✅ Successfully connected to MT5!"
    print(success_message)
    send_telegram_message(success_message)

    account_info = mt5.account_info()
    if account_info:
        balance_message = f"💰 Account Balance: {account_info.balance}\n🟢 Equity: {account_info.equity}"
        print(balance_message)
        send_telegram_message(balance_message)
        return True

    return False


# Function to fetch OHLCV  data FOR 5MIM TIMEFRAME from MT5
def fetch_candles(symbol, timeframe, num_candles):
    """Fetch OHLCV data from MT5"""
    if not mt5.terminal_info():
        print("❌ MT5 is not initialized!")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    
    if rates is None or len(rates) == 0:
        print("❌ Failed to fetch data from MT5:", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={"time": "Time", "open": "Open", "high": "High", "low": "Low", 
                       "close": "Close", "tick_volume": "Volume"}, inplace=True)
    
    df.sort_values("Time").reset_index(drop=True)
    
    
    # print("✅ Fetched Candles:\n", df)
    return df

def calculate_features(df, direction=1, pair_code=None):
  # Parse datetime column (adjust name if needed)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['hour'] = df['Time'].dt.hour

    df['hour_sin'] = np.sin(2*np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2*np.pi * df['hour'] / 24)

    # df['is_tokyo_session']    = ((df['hour'] >=  2) & (df['hour'] <  9)).astype(int)
    # df['is_london_session']   = ((df['hour'] >=  8) & (df['hour'] < 17)).astype(int)
    # df['is_new_york_session'] = ((df['hour'] >= 16) & (df['hour'] < 22)).astype(int)
    df['valid_hour'] = df['hour'].between(2, 20)
    df.drop(columns=['Time', 'hour','real_volume','spread'], inplace=True)

    # df.drop(columns=['Volume'], inplace=True)
    # Basic OHLC columns
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for w in [9, 20, 50, 100, 200]:
        sma_col = f'SMA_{w}'
        sma = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
        df[sma_col] = sma

        # 🔹 Price distance to SMA in %
        df[f'{sma_col}_pct_distance'] = ((df['Close'] - sma) / sma) * 100

        # 🔹 Slope (diff-based, over 5 periods)
        df[f'{sma_col}_slope_diff5'] = sma.diff(5)

        # 🔹 Slope (rate of change per period)
        df[f'{sma_col}_slope_diff'] = sma.diff() / sma.shift()

        # 🔹 Slope (smoothed gradient)
        df[f'{sma_col}_slope_grad'] = np.gradient(sma)

        # 🔹 Direction (+1 up, -1 down, 0 flat)
        df[f'{sma_col}_direction'] = np.sign(df[f'{sma_col}_slope_diff'])

        # 🔹 SMA cross signals (price crosses SMA)
        df[f'{sma_col}_cross'] = np.where(
            (df['Close'] > sma) & (df['Close'].shift(1) <= sma.shift(1)), 1,
            np.where((df['Close'] < sma) & (df['Close'].shift(1) >= sma.shift(1)), -1, 0)
        )

        # 🔹 Percentile rank of SMA in 5-bar rolling window
        df[f'{sma_col}_percentile'] = sma.rolling(window=5).apply(
            lambda x: (np.sum(x < x[-1]) / len(x)) * 100, raw=True
        )

        # 🔹 Upward movement ratio over last 5 bars
        df[f'{sma_col}_up_ratio_5'] = sma.diff().rolling(5).apply(
            lambda x: np.sum(x > 0) / 5, raw=True
        )
    def zone_duration(series):
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).cumcount()

    def add_rsi_features(df, rsi_window=14, div_shift=9):


        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_window).rsi()
        df['RSI'] = rsi.fillna(0)

        # Crossovers
        df['rsi_above_65'] = (df['RSI'] > 65).astype(int)
        df['rsi_below_45'] = (df['RSI'] < 45).astype(int)

        # Fix: use parentheses for each condition to avoid TypeError

        # Trend & Volatility
        df['rsi_change'] = df['RSI'] - df['RSI'].shift(1).fillna(0)
        df['rsi_slope'] = np.gradient(df['RSI'])
        df['rsi_slope_std_5'] = df['rsi_slope'].rolling(window=5).std().fillna(0)
        df['rsi_slope_std_10'] = df['rsi_slope'].rolling(window=10).std().fillna(0)
        df['rsi_rolling_mean'] = df['RSI'].rolling(5).mean().fillna(df['RSI'])
        df['rsi_std'] = df['RSI'].rolling(5).std().fillna(0)
        df['rsi_zscore'] = (df['RSI'] - df['RSI'].rolling(20).mean()) / (df['RSI'].rolling(20).std() + 1e-6)
        df['rsi_zscore'] = df['rsi_zscore'].fillna(0)

        # OB/OS durations
        df['rsi_above_65_duration'] = zone_duration(df['rsi_above_65'])
        df['rsi_below_45_duration'] = zone_duration(df['rsi_below_45'])

        df.drop(columns=['rsi_above_65', 'rsi_below_45'], inplace=True)

        return df



    def add_cci_features(df, cci_window=14, div_shift=9):
        cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=cci_window).cci()
        df['CCI'] = cci.fillna(0)

        # Crossovers
        df['cci_above_100'] = (df['CCI'] > 100).astype(int)
        df['cci_below_minus100'] = (df['CCI'] < -100).astype(int)

        # Trend
        df['cci_change'] = df['CCI'] - df['CCI'].shift(1).fillna(0)
        df['cci_slope'] = np.gradient(df['CCI'])
        df['cci_slope_std_5'] = df['cci_slope'].rolling(window=5).std().fillna(0)
        df['cci_slope_std_10'] = df['cci_slope'].rolling(window=10).std().fillna(0)
        df['cci_rolling_mean'] = df['CCI'].rolling(5).mean().fillna(df['CCI'])


        # Normalization
        df['cci_std'] = df['CCI'].rolling(5).std().fillna(0)
        df['cci_zscore'] = (df['CCI'] - df['CCI'].rolling(5).mean()) / (df['CCI'].rolling(5).std() + 1e-6)
        df['cci_zscore'] = df['cci_zscore'].fillna(0)


        # Duration in OB/OS zones
        df['cci_above_100_duration'] = zone_duration(df['cci_above_100'])
        df['cci_below_minus100_duration'] = zone_duration(df['cci_below_minus100'])

        # Distance from thresholds
        df['cci_to_100'] = (100 - df['CCI']).clip(lower=0)


        df.drop(columns=['cci_above_100', 'cci_below_minus100'], inplace=True)

        return df


    def add_macd_features(df, div_shift=9):
        macd = ta.trend.MACD(df['Close'])
        df['MACD_Line'] = macd.macd().fillna(0)
        df['MACD_Signal'] = macd.macd_signal().fillna(0)
        df['MACD_Histogram'] = macd.macd_diff().fillna(0)

        # Normalized


        df['macd_line_pct_change'] = df['MACD_Line'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['macd_hist_pct_change'] = df['MACD_Histogram'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['macd_line_zscore'] = (df['MACD_Line'] - df['MACD_Line'].rolling(5).mean()) / (df['MACD_Line'].rolling(5).std() + 1e-6)
        df['macd_hist_zscore'] = (df['MACD_Histogram'] - df['MACD_Histogram'].rolling(5).mean()) / (df['MACD_Histogram'].rolling(5).std() + 1e-6)
        df['macd_line_zscore'] = df['macd_line_zscore'].fillna(0)


        # Crossovers
        df['macd_cross_signal'] = ((df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)

        # Histogram direction
        df['macd_hist_direction'] = np.sign(df['MACD_Histogram'].diff().fillna(0)).astype(int)
        df['macd_hist_slope'] = np.gradient(df['MACD_Histogram']) 

        df['macd_hist_slope_std_10'] = df['macd_hist_slope'].rolling(10).std().fillna(0)

        df['macd_hist_slope_std_5'] = df['macd_hist_slope'].rolling(5).std().fillna(0) 
        df['MACD_Line_Slope'] = np.gradient(df['MACD_Line']) # new
        df['MACD_Line_Slope_std_10'] = df['MACD_Line_Slope'].rolling(10).std().fillna(0)
        df['MACD_Line_Slope_std_5'] = df['MACD_Line_Slope'].rolling(5).std().fillna(0)

        # ADX & its slope
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx().fillna(0)
        df['adx_slope'] = np.gradient(df['adx'])
        df['adx_slope_std_10'] = df['adx_slope'].rolling(10).std().fillna(0)
        df['adx_slope_std_5'] = df['adx_slope'].rolling(5).std().fillna(0)

        return df

    df = add_rsi_features(df, rsi_window=14, div_shift=9)
    df = add_cci_features(df, cci_window=20, div_shift=9)
    df = add_macd_features(df, div_shift=9)


    def add_volume_features(df):
        eps = 1e-6  # for division safety

        # === On-Balance Volume (OBV) ===
        obv = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        df['OBV'] = obv.fillna(0)

        # OBV Engineering
        df['obv_change'] = df['OBV'].diff().fillna(0)
        df['obv_pct_change'] = df['OBV'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['obv_slope'] = np.gradient(df['OBV'].fillna(0))

        return df
    df = add_volume_features(df)


    # === 15) Pain‐to‐Gain Ratios ===
    df['pain_ratio'] = df['Low'].rolling(5).min().pct_change() / (df['High'].rolling(5).max().pct_change() + 1e-6)
    df['gain_ratio'] = df['High'].rolling(5).max().pct_change() / (df['Low'].rolling(5).min().pct_change() + 1e-6)


    # === 18) Ichimoku‐Derived Features (unchanged) ===
    # def compute_ichimoku_features_from_5m(df: pd.DataFrame) -> pd.DataFrame:
    #     timeframes = {'15min': 3, '1h': 12, '4h': 48, '1d': 288}
    #     ichimoku = ta.trend.IchimokuIndicator(
    #         high=df['High'], low=df['Low'], window1=9, window2=26, window3=52
    #     )
    #     df['Tenkan']   = ichimoku.ichimoku_conversion_line()
    #     df['Kijun']    = ichimoku.ichimoku_base_line()
    #     df['Senkou_A'] = ichimoku.ichimoku_a()
    #     df['Senkou_B'] = ichimoku.ichimoku_b()
    #     df['tenkan_kijun_delta'] = df['Tenkan'] - df['Kijun']

    #     for label, window in timeframes.items():
    #         df[f'{label}_tk_delta_slope']       = (df['tenkan_kijun_delta'] - df['tenkan_kijun_delta'].shift(window)) / window
    #         df[f'{label}_tk_delta_mean']        = df['tenkan_kijun_delta'].rolling(window).mean()
    #         df[f'{label}_tk_delta_std']         = df['tenkan_kijun_delta'].rolling(window).std()
    #         df[f'{label}_tk_delta_pct_rank']    = df['tenkan_kijun_delta'].rank(pct=True).rolling(window).apply(lambda x: x[-1], raw=True)

    #     return df
    # df = compute_ichimoku_features_from_5m(df)

    df['feat_c1_lt_pre_s9']   = (df['Close'].shift(1) < df['SMA_9'].shift(1)).astype(int)
    df['feat_c0_gt_s9']       = (df['Close'] > df['SMA_9']).astype(int)
    df['feat_o1_gt_c1']       = (df['Open'].shift(1) > df['Close'].shift(1)).astype(int)
    df['feat_o0_lt_c0']       = (df['Open'] < df['Close']).astype(int)

    df['feat_c1_lt_pre_s20']  = (df['Close'].shift(1) < df['SMA_20'].shift(1)).astype(int)
    df['feat_c0_gt_s20']      = (df['Close'] > df['SMA_20']).astype(int)

    # Optional: also include cross-related features
    df['feat_cross_above_s9']  = ((df['Close'].shift(1) < df['SMA_9'].shift(1)) & (df['Close'] > df['SMA_9'])).astype(int)
    df['feat_cross_above_s20'] = ((df['Close'].shift(1) < df['SMA_20'].shift(1)) & (df['Close'] > df['SMA_20'])).astype(int)

    df['feat_cross_below_s9']  = ((df['Close'].shift(1) > df['SMA_9'].shift(1)) & (df['Close'] < df['SMA_9'])).astype(int)
    df['feat_cross_below_s20'] = ((df['Close'].shift(1) > df['SMA_20'].shift(1)) & (df['Close'] < df['SMA_20'])).astype(int)

    df['gk']      = (df['gain_ratio'].shift(1) > df['gain_ratio']).astype(int)
    df['pn']      = (df['pain_ratio'].shift(1) > df['gain_ratio']).astype(int)



    for i in range(1, 5):
        for name in ['MACD_Histogram', 'MACD_Signal', 'MACD_Line']:
            df[f'{name}{i}'] = df[name].shift(i)

    df['OC_ratio'] = (df['Close'] - df['Open']) / df['Open']
    for i in range(1, 5):
        df[f'OC_ratio_{i}'] = df['OC_ratio'].shift(i)

    for i in range(1, 5):
        df[f'Close_Change_{i}'] = df['Close'].pct_change(i)
        df[f'RSI_Change_{i}'] = df['RSI'].diff(i)


    df['volume_z'] = (df['Volume'] - df['Volume'].rolling(5).mean()) / df['Volume'].rolling(5).std()
    df['volume_slope'] = np.gradient(df['Volume'].fillna(0))


    timeframes = {'15min': 3, '1hr': 12, '4hr': 48, '1day': 288}
    def calculate_slope(series, lag):
        return (series - series.shift(lag)) / lag

    for name, period in timeframes.items():
        df[f'RSI_Slope_{name}'] = calculate_slope(df['RSI'], period)
        df[f'MACD_Slope_{name}'] = calculate_slope(df['MACD_Line'], period)
        df[f'RSI_Mean_{name}'] = df['RSI'].rolling(period).mean()
        df[f'MACD_Mean_{name}'] = df['MACD_Line'].rolling(period).mean()
        df[f'RSI_Std_{name}'] = df['RSI'].rolling(period).std()
        df[f'MACD_Std_{name}'] = df['MACD_Line'].rolling(period).std()

    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2.0)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Mid']   = bb.bollinger_mavg()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    df['BB_Close_Dist_Mid'] = df['Close'] - df['BB_Mid']
    df['BB_Close_Dist_Lower'] = df['Close'] - df['BB_Lower']
    df['BB_Close_Dist_Upper'] = df['BB_Upper'] - df['Close']
    df['BB_Mid_Slope'] = df['BB_Mid'].diff()
    df['BB_Is_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.35)
    df['BB_Expansion'] = df['BB_Width'].pct_change()

    for i in range(1, 5):
        df[f'BB_Width_shift_{i}'] = df['BB_Width'].shift(i)
        df[f'BB_Pct_Width_shift_{i}'] = df[f'BB_Width_shift_{i}'] / df['BB_Mid'].shift(i)
        df[f'BB_Close_Dist_Mid_shift_{i}'] = df['Close'].shift(i) - df['BB_Mid'].shift(i)

        EPS = 1e-9

        df['candle_body'] = (df['Close'] - df['Open']).abs()
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['candle_range'] = df['High'] - df['Low']

        df['bull_count'] = (df['Close'] > df['Open']).rolling(5).sum()
        df['bear_count'] = (df['Close'] < df['Open']).rolling(5).sum()
        df['momentum_unbalance'] = df['bull_count'] - df['bear_count']
        df.drop(columns=['bull_count', 'bear_count'], inplace=True)

        df['mean_close_10']  = df['Close'].rolling(10).mean()
        df['wick_dominance'] = (df['upper_wick'] + df['lower_wick']) / (df['candle_body'] + EPS)
        df['range_spike'] = df['candle_range'] / df['candle_range'].rolling(20).mean()
        df['price_surge'] = (df['Close'] - df['mean_close_10']) / df['candle_range'].rolling(10).mean()
        
        df.drop(columns=['mean_close_10'], inplace=True)

        df['PGI_alt'] = 2.0 * df['momentum_unbalance'].fillna(0) + 1.5 * df['range_spike'].fillna(0) + 1.2 * df['wick_dominance'].fillna(0) + 2.5 * df['price_surge'].fillna(0)

    # ────────────── STATISTICAL FEATURE ENGINE ──────────────
    # Price Distribution
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility_5'] = df['log_returns'].rolling(5).std()
    df['skewness'] = df['log_returns'].rolling(5).skew()
    df['kurtosis'] = df['log_returns'].rolling(5).kurt()
    df['volatility_20'] = df['log_returns'].rolling(20).std()
    df['skewness_20'] = df['log_returns'].rolling(20).skew()
    df['kurtosis_20'] = df['log_returns'].rolling(20).kurt()
    df['volatility_10'] = df['log_returns'].rolling(10).std()
    df['skewness_10'] = df['log_returns'].rolling(10).skew()
    df['kurtosis_10'] = df['log_returns'].rolling(10).kurt()


    df['price_slope'] = np.gradient(df['log_returns'])
    df['price_slope_std_20'] = df['price_slope'].rolling(20).std().fillna(0)
    df['price_slope_std_1'] = df['price_slope'].rolling(10).std().fillna(0)

    roll_max = df['Close'].rolling(5).max()
    df['drawdown'] = df['Close'] / roll_max - 1.0
    df['max_drawdown'] = df['drawdown'].rolling(5).min()


    # Fast time-domain motion
    df['velocity'] = df['Close'].diff()
    df['xtopher'] = (df['Close'].diff() + df['rsi_slope'] + df['obv_slope']) / df['adx_slope_std_10']
    df['acceleration'] = df['velocity'].diff()
    df['smoothed_velocity_5'] = df['velocity'].rolling(5).mean()
    df['smoothed_acceleration_5'] = df['acceleration'].rolling(5).mean()
    df['cum_log_returns'] = df['log_returns'].cumsum()


    df['returns'] = df['Close'].pct_change()
    df['rolling_std'] = df['returns'].rolling(5).std()
    df['god_oscillator'] = (
        0.6 * df['rsi_slope'] * 10 +
        0.4 * (df['obv_slope'] / (df['rolling_std'] + 1e-6))
    )

    def add_candlestick_features(df):
        # === 1. Range-bound Detection ===
        def is_range_bound(window=5, threshold=0.015):
            max_high = df['High'].rolling(window).max()
            min_low = df['Low'].rolling(window).min()
            range_pct = (max_high - min_low) / df['Close']
            return (range_pct < threshold).astype(int)

        df['range_5'] = is_range_bound(14)
        df['range_10'] = is_range_bound(48)

        # === 2. Engulfing Patterns ===
        df['bullish_engulf'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open']) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
        ).astype(int)

        df['bearish_engulf'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
        ).astype(int)

        # === 3. Pin Bar Detection ===
        body = abs(df['Close'] - df['Open'])
        range_ = df['High'] - df['Low']
        upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['pin_bar'] = ((body / range_ < 0.3) & ((upper_wick > 2 * body) | (lower_wick > 2 * body))).astype(int)

        # === 4. Inside/Outside Bars ===
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))).astype(int)

        # === 5. Clustering / Candle Compression ===
        def cluster_score(window=10, threshold=0.01):
            max_high = df['High'].rolling(window).max()
            min_low = df['Low'].rolling(window).min()
            spread = (max_high - min_low) / df['Close']
            return (spread < threshold).astype(int)

        df['cluster_10'] = cluster_score(14)

        # === Add memory with lag features (shifts) ===
        features_to_shift = ['range_5', 'range_10', 'bullish_engulf', 'bearish_engulf',
                    'pin_bar', 'inside_bar', 'outside_bar', 'cluster_10']

        for f in features_to_shift:
            for lag in range(1, 6):  # shift 1 to 5
                df[f'{f}_lag{lag}'] = df[f].shift(lag).fillna(0).astype(int)

        return df
    df = add_candlestick_features(df)

    def detect_bos(df, lookback=22):
        bos_up = []
        bos_down = []

        for i in range(len(df)):
            if i < lookback:
                bos_up.append(0)
                bos_down.append(0)
                continue

            recent_highs = df['High'].iloc[i-lookback:i]
            recent_lows = df['Low'].iloc[i-lookback:i]
            bos_up.append(int(df['High'].iloc[i] > recent_highs.max()))
            bos_down.append(int(df['Low'].iloc[i] < recent_lows.min()))

        df['BoS_Up'] = bos_up
        df['BoS_Down'] = bos_down

    df['SMA_9_lt_SMA_20'] = (df['SMA_9'] < df['SMA_20']).astype(int)
    from scipy.signal import find_peaks

    def detect_double_top_bottom(df, distance=22, threshold=0.002):
        highs = df['High'].values
        lows = df['Low'].values

        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)

        dt_mask = np.zeros(len(df))
        db_mask = np.zeros(len(df))

        for i in range(1, len(peaks)):
            if abs(highs[peaks[i]] - highs[peaks[i - 1]]) / highs[peaks[i]] < threshold:
                dt_mask[peaks[i]] = 1

        for i in range(1, len(troughs)):
            if abs(lows[troughs[i]] - lows[troughs[i - 1]]) / lows[troughs[i]] < threshold:
                db_mask[troughs[i]] = 1

        df['Double_Top'] = dt_mask.astype(int)
        df['Double_Bottom'] = db_mask.astype(int)


    def candles_since_bollinger_touch(df):
        last_touch_upper = np.full(len(df), np.nan)
        last_touch_lower = np.full(len(df), np.nan)

        last_upper = -1
        last_lower = -1

        for i in range(len(df)):
            if df['High'].iloc[i] > df['BB_Upper'].iloc[i]:
                last_upper = i
            if df['Low'].iloc[i] < df['BB_Lower'].iloc[i]:
                last_lower = i

            last_touch_upper[i] = i - last_upper if last_upper != -1 else np.nan
            last_touch_lower[i] = i - last_lower if last_lower != -1 else np.nan

        df['Candles_Since_BB_Upper'] = last_touch_upper
        df['Candles_Since_BB_Lower'] = last_touch_lower


    detect_bos(df, lookback=22)
    detect_double_top_bottom(df)
    candles_since_bollinger_touch(df)

   # ─── NEW: compute the 5 trade‐cols based on direction ───
    vol_window = 50
    df['volatility'] = df['log_returns'].rolling(vol_window).std()

    EPS = 1e-9
    
    if direction == 1:
        # Long setup - maintain original order
        df['entry_price'] = df['High']
        df['stop_loss_price'] = (df['entry_price'] - 6 * df['volatility'])
        # Drop volatility here exactly as in original
        df = df.drop(columns=['volatility'], errors='ignore')  # Safe drop
        df['stop_loss_distance'] = (df['entry_price'] - df['stop_loss_price'])
        df['sl_ratio_to_entry'] = df['stop_loss_distance'] / (df['entry_price'] + EPS)
        df['side'] = 1
    else:
        # Short setup - maintain original order
        df['entry_price'] = df['Low']
        df['stop_loss_price'] = (df['entry_price'] + 6 * df['volatility'])
        # Drop volatility here exactly as in original
        df = df.drop(columns=['volatility'], errors='ignore')  # Safe drop
        df['stop_loss_distance'] = (df['stop_loss_price'] - df['entry_price'])
        df['sl_ratio_to_entry'] = df['stop_loss_distance'] / (df['entry_price'] + EPS)
        df['side'] = 0
    # distances and ratios

    # distances and ratios
  
    vol_window = 50
    df['volatility'] = df['log_returns'].rolling(vol_window).std()
   
    if pair_code:
        df['pair'] = pair_code

    # ─── keep your existing slicing & clean‐up ───
    df = df.iloc[300:].reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    
    return df




def has_open_trade(symbol, trade_type):
    """
    Check if a trade is already open for the symbol, and if trade_type ('buy' or 'sell')
    has reached the max allowed (4). If more than 4 of a type are open, return True and skip new trade.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False

    # Count trade types
    buy_count = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_BUY)
    sell_count = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_SELL)

    if trade_type == "buy" and buy_count >= 4:
        print(f"🚫 Max BUY trades (4) already open for {symbol}. Skipping.")
        send_telegram_message(f"🚫 Max BUY trades (4) already open for {symbol}. Skipping.")
        return True
    elif trade_type == "sell" and sell_count >= 4:
        print(f"🚫 Max SELL trades (4) already open for {symbol}. Skipping.")
        send_telegram_message(f"🚫 Max SELL trades (4) already open for {symbol}. Skipping.")
        return True

    return False




def calculate_lot_size(account_balance, sl_pips, symbol, risk_percent=2):
    """Calculate optimal lot size per symbol based on risk, pip value, and SL size."""
    if sl_pips <= 0:
        print(f"❌ Invalid SL: {sl_pips}")
        return 0.01

    # Determine pip value based on symbol category
    if "XAU" in symbol:
        pip_value = 100.0       # $1 per lot per $1 move for XAU
    elif "JPY" in symbol:
        pip_value = 1000.0    # $10 per pip (0.01) per 1 lot
    elif "USD" in symbol and "JPY" not in symbol:
        pip_value = 100000.0  # $10 per pip (0.0001) per 1 lot for EURUSD, GBPUSD etc.
    else:
        pip_value = 100000.0  # default fallback

    # Get broker constraints
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"⚠️ Failed to get symbol info for {symbol}.")
        return 0.01

    # Risk-based calculation
    risk_amount = (risk_percent / 100.0) * account_balance
    lot_size = risk_amount / (sl_pips * pip_value)

    # Enforce min lot and rounding
    lot_size = max(round(lot_size, 2), symbol_info.volume_min)
    if lot_size  <= 0.01:
        lot_size = 0.01
    print(f"🧮 {symbol} | Balance: {account_balance}, SL: {sl_pips}, Lot: {lot_size}")
    send_telegram_message(f"🧮 {symbol} | Calculated Lot Size: {lot_size} with SL {sl_pips}")
    return lot_size




symbol_locks = {}  # Global lock dictionary

def is_symbol_locked(symbol):
    return symbol_locks.get(symbol, False)

def lock_symbol(symbol, duration=1):
    symbol_locks[symbol] = True
    threading.Timer(duration, lambda: unlock_symbol(symbol)).start()

def unlock_symbol(symbol):
    symbol_locks[symbol] = False
    send_telegram_message(f"🔓 Trading unlocked for {symbol}")

spread_limits = {
    "XAU": 0.15, "JPY": 0.015,
    "GBP": 0.00010, "EUR": 0.00010,
    "CAD": 0.00010
}

def get_spread_limit(sym):
    return next((v for k, v in spread_limits.items() if k in sym), 0.00030)
   

def handle_trigger_or_watch(symbol, trade_type, trigger_price, sl_price, tp_price,
                            regress_pred, classifier_conf, meta_conf, funx, meta_class):
    if is_symbol_locked(symbol) or has_open_trade(symbol, trade_type):
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    bid, ask = tick.bid, tick.ask
    spread = ask - bid
    if spread > get_spread_limit(symbol):
        send_telegram_message(f"❌ Spread too high for {symbol}.")
        return

    price = ask if trade_type == "buy" else bid
    if (trade_type == "buy" and price >= trigger_price) or (trade_type == "sell" and price <= trigger_price):
        place_market_order(symbol, trade_type, price, sl_price, tp_price,
                           regress_pred, classifier_conf, meta_conf, funx, meta_class)
    else:
        threading.Thread(
            target=watch_price_and_execute,
            args=(symbol, trade_type, trigger_price, sl_price, tp_price,
                  regress_pred, classifier_conf, meta_conf, funx, meta_class),
            daemon=True
        ).start()


def watch_price_and_execute(symbol, trade_type, trigger_price, sl_price, tp_price,
                            regress_pred, classifier_conf, meta_conf, funx, meta_class):
    timeout = 300  # 5 minutes
    interval = 1
    waited = 0

    while waited < timeout:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            time.sleep(interval)
            waited += interval
            continue

        bid, ask = tick.bid, tick.ask
        spread = ask - bid
        if spread > get_spread_limit(symbol):
            time.sleep(interval)
            waited += interval
            continue

        price = ask if trade_type == "buy" else bid
        if (trade_type == "buy" and price >= trigger_price) or (trade_type == "sell" and price <= trigger_price):
            place_market_order(symbol, trade_type, price, sl_price, tp_price,
                               regress_pred, classifier_conf, meta_conf, funx, meta_class)
            return

        time.sleep(interval)
        waited += interval

    send_telegram_message(f"⌛ Trade expired for {symbol} — trigger not hit.")


def place_market_order(symbol, trade_type, price, sl_price, tp_price,
                       regress_pred, classifier_conf, meta_conf, funx, meta_class):
    acct = mt5.account_info()
    sl_dist = abs(price - sl_price)
    lot = calculate_lot_size(acct.balance, sl_dist, symbol, risk_percent=2)
    order_type = mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 123456,
        "comment": "InstantTriggerOrder",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        msg = f"✅ {trade_type.upper()} order placed for {symbol} @ {price:.5f}"
        send_telegram_message(msg)
        lock_symbol(symbol, duration=300)
        close_opposite_trades(trade_type, symbol)

        trade_data = {
            "id": str(res.order),
            "symbol": symbol,
            "trade_type": trade_type,
            "entry_price": price,
            "sl": sl_price,
            "tp": tp_price,
            "entry_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "regress_pred": regress_pred,
            "classifier_conf": classifier_conf,
            "meta_classifier": meta_conf,
            "meta_class": meta_class,
            "funx": funx,
            "balance_at_entry": acct.balance,
            "equity_at_entry": acct.equity
        }
        update_trade(str(res.order), trade_data)
    else:
        send_telegram_message(f"❌ MARKET order failed for {symbol}: {res.comment if res else 'No response'}")


#FUNCTION TO DOCUMENT TRADE 
TRADE_FILE = "trades_gbp.json"

# Load existing trade data from JSON
def load_trades():
    try:
        with open(TRADE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Return empty if file is missing or corrupted

# Save trade data to JSON
def save_trades(trades):
    with open(TRADE_FILE, "w") as f:
        json.dump(trades, f, indent=4)

# Retrieve a specific trade by ID
def get_trade(trade_id):
    trades = load_trades()
    return trades.get(str(trade_id), None)

# Add or update a trade in JSON storage
def update_trade(trade_id, trade_data):
    trades = load_trades()
    trades[str(trade_id)] = trade_data
    save_trades(trades)


def refresh_trades_on_start():
    positions = mt5.positions_get()
    trades = load_trades()

    for position in positions:
        trade_id = str(position.ticket)
        if trade_id not in trades:
            broker_entry_time = datetime.datetime.fromtimestamp(position.time).replace(tzinfo=datetime.timezone.utc)
            entry_time = broker_entry_time - datetime.timedelta(minutes=BROKER_TIME_OFFSET_MINUTES)
            trade_data = {
                "id": trade_id,
                "symbol": position.symbol,
                "trade_type": "buy" if position.type == mt5.ORDER_TYPE_BUY else "sell",
                "entry_price": position.price_open,
                "exit_price": None,
                "sl": position.sl,
                "tp": position.tp,
                "profit": position.profit,
                "trade_duration": 0.0,
                "entry_time": entry_time.isoformat(),
                "exit_time": None
            }
            trades[trade_id] = trade_data

    save_trades(trades)


def close_trade(trade_id, symbol, trade_type, current_price, profit):
    trades = load_trades()

    # Initialize MetaTrader5 if not already initialized
    if not mt5.initialize():
        print("❌ MetaTrader5 connection failed!")
        logger.error("MetaTrader5 connection failed!")
        return

    # Check if the position exists
    positions = mt5.positions_get(ticket=int(trade_id))
    if not positions:
        print(f"⚠️ Position {trade_id} not found or already closed.")
        logger.warning(f"Position {trade_id} not found or already closed.")
        return

    position = positions[0]
    volume = position.volume
    print(f"Position found: {position}")
    logger.debug(f"Position found: {position}")

    # Prepare close order
    close_type = mt5.ORDER_TYPE_SELL if trade_type.lower() == "buy" else mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": int(trade_id),
        "price": current_price,  # This will be overridden by broker execution
        "deviation": 20,
        "magic": 123456,
        "comment": "Trade closed by model",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    print(f"Attempting to close trade {trade_id} at price {current_price} with volume {volume}")
    logger.debug(f"Attempting to close trade {trade_id} at price {current_price} with volume {volume}")

    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade closed successfully. Retcode: {result.retcode}, Price: {result.price}")
        logger.info(f"Trade {trade_id} closed at {result.price}, retcode: {result.retcode}")

        account_info = mt5.account_info()

        # Update trade record in JSON
        if trade_id in trades:
            trade_data = trades[trade_id]
            trade_data["exit_price"] = result.price  # Use actual exit price from result
            trade_data["exit_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            trade_data["profit"] = profit
            trade_data["balance_at_exit"] = account_info.balance if account_info else None
            trade_data["equity_at_exit"] = account_info.equity if account_info else None

            save_trades(trades)
            send_telegram_message(
                f"✅ Closed {trade_type.upper()} {symbol} trade (ID: {trade_id}) at {result.price:.5f} with profit: {profit:.2f}"
            )
    else:
        error_code, error_msg = mt5.last_error()
        print(f"❌ Failed to close trade {trade_id}. Error: {error_code}, Message: {error_msg}")
        logger.error(f"❌ Failed to close trade {trade_id}. Error code: {error_code}, message: {error_msg}")
        send_telegram_message(f"❌ Failed to close trade {trade_id}. Error: {error_code}, Message: {error_msg}")

def close_trade_partial(trade_id, symbol, trade_type, current_price, profit, fraction):
    trades = load_trades()

    if not mt5.initialize():
        print("❌ MetaTrader5 connection failed!")
        logger.error("MetaTrader5 connection failed!")
        return

    positions = mt5.positions_get(ticket=int(trade_id))
    if not positions:
        print(f"⚠️ Position {trade_id} not found or already closed.")
        logger.warning(f"Position {trade_id} not found or already closed.")
        return

    position = positions[0]
    volume = position.volume
    close_volume = round(volume * fraction, 2)  # MT5 typically supports 2 decimal places

    if close_volume <= 0:
        print(f"❌ Invalid close volume: {close_volume}")
        logger.error(f"Invalid close volume: {close_volume}")
        return

    close_type = mt5.ORDER_TYPE_SELL if trade_type.lower() == "buy" else mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": close_volume,
        "type": close_type,
        "position": int(trade_id),
        "price": current_price,
        "deviation": 20,
        "magic": 123456,
        "comment": f"Partial close {int(fraction*100)}pct by model",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    print(f"Attempting to close {fraction*100:.0f}% of trade {trade_id} at price {current_price} with volume {close_volume}")
    logger.debug(f"Attempting to close {fraction*100:.0f}% of trade {trade_id} at price {current_price} with volume {close_volume}")

    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Partial close successful. Retcode: {result.retcode}, Price: {result.price}")
        logger.info(f"Trade {trade_id} partially closed at {result.price}, volume: {close_volume}")

        account_info = mt5.account_info()

        # Update trade record in JSON
        if trade_id in trades:
            trade_data = trades[trade_id]
            trade_data["exit_price"] = result.price  # Use actual exit price from result
            trade_data["exit_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            trade_data["profit"] = profit
            trade_data["balance_at_exit"] = account_info.balance if account_info else None
            trade_data["equity_at_exit"] = account_info.equity if account_info else None

            save_trades(trades)
            send_telegram_message(
                f"✅ Closed {trade_type.upper()} {symbol} trade (ID: {trade_id}) at {result.price:.5f} with profit: {profit:.2f}"
            )

        send_telegram_message(
            f"✅ Closed {fraction*100:.0f}% of {trade_type.upper()} {symbol} trade (ID: {trade_id}) at {result.price:.5f} with profit: {profit:.2f}"
        )
    else:
        error_code, error_msg = mt5.last_error()
        print(f"❌ Failed to partially close trade {trade_id}. Error: {error_code}, Message: {error_msg}")
        logger.error(f"❌ Failed to partially close trade {trade_id}. Error code: {error_code}, message: {error_msg}")
        send_telegram_message(f"❌ Failed to close trade {trade_id}. Error: {error_code}, Message: {error_msg}")

def close_opposite_trades(new_trade_type, symbol):
    """
    Close all existing positions of the opposite type before opening a new trade.
    """
    if not mt5.initialize():
        print("❌ MetaTrader5 connection failed!")
        return

    open_positions = mt5.positions_get(symbol=symbol)
    if not open_positions:
        print("ℹ️ No open positions to check.")
        return

    opposite_type = mt5.ORDER_TYPE_SELL if new_trade_type.lower() == "buy" else mt5.ORDER_TYPE_BUY

    for pos in open_positions:
        if pos.type == opposite_type:
            trade_type_str = "sell" if pos.type == mt5.ORDER_TYPE_SELL else "buy"
            current_price = mt5.symbol_info_tick(symbol).bid if trade_type_str == "sell" else mt5.symbol_info_tick(symbol).ask
            close_trade(
                trade_id=pos.ticket,
                symbol=symbol,
                trade_type=trade_type_str,
                current_price=current_price,
                profit=pos.profit
            )


def repair_missing_trades():
    print("🔧 Repairing missing trades from history...")
    trades = load_trades()
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=3)

    closed_deals = mt5.history_deals_get(start, now)
    if closed_deals is None:
        print("⚠️ No deal history found.")
        return

    for deal in closed_deals:
        trade_id = str(deal.position_id)
        if trade_id not in trades:
            continue
        
        trade = trades[trade_id]
        if trade.get("exit_price") is None or trade.get("profit") is None:
            trade["exit_price"] = deal.price
            trade["exit_time"] = datetime.datetime.fromtimestamp(deal.time).isoformat()
            trade["profit"] = deal.profit
            print(f"✅ Updated trade ID {trade_id} with profit: {deal.profit}")
    
    save_trades(trades)


def send_limited_telegram_message(message):
    global last_trade_summary_time

    current_time = time.time()
    if current_time - last_trade_summary_time >= TRADE_SUMMARY_INTERVAL:
        send_telegram_message(message)
        last_trade_summary_time = current_time
    else:
        print("⏱️ Message skipped to respect 5-minute interval.")

def manage_trades(symbol, df_buy, df_sell):
    """Enhanced trade management using the correct directional features with meta classifier TP adjustment."""
    trades = load_trades()
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[ERROR] Could not fetch symbol info for {symbol}")
        return

    point = symbol_info.point
    digits = symbol_info.digits
    positions = mt5.positions_get(symbol=symbol)

    if positions:
        for position in positions:
            trade_id = str(position.ticket)
            is_buy = position.type == mt5.ORDER_TYPE_BUY
            df = df_buy if is_buy else df_sell
            latest_5m = df.iloc[-2]
            latest_5m_sma9 = latest_5m["SMA_9"]
            valid_hour = latest_5m["valid_hour"]
            current_price = mt5.symbol_info_tick(symbol).ask if is_buy else mt5.symbol_info_tick(symbol).bid
            profit = position.profit
            entry_price = position.price_open
            current_sl = position.sl
            current_tp = position.tp

            # Load or initialize trade data
            if trade_id in trades:
                trade_data = trades[trade_id]
                entry_time = datetime.datetime.fromisoformat(trade_data["entry_time"])
            else:
                entry_time = datetime.datetime.fromtimestamp(position.time).replace(tzinfo=datetime.timezone.utc)
                trade_data = {
                    "id": trade_id,
                    "symbol": symbol,
                    "trade_type": "buy" if is_buy else "sell",
                    "entry_price": entry_price,
                    "exit_price": None,
                    "sl": current_sl,
                    "tp": current_tp,
                    "profit": profit,
                    "entry_time": entry_time.isoformat(),
                    "exit_time": None,
                    "regress_pred": None,
                    "classifier_conf": None,
                    "funx": None,
                    "max_rr": 0,
                    "current_rr": 0,
                    "close_reason": None,
                    "closed_fraction": 0.0,
                    "hit_1_to_3": False,
                    "meta_classifier": None,
                    "meta_class": None,
                    "balance_at_entry": mt5.account_info().balance,
                    "equity_at_entry": mt5.account_info().equity,
                    "current_price": current_price,
                    "trade_duration": 0,
                    "milestone_comment": None,
                    "Target Hit": False
                }

            current_time = datetime.datetime.now(datetime.timezone.utc)
            trade_duration = (current_time - entry_time).total_seconds() / 60
            rr_unit = abs(entry_price - current_sl)

            if rr_unit == 0:
                print(f"[ERROR] RR Unit is zero for trade {trade_id}. Skipping RR calculation.")
                continue

            # === Meta Classifier-based TP Adjustment ===
            meta_class = trade_data.get("meta_class")
            if meta_class is not None:
                rr_targets = {
                    0: 0.0,
                    1: 1.10,
                    2: 1.10,
                }
                target_rr = rr_targets.get(meta_class, 1.0)
                expected_tp = entry_price + (target_rr * rr_unit) if is_buy else entry_price - (target_rr * rr_unit)
                rounded_expected_tp = round(expected_tp, digits)
                tp_deviation = abs((current_tp or 0) - rounded_expected_tp)

                if tp_deviation > 0.1 * rr_unit:
                    print(f"[⚙️ FIX] Adjusting TP for {symbol} trade {trade_id} to {target_rr}RR based on meta_class {meta_class}")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": int(trade_id),
                        "sl": current_sl,
                        "tp": rounded_expected_tp,
                        "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                        "magic": 123456,
                        "comment": f"Set TP to fixed {target_rr} RR",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        trade_data["tp"] = rounded_expected_tp
                        send_telegram_message(
                            f"🎯 TP updated for {symbol} trade {trade_id} → {target_rr}RR @ {rounded_expected_tp:.5f} (meta_class: {meta_class})"
                        )
                    else:
                        print(f"[WARNING] Failed to adjust TP for {trade_id}: {result.comment}")

            # === Calculate RR and update stats ===
            price_diff = (current_price - entry_price) if is_buy else (entry_price - current_price)
            current_rr = price_diff / rr_unit
            max_rr = max(current_rr, trade_data.get("max_rr", 0))

            trade_data.update({
                "current_price": current_price,
                "profit": profit,
                "trade_duration": trade_duration,
                "current_rr": current_rr,
                "max_rr": max_rr,
                "sl": current_sl,
                "tp": current_tp
            })

            # Milestone
            # if max_rr >= 1.5 and not trade_data.get("Target Hit"):
            #     trade_data["hit_1_to_2"] = True
            #     trade_data["milestone_comment"] = "🎯 Hit 1.5RR"
            #     trade_data["Target Hit"] = True
            #     send_telegram_message(f"🎯 Trade {symbol} (ID: {trade_id}) Hit 1.5RR Target! Current RR: {current_rr:.2f}")

            # === Exit Rules ===
            exit_triggered = False

            # Rule 1: SMA9 crossover against position
            if (is_buy and latest_5m["Close"] < latest_5m_sma9) or (not is_buy and latest_5m["Close"] > latest_5m_sma9):
                if current_rr >= 0.5 and trade_duration >= 60 :
                    close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                    trade_data["close_reason"] = "SMA9 crossover against position"
                    exit_triggered = True
            
            if max_rr >= 0.9 and current_rr <= 0.1 :
                    close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                    trade_data["close_reason"] = "Negative Drop"
                    exit_triggered = True

            # Rule 2: Time decay (4 hours)
            if not exit_triggered and trade_duration >= 240:
                close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                trade_data["close_reason"] = "Time decay exit (4h)"
                exit_triggered = True
                send_telegram_message(f"📉 Time decay exit for {symbol} after 4 hours")
           
            if not valid_hour: 
                close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                trade_data["close_reason"] = "📉 Closed all trade for the day"
                exit_triggered = True
                send_telegram_message(f"📉 Closed all trade for the day")


            # Rule 3: Profit drop from peak
            # if not exit_triggered and max_rr >= 1.5 and current_rr <= 0.3:
            #     close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
            #     trade_data["close_reason"] = f"Profit drop from {max_rr:.2f}RR to {current_rr:.2f}RR"
            #     exit_triggered = True
            #     send_telegram_message(f"📉 Profit drop exit for {symbol} from {max_rr:.2f}RR to {current_rr:.2f}RR")
            
            # if not exit_triggered and max_rr >= 2.5 and current_rr <= 1.5:
            #     close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
            #     trade_data["close_reason"] = f"Profit drop from {max_rr:.2f}RR to {current_rr:.2f}RR"
            #     exit_triggered = True
            #     send_telegram_message(f"📉 Profit drop exit for {symbol} from {max_rr:.2f}RR to {current_rr:.2f}RR")


            if exit_triggered:
                trade_data.update({
                    "exit_price": current_price,
                    "exit_time": current_time.isoformat(),
                    "profit": profit
                })
                trades[trade_id] = trade_data
                save_trades(trades)
                continue

            trades[trade_id] = trade_data
  # === Sync broker history for closed deals ===
    closed_deals = mt5.history_deals_get(
        datetime.datetime.now() - datetime.timedelta(days=1), 
        datetime.datetime.now()
    )
    
    if closed_deals:
        for deal in closed_deals:
            if deal.entry != 1:  # 1 means entry deal, we want exit deals
                continue
                
            position_id = str(deal.position_id)
            if position_id in trades:
                trade_data = trades[position_id]
                if not trade_data.get("exit_time"):
                    trade_data.update({
                        "exit_price": deal.price,
                        "exit_time": datetime.datetime.fromtimestamp(deal.time).isoformat(),
                        "profit": deal.profit,
                        "close_reason": trade_data.get("close_reason") or "Closed by broker"
                    })
                    trades[position_id] = trade_data
                    print(f"Updated closed trade {position_id} with exit data")

    save_trades(trades)



def extract_features(df):
    latest_data_df = df.iloc[-2]
    # print("🔍 Latest feature extracted:")
    # print(latest_data_df)

    nan_features = [feat for feat in feature_list if pd.isna(latest_data_df[feat])]
    
    if nan_features:
        message = f"🚨 NaN detected in features: {', '.join(nan_features)}"
        print(message)
        send_telegram_message(message)
        return None

    feature_values = [float(latest_data_df[feat]) for feat in feature_list]

    return feature_values

spread_limits_low = {
        "xau": 01.00, "jpy": 0.050,
        "gbpusd": 0.00050, 
        "usdcad": 0.00050
}
spread_limits_high = {
        "xau": 09.00, "jpy": 0.350,
        "gbpusd": 0.00350, 
        "usdcad": 0.00350
}



 # === Entry Logics ===
        #----------------------------------------
        #       Buy 5Min TF
        #---------------------------------------- 
 # ─── Define per‑symbol R:R thresholds ───
def buyM5(df_buy, response_data, symbol):
    RR_THRESHOLDS_BUY = {
        "XAUUSD+": 1.0,
        "GBPUSD+": 1.0,
        "USDJPY+": 1.0,
        "USDCAD+": 1.0,
    }

    df = df_buy
    result = response_data.get("buy", {})

    if not result or not result.get("accepted"):
        logger.info(f"{symbol}: buy skipped — not accepted by ML stack")
        return

    rr_pred = result.get("reg_pred")
    clf_probs = result.get("classifier_probs", {})
    meta_class = result.get("final_class")
    meta_probs = result.get("class_probabilities", [])
    trade_type = result.get("trade_type", "buy")

    if rr_pred is None or meta_class is None:
        logger.warning(f"{symbol}: buy skipped — incomplete ML prediction result")
        return

    if symbol not in RR_THRESHOLDS_BUY:
        raise KeyError(f"❌ No RR threshold defined for symbol '{symbol}'")

    rr_thresh = RR_THRESHOLDS_BUY[symbol]

    prev = df.iloc[-3]
    curr = df.iloc[-2]
    pair_code = curr['pair']
    log_returns = curr['log_returns'] 

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"❌ No spread limit defined for pair '{pair_code}'")

    if not curr.get("valid_hour", False):
        logger.info(f"{symbol}: buy skipped — outside valid trading hours")
        return

    if rr_pred < rr_thresh:
        logger.info(
            f"{symbol}: Warning — RR_pred {rr_pred:.2f} below threshold {rr_thresh:.2f}, "
            "but meta-model approved. Proceeding with trade."
        )
   
    # if log_returns <= 0.0001:
    #     logger.info(
    #         f"{symbol}: Alert skipping trade log returns too low "        
    #     )
    #     return 
    
    META_CONF_THRESH = {1: 0.85, 2: 0.85}

    if meta_class in META_CONF_THRESH:
        if meta_probs[meta_class] < META_CONF_THRESH[meta_class]:
            logger.info(
                f"{symbol}: Buy skipped — Meta class {meta_class} prob {meta_probs[meta_class]:.2f} "
                f"below threshold {META_CONF_THRESH[meta_class]}"
            )
            return  

    # === Trade Prices ===
    trigger_price = curr["Close"]
    sl_price = curr["stop_loss_price"]
    sl_dist = trigger_price - sl_price
    tp_price = trigger_price + (sl_dist * meta_class)  # final_class = TP multiplier

    spread_low = spread_limits_low[pair_code]
    spread_high = spread_limits_high[pair_code]

    if sl_dist < spread_low or sl_dist > spread_high:
        msg = (
            f"{pair_code.upper()}: BUY skipped — SL distance {sl_dist:.5f} "
            f"not in range ({spread_low:.5f} – {spread_high:.5f})"
        )
        logger.info(msg)
        send_telegram_message(msg)
        return

    # === Execute Trade ===
    handle_trigger_or_watch(
        symbol=symbol,
        trade_type="buy",
        trigger_price=trigger_price,
        sl_price=sl_price,
        tp_price=tp_price,
        regress_pred=rr_pred,
        classifier_conf=clf_probs,
        meta_conf=meta_probs,
        funx="White-Arrow A.I Model Buy",
        meta_class=meta_class,
    )

    # === Telegram Message ===
    msg = (
        f"{'-'*42}\n"
        f"✅ ML BUY SIGNAL (5m) {symbol}\n"
        f"🎯 Entry      : {curr['entry_price']:.5f}\n"
        f"🛡️ StopLoss   : {sl_price:.5f}\n"
        f"📈 TakeProfit : {tp_price:.5f} (x{meta_class})\n"
        f"🤖 Pred RR    : {rr_pred:.2f}\n"
        f"📊 Classifier Probs:\n"
        f"   ├─ 1:1 : {clf_probs.get('clf_1_1_prob', 0):.2f}\n"
        f"   ├─ 1:2 : {clf_probs.get('clf_1_2_prob', 0):.2f}\n"
        f"📊 Meta Probabilities:\n"
        f"   ├─ Reject : {meta_probs[0]:.2f}\n"
        f"   ├─ 1:1    : {meta_probs[1]:.2f}\n"
        f"   └─ 1:2    : {meta_probs[2]:.2f}\n"
        f"📊 Candle:\n"
        f"   ├─ Open   : {curr['Open']:.5f}\n"
        f"   ├─ High   : {curr['High']:.5f}\n"
        f"   ├─ Low    : {curr['Low']:.5f}\n"
        f"   └─ Close  : {curr['Close']:.5f}\n"
        f"{'-'*42}"
    )
    send_telegram_message(msg)
    logger.info(msg)

def sellM5(df_sell, response_data, symbol):
    RR_THRESHOLDS_SELL = {
        "XAUUSD+": 1.0,
        "GBPUSD+": 1.0,
        "USDJPY+": 1.0,
        "USDCAD+": 1.0,
    }

    df = df_sell
    result = response_data.get("sell", {})

    if not result or not result.get("accepted"):
        logger.info(f"{symbol}: sell skipped — not accepted by ML stack")
        return

    rr_pred = result.get("reg_pred")
    clf_probs = result.get("classifier_probs", {})
    meta_class = result.get("final_class")
    meta_probs = result.get("class_probabilities", [])
    trade_type = result.get("trade_type", "sell")

    if rr_pred is None or meta_class is None:
        logger.warning(f"{symbol}: sell skipped — incomplete ML prediction result")
        return

    if symbol not in RR_THRESHOLDS_SELL:
        raise KeyError(f"❌ No RR threshold defined for symbol '{symbol}'")

    rr_thresh = RR_THRESHOLDS_SELL[symbol]

    prev = df.iloc[-3]
    curr = df.iloc[-2]
    pair_code = curr['pair']
    log_returns = curr['log_returns'] 

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"❌ No spread limit defined for pair '{pair_code}'")

    if not curr.get("valid_hour", False):
        logger.info(f"{symbol}: sell skipped — outside valid trading hours")
        return

    if rr_pred < rr_thresh:
        logger.info(
            f"{symbol}: Warning — RR_pred {rr_pred:.2f} below threshold {rr_thresh:.2f}, "
            "but meta-model approved. Proceeding with trade."
        )

    # if log_returns >= -0.0001:
    #     logger.info(
    #         f"{symbol}: Alert skipping trade log returns too  high"      
    #     )
    #     return
    META_CONF_THRESH = {1: 0.85, 2: 0.85}
    if meta_class in META_CONF_THRESH:
        if meta_probs[meta_class] < META_CONF_THRESH[meta_class]:
            logger.info(
                f"{symbol}: Buy skipped — Meta class {meta_class} prob {meta_probs[meta_class]:.2f} "
                f"below threshold {META_CONF_THRESH[meta_class]}"
            )
            return  


    # === Trade Prices ===
    trigger_price = curr["Close"]
    sl_price = curr["stop_loss_price"]
    sl_dist = sl_price - trigger_price
    tp_price = trigger_price - (sl_dist * meta_class)  # final_class = TP multiplier

    spread_low = spread_limits_low[pair_code]
    spread_high = spread_limits_high[pair_code]

    if sl_dist < spread_low or sl_dist > spread_high:
        msg = (
            f"{pair_code.upper()}: SELL skipped — SL distance {sl_dist:.5f} "
            f"not in range ({spread_low:.5f} – {spread_high:.5f})"
        )
        logger.info(msg)
        send_telegram_message(msg)
        return

    # === Execute Trade ===
    handle_trigger_or_watch(
        symbol=symbol,
        trade_type="sell",
        trigger_price=trigger_price,
        sl_price=sl_price,
        tp_price=tp_price,
        regress_pred=rr_pred,
        classifier_conf=clf_probs,
        meta_conf=meta_probs,
        funx="White-Arrow A.I Model Sell",
        meta_class=meta_class,
    )

    # === Telegram Message ===
    msg = (
        f"{'-'*42}\n"
        f"✅ ML SELL SIGNAL (5m) {symbol}\n"
        f"🎯 Entry      : {curr['entry_price']:.5f}\n"
        f"🛡️ StopLoss   : {sl_price:.5f}\n"
        f"📈 TakeProfit : {tp_price:.5f} (x{meta_class})\n"
        f"🤖 Pred RR    : {rr_pred:.2f}\n"
        f"📊 Classifier Probs:\n"
        f"   ├─ 1:1 : {clf_probs.get('clf_1_1_prob', 0):.2f}\n"
        f"   ├─ 1:2 : {clf_probs.get('clf_1_2_prob', 0):.2f}\n"
        f"📊 Meta Probabilities:\n"
        f"   ├─ Reject : {meta_probs[0]:.2f}\n"
        f"   ├─ 1:1    : {meta_probs[1]:.2f}\n"
        f"   └─ 1:2    : {meta_probs[2]:.2f}\n"
        f"📊 Candle:\n"
        f"   ├─ Open   : {curr['Open']:.5f}\n"
        f"   ├─ High   : {curr['High']:.5f}\n"
        f"   ├─ Low    : {curr['Low']:.5f}\n"
        f"   └─ Close  : {curr['Close']:.5f}\n"
        f"{'-'*42}"
    )
    send_telegram_message(msg)
    logger.info(msg)


def send_telegram_message(text):
    """Send a message to Telegram"""
    # print(f"\n📨 Sending message to Telegram: {text}")
    print(f"\n📨 Sending message to Telegram")
    
    payload = {"chat_id": CHAT_ID, "text": text}
    
    try:
        response = requests.post(TELEGRAM_URL, json=payload)
        
        if response.status_code == 200:
            print("✅ Telegram message sent successfully!")
        else:
            print(f"❌ Telegram error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"🚨 Telegram request failed: {e}")

import os

csv_write_lock = threading.Lock()
def save_prediction_row_async(row_data: pd.Series, symbol: str, base_path: str = "latest_predictions", max_rows: int = 20000):
    def save():
        try:
            os.makedirs(base_path, exist_ok=True)
            csv_path = os.path.join(base_path, f"{symbol.upper()}.csv")

            # Wrap row into a DataFrame
            row = pd.DataFrame([row_data])

            with csv_write_lock:
                if os.path.exists(csv_path):
                    existing = pd.read_csv(csv_path)

                    row = row.reindex(columns=existing.columns, fill_value=None)

                    is_duplicate = (
                        not existing.empty and
                        existing.iloc[0:1].drop(columns=["timestamp"], errors="ignore").equals(
                            row.iloc[0:1].drop(columns=["timestamp"], errors="ignore")
                        )
                    )

                    if not is_duplicate:
                        new_df = pd.concat([row, existing], ignore_index=True)
                        new_df = new_df.head(max_rows)
                        new_df.to_csv(csv_path, index=False)
                else:
                    row.to_csv(csv_path, index=False)

        except Exception as e:
            print(f"⚠️ Threaded CSV write failed for {symbol}: {e}")

    threading.Thread(target=save).start()

def get_predictions(features: pd.DataFrame, symbol: str, trade_type="buy",
                    save_csv=False, csv_path="latest_predictions.csv", max_rows=20000):
    """
    Full stacked prediction pipeline:
    - 2 Base Classifiers → Regression → Meta (Multi-Class) → Final Decision
    """
    if not isinstance(features, pd.DataFrame):
        print(f"❌ 'features' must be a DataFrame for {symbol}")
        return None

    if trade_type not in ("buy", "sell"):
        print(f"❌ Invalid trade_type: '{trade_type}'")
        return None

    try:
        models = {
            "clf_1_1_prob": loaded_models[f"clf_{trade_type}_1.1"],
            "clf_1_2_prob": loaded_models[f"clf_{trade_type}_1.2"],
            "reg_pred": loaded_models[f"reg_{trade_type}"],
            "meta": loaded_models[f"meta_{trade_type}"]
        }
    except KeyError as e:
        print(f"❌ Missing models for {trade_type.upper()}: {e}")
        return None

    result = {
        "symbol": symbol,
        "trade_type": trade_type,
        "accepted": False,
        "final_class": None,
        "class_probabilities": None,
        "classifier_probs": {},
        "reg_pred": None
    }

    try:
        # === Predict classifier probabilities ===
        for clf_key in ["clf_1_1_prob", "clf_1_2_prob"]:
            clf_model = models[clf_key]
            clf_cols = clf_model.feature_name()
            clf_input = features.reindex(columns=clf_cols, fill_value=0)

            if clf_input.shape[1] != len(clf_cols):
                print(f"⚠️ {clf_key} feature count mismatch. Expected {len(clf_cols)}, got {clf_input.shape[1]}")
                continue

            prob = clf_model.predict(clf_input)
            # Ensure we get a single float value
            if isinstance(prob, (np.ndarray, list)):
                prob = float(prob[0])
            result["classifier_probs"][clf_key] = float(prob)

        # === Predict regression output ===
        reg_model = models["reg_pred"]
        reg_cols = reg_model.feature_name()
        reg_input = features.copy()
        reg_input["clf_1_1_prob"] = result["classifier_probs"].get("clf_1_1_prob", 0.0)
        reg_input["clf_1_2_prob"] = result["classifier_probs"].get("clf_1_2_prob", 0.0)
        reg_input_aligned = reg_input.reindex(columns=reg_cols, fill_value=0)

        if reg_input_aligned.shape[1] != len(reg_cols):
            print(f"⚠️ Regressor feature count mismatch. Expected {len(reg_cols)}, got {reg_input_aligned.shape[1]}")
            return result

        reg_value = reg_model.predict(reg_input_aligned)
        # Ensure we get a single float value
        if isinstance(reg_value, (np.ndarray, list)):
            reg_value = float(reg_value[0])
        result["reg_pred"] = float(reg_value)

        # === Predict meta classification ===
        meta_model = models["meta"]
        meta_cols = meta_model.feature_name()
        meta_input = features.copy()
        meta_input["clf_1_1_prob"] = result["classifier_probs"].get("clf_1_1_prob", 0.0)
        meta_input["clf_1_2_prob"] = result["classifier_probs"].get("clf_1_2_prob", 0.0)
        meta_input["reg_pred"] = reg_value
        meta_input_aligned = meta_input.reindex(columns=meta_cols, fill_value=0)

        if meta_input_aligned.shape[1] != len(meta_cols):
            print(f"⚠️ Meta-model feature count mismatch. Expected {len(meta_cols)}, got {meta_input_aligned.shape[1]}")
            return result

        meta_raw = meta_model.predict(meta_input_aligned)
        meta_class = int(np.argmax(meta_raw))
        meta_probs = meta_raw.flatten().tolist()


        result["final_class"] = meta_class
        result["class_probabilities"] = meta_probs
        result["accepted"] = meta_class > 0

        if result["accepted"]:
            print(f"✅ {symbol} {trade_type.upper()} accepted → Class {meta_class} | R:R: {reg_value:.2f}")
        else:
            print(f"🔕 {symbol} {trade_type.upper()} rejected → Class {meta_class} | R:R: {reg_value:.2f}")

        # === Optional threaded CSV saving ===
        if save_csv:
            try:
                row_data = features.iloc[0].copy()  # Use first row if multiple rows
                row_data["timestamp"] = pd.Timestamp.utcnow()
                row_data["symbol"] = symbol
                row_data["trade_type"] = trade_type
                row_data["clf_1_1_prob"] = result["classifier_probs"].get("clf_1_1_prob", 0.0)
                row_data["clf_1_2_prob"] = result["classifier_probs"].get("clf_1_2_prob", 0.0)
                row_data["reg_pred"] = reg_value
                row_data["meta_class"] = meta_class
                row_data["accepted"] = result["accepted"]

                save_prediction_row_async(row_data=row_data, symbol=symbol, base_path="latest_predictions", max_rows=max_rows)
            except Exception as e:
                print(f"⚠️ Failed to prepare prediction row for {symbol}: {e}")

        return result

    except Exception as e:
        print(f"❌ Pipeline error: {type(e).__name__} - {str(e)}")
        return None
# === CONFIG ===
SYMBOLS = ["GBPUSD+", "USDCAD+"] #"USDJPY+","XAUUSD+", 

TIMEFRAME = mt5.TIMEFRAME_M5
POLL_INTERVAL = 1  # seconds

# Dynamic worker calculation
MAX_WORKERS_PER_SYMBOL = min(2, max(1, cpu_count() // len(SYMBOLS)))
print(f"System CPU cores: {cpu_count()}")
print(f"Using {MAX_WORKERS_PER_SYMBOL} workers per symbol")

# === GLOBALS ===
last_timestamps = {}
latest_features = {}  
feature_lock = threading.Lock()
thread_map = {}  
executors = {}  
resource_log_interval = 3600  # Log resources every 5 minutes
last_resource_log = 0

# === Configure Logging ===
# Main trading logger
trading_logger = logging.getLogger("TradingBot")
trading_logger.setLevel(logging.INFO)

# System monitoring logger
system_logger = logging.getLogger("SystemMonitor")
system_logger.setLevel(logging.INFO)

# Create formatters
trade_formatter = logging.Formatter("%(asctime)s - TRADING - %(levelname)s - %(message)s")
system_formatter = logging.Formatter("%(asctime)s - SYSTEM - %(levelname)s - %(message)s")

# File handlers
trade_file_handler = logging.FileHandler("trading_operations.log", encoding="utf-8")
trade_file_handler.setFormatter(trade_formatter)

system_file_handler = logging.FileHandler("system_monitor.log", encoding="utf-8")
system_file_handler.setFormatter(system_formatter)

# Console handlers
trade_console_handler = logging.StreamHandler()
trade_console_handler.setFormatter(trade_formatter)

system_console_handler = logging.StreamHandler()
system_console_handler.setFormatter(system_formatter)

# Add handlers if none exist
if not trading_logger.handlers:
    trading_logger.addHandler(trade_file_handler)
    trading_logger.addHandler(trade_console_handler)

if not system_logger.handlers:
    system_logger.addHandler(system_file_handler)
    system_logger.addHandler(system_console_handler)



# === Updated Resource Monitoring ===
def log_system_resources():
    """Log system resource usage using system logger"""
    global last_resource_log
    now = time.time()
    if now - last_resource_log >= resource_log_interval:
        mem = psutil.virtual_memory()
        load = psutil.getloadavg()
        system_logger.info(
            f"Resource Usage | "
            f"CPU: {load[0]:.1f}/{cpu_count()} | "
            f"Mem: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB | "
            f"Threads: {threading.active_count()}"
        )
        last_resource_log = now


def symbol_loop(symbol):
    global last_timestamps
    trading_logger.info(f"[{symbol}] Starting symbol loop")
    PAIR_CATS = ["xau", "gbpusd", "jpy", "usdcad"]



    while True:
        try:
            log_system_resources()
            trading_logger.debug(f"[{symbol}] 🔄 Starting new iteration...")

            df_head = fetch_candles(symbol, TIMEFRAME, 2)
            if df_head is None or len(df_head) < 1:
                trading_logger.warning(f"[{symbol}] ⚠️ Insufficient head data.")
                send_telegram_message(f"[{symbol}] ⚠️ Fetch issue.")
                time.sleep(POLL_INTERVAL)
                continue

            df_head["Time"] = pd.to_datetime(df_head["Time"], unit="s")
            last_closed_time = df_head.iloc[-2]["Time"]

            if last_timestamps.get(symbol) == last_closed_time:
                time.sleep(POLL_INTERVAL)
                continue

            last_timestamps[symbol] = last_closed_time
            trading_logger.info(f"[{symbol}] 🆕 Candle at {last_closed_time}")

            df_raw = fetch_candles(symbol, TIMEFRAME, 320)
            if df_raw is None or len(df_raw) < 300:
                trading_logger.warning(f"[{symbol}] ⚠️ Insufficient history.")
                continue

            df_raw["Time"] = pd.to_datetime(df_raw["Time"], unit="s")
           
            pair_code = 'unkn'
            if symbol == 'XAUUSD+':
                pair_code = "xau"
            elif symbol == "GBPUSD+":
                pair_code = "gbpusd"
            elif symbol == "USDJPY+":
                pair_code = "jpy"  # fixed
            elif symbol == "USDCAD+":
                pair_code = "usdcad"  # fixed       


            # Submit feature computation
            t0 = time.time()
            future_buy = executors[symbol].submit(calculate_features, df_raw.copy(), 1, pair_code)
            future_sell = executors[symbol].submit(calculate_features, df_raw.copy(), 0, pair_code)

            try:
                df_buy = future_buy.result(timeout=20)
                df_sell = future_sell.result(timeout=20)
            except FuturesTimeoutError:
                trading_logger.error(f"[{symbol}] ⏱️ Feature timeout.")
                send_telegram_message(f"[{symbol}] ⏱️ Timeout.")
                continue

            elapsed = time.time() - t0
            trading_logger.info(f"[{symbol}] ✅ Features in {elapsed:.2f}s")

            with feature_lock:
                latest_features[symbol] = {"buy": df_buy, "sell": df_sell}

            response_data = {}

            # ========== BUY ==========
            if df_buy.iloc[-2]["side"] == 1:
                buy_row = df_buy.iloc[[-2]].copy()
                buy_row["pair"] = pd.Categorical([pair_code], categories=PAIR_CATS)
                prediction = get_predictions(buy_row, symbol, trade_type="buy", save_csv=True)
                if prediction and prediction.get("accepted"):
                    response_data["buy"] = prediction
                    buyM5(df_buy, response_data, symbol)
                else:
                    trading_logger.info(f"[{symbol}] 🚫 Buy rejected or not valid")

            # ========== SELL ==========
            if df_sell.iloc[-2]["side"] == 0:
                sell_row = df_sell.iloc[[-2]].copy()
                sell_row["pair"] = pd.Categorical([pair_code], categories=PAIR_CATS)
                prediction = get_predictions(sell_row, symbol, trade_type="sell", save_csv=True)
                if prediction and prediction.get("accepted"):
                    response_data["sell"] = prediction
                    sellM5(df_sell, response_data, symbol)
                else:
                    trading_logger.info(f"[{symbol}] 🚫 Sell rejected or not valid")


            if elapsed >= 5:
                threading.Thread(
                    target=send_telegram_message,
                    args=(f"[{symbol}] ✅ Done in {elapsed:.2f}s",),
                    daemon=True
                ).start()


        except Exception as e:
            trading_logger.exception(f"[{symbol}] ❌ Loop error: {e}")
            send_telegram_message(f"[{symbol}] ❌ Error: {e}")
            time.sleep(5)

        time.sleep(POLL_INTERVAL)



def trade_manager_loop():
    while True:
        try:
            with feature_lock:
                for sym in SYMBOLS:
                    if sym not in latest_features:
                        continue
                    df_buy = latest_features[sym].get("buy")
                    df_sell = latest_features[sym].get("sell")
                    if df_buy is not None and df_sell is not None:
                        manage_trades(sym, df_buy, df_sell)
        except Exception as e:
            trading_logger.exception(f"[{sym}] ❌ Trade manager loop error: {e}")
        time.sleep(2)



def thread_refresher_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                thread = thread_map.get(symbol)
                if thread is None or not thread.is_alive():
                    trading_logger.warning(f"[{symbol}]   Dead thread detected. Restarting.")
                    restart_symbol_thread(symbol)
        except Exception as e:
            trading_logger.exception(f"[{symbol}] ❌ Error during thread health check: {e}")
        time.sleep(60)



def restart_symbol_thread(symbol):
    try:
        trading_logger.warning(f"[{symbol}] 🔁 Restarting symbol thread.")
        executors[symbol] = ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_SYMBOL)
        new_thread = threading.Thread(
            target=symbol_loop,
            args=(symbol,),
            daemon=True,
            name=f"Restarted_{symbol}"
        )
        new_thread.start()
        thread_map[symbol] = new_thread
        trading_logger.info(f"[{symbol}] 🧵 Thread restarted at {pd.Timestamp.now()}")
    except Exception as e:
        trading_logger.error(f"[{symbol}] ❌ Failed to restart thread: {e}")
        send_telegram_message(f"[{symbol}] ❌ Failed to restart thread: {e}")




def run_parallel_symbol_loops():
    """Initialize all symbol threads with dynamic worker pools"""
    for symbol in SYMBOLS:
        last_timestamps[symbol] = None
        executors[symbol] = ThreadPoolExecutor(
            max_workers=MAX_WORKERS_PER_SYMBOL,
            thread_name_prefix=f"Work_{symbol}"
        )
        t = threading.Thread(
            target=symbol_loop,
            args=(symbol,),
            name=f"Thread_{symbol}",
            daemon=True
        )
        t.start()
        thread_map[symbol] = t
        trading_logger.info(f"[{symbol}] 🧵 Started with {MAX_WORKERS_PER_SYMBOL} workers")




if __name__ == "__main__":
    # System info diagnostics
    system_logger.info(f"🚀 Starting trading bot on {cpu_count()} CPU cores")
    trading_logger.info(f"📌 Symbols: {', '.join(SYMBOLS)}")
    trading_logger.info(f"🧵 {MAX_WORKERS_PER_SYMBOL} workers per symbol")

    if initialize_mt5(login, password, server):
        trading_logger.info("✅ MT5 Initialized Successfully")
        
        # Start all components
        run_parallel_symbol_loops()
        threading.Thread(
            target=trade_manager_loop,
            name="TradeManager",
            daemon=True
        ).start()
        
        threading.Thread(
            target=thread_refresher_loop,
            name="ThreadRefresher",
            daemon=True
        ).start()

        try:
            while True:
                log_system_resources()
                time.sleep(60)
        except KeyboardInterrupt:
            trading_logger.info("🛑 Shutting down gracefully...")
        except Exception as e:
            system_logger.critical(f"💥 Fatal error: {e}", exc_info=True)
    else:
        trading_logger.error("❌ MT5 login failed")