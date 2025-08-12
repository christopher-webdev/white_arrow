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
from dotenv import load_dotenv
import logging
from pathlib import Path

import shutil 
from contextlib import contextmanager

load_dotenv()

# Fix UnicodeEncodeError on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# === Configure Logging (Two Independent Loggers) ===
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Formats (customize if you like)
APP_FORMAT = logging.Formatter("%(asctime)s - APP - %(levelname)s - %(message)s")
TRADE_FORMAT = logging.Formatter("%(asctime)s - TRADE - %(levelname)s - %(message)s")

# Named loggers (avoid using the root logger for both)
logger = logging.getLogger("app")      # general app logger
t_logger = logging.getLogger("trade")  # trade management logger

logger.setLevel(logging.INFO)
t_logger.setLevel(logging.INFO)

# File handlers
app_file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
app_file_handler.setFormatter(APP_FORMAT)
app_file_handler.setLevel(logging.INFO)

trade_file_handler = logging.FileHandler(LOG_DIR / "trade_management.log", encoding="utf-8")
trade_file_handler.setFormatter(TRADE_FORMAT)
trade_file_handler.setLevel(logging.INFO)

# Console handlers
app_console_handler = logging.StreamHandler()
app_console_handler.setFormatter(APP_FORMAT)
app_console_handler.setLevel(logging.INFO)

trade_console_handler = logging.StreamHandler()
trade_console_handler.setFormatter(TRADE_FORMAT)
trade_console_handler.setLevel(logging.INFO)

# Avoid duplicate handlers on hot-reload / re-import
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("app.log")
    for h in logger.handlers
):
    logger.addHandler(app_file_handler)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(app_console_handler)

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("trade_management.log")
    for h in t_logger.handlers
):
    t_logger.addHandler(trade_file_handler)
if not any(isinstance(h, logging.StreamHandler) for h in t_logger.handlers):
    t_logger.addHandler(trade_console_handler)

# Prevent propagation to root (avoids double logs)
logger.propagate = False
t_logger.propagate = False

# === Usage examples ===
logger.info("App started")
t_logger.info("Trade manager initialized")

risk_percent = 2  # Risk percentage for lot size calculation

trading_enabled = True 

last_summary_sent_time = 0  # global or somewhere accessible
SUMMARY_SEND_INTERVAL = 5 * 60  # 5 minutes in seconds
last_trade_summary_time = 0
TRADE_SUMMARY_INTERVAL = 5 * 60  # 5 minutes in seconds

# Telegram Bot Details


MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

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
    "clf_buy_1.1": "./tmodel_artifacts_buy/classifier_1_1.txt",
    "clf_buy_1.2": "./tmodel_artifacts_buy/classifier_1_2.txt",
    "reg_buy": "./tmodel_artifacts_buy/regressor.txt",
    "meta_buy": "./tmodel_artifacts_buy/meta_model.txt",

    # SELL side
    "clf_sell_1.1": "./tmodel_artifacts_sell/classifier_1_1.txt",
    "clf_sell_1.2": "./tmodel_artifacts_sell/classifier_1_2.txt",
    "reg_sell": "./tmodel_artifacts_sell/regressor.txt",
    "meta_sell": "./tmodel_artifacts_sell/meta_model.txt"
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
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(TELEGRAM_URL, json=payload)

# Function to initialize MT5 
def initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
    """Initialize MT5 and login with provided credentials."""
    if not mt5.initialize():
        error_message = "❌ MT5 Initialization Failed"
        print(error_message)
        send_telegram_message(error_message)
        return False

    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    
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
    df['valid_hour'] = df['hour'].between(2, 18)
    df.drop(columns=['Time', 'hour'], inplace=True)

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

        df['bull_count'] = (df['Close'] > df['Open']).rolling(3).sum()
        df['bear_count'] = (df['Close'] < df['Open']).rolling(3).sum()
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
        0.5 * df['rsi_slope'] * 10 +
        0.5 * (df['obv_slope'] / (df['rolling_std'] + 1e-6))
    )

    def add_candlestick_features(df,
                            range_thr=0.002,   # ~0.2% for FX 5m; try 0.003–0.006 for XAUUSD
                            cluster_window=10,
                            cluster_lookback=200,
                            cluster_q=0.20):
        EPS = 1e-9

        # === 1) Range-bound Detection (percent-of-price) ===
        def is_range_bound(window=5, threshold=range_thr):
            max_high = df['High'].rolling(window, min_periods=window).max()
            min_low  = df['Low'].rolling(window,  min_periods=window).min()
            range_pct = (max_high - min_low) / (df['Close'].abs() + EPS)
            return (range_pct < threshold).astype(int)

        df['range_5']  = is_range_bound(5)
        df['range_10'] = is_range_bound(10)

        # === 2) Engulfing Patterns ===
        df['bullish_engulf'] = (
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Open']  < df['Close'].shift(1))
        ).astype(int)

        df['bearish_engulf'] = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (df['Open']  > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        ).astype(int)

        # === 3) Pin Bar Detection ===
        body = (df['Close'] - df['Open']).abs()
        rng  = (df['High'] - df['Low']).clip(lower=EPS)
        upper_wick = df['High'] - df[['Close','Open']].max(axis=1)
        lower_wick = df[['Close','Open']].min(axis=1) - df['Low']
        df['pin_bar'] = ((body / rng < 0.3) & ((upper_wick > 2*body) | (lower_wick > 2*body))).astype(int)

        # === 4) Inside / Outside Bars ===
        df['inside_bar']  = ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))).astype(int)

        # === 5) Clustering / Candle Compression (percentile-based, adaptive) ===
        def cluster_score_pctile(window=cluster_window, lookback=cluster_lookback, q=cluster_q):
            win_max = df['High'].rolling(window, min_periods=window).max()
            win_min = df['Low'].rolling(window,  min_periods=window).min()
            rng = win_max - win_min
            thr = rng.rolling(lookback, min_periods=lookback).quantile(q)
            return (rng <= thr).astype(int)

        df['cluster_10'] = cluster_score_pctile()

        # === Lag features (short memory) ===
        features_to_shift = ['range_5','range_10','bullish_engulf','bearish_engulf',
                            'pin_bar','inside_bar','outside_bar','cluster_10']
        for f in features_to_shift:
            for lag in range(1, 3):   # lags 1 and 2
                df[f'{f}_lag{lag}'] = df[f].shift(lag).fillna(0).astype(int)

        return df

    df = add_candlestick_features(df)

    def _atr14(df):
        prev_close = df['Close'].shift(1)
        tr = np.maximum(df['High'] - df['Low'],
                        np.maximum((df['High'] - prev_close).abs(),
                                (df['Low'] - prev_close).abs()))
        return tr.rolling(14, min_periods=1).mean()

    def detect_bos(df, lookback=60, swing_k=2, min_break_atr=0.25, confirm_with_close=True):
        """
        5m-friendly BOS:
        - lookback: bars to look back for last swing (60 ~= 5h)
        - swing_k: fractal strength (2 = 2 bars on each side)
        - min_break_atr: close/high must exceed last swing by this many ATRs
        - confirm_with_close: True = use Close to confirm, else use High/Low
        Creates df['BoS_Up'], df['BoS_Down'] in-place.
        """
        atr = _atr14(df).ffill()

        # Fractal swing points (vectorized, center-rolling)
        win = 2 * swing_k + 1
        sw_hi = (df['High'] == df['High'].rolling(win, center=True).max()).astype(int)
        sw_lo = (df['Low']  == df['Low'].rolling(win, center=True).min()).astype(int)

        # Last swing price memory (forward fill)
        last_sw_hi_price = df['High'].where(sw_hi.eq(1)).ffill()
        last_sw_lo_price = df['Low'].where(sw_lo.eq(1)).ffill()

        # Bars since last swing index
        idx = np.arange(len(df))
        last_sw_hi_idx = np.where(sw_hi.values==1, idx, np.nan)
        last_sw_lo_idx = np.where(sw_lo.values==1, idx, np.nan)
        last_sw_hi_idx = pd.Series(last_sw_hi_idx).ffill().values
        last_sw_lo_idx = pd.Series(last_sw_lo_idx).ffill().values

        bars_since_hi = idx - np.nan_to_num(last_sw_hi_idx, nan=-1)
        bars_since_lo = idx - np.nan_to_num(last_sw_lo_idx, nan=-1)

        # Confirmation price
        up_price  = df['Close'] if confirm_with_close else df['High']
        down_price= df['Close'] if confirm_with_close else df['Low']

        # BOS conditions: break by >= min_break_atr * ATR and last swing is within lookback
        bos_up = (
            (bars_since_hi > 0) & (bars_since_hi <= lookback) &
            (up_price >= (last_sw_hi_price + min_break_atr * atr))
        ).astype(int)

        bos_down = (
            (bars_since_lo > 0) & (bars_since_lo <= lookback) &
            (down_price <= (last_sw_lo_price - min_break_atr * atr))
        ).astype(int)

        df['BoS_Up'] = bos_up.fillna(0).astype(int)
        df['BoS_Down'] = bos_down.fillna(0).astype(int)
        return df


    from scipy.signal import find_peaks

    def detect_double_top_bottom(df, distance=18, tol_atr_mult=0.25, min_sep_bars=12):
        """
        5m-friendly DT/DB:
        - distance: min bars between detected peaks/troughs (18 ~= 90 min)
        - tol_atr_mult: two peaks are 'equal' if |Δprice| <= tol_atr_mult * ATR at the 2nd peak
        - min_sep_bars: additional guard for minimum spacing
        Creates df['Double_Top'], df['Double_Bottom'] in-place.
        """
        atr = _atr14(df).ffill()
        highs = df['High'].values
        lows  = df['Low'].values

        # Peaks and troughs
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)

        dt_mask = np.zeros(len(df), dtype=int)
        db_mask = np.zeros(len(df), dtype=int)

        # Double Top: adjacent peaks with similar heights (ATR tolerance)
        for i in range(1, len(peaks)):
            p1, p2 = peaks[i-1], peaks[i]
            if (p2 - p1) < min_sep_bars: 
                continue
            tol = tol_atr_mult * atr.iloc[p2]
            if np.abs(highs[p2] - highs[p1]) <= float(tol):
                dt_mask[p2] = 1

        # Double Bottom: adjacent troughs with similar depths (ATR tolerance)
        for i in range(1, len(troughs)):
            t1, t2 = troughs[i-1], troughs[i]
            if (t2 - t1) < min_sep_bars: 
                continue
            tol = tol_atr_mult * atr.iloc[t2]
            if np.abs(lows[t2] - lows[t1]) <= float(tol):
                db_mask[t2] = 1

        df['Double_Top'] = dt_mask
        df['Double_Bottom'] = db_mask
        return df

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
        return df

    def label_choch_from_bos(df, lookback=100):
        """
        Marks the first opposite-direction break as CHoCH.
        Requires df['BoS_Up'] and df['BoS_Down'] (0/1) already computed.
        - lookback: only consider flips where the previous BOS happened within N bars.
        Creates: CHoCH_Up, CHoCH_Down (0/1)
        """
        bos_dir = np.where(df['BoS_Up'].astype(int)==1, 1,
                np.where(df['BoS_Down'].astype(int)==1, -1, 0))
        idx = np.arange(len(df))

        last_dir = pd.Series(bos_dir).replace(0, np.nan).ffill().shift(1)
        last_idx = pd.Series(np.where(bos_dir!=0, idx, np.nan)).ffill().shift(1)

        within = (idx - last_idx) <= lookback

        choch_up = (df['BoS_Up'].astype(int)==1) & (last_dir==-1) & within
        choch_down = (df['BoS_Down'].astype(int)==1) & (last_dir== 1) & within

        df['CHoCH_Up'] = choch_up.astype(int)
        df['CHoCH_Down'] = choch_down.astype(int)
        return df

    def _atr(df, n=14):
        pc = df['Close'].shift(1)
        tr = np.maximum(df['High'] - df['Low'],
                np.maximum((df['High'] - pc).abs(), (df['Low'] - pc).abs()))
        return tr.rolling(n, min_periods=1).mean()

    def label_hh_hl_lh_ll(df, k=2):
        """
        Fractal swing highs/lows and next-swing relationship:
        HH/HL/LH/LL columns (0/1)
        k=2 -> swing needs 2 bars on each side.
        """
        win = 2*k + 1
        sh = (df['High'] == df['High'].rolling(win, center=True).max()).astype(int)
        sl = (df['Low']  == df['Low'].rolling(win, center=True).min()).astype(int)

        # record last swing highs/lows values & side
        last_sh_val = df['High'].where(sh.eq(1)).ffill()
        last_sl_val = df['Low'].where(sl.eq(1)).ffill()
        last_swing  = np.where(sh.eq(1),  1, np.where(sl.eq(1), -1, np.nan))
        last_swing  = pd.Series(last_swing).ffill()

        # next swing relative to the previous swing of the same type
        # build series of swing-only rows
        swings = df.assign(sw_type=np.where(sh.eq(1),1,np.where(sl.eq(1),-1,0)),
                        sw_price=np.where(sh.eq(1), df['High'],
                                    np.where(sl.eq(1), df['Low'], np.nan)))
        swings = swings[swings['sw_type']!=0].copy()

        # compare consecutive swings
        swings['prev_type']  = swings['sw_type'].shift(1)
        swings['prev_price'] = swings['sw_price'].shift(1)

        swings['HH'] = ((swings['sw_type']==1) & (swings['prev_type']==1) &
                        (swings['sw_price'] > swings['prev_price'])).astype(int)
        swings['HL'] = ((swings['sw_type']==-1) & (swings['prev_type']==-1) &
                        (swings['sw_price'] > swings['prev_price'])).astype(int)
        swings['LH'] = ((swings['sw_type']==1) & (swings['prev_type']==1) &
                        (swings['sw_price'] < swings['prev_price'])).astype(int)
        swings['LL'] = ((swings['sw_type']==-1) & (swings['prev_type']==-1) &
                        (swings['sw_price'] < swings['prev_price'])).astype(int)

        # map back to full df index
        df['HH'] = 0; df['HL'] = 0; df['LH'] = 0; df['LL'] = 0
        df.loc[swings.index, ['HH','HL','LH','LL']] = swings[['HH','HL','LH','LL']].values
        return df

    def detect_sfp(df, k=2, tol_atr_mult=0.2, confirm_with_close=True):
        """
        SFP_Up: takes out previous swing high by wick, closes back below (uptrap).
        SFP_Down: takes out previous swing low, closes back above (downtrap).
        """
        atr = _atr(df).ffill()
        win = 2*k + 1
        swing_high = (df['High'] == df['High'].rolling(win, center=True).max())
        swing_low  = (df['Low']  == df['Low'].rolling(win, center=True).min())

        last_sh = df['High'].where(swing_high).ffill()
        last_sl = df['Low'].where(swing_low).ffill()

        tol_up = tol_atr_mult * atr
        tol_dn = tol_atr_mult * atr

        # price pokes above last swing high by >= tol, but fails
        if confirm_with_close:
            sfp_up = (df['High'] >= last_sh + tol_up) & (df['Close'] < last_sh)
            sfp_dn = (df['Low']  <= last_sl - tol_dn) & (df['Close'] > last_sl)
        else:
            sfp_up = (df['High'] >= last_sh + tol_up) & (df['High']  < last_sh + 2*tol_up)
            sfp_dn = (df['Low']  <= last_sl - tol_dn) & (df['Low']   > last_sl - 2*tol_dn)

        df['SFP_Up'] = sfp_up.fillna(False).astype(int)
        df['SFP_Down'] = sfp_dn.fillna(False).astype(int)
        return df

    def detect_fvg(df, min_atr_mult=0.1):
        """
        3-candle FVG:
        Bull FVG when: Low[n] > High[n-2]  (gap) and body is impulsive
        Bear FVG when: High[n] < Low[n-2]
        min_atr_mult filters tiny gaps.
        """
        atr = _atr(df).ffill()
        high_m2 = df['High'].shift(2)
        low_m2  = df['Low'].shift(2)

        bull_gap = (df['Low'] > high_m2)
        bear_gap = (df['High'] < low_m2)

        # width threshold
        bull_w = (df['Low'] - high_m2)
        bear_w = (low_m2 - df['High'])

        bull_fvg = bull_gap & (bull_w >= min_atr_mult * atr)
        bear_fvg = bear_gap & (bear_w >= min_atr_mult * atr)

        df['FVG_Up'] = bull_fvg.fillna(False).astype(int)
        df['FVG_Down'] = bear_fvg.fillna(False).astype(int)
        return df

    def detect_break_retest(df, lookahead=12, tol_atr_mult=0.25):
        """
        After a BOS, mark a retest of the breakout level within 'lookahead' bars.
        Requires df['BoS_Up'] / df['BoS_Down'] (0/1).
        """
        atr = _atr(df).ffill()
        ret_up = np.zeros(len(df), dtype=int)
        ret_dn = np.zeros(len(df), dtype=int)

        # breakout levels: last swing that was broken
        last_high = df['High'].cummax()  # simple proxy; replace with stored swing if you have it
        last_low  = (-df['Low']).cummax() * -1

        for i in range(len(df)):
            if df['BoS_Up'].iloc[i] == 1:
                level = last_high.iloc[i]
                tol = float(tol_atr_mult * atr.iloc[i])
                hi = min(len(df), i + lookahead + 1)
                # retest if price trades back to level +/- tol
                if (df['Low'].iloc[i+1:hi] <= level + tol).any():
                    ret_up[i] = 1

            if df['BoS_Down'].iloc[i] == 1:
                level = last_low.iloc[i]
                tol = float(tol_atr_mult * atr.iloc[i])
                hi = min(len(df), i + lookahead + 1)
                if (df['High'].iloc[i+1:hi] >= level - tol).any():
                    ret_dn[i] = 1

        df['Retest_Up'] = ret_up
        df['Retest_Down'] = ret_dn
        return df

    df = detect_bos(df, lookback=60, swing_k=2, min_break_atr=0.25, confirm_with_close=True)
    df = detect_double_top_bottom(df, distance=18, tol_atr_mult=0.25, min_sep_bars=12)
    df = candles_since_bollinger_touch(df)
    df = label_choch_from_bos(df, lookback=100)
    df = label_hh_hl_lh_ll(df, k=2)
    df = detect_sfp(df, k=2, tol_atr_mult=0.2, confirm_with_close=True)
    df = detect_fvg(df, min_atr_mult=0.1)
    df = detect_break_retest(df, lookahead=12, tol_atr_mult=0.25)

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
   


# ===== CONFIG =====
TRADE_FILE = "trades.json"
LOCK_FILE  = TRADE_FILE + ".lock"

# ===== FILE LOCK (Unix/WSL & Windows) =====
try:
    import fcntl
    def _lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def _unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:
    import msvcrt
    def _lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def _unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

@contextmanager
def file_lock(lock_path=LOCK_FILE):
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as lf:
        _lock_file(lf)
        try:
            yield
        finally:
            _unlock_file(lf)

# ===== SAFE LOAD / ATOMIC WRITE / UPSERT =====
def load_trades():
    """Safe loader. If file missing or corrupted, return {} (and try .bak)."""
    if not os.path.exists(TRADE_FILE):
        return {}
    try:
        with open(TRADE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        bak = TRADE_FILE + ".bak"
        if os.path.exists(bak):
            try:
                with open(bak, "r", encoding="utf-8") as fb:
                    return json.load(fb)
            except Exception:
                pass
        # fall back to empty (and keep corrupted file for debugging)
        return {}

def _atomic_write_json(path, data):
    """Write JSON atomically: tmp -> fsync -> replace; also keep .bak."""
    tmp_path = path + ".tmp"
    # backup current file under lock to .bak (best-effort)
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except Exception:
            pass
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)  # atomic on same filesystem

def safe_upsert_trade(trade_id, trade_data):
    """
    Read-modify-write under a file lock, with atomic replace.
    Prevents concurrent writers from truncating/overwriting each other.
    """
    with file_lock():
        trades = load_trades()
        trades[str(trade_id)] = trade_data
        _atomic_write_json(TRADE_FILE, trades)

# ===== LEGACY APIS (kept for compatibility) =====
def save_trades(trades):
    """Legacy writer (not recommended for live). Left for compatibility."""
    # with file_lock():  # (You could lock even here if you choose to keep it)
    #     _atomic_write_json(TRADE_FILE, trades)
    with open(TRADE_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=4)

def update_trade(trade_id, trade_data):
    """Use safe upsert instead of whole-file rewrite."""
    safe_upsert_trade(trade_id, trade_data)

def get_trade(trade_id):
    trades = load_trades()
    return trades.get(str(trade_id), None)

# ===== YOUR manage_trades, modified to call safe_upsert_trade() =====
def manage_trades(symbol, df_buy, df_sell):
    """Enhanced trade management using the correct directional features with meta classifier TP adjustment + timed breakeven."""
    trades = load_trades()
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[ERROR] Could not fetch symbol info for {symbol}")
        return

    point = symbol_info.point
    digits = symbol_info.digits
    positions = mt5.positions_get(symbol=symbol)

    def _round_price(p: float) -> float:
        return round(p, digits)

    def _improves_sl(is_buy: bool, new_sl: float, current_sl: float | None) -> bool:
        """Return True if new_sl tightens risk (never loosens)."""
        if current_sl is None or current_sl == 0:
            return True
        return (new_sl > current_sl) if is_buy else (new_sl < current_sl)

    def _valid_sl_vs_market(is_buy: bool, new_sl: float, current_price: float) -> bool:
        """Ensure SL is on the correct side of market to avoid immediate stop."""
        if is_buy:
            return new_sl < (current_price - point)
        else:
            return new_sl > (current_price + point)

    if positions:
        for position in positions:
            trade_id = str(position.ticket)
            is_buy = position.type == mt5.ORDER_TYPE_BUY
            df = df_buy if is_buy else df_sell

            latest_5m = df.iloc[-2]
            latest_5m_sma9 = latest_5m["SMA_9"]
            valid_hour = latest_5m["valid_hour"]

            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"[ERROR] No tick for {symbol}")
                continue
            current_price = tick.ask if is_buy else tick.bid

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
                    "Target Hit": False,
                    # New fields
                    "rr_unit": None,            # lock initial RR distance
                    "be_stage": 0               # 0:none, 1: -0.5RR applied, 2: +0.2RR applied
                }
                # persist new trade immediately (append-safe)
                safe_upsert_trade(trade_id, trade_data)

            # Lock RR to the initial stop distance so later SL edits don't break RR math
            if not trade_data.get("rr_unit"):
                initial_rr_unit = abs(entry_price - (current_sl or entry_price))
                if initial_rr_unit == 0:
                    print(f"[ERROR] Initial RR Unit is zero for trade {trade_id}. Skipping.")
                    continue
                trade_data["rr_unit"] = initial_rr_unit

            base_rr_unit = trade_data["rr_unit"]  # always use this for RR/TP math

            current_time = datetime.datetime.now(datetime.timezone.utc)
            trade_duration = (current_time - entry_time).total_seconds() / 60

            # === Meta Classifier-based TP Adjustment (uses base_rr_unit) ===
            meta_class = trade_data.get("meta_class")
            if meta_class is not None:
                rr_targets = {
                    0: 0.0,
                    1: 1.05,
                    2: 1.05,
                }
                target_rr = rr_targets.get(meta_class, 1.0)
                expected_tp = entry_price + (target_rr * base_rr_unit) if is_buy else entry_price - (target_rr * base_rr_unit)
                rounded_expected_tp = _round_price(expected_tp)
                tp_deviation = abs((current_tp or 0) - rounded_expected_tp)

                if tp_deviation > 0.1 * base_rr_unit:
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

            # === Calculate RR and update stats (uses base_rr_unit) ===
            price_diff = (current_price - entry_price) if is_buy else (entry_price - current_price)
            current_rr = price_diff / base_rr_unit
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
            if max_rr >= 1.04 and not trade_data.get("Target Hit"):
                trade_data["hit_1:0"] = True
                trade_data["milestone_comment"] = "🎯 Hit 1.0RR"
                trade_data["Target Hit"] = True
                send_telegram_message(f"🎯 Trade {symbol} (ID: {trade_id}) Hit 1.0RR Target! Current RR: {current_rr:.2f}")

            # === Timed Breakeven (won't break RR math; uses base_rr_unit) ===
            be_stage = int(trade_data.get("be_stage", 0))

            def _try_move_sl(target_rr_for_sl: float, stage_to_set: int, label: str):
                nonlocal current_sl  # track if we changed it
                new_sl = entry_price + (target_rr_for_sl * base_rr_unit) if is_buy else entry_price - (target_rr_for_sl * base_rr_unit)
                new_sl = _round_price(new_sl)

                # Only tighten SL, and keep it on the correct side of market
                if not _improves_sl(is_buy, new_sl, current_sl):
                    return False
                if not _valid_sl_vs_market(is_buy, new_sl, current_price):
                    return False

                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": int(trade_id),
                    "sl": new_sl,
                    "tp": current_tp,  # don't touch TP here
                    "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                    "magic": 123456,
                    "comment": label,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    trade_data["sl"] = new_sl
                    trade_data["be_stage"] = stage_to_set
                    # Update local for subsequent decisions
                    current_sl = new_sl
                    send_telegram_message(f"🛡️ {label}: {symbol} trade {trade_id} SL → {new_sl:.5f}")
                    return True
                else:
                    print(f"[WARNING] Failed SL move ({label}) for {trade_id}: {res.comment}")
                    return False

            # Stage 1: after 15 minutes → SL to -0.5 RR (reduce risk)
            # if trade_duration >= 60 and be_stage < 1:
            #     _try_move_sl(target_rr_for_sl=-0.5, stage_to_set=1, label="Breakeven Stage 1 (-0.5RR)")

            # # Stage 2: after 30 minutes → SL to +0.2 RR (lock small profit)
            # if trade_duration >= 120 and be_stage < 2:
            #     _try_move_sl(target_rr_for_sl=+0.2, stage_to_set=2, label="Breakeven Stage 2 (+0.2RR)")

            # === Exit Rules ===
            exit_triggered = False

            # # Rule 1: SMA9 crossover against position
            # if (is_buy and latest_5m["Close"] < latest_5m_sma9) or (not is_buy and latest_5m["Close"] > latest_5m_sma9):
            #     if current_rr >= 0.5 and trade_duration >= 60:
            #         close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
            #         trade_data["close_reason"] = "SMA9 crossover against position"
            #         exit_triggered = True

            # Rule: Negative Drop (after good run)
            if not exit_triggered and max_rr >= 0.9 and current_rr <= 0.5:
                close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                trade_data["close_reason"] = "Negative Drop"
                exit_triggered = True

            # Rule 2: Time decay (4 hours)
            if not exit_triggered and trade_duration >= 240:
                close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                trade_data["close_reason"] = "Time decay exit (4h)"
                exit_triggered = True
                send_telegram_message(f"📉 Time decay exit for {symbol} after 4 hours")

            # # Rule 3: End-of-day guard
            if not exit_triggered and not valid_hour:
                close_trade(trade_id, symbol, "buy" if is_buy else "sell", current_price, profit)
                trade_data["close_reason"] = "📉 Closed all trade for the day"
                exit_triggered = True
                send_telegram_message(f"📉 Closed all trade for the day")

            # persist each step safely instead of rewriting the whole file
            if exit_triggered:
                trade_data.update({
                    "exit_price": current_price,
                    "exit_time": current_time.isoformat(),
                    "profit": profit
                })
                # trades[trade_id] = trade_data
                # save_trades(trades)
                safe_upsert_trade(trade_id, trade_data)
                continue

            # trades[trade_id] = trade_data
            # save_trades(trades)
            safe_upsert_trade(trade_id, trade_data)

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
                    # trades[position_id] = trade_data
                    # print(f"Updated closed trade {position_id} with exit data")
                    safe_upsert_trade(position_id, trade_data)

    # save_trades(trades)  # ❌ do not bulk-rewrite; each update is persisted atomically

    # === Sync broker history for closed deals (net profit from MT5) ===
    def _account_money_digits():
        ai = mt5.account_info()
        return getattr(ai, "currency_digits", 2) if ai else 2

    def _round_money(x: float) -> float:
        return round(float(x), _account_money_digits())

    # Some brokers produce multiple "exit-like" entries on a position.
    # Deal.entry values: 0=IN, 1=OUT, 2=INOUT (close by reversal), 3=OUT_BY (closed by)
    EXIT_ENTRIES = {1, 2, 3}

    # Look back far enough to catch anything still in your trades book.
    hist_from = datetime.datetime.now() - datetime.timedelta(days=7)
    hist_to   = datetime.datetime.now()

    deals = mt5.history_deals_get(hist_from, hist_to)
    if deals is None:
        print("[WARNING] history_deals_get returned None")
    else:
        # Group deals by position_id for quick lookup
        by_pos = {}
        for d in deals:
            by_pos.setdefault(d.position_id, []).append(d)

        # Update any open-in-JSON (no exit_time yet) that were closed by broker
        for position_id, trade_data in list(trades.items()):
            if trade_data.get("exit_time"):
                continue  # already finalized

            pid = int(position_id)
            pos_deals = by_pos.get(pid, [])
            if not pos_deals:
                continue

            # Filter to exit-side deals only
            exits = [d for d in pos_deals if int(getattr(d, "entry", -1)) in EXIT_ENTRIES]
            if not exits:
                continue

            # Sum realized P/L to match MT5 "net": profit + commission + swap (+ fee if present)
            net_profit = 0.0
            for d in exits:
                profit = float(getattr(d, "profit", 0.0))
                commission = float(getattr(d, "commission", 0.0))
                swap = float(getattr(d, "swap", 0.0))
                # Some brokers have 'fee' too; guard with getattr
                fee = float(getattr(d, "fee", 0.0))
                net_profit += (profit + commission + swap + fee)

            # Use the last exit deal as the canonical exit
            last_exit = max(
                exits,
                key=lambda x: getattr(x, "time_msc", 0) or getattr(x, "time", 0)
            )

            exit_price = float(getattr(last_exit, "price", trade_data.get("exit_price") or 0.0))
            exit_ts = getattr(last_exit, "time", None)
            if exit_ts is None:
                # fall back to now if time is missing (rare)
                exit_dt = datetime.datetime.now(datetime.timezone.utc)
            else:
                exit_dt = datetime.datetime.fromtimestamp(exit_ts, tz=datetime.timezone.utc)

            # Overwrite with broker values for 1:1 parity with MT5
            trade_data.update({
                "exit_price": exit_price,
                "exit_time": exit_dt.isoformat(),
                "profit": _round_money(net_profit),
                "close_reason": trade_data.get("close_reason") or "Closed by broker",
                "profit_source": "broker"  # new flag so you know it's synced
            })

            safe_upsert_trade(position_id, trade_data)

def _round_volume_to_step(symbol: str, volume: float) -> float:
    """
    Round volume to the symbol's allowed step (safer than fixed 2 decimals).
    Falls back to 2dp if step not available.
    """
    info = mt5.symbol_info(symbol)
    if not info or not info.volume_step:
        return round(volume, 2)
    step = info.volume_step
    v = max(info.volume_min or step, min(volume, info.volume_max or volume))
    steps = round(v / step)
    return round(steps * step, 2)


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

    # Compute correct side & refresh price if needed
    is_buy = trade_type.lower() == "buy"
    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        market_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        current_price = market_price  # broker will still execute at market
    # else: keep passed current_price

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": int(trade_id),
        "price": current_price,
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

            # save_trades(trades)
            safe_upsert_trade(trade_id, trade_data)

            send_telegram_message(
                f"✅ Closed {trade_type.upper()} {symbol} trade (ID: {trade_id}) at {result.price:.5f} with profit: {profit:.2f}"
            )
    else:
        error_code, error_msg = mt5.last_error()
        print(f"❌ Failed to close trade {trade_id}. Error: {error_code}, Message: {error_msg}")
        logger.error(f"❌ Failed to close trade {trade_id}. Error code: {error_code}, message: {error_msg}")
        send_telegram_message(f"❌ Failed to close trade {trade_id}. Error: {error_code}, Message: {error_msg}")

def _round_price(symbol: str, p: float) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return p
    return round(p, info.digits)

def _round_volume_to_step(symbol: str, volume: float) -> float:
    info = mt5.symbol_info(symbol)
    if not info or not info.volume_step:
        return round(volume, 2)
    step = info.volume_step
    v = max(info.volume_min or step, min(volume, info.volume_max or volume))
    steps = round(v / step)
    # keep two decimals typical for FX; adjust if your broker differs
    return round(steps * step, 2)

def _current_price_for_side(symbol: str, trade_type: str) -> float | None:
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None
    return tick.ask if trade_type.lower() == "buy" else tick.bid

def _within_spread_limit(symbol: str) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    return (tick.ask - tick.bid) <= get_spread_limit(symbol)

def _find_position_ticket_after_order(symbol: str, order_ticket: int, trade_type: str, magic=123456, tries=15, sleep_s=0.2):
    """
    Try to map the just-sent market order to a live position ticket.
    Works for both hedging & netting accounts.
    """
    # direct lookup by ticket (some brokers map order->position in hedging)
    for _ in range(tries):
        pos = mt5.positions_get(ticket=order_ticket)
        if pos:
            return pos[0].ticket
        time.sleep(sleep_s)

    # fallback: find by symbol + magic + type near now
    t_end = time.time() + tries * sleep_s
    side = mt5.ORDER_TYPE_BUY if trade_type.lower() == "buy" else mt5.ORDER_TYPE_SELL
    while time.time() < t_end:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            # pick the latest that matches magic & side
            candidates = [p for p in positions if getattr(p, "magic", 0) == magic and p.type == side]
            if candidates:
                # most recent by time_msc
                candidates.sort(key=lambda p: getattr(p, "time_msc", 0), reverse=True)
                return candidates[0].ticket
        time.sleep(sleep_s)
    return None


# ================== YOUR FLOW (upgraded) ==================

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
    # round SL/TP to symbol digits up front
    sl_price = _round_price(symbol, sl_price)
    tp_price = _round_price(symbol, tp_price)

    if (trade_type == "buy" and price >= trigger_price) or (trade_type == "sell" and price <= trigger_price):
        # re-check lock & spread *right before* sending
        if is_symbol_locked(symbol) or not _within_spread_limit(symbol):
            return
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
        # bail if something else already took the symbol
        if is_symbol_locked(symbol) or has_open_trade(symbol, trade_type):
            return

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
            # re-check lock & spread right before sending
            if is_symbol_locked(symbol) or not _within_spread_limit(symbol):
                return
            place_market_order(symbol, trade_type, price, sl_price, tp_price,
                               regress_pred, classifier_conf, meta_conf, funx, meta_class)
            return

        time.sleep(interval)
        waited += interval

    send_telegram_message(f"⌛ Trade expired for {symbol} — trigger not hit.")


def place_market_order(symbol, trade_type, price, sl_price, tp_price,
                       regress_pred, classifier_conf, meta_conf, funx, meta_class):
    # refresh price just in case
    live_price = _current_price_for_side(symbol, trade_type)
    if live_price is not None:
        price = live_price

    acct = mt5.account_info()
    if acct is None:
        send_telegram_message("❌ Could not read account info.")
        return

    sl_dist = abs(price - sl_price)
    if sl_dist <= 0:
        send_telegram_message("❌ Invalid SL distance.")
        return

    # lot sizing + volume step
    lot_raw = calculate_lot_size(acct.balance, sl_dist, symbol, risk_percent=2)
    lot = _round_volume_to_step(symbol, lot_raw)
    if lot <= 0:
        send_telegram_message("❌ Computed lot size is zero.")
        return

    # sanity on stop sides
    is_buy = trade_type.lower() == "buy"
    if is_buy and not (sl_price < price and tp_price > price):
        send_telegram_message("❌ SL/TP not on correct sides for BUY.")
        return
    if (not is_buy) and not (sl_price > price and tp_price < price):
        send_telegram_message("❌ SL/TP not on correct sides for SELL.")
        return

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,  # points
        "magic": 123456,
        "comment": "InstantTriggerOrder",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        msg = f"✅ {trade_type.upper()} order placed for {symbol} @ {price:.5f} (lot {lot})"
        send_telegram_message(msg)

        # lock quickly to prevent duplicates
        lock_symbol(symbol, duration=300)

        # optional: in netting mode this can change the just-opened position; keep if you intend that
        close_opposite_trades(trade_type, symbol)

        # try to map to a real position ticket
        pos_ticket = _find_position_ticket_after_order(symbol, res.order, trade_type, magic=123456)
        record_id = str(pos_ticket if pos_ticket is not None else res.order)

        trade_data = {
            "id": record_id,
            "order_ticket": int(res.order),
            "deal_ticket": int(getattr(res, "deal", 0) or 0),
            "position_ticket": int(pos_ticket) if pos_ticket is not None else None,
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
            "equity_at_entry": acct.equity,
            "volume": float(lot)
        }

        # update_trade(str(res.order), trade_data)
        safe_upsert_trade(record_id, trade_data)

    else:
        err = mt5.last_error()
        msg = f"❌ MARKET order failed for {symbol}: {res.comment if res else err}"
        send_telegram_message(msg)

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
    raw_close_volume = volume * float(fraction)
    close_volume = _round_volume_to_step(symbol, raw_close_volume)

    if close_volume <= 0:
        print(f"❌ Invalid close volume: {close_volume}")
        logger.error(f"Invalid close volume: {close_volume}")
        return

    is_buy = trade_type.lower() == "buy"
    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

    tick = mt5.symbol_info_tick(symbol)
    if tick:
        market_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        current_price = market_price

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
            # NOTE: For partial closes, we generally *don't* set final exit fields.
            # Keep last partial info and accumulate fraction.
            trade_data["last_partial_price"] = result.price
            trade_data["last_partial_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            trade_data["last_partial_fraction"] = float(fraction)
            trade_data["closed_fraction"] = float(trade_data.get("closed_fraction", 0.0) + fraction)
            trade_data["profit"] = profit  # running profit
            trade_data["balance_at_exit"] = account_info.balance if account_info else None
            trade_data["equity_at_exit"] = account_info.equity if account_info else None

            # If fully closed due to partial consuming whole volume, set exit fields
            if abs(trade_data["closed_fraction"] - 1.0) < 1e-6:
                trade_data["exit_price"] = result.price
                trade_data["exit_time"] = trade_data["last_partial_time"]
                trade_data["close_reason"] = trade_data.get("close_reason") or "Fully closed via partials"

            # save_trades(trades)
            safe_upsert_trade(trade_id, trade_data)

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
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"[ERROR] No tick for {symbol} when closing opposite trades")
                continue
            current_price = tick.bid if trade_type_str == "sell" else tick.ask
            close_trade(
                trade_id=pos.ticket,
                symbol=symbol,
                trade_type=trade_type_str,
                current_price=current_price,
                profit=pos.profit
            )



def extract_features(df):
    latest_data_df = df.iloc[-2]

    nan_features = [feat for feat in feature_list if pd.isna(latest_data_df[feat])]
    
    if nan_features:
        message = f"🚨 NaN detected in features: {', '.join(nan_features)}"
        print(message)
        send_telegram_message(message)
        return None

    feature_values = [float(latest_data_df[feat]) for feat in feature_list]

    return feature_values

spread_limits_low = {
        "gbpusd": 0.00050, 
        "usdcad": 0.00050
}
spread_limits_high = {
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
        t_logger.info(f"{symbol}: buy skipped — not accepted by ML stack")
        return

    rr_pred = result.get("reg_pred")
    clf_probs = result.get("classifier_probs", {})
    meta_class = result.get("final_class")
    meta_probs = result.get("class_probabilities", [])
    trade_type = result.get("trade_type", "buy")

    if rr_pred is None or meta_class is None:
        t_logger.info(f"{symbol}: buy skipped — incomplete ML prediction result")
        return

    if symbol not in RR_THRESHOLDS_BUY:
        raise KeyError(f"❌ No RR threshold defined for symbol '{symbol}'")

    rr_thresh = RR_THRESHOLDS_BUY[symbol]

    prev = df.iloc[-3]
    curr = df.iloc[-2]
    pair_code = curr['pair']
    log_returns = curr['log_returns']
    sma_200 = curr['SMA_200']
    low = curr['Low'] 

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"❌ No spread limit defined for pair '{pair_code}'")

    if not curr.get("valid_hour", False):
        return

    if rr_pred < rr_thresh:
        return

   
    if log_returns <= 0.001 and  sma_200 >=  low:
        return

    # if meta_class == 1:
    #     return 
    
    META_CONF_THRESH = {1: 0.80, 2: 0.70}

    if meta_class in META_CONF_THRESH:
        if meta_probs[meta_class] < META_CONF_THRESH[meta_class]:
            return  

    # === Trade Prices ===
    trigger_price = curr["Close"]
    sl_price = curr["stop_loss_price"]
    sl_dist = trigger_price - sl_price
    tp_price = trigger_price + (sl_dist * meta_class)  # final_class = TP multiplier

    spread_low = spread_limits_low[pair_code]
    spread_high = spread_limits_high[pair_code]

    if sl_dist < spread_low or sl_dist > spread_high:
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
        f"{'-'*42} TRADE\n"
        f"✅ ML BUY SIGNAL (5m) {symbol}\n"
        f"🎯 Entry      : {curr['entry_price']:.5f}\n"
        f"🛡️ StopLoss   : {sl_price:.5f}\n"
        f"📈 TakeProfit : {tp_price:.5f} (x{meta_class})\n"
        f"🤖 Pred RR    : {rr_pred:.2f}\n"
        f"📊 Classifier Probs:\n"
        f"   ├─ 1:1 : {clf_probs.get('clf_1_1_prob', 0):.2f}\n"
        f"   ├─ 1:2 : {clf_probs.get('clf_1_2_prob', 0):.2f}\n"
        f"📊 Meta Probabilities: {meta_class}\n"
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
    t_logger.info(msg)

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
        t_logger.info(f"{symbol}: sell skipped — not accepted by ML stack")
        return

    rr_pred = result.get("reg_pred")
    clf_probs = result.get("classifier_probs", {})
    meta_class = result.get("final_class")
    meta_probs = result.get("class_probabilities", [])
    trade_type = result.get("trade_type", "sell")

    if rr_pred is None or meta_class is None:
        t_logger.warning(f"{symbol}: sell skipped — incomplete ML prediction result")
        return

    if symbol not in RR_THRESHOLDS_SELL:
        raise KeyError(f"❌ No RR threshold defined for symbol '{symbol}'")

    rr_thresh = RR_THRESHOLDS_SELL[symbol]

    prev = df.iloc[-3]
    curr = df.iloc[-2]
    pair_code = curr['pair']
    log_returns = curr['log_returns'] 
    sma_200 = curr['SMA_200']
    high = curr['High'] 

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"❌ No spread limit defined for pair '{pair_code}'")

    if not curr.get("valid_hour", False):
        return

    if rr_pred < rr_thresh:
        return

    if log_returns >= -0.0001 and sma_200 <= high:
        return

    # if meta_class == 1:
    #     return 
        
    META_CONF_THRESH = {1: 0.80, 2: 0.70}
    if meta_class in META_CONF_THRESH:
        if meta_probs[meta_class] < META_CONF_THRESH[meta_class]:
            return  


    # === Trade Prices ===
    trigger_price = curr["Close"]
    sl_price = curr["stop_loss_price"]
    sl_dist = sl_price - trigger_price
    tp_price = trigger_price - (sl_dist * meta_class)  # final_class = TP multiplier

    spread_low = spread_limits_low[pair_code]
    spread_high = spread_limits_high[pair_code]

    if sl_dist < spread_low or sl_dist > spread_high:
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
        f"{'-'*42} TRADE\n"   
        f"✅ ML SELL SIGNAL (5m) {symbol}\n"
        f"🎯 Entry      : {curr['entry_price']:.5f}\n"
        f"🛡️ StopLoss   : {sl_price:.5f}\n"
        f"📈 TakeProfit : {tp_price:.5f} (x{meta_class})\n"
        f"🤖 Pred RR    : {rr_pred:.2f}\n"
        f"📊 Classifier Probs:\n"
        f"   ├─ 1:1 : {clf_probs.get('clf_1_1_prob', 0):.2f}\n"
        f"   ├─ 1:2 : {clf_probs.get('clf_1_2_prob', 0):.2f}\n"
        f"📊 Meta Probabilities: {meta_class}\n"
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
    t_logger.info(msg)


def send_telegram_message(text):
    """Send a message to Telegram"""
    # print(f"\n📨 Sending message to Telegram: {text}")
    print(f"\n📨 Sending message to Telegram")
    
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    
    try:
        response = requests.post(TELEGRAM_URL, json=payload)
        
        if response.status_code == 200:
            print("✅ Telegram message sent successfully!")
        else:
            print(f"❌ Telegram error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"🚨 Telegram request failed: {e}")



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
def get_predictions(
    features: pd.DataFrame,
    symbol: str,
    trade_type="buy",
    save_csv=False,
    csv_path="latest_predictions.csv",
    max_rows=20000,
    thresh_meta: float | None = None,     # e.g., 0.85 if you want a confidence gate
    metadata: dict | None = None          # pass your loaded model_metadata.pkl if available
):
    """
    Full stacked prediction pipeline (batch-safe):
    - Base Classifiers -> Regressor -> Meta
    - Returns a dict (single row) or a list of dicts (batch)
    """
    if not isinstance(features, pd.DataFrame):
        print(f"❌ 'features' must be a DataFrame for {symbol}")
        return None
    if trade_type not in ("buy", "sell"):
        print(f"❌ Invalid trade_type: '{trade_type}'")
        return None

    # Ensure 'pair' dtype matches training (important for LightGBM categorical splits)
    if 'pair' in features.columns:
        if metadata and 'pair_categories' in metadata:
            features = features.copy()
            features['pair'] = pd.Categorical(features['pair'], categories=metadata['pair_categories'])
        else:
            # Best-effort: cast to categorical to avoid object dtype
            features = features.copy()
            features['pair'] = features['pair'].astype('category')

    # Load boosters
    try:
        models = {
            "clf_1_1_prob": loaded_models[f"clf_{trade_type}_1.1"],
            "clf_1_2_prob": loaded_models[f"clf_{trade_type}_1.2"],
            "reg_pred":     loaded_models[f"reg_{trade_type}"],
            "meta":         loaded_models[f"meta_{trade_type}"]
        }
    except KeyError as e:
        print(f"❌ Missing models for {trade_type.upper()}: {e}")
        return None

    n = len(features)

    # --- 1) Base classifier probabilities (vectorized) ---
    clf_probs = {}
    for key in ("clf_1_1_prob", "clf_1_2_prob"):
        booster = models[key]
        cols = booster.feature_name()
        X = features.reindex(columns=cols, fill_value=0)

        if X.shape[1] != len(cols):
            missing = [c for c in cols if c not in X.columns]
            extra   = [c for c in X.columns if c not in cols]
            print(f"⚠️ {key} feature mismatch. Expected {len(cols)} got {X.shape[1]}. "
                  f"Missing: {missing[:5]}... Extra: {extra[:5]}...")
        # LightGBM Booster.predict on binary returns prob for class 1
        p = booster.predict(X)
        p = np.asarray(p).reshape(-1)  # shape (n,)
        if p.shape[0] != n:
            raise RuntimeError(f"{key} produced {p.shape[0]} probs for {n} rows.")
        clf_probs[key] = p

    # --- 2) Regressor (needs base probs as features) ---
    reg_booster = models["reg_pred"]
    reg_cols = reg_booster.feature_name()
    X_reg = features.copy()
    X_reg["clf_1_1_prob"] = clf_probs["clf_1_1_prob"]
    X_reg["clf_1_2_prob"] = clf_probs["clf_1_2_prob"]
    X_reg = X_reg.reindex(columns=reg_cols, fill_value=0)
    rr_pred = reg_booster.predict(X_reg).reshape(-1)

    # --- 3) Meta (multiclass) ---
    meta_booster = models["meta"]
    meta_cols = meta_booster.feature_name()
    X_meta = features.copy()
    X_meta["clf_1_1_prob"] = clf_probs["clf_1_1_prob"]
    X_meta["clf_1_2_prob"] = clf_probs["clf_1_2_prob"]
    X_meta["reg_pred"]      = rr_pred
    X_meta = X_meta.reindex(columns=meta_cols, fill_value=0)

    meta_raw = meta_booster.predict(X_meta)  # (n, C) or (C,) if n==1
    meta_raw = np.asarray(meta_raw)
    if meta_raw.ndim == 1:   # single row -> (C,)
        meta_raw = meta_raw.reshape(1, -1)
    meta_class = meta_raw.argmax(axis=1).astype(int)          # 0=Reject, 1=1:1, 2=1:2
    meta_conf  = meta_raw.max(axis=1)

    # Optional confidence gate
    if thresh_meta is not None:
        meta_class = np.where(meta_conf >= thresh_meta, meta_class, 0)

    # --- 4) Build outputs ---
    out = []
    for i in range(n):
        row = {
            "symbol": symbol,
            "trade_type": trade_type,
            "accepted": bool(meta_class[i] > 0),
            "final_class": int(meta_class[i]),
            "class_probabilities": meta_raw[i].tolist(),
            "classifier_probs": {
                "clf_1_1_prob": float(clf_probs["clf_1_1_prob"][i]),
                "clf_1_2_prob": float(clf_probs["clf_1_2_prob"][i]),
            },
            "reg_pred": float(rr_pred[i]),
            "meta_conf": float(meta_conf[i]),
        }
        out.append(row)

    # Console summary (single row)
    if n == 1:
        r = out[0]
        cls = r["final_class"]
        print(
            f"{'✅' if r['accepted'] else '🔕'} "
            f"{symbol} {trade_type.upper()} → Class {cls} | "
            f"Conf {r['meta_conf']:.2f} | R:R {r['reg_pred']:.2f} | "
            f"p1R {r['classifier_probs']['clf_1_1_prob']:.2f} p2R {r['classifier_probs']['clf_1_2_prob']:.2f}"
        )

    # --- 5) Optional CSV logging (first row or all rows) ---
    if save_csv:
        try:
            with csv_write_lock:
                # Prepare new log rows
                rows = []
                for i in range(n):
                    base = features.iloc[i].copy()
                    base["timestamp"] = pd.Timestamp.utcnow()
                    base["symbol"] = symbol
                    base["trade_type"] = trade_type
                    base["clf_1_1_prob"] = clf_probs["clf_1_1_prob"][i]
                    base["clf_1_2_prob"] = clf_probs["clf_1_2_prob"][i]
                    base["reg_pred"]     = rr_pred[i]
                    base["meta_conf"]    = meta_conf[i]
                    base["meta_class"]   = int(meta_class[i])
                    base["accepted"]     = bool(meta_class[i] > 0)
                    rows.append(base)

                df_log = pd.DataFrame(rows)

                # If file exists, append & trim
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    try:
                        old = pd.read_csv(csv_path)
                        cat_cols = ['pair'] if 'pair' in old.columns else []
                        for c in cat_cols:
                            if c in df_log.columns:
                                df_log[c] = df_log[c].astype(old[c].dtype)
                        new = pd.concat([old, df_log], ignore_index=True)
                        if len(new) > max_rows:
                            new = new.iloc[-max_rows:]
                        new.to_csv(csv_path, index=False)
                    except pd.errors.EmptyDataError:
                        # If file is empty, just write fresh
                        df_log.to_csv(csv_path, index=False)
                else:
                    df_log.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"⚠️ CSV logging failed: {e}")


    # Return single dict or list of dicts
    return out[0] if n == 1 else out

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

    if initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
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