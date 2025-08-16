import pandas as pd
import numpy as np
import ta  # Technical Analysis Library
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Load CSV
# df = pd.read_csv('ggt.csv')
def calculate_indicators(df):
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['hour'] = df['Time'].dt.hour

    df['hour_sin'] = np.sin(2*np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2*np.pi * df['hour'] / 24)

    df['is_tokyo_session']    = ((df['hour'] >=  2) & (df['hour'] <  9)).astype(int)
    df['is_london_session']   = ((df['hour'] >=  8) & (df['hour'] < 17)).astype(int)
    df['is_new_york_session'] = ((df['hour'] >= 16) & (df['hour'] < 22)).astype(int)
    df['valid_hour'] = df['hour'].between(2, 20)
    df.drop(columns=['Time', 'hour',], inplace=True)

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


    timeframes = {'15min': 3,  '30hr': 6, '1hr': 12}
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

    df['bull_count'] = (df['Close'] > df['Open']).rolling(2).sum()
    df['bear_count'] = (df['Close'] < df['Open']).rolling(2).sum()
    df['momentum_unbalance'] = df['bull_count'] - df['bear_count']
    df.drop(columns=['bull_count', 'bear_count'], inplace=True)

    df['mean_close_10']  = df['Close'].rolling(10).mean()
    df['wick_dominance'] = (df['upper_wick'] + df['lower_wick']) / (df['candle_body'] + EPS)
    df['range_spike'] = df['candle_range'] / df['candle_range'].rolling(2).mean()
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

    def label_choch_from_bos(df, lookback=24):
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


    df = detect_bos(df, lookback=60, swing_k=2, min_break_atr=0.25, confirm_with_close=True)
    df = detect_double_top_bottom(df, distance=18, tol_atr_mult=0.25, min_sep_bars=12)
    df = candles_since_bollinger_touch(df)
    df = label_choch_from_bos(df, lookback=100)
    df = label_hh_hl_lh_ll(df, k=2)
    df = detect_sfp(df, k=9, tol_atr_mult=0.02, confirm_with_close=True)
    df = detect_fvg(df, min_atr_mult=0.1)


    df['entry_price']        = 0.0
    df['stop_loss_price']    = 0.0
    df['stop_loss_distance'] = 0.0
    df['sl_ratio_to_entry']  = 0.0
    df['side']               = -1
    
    vol_window = 15
    df['volatility'] = df['log_returns'].rolling(vol_window).std()


    # Drop rows with NaNs in critical columns
    # df = df.dropna(subset=ohlc_cols)

    # # Compute volatility (rolling standard deviation of returns)
    
    # # Drop initial NaNs
    # df.dropna(inplace=True)

    # df = df.dropna().reset_index(drop=True)
    df = df.iloc[300:].reset_index(drop=True)
   
    return df
# ==== PLUG & PLAY: Multi-threshold triple-barrier (long & short) + metrics + loader ====
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------
# Helpers / sanity checks
# -------------------------
def _ensure_required_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

# -------------------------
# LONG: multi-threshold (-1/0/1) labels + rr_label
# -------------------------
def apply_triple_barrier_with_long_multi(
    df: pd.DataFrame,
    file: str,
    pt_mult_list=(1.2, 2.2, 3.2),
    sl_mult: float = 4.0,
    horizon: int = 50,
):
    EPS = 1e-9
    pair_code = os.path.splitext(os.path.basename(file))[0].lower()
    df = df.copy()
    df['pair'] = pair_code

    # Required columns
    _ensure_required_columns(df, ['valid_hour', 'SFP_Up', 'volatility', 'Close', 'High', 'Low'])

    # Spread guardrails (adjust as needed)
    spread_limits_low = {"gbpusd": 0.00040, "usdcad": 0.00040, "xauusd": 1.0000}
    spread_limits_high = {"gbpusd": 0.00350, "usdcad": 0.00350, "xauusd": 9.0000}
    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"No spread limits for '{pair_code}'")

    rr_labels = []
    y_labels_dict = {f"y_{pt}R": [] for pt in pt_mult_list}
    entry_prices, stop_loss_prices, stop_loss_distances, sl_ratios, sides = [], [], [], [], []

    for i in tqdm(range(len(df)), desc='Sim LONG trades'):
        # Not enough candles ahead
        if i > len(df) - horizon - 1:
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        # Entry filters
        if not bool(df['valid_hour'].iloc[i]) and not bool(df['SFP_Up'].iloc[i]):
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        spread_low = spread_limits_low[pair_code]
        spread_high = spread_limits_high[pair_code]

        entry = float(df['Close'].iloc[i])
        vol = float(df['volatility'].iloc[i])

        # Long: SL below entry, TP above
        sl = entry - sl_mult * vol
        sl_dist = entry - sl  # > 0

        if sl_dist < spread_low or sl_dist > spread_high:
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        max_rr = -np.inf
        hit_sl = False
        hit_tp_dict = {pt: False for pt in pt_mult_list}

        for j in range(1, horizon + 1):
            fp_high = float(df['High'].iloc[i + j])
            fp_low = float(df['Low'].iloc[i + j])

            # SL hit?
            if fp_low <= sl:
                max_rr = max(max_rr, -1.0)
                hit_sl = True
                break

            # TP checks
            for pt in pt_mult_list:
                tp = entry + (pt * sl_dist)
                if fp_high >= tp:
                    max_rr = max(max_rr, pt)
                    hit_tp_dict[pt] = True

            # Update max RR (mark-to-high for long)
            rr = (fp_high - entry) / (sl_dist + EPS)
            max_rr = max(max_rr, rr)

        rr_labels.append(max_rr)

        # -1/0/1 per threshold
        for pt in pt_mult_list:
            if hit_tp_dict[pt]:
                y_labels_dict[f"y_{pt}R"].append(1)
            elif hit_sl:
                y_labels_dict[f"y_{pt}R"].append(-1)
            else:
                y_labels_dict[f"y_{pt}R"].append(0)

        entry_prices.append(entry)
        stop_loss_prices.append(sl)
        stop_loss_distances.append(sl_dist)
        sl_ratios.append(sl_dist / (entry + EPS))
        sides.append(1)  # long

    df['rr_label'] = rr_labels
    for col, vals in y_labels_dict.items():
        df[col] = vals
    df['entry_price'] = entry_prices
    df['stop_loss_price'] = stop_loss_prices
    df['stop_loss_distance'] = stop_loss_distances
    df['sl_ratio_to_entry'] = sl_ratios
    df['side'] = sides

    return df

# -------------------------
# SHORT: multi-threshold (-1/0/1) labels + rr_label
# -------------------------
def apply_triple_barrier_with_short_multi(
    df: pd.DataFrame,
    file: str,
    pt_mult_list=(1.2, 2.2, 3.2),
    sl_mult: float = 4.0,
    horizon: int = 50,
):
    EPS = 1e-9
    pair_code = os.path.splitext(os.path.basename(file))[0].lower()
    df = df.copy()
    df['pair'] = pair_code

    _ensure_required_columns(df, ['valid_hour', 'SFP_Down', 'volatility', 'Close', 'High', 'Low'])

    spread_limits_low = {"gbpusd": 0.00040, "usdcad": 0.00040, "xauusd": 1.0000}
    spread_limits_high = {"gbpusd": 0.00350, "usdcad": 0.00350, "xauusd": 9.0000}
    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"No spread limits for '{pair_code}'")

    rr_labels = []
    y_labels_dict = {f"y_{pt}R": [] for pt in pt_mult_list}
    entry_prices, stop_loss_prices, stop_loss_distances, sl_ratios, sides = [], [], [], [], []

    for i in tqdm(range(len(df)), desc='Sim SHORT trades'):
        if i > len(df) - horizon - 1:
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        if not bool(df['valid_hour'].iloc[i]) and not bool(df['SFP_Down'].iloc[i]):
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        spread_low = spread_limits_low[pair_code]
        spread_high = spread_limits_high[pair_code]

        entry = float(df['Close'].iloc[i])
        vol = float(df['volatility'].iloc[i])

        # Short: SL above entry, TP below
        sl = entry + sl_mult * vol
        sl_dist = sl - entry  # > 0

        if sl_dist < spread_low or sl_dist > spread_high:
            rr_labels.append(np.nan)
            for pt in pt_mult_list:
                y_labels_dict[f"y_{pt}R"].append(np.nan)
            entry_prices.append(np.nan)
            stop_loss_prices.append(np.nan)
            stop_loss_distances.append(np.nan)
            sl_ratios.append(np.nan)
            sides.append(-1)
            continue

        max_rr = -np.inf
        hit_sl = False
        hit_tp_dict = {pt: False for pt in pt_mult_list}

        for j in range(1, horizon + 1):
            fp_high = float(df['High'].iloc[i + j])
            fp_low = float(df['Low'].iloc[i + j])

            # SL hit?
            if fp_high >= sl:
                max_rr = max(max_rr, -1.0)
                hit_sl = True
                break

            # TP checks
            for pt in pt_mult_list:
                tp = entry - (pt * sl_dist)
                if fp_low <= tp:
                    max_rr = max(max_rr, pt)
                    hit_tp_dict[pt] = True

            # Update max RR (mark-to-low for short)
            rr = (entry - fp_low) / (sl_dist + EPS)
            max_rr = max(max_rr, rr)

        rr_labels.append(max_rr)

        for pt in pt_mult_list:
            if hit_tp_dict[pt]:
                y_labels_dict[f"y_{pt}R"].append(1)
            elif hit_sl:
                y_labels_dict[f"y_{pt}R"].append(-1)
            else:
                y_labels_dict[f"y_{pt}R"].append(0)

        entry_prices.append(entry)
        stop_loss_prices.append(sl)
        stop_loss_distances.append(sl_dist)
        sl_ratios.append(sl_dist / (entry + EPS))
        sides.append(0)  # short

    df['rr_label'] = rr_labels
    for col, vals in y_labels_dict.items():
        df[col] = vals
    df['entry_price'] = entry_prices
    df['stop_loss_price'] = stop_loss_prices
    df['stop_loss_distance'] = stop_loss_distances
    df['sl_ratio_to_entry'] = sl_ratios
    df['side'] = sides

    return df

# -------------------------
# Metrics printer (works for any y_*R column)
# -------------------------
def print_trade_metrics(df: pd.DataFrame, target_col: str):
    df_valid = df[df['valid_hour']].copy()
    if df_valid.empty:
        print("⚠️ No valid-hour trades found.")
        return

    vc = df_valid[target_col].value_counts().sort_index()
    wins   = int(vc.get(1, 0))
    losses = int(vc.get(-1, 0))
    neutrals = int(vc.get(0, 0))
    total_trades = wins + losses  # excludes neutrals for WR/LR

    win_rate = wins / total_trades * 100 if total_trades else 0.0
    loss_rate = losses / total_trades * 100 if total_trades else 0.0
    neutral_rate = neutrals / (total_trades + neutrals) * 100 if (total_trades + neutrals) else 0.0

    RR_levels = [1.5]  # expectancy snapshot
    expectancies = {rr: (win_rate / 100 * rr) - (loss_rate / 100 * 1.0) for rr in RR_levels}

    print(f"\n🔎 Metrics for {target_col}:")
    print(vc)
    print(f"\n📈 Win Rate:     {win_rate:.2f}%")
    print(f"📉 Loss Rate:    {loss_rate:.2f}%")
    print(f"⏸️ Neutral Rate: {neutral_rate:.2f}%")
    print(f"📊 Total Trades (W+L): {total_trades}")

    print("\n💰 Expectancy by R:R:")
    for rr in RR_levels:
        print(f" - R:R {rr}:1 → Expectancy = {expectancies[rr]:.3f}")

    print("\n📈 rr_label stats:")
    print(f" - Max:    {pd.to_numeric(df_valid['rr_label'], errors='coerce').max():.3f}")
    print(f" - Mean:   {pd.to_numeric(df_valid['rr_label'], errors='coerce').mean():.3f}")
    print(f" - Median: {pd.to_numeric(df_valid['rr_label'], errors='coerce').median():.3f}")

# -------------------------
# Loader / runner: labels both sides, prints metrics, saves combined CSV
# -------------------------
def load_and_label_data(
    csv_files,
    save_path="test-combined.csv",
    pt_mult_list=(1.2, 2.2, 3.2),
    sl_mult=4.0,
    horizon=50,
):
    all_trades = []

    for file in csv_files:
        print(f"\n📂 Processing {file}")
        try:
            df_raw = pd.read_csv(file)

            # If you normally build features here, uncomment:
            df_raw = calculate_indicators(df_raw)

            long_trades = apply_triple_barrier_with_long_multi(
                df=df_raw.copy(),
                file=file,
                pt_mult_list=pt_mult_list,
                sl_mult=sl_mult,
                horizon=horizon
            )
            short_trades = apply_triple_barrier_with_short_multi(
                df=df_raw.copy(),
                file=file,
                pt_mult_list=pt_mult_list,
                sl_mult=sl_mult,
                horizon=horizon
            )

            both = pd.concat([long_trades, short_trades], ignore_index=True)
            print(f"✅ {file} → {len(both)} labeled rows")
            all_trades.append(both)

        except Exception as e:
            print(f"❌ Failed on {file}: {e}")

    if not all_trades:
        print("⚠️ No trades were collected.")
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True)

    # Save combined dataset
    combined.to_csv(save_path, index=False)
    print(f"\n✅ Final dataset saved: {len(combined)} rows from {len(csv_files)} files → {save_path}")

    # Print metrics for each threshold
    for pt in pt_mult_list:
        col = f"y_{pt}R"
        if col in combined.columns:
            print_trade_metrics(combined, col)

    return combined

# --- Example usage ---
if __name__ == "__main__":
    csv_files = ["gbpusd.csv", "usdcad.csv"]  # add more as needed
    final_dataset = load_and_label_data(
        csv_files,
        save_path="test-combined.csv",
        pt_mult_list=(1.2, 2.2, 3.2),
        sl_mult=4.0,
        horizon=50
    )
