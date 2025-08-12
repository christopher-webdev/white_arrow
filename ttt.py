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
    # Parse datetime column (adjust name if needed)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['hour'] = df['Time'].dt.hour

    df['hour_sin'] = np.sin(2*np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2*np.pi * df['hour'] / 24)

    # df['is_tokyo_session']    = ((df['hour'] >=  2) & (df['hour'] <  9)).astype(int)
    # df['is_london_session']   = ((df['hour'] >=  8) & (df['hour'] < 17)).astype(int)
    # df['is_new_york_session'] = ((df['hour'] >= 16) & (df['hour'] < 22)).astype(int)
    df['valid_hour'] = df['hour'].between(2, 18)
    # df.drop(columns=['Time', 'hour'], inplace=True)

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
        # df['Senkou_A'] = ichimoku.ichimoku_a()
        # df['Senkou_B'] = ichimoku.ichimoku_b()
        # df['tenkan_kijun_delta'] = df['Tenkan'] - df['Kijun']

        # for label, window in timeframes.items():
        #     df[f'{label}_tk_delta_slope']       = (df['tenkan_kijun_delta'] - df['tenkan_kijun_delta'].shift(window)) / window
        #     df[f'{label}_tk_delta_mean']        = df['tenkan_kijun_delta'].rolling(window).mean()
        #     df[f'{label}_tk_delta_std']         = df['tenkan_kijun_delta'].rolling(window).std()
        #     df[f'{label}_tk_delta_pct_rank']    = df['tenkan_kijun_delta'].rank(pct=True).rolling(window).apply(lambda x: x[-1], raw=True)

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

    df['entry_price']        = 0.0
    df['stop_loss_price']    = 0.0
    df['stop_loss_distance'] = 0.0
    df['sl_ratio_to_entry']  = 0.0
    df['side']               = -1
    
    vol_window = 50
    df['volatility_log'] = df['log_returns'].rolling(vol_window).std()
       # Simple slopes (n-step delta)
    df['SMA_9_slope_3']  = df['SMA_9']  - df['SMA_9'].shift(3)
    df['SMA_20_slope_5'] = df['SMA_20'] - df['SMA_20'].shift(5)

    # Distance from SMA20 (for pullback-to-trend entries)
    df['dist_to_sma20_pct'] = (df['Close'] - df['SMA_20']) / df['SMA_20']


    # Drop rows with NaNs in critical columns
    # df = df.dropna(subset=ohlc_cols)
    df = df.iloc[300:].reset_index(drop=True)
    # Compute volatility (rolling standard deviation of returns)
    
    # Drop initial NaNs
    # df.dropna(inplace=True)

    # df = df.dropna().reset_index(drop=True)
   
    return df
def build_long_entry_mask(df: pd.DataFrame,
                          pair_code: str,
                          sl_mult: float,
                          min_sl_distance: dict,
                          max_sl_distance: dict) -> pd.Series:

    cond_session = df['valid_hour'] 

    cond_trend = (
        (df['Close'] > df['SMA_50']) &
        (df['SMA_9'] > df['SMA_20']) &
        (df['SMA_9_slope_3'] > 0) &
        (df['SMA_20_slope_5'] >= 0)
    )

    cond_pullback = df['dist_to_sma20_pct'].abs() <= 0.004

    # 🔁 compute vol percentiles here (vol already exists)
    win = 100
    vol = df['volatility'].astype(float)
    df['vol_p40'] = vol.rolling(win, min_periods=win).quantile(0.40)
    df['vol_p90'] = vol.rolling(win, min_periods=win).quantile(0.90)
    # replace your 4 lines that compute vol_p40/p90 and cond_vol_regime with this single line:
    cond_vol_regime = (df['volatility'] >= df['vol_p40']) & (df['volatility'] <= df['vol_p90'])


    proposed_sl_dist = sl_mult * df['volatility']
    min_sl = min_sl_distance.get(pair_code)
    max_sl = max_sl_distance.get(pair_code)
    if (min_sl is None) or (max_sl is None):
        raise KeyError(f"Missing SL distance bounds for '{pair_code}'")
    cond_sl_bounds = (proposed_sl_dist >= min_sl) & (proposed_sl_dist <= max_sl)

    cond_bar_quality = (df['High'] - df['Low']) > 1e-9

    entry_mask = cond_trend & cond_pullback & cond_vol_regime & cond_sl_bounds & cond_bar_quality
    return entry_mask.fillna(False)

def compute_dynamic_vol(df: pd.DataFrame,
                        ewma_alpha: float = 0.15,
                        use_parkinson: bool = False,
                        atr_window: int = 14,
                        vol_floor: float = None,
                        vol_cap_pct: float = 0.02):
    """
    Returns df with columns:
      TR, ATR, parkinson_vol, gk_vol, range_vol_raw, range_vol_ewma, dyn_vol
    dyn_vol is what you should use for sizing SL/TP.
    """

    # --- True Range & ATR ---
    prev_close = df['Close'].shift(1)
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum((df['High'] - prev_close).abs(),
                               (df['Low'] - prev_close).abs()))
    atr = tr.rolling(atr_window, min_periods=1).mean()

    # --- Parkinson (uses High-Low only) ---
    # sigma_p^2 = (1/(4*ln2)) * ln(High/Low)^2
    hl_ratio = (df['High'] / df['Low']).clip(lower=1e-9)
    parkinson_var = (np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0))
    parkinson_vol = np.sqrt(parkinson_var)

    # --- Garman–Klass (uses O, H, L, C) if you have Open; else skip ---
    if 'Open' in df.columns:
        rs1 = 0.5 * (np.log(df['High'] / df['Low']) ** 2)
        rs2 = (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
        gk_var = (rs1 - rs2).clip(lower=0)
        gk_vol = np.sqrt(gk_var)
    else:
        gk_vol = pd.Series(index=df.index, dtype=float)

    # --- Pick a range estimator & scale to price units (like ATR) ---
    # Range estimators are dimensionless (in log terms). Map to price by multiplying by price level.
    price = df['Close']
    range_vol_raw = (parkinson_vol if use_parkinson else gk_vol).fillna(method='ffill') * price

    # --- EWMA smooth both ATR and range_vol ---
    atr_ewma = atr.ewm(alpha=ewma_alpha, adjust=False).mean()
    range_vol_ewma = pd.Series(range_vol_raw).ewm(alpha=ewma_alpha, adjust=False).mean()

    # --- Blend: slightly favor ATR because it’s in price terms and broker constraints align with it ---
    dyn_vol = 0.6 * atr_ewma + 0.4 * range_vol_ewma

    # --- Floor/Cap to keep distances sane ---
    if vol_floor is not None:
        dyn_vol = np.maximum(dyn_vol, vol_floor)
    # Cap at a % of price (e.g., 2%)
    dyn_vol = np.minimum(dyn_vol, vol_cap_pct * price)

    df = df.copy()
    df['TR'] = tr
    df['ATR'] = atr
    df['parkinson_vol'] = parkinson_vol
    df['gk_vol'] = gk_vol
    df['range_vol_raw'] = range_vol_raw
    df['range_vol_ewma'] = range_vol_ewma
    df['volatility'] = dyn_vol  # <- use this
    return df
def apply_triple_barrier_with_long(df,
                                   file,
                                   pt_mult: float = 1.0,
                                   sl_mult: float = 1.0,
                                   horizon: int = 48,
                                   rr_threshold: float = None):
    EPS = 1e-9
    pair_code = os.path.splitext(os.path.basename(file))[0].lower()
    df = df.copy()
    df['pair'] = pair_code

    # Per-pair bounds
    min_sl_distance = {
        "gbpusd": 0.00050,
        "usdcad": 0.00050,
        "audusd": 0.00050,
        "xauusd": 1.0,
    }
    max_sl_distance = {
        "gbpusd": 0.00350,
        "usdcad": 0.00350,
        "audusd": 0.00350,
        "xauusd": 9.0,
    }
    if pair_code not in min_sl_distance or pair_code not in max_sl_distance:
        raise KeyError(f"No SL distance bounds defined for '{pair_code}'")

    # 1) Core features first (no volatility dependency)
    df = calculate_indicators(df)

    # 2) Now compute dynamic volatility
    df = compute_dynamic_vol(
        df,
        ewma_alpha=0.2,
        use_parkinson=True,
        atr_window=14,
        vol_floor=min_sl_distance.get(pair_code, None),
        vol_cap_pct=0.02
    )

    # 3) Build entry mask (uses volatility percentiles inside)
    entry_mask = build_long_entry_mask(df, pair_code, sl_mult, min_sl_distance, max_sl_distance)

    # --- simulate ---
    n = len(df)
    labels = [np.nan] * n
    rr_labels = [np.nan] * n
    entry_prices = [np.nan] * n
    stop_loss_prices = [np.nan] * n
    stop_loss_distances = [np.nan] * n
    sl_ratios = [np.nan] * n
    sides = [-1] * n

    for i in range(n):
        if i > n - horizon - 1:
            continue
        if not entry_mask.iloc[i]:
            continue
        if 'valid_hour' in df.columns and not bool(df['valid_hour'].iloc[i]):
            continue
        if 'log_returns' in df.columns and (df['log_returns'].iloc[i] <= 0.0002):
            continue

        entry = float(df['Close'].iloc[i])
        vol   = float(df['volatility'].iloc[i])

        sl = entry - sl_mult * vol
        sl_dist = entry - sl
        tp = entry + pt_mult * sl_dist

        if (sl_dist < min_sl_distance[pair_code]) or (sl_dist > max_sl_distance[pair_code]):
            continue

        label = 0
        max_rr = -np.inf
        for j in range(1, horizon + 1):
            fp_high = float(df['High'].iloc[i + j])
            fp_low  = float(df['Low'].iloc[i + j])

            if fp_low <= sl:
                rr = -1.0
                max_rr = max(max_rr, rr)
                label = -1
                break
            if fp_high >= tp:
                rr = pt_mult
                max_rr = max(max_rr, rr)
                label = 1
                break

            rr = (fp_high - entry) / (entry - sl + EPS)
            max_rr = max(max_rr, rr)

        labels[i] = label
        rr_labels[i] = max_rr
        entry_prices[i] = entry
        stop_loss_prices[i] = sl
        stop_loss_distances[i] = sl_dist
        sl_ratios[i] = sl_dist / (entry + EPS)
        sides[i] = 1

    df['label'] = labels
    df['rr_label'] = rr_labels
    df['entry_price'] = entry_prices
    df['stop_loss_price'] = stop_loss_prices
    df['stop_loss_distance'] = stop_loss_distances
    df['sl_ratio_to_entry'] = sl_ratios
    df['side'] = sides

    if rr_threshold is not None:
        df['label_rr_threshold'] = np.where(df['rr_label'] >= rr_threshold, 1,
                                    np.where(df['rr_label'] <= -1.0, -1, 0))
    return df


def generate_rr_classification_labels(df):
    df['label_rr_1.0'] = (df['rr_label'] >= 1.1).astype(int)
    df['label_rr_2.0'] = (df['rr_label'] >= 2.0).astype(int)
    # df['label_rr_3.0'] = (df['rr_label'] >= 3.0).astype(int)
    return df

def label_long_trades(df, file, pt_mult, sl_mult, horizon, rr_threshold=2):
    df = calculate_indicators(df)
    df = apply_triple_barrier_with_long(df, file, pt_mult, sl_mult, horizon, rr_threshold)
    df.dropna(subset=['label'], inplace=True)
    # df['pair'] = df['pair'].astype('category')

    df = generate_rr_classification_labels(df)
    df_valid = df[df['valid_hour']].copy()

    print(df_valid.head())
    if df_valid.empty:
        print("⚠️ No valid-hour trades found in:", file)
        return pd.DataFrame()

    # Label stats
    label_counts = df_valid['label'].value_counts().sort_index()
    wins = label_counts.get(1.0, 0)
    losses = label_counts.get(-1.0, 0)
    neutrals = label_counts.get(0.0, 0)
    total_trades = wins + losses  # + neutrals

    win_rate = wins / total_trades * 100 if total_trades else 0
    loss_rate = losses / total_trades * 100 if total_trades else 0
    neutral_rate = neutrals / total_trades * 100 if total_trades else 0

    # Expectancy
    RR_levels = [2]
    expectancies = {rr: (win_rate / 100 * rr) - (loss_rate / 100 * 1) for rr in RR_levels}

    # 📊 Print Summary
    print("\n🔎 Trade Label Distribution (VALID HOURS ONLY):")
    print(label_counts)
    print(f"\n📈 Win Rate:     {win_rate:.2f}%")
    print(f"📉 Loss Rate:    {loss_rate:.2f}%")
    print(f"⏸️ Neutral Rate: {neutral_rate:.2f}%")
    print(f"📊 Total Trades: {total_trades}")

    print("\n💰 Expectancy by R:R:")
    for rr in RR_levels:
        print(f" - R:R {rr}:1 → Expectancy = {expectancies[rr]:.3f}")

    print(f"\n📈 Max R:R label summary (from 'rr_label'):")
    print(f" - Max RR:  {df_valid['rr_label'].max():.2f}")
    print(f" - Mean RR: {df_valid['rr_label'].mean():.2f}")
    print(f" - Median:  {df_valid['rr_label'].median():.2f}")

    return df_valid

def load_and_label_data(csv_files, save_path="testbuy.csv"):
    all_trades = []

    for file in csv_files:
        print(f"\n📂 Processing {file}")
        try:
            df = pd.read_csv(file)
         
            long_trades = label_long_trades(df.copy(), file, pt_mult=2, sl_mult=8, horizon=48, rr_threshold=2)

            both = pd.concat([long_trades], ignore_index=True)#short_trades,
            print(f"✅ {file} → {len(both)} trades labeled")
            all_trades.append(both)

        except Exception as e:
            print(f"❌ Failed on {file}: {e}")

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(save_path, index=False)
        print(f"\n✅ Final dataset saved: {len(combined)} trades from {len(csv_files)} files")
        print(combined['rr_label'].describe())
        return combined
    else:
        print("⚠️ No trades were collected.")
        return pd.DataFrame()

csv_files = ["gbpusd.csv","usdcad.csv" ] #" "usdcad.csv",,, "eurusd.csv","usdcad.csv"xau.csv", "jpy.csv"
final_dataset = load_and_label_data(csv_files)
