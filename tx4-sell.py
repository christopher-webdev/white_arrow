
import pandas as pd
import numpy as np
import ta  # Technical Analysis Library


# df = pd.read_csv("gg.csv")  # sep="\t", encoding="utf-16" 

from scipy.stats import normaltest, trim_mean
from scipy.stats import entropy, kurtosis, skew, gmean, hmean, iqr
from scipy.signal import hilbert
from scipy.fftpack import fft

def calculate_indicators(df, window=10, window_dev=2):
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['hour'] = df['Time'].dt.hour

    df['hour_sin'] = np.sin(2*np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2*np.pi * df['hour'] / 24)

    df['is_tokyo_session']    = ((df['hour'] >=  2) & (df['hour'] <  9)).astype(int)
    df['is_london_session']   = ((df['hour'] >=  8) & (df['hour'] < 17)).astype(int)
    df['is_new_york_session'] = ((df['hour'] >= 16) & (df['hour'] < 22)).astype(int)
    df['valid_hour'] = df['hour'].between(2, 21)
    df.drop(columns=['Time', 'hour'], inplace=True)

    for w in [9, 20,50,100,200]:
        df[f'SMA_{w}'] = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
        df[f'SMA{w}_Distance'] = ((df['Close'] - df[f'SMA_{w}']) / df[f'SMA_{w}']) * 100
        df[f'SMA{w}_Slope'] = df[f'SMA_{w}'].diff(5)

    for w in [9, 20, 50, 100, 200]:
        sma_col = f'SMA_{w}'
        sma = df[sma_col]

        # % distance of price to SMA
        df[f'{sma_col}_pct_distance'] = ((df['Close'] - sma) / sma) * 100

        # 🔹 Diff-based slope (rate of change)
        df[f'{sma_col}_slope_diff'] = sma.diff() / sma.shift()

        # 🔹 Gradient-based slope (smoother)
        df[f'{sma_col}_slope_grad'] = np.gradient(sma)

        # 🔹 Direction (+1, 0, -1)
        df[f'{sma_col}_direction'] = np.sign(df[f'{sma_col}_slope_diff'])

        # 🔹 Cross detection (price crossing SMA)
        cross = np.where(
            (df['Close'] > sma) & (df['Close'].shift(1) <= sma.shift(1)), 1,
            np.where((df['Close'] < sma) & (df['Close'].shift(1) >= sma.shift(1)), -1, 0)
        )
        df[f'{sma_col}_cross'] = cross

        # 🔹 Percentile rank of SMA in 5-bar rolling window
        df[f'{sma_col}_percentile'] = sma.rolling(window=5).apply(
            lambda x: (np.sum(x < x[-1]) / len(x)) * 100, raw=True
        )

        # 🔹 Upward movement ratio in last 5 bars
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
    def compute_ichimoku_features_from_5m(df: pd.DataFrame) -> pd.DataFrame:
        timeframes = {'15min': 3, '1h': 12, '4h': 48, '1d': 288}
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['High'], low=df['Low'], window1=9, window2=26, window3=52
        )
        df['Tenkan']   = ichimoku.ichimoku_conversion_line()
        df['Kijun']    = ichimoku.ichimoku_base_line()
        df['Senkou_A'] = ichimoku.ichimoku_a()
        df['Senkou_B'] = ichimoku.ichimoku_b()
        df['tenkan_kijun_delta'] = df['Tenkan'] - df['Kijun']

        for label, window in timeframes.items():
            df[f'{label}_tk_delta_slope']       = (df['tenkan_kijun_delta'] - df['tenkan_kijun_delta'].shift(window)) / window
            df[f'{label}_tk_delta_mean']        = df['tenkan_kijun_delta'].rolling(window).mean()
            df[f'{label}_tk_delta_std']         = df['tenkan_kijun_delta'].rolling(window).std()
            df[f'{label}_tk_delta_pct_rank']    = df['tenkan_kijun_delta'].rank(pct=True).rolling(window).apply(lambda x: x[-1], raw=True)

        return df
    df = compute_ichimoku_features_from_5m(df)
    
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
      
    bb = ta.volatility.BollingerBands(close=df['Close'], window=10, window_dev=2.0)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Mid']   = bb.bollinger_mavg()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    df['BB_Close_Dist_Mid'] = df['Close'] - df['BB_Mid']
    df['BB_Close_Dist_Lower'] = df['Close'] - df['BB_Lower']
    df['BB_Close_Dist_Upper'] = df['BB_Upper'] - df['Close']
    df['BB_Mid_Slope'] = df['BB_Mid'].diff()
    df['BB_Is_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(window=window).quantile(0.15)
    df['BB_Expansion'] = df['BB_Width'].pct_change()

    for i in range(1, 12):
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
    df['session_intensity'] = 0.7 * df['is_london_session'] + 1.2 * df['is_new_york_session'] + 0.4 * df['is_tokyo_session']
    df.drop(columns=['mean_close_10'], inplace=True)

    df['PGI_alt'] = 2.0 * df['momentum_unbalance'].fillna(0) + 1.5 * df['range_spike'].fillna(0) + 1.2 * df['wick_dominance'].fillna(0) + 2.5 * df['price_surge'].fillna(0)
 
# ────────────── STATISTICAL FEATURE ENGINE ──────────────
    # Price Distribution
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(5).std()
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
        0.5 * (df['SMA9_Distance'] - 50) +
        0.3 * (df['rsi_slope'] * 10) +
        0.2 * (df['obv_slope'] / (df['rolling_std'] + 1e-6))
    )
    def add_candlestick_features(df):
        # === 1. Range-bound Detection ===
        def is_range_bound(window=5, threshold=0.015):
            max_high = df['High'].rolling(window).max()
            min_low = df['Low'].rolling(window).min()
            range_pct = (max_high - min_low) / df['Close']
            return (range_pct < threshold).astype(int)

        df['range_5'] = is_range_bound(5)
        df['range_10'] = is_range_bound(10)

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

        df['cluster_10'] = cluster_score(10)

        # === Add memory with lag features (shifts) ===
        features_to_shift = ['range_5', 'range_10', 'bullish_engulf', 'bearish_engulf',
                            'pin_bar', 'inside_bar', 'outside_bar', 'cluster_10']

        for f in features_to_shift:
            for lag in range(1, 6):  # shift 1 to 5
                df[f'{f}_lag{lag}'] = df[f].shift(lag).fillna(0).astype(int)

        return df
    df = add_candlestick_features(df)


    df['entry_price']        = 0.0
    df['stop_loss_price']    = 0.0
    df['stop_loss_distance'] = 0.0
    df['sl_ratio_to_entry']  = 0.0
    df['side']               = -1
  
    df = df.iloc[300:].reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


from tqdm import tqdm
def filter_ml_trades_sell(df, file, rr_cap=3, max_lookahead=48):
    EPS = 1e-9
    df = calculate_indicators(df)
    pair_code = os.path.splitext(os.path.basename(file))[0]
    df['pair'] = pair_code

    spread_limits_low = {
        "xau": 3.00, "jpy": 0.050,
        "gbp": 0.00050, 
        "cad": 0.00060
    }
    spread_limits_high = {
        "xau": 09.00, "jpy": 0.350,
        "gbp": 0.00350, 
        "cad": 0.00350
    }

    if pair_code not in spread_limits_low or pair_code not in spread_limits_high:
        raise KeyError(f"No spread limit defined for pair '{pair_code}'")

    bbwidth_thresh = df['BB_Width'].quantile(1)
    volatility_thresh = df['rolling_std'].quantile(1)

    def passes_filter(row):
        try:
            return (
                row['BB_Width'] <= bbwidth_thresh and
                row['rolling_std'] <= volatility_thresh
            )
        except:
            return False

    trades = []
    filtered_in = 0

    for i in tqdm(range(1, len(df) - max_lookahead - 1), desc='Sim short trades'):
        row = df.iloc[i]
        prev_close, curr_close = df['Close'].iloc[i - 1], df['Close'].iloc[i]
        prev_sma9, curr_sma9 = df['SMA_9'].iloc[i - 1], df['SMA_9'].iloc[i]
        valid_hour = df['valid_hour'].iloc[i]
        next_low = df['Low'].iloc[i+1] < df['Low'].iloc[i]

        # === Short entry condition ===
        if  prev_close > prev_sma9 and curr_close < curr_sma9 and valid_hour and next_low:  #
            entry_price = df['Low'].iloc[i]
            sl_price = df['BB_Upper'].iloc[i]

            if np.isnan(sl_price) or sl_price <= entry_price:
                continue

            sl_dist = sl_price - entry_price
            
            spread_low = spread_limits_low[pair_code]
            spread_high = spread_limits_high[pair_code]
           
            if sl_dist < spread_low:
                continue

            if sl_dist > spread_high:
                continue

            # if not passes_filter(row):
            #     continue
            # filtered_in += 1

            rr_achieved = 0
            time_to_outcome = max_lookahead
            hit_type = -1  # Default to -1 (NoHit)

            for j in range(1, max_lookahead + 1):
                future = df.iloc[i + j]
                if future['High'] >= sl_price:
                    hit_type = 0  # SL
                    time_to_outcome = j
                    break
                rr = (entry_price - future['Close']) / sl_dist
                if rr >= rr_cap:
                    rr_achieved = rr_cap
                    hit_type = 1  # TP
                    time_to_outcome = j
                    break
                rr_achieved = max(rr_achieved, rr)

            if hit_type == -1 and rr_achieved > 1.5:
                hit_type = 1 if rr_achieved >= 1 else 0  # fallback classification

            row = df.iloc[i].copy()
            row['entry_price'] = entry_price
            row['stop_loss_price'] = sl_price
            row['stop_loss_distance'] = sl_dist
            row['sl_ratio_to_entry'] = sl_dist / (entry_price + EPS)
            row['rr_label'] = round(min(rr_achieved, rr_cap), 3)
            row['rr_class'] = hit_type
            row['time_to_outcome'] = time_to_outcome
            row['side'] = 0  # Short
            trades.append(row)

    trades_df = pd.DataFrame(trades)
    # trades_df.to_csv("ml_trainingregress_sell.csv", index=False)
    # print(f"✅ Saved {len(trades_df)} short ML samples to 'ml_trainingregress_sell.csv'")
    print(f"📊 RR label stats:\n{trades_df['rr_label'].describe()}")
    print(f"📊 Class distribution:\n{trades_df['rr_class'].value_counts()}")
    print(f"Filtered-In Trades: {filtered_in}")
    return trades_df


from tqdm import tqdm
import os

# Import calculate_indicators() and filter_ml_trades() here (already defined in your script)

# Add this new function to process multiple pairs
import traceback

def run_on_multiple_csvs(csv_files, rr_cap=3, max_lookahead=48):
    all_trades = []

    for file in csv_files:
        print(f"📂 Processing {file}")
        try:
            df = pd.read_csv(file)
            trades_df = filter_ml_trades_sell(df, file, rr_cap=rr_cap, max_lookahead=max_lookahead)
            all_trades.append(trades_df)
        except Exception as e:
            print(f"❌ Failed on {file}: {e}")
            traceback.print_exc()  # Print full traceback for debugging

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv("jpy-sell.csv", index=False)
        print(f"✅ Final dataset saved: {len(combined)} trades from {len(csv_files)} pairs")
        print(combined['rr_label'].describe())
        return combined
    else:
        print("⚠️ No trades were collected.")
        return pd.DataFrame()

# Example usage
csv_files = ['jpy.csv']  # Add more files if needed
final_combined = run_on_multiple_csvs(csv_files)
