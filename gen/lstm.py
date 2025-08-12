# import pandas as pd
# import numpy as np
# import ta
# from ta.trend import SMAIndicator, EMAIndicator
# from ta.momentum import RSIIndicator, StochasticOscillator
# from ta.volatility import BollingerBands, AverageTrueRange
# from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
# from tqdm import tqdm
# import pandas as pd
# import numpy as np
# import ta

# import pandas as pd
# import numpy as np
# import ta
# from tqdm import tqdm

# def calculate_indicators(df, window=20, window_dev=2):
#     # === 1) Time & Session Features ===
#     df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
#     df['hour'] = df['Time'].dt.hour

#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

#     # df['is_opening_session'] = ((df['hour'] >= 9) & (df['hour'] <= 11)).astype(int)
#     # df['is_closing_session'] = ((df['hour'] >= 15) & (df['hour'] <= 17)).astype(int)

#     # UTC‐based Forex sessions
#     df['is_tokyo_session']   = ((df['hour'] >= 0)  & (df['hour'] <  9)).astype(int)
#     df['is_london_session']  = ((df['hour'] >= 8)  & (df['hour'] < 17)).astype(int)
#     df['is_new_york_session']= ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

#     df.drop(columns=['Time', 'hour'], inplace=True)


#     # === 2) Simple Moving Averages, Slopes, and Distances ===
#     for w in [9, 20]:
#         df[f'SMA_{w}'] = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
#         df[f'SMA{w}_Distance'] = ((df['Close'] - df[f'SMA_{w}']) / df[f'SMA_{w}']) * 100
#         df[f'SMA{w}_Slope'] = df[f'SMA_{w}'].diff(5)

  


#     def zone_duration(series):
#         groups = (series != series.shift()).cumsum()
#         return series.groupby(groups).cumcount()

#     def add_rsi_features(df, rsi_window=14, div_shift=9):
#         rsi = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_window).rsi()
#         df['RSI'] = rsi.fillna(0)

#         # Crossovers
#         df['rsi_above_70'] = (df['RSI'] > 70).astype(int)
#         df['rsi_below_30'] = (df['RSI'] < 30).astype(int)
#         df['rsi_cross_70'] = ((df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)).astype(int)
#         df['rsi_cross_30'] = ((df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)).astype(int)

       
#         df['rsi_cross_50_up'] = ((df['RSI'] > 50) & (df['RSI'].shift(1) <= 50)).astype(int)
#         df['rsi_cross_50_down'] = ((df['RSI'] < 50) & (df['RSI'].shift(1) >= 50)).astype(int)

#         # Trend & Volatility
#         df['rsi_change'] = df['RSI'] - df['RSI'].shift(1).fillna(0)
#         df['rsi_slope'] = np.gradient(df['RSI'])
#         df['rsi_rolling_mean'] = df['RSI'].rolling(5).mean().fillna(df['RSI'])
#         df['rsi_above_mean'] = (df['RSI'] > df['rsi_rolling_mean']).astype(int)
#         df['rsi_std'] = df['RSI'].rolling(5).std().fillna(0)
#         df['rsi_zscore'] = (df['RSI'] - df['RSI'].rolling(20).mean()) / (df['RSI'].rolling(20).std() + 1e-6)
#         df['rsi_zscore'] = df['rsi_zscore'].fillna(0)

#         # Divergence with proper shift
#         df['price_up_rsi_down'] = ((df['Close'] > df['Close'].shift(div_shift)) & (df['RSI'] < df['RSI'].shift(div_shift))).astype(int)
#         df['price_down_rsi_up'] = ((df['Close'] < df['Close'].shift(div_shift)) & (df['RSI'] > df['RSI'].shift(div_shift))).astype(int)

#         # OB/OS durations
#         df['rsi_above_70_duration'] = zone_duration(df['rsi_above_70'])
#         df['rsi_below_30_duration'] = zone_duration(df['rsi_below_30'])

#         # Distance to thresholds
#         df['rsi_to_70'] = (70 - df['RSI']).clip(lower=0)
#         df['rsi_to_30'] = (df['RSI'] - 30).clip(lower=0)
#         return df


#     def add_cci_features(df, cci_window=20, div_shift=9):
#         cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=cci_window).cci()
#         df['CCI'] = cci.fillna(0)

#         # Crossovers
#         df['cci_above_100'] = (df['CCI'] > 100).astype(int)
#         df['cci_below_minus100'] = (df['CCI'] < -100).astype(int)
#         df['cci_cross_100'] = ((df['CCI'] > 100) & (df['CCI'].shift(1) <= 100)).astype(int)
#         df['cci_cross_minus100'] = ((df['CCI'] < -100) & (df['CCI'].shift(1) >= -100)).astype(int)
#         df['cci_cross_0_up'] = ((df['CCI'] > 0) & (df['CCI'].shift(1) <= 0)).astype(int)
#         df['cci_cross_0_down'] = ((df['CCI'] < 0) & (df['CCI'].shift(1) >= 0)).astype(int)

#         # Trend
#         df['cci_change'] = df['CCI'] - df['CCI'].shift(1).fillna(0)
#         df['cci_slope'] = np.gradient(df['CCI'])
#         df['cci_rolling_mean'] = df['CCI'].rolling(5).mean().fillna(df['CCI'])
#         df['cci_above_mean'] = (df['CCI'] > df['cci_rolling_mean']).astype(int)

#         # Normalization
#         df['cci_std'] = df['CCI'].rolling(5).std().fillna(0)
#         df['cci_zscore'] = (df['CCI'] - df['CCI'].rolling(20).mean()) / (df['CCI'].rolling(20).std() + 1e-6)
#         df['cci_zscore'] = df['cci_zscore'].fillna(0)

#         # Divergence
#         df['price_up_cci_down'] = ((df['Close'] > df['Close'].shift(div_shift)) & (df['CCI'] < df['CCI'].shift(div_shift))).astype(int)
#         df['price_down_cci_up'] = ((df['Close'] < df['Close'].shift(div_shift)) & (df['CCI'] > df['CCI'].shift(div_shift))).astype(int)

#         # Duration in OB/OS zones
#         df['cci_above_100_duration'] = zone_duration(df['cci_above_100'])
#         df['cci_below_minus100_duration'] = zone_duration(df['cci_below_minus100'])

#         # Distance from thresholds
#         df['cci_to_100'] = (100 - df['CCI']).clip(lower=0)
#         df['cci_to_minus100'] = (df['CCI'] + 100).clip(lower=0)
#         return df


#     def add_macd_features(df, div_shift=9):
#         macd = ta.trend.MACD(df['Close'])
#         df['MACD_Line'] = macd.macd().fillna(0)
#         df['MACD_Signal'] = macd.macd_signal().fillna(0)
#         df['MACD_Histogram'] = macd.macd_diff().fillna(0)

#         # Normalized
#         df['macd_line_pct_change'] = df['MACD_Line'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
#         df['macd_hist_pct_change'] = df['MACD_Histogram'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
#         df['macd_line_zscore'] = (df['MACD_Line'] - df['MACD_Line'].rolling(20).mean()) / (df['MACD_Line'].rolling(20).std() + 1e-6)
#         df['macd_hist_zscore'] = (df['MACD_Histogram'] - df['MACD_Histogram'].rolling(20).mean()) / (df['MACD_Histogram'].rolling(20).std() + 1e-6)
#         df['macd_line_zscore'] = df['macd_line_zscore'].fillna(0)
#         df['macd_hist_zscore'] = df['macd_hist_zscore'].fillna(0)

#         # Crossovers
#         df['macd_cross_signal'] = ((df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
#         df['macd_cross_zero_up'] = ((df['MACD_Line'] > 0) & (df['MACD_Line'].shift(1) <= 0)).astype(int)
#         df['macd_cross_zero_down'] = ((df['MACD_Line'] < 0) & (df['MACD_Line'].shift(1) >= 0)).astype(int)

#         # Histogram direction
#         df['macd_hist_direction'] = np.sign(df['MACD_Histogram'].diff().fillna(0)).astype(int)
#         df['macd_hist_slope'] = np.gradient(df['MACD_Histogram'])

#         # Divergence
#         df['price_up_hist_down'] = ((df['Close'] > df['Close'].shift(div_shift)) & (df['MACD_Histogram'] < df['MACD_Histogram'].shift(div_shift))).astype(int)
#         df['price_down_hist_up'] = ((df['Close'] < df['Close'].shift(div_shift)) & (df['MACD_Histogram'] > df['MACD_Histogram'].shift(div_shift))).astype(int)

#         return df

#     df = add_rsi_features(df, rsi_window=14, div_shift=9)
#     df = add_cci_features(df, cci_window=20, div_shift=9)
#     df = add_macd_features(df, div_shift=9)




#     def add_price_volume_features(df):
#         eps = 1e-6  # for division safety

#         # === 1. Intrabar Movement ===
#         df['OC_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + eps)
#         for shift in [1, 2, 3, 5, 9]:
#             df[f'OC_ratio_shift_{shift}'] = df['OC_ratio'].shift(shift).fillna(0)

#         # === 2. Close % Change (Momentum) ===
#         for shift in [1, 2, 3, 5, 9]:
#             df[f'Close_Change_{shift}'] = df['Close'].pct_change(shift).replace([np.inf, -np.inf], 0).fillna(0)

#         # === 3. Candle Shape ===
#         df['body'] = abs(df['Close'] - df['Open'])
#         df['range'] = df['High'] - df['Low'] + eps
#         df['body_ratio'] = df['body'] / df['range']

#         # === 4. Wick Logic (optional, useful for pattern learning) ===
#         df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
#         df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
#         df['upper_wick_ratio'] = df['upper_wick'] / df['range']
#         df['lower_wick_ratio'] = df['lower_wick'] / df['range']

#         # === 5. Volume Normalization ===
#         df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps)
#         df['volume_zscore'] = df['volume_zscore'].fillna(0)

#         # === 6. Volume-Weighted OC Move ===
#         df['vol_weighted_oc'] = df['OC_ratio'] * df['Volume']
#         df['vol_weighted_body'] = df['body'] * df['Volume']

#         return df
#     df = add_price_volume_features(df)



#     def add_volume_features(df):
#         eps = 1e-6  # for division safety

#         # === On-Balance Volume (OBV) ===
#         obv = ta.volume.OnBalanceVolumeIndicator(
#             close=df['Close'], volume=df['Volume']
#         ).on_balance_volume()
#         df['OBV'] = obv.fillna(0)

#         # OBV Engineering
#         df['obv_change'] = df['OBV'].diff().fillna(0)
#         df['obv_pct_change'] = df['OBV'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
#         df['obv_slope'] = np.gradient(df['OBV'].fillna(0))
#         df['obv_zscore'] = (
#             (df['OBV'] - df['OBV'].rolling(20).mean()) /
#             (df['OBV'].rolling(20).std() + eps)
#         ).fillna(0)

#         # === Money Flow Index (MFI) ===
#         mfi = ta.volume.MFIIndicator(
#             high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
#         ).money_flow_index()
#         df['MFI_14'] = mfi.fillna(50)  # center default to neutral

#         # MFI Engineering
#         df['mfi_slope'] = np.gradient(df['MFI_14'].fillna(0))
#         df['mfi_zscore'] = (
#             (df['MFI_14'] - df['MFI_14'].rolling(20).mean()) /
#             (df['MFI_14'].rolling(20).std() + eps)
#         ).fillna(0)

#         return df
#     df = add_volume_features(df)


#     # ─── NEW: Bearish / Bullish Engulfing Pattern (binary flags) ───
#     df['bull_engulf'] = (
#         ((df['Close'].shift(1) < df['Open'].shift(1)) &  # prior candle bearish
#          (df['Close'] > df['Open']) &                    # current candle bullish
#          (df['Open'] < df['Close'].shift(1)) & 
#          (df['Close'] > df['Open'].shift(1)))
#     ).astype(int)
#     df['bear_engulf'] = (
#         ((df['Close'].shift(1) > df['Open'].shift(1)) &  # prior candle bullish
#          (df['Close'] < df['Open']) &                    # current candle bearish
#          (df['Open'] > df['Close'].shift(1)) &
#          (df['Close'] < df['Open'].shift(1)))
#     ).astype(int)


#     # === 6) Multi‐Timeframe Slopes & Statistics ===
#     timeframes = {'15min': 3, '1hr': 12, '4hr': 48, '1day': 288}
#     def calculate_slope(series, lag):
#         return (series - series.shift(lag)) / lag

#     for name, period in timeframes.items():
#         df[f'RSI_Slope_{name}']   = calculate_slope(df['RSI'], period)
#         df[f'MACD_Slope_{name}']  = calculate_slope(df['MACD_Line'], period)

#         df[f'RSI_Mean_{name}']    = df['RSI'].rolling(period).mean()
#         df[f'MACD_Mean_{name}']   = df['MACD_Line'].rolling(period).mean()

#         df[f'RSI_Std_{name}']     = df['RSI'].rolling(period).std()
#         df[f'MACD_Std_{name}']    = df['MACD_Line'].rolling(period).std()

#         df[f'RSI_PctRank_{name}'] = df['RSI'].rank(pct=True).rolling(period)\
#                                       .apply(lambda x: x[-1], raw=True)
#         df[f'MACD_PctRank_{name}'] = df['MACD_Line'].rank(pct=True).rolling(period)\
#                                       .apply(lambda x: x[-1], raw=True)


#     # === 7) Bollinger Bands & Related Features ===
#     def add_bollinger_features(df, window=20, window_dev=2.0):
#         eps = 1e-6

#         bb = ta.volatility.BollingerBands(close=df['Close'], window=window, window_dev=window_dev)
#         df['BB_Lower'] = bb.bollinger_lband()
#         df['BB_Upper'] = bb.bollinger_hband()
#         df['BB_Mid'] = bb.bollinger_mavg()

#         # Core Band Metrics
#         df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
#         df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + eps)
#         df['BB_Close_Dist_Mid'] = df['Close'] - df['BB_Mid']
#         df['BB_Close_Dist_Lower'] = df['Close'] - df['BB_Lower']
#         df['BB_Close_Dist_Upper'] = df['BB_Upper'] - df['Close']

#         # Slopes and Expansion
#         df['BB_Mid_Slope'] = df['BB_Mid'].diff().fillna(0)
#         df['BB_Upper_Slope'] = df['BB_Upper'].diff().fillna(0)
#         df['BB_Lower_Slope'] = df['BB_Lower'].diff().fillna(0)
#         df['BB_Expansion'] = df['BB_Width'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

#         # Volatility Regimes
#         squeeze_thresh = df['BB_Width'].rolling(window).quantile(0.15)
#         expand_thresh = df['BB_Width'].rolling(window).quantile(0.85)
#         df['BB_Is_Squeeze'] = (df['BB_Width'] < squeeze_thresh).astype(int)
#         df['BB_Is_Expansion'] = (df['BB_Width'] > expand_thresh).astype(int)

#         # Squeeze Duration
#         def squeeze_duration(series):
#             groups = (series != series.shift()).cumsum()
#             return series.groupby(groups).cumcount()

#         df['BB_Squeeze_Duration'] = squeeze_duration(df['BB_Is_Squeeze'])

#         # Touches and Reactions
#         df['BB_Touch_Upper'] = (df['High'] >= df['BB_Upper']).astype(int)
#         df['BB_Touch_Lower'] = (df['Low'] <= df['BB_Lower']).astype(int)
#         df['BB_Close_Outside'] = ((df['Close'] > df['BB_Upper']) | (df['Close'] < df['BB_Lower'])).astype(int)

#         # Band Position (0 = bottom, 1 = top, >1 = breakout)
#         df['BB_Band_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + eps)

#         return df
#     df = add_bollinger_features(df)

#     # === 8) VWAP & Price‐vs‐VWAP ===
#     df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-6)
#     df['Price_vs_VWAP'] = df['Close'] - df['VWAP']

#     # === 9) Mean Reversion & Candle Pattern Features ===
#     df['Deviation_From_Mean'] = df['Close'] - df['Close'].rolling(20).mean()

#     # === 10) Volatility Context (ATR multiples, Range Expansion) ===
#     df['rolling_std'] = df['Close'].rolling(window=14).std()
#     df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
#     df['ATR_14'] = (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) / df['Close'].rolling(14).mean()
#     df['ATR_48'] = (df['High'].rolling(48).max() - df['Low'].rolling(48).min()) / df['Close'].rolling(48).mean()
#     df['ATR_96'] = (df['High'].rolling(96).max() - df['Low'].rolling(96).min()) / df['Close'].rolling(96).mean()

#     df['range_expansion_14'] = (df['High'] - df['Low']) / df['ATR_14']
#     df['range_expansion_48'] = (df['High'] - df['Low']) / df['ATR_48']
#     df['range_expansion_96'] = (df['High'] - df['Low']) / df['ATR_96']

   
#     df['fractal_energy'] = df['Close'].rolling(5).apply(
#         lambda x: np.sum(np.abs(np.diff(np.log(x)))) / (np.max(x) - np.min(x) + 1e-6)
#     )

#     # === 12) Timeframe Synergy Score (MACD Alignment) ===
#     for tf in ['15min', '1hr', '4hr']:
#         df[f'{tf}_macd_sync'] = (
#             df['MACD_Line'].rolling(timeframes[tf]).mean() /
#             (df['MACD_Line'].rolling(timeframes[tf]).std() + 1e-6)
#         )

#     # === 13) Composite Momentum Oscillator (GOD Oscillator) ===
#     df['god_oscillator'] = (
#         0.5 * (df['RSI'] - 50) +
#         0.3 * (df['MACD_Histogram'] * 10) +
#         0.2 * (df['BB_Width'] / df['ATR'])
#     )

#     # === 15) Pain‐to‐Gain Ratios ===
#     df['pain_ratio'] = df['Low'].rolling(5).min().pct_change() / (df['High'].rolling(5).max().pct_change() + 1e-6)
#     df['gain_ratio'] = df['High'].rolling(5).max().pct_change() / (df['Low'].rolling(5).min().pct_change() + 1e-6)

   
#     # === 18) Ichimoku‐Derived Features (unchanged) ===
#     def compute_ichimoku_features_from_5m(df: pd.DataFrame) -> pd.DataFrame:
#         timeframes = {'15min': 3, '1h': 12, '4h': 48, '1d': 288}
#         ichimoku = ta.trend.IchimokuIndicator(
#             high=df['High'], low=df['Low'], window1=9, window2=26, window3=52
#         )
#         df['Tenkan']   = ichimoku.ichimoku_conversion_line()
#         df['Kijun']    = ichimoku.ichimoku_base_line()
#         df['Senkou_A'] = ichimoku.ichimoku_a()
#         df['Senkou_B'] = ichimoku.ichimoku_b()
#         df['tenkan_kijun_delta'] = df['Tenkan'] - df['Kijun']

#         for label, window in timeframes.items():
#             df[f'{label}_tk_delta_slope']       = (df['tenkan_kijun_delta'] - df['tenkan_kijun_delta'].shift(window)) / window
#             df[f'{label}_tk_delta_mean']        = df['tenkan_kijun_delta'].rolling(window).mean()
#             df[f'{label}_tk_delta_std']         = df['tenkan_kijun_delta'].rolling(window).std()
#             df[f'{label}_tk_delta_pct_rank']    = df['tenkan_kijun_delta'].rank(pct=True).rolling(window).apply(lambda x: x[-1], raw=True)

#         return df

#     df = compute_ichimoku_features_from_5m(df)


#     # === 19) Initialize Columns for Strategy Logic ===
#     df['entry_price']         = 0.0
#     df['stop_loss_price']     = 0.0
#     df['stop_loss_distance']  = 0.0
#     df['sl_ratio_to_entry']   = 0.0
#     df['count_above']         = 0
#     df['count_below']         = 0
#     df['count_above_9']       = 0
#     df['count_above_20']      = 0
#     df['count_below_9']       = 0
#     df['count_below_20']      = 0
#     df['side']                = -1     # 1=Long, 0=Short, -1=No Trade
#     df['strategy']            = -1     # 1..5 for specific patterns, 6=divergence_long, 7=divergence_short

#     df['feat_c1_lt_pre_s9']   = (df['Close'].shift(1) < df['SMA_9'].shift(1)).astype(int)
#     df['feat_c0_gt_s9']       = (df['Close'] > df['SMA_9']).astype(int)
#     df['feat_o1_gt_c1']       = (df['Open'].shift(1) > df['Close'].shift(1)).astype(int)
#     df['feat_o0_lt_c0']       = (df['Open'] < df['Close']).astype(int)

#     df['feat_c1_lt_pre_s20']  = (df['Close'].shift(1) < df['SMA_20'].shift(1)).astype(int)
#     df['feat_c0_gt_s20']      = (df['Close'] > df['SMA_20']).astype(int)

#     # Optional: also include cross-related features
#     df['feat_cross_above_s9']  = ((df['Close'].shift(1) < df['SMA_9'].shift(1)) & (df['Close'] > df['SMA_9'])).astype(int)
#     df['feat_cross_above_s20'] = ((df['Close'].shift(1) < df['SMA_20'].shift(1)) & (df['Close'] > df['SMA_20'])).astype(int)
    
#     df['feat_cross_below_s9']  = ((df['Close'].shift(1) > df['SMA_9'].shift(1)) & (df['Close'] < df['SMA_9'])).astype(int)
#     df['feat_cross_below_s20'] = ((df['Close'].shift(1) > df['SMA_20'].shift(1)) & (df['Close'] < df['SMA_20'])).astype(int)
    
#     df['gk']      = (df['gain_ratio'].shift(9) > df['gain_ratio']).astype(int)
#     df['pn']      = (df['pain_ratio'].shift(9) > df['gain_ratio']).astype(int)

#         # === 21) Human-Contextual Pattern Features (Human Eyes Simulation) ===
#     eps = 1e-6

#     # 1. Relative Candle Strength
#     df['bullish_strength'] = ((df['Close'] - df['Open']) / (df['High'] - df['Low'] + eps)).clip(0, 1)
#     df['bearish_strength'] = ((df['Open'] - df['Close']) / (df['High'] - df['Low'] + eps)).clip(0, 1)

#     # 2. Position in Local Swing
#     df['is_near_local_high'] = (df['Close'] >= df['High'].rolling(10).max()).astype(int)
#     df['is_near_local_low'] = (df['Close'] <= df['Low'].rolling(10).min()).astype(int)

#     # 3. Trend Consistency
#     df['above_sma9_5x'] = (
#         (df['Close'] > df['SMA_9'])
#         .rolling(5)
#         .apply(lambda x: int(x.all()), raw=True)
#         .fillna(0)
#     )
#     df['below_sma9_5x'] = (
#     (df['Close'] < df['SMA_9'])
#     .rolling(5)
#     .apply(lambda x: int(x.all()), raw=True)
#     .fillna(0)
#     )

#     # 4. Streak Memory
#     df['green_candle'] = (df['Close'] > df['Open']).astype(int)
#     df['red_candle'] = (df['Close'] < df['Open']).astype(int)
#     df['green_streak'] = df['green_candle'].rolling(3).sum()
#     df['red_streak'] = df['red_candle'].rolling(3).sum()

#     # 5. Volatility Spike
#     df['volatility_explosion'] = (df['ATR'] > df['ATR'].rolling(10).mean() * 1.5).astype(int)

#     # 6. Wick Rejections
#     df['upper_rejection'] = (df['upper_wick'] > df['body'] * 1.5).astype(int)
#     df['lower_rejection'] = (df['lower_wick'] > df['body'] * 1.5).astype(int)

#     # 7. Session Overlap (London + NY)
#     df['is_overlap_session'] = ((df['is_london_session'] == 1) & (df['is_new_york_session'] == 1)).astype(int)

#     # 8. Bollinger Band Rejection
#     df['upper_bb_reject'] = ((df['BB_Touch_Upper'] == 1) & (df['upper_rejection'] == 1)).astype(int)
#     df['lower_bb_reject'] = ((df['BB_Touch_Lower'] == 1) & (df['lower_rejection'] == 1)).astype(int)

#     # 9. Demand/Supply Pressure Zones
#     df['bull_pressure'] = (df['lower_wick_ratio'] > 0.6).astype(int)
#     df['bear_pressure'] = (df['upper_wick_ratio'] > 0.6).astype(int)


#     lookback            = 48
#     stoplossmarginbuy   = 0.2
#     stoplossmarginsell  = 0.2

#     low_arr    = df['Low'].values
#     high_arr   = df['High'].values
#     close_arr  = df['Close'].values
#     open_arr   = df['Open'].values
#     sma9_arr   = df['SMA_9'].values
#     sma20_arr  = df['SMA_20'].values
#     rsi_arr    = df['RSI'].values
#     tenken = df['Tenkan'].values
#     kenji = df['Kijun'].values
#     bolli_buy = df['BB_Lower'].values
#     bolli_sell = df['BB_Upper'].values

  
#     entry_price      = np.zeros(len(df))
#     stop_loss_price  = np.zeros(len(df))
#     stop_loss_distance = np.zeros(len(df))
#     sl_ratio_to_entry = np.zeros(len(df))
#     side_arr         = np.full(len(df), -1)
#     # strategy_arr     = np.full(len(df), -1)
#     count_above      = np.zeros(len(df))
#     count_below      = np.zeros(len(df))
#     count_above_9    = np.zeros(len(df))
#     count_above_20   = np.zeros(len(df))
#     count_below_9    = np.zeros(len(df))
#     count_below_20   = np.zeros(len(df))
    

#     # === 20) Strategy Logic (Long + Short) ===
#     for i in tqdm(range(3, len(df) - 1), desc="Fast strategy logic"):
#         l0, l1, l2 = low_arr[i], low_arr[i-1], low_arr[i-2]
#         h0, h1, h2 = high_arr[i], high_arr[i-1], high_arr[i-2]
#         c0, c1, c2 = close_arr[i], close_arr[i-1], close_arr[i-2]
#         o0, o1, o2 = open_arr[i], open_arr[i-1], open_arr[i-2]
#         s9, pre_s9   = sma9_arr[i], sma9_arr[i-1]
#         s20, pre_s20 = sma20_arr[i], sma20_arr[i-1]
#         slb = bolli_buy[i]
#         sls = bolli_sell[i]
#         tenken1 = tenken[i-1]
#         tenken0 =tenken[i]
#         kenji0 = kenji[i]
#         kenji1 = kenji[i-1]


#         # ─── Common Pattern Conditions ───
#         cond_1 = (c1 < pre_s9  and  c0 > s9) 
#         cond_2 = (c1 < pre_s20 and  c0 > s20)

#         # ─── 1) LONG Setups ─────────────────────────────────────────────────────
#         if cond_1 or cond_2:#or cond_3 or cond_4 or cond_5:
         
#             # Long entry condition: next bar’s high > this bar’s high
#             if True: 
#                 entry = c0
                
#                 sl = slb 
#                 dist = abs(entry - slb)

#                 ca_9  = np.sum(sma9_arr[i-lookback:i]   > close_arr[i-lookback:i])
#                 ca_20 = np.sum(sma20_arr[i-lookback:i]  > close_arr[i-lookback:i])
#                 cb_9  = np.sum(sma9_arr[i-lookback:i]   < close_arr[i-lookback:i])
#                 cb_20 = np.sum(sma20_arr[i-lookback:i]  < close_arr[i-lookback:i])
#                 ca    = np.sum(sma9_arr[i-lookback:i]   > sma20_arr[i-lookback:i])
#                 cb    = np.sum(sma9_arr[i-lookback:i]   < sma20_arr[i-lookback:i])

#                 entry_price[i]      = entry
#                 stop_loss_price[i]  = sl
#                 stop_loss_distance[i] = dist
#                 sl_ratio_to_entry[i]  = dist / entry if entry != 0 else 0
#                 side_arr[i]           = 1
#                 count_above[i]        = ca
#                 count_below[i]        = cb
#                 count_above_9[i]      = ca_9
#                 count_above_20[i]     = ca_20
#                 count_below_9[i]      = cb_9
#                 count_below_20[i]     = cb_20
       

#         # ─── 2) SHORT Setups (mirror of Long logic) ──────────────────────────────
        
#         cond_s1 = (c1 > pre_s9   and  c0 < s9)  # and (o1 < c1) and (o0 > c0)
#         cond_s2 = (c1 > pre_s20  and  c0 < s20) # and (o1 < c1) and (o0 > c0)

#         if cond_s1 or cond_s2:
            
#             if True:
#                 entry = c0
                
#                 sl = sls 
#                 dist = abs(entry - sls)

#                 ca_9  = np.sum(sma9_arr[i-lookback:i]   > close_arr[i-lookback:i])
#                 ca_20 = np.sum(sma20_arr[i-lookback:i]  > close_arr[i-lookback:i])
#                 cb_9  = np.sum(sma9_arr[i-lookback:i]   < close_arr[i-lookback:i])
#                 cb_20 = np.sum(sma20_arr[i-lookback:i]  < close_arr[i-lookback:i])
#                 ca    = np.sum(sma9_arr[i-lookback:i]   > sma20_arr[i-lookback:i])
#                 cb    = np.sum(sma9_arr[i-lookback:i]   < sma20_arr[i-lookback:i])

#                 entry_price[i]      = entry
#                 stop_loss_price[i]  = sl
#                 stop_loss_distance[i] = dist
#                 sl_ratio_to_entry[i]  = dist / entry if entry != 0 else 0
#                 side_arr[i]           = 0
#                 count_above[i]        = ca
#                 count_below[i]        = cb
#                 count_above_9[i]      = ca_9
#                 count_above_20[i]     = ca_20
#                 count_below_9[i]      = cb_9
#                 count_below_20[i]     = cb_20
        
        
#     # === 21) Assign back into DataFrame ===
#     df['entry_price']        = entry_price
#     df['stop_loss_price']    = stop_loss_price
#     df['stop_loss_distance'] = stop_loss_distance
#     df['sl_ratio_to_entry']  = sl_ratio_to_entry
#     df['side']               = side_arr
#     # df['strategy']           = strategy_arr
#     df['count_above']        = count_above
#     df['count_below']        = count_below
#     df['count_above_9']      = count_above_9
#     df['count_above_20']     = count_above_20
#     df['count_below_9']      = count_below_9
#     df['count_below_20']     = count_below_20

#     # Trim off any warm‐up rows if needed
#     df = df.iloc[290:].reset_index(drop=True)
#     return df




# def simulate_strategy(df, lookahead=96):
#     df = calculate_indicators(df)
    
#     df['label'] = np.nan
#     df['max_rr'] = np.nan
#     df['candle_to_outcome'] = np.nan
    
#     sides = df['side'].values
#     entrys = df['entry_price'].values
#     sls = df['stop_loss_price'].values
#     highs = df['High'].values
#     lows = df['Low'].values
#     closes = df['Close'].values

#     tp_points = 2  # TP points for long trades
    
#     for i in tqdm(range(len(df)), desc="Simulating trades"):
#         if sides[i] == -1:  # Skip if no trade
#             continue
            
#         side = sides[i]
#         entry = entrys[i]
#         sl = sls[i]
#         # Compute TP level
#         tp = entry + tp_points * (entry - sl) if side == 1 else entry - tp_points * (sl - entry)

#         rr_max = 0.0
#         candles_to_outcome = 0
#         labeled = False

#         for k in range(i + 1, min(i + lookahead + 1, len(df))):
#             h = highs[k]
#             l = lows[k]
#             c = closes[k]
#             candles_to_outcome += 1

#             # Calculate current RR
#             if side == 1:
#                 rr = (h - entry) / (entry - sl) if (entry - sl) != 0 else 0
#             else:
#                 rr = (entry - l) / (sl - entry) if (sl - entry) != 0 else 0
                
#             rr_max = max(rr_max, rr)

#             # Check for stop loss hit
#             if (side == 1 and l <= sl) or (side == 0 and h >= sl):
#                 # If SL is hit, check whether rr_max ever reached >= 1
#                 if rr_max >= 1.0:
#                     df.at[i, 'label'] = -1
#                 else:
#                     df.at[i, 'label'] = 0
#                 labeled = True
#                 break

#             # Check for take profit hit
#             if (side == 1 and h >= tp) or (side == 0 and l <= tp):
#                 df.at[i, 'label'] = 1
#                 labeled = True
#                 break

#         if not labeled:
#             # Neither TP nor SL hit within lookahead → expired
#             final_px = closes[min(i + lookahead, len(df) - 1)]
#             if side == 1:
#                 df.at[i, 'label'] = -1 if final_px > entry else 0
#             else:
#                 df.at[i, 'label'] = -1 if final_px < entry else 0

#         df.at[i, 'candle_to_outcome'] = candles_to_outcome
#         df.at[i, 'max_rr'] = rr_max

#     df = df.dropna(subset=['side'])  # Only keep rows with trades
#     df.dropna(inplace=True)
#     return df


# # === Usage Example ===
# # df = pd.read_csv("5mgbpfortest.csv", ), sep="\t", encoding = "utf-16"
# df = pd.read_csv("200g.csv")


# df = simulate_strategy(df)
# df.to_csv("200gx.csv", index=False)

# # === Summary ===
# print("Win Count:", sum(df['label'] == 1))
# print("Loss Count:", sum(df['label'] == 0))
# print("Beakeven Count:", sum(df['label'] == -1))
# print("Long Trades:", sum(df['side'] == 1))
# print("Short Trades:", sum(df['side'] == 0))
# print("Long Wins:", sum((df['side'] == 1) & (df['label'] == 1)))
# print("Short Wins:", sum((df['side'] == 0) & (df['label'] == 1)))
# print("\n=== Max RR Stats ===")
# print("Avg RR:", df['max_rr'].mean())
# print("Median RR:", df['max_rr'].median())
# print("Top RR:", df['max_rr'].max())
# win_count = sum(df['label'] == 1)
# loss_count = sum(df['label'] == 0)
# total_trades = win_count + loss_count

# if total_trades > 0:
#     win_rate = (win_count / total_trades) * 100
#     print(f"Win Rate: {win_rate:.2f}%")
# else:
#     print("No trades found.")

# print(df.head())  # Should now show the earliest time first

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# # === 1. Load your data (keep label -1 for future use) ===
# df = pd.read_csv("200gx.csv")

# # === 2. Prepare LSTM Input (using only label 0 and 1 for training) ===
# def get_lstm_ready_data(df, label_column="rr_class", window_size=48):
#     df_labeled = df[df[label_column].isin([0, 1])].copy()
#     drop_cols = ["rr_label", "rr_class", "time_to_outcome", "pair"]
#     X = df_labeled.drop(columns=[col for col in drop_cols if col in df_labeled.columns], errors='ignore')
#     y = df_labeled[label_column].values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_seq, y_seq = [], []
#     for i in range(window_size, len(X_scaled)):
#         X_seq.append(X_scaled[i - window_size:i])
#         y_seq.append(y[i])
#     return np.array(X_seq), np.array(y_seq), scaler

# X_lstm, y_lstm, scaler = get_lstm_ready_data(df)

# # === 3. Use Best Hyperparameters ===
# best_params = {
#     'lstm_units': 100,
#     'dense_units': 41,
#     'dropout_rate': 0.3753223118115935,
#     'learning_rate': 0.0001976054212468114
# }

# # === 4. Train/Val Split ===
# X_train, X_val, y_train, y_val = train_test_split(
#     X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm
# )

# # === 5. Build Model ===
# model = Sequential([
#     LSTM(best_params["lstm_units"], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
#     Dropout(best_params["dropout_rate"]),
#     Dense(best_params["dense_units"], activation='relu'),
#     Dropout(best_params["dropout_rate"]),
#     Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
#               loss='binary_crossentropy', metrics=['accuracy'])

# early_stop = EarlyStopping(patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, validation_data=(X_val, y_val),
#           epochs=20, batch_size=64, callbacks=[early_stop])

# # === 6. Evaluation ===
# pred_probs = model.predict(X_val).flatten()
# pred_labels = (pred_probs > 0.5).astype(int)

# print("\n✅ Classification Report:")
# print(classification_report(y_val, pred_labels))
# print("✅ AUC:", roc_auc_score(y_val, pred_probs))

# # === 7. Confusion Matrix ===
# cm = confusion_matrix(y_val, pred_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Buy"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()

# # === 8. Optional: Probabilities Plot ===
# plt.figure(figsize=(10, 4))
# plt.hist(pred_probs[y_val == 0], bins=50, alpha=0.6, label='Class 0', color='red')
# plt.hist(pred_probs[y_val == 1], bins=50, alpha=0.6, label='Class 1', color='green')
# plt.title("Predicted Probabilities")
# plt.xlabel("Probability")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # === 9. Save Model ===
# model.save("lstm_trading_model.h5")
# print("✅ Model saved to lstm_trading_model.h5")




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import optuna



# # === 1. Load your data ===
# df = pd.read_csv("./csv/gbp.csv")

# # === 2. Prepare Data for LSTM Regression ===
# def get_lstm_regression_data(df, target_column="rr_label", window_size=22):
#     df = df[df[target_column] >= 0].copy()  # filter out invalid or missing rr_labels
#     drop_cols = ["rr_label", "rr_class", "time_to_outcome","pair"]
#     X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
#     y = df[target_column].values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_seq, y_seq = [], []
#     for i in range(window_size, len(X_scaled)):
#         X_seq.append(X_scaled[i - window_size:i])
#         y_seq.append(y[i])
#     return np.array(X_seq), np.array(y_seq), scaler

# X_lstm, y_lstm, scaler = get_lstm_regression_data(df)

# # === 3. Optuna Tuning ===
# def objective(trial):
#     lstm_units = trial.suggest_int("lstm_units", 32, 128)
#     dense_units = trial.suggest_int("dense_units", 16, 64)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

#     X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

#     model = Sequential([
#         LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
#         Dropout(dropout_rate),
#         Dense(dense_units, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='linear')  # Linear activation for regression
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                   loss='mean_squared_error', metrics=['mae'])

#     early_stop = EarlyStopping(patience=20, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_val, y_val),
#               epochs=100, batch_size=64, callbacks=[early_stop], verbose=0)

#     y_pred = model.predict(X_val).flatten()
#     return -mean_squared_error(y_val, y_pred)  # minimize MSE

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=12)

# # === 4. Train Final Regressor ===
# best_params = study.best_params
# print("✅ Best Params:", best_params)

# X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# model = Sequential([
#     LSTM(best_params["lstm_units"], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
#     Dropout(best_params["dropout_rate"]),
#     Dense(best_params["dense_units"], activation='relu'),
#     Dropout(best_params["dropout_rate"]),
#     Dense(1, activation='linear')
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
#               loss='mean_squared_error', metrics=['mae'])

# early_stop = EarlyStopping(patience=20, restore_best_weights=True)
# model.fit(X_train, y_train, validation_data=(X_val, y_val),
#           epochs=100, batch_size=64, callbacks=[early_stop])

# # === 5. Evaluate ===
# y_pred = model.predict(X_val).flatten()
# lstm_rmse = mean_squared_error(y_val, y_pred, squared=False)
# lstm_r2 = r2_score(y_val, y_pred)
# lstm_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# # Save the model
# model.save("lstm_regressor_trading.h5")
# print("✅ Model saved as lstm_regressor_trading.h5")
# print(f"   📉 RMSE: {lstm_rmse:.4f}")
# print(f"   📈 R² Score: {lstm_r2:.4f}")
# print("📉 MSE:", mean_squared_error(y_val, y_pred))

























import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import optuna

# === 1. Load your labeled data with features ===
df = pd.read_csv("./csv/ml_trainingregressgbp1.csv")  # sep="\t", Replace with your file

# === 2. LSTM Input Preparation ===
def get_lstm_ready_data(df, label_column="rr_class", window_size=48):
    df = df[df[label_column].isin([0, 1])].copy()  # exclude -1
    drop_cols = ["rr_label", "rr_class", "time_to_outcome"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    y = df[label_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(window_size, len(X_scaled)):
        X_seq.append(X_scaled[i - window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq), scaler

X_lstm, y_lstm, scaler = get_lstm_ready_data(df)

# === 3. Optuna Tuning ===
def objective(trial):
    # Suggest hyperparameters
    lstm_units = trial.suggest_int("lstm_units", 32, 128)
    dense_units = trial.suggest_int("dense_units", 16, 64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm)

    # Model
    model = Sequential([
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=64, callbacks=[early_stop], verbose=0)

    pred_probs = model.predict(X_val)
    auc = roc_auc_score(y_val, pred_probs)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# === 4. Train Final Model with Best Params ===
best_params = study.best_params
print("✅ Best Params:", best_params)

X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm)

model = Sequential([
    LSTM(best_params["lstm_units"], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(best_params["dropout_rate"]),
    Dense(best_params["dense_units"], activation='relu'),
    Dropout(best_params["dropout_rate"]),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
              loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=64, callbacks=[early_stop])

# === 5. Evaluate and Save ===
pred_probs = model.predict(X_val)
pred_labels = (pred_probs > 0.5).astype(int)
pred_labels_70 = (pred_probs > 0.7).astype(int)

print("\nClassification Report:")
print(classification_report(y_val, pred_labels))
print(classification_report(y_val, pred_labels_70))

print("AUC:", roc_auc_score(y_val, pred_probs))

model.save("lstm_trading_model.h5")
print("✅ Model saved to lstm_trading_model.h5")
