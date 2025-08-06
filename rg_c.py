def calculate_features_buy(df, window=20, window_dev=2):
    """Calculate indicators & features"""
    # df = df.iloc[::-1].reset_index(drop=True)
    # print("\n🔹 Calculating Features...")
    # df.sort_values("Time", inplace=True)
    # df.sort_values("Time").reset_index(drop=True)
    #print(df.head())  # Should now show the earliest time first

    # === 1) Time & Session Features ===
   # === 1) Time & Session Features ===
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['hour'] = df['Time'].dt.hour

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # df['is_opening_session'] = ((df['hour'] >= 9) & (df['hour'] <= 11)).astype(int)
    # df['is_closing_session'] = ((df['hour'] >= 15) & (df['hour'] <= 17)).astype(int)

    # UTC‐based Forex sessions
    df['is_tokyo_session']   = ((df['hour'] >= 0)  & (df['hour'] <  9)).astype(int)
    df['is_london_session']  = ((df['hour'] >= 8)  & (df['hour'] < 17)).astype(int)
    df['is_new_york_session']= ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

    df.drop(columns=['Time', 'hour'], inplace=True)


    # === 2) Simple Moving Averages, Slopes, and Distances ===
    for w in [9, 20,50,100,200]:
        df[f'SMA_{w}'] = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
        df[f'SMA{w}_Distance'] = ((df['Close'] - df[f'SMA_{w}']) / df[f'SMA_{w}']) * 100
        df[f'SMA{w}_Slope'] = df[f'SMA_{w}'].diff(5)

  


    def zone_duration(series):
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).cumcount()

    def add_rsi_features(df, rsi_window=14, div_shift=9):
   

        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_window).rsi()
        df['RSI'] = rsi.fillna(0)

        # Crossovers
        df['rsi_above_70'] = (df['RSI'] > 70).astype(int)
        df['rsi_below_30'] = (df['RSI'] < 30).astype(int)

        # Fix: use parentheses for each condition to avoid TypeError
        cross_up_recent = (
            (df['RSI'].shift(1) <= 50) |
            (df['RSI'].shift(2) <= 50) |
            (df['RSI'].shift(3) <= 50)
        )
        cross_down_recent = (
            (df['RSI'].shift(1) >= 50) |
            (df['RSI'].shift(2) >= 50) |
            (df['RSI'].shift(3) >= 50)
        )

        df['rsi_cross_50_up'] = ((df['RSI'] > 50) & cross_up_recent).astype(int)
        df['rsi_cross_50_down'] = ((df['RSI'] < 50) & cross_down_recent).astype(int)

        # Trend & Volatility
        df['rsi_change'] = df['RSI'] - df['RSI'].shift(1).fillna(0)
        df['rsi_slope'] = np.gradient(df['RSI'])
        df['rsi_rolling_mean'] = df['RSI'].rolling(5).mean().fillna(df['RSI'])
        df['rsi_std'] = df['RSI'].rolling(5).std().fillna(0)
        df['rsi_zscore'] = (df['RSI'] - df['RSI'].rolling(20).mean()) / (df['RSI'].rolling(20).std() + 1e-6)
        df['rsi_zscore'] = df['rsi_zscore'].fillna(0)

        # OB/OS durations
        df['rsi_above_70_duration'] = zone_duration(df['rsi_above_70'])
        df['rsi_below_30_duration'] = zone_duration(df['rsi_below_30'])

        df.drop(columns=['rsi_above_70', 'rsi_below_30'], inplace=True)

        return df



    def add_cci_features(df, cci_window=20, div_shift=9):
        cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=cci_window).cci()
        df['CCI'] = cci.fillna(0)

        # Crossovers
        df['cci_above_100'] = (df['CCI'] > 100).astype(int)
        df['cci_below_minus100'] = (df['CCI'] < -100).astype(int)
        df['cci_cross_0_up'] = ((df['CCI'] > 0) & (df['CCI'].shift(1) <= 0)).astype(int)
        df['cci_cross_0_down'] = ((df['CCI'] < 0) & (df['CCI'].shift(1) >= 0)).astype(int)

        # Trend
        df['cci_change'] = df['CCI'] - df['CCI'].shift(1).fillna(0)
        df['cci_slope'] = np.gradient(df['CCI'])
        df['cci_rolling_mean'] = df['CCI'].rolling(5).mean().fillna(df['CCI'])
        df['cci_above_mean'] = (df['CCI'] > df['cci_rolling_mean']).astype(int)

        # Normalization
        df['cci_std'] = df['CCI'].rolling(5).std().fillna(0)
        df['cci_zscore'] = (df['CCI'] - df['CCI'].rolling(20).mean()) / (df['CCI'].rolling(20).std() + 1e-6)
        df['cci_zscore'] = df['cci_zscore'].fillna(0)


        # Duration in OB/OS zones
        df['cci_above_100_duration'] = zone_duration(df['cci_above_100'])
        df['cci_below_minus100_duration'] = zone_duration(df['cci_below_minus100'])

        # Distance from thresholds
        df['cci_to_100'] = (100 - df['CCI']).clip(lower=0)
        df['cci_to_minus100'] = (df['CCI'] + 100).clip(lower=0)

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
        df['macd_line_zscore'] = (df['MACD_Line'] - df['MACD_Line'].rolling(20).mean()) / (df['MACD_Line'].rolling(20).std() + 1e-6)
        df['macd_hist_zscore'] = (df['MACD_Histogram'] - df['MACD_Histogram'].rolling(20).mean()) / (df['MACD_Histogram'].rolling(20).std() + 1e-6)
        df['macd_line_zscore'] = df['macd_line_zscore'].fillna(0)
        

        # Crossovers
        df['macd_cross_signal'] = ((df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
       
        # Histogram direction
        df['macd_hist_direction'] = np.sign(df['MACD_Histogram'].diff().fillna(0)).astype(int)
        df['macd_hist_slope'] = np.gradient(df['MACD_Histogram'])

        df.drop(columns=['MACD_Histogram'], inplace=True)
        return df

    df = add_rsi_features(df, rsi_window=14, div_shift=9)
    df = add_cci_features(df, cci_window=20, div_shift=9)
    df = add_macd_features(df, div_shift=9)




    def add_price_volume_features(df):
        eps = 1e-6  # for division safety

        # === 1. Intrabar Movement ===
        df['OC_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + eps)
        for shift in [1, 2, 5]:
            df[f'OC_ratio_shift_{shift}'] = df['OC_ratio'].shift(shift).fillna(0)

        # === 2. Close % Change (Momentum) ===
        for shift in [1, 2,5]:
            df[f'Close_Change_{shift}'] = df['Close'].pct_change(shift).replace([np.inf, -np.inf], 0).fillna(0)

        # === 3. Candle Shape ===
        df['body'] = abs(df['Close'] - df['Open'])
        df['range'] = df['High'] - df['Low'] + eps
        df['body_ratio'] = df['body'] / df['range']
        df.drop(columns=['body'], inplace=True)

        # === 4. Wick Logic (optional, useful for pattern learning) ===
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['upper_wick_ratio'] = df['upper_wick'] / df['range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['range']

        # === 5. Volume Normalization ===
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps)
        df['volume_zscore'] = df['volume_zscore'].fillna(0)

        # === 6. Volume-Weighted OC Move ===
        df['vol_weighted_oc'] = df['OC_ratio'] * df['Volume']


        return df
    df = add_price_volume_features(df)



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

    # === 6) Multi‐Timeframe Slopes & Statistics ===
    timeframes = {'15min': 3, '1hr': 12, '4hr': 48, '1day': 288}
    def calculate_slope(series, lag):
        return (series - series.shift(lag)) / lag

    for name, period in timeframes.items():
        df[f'RSI_Slope_{name}']   = calculate_slope(df['RSI'], period)
        df[f'OBV_Slope_{name}']  = calculate_slope(df['OBV'], period)

        df[f'RSI_Mean_{name}']    = df['RSI'].rolling(period).mean()
        df[f'OBV_Mean_{name}']   = df['OBV'].rolling(period).mean()

        df[f'RSI_PctRank_{name}'] = df['RSI'].rank(pct=True).rolling(period)\
                                      .apply(lambda x: x[-1], raw=True)
        df[f'OBV_PctRank_{name}'] = df['OBV'].rank(pct=True).rolling(period)\
                                      .apply(lambda x: x[-1], raw=True)
    for label, period in timeframes.items():
        df[f'SMA9_{label}'] = df['Close'].rolling(period).mean()
        df[f'SMA9_{label}_distance'] = ((df['Close'] - df[f'SMA9_{label}']) / (df[f'SMA9_{label}'] + 1e-6)) * 100
        df[f'SMA9_{label}_above'] = (df['Close'] > df[f'SMA9_{label}']).astype(int)
        df[f'SMA9_{label}_below'] = (df['Close'] < df[f'SMA9_{label}']).astype(int)
        df[f'SMA9_{label}_slope'] = df[f'SMA9_{label}'].diff()



    # === 7) Bollinger Bands & Related Features ===
    def add_bollinger_features(df, window=10, window_dev=2.0):
        eps = 1e-6

        bb = ta.volatility.BollingerBands(close=df['Close'], window=window, window_dev=window_dev)
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Mid'] = bb.bollinger_mavg()

        # Core Band Metrics
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + eps)
        df['BB_Close_Dist_Mid'] = df['Close'] - df['BB_Mid']
        df['BB_Close_Dist_Lower'] = df['Close'] - df['BB_Lower']
        df['BB_Close_Dist_Upper'] = df['BB_Upper'] - df['Close']

        # Slopes and Expansion
        
        df['BB_Upper_Slope'] = df['BB_Upper'].diff().fillna(0)
        df['BB_Lower_Slope'] = df['BB_Lower'].diff().fillna(0)
       
        # Volatility Regimes
        squeeze_thresh = df['BB_Width'].rolling(window).quantile(0.15)
        expand_thresh = df['BB_Width'].rolling(window).quantile(0.85)
        df['BB_Is_Squeeze'] = (df['BB_Width'] < squeeze_thresh).astype(int)
        df['BB_Is_Expansion'] = (df['BB_Width'] > expand_thresh).astype(int)

        # Touches and Reactions
        df['BB_Touch_Upper'] = (df['High'] >= df['BB_Upper']).astype(int)
        df['BB_Touch_Lower'] = (df['Low'] <= df['BB_Lower']).astype(int)
        df['BB_Close_Upper'] = ((df['Close'] > df['BB_Upper'])).astype(int)
        df['BB_Close_Lower'] = ((df['Close'] < df['BB_Lower'])).astype(int)
        # Band Position (0 = bottom, 1 = top, >1 = breakout)
        df['BB_Band_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + eps)
        df.drop(columns=['BB_Mid'], inplace=True)
        
        return df
    df = add_bollinger_features(df)

    # === 8) VWAP & Price‐vs‐VWAP ===
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-6)
    df['Price_vs_VWAP'] = df['Close'] - df['VWAP']
    df.drop(columns=['VWAP'], inplace=True)
    # === 9) Mean Reversion & Candle Pattern Features ===
    df['Deviation_From_Mean'] = df['Close'] - df['Close'].rolling(20).mean()

    # === 10) Volatility Context (ATR multiples, Range Expansion) ===
    df['rolling_std'] = df['Close'].rolling(window=14).std()
    # df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    # df['ATR_14'] = (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) / df['Close'].rolling(14).mean()
    # df['ATR_48'] = (df['High'].rolling(48).max() - df['Low'].rolling(48).min()) / df['Close'].rolling(48).mean()
    # df['ATR_96'] = (df['High'].rolling(96).max() - df['Low'].rolling(96).min()) / df['Close'].rolling(96).mean()

    # df['range_expansion_14'] = (df['High'] - df['Low']) / df['ATR_14']
    # df['range_expansion_48'] = (df['High'] - df['Low']) / df['ATR_48']
    # df['range_expansion_96'] = (df['High'] - df['Low']) / df['ATR_96']

   
    # df['fractal_energy'] = df['Close'].rolling(5).apply(
    #     lambda x: np.sum(np.abs(np.diff(np.log(x)))) / (np.max(x) - np.min(x) + 1e-6)
    # )

    # # === 12) Timeframe Synergy Score (MACD Alignment) ===
    # for tf in ['15min', '1hr', '4hr']:
    #     df[f'{tf}_macd_sync'] = (
    #         df['MACD_Line'].rolling(timeframes[tf]).mean() /
    #         (df['MACD_Line'].rolling(timeframes[tf]).std() + 1e-6)
    #     )

    # === 13) Composite Momentum Oscillator (GOD Oscillator) ===
    # df['god_oscillator'] = (
    #     0.5 * (df['SMA9_Distance'] - 50) +
    #     0.3 * (df['rsi_slope'] * 10) +
    #     0.2 * (df['obv_slope'] / df['ATR'])
    # )
    df['god_oscillator'] = (
        0.5 * (df['SMA9_Distance'] - 50) +
        0.3 * (df['rsi_slope'] * 10) +
        0.2 * (df['obv_slope'] / (df['rolling_std'] + 1e-6))
    )


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


    # === 19) Initialize Columns for Strategy Logic ===
    df['entry_price']         = 0.0
    df['stop_loss_price']     = 0.0
    df['stop_loss_distance']  = 0.0
    df['sl_ratio_to_entry']   = 0.0
    df['count_above']         = 0
    df['count_below']         = 0
    df['count_above_9']       = 0
    df['count_above_20']      = 0
    df['count_below_9']       = 0
    df['count_below_20']      = 0
    df['side']                = -1     # 1=Long, 0=Short, -1=No Trade
   
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

        # === 21) Human-Contextual Pattern Features (Human Eyes Simulation) ===
    eps = 1e-6

    # 1. Relative Candle Strength
    df['bullish_strength'] = ((df['Close'] - df['Open']) / (df['High'] - df['Low'] + eps)).clip(0, 1)
    df['bearish_strength'] = ((df['Open'] - df['Close']) / (df['High'] - df['Low'] + eps)).clip(0, 1)

  
    # 4. Streak Memory
    df['green_candle'] = (df['Close'] > df['Open']).astype(int)
    df['red_candle'] = (df['Close'] < df['Open']).astype(int)
    df['green_streak'] = df['green_candle'].rolling(3).sum()
    df['red_streak'] = df['red_candle'].rolling(3).sum()

    # ────────────── STATISTICAL FEATURE ENGINE ──────────────
    # Price Distribution
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(5).std()
    df['skewness'] = df['log_returns'].rolling(5).skew()
    df['kurtosis'] = df['log_returns'].rolling(5).kurt()
    df['volatility_20'] = df['log_returns'].rolling(5).std()
    df['skewness_20'] = df['log_returns'].rolling(5).skew()
    df['kurtosis_20'] = df['log_returns'].rolling(5).kurt()
    df['volatility_10'] = df['log_returns'].rolling(5).std()
    df['skewness_10'] = df['log_returns'].rolling(5).skew()
    df['kurtosis_10'] = df['log_returns'].rolling(5).kurt()

    # Quantile Levels
    df['quantile_20'] = df['Close'].rolling(5).quantile(0.1)
    df['quantile_80'] = df['Close'].rolling(5).quantile(0.9)

    roll_max = df['Close'].rolling(5).max()
    df['drawdown'] = df['Close'] / roll_max - 1.0
    df['max_drawdown'] = df['drawdown'].rolling(5).min()


    # Fast time-domain motion
    df['velocity'] = df['Close'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['smoothed_velocity_5'] = df['velocity'].rolling(5).mean()
    df['smoothed_acceleration_5'] = df['acceleration'].rolling(5).mean()
    df['cum_log_returns'] = df['log_returns'].cumsum()

    
    df['returns'] = df['Close'].pct_change()
    df['rolling_std'] = df['returns'].rolling(5).std()
  

    lookback            = 48
  
    low_arr    = df['Low'].values
    high_arr   = df['High'].values
    close_arr  = df['Close'].values
    open_arr   = df['Open'].values
    sma9_arr   = df['SMA_9'].values
    sma20_arr  = df['SMA_20'].values

    bolli_buy  = df['BB_Lower'].values
    bolli_sell = df['BB_Upper'].values

    entry_price         = np.zeros(len(df))
    stop_loss_price     = np.zeros(len(df))
    stop_loss_distance  = np.zeros(len(df))
    sl_ratio_to_entry   = np.zeros(len(df))
    side_arr            = np.full(len(df), -1)

    count_above         = np.zeros(len(df))
    count_below         = np.zeros(len(df))
    count_above_9       = np.zeros(len(df))
    count_above_20      = np.zeros(len(df))
    count_below_9       = np.zeros(len(df))
    count_below_20      = np.zeros(len(df))


    # === 20) Strategy Logic (Long + Short) ===
    for i in (range(3, len(df) - 1)):
        EPS = 1e-9
        
    bbwidth_thresh = df['BB_Width'].quantile(0.15)
    volatility_thresh = df['rolling_std'].quantile(0.15)
    
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

        for i in tqdm(range(1, len(df) - max_lookahead - 1), desc='Sim trades'):
            row = df.iloc[i]
            prev_close, curr_close = df['Close'].iloc[i - 1], df['Close'].iloc[i]
            prev_sma9, curr_sma9 = df['SMA_9'].iloc[i - 1], df['SMA_9'].iloc[i]
            valid_hour = df['valid_hour'].iloc[i]
            next_high = df['High'].iloc[i+1] > df['High'].iloc[i]
            
            if prev_close < prev_sma9 and curr_close > curr_sma9 and valid_hour and next_high:
                entry_price = df['High'].iloc[i]
                sl_price = df['BB_Lower'].iloc[i]

                if np.isnan(sl_price) or sl_price >= entry_price:
                    continue

                sl_dist = entry_price - sl_price
                if sl_dist <= 0:
                    continue

                if not passes_filter(row):
                    continue
                filtered_in += 1

                rr_achieved = 0
                time_to_outcome = max_lookahead
                hit_type = -1  # Default to -1 (NoHit)

                for j in range(1, max_lookahead + 1):
                    future = df.iloc[i + j]
                    if future['Low'] <= sl_price:
                        hit_type = 0  # SL
                        time_to_outcome = j
                        break
                    rr = (future['High'] - entry_price) / sl_dist
                    if rr >= rr_cap:
                        rr_achieved = rr_cap
                        hit_type = 1  # TP
                        time_to_outcome = j
                        break
                    rr_achieved = max(rr_achieved, rr)

                if hit_type == -1 and rr_achieved > 0:
                    hit_type = 1 if rr_achieved >= 1 else 0  # optional fallback classification

                row = df.iloc[i].copy()
                row['entry_price'] = entry_price
                row['stop_loss_price'] = sl_price
                row['stop_loss_distance'] = sl_dist
                row['sl_ratio_to_entry'] = sl_dist / (entry_price + EPS)
                row['rr_label'] = round(min(rr_achieved, rr_cap), 3)
                row['rr_class'] = hit_type  # <-- Classification label
                row['time_to_outcome'] = time_to_outcome
                row['side'] = 1
                
                trades.append(row)

        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("ml_trainingregress.csv", index=False)
        print(f"✅ Saved {len(trades_df)} ML samples to 'ml_trainingregress.csv'")
        print(f"📊 RR label stats:\n{trades_df['rr_label'].describe()}")
        print(f"📊 Class distribution:\n{trades_df['rr_class'].value_counts()}")
        print(f"Filtered-In Trades: {filtered_in}")
        return trades_df

        l0, l1, l2 = low_arr[i], low_arr[i-1], low_arr[i-2]
        h0, h1, h2 = high_arr[i], high_arr[i-1], high_arr[i-2]
        c0, c1, c2 = close_arr[i], close_arr[i-1], close_arr[i-2]
        o0, o1, o2 = open_arr[i], open_arr[i-1], open_arr[i-2]
        s9, pre_s9   = sma9_arr[i], sma9_arr[i-1]
        s20, pre_s20 = sma20_arr[i], sma20_arr[i-1]

            ca_9  = np.sum(sma9_arr[i-lookback:i]   > close_arr[i-lookback:i])
            ca_20 = np.sum(sma20_arr[i-lookback:i]  > close_arr[i-lookback:i])
            cb_9  = np.sum(sma9_arr[i-lookback:i]   < close_arr[i-lookback:i])
            cb_20 = np.sum(sma20_arr[i-lookback:i]  < close_arr[i-lookback:i])
            ca    = np.sum(sma9_arr[i-lookback:i]   > sma20_arr[i-lookback:i])
            cb    = np.sum(sma9_arr[i-lookback:i]   < sma20_arr[i-lookback:i])

            entry_price[i]        = entry
            stop_loss_price[i]    = sl
            stop_loss_distance[i] = dist
            sl_ratio_to_entry[i]  = dist / entry if entry != 0 else 0
            side_arr[i]           = 1
            count_above[i]        = ca
            count_below[i]        = cb
            count_above_9[i]      = ca_9
            count_above_20[i]     = ca_20
            count_below_9[i]      = cb_9
            count_below_20[i]     = cb_20
    # === 21) Assign back to df ===
    df['entry_price']        = entry_price
    df['stop_loss_price']    = stop_loss_price
    df['stop_loss_distance'] = stop_loss_distance
    df['sl_ratio_to_entry']  = sl_ratio_to_entry
    df['side']               = side_arr

    df['count_above']        = count_above
    df['count_below']        = count_below
    df['count_above_9']      = count_above_9
    df['count_above_20']     = count_above_20
    df['count_below_9']      = count_below_9
    df['count_below_20']     = count_below_20

  
    def create_temporal_features(df, window=12):
        # Only lag key indicators
        important_cols = [
    
        'BB_Is_Expansion','BB_Is_Squeeze', 'BB_Touch_Lower','BB_Touch_Upper', 'OBV',
        'cci_above_mean', 'cci_slope', 'feat_c0_gt_s20', 'feat_cross_above_s20', 'feat_o1_gt_c1', 'gain_ratio', 'gk',
        'hour_cos', 'hour_sin', 'is_london_session', 'is_new_york_session', 'is_tokyo_session',
        'obv_slope', 'obv_zscore', 'pain_ratio', 'pn','upper_wick_ratio', 'volume_zscore'
        ]     


        for col in important_cols:
            if col in df.columns:
                for i in range(1, window + 1):
                    df[f"{col}_lag_{i}"] = df[col].shift(i)

        return df


