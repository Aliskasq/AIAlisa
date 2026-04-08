import pandas as pd
import numpy as np

def rma(series, length):
    """Running Moving Average (Wilder's Smoothing) used for RSI, ATR, ADX"""
    alpha = 1.0 / length
    res = np.zeros_like(series, dtype=float)
    res[0] = series.iloc[0] if isinstance(series, pd.Series) else series[0]
    for i in range(1, len(series)):
        val = series.iloc[i] if isinstance(series, pd.Series) else series[i]
        res[i] = alpha * val + (1 - alpha) * res[i - 1]
    return pd.Series(res, index=series.index)


def calc_dema(series, length):
    """Double EMA — used by CCMI (Chande Composite Momentum Index)"""
    e1 = series.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    return 2 * e1 - e2


def calc_cmo_component(src, period):
    """Chande Momentum Oscillator for given period, DEMA-smoothed (period 3)."""
    diff = src.diff()
    up_sum = diff.where(diff > 0, 0.0).rolling(period).sum()
    dn_sum = (-diff.where(diff < 0, 0.0)).rolling(period).sum()
    total = up_sum + dn_sum
    raw_cmo = 100 * (up_sum - dn_sum) / total.replace(0, np.nan)
    return calc_dema(raw_cmo.fillna(0), 3)

def calculate_binance_indicators(df: pd.DataFrame, tf_key: str):
    """
    Calculates basic (EMA, RSI, MACD, Fibo) + advanced indicators
    (SuperTrend, ADX, MFI, StochRSI, Ichimoku, OBV) + SMC (Order Blocks, FVG)
    for AI analysis.
    
    Returns: (dict with last candle + historical context, df with all indicators)
    """
    df = df.copy()

    # Convert types to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. EMA (7, 25, 99)
    df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema25'] = df['close'].ewm(span=25, adjust=False).mean()
    df['ema99'] = df['close'].ewm(span=99, adjust=False).mean()

    # 2. RSI (14) — standard period
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_g14 = rma(gain, 14)
    avg_l14 = rma(loss, 14)
    rs14 = avg_g14 / avg_l14.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs14))
    df['rsi14'] = df['rsi14'].fillna(50)

    # 3. MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # 4. (Volume Decay & Fibonacci removed — low value, noise)

    # ==========================================
    # 🚀 ADVANCED INDICATORS (NEW)
    # ==========================================

    # 6. ATR (Average True Range, param 14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = rma(tr, 14)  # ATR 14 for ADX
    df['atr'] = rma(tr, 10)    # ATR 10 for SuperTrend (matches Pine Script)

    # 7. SuperTrend (ATR 10, Multiplier 3) — matches Pine Script
    multiplier = 3.0
    hl2 = (df['high'] + df['low']) / 2
    basic_ub = hl2 + multiplier * df['atr']
    basic_lb = hl2 - multiplier * df['atr']

    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    st_dir = np.ones(len(df))

    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub[i-1] or df['close'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]

        if basic_lb.iloc[i] > final_lb[i-1] or df['close'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]

        if supertrend[i-1] == final_ub[i-1] and df['close'].iloc[i] < final_ub[i]:
            st_dir[i] = -1
        elif supertrend[i-1] == final_ub[i-1] and df['close'].iloc[i] > final_ub[i]:
            st_dir[i] = 1
        elif supertrend[i-1] == final_lb[i-1] and df['close'].iloc[i] > final_lb[i]:
            st_dir[i] = 1
        elif supertrend[i-1] == final_lb[i-1] and df['close'].iloc[i] < final_lb[i]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]

        if st_dir[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]

    df['supertrend_dir'] = st_dir
    df['supertrend'] = supertrend

    # 8. ADX (Average Directional Index, 14) with DI+ and DI-
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_safe = df['atr14'].replace(0, np.nan)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), 14) / atr_safe
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), 14) / atr_safe
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    dx = dx.fillna(0)
    df['adx'] = rma(dx, 14)
    df['di_plus'] = plus_di
    df['di_minus'] = minus_di

    # 9. Stochastic RSI (14, K=3, D=3)
    avg_g14 = rma(gain, 14)
    avg_l14 = rma(loss, 14)
    rs14 = avg_g14 / avg_l14.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs14))
    rsi14 = rsi14.fillna(50)
    stoch_rsi_min = rsi14.rolling(14).min()
    stoch_rsi_max = rsi14.rolling(14).max()
    stoch_range = stoch_rsi_max - stoch_rsi_min
    stoch_range = stoch_range.replace(0, np.nan)
    stoch_rsi_k = 100 * (rsi14 - stoch_rsi_min) / stoch_range
    df['stoch_k'] = stoch_rsi_k.rolling(3).mean().fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(50)

    # 10. MFI (Money Flow Index, 14)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']

    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0.0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0.0)

    pos_flow_sum = pd.Series(pos_flow).rolling(14).sum()
    neg_flow_sum = pd.Series(neg_flow).rolling(14).sum()

    money_ratio = pos_flow_sum / neg_flow_sum
    df['mfi'] = 100 - (100 / (1 + money_ratio))

    # 11. Ichimoku Cloud (9, 26, 52)
    tenkan_max = df['high'].rolling(9).max()
    tenkan_min = df['low'].rolling(9).min()
    df['tenkan_sen'] = (tenkan_max + tenkan_min) / 2

    kijun_max = df['high'].rolling(26).max()
    kijun_min = df['low'].rolling(26).min()
    df['kijun_sen'] = (kijun_max + kijun_min) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    senkou_max = df['high'].rolling(52).max()
    senkou_min = df['low'].rolling(52).min()
    df['senkou_span_b'] = ((senkou_max + senkou_min) / 2).shift(26)

    # 12. OBV (On-Balance Volume)
    obv_change = np.sign(df['close'].diff()) * df['volume']
    df['obv'] = obv_change.fillna(0).cumsum()
    df['obv_sma20'] = df['obv'].rolling(20).mean()

    # 13. Bollinger Bands (20, 2)
    bb_sma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_sma + 2 * bb_std
    df['bb_lower'] = bb_sma - 2 * bb_std
    df['bb_mid'] = bb_sma

    # 14. Keltner Channels + TTM Squeeze (20, ATR mult 1.5)
    kc_mid = bb_sma  # same 20-period SMA as BB
    kc_atr = rma(tr, 20)  # 20-period ATR for Keltner
    kc_mult = 1.5
    df['kc_upper'] = kc_mid + kc_mult * kc_atr
    df['kc_lower'] = kc_mid - kc_mult * kc_atr

    # TTM Squeeze: BB inside Keltner = squeeze ON, BB outside = squeeze OFF (fired)
    df['ttm_squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

    # TTM Squeeze Momentum histogram (close - midline of Donchian/KC blend)
    highest_20 = df['high'].rolling(20).max()
    lowest_20 = df['low'].rolling(20).min()
    donchian_mid = (highest_20 + lowest_20) / 2
    ttm_mom_raw = df['close'] - (donchian_mid + bb_sma) / 2
    df['ttm_mom'] = ttm_mom_raw.ewm(span=5, adjust=False).mean()  # smoothed

    # 15. CMF (Chaikin Money Flow, 20)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df['volume']
    df['cmf'] = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

    # ==========================================
    # 🆕 CCMI — Chande Composite Momentum Index
    # ==========================================
    close_s = df['close']

    cmo5 = calc_cmo_component(close_s, 5)
    cmo10 = calc_cmo_component(close_s, 10)
    cmo20 = calc_cmo_component(close_s, 20)

    std5 = close_s.rolling(5).std().fillna(0)
    std10 = close_s.rolling(10).std().fillna(0)
    std20 = close_s.rolling(20).std().fillna(0)

    std_sum = std5 + std10 + std20
    ccmi_raw = (std5 * cmo5 + std10 * cmo10 + std20 * cmo20) / std_sum.replace(0, np.nan)
    ccmi_raw = ccmi_raw.fillna(0)

    df['ccmi'] = ccmi_raw.ewm(span=3, adjust=False).mean()
    df['ccmi_signal'] = ccmi_raw.rolling(5).mean()

    # ==========================================
    # 🆕 IMI — Intraday Momentum Index
    # ==========================================
    imi_length = 14
    body_gain = (close_s - df['open']).where(close_s > df['open'], 0.0)
    body_loss = (df['open'] - close_s).where(close_s < df['open'], 0.0)

    imi_upt = body_gain.rolling(imi_length).sum()
    imi_dnt = body_loss.rolling(imi_length).sum()
    imi_total = imi_upt + imi_dnt
    df['imi'] = (100 * imi_upt / imi_total.replace(0, np.nan)).fillna(50)
    df['imi_ma'] = df['imi'].ewm(span=6, adjust=False).mean()

    # ==========================================
    # 🆕 RSI-Momentum Divergence
    # ==========================================
    mom10 = close_s.diff(10)
    delta_mom = mom10.diff()
    gain_mom = delta_mom.where(delta_mom > 0, 0.0)
    loss_mom = -delta_mom.where(delta_mom < 0, 0.0)
    avg_g_mom = rma(gain_mom.fillna(0), 14)
    avg_l_mom = rma(loss_mom.fillna(0), 14)
    rs_mom = avg_g_mom / avg_l_mom.replace(0, np.nan)
    df['rsi_mom'] = (100 - (100 / (1 + rs_mom))).fillna(50)

    rsi_mom_bull_div = False
    rsi_mom_bear_div = False
    rsi_mom_bull_div_detail = ""
    rsi_mom_bear_div_detail = ""

    if len(df) >= 60:
        rsi_mom_arr = df['rsi_mom'].values
        low_arr = df['low'].values
        high_arr = df['high'].values
        n = len(df)
        pivot_lr = 5

        pivot_lows = []
        for i in range(pivot_lr, n - pivot_lr):
            is_pivot = True
            for j in range(1, pivot_lr + 1):
                if rsi_mom_arr[i] > rsi_mom_arr[i - j] or rsi_mom_arr[i] > rsi_mom_arr[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_lows.append(i)

        pivot_highs = []
        for i in range(pivot_lr, n - pivot_lr):
            is_pivot = True
            for j in range(1, pivot_lr + 1):
                if rsi_mom_arr[i] < rsi_mom_arr[i - j] or rsi_mom_arr[i] < rsi_mom_arr[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_highs.append(i)

        if len(pivot_lows) >= 2:
            curr_pl = pivot_lows[-1]
            prev_pl = pivot_lows[-2]
            bars_between = curr_pl - prev_pl
            if 5 <= bars_between <= 50:
                if low_arr[curr_pl] < low_arr[prev_pl] and rsi_mom_arr[curr_pl] > rsi_mom_arr[prev_pl]:
                    rsi_mom_bull_div = True
                    bars_ago = n - 1 - curr_pl
                    rsi_mom_bull_div_detail = (
                        f"Price LL ({low_arr[prev_pl]:.6f}->{low_arr[curr_pl]:.6f}) "
                        f"but RSI-Mom HL ({rsi_mom_arr[prev_pl]:.1f}->{rsi_mom_arr[curr_pl]:.1f}), "
                        f"{bars_ago} bars ago"
                    )

        if len(pivot_highs) >= 2:
            curr_ph = pivot_highs[-1]
            prev_ph = pivot_highs[-2]
            bars_between = curr_ph - prev_ph
            if 5 <= bars_between <= 50:
                if high_arr[curr_ph] > high_arr[prev_ph] and rsi_mom_arr[curr_ph] < rsi_mom_arr[prev_ph]:
                    rsi_mom_bear_div = True
                    bars_ago = n - 1 - curr_ph
                    rsi_mom_bear_div_detail = (
                        f"Price HH ({high_arr[prev_ph]:.6f}->{high_arr[curr_ph]:.6f}) "
                        f"but RSI-Mom LH ({rsi_mom_arr[prev_ph]:.1f}->{rsi_mom_arr[curr_ph]:.1f}), "
                        f"{bars_ago} bars ago"
                    )

    # ==========================================
    # ⏳ DYNAMIC PRICE CHANGE (1H/4H & 24H)
    # ==========================================
    current_price = df['close'].iloc[-1]

    # Calculate recent candle change
    if len(df) >= 2:
        prev_price = df['close'].iloc[-2]
        change_recent = ((current_price - prev_price) / prev_price) * 100
    else:
        change_recent = 0.0

    # Calculate 24h change dynamically based on timeframe
    if tf_key.lower() == '1d' and len(df) >= 2:
        price_24h_ago = df['close'].iloc[-2]  # 1 candle of 1D = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "1D"
    elif tf_key.lower() == '4h' and len(df) >= 7:
        price_24h_ago = df['close'].iloc[-7]  # 6 candles of 4H = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "4H"
    elif tf_key.lower() == '1h' and len(df) >= 25:
        price_24h_ago = df['close'].iloc[-25]  # 24 candles of 1H = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "1H"
    elif tf_key.lower() == '15m' and len(df) >= 97:
        price_24h_ago = df['close'].iloc[-97]  # 96 candles of 15m = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "15m"
    else:
        change_24h = 0.0
        recent_label = "Period"

    # ==========================================
    # 🔍 HISTORICAL CONTEXT ANALYSIS (NEW)
    # ==========================================
    
    # Pack the last candle data with all current indicators
    last = df.iloc[-1]
    cloud_top = max(last['senkou_span_a'], last['senkou_span_b']) if not pd.isna(last['senkou_span_a']) else 0
    cloud_bottom = min(last['senkou_span_a'], last['senkou_span_b']) if not pd.isna(last['senkou_span_a']) else 0

    if last['close'] > cloud_top: ichi_status = "ABOVE CLOUD (Bullish)"
    elif last['close'] < cloud_bottom: ichi_status = "BELOW CLOUD (Bearish)"
    else: ichi_status = "INSIDE CLOUD (Neutral/Chop)"
    
    # Assess OBV Trend Status
    obv_status = "Bullish (Accumulation)" if last['obv'] > last['obv_sma20'] else "Bearish (Distribution)"

    # ==========================================
    # ⭐ HISTORICAL CONTEXT FIELDS (NEW)
    # ==========================================
    
    # EMA History (50 candles)
    ema7_slope_50 = 0
    ema25_slope_50 = 0
    ema99_slope_50 = 0
    ema7_25_cross_bars = None
    ema7_25_cross_dir = None
    price_above_ema7_count = 0
    price_above_ema25_count = 0
    price_above_ema99_count = 0
    
    if len(df) >= 51:
        # EMA slopes over last 50 candles
        ema7_50_ago = df['ema7'].iloc[-51]
        ema25_50_ago = df['ema25'].iloc[-51]
        ema99_50_ago = df['ema99'].iloc[-51]
        
        if ema7_50_ago > 0:
            ema7_slope_50 = ((last['ema7'] - ema7_50_ago) / ema7_50_ago) * 100
        if ema25_50_ago > 0:
            ema25_slope_50 = ((last['ema25'] - ema25_50_ago) / ema25_50_ago) * 100
        if ema99_50_ago > 0:
            ema99_slope_50 = ((last['ema99'] - ema99_50_ago) / ema99_50_ago) * 100
            
        # Price above EMA counts (last 50 candles)
        last_50 = df.tail(50)
        price_above_ema7_count = (last_50['close'] > last_50['ema7']).sum()
        price_above_ema25_count = (last_50['close'] > last_50['ema25']).sum()
        price_above_ema99_count = (last_50['close'] > last_50['ema99']).sum()
    
    # EMA7-EMA25 cross detection (last 50 bars)
    if len(df) >= 50:
        ema7_last_50 = df['ema7'].tail(50)
        ema25_last_50 = df['ema25'].tail(50)
        
        # Find where EMA7 crosses EMA25
        for i in range(1, 50):
            prev_diff = ema7_last_50.iloc[-i-1] - ema25_last_50.iloc[-i-1]
            curr_diff = ema7_last_50.iloc[-i] - ema25_last_50.iloc[-i]
            
            # Golden cross (EMA7 crosses above EMA25)
            if prev_diff <= 0 and curr_diff > 0:
                ema7_25_cross_bars = i
                ema7_25_cross_dir = "golden"
                break
            # Death cross (EMA7 crosses below EMA25)
            elif prev_diff >= 0 and curr_diff < 0:
                ema7_25_cross_bars = i
                ema7_25_cross_dir = "death"
                break
    
    # MACD History (50 candles)
    macd_hist_trend_50 = []
    macd_hist_direction = "unknown"
    macd_bars_since_cross = None
    macd_hist_accel = 0
    macd_hist_min_50 = 0
    macd_hist_max_50 = 0
    macd_hist_avg_50 = 0
    macd_zero_crosses_50 = 0
    
    if len(df) >= 50:
        macd_hist_trend_50 = df['macd_hist'].tail(50).tolist()
        macd_hist_min_50 = min(macd_hist_trend_50)
        macd_hist_max_50 = max(macd_hist_trend_50)
        macd_hist_avg_50 = sum(macd_hist_trend_50) / len(macd_hist_trend_50)
        
        # Count zero-line crosses in last 50 bars (choppy = many crosses)
        for i in range(1, len(macd_hist_trend_50)):
            if (macd_hist_trend_50[i-1] <= 0 < macd_hist_trend_50[i]) or \
               (macd_hist_trend_50[i-1] >= 0 > macd_hist_trend_50[i]):
                macd_zero_crosses_50 += 1
        
        # Determine histogram direction (critical for momentum analysis)
        recent_trend = macd_hist_trend_50[-3:]
        last_h = recent_trend[-1]
        prev_h = recent_trend[-2]
        first_h = recent_trend[0]
        
        if prev_h <= 0 < last_h:
            macd_hist_direction = "turned_positive"
        elif prev_h >= 0 > last_h:
            macd_hist_direction = "turned_negative"
        elif last_h > 0 and last_h > first_h:
            macd_hist_direction = "growing"
        elif last_h > 0 and last_h < first_h:
            macd_hist_direction = "fading_bullish"
        elif last_h < 0 and last_h < first_h:
            macd_hist_direction = "shrinking"
        elif last_h < 0 and last_h > first_h:
            macd_hist_direction = "fading_bearish"
        else:
            macd_hist_direction = "stable"
        
        # MACD histogram acceleration
        change1 = macd_hist_trend_50[-1] - macd_hist_trend_50[-2]
        change2 = macd_hist_trend_50[-2] - macd_hist_trend_50[-3]
        macd_hist_accel = change1 - change2
    elif len(df) >= 5:
        macd_hist_trend_50 = df['macd_hist'].tail(5).tolist()
        if len(macd_hist_trend_50) >= 3:
            recent_trend = macd_hist_trend_50[-3:]
            last_h = recent_trend[-1]
            prev_h = recent_trend[-2]
            first_h = recent_trend[0]
            if prev_h <= 0 < last_h:
                macd_hist_direction = "turned_positive"
            elif prev_h >= 0 > last_h:
                macd_hist_direction = "turned_negative"
            elif last_h > 0 and last_h > first_h:
                macd_hist_direction = "growing"
            elif last_h > 0 and last_h < first_h:
                macd_hist_direction = "fading_bullish"
            elif last_h < 0 and last_h < first_h:
                macd_hist_direction = "shrinking"
            elif last_h < 0 and last_h > first_h:
                macd_hist_direction = "fading_bearish"
            else:
                macd_hist_direction = "stable"
            change1 = macd_hist_trend_50[-1] - macd_hist_trend_50[-2]
            change2 = macd_hist_trend_50[-2] - macd_hist_trend_50[-3]
            macd_hist_accel = change1 - change2
    
    # MACD line/signal cross detection (last 50 bars)
    if len(df) >= 50:
        macd_line_50 = df['macd_line'].tail(50)
        macd_signal_50 = df['macd_signal'].tail(50)
        
        for i in range(1, 50):
            prev_diff = macd_line_50.iloc[-i-1] - macd_signal_50.iloc[-i-1]
            curr_diff = macd_line_50.iloc[-i] - macd_signal_50.iloc[-i]
            
            if (prev_diff <= 0 < curr_diff) or (prev_diff >= 0 > curr_diff):
                macd_bars_since_cross = i
                break
    
    # OBV History (50 candles)
    obv_roc_50 = 0
    obv_roc_10 = 0
    obv_spike = False
    obv_price_divergence = "none"
    obv_trend_50 = "stable"
    
    if len(df) >= 51:
        obv_50_ago = df['obv'].iloc[-51]
        if obv_50_ago != 0:
            obv_roc_50 = ((last['obv'] - obv_50_ago) / abs(obv_50_ago)) * 100
    
    if len(df) >= 11:
        obv_10_ago = df['obv'].iloc[-11]
        if obv_10_ago != 0:
            obv_roc_10 = ((last['obv'] - obv_10_ago) / abs(obv_10_ago)) * 100
    
    # OBV trend over 50 candles (split into halves)
    if len(df) >= 50:
        obv_first_half = df['obv'].tail(50).head(25).mean()
        obv_second_half = df['obv'].tail(25).mean()
        if obv_second_half > obv_first_half * 1.02:
            obv_trend_50 = "rising"
        elif obv_second_half < obv_first_half * 0.98:
            obv_trend_50 = "falling"
    
    # OBV spike detection (2 std deviations over 50 candles)
    if len(df) >= 50:
        obv_changes = df['obv'].diff().tail(50)
        recent_obv_change = df['obv'].iloc[-1] - df['obv'].iloc[-3]  # Last 2 candles
        obv_std = obv_changes.std()
        if obv_std > 0 and abs(recent_obv_change) > 2 * obv_std:
            obv_spike = True
    
    # OBV-Price divergence (last 50 candles)
    if len(df) >= 50:
        last_50_obv = df.tail(50)
        price_change = (last_50_obv['close'].iloc[-1] - last_50_obv['close'].iloc[0]) / last_50_obv['close'].iloc[0]
        obv_val_start = last_50_obv['obv'].iloc[0]
        if abs(obv_val_start) > 0:
            obv_change = (last_50_obv['obv'].iloc[-1] - obv_val_start) / abs(obv_val_start)
        else:
            obv_change = 0
        
        if price_change > 0.02 and obv_change < -0.02:  # Price up, OBV down
            obv_price_divergence = "bearish"
        elif price_change < -0.02 and obv_change > 0.02:  # Price down, OBV up
            obv_price_divergence = "bullish"
    
    # RSI History (50 candles)
    rsi_trend_50 = "stable"
    rsi_values_50 = []
    rsi_min_50 = 50
    rsi_max_50 = 50
    rsi_avg_50 = 50
    rsi_time_overbought = 0  # candles above 70
    rsi_time_oversold = 0    # candles below 30
    rsi_price_divergence = "none"
    
    if len(df) >= 50:
        rsi_values_50 = df['rsi14'].tail(50).tolist()
        rsi_min_50 = min(rsi_values_50)
        rsi_max_50 = max(rsi_values_50)
        rsi_avg_50 = sum(rsi_values_50) / len(rsi_values_50)
        rsi_time_overbought = sum(1 for r in rsi_values_50 if r > 70)
        rsi_time_oversold = sum(1 for r in rsi_values_50 if r < 30)
        
        # RSI trend: compare first 10 avg vs last 10 avg
        first_10_avg = sum(rsi_values_50[:10]) / 10
        last_10_avg = sum(rsi_values_50[-10:]) / 10
        if last_10_avg > first_10_avg + 5:
            rsi_trend_50 = "rising"
        elif last_10_avg < first_10_avg - 5:
            rsi_trend_50 = "falling"
        
        # RSI-Price divergence (price making new highs but RSI declining, or vice versa)
        price_50 = df['close'].tail(50)
        rsi_50 = df['rsi14'].tail(50)
        price_first_half_max = price_50.head(25).max()
        price_second_half_max = price_50.tail(25).max()
        rsi_first_half_max = rsi_50.head(25).max()
        rsi_second_half_max = rsi_50.tail(25).max()
        
        if price_second_half_max > price_first_half_max and rsi_second_half_max < rsi_first_half_max - 5:
            rsi_price_divergence = "bearish"  # Higher price, lower RSI
        elif price_50.tail(25).min() < price_50.head(25).min() and rsi_50.tail(25).min() > rsi_50.head(25).min() + 5:
            rsi_price_divergence = "bullish"  # Lower price, higher RSI
    elif len(df) >= 5:
        rsi_values_50 = df['rsi14'].tail(5).tolist()
        if len(rsi_values_50) >= 3:
            if rsi_values_50[-1] > rsi_values_50[-3] + 2:
                rsi_trend_50 = "rising"
            elif rsi_values_50[-1] < rsi_values_50[-3] - 2:
                rsi_trend_50 = "falling"
    
    # RSI Pullback Peak Detection (last 50 candles)
    # Find the highest RSI value that was followed by a meaningful drop (pullback)
    # This gives a dynamic "danger level" — e.g. "last pullback started from RSI 82"
    rsi_pullback_peak = 0
    rsi_pullback_drop = 0
    if len(rsi_values_50) >= 10:
        # Scan backwards: find peaks where RSI dropped by ≥8 points after
        for i in range(len(rsi_values_50) - 3, 0, -1):
            val = rsi_values_50[i]
            if val >= 70:  # only care about overbought peaks
                # Check if RSI dropped significantly after this point
                future_min = min(rsi_values_50[i+1:min(i+10, len(rsi_values_50))])
                drop = val - future_min
                if drop >= 8:  # meaningful pullback (at least 8 RSI points)
                    rsi_pullback_peak = round(val, 1)
                    rsi_pullback_drop = round(drop, 1)
                    break  # take the most recent one

    # SuperTrend History (50 candles)
    st_bars_since_flip = 0
    st_distance_pct = 0
    st_flips_50 = 0  # number of direction changes in 50 bars (choppy = many)
    st_bullish_bars_50 = 0
    
    if len(df) >= 50:
        st_dir_series = df['supertrend_dir'].tail(50)
        # Find last flip
        for i in range(1, 50):
            if st_dir_series.iloc[-i] != st_dir_series.iloc[-i-1]:
                if st_bars_since_flip == 0:
                    st_bars_since_flip = i
                st_flips_50 += 1
        st_bullish_bars_50 = (st_dir_series == 1).sum()
    
    # Distance from SuperTrend line
    if last['supertrend'] > 0:
        st_distance_pct = ((current_price - last['supertrend']) / last['supertrend']) * 100
    
    # Bollinger Bands History
    bb_pctb = 0
    bb_squeeze = False
    bb_expanding = False
    bb_walking_upper = 0
    bb_walking_lower = 0
    
    if len(df) >= 10:
        # %B calculation
        bb_range = last['bb_upper'] - last['bb_lower']
        if bb_range > 0:
            bb_pctb = (current_price - last['bb_lower']) / bb_range
        
        # BB squeeze detection (width in lowest 20% of last 50 candles)
        if len(df) >= 50:
            bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
            current_width = bb_width.iloc[-1]
            width_percentile = bb_width.tail(50).rank(pct=True).iloc[-1]
            bb_squeeze = width_percentile <= 0.20
        
        # BB expanding (current width > previous 3 candles average)
        if len(df) >= 4:
            current_width = (last['bb_upper'] - last['bb_lower']) / last['bb_mid']
            prev_3_avg = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']).tail(4).iloc[:-1].mean()
            bb_expanding = current_width > prev_3_avg
        
        # Walking bands (last 50 candles)
        last_50_bb = df.tail(50)
        bb_walking_upper = (last_50_bb['close'] > last_50_bb['bb_upper'] * 0.998).sum()
        bb_walking_lower = (last_50_bb['close'] < last_50_bb['bb_lower'] * 1.002).sum()

    # TTM Squeeze History
    ttm_squeeze_on_now = False
    ttm_squeeze_bars = 0       # how many consecutive bars squeeze has been ON
    ttm_squeeze_fired = False  # squeeze just turned OFF (= fired/breakout)
    ttm_mom_val = 0.0
    ttm_mom_rising = False
    ttm_mom_direction = "neutral"

    if len(df) >= 20 and 'ttm_squeeze_on' in df.columns:
        ttm_squeeze_on_now = bool(df['ttm_squeeze_on'].iloc[-1])
        # Count consecutive squeeze bars
        for i in range(1, min(51, len(df))):
            if df['ttm_squeeze_on'].iloc[-i]:
                ttm_squeeze_bars += 1
            else:
                break
        # Squeeze just fired? (was ON, now OFF)
        if len(df) >= 2:
            was_on = bool(df['ttm_squeeze_on'].iloc[-2])
            is_off = not ttm_squeeze_on_now
            ttm_squeeze_fired = was_on and is_off

        ttm_mom_val = float(df['ttm_mom'].iloc[-1]) if not pd.isna(df['ttm_mom'].iloc[-1]) else 0
        if len(df) >= 3:
            prev_mom = float(df['ttm_mom'].iloc[-2]) if not pd.isna(df['ttm_mom'].iloc[-2]) else 0
            ttm_mom_rising = ttm_mom_val > prev_mom
        if ttm_mom_val > 0:
            ttm_mom_direction = "bullish"
        elif ttm_mom_val < 0:
            ttm_mom_direction = "bearish"

    # Ichimoku History
    tk_cross = "neutral"
    tk_cross_bars = None
    cloud_thickness_pct = 0
    future_cloud = "neutral"
    
    if not pd.isna(last['tenkan_sen']) and not pd.isna(last['kijun_sen']):
        if last['tenkan_sen'] > last['kijun_sen']:
            tk_cross = "bullish"
        elif last['tenkan_sen'] < last['kijun_sen']:
            tk_cross = "bearish"
        
        # Find TK cross
        if len(df) >= 30:
            tk_diff = df['tenkan_sen'] - df['kijun_sen']
            for i in range(1, min(30, len(tk_diff))):
                if not pd.isna(tk_diff.iloc[-i]) and not pd.isna(tk_diff.iloc[-i-1]):
                    if (tk_diff.iloc[-i] > 0) != (tk_diff.iloc[-i-1] > 0):
                        tk_cross_bars = i
                        break
    
    # Cloud thickness
    if cloud_top > 0 and cloud_bottom > 0:
        cloud_thickness_pct = ((cloud_top - cloud_bottom) / current_price) * 100
    
    # Future cloud (senkou A vs B without shift)
    if not pd.isna(last['tenkan_sen']) and not pd.isna(last['kijun_sen']):
        senkou_a_future = (last['tenkan_sen'] + last['kijun_sen']) / 2
        if len(df) >= 52:
            senkou_b_future = (df['high'].tail(52).max() + df['low'].tail(52).min()) / 2
            future_cloud = "bullish" if senkou_a_future > senkou_b_future else "bearish"
    
    # ADX Trend (50 candles)
    adx_trend = "stable"
    adx_avg_50 = 0
    adx_max_50 = 0
    di_cross_bars = None
    di_cross_dir = None
    
    if len(df) >= 50:
        adx_50 = df['adx'].tail(50)
        adx_avg_50 = adx_50.mean()
        adx_max_50 = adx_50.max()
        
        # ADX trend: compare first 10 avg vs last 10 avg
        adx_first_10 = adx_50.head(10).mean()
        adx_last_10 = adx_50.tail(10).mean()
        if adx_last_10 > adx_first_10 + 3:
            adx_trend = "rising"
        elif adx_last_10 < adx_first_10 - 3:
            adx_trend = "falling"
        
        # DI+/DI- cross detection (last 50 bars)
        di_plus_50 = df['di_plus'].tail(50)
        di_minus_50 = df['di_minus'].tail(50)
        for i in range(1, 50):
            prev_diff = di_plus_50.iloc[-i-1] - di_minus_50.iloc[-i-1]
            curr_diff = di_plus_50.iloc[-i] - di_minus_50.iloc[-i]
            if prev_diff <= 0 < curr_diff:
                di_cross_bars = i
                di_cross_dir = "bullish"
                break
            elif prev_diff >= 0 > curr_diff:
                di_cross_bars = i
                di_cross_dir = "bearish"
                break
    elif len(df) >= 5:
        adx_5 = df['adx'].tail(5)
        if adx_5.iloc[-1] > adx_5.iloc[-3] + 2:
            adx_trend = "rising"
        elif adx_5.iloc[-1] < adx_5.iloc[-3] - 2:
            adx_trend = "falling"
    
    # StochRSI Trend (15 candles — fast oscillator, 50 is noise)
    stoch_k_trend = "stable"
    stoch_overbought_bars_15 = 0
    stoch_oversold_bars_15 = 0
    stoch_kd_crosses_15 = 0
    
    if len(df) >= 15:
        stoch_k_15 = df['stoch_k'].tail(15)
        stoch_d_15 = df['stoch_d'].tail(15)
        stoch_overbought_bars_15 = (stoch_k_15 > 80).sum()
        stoch_oversold_bars_15 = (stoch_k_15 < 20).sum()
        
        # K/D crosses count
        for i in range(1, 15):
            prev_diff = stoch_k_15.iloc[i-1] - stoch_d_15.iloc[i-1]
            curr_diff = stoch_k_15.iloc[i] - stoch_d_15.iloc[i]
            if (prev_diff <= 0 < curr_diff) or (prev_diff >= 0 > curr_diff):
                stoch_kd_crosses_15 += 1
        
        # Trend: compare first 5 avg vs last 5 avg
        sk_first_5 = stoch_k_15.head(5).mean()
        sk_last_5 = stoch_k_15.tail(5).mean()
        if sk_last_5 > sk_first_5 + 10:
            stoch_k_trend = "rising"
        elif sk_last_5 < sk_first_5 - 10:
            stoch_k_trend = "falling"
    elif len(df) >= 5:
        stoch_k_5 = df['stoch_k'].tail(5)
        if stoch_k_5.iloc[-1] > stoch_k_5.iloc[-3] + 5:
            stoch_k_trend = "rising"
        elif stoch_k_5.iloc[-1] < stoch_k_5.iloc[-3] - 5:
            stoch_k_trend = "falling"
    
    # MFI Trend (20 candles — volume oscillator, 50 is overkill for OB/OS counts)
    mfi_trend_20 = "stable"
    mfi_avg_20 = 50
    mfi_overbought_bars_20 = 0
    mfi_oversold_bars_20 = 0
    
    if len(df) >= 20:
        mfi_20 = df['mfi'].tail(20)
        mfi_avg_20 = mfi_20.mean()
        mfi_overbought_bars_20 = (mfi_20 > 80).sum()
        mfi_oversold_bars_20 = (mfi_20 < 20).sum()
        
        mfi_first_7 = mfi_20.head(7).mean()
        mfi_last_7 = mfi_20.tail(7).mean()
        if mfi_last_7 > mfi_first_7 + 5:
            mfi_trend_20 = "rising"
        elif mfi_last_7 < mfi_first_7 - 5:
            mfi_trend_20 = "falling"
    elif len(df) >= 5:
        mfi_5 = df['mfi'].tail(5)
        if mfi_5.iloc[-1] > mfi_5.iloc[-3] + 3:
            mfi_trend_20 = "rising"
        elif mfi_5.iloc[-1] < mfi_5.iloc[-3] - 3:
            mfi_trend_20 = "falling"
    
    # CMF Trend (30 candles — 20-period window inside, 50 too far)
    cmf_trend_30 = "stable"
    cmf_avg_30 = 0
    cmf_positive_bars_30 = 0
    
    if len(df) >= 30:
        cmf_30 = df['cmf'].tail(30)
        cmf_avg_30 = cmf_30.mean()
        cmf_positive_bars_30 = (cmf_30 > 0).sum()
        
        cmf_first_10 = cmf_30.head(10).mean()
        cmf_last_10 = cmf_30.tail(10).mean()
        if cmf_last_10 > cmf_first_10 + 0.03:
            cmf_trend_30 = "rising"
        elif cmf_last_10 < cmf_first_10 - 0.03:
            cmf_trend_30 = "falling"
    elif len(df) >= 5:
        cmf_5 = df['cmf'].tail(5)
        if cmf_5.iloc[-1] > cmf_5.iloc[-3] + 0.02:
            cmf_trend_30 = "rising"
        elif cmf_5.iloc[-1] < cmf_5.iloc[-3] - 0.02:
            cmf_trend_30 = "falling"

    last_indic_row = {
        "close": last['close'],
        "change_recent": round(change_recent, 2),
        "change_24h": round(change_24h, 2),
        "recent_label": recent_label,
        "ema7": last['ema7'], "ema25": last['ema25'], "ema99": last['ema99'],
        "rsi14": last['rsi14'],
        "macd_line": last['macd_line'],
        "macd_signal": last['macd_signal'],
        "macd_hist": last['macd_hist'],
        "obv_status": obv_status,
        "supertrend": "🟢 BULLISH" if last['supertrend_dir'] == 1 else "🔴 BEARISH",
        "supertrend_dir_raw": int(last['supertrend_dir']),  # 1 for bullish, -1 for bearish
        "supertrend_price": last['supertrend'],
        "adx": last['adx'],
        "di_plus": last['di_plus'],
        "di_minus": last['di_minus'],
        "stoch_k": last['stoch_k'], "stoch_d": last['stoch_d'],
        "mfi": last['mfi'],
        "ichimoku_status": ichi_status,
        "ichi_cloud_top": cloud_top,
        "ichi_cloud_bottom": cloud_bottom,
        "funding_rate": df.get("funding_rate", "Unknown"),
        "bb_upper": last['bb_upper'],
        "bb_lower": last['bb_lower'],
        "bb_mid": last['bb_mid'],
        "cmf": last['cmf'],
        
        # HISTORICAL CONTEXT FIELDS (50 candles)
        # EMA History
        "ema7_slope_50": round(ema7_slope_50, 2),
        "ema25_slope_50": round(ema25_slope_50, 2),
        "ema99_slope_50": round(ema99_slope_50, 2),
        "ema7_25_cross_bars": ema7_25_cross_bars,
        "ema7_25_cross_dir": ema7_25_cross_dir,
        "price_above_ema7_count": price_above_ema7_count,
        "price_above_ema25_count": price_above_ema25_count,
        "price_above_ema99_count": price_above_ema99_count,
        
        # MACD History (50 candles)
        "macd_hist_trend_50": [round(h, 6) for h in macd_hist_trend_50[-10:]],  # last 10 values for prompt
        "macd_hist_direction": macd_hist_direction,
        "macd_bars_since_cross": macd_bars_since_cross,
        "macd_hist_accel": round(macd_hist_accel, 6),
        "macd_hist_min_50": round(macd_hist_min_50, 6),
        "macd_hist_max_50": round(macd_hist_max_50, 6),
        "macd_hist_avg_50": round(macd_hist_avg_50, 6),
        "macd_zero_crosses_50": macd_zero_crosses_50,
        
        # OBV History (50 candles)
        "obv_roc_50": round(obv_roc_50, 2),
        "obv_roc_10": round(obv_roc_10, 2),
        "obv_trend_50": obv_trend_50,
        "obv_spike": obv_spike,
        "obv_price_divergence": obv_price_divergence,
        
        # RSI History (50 candles)
        "rsi_trend_50": rsi_trend_50,
        "rsi_values_50": [round(r, 1) for r in rsi_values_50[-10:]],  # last 10 values for prompt
        "rsi_min_50": round(rsi_min_50, 1),
        "rsi_max_50": round(rsi_max_50, 1),
        "rsi_avg_50": round(rsi_avg_50, 1),
        "rsi_time_overbought": rsi_time_overbought,
        "rsi_time_oversold": rsi_time_oversold,
        "rsi_price_divergence": rsi_price_divergence,
        "rsi_pullback_peak": rsi_pullback_peak,
        "rsi_pullback_drop": rsi_pullback_drop,
        
        # SuperTrend History (50 candles)
        "st_bars_since_flip": st_bars_since_flip,
        "st_distance_pct": round(st_distance_pct, 2),
        "st_flips_50": st_flips_50,
        "st_bullish_bars_50": st_bullish_bars_50,
        
        # Bollinger Bands History (50 candles)
        "bb_pctb": round(bb_pctb, 3),
        "bb_squeeze": bb_squeeze,
        "bb_expanding": bb_expanding,
        "bb_walking_upper": bb_walking_upper,
        "bb_walking_lower": bb_walking_lower,

        # TTM Squeeze
        "ttm_squeeze_on": ttm_squeeze_on_now,
        "ttm_squeeze_bars": ttm_squeeze_bars,
        "ttm_squeeze_fired": ttm_squeeze_fired,
        "ttm_mom": round(ttm_mom_val, 8),
        "ttm_mom_rising": ttm_mom_rising,
        "ttm_mom_direction": ttm_mom_direction,
        
        # Ichimoku History
        "tk_cross": tk_cross,
        "tk_cross_bars": tk_cross_bars,
        "cloud_thickness_pct": round(cloud_thickness_pct, 2),
        "future_cloud": future_cloud,
        
        # ADX/DI History (50 candles)
        "adx_trend": adx_trend,
        "adx_avg_50": round(adx_avg_50, 1),
        "adx_max_50": round(adx_max_50, 1),
        "di_cross_bars": di_cross_bars,
        "di_cross_dir": di_cross_dir,
        
        # StochRSI History (15 candles)
        "stoch_k_trend": stoch_k_trend,
        "stoch_overbought_bars_15": stoch_overbought_bars_15,
        "stoch_oversold_bars_15": stoch_oversold_bars_15,
        "stoch_kd_crosses_15": stoch_kd_crosses_15,
        
        # MFI History (20 candles)
        "mfi_trend_20": mfi_trend_20,
        "mfi_avg_20": round(mfi_avg_20, 1),
        "mfi_overbought_bars_20": mfi_overbought_bars_20,
        "mfi_oversold_bars_20": mfi_oversold_bars_20,
        
        # CMF History (30 candles)
        "cmf_trend_30": cmf_trend_30,
        "cmf_avg_30": round(cmf_avg_30, 3),
        "cmf_positive_bars_30": cmf_positive_bars_30,

        # CCMI (Chande Composite Momentum Index)
        "ccmi": round(float(last['ccmi']), 2) if not pd.isna(last['ccmi']) else 0,
        "ccmi_signal": round(float(last['ccmi_signal']), 2) if not pd.isna(last['ccmi_signal']) else 0,

        # IMI (Intraday Momentum Index)
        "imi": round(float(last['imi']), 1) if not pd.isna(last['imi']) else 50,
        "imi_ma": round(float(last['imi_ma']), 1) if not pd.isna(last['imi_ma']) else 50,

        # RSI-Momentum Divergence
        "rsi_mom": round(float(last['rsi_mom']), 1) if not pd.isna(last['rsi_mom']) else 50,
        "rsi_mom_bull_div": rsi_mom_bull_div,
        "rsi_mom_bear_div": rsi_mom_bear_div,
        "rsi_mom_bull_div_detail": rsi_mom_bull_div_detail,
        "rsi_mom_bear_div_detail": rsi_mom_bear_div_detail,

        # ATR (Average True Range)
        "atr14_value": round(float(last['atr14']), 8) if not pd.isna(last['atr14']) else 0,
    }

    # --- ATR Historical Context ---
    atr_pct = 0
    atr_trend = "stable"
    atr_expanding = False
    atr_percentile_50 = 0.5

    if len(df) >= 50:
        atr_50 = df['atr14'].tail(50)
        atr_current = float(last['atr14']) if not pd.isna(last['atr14']) else 0
        if current_price > 0 and atr_current > 0:
            atr_pct = (atr_current / current_price) * 100
        # ATR trend: first 10 avg vs last 10 avg
        atr_first_10 = atr_50.head(10).mean()
        atr_last_10 = atr_50.tail(10).mean()
        if atr_first_10 > 0:
            atr_change = ((atr_last_10 - atr_first_10) / atr_first_10) * 100
            if atr_change > 10:
                atr_trend = "rising"
                atr_expanding = True
            elif atr_change < -10:
                atr_trend = "falling"
        # Percentile rank within 50 bars
        atr_percentile_50 = float(atr_50.rank(pct=True).iloc[-1])

    last_indic_row["atr_pct"] = round(atr_pct, 3)
    last_indic_row["atr_trend"] = atr_trend
    last_indic_row["atr_expanding"] = atr_expanding
    last_indic_row["atr_percentile_50"] = round(atr_percentile_50, 2)

    return last_indic_row, df


def format_tf_summary(indic: dict, tf_label: str) -> str:
    """Format one timeframe's indicators with detailed historical analysis for AI prompt."""
    price = indic['close']
    ema7 = indic['ema7']
    ema25 = indic['ema25']
    ema99 = indic['ema99']

    # ── EMA ANALYSIS WITH HISTORICAL CONTEXT ──
    ema7_slope = indic.get('ema7_slope_50', 0)
    ema25_slope = indic.get('ema25_slope_50', 0)
    ema99_slope = indic.get('ema99_slope_50', 0)
    cross_bars = indic.get('ema7_25_cross_bars')
    cross_dir = indic.get('ema7_25_cross_dir')
    price_above_7 = indic.get('price_above_ema7_count', 0)
    price_above_25 = indic.get('price_above_ema25_count', 0)
    price_above_99 = indic.get('price_above_ema99_count', 0)
    
    # Build EMA slope indicators
    slope7_ind = "↗" if ema7_slope > 1 else ("↘" if ema7_slope < -1 else "→")
    slope25_ind = "↗" if ema25_slope > 1 else ("↘" if ema25_slope < -1 else "→")
    slope99_ind = "↗" if ema99_slope > 1 else ("↘" if ema99_slope < -1 else "→")
    
    cross_text = ""
    if cross_bars and cross_dir:
        cross_text = f" | EMA7×25 {cross_dir} cross {cross_bars} bars ago"
    
    if ema7 > ema25 > ema99:
        ema_signal = (f"🟢 BULLISH (aligned, above EMAs){' but EMA7 slope decelerating' if ema7_slope < 0 else ''}")
    elif ema7 < ema25 < ema99:
        ema_signal = (f"🔴 BEARISH (aligned, below EMAs){' but EMA7 slope accelerating up' if ema7_slope > 1 else ''}")
    else:
        ema_signal = f"⚪ MIXED (EMA7={slope7_ind}, EMA25={slope25_ind}, EMA99={slope99_ind})"
    
    idx = 1
    ema_analysis = (f"{idx}. EMA: 7={ema7:.6f}({slope7_ind}{ema7_slope:+.1f}%/50bars) {'>' if ema7 > ema25 else '<'} "
                   f"25={ema25:.6f}({slope25_ind}{ema25_slope:+.1f}%) {'>' if ema25 > ema99 else '<'} "
                   f"99={ema99:.6f}({slope99_ind}{'flat' if abs(ema99_slope) < 0.5 else f'{ema99_slope:+.1f}%'})\n"
                   f"   Price>EMA7: {price_above_7}/50 candles | Price>EMA25: {price_above_25}/50 | Price>EMA99: {price_above_99}/50{cross_text}\n"
                   f"   → {ema_signal}")

    # ── MACD ANALYSIS WITH HISTOGRAM DYNAMICS ──
    macd_line = indic['macd_line']
    macd_signal_val = indic['macd_signal']
    macd_hist = indic['macd_hist']
    hist_direction = indic.get('macd_hist_direction', 'unknown')
    hist_trend = indic.get('macd_hist_trend_50', [])
    bars_since_cross = indic.get('macd_bars_since_cross')
    hist_accel = indic.get('macd_hist_accel', 0)
    hist_min_50 = indic.get('macd_hist_min_50', 0)
    hist_max_50 = indic.get('macd_hist_max_50', 0)
    hist_avg_50 = indic.get('macd_hist_avg_50', 0)
    zero_crosses = indic.get('macd_zero_crosses_50', 0)
    
    hist_peak = max(hist_trend) if hist_trend else macd_hist
    cross_text = f"Cross {bars_since_cross} bars ago" if bars_since_cross else "No recent cross"
    
    if macd_line > macd_signal_val:
        if hist_direction in ["growing", "turned_positive"]:
            macd_signal = f"🟢 BULLISH (momentum growing)"
            macd_vote_weight = 1.0
        elif hist_direction == "fading_bullish":
            macd_signal = f"🟡 WEAK BULLISH (hist positive but FALLING — momentum exhausting)"
            macd_vote_weight = 0.5
        else:
            macd_signal = f"🟡 WEAK BULLISH (momentum fading)"
            macd_vote_weight = 0.5
    else:
        if hist_direction in ["shrinking", "turned_negative"]:
            macd_signal = f"🔴 BEARISH (momentum growing)"
            macd_vote_weight = 1.0
        elif hist_direction == "fading_bearish":
            macd_signal = f"🟡 WEAK BEARISH (hist negative but RISING toward 0 — bearish exhausting)"
            macd_vote_weight = 0.5
        else:
            macd_signal = f"🟡 WEAK BEARISH (momentum fading)"
            macd_vote_weight = 0.5
    
    idx += 1
    macd_analysis = (f"{idx}. MACD: DIF={macd_line:.6f} {'>' if macd_line > macd_signal_val else '<'} "
                    f"DEA={macd_signal_val:.6f} | Hist={macd_hist:.6f} {hist_direction.upper()}\n"
                    f"   {cross_text} | Hist range 50bars: [{hist_min_50:.6f}..{hist_max_50:.6f}] avg={hist_avg_50:.6f} | Zero crosses: {zero_crosses}\n"
                    f"   Accel={hist_accel:.6f} | {'CHOPPY' if zero_crosses > 4 else 'TRENDING'}\n"
                    f"   → {macd_signal}")

    # ── OBV ANALYSIS WITH SPIKE DETECTION ──
    obv_status = indic['obv_status']
    obv_roc_50 = indic.get('obv_roc_50', 0)
    obv_roc_10 = indic.get('obv_roc_10', 0)
    obv_trend_50 = indic.get('obv_trend_50', 'stable')
    obv_spike = indic.get('obv_spike', False)
    obv_divergence = indic.get('obv_price_divergence', 'none')
    
    spike_text = f" ⚠️ SPIKE detected (2σ over 50 bars)" if obv_spike else ""
    divergence_text = f"Price-OBV 50bar: {'NO DIVERGENCE (confirmed)' if obv_divergence == 'none' else f'{obv_divergence.upper()} DIVERGENCE ⚠️'}"
    
    if "Accumulation" in obv_status:
        obv_signal = f"🟢 BULLISH{' but spike=caution' if obv_spike else ''}"
        obv_vote_weight = 0.5 if obv_spike else 1.0
    else:
        obv_signal = f"🔴 BEARISH{' but spike=caution' if obv_spike else ''}"
        obv_vote_weight = 0.5 if obv_spike else 1.0
    
    idx += 1
    obv_analysis = (f"{idx}. OBV: {obv_status.split('(')[1].replace(')', '')} (>SMA20) | ROC(10)={obv_roc_10:+.1f}% | ROC(50)={obv_roc_50:+.1f}% | Trend 50bar: {obv_trend_50}{spike_text}\n"
                   f"   {divergence_text}\n"
                   f"   → {obv_signal}")

    # ── RSI ANALYSIS WITH PENALTY SYSTEM ──
    rsi = indic['rsi14']
    rsi_trend = indic.get('rsi_trend_50', 'stable')
    rsi_values = indic.get('rsi_values_50', [])
    rsi_min = indic.get('rsi_min_50', rsi)
    rsi_max = indic.get('rsi_max_50', rsi)
    rsi_avg = indic.get('rsi_avg_50', rsi)
    rsi_time_ob = indic.get('rsi_time_overbought', 0)
    rsi_time_os = indic.get('rsi_time_oversold', 0)
    rsi_div = indic.get('rsi_price_divergence', 'none')
    
    values_str = "→".join([f"{v:.1f}" for v in rsi_values[-5:]]) if rsi_values else f"{rsi:.1f}"
    
    rsi_penalty = 0  # NO penalty — AI decides based on ADX context
    if rsi > 80:
        rsi_signal = f"⚠️ OVERBOUGHT ({rsi:.0f}) — check ADX: if strong trend this is normal"
    elif rsi > 70:
        rsi_signal = f"⚠️ OVERBOUGHT ({rsi:.0f}) — warn only, no auto-penalty"
    elif rsi < 20:
        rsi_signal = f"⚠️ OVERSOLD ({rsi:.0f}) — check ADX: if strong downtrend this is normal"
    elif rsi < 30:
        rsi_signal = f"⚠️ OVERSOLD ({rsi:.0f}) — warn only, no auto-penalty"
    elif rsi > 55:
        rsi_signal = f"🟢 BULLISH"
    elif rsi < 45:
        rsi_signal = f"🔴 BEARISH"
    else:
        rsi_signal = f"⚪ NEUTRAL"
    
    penalty_text = ""  # No penalty applied in scorecard
    div_text = f" | RSI-Price DIVERGENCE: {rsi_div.upper()} ⚠️" if rsi_div != "none" else ""
    
    # Pullback warning text
    pullback_text = ""
    rsi_pullback_peak = indic.get("rsi_pullback_peak", 0)
    rsi_pullback_drop = indic.get("rsi_pullback_drop", 0)
    if rsi_pullback_peak > 0 and rsi > 70:
        pullback_text = f"\n   ⚠️ PULLBACK HISTORY: last pullback started from RSI {rsi_pullback_peak} (dropped {rsi_pullback_drop} pts)"
        if rsi >= rsi_pullback_peak - 2:
            pullback_text += f" — CURRENT RSI {rsi:.0f} IS NEAR THAT LEVEL!"
    
    idx += 1
    rsi_analysis = (f"{idx}. RSI: {rsi:.1f} {'OVERBOUGHT' if rsi > 70 else ('OVERSOLD' if rsi < 30 else 'NORMAL')} | "
                   f"Trend 50bar: {rsi_trend} ({values_str})\n"
                   f"   50bar range: [{rsi_min:.1f}..{rsi_max:.1f}] avg={rsi_avg:.1f} | OB bars: {rsi_time_ob}/50 | OS bars: {rsi_time_os}/50{div_text}{pullback_text}\n"
                   f"   → {rsi_signal}")

    # ── SUPERTREND ANALYSIS WITH FLIP TIMING ──
    st_status = indic['supertrend']
    st_price = indic['supertrend_price']
    st_flip_bars = indic.get('st_bars_since_flip', 0)
    st_distance = indic.get('st_distance_pct', 0)
    st_flips = indic.get('st_flips_50', 0)
    st_bull_bars = indic.get('st_bullish_bars_50', 0)
    
    # Adjust vote weight based on flip timing
    if st_flip_bars < 3:
        st_vote_weight = 1.5  # Fresh signal
        flip_note = "(fresh signal)"
    elif st_flip_bars > 30:
        st_vote_weight = 0.5   # Stale signal
        flip_note = "(stale signal)"
    else:
        st_vote_weight = 1.0   # Normal signal
        flip_note = "(established trend, not fresh signal)"
    
    # SuperTrend analysis text (idx assigned later when added to list)
    st_analysis_text = (f"SuperTrend: {st_status} @ {st_price:.6f} | Flipped {st_flip_bars} bars ago | "
                       f"Price {st_distance:+.1f}% from ST line\n"
                       f"   50bar: {st_flips} flips | Bullish {st_bull_bars}/50 bars | {'CHOPPY' if st_flips > 4 else 'TRENDING'}\n"
                       f"   → {st_status.split()[1]} {flip_note}")

    # ── BOLLINGER BANDS WITH SQUEEZE/EXPANSION ──
    bb_upper = indic['bb_upper']
    bb_mid = indic['bb_mid']
    bb_lower = indic['bb_lower']
    bb_pctb = indic.get('bb_pctb', 0)
    bb_squeeze = indic.get('bb_squeeze', False)
    bb_expanding = indic.get('bb_expanding', False)
    bb_walk_upper = indic.get('bb_walking_upper', 0)
    bb_walk_lower = indic.get('bb_walking_lower', 0)
    
    bb_width_pct = ((bb_upper - bb_lower) / bb_mid) * 100 if bb_mid > 0 else 0
    
    if price >= bb_upper * 0.998:
        if bb_walk_upper > 5:
            bb_signal = f"🟢 BULLISH (walking upper band {bb_walk_upper}/50 candles — strong trend, not resistance)"
        else:
            bb_signal = f"⚠️ INFO: at upper band — no vote, check walking band context"
    elif price <= bb_lower * 1.002:
        bb_signal = f"⚠️ BULLISH (at lower band, potential support)"
    elif bb_pctb > 0.5:
        bb_signal = f"🟢 BULLISH (above mid, expanding, trend walk)" if bb_expanding else f"🟢 BULLISH (above mid)"
    else:
        bb_signal = f"🔴 BEARISH (below mid)"
    
    # TTM Squeeze analysis
    ttm_squeeze_on = indic.get('ttm_squeeze_on', False)
    ttm_squeeze_bars = indic.get('ttm_squeeze_bars', 0)
    ttm_squeeze_fired = indic.get('ttm_squeeze_fired', False)

    if ttm_squeeze_fired:
        ttm_text = "🔥 FIRED — breakout initiated!"
    elif ttm_squeeze_on:
        ttm_text = f"⚠️ SQUEEZE ON ({ttm_squeeze_bars} bars) — compression building, breakout brewing"
    else:
        ttm_text = "No squeeze"

    idx += 1
    bb_analysis = (f"{idx}. BB: Upper={bb_upper:.6f} Mid={bb_mid:.6f} Lower={bb_lower:.6f} | %B={bb_pctb:.3f} | Width={bb_width_pct:.1f}%\n"
                  f"   Squeeze: {'YES' if bb_squeeze else 'NO'} | Expanding: {'YES' if bb_expanding else 'NO'} | "
                  f"Walking upper band: {bb_walk_upper}/50 candles\n"
                  f"   TTM Squeeze: {ttm_text}\n"
                  f"   → {bb_signal}")

    # Build indicator lines based on timeframe
    tf_upper = tf_label.upper()
    is_higher_tf = tf_upper in ("1D", "4H")
    is_1h_or_higher = tf_upper in ("1D", "4H", "1H")

    raw_lines = [ema_analysis, macd_analysis, obv_analysis, rsi_analysis, bb_analysis]
    
    # SuperTrend — all timeframes (early reversal signal on 15m)
    idx += 1
    st_analysis = f"{idx}. {st_analysis_text}"
    raw_lines.append(st_analysis)
    
    # Ichimoku votes on 4H+ only
    if is_higher_tf:
        ichi_status = indic['ichimoku_status']
        tk_cross = indic.get('tk_cross', 'neutral')
        tk_cross_bars = indic.get('tk_cross_bars')
        cloud_thickness = indic.get('cloud_thickness_pct', 0)
        future_cloud = indic.get('future_cloud', 'neutral')
        
        tk_text = f"TK: {tk_cross} cross {tk_cross_bars} bars ago" if tk_cross_bars else f"TK: {tk_cross}"
        
        if "ABOVE CLOUD" in ichi_status:
            ichi_signal = f"🟢 BULLISH (above cloud, TK {tk_cross}, thick {'support' if cloud_thickness > 1 else 'cloud'})"
        elif "BELOW CLOUD" in ichi_status:
            ichi_signal = f"🔴 BEARISH (below cloud, TK {tk_cross}, thick {'resistance' if cloud_thickness > 1 else 'cloud'})"
        else:
            ichi_signal = f"⚪ NEUTRAL (inside cloud, choppy)"
        
        idx += 1
        ichi_analysis = (f"{idx}. Ichimoku: {ichi_status.split('(')[0].strip()} | {tk_text} | Cloud thickness: {cloud_thickness:.1f}%\n"
                        f"   Future cloud: {future_cloud} (Senkou A {'>' if future_cloud == 'bullish' else '<'} B)\n"
                        f"   → {ichi_signal}")
        raw_lines.append(ichi_analysis)
    
    # ADX with DI analysis (NOW VOTES)
    adx = indic['adx']
    di_plus = indic.get('di_plus', 0)
    di_minus = indic.get('di_minus', 0)
    adx_trend = indic.get('adx_trend', 'stable')
    adx_avg = indic.get('adx_avg_50', 0)
    adx_max = indic.get('adx_max_50', 0)
    di_cross_b = indic.get('di_cross_bars')
    di_cross_d = indic.get('di_cross_dir')
    
    if adx > 40:
        trend_strength = "STRONG TREND"
    elif adx > 25:
        trend_strength = "MODERATE TREND"
    else:
        trend_strength = "WEAK/NO TREND"
    
    di_cross_text = f" | DI cross: {di_cross_d} {di_cross_b} bars ago" if di_cross_b else ""
    
    if di_plus > di_minus:
        adx_signal = f"🟢 BULLISH (DI+ dominant, trend {'strengthening' if adx_trend == 'rising' else 'stable'})"
    else:
        adx_signal = f"🔴 BEARISH (DI- dominant, trend {'strengthening' if adx_trend == 'rising' else 'stable'})"
    
    idx += 1
    adx_analysis = (f"{idx}. ADX: {adx:.0f} {trend_strength} | DI+: {di_plus:.1f} {'>' if di_plus > di_minus else '<'} "
                   f"DI-: {di_minus:.1f} | ADX {adx_trend}\n"
                   f"   50bar: avg={adx_avg:.0f} max={adx_max:.0f}{di_cross_text}\n"
                   f"   → {adx_signal}")
    raw_lines.append(adx_analysis)
    
    # StochRSI (NOW VOTES)
    stoch_k = indic['stoch_k']
    stoch_d = indic['stoch_d']
    stoch_k_trend = indic.get('stoch_k_trend', 'stable')
    stoch_ob_bars = indic.get('stoch_overbought_bars_15', 0)
    stoch_os_bars = indic.get('stoch_oversold_bars_15', 0)
    stoch_crosses = indic.get('stoch_kd_crosses_15', 0)
    
    if stoch_k > 80:
        stoch_signal = f"⚠️ INFO: OVERBOUGHT ({stoch_k:.0f}) — no vote, AI decides with ADX context"
    elif stoch_k < 20:
        stoch_signal = f"⚠️ INFO: OVERSOLD ({stoch_k:.0f}) — no vote, AI decides with ADX context"
    elif stoch_k > stoch_d:
        stoch_signal = f"🟢 BULLISH (K>D cross)"
    else:
        stoch_signal = f"🔴 BEARISH (K<D cross)"
    
    idx += 1
    stoch_analysis = (f"{idx}. StochRSI: K={stoch_k:.0f} D={stoch_d:.0f} "
                     f"{'OVERBOUGHT' if stoch_k > 80 else ('OVERSOLD' if stoch_k < 20 else 'NORMAL')} | "
                     f"K trend 15bar: {stoch_k_trend}\n"
                     f"   15bar: OB bars={stoch_ob_bars}/15 | OS bars={stoch_os_bars}/15 | K/D crosses: {stoch_crosses} | {'CHOPPY' if stoch_crosses > 4 else 'TRENDING'}\n"
                     f"   → {stoch_signal}")
    raw_lines.append(stoch_analysis)
    
    # MFI (NOW VOTES)
    mfi = indic['mfi']
    mfi_trend = indic.get('mfi_trend_20', 'stable')
    mfi_avg = indic.get('mfi_avg_20', 50)
    mfi_ob = indic.get('mfi_overbought_bars_20', 0)
    mfi_os = indic.get('mfi_oversold_bars_20', 0)
    
    if mfi > 80:
        mfi_signal = f"⚠️ INFO: OVERBOUGHT ({mfi:.0f}) — no vote, AI decides with ADX context"
    elif mfi < 20:
        mfi_signal = f"⚠️ INFO: OVERSOLD ({mfi:.0f}) — no vote, AI decides with ADX context"
    elif mfi > 50:
        mfi_signal = f"🟢 BULLISH"
    else:
        mfi_signal = f"🔴 BEARISH"
    
    idx += 1
    mfi_analysis = (f"{idx}. MFI: {mfi:.0f} {'BULLISH' if mfi > 50 else 'BEARISH'} | Trend 20bar: {mfi_trend} | Avg={mfi_avg:.0f}\n"
                   f"   20bar: OB bars={mfi_ob}/20 | OS bars={mfi_os}/20\n"
                   f"   → {mfi_signal}")
    raw_lines.append(mfi_analysis)
    
    # CMF (NOW VOTES)
    cmf = indic['cmf']
    cmf_trend = indic.get('cmf_trend_30', 'stable')
    cmf_avg = indic.get('cmf_avg_30', 0)
    cmf_pos_bars = indic.get('cmf_positive_bars_30', 0)
    
    if cmf > 0.1:
        cmf_signal = f"🟢 BULLISH (strong buying)"
    elif cmf > 0:
        cmf_signal = f"🟢 BULLISH (mild buying)"
    elif cmf < -0.1:
        cmf_signal = f"🔴 BEARISH (strong selling)"
    elif cmf < 0:
        cmf_signal = f"🔴 BEARISH (mild selling)"
    else:
        cmf_signal = f"⚪ NEUTRAL"
    
    idx += 1
    cmf_analysis = (f"{idx}. CMF: {cmf:.3f} {'STRONG' if abs(cmf) > 0.1 else ''} "
                   f"{'BUYING' if cmf > 0 else 'SELLING'} | Trend 30bar: {cmf_trend} | Avg={cmf_avg:.3f}\n"
                   f"   Positive bars: {cmf_pos_bars}/30 ({'buying dominant' if cmf_pos_bars > 18 else ('selling dominant' if cmf_pos_bars < 12 else 'mixed')})\n"
                   f"   → {cmf_signal}")
    raw_lines.append(cmf_analysis)

    # ── CCMI (Chande Composite Momentum Index) ──
    ccmi_val = indic.get('ccmi', 0)
    ccmi_sig = indic.get('ccmi_signal', 0)
    ccmi_crossover = ccmi_val > ccmi_sig

    if ccmi_val > 70:
        ccmi_signal_txt = f"⚠️ OVERBOUGHT ({ccmi_val:.0f}) — momentum extreme"
    elif ccmi_val < -70:
        ccmi_signal_txt = f"⚠️ OVERSOLD ({ccmi_val:.0f}) — momentum extreme"
    elif ccmi_val > 30 and ccmi_crossover:
        ccmi_signal_txt = f"🟢 BULLISH (above signal, momentum rising)"
    elif ccmi_val > 0:
        ccmi_signal_txt = f"🟢 BULLISH ({'accelerating' if ccmi_crossover else 'decelerating'})"
    elif ccmi_val < -30 and not ccmi_crossover:
        ccmi_signal_txt = f"🔴 BEARISH (below signal, momentum falling)"
    elif ccmi_val < 0:
        ccmi_signal_txt = f"🔴 BEARISH ({'recovering' if ccmi_crossover else 'accelerating down'})"
    else:
        ccmi_signal_txt = f"⚪ NEUTRAL"

    idx += 1
    ccmi_analysis = (f"{idx}. CCMI: {ccmi_val:.1f} | Signal: {ccmi_sig:.1f} | "
                    f"Cross: {'ABOVE ↑' if ccmi_crossover else 'BELOW ↓'}\n"
                    f"   Multi-period momentum (5/10/20) volatility-weighted\n"
                    f"   → {ccmi_signal_txt}")
    raw_lines.append(ccmi_analysis)

    # ── IMI (Intraday Momentum Index) ──
    imi_val = indic.get('imi', 50)
    imi_ma_val = indic.get('imi_ma', 50)

    if imi_val > 70:
        imi_signal_txt = f"🟢 STRONG BUYING pressure ({imi_val:.0f}) — buyers dominate candle bodies"
    elif imi_val > 50:
        imi_signal_txt = f"🟢 BULLISH pressure ({imi_val:.0f})"
    elif imi_val < 20:
        imi_signal_txt = f"🔴 STRONG SELLING pressure ({imi_val:.0f}) — sellers dominate candle bodies"
    elif imi_val < 50:
        imi_signal_txt = f"🔴 BEARISH pressure ({imi_val:.0f})"
    else:
        imi_signal_txt = f"⚪ NEUTRAL (balanced)"

    idx += 1
    imi_analysis = (f"{idx}. IMI: {imi_val:.1f} | MA(6): {imi_ma_val:.1f} | "
                   f"{'OB zone' if imi_val > 70 else ('OS zone' if imi_val < 20 else 'Normal')}\n"
                   f"   Candle body pressure (close vs open) — what RSI doesn't see\n"
                   f"   → {imi_signal_txt}")
    raw_lines.append(imi_analysis)

    # ── RSI-Momentum Divergence ──
    rsi_mom_val = indic.get('rsi_mom', 50)
    rsi_mom_bull = indic.get('rsi_mom_bull_div', False)
    rsi_mom_bear = indic.get('rsi_mom_bear_div', False)
    rsi_mom_bull_detail = indic.get('rsi_mom_bull_div_detail', '')
    rsi_mom_bear_detail = indic.get('rsi_mom_bear_div_detail', '')

    div_lines = []
    if rsi_mom_bull:
        div_lines.append(f"   🟢 BULLISH DIVERGENCE: {rsi_mom_bull_detail}")
    if rsi_mom_bear:
        div_lines.append(f"   🔴 BEARISH DIVERGENCE: {rsi_mom_bear_detail}")
    if not div_lines:
        div_lines.append(f"   No divergence detected")

    if rsi_mom_bear:
        rsi_mom_signal_txt = f"⚠️ BEARISH REVERSAL WARNING — price momentum diverging"
    elif rsi_mom_bull:
        rsi_mom_signal_txt = f"⚠️ BULLISH REVERSAL SIGNAL — price momentum diverging"
    elif rsi_mom_val > 60:
        rsi_mom_signal_txt = f"🟢 BULLISH (momentum accelerating)"
    elif rsi_mom_val < 40:
        rsi_mom_signal_txt = f"🔴 BEARISH (momentum decelerating)"
    else:
        rsi_mom_signal_txt = f"⚪ NEUTRAL"

    idx += 1
    rsi_mom_analysis = (f"{idx}. RSI-Mom: {rsi_mom_val:.1f} (RSI of momentum(10) — 2nd derivative of price)\n"
                       + "\n".join(div_lines) + "\n"
                       f"   → {rsi_mom_signal_txt}")
    raw_lines.append(rsi_mom_analysis)

    # ── ATR VOLATILITY ANALYSIS ──
    atr_val = indic.get('atr14_value', 0)
    atr_pct_val = indic.get('atr_pct', 0)
    atr_trend_val = indic.get('atr_trend', 'stable')
    atr_expanding_val = indic.get('atr_expanding', False)
    atr_percentile = indic.get('atr_percentile_50', 0.5)

    if atr_expanding_val and adx_trend == "rising":
        atr_signal = f"🟢 BREAKOUT CONDITIONS (ATR rising + ADX rising = volatility expansion with trend)"
    elif atr_expanding_val:
        atr_signal = f"⚠️ VOLATILITY EXPANDING (ATR rising but ADX not — choppy breakout or whipsaw?)"
    elif atr_trend_val == "falling" and adx < 20:
        atr_signal = f"🔴 FLAT MARKET (ATR falling + ADX weak = no volatility, no trend)"
    elif atr_trend_val == "falling":
        atr_signal = f"⚪ VOLATILITY CONTRACTING (potential squeeze building)"
    else:
        atr_signal = f"⚪ STABLE VOLATILITY"

    idx += 1
    atr_analysis = (f"{idx}. ATR: {atr_val:.8f} ({atr_pct_val:.2f}% of price) | Trend: {atr_trend_val} | "
                   f"Percentile: {atr_percentile:.0%} of 50bar range\n"
                   f"   {'EXPANDING ↑' if atr_expanding_val else ('CONTRACTING ↓' if atr_trend_val == 'falling' else 'STABLE →')}"
                   f" | ADX {adx_trend}: {'BREAKOUT COMBO' if atr_expanding_val and adx_trend == 'rising' else 'normal'}\n"
                   f"   → {atr_signal}")
    raw_lines.append(atr_analysis)

    # ── INTER-INDICATOR CONFLUENCES ──
    # Detect powerful multi-indicator alignments that are stronger than sum of parts
    confluences = []
    confluence_bull_bonus = 0.0
    confluence_bear_bonus = 0.0

    # --- BULLISH CONFLUENCES ---
    # 1. EMA golden cross + MACD turned positive + OBV rising = strong bull
    ema_bull = (ema7 > ema25 > ema99)
    macd_bull = ("BULLISH" in macd_signal and hist_direction in ["growing", "turned_positive"])
    obv_bull = ("Accumulation" in obv_status)
    if ema_bull and macd_bull and obv_bull:
        confluences.append("🟢 BULL CONFLUENCE: EMA aligned + MACD growing + OBV accumulation")
        confluence_bull_bonus += 0.5

    # 2. SuperTrend bullish + ADX rising + ATR expanding = breakout confirmed
    st_bull = ("BULLISH" in st_status)
    adx_rising = (adx_trend == "rising" and adx > 20)
    if st_bull and adx_rising and atr_expanding_val:
        confluences.append("🟢 BREAKOUT CONFLUENCE: SuperTrend bull + ADX rising + ATR expanding")
        confluence_bull_bonus += 0.5

    # 3. Price above all EMAs + OBV rising + CMF positive = accumulation trend
    price_above_emas = (price > ema7 and price > ema25 and price > ema99)
    cmf_positive = (cmf > 0.05)
    if price_above_emas and obv_bull and cmf_positive:
        confluences.append("🟢 ACCUMULATION CONFLUENCE: Price>all EMAs + OBV rising + CMF buying")
        confluence_bull_bonus += 0.3

    # --- BEARISH CONFLUENCES ---
    # 4. RSI overbought + OBV bearish divergence + MACD fading = reversal warning
    rsi_ob = (rsi > 70)
    obv_bear_div = (obv_divergence == "bearish")
    macd_fading = (hist_direction == "fading_bullish")
    if rsi_ob and obv_bear_div and macd_fading:
        confluences.append("🔴 REVERSAL CONFLUENCE: RSI overbought + OBV bearish divergence + MACD fading")
        confluence_bear_bonus += 0.5

    # 5. EMA death cross + SuperTrend bearish + CMF negative = strong bear
    ema_bear = (ema7 < ema25 < ema99)
    st_bear = ("BEARISH" in st_status)
    cmf_negative = (cmf < -0.05)
    if ema_bear and st_bear and cmf_negative:
        confluences.append("🔴 BEAR CONFLUENCE: EMA aligned down + SuperTrend bear + CMF selling")
        confluence_bear_bonus += 0.5

    # 6. RSI-Mom bearish divergence + MACD turning negative + ADX falling = trend exhaustion
    rsi_mom_bear_div = indic.get('rsi_mom_bear_div', False)
    macd_turning_neg = (hist_direction == "turned_negative")
    adx_falling = (adx_trend == "falling")
    if rsi_mom_bear_div and (macd_turning_neg or macd_fading) and adx_falling:
        confluences.append("🔴 EXHAUSTION CONFLUENCE: RSI-Mom divergence + MACD weakening + ADX falling")
        confluence_bear_bonus += 0.3

    if confluences:
        confluence_text = "\n".join(f"   {c}" for c in confluences)
        idx += 1
        confluence_analysis = (f"{idx}. CONFLUENCES DETECTED ({len(confluences)}):\n{confluence_text}\n"
                              f"   → Bull bonus: +{confluence_bull_bonus:.1f} | Bear bonus: +{confluence_bear_bonus:.1f}")
        raw_lines.append(confluence_analysis)

    # ── GROUPED VOTING SCORECARD WITH DYNAMIC ADX WEIGHTS ──
    # Each group votes as ONE unit (majority within group decides).
    # Dynamic weights: ADX>25 → trend/momentum ×1.5, oscillators ×0.5
    #                  ADX<20 → oscillators ×1.5, trend/momentum ×0.5
    # OB/OS "INFO" signals = neutral (don't vote, AI decides with context)

    def _classify_vote(sig):
        """Classify signal as bull/bear/neutral. INFO signals = neutral."""
        if "INFO" in sig:
            return "neutral"
        if "🟢" in sig or "BULLISH" in sig:
            return "bull"
        if "🔴" in sig or "BEARISH" in sig:
            return "bear"
        return "neutral"

    def _group_majority(members):
        """Group vote = majority direction. Tie = neutral."""
        bulls = sum(1 for _, v in members if v == "bull")
        bears = sum(1 for _, v in members if v == "bear")
        if bulls > bears:
            return "bull", bulls, bears
        elif bears > bulls:
            return "bear", bulls, bears
        return "neutral", bulls, bears

    # --- Classify all individual indicators ---
    ema_v = _classify_vote(ema_signal)
    st_v = "bull" if "BULLISH" in st_status else ("bear" if "BEARISH" in st_status else "neutral")
    ichi_v = _classify_vote(ichi_signal) if is_higher_tf else None

    macd_v = _classify_vote(macd_signal)
    ccmi_v = _classify_vote(ccmi_signal_txt)
    rsi_mom_v = _classify_vote(rsi_mom_signal_txt)

    rsi_v = _classify_vote(rsi_signal)
    stoch_v = _classify_vote(stoch_signal)
    mfi_v = _classify_vote(mfi_signal)
    imi_v = _classify_vote(imi_signal_txt)

    obv_v = _classify_vote(obv_signal)
    cmf_v = _classify_vote(cmf_signal)

    bb_v = _classify_vote(bb_signal)
    adx_v = _classify_vote(adx_signal)

    # --- Build groups ---
    trend_members = [("EMA", ema_v), ("SuperTrend", st_v)]
    if ichi_v is not None:
        trend_members.append(("Ichimoku", ichi_v))

    momentum_members = [("MACD", macd_v), ("CCMI", ccmi_v), ("RSI-Mom", rsi_mom_v)]
    oscillator_members = [("RSI", rsi_v), ("StochRSI", stoch_v), ("MFI", mfi_v), ("IMI", imi_v)]
    volume_members = [("OBV", obv_v), ("CMF", cmf_v)]
    volatility_members = [("BB", bb_v)]
    direction_members = [("ADX/DI", adx_v)]

    groups = [
        ("Trend",      trend_members,      "trend"),
        ("Momentum",   momentum_members,   "trend"),       # trend-like behaviour
        ("Oscillators", oscillator_members, "oscillator"),
        ("Volume",     volume_members,     "volume"),
        ("Volatility", volatility_members, "volatility"),
        ("Direction",  direction_members,  "direction"),
    ]

    # --- Dynamic weight multipliers based on ADX regime ---
    if adx > 25:
        w_mult = {"trend": 1.5, "oscillator": 0.5, "volume": 1.0, "volatility": 1.0, "direction": 1.0}
        regime = "TRENDING"
    elif adx < 20:
        w_mult = {"trend": 0.5, "oscillator": 1.5, "volume": 1.0, "volatility": 1.0, "direction": 1.0}
        regime = "RANGING"
    else:
        w_mult = {"trend": 1.0, "oscillator": 1.0, "volume": 1.0, "volatility": 1.0, "direction": 1.0}
        regime = "TRANSITION"

    # --- Compute grouped scorecard ---
    bullish_weight = 0.0
    bearish_weight = 0.0
    group_details = []
    bull_groups = 0
    bear_groups = 0
    neutral_groups = 0

    for grp_name, members, category in groups:
        direction, bulls, bears = _group_majority(members)
        w = w_mult[category]

        member_icons = "/".join(
            f"{n}{'🟢' if v == 'bull' else ('🔴' if v == 'bear' else '⚪')}" for n, v in members
        )

        if direction == "bull":
            bullish_weight += w
            bull_groups += 1
            group_details.append(f"  {grp_name}=🟢(×{w:.1f}) [{member_icons}]")
        elif direction == "bear":
            bearish_weight += w
            bear_groups += 1
            group_details.append(f"  {grp_name}=🔴(×{w:.1f}) [{member_icons}]")
        else:
            neutral_groups += 1
            group_details.append(f"  {grp_name}=⚪ [{member_icons}]")

    # --- Open Interest bonus (if available) ---
    oi_impact = ""
    positioning = indic.get("positioning", {})
    if positioning and "oi_change_pct" in positioning:
        oi_change = positioning.get("oi_change_pct", 0)
        if oi_change > 5:
            bullish_weight += 0.5
            oi_impact = " +OI📈"
        elif oi_change < -5:
            bearish_weight += 0.5
            oi_impact = " +OI📉"

    # --- Confluence bonuses ---
    bullish_weight += confluence_bull_bonus
    bearish_weight += confluence_bear_bonus

    # --- PENALTY SYSTEM: reality checks that cap unrealistic confidence ---
    penalties = []
    penalty_pct = 0  # total percentage points to subtract from bull_pct (or add if negative = bear penalties)

    rsi = indic.get('rsi14', 50)
    stoch_k = indic.get('stoch_k', 50)
    macd_hist = indic.get('macd_hist', 0)
    hist_direction = indic.get('hist_direction', 'stable')
    rsi_mom_bear_div = indic.get('rsi_mom_bear_div', False)
    rsi_mom_bull_div = indic.get('rsi_mom_bull_div', False)
    stoch_k_trend = indic.get('stoch_k_trend', 'stable')

    # 1. RSI extreme overbought → penalty on LONG
    if rsi > 85:
        p = 10
        penalties.append(f"RSI {rsi:.0f} >85 → LONG -{p}%")
        penalty_pct += p
    elif rsi > 80:
        p = 5
        penalties.append(f"RSI {rsi:.0f} >80 → LONG -{p}%")
        penalty_pct += p

    # 2. RSI extreme oversold → penalty on SHORT
    if rsi < 15:
        p = -10
        penalties.append(f"RSI {rsi:.0f} <15 → SHORT -10%")
        penalty_pct += p
    elif rsi < 20:
        p = -5
        penalties.append(f"RSI {rsi:.0f} <20 → SHORT -5%")
        penalty_pct += p

    # 3. RSI-Mom bearish divergence → penalty on LONG
    if rsi_mom_bear_div:
        p = 5
        penalties.append(f"RSI-Mom bearish divergence → LONG -{p}%")
        penalty_pct += p

    # 4. RSI-Mom bullish divergence → penalty on SHORT
    if rsi_mom_bull_div:
        p = -5
        penalties.append(f"RSI-Mom bullish divergence → SHORT -5%")
        penalty_pct += p

    # 5. MACD fading/weakening → penalty on dominant side
    if hist_direction == "fading_bullish":
        p = 5
        penalties.append(f"MACD fading bullish → LONG -{p}%")
        penalty_pct += p
    elif hist_direction == "fading_bearish":
        p = -5
        penalties.append(f"MACD fading bearish → SHORT -5%")
        penalty_pct += p

    # 6. StochRSI bearish cross while overbought → penalty on LONG
    if stoch_k > 70 and stoch_k_trend == 'falling':
        p = 5
        penalties.append(f"StochRSI bearish cross OB ({stoch_k:.0f}, falling) → LONG -{p}%")
        penalty_pct += p
    # StochRSI bullish cross while oversold → penalty on SHORT
    elif stoch_k < 30 and stoch_k_trend == 'rising':
        p = -5
        penalties.append(f"StochRSI bullish cross OS ({stoch_k:.0f}, rising) → SHORT -5%")
        penalty_pct += p

    # --- Final LONG/SHORT percentage ---
    total_weight = bullish_weight + bearish_weight
    if total_weight > 0:
        bull_pct = bullish_weight / total_weight * 100
        bear_pct = bearish_weight / total_weight * 100
    else:
        bull_pct = bear_pct = 50

    # Apply penalties
    if penalty_pct != 0:
        bull_pct -= penalty_pct
        bear_pct += penalty_pct

    bull_pct = max(0, min(100, round(bull_pct)))
    bear_pct = 100 - bull_pct

    adx_note = f"ADX={adx:.0f}({trend_strength.lower()}, DI+ {'dominant' if di_plus > di_minus else 'weak'})"

    confluence_note = ""
    if confluence_bull_bonus > 0 or confluence_bear_bonus > 0:
        confluence_note = f" | Confluence: +{confluence_bull_bonus:.1f}🟢 +{confluence_bear_bonus:.1f}🔴"

    penalty_note = ""
    if penalties:
        penalty_note = f" | ⚠️ Penalties: {', '.join(penalties)}"

    consensus_lines = [
        f"📊 GROUPED SCORECARD (6 groups, regime={regime}): {bull_groups}🟢 vs {bear_groups}🔴 vs {neutral_groups}⚪{oi_impact}{confluence_note}{penalty_note}",
        f"   Weights: Trend/Mom ×{w_mult['trend']:.1f} | Oscillators ×{w_mult['oscillator']:.1f} | Vol/Volat/Dir ×1.0",
        f"→ LONG {bull_pct}% / SHORT {bear_pct}% | {adx_note}",
    ] + group_details

    consensus = "\n".join(consensus_lines)

    return (
        f"=== {tf_label} ===\n"
        f"Price: {price:.6f} | Change: {indic.get('change_recent', 0):+.2f}% | 24h: {indic.get('change_24h', 0):+.2f}%\n\n"
        + "\n\n".join(raw_lines) + "\n\n"
        + consensus
    )