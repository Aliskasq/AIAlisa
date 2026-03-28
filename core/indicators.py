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

    # 14. (VWAP removed — meaningless on 4H futures)

    # 15. CMF (Chaikin Money Flow, 20)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df['volume']
    df['cmf'] = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

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
    
    # EMA History (10 candles)
    ema7_slope_10 = 0
    ema25_slope_10 = 0
    ema99_slope_10 = 0
    ema7_25_cross_bars = None
    ema7_25_cross_dir = None
    price_above_ema7_count = 0
    price_above_ema25_count = 0
    price_above_ema99_count = 0
    
    if len(df) >= 10:
        # EMA slopes over last 10 candles
        ema7_10_ago = df['ema7'].iloc[-11]
        ema25_10_ago = df['ema25'].iloc[-11]
        ema99_10_ago = df['ema99'].iloc[-11]
        
        if ema7_10_ago > 0:
            ema7_slope_10 = ((last['ema7'] - ema7_10_ago) / ema7_10_ago) * 100
        if ema25_10_ago > 0:
            ema25_slope_10 = ((last['ema25'] - ema25_10_ago) / ema25_10_ago) * 100
        if ema99_10_ago > 0:
            ema99_slope_10 = ((last['ema99'] - ema99_10_ago) / ema99_10_ago) * 100
            
        # Price above EMA counts (last 10 candles)
        last_10 = df.tail(10)
        price_above_ema7_count = (last_10['close'] > last_10['ema7']).sum()
        price_above_ema25_count = (last_10['close'] > last_10['ema25']).sum()
        price_above_ema99_count = (last_10['close'] > last_10['ema99']).sum()
    
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
    
    # MACD History
    macd_hist_trend_5 = []
    macd_hist_direction = "unknown"
    macd_bars_since_cross = None
    macd_hist_accel = 0
    
    if len(df) >= 5:
        macd_hist_trend_5 = df['macd_hist'].tail(5).tolist()
        
        # Determine histogram direction (critical for momentum analysis)
        if len(macd_hist_trend_5) >= 3:
            recent_trend = macd_hist_trend_5[-3:]
            last_h = recent_trend[-1]
            prev_h = recent_trend[-2]
            first_h = recent_trend[0]
            
            if prev_h <= 0 < last_h:
                macd_hist_direction = "turned_positive"
            elif prev_h >= 0 > last_h:
                macd_hist_direction = "turned_negative"
            elif last_h > 0 and last_h > first_h:
                macd_hist_direction = "growing"        # Positive and increasing = strong bullish momentum
            elif last_h > 0 and last_h < first_h:
                macd_hist_direction = "fading_bullish"  # Positive but decreasing = bullish momentum FADING
            elif last_h < 0 and last_h < first_h:
                macd_hist_direction = "shrinking"       # Negative and decreasing = strong bearish momentum
            elif last_h < 0 and last_h > first_h:
                macd_hist_direction = "fading_bearish"  # Negative but increasing toward 0 = bearish momentum FADING
            else:
                macd_hist_direction = "stable"
        
        # MACD histogram acceleration
        if len(macd_hist_trend_5) >= 3:
            change1 = macd_hist_trend_5[-1] - macd_hist_trend_5[-2]
            change2 = macd_hist_trend_5[-2] - macd_hist_trend_5[-3]
            macd_hist_accel = change1 - change2
    
    # MACD line/signal cross detection (last 30 bars)
    if len(df) >= 30:
        macd_line_30 = df['macd_line'].tail(30)
        macd_signal_30 = df['macd_signal'].tail(30)
        
        for i in range(1, 30):
            prev_diff = macd_line_30.iloc[-i-1] - macd_signal_30.iloc[-i-1]
            curr_diff = macd_line_30.iloc[-i] - macd_signal_30.iloc[-i]
            
            if (prev_diff <= 0 < curr_diff) or (prev_diff >= 0 > curr_diff):
                macd_bars_since_cross = i
                break
    
    # OBV History
    obv_roc_5 = 0
    obv_spike = False
    obv_price_divergence = "none"
    
    if len(df) >= 5:
        obv_5_ago = df['obv'].iloc[-6]
        if obv_5_ago != 0:
            obv_roc_5 = ((last['obv'] - obv_5_ago) / abs(obv_5_ago)) * 100
    
    # OBV spike detection (2 std deviations over 20 candles)
    if len(df) >= 22:
        obv_changes = df['obv'].diff().tail(20)
        recent_obv_change = df['obv'].iloc[-1] - df['obv'].iloc[-3]  # Last 2 candles
        obv_std = obv_changes.std()
        if abs(recent_obv_change) > 2 * obv_std:
            obv_spike = True
    
    # OBV-Price divergence (last 10 candles)
    if len(df) >= 10:
        last_10 = df.tail(10)
        price_change = (last_10['close'].iloc[-1] - last_10['close'].iloc[0]) / last_10['close'].iloc[0]
        obv_change = (last_10['obv'].iloc[-1] - last_10['obv'].iloc[0]) / abs(last_10['obv'].iloc[0])
        
        if price_change > 0.02 and obv_change < -0.02:  # Price up, OBV down
            obv_price_divergence = "bearish"
        elif price_change < -0.02 and obv_change > 0.02:  # Price down, OBV up
            obv_price_divergence = "bullish"
    
    # RSI History
    rsi_trend_5 = "stable"
    rsi_values_5 = []
    
    if len(df) >= 5:
        rsi_values_5 = df['rsi14'].tail(5).tolist()
        if len(rsi_values_5) >= 3:
            if rsi_values_5[-1] > rsi_values_5[-3] + 2:
                rsi_trend_5 = "rising"
            elif rsi_values_5[-1] < rsi_values_5[-3] - 2:
                rsi_trend_5 = "falling"
    
    # SuperTrend History
    st_bars_since_flip = 0
    st_distance_pct = 0
    
    if len(df) >= 30:
        # Find last SuperTrend flip
        st_dir_series = df['supertrend_dir'].tail(30)
        for i in range(1, 30):
            if st_dir_series.iloc[-i] != st_dir_series.iloc[-i-1]:
                st_bars_since_flip = i
                break
    
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
        
        # Walking bands (last 10 candles)
        last_10 = df.tail(10)
        bb_walking_upper = (last_10['close'] > last_10['bb_upper'] * 0.998).sum()
        bb_walking_lower = (last_10['close'] < last_10['bb_lower'] * 1.002).sum()
    
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
    
    # ADX Trend
    adx_trend = "stable"
    if len(df) >= 5:
        adx_5 = df['adx'].tail(5)
        if adx_5.iloc[-1] > adx_5.iloc[-3] + 2:
            adx_trend = "rising"
        elif adx_5.iloc[-1] < adx_5.iloc[-3] - 2:
            adx_trend = "falling"
    
    # StochRSI Trend
    stoch_k_trend = "stable"
    if len(df) >= 5:
        stoch_k_5 = df['stoch_k'].tail(5)
        if stoch_k_5.iloc[-1] > stoch_k_5.iloc[-3] + 5:
            stoch_k_trend = "rising"
        elif stoch_k_5.iloc[-1] < stoch_k_5.iloc[-3] - 5:
            stoch_k_trend = "falling"
    
    # MFI Trend
    mfi_trend_5 = "stable"
    if len(df) >= 5:
        mfi_5 = df['mfi'].tail(5)
        if mfi_5.iloc[-1] > mfi_5.iloc[-3] + 3:
            mfi_trend_5 = "rising"
        elif mfi_5.iloc[-1] < mfi_5.iloc[-3] - 3:
            mfi_trend_5 = "falling"
    
    # CMF Trend
    cmf_trend_5 = "stable"
    if len(df) >= 5:
        cmf_5 = df['cmf'].tail(5)
        if cmf_5.iloc[-1] > cmf_5.iloc[-3] + 0.02:
            cmf_trend_5 = "rising"
        elif cmf_5.iloc[-1] < cmf_5.iloc[-3] - 0.02:
            cmf_trend_5 = "falling"

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
        
        # NEW HISTORICAL CONTEXT FIELDS
        # EMA History
        "ema7_slope_10": round(ema7_slope_10, 2),
        "ema25_slope_10": round(ema25_slope_10, 2),
        "ema99_slope_10": round(ema99_slope_10, 2),
        "ema7_25_cross_bars": ema7_25_cross_bars,
        "ema7_25_cross_dir": ema7_25_cross_dir,
        "price_above_ema7_count": price_above_ema7_count,
        "price_above_ema25_count": price_above_ema25_count,
        "price_above_ema99_count": price_above_ema99_count,
        
        # MACD History
        "macd_hist_trend_5": [round(h, 6) for h in macd_hist_trend_5],
        "macd_hist_direction": macd_hist_direction,
        "macd_bars_since_cross": macd_bars_since_cross,
        "macd_hist_accel": round(macd_hist_accel, 6),
        
        # OBV History
        "obv_roc_5": round(obv_roc_5, 2),
        "obv_spike": obv_spike,
        "obv_price_divergence": obv_price_divergence,
        
        # RSI History
        "rsi_trend_5": rsi_trend_5,
        "rsi_values_5": [round(r, 1) for r in rsi_values_5],
        
        # SuperTrend History
        "st_bars_since_flip": st_bars_since_flip,
        "st_distance_pct": round(st_distance_pct, 2),
        
        # Bollinger Bands History
        "bb_pctb": round(bb_pctb, 3),
        "bb_squeeze": bb_squeeze,
        "bb_expanding": bb_expanding,
        "bb_walking_upper": bb_walking_upper,
        "bb_walking_lower": bb_walking_lower,
        
        # Ichimoku History
        "tk_cross": tk_cross,
        "tk_cross_bars": tk_cross_bars,
        "cloud_thickness_pct": round(cloud_thickness_pct, 2),
        "future_cloud": future_cloud,
        
        # ADX/DI History
        "adx_trend": adx_trend,
        
        # StochRSI History
        "stoch_k_trend": stoch_k_trend,
        
        # MFI History
        "mfi_trend_5": mfi_trend_5,
        
        # CMF History
        "cmf_trend_5": cmf_trend_5,
    }

    return last_indic_row, df


def format_tf_summary(indic: dict, tf_label: str) -> str:
    """Format one timeframe's indicators with detailed historical analysis for AI prompt."""
    price = indic['close']
    ema7 = indic['ema7']
    ema25 = indic['ema25']
    ema99 = indic['ema99']

    # ── EMA ANALYSIS WITH HISTORICAL CONTEXT ──
    ema7_slope = indic.get('ema7_slope_10', 0)
    ema25_slope = indic.get('ema25_slope_10', 0)
    ema99_slope = indic.get('ema99_slope_10', 0)
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
    ema_analysis = (f"{idx}. EMA: 7={ema7:.6f}({slope7_ind}{ema7_slope:+.1f}%/10bars) {'>' if ema7 > ema25 else '<'} "
                   f"25={ema25:.6f}({slope25_ind}{ema25_slope:+.1f}%) {'>' if ema25 > ema99 else '<'} "
                   f"99={ema99:.6f}({slope99_ind}{'flat' if abs(ema99_slope) < 0.5 else f'{ema99_slope:+.1f}%'})\n"
                   f"   Price>EMA7: {price_above_7}/10 candles | Price>EMA25: {price_above_25}/10 | Price>EMA99: {price_above_99}/10{cross_text}\n"
                   f"   → {ema_signal}")

    # ── MACD ANALYSIS WITH HISTOGRAM DYNAMICS ──
    macd_line = indic['macd_line']
    macd_signal_val = indic['macd_signal']
    macd_hist = indic['macd_hist']
    hist_direction = indic.get('macd_hist_direction', 'unknown')
    hist_trend = indic.get('macd_hist_trend_5', [])
    bars_since_cross = indic.get('macd_bars_since_cross')
    hist_accel = indic.get('macd_hist_accel', 0)
    
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
                    f"   {cross_text}, hist peaked at {hist_peak:.6f}, accel={hist_accel:.6f}\n"
                    f"   → {macd_signal}")

    # ── OBV ANALYSIS WITH SPIKE DETECTION ──
    obv_status = indic['obv_status']
    obv_roc = indic.get('obv_roc_5', 0)
    obv_spike = indic.get('obv_spike', False)
    obv_divergence = indic.get('obv_price_divergence', 'none')
    
    spike_text = f" ⚠️ SPIKE: +{abs(obv_roc):.1f}% last 5 bars (possible short-term {'pump' if obv_roc > 0 else 'dump'})" if obv_spike else ""
    divergence_text = f"Price-OBV: {'NO DIVERGENCE (confirmed move)' if obv_divergence == 'none' else f'{obv_divergence.upper()} DIVERGENCE'}"
    
    if "Accumulation" in obv_status:
        obv_signal = f"🟢 BULLISH{'but spike=caution' if obv_spike else ''}"
        obv_vote_weight = 0.5 if obv_spike else 1.0
    else:
        obv_signal = f"🔴 BEARISH{'but spike=caution' if obv_spike else ''}"
        obv_vote_weight = 0.5 if obv_spike else 1.0
    
    idx += 1
    obv_analysis = (f"{idx}. OBV: {obv_status.split('(')[1].replace(')', '')} (>SMA20) | ROC(5)={obv_roc:+.1f}%{spike_text}\n"
                   f"   {divergence_text}\n"
                   f"   → {obv_signal}")

    # ── RSI ANALYSIS WITH PENALTY SYSTEM ──
    rsi = indic['rsi14']
    rsi_trend = indic.get('rsi_trend_5', 'stable')
    rsi_values = indic.get('rsi_values_5', [])
    
    values_str = "→".join([f"{v:.1f}" for v in rsi_values[-5:]]) if rsi_values else f"{rsi:.1f}"
    
    rsi_penalty = 0
    if rsi > 80:
        rsi_signal = f"⚠️ BEARISH VOTE (extremely overbought override)"
        rsi_penalty = -20  # -20% penalty from LONG
    elif rsi > 70:
        rsi_signal = f"⚠️ BEARISH VOTE (overbought override)"
        rsi_penalty = -10  # -10% penalty from LONG
    elif rsi < 20:
        rsi_signal = f"⚠️ BULLISH VOTE (extremely oversold override)"
        rsi_penalty = 20  # +20% bonus to LONG
    elif rsi < 30:
        rsi_signal = f"⚠️ BULLISH VOTE (oversold override)"
        rsi_penalty = 10  # +10% bonus to LONG
    elif rsi > 55:
        rsi_signal = f"🟢 BULLISH"
        rsi_penalty = 0
    elif rsi < 45:
        rsi_signal = f"🔴 BEARISH"
        rsi_penalty = 0
    else:
        rsi_signal = f"⚪ NEUTRAL"
        rsi_penalty = 0
    
    penalty_text = f"\n   PENALTY: {rsi_penalty:+d}% from LONG score" if rsi_penalty != 0 else ""
    
    idx += 1
    rsi_analysis = (f"{idx}. RSI: {rsi:.1f} {'OVERBOUGHT' if rsi > 70 else ('OVERSOLD' if rsi < 30 else 'NORMAL')} | "
                   f"Trend: {rsi_trend} 5 bars ({values_str}){penalty_text}\n"
                   f"   → {rsi_signal}")

    # ── SUPERTREND ANALYSIS WITH FLIP TIMING ──
    st_status = indic['supertrend']
    st_price = indic['supertrend_price']
    st_flip_bars = indic.get('st_bars_since_flip', 0)
    st_distance = indic.get('st_distance_pct', 0)
    
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
        bb_signal = f"⚠️ BEARISH (at upper band, potential resistance)"
    elif price <= bb_lower * 1.002:
        bb_signal = f"⚠️ BULLISH (at lower band, potential support)"
    elif bb_pctb > 0.5:
        bb_signal = f"🟢 BULLISH (above mid, expanding, trend walk)" if bb_expanding else f"🟢 BULLISH (above mid)"
    else:
        bb_signal = f"🔴 BEARISH (below mid)"
    
    idx += 1
    bb_analysis = (f"{idx}. BB: Upper={bb_upper:.6f} Mid={bb_mid:.6f} Lower={bb_lower:.6f} | %B={bb_pctb:.3f} | Width={bb_width_pct:.1f}%\n"
                  f"   Squeeze: {'YES' if bb_squeeze else 'NO'} | Expanding: {'YES' if bb_expanding else 'NO'} | "
                  f"Walking upper band: {bb_walk_upper}/10 candles\n"
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
    
    if adx > 40:
        trend_strength = "STRONG TREND"
    elif adx > 25:
        trend_strength = "MODERATE TREND"
    else:
        trend_strength = "WEAK/NO TREND"
    
    if di_plus > di_minus:
        adx_signal = f"🟢 BULLISH (DI+ dominant, trend {'strengthening' if adx_trend == 'rising' else 'stable'})"
    else:
        adx_signal = f"🔴 BEARISH (DI- dominant, trend {'strengthening' if adx_trend == 'rising' else 'stable'})"
    
    idx += 1
    adx_analysis = (f"{idx}. ADX: {adx:.0f} {trend_strength} | DI+: {di_plus:.1f} {'>' if di_plus > di_minus else '<'} "
                   f"DI-: {di_minus:.1f} | ADX {adx_trend}\n"
                   f"   → {adx_signal}")
    raw_lines.append(adx_analysis)
    
    # StochRSI (NOW VOTES)
    stoch_k = indic['stoch_k']
    stoch_d = indic['stoch_d']
    stoch_k_trend = indic.get('stoch_k_trend', 'stable')
    
    if stoch_k > 80:
        stoch_signal = f"⚠️ BEARISH VOTE (overbought zone)"
    elif stoch_k < 20:
        stoch_signal = f"⚠️ BULLISH VOTE (oversold zone)"
    elif stoch_k > stoch_d:
        stoch_signal = f"🟢 BULLISH (K>D cross)"
    else:
        stoch_signal = f"🔴 BEARISH (K<D cross)"
    
    idx += 1
    stoch_analysis = (f"{idx}. StochRSI: K={stoch_k:.0f} D={stoch_d:.0f} "
                     f"{'OVERBOUGHT' if stoch_k > 80 else ('OVERSOLD' if stoch_k < 20 else 'NORMAL')} | "
                     f"K trend: {stoch_k_trend}\n"
                     f"   → {stoch_signal}")
    raw_lines.append(stoch_analysis)
    
    # MFI (NOW VOTES)
    mfi = indic['mfi']
    mfi_trend = indic.get('mfi_trend_5', 'stable')
    
    if mfi > 80:
        mfi_signal = f"⚠️ BEARISH (overbought)"
    elif mfi < 20:
        mfi_signal = f"⚠️ BULLISH (oversold)"
    elif mfi > 50:
        mfi_signal = f"🟢 BULLISH"
    else:
        mfi_signal = f"🔴 BEARISH"
    
    idx += 1
    mfi_analysis = (f"{idx}. MFI: {mfi:.0f} {'BULLISH' if mfi > 50 else 'BEARISH'} | Trend: {mfi_trend} 5 bars\n"
                   f"   → {mfi_signal}")
    raw_lines.append(mfi_analysis)
    
    # CMF (NOW VOTES)
    cmf = indic['cmf']
    cmf_trend = indic.get('cmf_trend_5', 'stable')
    
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
                   f"{'BUYING' if cmf > 0 else 'SELLING'} | Trend: {cmf_trend}\n"
                   f"   → {cmf_signal}")
    raw_lines.append(cmf_analysis)

    # ── WEIGHTED VOTING SCORECARD ──
    bullish_votes = 0
    bearish_votes = 0
    neutral_votes = 0
    bullish_weight = 0.0
    bearish_weight = 0.0
    indicator_votes = []
    
    # Count votes with weights
    voting_indicators = [
        ("EMA", ema_signal, 0.5 if ema7_slope < 0 and "BULLISH" in ema_signal else 1.0),
        ("RSI", rsi_signal, 1.0),  # RSI penalty handled separately
        ("MACD", macd_signal, macd_vote_weight),
        ("OBV", obv_signal, obv_vote_weight),
        ("BB", bb_signal, 1.0),
    ]
    
    # SuperTrend votes on all TFs; reduced weight on 15m (noisy but catches early reversals)
    _st_weight = st_vote_weight * (0.5 if tf_upper == "15M" else 1.0)
    voting_indicators.append(("SuperTrend", st_status.split()[1], _st_weight))
    
    if is_higher_tf:
        ichi_vote = "🟢" if "🟢" in ichi_signal else ("🔴" if "🔴" in ichi_signal else "⚪")
        voting_indicators.append(("Ichimoku", ichi_vote, 1.0))
    
    # Add new voting indicators
    voting_indicators.extend([
        ("ADX", adx_signal, 1.0),
        ("StochRSI", stoch_signal, 1.0),
        ("MFI", mfi_signal, 1.0),
        ("CMF", cmf_signal, 1.0),
    ])
    
    for name, signal, weight in voting_indicators:
        if "🟢" in signal or "BULLISH" in signal:
            bullish_votes += 1
            bullish_weight += weight
            if weight != 1.0:
                indicator_votes.append(f"{name}=🟢½")
            else:
                indicator_votes.append(f"{name}=🟢")
        elif "🔴" in signal or "BEARISH" in signal:
            bearish_votes += 1
            bearish_weight += weight
            if "⚠️" in signal and ("overbought" in signal.lower() or "oversold" in signal.lower()):
                indicator_votes.append(f"{name}=⚠️{'OB' if 'overbought' in signal.lower() else 'OS'}")
            else:
                indicator_votes.append(f"{name}=🔴")
        else:
            neutral_votes += 1
            indicator_votes.append(f"{name}=⚪")
    
    # Calculate base percentages
    total_weight = bullish_weight + bearish_weight
    if total_weight > 0:
        bull_pct = bullish_weight / total_weight * 100
        bear_pct = bearish_weight / total_weight * 100
    else:
        bull_pct = bear_pct = 50
    
    # Apply RSI penalties
    if rsi_penalty != 0:
        penalty_text = f"RSI PENALTY: {rsi_penalty:+d}% ({'extremely ' if abs(rsi_penalty) > 15 else ''}{'overbought' if rsi_penalty < 0 else 'oversold'})"
        if rsi_penalty < 0:  # Reduce LONG
            bull_pct = max(0, bull_pct + rsi_penalty)
            bear_pct = min(100, 100 - bull_pct)
        else:  # Increase LONG
            bull_pct = min(100, bull_pct + rsi_penalty)
            bear_pct = max(0, 100 - bull_pct)
    else:
        penalty_text = ""
    
    # Check for Open Interest impact (if positioning data available)
    oi_impact = ""
    positioning = indic.get("positioning", {})
    if positioning and "oi_change_pct" in positioning:
        oi_change = positioning.get("oi_change_pct", 0)
        if oi_change > 5:
            bullish_weight += 0.5  # OI rising = bullish vote
            oi_impact = " +OI📈"
        elif oi_change < -5:
            bearish_weight += 0.5  # OI falling = bearish vote
            oi_impact = " +OI📉"
    
    # Final percentage calculation with OI
    if oi_impact:
        total_weight = bullish_weight + bearish_weight
        if total_weight > 0:
            bull_pct = bullish_weight / total_weight * 100
            bear_pct = bearish_weight / total_weight * 100
    
    bull_pct = max(0, min(100, round(bull_pct)))
    bear_pct = 100 - bull_pct
    
    votes_str = " ".join(indicator_votes)
    voting_count = len(voting_indicators)
    
    # ADX trend strength note
    adx_note = f"ADX={adx:.0f}({trend_strength.lower()}, DI+ {'dominant' if di_plus > di_minus else 'weak'})"
    
    consensus_line1 = (f"📊 SCORECARD ({voting_count} voting): {bullish_votes}🟢({bullish_weight:.1f} weighted) vs "
                      f"{bearish_votes}🔴({bearish_weight:.1f} weighted) vs {neutral_votes}⚪{oi_impact}")
    consensus_line2 = penalty_text if penalty_text else ""
    consensus_line3 = f"→ LONG {bull_pct}% / SHORT {bear_pct}% | {adx_note}"
    consensus_line4 = f"[{votes_str}]"
    
    consensus = "\n".join(filter(None, [consensus_line1, consensus_line2, consensus_line3, consensus_line4]))

    return (
        f"=== {tf_label} ===\n"
        f"Price: {price:.6f} | Change: {indic.get('change_recent', 0):+.2f}% | 24h: {indic.get('change_24h', 0):+.2f}%\n\n"
        + "\n\n".join(raw_lines) + "\n\n"
        + consensus
    )