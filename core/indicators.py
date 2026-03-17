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
    """
    df = df.copy()

    # Convert types to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. EMA (7, 25, 99)
    df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema25'] = df['close'].ewm(span=25, adjust=False).mean()
    df['ema99'] = df['close'].ewm(span=99, adjust=False).mean()

    # 2. RSI (6, 12, 24) — Binance standard periods
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    for period in [6, 12, 24]:
        avg_g = rma(gain, period)
        avg_l = rma(loss, period)
        rs = avg_g / avg_l.replace(0, np.nan)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi{period}'] = df[f'rsi{period}'].fillna(50)

    # 3. MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # 4. Volume Trend (Decay)
    df['vol_sma5'] = df['volume'].rolling(5).mean()
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['volume_decay'] = df['vol_sma5'] < df['vol_sma20']

    # 5. Local Min/Max & Fibonacci Grid
    local_min = df['low'].rolling(10).min().iloc[-1]
    local_max = df['high'].rolling(10).max().iloc[-1]
    diff_f = local_max - local_min if local_max != local_min else 1.0
    fibo = {
        "1.000": local_max,
        "0.786": local_max - diff_f * 0.214,
        "0.618": local_max - diff_f * 0.382,
        "0.500": local_max - diff_f * 0.500,
        "0.382": local_max - diff_f * 0.618,
        "0.236": local_max - diff_f * 0.764,
        "0.000": local_min
    }

    # ==========================================
    # 🚀 ADVANCED INDICATORS (NEW)
    # ==========================================

    # 6. ATR (Average True Range, param 14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = rma(tr, 14)

    # 7. SuperTrend (10, 3)
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

    # 8. ADX (Average Directional Index, 14)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_safe = df['atr'].replace(0, np.nan)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), 14) / atr_safe
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), 14) / atr_safe
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    dx = dx.fillna(0)
    df['adx'] = rma(dx, 14)

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

    # 14. VWAP (Volume Weighted Average Price) — approximation over window
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # 15. CMF (Chaikin Money Flow, 20)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df['volume']
    df['cmf'] = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

    # 16. Volume Block Analysis (2 blocks of 10 candles)
    vol_analysis = {"block1": {}, "block2": {}, "shift": ""}
    if len(df) >= 20:
        block1 = df.iloc[-20:-10]  # older 10 candles
        block2 = df.iloc[-10:]     # recent 10 candles

        b1_green_vol = block1.loc[block1['close'] > block1['open'], 'volume'].sum()
        b1_red_vol = block1.loc[block1['close'] <= block1['open'], 'volume'].sum()
        b2_green_vol = block2.loc[block2['close'] > block2['open'], 'volume'].sum()
        b2_red_vol = block2.loc[block2['close'] <= block2['open'], 'volume'].sum()

        b1_total = b1_green_vol + b1_red_vol if (b1_green_vol + b1_red_vol) > 0 else 1
        b2_total = b2_green_vol + b2_red_vol if (b2_green_vol + b2_red_vol) > 0 else 1

        b1_buy_pct = round(b1_green_vol / b1_total * 100, 1)
        b2_buy_pct = round(b2_green_vol / b2_total * 100, 1)

        # Determine power shift
        if b2_buy_pct > b1_buy_pct + 5:
            shift = "🟢 Buyers gaining strength"
        elif b1_buy_pct > b2_buy_pct + 5:
            shift = "🔴 Sellers gaining strength"
        else:
            shift = "⚪ Balanced / No clear shift"

        vol_analysis = {
            "block1_buy_pct": b1_buy_pct,
            "block1_sell_pct": round(100 - b1_buy_pct, 1),
            "block2_buy_pct": b2_buy_pct,
            "block2_sell_pct": round(100 - b2_buy_pct, 1),
            "shift": shift
        }

    # ==========================================
    # 🧠 SMART MONEY CONCEPTS (SMC)
    # ==========================================

    current_price = df['close'].iloc[-1]

    # --- FVG (Fair Value Gap) ---
    # Bullish FVG: candle[i] low > candle[i-2] high (gap up — unfilled = support)
    # Bearish FVG: candle[i-2] low > candle[i] high (gap down — unfilled = resistance)
    last_bull_fvg = "None"
    last_bear_fvg = "None"

    for i in range(len(df) - 1, 2, -1):
        gap_bottom = df['high'].iloc[i - 2]
        gap_top = df['low'].iloc[i]
        if gap_top > gap_bottom:
            # Bullish FVG exists — check if still unfilled (price hasn't dropped through it)
            if current_price >= gap_bottom:
                last_bull_fvg = f"{gap_bottom:.5f} - {gap_top:.5f}"
                break

    for i in range(len(df) - 1, 2, -1):
        gap_top = df['low'].iloc[i - 2]
        gap_bottom = df['high'].iloc[i]
        if gap_top > gap_bottom:
            # Bearish FVG exists — check if still unfilled (price hasn't risen through it)
            if current_price <= gap_top:
                last_bear_fvg = f"{gap_bottom:.5f} - {gap_top:.5f}"
                break

    # --- Order Blocks ---
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = np.abs(df['close'] - df['open'])

    strong_bull = df[(df['is_green']) & (df['body_size'] > df['atr'] * 1.5)]
    strong_bear = df[(df['is_red']) & (df['body_size'] > df['atr'] * 1.5)]

    bullish_ob = "None"
    bearish_ob = "None"

    # Bullish OB: last red candle before a strong green move, still holding as support
    if not strong_bull.empty:
        for sb_idx in reversed(strong_bull.index.tolist()):
            reds_before = df.loc[:sb_idx][df.loc[:sb_idx]['is_red']]
            if not reds_before.empty:
                ob_low = reds_before['low'].iloc[-1]
                ob_high = reds_before['high'].iloc[-1]
                # OB is valid only if price is still above it (not broken)
                if current_price >= ob_low:
                    bullish_ob = f"{ob_low:.5f} - {ob_high:.5f} (Support OB)"
                    break

    # Bearish OB: last green candle before a strong red move, still holding as resistance
    if not strong_bear.empty:
        for sb_idx in reversed(strong_bear.index.tolist()):
            greens_before = df.loc[:sb_idx][df.loc[:sb_idx]['is_green']]
            if not greens_before.empty:
                ob_low = greens_before['low'].iloc[-1]
                ob_high = greens_before['high'].iloc[-1]
                # OB is valid only if price is still below it (not broken)
                if current_price <= ob_high:
                    bearish_ob = f"{ob_low:.5f} - {ob_high:.5f} (Resist OB)"
                    break

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

    # Calculate 24h change dynamically
    if tf_key.lower() == '4h' and len(df) >= 7:
        price_24h_ago = df['close'].iloc[-7] # 6 candles of 4H = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "4H"
    elif tf_key.lower() == '1d' and len(df) >= 2:
        price_24h_ago = df['close'].iloc[-2] # 1 candle of 1D = 24H
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        recent_label = "1D"
    else:
        change_24h = 0.0
        recent_label = "Period"

    # Pack the last candle data with all current indicators
    last = df.iloc[-1]
    cloud_top = max(last['senkou_span_a'], last['senkou_span_b']) if not pd.isna(last['senkou_span_a']) else 0
    cloud_bottom = min(last['senkou_span_a'], last['senkou_span_b']) if not pd.isna(last['senkou_span_a']) else 0

    if last['close'] > cloud_top: ichi_status = "ABOVE CLOUD (Bullish)"
    elif last['close'] < cloud_bottom: ichi_status = "BELOW CLOUD (Bearish)"
    else: ichi_status = "INSIDE CLOUD (Neutral/Chop)"
    
    # Assess OBV Trend Status
    obv_status = "Bullish (Accumulation)" if last['obv'] > last['obv_sma20'] else "Bearish (Distribution)"

    last_indic_row = {
        "close": last['close'],
        "change_recent": round(change_recent, 2),
        "change_24h": round(change_24h, 2),
        "recent_label": recent_label,
        "ema7": last['ema7'], "ema25": last['ema25'], "ema99": last['ema99'],
        "rsi6": last['rsi6'],
        "rsi12": last['rsi12'],
        "rsi24": last['rsi24'],
        "macd_line": last['macd_line'],
        "macd_signal": last['macd_signal'],
        "obv_status": obv_status,
        "macd_hist": last['macd_hist'],
        "volume_decay": "Yes" if last['volume_decay'] else "No",
        "fibo_levels": fibo,
        "supertrend": "🟢 BULLISH" if last['supertrend_dir'] == 1 else "🔴 BEARISH",
        "supertrend_price": last['supertrend'],
        "adx": last['adx'],
        "stoch_k": last['stoch_k'], "stoch_d": last['stoch_d'],
        "mfi": last['mfi'],
        "ichimoku_status": ichi_status,
        "funding_rate": df.get("funding_rate", "Unknown"),
        "vol_blocks": vol_analysis,
        "bb_upper": last['bb_upper'],
        "bb_lower": last['bb_lower'],
        "bb_mid": last['bb_mid'],
        "vwap": last['vwap'],
        "cmf": last['cmf'],
        "smc_bullish_ob": bullish_ob,
        "smc_bearish_ob": bearish_ob,
        "smc_bullish_fvg": last_bull_fvg,
        "smc_bearish_fvg": last_bear_fvg
    }

    return last_indic_row, df
