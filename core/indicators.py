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

    # 14. (VWAP removed — meaningless on 4H futures)

    # 15. CMF (Chaikin Money Flow, 20)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df['volume']
    df['cmf'] = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

    # (Old SMC code removed — replaced by core/smc.py with full LuxAlgo port)

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
        "rsi14": last['rsi14'],
        "macd_line": last['macd_line'],
        "macd_signal": last['macd_signal'],
        "macd_hist": last['macd_hist'],
        "obv_status": obv_status,
        "supertrend": "🟢 BULLISH" if last['supertrend_dir'] == 1 else "🔴 BEARISH",
        "supertrend_price": last['supertrend'],
        "adx": last['adx'],
        "stoch_k": last['stoch_k'], "stoch_d": last['stoch_d'],
        "mfi": last['mfi'],
        "ichimoku_status": ichi_status,
        "funding_rate": df.get("funding_rate", "Unknown"),
        "bb_upper": last['bb_upper'],
        "bb_lower": last['bb_lower'],
        "bb_mid": last['bb_mid'],
        "cmf": last['cmf']
    }

    return last_indic_row, df


def format_tf_summary(indic: dict, tf_label: str) -> str:
    """Format one timeframe's indicators into a compact text block for AI prompt."""
    price = indic['close']
    ema7 = indic['ema7']
    ema25 = indic['ema25']
    ema99 = indic['ema99']

    # EMA position analysis
    ema_positions = []
    if price > ema7: ema_positions.append("Price>EMA7")
    else: ema_positions.append("Price<EMA7")
    if price > ema25: ema_positions.append("Price>EMA25")
    else: ema_positions.append("Price<EMA25")
    if price > ema99: ema_positions.append("Price>EMA99")
    else: ema_positions.append("Price<EMA99")

    # EMA crosses
    ema_crosses = []
    if ema7 > ema25: ema_crosses.append("EMA7>EMA25 (bullish)")
    else: ema_crosses.append("EMA7<EMA25 (bearish)")
    if ema25 > ema99: ema_crosses.append("EMA25>EMA99 (bullish)")
    else: ema_crosses.append("EMA25<EMA99 (bearish)")

    # Golden/Death cross
    if ema7 > ema25 > ema99:
        alignment = "🟢 BULLISH ALIGNMENT (EMA7>25>99)"
    elif ema7 < ema25 < ema99:
        alignment = "🔴 BEARISH ALIGNMENT (EMA7<25<99)"
    else:
        alignment = "⚪ MIXED (no full alignment)"

    return (
        f"=== {tf_label} ===\n"
        f"Price: {price:.6f} | Change: {indic.get('change_recent', 0):+.2f}% ({indic.get('recent_label', tf_label)}) | 24h: {indic.get('change_24h', 0):+.2f}%\n"
        f"EMA7: {ema7:.6f} | EMA25: {ema25:.6f} | EMA99: {ema99:.6f}\n"
        f"EMA Position: {', '.join(ema_positions)} | {alignment}\n"
        f"EMA Crosses: {', '.join(ema_crosses)}\n"
        f"RSI(14): {indic['rsi14']:.1f} | StochRSI K/D: {indic['stoch_k']:.1f}/{indic['stoch_d']:.1f} | MFI: {indic['mfi']:.1f}\n"
        f"MACD: {indic['macd_line']:.6f} / Signal: {indic['macd_signal']:.6f} / Hist: {indic['macd_hist']:.6f}\n"
        f"SuperTrend: {indic['supertrend']} @ {indic['supertrend_price']:.6f} | ADX: {indic['adx']:.1f}\n"
        f"Bollinger: {indic['bb_lower']:.6f} / {indic['bb_mid']:.6f} / {indic['bb_upper']:.6f}\n"
        f"Ichimoku: {indic['ichimoku_status']} | OBV: {indic['obv_status']} | CMF: {indic['cmf']:.4f}"
    )
