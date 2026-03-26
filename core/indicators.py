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

    # 8. ADX (Average Directional Index, 14)
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
        "supertrend_dir_raw": int(last['supertrend_dir']),  # 1 for bullish, -1 for bearish
        "supertrend_price": last['supertrend'],
        "adx": last['adx'],
        "stoch_k": last['stoch_k'], "stoch_d": last['stoch_d'],
        "mfi": last['mfi'],
        "ichimoku_status": ichi_status,
        "ichi_cloud_top": cloud_top,
        "ichi_cloud_bottom": cloud_bottom,
        "funding_rate": df.get("funding_rate", "Unknown"),
        "bb_upper": last['bb_upper'],
        "bb_lower": last['bb_lower'],
        "bb_mid": last['bb_mid'],
        "cmf": last['cmf']
    }

    return last_indic_row, df


def format_tf_summary(indic: dict, tf_label: str) -> str:
    """Format one timeframe's indicators into pre-interpreted signals for AI prompt."""
    price = indic['close']
    ema7 = indic['ema7']
    ema25 = indic['ema25']
    ema99 = indic['ema99']

    # ── EMA ANALYSIS ──
    if ema7 > ema25 > ema99:
        ema_signal = f"🟢 BULLISH ALIGNMENT (EMA7={ema7:.6f} > EMA25={ema25:.6f} > EMA99={ema99:.6f})"
    elif ema7 < ema25 < ema99:
        ema_signal = f"🔴 BEARISH ALIGNMENT (EMA7={ema7:.6f} < EMA25={ema25:.6f} < EMA99={ema99:.6f})"
    else:
        ema_signal = f"⚪ MIXED (EMA7={ema7:.6f}, EMA25={ema25:.6f}, EMA99={ema99:.6f})"

    # Distance from key EMAs (for SL/TP placement)
    ema99_dist_pct = ((price - ema99) / ema99) * 100 if ema99 > 0 else 0
    ema25_dist_pct = ((price - ema25) / ema25) * 100 if ema25 > 0 else 0

    # ── RSI ANALYSIS ──
    rsi = indic['rsi14']
    if rsi > 80: rsi_signal = f"⛔ EXTREMELY OVERBOUGHT ({rsi:.1f}) — reversal likely"
    elif rsi > 70: rsi_signal = f"⚠️ OVERBOUGHT ({rsi:.1f}) — caution for longs"
    elif rsi < 20: rsi_signal = f"⛔ EXTREMELY OVERSOLD ({rsi:.1f}) — reversal likely"
    elif rsi < 30: rsi_signal = f"⚠️ OVERSOLD ({rsi:.1f}) — caution for shorts"
    elif rsi > 55: rsi_signal = f"🟢 BULLISH ({rsi:.1f})"
    elif rsi < 45: rsi_signal = f"🔴 BEARISH ({rsi:.1f})"
    else: rsi_signal = f"⚪ NEUTRAL ({rsi:.1f})"

    # ── STOCHASTIC RSI ──
    stoch_k = indic['stoch_k']
    stoch_d = indic['stoch_d']
    if stoch_k > 80:
        stoch_signal = f"⚠️ OVERBOUGHT K={stoch_k:.0f}"
        if stoch_k < stoch_d: stoch_signal += " + BEARISH CROSS (K<D)"
    elif stoch_k < 20:
        stoch_signal = f"⚠️ OVERSOLD K={stoch_k:.0f}"
        if stoch_k > stoch_d: stoch_signal += " + BULLISH CROSS (K>D)"
    elif stoch_k > stoch_d:
        stoch_signal = f"🟢 BULLISH CROSS K={stoch_k:.0f}>D={stoch_d:.0f}"
    else:
        stoch_signal = f"🔴 BEARISH CROSS K={stoch_k:.0f}<D={stoch_d:.0f}"

    # ── MACD ANALYSIS ──
    macd_line = indic['macd_line']
    macd_signal_val = indic['macd_signal']
    macd_hist = indic['macd_hist']
    if macd_line > macd_signal_val:
        if macd_hist > 0:
            macd_signal = f"🟢 BULLISH (line={macd_line:.6f} > signal={macd_signal_val:.6f}, hist={macd_hist:.6f}"
            macd_signal += ", GROWING)" if macd_hist > abs(macd_line - macd_signal_val) * 0.5 else ", weakening)"
        else:
            macd_signal = f"🟡 BULLISH CROSS but hist negative (line={macd_line:.6f}, signal={macd_signal_val:.6f}, hist={macd_hist:.6f})"
    else:
        if macd_hist < 0:
            macd_signal = f"🔴 BEARISH (line={macd_line:.6f} < signal={macd_signal_val:.6f}, hist={macd_hist:.6f}"
            macd_signal += ", GROWING)" if abs(macd_hist) > abs(macd_signal_val - macd_line) * 0.5 else ", weakening)"
        else:
            macd_signal = f"🟡 BEARISH CROSS but hist positive (line={macd_line:.6f}, signal={macd_signal_val:.6f}, hist={macd_hist:.6f})"

    # ── OBV ANALYSIS ──
    obv_status = indic['obv_status']
    if "Accumulation" in obv_status:
        obv_signal = "🟢 ACCUMULATION (OBV > SMA20 — buyers dominating)"
    else:
        obv_signal = "🔴 DISTRIBUTION (OBV < SMA20 — sellers dominating)"

    # ── CMF ANALYSIS ──
    cmf = indic['cmf']
    if cmf > 0.1: cmf_signal = f"🟢 STRONG BUYING PRESSURE (CMF={cmf:.3f})"
    elif cmf > 0.0: cmf_signal = f"🟢 MILD BUYING (CMF={cmf:.3f})"
    elif cmf < -0.1: cmf_signal = f"🔴 STRONG SELLING PRESSURE (CMF={cmf:.3f})"
    elif cmf < 0.0: cmf_signal = f"🔴 MILD SELLING (CMF={cmf:.3f})"
    else: cmf_signal = f"⚪ NEUTRAL (CMF={cmf:.3f})"

    # ── MFI ANALYSIS ──
    mfi = indic['mfi']
    if mfi > 80: mfi_signal = f"⚠️ MFI OVERBOUGHT ({mfi:.0f}) — money flow exhausted"
    elif mfi < 20: mfi_signal = f"⚠️ MFI OVERSOLD ({mfi:.0f}) — money flow depleted"
    elif mfi > 50: mfi_signal = f"🟢 MFI BULLISH ({mfi:.0f})"
    else: mfi_signal = f"🔴 MFI BEARISH ({mfi:.0f})"

    # ── SUPERTREND ──
    st = indic['supertrend']
    st_price = indic['supertrend_price']

    # ── ADX (trend strength) ──
    adx = indic['adx']
    if adx > 40: adx_signal = f"💪 STRONG TREND (ADX={adx:.0f})"
    elif adx > 25: adx_signal = f"📈 MODERATE TREND (ADX={adx:.0f})"
    else: adx_signal = f"😴 WEAK/NO TREND (ADX={adx:.0f})"

    # ── BOLLINGER BANDS ──
    bb_upper = indic['bb_upper']
    bb_lower = indic['bb_lower']
    bb_mid = indic['bb_mid']
    bb_width_pct = ((bb_upper - bb_lower) / bb_mid) * 100 if bb_mid > 0 else 0
    if price >= bb_upper * 0.998:
        bb_signal = f"⚠️ AT UPPER BAND — potential reversal/resistance"
    elif price <= bb_lower * 1.002:
        bb_signal = f"⚠️ AT LOWER BAND — potential support/bounce"
    elif price > bb_mid:
        bb_signal = f"🟢 ABOVE MID (bullish side)"
    else:
        bb_signal = f"🔴 BELOW MID (bearish side)"

    # ── ICHIMOKU ──
    ichi = indic['ichimoku_status']

    # ── INDICATOR SCORECARD (count ALL 12 base indicators) ──
    # ── SCORECARD: only reliable indicators vote ──
    # Excluded from voting (shown as info only):
    #   - StochRSI: duplicates RSI, too noisy on lower TFs
    #   - CMF: duplicates OBV
    #   - MFI: duplicates RSI+OBV
    #   - Ichimoku: unreliable on 15m and 1H (designed for daily)
    #   - SuperTrend: unreliable on 15m (whipsaw)
    tf_upper = tf_label.upper()
    is_higher_tf = tf_upper in ("1D", "4H")
    is_1h_or_higher = tf_upper in ("1D", "4H", "1H")

    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    indicator_votes = []

    # Core voting indicators (always vote)
    voting_indicators = {
        "EMA": ema_signal,
        "RSI": rsi_signal,
        "MACD": macd_signal,
        "OBV": obv_signal,
        "BB": bb_signal,
    }
    # SuperTrend votes on 1H+ only
    if is_1h_or_higher:
        voting_indicators["SuperTrend"] = st
    # Ichimoku votes on 4H+ only
    if is_higher_tf:
        voting_indicators["Ichimoku"] = ichi

    for name, sig in voting_indicators.items():
        if "🟢" in sig or "BULLISH" in sig or "ACCUMULATION" in sig or "BUYING" in sig:
            bullish_count += 1
            indicator_votes.append(f"{name}=🟢")
        elif "🔴" in sig or "BEARISH" in sig or "DISTRIBUTION" in sig or "SELLING" in sig:
            bearish_count += 1
            indicator_votes.append(f"{name}=🔴")
        elif "⚠️" in sig:
            if "OVERBOUGHT" in sig:
                bearish_count += 1
                indicator_votes.append(f"{name}=⚠️OB")
            elif "OVERSOLD" in sig:
                bullish_count += 1
                indicator_votes.append(f"{name}=⚠️OS")
            else:
                neutral_count += 1
                indicator_votes.append(f"{name}=⚪")
        else:
            neutral_count += 1
            indicator_votes.append(f"{name}=⚪")

    # ADX doesn't vote direction but shows trend strength
    adx_note = f"ADX={adx:.0f}({'strong' if adx > 25 else 'weak'})"

    total = bullish_count + bearish_count
    if total > 0:
        bull_pct = round(bullish_count / total * 100)
        bear_pct = 100 - bull_pct
    else:
        bull_pct = bear_pct = 50

    votes_str = " ".join(indicator_votes)
    voting_count = bullish_count + bearish_count + neutral_count
    consensus = (
        f"📊 SCORECARD ({voting_count} voting): {bullish_count}🟢 vs {bearish_count}🔴 vs {neutral_count}⚪ "
        f"→ LONG {bull_pct}% / SHORT {bear_pct}% | {adx_note}\n"
        f"   [{votes_str}]"
    )

    # Build indicator lines: RAW VALUES + interpretation + scorecard
    st_dist_pct = ((price - st_price) / st_price) * 100 if st_price > 0 else 0

    raw_lines = [
        f"1. EMA: EMA7={ema7:.6f} | EMA25={ema25:.6f} | EMA99={ema99:.6f} | Price {ema25_dist_pct:+.1f}% from EMA25, {ema99_dist_pct:+.1f}% from EMA99 → {ema_signal}",
        f"2. RSI: {rsi:.1f} → {rsi_signal}",
        f"3. MACD: line={macd_line:.6f} signal={macd_signal_val:.6f} hist={macd_hist:.6f} → {macd_signal}",
        f"4. OBV: {obv_signal}",
        f"5. BB: Upper={bb_upper:.6f} Mid={bb_mid:.6f} Lower={bb_lower:.6f} Width={bb_width_pct:.1f}% → {bb_signal}",
    ]
    idx = 6
    if is_1h_or_higher:
        raw_lines.append(f"{idx}. SuperTrend: {st} @ {st_price:.6f} (price {st_dist_pct:+.1f}% from ST line)")
        idx += 1
    if is_higher_tf:
        raw_lines.append(f"{idx}. Ichimoku: {ichi}")
        idx += 1
    raw_lines.append(f"{idx}. ADX: {adx:.1f} → {adx_signal}")

    # Trend Filter Consensus
    st_dir_raw = indic.get('supertrend_dir_raw', 1)  # 1 bullish, -1 bearish
    ichi_cloud_top = indic.get('ichi_cloud_top', 0)
    ichi_cloud_bottom = indic.get('ichi_cloud_bottom', 0)
    
    trend_bull = sum([ema7 > ema25 > ema99, st_dir_raw == 1, price > ichi_cloud_top]) >= 2
    trend_bear = sum([ema7 < ema25 < ema99, st_dir_raw == -1, price < ichi_cloud_bottom]) >= 2
    
    if trend_bull and adx > 25:
        trend_filter = "STRONG BULLISH TREND — ONLY LONG"
    elif trend_bear and adx > 25:
        trend_filter = "STRONG BEARISH TREND — ONLY SHORT"
    elif adx < 25:
        trend_filter = "WEAK TREND (ADX < 25) — BOTH DIRECTIONS POSSIBLE"
    else:
        trend_filter = "TREND NEUTRAL — BOTH DIRECTIONS POSSIBLE"
    
    # MACD dynamic analysis
    macd_dynamic = ""
    if macd_line > macd_signal_val and macd_hist > 0:
        macd_dynamic = "MACD: bullish, histogram growing"
    elif macd_line > macd_signal_val and macd_hist <= 0:
        macd_dynamic = "MACD: bullish crossover but histogram weak"
    elif macd_line < macd_signal_val and macd_hist < 0:
        macd_dynamic = "MACD: bearish, histogram growing"
    elif macd_line < macd_signal_val and macd_hist >= 0:
        macd_dynamic = "MACD: bearish crossover but histogram weak"
    else:
        macd_dynamic = "MACD: neutral"
    
    # RSI dynamic context
    rsi_dynamic = ""
    if rsi > 70:
        rsi_dynamic = "RSI overbought — caution for longs"
    elif rsi < 30:
        rsi_dynamic = "RSI oversold — caution for shorts"
    elif rsi > 55:
        rsi_dynamic = "RSI bullish zone"
    elif rsi < 45:
        rsi_dynamic = "RSI bearish zone"
    else:
        rsi_dynamic = "RSI neutral"
    
    # OBV dynamic (already in obv_signal)
    
    trend_block = f"\n🎯 TREND FILTER: {trend_filter}\n📈 DYNAMIC: {macd_dynamic} | {rsi_dynamic}"

    return (
        f"=== {tf_label} ===\n"
        f"Price: {price:.6f} | Change: {indic.get('change_recent', 0):+.2f}% | 24h: {indic.get('change_24h', 0):+.2f}%\n"
        + "\n".join(raw_lines) + trend_block + "\n"
        + consensus
    )
