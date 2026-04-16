"""
Signal Pipeline — two-tier signal system with 30-min re-monitoring.

FULL SIGNAL (🟢): confidence ≥ 65% AND ADX > 20 → trade entry (+ momentum override)
MONITOR   (🔵): everything else                  → 30-min re-check (indicators first, AI only if promising)

Monitor loop: Phase 1 = indicators only (free). Phase 2 = AI only when scorecard improves.
Expires after 24h without upgrade.
"""

import json
import os
import logging
import time
from datetime import datetime, timezone

MONITOR_FILE = "data/signal_monitor.json"
VOLUME_WAITLIST_FILE = "data/volume_waitlist.json"
RECHECK_INTERVAL_SEC = 1800  # 30 minutes from each signal's OWN detection time
MAX_MONITOR_HOURS = 24  # expire signal monitors after 24h

# --- Confidence threshold ---
CONFIDENCE_FULL = 60      # % to issue full signal (consensus across TFs can go as low as 60%)
ADX_TRENDING = 20         # minimum ADX for trending market

# --- Fixed risk params ---
FIXED_LEVERAGE = 1
FIXED_DEPOSIT_PCT = 2  # 2% of bank per trade


def classify_signal(long_pct: float, short_pct: float, adx: float,
                    adx_trend: str = "stable", adx_avg_50: float = 0,
                    mtf_data: dict = None, indicators: dict = None) -> str:
    """
    Classify signal tier based on confidence and market regime.
    
    Uses ADX from 4H (primary breakout TF) + its 50-candle dynamics + MTF context.
    
    KEY INSIGHT: Low ADX on 4H doesn't always mean flat market.
    - ADX low + RISING → trend just starting (breakout from consolidation) → DON'T penalize
    - ADX low + FALLING → trend dying → penalize
    - ADX low on 4H but HIGH on 1H/15m → early trend, 4H hasn't caught up yet → DON'T penalize
    
    PENALTIES (force monitor even with high confidence):
    - RSI > 85 on 4H + escalating across TFs (4H < 1H < 15m) = overbought → monitor LONG
    - RSI < 15 on 4H + escalating down (4H > 1H > 15m) = oversold → monitor SHORT
    - 15m candle pump > 10% = too volatile → monitor
    
    Returns: "full" or "monitor"
    """
    confidence = max(long_pct, short_pct)
    direction = "LONG" if long_pct > short_pct else "SHORT"
    
    # === RSI PENALTY (only if escalating across TFs: 4H→1H→15m) ===
    if indicators and mtf_data:
        rsi_4h = indicators.get("rsi14", 50)
        rsi_1h = mtf_data.get("1H", {}).get("rsi14", 0)
        rsi_15m = mtf_data.get("15m", {}).get("rsi14", 0)
        # Overbought: RSI > 85 on 4H AND escalating (1H > 4H AND 15m > 1H)
        if rsi_4h > 85 and direction == "LONG":
            if rsi_1h > rsi_4h and rsi_15m > rsi_1h:
                logging.info(f"⚠️ RSI penalty: escalating overbought 4H={rsi_4h:.1f}→1H={rsi_1h:.1f}→15m={rsi_15m:.1f} — forcing monitor for LONG")
                return "monitor"
        # Oversold: RSI < 15 on 4H AND escalating down
        if rsi_4h < 15 and direction == "SHORT":
            if rsi_1h < rsi_4h and rsi_15m < rsi_1h:
                logging.info(f"⚠️ RSI penalty: escalating oversold 4H={rsi_4h:.1f}→1H={rsi_1h:.1f}→15m={rsi_15m:.1f} — forcing monitor for SHORT")
                return "monitor"
    
    # === 15m PUMP PENALTY ===
    if mtf_data:
        indic_15m = mtf_data.get("15m", {})
        change_15m = indic_15m.get("change_recent", 0)
        if abs(change_15m) > 10:
            logging.info(f"⚠️ Pump penalty: 15m candle moved {change_15m:+.1f}% (>10%) — forcing monitor")
            return "monitor"
    
    # === MULTI-TF ADX CONTEXT ===
    # 4H ADX lags behind — check if lower TFs already show strong trend
    mtf_adx_boost = False
    if mtf_data:
        adx_1h = mtf_data.get("1H", {}).get("adx", 0)
        adx_15m = mtf_data.get("15m", {}).get("adx", 0)
        # If 1H or 15m shows strong trend (ADX > 25), the 4H is just lagging
        if adx_1h >= 25 or adx_15m >= 30:
            mtf_adx_boost = True
    
    # === EFFECTIVE ADX CALCULATION ===
    effective_adx = adx
    
    # Rising ADX = trend is STARTING or STRENGTHENING
    if adx_trend == "rising":
        if adx >= 15:
            effective_adx = adx + 8  # rising from 15+ = strong start signal
        elif adx >= 10:
            effective_adx = adx + 5  # rising from very low = early but promising
    
    # Lower TFs show trend that 4H hasn't caught up to yet
    if mtf_adx_boost and effective_adx < ADX_TRENDING:
        effective_adx = max(effective_adx, ADX_TRENDING)  # treat as trending
    
    # ADX was high recently (avg_50 > 25) but currently low = pullback, not dead market
    if adx_avg_50 >= 25 and adx < ADX_TRENDING and adx_trend != "falling":
        effective_adx = max(effective_adx, ADX_TRENDING)  # was trending recently
    
    # === CLASSIFICATION ===
    # Momentum override: strong trend = lower confidence threshold
    if effective_adx >= 30 and confidence >= 55:
        return "full"  # Strong trend + decent confidence = go
    
    # Standard: confidence ≥ 65% + trending market
    if confidence >= CONFIDENCE_FULL and effective_adx >= ADX_TRENDING:
        return "full"
    
    # Lowered threshold when ALL timeframes agree on direction
    # Uses lightweight DI+/DI- check instead of full format_tf_summary
    if mtf_data and confidence >= 55 and effective_adx >= 15:
        direction = "LONG" if long_pct > short_pct else "SHORT"
        all_agree = True
        for tf_name in ["1H", "15m"]:
            tf_indic = mtf_data.get(tf_name, {})
            if tf_indic:
                di_plus = tf_indic.get("di_plus", 0)
                di_minus = tf_indic.get("di_minus", 0)
                ema7 = tf_indic.get("ema7", 0)
                ema25 = tf_indic.get("ema25", 0)
                # Check if TF agrees with direction
                if direction == "LONG":
                    if di_minus > di_plus and ema7 < ema25:
                        all_agree = False
                        break
                else:  # SHORT
                    if di_plus > di_minus and ema7 > ema25:
                        all_agree = False
                        break
        if all_agree:
            return "full"  # All TFs agree + decent confidence = go
    
    return "monitor"


def parse_confidence_from_ai(ai_text: str) -> tuple:
    """
    Extract LONG% and SHORT% from AI response text.
    Supports both English and Russian formats:
    - "LONG 62% / SHORT 38%"
    - "ЛОНГ: 62% / ШОРТ: 38%"
    - "ЛОНГ 62% / ШОРТ 38%"
    
    Returns: (long_pct, short_pct) or (50, 50) if not found
    """
    import re
    
    # English: "LONG XX% / SHORT YY%"
    pattern_en = r'LONG[:\s]*(\d+)%?\s*/\s*SHORT[:\s]*(\d+)%?'
    matches = re.findall(pattern_en, ai_text, re.IGNORECASE)
    
    if matches:
        last_match = matches[-1]
        return (float(last_match[0]), float(last_match[1]))
    
    # Russian: "ЛОНГ XX% / ШОРТ YY%"
    pattern_ru = r'ЛОНГ[:\s]*(\d+)%?\s*/\s*ШОРТ[:\s]*(\d+)%?'
    matches = re.findall(pattern_ru, ai_text, re.IGNORECASE)
    
    if matches:
        last_match = matches[-1]
        return (float(last_match[0]), float(last_match[1]))
    
    return (50, 50)


def calculate_atr_sl_tp(indicators: dict, direction: str, entry_price: float) -> dict:
    """
    Calculate SL/TP based on ATR with minimum floor and 2:1 R:R.
    
    SL = max(1.5 × ATR, 2% of price) — minimum 2% so SL is never too tight
    TP = 2 × SL distance — guaranteed 2:1 reward-to-risk ratio
    
    Returns dict with sl, tp, sl_pct, tp_pct, rr_ratio
    """
    # ATR14 from indicators
    atr = indicators.get("atr14_value", 0)
    if not atr or atr <= 0:
        # Fallback: estimate ATR as 2% of price
        atr = entry_price * 0.02
    
    # SL distance: 1.5 × ATR, but minimum 2% of price (never too tight)
    sl_from_atr = 1.5 * atr
    sl_min_floor = entry_price * 0.02  # 2% minimum
    sl_distance = max(sl_from_atr, sl_min_floor)
    
    # TP distance: 2 × SL distance (guaranteed 2:1 R:R)
    tp_distance = 2.0 * sl_distance
    
    if direction == "LONG":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    elif direction == "SHORT":
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    else:
        return {"sl": 0, "tp": 0, "sl_pct": 0, "tp_pct": 0}
    
    sl_pct = abs(sl_distance / entry_price) * 100
    tp_pct = abs(tp_distance / entry_price) * 100
    
    return {
        "sl": round(sl, 8),
        "tp": round(tp, 8),
        "sl_pct": round(sl_pct, 2),
        "tp_pct": round(tp_pct, 2),
        "rr_ratio": round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0
    }


async def get_current_1h_candle(session, symbol: str) -> dict:
    """
    Fetch the CURRENT OPEN 1h candle from Binance Futures.
    Returns: {"volume": float, "is_green": bool, "open": float, "close": float}

    Uses limit=1 to get the currently forming candle.
    Binance kline: [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
    quoteVolume (index 7) is in USDT.
    is_green = float(close) > float(open)
    """
    import aiohttp
    try:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval=1h&limit=1")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return {"volume": 0, "is_green": False, "open": 0, "close": 0}
            candles = await resp.json()
            if not candles:
                return {"volume": 0, "is_green": False, "open": 0, "close": 0}
            c = candles[0]
            open_price = float(c[1])
            close_price = float(c[4])
            quote_volume = float(c[7])
            return {
                "volume": quote_volume,
                "is_green": close_price > open_price,
                "open": open_price,
                "close": close_price
            }
    except Exception:
        return {"volume": 0, "is_green": False, "open": 0, "close": 0}


async def check_volume_pass(session, symbol: str) -> dict:
    """
    Check if coin passes volume filter.
    Returns: {"pass": bool, "vol_12h": float, "vol_1h": float, "candle_green": bool}

    Pass conditions (OR):
    1. 12h volume >= $2M
    2. Current open 1h candle: volume >= $170K AND green (close > open)
    """
    from config import SIGNAL_MIN_VOLUME_12H, SIGNAL_MIN_VOLUME_1H

    vol_12h = await get_volume_12h(session, symbol)
    candle_1h = await get_current_1h_candle(session, symbol)
    vol_1h = candle_1h["volume"]
    candle_green = candle_1h["is_green"]

    passed = False
    if vol_12h >= SIGNAL_MIN_VOLUME_12H:
        passed = True
    elif vol_1h >= SIGNAL_MIN_VOLUME_1H and candle_green:
        passed = True

    return {
        "pass": passed,
        "vol_12h": vol_12h,
        "vol_1h": vol_1h,
        "candle_green": candle_green
    }


async def get_volume_12h(session, symbol: str) -> float:
    """
    Calculate 12-hour quote volume from last 3 × 4H candles.
    
    Why not 24h ticker: it includes yesterday's pump/dump.
    Why 3 × 4H candles: exactly 12 hours of CURRENT activity.
    
    Returns: quote volume in USDT for last 12 hours.
    """
    import aiohttp
    try:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval=4h&limit=3")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return 0
            candles = await resp.json()
            if not candles:
                return 0
            # quoteVolume is index 7 in Binance kline array
            total = sum(float(c[7]) for c in candles)
            return total
    except Exception:
        return 0


# --- Volume Waitlist file operations ---
# Coins that failed volume check: recheck every 5 min in main loop.
# Clear at 23:58 UTC (00:00 = full rescan).

def load_volume_waitlist() -> list:
    try:
        if os.path.exists(VOLUME_WAITLIST_FILE):
            with open(VOLUME_WAITLIST_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ load_volume_waitlist: {e}")
    return []


def save_volume_waitlist(waitlist: list):
    try:
        os.makedirs("data", exist_ok=True)
        with open(VOLUME_WAITLIST_FILE, "w") as f:
            json.dump(waitlist, f, indent=2)
    except Exception as e:
        logging.error(f"❌ save_volume_waitlist: {e}")


def add_to_volume_waitlist(symbol, tf, alert_data, vol_12h, vol_1h, candle_green):
    """Add coin that failed volume check. Store breakout data so we can process it when volume arrives."""
    waitlist = load_volume_waitlist()
    key = f"{symbol}_{tf}"
    # No duplicates — replace existing
    waitlist = [w for w in waitlist if w.get("key") != key]
    waitlist.append({
        "key": key,
        "symbol": symbol,
        "tf": tf,
        "alert": alert_data,
        "vol_12h": vol_12h,
        "vol_1h": vol_1h,
        "candle_green": candle_green,
        "added_at": datetime.now(timezone.utc).isoformat()
    })
    save_volume_waitlist(waitlist)
    logging.info(f"📊 VOL WAITLIST added: {symbol} {tf} (12h=${vol_12h:,.0f}, 1h=${vol_1h:,.0f})")


def remove_from_volume_waitlist(key: str):
    """Remove coin from volume waitlist (passed volume check)."""
    waitlist = load_volume_waitlist()
    waitlist = [w for w in waitlist if w.get("key") != key]
    save_volume_waitlist(waitlist)
    logging.info(f"📊 VOL WAITLIST removed: {key}")


def clear_volume_waitlist():
    """Clear all — called at 23:58 UTC."""
    save_volume_waitlist([])
    logging.info("🧹 Volume waitlist cleared")


def get_volume_waitlist() -> list:
    """Return full waitlist for display and recheck."""
    return load_volume_waitlist()


# --- Monitor file operations ---

def load_monitors() -> list:
    """Load monitored signals from file."""
    try:
        if os.path.exists(MONITOR_FILE):
            with open(MONITOR_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ load_monitors: {e}")
    return []


def save_monitors(monitors: list):
    """Save monitored signals to file."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(MONITOR_FILE, "w") as f:
            json.dump(monitors, f, indent=2)
    except Exception as e:
        logging.error(f"❌ save_monitors: {e}")


def add_monitor(symbol: str, tf: str, direction: str, long_pct: float, 
                short_pct: float, entry_price: float, reason: str,
                recheck_sec: int = None,
                rsi_4h: float = 0, rsi_1h: float = 0, rsi_15m: float = 0,
                adx_4h: float = 0):
    """
    Add a signal to the monitor queue.
    
    Each signal has its OWN next_check_ts = now + 30 min.
    So breakouts at 15:00, 15:05, 15:10 get re-checked at 15:30, 15:35, 15:40.
    NOT all at once. This prevents API spikes when 100 breakouts happen.
    
    Monitors survive 03:00 trendline redraw — they live for 24h independently.
    
    reason types:
    - "high_rsi"     → monitor RSI only, wait for RSI drop of 15-20 points
    - "flat_market"  → monitor ADX only, wait for ADX > 20
    - "low_confidence" → monitor full scorecard, wait for improvement
    - "pump_15m"     → monitor 15m volatility, wait for calm down
    
    Saves initial RSI/ADX values so we can compare on recheck without AI.
    """
    monitors = load_monitors()
    
    # Don't duplicate same symbol+tf
    key = f"{symbol}_{tf}"
    monitors = [m for m in monitors if m.get("key") != key]
    
    now = time.time()
    monitors.append({
        "key": key,
        "symbol": symbol,
        "tf": tf,
        "direction": direction,
        "long_pct": long_pct,
        "short_pct": short_pct,
        "entry_price": entry_price,
        "reason": reason,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "added_ts": now,
        "next_check_ts": now + (recheck_sec or RECHECK_INTERVAL_SEC),
        "recheck_sec": recheck_sec or RECHECK_INTERVAL_SEC,
        "check_count": 0,
        # Snapshot values for lightweight recheck
        "initial_rsi_4h": rsi_4h,
        "initial_rsi_1h": rsi_1h,
        "initial_rsi_15m": rsi_15m,
        "initial_adx_4h": adx_4h,
    })
    
    save_monitors(monitors)
    logging.info(f"🔵 MONITOR added: {symbol} {tf} {direction} ({reason}, RSI4H={rsi_4h:.0f}, ADX={adx_4h:.0f}) — next check in {(recheck_sec or RECHECK_INTERVAL_SEC)//60}min")


def get_due_monitors() -> list:
    """
    Get monitors that are due for re-check RIGHT NOW.
    
    Each monitor has its own next_check_ts based on when IT was created.
    Example: breakouts at 15:00, 15:05, 15:10 → due at 15:30, 15:35, 15:40.
    They DON'T pile up — each has its own timer.
    
    Monitors live 24h, independent of 03:00 trendline redraw.
    """
    monitors = load_monitors()
    now = time.time()
    due = []
    
    for m in monitors:
        age_hours = (now - m["added_ts"]) / 3600
        
        if age_hours > MAX_MONITOR_HOURS:
            continue  # Expired (>24h)
        
        # Each monitor has its OWN next_check_ts
        if now >= m.get("next_check_ts", 0):
            due.append(m)
    
    return due


def update_monitor_checked(key: str):
    """
    Mark monitor as checked, schedule next check in 30 min from NOW.
    
    This keeps the staggered timing — each signal stays on its own schedule.
    """
    monitors = load_monitors()
    now = time.time()
    for m in monitors:
        if m["key"] == key:
            m["check_count"] = m.get("check_count", 0) + 1
            m["next_check_ts"] = now + m.get("recheck_sec", RECHECK_INTERVAL_SEC)
            m["last_checked_at"] = datetime.now(timezone.utc).isoformat()
    save_monitors(monitors)


def remove_monitor(key: str):
    """Remove a signal from monitoring (upgraded or expired)."""
    monitors = load_monitors()
    monitors = [m for m in monitors if m["key"] != key]
    save_monitors(monitors)
    logging.info(f"🟢 MONITOR removed: {key}")


def cleanup_expired_monitors():
    """Remove monitors older than MAX_MONITOR_HOURS."""
    monitors = load_monitors()
    now = time.time()
    active = [m for m in monitors if (now - m["added_ts"]) / 3600 <= MAX_MONITOR_HOURS]
    
    expired = len(monitors) - len(active)
    if expired > 0:
        logging.info(f"🕐 Cleaned {expired} expired monitors")
        save_monitors(active)


def get_1d_emergency_warnings(indicators_1d: dict) -> list:
    """
    Extract emergency-level warnings from 1D indicators.
    
    These are macro-level red flags that the 4H analysis can't see.
    Only fires on EXTREME conditions — not normal overbought/oversold.
    
    Checked ONLY for FULL signals (not monitor/info — they don't trade).
    Appended to AI prompt so the model can factor in macro risk.
    
    Returns: list of warning strings (empty = all clear)
    """
    warnings = []
    
    rsi = indicators_1d.get("rsi14", 50)
    adx = indicators_1d.get("adx", 25)
    bb_pctb = indicators_1d.get("bb_pctb", 0.5)
    rsi_div = indicators_1d.get("rsi_price_divergence", "none")
    obv_div = indicators_1d.get("obv_price_divergence", "none")
    
    # RSI > 85 on 1D = extreme overbought (very rare, usually precedes dump)
    if rsi > 85:
        warnings.append(f"⚠️ 1D RSI={rsi:.1f} EXTREME OVERBOUGHT — high reversal risk")
    
    # RSI < 15 on 1D = extreme oversold (capitulation)
    if rsi < 15:
        warnings.append(f"⚠️ 1D RSI={rsi:.1f} EXTREME OVERSOLD — capitulation zone")
    
    # ADX < 12 on 1D = dead market, no macro trend
    if adx < 12:
        warnings.append(f"⚠️ 1D ADX={adx:.1f} — no macro trend (flat market on daily)")
    
    # BB %B > 1.05 = price way above upper BB on daily (extended)
    if bb_pctb > 1.05:
        warnings.append(f"⚠️ 1D BB %B={bb_pctb:.2f} — price extended above daily Bollinger upper band")
    
    # BB %B < -0.05 = price way below lower BB on daily (capitulation)
    if bb_pctb < -0.05:
        warnings.append(f"⚠️ 1D BB %B={bb_pctb:.2f} — price below daily Bollinger lower band")
    
    # Bearish RSI divergence on 1D = major warning for longs
    if rsi_div == "bearish":
        warnings.append("⚠️ 1D BEARISH RSI DIVERGENCE — price rising but daily RSI falling")
    
    # Bullish RSI divergence on 1D = major warning for shorts
    if rsi_div == "bullish":
        warnings.append("⚠️ 1D BULLISH RSI DIVERGENCE — price falling but daily RSI rising")
    
    # OBV divergence on 1D
    if obv_div == "bearish":
        warnings.append("⚠️ 1D BEARISH OBV DIVERGENCE — volume not confirming price rise")
    
    return warnings


async def _quick_rsi_check(session, symbol: str, intervals: list = None) -> dict:
    """
    Lightweight RSI-only check for monitors.
    Fetches only 20 candles per TF (enough for RSI14) instead of 250.
    Returns: {"4H": rsi_value, "1H": rsi_value, "15m": rsi_value}
    
    Cost: 3 tiny kline requests vs 3×250 + full indicator calculation.
    """
    import aiohttp
    import asyncio

    if intervals is None:
        intervals = [("4h", "4H"), ("1h", "1H"), ("15m", "15m")]

    results = {}

    async def _fetch_rsi(interval_api, tf_label):
        try:
            url = (f"https://fapi.binance.com/fapi/v1/klines"
                   f"?symbol={symbol}&interval={interval_api}&limit=20")
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return
                candles = await resp.json()
                if not candles or len(candles) < 15:
                    return
                closes = [float(c[4]) for c in candles]
                # RSI(14) calculation
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [d if d > 0 else 0 for d in deltas]
                losses = [-d if d < 0 else 0 for d in deltas]
                avg_gain = sum(gains[:14]) / 14
                avg_loss = sum(losses[:14]) / 14
                for i in range(14, len(deltas)):
                    avg_gain = (avg_gain * 13 + gains[i]) / 14
                    avg_loss = (avg_loss * 13 + losses[i]) / 14
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                results[tf_label] = round(rsi, 1)
        except Exception:
            pass

    tasks = [_fetch_rsi(api, label) for api, label in intervals]
    await asyncio.gather(*tasks)
    return results


async def _quick_adx_check(session, symbol: str, interval: str = "4h") -> float:
    """
    Lightweight ADX-only check for monitors.
    Fetches 30 candles (enough for ADX14).
    Returns: ADX value or 0 on error.
    """
    try:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval={interval}&limit=30")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return 0
            candles = await resp.json()
            if not candles or len(candles) < 20:
                return 0
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            
            # True Range
            tr_list = [highs[0] - lows[0]]
            for i in range(1, len(candles)):
                tr_list.append(max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                ))
            
            # +DM, -DM
            plus_dm = [0.0]
            minus_dm = [0.0]
            for i in range(1, len(candles)):
                up = highs[i] - highs[i-1]
                down = lows[i-1] - lows[i]
                plus_dm.append(up if up > down and up > 0 else 0)
                minus_dm.append(down if down > up and down > 0 else 0)
            
            # Smoothed (Wilder, period 14)
            period = 14
            if len(tr_list) < period + 1:
                return 0
            atr = sum(tr_list[1:period+1]) / period
            plus_di_smooth = sum(plus_dm[1:period+1]) / period
            minus_di_smooth = sum(minus_dm[1:period+1]) / period
            
            dx_list = []
            for i in range(period + 1, len(tr_list)):
                atr = (atr * (period - 1) + tr_list[i]) / period
                plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm[i]) / period
                minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm[i]) / period
                
                if atr > 0:
                    plus_di = (plus_di_smooth / atr) * 100
                    minus_di = (minus_di_smooth / atr) * 100
                else:
                    plus_di = minus_di = 0
                
                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx_list.append(abs(plus_di - minus_di) / di_sum * 100)
            
            if len(dx_list) < period:
                return sum(dx_list) / len(dx_list) if dx_list else 0
            
            adx = sum(dx_list[:period]) / period
            for i in range(period, len(dx_list)):
                adx = (adx * (period - 1) + dx_list[i]) / period
            
            return round(adx, 1)
    except Exception:
        return 0


def _scorecard_looks_promising(indicators: dict, mtf_data: dict = None) -> tuple:
    """
    Quick indicator-only check: does the scorecard suggest improvement?
    
    Weighted assessment: 4H=50%, 1H=30%, 15m=20% (matches AI weights).
    NO AI call needed — just math on existing indicators.
    
    Returns: (promising: bool, weighted_bull_pct: float, adx: float, reason: str)
    """
    from core.indicators import format_tf_summary
    import re
    
    def _extract_bull_pct(indic, tf_label):
        summary = format_tf_summary(indic, tf_label)
        match = re.search(r'LONG\s*(\d+)%\s*/\s*SHORT\s*(\d+)%', summary)
        if match:
            return float(match.group(1))
        return 50
    
    # 4H (primary)
    bull_4h = _extract_bull_pct(indicators, "4H")
    adx = indicators.get("adx", 0)
    adx_trend = indicators.get("adx_trend", "stable")
    
    # 1H + 15m from MTF data
    bull_1h = 50
    bull_15m = 50
    if mtf_data:
        if "1H" in mtf_data:
            bull_1h = _extract_bull_pct(mtf_data["1H"], "1H")
        if "15m" in mtf_data:
            bull_15m = _extract_bull_pct(mtf_data["15m"], "15m")
    
    # Weighted bull %: 4H=50%, 1H=30%, 15m=20%
    bull_pct = bull_4h * 0.50 + bull_1h * 0.30 + bull_15m * 0.20
    
    # Effective ADX with rising bonus
    effective_adx = adx + 5 if (adx_trend == "rising" and adx >= 20) else adx
    
    # Is it worth calling AI?
    if effective_adx >= 30 and bull_pct >= 55:
        return (True, bull_pct, adx, f"strong_trend_adx{adx:.0f}")
    if bull_pct >= 60 and effective_adx >= 20:
        return (True, bull_pct, adx, f"improving_conf{bull_pct:.0f}")
    if bull_pct >= 65:
        return (True, bull_pct, adx, f"high_conf{bull_pct:.0f}")
    
    return (False, bull_pct, adx, "no_improvement")


async def monitor_recheck_loop(session):
    """
    Background loop: every 60s checks if any monitors are due for re-analysis.
    
    SMART TWO-PHASE CHECK (saves API + bandwidth):
    
    Phase 1 — LIGHTWEIGHT (reason-specific, ~20 candles per TF):
      - high_rsi:       fetch RSI only (20 candles × 3 TFs). Check if RSI dropped 15-20 pts.
      - flat_market:    fetch ADX only (30 candles × 1 TF). Check if ADX > 20.
      - pump_15m:       fetch 15m close only (5 candles). Check if volatility calmed.
      - low_confidence: fetch RSI + ADX lightweight. Check basic improvement.
      If condition NOT improved → update timer, move on. NO full indicators, NO AI.
    
    Phase 2 — FULL (only when Phase 1 passes):
      Load full 250 candles, all indicators, funding, positioning, SMC (500 candles).
      Call AI for verdict. If FULL → send signal with chart.
    
    This means 50 monitors = 50 tiny fetches (3-5 KB each) but only 2-5 full loads + AI calls.
    """
    import asyncio
    import aiohttp
    import pandas as pd

    # Delayed imports to avoid circular dependency
    from core.binance_api import fetch_klines, fetch_funding_history, fetch_market_positioning
    from core.indicators import calculate_binance_indicators
    from agent.analyzer import ask_ai_analysis
    from core.chart_drawer import send_breakout_notification
    from core.geometry_scanner import find_trend_line
    from core.chart_drawer import draw_scan_chart, draw_simple_chart
    from config import (BOT_TOKEN, GROUP_CHAT_ID, MONITOR_GROUP_CHAT_ID,
                         OPENROUTER_API_KEY_MONITOR, OPENROUTER_MODEL_MONITOR,
                         add_breakout_entry, upgrade_breakout_entry, add_monitor_breakout_entry, parse_ai_trade_params)

    logging.info("🔄 Monitor recheck loop started (smart two-phase: lightweight → full + AI)")

    while True:
        try:
            await asyncio.sleep(60)  # check every minute
            cleanup_expired_monitors()
            due = get_due_monitors()
            if not due:
                continue

            logging.info(f"🔄 Monitor: {len(due)} due for re-check")
            # No AI calls in monitor — auto-trade mode

            for m in due:
                try:
                    sym = m["symbol"]
                    tf = m["tf"]
                    interval = '1d' if tf == "1D" else '4h'
                    reason = m.get("reason", "low_confidence")

                    # ═══════════════════════════════════════════════════════
                    # PHASE 1: LIGHTWEIGHT CHECK (reason-specific, no AI)
                    # ═══════════════════════════════════════════════════════
                    phase1_pass = False

                    if reason == "high_rsi":
                        # Only fetch RSI (20 candles × 3 TFs)
                        rsi_now = await _quick_rsi_check(session, sym)
                        rsi_4h_now = rsi_now.get("4H", 99)
                        rsi_1h_now = rsi_now.get("1H", 99)
                        initial_rsi = m.get("initial_rsi_4h", 99)
                        rsi_drop = initial_rsi - rsi_4h_now

                        if rsi_drop >= 10:
                            phase1_pass = True
                            logging.info(f"🔵 MONITOR RSI improved: {sym} RSI {initial_rsi:.0f}→{rsi_4h_now:.0f} (dropped {rsi_drop:.0f} pts) → Phase 2")
                        else:
                            logging.info(f"🔵 MONITOR RSI still high: {sym} RSI {initial_rsi:.0f}→{rsi_4h_now:.0f} (need -{10-rsi_drop:.0f} more)")

                    elif reason == "flat_market":
                        # Only fetch ADX (30 candles × 1 TF)
                        adx_now = await _quick_adx_check(session, sym, interval)
                        initial_adx = m.get("initial_adx_4h", 0)

                        if adx_now >= ADX_TRENDING:
                            phase1_pass = True
                            logging.info(f"🔵 MONITOR ADX improved: {sym} ADX {initial_adx:.0f}→{adx_now:.0f} (≥{ADX_TRENDING}) → Phase 2")
                        else:
                            logging.info(f"🔵 MONITOR ADX still flat: {sym} ADX {initial_adx:.0f}→{adx_now:.0f} (need ≥{ADX_TRENDING})")

                    elif reason == "pump_15m":
                        # Check if 15m volatility calmed down
                        try:
                            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=15m&limit=5"
                            async with session.get(url, timeout=10) as resp:
                                if resp.status == 200:
                                    candles = await resp.json()
                                    if candles and len(candles) >= 2:
                                        last_open = float(candles[-1][1])
                                        last_close = float(candles[-1][4])
                                        change_pct = abs((last_close - last_open) / last_open * 100) if last_open > 0 else 0
                                        if change_pct < 5:
                                            phase1_pass = True
                                            logging.info(f"🔵 MONITOR 15m calmed: {sym} candle {change_pct:.1f}% (<5%) → Phase 2")
                                        else:
                                            logging.info(f"🔵 MONITOR 15m still volatile: {sym} candle {change_pct:.1f}%")
                        except Exception:
                            pass

                    else:  # "low_confidence" or unknown
                        # Lightweight: check RSI + ADX together
                        rsi_now = await _quick_rsi_check(session, sym)
                        adx_now = await _quick_adx_check(session, sym, interval)
                        rsi_4h_now = rsi_now.get("4H", 50)

                        # Pass if: RSI not extreme AND ADX shows trend
                        if 30 < rsi_4h_now < 75 and adx_now >= 18:
                            phase1_pass = True
                            logging.info(f"🔵 MONITOR conditions improved: {sym} RSI={rsi_4h_now:.0f} ADX={adx_now:.0f} → Phase 2")
                        else:
                            logging.info(f"🔵 MONITOR still weak: {sym} RSI={rsi_4h_now:.0f} ADX={adx_now:.0f}")

                    if not phase1_pass:
                        update_monitor_checked(m["key"])
                        await asyncio.sleep(0.5)
                        continue

                    # ═══════════════════════════════════════════════════════
                    # PHASE 2: AUTO-TRADE (no AI — use stored direction)
                    # ═══════════════════════════════════════════════════════
                    # Direction comes from the original AI analysis when signal entered monitor
                    direction = m.get("direction", "LONG")
                    conf = max(m.get("long_pct", 50), m.get("short_pct", 50))

                    # Fetch current price (lightweight — just 1 candle)
                    try:
                        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit=5"
                        async with session.get(url, timeout=10) as resp:
                            if resp.status == 200:
                                candles = await resp.json()
                                current_price = float(candles[-1][4]) if candles else m.get("entry_price", 0)
                            else:
                                current_price = m.get("entry_price", 0)
                    except Exception:
                        current_price = m.get("entry_price", 0)

                    if current_price <= 0:
                        update_monitor_checked(m["key"])
                        continue

                    # Fetch full indicators for ATR calculation (250 candles)
                    try:
                        raw = await fetch_klines(session, sym, interval, 250)
                        if raw:
                            df = pd.DataFrame(raw)
                            indicators, _ = calculate_binance_indicators(df, tf)
                        else:
                            indicators = {}
                    except Exception:
                        indicators = {}

                    # Calculate SL/TP from ATR (SL max 10%, TP = 2:1)
                    atr_params = calculate_atr_sl_tp(indicators, direction, current_price)
                    sl = atr_params["sl"]
                    tp = atr_params["tp"]
                    sl_pct = atr_params["sl_pct"]

                    # Cap SL at 10% max
                    if sl_pct > 10:
                        sl_distance = current_price * 0.10
                        if direction == "LONG":
                            sl = current_price - sl_distance
                            tp = current_price + sl_distance * 2
                        else:
                            sl = current_price + sl_distance
                            tp = current_price - sl_distance * 2
                        sl_pct = 10.0

                    tp_pct = abs(tp - current_price) / current_price * 100

                    remove_monitor(m["key"])
                    logging.info(f"🟢 MONITOR AUTO-TRADE: {sym} {tf} {direction} {conf:.0f}% | Entry: {current_price} | SL: {sl} ({sl_pct:.1f}%) | TP: {tp} ({tp_pct:.1f}%)")

                    # Send simple text push to monitor group
                    _upgrade_chat = MONITOR_GROUP_CHAT_ID or GROUP_CHAT_ID
                    short_sym = sym.replace("USDT", "")
                    push_text = (
                        f"🟢 MONITOR → TRADE\n"
                        f"${short_sym} | {tf} | {direction} {conf:.0f}%\n"
                        f"💰 Entry: {current_price}\n"
                        f"🚫 SL: {sl} ({sl_pct:.1f}%)\n"
                        f"🎯 TP: {tp} ({tp_pct:.1f}%)\n"
                        f"📊 Was: {reason} | Original: {m.get('entry_price', 0)}"
                    )
                    try:
                        tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                        await session.post(tg_url, json={
                            "chat_id": _upgrade_chat, "text": push_text
                        }, timeout=10)
                    except Exception as e:
                        logging.error(f"❌ Monitor trade push: {e}")

                    # Add to MONITOR bank (bankm) — NOT to signals close
                    try:
                        add_monitor_breakout_entry(sym, tf, m.get("entry_price", 0), current_price,
                                          "monitor_auto_trade",
                                          ai_direction=direction,
                                          ai_entry=current_price,
                                          ai_sl=sl,
                                          ai_tp=tp,
                                          ai_leverage=FIXED_LEVERAGE,
                                          ai_deposit_pct=FIXED_DEPOSIT_PCT)
                        logging.info(f"✅ Monitor bank entry: {sym} {direction} @ {current_price}")
                    except Exception as e:
                        logging.error(f"❌ Monitor bank entry failed for {sym}: {e}")

                    await asyncio.sleep(2)  # small cooldown between checks (no AI needed)

                except Exception as e:
                    logging.error(f"❌ Monitor {m.get('symbol','?')}: {e}")
                    update_monitor_checked(m["key"])

            if len(due) > 0:
                logging.info(f"🔄 Monitor cycle done: {len(due)} checked (no AI — auto-trade mode)")

        except Exception as e:
            logging.error(f"❌ Monitor loop error: {e}")
            await asyncio.sleep(60)
