"""
Signal Pipeline — AI signal system with trailing stops.

AI verdict is the sole authority:
- LONG/SHORT → trade entry (signals bank)
- SKIP → no trade
- Confidence < 62.5% → forced SKIP


Trailing stop (3%) replaces fixed TP. Initial SL from AI (capped 10%).
"""

import json
import os
import logging
import time
from datetime import datetime, timezone

VOLUME_WAITLIST_FILE = "data/volume_waitlist.json"

# --- Fixed risk params ---
FIXED_LEVERAGE = 1
FIXED_DEPOSIT_PCT = 2  # 2% of bank per trade


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
    
    # English: "LONG XX.X% / SHORT YY.Y%" (supports decimals)
    pattern_en = r'LONG[:\s]*(\d+(?:\.\d+)?)%?\s*/\s*SHORT[:\s]*(\d+(?:\.\d+)?)%?'
    matches = re.findall(pattern_en, ai_text, re.IGNORECASE)
    
    if matches:
        last_match = matches[-1]
        return (float(last_match[0]), float(last_match[1]))
    
    # Russian: "ЛОНГ XX.X% / ШОРТ YY.Y%" (supports decimals)
    pattern_ru = r'ЛОНГ[:\s]*(\d+(?:\.\d+)?)%?\s*/\s*ШОРТ[:\s]*(\d+(?:\.\d+)?)%?'
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
    atr = indicators.get("atr14_value", 0)
    if not atr or atr <= 0:
        atr = entry_price * 0.02
    
    sl_from_atr = 1.5 * atr
    sl_min_floor = entry_price * 0.02
    sl_distance = max(sl_from_atr, sl_min_floor)
    
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


async def check_btc_shield(session, ema_period: int = 25) -> dict:
    """
    Check BTC EMA25 on 15m — is the last closed candle fully below/above EMA?

    Returns: {
        "bearish": bool,   # candle fully below EMA → close losing LONGs
        "bullish": bool,   # candle fully above EMA → close losing SHORTs
        "btc_price": float,
        "ema_value": float,
        "candle_high": float,
        "candle_low": float,
    }
    """
    try:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol=BTCUSDT&interval=15m&limit={ema_period + 5}")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return {"bearish": False, "bullish": False}
            candles = await resp.json()
            if not candles or len(candles) < ema_period + 2:
                return {"bearish": False, "bullish": False}

        closes = [float(c[4]) for c in candles]
        ema_values = _compute_ema(closes, ema_period)

        # Last CLOSED candle = candles[-2] (candles[-1] is still open)
        last_closed_high = float(candles[-2][2])
        last_closed_low = float(candles[-2][3])
        last_closed_close = float(candles[-2][4])
        last_ema = ema_values[-2]

        # "Fully below EMA" = candle high < EMA → bearish
        is_bearish = last_closed_high < last_ema
        # "Fully above EMA" = candle low > EMA → bullish
        is_bullish = last_closed_low > last_ema

        return {
            "bearish": is_bearish,
            "bullish": is_bullish,
            "btc_price": last_closed_close,
            "ema_value": round(last_ema, 2),
            "candle_high": last_closed_high,
            "candle_low": last_closed_low,
        }
    except Exception as e:
        logging.error(f"❌ check_btc_shield: {e}")
        return {"bearish": False, "bullish": False}


def check_fixed_atr_sl_tp(candles: list, direction: str, entry_price: float,
                          atr_value: float, sl_atr: float, tp_atr: float) -> tuple:
    """
    Fixed SL/TP based on ATR multipliers.
    SL = entry ± (atr_value × sl_atr), TP = entry ± (atr_value × tp_atr)
    Returns: (status, close_price, peak_price, sl_used)
    """
    if not candles or not direction or entry_price <= 0 or not atr_value or atr_value <= 0:
        return ("open", entry_price, entry_price, 0)

    sl_dist = atr_value * sl_atr
    tp_dist = atr_value * tp_atr

    if direction == "LONG":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    elif direction == "SHORT":
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
    else:
        return ("open", entry_price, entry_price, 0)

    return check_fixed_sl_tp_from_candles(candles, direction, entry_price, sl, tp)


def check_candle_trailing(candles: list, direction: str, entry_price: float,
                          atr_value: float, initial_sl_atr: float, trail_atr: float,
                          anchor: str = "low-1", activation_candles: int = 3) -> tuple:
    """
    Candle-based trailing stop.

    Phase 1 (first N candles): initial SL = entry ± (atr × initial_sl_atr)
    Phase 2 (after N candles): move SL to candle high/low ± (atr × trail_atr)
        anchor: high-1/low-1/high-2/low-2 — which closed candle level to track.

    For LONG: SL under the candle low (or high) minus trail buffer.
    For SHORT: SL above the candle high (or low) plus trail buffer.

    Returns: (status, close_price, peak_price, final_sl)
    """
    if not candles or not direction or entry_price <= 0:
        return ("open", entry_price, entry_price, 0)

    if not atr_value or atr_value <= 0:
        atr_value = entry_price * 0.03  # fallback 3%

    trail_buffer = atr_value * trail_atr
    init_sl_dist = atr_value * initial_sl_atr

    # Parse anchor: "high-1" → field=high, offset=1
    anchor_parts = anchor.split("-")
    anchor_field = anchor_parts[0]  # "high" or "low"
    anchor_offset = int(anchor_parts[1]) if len(anchor_parts) > 1 else 1  # 1 or 2

    # Candle indices: [2]=high, [3]=low
    field_idx = 2 if anchor_field == "high" else 3

    if direction == "LONG":
        initial_sl = entry_price - init_sl_dist
        current_sl = initial_sl
        peak = entry_price

        # Store closed candle levels for lookback
        closed_levels = []

        for i, candle in enumerate(candles):
            high = float(candle[2])
            low = float(candle[3])

            if high > peak:
                peak = high

            # Check SL hit
            if low <= current_sl:
                status = "sl" if i < activation_candles else "trail"
                return (status, current_sl, peak, current_sl)

            # Record this candle's level for next candle's reference
            closed_levels.append(float(candle[field_idx]))

            # After activation period, update trailing SL
            if i >= activation_candles and len(closed_levels) >= anchor_offset + 1:
                # -1 offset means last closed = closed_levels[-2] (current candle is [-1] but still open-ish)
                ref_idx = -(anchor_offset + 1)
                if abs(ref_idx) <= len(closed_levels):
                    ref_level = closed_levels[ref_idx]
                    new_sl = ref_level - trail_buffer
                    if new_sl > current_sl:
                        current_sl = new_sl

        last_close = float(candles[-1][4])
        return ("open", last_close, peak, current_sl)

    elif direction == "SHORT":
        initial_sl = entry_price + init_sl_dist
        current_sl = initial_sl
        trough = entry_price

        closed_levels = []

        for i, candle in enumerate(candles):
            high = float(candle[2])
            low = float(candle[3])

            if low < trough:
                trough = low

            if high >= current_sl:
                status = "sl" if i < activation_candles else "trail"
                return (status, current_sl, trough, current_sl)

            closed_levels.append(float(candle[field_idx]))

            if i >= activation_candles and len(closed_levels) >= anchor_offset + 1:
                ref_idx = -(anchor_offset + 1)
                if abs(ref_idx) <= len(closed_levels):
                    ref_level = closed_levels[ref_idx]
                    new_sl = ref_level + trail_buffer
                    if new_sl < current_sl:
                        current_sl = new_sl

        last_close = float(candles[-1][4])
        return ("open", last_close, trough, current_sl)

    return ("open", entry_price, entry_price, 0)


def check_ema_sl(candles_5m: list, candles_ema_tf: list, direction: str,
                 entry_price: float, atr_value: float, initial_sl_atr: float,
                 ema_period: int = 25) -> tuple:
    """
    EMA-based stop-loss.

    Phase 1: Initial SL = entry ± (atr × initial_sl_atr)
    Phase 2: Check EMA on the ema_tf candles.
        LONG close: if prev_candle_close < EMA AND prev_prev_candle_close >= EMA → trend broken
        SHORT close: if prev_candle_close > EMA AND prev_prev_candle_close <= EMA → trend broken

    Uses 5m candles for SL check, ema_tf candles for EMA signal.

    Returns: (status, close_price, peak_price, sl_used)
    """
    if not candles_5m or not direction or entry_price <= 0:
        return ("open", entry_price, entry_price, 0)

    if not atr_value or atr_value <= 0:
        atr_value = entry_price * 0.03

    init_sl_dist = atr_value * initial_sl_atr

    # Calculate EMA on the ema_tf candles
    if not candles_ema_tf or len(candles_ema_tf) < ema_period + 3:
        # Not enough data for EMA — fall back to initial SL only
        if direction == "LONG":
            sl = entry_price - init_sl_dist
        else:
            sl = entry_price + init_sl_dist
        return check_fixed_sl_tp_from_candles(candles_5m, direction, entry_price, sl, 0)

    # Compute EMA from close prices
    closes = [float(c[4]) for c in candles_ema_tf]
    ema_values = _compute_ema(closes, ema_period)

    # Check last 3 ema_tf candles: current(-1), prev(-2), prev_prev(-3)
    ema_signal_close = False
    if len(ema_values) >= 3:
        prev_close = closes[-2]
        prev_prev_close = closes[-3]
        prev_ema = ema_values[-2]
        prev_prev_ema = ema_values[-3]

        if direction == "LONG":
            # Trend broken: prev candle went below EMA while prev_prev was above
            if prev_close < prev_ema and prev_prev_close >= prev_prev_ema:
                ema_signal_close = True
        elif direction == "SHORT":
            # Trend broken: prev candle went above EMA while prev_prev was below
            if prev_close > prev_ema and prev_prev_close <= prev_prev_ema:
                ema_signal_close = True

    # Walk 5m candles for initial SL check
    if direction == "LONG":
        sl = entry_price - init_sl_dist
        peak = entry_price
        for candle in candles_5m:
            high = float(candle[2])
            low = float(candle[3])
            if high > peak:
                peak = high
            if low <= sl:
                return ("sl", sl, peak, sl)

        last_close = float(candles_5m[-1][4])
        if ema_signal_close:
            return ("ema", last_close, peak, sl)
        return ("open", last_close, peak, sl)

    elif direction == "SHORT":
        sl = entry_price + init_sl_dist
        trough = entry_price
        for candle in candles_5m:
            high = float(candle[2])
            low = float(candle[3])
            if low < trough:
                trough = low
            if high >= sl:
                return ("sl", sl, trough, sl)

        last_close = float(candles_5m[-1][4])
        if ema_signal_close:
            return ("ema", last_close, trough, sl)
        return ("open", last_close, trough, sl)

    return ("open", entry_price, entry_price, 0)


def _compute_ema(values: list, period: int) -> list:
    """Compute EMA over a list of floats. Returns list same length as input."""
    if not values or period <= 0:
        return values
    ema = [values[0]]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        ema.append(values[i] * k + ema[-1] * (1 - k))
    return ema


def check_fixed_sl_tp_from_candles(candles: list, direction: str, entry_price: float,
                                    initial_sl: float, tp_price: float) -> tuple:
    """
    Walk 5m candles and check FIXED SL and TP levels (no trailing).
    
    - If price hits SL → loss
    - If price hits TP → win  
    - If neither hit → still open at last close
    
    Returns: (status, close_price, peak_price, sl_used)
        status: "sl" (SL hit), "tp" (TP hit), "open" (still open)
    """
    if not candles or not direction or entry_price <= 0:
        return ("open", entry_price, entry_price, initial_sl)
    
    if direction == "LONG":
        peak = entry_price
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            
            if high > peak:
                peak = high
            
            # Check SL first (worst case)
            if initial_sl and initial_sl > 0 and low <= initial_sl:
                return ("sl", initial_sl, peak, initial_sl)
            
            # Check TP
            if tp_price and tp_price > 0 and high >= tp_price:
                return ("tp", tp_price, peak, initial_sl)
        
        last_close = float(candles[-1][4])
        return ("open", last_close, peak, initial_sl)
    
    elif direction == "SHORT":
        trough = entry_price
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            
            if low < trough:
                trough = low
            
            # Check SL first
            if initial_sl and initial_sl > 0 and high >= initial_sl:
                return ("sl", initial_sl, trough, initial_sl)
            
            # Check TP
            if tp_price and tp_price > 0 and low <= tp_price:
                return ("tp", tp_price, trough, initial_sl)
        
        last_close = float(candles[-1][4])
        return ("open", last_close, trough, initial_sl)
    
    return ("open", entry_price, entry_price, initial_sl)


def check_trailing_stop_from_candles(candles: list, direction: str, entry_price: float,
                                      initial_sl: float, trail_pct: float = 3.0) -> tuple:
    """
    Walk 5m candles and apply trailing stop with breakeven logic.
    
    Phase 1: Only initial SL protects (no trailing yet)
    Phase 2: Price moves BREAKEVEN_TRIGGER_PCT% in our favor →
             SL jumps to entry + BREAKEVEN_PROFIT_PCT% (guaranteed profit)
    Phase 3: Trailing stop at trail_pct% from peak (never below breakeven SL)
    
    Returns: (status, close_price, peak_price, final_trailing_sl)
        status: "sl" (initial SL hit), "trail" (trailing SL hit), "open" (still open)
    """
    from config import BREAKEVEN_TRIGGER_PCT, BREAKEVEN_PROFIT_PCT, BREAKEVEN_TIME_TRIGGER_PCT, BREAKEVEN_TIME_CANDLES
    
    if not candles or not direction or entry_price <= 0:
        return ("open", entry_price, entry_price, initial_sl)
    
    trail_mult = trail_pct / 100.0
    be_trigger_mult = BREAKEVEN_TRIGGER_PCT / 100.0
    be_profit_mult = BREAKEVEN_PROFIT_PCT / 100.0
    be_time_trigger_mult = BREAKEVEN_TIME_TRIGGER_PCT / 100.0
    
    if direction == "LONG":
        peak = entry_price
        breakeven_activated = False
        breakeven_sl = entry_price * (1 + be_profit_mult)  # entry + 0.5%
        candle_count = 0
        
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            candle_count += 1
            
            # Update peak
            if high > peak:
                peak = high
            
            # Check if breakeven should activate
            # Option 1: price reached +5% from entry (instant)
            # Option 2: price is +3% AND 20+ candles passed (time-based)
            if not breakeven_activated:
                if peak >= entry_price * (1 + be_trigger_mult):
                    breakeven_activated = True
                elif (candle_count >= BREAKEVEN_TIME_CANDLES and
                      peak >= entry_price * (1 + be_time_trigger_mult)):
                    breakeven_activated = True
            
            if not breakeven_activated:
                # Phase 1: only initial SL protects
                if initial_sl and initial_sl > 0 and low <= initial_sl:
                    return ("sl", initial_sl, peak, initial_sl)
            else:
                # Phase 2/3: trailing stop, floor = breakeven SL
                trailing_sl = peak * (1 - trail_mult)
                # Never below breakeven profit level
                effective_sl = max(trailing_sl, breakeven_sl)
                # Also never below initial SL (shouldn't happen but safety)
                if initial_sl and initial_sl > 0:
                    effective_sl = max(effective_sl, initial_sl)
                
                if low <= effective_sl:
                    return ("trail", effective_sl, peak, effective_sl)
        
        # Still open
        last_close = float(candles[-1][4])
        if breakeven_activated:
            trailing_sl = peak * (1 - trail_mult)
            effective_sl = max(trailing_sl, breakeven_sl)
            if initial_sl and initial_sl > 0:
                effective_sl = max(effective_sl, initial_sl)
        else:
            effective_sl = initial_sl if (initial_sl and initial_sl > 0) else entry_price * (1 - trail_mult)
        return ("open", last_close, peak, effective_sl)
    
    elif direction == "SHORT":
        trough = entry_price
        breakeven_activated = False
        breakeven_sl = entry_price * (1 - be_profit_mult)  # entry - 0.5%
        candle_count = 0
        
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            candle_count += 1
            
            # Update trough
            if low < trough:
                trough = low
            
            # Check if breakeven should activate
            if not breakeven_activated:
                if trough <= entry_price * (1 - be_trigger_mult):
                    breakeven_activated = True
                elif (candle_count >= BREAKEVEN_TIME_CANDLES and
                      trough <= entry_price * (1 - be_time_trigger_mult)):
                    breakeven_activated = True
            
            if not breakeven_activated:
                # Phase 1: only initial SL protects
                if initial_sl and initial_sl > 0 and high >= initial_sl:
                    return ("sl", initial_sl, trough, initial_sl)
            else:
                # Phase 2/3: trailing stop, ceiling = breakeven SL
                trailing_sl = trough * (1 + trail_mult)
                # Never above breakeven profit level
                effective_sl = min(trailing_sl, breakeven_sl)
                # Also never above initial SL
                if initial_sl and initial_sl > 0:
                    effective_sl = min(effective_sl, initial_sl)
                
                if high >= effective_sl:
                    return ("trail", effective_sl, trough, effective_sl)
        
        # Still open
        last_close = float(candles[-1][4])
        if breakeven_activated:
            trailing_sl = trough * (1 + trail_mult)
            effective_sl = min(trailing_sl, breakeven_sl)
            if initial_sl and initial_sl > 0:
                effective_sl = min(effective_sl, initial_sl)
        else:
            effective_sl = initial_sl if (initial_sl and initial_sl > 0) else entry_price * (1 + trail_mult)
        return ("open", last_close, trough, effective_sl)
    
    return ("open", entry_price, entry_price, initial_sl)


async def get_current_1h_candle(session, symbol: str) -> dict:
    """
    Fetch the CURRENT OPEN 1h candle from Binance Futures.
    Returns: {"volume": float, "is_green": bool, "open": float, "close": float}
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
        "candle_green": candle_green,
        "current_price": candle_1h["close"],
    }


async def get_volume_12h(session, symbol: str) -> float:
    """
    Calculate 12-hour quote volume from last 3 × 4H candles.
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


def get_1d_emergency_warnings(indicators_1d: dict) -> list:
    """
    Extract emergency-level warnings from 1D indicators.
    These are macro-level red flags that the 4H analysis can't see.
    Only fires on EXTREME conditions.
    Returns: list of warning strings (empty = all clear)
    """
    warnings = []
    
    rsi = indicators_1d.get("rsi14", 50)
    adx = indicators_1d.get("adx", 25)
    bb_pctb = indicators_1d.get("bb_pctb", 0.5)
    rsi_div = indicators_1d.get("rsi_price_divergence", "none")
    obv_div = indicators_1d.get("obv_price_divergence", "none")
    
    if rsi > 85:
        warnings.append(f"⚠️ 1D RSI={rsi:.1f} EXTREME OVERBOUGHT — high reversal risk")
    if rsi < 15:
        warnings.append(f"⚠️ 1D RSI={rsi:.1f} EXTREME OVERSOLD — capitulation zone")
    if adx < 12:
        warnings.append(f"⚠️ 1D ADX={adx:.1f} — no macro trend (flat market on daily)")
    if bb_pctb > 1.05:
        warnings.append(f"⚠️ 1D BB %B={bb_pctb:.2f} — price extended above daily Bollinger upper band")
    if bb_pctb < -0.05:
        warnings.append(f"⚠️ 1D BB %B={bb_pctb:.2f} — price below daily Bollinger lower band")
    if rsi_div == "bearish":
        warnings.append("⚠️ 1D BEARISH RSI DIVERGENCE — price rising but daily RSI falling")
    if rsi_div == "bullish":
        warnings.append("⚠️ 1D BULLISH RSI DIVERGENCE — price falling but daily RSI rising")
    if obv_div == "bearish":
        warnings.append("⚠️ 1D BEARISH OBV DIVERGENCE — volume not confirming price rise")
    
    return warnings
