"""
Signal Pipeline — AI signal system with trailing stops.

AI verdict is the sole authority:
- LONG/SHORT → trade entry (signals bank)
- SKIP → no trade
- Confidence < 62.5% → forced SKIP

ML predictions tracked in separate ML bank.
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


def calculate_ml_sl(indicators: dict, direction: str, entry_price: float) -> float:
    """
    Calculate ML SL based on ATR, clamped between 5% and 10%.
    
    SL = 1.5 × ATR, but min 5%, max 10% from entry.
    Returns: SL price level.
    """
    from config import ML_SL_ATR_MULT, ML_SL_MIN_PCT, ML_SL_MAX_PCT
    
    atr = indicators.get("atr14_value", 0)
    if not atr or atr <= 0:
        atr = entry_price * 0.05  # fallback: 5%
    
    sl_distance = ML_SL_ATR_MULT * atr
    
    # Clamp between 5% and 10%
    sl_min = entry_price * (ML_SL_MIN_PCT / 100)
    sl_max = entry_price * (ML_SL_MAX_PCT / 100)
    sl_distance = max(sl_min, min(sl_distance, sl_max))
    
    if direction == "LONG":
        return round(entry_price - sl_distance, 8)
    elif direction == "SHORT":
        return round(entry_price + sl_distance, 8)
    else:
        return 0


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
        "candle_green": candle_green
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
