"""
Signal Pipeline — two-tier signal system with hourly re-monitoring.

FULL SIGNAL (🟢): confidence ≥ 65% AND ADX > 20 → trade entry
MONITOR   (🔵): confidence 50-64% OR ADX < 20  → hourly re-check
INFO ONLY (⚫): confidence < 50%                → show, don't trade

Monitor loop re-analyzes every hour. If conditions improve → upgrade to FULL.
Expires after 24h without upgrade.
"""

import json
import os
import logging
import time
from datetime import datetime, timezone

MONITOR_FILE = "data/signal_monitor.json"
MAX_MONITOR_HOURS = 24  # expire after 24h — independent of 03:00 trendline redraw
RECHECK_INTERVAL_SEC = 1800  # 30 minutes from each signal's OWN detection time

# --- Confidence threshold ---
CONFIDENCE_FULL = 65      # % to issue full signal
CONFIDENCE_MONITOR = 50   # % to put on watch
ADX_TRENDING = 20         # minimum ADX for trending market

# --- Fixed risk params ---
FIXED_LEVERAGE = 1
FIXED_DEPOSIT_PCT = 2  # 2% of bank per trade


def classify_signal(long_pct: float, short_pct: float, adx: float) -> str:
    """
    Classify signal tier based on confidence and market regime.
    
    Returns: "full", "monitor", or "info"
    """
    confidence = max(long_pct, short_pct)
    
    if confidence >= CONFIDENCE_FULL and adx >= ADX_TRENDING:
        return "full"
    elif confidence >= CONFIDENCE_MONITOR:
        return "monitor"
    else:
        return "info"


def parse_confidence_from_ai(ai_text: str) -> tuple:
    """
    Extract LONG% and SHORT% from AI response text.
    Looks for pattern like "LONG 62% / SHORT 38%" or "Overall: LONG 55% / SHORT 45%"
    
    Returns: (long_pct, short_pct) or (50, 50) if not found
    """
    import re
    
    # Pattern: "LONG XX% / SHORT YY%" or "LONG XX / SHORT YY"
    pattern = r'LONG\s*(\d+)%?\s*/\s*SHORT\s*(\d+)%?'
    matches = re.findall(pattern, ai_text, re.IGNORECASE)
    
    if matches:
        # Take the LAST match (usually the "Overall" line)
        last_match = matches[-1]
        long_pct = float(last_match[0])
        short_pct = float(last_match[1])
        return (long_pct, short_pct)
    
    return (50, 50)


def calculate_atr_sl_tp(indicators: dict, direction: str, entry_price: float) -> dict:
    """
    Calculate SL/TP based on ATR (not SMC levels which can be too far).
    
    SL = 2 × ATR from entry (reasonable for 4H)
    TP = 3 × ATR from entry (minimum 1:1.5 R:R)
    
    Returns dict with sl, tp, sl_pct, tp_pct
    """
    # ATR14 from indicators
    atr = indicators.get("atr14_value", 0)
    if not atr or atr <= 0:
        # Fallback: estimate ATR as 2% of price
        atr = entry_price * 0.02
    
    sl_distance = 2.0 * atr
    tp_distance = 3.0 * atr  # 1:1.5 R:R minimum
    
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


def check_volume_filter(volume_12h: float) -> bool:
    """
    Check if coin has sufficient volume for trading.
    
    Uses 12-hour volume (sum of last 3 × 4H candle quoteVolumes).
    This catches current activity, not inflated by old pumps.
    
    Rule: 12h volume must be ≥ $500K
    """
    MIN_VOLUME_12H = 500_000  # $500K in 12 hours
    return volume_12h >= MIN_VOLUME_12H


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
                short_pct: float, entry_price: float, reason: str):
    """
    Add a signal to the monitor queue.
    
    Each signal has its OWN next_check_ts = now + 30 min.
    So breakouts at 15:00, 15:05, 15:10 get re-checked at 15:30, 15:35, 15:40.
    NOT all at once. This prevents API spikes when 100 breakouts happen.
    
    Monitors survive 03:00 trendline redraw — they live for 24h independently.
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
        "reason": reason,  # "low_confidence" or "flat_market"
        "added_at": datetime.now(timezone.utc).isoformat(),
        "added_ts": now,
        "next_check_ts": now + RECHECK_INTERVAL_SEC,  # first re-check in 30 min
        "check_count": 0
    })
    
    save_monitors(monitors)
    logging.info(f"🔵 MONITOR added: {symbol} {tf} {direction} ({reason}) — next check in 30min")


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
            m["next_check_ts"] = now + RECHECK_INTERVAL_SEC  # next in 30 min
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
