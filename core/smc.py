"""
Smart Money Concepts (SMC) module — ported from LuxAlgo Pine Script.
Detects: Market Structure (BOS/CHoCH), Order Blocks, Fair Value Gaps,
Equal Highs/Lows, Premium/Discount Zones.

Input: pandas DataFrame with columns [open, high, low, close, volume]
Output: dict with all SMC data + formatted text summary for AI prompt.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

BULLISH = 1
BEARISH = -1
BULLISH_LEG = 1
BEARISH_LEG = 0


# ─── PIVOT / SWING DETECTION ────────────────────────────────────────────────

def find_pivots(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """
    Detect pivot highs and lows using LuxAlgo leg() logic.
    A pivot high at bar[i] if high[i] > max(high[i-size:i]) (bar is higher than next `size` bars before it).
    Returns df with added columns: pivot_high (bool), pivot_low (bool).
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    for i in range(size, n - size):
        # Check if high[i] is the highest in window [i-size, i+size]
        left_max_h = np.max(highs[i - size:i])
        right_max_h = np.max(highs[i + 1:i + size + 1])
        if highs[i] >= left_max_h and highs[i] >= right_max_h:
            pivot_high[i] = True

        # Check if low[i] is the lowest in window [i-size, i+size]
        left_min_l = np.min(lows[i - size:i])
        right_min_l = np.min(lows[i + 1:i + size + 1])
        if lows[i] <= left_min_l and lows[i] <= right_min_l:
            pivot_low[i] = True

    result = df.copy()
    result["pivot_high"] = pivot_high
    result["pivot_low"] = pivot_low
    return result


# ─── MARKET STRUCTURE (BOS / CHoCH) ─────────────────────────────────────────

def detect_structure(df: pd.DataFrame, size: int) -> List[Dict]:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).
    
    Logic (from LuxAlgo):
    - Track last swing high and swing low pivots.
    - When close crosses above last swing high:
        If trend was BULLISH → BOS (continuation)
        If trend was BEARISH → CHoCH (reversal)
    - When close crosses below last swing low:
        If trend was BEARISH → BOS (continuation)
        If trend was BULLISH → CHoCH (reversal)
    """
    pivoted = find_pivots(df, size)
    closes = pivoted["close"].values
    highs = pivoted["high"].values
    lows = pivoted["low"].values
    n = len(pivoted)

    structures = []
    trend = 0  # 0 = undefined, BULLISH = 1, BEARISH = -1

    last_swing_high = None  # {"price": float, "index": int, "crossed": bool}
    last_swing_low = None

    for i in range(n):
        # Update pivots
        if pivoted["pivot_high"].iloc[i]:
            last_swing_high = {"price": highs[i], "index": i, "crossed": False}
        if pivoted["pivot_low"].iloc[i]:
            last_swing_low = {"price": lows[i], "index": i, "crossed": False}

        # Check bullish break (close > last swing high)
        if last_swing_high and not last_swing_high["crossed"]:
            if closes[i] > last_swing_high["price"]:
                tag = "CHoCH" if trend == BEARISH else "BOS"
                bias = BULLISH
                last_swing_high["crossed"] = True
                trend = BULLISH
                structures.append({
                    "type": tag,
                    "bias": bias,
                    "price": last_swing_high["price"],
                    "break_index": i,
                    "pivot_index": last_swing_high["index"],
                })

        # Check bearish break (close < last swing low)
        if last_swing_low and not last_swing_low["crossed"]:
            if closes[i] < last_swing_low["price"]:
                tag = "CHoCH" if trend == BULLISH else "BOS"
                bias = BEARISH
                last_swing_low["crossed"] = True
                trend = BEARISH
                structures.append({
                    "type": tag,
                    "bias": bias,
                    "price": last_swing_low["price"],
                    "break_index": i,
                    "pivot_index": last_swing_low["index"],
                })

    return structures


# ─── ORDER BLOCKS ────────────────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, structures: List[Dict], max_blocks: int = 5) -> List[Dict]:
    """
    Find Order Blocks at structure breaks.
    
    At a bullish break: find the candle with the lowest low between the pivot and the break → bullish OB.
    At a bearish break: find the candle with the highest high between the pivot and the break → bearish OB.
    
    Then check mitigation (price has passed through OB).
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    n = len(df)

    # ATR for volatility filter
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(200, min_periods=1).mean().values

    order_blocks = []

    for s in structures:
        pivot_idx = s["pivot_index"]
        break_idx = s["break_index"]

        if pivot_idx >= break_idx or pivot_idx < 0:
            continue

        # High volatility bar parsing (from LuxAlgo)
        segment_highs = []
        segment_lows = []
        for j in range(pivot_idx, break_idx):
            hv = (highs[j] - lows[j]) >= (2 * atr[j]) if atr[j] > 0 else False
            parsed_h = lows[j] if hv else highs[j]
            parsed_l = highs[j] if hv else lows[j]
            segment_highs.append(parsed_h)
            segment_lows.append(parsed_l)

        if not segment_highs:
            continue

        if s["bias"] == BULLISH:
            # Bullish OB: candle with lowest parsed low
            min_idx = int(np.argmin(segment_lows))
            ob_idx = pivot_idx + min_idx
            ob = {
                "bias": BULLISH,
                "high": highs[ob_idx],
                "low": lows[ob_idx],
                "index": ob_idx,
                "mitigated": False,
            }
        else:
            # Bearish OB: candle with highest parsed high
            max_idx = int(np.argmax(segment_highs))
            ob_idx = pivot_idx + max_idx
            ob = {
                "bias": BEARISH,
                "high": highs[ob_idx],
                "low": lows[ob_idx],
                "index": ob_idx,
                "mitigated": False,
            }

        # Check mitigation (after OB formed, did price pass through it?)
        for k in range(break_idx + 1, n):
            if ob["bias"] == BEARISH and highs[k] > ob["high"]:
                ob["mitigated"] = True
                break
            elif ob["bias"] == BULLISH and lows[k] < ob["low"]:
                ob["mitigated"] = True
                break

        order_blocks.append(ob)

    # Return only unmitigated, most recent blocks
    active = [ob for ob in order_blocks if not ob["mitigated"]]
    # Keep last N
    return active[-max_blocks:]


# ─── FAIR VALUE GAPS (FVG) ──────────────────────────────────────────────────

def find_fair_value_gaps(df: pd.DataFrame) -> List[Dict]:
    """
    Detect Fair Value Gaps (3-candle imbalance).
    
    Bullish FVG: current_low > 2-bars-ago high (gap between candle 0 and candle 2)
    Bearish FVG: current_high < 2-bars-ago low
    
    Filter by ATR threshold to remove insignificant gaps.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    n = len(df)

    # ATR threshold
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    fvgs = []

    for i in range(2, n):
        # Bullish FVG: gap up
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            if gap_size > 0.5 * atr[i]:  # threshold filter
                mitigated = False
                for k in range(i + 1, n):
                    if lows[k] < highs[i - 2]:
                        mitigated = True
                        break
                fvgs.append({
                    "bias": BULLISH,
                    "top": lows[i],
                    "bottom": highs[i - 2],
                    "index": i,
                    "mitigated": mitigated,
                })

        # Bearish FVG: gap down
        if highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            if gap_size > 0.5 * atr[i]:
                mitigated = False
                for k in range(i + 1, n):
                    if highs[k] > lows[i - 2]:
                        mitigated = True
                        break
                fvgs.append({
                    "bias": BEARISH,
                    "top": lows[i - 2],
                    "bottom": highs[i],
                    "index": i,
                    "mitigated": mitigated,
                })

    return fvgs


# ─── EQUAL HIGHS / LOWS ─────────────────────────────────────────────────────

def find_equal_highs_lows(df: pd.DataFrame, size: int = 3, threshold: float = 0.1) -> List[Dict]:
    """
    Detect Equal Highs and Equal Lows.
    Two swing points at approximately the same price level → liquidity pool.
    """
    pivoted = find_pivots(df, size)
    highs = pivoted["high"].values
    lows = pivoted["low"].values
    n = len(pivoted)

    # ATR for threshold
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(df["close"].values, 1)), np.abs(lows - np.roll(df["close"].values, 1))))
    atr = pd.Series(tr).rolling(200, min_periods=1).mean().values

    equals = []

    # Collect pivot highs and lows
    pivot_highs = [(i, highs[i]) for i in range(n) if pivoted["pivot_high"].iloc[i]]
    pivot_lows = [(i, lows[i]) for i in range(n) if pivoted["pivot_low"].iloc[i]]

    # Check consecutive pivot highs for equal levels
    for j in range(1, len(pivot_highs)):
        idx_prev, price_prev = pivot_highs[j - 1]
        idx_curr, price_curr = pivot_highs[j]
        if atr[idx_curr] > 0 and abs(price_curr - price_prev) < threshold * atr[idx_curr]:
            equals.append({
                "type": "EQH",
                "price": round((price_curr + price_prev) / 2, 8),
                "index1": idx_prev,
                "index2": idx_curr,
            })

    # Check consecutive pivot lows
    for j in range(1, len(pivot_lows)):
        idx_prev, price_prev = pivot_lows[j - 1]
        idx_curr, price_curr = pivot_lows[j]
        if atr[idx_curr] > 0 and abs(price_curr - price_prev) < threshold * atr[idx_curr]:
            equals.append({
                "type": "EQL",
                "price": round((price_curr + price_prev) / 2, 8),
                "index1": idx_prev,
                "index2": idx_curr,
            })

    return equals


# ─── PREMIUM / DISCOUNT ZONES ───────────────────────────────────────────────

def get_premium_discount(df: pd.DataFrame, swing_size: int = 50) -> Dict:
    """
    Calculate Premium/Discount/Equilibrium zones from trailing swing high/low.
    Premium = top 5% (overpriced), Discount = bottom 5% (underpriced).
    """
    pivoted = find_pivots(df, swing_size)
    n = len(pivoted)

    # Find the last swing high and swing low
    trailing_high = df["high"].max()
    trailing_low = df["low"].min()

    # More precise: track from last significant swing
    for i in range(n - 1, -1, -1):
        if pivoted["pivot_high"].iloc[i]:
            trailing_high = pivoted["high"].iloc[i]
            break

    for i in range(n - 1, -1, -1):
        if pivoted["pivot_low"].iloc[i]:
            trailing_low = pivoted["low"].iloc[i]
            break

    equilibrium = (trailing_high + trailing_low) / 2
    current_price = df["close"].iloc[-1]

    if trailing_high == trailing_low:
        zone = "Equilibrium"
    elif current_price > equilibrium + (trailing_high - equilibrium) * 0.5:
        zone = "Premium"
    elif current_price < equilibrium - (equilibrium - trailing_low) * 0.5:
        zone = "Discount"
    else:
        zone = "Equilibrium"

    return {
        "swing_high": round(trailing_high, 8),
        "swing_low": round(trailing_low, 8),
        "equilibrium": round(equilibrium, 8),
        "premium_start": round(0.95 * trailing_high + 0.05 * trailing_low, 8),
        "discount_end": round(0.95 * trailing_low + 0.05 * trailing_high, 8),
        "current_zone": zone,
        "current_price": round(current_price, 8),
    }


# ─── MAIN ANALYSIS FUNCTION ─────────────────────────────────────────────────

def analyze_smc(df: pd.DataFrame, tf_label: str = "4H",
                internal_size: int = 5, swing_size: int = 50) -> Dict:
    """
    Full SMC analysis on a DataFrame.
    
    Returns dict with:
    - structures: list of BOS/CHoCH events
    - order_blocks: list of active (unmitigated) OBs
    - fvgs: list of active FVGs
    - equal_hl: list of EQH/EQL
    - zones: premium/discount/equilibrium
    - summary: formatted text for AI prompt
    """
    if df is None or len(df) < 30:
        return {"summary": f"[{tf_label}] Insufficient data for SMC analysis."}

    try:
        # Ensure numeric types
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        if len(df) < 30:
            return {"summary": f"[{tf_label}] Insufficient data for SMC analysis."}

        # Internal structure (size=5 like LuxAlgo)
        internal_structures = detect_structure(df, internal_size)
        # Swing structure (size=50 like LuxAlgo default)
        swing_structures = detect_structure(df, swing_size)

        # Order blocks from both
        internal_obs = find_order_blocks(df, internal_structures, max_blocks=5)
        swing_obs = find_order_blocks(df, swing_structures, max_blocks=3)

        # Fair value gaps
        fvgs = find_fair_value_gaps(df)
        active_fvgs = [f for f in fvgs if not f["mitigated"]]

        # Equal highs/lows
        equal_hl = find_equal_highs_lows(df, size=3, threshold=0.1)

        # Premium/Discount zones
        zones = get_premium_discount(df, swing_size)

        # ── BUILD TEXT SUMMARY ──
        lines = [f"📐 SMC Indicator [{tf_label}]:"]

        # Bars since last structure
        all_structures = swing_structures + internal_structures
        bars_since_last_structure = 0
        if all_structures:
            all_structures.sort(key=lambda x: x["break_index"])
            last_structure = all_structures[-1]
            bars_since_last_structure = len(df) - 1 - last_structure["break_index"]
            lines.append(f"Last Structure: {bars_since_last_structure} bars ago")

        # Current trend from last swing structure
        if swing_structures:
            last_swing = swing_structures[-1]
            trend_str = "BULLISH" if last_swing["bias"] == BULLISH else "BEARISH"
            lines.append(f"Trend: {trend_str} (last: {last_swing['type']})")

        # Internal trend
        if internal_structures:
            last_internal = internal_structures[-1]
            int_trend = "BULLISH" if last_internal["bias"] == BULLISH else "BEARISH"
            lines.append(f"Internal: {int_trend} (last: {last_internal['type']})")

        # Recent structure breaks (last 3)
        recent_breaks = (swing_structures + internal_structures)
        recent_breaks.sort(key=lambda x: x["break_index"])
        for s in recent_breaks[-3:]:
            bias = "Bull" if s["bias"] == BULLISH else "Bear"
            lines.append(f"  {s['type']} {bias} @ {s['price']:.6f}")

        # Active Order Blocks (ranked by distance from current price)
        current_price = df['close'].iloc[-1]
        all_obs = swing_obs + internal_obs
        if all_obs:
            # Calculate distance from current price
            for ob in all_obs:
                ob_mid = (ob['high'] + ob['low']) / 2
                ob['distance_pct'] = abs((current_price - ob_mid) / current_price) * 100
            
            # Sort by distance (closest first)
            all_obs.sort(key=lambda x: x['distance_pct'])
            
            lines.append("Order Blocks (by distance):")
            for ob in all_obs[:5]:  # Show top 5 closest
                tag = "Bull OB" if ob["bias"] == BULLISH else "Bear OB"
                source = "Swing" if ob in swing_obs else "Int"
                lines.append(f"  {source} {tag}: {ob['low']:.6f}-{ob['high']:.6f} ({ob['distance_pct']:.1f}% away)")

        # Active FVGs (last 3)
        if active_fvgs:
            lines.append("FVG:")
            for f in active_fvgs[-3:]:
                tag = "Bull" if f["bias"] == BULLISH else "Bear"
                lines.append(f"  {tag} FVG: {f['bottom']:.6f}-{f['top']:.6f}")

        # Equal Highs/Lows (last 2 each)
        recent_eqh = [e for e in equal_hl if e["type"] == "EQH"][-2:]
        recent_eql = [e for e in equal_hl if e["type"] == "EQL"][-2:]
        if recent_eqh or recent_eql:
            lines.append("Liquidity:")
            for e in recent_eqh:
                dist_pct = abs((current_price - e['price']) / current_price) * 100
                lines.append(f"  EQH @ {e['price']:.6f} ({dist_pct:.1f}% away)")
            for e in recent_eql:
                dist_pct = abs((current_price - e['price']) / current_price) * 100
                lines.append(f"  EQL @ {e['price']:.6f} ({dist_pct:.1f}% away)")

        # Zones
        lines.append(f"Zone: {zones['current_zone']} (H:{zones['swing_high']:.6f} L:{zones['swing_low']:.6f} EQ:{zones['equilibrium']:.6f})")

        summary = "\n".join(lines)

        return {
            "swing_structures": swing_structures,
            "internal_structures": internal_structures,
            "swing_order_blocks": swing_obs,
            "internal_order_blocks": internal_obs,
            "fvgs": active_fvgs,
            "equal_hl": equal_hl,
            "zones": zones,
            "summary": summary,
        }

    except Exception as e:
        logging.error(f"❌ SMC analysis error ({tf_label}): {e}")
        return {"summary": f"[{tf_label}] SMC error: {str(e)[:100]}"}
