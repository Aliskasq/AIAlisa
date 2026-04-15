"""
Smart Money Concepts (SMC) module — exact port of LuxAlgo Pine Script v5.

Detects: Market Structure (BOS/CHoCH), Order Blocks (internal + swing),
Fair Value Gaps, Equal Highs/Lows, Strong/Weak High/Low,
Premium/Discount Zones, Trailing Extremes.

Input:  pandas DataFrame with columns [open, high, low, close, volume]
Output: dict with all SMC data + formatted text summary for AI prompt.

Reference: 'Smart Money Concepts [LuxAlgo]' indicator for TradingView.
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


# ─── LEG DETECTION (exact LuxAlgo port) ─────────────────────────────────────

def _compute_legs(highs: np.ndarray, lows: np.ndarray, size: int) -> np.ndarray:
    """
    Port of LuxAlgo leg() function.

    leg(int size) =>
        var leg = 0
        newLegHigh = high[size] > ta.highest(size)   // bar `size` ago is higher than highest of bars 0..size-1
        newLegLow  = low[size]  < ta.lowest(size)    // bar `size` ago is lower  than lowest  of bars 0..size-1
        if newLegHigh → leg := BEARISH_LEG (0)
        if newLegLow  → leg := BULLISH_LEG (1)

    Returns array of leg values (0 or 1) for each bar.
    """
    n = len(highs)
    legs = np.zeros(n, dtype=int)
    current_leg = 0

    for i in range(size, n):
        # high[size] in Pine = highs[i - size] looking back from bar i
        bar_high = highs[i - size]
        bar_low = lows[i - size]

        # ta.highest(size) = max of bars [i-size+1 .. i] (the most recent `size` bars)
        window_high = np.max(highs[i - size + 1: i + 1])
        # ta.lowest(size) = min of bars [i-size+1 .. i]
        window_low = np.min(lows[i - size + 1: i + 1])

        if bar_high > window_high:
            current_leg = BEARISH_LEG  # pivot high detected → start of bearish leg
        elif bar_low < window_low:
            current_leg = BULLISH_LEG  # pivot low detected → start of bullish leg

        legs[i] = current_leg

    return legs


def _detect_pivots_from_legs(highs: np.ndarray, lows: np.ndarray,
                             legs: np.ndarray, size: int) -> List[Dict]:
    """
    Detect pivot points from leg changes (exact LuxAlgo getCurrentStructure logic).

    When leg changes:
    - From bearish→bullish (startOfBullishLeg): pivot LOW at bar[i-size] (low)
    - From bullish→bearish (startOfBearishLeg): pivot HIGH at bar[i-size] (high)

    Returns list of pivots: {"type": "high"|"low", "price": float, "index": int}
    """
    n = len(legs)
    pivots = []

    for i in range(size + 1, n):
        change = legs[i] - legs[i - 1]
        if change == 0:
            continue

        if change > 0:
            # startOfBullishLeg → new pivot LOW at bar[i - size]
            idx = i - size
            if idx >= 0:
                pivots.append({
                    "type": "low",
                    "price": float(lows[idx]),
                    "index": idx,
                })
        elif change < 0:
            # startOfBearishLeg → new pivot HIGH at bar[i - size]
            idx = i - size
            if idx >= 0:
                pivots.append({
                    "type": "high",
                    "price": float(highs[idx]),
                    "index": idx,
                })

    return pivots


# ─── MARKET STRUCTURE (BOS / CHoCH) — exact LuxAlgo displayStructure ────────

def detect_structure(df: pd.DataFrame, size: int,
                     internal: bool = False,
                     swing_high_level: float = None,
                     swing_low_level: float = None) -> Tuple[List[Dict], List[Dict], int]:
    """
    Detect BOS and CHoCH from structure breaks.

    Exact LuxAlgo logic:
    - Track last pivot high (swingHigh / internalHigh) and last pivot low.
    - On each bar, check:
      * close crosses ABOVE last pivot high (crossover, not just above):
        trend was BEARISH → CHoCH; trend was BULLISH → BOS
      * close crosses BELOW last pivot low (crossunder):
        trend was BULLISH → CHoCH; trend was BEARISH → BOS
    - Internal filter: if internal, skip if pivot level == swing level (confluence filter).

    Returns:
        (structures, pivots, final_trend_bias)
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    n = len(df)

    # Compute legs and pivots
    legs = _compute_legs(highs, lows, size)
    raw_pivots = _detect_pivots_from_legs(highs, lows, legs, size)

    structures = []
    trend = 0  # 0 = undefined

    # State tracking (exact LuxAlgo pivot UDT)
    last_high = {"price": None, "last_price": None, "index": 0, "crossed": False}
    last_low = {"price": None, "last_price": None, "index": 0, "crossed": False}

    # Process pivots in bar order, interleave with close checks
    # Build a per-bar event map
    pivot_at_bar = {}
    for p in raw_pivots:
        idx = p["index"]
        if idx not in pivot_at_bar:
            pivot_at_bar[idx] = []
        pivot_at_bar[idx].append(p)

    for i in range(n):
        # Update pivots that occur at this bar
        if i in pivot_at_bar:
            for p in pivot_at_bar[i]:
                if p["type"] == "low":
                    last_low["last_price"] = last_low["price"]
                    last_low["price"] = p["price"]
                    last_low["index"] = p["index"]
                    last_low["crossed"] = False
                elif p["type"] == "high":
                    last_high["last_price"] = last_high["price"]
                    last_high["price"] = p["price"]
                    last_high["index"] = p["index"]
                    last_high["crossed"] = False

        # Check bullish break: close crosses above last high
        if last_high["price"] is not None and not last_high["crossed"]:
            # Crossover: previous close <= level AND current close > level
            prev_close = closes[i - 1] if i > 0 else 0
            if prev_close <= last_high["price"] and closes[i] > last_high["price"]:
                # Internal confluence filter
                if internal and swing_high_level is not None:
                    if last_high["price"] == swing_high_level:
                        pass  # skip — same as swing level
                    else:
                        tag = "CHoCH" if trend == BEARISH else "BOS"
                        last_high["crossed"] = True
                        trend = BULLISH
                        structures.append({
                            "type": tag,
                            "bias": BULLISH,
                            "price": last_high["price"],
                            "break_index": i,
                            "pivot_index": last_high["index"],
                        })
                else:
                    tag = "CHoCH" if trend == BEARISH else "BOS"
                    last_high["crossed"] = True
                    trend = BULLISH
                    structures.append({
                        "type": tag,
                        "bias": BULLISH,
                        "price": last_high["price"],
                        "break_index": i,
                        "pivot_index": last_high["index"],
                    })

        # Check bearish break: close crosses below last low
        if last_low["price"] is not None and not last_low["crossed"]:
            prev_close = closes[i - 1] if i > 0 else float('inf')
            if prev_close >= last_low["price"] and closes[i] < last_low["price"]:
                # Internal confluence filter
                if internal and swing_low_level is not None:
                    if last_low["price"] == swing_low_level:
                        pass  # skip
                    else:
                        tag = "CHoCH" if trend == BULLISH else "BOS"
                        last_low["crossed"] = True
                        trend = BEARISH
                        structures.append({
                            "type": tag,
                            "bias": BEARISH,
                            "price": last_low["price"],
                            "break_index": i,
                            "pivot_index": last_low["index"],
                        })
                else:
                    tag = "CHoCH" if trend == BULLISH else "BOS"
                    last_low["crossed"] = True
                    trend = BEARISH
                    structures.append({
                        "type": tag,
                        "bias": BEARISH,
                        "price": last_low["price"],
                        "break_index": i,
                        "pivot_index": last_low["index"],
                    })

    return structures, raw_pivots, trend


# ─── ORDER BLOCKS (exact LuxAlgo storeOrderBlock + deleteOrderBlocks) ───────

def find_order_blocks(df: pd.DataFrame, structures: List[Dict],
                      max_blocks: int = 5,
                      mitigation: str = "highlow") -> List[Dict]:
    """
    Find and manage Order Blocks at structure breaks.

    LuxAlgo logic:
    - At bullish break: find candle with min(parsedLow) between pivot and break.
    - At bearish break: find candle with max(parsedHigh) between pivot and break.
    - parsedHigh/parsedLow: if bar is high-volatility (range >= 2*ATR), swap high↔low.
    - OB boundaries use parsedHighs/parsedLows (not raw).
    - Mitigation: bearish OB mitigated when high > OB.barHigh (or close > OB.barHigh).
                  bullish OB mitigated when low < OB.barLow (or close < OB.barLow).
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    # ATR(200) for volatility filter (exact LuxAlgo: ta.atr(200))
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr200 = pd.Series(tr).rolling(200, min_periods=1).mean().values

    # Build parsed highs/lows arrays (exact LuxAlgo high-volatility bar logic)
    parsed_highs = np.copy(highs)
    parsed_lows = np.copy(lows)
    for i in range(n):
        if (highs[i] - lows[i]) >= 2 * atr200[i] and atr200[i] > 0:
            # High volatility bar: swap
            parsed_highs[i] = lows[i]
            parsed_lows[i] = highs[i]

    order_blocks = []

    for s in structures:
        pivot_idx = s["pivot_index"]
        break_idx = s["break_index"]

        if pivot_idx >= break_idx or pivot_idx < 0:
            continue

        if s["bias"] == BEARISH:
            # Bearish OB: find max parsedHigh between pivot and break
            segment = parsed_highs[pivot_idx:break_idx]
            if len(segment) == 0:
                continue
            local_idx = int(np.argmax(segment))
            ob_idx = pivot_idx + local_idx
        else:
            # Bullish OB: find min parsedLow between pivot and break
            segment = parsed_lows[pivot_idx:break_idx]
            if len(segment) == 0:
                continue
            local_idx = int(np.argmin(segment))
            ob_idx = pivot_idx + local_idx

        # OB boundaries use PARSED values (exact LuxAlgo)
        ob = {
            "bias": s["bias"],
            "high": float(parsed_highs[ob_idx]),
            "low": float(parsed_lows[ob_idx]),
            "raw_high": float(highs[ob_idx]),
            "raw_low": float(lows[ob_idx]),
            "index": ob_idx,
            "break_index": break_idx,
            "mitigated": False,
            "mitigated_index": None,
        }

        # Check mitigation after the break
        for k in range(break_idx + 1, n):
            if mitigation == "close":
                mit_bear_src = closes[k]
                mit_bull_src = closes[k]
            else:
                mit_bear_src = highs[k]
                mit_bull_src = lows[k]

            if ob["bias"] == BEARISH and mit_bear_src > ob["high"]:
                ob["mitigated"] = True
                ob["mitigated_index"] = k
                break
            elif ob["bias"] == BULLISH and mit_bull_src < ob["low"]:
                ob["mitigated"] = True
                ob["mitigated_index"] = k
                break

        order_blocks.append(ob)

    # Return unmitigated, most recent blocks (capped)
    active = [ob for ob in order_blocks if not ob["mitigated"]]
    if len(active) > max_blocks:
        active = active[-max_blocks:]
    return active


# ─── FAIR VALUE GAPS (exact LuxAlgo drawFairValueGaps) ──────────────────────

def find_fair_value_gaps(df: pd.DataFrame) -> List[Dict]:
    """
    Detect Fair Value Gaps (3-candle imbalance).

    LuxAlgo logic:
    Bullish FVG:  currentLow > last2High AND lastClose > last2High AND barDelta > threshold
    Bearish FVG:  currentHigh < last2Low AND lastClose < last2Low AND -barDelta > threshold

    Threshold: cumulative mean of |bar delta %| * 2  (auto threshold)
    Mitigation: bullish → low < FVG.bottom;  bearish → high > FVG.top
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    n = len(df)

    # Cumulative mean range threshold (LuxAlgo auto threshold)
    cum_abs_delta = 0.0
    fvgs = []

    for i in range(2, n):
        # barDeltaPercent = (close[i-1] - open[i-1]) / (open[i-1] * 100)
        last_close = closes[i - 1]
        last_open = opens[i - 1]
        if last_open != 0:
            bar_delta_pct = (last_close - last_open) / (last_open * 100)
        else:
            bar_delta_pct = 0

        cum_abs_delta += abs(bar_delta_pct)
        threshold = (cum_abs_delta / (i - 1)) * 2 if i > 1 else 0

        current_low = lows[i]
        current_high = highs[i]
        last2_high = highs[i - 2]
        last2_low = lows[i - 2]

        # Bullish FVG
        if current_low > last2_high and last_close > last2_high and bar_delta_pct > threshold:
            mitigated = False
            for k in range(i + 1, n):
                if lows[k] < last2_high:  # low < FVG.bottom
                    mitigated = True
                    break
            fvgs.append({
                "bias": BULLISH,
                "top": float(current_low),
                "bottom": float(last2_high),
                "index": i,
                "mitigated": mitigated,
            })

        # Bearish FVG
        if current_high < last2_low and last_close < last2_low and (-bar_delta_pct) > threshold:
            mitigated = False
            for k in range(i + 1, n):
                if highs[k] > last2_low:  # high > FVG.top
                    mitigated = True
                    break
            fvgs.append({
                "bias": BEARISH,
                "top": float(last2_low),
                "bottom": float(current_high),
                "index": i,
                "mitigated": mitigated,
            })

    return fvgs


# ─── EQUAL HIGHS / LOWS (exact LuxAlgo) ─────────────────────────────────────

def find_equal_highs_lows(df: pd.DataFrame, pivots: List[Dict],
                          threshold: float = 0.1) -> List[Dict]:
    """
    Detect EQH and EQL from consecutive pivots at similar price levels.

    LuxAlgo: uses separate getCurrentStructure(equalHighsLowsLengthInput, true)
    with threshold * ATR(200) for comparison.

    Uses pivots already detected (avoids recomputation).
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    # ATR(200)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr200 = pd.Series(tr).rolling(200, min_periods=1).mean().values

    equals = []

    # Separate pivot highs and lows
    pivot_highs = [p for p in pivots if p["type"] == "high"]
    pivot_lows = [p for p in pivots if p["type"] == "low"]

    # Consecutive pivot highs
    for j in range(1, len(pivot_highs)):
        prev = pivot_highs[j - 1]
        curr = pivot_highs[j]
        idx = curr["index"]
        if idx < n and atr200[idx] > 0:
            if abs(curr["price"] - prev["price"]) < threshold * atr200[idx]:
                equals.append({
                    "type": "EQH",
                    "price": round((curr["price"] + prev["price"]) / 2, 8),
                    "index1": prev["index"],
                    "index2": curr["index"],
                })

    # Consecutive pivot lows
    for j in range(1, len(pivot_lows)):
        prev = pivot_lows[j - 1]
        curr = pivot_lows[j]
        idx = curr["index"]
        if idx < n and atr200[idx] > 0:
            if abs(curr["price"] - prev["price"]) < threshold * atr200[idx]:
                equals.append({
                    "type": "EQL",
                    "price": round((curr["price"] + prev["price"]) / 2, 8),
                    "index1": prev["index"],
                    "index2": curr["index"],
                })

    return equals


# ─── TRAILING EXTREMES + STRONG/WEAK HIGH/LOW (exact LuxAlgo) ───────────────

def compute_trailing_extremes(df: pd.DataFrame, swing_pivots: List[Dict],
                              swing_trend: int) -> Dict:
    """
    Port of LuxAlgo updateTrailingExtremes() + drawHighLowSwings().

    trailing.top    = running max(high) since last swing pivot update
    trailing.bottom = running min(low)  since last swing pivot update

    Strong/Weak logic:
    - swingTrend == BEARISH → top is 'Strong High', bottom is 'Weak Low'
    - swingTrend == BULLISH → top is 'Weak High',   bottom is 'Strong Low'
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    # Find the last swing pivot (the point where trailing extremes reset)
    last_swing_idx = 0
    if swing_pivots:
        last_swing_idx = max(p["index"] for p in swing_pivots)

    # Track trailing high/low from last swing pivot to end
    trailing_high = highs[last_swing_idx] if last_swing_idx < n else highs[-1]
    trailing_low = lows[last_swing_idx] if last_swing_idx < n else lows[-1]
    trailing_high_idx = last_swing_idx
    trailing_low_idx = last_swing_idx

    for i in range(last_swing_idx, n):
        if highs[i] >= trailing_high:
            trailing_high = highs[i]
            trailing_high_idx = i
        if lows[i] <= trailing_low:
            trailing_low = lows[i]
            trailing_low_idx = i

    # Strong/Weak labels (exact LuxAlgo logic)
    if swing_trend == BEARISH:
        high_label = "Strong High"
        low_label = "Weak Low"
    elif swing_trend == BULLISH:
        high_label = "Weak High"
        low_label = "Strong Low"
    else:
        high_label = "High"
        low_label = "Low"

    return {
        "trailing_high": float(trailing_high),
        "trailing_low": float(trailing_low),
        "trailing_high_index": trailing_high_idx,
        "trailing_low_index": trailing_low_idx,
        "high_label": high_label,
        "low_label": low_label,
    }


# ─── PREMIUM / DISCOUNT ZONES (exact LuxAlgo drawPremiumDiscountZones) ──────

def get_premium_discount(trailing: Dict, current_price: float) -> Dict:
    """
    Exact LuxAlgo zones:
    Premium:     [0.95*high + 0.05*low, high]           (top 5%)
    Equilibrium: [0.525*low + 0.475*high, 0.525*high + 0.475*low]  (middle ~5%)
    Discount:    [low, 0.95*low + 0.05*high]             (bottom 5%)
    """
    h = trailing["trailing_high"]
    l = trailing["trailing_low"]

    if h == l:
        zone = "Equilibrium"
    else:
        premium_start = 0.95 * h + 0.05 * l
        discount_end = 0.95 * l + 0.05 * h
        eq_top = 0.525 * h + 0.475 * l
        eq_bottom = 0.525 * l + 0.475 * h

        if current_price >= premium_start:
            zone = "Premium"
        elif current_price <= discount_end:
            zone = "Discount"
        else:
            zone = "Equilibrium"

    equilibrium = (h + l) / 2

    return {
        "swing_high": round(h, 8),
        "swing_low": round(l, 8),
        "equilibrium": round(equilibrium, 8),
        "premium_start": round(0.95 * h + 0.05 * l, 8),
        "discount_end": round(0.95 * l + 0.05 * h, 8),
        "current_zone": zone,
        "current_price": round(current_price, 8),
    }


# ─── MAIN ANALYSIS FUNCTION ─────────────────────────────────────────────────

def analyze_smc(df: pd.DataFrame, tf_label: str = "4H",
                internal_size: int = 5, swing_size: int = 50,
                ob_mitigation: str = "highlow") -> Dict:
    """
    Full SMC analysis — exact LuxAlgo port.

    Pipeline (matches LuxAlgo execution order):
    1. getCurrentStructure(swingsLengthInput)     → swing pivots
    2. getCurrentStructure(5, internal=True)       → internal pivots
    3. getCurrentStructure(eqhlLength, eqhl=True)  → equal highs/lows pivots
    4. displayStructure(internal=True)             → internal BOS/CHoCH + internal OBs
    5. displayStructure()                          → swing BOS/CHoCH + swing OBs
    6. deleteOrderBlocks (mitigation check)
    7. updateTrailingExtremes + Strong/Weak High/Low
    8. Premium/Discount zones
    9. Fair Value Gaps

    Returns dict with all SMC data + formatted text summary.
    """
    if df is None or len(df) < 30:
        return {"summary": f"[{tf_label}] Insufficient data for SMC analysis."}

    try:
        # Ensure numeric and clean
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        if len(df) < 30:
            return {"summary": f"[{tf_label}] Insufficient data for SMC analysis."}

        n = len(df)
        current_price = float(df["close"].iloc[-1])

        # ── 1. SWING STRUCTURE ──
        swing_structures, swing_pivots, swing_trend = detect_structure(
            df, swing_size, internal=False
        )

        # Get last swing high/low levels for internal confluence filter
        swing_high_level = None
        swing_low_level = None
        for p in reversed(swing_pivots):
            if p["type"] == "high" and swing_high_level is None:
                swing_high_level = p["price"]
            if p["type"] == "low" and swing_low_level is None:
                swing_low_level = p["price"]
            if swing_high_level is not None and swing_low_level is not None:
                break

        # ── 2. INTERNAL STRUCTURE (with confluence filter) ──
        internal_structures, internal_pivots, internal_trend = detect_structure(
            df, internal_size, internal=True,
            swing_high_level=swing_high_level,
            swing_low_level=swing_low_level
        )

        # ── 3. EQUAL HIGHS / LOWS (using size=3 pivots, like LuxAlgo default) ──
        eqhl_legs = _compute_legs(df["high"].values, df["low"].values, 3)
        eqhl_pivots = _detect_pivots_from_legs(
            df["high"].values, df["low"].values, eqhl_legs, 3
        )
        equal_hl = find_equal_highs_lows(df, eqhl_pivots, threshold=0.1)

        # ── 4-5. ORDER BLOCKS ──
        internal_obs = find_order_blocks(
            df, internal_structures, max_blocks=5, mitigation=ob_mitigation
        )
        swing_obs = find_order_blocks(
            df, swing_structures, max_blocks=5, mitigation=ob_mitigation
        )

        # ── 6. FAIR VALUE GAPS ──
        all_fvgs = find_fair_value_gaps(df)
        active_fvgs = [f for f in all_fvgs if not f["mitigated"]]

        # ── 7. TRAILING EXTREMES + STRONG/WEAK HIGH/LOW ──
        trailing = compute_trailing_extremes(df, swing_pivots, swing_trend)

        # ── 8. PREMIUM / DISCOUNT ZONES ──
        zones = get_premium_discount(trailing, current_price)

        # ── BUILD TEXT SUMMARY FOR AI PROMPT ──
        lines = [f"📐 SMC [{tf_label}]:"]

        # Swing trend + last structure
        if swing_structures:
            last_swing_s = swing_structures[-1]
            trend_str = "BULLISH" if last_swing_s["bias"] == BULLISH else "BEARISH"
            bars_ago = n - 1 - last_swing_s["break_index"]
            lines.append(f"Swing Trend: {trend_str} (last: {last_swing_s['type']} {bars_ago} bars ago)")

        # Internal trend
        if internal_structures:
            last_int_s = internal_structures[-1]
            int_trend_str = "BULLISH" if last_int_s["bias"] == BULLISH else "BEARISH"
            int_bars_ago = n - 1 - last_int_s["break_index"]
            lines.append(f"Internal Trend: {int_trend_str} (last: {last_int_s['type']} {int_bars_ago} bars ago)")

        # Trend agreement / divergence
        if swing_structures and internal_structures:
            if last_swing_s["bias"] != last_int_s["bias"]:
                lines.append("⚠️ DIVERGENCE: Swing vs Internal trend disagree!")

        # Strong/Weak High/Low
        dist_high = abs(current_price - trailing["trailing_high"]) / current_price * 100
        dist_low = abs(current_price - trailing["trailing_low"]) / current_price * 100
        lines.append(
            f"{trailing['high_label']}: {trailing['trailing_high']:.6f} ({dist_high:.1f}% away) | "
            f"{trailing['low_label']}: {trailing['trailing_low']:.6f} ({dist_low:.1f}% away)"
        )

        # Recent structure breaks (last 5 combined, sorted by time)
        all_structs = []
        for s in swing_structures:
            s["_source"] = "Swing"
            all_structs.append(s)
        for s in internal_structures:
            s["_source"] = "Int"
            all_structs.append(s)
        all_structs.sort(key=lambda x: x["break_index"])

        if all_structs:
            lines.append("Recent Structures:")
            for s in all_structs[-5:]:
                bias_str = "Bull" if s["bias"] == BULLISH else "Bear"
                bars = n - 1 - s["break_index"]
                lines.append(f"  {s['_source']} {s['type']} {bias_str} @ {s['price']:.6f} ({bars} bars ago)")

        # Active Order Blocks (sorted by distance)
        all_obs = []
        for ob in swing_obs:
            ob["_source"] = "Swing"
            all_obs.append(ob)
        for ob in internal_obs:
            ob["_source"] = "Int"
            all_obs.append(ob)

        if all_obs:
            for ob in all_obs:
                ob_mid = (ob["high"] + ob["low"]) / 2
                ob["_distance_pct"] = abs((current_price - ob_mid) / current_price) * 100
                ob["_side"] = "below" if ob_mid < current_price else "above"
            all_obs.sort(key=lambda x: x["_distance_pct"])

            lines.append("Order Blocks:")
            for ob in all_obs[:6]:
                tag = "🟦 Bull OB" if ob["bias"] == BULLISH else "🟥 Bear OB"
                lines.append(
                    f"  {ob['_source']} {tag}: {ob['low']:.6f}-{ob['high']:.6f} "
                    f"({ob['_distance_pct']:.1f}% {ob['_side']})"
                )

        # Active FVGs (closest to price)
        if active_fvgs:
            for f in active_fvgs:
                f_mid = (f["top"] + f["bottom"]) / 2
                f["_distance_pct"] = abs((current_price - f_mid) / current_price) * 100
            active_fvgs.sort(key=lambda x: x["_distance_pct"])

            lines.append("Fair Value Gaps:")
            for f in active_fvgs[:4]:
                tag = "Bull" if f["bias"] == BULLISH else "Bear"
                lines.append(
                    f"  {tag} FVG: {f['bottom']:.6f}-{f['top']:.6f} ({f['_distance_pct']:.1f}% away)"
                )

        # Equal Highs/Lows (closest to price)
        recent_eqh = [e for e in equal_hl if e["type"] == "EQH"]
        recent_eql = [e for e in equal_hl if e["type"] == "EQL"]
        for e in recent_eqh + recent_eql:
            e["_distance_pct"] = abs((current_price - e["price"]) / current_price) * 100

        eqh_close = sorted(recent_eqh, key=lambda x: x["_distance_pct"])[:2]
        eql_close = sorted(recent_eql, key=lambda x: x["_distance_pct"])[:2]

        if eqh_close or eql_close:
            lines.append("Liquidity Pools:")
            for e in eqh_close:
                side = "above" if e["price"] > current_price else "below"
                lines.append(f"  EQH @ {e['price']:.6f} ({e['_distance_pct']:.1f}% {side})")
            for e in eql_close:
                side = "above" if e["price"] > current_price else "below"
                lines.append(f"  EQL @ {e['price']:.6f} ({e['_distance_pct']:.1f}% {side})")

        # Premium/Discount zone
        lines.append(
            f"Zone: {zones['current_zone']} "
            f"(H:{zones['swing_high']:.6f} L:{zones['swing_low']:.6f} EQ:{zones['equilibrium']:.6f})"
        )

        summary = "\n".join(lines)

        return {
            "swing_structures": swing_structures,
            "internal_structures": internal_structures,
            "swing_order_blocks": swing_obs,
            "internal_order_blocks": internal_obs,
            "fvgs": active_fvgs,
            "equal_hl": equal_hl,
            "zones": zones,
            "trailing": trailing,
            "swing_trend": swing_trend,
            "internal_trend": internal_trend,
            "summary": summary,
        }

    except Exception as e:
        logging.error(f"❌ SMC analysis error ({tf_label}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"summary": f"[{tf_label}] SMC error: {str(e)[:100]}"}


def score_smc(smc_data: dict, current_price: float) -> dict:
    """
    Score SMC data for the 5-group scorecard.
    
    Returns dict with:
      long_score, short_score (raw points),
      long_pct, short_pct (0-100),
      details (list of strings explaining each score component)
    """
    long_pts = 0
    short_pts = 0
    details = []
    max_possible = 0  # track max for percentage calc

    if not smc_data or "swing_structures" not in smc_data:
        return {"long_pct": 50, "short_pct": 50, "details": ["No SMC data"]}

    # ============================================
    # 1. BOS / CHoCH — Swing Structure (+3 CHoCH, +2 BOS)
    # ============================================
    max_possible += 3
    swing_structs = smc_data.get("swing_structures", [])
    if swing_structs:
        last_swing = swing_structs[-1]
        swing_type = last_swing.get("type", "")  # "BOS" or "CHoCH"
        swing_bias = last_swing.get("bias", "")   # "Bullish" or "Bearish"
        bars_since = last_swing.get("bars_since", 999)

        # Freshness multiplier
        if bars_since < 10:
            fresh_mult = 1.0
        elif bars_since < 30:
            fresh_mult = 0.5
        else:
            fresh_mult = 0.3

        base_pts = 3 if "CHoCH" in swing_type else 2

        if "Bullish" in swing_bias or "bullish" in swing_bias:
            pts = round(base_pts * fresh_mult, 1)
            long_pts += pts
            details.append(f"Swing {swing_type} Bullish ({bars_since}b ago) → LONG +{pts}")
        elif "Bearish" in swing_bias or "bearish" in swing_bias:
            pts = round(base_pts * fresh_mult, 1)
            short_pts += pts
            details.append(f"Swing {swing_type} Bearish ({bars_since}b ago) → SHORT +{pts}")

    # ============================================
    # 2. Internal Structure (×0.5 weight of swing)
    # ============================================
    max_possible += 1.5
    internal_structs = smc_data.get("internal_structures", [])
    if internal_structs:
        last_internal = internal_structs[-1]
        int_type = last_internal.get("type", "")
        int_bias = last_internal.get("bias", "")
        int_bars = last_internal.get("bars_since", 999)

        if int_bars < 10:
            int_fresh = 0.5
        elif int_bars < 30:
            int_fresh = 0.25
        else:
            int_fresh = 0.15

        int_base = 1.5 if "CHoCH" in int_type else 1.0

        if "Bullish" in int_bias or "bullish" in int_bias:
            pts = round(int_base * int_fresh / 0.5, 1)  # normalized
            long_pts += pts
            details.append(f"Internal {int_type} Bullish ({int_bars}b) → LONG +{pts}")
        elif "Bearish" in int_bias or "bearish" in int_bias:
            pts = round(int_base * int_fresh / 0.5, 1)
            short_pts += pts
            details.append(f"Internal {int_type} Bearish ({int_bars}b) → SHORT +{pts}")

    # Internal vs Swing conflict
    if swing_structs and internal_structs:
        swing_bias = swing_structs[-1].get("bias", "")
        int_bias = internal_structs[-1].get("bias", "")
        if ("Bullish" in swing_bias and "Bearish" in int_bias):
            long_pts -= 1
            details.append("⚠️ Swing↑ vs Internal↓ conflict → LONG -1")
        elif ("Bearish" in swing_bias and "Bullish" in int_bias):
            short_pts -= 1
            details.append("⚠️ Swing↓ vs Internal↑ conflict → SHORT -1")

    # ============================================
    # 3. Order Blocks (proximity scoring)
    # ============================================
    max_possible += 4  # up to ±2 per side
    all_obs = smc_data.get("swing_order_blocks", []) + smc_data.get("internal_order_blocks", [])

    for ob in all_obs:
        ob_top = ob.get("top", 0)
        ob_bottom = ob.get("bottom", 0)
        ob_type = ob.get("type", "")  # "bull" or "bear"
        ob_mid = (ob_top + ob_bottom) / 2 if ob_top and ob_bottom else 0

        if ob_mid <= 0 or current_price <= 0:
            continue

        dist_pct = abs(current_price - ob_mid) / current_price * 100

        if dist_pct > 5:
            continue  # too far

        if "bull" in ob_type.lower():
            # Bullish OB = support zone (below price = good for LONG)
            if ob_mid < current_price:
                pts = 2 if dist_pct < 2 else 1
                long_pts += pts
                details.append(f"🟦 Bull OB {dist_pct:.1f}% below → LONG +{pts}")
            else:
                # Bull OB above price = less relevant
                pass
        elif "bear" in ob_type.lower():
            # Bearish OB = resistance zone (above price = bad for LONG)
            if ob_mid > current_price:
                pts = 2 if dist_pct < 2 else 1
                long_pts -= pts
                short_pts += pts
                details.append(f"🟥 Bear OB {dist_pct:.1f}% above → LONG -{pts}, SHORT +{pts}")
            else:
                # Bear OB below price = less relevant (already broken)
                pass

    # ============================================
    # 4. Strong/Weak High/Low
    # ============================================
    max_possible += 2
    trailing = smc_data.get("trailing", {})

    strong_high = trailing.get("strong_high")
    weak_high = trailing.get("weak_high")
    strong_low = trailing.get("strong_low")
    weak_low = trailing.get("weak_low")

    if weak_high and current_price > 0:
        dist = abs(current_price - weak_high) / current_price * 100
        if dist < 2:
            long_pts += 2
            details.append(f"Weak High {dist:.1f}% away → LONG +2 (breakout likely)")

    if strong_high and current_price > 0:
        dist = abs(current_price - strong_high) / current_price * 100
        if dist < 2:
            long_pts -= 2
            short_pts += 1
            details.append(f"Strong High {dist:.1f}% away → LONG -2 (wall)")

    if weak_low and current_price > 0:
        dist = abs(current_price - weak_low) / current_price * 100
        if dist < 2:
            short_pts += 2
            details.append(f"Weak Low {dist:.1f}% away → SHORT +2 (breakdown likely)")

    if strong_low and current_price > 0:
        dist = abs(current_price - strong_low) / current_price * 100
        if dist < 2:
            short_pts -= 2
            long_pts += 1
            details.append(f"Strong Low {dist:.1f}% away → SHORT -2 (floor)")

    # ============================================
    # 5. Premium / Discount Zones
    # ============================================
    max_possible += 3
    zones = smc_data.get("zones", {})
    current_zone = zones.get("current_zone", "")

    if "Premium" in current_zone:
        long_pts -= 3
        details.append("Premium zone → LONG -3 (buying high)")
    elif "Discount" in current_zone:
        short_pts -= 3
        details.append("Discount zone → SHORT -3 (shorting low)")
    # Equilibrium = neutral, no score

    # ============================================
    # 6. EQH / EQL (liquidity magnets)
    # ============================================
    max_possible += 1
    equal_hl = smc_data.get("equal_hl", {})
    eqh_list = equal_hl.get("eqh", [])
    eql_list = equal_hl.get("eql", [])

    for eqh in (eqh_list if isinstance(eqh_list, list) else []):
        eqh_price = eqh.get("price", 0) if isinstance(eqh, dict) else 0
        if eqh_price > current_price and current_price > 0:
            dist = (eqh_price - current_price) / current_price * 100
            if dist < 3:
                long_pts += 1
                details.append(f"EQH {dist:.1f}% above → LONG +1 (liquidity magnet)")
                break

    for eql in (eql_list if isinstance(eql_list, list) else []):
        eql_price = eql.get("price", 0) if isinstance(eql, dict) else 0
        if eql_price < current_price and current_price > 0:
            dist = (current_price - eql_price) / current_price * 100
            if dist < 3:
                short_pts += 1
                details.append(f"EQL {dist:.1f}% below → SHORT +1 (liquidity magnet)")
                break

    # ============================================
    # Convert to percentages
    # ============================================
    total = abs(long_pts) + abs(short_pts)
    if total == 0:
        long_pct = 50
        short_pct = 50
    else:
        # Shift from raw score to 0-100
        net = long_pts - short_pts
        # Map net score to percentage: positive net = LONG bias
        # Scale: max realistic net is ~±12
        max_net = max(max_possible, 12)
        ratio = net / max_net  # -1 to +1
        long_pct = round(50 + ratio * 50, 1)
        long_pct = max(5, min(95, long_pct))  # clamp
        short_pct = round(100 - long_pct, 1)

    return {
        "long_pts": round(long_pts, 1),
        "short_pts": round(short_pts, 1),
        "long_pct": long_pct,
        "short_pct": short_pct,
        "details": details,
    }
