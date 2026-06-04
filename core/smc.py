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
                     confluence_filter: bool = False,
                     swing_high_level: float = None,
                     swing_low_level: float = None,
                     swing_high_per_bar: np.ndarray = None,
                     swing_low_per_bar: np.ndarray = None,
                     strict_luxalgo: bool = True) -> Tuple[List[Dict], List[Dict], int]:
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
    # Build a per-bar event map keyed by DETECTION bar (not pivot bar).
    # Pine: pivot at bar X is detected at bar X+size (when leg() changes).
    # The pivot level only becomes available for crossover checks from that bar.
    pivot_at_bar = {}
    for p in raw_pivots:
        detection_bar = p["index"] + size  # bar when Pine detects this pivot
        if detection_bar < n:
            if detection_bar not in pivot_at_bar:
                pivot_at_bar[detection_bar] = []
            pivot_at_bar[detection_bar].append(p)

    for i in range(n):
        # Update pivots that occur at this bar
        if i in pivot_at_bar:
            for p in pivot_at_bar[i]:
                if p["type"] == "low":
                    last_low["last_price"] = last_low["price"]
                    last_low["price"] = p["price"]
                    last_low["index"] = p["index"]
                    last_low["crossed"] = False
                    # Retroactive crossover: check if close already crossed below
                    # this level between pivot bar and detection bar.
                    # In extreme downtrends, price moves past pivot level before
                    # detection (bar + size), so the normal crossover check misses it.
                    if not internal:
                        _retro_level = p["price"]
                        for _rb in range(p["index"] + 1, i):
                            _rc = closes[_rb]
                            _rc_prev = closes[_rb - 1] if _rb > 0 else float('inf')
                            if _rc_prev >= _retro_level and _rc < _retro_level:
                                # Found missed crossover — mark as already crossed
                                # and record the structure break
                                tag = "CHoCH" if trend == BULLISH else "BOS"
                                last_low["crossed"] = True
                                trend = BEARISH
                                structures.append({
                                    "type": tag,
                                    "bias": BEARISH,
                                    "price": _retro_level,
                                    "break_index": _rb,
                                    "pivot_index": p["index"],
                                })
                                break
                elif p["type"] == "high":
                    last_high["last_price"] = last_high["price"]
                    last_high["price"] = p["price"]
                    last_high["index"] = p["index"]
                    last_high["crossed"] = False
                    # Retroactive crossover: check if close already crossed above
                    if not internal:
                        _retro_level = p["price"]
                        for _rb in range(p["index"] + 1, i):
                            _rc = closes[_rb]
                            _rc_prev = closes[_rb - 1] if _rb > 0 else 0
                            if _rc_prev <= _retro_level and _rc > _retro_level:
                                tag = "CHoCH" if trend == BEARISH else "BOS"
                                last_high["crossed"] = True
                                trend = BULLISH
                                structures.append({
                                    "type": tag,
                                    "bias": BULLISH,
                                    "price": _retro_level,
                                    "break_index": _rb,
                                    "pivot_index": p["index"],
                                })
                                break

        # Candle shape filter (Pine: bullishBar / bearishBar)
        # Only active when confluence_filter=True AND internal=True
        # Pine: bullishBar := (high - max(close,open)) > (min(close,open) - low)
        if confluence_filter and internal:
            upper_wick = highs[i] - max(closes[i], opens[i])
            lower_wick = min(closes[i], opens[i]) - lows[i]
            bullish_bar = upper_wick > lower_wick
            bearish_bar = upper_wick < lower_wick
        else:
            bullish_bar = True
            bearish_bar = True

        # Check bullish break: close crosses above last high
        if last_high["price"] is not None and not last_high["crossed"]:
            # Crossover: previous close <= level AND current close > level
            prev_close = closes[i - 1] if i > 0 else 0
            if prev_close <= last_high["price"] and closes[i] > last_high["price"]:
                # Internal confluence filter:
                # Pine v5: extraCondition = internalHigh.currentLevel != swingHigh.currentLevel
                # Pine v5 float semantics: (x != na) → true, so when swingLevel is na
                # (before first swing pivot), extraCondition = true → break is ALLOWED.
                # Only skip when swing level IS set and equals internal level (real confluence).
                skip = False
                if internal:
                    if swing_high_per_bar is not None and i < len(swing_high_per_bar):
                        swing_level = swing_high_per_bar[i]
                        # Pine v5: x != na → true (allow), x == x → skip (confluence)
                        if not np.isnan(swing_level) and last_high["price"] == swing_level:
                            skip = True
                    elif swing_high_level is not None:
                        if last_high["price"] == swing_high_level:
                            skip = True
                    # else: no swing reference → allow (Pine v5: x != na → true)
                    # Candle shape filter (when confluence_filter enabled)
                    if not skip and not bullish_bar:
                        skip = True

                if not skip:
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
                # Internal confluence filter:
                # Pine v5: extraCondition = internalLow.currentLevel != swingLow.currentLevel
                # Pine v5 float semantics: (x != na) → true, so when swingLevel is na
                # (before first swing pivot), extraCondition = true → break is ALLOWED.
                # Only skip when swing level IS set and equals internal level (real confluence).
                skip = False
                if internal:
                    if swing_low_per_bar is not None and i < len(swing_low_per_bar):
                        swing_level = swing_low_per_bar[i]
                        # Pine v5: x != na → true (allow), x == x → skip (confluence)
                        if not np.isnan(swing_level) and last_low["price"] == swing_level:
                            skip = True
                    elif swing_low_level is not None:
                        if last_low["price"] == swing_low_level:
                            skip = True
                    # else: no swing reference → allow (Pine v5: x != na → true)
                    # Candle shape filter (when confluence_filter enabled)
                    if not skip and not bearish_bar:
                        skip = True

                if not skip:
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
                      mitigation: str = "highlow",
                      symbol: str = "",
                      label: str = "",
                      tf_label: str = "") -> List[Dict]:
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

    # ATR(200) for volatility filter — exact Pine ta.atr(200) = ta.rma(tr, 200)
    # Pine ta.rma: returns na for first (length-1) bars, SMA at bar length-1,
    # then alpha=1/length exponential from bar length onward.
    atr_length = 200
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # Pine RMA: na for bars 0..length-2, SMA at bar length-1, then EWM
    atr200 = np.full(n, np.nan)
    if n >= atr_length:
        atr200[atr_length - 1] = np.mean(tr[:atr_length])  # SMA seed
        alpha = 1.0 / atr_length
        for i in range(atr_length, n):
            atr200[i] = alpha * tr[i] + (1 - alpha) * atr200[i - 1]

    # Build parsed highs/lows arrays (exact LuxAlgo high-volatility bar logic)
    # When ATR is nan (first 199 bars), comparison is False → no swap (matches Pine)
    parsed_highs = np.copy(highs)
    parsed_lows = np.copy(lows)
    for i in range(n):
        if not np.isnan(atr200[i]) and (highs[i] - lows[i]) >= 2 * atr200[i]:
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

        # Check mitigation from the break bar (Pine: deleteOrderBlocks runs every bar including break bar)
        for k in range(break_idx, n):
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

        mit_idx = ob.get("mitigated_index", "")
        mit_info = f" @bar {mit_idx}" if ob["mitigated"] else ""
        sym_tag = f" [{symbol}]" if symbol else ""
        if tf_label in ("1D", "4H", ""):
            lbl_tag = f" {label}" if label else ""
            logging.info(
                f"📦 OB{lbl_tag} {'BULL' if ob['bias']==BULLISH else 'BEAR'}{sym_tag} idx={ob_idx} "
                f"[{ob['low']:.6f}-{ob['high']:.6f}] break@{break_idx} "
                f"mitigated={ob['mitigated']}{mit_info}"
            )
        order_blocks.append(ob)

    # Return unmitigated, most recent blocks (capped)
    # Exact Pine logic: no dedup, just keep the N most recent unmitigated OBs
    # Pine uses unshift (prepend) + slice(0, N) = most recent N
    # Python: order_blocks are in chronological order, so [-N:] = most recent N
    active = [ob for ob in order_blocks if not ob["mitigated"]]
    mitigated_count = len(order_blocks) - len(active)

    sym_tag = f" [{symbol}]" if symbol else ""
    if tf_label in ("1D", "4H", ""):
        lbl_tag = f" {label}" if label else ""
        logging.info(f"📦 OB{lbl_tag} summary{sym_tag}: total={len(order_blocks)} active={len(active)} mitigated={mitigated_count} max={max_blocks}")
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

    # ATR(200) — exact Pine ta.rma(tr, 200): na for first 199 bars, SMA seed, then EWM
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr_length = 200
    atr200 = np.full(n, np.nan)
    if n >= atr_length:
        atr200[atr_length - 1] = np.mean(tr[:atr_length])  # SMA seed
        alpha = 1.0 / atr_length
        for i in range(atr_length, n):
            atr200[i] = alpha * tr[i] + (1 - alpha) * atr200[i - 1]

    equals = []

    # Separate pivot highs and lows
    pivot_highs = [p for p in pivots if p["type"] == "high"]
    pivot_lows = [p for p in pivots if p["type"] == "low"]

    # Consecutive pivot highs
    for j in range(1, len(pivot_highs)):
        prev = pivot_highs[j - 1]
        curr = pivot_highs[j]
        idx = curr["index"]
        if idx < n and not np.isnan(atr200[idx]) and atr200[idx] > 0:
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
        if idx < n and not np.isnan(atr200[idx]) and atr200[idx] > 0:
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
                              swing_trend: int, swing_size: int = 50) -> Dict:
    """
    Port of LuxAlgo updateTrailingExtremes() + drawHighLowSwings().

    Exact Pine Script logic:
    - getCurrentStructure() sets trailing.top/bottom to new pivot price on each new swing
    - updateTrailingExtremes() runs EVERY bar: trailing.top = max(high, trailing.top),
      trailing.bottom = min(low, trailing.bottom)
    - So trailing values are running max/min that RESET at each new swing pivot

    Strong/Weak logic:
    - swingTrend == BEARISH → top is 'Strong High', bottom is 'Weak Low'
    - swingTrend == BULLISH → top is 'Weak High',   bottom is 'Strong Low'
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    # Build pivot events map: DETECTION bar → pivot
    # Pine: getCurrentStructure updates trailing on detection bar (pivot bar + swing_size)
    # swing_size is passed via the pivots' detection offset
    # We infer swing_size from the calling context (swing pivots use swing_size=50)
    # For safety, accept swing_size as parameter or infer from pivot spacing
    pivot_events = {}
    for p in swing_pivots:
        # Detection bar = pivot bar + swing_size
        detection_bar = p["index"] + swing_size
        if detection_bar < n:
            if detection_bar not in pivot_events:
                pivot_events[detection_bar] = []
            pivot_events[detection_bar].append(p)

    # Initialize trailing values
    trailing_high = highs[0] if n > 0 else 0.0
    trailing_low = lows[0] if n > 0 else 0.0
    trailing_high_idx = 0
    trailing_low_idx = 0

    # Walk through every bar (exact Pine Script execution order)
    # Pine: updateTrailingExtremes() runs BEFORE getCurrentStructure()
    # So trailing update happens first, then pivot reset OVERWRITES it.
    for i in range(n):
        # 1. updateTrailingExtremes runs FIRST (exact Pine execution order)
        if highs[i] >= trailing_high:
            trailing_high = float(highs[i])
            trailing_high_idx = i
        if lows[i] <= trailing_low:
            trailing_low = float(lows[i])
            trailing_low_idx = i

        # 2. getCurrentStructure resets trailing AFTER update (overwrites)
        # Use p["index"] (actual pivot bar) for visual line start, not detection bar i
        if i in pivot_events:
            for p in pivot_events[i]:
                if p["type"] == "high":
                    trailing_high = p["price"]
                    trailing_high_idx = p["index"]  # actual candle, not detection bar
                elif p["type"] == "low":
                    trailing_low = p["price"]
                    trailing_low_idx = p["index"]  # actual candle, not detection bar

    # Strong/Weak labels (exact LuxAlgo ternary logic)
    # Pine: swingTrend.bias == BEARISH ? 'Strong High' : 'Weak High'
    # Pine: swingTrend.bias == BULLISH ? 'Strong Low'  : 'Weak Low'
    # When bias is 0 (undefined), Pine ternary falls to the else branch:
    #   → 'Weak High' and 'Weak Low' (NOT bare "High"/"Low")
    if swing_trend == BEARISH:
        high_label = "Strong High"
        low_label = "Weak Low"
    else:
        # BULLISH or undefined (0) — both produce "Weak High"
        high_label = "Weak High"
        low_label = "Strong Low" if swing_trend == BULLISH else "Weak Low"

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
                ob_mitigation: str = "highlow",
                show_internal_obs: bool = None,
                show_swing_obs: bool = None,
                max_internal_obs: int = None,
                max_swing_obs: int = None,
                confluence_filter: bool = False,
                symbol: str = "",
                strict_luxalgo: bool = True) -> Dict:
    """
    Full SMC analysis — exact LuxAlgo port.

    Pipeline (exact LuxAlgo execution order):
    1. updateTrailingExtremes()                    → trailing max/min BEFORE pivots
    2. getCurrentStructure(swingsLengthInput)       → swing pivots (resets trailing)
    3. getCurrentStructure(5, internal=True)        → internal pivots
    4. getCurrentStructure(eqhlLength, eqhl=True)  → equal highs/lows pivots
    5. displayStructure(internal=True)             → internal BOS/CHoCH + internal OBs
    6. displayStructure()                          → swing BOS/CHoCH + swing OBs
    7. deleteOrderBlocks (mitigation check)
    8. Premium/Discount zones
    9. Fair Value Gaps

    LuxAlgo defaults: show_internal_obs=True, show_swing_obs=False.

    Returns dict with all SMC data + formatted text summary.
    """
    # Apply saved settings for OB display when not explicitly passed
    if show_internal_obs is None or show_swing_obs is None or max_internal_obs is None or max_swing_obs is None:
        _smc_cfg = get_smc_settings()
        if show_internal_obs is None:
            show_internal_obs = _smc_cfg.get("internal_obs", 5) > 0
        if show_swing_obs is None:
            show_swing_obs = _smc_cfg.get("swing_obs", 0) > 0
        if max_internal_obs is None:
            max_internal_obs = _smc_cfg.get("internal_obs", 5) or 5
        if max_swing_obs is None:
            max_swing_obs = _smc_cfg.get("swing_obs", 0) or 5
        # AIAlisa mode: use custom internal_size; TView mode: always LuxAlgo default (5)
        if not strict_luxalgo:
            internal_size = _smc_cfg.get("internal_size", 5)

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

        # Build per-bar swing levels for internal confluence filter
        # Pine: swingHigh.currentLevel / swingLow.currentLevel change on each new pivot detection
        # Pivot at bar idx is DETECTED at bar idx + swing_size
        swing_high_per_bar = np.full(n, np.nan)
        swing_low_per_bar = np.full(n, np.nan)
        swing_legs = _compute_legs(df["high"].values, df["low"].values, swing_size)

        current_sh = np.nan
        current_sl = np.nan
        # Build detection bar → pivot level map
        for p in swing_pivots:
            detection_bar = p["index"] + swing_size
            if detection_bar < n:
                if p["type"] == "high":
                    current_sh = p["price"]
                elif p["type"] == "low":
                    current_sl = p["price"]
            # Fill forward from detection_bar
        # Simpler approach: walk swing pivots in order and fill forward
        current_sh = np.nan
        current_sl = np.nan
        pivot_events = {}
        for p in swing_pivots:
            det = p["index"] + swing_size
            if det not in pivot_events:
                pivot_events[det] = []
            pivot_events[det].append(p)

        for i in range(n):
            if i in pivot_events:
                for p in pivot_events[i]:
                    if p["type"] == "high":
                        current_sh = p["price"]
                    elif p["type"] == "low":
                        current_sl = p["price"]
            swing_high_per_bar[i] = current_sh
            swing_low_per_bar[i] = current_sl

        # Debug: log swing structures (only 1D/4H to reduce noise)
        _smc_debug = tf_label in ("1D", "4H")
        sym_tag = f" [{symbol}]" if symbol else ""
        if _smc_debug:
            logging.info(f"🔍 SMC{sym_tag} {tf_label}: {n} candles, swing_structures={len(swing_structures)}, swing_pivots={len(swing_pivots)}")
            for s in swing_structures[-10:]:
                logging.info(f"  🔹 Swing {s['type']} {'BULL' if s['bias']==BULLISH else 'BEAR'} price={s['price']:.6f} pivot@{s['pivot_index']} break@{s['break_index']}")

        # ── 2. INTERNAL STRUCTURE (with confluence filter) ──
        internal_structures, internal_pivots, internal_trend = detect_structure(
            df, internal_size, internal=True,
            confluence_filter=confluence_filter,
            swing_high_per_bar=swing_high_per_bar,
            swing_low_per_bar=swing_low_per_bar,
            strict_luxalgo=strict_luxalgo,
        )

        if _smc_debug:
            logging.info(f"🔍 SMC{sym_tag} {tf_label}: internal_structures={len(internal_structures)}, internal_pivots={len(internal_pivots)}")

        # ── 3. EQUAL HIGHS / LOWS (using size=3 pivots, like LuxAlgo default) ──
        eqhl_legs = _compute_legs(df["high"].values, df["low"].values, 3)
        eqhl_pivots = _detect_pivots_from_legs(
            df["high"].values, df["low"].values, eqhl_legs, 3
        )
        equal_hl = find_equal_highs_lows(df, eqhl_pivots, threshold=0.1)

        # ── 4-5. ORDER BLOCKS ──
        # Exact LuxAlgo defaults: internal OBs ON (max 5), swing OBs OFF
        # OBs: use configured max blocks count
        internal_obs = []
        if show_internal_obs:
            internal_obs = find_order_blocks(
                df, internal_structures, max_blocks=max_internal_obs, mitigation=ob_mitigation, symbol=symbol, label="INT", tf_label=tf_label
            )
        swing_obs = []
        if show_swing_obs:
            swing_obs = find_order_blocks(
                df, swing_structures, max_blocks=max_swing_obs, mitigation=ob_mitigation, symbol=symbol, label="SWING", tf_label=tf_label
            )

        # ── 6. FAIR VALUE GAPS ──
        all_fvgs = find_fair_value_gaps(df)
        active_fvgs = [f for f in all_fvgs if not f["mitigated"]]

        # ── 7. TRAILING EXTREMES + STRONG/WEAK HIGH/LOW ──
        # When swing_trend is undefined (0, no swing structures detected),
        # fall back to internal_trend for Strong/Weak labels.
        # This handles coins in extreme trends where swing crossovers are missed
        # because price moves too far before swing pivots are confirmed (swing_size=50).
        _effective_trend = swing_trend if swing_trend != 0 else internal_trend
        trailing = compute_trailing_extremes(df, swing_pivots, _effective_trend, swing_size=swing_size)

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
        # Only include OBs that are enabled (matching LuxAlgo display toggles)
        all_obs = []
        if show_swing_obs:
            for ob in swing_obs:
                ob["_source"] = "Swing"
                all_obs.append(ob)
        if show_internal_obs:
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
            "n": n,  # total candles used for SMC — needed for chart overlay index offset
            "summary": summary,
        }

    except Exception as e:
        logging.error(f"❌ SMC analysis error ({tf_label}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"summary": f"[{tf_label}] SMC error: {str(e)[:100]}"}


def get_smc_mode() -> bool:
    """Load the current SMC mode from config. Returns strict_luxalgo flag."""
    try:
        from config import load_smc_mode
        return load_smc_mode()
    except Exception:
        return True  # default: TradingView mode


def get_smc_settings() -> dict:
    """Load full SMC settings (strict_luxalgo, internal_obs, swing_obs)."""
    try:
        from config import load_smc_settings
        return load_smc_settings()
    except Exception:
        return {"strict_luxalgo": True, "internal_obs": 5, "swing_obs": 0}


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
        swing_bias = last_swing.get("bias", 0)
        bars_since = last_swing.get("bars_since", 999)

        # Freshness multiplier
        if bars_since < 10:
            fresh_mult = 1.0
        elif bars_since < 30:
            fresh_mult = 0.5
        else:
            fresh_mult = 0.3

        base_pts = 3 if "CHoCH" in str(swing_type) else 2

        if swing_bias == BULLISH:
            pts = round(base_pts * fresh_mult, 1)
            long_pts += pts
            details.append(f"Swing {swing_type} Bullish ({bars_since}b ago) → LONG +{pts}")
        elif swing_bias == BEARISH:
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
        int_bias = last_internal.get("bias", 0)
        int_bars = last_internal.get("bars_since", 999)

        if int_bars < 10:
            int_fresh = 0.5
        elif int_bars < 30:
            int_fresh = 0.25
        else:
            int_fresh = 0.15

        int_base = 1.5 if "CHoCH" in str(int_type) else 1.0

        if int_bias == BULLISH:
            pts = round(int_base * int_fresh / 0.5, 1)  # normalized
            long_pts += pts
            details.append(f"Internal {int_type} Bullish ({int_bars}b) → LONG +{pts}")
        elif int_bias == BEARISH:
            pts = round(int_base * int_fresh / 0.5, 1)
            short_pts += pts
            details.append(f"Internal {int_type} Bearish ({int_bars}b) → SHORT +{pts}")

    # Internal vs Swing conflict
    if swing_structs and internal_structs:
        swing_bias = swing_structs[-1].get("bias", 0)
        int_bias = internal_structs[-1].get("bias", 0)
        if swing_bias == BULLISH and int_bias == BEARISH:
            long_pts -= 1
            details.append("⚠️ Swing↑ vs Internal↓ conflict → LONG -1")
        elif swing_bias == BEARISH and int_bias == BULLISH:
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
    equal_hl = smc_data.get("equal_hl", [])
    # equal_hl is a flat list of dicts with "type" = "EQH" or "EQL"
    if isinstance(equal_hl, list):
        eqh_list = [e for e in equal_hl if isinstance(e, dict) and e.get("type") == "EQH"]
        eql_list = [e for e in equal_hl if isinstance(e, dict) and e.get("type") == "EQL"]
    elif isinstance(equal_hl, dict):
        eqh_list = equal_hl.get("eqh", [])
        eql_list = equal_hl.get("eql", [])
    else:
        eqh_list = []
        eql_list = []

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
