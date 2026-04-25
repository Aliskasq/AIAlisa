"""
ML Feature Extraction — bridge between indicators.py/smc.py and XGBoost.

Two modes:
  1. extract_features_from_df(df)   → for TRAINING (all rows, returns DataFrame of features)
  2. extract_features_from_dict(d, smc=None)  → for PREDICTION (single candle, from indicator dict + optional SMC)

Features are normalized / relative so the model generalizes across coins.
All features are price-agnostic (percentages, ratios, oscillator values).

v2: Added ~55 dynamic/history features (50-candle context) + SMC features.
"""

import numpy as np
import pandas as pd

# ─── FEATURE DEFINITIONS ────────────────────────────────────────────
# These are the columns XGBoost will see.  Order matters (must match train & predict).

FEATURE_NAMES = [
    # ══════════════════════════════════════════
    # GROUP 1: SNAPSHOT (original 37 features)
    # ══════════════════════════════════════════

    # ── Trend (EMA relative positions) ──
    "price_vs_ema7_pct",      # (close - ema7) / close * 100
    "price_vs_ema25_pct",     # (close - ema25) / close * 100
    "price_vs_ema99_pct",     # (close - ema99) / close * 100
    "ema7_vs_ema25_pct",      # (ema7 - ema25) / close * 100
    "ema25_vs_ema99_pct",     # (ema25 - ema99) / close * 100

    # ── Momentum ──
    "rsi14",
    "rsi6",
    "macd_hist_norm",         # macd_hist / close * 10000 (normalized)
    "macd_line_norm",         # macd_line / close * 10000
    "macd_signal_norm",       # macd_signal / close * 10000

    # ── Trend strength ──
    "adx",
    "di_plus",
    "di_minus",
    "di_diff",                # di_plus - di_minus

    # ── SuperTrend ──
    "supertrend_dir",         # 1 (bull) or -1 (bear)
    "st_distance_pct",        # (close - supertrend) / close * 100

    # ── Oscillators ──
    "stoch_k",
    "stoch_d",
    "mfi",
    "cmf",

    # ── Bollinger Bands ──
    "bb_pctb",                # (close - bb_lower) / (bb_upper - bb_lower)
    "bb_width_pct",           # (bb_upper - bb_lower) / bb_mid * 100

    # ── Volume ──
    "obv_roc_10",             # OBV rate of change over 10 bars (%)
    "obv_above_sma",          # 1 if obv > obv_sma20, else 0

    # ── Volatility ──
    "atr_pct",                # atr14 / close * 100
    "ttm_squeeze_on",         # 1 if squeeze active, 0 otherwise
    "ttm_mom_norm",           # ttm_mom / close * 10000

    # ── Advanced momentum ──
    "ccmi",
    "ccmi_vs_signal",         # ccmi - ccmi_signal
    "imi",
    "rsi_mom",

    # ── Funding Rate ──
    "funding_rate_scaled",     # fundingRate * 10000 (0.01% → 1.0)
    "funding_rate_ma3_scaled", # average of last 3 funding payments * 10000
    "funding_rate_trend",      # (current - ma3) * 10000

    # ── Candle shape ──
    "body_pct",               # abs(close - open) / close * 100
    "upper_wick_pct",         # (high - max(open,close)) / close * 100
    "lower_wick_pct",         # (min(open,close) - low) / close * 100

    # ══════════════════════════════════════════
    # GROUP 2: DYNAMIC / HISTORY (50-candle context)
    # ══════════════════════════════════════════

    # ── EMA Dynamics ──
    "ema7_slope_50",           # EMA7 slope over 50 candles (%)
    "ema25_slope_50",          # EMA25 slope over 50 candles (%)
    "ema99_slope_50",          # EMA99 slope over 50 candles (%)
    "ema7_25_cross_bars",      # bars since golden/death cross (0 = no cross)
    "ema7_25_cross_dir_enc",   # 1=golden, -1=death, 0=none
    "price_above_ema7_ratio",  # % of last 50 candles above EMA7 (0-1)
    "price_above_ema25_ratio", # % of last 50 candles above EMA25 (0-1)
    "price_above_ema99_ratio", # % of last 50 candles above EMA99 (0-1)
    "ema_body_score_long",     # 0-9 body position score
    "ema_body_score_short",    # 0-9 body position score

    # ── MACD Dynamics ──
    "macd_hist_direction_enc", # 1=growing, -1=shrinking, 0=mixed
    "macd_bars_since_cross",   # bars since MACD/signal cross
    "macd_zero_cross_bars",    # bars since MACD crossed zero
    "macd_zero_cross_dir_enc", # 1=bull, -1=bear, 0=none
    "macd_hist_accel_norm",    # histogram acceleration / close * 10000
    "macd_hist_range_norm",    # (max - min) / close * 10000
    "macd_zero_crosses_50",    # count of zero crosses in 50 bars (choppiness)

    # ── OBV Dynamics ──
    "obv_roc_50",              # OBV rate of change over 50 bars
    "obv_trend_50_enc",        # 1=rising, -1=falling, 0=stable
    "obv_spike_enc",           # 1=spike detected, 0=no
    "obv_price_div_enc",       # 1=bullish div, -1=bearish div, 0=none

    # ── RSI Dynamics ──
    "rsi_trend_50_enc",        # 1=rising, -1=falling, 0=stable
    "rsi_min_50",              # RSI min over 50 candles
    "rsi_max_50",              # RSI max over 50 candles
    "rsi_avg_50",              # RSI avg over 50 candles
    "rsi_range_50",            # max - min (volatility of RSI)
    "rsi_time_overbought",     # candles in OB zone (out of 50)
    "rsi_time_oversold",       # candles in OS zone (out of 50)
    "rsi_price_div_enc",       # 1=bullish, -1=bearish, 0=none
    "rsi_pullback_peak",       # peak RSI before last major pullback
    "rsi_mom_bull_div_enc",    # 1=bullish RSI-Mom divergence, 0=none
    "rsi_mom_bear_div_enc",    # 1=bearish RSI-Mom divergence, 0=none

    # ── SuperTrend Dynamics ──
    "st_bars_since_flip",      # bars since last direction change
    "st_flips_50",             # number of flips in 50 bars (choppy = many)
    "st_bullish_ratio_50",     # % of bullish bars in last 50 (0-1)

    # ── Bollinger Bands Dynamics ──
    "bb_squeeze_enc",          # 1=squeeze active, 0=no
    "bb_expanding_enc",        # 1=expanding, 0=no
    "bb_walking_upper_enc",    # 1=walking upper band, 0=no
    "bb_walking_lower_enc",    # 1=walking lower band, 0=no

    # ── TTM Squeeze Dynamics ──
    "ttm_squeeze_bars",        # how many bars in squeeze
    "ttm_squeeze_fired_enc",   # 1=just fired (released), 0=no
    "ttm_mom_rising_enc",      # 1=momentum rising, 0=no
    "ttm_mom_direction_enc",   # 1=positive, -1=negative, 0=neutral

    # ── Ichimoku Dynamics ──
    "tk_cross_enc",            # 1=bullish TK cross, -1=bearish, 0=none
    "tk_cross_bars",           # bars since TK cross
    "cloud_thickness_pct",     # cloud thickness as % of price
    "future_cloud_enc",        # 1=bullish, -1=bearish, 0=neutral

    # ── ADX/DI Dynamics ──
    "adx_trend_enc",           # 1=rising, -1=falling, 0=flat
    "adx_avg_50",              # average ADX over 50 bars
    "adx_max_50",              # max ADX over 50 bars
    "di_cross_bars",           # bars since DI+/DI- cross
    "di_cross_dir_enc",        # 1=DI+ crossed above, -1=DI- crossed above, 0=none

    # ── StochRSI Dynamics ──
    "stoch_k_trend_enc",       # 1=rising, -1=falling, 0=flat
    "stoch_overbought_bars",   # bars in OB zone (last 15)
    "stoch_oversold_bars",     # bars in OS zone (last 15)
    "stoch_kd_crosses",        # K/D crosses in last 15 bars

    # ── MFI Dynamics ──
    "mfi_trend_enc",           # 1=rising, -1=falling, 0=flat
    "mfi_avg_20",              # average MFI over 20 bars
    "mfi_overbought_bars",     # bars in OB zone (last 20)
    "mfi_oversold_bars",       # bars in OS zone (last 20)

    # ── CMF Dynamics ──
    "cmf_trend_enc",           # 1=rising, -1=falling, 0=flat
    "cmf_avg_30",              # average CMF over 30 bars
    "cmf_positive_ratio_30",   # % positive bars in last 30 (0-1)

    # ══════════════════════════════════════════
    # GROUP 3: SMART MONEY CONCEPTS (SMC)
    # ══════════════════════════════════════════

    "smc_swing_trend_enc",     # 1=bullish, -1=bearish, 0=unknown
    "smc_internal_trend_enc",  # 1=bullish, -1=bearish, 0=unknown
    "smc_trend_conflict_enc",  # 1=swing and internal disagree, 0=agree
    "smc_last_swing_type_enc", # 1=CHoCH, 0=BOS
    "smc_last_swing_bias_enc", # 1=bullish, -1=bearish, 0=none
    "smc_last_swing_bars",     # bars since last swing structure
    "smc_last_int_bias_enc",   # 1=bullish, -1=bearish, 0=none
    "smc_last_int_bars",       # bars since last internal structure
    "smc_bull_ob_dist_pct",    # distance to nearest bullish OB (%)  — 0 if none
    "smc_bear_ob_dist_pct",    # distance to nearest bearish OB (%) — 0 if none
    "smc_bull_fvg_dist_pct",   # distance to nearest bullish FVG (%)
    "smc_bear_fvg_dist_pct",   # distance to nearest bearish FVG (%)
    "smc_eqh_dist_pct",        # distance to nearest EQH (%)
    "smc_eql_dist_pct",        # distance to nearest EQL (%)
    "smc_zone_enc",            # 1=discount, 0=equilibrium, -1=premium
    "smc_score_long_pct",      # SMC long score (0-100) from score_smc
    "smc_score_short_pct",     # SMC short score (0-100)
]

NUM_FEATURES = len(FEATURE_NAMES)  # ~92


# ─── ENCODING HELPERS ────────────────────────────────────────────

def _enc_trend(val):
    """Encode trend strings: rising/growing→1, falling/shrinking→-1, else→0."""
    if not val or not isinstance(val, str):
        return 0
    v = val.lower()
    if v in ("rising", "growing", "bullish", "golden", "bull", "positive"):
        return 1
    if v in ("falling", "shrinking", "bearish", "death", "bear", "negative"):
        return -1
    return 0


def _enc_bool(val):
    """Encode bool/truthy to 1/0."""
    return 1 if val else 0


def _parse_funding_from_dict(indicators: dict) -> tuple:
    """Parse funding rate from indicator dict.
    
    Handles both formats:
      - numeric: indicators["funding_rate"] = 0.0001 (from ML training)
      - string:  indicators["funding_rate"] = "0.0100% → 0.0150% → 0.0250%" (from bot)
    
    Returns: (current_rate, ma3_rate) as floats (decimal, e.g. 0.0001)
    """
    raw = indicators.get("funding_rate", 0)
    ma3 = indicators.get("funding_rate_ma3", None)
    
    if isinstance(raw, (int, float)):
        current = float(raw)
        if ma3 is not None:
            return current, float(ma3)
        return current, current
    
    if isinstance(raw, str) and "→" in raw:
        try:
            parts = [p.strip().replace("%", "") for p in raw.split("→")]
            rates = [float(p) / 100 for p in parts]
            current = rates[-1] if rates else 0.0
            avg = sum(rates) / len(rates) if rates else 0.0
            return current, avg
        except (ValueError, IndexError):
            return 0.0, 0.0
    
    if isinstance(raw, str):
        try:
            return float(raw.replace("%", "")) / 100, 0.0
        except ValueError:
            return 0.0, 0.0
    
    return 0.0, 0.0


def _safe_div(a, b, default=0.0):
    """Safe division, returns default on zero/nan."""
    if b == 0 or (isinstance(b, float) and (np.isnan(b) or np.isinf(b))):
        return default
    result = a / b
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def _safe_float(val, default=0.0):
    """Safe float conversion."""
    try:
        v = float(val) if val is not None else default
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ─── MAIN EXTRACTION: from indicator dict (PREDICTION) ────────────

def extract_features_from_dict(indicators: dict, smc_data: dict = None) -> np.ndarray:
    """
    Extract feature vector from a single indicator dict + optional SMC data.
    
    Args:
        indicators: dict from calculate_binance_indicators (single candle)
        smc_data: dict from analyze_smc (optional, for SMC features)
    
    Returns: 1D numpy array of shape (NUM_FEATURES,).
    """
    if not indicators:
        return np.zeros(NUM_FEATURES)
    
    c = float(indicators.get("close", 0))
    if c <= 0:
        return np.zeros(NUM_FEATURES)
    
    ema7 = float(indicators.get("ema7", c))
    ema25 = float(indicators.get("ema25", c))
    ema99 = float(indicators.get("ema99", c))
    macd_line = float(indicators.get("macd_line", 0))
    macd_signal = float(indicators.get("macd_signal", 0))
    macd_hist = float(indicators.get("macd_hist", 0))
    st_price = float(indicators.get("supertrend_price", c))
    st_dir = int(indicators.get("supertrend_dir_raw", 1))
    bb_upper = float(indicators.get("bb_upper", c))
    bb_lower = float(indicators.get("bb_lower", c))
    bb_mid = float(indicators.get("bb_mid", c))
    bb_range = bb_upper - bb_lower
    funding, funding_ma3 = _parse_funding_from_dict(indicators)

    # ── GROUP 1: SNAPSHOT (37 features) ──
    snapshot = [
        # Trend
        (c - ema7) / c * 100,
        (c - ema25) / c * 100,
        (c - ema99) / c * 100,
        (ema7 - ema25) / c * 100,
        (ema25 - ema99) / c * 100,
        # Momentum
        _safe_float(indicators.get("rsi14", 50)),
        _safe_float(indicators.get("rsi6", 50)),
        macd_hist / c * 10000,
        macd_line / c * 10000,
        macd_signal / c * 10000,
        # Trend strength
        _safe_float(indicators.get("adx", 20)),
        _safe_float(indicators.get("di_plus", 0)),
        _safe_float(indicators.get("di_minus", 0)),
        _safe_float(indicators.get("di_plus", 0)) - _safe_float(indicators.get("di_minus", 0)),
        # SuperTrend
        st_dir,
        (c - st_price) / c * 100,
        # Oscillators
        _safe_float(indicators.get("stoch_k", 50)),
        _safe_float(indicators.get("stoch_d", 50)),
        _safe_float(indicators.get("mfi", 50)),
        _safe_float(indicators.get("cmf", 0)),
        # Bollinger Bands
        _safe_div(c - bb_lower, bb_range, 0.5),
        _safe_div(bb_range, bb_mid, 0) * 100,
        # Volume
        _safe_float(indicators.get("obv_roc_10", 0)),
        1.0 if "Accumulation" in str(indicators.get("obv_status", "")) else 0.0,
        # Volatility
        _safe_float(indicators.get("atr_pct", 0)) if "atr_pct" in indicators
            else _safe_float(indicators.get("atr14_value", 0)) / c * 100,
        1.0 if indicators.get("ttm_squeeze_on", False) else 0.0,
        _safe_float(indicators.get("ttm_mom", 0)) / c * 10000,
        # Advanced momentum
        _safe_float(indicators.get("ccmi", 0)),
        _safe_float(indicators.get("ccmi", 0)) - _safe_float(indicators.get("ccmi_signal", 0)),
        _safe_float(indicators.get("imi", 50)),
        _safe_float(indicators.get("rsi_mom", 50)),
        # Funding Rate
        funding * 10000,
        funding_ma3 * 10000,
        (funding - funding_ma3) * 10000,
        # Candle shape
        0.0,  # body_pct — not in dict
        0.0,  # upper_wick_pct
        0.0,  # lower_wick_pct
    ]

    # ── GROUP 2: DYNAMIC / HISTORY ──
    
    # EMA Dynamics
    ema7_25_cross_bars_raw = indicators.get("ema7_25_cross_bars")
    dynamics = [
        _safe_float(indicators.get("ema7_slope_50", 0)),
        _safe_float(indicators.get("ema25_slope_50", 0)),
        _safe_float(indicators.get("ema99_slope_50", 0)),
        _safe_float(ema7_25_cross_bars_raw, 0),
        _enc_trend(indicators.get("ema7_25_cross_dir")),
        _safe_float(indicators.get("price_above_ema7_count", 0)) / 50,
        _safe_float(indicators.get("price_above_ema25_count", 0)) / 50,
        _safe_float(indicators.get("price_above_ema99_count", 0)) / 50,
        _safe_float(indicators.get("ema_body_score_long", 0)),
        _safe_float(indicators.get("ema_body_score_short", 0)),
        
        # MACD Dynamics
        _enc_trend(indicators.get("macd_hist_direction")),
        _safe_float(indicators.get("macd_bars_since_cross"), 0),
        _safe_float(indicators.get("macd_zero_cross_bars"), 0),
        _enc_trend(indicators.get("macd_zero_cross_dir")),
        _safe_float(indicators.get("macd_hist_accel", 0)) / c * 10000 if c > 0 else 0,
        (_safe_float(indicators.get("macd_hist_max_50", 0)) - _safe_float(indicators.get("macd_hist_min_50", 0))) / c * 10000 if c > 0 else 0,
        _safe_float(indicators.get("macd_zero_crosses_50", 0)),
        
        # OBV Dynamics
        _safe_float(indicators.get("obv_roc_50", 0)),
        _enc_trend(indicators.get("obv_trend_50")),
        _enc_bool(indicators.get("obv_spike")),
        _enc_trend(indicators.get("obv_price_divergence")),  # "bullish"→1, "bearish"→-1
        
        # RSI Dynamics
        _enc_trend(indicators.get("rsi_trend_50")),
        _safe_float(indicators.get("rsi_min_50", 50)),
        _safe_float(indicators.get("rsi_max_50", 50)),
        _safe_float(indicators.get("rsi_avg_50", 50)),
        _safe_float(indicators.get("rsi_max_50", 50)) - _safe_float(indicators.get("rsi_min_50", 50)),
        _safe_float(indicators.get("rsi_time_overbought", 0)),
        _safe_float(indicators.get("rsi_time_oversold", 0)),
        _enc_trend(indicators.get("rsi_price_divergence")),
        _safe_float(indicators.get("rsi_pullback_peak", 0)),
        _enc_bool(indicators.get("rsi_mom_bull_div")),
        _enc_bool(indicators.get("rsi_mom_bear_div")),
        
        # SuperTrend Dynamics
        _safe_float(indicators.get("st_bars_since_flip", 0)),
        _safe_float(indicators.get("st_flips_50", 0)),
        _safe_float(indicators.get("st_bullish_bars_50", 0)) / 50,
        
        # Bollinger Bands Dynamics
        _enc_bool(indicators.get("bb_squeeze")),
        _enc_bool(indicators.get("bb_expanding")),
        _enc_bool(indicators.get("bb_walking_upper")),
        _enc_bool(indicators.get("bb_walking_lower")),
        
        # TTM Squeeze Dynamics
        _safe_float(indicators.get("ttm_squeeze_bars", 0)),
        _enc_bool(indicators.get("ttm_squeeze_fired")),
        _enc_bool(indicators.get("ttm_mom_rising")),
        _enc_trend(indicators.get("ttm_mom_direction")),
        
        # Ichimoku Dynamics
        _enc_trend(indicators.get("tk_cross")),
        _safe_float(indicators.get("tk_cross_bars"), 0),
        _safe_float(indicators.get("cloud_thickness_pct", 0)),
        _enc_trend(indicators.get("future_cloud")),
        
        # ADX/DI Dynamics
        _enc_trend(indicators.get("adx_trend")),
        _safe_float(indicators.get("adx_avg_50", 20)),
        _safe_float(indicators.get("adx_max_50", 20)),
        _safe_float(indicators.get("di_cross_bars"), 0),
        _enc_trend(indicators.get("di_cross_dir")),
        
        # StochRSI Dynamics
        _enc_trend(indicators.get("stoch_k_trend")),
        _safe_float(indicators.get("stoch_overbought_bars_15", 0)),
        _safe_float(indicators.get("stoch_oversold_bars_15", 0)),
        _safe_float(indicators.get("stoch_kd_crosses_15", 0)),
        
        # MFI Dynamics
        _enc_trend(indicators.get("mfi_trend_20")),
        _safe_float(indicators.get("mfi_avg_20", 50)),
        _safe_float(indicators.get("mfi_overbought_bars_20", 0)),
        _safe_float(indicators.get("mfi_oversold_bars_20", 0)),
        
        # CMF Dynamics
        _enc_trend(indicators.get("cmf_trend_30")),
        _safe_float(indicators.get("cmf_avg_30", 0)),
        _safe_float(indicators.get("cmf_positive_bars_30", 0)) / 30,
    ]

    # ── GROUP 3: SMC FEATURES ──
    smc_features = _extract_smc_features(smc_data, c)

    all_features = snapshot + dynamics + smc_features
    
    result = np.array(all_features, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    assert len(result) == NUM_FEATURES, f"Feature count mismatch: {len(result)} vs {NUM_FEATURES}"
    return result


def _extract_smc_features(smc_data: dict, current_price: float) -> list:
    """Extract SMC features from analyze_smc output. Returns list of 17 floats."""
    if not smc_data or "swing_structures" not in smc_data:
        return [0.0] * 17
    
    BULLISH, BEARISH = 1, -1  # match smc.py constants
    
    # Swing trend
    swing_trend = smc_data.get("swing_trend", 0)
    swing_trend_enc = 1 if swing_trend == BULLISH else (-1 if swing_trend == BEARISH else 0)
    
    # Internal trend
    internal_trend = smc_data.get("internal_trend", 0)
    internal_trend_enc = 1 if internal_trend == BULLISH else (-1 if internal_trend == BEARISH else 0)
    
    # Conflict
    conflict = 1 if (swing_trend != 0 and internal_trend != 0 and swing_trend != internal_trend) else 0
    
    # Last swing structure
    swing_structs = smc_data.get("swing_structures", [])
    if swing_structs:
        last_s = swing_structs[-1]
        last_swing_type = 1 if "CHoCH" in str(last_s.get("type", "")) else 0
        last_swing_bias = 1 if last_s.get("bias") == BULLISH else (-1 if last_s.get("bias") == BEARISH else 0)
        last_swing_bars = _safe_float(last_s.get("bars_since", 0))
    else:
        last_swing_type, last_swing_bias, last_swing_bars = 0, 0, 0
    
    # Last internal structure
    int_structs = smc_data.get("internal_structures", [])
    if int_structs:
        last_i = int_structs[-1]
        last_int_bias = 1 if last_i.get("bias") == BULLISH else (-1 if last_i.get("bias") == BEARISH else 0)
        last_int_bars = _safe_float(last_i.get("bars_since", 0))
    else:
        last_int_bias, last_int_bars = 0, 0
    
    # Order Blocks — nearest bullish and bearish
    all_obs = smc_data.get("swing_order_blocks", []) + smc_data.get("internal_order_blocks", [])
    bull_ob_dist = 0.0
    bear_ob_dist = 0.0
    if current_price > 0:
        for ob in all_obs:
            ob_mid = (ob.get("top", 0) + ob.get("bottom", 0)) / 2
            if ob_mid <= 0:
                continue
            dist = abs(current_price - ob_mid) / current_price * 100
            if dist > 10:
                continue
            ob_type = str(ob.get("type", "")).lower()
            if "bull" in ob_type and (bull_ob_dist == 0 or dist < bull_ob_dist):
                bull_ob_dist = dist
            elif "bear" in ob_type and (bear_ob_dist == 0 or dist < bear_ob_dist):
                bear_ob_dist = dist
    
    # FVGs — nearest bullish and bearish
    fvgs = smc_data.get("fvgs", [])
    bull_fvg_dist = 0.0
    bear_fvg_dist = 0.0
    if current_price > 0:
        for f in (fvgs if isinstance(fvgs, list) else []):
            f_mid = (f.get("top", 0) + f.get("bottom", 0)) / 2
            if f_mid <= 0:
                continue
            dist = abs(current_price - f_mid) / current_price * 100
            if dist > 10:
                continue
            f_type = str(f.get("type", "")).lower()
            if "bull" in f_type and (bull_fvg_dist == 0 or dist < bull_fvg_dist):
                bull_fvg_dist = dist
            elif "bear" in f_type and (bear_fvg_dist == 0 or dist < bear_fvg_dist):
                bear_fvg_dist = dist
    
    # EQH / EQL — nearest
    equal_hl = smc_data.get("equal_hl", [])
    eqh_dist = 0.0
    eql_dist = 0.0
    if current_price > 0:
        eq_list = equal_hl if isinstance(equal_hl, list) else []
        for e in eq_list:
            if not isinstance(e, dict):
                continue
            e_price = e.get("price", 0)
            if e_price <= 0:
                continue
            dist = abs(current_price - e_price) / current_price * 100
            if dist > 10:
                continue
            if e.get("type") == "EQH" and (eqh_dist == 0 or dist < eqh_dist):
                eqh_dist = dist
            elif e.get("type") == "EQL" and (eql_dist == 0 or dist < eql_dist):
                eql_dist = dist
    
    # Premium / Discount zone
    zones = smc_data.get("zones", {})
    zone_str = str(zones.get("current_zone", ""))
    zone_enc = 1 if "Discount" in zone_str else (-1 if "Premium" in zone_str else 0)
    
    # SMC score (pre-computed)
    # We'll compute it inline if not provided
    smc_long_pct = 50.0
    smc_short_pct = 50.0
    try:
        from core.smc import score_smc
        score = score_smc(smc_data, current_price)
        smc_long_pct = score.get("long_pct", 50)
        smc_short_pct = score.get("short_pct", 50)
    except Exception:
        pass
    
    return [
        swing_trend_enc,
        internal_trend_enc,
        conflict,
        last_swing_type,
        last_swing_bias,
        last_swing_bars,
        last_int_bias,
        last_int_bars,
        bull_ob_dist,
        bear_ob_dist,
        bull_fvg_dist,
        bear_fvg_dist,
        eqh_dist,
        eql_dist,
        zone_enc,
        smc_long_pct,
        smc_short_pct,
    ]


# ─── TRAINING EXTRACTION: from DataFrame ────────────────────────

def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ML features from a DataFrame that already has indicator columns.
    Returns DataFrame with FEATURE_NAMES columns (only snapshot + partial dynamics).
    
    Note: Full dynamic features (50-candle history) and SMC features require
    the indicator dict, not raw DataFrame. For training, we compute what we can
    from the DataFrame and fill the rest with defaults. The trainer should
    use extract_features_from_dict when indicator dicts are available.
    """
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    
    # ── GROUP 1: SNAPSHOT ──
    out["price_vs_ema7_pct"] = (c - df["ema7"]) / c * 100
    out["price_vs_ema25_pct"] = (c - df["ema25"]) / c * 100
    out["price_vs_ema99_pct"] = (c - df["ema99"]) / c * 100
    out["ema7_vs_ema25_pct"] = (df["ema7"] - df["ema25"]) / c * 100
    out["ema25_vs_ema99_pct"] = (df["ema25"] - df["ema99"]) / c * 100
    out["rsi14"] = df["rsi14"]
    out["rsi6"] = df["rsi6"]
    out["macd_hist_norm"] = df["macd_hist"] / c * 10000
    out["macd_line_norm"] = df["macd_line"] / c * 10000
    out["macd_signal_norm"] = df["macd_signal"] / c * 10000
    out["adx"] = df["adx"]
    out["di_plus"] = df["di_plus"]
    out["di_minus"] = df["di_minus"]
    out["di_diff"] = df["di_plus"] - df["di_minus"]
    out["supertrend_dir"] = df["supertrend_dir"]
    out["st_distance_pct"] = (c - df["supertrend"]) / c * 100
    out["stoch_k"] = df["stoch_k"]
    out["stoch_d"] = df["stoch_d"]
    out["mfi"] = df["mfi"]
    out["cmf"] = df["cmf"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    bb_range_safe = bb_range.replace(0, np.nan)
    out["bb_pctb"] = (c - df["bb_lower"]) / bb_range_safe
    bb_mid_safe = df["bb_mid"].replace(0, np.nan)
    out["bb_width_pct"] = bb_range / bb_mid_safe * 100
    obv_10_ago = df["obv"].shift(10)
    obv_10_abs = obv_10_ago.abs().replace(0, np.nan)
    out["obv_roc_10"] = (df["obv"] - obv_10_ago) / obv_10_abs * 100
    out["obv_above_sma"] = (df["obv"] > df["obv_sma20"]).astype(int)
    out["atr_pct"] = df["atr14"] / c * 100
    out["ttm_squeeze_on"] = df["ttm_squeeze_on"].astype(int)
    out["ttm_mom_norm"] = df["ttm_mom"] / c * 10000
    out["ccmi"] = df["ccmi"]
    out["ccmi_vs_signal"] = df["ccmi"] - df["ccmi_signal"]
    out["imi"] = df["imi"]
    out["rsi_mom"] = df["rsi_mom"]
    if "funding_rate" in df.columns:
        out["funding_rate_scaled"] = df["funding_rate"] * 10000
        out["funding_rate_ma3_scaled"] = df["funding_rate_ma3"] * 10000
        out["funding_rate_trend"] = (df["funding_rate"] - df["funding_rate_ma3"]) * 10000
    else:
        out["funding_rate_scaled"] = 0.0
        out["funding_rate_ma3_scaled"] = 0.0
        out["funding_rate_trend"] = 0.0
    out["body_pct"] = (df["close"] - df["open"]).abs() / c * 100
    body_top = df[["close", "open"]].max(axis=1)
    body_bottom = df[["close", "open"]].min(axis=1)
    out["upper_wick_pct"] = (df["high"] - body_top) / c * 100
    out["lower_wick_pct"] = (body_bottom - df["low"]) / c * 100

    # ── GROUP 2: DYNAMIC FEATURES (computed from DataFrame rolling windows) ──
    
    # EMA slopes (50-bar change %)
    for ema_name in ["ema7", "ema25", "ema99"]:
        ema_50_ago = df[ema_name].shift(50)
        ema_50_safe = ema_50_ago.replace(0, np.nan)
        out[f"{ema_name}_slope_50"] = (df[ema_name] - ema_50_ago) / ema_50_safe * 100
    
    # EMA cross: bars since ema7 crossed ema25
    ema_cross_signal = (df["ema7"] > df["ema25"]).astype(int)
    ema_cross_change = ema_cross_signal.diff().abs()
    _cross_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if ema_cross_change.iloc[i] == 1:
            _cross_bars.iloc[i] = 0
        else:
            _cross_bars.iloc[i] = _cross_bars.iloc[i-1] + 1
    out["ema7_25_cross_bars"] = _cross_bars
    out["ema7_25_cross_dir_enc"] = ema_cross_signal.map({1: 1, 0: -1})  # 1=golden, -1=death
    
    # Price above EMA ratios (rolling 50)
    for ema_name, col in [("ema7", "price_above_ema7_ratio"), 
                           ("ema25", "price_above_ema25_ratio"),
                           ("ema99", "price_above_ema99_ratio")]:
        above = (c > df[ema_name]).astype(float)
        out[col] = above.rolling(50, min_periods=1).mean()
    
    # EMA body scores (simplified for DataFrame — use 0 as placeholder)
    out["ema_body_score_long"] = 0.0
    out["ema_body_score_short"] = 0.0
    
    # MACD dynamics
    macd_hist = df["macd_hist"]
    out["macd_hist_direction_enc"] = np.where(
        macd_hist > macd_hist.shift(1), 1,
        np.where(macd_hist < macd_hist.shift(1), -1, 0)
    )
    
    # MACD bars since line/signal cross
    macd_cross_signal = (df["macd_line"] > df["macd_signal"]).astype(int)
    macd_cross_change = macd_cross_signal.diff().abs()
    _macd_cross_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if macd_cross_change.iloc[i] == 1:
            _macd_cross_bars.iloc[i] = 0
        else:
            _macd_cross_bars.iloc[i] = _macd_cross_bars.iloc[i-1] + 1
    out["macd_bars_since_cross"] = _macd_cross_bars
    
    # MACD zero cross bars
    macd_above_zero = (df["macd_line"] > 0).astype(int)
    macd_zero_change = macd_above_zero.diff().abs()
    _macd_zero_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if macd_zero_change.iloc[i] == 1:
            _macd_zero_bars.iloc[i] = 0
        else:
            _macd_zero_bars.iloc[i] = _macd_zero_bars.iloc[i-1] + 1
    out["macd_zero_cross_bars"] = _macd_zero_bars
    out["macd_zero_cross_dir_enc"] = macd_above_zero.map({1: 1, 0: -1})
    
    # MACD histogram acceleration
    macd_hist_diff = macd_hist.diff()
    out["macd_hist_accel_norm"] = macd_hist_diff.diff() / c * 10000
    
    # MACD histogram range over 50 bars
    out["macd_hist_range_norm"] = (macd_hist.rolling(50, min_periods=1).max() - 
                                    macd_hist.rolling(50, min_periods=1).min()) / c * 10000
    
    # MACD zero crosses count in 50 bars
    macd_zero_crosses = (macd_hist.shift(1) * macd_hist < 0).astype(float)
    out["macd_zero_crosses_50"] = macd_zero_crosses.rolling(50, min_periods=1).sum()
    
    # OBV dynamics
    obv = df["obv"]
    obv_50_ago = obv.shift(50)
    obv_50_abs = obv_50_ago.abs().replace(0, np.nan)
    out["obv_roc_50"] = (obv - obv_50_ago) / obv_50_abs * 100
    
    # OBV trend (compare first half vs second half of 50-bar window)
    obv_ma25_now = obv.rolling(25, min_periods=1).mean()
    obv_ma25_prev = obv.shift(25).rolling(25, min_periods=1).mean()
    out["obv_trend_50_enc"] = np.where(obv_ma25_now > obv_ma25_prev * 1.01, 1,
                                        np.where(obv_ma25_now < obv_ma25_prev * 0.99, -1, 0))
    
    # OBV spike (>2 std dev)
    obv_diff = obv.diff()
    obv_std = obv_diff.rolling(50, min_periods=10).std()
    out["obv_spike_enc"] = (obv_diff.abs() > 2 * obv_std).astype(int)
    
    # OBV-price divergence (simplified)
    price_change_50 = c.pct_change(50)
    obv_change_50 = obv.pct_change(50)
    out["obv_price_div_enc"] = np.where(
        (price_change_50 < 0) & (obv_change_50 > 0), 1,   # bullish div
        np.where((price_change_50 > 0) & (obv_change_50 < 0), -1, 0)  # bearish div
    )
    
    # RSI dynamics
    rsi = df["rsi14"]
    rsi_diff = rsi.diff(3)
    out["rsi_trend_50_enc"] = np.where(rsi_diff > 2, 1, np.where(rsi_diff < -2, -1, 0))
    out["rsi_min_50"] = rsi.rolling(50, min_periods=1).min()
    out["rsi_max_50"] = rsi.rolling(50, min_periods=1).max()
    out["rsi_avg_50"] = rsi.rolling(50, min_periods=1).mean()
    out["rsi_range_50"] = out["rsi_max_50"] - out["rsi_min_50"]
    out["rsi_time_overbought"] = (rsi > 70).astype(float).rolling(50, min_periods=1).sum()
    out["rsi_time_oversold"] = (rsi < 30).astype(float).rolling(50, min_periods=1).sum()
    
    # RSI-price divergence (simplified)
    rsi_change_50 = rsi.diff(50)
    out["rsi_price_div_enc"] = np.where(
        (price_change_50 < 0) & (rsi_change_50 > 0), 1,
        np.where((price_change_50 > 0) & (rsi_change_50 < 0), -1, 0)
    )
    
    out["rsi_pullback_peak"] = 0.0  # requires scan logic, filled with 0 for training
    out["rsi_mom_bull_div_enc"] = 0.0
    out["rsi_mom_bear_div_enc"] = 0.0
    
    # SuperTrend dynamics
    st_dir = df["supertrend_dir"]
    st_change = st_dir.diff().abs()
    _st_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if st_change.iloc[i] != 0:
            _st_bars.iloc[i] = 0
        else:
            _st_bars.iloc[i] = _st_bars.iloc[i-1] + 1
    out["st_bars_since_flip"] = _st_bars
    out["st_flips_50"] = st_change.rolling(50, min_periods=1).sum()
    out["st_bullish_ratio_50"] = (st_dir == 1).astype(float).rolling(50, min_periods=1).mean()
    
    # BB dynamics
    bb_width = bb_range / bb_mid_safe * 100
    bb_width_pct20 = bb_width.rolling(50, min_periods=10).quantile(0.2)
    out["bb_squeeze_enc"] = (bb_width < bb_width_pct20).astype(int)
    out["bb_expanding_enc"] = (bb_width > bb_width.shift(3).rolling(3).mean()).astype(int)
    
    # BB walking bands
    out["bb_walking_upper_enc"] = ((c > df["bb_upper"] - bb_range * 0.1) & (c.shift(1) > df["bb_upper"].shift(1) - bb_range.shift(1) * 0.1)).astype(int)
    out["bb_walking_lower_enc"] = ((c < df["bb_lower"] + bb_range * 0.1) & (c.shift(1) < df["bb_lower"].shift(1) + bb_range.shift(1) * 0.1)).astype(int)
    
    # TTM dynamics
    ttm_sq = df["ttm_squeeze_on"].astype(int)
    _ttm_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if ttm_sq.iloc[i] == 1:
            _ttm_bars.iloc[i] = _ttm_bars.iloc[i-1] + 1
        else:
            _ttm_bars.iloc[i] = 0
    out["ttm_squeeze_bars"] = _ttm_bars
    out["ttm_squeeze_fired_enc"] = ((ttm_sq == 0) & (ttm_sq.shift(1) == 1)).astype(int)
    ttm_mom = df["ttm_mom"]
    out["ttm_mom_rising_enc"] = (ttm_mom > ttm_mom.shift(1)).astype(int)
    out["ttm_mom_direction_enc"] = np.where(ttm_mom > 0, 1, np.where(ttm_mom < 0, -1, 0))
    
    # Ichimoku dynamics
    if "tenkan" in df.columns and "kijun" in df.columns:
        tk_bullish = (df["tenkan"] > df["kijun"]).astype(int)
        tk_change = tk_bullish.diff().abs()
        out["tk_cross_enc"] = tk_bullish.map({1: 1, 0: -1})
        _tk_bars = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if tk_change.iloc[i] == 1:
                _tk_bars.iloc[i] = 0
            else:
                _tk_bars.iloc[i] = _tk_bars.iloc[i-1] + 1
        out["tk_cross_bars"] = _tk_bars
        senkou_a = df.get("senkou_a", pd.Series(0, index=df.index))
        senkou_b = df.get("senkou_b", pd.Series(0, index=df.index))
        cloud = (senkou_a - senkou_b).abs()
        out["cloud_thickness_pct"] = cloud / c * 100
        out["future_cloud_enc"] = np.where(senkou_a > senkou_b, 1, np.where(senkou_a < senkou_b, -1, 0))
    else:
        out["tk_cross_enc"] = 0
        out["tk_cross_bars"] = 0
        out["cloud_thickness_pct"] = 0.0
        out["future_cloud_enc"] = 0
    
    # ADX dynamics
    adx = df["adx"]
    adx_diff = adx.diff(5)
    out["adx_trend_enc"] = np.where(adx_diff > 2, 1, np.where(adx_diff < -2, -1, 0))
    out["adx_avg_50"] = adx.rolling(50, min_periods=1).mean()
    out["adx_max_50"] = adx.rolling(50, min_periods=1).max()
    
    # DI cross bars
    di_cross_signal = (df["di_plus"] > df["di_minus"]).astype(int)
    di_cross_change = di_cross_signal.diff().abs()
    _di_bars = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if di_cross_change.iloc[i] == 1:
            _di_bars.iloc[i] = 0
        else:
            _di_bars.iloc[i] = _di_bars.iloc[i-1] + 1
    out["di_cross_bars"] = _di_bars
    out["di_cross_dir_enc"] = di_cross_signal.map({1: 1, 0: -1})
    
    # StochRSI dynamics
    stoch_k = df["stoch_k"]
    stoch_k_diff = stoch_k.diff(3)
    out["stoch_k_trend_enc"] = np.where(stoch_k_diff > 3, 1, np.where(stoch_k_diff < -3, -1, 0))
    out["stoch_overbought_bars"] = (stoch_k > 80).astype(float).rolling(15, min_periods=1).sum()
    out["stoch_oversold_bars"] = (stoch_k < 20).astype(float).rolling(15, min_periods=1).sum()
    stoch_d = df["stoch_d"]
    stoch_kd_cross = ((stoch_k.shift(1) < stoch_d.shift(1)) & (stoch_k > stoch_d)) | \
                     ((stoch_k.shift(1) > stoch_d.shift(1)) & (stoch_k < stoch_d))
    out["stoch_kd_crosses"] = stoch_kd_cross.astype(float).rolling(15, min_periods=1).sum()
    
    # MFI dynamics
    mfi = df["mfi"]
    mfi_diff = mfi.diff(5)
    out["mfi_trend_enc"] = np.where(mfi_diff > 3, 1, np.where(mfi_diff < -3, -1, 0))
    out["mfi_avg_20"] = mfi.rolling(20, min_periods=1).mean()
    out["mfi_overbought_bars"] = (mfi > 80).astype(float).rolling(20, min_periods=1).sum()
    out["mfi_oversold_bars"] = (mfi < 20).astype(float).rolling(20, min_periods=1).sum()
    
    # CMF dynamics
    cmf = df["cmf"]
    cmf_diff = cmf.diff(5)
    out["cmf_trend_enc"] = np.where(cmf_diff > 0.02, 1, np.where(cmf_diff < -0.02, -1, 0))
    out["cmf_avg_30"] = cmf.rolling(30, min_periods=1).mean()
    cmf_positive = (cmf > 0).astype(float)
    out["cmf_positive_ratio_30"] = cmf_positive.rolling(30, min_periods=1).mean()
    
    # ── GROUP 3: SMC features (zeros for DataFrame training — SMC requires full analysis) ──
    for smc_col in [
        "smc_swing_trend_enc", "smc_internal_trend_enc", "smc_trend_conflict_enc",
        "smc_last_swing_type_enc", "smc_last_swing_bias_enc", "smc_last_swing_bars",
        "smc_last_int_bias_enc", "smc_last_int_bars",
        "smc_bull_ob_dist_pct", "smc_bear_ob_dist_pct",
        "smc_bull_fvg_dist_pct", "smc_bear_fvg_dist_pct",
        "smc_eqh_dist_pct", "smc_eql_dist_pct",
        "smc_zone_enc", "smc_score_long_pct", "smc_score_short_pct",
    ]:
        out[smc_col] = 0.0

    # Replace inf with NaN
    out = out.replace([np.inf, -np.inf], np.nan)

    return out[FEATURE_NAMES]


def create_labels(df: pd.DataFrame, horizon: int = 4, threshold_pct: float = 0.3) -> pd.Series:
    """
    Create binary labels for training.
    
    Label = 1 (LONG) if price rises > threshold_pct% in `horizon` candles.
    Label = 0 (SHORT) if price drops > threshold_pct% in `horizon` candles.
    Label = NaN otherwise (neutral — excluded from training).
    """
    future_close = df["close"].shift(-horizon)
    pct_change = (future_close - df["close"]) / df["close"] * 100
    
    labels = pd.Series(np.nan, index=df.index)
    labels[pct_change > threshold_pct] = 1   # LONG
    labels[pct_change < -threshold_pct] = 0  # SHORT
    
    return labels
