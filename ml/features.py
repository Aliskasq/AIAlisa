"""
ML Feature Extraction — bridge between indicators.py and XGBoost.

Two modes:
  1. extract_features_from_df(df)   → for TRAINING (all rows, returns DataFrame of features)
  2. extract_features_from_dict(d)  → for PREDICTION (single candle, from indicator dict)

Features are normalized / relative so the model generalizes across coins.
All features are price-agnostic (percentages, ratios, oscillator values).
"""

import numpy as np
import pandas as pd

# ─── FEATURE DEFINITIONS ────────────────────────────────────────────
# These are the columns XGBoost will see.  Order matters (must match train & predict).

FEATURE_NAMES = [
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

    # ── Candle shape ──
    "body_pct",               # abs(close - open) / close * 100
    "upper_wick_pct",         # (high - max(open,close)) / close * 100
    "lower_wick_pct",         # (min(open,close) - low) / close * 100
]

NUM_FEATURES = len(FEATURE_NAMES)  # should be 33


def _safe_div(a, b, default=0.0):
    """Safe division, returns default on zero/nan."""
    if b == 0 or (isinstance(b, float) and (np.isnan(b) or np.isinf(b))):
        return default
    result = a / b
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ML features from a DataFrame that already has indicator columns
    (output of calculate_binance_indicators's df).
    
    Returns: DataFrame with FEATURE_NAMES columns, same index as input.
    Rows with NaN features are NOT dropped (caller decides).
    """
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    
    # ── Trend ──
    out["price_vs_ema7_pct"] = (c - df["ema7"]) / c * 100
    out["price_vs_ema25_pct"] = (c - df["ema25"]) / c * 100
    out["price_vs_ema99_pct"] = (c - df["ema99"]) / c * 100
    out["ema7_vs_ema25_pct"] = (df["ema7"] - df["ema25"]) / c * 100
    out["ema25_vs_ema99_pct"] = (df["ema25"] - df["ema99"]) / c * 100

    # ── Momentum ──
    out["rsi14"] = df["rsi14"]
    out["rsi6"] = df["rsi6"]
    out["macd_hist_norm"] = df["macd_hist"] / c * 10000
    out["macd_line_norm"] = df["macd_line"] / c * 10000
    out["macd_signal_norm"] = df["macd_signal"] / c * 10000

    # ── Trend strength ──
    out["adx"] = df["adx"]
    out["di_plus"] = df["di_plus"]
    out["di_minus"] = df["di_minus"]
    out["di_diff"] = df["di_plus"] - df["di_minus"]

    # ── SuperTrend ──
    out["supertrend_dir"] = df["supertrend_dir"]
    out["st_distance_pct"] = (c - df["supertrend"]) / c * 100

    # ── Oscillators ──
    out["stoch_k"] = df["stoch_k"]
    out["stoch_d"] = df["stoch_d"]
    out["mfi"] = df["mfi"]
    out["cmf"] = df["cmf"]

    # ── Bollinger Bands ──
    bb_range = df["bb_upper"] - df["bb_lower"]
    bb_range_safe = bb_range.replace(0, np.nan)
    out["bb_pctb"] = (c - df["bb_lower"]) / bb_range_safe
    bb_mid_safe = df["bb_mid"].replace(0, np.nan)
    out["bb_width_pct"] = bb_range / bb_mid_safe * 100

    # ── Volume ──
    obv_10_ago = df["obv"].shift(10)
    obv_10_abs = obv_10_ago.abs().replace(0, np.nan)
    out["obv_roc_10"] = (df["obv"] - obv_10_ago) / obv_10_abs * 100
    out["obv_above_sma"] = (df["obv"] > df["obv_sma20"]).astype(int)

    # ── Volatility ──
    out["atr_pct"] = df["atr14"] / c * 100
    out["ttm_squeeze_on"] = df["ttm_squeeze_on"].astype(int)
    out["ttm_mom_norm"] = df["ttm_mom"] / c * 10000

    # ── Advanced momentum ──
    out["ccmi"] = df["ccmi"]
    out["ccmi_vs_signal"] = df["ccmi"] - df["ccmi_signal"]
    out["imi"] = df["imi"]
    out["rsi_mom"] = df["rsi_mom"]

    # ── Candle shape ──
    out["body_pct"] = (df["close"] - df["open"]).abs() / c * 100
    body_top = df[["close", "open"]].max(axis=1)
    body_bottom = df[["close", "open"]].min(axis=1)
    out["upper_wick_pct"] = (df["high"] - body_top) / c * 100
    out["lower_wick_pct"] = (body_bottom - df["low"]) / c * 100

    # Replace inf with NaN
    out = out.replace([np.inf, -np.inf], np.nan)

    return out[FEATURE_NAMES]


def extract_features_from_dict(indicators: dict) -> np.ndarray:
    """
    Extract feature vector from a single indicator dict
    (as returned by calculate_binance_indicators's first element).
    
    Returns: 1D numpy array of shape (NUM_FEATURES,).
    """
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

    features = np.array([
        # Trend
        (c - ema7) / c * 100,
        (c - ema25) / c * 100,
        (c - ema99) / c * 100,
        (ema7 - ema25) / c * 100,
        (ema25 - ema99) / c * 100,

        # Momentum
        float(indicators.get("rsi14", 50)),
        float(indicators.get("rsi6", 50)),
        macd_hist / c * 10000,
        macd_line / c * 10000,
        macd_signal / c * 10000,

        # Trend strength
        float(indicators.get("adx", 20)),
        float(indicators.get("di_plus", 0)),
        float(indicators.get("di_minus", 0)),
        float(indicators.get("di_plus", 0)) - float(indicators.get("di_minus", 0)),

        # SuperTrend
        st_dir,
        (c - st_price) / c * 100,

        # Oscillators
        float(indicators.get("stoch_k", 50)),
        float(indicators.get("stoch_d", 50)),
        float(indicators.get("mfi", 50)),
        float(indicators.get("cmf", 0)),

        # Bollinger Bands
        _safe_div(c - bb_lower, bb_range, 0.5),
        _safe_div(bb_range, bb_mid, 0) * 100,

        # Volume
        float(indicators.get("obv_roc_10", 0)),
        1.0 if "Accumulation" in str(indicators.get("obv_status", "")) else 0.0,

        # Volatility
        float(indicators.get("atr_pct", 0)) if "atr_pct" in indicators else float(indicators.get("atr14_value", 0)) / c * 100,
        1.0 if indicators.get("ttm_squeeze_on", False) else 0.0,
        float(indicators.get("ttm_mom", 0)) / c * 10000,

        # Advanced momentum
        float(indicators.get("ccmi", 0)),
        float(indicators.get("ccmi", 0)) - float(indicators.get("ccmi_signal", 0)),
        float(indicators.get("imi", 50)),
        float(indicators.get("rsi_mom", 50)),

        # Candle shape — not available from dict (only last candle summary)
        # Use zeros as fallback; these features are less critical for single prediction
        0.0,  # body_pct
        0.0,  # upper_wick_pct
        0.0,  # lower_wick_pct
    ], dtype=np.float64)

    # Clean NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def create_labels(df: pd.DataFrame, horizon: int = 4, threshold_pct: float = 0.3) -> pd.Series:
    """
    Create binary labels for training.
    
    Label = 1 (LONG) if price rises > threshold_pct% in `horizon` candles.
    Label = 0 (SHORT) if price drops > threshold_pct% in `horizon` candles.
    Label = NaN otherwise (neutral — excluded from training).
    
    Args:
        df: DataFrame with 'close' column
        horizon: candles to look ahead (4 for 4H = 16h, 4 for 1H = 4h, 4 for 15m = 1h)
        threshold_pct: minimum move % to count as signal (0.3%)
    
    Returns: Series with 1, 0, or NaN
    """
    future_close = df["close"].shift(-horizon)
    pct_change = (future_close - df["close"]) / df["close"] * 100
    
    labels = pd.Series(np.nan, index=df.index)
    labels[pct_change > threshold_pct] = 1   # LONG
    labels[pct_change < -threshold_pct] = 0  # SHORT
    
    return labels
