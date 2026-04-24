#!/usr/bin/env python3
"""
ML Trainer — standalone script, runs via cron (Wed + Sun 04:00 UTC).

Trains 3 XGBoost classifiers (4H, 1H, 15m) on ALL 540+ Binance Futures pairs.
Sequential pair processing → peak RAM ~300-400 MB.

Usage:
    python -m ml.trainer              # train all 3 models
    python -m ml.trainer --tf 4h      # train single timeframe
    python -m ml.trainer --dry-run    # show stats without saving

Cron entry:
    0 4 * * 0,3 cd /path/to/AIAlisa && python -m ml.trainer >> ml/train.log 2>&1
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
import gc

import numpy as np
import pandas as pd
import aiohttp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import calculate_binance_indicators
from ml.features import extract_features_from_df, create_labels, FEATURE_NAMES, NUM_FEATURES

# ─── CONFIG ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

TIMEFRAME_CONFIG = {
    "4H": {
        "interval": "4h",
        "limit": 1500,       # 1500 candles = ~250 days
        "requests": 1,       # single request (Binance max = 1500)
        "horizon": 4,        # look-ahead: 4 candles = 16h
        "threshold": 0.3,    # min move % for label
        "model_file": "xgb_4h.pkl",
    },
    "1H": {
        "interval": "1h",
        "limit": 1500,       # 1500 candles = ~62 days
        "requests": 1,
        "horizon": 4,        # 4 candles = 4h
        "threshold": 0.3,
        "model_file": "xgb_1h.pkl",
    },
    "15m": {
        "interval": "15m",
        "limit": 4500,       # 3 × 1500 = ~46 days
        "requests": 3,       # 3 Binance requests stitched together
        "horizon": 4,        # 4 candles = 1h
        "threshold": 0.3,
        "model_file": "xgb_15m.pkl",
    },
}

# XGBoost hyperparameters — conservative to avoid overfitting on 540 pairs
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,        # L1 regularization
    "reg_lambda": 1.0,       # L2 regularization
    "scale_pos_weight": 1.0, # will be recalculated per dataset
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,             # single-threaded to save RAM
    "tree_method": "hist",   # memory-efficient
}

# Rate limiting — trainer uses max 500 weight/min (leaves rest for bot)
# Each klines request ≈ 10 weight → max ~50 requests/min → ~1.2s between requests
API_DELAY_SEC = 1.3          # delay between API calls within batch
MAX_WEIGHT_TRAINER = 500     # max weight per minute for trainer
BATCH_SIZE = 50              # pairs to process before gc.collect()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ML-TRAIN] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "train.log"),
                           mode="a", encoding="utf-8"),
    ]
)


async def fetch_klines_raw(session: aiohttp.ClientSession, symbol: str,
                           interval: str, limit: int = 1500,
                           end_time: int = None) -> list:
    """Fetch raw klines from Binance Futures.  Returns list of dicts or None."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if end_time:
        url += f"&endTime={end_time}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            # Track API weight — pause if too high (bot also uses weight)
            weight = resp.headers.get('X-MBX-USED-WEIGHT-1M', '0')
            weight_int = int(weight) if weight.isdigit() else 0
            
            # Hard pause: total weight approaching Binance limit (bot sends alert at 2350)
            if weight_int > 2200:
                wait_sec = 62 - (time.time() % 60)
                logging.warning(f"⚠️ API weight {weight_int}/2400 — pausing {wait_sec:.0f}s for minute reset")
                await asyncio.sleep(wait_sec)
            
            if resp.status == 429:
                retry = int(resp.headers.get("Retry-After", "30"))
                logging.warning(f"⚠️ 429 rate limited on {symbol}, waiting {retry}s...")
                await asyncio.sleep(retry + 1)
                return None
            if resp.status != 200:
                logging.warning(f"⚠️ Klines {symbol} {interval}: HTTP {resp.status}")
                return None
            raw = await resp.json()
            if not raw:
                logging.warning(f"⚠️ Klines {symbol} {interval}: empty response")
                return None
            return [
                {
                    "open_time": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
                for c in raw
            ]
    except asyncio.TimeoutError:
        logging.warning(f"⚠️ Timeout fetching {symbol} {interval}")
        return None
    except Exception as e:
        logging.warning(f"⚠️ Fetch error {symbol} {interval}: {type(e).__name__}: {e}")
        return None


async def fetch_klines_multi(session: aiohttp.ClientSession, symbol: str,
                              interval: str, total_limit: int,
                              num_requests: int) -> list:
    """Fetch klines that require multiple requests (e.g., 15m × 3).
    Stitches together oldest→newest with no overlap."""
    if num_requests == 1:
        data = await fetch_klines_raw(session, symbol, interval, total_limit)
        return data or []
    
    per_request = total_limit // num_requests  # 1500 each
    all_candles = []
    end_time = None  # start from latest
    
    for i in range(num_requests):
        await asyncio.sleep(API_DELAY_SEC)
        data = await fetch_klines_raw(session, symbol, interval, per_request, end_time)
        if not data:
            break
        all_candles = data + all_candles  # prepend (oldest first)
        end_time = data[0]["open_time"] - 1  # next batch ends before this
    
    # Remove duplicates by open_time
    seen = set()
    unique = []
    for c in all_candles:
        if c["open_time"] not in seen:
            seen.add(c["open_time"])
            unique.append(c)
    
    return sorted(unique, key=lambda x: x["open_time"])


async def get_all_futures_symbols(session: aiohttp.ClientSession) -> list:
    """Get all USDT perpetual futures symbols."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            symbols = []
            for s in data.get("symbols", []):
                if (s.get("contractType") == "PERPETUAL"
                        and s.get("quoteAsset") == "USDT"
                        and s.get("status") == "TRADING"):
                    symbols.append(s["symbol"])
            return sorted(symbols)
    except Exception as e:
        logging.error(f"❌ Failed to get symbols: {e}")
        return []


def process_pair(candles: list, tf_key: str, config: dict) -> tuple:
    """
    Process a single pair: calculate indicators → extract features → create labels.
    
    Returns: (features_df, labels_series) or (None, None) on failure.
    Both have NaN rows already dropped.
    """
    if len(candles) < 150:  # not enough data for indicators
        return None, None
    
    try:
        df = pd.DataFrame(candles)
        result = calculate_binance_indicators(df, tf_key)
        
        # calculate_binance_indicators returns (last_row_dict, df) or just dict
        if isinstance(result, tuple):
            _, df_with_indicators = result
        else:
            logging.warning(f"Unexpected return type from calculate_binance_indicators: {type(result)}")
            return None, None
        
        features = extract_features_from_df(df_with_indicators)
        labels = create_labels(df_with_indicators, 
                              horizon=config["horizon"],
                              threshold_pct=config["threshold"])
        
        # Drop NaN rows (from indicator warmup + future label lookahead)
        valid_mask = features.notna().all(axis=1) & labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        if len(features) < 50:  # too few valid samples
            return None, None
        
        return features, labels
    except KeyError as e:
        logging.warning(f"Missing column in df: {e}")
        return None, None
    except Exception as e:
        logging.warning(f"Process error: {type(e).__name__}: {e}")
        return None, None


async def train_timeframe(tf_key: str, config: dict, symbols: list,
                          session: aiohttp.ClientSession,
                          dry_run: bool = False) -> dict:
    """
    Train XGBoost for one timeframe on all symbols.
    Sequential processing: one pair at a time → collect features → train.
    
    Returns: dict with stats (accuracy, feature importance, etc.)
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"🧠 Training {tf_key} model on {len(symbols)} pairs...")
    logging.info(f"   Config: {config['limit']} candles, horizon={config['horizon']}, "
                f"threshold={config['threshold']}%, requests={config['requests']}")
    
    start_time = time.time()
    
    # Phase 1: Collect features — batch API calls (10 parallel), sequential processing
    all_features = []
    all_labels = []
    pairs_ok = 0
    pairs_fail = 0
    total_samples = 0
    
    FETCH_BATCH = 3   # small batches to stay within 500 weight/min
    _first_errors_logged = 0
    
    for batch_start in range(0, len(symbols), FETCH_BATCH):
        batch_symbols = symbols[batch_start:batch_start + FETCH_BATCH]
        
        # Staggered fetch — 1.3s between requests to stay under 500 weight/min
        fetch_tasks = []
        for idx, sym in enumerate(batch_symbols):
            async def _fetch_one(_sym=sym, _delay=idx * API_DELAY_SEC):
                await asyncio.sleep(_delay)
                return _sym, await fetch_klines_multi(
                    session, _sym, config["interval"], config["limit"], config["requests"]
                )
            fetch_tasks.append(_fetch_one())
        
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Sequential processing (indicator calc is CPU-bound)
        for result in results:
            if isinstance(result, Exception):
                pairs_fail += 1
                if _first_errors_logged < 3:
                    logging.warning(f"⚠️ Fetch exception: {type(result).__name__}: {result}")
                    _first_errors_logged += 1
                continue
            sym, candles = result
            if not candles:
                pairs_fail += 1
                continue
            
            features, labels = process_pair(candles, tf_key, config)
            if features is None:
                pairs_fail += 1
                del candles
                continue
            
            all_features.append(features.values)
            all_labels.append(labels.values)
            total_samples += len(features)
            pairs_ok += 1
            
            del candles, features, labels
        
        # Periodic GC + progress
        i = batch_start + len(batch_symbols)
        if i % BATCH_SIZE == 0 or i >= len(symbols):
            gc.collect()
            logging.info(f"   Progress: {i}/{len(symbols)} pairs, "
                        f"{total_samples} samples, {pairs_ok} ok / {pairs_fail} fail")
    
    if not all_features:
        logging.error(f"❌ No valid data for {tf_key}!")
        return {"error": "no data"}
    
    # Phase 2: Concatenate all features
    logging.info(f"   Concatenating {total_samples} samples from {pairs_ok} pairs...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    # Free the lists
    del all_features, all_labels
    gc.collect()
    
    logging.info(f"   Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    logging.info(f"   Labels: LONG={int((y==1).sum())} ({(y==1).mean()*100:.1f}%), "
                f"SHORT={int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")
    
    if dry_run:
        logging.info(f"   🏃 DRY RUN — skipping training")
        return {
            "tf": tf_key,
            "pairs": pairs_ok,
            "samples": int(X.shape[0]),
            "long_pct": round((y==1).mean() * 100, 1),
            "dry_run": True,
        }
    
    # Phase 3: Train XGBoost
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    
    # Balance classes
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
    
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = round(scale_pos, 2)
    
    # Train/validation split (80/20, stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logging.info(f"   Training XGBoost ({params['n_estimators']} trees, depth={params['max_depth']})...")
    
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Phase 4: Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logging.info(f"   ✅ Validation accuracy: {accuracy*100:.1f}%")
    logging.info(f"   Classification report:\n{classification_report(y_val, y_pred, target_names=['SHORT','LONG'])}")
    
    # Feature importance (top 10)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    logging.info(f"   Top 10 features:")
    for idx in top_idx:
        logging.info(f"     {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")
    
    # Phase 5: Save model
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, config["model_file"])
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path) / 1024 / 1024
    logging.info(f"   💾 Saved: {model_path} ({model_size:.1f} MB)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "tf": tf_key,
        "pairs": pairs_ok,
        "pairs_failed": pairs_fail,
        "samples_total": int(X.shape[0]),
        "samples_train": int(X_train.shape[0]),
        "samples_val": int(X_val.shape[0]),
        "long_pct": round((y==1).mean() * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "model_size_mb": round(model_size, 1),
        "elapsed_sec": round(elapsed, 1),
        "top_features": {FEATURE_NAMES[i]: round(float(importances[i]), 4) for i in top_idx[:5]},
    }
    
    # Free memory
    del X, y, X_train, X_val, y_train, y_val, model
    gc.collect()
    
    return stats


async def main(args):
    """Main training entry point."""
    logging.info("=" * 60)
    logging.info("🧠 AIAlisa ML Trainer started")
    logging.info(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    logging.info("=" * 60)
    
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Get all futures symbols
        symbols = await get_all_futures_symbols(session)
        if not symbols:
            logging.error("❌ Failed to get symbols!")
            return
        
        logging.info(f"📊 Found {len(symbols)} USDT perpetual pairs")
        
        # Determine which timeframes to train
        if args.tf:
            tf_key = args.tf.upper()
            if tf_key == "15M":
                tf_key = "15m"
            if tf_key not in TIMEFRAME_CONFIG:
                logging.error(f"❌ Unknown timeframe: {tf_key}")
                return
            timeframes = {tf_key: TIMEFRAME_CONFIG[tf_key]}
        else:
            timeframes = TIMEFRAME_CONFIG
        
        all_stats = {}
        for tf_key, config in timeframes.items():
            stats = await train_timeframe(tf_key, config, symbols, session, 
                                          dry_run=args.dry_run)
            all_stats[tf_key] = stats
            gc.collect()
        
    # Summary
    total_elapsed = time.time() - start
    logging.info(f"\n{'='*60}")
    logging.info(f"🏁 Training complete in {total_elapsed/60:.1f} minutes")
    for tf, stats in all_stats.items():
        if "error" in stats:
            logging.info(f"   {tf}: ❌ {stats['error']}")
        elif stats.get("dry_run"):
            logging.info(f"   {tf}: {stats['pairs']} pairs, {stats['samples']} samples (dry run)")
        else:
            logging.info(f"   {tf}: {stats['accuracy']:.1f}% acc, "
                        f"{stats['pairs']} pairs, {stats['samples_total']} samples, "
                        f"{stats['model_size_mb']:.1f} MB, {stats['elapsed_sec']:.0f}s")
    
    # Save stats to JSON
    stats_path = os.path.join(MODEL_DIR, "train_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump({
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_elapsed_min": round(total_elapsed / 60, 1),
                "timeframes": all_stats,
            }, f, indent=2)
        logging.info(f"   📊 Stats saved: {stats_path}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIAlisa ML Trainer")
    parser.add_argument("--tf", type=str, help="Train single timeframe (4h, 1h, 15m)")
    parser.add_argument("--dry-run", action="store_true", help="Collect data but don't train")
    args = parser.parse_args()
    
    asyncio.run(main(args))
