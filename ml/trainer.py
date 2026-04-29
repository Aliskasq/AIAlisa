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
        "limit": 1500,       # 1500 candles = ~250 days (150 warmup + 1350 training)
        "requests": 1,
        "horizon": 4,        # look-ahead: 4 candles = 16h
        "threshold": 0.5,    # min move % for label (0.5% filters noise)
        "model_files": {"xgb": "xgb_4h.pkl", "lgb": "lgb_4h.pkl", "cat": "cat_4h.pkl"},
    },
    "1H": {
        "interval": "1h",
        "limit": 1500,       # 1500 candles = ~62 days
        "requests": 1,
        "horizon": 4,        # 4 candles = 4h
        "threshold": 0.5,    # min move % for label
        "model_files": {"xgb": "xgb_1h.pkl", "lgb": "lgb_1h.pkl", "cat": "cat_1h.pkl"},
    },
    "15m": {
        "interval": "15m",
        "limit": 1500,       # 1500 candles = ~15 days
        "requests": 1,
        "horizon": 4,        # 4 candles = 1h
        "threshold": 0.3,    # 15m keeps 0.3% (smaller moves are real on short TF)
        "model_files": {"xgb": "xgb_15m.pkl", "lgb": "lgb_15m.pkl", "cat": "cat_15m.pkl"},
    },
}

# XGBoost hyperparameters — conservative to avoid overfitting on 540 pairs
XGB_PARAMS = {
    "n_estimators": 250,       # balanced: more trees but fits 2 GB server
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

# LightGBM hyperparameters — similar regularization, faster training
LGB_PARAMS = {
    "n_estimators": 300,       # LGB trains faster → can afford more trees
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # recalculated per dataset
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,            # suppress LGB warnings
}

# CatBoost hyperparameters — handles categorical features natively, good regularization
CAT_PARAMS = {
    "iterations": 300,
    "depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "l2_leaf_reg": 3.0,       # L2 regularization (stronger than default)
    "random_seed": 42,
    "thread_count": 1,        # single-threaded to save RAM
    "verbose": 0,             # suppress training output
    "task_type": "CPU",
    "auto_class_weights": "Balanced",  # handles class imbalance
}

# Rate limiting — trainer uses max 500 weight/min (leaves rest for bot)
# Each klines request ≈ 10 weight → max ~50 requests/min → ~1.2s between requests
API_DELAY_SEC = 0.5          # delay between API calls within batch (800 candles = weight 5)
MAX_WEIGHT_TRAINER = 1000    # max weight per minute for trainer
BATCH_SIZE = 100             # pairs to process before gc.collect()

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


async def fetch_funding_history(session: aiohttp.ClientSession, symbol: str,
                                 limit: int = 1000) -> list:
    """Fetch funding rate history for a symbol. Weight = 1.
    Returns list of {funding_time, funding_rate} dicts or empty list."""
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [
                {"funding_time": int(r["fundingTime"]), "funding_rate": float(r["fundingRate"])}
                for r in data
            ]
    except Exception:
        return []


def merge_funding_into_df(df: pd.DataFrame, funding_data: list) -> pd.DataFrame:
    """Merge funding rate history into candle DataFrame using merge_asof.

    Each candle gets the most recent funding rate at its open_time.
    Also calculates MA3 (average of last 3 funding payments = ~24h).
    """
    if not funding_data:
        df["funding_rate"] = 0.0
        df["funding_rate_ma3"] = 0.0
        return df

    # Build funding DataFrame with pre-calculated MA3
    fdf = pd.DataFrame(funding_data).sort_values("funding_time").reset_index(drop=True)
    fdf["funding_rate_ma3"] = fdf["funding_rate"].rolling(3, min_periods=1).mean()

    # merge_asof: for each candle, find the most recent funding event
    df = df.sort_values("open_time").reset_index(drop=True)
    df = pd.merge_asof(
        df, fdf[["funding_time", "funding_rate", "funding_rate_ma3"]],
        left_on="open_time", right_on="funding_time", direction="backward"
    )

    # Fill NaN (candles before first funding event)
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df["funding_rate_ma3"] = df["funding_rate_ma3"].fillna(0.0)

    # Drop helper column
    if "funding_time" in df.columns:
        df.drop(columns=["funding_time"], inplace=True)

    return df


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


def process_pair(candles: list, tf_key: str, config: dict,
                 funding_data: list = None) -> tuple:
    """
    Process a single pair: calculate indicators → SMC analysis → extract features → create labels.
    
    Returns: (features_df, labels_series) or (None, None) on failure.
    Both have NaN rows already dropped.
    """
    if len(candles) < 150:  # not enough data for indicators
        return None, None
    
    try:
        df = pd.DataFrame(candles)

        # Merge funding rate into candle df before indicator calculation
        if funding_data:
            df = merge_funding_into_df(df, funding_data)

        result = calculate_binance_indicators(df, tf_key)
        
        # calculate_binance_indicators returns (last_row_dict, df) or just dict
        if isinstance(result, tuple):
            last_row_dict, df_with_indicators = result
        else:
            logging.warning(f"Unexpected return type from calculate_binance_indicators: {type(result)}")
            return None, None

        # Carry funding columns through to feature extraction
        if "funding_rate" in df.columns:
            df_with_indicators["funding_rate"] = df["funding_rate"].values
            df_with_indicators["funding_rate_ma3"] = df["funding_rate_ma3"].values
        
        # SMC analysis — run on the raw candle DataFrame
        smc_data = None
        try:
            from core.smc import analyze_smc
            smc_data = analyze_smc(df, tf_key)
        except Exception:
            pass  # SMC is optional, features will be zeros if missing
        
        features = extract_features_from_df(df_with_indicators, smc_data=smc_data)
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


def _train_single_model(model_type: str, X_train, y_train, X_val, y_val,
                        scale_pos: float, feature_names: list) -> dict:
    """
    Train a single model (xgb or lgb) and return stats + model object.
    
    Returns: {
        "model": trained_model,
        "accuracy": float,
        "y_proba": np.array (probabilities on val set),
        "top_features": {name: importance, ...},
        "model_size_mb": float (after save),
        "report": str (classification report),
    }
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    if model_type == "xgb":
        from xgboost import XGBClassifier
        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = round(scale_pos, 2)
        logging.info(f"   🔵 Training XGBoost ({params['n_estimators']} trees, "
                    f"depth={params['max_depth']}, {X_train.shape[1]} features)...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    elif model_type == "lgb":
        from lightgbm import LGBMClassifier
        params = LGB_PARAMS.copy()
        params["scale_pos_weight"] = round(scale_pos, 2)
        logging.info(f"   🟢 Training LightGBM ({params['n_estimators']} trees, "
                    f"depth={params['max_depth']}, {X_train.shape[1]} features)...")
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric="logloss")
    
    elif model_type == "cat":
        from catboost import CatBoostClassifier
        params = CAT_PARAMS.copy()
        logging.info(f"   🟠 Training CatBoost ({params['iterations']} trees, "
                    f"depth={params['depth']}, {X_train.shape[1]} features)...")
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=30)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]  # LONG probability
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=['SHORT', 'LONG'])
    
    logging.info(f"   ✅ {model_type.upper()} validation accuracy: {accuracy*100:.1f}%")
    logging.info(f"   Classification report:\n{report}")
    
    # Feature importance (top 10)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    logging.info(f"   Top 10 features ({model_type.upper()}):")
    for idx in top_idx:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        logging.info(f"     {feat_name}: {importances[idx]:.4f}")
    
    top_features = {}
    for idx in top_idx[:5]:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        top_features[fname] = round(float(importances[idx]), 4)
    
    return {
        "model": model,
        "accuracy": round(accuracy * 100, 1),
        "y_proba": y_proba,
        "top_features": top_features,
        "report": report,
    }


async def train_timeframe(tf_key: str, config: dict, symbols: list,
                          session: aiohttp.ClientSession,
                          dry_run: bool = False) -> dict:
    """
    Train XGBoost + LightGBM ensemble for one timeframe on all symbols.
    Data fetched once, models trained sequentially to save RAM.
    
    Returns: dict with per-model stats + ensemble accuracy
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"🧠 Training {tf_key} ensemble (XGB + LGB) on {len(symbols)} pairs...")
    logging.info(f"   Config: {config['limit']} candles, horizon={config['horizon']}, "
                f"threshold={config['threshold']}%, requests={config['requests']}")
    
    start_time = time.time()
    
    # Phase 1: Collect features — batch API calls, sequential processing
    all_features = []
    all_labels = []
    pairs_ok = 0
    pairs_fail = 0
    total_samples = 0
    MAX_SAMPLES = 500_000  # hard cap — with 2GB swap we can afford more data
    
    FETCH_BATCH = 10  # 10 pairs per batch × 6 weight each (klines 5 + funding 1) ≈ 60 weight
    _first_errors_logged = 0
    
    for batch_start in range(0, len(symbols), FETCH_BATCH):
        batch_symbols = symbols[batch_start:batch_start + FETCH_BATCH]
        
        # Staggered fetch — 1.3s between requests to stay under 500 weight/min
        fetch_tasks = []
        for idx, sym in enumerate(batch_symbols):
            async def _fetch_one(_sym=sym, _delay=idx * API_DELAY_SEC):
                await asyncio.sleep(_delay)
                candles = await fetch_klines_multi(
                    session, _sym, config["interval"], config["limit"], config["requests"]
                )
                # Fetch funding history (weight = 1, limit = 1000 covers ~333 days)
                funding = await fetch_funding_history(session, _sym, limit=1000)
                await asyncio.sleep(0.3)  # small delay after funding request
                return _sym, candles, funding
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
            sym, candles, funding_data = result
            if not candles:
                pairs_fail += 1
                continue
            
            features, labels = process_pair(candles, tf_key, config, funding_data)
            if features is None:
                pairs_fail += 1
                del candles, funding_data
                continue
            
            # Subsample per pair if needed to fit all pairs in RAM
            max_per_pair = MAX_SAMPLES // len(symbols) if len(symbols) > 0 else 9999
            feat_vals = features.values.astype(np.float32)
            lab_vals = labels.values.astype(np.float32)
            if len(feat_vals) > max_per_pair:
                idx = np.random.RandomState(42 + pairs_ok).choice(len(feat_vals), max_per_pair, replace=False)
                feat_vals = feat_vals[idx]
                lab_vals = lab_vals[idx]
            
            all_features.append(feat_vals)
            all_labels.append(lab_vals)
            total_samples += len(feat_vals)
            pairs_ok += 1
            
            del candles, funding_data, features, labels, feat_vals, lab_vals
        
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
    
    # Balance classes
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
    
    # ── Temporal split (no data leakage) ──
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logging.info(f"   Temporal split: train={len(X_train)}, val={len(X_val)} (last 20% by time)")
    
    # Free full arrays (train/val are views, but X/y can be freed after copy)
    # Actually X_train/X_val are slices of X, so keep X alive
    
    import joblib
    from sklearn.metrics import accuracy_score
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_files = config["model_files"]
    model_stats = {}
    ensemble_probas = {}  # model_type → y_proba on val set
    
    # ── Phase 3: Train models sequentially ──
    for model_type in ["xgb", "lgb", "cat"]:
        try:
            result = _train_single_model(
                model_type, X_train, y_train, X_val, y_val,
                scale_pos, FEATURE_NAMES
            )
        except Exception as e:
            logging.error(f"   ❌ {model_type.upper()} training failed: {type(e).__name__}: {e}")
            model_stats[model_type] = {"error": str(e)}
            continue
        
        # Save model
        model_path = os.path.join(MODEL_DIR, model_files[model_type])
        save_obj = {
            "model": result["model"],
            "feature_names": FEATURE_NAMES,
            "feature_mask": None,
            "model_type": model_type,
        }
        joblib.dump(save_obj, model_path)
        model_size = os.path.getsize(model_path) / 1024 / 1024
        logging.info(f"   💾 Saved: {model_path} ({model_size:.1f} MB)")
        
        ensemble_probas[model_type] = result["y_proba"]
        model_stats[model_type] = {
            "accuracy": result["accuracy"],
            "top_features": result["top_features"],
            "model_size_mb": round(model_size, 1),
        }
        
        # Free model to save RAM before training next one
        del result["model"]
        del result
        gc.collect()
    
    # ── Phase 4: Ensemble evaluation ──
    ensemble_accuracy = None
    ensemble_weights = {}
    
    if len(ensemble_probas) >= 2:
        # Weighted average by accuracy (higher accuracy → more weight)
        total_acc = sum(model_stats[mt]["accuracy"] for mt in ensemble_probas)
        for mt in ensemble_probas:
            ensemble_weights[mt] = round(model_stats[mt]["accuracy"] / total_acc, 3)
        
        # Weighted soft voting
        ensemble_proba = np.zeros(len(y_val))
        for mt, proba in ensemble_probas.items():
            ensemble_proba += proba * ensemble_weights[mt]
        
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        ensemble_accuracy = round(accuracy_score(y_val, ensemble_pred) * 100, 1)
        
        logging.info(f"\n   🏆 ENSEMBLE ({' + '.join(mt.upper() for mt in ensemble_probas)}):")
        logging.info(f"   Weights: {', '.join(f'{mt.upper()}={w:.3f}' for mt, w in ensemble_weights.items())}")
        logging.info(f"   ✅ Ensemble accuracy: {ensemble_accuracy}%")
        
        # Compare: ensemble vs best single model
        best_single = max(model_stats[mt]["accuracy"] for mt in ensemble_probas)
        diff = ensemble_accuracy - best_single
        if diff > 0:
            logging.info(f"   📈 Ensemble beats best single model by +{diff:.1f}%")
        elif diff == 0:
            logging.info(f"   ➡️ Ensemble matches best single model")
        else:
            logging.info(f"   📉 Ensemble is {abs(diff):.1f}% worse than best single (check for overfitting)")
    elif len(ensemble_probas) == 1:
        # Only one model succeeded — use its accuracy as ensemble
        mt = list(ensemble_probas.keys())[0]
        ensemble_accuracy = model_stats[mt]["accuracy"]
        ensemble_weights = {mt: 1.0}
        logging.info(f"   ⚠️ Only {mt.upper()} trained — no ensemble possible")
    
    # Save ensemble weights for engine.py
    if ensemble_weights:
        weights_path = os.path.join(MODEL_DIR, f"ensemble_weights_{tf_key.lower()}.json")
        with open(weights_path, "w") as f:
            json.dump(ensemble_weights, f)
        logging.info(f"   💾 Ensemble weights saved: {weights_path}")
    
    elapsed = time.time() - start_time
    
    stats = {
        "tf": tf_key,
        "pairs": pairs_ok,
        "pairs_failed": pairs_fail,
        "samples_total": int(X.shape[0]),
        "samples_train": int(X_train.shape[0]),
        "samples_val": int(X_val.shape[0]),
        "long_pct": round((y==1).mean() * 100, 1),
        "models": model_stats,
        "ensemble_accuracy": ensemble_accuracy,
        "ensemble_weights": ensemble_weights,
        "elapsed_sec": round(elapsed, 1),
    }
    
    # Free memory
    del X, y, X_train, X_val, y_train, y_val, ensemble_probas
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
            models = stats.get("models", {})
            parts = []
            for mt in ["xgb", "lgb"]:
                if mt in models and "accuracy" in models[mt]:
                    parts.append(f"{mt.upper()}={models[mt]['accuracy']:.1f}%")
            ens = stats.get("ensemble_accuracy")
            if ens:
                parts.append(f"Ensemble={ens:.1f}%")
            logging.info(f"   {tf}: {' | '.join(parts)}, "
                        f"{stats['pairs']} pairs, {stats['samples_total']} samples, "
                        f"{stats['elapsed_sec']:.0f}s")
    
    # Save stats to JSON — merge with existing (don't overwrite other TFs)
    stats_path = os.path.join(MODEL_DIR, "train_stats.json")
    try:
        existing = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                existing = json.load(f)
        
        # Merge: keep stats from other TFs, update only what we trained now
        merged_tfs = existing.get("timeframes", {})
        merged_tfs.update(all_stats)
        
        with open(stats_path, "w") as f:
            json.dump({
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_elapsed_min": round(total_elapsed / 60, 1),
                "timeframes": merged_tfs,
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
