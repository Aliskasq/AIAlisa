"""
ML Prediction Engine — loads trained ensemble models and provides predictions.

Supports XGBoost + LightGBM ensemble with weighted soft voting.
Falls back to single model if only one is available (backward compatible).

Usage in bot:
    from ml.engine import MLEngine
    ml = MLEngine()                     # loads models from ml/models/
    ml.predict(indicators_dict, "4H")   # → {"direction": "LONG", "confidence": 0.74}
    ml.predict_all(ind_4h, ind_1h, ind_15m) # → weighted score + per-TF breakdown
"""

import os
import json
import logging
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Timeframe weights for combined score (matches analyzer.py)
TF_WEIGHTS = {"4H": 0.40, "1H": 0.35, "15m": 0.25}

# Singleton instance
_instance = None

def get_ml_engine() -> "MLEngine":
    """Get or create singleton MLEngine instance."""
    global _instance
    if _instance is None:
        _instance = MLEngine()
    return _instance

# Timeframe → model filenames (ensemble: xgb + lgb)
TF_MODELS = {
    "4H": {"xgb": "xgb_4h.pkl"},
    "1H": {"xgb": "xgb_1h.pkl"},
    "15m": {"xgb": "xgb_15m.pkl"},
}

# Legacy single-model filenames (backward compat)
TF_MODELS_LEGACY = {
    "4H": "xgb_4h.pkl",
    "1H": "xgb_1h.pkl",
    "15m": "xgb_15m.pkl",
}


class MLEngine:
    """ML prediction engine with ensemble support.  ~50 MB RAM for 6 models (3 TF × 2 models)."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or MODEL_DIR
        self.models = {}       # tf_key → {model_type: model_object}
        self.feature_masks = {}  # tf_key → {model_type: mask_or_None}
        self.ensemble_weights = {}  # tf_key → {model_type: weight}
        self._loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all available models (XGB + LGB per TF).  Missing models silently skipped."""
        try:
            import joblib
        except ImportError:
            logging.warning("⚠️ ML: joblib not installed — ML predictions disabled")
            return
        
        total_loaded = 0
        
        for tf_key, model_files in TF_MODELS.items():
            tf_models = {}
            tf_masks = {}
            
            for model_type, filename in model_files.items():
                path = os.path.join(self.model_dir, filename)
                if not os.path.exists(path):
                    continue
                try:
                    loaded = joblib.load(path)
                    if isinstance(loaded, dict) and "model" in loaded:
                        tf_models[model_type] = loaded["model"]
                        mask = loaded.get("feature_mask")
                        tf_masks[model_type] = np.array(mask, dtype=bool) if mask else None
                        n_feat = len(loaded.get("feature_names", []))
                        logging.info(f"🧠 ML: loaded {filename} ({model_type.upper()}, {n_feat} features)")
                    else:
                        # Legacy format — raw model, treat as xgb
                        tf_models[model_type] = loaded
                        tf_masks[model_type] = None
                        logging.info(f"🧠 ML: loaded {filename} (legacy format)")
                    total_loaded += 1
                except Exception as e:
                    logging.error(f"❌ ML: failed to load {filename}: {e}")
            
            # Legacy fallback: try old single-file format
            if not tf_models and tf_key in TF_MODELS_LEGACY:
                legacy_path = os.path.join(self.model_dir, TF_MODELS_LEGACY[tf_key])
                if os.path.exists(legacy_path):
                    try:
                        loaded = joblib.load(legacy_path)
                        if isinstance(loaded, dict) and "model" in loaded:
                            tf_models["xgb"] = loaded["model"]
                            mask = loaded.get("feature_mask")
                            tf_masks["xgb"] = np.array(mask, dtype=bool) if mask else None
                        else:
                            tf_models["xgb"] = loaded
                            tf_masks["xgb"] = None
                        logging.info(f"🧠 ML: loaded {TF_MODELS_LEGACY[tf_key]} (legacy single)")
                        total_loaded += 1
                    except Exception as e:
                        logging.error(f"❌ ML: failed to load legacy {TF_MODELS_LEGACY[tf_key]}: {e}")
            
            if tf_models:
                self.models[tf_key] = tf_models
                self.feature_masks[tf_key] = tf_masks
            
            # Load ensemble weights
            weights_path = os.path.join(self.model_dir, f"ensemble_weights_{tf_key.lower()}.json")
            if os.path.exists(weights_path):
                try:
                    with open(weights_path) as f:
                        self.ensemble_weights[tf_key] = json.load(f)
                    logging.info(f"🧠 ML: loaded ensemble weights for {tf_key}: {self.ensemble_weights[tf_key]}")
                except Exception:
                    pass
        
        self._loaded = total_loaded > 0
        if self._loaded:
            summary = []
            for tf_key in ["4H", "1H", "15m"]:
                if tf_key in self.models:
                    types = "+".join(mt.upper() for mt in self.models[tf_key])
                    summary.append(f"{tf_key}({types})")
            logging.info(f"🧠 ML Engine ready: {', '.join(summary)}")
        else:
            logging.info("🧠 ML Engine: no models found yet (run trainer.py first)")
    
    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded."""
        return self._loaded
    
    def predict(self, indicators: dict, tf_key: str, smc_data: dict = None) -> dict:
        """
        Predict for a single timeframe using ensemble.
        
        Returns:
            {
                "direction": "LONG" or "SHORT",
                "long_prob": 0.74,
                "confidence": 74.0,
                "available": True,
                "models_used": ["xgb", "lgb"],
                "per_model": {"xgb": 0.72, "lgb": 0.76},  # long_prob per model
            }
        """
        if tf_key not in self.models:
            return {"available": False}
        
        from ml.features import extract_features_from_dict, NUM_FEATURES
        
        try:
            features = extract_features_from_dict(indicators, smc_data=smc_data)
            features_2d = features.reshape(1, -1)
            
            tf_models = self.models[tf_key]
            tf_masks = self.feature_masks.get(tf_key, {})
            weights = self.ensemble_weights.get(tf_key, {})
            
            per_model = {}
            models_used = []
            
            for model_type, model in tf_models.items():
                try:
                    feat = features_2d.copy()
                    
                    # Apply feature mask if exists
                    mask = tf_masks.get(model_type)
                    if mask is not None and len(mask) == feat.shape[1]:
                        feat = feat[:, mask]
                    
                    # Handle model trained with fewer features
                    expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else feat.shape[1]
                    if feat.shape[1] > expected:
                        feat = feat[:, :expected]
                    
                    proba = model.predict_proba(feat)[0]
                    long_prob = float(proba[1])
                    per_model[model_type] = long_prob
                    models_used.append(model_type)
                except Exception as e:
                    logging.error(f"❌ ML predict error ({tf_key}/{model_type}): {e}")
            
            if not per_model:
                return {"available": False}
            
            # Weighted ensemble
            if len(per_model) > 1 and weights:
                total_w = sum(weights.get(mt, 1.0) for mt in per_model)
                long_prob = sum(
                    prob * weights.get(mt, 1.0) / total_w 
                    for mt, prob in per_model.items()
                )
            elif len(per_model) > 1:
                # Equal weights if no weights file
                long_prob = sum(per_model.values()) / len(per_model)
            else:
                long_prob = list(per_model.values())[0]
            
            short_prob = 1.0 - long_prob
            direction = "LONG" if long_prob >= 0.5 else "SHORT"
            confidence = max(long_prob, short_prob) * 100
            
            return {
                "available": True,
                "direction": direction,
                "long_prob": round(long_prob, 4),
                "confidence": round(confidence, 1),
                "models_used": models_used,
                "per_model": {mt: round(p, 4) for mt, p in per_model.items()},
            }
        except Exception as e:
            logging.error(f"❌ ML predict error ({tf_key}): {e}")
            return {"available": False}
    
    def predict_all(self, ind_4h: dict = None, ind_1h: dict = None, ind_15m: dict = None,
                     smc_data: dict = None) -> dict:
        """
        Predict across all timeframes and compute weighted score.
        
        Returns:
            {
                "available": True,
                "per_tf": {
                    "4H": {"direction": "LONG", "long_prob": 0.74, "confidence": 74.0, ...},
                    ...
                },
                "weighted_long_pct": 67.0,
                "weighted_short_pct": 33.0,
                "direction": "LONG",
                "consensus": True,
            }
        """
        tf_inputs = {"4H": ind_4h, "1H": ind_1h, "15m": ind_15m}
        per_tf = {}
        weighted_long = 0.0
        total_weight = 0.0
        
        for tf_key, ind in tf_inputs.items():
            if ind is None:
                continue
            tf_smc = smc_data.get(tf_key) if smc_data else None
            result = self.predict(ind, tf_key, smc_data=tf_smc)
            if result.get("available"):
                per_tf[tf_key] = result
                w = TF_WEIGHTS.get(tf_key, 0)
                weighted_long += result["long_prob"] * 100 * w
                total_weight += w
        
        if not per_tf:
            return {"available": False}
        
        if total_weight > 0:
            weighted_long_pct = round(weighted_long / total_weight, 1)
        else:
            weighted_long_pct = 50.0
        
        weighted_short_pct = round(100 - weighted_long_pct, 1)
        direction = "LONG" if weighted_long_pct >= 50 else "SHORT"
        
        directions = [r["direction"] for r in per_tf.values()]
        consensus = len(set(directions)) == 1
        
        return {
            "available": True,
            "per_tf": per_tf,
            "weighted_long_pct": weighted_long_pct,
            "weighted_short_pct": weighted_short_pct,
            "direction": direction,
            "consensus": consensus,
        }
    
    def format_for_push(self, ml_result: dict) -> str:
        """
        Format ML predictions for Telegram push message.
        Per-TF + weighted summary.
        """
        if not ml_result.get("available"):
            return ""
        
        lines = []
        per_tf = ml_result.get("per_tf", {})
        tf_labels = {"4H": "4ч", "1H": "1ч", "15m": "15м"}
        weighted_dir = ml_result.get("direction", "LONG")
        
        for tf_key in ["4H", "1H", "15m"]:
            if tf_key not in per_tf:
                continue
            r = per_tf[tf_key]
            tf_label = tf_labels.get(tf_key, tf_key)
            conf = r["confidence"]
            d = r["direction"]
            
            warn = " ⚠️" if d != weighted_dir else ""
            
            # Show per-model breakdown if ensemble
            models_used = r.get("models_used", [])
            per_model = r.get("per_model", {})
            if len(models_used) > 1:
                model_parts = []
                for mt in models_used:
                    p = per_model.get(mt, 0)
                    mt_dir = "L" if p >= 0.5 else "S"
                    mt_conf = max(p, 1-p) * 100
                    model_parts.append(f"{mt.upper()}:{mt_dir}{mt_conf:.0f}%")
                model_info = f" [{' '.join(model_parts)}]"
            else:
                model_info = ""
            
            lines.append(f"🧠 ML {tf_label}: {d} {conf:.0f}%{model_info}{warn}")
        
        wl = ml_result["weighted_long_pct"]
        ws = ml_result["weighted_short_pct"]
        lines.append(f"🧠 ML: {weighted_dir} {max(wl, ws):.0f}% (взвеш: 4H×40% + 1H×35% + 15m×25%)")
        
        return "\n".join(lines)
    
    def format_consensus_line(self, ml_result: dict, ai_direction: str) -> str:
        """
        Format consensus line: does ML agree with AI?
        """
        if not ml_result.get("available"):
            return "ℹ️ ML в режиме наблюдения"
        
        ml_dir = ml_result.get("direction", "")
        if ml_dir == ai_direction:
            return "✅ AI + ML: КОНСЕНСУС"
        else:
            return f"⚠️ AI + ML: РАСХОЖДЕНИЕ — ML против ({ml_dir})"
