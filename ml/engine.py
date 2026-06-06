"""
ML Prediction Engine — loads trained XGBoost models and provides predictions.

Usage in bot:
    from ml.engine import MLEngine
    ml = MLEngine()                     # loads models from ml/models/
    ml.predict(indicators_dict, "4H")   # → {"direction": "LONG", "confidence": 0.74}
    ml.predict_all(ind_4h, ind_1h, ind_15m) # → weighted score + per-TF breakdown
"""

import os
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

# Timeframe → model filenames
TF_MODELS = {
    "4H": {"xgb": "xgb_4h.pkl"},
    "1H": {"xgb": "xgb_1h.pkl"},
    "15m": {"xgb": "xgb_15m.pkl"},
}


class MLEngine:
    """ML prediction engine with XGBoost models.  ~25 MB RAM for 3 models (3 TF × 1 XGBoost each)."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or MODEL_DIR
        self.models = {}       # tf_key → {model_type: model_object}
        self.feature_masks = {}  # tf_key → {model_type: mask_or_None}
        self._loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all available XGBoost models (one per TF).  Missing models silently skipped."""
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
                        # Legacy format — raw model
                        tf_models[model_type] = loaded
                        tf_masks[model_type] = None
                        logging.info(f"🧠 ML: loaded {filename} (legacy format)")
                    total_loaded += 1
                except Exception as e:
                    logging.error(f"❌ ML: failed to load {filename}: {e}")
            
            if tf_models:
                self.models[tf_key] = tf_models
                self.feature_masks[tf_key] = tf_masks
        
        self._loaded = total_loaded > 0
        if self._loaded:
            summary = []
            for tf_key in ["4H", "1H", "15m"]:
                if tf_key in self.models:
                    summary.append(f"{tf_key}(XGB)")
            logging.info(f"🧠 ML Engine ready: {', '.join(summary)}")
        else:
            logging.info("🧠 ML Engine: no models found yet (run trainer.py first)")
    
    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded."""
        return self._loaded
    
    def predict(self, indicators: dict, tf_key: str, smc_data: dict = None) -> dict:
        """
        Predict for a single timeframe using XGBoost model.
        
        Returns:
            {
                "direction": "LONG" or "SHORT",
                "long_prob": 0.74,
                "confidence": 74.0,
                "available": True,
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
            
            # Get XGBoost model (only model type)
            model = tf_models.get("xgb")
            if model is None:
                return {"available": False}
            
            feat = features_2d.copy()
            
            # Apply feature mask if exists
            mask = tf_masks.get("xgb")
            if mask is not None and len(mask) == feat.shape[1]:
                feat = feat[:, mask]
            
            # Handle model trained with fewer features
            expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else feat.shape[1]
            if feat.shape[1] > expected:
                feat = feat[:, :expected]
            
            proba = model.predict_proba(feat)[0]
            long_prob = float(proba[1])
            short_prob = 1.0 - long_prob
            direction = "LONG" if long_prob >= 0.5 else "SHORT"
            confidence = max(long_prob, short_prob) * 100
            
            return {
                "available": True,
                "direction": direction,
                "long_prob": round(long_prob, 4),
                "confidence": round(confidence, 1),
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
            
            lines.append(f"🧠 ML {tf_label}: {d} {conf:.0f}%{warn}")
        
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
