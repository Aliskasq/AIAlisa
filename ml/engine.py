"""
ML Prediction Engine — loads trained XGBoost models and provides predictions.

Usage in bot:
    from ml.engine import MLEngine
    ml = MLEngine()                     # loads 3 models from ml/models/
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

# Timeframe → model filename
TF_MODELS = {
    "4H": "xgb_4h.pkl",
    "1H": "xgb_1h.pkl",
    "15m": "xgb_15m.pkl",
}


class MLEngine:
    """Lightweight ML prediction engine.  ~25 MB RAM for 3 models."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or MODEL_DIR
        self.models = {}   # tf_key → XGBClassifier
        self._loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all available .pkl models.  Missing models are silently skipped."""
        try:
            import joblib
        except ImportError:
            logging.warning("⚠️ ML: joblib not installed — ML predictions disabled")
            return
        
        for tf_key, filename in TF_MODELS.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    self.models[tf_key] = joblib.load(path)
                    logging.info(f"🧠 ML: loaded {filename}")
                except Exception as e:
                    logging.error(f"❌ ML: failed to load {filename}: {e}")
        
        self._loaded = len(self.models) > 0
        if self._loaded:
            logging.info(f"🧠 ML Engine ready: {list(self.models.keys())} models loaded")
        else:
            logging.info("🧠 ML Engine: no models found yet (run trainer.py first)")
    
    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded."""
        return self._loaded
    
    def predict(self, indicators: dict, tf_key: str, smc_data: dict = None) -> dict:
        """
        Predict for a single timeframe.
        
        Args:
            indicators: dict from calculate_binance_indicators (first return value)
            tf_key: "4H", "1H", or "15m"
            smc_data: dict from analyze_smc (optional, for SMC features)
        
        Returns:
            {
                "direction": "LONG" or "SHORT",
                "long_prob": 0.74,         # probability of LONG
                "confidence": 74.0,        # confidence % in predicted direction
                "available": True
            }
            If model not available: {"available": False}
        """
        if tf_key not in self.models:
            return {"available": False}
        
        from ml.features import extract_features_from_dict, NUM_FEATURES
        
        try:
            features = extract_features_from_dict(indicators, smc_data=smc_data)
            features_2d = features.reshape(1, -1)
            
            # Handle model trained with fewer features (backward compat)
            model = self.models[tf_key]
            expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else features_2d.shape[1]
            if features_2d.shape[1] > expected:
                # Model was trained with fewer features — truncate to match
                features_2d = features_2d[:, :expected]
            
            proba = model.predict_proba(features_2d)[0]
            
            # proba[0] = SHORT prob, proba[1] = LONG prob
            long_prob = float(proba[1])
            short_prob = float(proba[0])
            
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
                    "4H": {"direction": "LONG", "long_prob": 0.74, "confidence": 74.0},
                    "1H": {"direction": "LONG", "long_prob": 0.68, "confidence": 68.0},
                    "15m": {"direction": "SHORT", "long_prob": 0.45, "confidence": 55.0},
                },
                "weighted_long_pct": 67.0,   # 4H×40% + 1H×35% + 15m×25%
                "weighted_short_pct": 33.0,
                "direction": "LONG",
                "consensus": True,           # all TFs agree on direction
            }
        """
        tf_inputs = {"4H": ind_4h, "1H": ind_1h, "15m": ind_15m}
        per_tf = {}
        weighted_long = 0.0
        total_weight = 0.0
        
        for tf_key, ind in tf_inputs.items():
            if ind is None:
                continue
            # Pass matching SMC data for this timeframe
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
        
        # Check consensus (all available TFs agree on direction)
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
        
        Example output:
            🧠 ML 4ч: LONG 74%
            🧠 ML 1ч: LONG 68%
            🧠 ML 15м: SHORT 55% ⚠️
            🧠 ML: LONG 67% (взвеш: 4H×40% + 1H×35% + 15m×25%)
            ✅ AI + ML: КОНСЕНСУС
        """
        if not ml_result.get("available"):
            return ""
        
        lines = []
        per_tf = ml_result.get("per_tf", {})
        
        # Per-TF labels
        tf_labels = {"4H": "4ч", "1H": "1ч", "15m": "15м"}
        
        # Determine AI-like direction for divergence check
        weighted_dir = ml_result.get("direction", "LONG")
        
        for tf_key in ["4H", "1H", "15m"]:
            if tf_key not in per_tf:
                continue
            r = per_tf[tf_key]
            tf_label = tf_labels.get(tf_key, tf_key)
            conf = r["confidence"]
            d = r["direction"]
            
            # ⚠️ if this TF disagrees with weighted direction
            warn = " ⚠️" if d != weighted_dir else ""
            lines.append(f"🧠 ML {tf_label}: {d} {conf:.0f}%{warn}")
        
        # Weighted summary
        wl = ml_result["weighted_long_pct"]
        ws = ml_result["weighted_short_pct"]
        lines.append(f"🧠 ML: {weighted_dir} {max(wl, ws):.0f}% (взвеш: 4H×40% + 1H×35% + 15m×25%)")
        
        return "\n".join(lines)
    
    def format_consensus_line(self, ml_result: dict, ai_direction: str) -> str:
        """
        Format consensus line: does ML agree with AI?
        
        Returns:
            "✅ AI + ML: КОНСЕНСУС" or "⚠️ AI + ML: РАСХОЖДЕНИЕ — ML против"
        """
        if not ml_result.get("available"):
            return "ℹ️ ML в режиме наблюдения"
        
        ml_dir = ml_result.get("direction", "")
        if ml_dir == ai_direction:
            return "✅ AI + ML: КОНСЕНСУС"
        else:
            return f"⚠️ AI + ML: РАСХОЖДЕНИЕ — ML против ({ml_dir})"
