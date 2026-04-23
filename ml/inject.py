"""
ML Push Injector — inserts ML prediction lines into AI verdict text.

Takes AI-generated caption and injects 🧠 ML lines after each ⏱ timeframe line,
plus a summary block after the verdict.

Approved format:
    ⏱ 4ч: LONG 63% / SHORT 37% (EMA выше, MACD растёт, ADX 64)
    🧠 ML 4ч: LONG 74%
    
    ⏱ 1ч: LONG 60% / SHORT 40% (EMA выше, MACD растёт, ADX 53)
    🧠 ML 1ч: LONG 68%
    
    ⏱ 15м: LONG 63% / SHORT 37% (EMA выше, SuperTrend свежий)
    🧠 ML 15м: SHORT 55% ⚠️
    
    🏆 ВЕРДИКТ: LONG
    📊 AI: LONG 63% / SHORT 37%
    🧠 ML: LONG 58% (взвеш: 4H×50% + 1H×30% + 15m×20%)
    ✅ AI + ML: КОНСЕНСУС
"""

import re
import logging


# Map timeframe patterns in AI text → ML keys
_TF_PATTERNS = {
    "4H": [r"⏱\s*4[Hhч]", r"⏱\s*4\s*(?:час|Hour)"],
    "1H": [r"⏱\s*1[Hhч]", r"⏱\s*1\s*(?:час|Hour)"],
    "15m": [r"⏱\s*15[mмMМ]", r"⏱\s*15\s*(?:мин|min)"],
}

_TF_LABELS_RU = {"4H": "4ч", "1H": "1ч", "15m": "15м"}


def inject_ml_into_caption(ai_text: str, ml_result: dict) -> str:
    """
    Inject ML predictions into AI caption text.
    
    Args:
        ai_text: original AI verdict text (caption for Telegram)
        ml_result: output from MLEngine.predict_all()
    
    Returns: modified text with ML lines injected
    """
    if not ai_text or not ml_result or not ml_result.get("available"):
        return ai_text
    
    per_tf = ml_result.get("per_tf", {})
    if not per_tf:
        return ai_text
    
    lines = ai_text.split("\n")
    new_lines = []
    injected_tfs = set()
    
    for line in lines:
        new_lines.append(line)
        
        # Check if this line is a ⏱ timeframe line
        for tf_key, patterns in _TF_PATTERNS.items():
            if tf_key in per_tf and tf_key not in injected_tfs:
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Inject ML line right after this ⏱ line
                        r = per_tf[tf_key]
                        tf_label = _TF_LABELS_RU.get(tf_key, tf_key)
                        d = r["direction"]
                        conf = r["confidence"]
                        
                        # ⚠️ if ML disagrees with weighted ML direction
                        weighted_dir = ml_result.get("direction", d)
                        warn = " ⚠️" if d != weighted_dir else ""
                        
                        new_lines.append(f"🧠 ML {tf_label}: {d} {conf:.0f}%{warn}")
                        injected_tfs.add(tf_key)
                        break
    
    # Now inject summary block after VERDICT/ВЕРДИКТ line
    result_lines = []
    verdict_injected = False
    
    for line in new_lines:
        result_lines.append(line)
        
        # Look for the "Overall" / "Общий" line with LONG/SHORT percentages
        # This is where we add the ML summary
        if not verdict_injected and re.search(
            r"(?:Overall|Общий)[:\s]*(?:LONG|ЛОНГ)\s+\d+%", line, re.IGNORECASE
        ):
            wl = ml_result["weighted_long_pct"]
            ws = ml_result["weighted_short_pct"]
            d = ml_result["direction"]
            result_lines.append(
                f"🧠 ML: {d} {max(wl, ws):.0f}% (взвеш: 4H×50% + 1H×30% + 15m×20%)"
            )
            
            # Add consensus line — need AI direction
            ai_dir = _extract_ai_direction(ai_text)
            if ai_dir:
                if ai_dir == d:
                    result_lines.append("✅ AI + ML: КОНСЕНСУС")
                else:
                    result_lines.append(f"⚠️ AI + ML: РАСХОЖДЕНИЕ — ML против ({d})")
            
            verdict_injected = True
    
    # If we couldn't find the Overall line, append ML summary at the end
    if not verdict_injected:
        wl = ml_result["weighted_long_pct"]
        ws = ml_result["weighted_short_pct"]
        d = ml_result["direction"]
        result_lines.append("")
        result_lines.append(
            f"🧠 ML: {d} {max(wl, ws):.0f}% (взвеш: 4H×50% + 1H×30% + 15m×20%)"
        )
        ai_dir = _extract_ai_direction(ai_text)
        if ai_dir:
            if ai_dir == d:
                result_lines.append("✅ AI + ML: КОНСЕНСУС")
            else:
                result_lines.append(f"⚠️ AI + ML: РАСХОЖДЕНИЕ — ML против ({d})")
    
    return "\n".join(result_lines)


def _extract_ai_direction(text: str) -> str:
    """Extract AI direction from verdict text. Returns 'LONG', 'SHORT', or ''."""
    _verdict_map = {
        "ЛОНГ": "LONG", "ДЛГО": "LONG",
        "ШОРТ": "SHORT", "КОРОТКО": "SHORT",
    }
    m = re.search(
        r"(?:VERDICT|ВЕРДИКТ)[:\s]*(LONG|SHORT|ЛОНГ|ШОРТ|ДЛГО|КОРОТКО)",
        text, re.IGNORECASE
    )
    if m:
        raw = m.group(1).upper()
        return _verdict_map.get(raw, raw)
    return ""


def format_ml_for_prompt(ml_result: dict) -> str:
    """
    Format ML predictions as additional context for AI prompt.
    Added to the user prompt so AI sees ML opinion.
    
    Returns text block like:
        [ML MODEL PREDICTIONS (XGBoost, second opinion)]
        4H: LONG 74% | 1H: LONG 68% | 15m: SHORT 55%
        Weighted: LONG 67% (4H×50% + 1H×30% + 15m×20%)
        Consensus: NO (15m disagrees)
        Note: ML is statistical pattern recognition. Use as additional signal, not override.
    """
    if not ml_result or not ml_result.get("available"):
        return ""
    
    per_tf = ml_result.get("per_tf", {})
    if not per_tf:
        return ""
    
    tf_parts = []
    for tf_key in ["4H", "1H", "15m"]:
        if tf_key in per_tf:
            r = per_tf[tf_key]
            tf_parts.append(f"{tf_key}: {r['direction']} {r['confidence']:.0f}%")
    
    wl = ml_result["weighted_long_pct"]
    ws = ml_result["weighted_short_pct"]
    d = ml_result["direction"]
    consensus = "YES (all TFs agree)" if ml_result.get("consensus") else "NO (TFs disagree)"
    
    return (
        f"\n[ML MODEL PREDICTIONS (XGBoost, second opinion)]\n"
        f"{' | '.join(tf_parts)}\n"
        f"Weighted: {d} {max(wl, ws):.0f}% (4H×50% + 1H×30% + 15m×20%)\n"
        f"ML Consensus: {consensus}\n"
        f"Note: ML is statistical pattern recognition trained on 540+ pairs. "
        f"Use as additional confirmation signal. If ML strongly disagrees with indicators, mention it.\n"
    )
