import json
import logging
import aiohttp
import asyncio
import math
import os
import time as _time
from typing import Optional
from pydantic import BaseModel
from config import (OPENROUTER_API_KEY, OPENROUTER_MODEL,
                     GROQ_API_KEYS, GEMINI_API_KEYS, KEY_ACCOUNT_LABELS,
                     load_ai_settings, save_ai_settings)



# Global lock — serializes ALL AI requests (auto-push + manual scan).
# Whoever acquires first goes first; 15s cooldown between releases prevents 429.
_ai_request_lock = asyncio.Lock()
_ai_last_request_time = 0  # timestamp of last completed AI request

# --- Multi-provider AI settings (loaded on first use) ---
_ai_settings = None

def _get_ai_settings():
    """Get cached AI settings, loading from disk on first call."""
    global _ai_settings
    if _ai_settings is None:
        _ai_settings = load_ai_settings()
    return _ai_settings

def _save_and_cache_settings(settings):
    """Save settings to disk and update cache."""
    global _ai_settings
    _ai_settings = settings
    save_ai_settings(settings)

def get_active_provider_info():
    """Return (provider, model, key_index) for external use (e.g. tg_listener)."""
    s = _get_ai_settings()
    provider = s["active_provider"]
    if provider == "openrouter":
        return provider, s["openrouter_model"], 0
    elif provider == "gemini":
        return provider, s["gemini_model"], s["active_key_index"]
    elif provider == "groq":
        return provider, s["groq_model"], s["active_key_index"]
    return provider, s.get("openrouter_model", ""), 0

def set_active_provider(provider, model=None, key_index=None):
    """Switch active provider/model/key and persist."""
    s = _get_ai_settings()
    s["active_provider"] = provider
    if model:
        s[f"{provider}_model"] = model
    if key_index is not None:
        s["active_key_index"] = key_index
    _save_and_cache_settings(s)

# --- Provider API call helpers ---

async def _call_openrouter(messages, api_key, model, timeout_sec=240):
    """Call OpenRouter API. Returns response text or None on failure."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.15}
    req_timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                    headers=headers, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content.strip()
                elif resp.status == 429:
                    retry = int(resp.headers.get("Retry-After", "5"))
                    logging.warning(f"⚠️ OpenRouter 429, waiting {retry}s")
                    await asyncio.sleep(retry + 1)
                else:
                    body = await resp.text()
                    logging.warning(f"⚠️ OpenRouter HTTP {resp.status}: {body[:200]}")
    except Exception as e:
        logging.warning(f"⚠️ OpenRouter error: {type(e).__name__}: {e}")
    return None

async def _call_gemini(messages, api_key, model, timeout_sec=240):
    """Call Gemini REST API. Converts OpenAI-style messages to Gemini format."""
    # Extract system instruction and user messages
    system_text = ""
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

    payload = {"contents": contents}
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}
    payload["generationConfig"] = {"temperature": 0.15}

    url = f"https://botgem.zhoriha.workers.dev/v1beta/models/{model}:generateContent?key={api_key}"
    req_timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                        if text.strip():
                            return text.strip()
                else:
                    body = await resp.text()
                    logging.warning(f"⚠️ Gemini HTTP {resp.status}: {body[:200]}")
    except Exception as e:
        logging.warning(f"⚠️ Gemini error: {type(e).__name__}: {e}")
    return None

async def _call_groq(messages, api_key, model, timeout_sec=240):
    """Call Groq API (OpenAI-compatible). Returns response text or None."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.15}
    req_timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.groq.com/openai/v1/chat/completions",
                                    headers=headers, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content.strip()
                elif resp.status == 429:
                    logging.warning(f"⚠️ Groq 429 rate limited")
                else:
                    body = await resp.text()
                    logging.warning(f"⚠️ Groq HTTP {resp.status}: {body[:200]}")
    except Exception as e:
        logging.warning(f"⚠️ Groq error: {type(e).__name__}: {e}")
    return None

async def _call_provider(provider, messages, api_key, model, timeout_sec=240):
    """Dispatch to the right provider call function."""
    if provider == "openrouter":
        return await _call_openrouter(messages, api_key, model, timeout_sec)
    elif provider == "gemini":
        return await _call_gemini(messages, api_key, model, timeout_sec)
    elif provider == "groq":
        return await _call_groq(messages, api_key, model, timeout_sec)
    return None

def _get_keys_for_provider(provider):
    """Return list of API keys for the given provider."""
    if provider == "openrouter":
        return [OPENROUTER_API_KEY] if OPENROUTER_API_KEY else []
    elif provider == "gemini":
        return GEMINI_API_KEYS
    elif provider == "groq":
        return GROQ_API_KEYS
    return []

async def call_ai_with_fallback(messages, timeout_sec=240):
    """Main AI call with full fallback chain.
    
    Chain:
    1. Gemini (21 keys, rotate forward from saved index) - ALWAYS PRIORITY #1
    2. Groq (3 keys: gsk_rXNL..., gsk_Fl3x..., gsk_IN09...)
    3. OpenRouter openai/gpt-oss-120b:free
    4. OpenRouter openrouter/free
    
    Rotation logic (CRITICAL):
    - Start from saved active_key_index (persisted in ai_settings.json)
    - If fails (429, error, timeout) → move to NEXT key (index + 1)
    - Keep going forward only. From key #21, wrap to key #1.
    - When a key WORKS → save its index as active. Next request starts from this key.
    
    Returns (response_text, provider_used, model_used) or (None, None, None).
    """
    s = _get_ai_settings()

    # === STEP 1: Try ALL 21 Gemini keys (ALWAYS PRIORITY #1) ===
    if GEMINI_API_KEYS:
        gemini_model = s.get("gemini_model", "gemini-2.5-flash")
        saved_index = s.get("active_key_index", 0)
        
        # Try all 21 keys starting from saved index, one attempt each
        for i in range(len(GEMINI_API_KEYS)):
            key_idx = (saved_index + i) % len(GEMINI_API_KEYS)
            logging.info(f"🔄 Trying Gemini key #{key_idx+1} ({KEY_ACCOUNT_LABELS.get(key_idx, f'#{key_idx+1}')}) model={gemini_model}")
            result = await _call_gemini(messages, GEMINI_API_KEYS[key_idx], gemini_model, timeout_sec)
            if result:
                # Key worked - save it as active
                s["active_provider"] = "gemini"
                s["active_key_index"] = key_idx
                s["gemini_model"] = gemini_model
                _save_and_cache_settings(s)
                logging.info(f"✅ Gemini key #{key_idx+1} succeeded, saved as active")
                return result, "gemini", gemini_model
        
        logging.warning("⚠️ All 21 Gemini keys failed, falling back to Groq...")

    # === STEP 2: Groq (3 keys) ===
    if GROQ_API_KEYS:
        groq_model = "llama-3.3-70b-versatile"
        for i, key in enumerate(GROQ_API_KEYS):
            logging.info(f"🔄 Fallback: Groq key #{i+1} model={groq_model}")
            result = await _call_groq(messages, key, groq_model, timeout_sec)
            if result:
                s["active_provider"] = "groq"
                s["groq_model"] = groq_model
                s["active_key_index"] = i
                _save_and_cache_settings(s)
                logging.info(f"✅ Groq fallback succeeded (key #{i+1}), saved as active")
                return result, "groq", groq_model

    # === STEP 3: OpenRouter openai/gpt-oss-120b:free ===
    if OPENROUTER_API_KEY:
        or_model = "openai/gpt-oss-120b:free"
        logging.info(f"🔄 Fallback: OpenRouter model={or_model}")
        result = await _call_openrouter(messages, OPENROUTER_API_KEY, or_model, timeout_sec)
        if result:
            s["active_provider"] = "openrouter"
            s["openrouter_model"] = or_model
            _save_and_cache_settings(s)
            logging.info(f"✅ OpenRouter fallback succeeded ({or_model}), saved as active")
            return result, "openrouter", or_model

    # === STEP 4: OpenRouter openrouter/free (last resort) ===
    if OPENROUTER_API_KEY:
        or_model = "openrouter/free"
        logging.info(f"🔄 Last resort: OpenRouter model={or_model}")
        result = await _call_openrouter(messages, OPENROUTER_API_KEY, or_model, timeout_sec)
        if result:
            s["active_provider"] = "openrouter"
            s["openrouter_model"] = or_model
            _save_and_cache_settings(s)
            logging.info(f"✅ OpenRouter last resort succeeded ({or_model}), saved as active")
            return result, "openrouter", or_model

    logging.warning("❌ All AI providers failed — sending without AI verdict")
    return None, None, None

async def test_provider_key(provider, api_key, model):
    """Test a single provider key with a tiny prompt. Returns True/False."""
    messages = [{"role": "user", "content": "Say OK"}]
    try:
        result = await _call_provider(provider, messages, api_key, model, timeout_sec=15)
        return result is not None
    except Exception:
        return False

# Fast verdict prompt for auto/monitor modes — concise but complete
FAST_VERDICT_PROMPT_EN = """You are a crypto trading analyst. Analyze the scorecard data and give a verdict.

IMPORTANT: Think concisely. Your entire response (including any internal reasoning) should stay under 5000 tokens total. Do NOT write long explanations — be sharp and direct.

Output format:

VERDICT: LONG or SHORT or SKIP
LONG: [number]% / SHORT: [number]%
ENTRY: [current price]
SL: [minimum 2% from entry, use 1.5×ATR if larger]
TP: [minimum 2× SL distance — R:R must be ≥ 2:1]
LOGIC: [2-3 sentences — key factors driving the verdict]
RISK: [1 sentence — main risk to watch]

Rules:
- Weights: 4H=50%, 1H=30%, 15m=20%. 1D is context-only (risk warnings like overbought RSI) — do NOT include 1D in the weighted score.
- If both below 65% → SKIP. If ADX<20 → SKIP (FLAT).
- Leverage: always 1x. Deposit: always 2%.
- Do NOT assume direction from indicators alone. When 8+ indicators are bullish, ACTIVELY look for exhaustion signals:
  * RSI > 70 on ANY timeframe = potential reversal, consider SHORT or SKIP
  * Bearish RSI-Mom divergence = momentum exhausting, strongly consider SHORT
  * BB %B > 1.0 = price extended beyond bands, mean reversion likely
  * OBV spike = climactic volume, often marks tops
  * If 2+ exhaustion signals found: SHORT or SKIP, even if majority says LONG
- Similarly for bearish consensus: look for oversold bounces.
- Fresh trend (ADX rising from <20) is different from exhausted trend (RSI >80 + divergence). Distinguish them.
- Respond in ENGLISH."""

FAST_VERDICT_PROMPT_RU = """Ты крипто-трейдинг аналитик. Проанализируй данные scorecard и дай вердикт.

ВАЖНО: Думай кратко. Весь твой ответ (включая любые внутренние рассуждения) должен уложиться в 5000 токенов. НЕ пиши длинные объяснения — будь чётким и конкретным.

Формат ответа:

ВЕРДИКТ: ЛОНГ или ШОРТ или ПРОПУСК
ЛОНГ: [число]% / ШОРТ: [число]%
ВХОД: [текущая цена]
СЛ: [минимум 2% от входа, 1.5×ATR если больше]
ТП: [минимум 2× расстояние СЛ — R:R должен быть ≥ 2:1]
ЛОГИКА: [2-3 предложения — ключевые факторы вердикта]
РИСК: [1 предложение — главный риск]

Правила:
- Веса: 4H=50%, 1H=30%, 15m=20%. 1D — только контекст (предупреждения о рисках, перекупленность RSI) — НЕ включай 1D в взвешенный счёт.
- Если оба ниже 65% → ПРОПУСК. Если ADX<20 → ПРОПУСК (ФЛЭТ).
- Плечо: всегда 1x. Депозит: всегда 2%.
- НЕ следуй за большинством индикаторов слепо. Когда 8+ индикаторов бычьи, АКТИВНО ищи сигналы истощения:
  * RSI > 70 на ЛЮБОМ таймфрейме = потенциальный разворот, рассмотри ШОРТ или ПРОПУСК
  * Медвежья дивергенция RSI-Mom = импульс угасает, серьёзно рассмотри ШОРТ
  * BB %B > 1.0 = цена вышла за полосы, вероятен возврат к среднему
  * Всплеск OBV = кульминационный объём, часто отмечает вершины
  * Если 2+ сигнала истощения: ШОРТ или ПРОПУСК, даже если большинство за ЛОНГ
- Аналогично для медвежьего консенсуса: ищи отскоки от перепроданности.
- Свежий тренд (ADX растёт от <20) отличается от истощённого (RSI >80 + дивергенция). Различай их.
- Отвечай СТРОГО на РУССКОМ."""

def get_fast_verdict_prompt(lang: str = "en") -> str:
    return FAST_VERDICT_PROMPT_RU if lang == "ru" else FAST_VERDICT_PROMPT_EN


# ---------------------------------------------------------
# STRUCTURED OUTPUT: AI TRADING VERDICT MODEL
# ---------------------------------------------------------
class TradeVerdict(BaseModel):
    """Structured Trading Verdict — Binance AI Analysis."""
    direction: str          # "LONG" or "SHORT"
    entry_price: float      # Recommended entry price
    stop_loss: float        # Stop-loss price level
    take_profit: float      # Take-profit price level
    risk_percent: float     # Estimated risk as % of margin
    leverage_rec: str       # e.g. "5x", "10x"
    deposit_rec: str        # e.g. "10%", "15%"
    logic: str              # AI reasoning (max 5 sentences)
    risk_note: Optional[str] = None  # Stop-loss calculation for margin queries


def _is_valid_analysis(text: str) -> bool:
    """Check if AI response looks like a real trading analysis, not garbage/error."""
    if not text or len(text) < 100:
        return False
    # Must contain at least 2 of these key markers
    markers = ["VERDICT", "LONG", "SHORT", "Entry", "SL", "TP", "REC",
               "ВЕРДИКТ", "Вход", "Стоп", "Тейк", "Analysis", "Анализ"]
    found = sum(1 for m in markers if m.lower() in text.lower())
    if found < 2:
        return False
    # Reject if it's clearly an error message
    error_signs = ["error", "failed", "timeout", "exception", "traceback",
                   "request_id=", "502", "503", "429"]
    error_found = sum(1 for s in error_signs if s.lower() in text.lower()[:200])
    if error_found >= 2:
        return False
    return True


def _has_placeholder_percentages(text: str) -> bool:
    """Check if AI response contains literal X%/Y% placeholder patterns instead of real numbers."""
    import re
    # Match patterns like "LONG X%" or "SHORT Y%" or "ДЛГО X%" or "КОРОТКО Y%"
    placeholder_patterns = [
        r'LONG\s+X%',
        r'SHORT\s+Y%',
        r'ДЛГО\s+X%',
        r'КОРОТКО\s+Y%',
        r'ЛОНГ\s+X%',
        r'ШОРТ\s+Y%',
        r'LONG\s+\[calculated_number\]%',
        r'SHORT\s+\[calculated_number\]%',
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _fix_placeholder_percentages(text: str, mtf_data: dict = None, indicators: dict = None, tf_key: str = "4H") -> str:
    """Replace literal X%/Y% placeholders in AI response with real scorecard percentages.
    
    Extracts LONG/SHORT % from format_tf_summary scorecard for each TF,
    then substitutes them into the AI text.
    """
    import re
    from core.indicators import format_tf_summary

    if not _has_placeholder_percentages(text):
        return text

    logging.warning("⚠️ AI returned literal X%/Y% placeholders — substituting from scorecard")

    # Collect scorecard percentages per TF
    tf_pcts = {}

    # Primary TF
    if indicators:
        summary = format_tf_summary(indicators, tf_key)
        match = re.search(r'LONG\s+(\d+)%\s*/\s*SHORT\s+(\d+)%', summary)
        if match:
            tf_pcts[tf_key] = (int(match.group(1)), int(match.group(2)))

    # MTF data
    if mtf_data:
        for tf_label, tf_indic in mtf_data.items():
            summary = format_tf_summary(tf_indic, tf_label)
            match = re.search(r'LONG\s+(\d+)%\s*/\s*SHORT\s+(\d+)%', summary)
            if match:
                tf_pcts[tf_label] = (int(match.group(1)), int(match.group(2)))

    if not tf_pcts:
        return text

    # Replace per-TF placeholders: "⏱ 4H: LONG X% / SHORT Y%" → "⏱ 4H: LONG 72% / SHORT 28%"
    # Also handle Russian variants: ДЛГО/КОРОТКО, ЛОНГ/ШОРТ
    for tf_label, (long_pct, short_pct) in tf_pcts.items():
        for long_word, short_word in [("LONG", "SHORT"), ("ДЛГО", "КОРОТКО"), ("ЛОНГ", "ШОРТ")]:
            # Pattern: TF label followed by LONG X% / SHORT Y%
            pattern = rf'(⏱\s*{re.escape(tf_label)}[:\s]*){long_word}\s+X%\s*/\s*{short_word}\s+Y%'
            replacement = rf'\g<1>{long_word} {long_pct}% / {short_word} {short_pct}%'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Replace "Overall" / "Общий" line placeholder
    if tf_pcts:
        # Weighted average: 4H=50%, 1H=30%, 15m=20% (1D is context-only, not a voting TF)
        weights = {"4H": 0.50, "1H": 0.30, "15m": 0.20}
        total_w = 0
        weighted_long = 0
        for tf_label, (long_pct, short_pct) in tf_pcts.items():
            if tf_label not in weights:
                continue  # Skip 1D and any unknown TFs — context only
            w = weights[tf_label]
            weighted_long += long_pct * w
            total_w += w
        if total_w > 0:
            overall_long = round(weighted_long / total_w)
            overall_short = 100 - overall_long
            for long_word, short_word in [("LONG", "SHORT"), ("ДЛГО", "КОРОТКО"), ("ЛОНГ", "ШОРТ")]:
                # Match both placeholders (X%/Y%) and any numbers AI may have inserted
                for pct_pat in [r'X%', r'\d+%']:
                    # Fix "Overall" / "Общий" line
                    pattern = rf'((?:Overall|Общий)[:\s]*){long_word}\s+{pct_pat}\s*/\s*{short_word}\s+{pct_pat}'
                    replacement = rf'\g<1>{long_word} {overall_long}% / {short_word} {overall_short}%'
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    # Fix VERDICT / ВЕРДИКТ line parenthetical: e.g. "SKIP (LONG 56% / SHORT 34%)"
                    pattern_v = rf'((?:VERDICT|ВЕРДИКТ)[:\s]*(?:LONG|SHORT|SKIP|ЛОНГ|ШОРТ|ПРОПУСК|MONITOR|МОНИТОРИНГ)[^(]*\(){long_word}\s+{pct_pat}\s*/\s*{short_word}\s+{pct_pat}'
                    replacement_v = rf'\g<1>{long_word} {overall_long}% / {short_word} {overall_short}%'
                    text = re.sub(pattern_v, replacement_v, text, flags=re.IGNORECASE)
                    # Fix "Weighted verdict" / "Взвешенный вердикт" line
                    pattern_wv = rf'((?:Weighted verdict|Взвешенный вердикт)[:\s]*){long_word}\s+{pct_pat}\s*/\s*{short_word}\s+{pct_pat}'
                    replacement_wv = rf'\g<1>{long_word} {overall_long}% / {short_word} {overall_short}%'
                    text = re.sub(pattern_wv, replacement_wv, text, flags=re.IGNORECASE)

    return text


def _validate_sl_tp_in_text(text: str) -> str:
    """Post-validate SL/TP consistency in free-text AI response.
    Checks: SL direction, minimum 2% SL distance, minimum 2:1 R:R ratio."""
    import re
    direction_m = re.search(r'VERDICT[:\s]*(LONG|SHORT)', text, re.IGNORECASE)
    entry_m = re.search(r'(?:Entry|Вход)[:\s]*\$?([\d.]+)', text, re.IGNORECASE)
    sl_m = re.search(r'(?:SL|Стоп|🚫 SL)[:\s]*\$?([\d.]+)', text, re.IGNORECASE)
    tp_m = re.search(r'(?:TP|Тейк|🎯 TP)[:\s]*\$?([\d.]+)', text, re.IGNORECASE)
    if direction_m and entry_m and sl_m:
        direction = direction_m.group(1).upper()
        entry = float(entry_m.group(1))
        sl = float(sl_m.group(1))
        tp = float(tp_m.group(1)) if tp_m else 0
        
        # Check SL on wrong side
        if direction == "LONG" and sl > entry:
            logging.warning(f"⚠️ [SL WARN] LONG but SL({sl}) > Entry({entry}) in text response!")
            text += "\n\n⚠️ ВНИМАНИЕ: SL выше входа для LONG — проверьте уровни вручную!"
        elif direction == "SHORT" and sl < entry:
            logging.warning(f"⚠️ [SL WARN] SHORT but SL({sl}) < Entry({entry}) in text response!")
            text += "\n\n⚠️ ВНИМАНИЕ: SL ниже входа для SHORT — проверьте уровни вручную!"
        
        # Check minimum 2% SL distance
        if entry > 0:
            sl_pct = abs(sl - entry) / entry * 100
            if sl_pct < 1.5:  # warn if SL is too tight (< 1.5%)
                logging.warning(f"⚠️ [SL TIGHT] {direction} SL distance only {sl_pct:.1f}% (entry={entry}, sl={sl})")
                text += f"\n\n⚠️ SL слишком близко ({sl_pct:.1f}%) — рекомендуется минимум 2%"
        
        # Check R:R ratio (minimum 2:1)
        if entry > 0 and tp > 0 and sl > 0:
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            if sl_dist > 0:
                rr = tp_dist / sl_dist
                if rr < 1.8:  # warn if R:R is below 1.8 (some tolerance)
                    logging.warning(f"⚠️ [R:R WARN] {direction} R:R={rr:.1f} (entry={entry}, sl={sl}, tp={tp})")
                    text += f"\n\n⚠️ R:R = {rr:.1f}:1 — рекомендуется минимум 2:1"
    return text


def _format_verdict(v: TradeVerdict, base_coin: str, price: float, dynamics_text: str) -> str:
    """Convert structured TradeVerdict back to display text for Telegram."""
    lines = []
    if v.risk_note:
        lines.append(v.risk_note)
        lines.append("")
    lines.append(f"${base_coin} 📊 Current Price: ${price:.6f}. {dynamics_text}")
    lines.append(f"🏆 VERDICT: {v.direction}")
    lines.append(f"🧠 LOGIC: {v.logic}")
    lines.append(f"🎯 TRADE: 💰 Entry: {v.entry_price} | 🚫 SL: {v.stop_loss} | 🎯 TP: {v.take_profit}")
    lines.append(f"🚫 RISK: {v.risk_percent}%")
    lines.append(f"💼 REC: {v.leverage_rec} | {v.deposit_rec}")
    return "\n".join(lines)

# ---------------------------------------------------------
# LIVE AI STREAMING IN TELEGRAM
# ---------------------------------------------------------
async def _edit_telegram_msg(session, chat_id, message_id, text, bot_token, parse_mode=None):
    """Edit a Telegram message. Used for real-time streaming updates."""
    url = f"https://api.telegram.org/bot{bot_token}/editMessageText"
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text[:4096]}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        async with session.post(url, json=payload, timeout=5) as resp:
            pass  # Silently ignore edit errors (rate limits, unchanged text)
    except Exception:
        pass


async def _progressive_display(text, telegram_stream):
    """Progressively reveal AI text in Telegram via editMessageText.
    
    Takes already-obtained AI response and reveals it word-by-word
    with a typing cursor effect.
    """
    session = telegram_stream["session"]
    chat_id = telegram_stream["chat_id"]
    message_id = telegram_stream["message_id"]
    bot_token = telegram_stream["bot_token"]

    try:
        words = text.split()
        if len(words) < 5:
            # Too short for progressive display
            await _edit_telegram_msg(session, chat_id, message_id,
                f"⚡ *AiAlisa AI Complete* ✅\n\n{text}",
                bot_token, parse_mode="Markdown")
            return

        # Build progressive chunks (~8 updates)
        chunk_size = max(3, len(words) // 8)
        for i in range(chunk_size, len(words), chunk_size):
            partial = " ".join(words[:i])
            display = partial + " ▌"
            if len(display) > 4000:
                display = display[:4000]
            await _edit_telegram_msg(session, chat_id, message_id,
                f"⚡ *AiAlisa Live* ({i}/{len(words)})\n\n{display}",
                bot_token)
            await asyncio.sleep(0.7)

        # Final — complete text, no cursor
        final_display = text
        if len(final_display) > 3900:
            final_display = final_display[:3900] + "..."
        await _edit_telegram_msg(session, chat_id, message_id,
            f"⚡ *AiAlisa AI Complete* ✅\n\n{final_display}",
            bot_token, parse_mode="Markdown")

        logging.info(f"✅ AiAlisa Progressive Stream: {len(words)} words → Telegram")

    except Exception as e:
        logging.info(f"⚙️ Progressive display error ({e}), skipping...")


# Load the FULL arsenal of Native Binance Skills
from agent.skills import (
    get_smart_money_signals,
    get_unified_token_rank,
    get_social_hype_leaderboard,
    get_smart_money_inflow_rank,
    get_meme_rank,
    get_address_pnl_rank
)

async def ask_ai_analysis(symbol: str, tf_key: str, indicators: dict, line_price: float = None, user_margin: dict = None, lang: str = "en", telegram_stream: dict = None, extended: bool = False, square: bool = False, mtf_data: dict = None, smc_data: dict = None, mode: str = "scan", api_key_override: str = None, model_override: str = None) -> str:
    """
    AI Agent: Executes Binance Market Intelligence Skills
    and sends aggregated context to OpenRouter.
    
    Args:
        telegram_stream: Optional dict for live Telegram streaming:
            {"session": aiohttp_session, "chat_id": int, "message_id": int, "bot_token": str}
            If provided, streams AI tokens to Telegram in real-time via editMessageText.
    """
    base_coin = symbol.upper().replace("USDT", "")

    # CLEAN GARBAGE DATA (NaN / Infinity)
    clean_indic = {}
    for k, v in indicators.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_indic[k] = 0.0
        else:
            clean_indic[k] = v

    price = clean_indic.get("close", 0.0)
    change_recent = clean_indic.get("change_recent", 0.0)
    change_24h = clean_indic.get("change_24h", 0.0)
    recent_label = clean_indic.get("recent_label", "Recent")

    # Build dynamic text to avoid duplicate stats on 1D timeframe
    if recent_label.upper() == "1D" or tf_key.upper() == "1D":
        dynamics_text = f"Over the last 24H changed by {change_24h}%."
    else:
        dynamics_text = f"Over the last {recent_label} changed by {change_recent}%, and over 24H by {change_24h}%."

    # Extract 1H price change for header display
    change_1h = 0.0
    if mtf_data and "1H" in mtf_data:
        change_1h = mtf_data["1H"].get("change_recent", 0.0)
        if isinstance(change_1h, float) and (math.isnan(change_1h) or math.isinf(change_1h)):
            change_1h = 0.0
    elif tf_key.upper() == "1H":
        change_1h = change_recent

    # User Risk and Breakout context
    user_risk_text = ""
    risk_prompt_rule = ""
    if user_margin:
        margin = user_margin.get("margin", 0)
        lev = user_margin.get("leverage", 1)
        max_loss = user_margin.get("max_loss")
        user_risk_text = f"USER RISK SIMULATION: ${margin} and {lev}x leverage."

        if max_loss:
            user_risk_text += f" Max allowed loss limit: {max_loss}% of margin."
            if lang == "ru":
                risk_prompt_rule = f"\nCALCULATE EXACT STOP LOSS. Start your entire response exactly like this:\n'Чтобы не потерять более {max_loss}% от маржи (${margin}) при {lev}x плече, ваш стоп-лосс должен быть [CALCULATED PRICE].'\nLeave a blank line, then continue with the standard VERDICT structure."
            else:
                risk_prompt_rule = f"\nCALCULATE EXACT STOP LOSS. Start your entire response exactly like this:\n'To avoid losing more than {max_loss}% of your margin (${margin}) at {lev}x leverage, your Stop Loss must be [CALCULATED PRICE].'\nLeave a blank line, then continue with the standard VERDICT structure."

    breakout_context = ""

    if lang == "en":
        lang_directive = "Respond strictly in ENGLISH."
    else:
        lang_directive = "Respond strictly in RUSSIAN. Translate all output labels (like Current Price, VERDICT, LOGIC, TRADE, RISK, REC) to Russian."

    # =========================================================
    # AGENTIC ACTION: NATIVELY CALLING BINANCE SKILLS
    # =========================================================
    # Forced initialization logs for Architectural visibility
    logging.info(f"🤖 AiAlisa Agent initialized for {symbol}...")
    logging.info(f"⚙️ [AiAlisa] Executing Agentic Binance Skills...")

    smart_money_context = await get_smart_money_signals(symbol)
    hype_context = await get_social_hype_leaderboard()

    # Only include Smart Money / Hype in prompt if they have real data
    skills_block = ""
    if smart_money_context and "no data" not in smart_money_context.lower() and "error" not in smart_money_context.lower() and "none" not in smart_money_context.lower():
        skills_block += f"\n- SMART MONEY ACTIVITY: {smart_money_context}"
    if hype_context and "no data" not in hype_context.lower() and "error" not in hype_context.lower() and "none" not in hype_context.lower():
        skills_block += f"\n- SOCIAL HYPE TRENDS: {hype_context}"

    skills_note = ""
    if skills_block:
        skills_note = f"\nWeb3 Plugin Results:{skills_block}\n"

    # =========================================================
    # 1. SYSTEM SETUP — UNIFIED MULTI-TF FORMAT
    # =========================================================
    # Build dynamic TF list based on available data
    available_tfs = [tf_key]
    if mtf_data:
        available_tfs = list(mtf_data.keys()) + [tf_key]
    # Deduplicate and order: 1D, 4H, 1H, 15m
    tf_order = ["1D", "4H", "1H", "15m"]
    available_tfs = [t for t in tf_order if t in available_tfs]
    tf_list_str = " + ".join(available_tfs)

    # TF lines for output format — use [number] to make it clear AI must calculate real %
    tf_format_lines = "\n".join([f"⏱ {t}: LONG [calculated_number]% / SHORT [calculated_number]% (key reasons)" for t in available_tfs])
    tf_format_lines_short = "\n".join([f"⏱ {t}: LONG [calculated_number]% / SHORT [calculated_number]% (brief reason)" for t in available_tfs])

    if extended:
        system_instruction = f"""You are AiAlisa, an advanced AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. You MUST analyze EVERY indicator on EVERY timeframe.

HOW TO CALCULATE LONG/SHORT PERCENTAGE PER TIMEFRAME:
The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TWO-STEP ANALYSIS:

STEP 1 — DIRECTION (which way to trade):
- 4H = 50% weight (PRIMARY — breakout detected here)
- 1H = 30% weight (CONFIRMATION — momentum developing?)
- 15m = 20% weight (ENTRY TIMING — confirms momentum, if 1H+15m disagree with 4H, lean toward 1H+15m as 4H may lag)
- 1D = CONTEXT ONLY (NOT included in weighted score — use for risk warnings: overbought RSI, macro trend, etc.)

CRITICAL: 1D is LAGGING! 1D uptrend may mean coin is at the TOP. 1D downtrend may mean reversal JUST started.
Do NOT let 1D override a fresh 4H breakout. Use 1D only for risk notes.

STEP 2 — ENTRY TIMING (after direction decided):
Use 15m + 1H: enter NOW if momentum confirms, or wait for SAFE entry if 15m shows pullback.

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money = trend continuation, OI falling = positions closing = trend weakening
- RSI MOMENTUM RULE: If ADX > 30 (strong trend), high RSI is normal momentum — do NOT reduce confidence.
- RSI WARNING (ALWAYS): If RSI > 70 on ANY TF, ALWAYS include "⚠️ RSI [value]" in your output. This is mandatory.
- RSI PULLBACK HISTORY: The scorecard shows "PULLBACK HISTORY: last pullback started from RSI X". Use this! If current RSI is near that peak, warn: "⚠️ RSI [value], previous pullback from [peak]"
- If ADX < 30 (weak trend) AND RSI > 82 on 4H: reduce confidence by 10%.
- Each indicator has historical context showing dynamics over multiple candles

YOUR RESPONSE MUST CONTAIN TWO PARTS SEPARATED BY --- ON ITS OWN LINE.
Do NOT write any headers like "PART 1", "PART 2", "=== PART ===", "Brief", "Extended" etc. Just the content.

FIRST PART (brief chart caption — keep it concise):

${base_coin} Analysis🤔
📊 Price: ${price:.6f} 24H {change_24h:+.0f}% 1H {change_1h:+.0f}%

{tf_format_lines_short}

🏆 VERDICT: LONG or SHORT
📊 Overall: LONG X% / SHORT Y%
💰 Funding: [rate]
📊 L/S: [ratio]
⚠️ [short risk note referencing lower TF]

💰 Entry: ${price:.6f}
🔰 Safe: $X.XXXX
🚫 SL: $X.XXXX ([reason])
🎯 TP: $X.XXXX ([reason])
💼 REC: Xx | X%

---

SECOND PART (extended analysis, STRICT 2000-3800 characters — DO NOT EXCEED 3800):

*🔬 ${base_coin} Extended Analysis*

FOR EACH TIMEFRAME (⏱ 1D, 4H, 1H, 15m) write ONE compact block:
⏱ [TF]: LONG X% / SHORT Y% (key reasons)
- EMA: [cluster, death/golden cross, slope]
- MACD: [direction, histogram trend]
- OBV: [accumulation/distribution, ROC]
- RSI: [value, trend, overbought/oversold]
- BB: [%B, squeeze/expanding]
- SuperTrend: [direction, freshness, distance]
- Ichimoku: [cloud position, TK cross]
- ADX: [strength, DI+/DI- dominant]
- StochRSI/MFI/CMF: [brief combined]
- Score: X bullish vs Y bearish

📐 SMC (key levels only, 2-3 lines total):
- Key support: [nearest bull OB/EQL with price and distance %]
- Key resistance: [nearest bear OB/EQH with price and distance %]
- Structure: [BOS/CHoCH direction across TFs]

📊 Weighted verdict: LONG X% / SHORT Y% (1-2 sentences why)

⚠️ Risks (2-3 bullet points max)

💰 Entry/SL/TP reasoning: [which indicators converge at these levels]{risk_prompt_rule}

COMPACT FORMAT IS CRITICAL — use abbreviations, skip redundant details, keep each indicator to ONE short line. Total MUST be 2000-3800 characters.

CRITICAL RULES:
1. First part: brief signal summary following the template above. Do NOT count characters.
2. Second part MUST be 2000-3800 characters. No more, no less.
3. Separate parts with exactly --- on its own line
4. Do NOT write any labels, headers, or markers like "Part 1", "Part 2", "=== PART ===" etc.
5. Entry = CURRENT PRICE. Safe Entry = better entry from support/OB
6. SL/TP: CONFLUENCE of multiple indicators. SL distance: MINIMUM 2% from entry (use 1.5×ATR if larger). TP distance: MINIMUM 2× SL distance (R:R ≥ 2:1 ALWAYS). CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
7. LEVERAGE: ALWAYS 1x. DEPOSIT: ALWAYS 2%. These are FIXED. In REC always write: 1x | 2%
8. DO NOT ADD HASHTAGS
9. RSI RULES (MOMENTUM-AWARE):
   - ALWAYS show RSI value when >70: "⚠️ RSI [value]" — this is MANDATORY in every signal.
   - Check PULLBACK HISTORY in scorecard: if current RSI is near the level where last pullback happened, warn explicitly: "⚠️ RSI 80, previous pullback from 82!"
   - ADX > 30 (STRONG TREND): high RSI (up to ~82 on RSI14) is normal momentum. Do NOT reduce confidence. Warn but trade.
   - ADX < 30 (WEAK TREND): RSI > 82 on 4H = reduce confidence 10%. RSI > 82 on 2+ TFs = consider SKIP.
   - 15m RSI spikes are normal in breakouts — never penalize 15m RSI alone.
10. SKIP only if truly 50/50 or ADX < 20 (flat). Do NOT skip just because RSI is high in a strong trend.
12. EXHAUSTION CHECK (MANDATORY when 8+ indicators bullish): Actively look for reversal signals:
   - RSI > 70 on ANY TF = potential reversal, consider SHORT or SKIP
   - Bearish RSI-Mom divergence = momentum exhausting, strongly consider SHORT
   - BB %B > 1.0 = price extended beyond bands, mean reversion likely
   - OBV spike (2σ) = climactic volume, often marks tops
   - If 2+ exhaustion signals found: SHORT or SKIP, even if majority says LONG
   - Fresh trend (ADX rising from <20) ≠ exhausted trend (RSI >80 + divergence). Distinguish them.
11. You MUST always output BOTH parts. Never skip the second part.
CRITICAL — REAL NUMBERS ONLY:
- NEVER output literal "X%" or "Y%" in your response. Every percentage MUST be a real calculated number (e.g. "LONG 72%" not "LONG X%").
- Use the SCORECARD at the bottom of each TF data block — it already shows "LONG XX% / SHORT YY%" with real numbers. Use those as your starting point.
- If you output "X%" or "Y%" literally, your response will be REJECTED.

SKIP RULES:
- If Overall LONG% and SHORT% are both below 65%, set VERDICT: SKIP (LONG X% / SHORT Y%)
- If ADX < 20 on 4H, set VERDICT: SKIP (FLAT) — market ranging, add note "⚠️ ADX flat, monitoring"
- SKIP signals go to hourly monitoring and may upgrade later
"""
    elif square:
        system_instruction = f"""You are AiAlisa, an advanced AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. Analyze EVERY indicator on EVERY timeframe.

The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TWO-STEP ANALYSIS:

STEP 1 — DIRECTION (which way to trade):
- 4H = 50% weight (PRIMARY — breakout detected here)
- 1H = 30% weight (CONFIRMATION — momentum developing?)
- 15m = 20% weight (ENTRY TIMING — confirms momentum, if 1H+15m disagree with 4H, lean toward 1H+15m as 4H may lag)
- 1D = CONTEXT ONLY (NOT included in weighted score — use for risk warnings: overbought RSI, macro trend, etc.)

CRITICAL: 1D is LAGGING! 1D uptrend may mean coin is at the TOP. 1D downtrend may mean reversal JUST started.
Do NOT let 1D override a fresh 4H breakout. Use 1D only for risk notes.

STEP 2 — ENTRY TIMING (after direction decided):
Use 15m + 1H: enter NOW if momentum confirms, or wait for SAFE entry if 15m shows pullback.

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money = trend continuation, OI falling = positions closing
- RSI MOMENTUM RULE: If ADX > 30 (strong trend), high RSI is normal momentum — do NOT reduce confidence.
- RSI WARNING (ALWAYS): If RSI > 70 on ANY TF, ALWAYS include "⚠️ RSI [value]" in your output. This is mandatory.
- RSI PULLBACK HISTORY: The scorecard shows "PULLBACK HISTORY: last pullback started from RSI X". Use this! If current RSI is near that peak, warn: "⚠️ RSI [value], previous pullback from [peak]"
- If ADX < 30 (weak trend) AND RSI > 82 on 4H: reduce confidence by 10%.
- Each indicator has historical context showing dynamics over multiple candles

THIS IS FOR BINANCE SQUARE POST — PLAIN TEXT ONLY, NO BOLD, NO MARKDOWN, NO * symbols.
Your response MUST be between 1300 and 1900 characters (counting all characters, spaces, emoji). Count carefully!

MANDATORY OUTPUT FORMAT:

${base_coin} Analysis
📊 Price: ${price:.6f} 24H {change_24h:+.0f}% 1H {change_1h:+.0f}%

{tf_format_lines_short}

🏆 VERDICT: LONG or SHORT
📊 Overall: LONG X% / SHORT Y%
💰 Funding: [rate]
📊 L/S: [ratio]

📊 Key indicators:
- Trend (ADX/EMA): [brief cross-TF summary]
- Momentum (MACD/RSI): [brief cross-TF summary]
- Volume (OBV/CMF): [brief cross-TF summary]
- SMC: [key levels, OB, FVG near price]

⚠️ Risks:
1. [main risk]
2. [secondary risk]

💰 Entry: ${price:.6f}
🔰 Safe: $X.XXXX ([reason])
🚫 SL: $X.XXXX ([reason])
🎯 TP: $X.XXXX ([reason])
💼 REC: Xx | X%{risk_prompt_rule}

RULES:
1. PLAIN TEXT ONLY — no bold, no markdown, no * or ** symbols. Binance Square does not render formatting.
2. Entry = current price. Safe = better entry from support/OB
3. SL/TP: CONFLUENCE of multiple indicators. SL distance: MINIMUM 2% from entry (use 1.5×ATR if larger). TP distance: MINIMUM 2× SL distance (R:R ≥ 2:1 ALWAYS). CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
4. LEVERAGE: ALWAYS 1x. DEPOSIT: ALWAYS 2%. In REC always write: 1x | 2%
5. Response MUST be 1300-1900 characters. Header/footer will add ~200 chars to reach 1500-2100 total.
6. DO NOT ADD HASHTAGS — they are added automatically
7. RSI RULES (MOMENTUM-AWARE):
   - ALWAYS show RSI value when >70: "⚠️ RSI [value]" — MANDATORY.
   - Check PULLBACK HISTORY: if current RSI near last pullback peak, warn: "⚠️ RSI 80, previous pullback from 82!"
   - ADX > 30: high RSI is normal momentum, don't reduce confidence.
   - ADX < 30: RSI > 82 on 4H = reduce 10%.
   - 15m RSI spikes = normal in breakouts, never penalize alone.
8. SKIP only if truly 50/50 or ADX < 20 (flat). Do NOT skip just because RSI is high in a strong trend.
9. EXHAUSTION CHECK (MANDATORY when 8+ indicators bullish): Actively look for reversal signals:
   - RSI > 70 on ANY TF = potential reversal, consider SHORT or SKIP
   - Bearish RSI-Mom divergence = momentum exhausting, strongly consider SHORT
   - BB %B > 1.0 = price extended beyond bands, mean reversion likely
   - OBV spike (2σ) = climactic volume, often marks tops
   - If 2+ exhaustion signals found: SHORT or SKIP, even if majority says LONG
   - Fresh trend (ADX rising from <20) ≠ exhausted trend (RSI >80 + divergence). Distinguish them.
CRITICAL — REAL NUMBERS ONLY:
- NEVER output literal "X%" or "Y%" in your response. Every percentage MUST be a real calculated number (e.g. "LONG 72%" not "LONG X%").
- Use the SCORECARD at the bottom of each TF data block — it already shows "LONG XX% / SHORT YY%" with real numbers. Use those as your starting point.
- If you output "X%" or "Y%" literally, your response will be REJECTED.

SKIP RULES:
- If Overall LONG% and SHORT% are both below 65%, set VERDICT: SKIP (LONG X% / SHORT Y%)
- If ADX < 20 on 4H, set VERDICT: SKIP (FLAT) — market ranging, add note "⚠️ ADX flat, monitoring"
- SKIP signals go to hourly monitoring and may upgrade later
"""
    else:
        system_instruction = f"""You are AiAlisa, an advanced AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. Analyze EVERY indicator on EVERY timeframe.

The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TWO-STEP ANALYSIS:

STEP 1 — DIRECTION (which way to trade):
- 4H = 50% weight (PRIMARY — breakout detected here)
- 1H = 30% weight (CONFIRMATION — momentum developing?)
- 15m = 20% weight (ENTRY TIMING — confirms momentum, if 1H+15m disagree with 4H, lean toward 1H+15m as 4H may lag)
- 1D = CONTEXT ONLY (NOT included in weighted score — use for risk warnings: overbought RSI, macro trend, etc.)

CRITICAL: 1D is LAGGING! 1D uptrend may mean coin is at the TOP. 1D downtrend may mean reversal JUST started.
Do NOT let 1D override a fresh 4H breakout. Use 1D only for risk notes.

STEP 2 — ENTRY TIMING (after direction decided):
Use 15m + 1H: enter NOW if momentum confirms, or wait for SAFE entry if 15m shows pullback.

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money, OI falling = positions closing
- RSI MOMENTUM RULE: If ADX > 30 (strong trend), high RSI is normal momentum — do NOT reduce confidence.
- RSI WARNING (ALWAYS): If RSI > 70 on ANY TF, ALWAYS include "⚠️ RSI [value]" in your output. This is mandatory.
- RSI PULLBACK HISTORY: The scorecard shows "PULLBACK HISTORY: last pullback started from RSI X". Use this! If current RSI is near that peak, warn: "⚠️ RSI [value], previous pullback from [peak]"
- If ADX < 30 (weak trend) AND RSI > 82 on 4H: reduce confidence by 10%.

MANDATORY OUTPUT FORMAT (follow this template, keep it concise, do NOT count characters):

${base_coin} Analysis🤔
📊 Price: ${price:.6f} 24H {change_24h:+.0f}% 1H {change_1h:+.0f}%

{tf_format_lines_short}

🏆 VERDICT: LONG or SHORT
📊 Overall: LONG X% / SHORT Y%
💰 Funding: [rate]
📊 L/S: [ratio]
⚠️ [short risk note]

💰 Entry: ${price:.6f}
🔰 Safe: $X.XXXX
🚫 SL: $X.XXXX ([reason])
🎯 TP: $X.XXXX ([reason])
💼 REC: Xx | X%{risk_prompt_rule}

RULES:
1. Entry = current price. Safe = better entry from support/OB
2. SL/TP: CONFLUENCE of multiple indicators. SL distance: MINIMUM 2% from entry (use 1.5×ATR if larger). TP distance: MINIMUM 2× SL distance (R:R ≥ 2:1 ALWAYS). CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
3. LEVERAGE: ALWAYS 1x. DEPOSIT: ALWAYS 2%. In REC always write: 1x | 2%
4. Each TF line: brief reason in parentheses
5. Keep response concise. Do NOT count characters.

CRITICAL — REAL NUMBERS ONLY:
- NEVER output literal "X%" or "Y%" in your response. Every percentage MUST be a real calculated number (e.g. "LONG 72%" not "LONG X%").
- Use the SCORECARD at the bottom of each TF data block — it already shows "LONG XX% / SHORT YY%" with real numbers. Use those as your starting point.
- If you output "X%" or "Y%" literally, your response will be REJECTED.

SKIP RULES:
- If Overall LONG% and SHORT% are both below 65%, set VERDICT: SKIP (LONG X% / SHORT Y%)
- If ADX < 20 on 4H, set VERDICT: SKIP (FLAT) — market ranging, add note "⚠️ ADX flat, monitoring"
- SKIP signals go to hourly monitoring and may upgrade later
"""


    # 2. BUILD MULTI-TIMEFRAME DATA BLOCK
    from core.indicators import format_tf_summary

    # Inject SMC scoring into indicator dicts for 5-group scorecard
    from core.smc import score_smc
    current_price_for_smc = clean_indic.get('close', 0)
    if smc_data and tf_key in smc_data:
        smc_result = smc_data[tf_key]
        if isinstance(smc_result, dict) and "swing_structures" in smc_result:
            clean_indic["_smc_score"] = score_smc(smc_result, current_price_for_smc)

    # Primary TF data (always present)
    primary_tf_text = format_tf_summary(clean_indic, tf_key)

    # Additional timeframes (if mtf_data provided)
    mtf_text = ""
    if mtf_data:
        for mtf_label, mtf_indic in mtf_data.items():
            # Clean NaN/Inf from each TF
            clean_mtf = {}
            for k, v in mtf_indic.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_mtf[k] = 0.0
                else:
                    clean_mtf[k] = v
            # Inject SMC scoring for this MTF
            if smc_data and mtf_label in smc_data:
                smc_mtf = smc_data[mtf_label]
                if isinstance(smc_mtf, dict) and "swing_structures" in smc_mtf:
                    mtf_price = clean_mtf.get('close', current_price_for_smc)
                    clean_mtf["_smc_score"] = score_smc(smc_mtf, mtf_price)
            mtf_text += "\n" + format_tf_summary(clean_mtf, mtf_label)

    # Build SMC block with per-TF directional scorecard
    smc_text = ""
    if smc_data:
        smc_parts = []
        for tf_lbl, smc_result in smc_data.items():
            if isinstance(smc_result, dict) and "summary" in smc_result:
                # Count SMC bullish/bearish signals
                smc_bull = 0
                smc_bear = 0
                summary_text = smc_result["summary"]

                # Swing trend
                if smc_result.get("swing_structures"):
                    last_swing = smc_result["swing_structures"][-1]
                    if last_swing["bias"] == 1: smc_bull += 1  # BULLISH
                    else: smc_bear += 1

                # Internal trend
                if smc_result.get("internal_structures"):
                    last_int = smc_result["internal_structures"][-1]
                    if last_int["bias"] == 1: smc_bull += 1
                    else: smc_bear += 1

                # Order blocks near price (within 3%)
                current_p = price
                for ob in (smc_result.get("swing_order_blocks", []) + smc_result.get("internal_order_blocks", [])):
                    ob_mid = (ob["low"] + ob["high"]) / 2
                    dist_pct = abs(current_p - ob_mid) / current_p * 100
                    if dist_pct < 3:  # within 3% of price
                        if ob["bias"] == 1: smc_bull += 1  # Bull OB = support
                        else: smc_bear += 1  # Bear OB = resistance

                # FVGs
                for fvg in smc_result.get("fvgs", []):
                    if fvg["bias"] == 1: smc_bull += 1
                    else: smc_bear += 1

                # Premium/Discount zone
                zones = smc_result.get("zones", {})
                zone_name = zones.get("current_zone", "")
                if "Discount" in zone_name: smc_bull += 1  # cheap = buy
                elif "Premium" in zone_name: smc_bear += 1  # expensive = sell

                smc_total = smc_bull + smc_bear
                if smc_total > 0:
                    smc_bull_pct = round(smc_bull / smc_total * 100)
                    smc_score = f"\n📊 SMC [{tf_lbl}] SCORECARD: {smc_bull}🟢 vs {smc_bear}🔴 → LONG {smc_bull_pct}% / SHORT {100-smc_bull_pct}%"
                else:
                    smc_score = ""

                smc_parts.append(f"{summary_text}{smc_score}")
        if smc_parts:
            smc_text = "\n\n[SMC INDICATORS 12-16: Structure, Order Blocks, FVG, Liquidity, Premium/Discount]\n" + "\n\n".join(smc_parts)

    # Funding rate — informational only, NOT a directional indicator
    funding = clean_indic.get("funding_rate", "Unknown")
    funding_text = f"Funding Rate: {funding} (INFO ONLY — funding does NOT predict direction. Coins with -0.5% or +1% funding can move explosively in either direction. Use for context, NOT for LONG/SHORT decision.)"

    # Market Positioning (OI, L/S Ratio, Taker Volume)
    from core.binance_api import format_positioning_text
    positioning = clean_indic.get("positioning", {})
    positioning_text = format_positioning_text(positioning, price) if positioning else ""

    user_prompt = f"""Evaluate {symbol}. {user_risk_text}

[MULTI-TIMEFRAME DATA — check SCORECARD per TF]
{primary_tf_text}
{mtf_text}
{smc_text}

{positioning_text}

[ADDITIONAL]
{funding_text}

INSTRUCTIONS: The SCORECARD at the bottom of each TF already counts bullish vs bearish indicators.
SMC SCORECARD counts structure, order blocks, FVG, zones separately.
MARKET POSITIONING shows crowd behavior (OI, L/S ratio, taker volume) — use to confirm or question your direction.
Combine ALL scorecards to derive your final LONG/SHORT %. DO NOT invent percentages — base them on actual indicator counts.
Cross-TF divergences = pullback risk. Entry = current price. Safe Entry = better entry from support/OB.
For SL/TP: cross-reference ALL data — find where indicators CONVERGE. Confluence = strongest levels. Use liquidation zones to identify potential price magnets.
"""

    # ---------------------------------------------------------
    # PHASE 1: SHOW LIVE STATUS IN TELEGRAM (if streaming)
    # ---------------------------------------------------------
    if telegram_stream:
        tg_session = telegram_stream["session"]
        tg_chat = telegram_stream["chat_id"]
        tg_msg = telegram_stream["message_id"]
        tg_token = telegram_stream["bot_token"]
        await _edit_telegram_msg(tg_session, tg_chat, tg_msg,
            "⚡ AiAlisa AI connected...\n🔍 Executing Binance Web3 Skills...", tg_token)
        await asyncio.sleep(0.5)
        await _edit_telegram_msg(tg_session, tg_chat, tg_msg,
            "⚡ AiAlisa AI connected...\n🧠 AI reasoning in progress...", tg_token)

    # ---------------------------------------------------------
    # PHASE 2: GET AI RESPONSE (multi-provider with fallback)
    # ---------------------------------------------------------
    ai_response = None

    # Build messages array based on mode
    if mode == "auto":
        messages = [
            {"role": "system", "content": get_fast_verdict_prompt(lang)},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

    global _ai_last_request_time
    await _ai_request_lock.acquire()
    try:
        elapsed = _time.monotonic() - _ai_last_request_time
        if elapsed < 10 and _ai_last_request_time > 0:
            wait = 10 - elapsed
            logging.info(f"⏳ AI queue: waiting {wait:.1f}s for rate limit cooldown...")
            await asyncio.sleep(wait)

        # Determine provider/model (override params take priority for monitor pipeline)
        if api_key_override or model_override:
            # Monitor pipeline or manual override — use OpenRouter directly
            _active_key = api_key_override or OPENROUTER_API_KEY
            _active_model = model_override or OPENROUTER_MODEL
            logging.info(f"⚡ [AiAlisa] AI inference via OpenRouter (override) model={_active_model}")

            # === REAL-TIME SSE STREAMING to Telegram (OpenRouter only) ===
            if telegram_stream:
                headers = {"Authorization": f"Bearer {_active_key}", "Content-Type": "application/json"}
                payload = {"model": _active_model, "messages": messages, "temperature": 0.15, "stream": True}
                if mode == "extended":
                    payload["reasoning"] = {"enabled": True}
                tg_s = telegram_stream["session"]
                tg_c = telegram_stream["chat_id"]
                tg_m = telegram_stream["message_id"]
                tg_t = telegram_stream["bot_token"]
                try:
                    accumulated = ""
                    last_edit = 0
                    token_count = 0
                    request_timeout = aiohttp.ClientTimeout(total=240)
                    async with aiohttp.ClientSession() as sess:
                        async with sess.post("https://openrouter.ai/api/v1/chat/completions",
                                             headers=headers, json=payload, timeout=request_timeout) as resp:
                            if resp.status == 200:
                                async for line in resp.content:
                                    line = line.decode("utf-8").strip()
                                    if not line.startswith("data: "):
                                        continue
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        delta = chunk["choices"][0].get("delta", {})
                                        token = delta.get("content", "")
                                        if token:
                                            accumulated += token
                                            token_count += 1
                                            now = _time.monotonic()
                                            if now - last_edit >= 1.5 and len(accumulated.strip()) > 10:
                                                display = accumulated.strip() + " ▌"
                                                if len(display) > 4000:
                                                    display = "..." + display[-3997:]
                                                await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                                    f"⚡ *AiAlisa AI Streaming* 🔴 LIVE\n\n{display}", tg_t)
                                                last_edit = now
                                    except (json.JSONDecodeError, KeyError, IndexError):
                                        continue
                                if accumulated.strip():
                                    final = accumulated.strip()
                                    if len(final) > 3900:
                                        final = final[:3900] + "..."
                                    await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                        f"⚡ *AiAlisa AI Complete* ✅\n\n{final}", tg_t, parse_mode="Markdown")
                                    ai_response = accumulated.strip()
                                    logging.info(f"✅ AiAlisa Live Stream: {token_count} tokens → Telegram (real-time)")
                            else:
                                logging.warning(f"⚠️ OpenRouter stream error: {resp.status}")
                except Exception as e:
                    logging.warning(f"⚠️ OpenRouter streaming error ({e}), trying non-stream...")

            # Non-streaming fallback for override mode
            if not ai_response:
                ai_response = await _call_openrouter(messages, _active_key, _active_model)

        else:
            # === MULTI-PROVIDER FALLBACK CHAIN ===
            s = _get_ai_settings()
            provider = s["active_provider"]
            logging.info(f"⚡ [AiAlisa] AI inference via {provider} (fallback chain)")

            # Streaming for OpenRouter manual scans
            if telegram_stream and provider == "openrouter":
                _or_key = OPENROUTER_API_KEY
                _or_model = s.get("openrouter_model", OPENROUTER_MODEL)
                headers = {"Authorization": f"Bearer {_or_key}", "Content-Type": "application/json"}
                payload = {"model": _or_model, "messages": messages, "temperature": 0.15, "stream": True}
                if mode == "extended":
                    payload["reasoning"] = {"enabled": True}
                tg_s = telegram_stream["session"]
                tg_c = telegram_stream["chat_id"]
                tg_m = telegram_stream["message_id"]
                tg_t = telegram_stream["bot_token"]
                try:
                    accumulated = ""
                    last_edit = 0
                    token_count = 0
                    request_timeout = aiohttp.ClientTimeout(total=240)
                    async with aiohttp.ClientSession() as sess:
                        async with sess.post("https://openrouter.ai/api/v1/chat/completions",
                                             headers=headers, json=payload, timeout=request_timeout) as resp:
                            if resp.status == 200:
                                async for line in resp.content:
                                    line = line.decode("utf-8").strip()
                                    if not line.startswith("data: "):
                                        continue
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        delta = chunk["choices"][0].get("delta", {})
                                        token = delta.get("content", "")
                                        if token:
                                            accumulated += token
                                            token_count += 1
                                            now = _time.monotonic()
                                            if now - last_edit >= 1.5 and len(accumulated.strip()) > 10:
                                                display = accumulated.strip() + " ▌"
                                                if len(display) > 4000:
                                                    display = "..." + display[-3997:]
                                                await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                                    f"⚡ *AiAlisa AI Streaming* 🔴 LIVE\n\n{display}", tg_t)
                                                last_edit = now
                                    except (json.JSONDecodeError, KeyError, IndexError):
                                        continue
                                if accumulated.strip():
                                    final = accumulated.strip()
                                    if len(final) > 3900:
                                        final = final[:3900] + "..."
                                    await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                        f"⚡ *AiAlisa AI Complete* ✅\n\n{final}", tg_t, parse_mode="Markdown")
                                    ai_response = accumulated.strip()
                                    logging.info(f"✅ AiAlisa Live Stream: {token_count} tokens → Telegram (real-time)")
                            else:
                                logging.warning(f"⚠️ OpenRouter stream error: {resp.status}")
                except Exception as e:
                    logging.warning(f"⚠️ OpenRouter streaming error ({e}), trying fallback chain...")

            # Non-streaming: full fallback chain
            if not ai_response:
                result, used_provider, used_model = await call_ai_with_fallback(messages)
                if result:
                    ai_response = result
                    logging.info(f"✅ AI response via {used_provider}/{used_model}")
                    # Update telegram status if streaming was attempted
                    if telegram_stream:
                        tg_s = telegram_stream["session"]
                        tg_c = telegram_stream["chat_id"]
                        tg_m = telegram_stream["message_id"]
                        tg_t = telegram_stream["bot_token"]
                        final = ai_response[:3900] + "..." if len(ai_response) > 3900 else ai_response
                        await _edit_telegram_msg(tg_s, tg_c, tg_m,
                            f"⚡ *AiAlisa AI Complete* ✅ ({used_provider})\n\n{final}", tg_t, parse_mode="Markdown")

    finally:
        _ai_last_request_time = _time.monotonic()
        _ai_request_lock.release()

    # ---------------------------------------------------------
    # PHASE 3: PROGRESSIVE DISPLAY (only if text obtained without streaming)
    # ---------------------------------------------------------
    if telegram_stream and ai_response and not ai_response.startswith("❌"):
        # Only do progressive display if we didn't already stream live
        # (Check: if streaming happened, the message already shows "AI Complete")
        pass  # Live streaming already handled above

    # Post-validate: fix literal X%/Y% placeholders from weak models
    if ai_response:
        ai_response = _fix_placeholder_percentages(ai_response, mtf_data=mtf_data, indicators=clean_indic, tf_key=tf_key)

    # Post-validate SL/TP direction in free-text responses
    if ai_response:
        ai_response = _validate_sl_tp_in_text(ai_response)

    return ai_response
