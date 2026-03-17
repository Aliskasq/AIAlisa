import json
import logging
import aiohttp
import asyncio
import math
import os
import time as _time
from typing import Optional
from pydantic import BaseModel
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

# Import OpenClaw Architecture dependencies
try:
    import openclaw
    import cmdop
    openclaw_installed = True
except ImportError:
    openclaw = None
    cmdop = None
    openclaw_installed = False


# ---------------------------------------------------------
# OPENCLAW STRUCTURED OUTPUT: AI TRADING VERDICT MODEL
# ---------------------------------------------------------
class TradeVerdict(BaseModel):
    """OpenClaw Structured Trading Verdict — Binance AI Analysis.
    
    Used with cmdop ExtractService to return typed, validated
    trading signals instead of raw unstructured text.
    """
    direction: str          # "LONG" or "SHORT"
    entry_price: float      # Recommended entry price
    stop_loss: float        # Stop-loss price level
    take_profit: float      # Take-profit price level
    risk_percent: float     # Estimated risk as % of margin
    leverage_rec: str       # e.g. "5x", "10x"
    deposit_rec: str        # e.g. "10%", "15%"
    logic: str              # AI reasoning (max 5 sentences)
    risk_note: Optional[str] = None  # Stop-loss calculation for margin queries


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
# OPENCLAW AGENT STREAMING: LIVE AI IN TELEGRAM
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


async def _stream_verdict(client, prompt, telegram_stream):
    """Stream AI verdict to Telegram with progressive display.
    
    Uses OpenClaw agent.run() for inference, then progressively
    reveals the response in Telegram via editMessageText — creating
    a real-time "thinking" effect for the user.
    
    Falls back gracefully if anything fails.
    Returns the final text, or None on failure.
    """
    session = telegram_stream["session"]
    chat_id = telegram_stream["chat_id"]
    message_id = telegram_stream["message_id"]
    bot_token = telegram_stream["bot_token"]

    try:
        # Phase 1: Show skill execution status
        await _edit_telegram_msg(session, chat_id, message_id,
            "⚡ OpenClaw AI Agent connected...\n🔍 Executing Binance Web3 Skills...", bot_token)
        await asyncio.sleep(0.5)

        # Phase 2: Show AI reasoning
        await _edit_telegram_msg(session, chat_id, message_id,
            "⚡ OpenClaw AI Agent connected...\n🧠 AI reasoning in progress...", bot_token)

        # Phase 3: Get full AI response via OpenClaw SDK
        result = await client.agent.run(prompt)

        # Extract response text (mirrors main fallback logic exactly)
        response = None
        if hasattr(result, 'content') and result.content:
            response = result.content
        elif hasattr(result, 'text') and result.text:
            response = result.text
        elif isinstance(result, str) and result:
            response = result
        else:
            response = str(result)

        if not response or len(response.strip()) < 10:
            return None

        response = response.strip()

        # Phase 4: Progressive reveal — show text appearing chunk by chunk
        chunks = []
        words = response.split()
        chunk_size = max(3, len(words) // 8)  # ~8 updates total
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[:i + chunk_size]))

        for i, partial in enumerate(chunks[:-1]):  # Skip last — will be final
            display = partial + " ▌"
            if len(display) > 4000:
                display = display[:4000]
            await _edit_telegram_msg(session, chat_id, message_id,
                f"⚡ *OpenClaw Live* ({len(partial.split())}/{len(words)} words)\n\n{display}",
                bot_token)
            await asyncio.sleep(0.8)

        # Phase 5: Final complete message
        final_display = response
        if len(final_display) > 3900:
            final_display = final_display[:3900] + "..."
        await _edit_telegram_msg(session, chat_id, message_id,
            f"⚡ *OpenClaw AI Complete* ✅\n\n{final_display}",
            bot_token, parse_mode="Markdown")

        logging.info(f"✅ OpenClaw Progressive Stream: {len(words)} words → Telegram")
        return response

    except Exception as e:
        logging.info(f"⚙️ Progressive stream error ({e}), falling back...")

    return None


# Load the FULL arsenal of Native Binance Skills
from agent.skills import (
    get_smart_money_signals,
    get_unified_token_rank,
    get_social_hype_leaderboard,
    get_smart_money_inflow_rank,
    get_meme_rank,
    get_address_pnl_rank
)

async def ask_ai_analysis(symbol: str, tf_key: str, indicators: dict, line_price: float = None, user_margin: dict = None, lang: str = "en", telegram_stream: dict = None) -> str:
    """
    OpenClaw Architectural Agent: Executes Binance Market Intelligence Skills natively
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

    # Format nice Fibonacci text
    fibo_text = ", ".join([f"{k}: {v:.4f}" for k, v in clean_indic.get("fibo_levels", {}).items()])

    # Build dynamic text to avoid duplicate stats on 1D timeframe
    if recent_label.upper() == "1D" or tf_key.upper() == "1D":
        dynamics_text = f"Over the last 24H changed by {change_24h}%."
    else:
        dynamics_text = f"Over the last {recent_label} changed by {change_recent}%, and over 24H by {change_24h}%."

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
    if line_price:
        breakout_context = f"Trigger line breakout at {line_price:.6f}."

    if lang == "en":
        lang_directive = "Respond strictly in ENGLISH."
    else:
        lang_directive = "Respond strictly in RUSSIAN. Translate all output labels (like Current Price, VERDICT, LOGIC, TRADE, RISK, REC) to Russian."

    # =========================================================
    # AGENTIC ACTION: NATIVELY CALLING BINANCE SKILLS
    # =========================================================
    # Forced initialization logs for Architectural visibility
    logging.info(f"🤖 AiAlisa Agent initialized for {symbol}...")
    logging.info(f"⚙️ [OpenClaw Architecture] Executing Agentic Binance Skills...")

    smart_money_context = await get_smart_money_signals(symbol)
    hype_context = await get_social_hype_leaderboard()

    # =========================================================
    # 1. SYSTEM SETUP
    # =========================================================
    # Length limits tuned to max 900 chars and tightly controlled logic (up to 5 sentences).
    system_instruction = f"""You are AiAlisa, an advanced OpenClaw AI Agent and Binance Crypto Influencer. ATTENTION: PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
You have executed your Internal Binance Web3 Plugins. Here are your findings:
- SMART MONEY ACTIVITY: {smart_money_context}
- SOCIAL HYPE TRENDS: {hype_context}

CRITICAL RULES:
1. Pick ONE direction (LONG or SHORT).
2. MUST EXPLAIN LOGIC: explicitly evaluate Funding Rate, Smart Money, Social Hype, and 2 Technical Indicators.
3. LOSS CALCULATION Formula: $loss = (|Entry - SL| / Entry) * Leverage * 100$.{risk_prompt_rule}
4. Your response must be extremely structured.
5. DO NOT ADD ANY HASHTAGS.
6. STRICT LENGTH LIMIT: YOUR ENTIRE RESPONSE MUST BE UNDER 900 CHARACTERS. Keep the LOGIC section concise (maximum 5 short sentences).

MANDATORY OUTPUT STRUCTURE (Append below custom risk sentence if applicable):
${base_coin} 📊 Current Price: ${price:.6f}. {dynamics_text}
🏆 VERDICT: [LONG or SHORT]
🧠 LOGIC: [Concise logic, max 5 short sentences referencing Funding, Smart Money, Hype + 2 indicators]
🎯 TRADE: 💰 Entry: [Price] | 🚫 SL: [Price] | 🎯 TP: [Price]
🚫 RISK: [State Margin Loss % here]
💼 REC: [Leverage]x | [Deposit]%
"""


    # 2. FULL MATHEMATICAL DATA
    user_prompt = f"""Evaluate {symbol} on {tf_key}. {breakout_context} {user_risk_text}

[BOT MATHEMATICAL DATA]
Price: {price:.6f}
Trend: {clean_indic.get("supertrend", "Unknown")} (Level: {clean_indic.get("supertrend_price", 0):.6f})
ADX: {clean_indic.get("adx", 0):.2f} | MFI: {clean_indic.get("mfi", 0):.2f} | StochRSI: {clean_indic.get("stoch_k", 0):.2f}
RSI(6): {clean_indic.get("rsi6", 0):.2f} | Volume Decay: {clean_indic.get("volume_decay", "Unknown")} | OBV Trend: {clean_indic.get("obv_status", "Unknown")}
Ichimoku: {clean_indic.get("ichimoku_status", "Unknown")} | MACD: {'Positive' if clean_indic.get("macd_hist", 0) > 0 else 'Negative'}
EMA: 7={clean_indic.get("ema7", 0):.4f}, 25={clean_indic.get("ema25", 0):.4f}, 99={clean_indic.get("ema99", 0):.4f}
Funding: {clean_indic.get("funding_rate", "Unknown")}
Bullish FVG: {clean_indic.get("smc_bullish_fvg", "None")} | Bearish FVG: {clean_indic.get("smc_bearish_fvg", "None")}
Support OB: {clean_indic.get("smc_bullish_ob", "None")} | Resist OB: {clean_indic.get("smc_bearish_ob", "None")}
Fibo: {fibo_text}
"""

    # ---------------------------------------------------------
    # ROUTING: PRIMARY INFERENCE VIA OPENCLAW SDK
    # ---------------------------------------------------------
    if openclaw_installed and openclaw:
        try:
            logging.info("🧠 Attempting inference via openclaw.AsyncOpenClaw()...")
            
            # Connect to CMDOP Cloud securely using .env key
            client = openclaw.AsyncOpenClaw.remote(api_key=os.getenv("CMDOP_API_KEY"))

            # Merge system and user prompts into a single string for the new SDK
            full_prompt = f"{system_instruction}\n\n{user_prompt}"

            # =====================================================
            # STEP 1: OpenClaw Agent Streaming (Live Telegram UI)
            # =====================================================
            # If telegram_stream is provided, stream AI tokens in real-time
            # to a Telegram message via editMessageText. Users see the AI
            # "thinking" live — massive UX improvement over static "⏳".
            if telegram_stream:
                try:
                    logging.info("⚡ [OpenClaw Stream] Starting live AI stream to Telegram...")
                    stream_result = await _stream_verdict(client, full_prompt, telegram_stream)
                    if stream_result:
                        logging.info("✅ OpenClaw Streaming completed successfully.")
                        return stream_result
                    else:
                        logging.info("⚙️ Streaming returned empty, falling back to Extract...")
                except Exception as stream_err:
                    logging.info(f"⚙️ Streaming error ({stream_err}), falling back to Extract...")

            # =====================================================
            # STEP 2: OpenClaw Extract — Structured TradeVerdict
            # =====================================================
            # Uses CMDOP ExtractService to return a validated Pydantic
            # model instead of raw text. This ensures typed entry/SL/TP
            # values and enables programmatic signal processing.
            try:
                logging.info("📊 [OpenClaw Extract] Requesting structured TradeVerdict...")
                extract_result = await client.extract.run(
                    model=TradeVerdict,
                    prompt=full_prompt,
                    options=cmdop.ExtractOptions(
                        temperature=0.2,
                        timeout_seconds=120,
                        max_tokens=4096
                    )
                )
                if extract_result.data:
                    response = _format_verdict(extract_result.data, base_coin, price, dynamics_text)
                    logging.info(f"✅ OpenClaw Extract: Structured verdict received → {extract_result.data.direction} "
                                 f"(Entry: {extract_result.data.entry_price}, SL: {extract_result.data.stop_loss}, "
                                 f"TP: {extract_result.data.take_profit})")
                    return response
                else:
                    logging.info("⚙️ Extract returned no structured data, falling back to agent.run...")
            except Exception as extract_err:
                logging.info(f"⚙️ Extract unavailable ({extract_err}), falling back to agent.run...")

            # =====================================================
            # STEP 3: Fallback — OpenClaw Agent (existing behavior)
            # =====================================================
            result = await client.agent.run(full_prompt)
            
            # Extract content handling potential wrapper classes returned by the library
            if hasattr(result, 'content'):
                response = result.content
            elif hasattr(result, 'text'):
                response = result.text
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)
                
            if response:
                logging.info("✅ OpenClaw agent.run inference successful.")
                return response
        except Exception as e:
            logging.warning(f"⚠️ OpenClaw SDK timeout/error ({e}). Activating failsafe routing...")
    else:
        logging.warning("⚠️ OpenClaw core detected missing OS dependencies. Activating built-in failsafe AI routing...")

    # ---------------------------------------------------------
    # ROUTING: FAILSAFE AIOHTTP EXECUTION VIA OPENROUTER
    # ---------------------------------------------------------
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    return f"❌ API Error: {response.status}"
    except Exception as e:
        return f"❌ Network Error: {e}"
