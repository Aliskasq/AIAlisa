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


async def _progressive_display(text, telegram_stream):
    """Progressively reveal AI text in Telegram via editMessageText.
    
    Takes already-obtained AI response and reveals it word-by-word
    with a typing cursor effect. Works regardless of text source
    (OpenClaw SDK, OpenRouter, or any other provider).
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
                f"⚡ *OpenClaw AI Complete* ✅\n\n{text}",
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
                f"⚡ *OpenClaw Live* ({i}/{len(words)})\n\n{display}",
                bot_token)
            await asyncio.sleep(0.7)

        # Final — complete text, no cursor
        final_display = text
        if len(final_display) > 3900:
            final_display = final_display[:3900] + "..."
        await _edit_telegram_msg(session, chat_id, message_id,
            f"⚡ *OpenClaw AI Complete* ✅\n\n{final_display}",
            bot_token, parse_mode="Markdown")

        logging.info(f"✅ OpenClaw Progressive Stream: {len(words)} words → Telegram")

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
    # PHASE 1: SHOW LIVE STATUS IN TELEGRAM (if streaming)
    # ---------------------------------------------------------
    if telegram_stream:
        tg_session = telegram_stream["session"]
        tg_chat = telegram_stream["chat_id"]
        tg_msg = telegram_stream["message_id"]
        tg_token = telegram_stream["bot_token"]
        await _edit_telegram_msg(tg_session, tg_chat, tg_msg,
            "⚡ OpenClaw AI Agent connected...\n🔍 Executing Binance Web3 Skills...", tg_token)
        await asyncio.sleep(0.5)
        await _edit_telegram_msg(tg_session, tg_chat, tg_msg,
            "⚡ OpenClaw AI Agent connected...\n🧠 AI reasoning in progress...", tg_token)

    # ---------------------------------------------------------
    # PHASE 2: GET AI RESPONSE (try all sources)
    # ---------------------------------------------------------
    ai_response = None
    full_prompt = f"{system_instruction}\n\n{user_prompt}"

    # --- STEP 1: OpenClaw SDK (Extract → Agent) ---
    if openclaw_installed and openclaw:
        try:
            logging.info("🧠 Attempting inference via openclaw.AsyncOpenClaw()...")
            client = openclaw.AsyncOpenClaw.remote(api_key=os.getenv("CMDOP_API_KEY"))

            # Try Extract (structured Pydantic output)
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
                    ai_response = _format_verdict(extract_result.data, base_coin, price, dynamics_text)
                    logging.info(f"✅ OpenClaw Extract: {extract_result.data.direction}")
                else:
                    logging.info("⚙️ Extract returned no data, trying agent.run...")
            except Exception as extract_err:
                logging.info(f"⚙️ Extract unavailable ({extract_err}), trying agent.run...")

            # Try Agent (free-text)
            if not ai_response:
                try:
                    result = await client.agent.run(full_prompt)
                    # Validate success before extracting
                    if hasattr(result, 'success') and result.success is False:
                        logging.info(f"⚙️ agent.run returned success=False, falling back to OpenRouter...")
                    elif hasattr(result, 'text') and result.text:
                        ai_response = result.text
                        logging.info("✅ OpenClaw agent.run inference successful.")
                    elif hasattr(result, 'content') and result.content:
                        ai_response = result.content
                        logging.info("✅ OpenClaw agent.run inference successful.")
                    elif isinstance(result, str) and len(result) > 20 and 'request_id=' not in result:
                        ai_response = result
                        logging.info("✅ OpenClaw agent.run inference successful.")
                    else:
                        logging.info("⚙️ agent.run returned empty/invalid, falling back to OpenRouter...")
                except Exception as agent_err:
                    logging.info(f"⚙️ agent.run error ({agent_err}), falling back to OpenRouter...")

        except Exception as e:
            logging.warning(f"⚠️ OpenClaw SDK error ({e}). Activating failsafe routing...")
    else:
        logging.warning("⚠️ OpenClaw not installed. Using failsafe AI routing...")

    # --- STEP 2: OpenRouter (with real-time streaming if Telegram active) ---
    if not ai_response:
        logging.info("🔄 Routing through OpenRouter...")
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
            "temperature": 0.2,
        }

        # === REAL-TIME SSE STREAMING to Telegram ===
        if telegram_stream:
            payload["stream"] = True
            tg_s = telegram_stream["session"]
            tg_c = telegram_stream["chat_id"]
            tg_m = telegram_stream["message_id"]
            tg_t = telegram_stream["bot_token"]
            try:
                accumulated = ""
                last_edit = 0
                token_count = 0
                async with aiohttp.ClientSession() as sess:
                    async with sess.post("https://openrouter.ai/api/v1/chat/completions",
                                         headers=headers, json=payload, timeout=180) as resp:
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
                                        # Update Telegram every 1.5s (rate limit safe)
                                        if now - last_edit >= 1.5 and len(accumulated.strip()) > 10:
                                            display = accumulated.strip() + " ▌"
                                            if len(display) > 4000:
                                                display = "..." + display[-3997:]
                                            await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                                f"⚡ *OpenClaw AI Streaming* 🔴 LIVE\n\n{display}", tg_t)
                                            last_edit = now
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue

                            if accumulated.strip():
                                # Final message — complete, no cursor
                                final = accumulated.strip()
                                if len(final) > 3900:
                                    final = final[:3900] + "..."
                                await _edit_telegram_msg(tg_s, tg_c, tg_m,
                                    f"⚡ *OpenClaw AI Complete* ✅\n\n{final}", tg_t, parse_mode="Markdown")
                                ai_response = accumulated.strip()
                                logging.info(f"✅ OpenClaw Live Stream: {token_count} tokens → Telegram (real-time)")
                        else:
                            logging.warning(f"⚠️ OpenRouter stream error: {resp.status}")
            except Exception as e:
                logging.warning(f"⚠️ OpenRouter streaming error ({e}), trying non-stream...")

        # === Non-streaming fallback (automated signals, no Telegram edit) ===
        if not ai_response:
            try:
                payload.pop("stream", None)
                async with aiohttp.ClientSession() as session:
                    async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                            headers=headers, json=payload, timeout=120) as response:
                        if response.status == 200:
                            data = await response.json()
                            ai_response = data["choices"][0]["message"]["content"].strip()
                            logging.info("✅ OpenRouter inference successful.")
                        else:
                            ai_response = f"❌ API Error: {response.status}"
            except Exception as e:
                ai_response = f"❌ Network Error: {e}"

    # ---------------------------------------------------------
    # PHASE 3: PROGRESSIVE DISPLAY (only if text obtained without streaming)
    # ---------------------------------------------------------
    if telegram_stream and ai_response and not ai_response.startswith("❌"):
        # Only do progressive display if we didn't already stream live
        # (Check: if streaming happened, the message already shows "AI Complete")
        pass  # Live streaming already handled above

    return ai_response
