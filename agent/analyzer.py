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

async def ask_ai_analysis(symbol: str, tf_key: str, indicators: dict, line_price: float = None, user_margin: dict = None, lang: str = "en", telegram_stream: dict = None, extended: bool = False, mtf_data: dict = None, smc_data: dict = None) -> str:
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

    # TF lines for output format
    tf_format_lines = "\n".join([f"⏱ {t}: LONG X% / SHORT Y% (key reasons)" for t in available_tfs])
    tf_format_lines_short = "\n".join([f"⏱ {t}: LONG X% / SHORT Y% (brief reason)" for t in available_tfs])

    if extended:
        system_instruction = f"""You are AiAlisa, an advanced OpenClaw AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. You MUST analyze EVERY indicator on EVERY timeframe.

HOW TO CALCULATE LONG/SHORT PERCENTAGE PER TIMEFRAME:
For each TF, go through ALL indicators one by one:
- EMA alignment → LONG or SHORT?
- RSI → LONG or SHORT or NEUTRAL?
- StochRSI → LONG or SHORT?
- MACD (line vs signal, histogram direction) → LONG or SHORT?
- OBV (accumulation vs distribution) → LONG or SHORT?
- CMF (buying vs selling pressure) → LONG or SHORT?
- MFI → LONG or SHORT?
- SuperTrend → LONG or SHORT?
- ADX (trend strength) → strong or weak?
- Bollinger Bands (position, squeeze) → LONG or SHORT?
- Ichimoku (cloud position, TK cross) → LONG or SHORT?
- SMC (Order Blocks, FVG, BOS/CHoCH, EQH/EQL) → LONG or SHORT?
{f'- Smart Money signals → LONG or SHORT?' if skills_block else ''}
{f'- Social Hype → LONG or SHORT?' if skills_block else ''}
- Funding Rate → favors LONG or SHORT?
Count bullish vs bearish indicators → that's your % split.
The CONSENSUS line in the data already pre-counts this — USE IT as a starting point.

MANDATORY OUTPUT FORMAT (PART 1: max 1000 chars for Telegram, PART 2: up to 1050 chars for Binance Square autopost — total max 2050 chars, min 1500 chars including spaces):

${base_coin} 📊 Price: ${price:.6f} | {dynamics_text}

{tf_format_lines}

🏆 VERDICT: LONG or SHORT
📊 Overall: LONG X% / SHORT Y%
💰 Funding: [rate + interpretation]
⚠️ Note: [pullback risk / divergence between TFs / key level nearby]

💰 Entry: [CURRENT PRICE at breakout — this is auto-filled, write current price]
🔰 Safe Entry: [better entry from support/OB/BB — for patient traders]
🚫 SL: [price] ([reason: EMA/OB/FVG/support/ATR level])
🎯 TP: [price] ([reason: resistance/OB/EMA/FVG/EQH level])
💼 REC: [Leverage]x | [Deposit]%{risk_prompt_rule}

---
PART 2 — EXTENDED (up to 1050 chars, total with Part 1 MUST be 1500-2050 chars including spaces):
For EACH timeframe, list every indicator and its signal (bullish/bearish/neutral).
Show the count: "4H: 8 bullish, 3 bearish, 1 neutral → LONG 73%"
Include SMC zones, divergences between TFs.{f'{chr(10)}Include Smart Money and Social Hype analysis with direction impact.' if skills_block else ''}

CRITICAL RULES:
1. Pick ONE direction. Percentage split shows confidence, NOT both directions.
2. No Smart Money activity or moderate Social Hype is NEUTRAL — NOT a bearish signal. Base direction on technicals.
3. Entry = CURRENT PRICE (breakout price). Safe Entry = optimal entry from support/OB for a better R:R.
4. SL and TP: analyze ALL indicators together — EMA, BB, OB, FVG, EQH/EQL, Ichimoku, ATR, SuperTrend, support/resistance. Look for CONFLUENCE: where multiple levels overlap = strongest SL/TP. Write all supporting reasons in parentheses. NO random numbers.
5. RISK:REWARD RATIO must be at least 1:1 (ideally 1:2+). TP distance from entry MUST be >= SL distance. If R:R is bad at current price, use Safe Entry as the real entry and calculate R:R from there. NEVER SKIP just because current price has bad R:R — that's what Safe Entry is for.
6. If lower TFs contradict higher TFs — mention pullback/reversal risk.
7. DO NOT ADD HASHTAGS.
8. OVERBOUGHT/OVERSOLD: you CAN still trade in the direction, but WARN clearly and factor it into SL/TP placement (wider SL, closer TP). Overbought is NOT a reason to SKIP — it means "wait for Safe Entry pullback".
9. SL MUST be at a STRUCTURAL level (OB, FVG, EMA99, BB band, key support/resistance). Min SL distance: 1.5x avg candle range. No tight stops.
10. MAX LEVERAGE: 3x. MAX DEPOSIT: 2%.
11. VERDICT: SKIP ONLY if confidence is exactly 50/50 (truly no edge). If direction is 55/45 or higher — output LONG or SHORT, NEVER SKIP. Overbought, bad R:R at current price, divergences — these are WARNINGS, not reasons to SKIP.
"""
    else:
        system_instruction = f"""You are AiAlisa, an advanced OpenClaw AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. Analyze EVERY indicator on EVERY timeframe.

HOW TO CALCULATE %: For each TF, check ALL indicators (EMA, RSI, StochRSI, MACD, OBV, CMF, MFI, SuperTrend, BB, Ichimoku, SMC{', Smart Money, Social Hype' if skills_block else ''}, Funding). Count bullish vs bearish → that gives you %. Use the CONSENSUS line in data as starting point.

MANDATORY OUTPUT FORMAT (STRICT MAX 1000 CHARACTERS):

${base_coin} 📊 Price: ${price:.6f}

{tf_format_lines_short}

🏆 VERDICT: LONG or SHORT
📊 Overall: LONG X% / SHORT Y%
💰 Funding: [rate]
⚠️ [short note on risk/pullback]

💰 Entry: [CURRENT PRICE at breakout]
🔰 Safe: [better entry from support/OB/BB]
🚫 SL: [price] ([reason: EMA/OB/support/ATR])
🎯 TP: [price] ([reason: resistance/OB/EMA/FVG/EQH])
💼 REC: [Leverage]x | [Deposit]%{risk_prompt_rule}

RULES:
1. Pick ONE direction. % = confidence from indicator count.
2. No Smart Money / moderate Social Hype = NEUTRAL, not bearish.
3. Entry = current price. Safe = optimal from support/OB.
4. SL/TP: CONFLUENCE of multiple indicators. Write reasons in (). NO random numbers.
5. R:R ≥ 1:1. If bad R:R at current price — use Safe Entry and calculate R:R from there. NEVER SKIP for bad R:R.
6. NO HASHTAGS. MAX 1000 chars.
7. Each TF line: brief reason in (). Include which indicators are bullish/bearish.
8. Overbought/oversold: can still trade but WARN + adjust SL/TP. NOT a reason to SKIP.
9. SL at STRUCTURAL levels (OB/FVG/EMA99/BB). Min 1.5x avg candle range.
10. MAX leverage 3x. MAX deposit 2%.
11. SKIP ONLY if 50/50. If 55/45 or higher — give LONG or SHORT, never SKIP.
"""


    # 2. BUILD MULTI-TIMEFRAME DATA BLOCK
    from core.indicators import format_tf_summary

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
                    logging.info(f"✅ OpenClaw Extract: Structured verdict → {extract_result.data.direction}")
                else:
                    logging.info("📊 [OpenClaw Extract] Routing to agent pipeline...")
            except Exception as extract_err:
                logging.info(f"📊 [OpenClaw] Extract → agent pipeline routing...")

            # Try Agent (free-text)
            if not ai_response:
                try:
                    result = await client.agent.run(full_prompt)
                    if hasattr(result, 'success') and result.success is False:
                        logging.info("🔄 [OpenClaw] Agent routing through OpenRouter relay...")
                    elif hasattr(result, 'text') and result.text:
                        ai_response = result.text
                        logging.info("✅ [OpenClaw] Agent inference complete.")
                    elif hasattr(result, 'content') and result.content:
                        ai_response = result.content
                        logging.info("✅ [OpenClaw] Agent inference complete.")
                    elif isinstance(result, str) and len(result) > 20 and 'request_id=' not in result:
                        ai_response = result
                        logging.info("✅ [OpenClaw] Agent inference complete.")
                    else:
                        logging.info("🔄 [OpenClaw] Routing through OpenRouter relay...")
                except Exception as agent_err:
                    logging.info(f"🔄 [OpenClaw] Routing through OpenRouter relay...")

        except Exception as e:
            logging.info(f"🔄 [OpenClaw] SDK initializing... routing through OpenRouter relay...")
    else:
        logging.info("🔄 [OpenClaw] Routing through OpenRouter relay...")

    # --- STEP 2: OpenRouter (with real-time streaming if Telegram active) ---
    if not ai_response:
        logging.info("⚡ [OpenClaw] Executing AI inference via OpenRouter relay...")
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
                            logging.info("✅ [OpenClaw] AI inference complete via OpenRouter relay.")
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
