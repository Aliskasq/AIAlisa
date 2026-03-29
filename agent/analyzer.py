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


def _validate_sl_tp_in_text(text: str) -> str:
    """Post-validate SL/TP consistency in free-text AI response.
    If LONG verdict has SL > Entry or SHORT has SL < Entry, add a warning."""
    import re
    direction_m = re.search(r'VERDICT[:\s]*(LONG|SHORT)', text, re.IGNORECASE)
    entry_m = re.search(r'(?:Entry|Вход)[:\s]*\$?([\d.]+)', text, re.IGNORECASE)
    sl_m = re.search(r'(?:SL|Стоп)[:\s]*\$?([\d.]+)', text, re.IGNORECASE)
    if direction_m and entry_m and sl_m:
        direction = direction_m.group(1).upper()
        entry = float(entry_m.group(1))
        sl = float(sl_m.group(1))
        if direction == "LONG" and sl > entry:
            logging.warning(f"⚠️ [SL WARN] LONG but SL({sl}) > Entry({entry}) in text response!")
            text += "\n\n⚠️ ВНИМАНИЕ: SL выше входа для LONG — проверьте уровни вручную!"
        elif direction == "SHORT" and sl < entry:
            logging.warning(f"⚠️ [SL WARN] SHORT but SL({sl}) < Entry({entry}) in text response!")
            text += "\n\n⚠️ ВНИМАНИЕ: SL ниже входа для SHORT — проверьте уровни вручную!"
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

async def ask_ai_analysis(symbol: str, tf_key: str, indicators: dict, line_price: float = None, user_margin: dict = None, lang: str = "en", telegram_stream: dict = None, extended: bool = False, square: bool = False, mtf_data: dict = None, smc_data: dict = None) -> str:
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
The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TIMEFRAME WEIGHTING FOR FINAL VERDICT (entry is at CURRENT price, so short-term matters most):
- 15m = 35% weight (most important — immediate momentum and entry timing)
- 1H = 30% weight (short-term trend confirmation)
- 4H = 20% weight (medium-term context)
- 1D = 15% weight (background trend, least weight for entry direction)
If 15m+1H disagree with 4H+1D, the SHORT-TERM view wins for the VERDICT direction.
Example: 1D LONG 100% + 4H LONG 100% but 1H SHORT 80% + 15m SHORT 90% = VERDICT should be SHORT (short-term bearish momentum dominates for current-price entry).

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money = trend continuation, OI falling = positions closing = trend weakening
- RSI overbought penalties are already calculated in the scorecards
- Each indicator has historical context showing dynamics over multiple candles

YOUR RESPONSE MUST CONTAIN TWO PARTS SEPARATED BY --- ON ITS OWN LINE.
Do NOT write any headers like "PART 1", "PART 2", "=== PART ===", "Brief", "Extended" etc. Just the content.

FIRST PART (chart caption, STRICT 600-828 characters including spaces):

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
1. First part MUST be 600-828 characters (including spaces). Count carefully!
2. Second part MUST be 2000-3800 characters. No more, no less.
3. Separate parts with exactly --- on its own line
4. Do NOT write any labels, headers, or markers like "Part 1", "Part 2", "=== PART ===" etc.
5. Entry = CURRENT PRICE. Safe Entry = better entry from support/OB
6. SL/TP: CONFLUENCE of multiple indicators. SL distance: 2-10% from entry, must be < TP distance. CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
7. MAX LEVERAGE: 3x. MAX DEPOSIT: 2%
8. DO NOT ADD HASHTAGS
9. OVERBOUGHT/OVERSOLD RULES: RSI >75 on 1 TF = warn, reduce 10%. RSI >75 on 2+ TFs = reduce 25%+, NEVER 100% LONG. RSI >75 on 3+ TFs = consider SKIP or SHORT.
10. SKIP only if truly 50/50 or 3+ TFs overbought/oversold. Otherwise give direction.
11. You MUST always output BOTH parts. Never skip the second part.
"""
    elif square:
        system_instruction = f"""You are AiAlisa, an advanced OpenClaw AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. Analyze EVERY indicator on EVERY timeframe.

The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TIMEFRAME WEIGHTING FOR FINAL VERDICT (entry is at CURRENT price, so short-term matters most):
- 15m = 35% weight (most important — immediate momentum and entry timing)
- 1H = 30% weight (short-term trend confirmation)
- 4H = 20% weight (medium-term context)
- 1D = 15% weight (background trend, least weight for entry direction)
If 15m+1H disagree with 4H+1D, the SHORT-TERM view wins for the VERDICT direction.

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money = trend continuation, OI falling = positions closing
- RSI overbought penalties are already calculated in the scorecards
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
3. SL/TP: CONFLUENCE of multiple indicators. SL distance: 2-10% from entry, must be < TP distance. CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
4. MAX leverage 3x. MAX deposit 2%
5. Response MUST be 1300-1900 characters. Header/footer will add ~200 chars to reach 1500-2100 total.
6. DO NOT ADD HASHTAGS — they are added automatically
7. OVERBOUGHT/OVERSOLD: RSI >75 on 2+ TFs = reduce confidence, warn clearly
8. SKIP only if truly 50/50 or 3+ TFs overbought/oversold. Otherwise give direction.
"""
    else:
        system_instruction = f"""You are AiAlisa, an advanced OpenClaw AI Agent and Binance Crypto Influencer. PAPER TRADING SIMULATION. NO REAL MONEY.
{lang_directive}
{skills_note}
You receive MULTI-TIMEFRAME data: {tf_list_str}. Analyze EVERY indicator on EVERY timeframe.

The SCORECARD at the bottom of each TF already pre-counts weighted bullish vs bearish indicators.
Use these scorecards as your starting point and combine across timeframes.

TIMEFRAME WEIGHTING FOR FINAL VERDICT (entry is at CURRENT price, so short-term matters most):
- 15m = 35% weight (most important — immediate momentum and entry timing)
- 1H = 30% weight (short-term trend confirmation)
- 4H = 20% weight (medium-term context)
- 1D = 15% weight (background trend, least weight for entry direction)
If 15m+1H disagree with 4H+1D, the SHORT-TERM view wins for the VERDICT direction.

IMPORTANT DIRECTIONAL RULES:
- Funding rate and L/S Account Ratio are INFO ONLY — display them but do NOT use for direction voting
- Open Interest change IS a directional indicator — OI rising = new money, OI falling = positions closing
- RSI overbought penalties are already calculated in the scorecards

MANDATORY OUTPUT FORMAT (Your response MUST be between 600 and 828 characters including spaces. Count carefully. Do NOT go under 600 or over 828.):

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
2. SL/TP: CONFLUENCE of multiple indicators. SL distance: 2-10% from entry, must be < TP distance. CRITICAL: For LONG — SL MUST be BELOW entry, TP MUST be ABOVE entry. For SHORT — SL MUST be ABOVE entry, TP MUST be BELOW entry. NEVER place SL on the wrong side of entry!
3. MAX leverage 3x. MAX deposit 2%
4. Each TF line: brief reason in parentheses
5. Response MUST be 600-828 characters exactly
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

    # --- STEP 1: OpenClaw SDK (Extract → Agent) with timeout + validation ---
    SDK_TIMEOUT = 60  # seconds — don't let SDK hang longer than this
    if openclaw_installed and openclaw:
        try:
            logging.info("🧠 Attempting inference via openclaw.AsyncOpenClaw()...")
            client = openclaw.AsyncOpenClaw.remote(api_key=os.getenv("CMDOP_API_KEY"))

            # Try Extract (structured Pydantic output)
            try:
                logging.info(f"📊 [OpenClaw Extract] Requesting structured TradeVerdict via {OPENROUTER_MODEL}...")
                extract_result = await asyncio.wait_for(
                    client.extract.run(
                        model=TradeVerdict,
                        prompt=full_prompt,
                        options=cmdop.ExtractOptions(
                            temperature=0.2,
                            timeout_seconds=SDK_TIMEOUT,
                            max_tokens=4096,
                            model=OPENROUTER_MODEL,
                        )
                    ),
                    timeout=SDK_TIMEOUT
                )
                if extract_result.data:
                    v = extract_result.data
                    # Validate SL/TP direction consistency
                    if v.direction.upper() == "LONG" and v.stop_loss > v.entry_price:
                        logging.warning(f"⚠️ [SL FIX] LONG but SL({v.stop_loss}) > Entry({v.entry_price}), swapping SL↔TP")
                        v.stop_loss, v.take_profit = v.take_profit, v.stop_loss
                    elif v.direction.upper() == "SHORT" and v.stop_loss < v.entry_price:
                        logging.warning(f"⚠️ [SL FIX] SHORT but SL({v.stop_loss}) < Entry({v.entry_price}), swapping SL↔TP")
                        v.stop_loss, v.take_profit = v.take_profit, v.stop_loss
                    candidate = _format_verdict(v, base_coin, price, dynamics_text)
                    if _is_valid_analysis(candidate):
                        ai_response = candidate
                        logging.info(f"✅ OpenClaw Extract: Structured verdict → {v.direction}")
                    else:
                        logging.warning(f"⚠️ [OpenClaw Extract] Response failed validation, skipping. Preview: {candidate[:120]}")
                else:
                    logging.info("📊 [OpenClaw Extract] No data returned, trying agent pipeline...")
            except asyncio.TimeoutError:
                logging.warning(f"⚠️ [OpenClaw Extract] Timed out after {SDK_TIMEOUT}s")
            except Exception as extract_err:
                logging.warning(f"⚠️ [OpenClaw Extract] Error: {type(extract_err).__name__}: {extract_err}")

            # Try Agent (free-text)
            if not ai_response:
                try:
                    result = await asyncio.wait_for(
                        client.agent.run(full_prompt),
                        timeout=SDK_TIMEOUT
                    )
                    candidate = None
                    if hasattr(result, 'success') and result.success is False:
                        logging.info("🔄 [OpenClaw Agent] success=False, falling through...")
                    elif hasattr(result, 'text') and result.text:
                        candidate = result.text
                    elif hasattr(result, 'content') and result.content:
                        candidate = result.content
                    elif isinstance(result, str) and len(result) > 20 and 'request_id=' not in result:
                        candidate = result

                    if candidate and _is_valid_analysis(candidate):
                        ai_response = candidate
                        logging.info(f"✅ [OpenClaw Agent] Inference complete ({len(candidate)} chars)")
                    elif candidate:
                        logging.warning(f"⚠️ [OpenClaw Agent] Response failed validation. Preview: {candidate[:200]}")
                    else:
                        logging.info("🔄 [OpenClaw Agent] No valid response, falling through to OpenRouter direct...")
                except asyncio.TimeoutError:
                    logging.warning(f"⚠️ [OpenClaw Agent] Timed out after {SDK_TIMEOUT}s")
                except Exception as agent_err:
                    logging.warning(f"⚠️ [OpenClaw Agent] Error: {type(agent_err).__name__}: {agent_err}")

        except Exception as e:
            logging.warning(f"⚠️ [OpenClaw SDK] Init error: {type(e).__name__}: {e}")
    else:
        logging.info("🔄 [OpenClaw SDK] Not installed, using OpenRouter direct...")

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
        # Enable reasoning only for manual analysis (not all models/providers support it)
        if telegram_stream:
            payload["reasoning"] = {"enabled": True}

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
            for attempt in range(2):  # retry once on failure
                try:
                    payload.pop("stream", None)
                    payload.pop("reasoning", None)  # strip reasoning for non-stream
                    async with aiohttp.ClientSession() as session:
                        async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                                headers=headers, json=payload, timeout=180) as response:
                            if response.status == 200:
                                data = await response.json()
                                candidate = data["choices"][0]["message"]["content"].strip()
                                if _is_valid_analysis(candidate):
                                    ai_response = candidate
                                    logging.info("✅ [OpenRouter Direct] AI inference complete.")
                                else:
                                    logging.warning(f"⚠️ [OpenRouter Direct] Response failed validation (attempt {attempt+1}). Preview: {candidate[:200]}")
                                    if attempt == 1:
                                        ai_response = candidate  # accept on last attempt anyway
                            elif response.status == 429:
                                retry_after = int(response.headers.get("Retry-After", "5"))
                                logging.warning(f"⚠️ [OpenRouter Direct] Rate limited, waiting {retry_after}s...")
                                await asyncio.sleep(retry_after)
                                continue
                            else:
                                body = await response.text()
                                logging.warning(f"⚠️ [OpenRouter Direct] HTTP {response.status}: {body[:300]}")
                                ai_response = f"❌ API Error: {response.status}"
                    if ai_response:
                        break
                except Exception as e:
                    logging.warning(f"⚠️ [OpenRouter Direct] Network error (attempt {attempt+1}): {type(e).__name__}: {e}")
                    if attempt == 0:
                        await asyncio.sleep(3)  # wait before retry
                    else:
                        ai_response = f"❌ Network Error: {e}"

    # ---------------------------------------------------------
    # PHASE 3: PROGRESSIVE DISPLAY (only if text obtained without streaming)
    # ---------------------------------------------------------
    if telegram_stream and ai_response and not ai_response.startswith("❌"):
        # Only do progressive display if we didn't already stream live
        # (Check: if streaming happened, the message already shows "AI Complete")
        pass  # Live streaming already handled above

    # Post-validate SL/TP direction in free-text responses
    if ai_response:
        ai_response = _validate_sl_tp_in_text(ai_response)

    # Hard-trim automated signals (no telegram_stream = auto-push caption)
    # StepFun free tier ignores max_tokens, so we must enforce length here
    if ai_response and not telegram_stream and not square and not ai_response.startswith("❌"):
        # For two-part responses (scan/look), split and trim separately
        if "---" in ai_response:
            parts = ai_response.split("---", 1)
            part1 = parts[0].strip()
            part2 = parts[1].strip() if len(parts) > 1 else ""
            if len(part1) > 828:
                # Trim to last newline before 828
                cut = part1[:828].rfind("\n")
                part1 = part1[:cut] if cut > 400 else part1[:828]
            if len(part2) > 3800:
                cut = part2[:3800].rfind("\n")
                part2 = part2[:cut] if cut > 2000 else part2[:3800]
            ai_response = f"{part1}\n---\n{part2}" if part2 else part1
        else:
            # Single-part auto-push caption: max 828 chars
            if len(ai_response) > 828:
                cut = ai_response[:828].rfind("\n")
                ai_response = ai_response[:cut] if cut > 400 else ai_response[:828]
                logging.info(f"✂️ [Auto-push] Trimmed response to {len(ai_response)} chars")

    return ai_response
