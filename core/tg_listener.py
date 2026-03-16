import asyncio
import aiohttp
import logging
import re
import pandas as pd
import json
from config import BOT_TOKEN, GROUP_CHAT_ID, CHAT_ID

from core.binance_api import fetch_klines, fetch_funding_rate
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
import agent.analyzer  
import agent.square_publisher
from agent.square_publisher import set_coins, set_times, get_coins, get_times, get_status_text
from agent.skills import post_to_binance_square
from core.geometry_scanner import find_trend_line
from core.chart_drawer import draw_scan_chart
SCAN_SCHEDULE = {"hour": 3, "minute": 0}

# --- ADMIN ACCESS CONTROL ---
ADMIN_ID = int(CHAT_ID) if CHAT_ID else 0

def is_admin(msg: dict) -> bool:
    """Check if the message sender is the bot admin."""
    user_id = msg.get("from", {}).get("id", 0)
    return user_id == ADMIN_ID

# --- Import all skills ---
from agent.skills import (
    get_smart_money_signals,
    get_unified_token_rank,
    get_social_hype_leaderboard,
    get_smart_money_inflow_rank,
    get_meme_rank,
    get_address_pnl_rank
)

SQUARE_CACHE = {}

async def send_response(session, chat_id, text, reply_to_msg_id=None, reply_markup=None, parse_mode=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_to_msg_id:
        payload["reply_to_message_id"] = reply_to_msg_id
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if parse_mode:
        payload["parse_mode"] = parse_mode
        
    await session.post(url, json=payload)

async def telegram_polling_loop(app_session):
    """Listens for messages and button presses from the Telegram group/chat"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    offset = 0
    while True:
        try:
            async with app_session.get(f"{url}?offset={offset}&timeout=10", timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1

                        # --- HANDLE BUTTON PRESSES (CALLBACK QUERY) ---
                        if "callback_query" in update:
                            cq = update["callback_query"]
                            cb_data = cq.get("data", "")
                            cq_id = cq.get("id")
                            chat_id = cq.get("message", {}).get("chat", {}).get("id")
                            
                            # 1. Square Integration (Admin check)
                            if cb_data.startswith("sq_"):
                                user_id = cq.get("from", {}).get("id")
                                chat_type = cq.get("message", {}).get("chat", {}).get("type", "")
                                
                                is_admin = True
                                if chat_type in ["group", "supergroup"]:
                                    admin_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getChatMember?chat_id={chat_id}&user_id={user_id}"
                                    async with app_session.get(admin_url) as chk_resp:
                                        if chk_resp.status == 200:
                                            chk_data = await chk_resp.json()
                                            status = chk_data.get("result", {}).get("status", "")
                                            if status not in ["creator", "administrator"]:
                                                is_admin = False
                                
                                if not is_admin:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⛔️ Only Group Admins can post to Square!", "show_alert": True}
                                    )
                                    continue
                                
                                post_id = cb_data.replace("sq_", "")
                                text_to_post = SQUARE_CACHE.get(post_id)
                                if text_to_post:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⏳ Publishing..."}
                                    )
                                    result_msg = await post_to_binance_square(text_to_post)
                                    await send_response(app_session, chat_id, result_msg)
                                    del SQUARE_CACHE[post_id]
                                else:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⚠️ Text is outdated.", "show_alert": True}
                                    )
                                continue

                            # 2. Binance Web3 Skills Buttons
                            if cb_data.startswith("sk_"):
                                await app_session.post(
                                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                    json={"callback_query_id": cq_id, "text": "⏳ Fetching Web3 data..."}
                                )
                                
                                result_text = ""
                                if cb_data == "sk_sm_BTC":
                                    result_text = await get_smart_money_signals("BTC")
                                elif cb_data == "sk_sm_ETH":
                                    result_text = await get_smart_money_signals("ETH")
                                elif cb_data == "sk_hype":
                                    result_text = await get_social_hype_leaderboard()
                                elif cb_data == "sk_inflow":
                                    result_text = await get_smart_money_inflow_rank()
                                elif cb_data == "sk_meme":
                                    result_text = await get_meme_rank()
                                elif cb_data == "sk_rank":
                                    result_text = await get_unified_token_rank(10)
                                elif cb_data == "sk_trader":
                                    result_text = await get_address_pnl_rank()
                                    
                                await send_response(app_session, chat_id, f"🛠 *Binance Web3 Skill:*\n{result_text}", parse_mode="Markdown")
                                continue

                        # --- HANDLE REGULAR MESSAGES ---
                        msg = update.get("message", {})
                        original_text = msg.get("text", "")
                        text = original_text.lower()
                        chat_id = msg.get("chat", {}).get("id")
                        msg_id = msg.get("message_id")

                        if not text:
                            continue
                            
                        # AUTO LANGUAGE DETECTION
                        lang_pref = "ru" if any('\u0400' <= char <= '\u04FF' for char in text) else "en"

                        # ==========================================
                        # BLOCK 1: AI MODEL COMMANDS (/models)
                        # ==========================================
                        if text.startswith("/models"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            models_text = (
                                "🧠 *Available / Popular Models:*\n\n"
                                "1. `stepfun/step-3.5-flash:free` (Fast, Free)\n"
                                "2. `meta-llama/llama-3-8b-instruct:free` (Free)\n"
                                "3. `google/gemini-2.5-flash` (Fast, requires balance)\n"
                                "4. `anthropic/claude-3-haiku` (Smart, requires balance)\n\n"
                                "To change, type:\n`/model model_name`"
                            )
                            await send_response(app_session, chat_id, models_text, msg_id, parse_mode="Markdown")
                            continue
                            
                        if text.startswith("/model"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            parts = original_text.split(maxsplit=1)
                            if len(parts) == 1:
                                current_m = agent.analyzer.OPENROUTER_MODEL
                                await send_response(app_session, chat_id, f"🤖 Current model:\n`{current_m}`", msg_id, parse_mode="Markdown")
                            else:
                                new_model = parts[1].strip()
                                agent.analyzer.OPENROUTER_MODEL = new_model
                                await send_response(app_session, chat_id, f"✅ Model successfully changed to:\n`{new_model}`", msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # BLOCK 2: BINANCE WEB3 SKILLS MENU
                        # ==========================================
                        if text.startswith("/skills") or text in ["skills", "скиллы"]:
                            skills_menu_text = "🛠 *Select Binance Web3 Agent Skill:*"
                            skills_markup = {
                                "inline_keyboard": [
                                    [{"text": "🐋 Smart Money (BTC)", "callback_data": "sk_sm_BTC"}, {"text": "🐋 SM (ETH)", "callback_data": "sk_sm_ETH"}],
                                    [{"text": "🔥 Social Hype", "callback_data": "sk_hype"}, {"text": "💸 Net Inflow", "callback_data": "sk_inflow"}],
                                    [{"text": "🏆 Top Tokens", "callback_data": "sk_rank"}, {"text": "🐶 Meme Rank", "callback_data": "sk_meme"}],
                                    [{"text": "👨‍💻 Top Traders PnL", "callback_data": "sk_trader"}]
                                ]
                            }
                            await send_response(app_session, chat_id, skills_menu_text, reply_to_msg_id=msg_id, reply_markup=skills_markup, parse_mode="Markdown")
                            continue

                        if text.startswith("skill ") or text.startswith("скилл ") or text.startswith("скил "):
                            cmd_body = text.split(" ", 1)[1].strip()
                            result_text = ""
                            if "smart money" in cmd_body or "смарт мани" in cmd_body:
                                parts = cmd_body.split()
                                coin = parts[-1].upper() if parts[-1] not in ["money", "мани"] else "BTC"
                                result_text = await get_smart_money_signals(coin)
                            elif "hype" in cmd_body or "хайп" in cmd_body:
                                result_text = await get_social_hype_leaderboard()
                            elif "inflow" in cmd_body or "приток" in cmd_body:
                                result_text = await get_smart_money_inflow_rank()
                            elif "meme" in cmd_body or "мем" in cmd_body:
                                result_text = await get_meme_rank()
                            elif "rank" in cmd_body or "рейтинг" in cmd_body:
                                result_text = await get_unified_token_rank(10)
                            elif "trader" in cmd_body or "трейдер" in cmd_body:
                                result_text = await get_address_pnl_rank()
                            else:
                                result_text = "⚠️ Unknown skill. Available: `smart money [coin]`, `hype`, `inflow`, `meme`, `rank`, `traders`"
                                
                            await send_response(app_session, chat_id, f"🛠 *Binance Web3 Skill:*\n{result_text}", msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # BLOCK 3: BASIC COMMANDS (/start, /time, /autopost)
                        # ==========================================
                        if text.startswith("/start") or text.startswith("/help") or text in ["привет", "hello"]:
                            welcome_text = (
                                "🤖 *Hello! I am AiAlisa Copilot.*\n\n"
                                "*Commands:*\n"
                                "🔍 `scan BTC` or `посмотри btc` - Run Tech Analysis\n"
                                "💰 `margin 100 leverage 10 max 20%` - Reply for exact stop-loss math\n"
                                "🛠 `/skills` - Open Web3 Skills Menu\n"
                                "📈 `/top gainers` - Top 10 growth 24h\n"
                                "📉 `/top losers` - Top 10 drops 24h"
                            )
                            if is_admin(msg):
                                welcome_text += (
                                    "\n\n🔐 *Admin Commands:*\n"
                                    "🧠 `/models` - Change AI Engine\n"
                                    "⏰ `/time 18:30` - Set global scan schedule\n"
                                    "📢 `/autopost` - Status / manage auto-posts\n"
                                    "📢 `/autopost on / off` - Toggle auto-posts\n"
                                    "🪙 `/autopost SOL BTC ETH` - Set coins\n"
                                    "⏰ `/autopost time 13:30 22:50` - Set schedule\n"
                                    "✏️ `/post текст` - Post to Binance Square"
                                )
                            await send_response(app_session, chat_id, welcome_text, msg_id, parse_mode="Markdown")
                            continue

                        if text.startswith("/time "):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            try:
                                # Parse format like "/time 18:15" or "/time 18 15"
                                time_str = text.split(" ", 1)[1].replace(":", " ").split()
                                new_h = int(time_str[0])
                                new_m = int(time_str[1])

                                if 0 <= new_h < 24 and 0 <= new_m < 60:
                                    SCAN_SCHEDULE["hour"] = new_h
                                    SCAN_SCHEDULE["minute"] = new_m
                                    msg_text = f"✅ Global scan time successfully changed to *{new_h:02d}:{new_m:02d}* (UTC+3)"
                                else:
                                    msg_text = "⚠️ Invalid time format. Use: `/time 18:15` or `/time 18 15`"
                            except Exception:
                                msg_text = "⚠️ Error parsing time. Example: `/time 18 15`"

                            await send_response(app_session, chat_id, msg_text, msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # CUSTOM POST TO BINANCE SQUARE (/post <text>)
                        # ==========================================
                        if text.startswith("/post"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            parts = original_text.split(maxsplit=1)
                            if len(parts) < 2 or not parts[1].strip():
                                await send_response(app_session, chat_id,
                                    "✏️ *Как использовать:*\n"
                                    "`/post Ваш текст для Binance Square`\n\n"
                                    "Пример:\n"
                                    "`/post Привет Бинанс! Сегодня BTC выглядит бычьим 🚀`",
                                    msg_id, parse_mode="Markdown")
                            else:
                                user_text = parts[1].strip()
                                await send_response(app_session, chat_id, "⏳ Публикую в Binance Square...", msg_id)
                                result = await post_to_binance_square(user_text)
                                await send_response(app_session, chat_id, result, msg_id)
                            continue

                        if text.startswith("/autopost"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            parts = original_text.split(maxsplit=1)
                            arg = parts[1].strip() if len(parts) > 1 else ""
                            arg_lower = arg.lower()

                            if arg_lower == "on":
                                agent.square_publisher.AUTO_SQUARE_ENABLED = True
                                msg_text = "✅ Auto-posting is now **ENABLED**."
                            elif arg_lower == "off":
                                agent.square_publisher.AUTO_SQUARE_ENABLED = False
                                msg_text = "⏸ Auto-posting is now **DISABLED**."

                            elif arg_lower.startswith("time"):
                                # /autopost time 13:30 22:50
                                time_parts = arg.split()[1:]  # skip "time"
                                if not time_parts:
                                    times = get_times()
                                    times_str = ", ".join(f"{t['hour']:02d}:{t['minute']:02d}" for t in times)
                                    msg_text = f"⏰ Current schedule: `{times_str}` UTC\n\nTo change: `/autopost time 13:30 22:50`"
                                else:
                                    new_times = []
                                    parse_ok = True
                                    for tp in time_parts:
                                        try:
                                            h, m = tp.replace(".", ":").split(":")
                                            h, m = int(h), int(m)
                                            if 0 <= h < 24 and 0 <= m < 60:
                                                new_times.append({"hour": h, "minute": m})
                                            else:
                                                parse_ok = False
                                        except Exception:
                                            parse_ok = False

                                    if parse_ok and new_times:
                                        set_times(new_times)
                                        times_str = ", ".join(f"{t['hour']:02d}:{t['minute']:02d} UTC" for t in new_times)
                                        msg_text = f"✅ Schedule updated!\n⏰ New times: `{times_str}`"
                                    else:
                                        msg_text = "⚠️ Wrong format. Example:\n`/autopost time 09:00 21:30`"

                            elif arg_lower in ("", "status"):
                                # No args — show full status
                                msg_text = get_status_text()

                            else:
                                # Treat everything else as coin list: /autopost SOL RIVER FHE
                                coin_args = arg.split()
                                if len(coin_args) >= 1:
                                    set_coins(coin_args)
                                    coins_str = ", ".join(get_coins())
                                    msg_text = f"✅ Coins updated!\n🪙 Auto-post list: `{coins_str}`"
                                else:
                                    msg_text = get_status_text()

                            await send_response(app_session, chat_id, msg_text, msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # TOP GAINERS / LOSERS (/top gainers, /top losers)
                        # ==========================================
                        if text.startswith("/top"):
                            top_parts = text.split()
                            mode = top_parts[1] if len(top_parts) > 1 else ""

                            if mode in ("gainers", "gainer", "рост", "gainers24"):
                                await send_response(app_session, chat_id, "⏳ Loading top gainers...", msg_id)
                                try:
                                    async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as resp:
                                        if resp.status == 200:
                                            tickers = await resp.json()
                                            usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]
                                            sorted_t = sorted(usdt_tickers, key=lambda x: float(x["priceChangePercent"]), reverse=True)[:10]
                                            lines = ["🟢 *Top 10 Gainers (24h Futures):*\n"]
                                            for i, t in enumerate(sorted_t, 1):
                                                sym = t["symbol"].replace("USDT", "")
                                                pct = float(t["priceChangePercent"])
                                                price = float(t["lastPrice"])
                                                vol = float(t["quoteVolume"])
                                                vol_m = vol / 1_000_000
                                                lines.append(f"{i}. `{sym}` → *+{pct:.2f}%*  |  ${price:,.4f}  |  Vol: ${vol_m:.1f}M")
                                            await send_response(app_session, chat_id, "\n".join(lines), msg_id, parse_mode="Markdown")
                                        else:
                                            await send_response(app_session, chat_id, "❌ Binance API error", msg_id)
                                except Exception as e:
                                    await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)

                            elif mode in ("losers", "loser", "падение", "losers24"):
                                await send_response(app_session, chat_id, "⏳ Loading top losers...", msg_id)
                                try:
                                    async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as resp:
                                        if resp.status == 200:
                                            tickers = await resp.json()
                                            usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]
                                            sorted_t = sorted(usdt_tickers, key=lambda x: float(x["priceChangePercent"]))[:10]
                                            lines = ["🔴 *Top 10 Losers (24h Futures):*\n"]
                                            for i, t in enumerate(sorted_t, 1):
                                                sym = t["symbol"].replace("USDT", "")
                                                pct = float(t["priceChangePercent"])
                                                price = float(t["lastPrice"])
                                                vol = float(t["quoteVolume"])
                                                vol_m = vol / 1_000_000
                                                lines.append(f"{i}. `{sym}` → *{pct:.2f}%*  |  ${price:,.4f}  |  Vol: ${vol_m:.1f}M")
                                            await send_response(app_session, chat_id, "\n".join(lines), msg_id, parse_mode="Markdown")
                                        else:
                                            await send_response(app_session, chat_id, "❌ Binance API error", msg_id)
                                except Exception as e:
                                    await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
                            else:
                                await send_response(app_session, chat_id,
                                    "📊 *Usage:*\n"
                                    "`/top gainers` — Top 10 growth (24h)\n"
                                    "`/top losers` — Top 10 drops (24h)",
                                    msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # BLOCK 4: CHART ANALYSIS (RU/EN TRIGGERS)
                        # ==========================================
                        analysis_prefixes = [
                            "scan ", "check ", "look ", "analyze ",
                            "посмотри ", "посмотри на ", "глянь ", "чекни ", "анализ "
                        ]
                        matched_prefix = next((p for p in analysis_prefixes if text.startswith(p)), None)

                        if matched_prefix:
                            symbol_raw = text.replace(matched_prefix, "").strip().split()[0].upper()
                            symbol = symbol_raw + "USDT" if not symbol_raw.endswith("USDT") else symbol_raw

                            await send_response(app_session, chat_id, f"⏳ Fetching chart data + building trend line... ({symbol})", msg_id)

                            # Fetch 199 candles for trend line construction (same as main scanner)
                            raw_df_full = await fetch_klines(app_session, symbol, "4h", 199)
                            # Also fetch 100 for indicators (lighter)
                            raw_df = await fetch_klines(app_session, symbol, "4h", 100)

                            if raw_df:
                                df = pd.DataFrame(raw_df)
                                last_row, full_df = calculate_binance_indicators(df, "4H")
                                funding = await fetch_funding_rate(app_session, symbol)
                                last_row["funding_rate"] = funding
                                ai_msg = await ask_ai_analysis(symbol, "4H", last_row, lang=lang_pref)

                                # --- BUILD TREND LINE & CHART ---
                                chart_path = None
                                if raw_df_full:
                                    df_full = pd.DataFrame(raw_df_full)
                                    line_data, _ = await find_trend_line(df_full, "4H", symbol)
                                    if line_data:
                                        chart_path = await draw_scan_chart(symbol, df_full, line_data, "4H")

                                # --- PREPARE BINANCE SQUARE PUBLICATION & BUTTONS ---
                                import uuid
                                post_id = str(uuid.uuid4())[:8]
                                square_text = f"🚀 ${symbol} AI Market Analysis!\n\n{ai_msg}\n\n#AIBinance #BinanceSquare #Write2Earn"
                                SQUARE_CACHE[post_id] = square_text

                                app_link = f"https://app.binance.com/en/futures/{symbol.upper()}"
                                web_link = f"https://www.binance.com/en/futures/{symbol.upper()}"

                                scan_markup = {
                                    "inline_keyboard": [
                                        [{"text": "📱 Open BINANCE App", "url": app_link}],
                                        [{"text": f"🖥 Open {symbol} Chart on Web", "url": web_link}],
                                        [{"text": "📢 Post to Binance Square", "callback_data": f"sq_{post_id}"}]
                                    ]
                                }

                                # --- SEND: chart + AI text, or just text if no line found ---
                                if chart_path:
                                    import os as _os
                                    safe_ai = ai_msg if len(ai_msg) < 800 else ai_msg[:800] + "...\n*[текст обрезан]*"
                                    caption = f"📊 *{symbol} — 4H Trend Analysis*\n\n{safe_ai}"
                                    photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                                    try:
                                        with open(chart_path, 'rb') as f:
                                            data = aiohttp.FormData()
                                            data.add_field('chat_id', str(chat_id))
                                            data.add_field('caption', caption)
                                            data.add_field('parse_mode', 'Markdown')
                                            data.add_field('reply_to_message_id', str(msg_id))
                                            data.add_field('reply_markup', json.dumps(scan_markup))
                                            data.add_field('photo', f, filename=f"{symbol}.png", content_type='image/png')
                                            async with app_session.post(photo_url, data=data, timeout=30) as resp:
                                                if resp.status != 200:
                                                    resp_text = await resp.text()
                                                    logging.error(f"❌ Scan photo send error: {resp.status} - {resp_text}")
                                                    # Fallback to text-only
                                                    await send_response(app_session, chat_id, ai_msg, msg_id, reply_markup=scan_markup)
                                    except Exception as e:
                                        logging.error(f"❌ Error sending scan chart: {e}")
                                        await send_response(app_session, chat_id, ai_msg, msg_id, reply_markup=scan_markup)
                                    finally:
                                        try:
                                            _os.remove(chart_path)
                                        except: pass
                                else:
                                    # No trend line found — send text only
                                    await send_response(app_session, chat_id, ai_msg, msg_id, reply_markup=scan_markup)

                            continue

                        # ==========================================
                        # BLOCK 5: RISK & OUT-OF-BOUNDS (THROUGH REPLIES)
                        # ==========================================
                        is_margin_en = "margin" in text and "leverage" in text
                        is_margin_ru = ("маржа" in text or "маржу" in text) and "плечо" in text
                        
                        if is_margin_en or is_margin_ru:
                            nums = re.findall(r'\d+', text)
                            if len(nums) >= 2:
                                margin = float(nums[0])
                                leverage = float(nums[1])
                                # Extract 3rd number if user asks for specific limit (e.g. max 20%)
                                max_loss = float(nums[2]) if len(nums) >= 3 else None
                                margin_data = {"margin": margin, "leverage": leverage, "max_loss": max_loss}
                                    
                                coin_to_analyze = "BTCUSDT"
                                reply_msg = msg.get("reply_to_message", {})
                                reply_text = reply_msg.get("caption", reply_msg.get("text", ""))
                                
                                match = re.search(r'[\$#]([A-Za-z0-9]+)', reply_text)
                                if match:
                                    coin_to_analyze = match.group(1).upper()
                                    if not coin_to_analyze.endswith("USDT"):
                                        coin_to_analyze += "USDT"

                                raw_df = await fetch_klines(app_session, coin_to_analyze, "4h", 100)
                                if raw_df:
                                    last_row, _ = calculate_binance_indicators(pd.DataFrame(raw_df), "4H")
                                    funding = await fetch_funding_rate(app_session, coin_to_analyze)
                                    last_row["funding_rate"] = funding
                                    ai_msg = await ask_ai_analysis(coin_to_analyze, "4H", last_row, user_margin=margin_data, lang=lang_pref)
                                    await send_response(app_session, chat_id, ai_msg, msg_id)

        except Exception as e:
            logging.error(f"TG Polling Error: {e}")
            await asyncio.sleep(2)
