import asyncio
import aiohttp
import logging
import re
import os
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
from config import BOT_TOKEN, GROUP_CHAT_ID, CHAT_ID, load_breakout_log, load_price_alerts, save_price_alerts

from core.binance_api import fetch_klines, fetch_funding_rate
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
import agent.analyzer  
import agent.square_publisher
from agent.square_publisher import set_coins, set_times, get_coins, get_times, get_status_text
from agent.skills import post_to_binance_square
# --- SQUARE CACHE (file-based, no shared dict issues) ---
SQUARE_CACHE_FILE = "data/square_cache.json"

def square_cache_put(post_id: str, text: str):
    """Save text to square cache file."""
    try:
        cache = {}
        if os.path.exists(SQUARE_CACHE_FILE):
            with open(SQUARE_CACHE_FILE, 'r') as f:
                cache = json.load(f)
        cache[post_id] = text
        # Keep only last 50 entries to prevent file bloat
        if len(cache) > 50:
            keys = list(cache.keys())
            for k in keys[:-50]:
                del cache[k]
        with open(SQUARE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"❌ square_cache_put error: {e}")

def square_cache_get(post_id: str) -> str | None:
    """Read text from square cache file."""
    try:
        if os.path.exists(SQUARE_CACHE_FILE):
            with open(SQUARE_CACHE_FILE, 'r') as f:
                cache = json.load(f)
                return cache.get(post_id)
    except Exception as e:
        logging.error(f"❌ square_cache_get error: {e}")
    return None

def square_cache_delete(post_id: str):
    """Remove entry from square cache file."""
    try:
        if os.path.exists(SQUARE_CACHE_FILE):
            with open(SQUARE_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            cache.pop(post_id, None)
            with open(SQUARE_CACHE_FILE, 'w') as f:
                json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"❌ square_cache_delete error: {e}")

from core.geometry_scanner import find_trend_line
from core.chart_drawer import draw_scan_chart, draw_simple_chart
SCAN_SCHEDULE_FILE = "data/scan_schedule.json"

def _load_scan_schedule():
    try:
        if os.path.exists(SCAN_SCHEDULE_FILE):
            with open(SCAN_SCHEDULE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {"hour": 3, "minute": 0}

def _save_scan_schedule():
    try:
        with open(SCAN_SCHEDULE_FILE, 'w') as f:
            json.dump(SCAN_SCHEDULE, f)
    except Exception:
        pass

SCAN_SCHEDULE = _load_scan_schedule()

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


async def build_trend_text(session: aiohttp.ClientSession) -> str:
    """Build a formatted list of all breakout coins with breakout price and current live price."""
    log = load_breakout_log()
    if not log:
        return "📭 Нет пробитий с последнего скана."

    lines = ["📊 *Пробития трендовых линий:*\n"]
    for entry in log:
        sym = entry["symbol"].replace("USDT", "")
        tf = entry["tf"]
        bp = entry["breakout_price"]
        # Fetch live price
        current_price = entry["current_price"]
        try:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={entry['symbol']}"
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    current_price = float(data["price"])
        except Exception:
            pass

        diff_pct = ((current_price / bp) - 1) * 100 if bp > 0 else 0
        arrow = "🟢" if diff_pct >= 0 else "🔴"
        lines.append(
            f"{arrow} `${sym}` ({tf})\n"
            f"    Пробитие: `${bp:.6f}`\n"
            f"    Сейчас: `${current_price:.6f}` (*{diff_pct:+.2f}%*)"
        )
    
    lines.append(f"\n_Всего: {len(log)} монет_")
    return "\n".join(lines)


async def auto_trend_sender(session: aiohttp.ClientSession):
    """Background task: sends /trend summary to group at 23:57 UTC daily."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            # Target: 23:57 UTC (= 02:57 UTC+3)
            target = now.replace(hour=23, minute=57, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            
            sleep_sec = (target - now).total_seconds()
            logging.info(f"📊 Trend auto-sender sleeping {sleep_sec:.0f}s until {target.strftime('%Y-%m-%d %H:%M')} UTC")
            await asyncio.sleep(sleep_sec)

            # Build and send trend summary
            trend_text = await build_trend_text(session)
            if "Нет пробитий" not in trend_text:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id": GROUP_CHAT_ID,
                    "text": f"🕐 *Ежедневный итог пробитий (23:57 UTC):*\n\n{trend_text}",
                    "parse_mode": "Markdown"
                }
                await session.post(url, json=payload)
                logging.info("✅ Auto trend summary sent to group.")
            else:
                logging.info("📭 No breakouts to report, skipping auto-trend.")
            
            await asyncio.sleep(120)  # Prevent double-trigger
        except Exception as e:
            logging.error(f"❌ Auto trend sender error: {e}")
            await asyncio.sleep(60)


async def price_alert_monitor(session: aiohttp.ClientSession):
    """Background task: checks price alerts every 30 seconds and notifies users."""
    logging.info("🔔 Price alert monitor started.")
    while True:
        try:
            alerts = load_price_alerts()
            if not alerts:
                await asyncio.sleep(30)
                continue

            # Get all unique symbols
            symbols = list(set(a["symbol"] for a in alerts))
            prices = {}

            # Fetch current prices (batch via futures ticker)
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                    if resp.status == 200:
                        tickers = await resp.json()
                        for t in tickers:
                            prices[t["symbol"]] = float(t["price"])
            except Exception:
                await asyncio.sleep(30)
                continue

            triggered = []
            remaining = []

            for alert in alerts:
                sym = alert["symbol"]
                target = alert["target_price"]
                direction = alert["direction"]  # "above" or "below"
                current = prices.get(sym)

                if current is None:
                    remaining.append(alert)
                    continue

                hit = False
                if direction == "above" and current >= target:
                    hit = True
                elif direction == "below" and current <= target:
                    hit = True

                if hit:
                    triggered.append((alert, current))
                else:
                    remaining.append(alert)

            # Send notifications for triggered alerts
            for alert, current in triggered:
                short_sym = alert["symbol"].replace("USDT", "")
                arrow = "🟢📈" if alert["direction"] == "above" else "🔴📉"
                notify_text = (
                    f"{arrow} *PRICE ALERT!*\n\n"
                    f"💰 `${short_sym}` reached `${current:.6f}`\n"
                    f"🎯 Your target: `${alert['target_price']:.6f}`"
                )
                try:
                    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": alert["chat_id"],
                        "text": notify_text,
                        "parse_mode": "Markdown"
                    }
                    await session.post(url, json=payload)
                except Exception as e:
                    logging.error(f"❌ Alert notification error: {e}")

            # Save remaining alerts
            if len(triggered) > 0:
                save_price_alerts(remaining)
                logging.info(f"🔔 {len(triggered)} price alert(s) triggered, {len(remaining)} remaining.")

            await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"❌ Price alert monitor error: {e}")
            await asyncio.sleep(30)


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


async def send_and_get_msg_id(session, chat_id, text, reply_to_msg_id=None):
    """Send a Telegram message and return its message_id (for streaming edits)."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_to_msg_id:
        payload["reply_to_message_id"] = reply_to_msg_id
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", {}).get("message_id")
    except Exception as e:
        logging.error(f"❌ send_and_get_msg_id error: {e}")
    return None

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
                                
                                user_is_admin = True
                                if chat_type in ["group", "supergroup"]:
                                    admin_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getChatMember?chat_id={chat_id}&user_id={user_id}"
                                    async with app_session.get(admin_url) as chk_resp:
                                        if chk_resp.status == 200:
                                            chk_data = await chk_resp.json()
                                            status = chk_data.get("result", {}).get("status", "")
                                            if status not in ["creator", "administrator"]:
                                                user_is_admin = False
                                
                                if not user_is_admin:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⛔️ Only Group Admins can post to Square!", "show_alert": True}
                                    )
                                    continue
                                
                                post_id = cb_data.replace("sq_", "")
                                text_to_post = square_cache_get(post_id)
                                if text_to_post:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⏳ Publishing..."}
                                    )
                                    result_msg = await post_to_binance_square(text_to_post)
                                    await send_response(app_session, chat_id, result_msg)
                                    square_cache_delete(post_id)
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

                            # 3. Model Selection Buttons (Admin only)
                            if cb_data.startswith("md_"):
                                if cb_data == "md_noop":
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id}
                                    )
                                    continue
                                user_id = cq.get("from", {}).get("id", 0)
                                if user_id != ADMIN_ID:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⛔️ Admin only.", "show_alert": True}
                                    )
                                    continue
                                new_model = cb_data[3:]  # strip "md_" prefix
                                agent.analyzer.OPENROUTER_MODEL = new_model
                                short_name = new_model.split("/")[-1] if "/" in new_model else new_model
                                await app_session.post(
                                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                    json={"callback_query_id": cq_id, "text": f"✅ Switched to {short_name}"}
                                )
                                await send_response(app_session, chat_id, f"✅ AI Engine changed to:\n`{new_model}`", parse_mode="Markdown")
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
                        if text.startswith("/models") or text.startswith("/model"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue
                            current_m = agent.analyzer.OPENROUTER_MODEL
                            model_text = f"🧠 *AI Engine Selection*\nCurrent: `{current_m}`"
                            model_markup = {
                                "inline_keyboard": [
                                    [{"text": "── FREE MODELS ──", "callback_data": "md_noop"}],
                                    [{"text": "⚡ StepFun 3.5 Flash", "callback_data": "md_stepfun/step-3.5-flash:free"},
                                     {"text": "🦙 Llama 3 8B", "callback_data": "md_meta-llama/llama-3-8b-instruct:free"}],
                                    [{"text": "🦙 Llama 4 Mav.", "callback_data": "md_meta-llama/llama-4-maverick:free"},
                                     {"text": "🔮 Mistral Small", "callback_data": "md_mistralai/mistral-small-3.1-24b-instruct:free"}],
                                    [{"text": "── GPT MODELS ──", "callback_data": "md_noop"}],
                                    [{"text": "🧠 GPT-4o", "callback_data": "md_openai/gpt-4o"},
                                     {"text": "⚡ GPT-4o Mini", "callback_data": "md_openai/gpt-4o-mini"}],
                                    [{"text": "🧠 GPT-4.1", "callback_data": "md_openai/gpt-4.1"},
                                     {"text": "⚡ GPT-4.1 Mini", "callback_data": "md_openai/gpt-4.1-mini"}],
                                    [{"text": "💎 o4-mini", "callback_data": "md_openai/o4-mini"}],
                                    [{"text": "── GEMINI MODELS ──", "callback_data": "md_noop"}],
                                    [{"text": "💎 Gemini 2.5 Pro", "callback_data": "md_google/gemini-2.5-pro-preview-06-05"},
                                     {"text": "⚡ Gemini 2.5 Flash", "callback_data": "md_google/gemini-2.5-flash-preview-05-20"}],
                                    [{"text": "⚡ Gemini 2.0 Flash", "callback_data": "md_google/gemini-2.0-flash-001"}],
                                ]
                            }
                            await send_response(app_session, chat_id, model_text, msg_id, reply_markup=model_markup, parse_mode="Markdown")
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
                                "📈 `/top gainers` - Top 10 Futures growth 24h\n"
                                "📉 `/top losers` - Top 10 Futures drops 24h\n"
                                "📊 `/trend` - All breakout coins since last scan\n"
                                "🔔 `/alert BTC 69500` - Price alert notification"
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
                                    _save_scan_schedule()
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
                                await send_response(app_session, chat_id, "⏳ Loading top gainers (Futures)...", msg_id)
                                try:
                                    async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as resp:
                                        if resp.status == 200:
                                            tickers = await resp.json()
                                            usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 5_000_000]
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
                                await send_response(app_session, chat_id, "⏳ Loading top losers (Futures)...", msg_id)
                                try:
                                    async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as resp:
                                        if resp.status == 200:
                                            tickers = await resp.json()
                                            usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 5_000_000]
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
                        # PRICE ALERTS (/alert) — PUBLIC
                        # ==========================================
                        if text.startswith("/alert"):
                            parts = original_text.split()
                            # /alert list — show active alerts
                            if len(parts) == 2 and parts[1].lower() in ("list", "список"):
                                alerts = load_price_alerts()
                                user_alerts = [a for a in alerts if a["chat_id"] == chat_id]
                                if not user_alerts:
                                    await send_response(app_session, chat_id, "📭 У вас нет активных алертов.\n\nИспользуйте:\n`/alert BTC 69500`", msg_id, parse_mode="Markdown")
                                else:
                                    lines = ["🔔 *Ваши алерты:*\n"]
                                    for i, a in enumerate(user_alerts, 1):
                                        short = a["symbol"].replace("USDT", "")
                                        arrow = "↗️" if a["direction"] == "above" else "↘️"
                                        lines.append(f"{i}. {arrow} `${short}` → `${a['target_price']:.6f}`")
                                    await send_response(app_session, chat_id, "\n".join(lines), msg_id, parse_mode="Markdown")
                                continue

                            # /alert clear — remove all user alerts
                            if len(parts) == 2 and parts[1].lower() in ("clear", "очистить"):
                                alerts = load_price_alerts()
                                remaining = [a for a in alerts if a["chat_id"] != chat_id]
                                save_price_alerts(remaining)
                                await send_response(app_session, chat_id, "✅ Все ваши алерты удалены.", msg_id)
                                continue

                            # /alert BTC 69500 — set new alert
                            if len(parts) >= 3:
                                coin_raw = parts[1].upper().strip()
                                symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
                                try:
                                    target_price = float(parts[2].replace(",", "."))
                                except ValueError:
                                    await send_response(app_session, chat_id, "⚠️ Неверная цена. Пример: `/alert BTC 69500`", msg_id, parse_mode="Markdown")
                                    continue

                                # Get current price to determine direction
                                current_price = 0
                                try:
                                    async with app_session.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5) as resp:
                                        if resp.status == 200:
                                            data = await resp.json()
                                            current_price = float(data["price"])
                                except Exception:
                                    pass

                                if current_price == 0:
                                    await send_response(app_session, chat_id, f"⚠️ Не найдена пара `{symbol}` на Binance Futures.", msg_id, parse_mode="Markdown")
                                    continue

                                direction = "above" if target_price > current_price else "below"
                                arrow = "📈" if direction == "above" else "📉"

                                alerts = load_price_alerts()
                                alerts.append({
                                    "symbol": symbol,
                                    "target_price": target_price,
                                    "direction": direction,
                                    "chat_id": chat_id,
                                    "user_id": msg.get("from", {}).get("id", 0),
                                    "set_price": current_price,
                                    "time": datetime.now(timezone.utc).isoformat()
                                })
                                save_price_alerts(alerts)

                                short = symbol.replace("USDT", "")
                                await send_response(app_session, chat_id,
                                    f"✅ Алерт установлен!\n\n"
                                    f"🪙 `${short}`\n"
                                    f"💰 Сейчас: `${current_price:.6f}`\n"
                                    f"{arrow} Цель: `${target_price:.6f}`\n"
                                    f"📩 Уведомлю когда цена {'поднимется' if direction == 'above' else 'опустится'} до цели.",
                                    msg_id, parse_mode="Markdown")
                                continue

                            # No args — show help
                            await send_response(app_session, chat_id,
                                "🔔 *Price Alert:*\n\n"
                                "Установить: `/alert BTC 69500`\n"
                                "Список: `/alert list`\n"
                                "Удалить все: `/alert clear`",
                                msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # TREND BREAKOUT LIST (/trend) — PUBLIC
                        # ==========================================
                        if text.startswith("/trend") or text in ["тренд", "тренды", "пробития"]:
                            await send_response(app_session, chat_id, "⏳ Загружаю пробития...", msg_id)
                            trend_text = await build_trend_text(app_session)
                            await send_response(app_session, chat_id, trend_text, msg_id, parse_mode="Markdown")
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

                            # Send status message and capture its ID for live streaming
                            stream_msg_id = await send_and_get_msg_id(
                                app_session, chat_id,
                                f"⏳ Fetching chart data + building trend line... ({symbol})", msg_id
                            )

                            # Fetch 199 candles for trend line construction (same as main scanner)
                            raw_df_full = await fetch_klines(app_session, symbol, "4h", 199)
                            # Also fetch 100 for indicators (lighter)
                            raw_df = await fetch_klines(app_session, symbol, "4h", 100)

                            if raw_df:
                                df = pd.DataFrame(raw_df)
                                last_row, full_df = calculate_binance_indicators(df, "4H")
                                funding = await fetch_funding_rate(app_session, symbol)
                                last_row["funding_rate"] = funding

                                # Build telegram_stream dict for live AI streaming
                                tg_stream = None
                                if stream_msg_id:
                                    tg_stream = {
                                        "session": app_session,
                                        "chat_id": chat_id,
                                        "message_id": stream_msg_id,
                                        "bot_token": BOT_TOKEN
                                    }

                                ai_msg = await ask_ai_analysis(symbol, "4H", last_row, lang=lang_pref, telegram_stream=tg_stream)

                                # Delete streaming message — final result goes in chart caption
                                if stream_msg_id:
                                    try:
                                        del_url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
                                        await app_session.post(del_url, json={
                                            "chat_id": chat_id, "message_id": stream_msg_id
                                        }, timeout=5)
                                    except Exception:
                                        pass  # Silently ignore if already deleted

                                # --- BUILD TREND LINE & CHART ---
                                chart_path = None
                                if raw_df_full:
                                    df_full = pd.DataFrame(raw_df_full)
                                    line_data, _ = await find_trend_line(df_full, "4H", symbol)
                                    if line_data:
                                        chart_path = await draw_scan_chart(symbol, df_full, line_data, "4H")
                                    else:
                                        # No trend line — draw simple chart without line
                                        chart_path = await draw_simple_chart(symbol, df_full, "4H")

                                # --- PREPARE BINANCE SQUARE PUBLICATION & BUTTONS ---
                                import uuid
                                post_id = str(uuid.uuid4())[:8]
                                square_text = f"🚀 ${symbol} AI Market Analysis!\n\n{ai_msg}\n\n#AIBinance #BinanceSquare #Write2Earn"
                                square_cache_put(post_id, square_text)

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
