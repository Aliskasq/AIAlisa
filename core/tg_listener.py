import asyncio
import aiohttp
import logging
import re
import os
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
from config import (BOT_TOKEN, GROUP_CHAT_ID, CHAT_ID, load_breakout_log, load_price_alerts,
                     save_price_alerts, load_virtual_bank, save_virtual_bank, update_bank_with_trades,
                     VIRTUAL_BANK_POSITION_SIZE)

# --- PAPER TRADING PORTFOLIO (per-user, persistent) ---
PAPER_FILE = "data/paper_portfolio.json"

def _load_paper():
    try:
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_paper(data):
    try:
        os.makedirs("data", exist_ok=True)
        with open(PAPER_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# --- LANGUAGE SETTINGS (per-chat, persistent) ---
LANG_FILE = "data/lang_settings.json"

def _load_langs():
    try:
        if os.path.exists(LANG_FILE):
            with open(LANG_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_langs(langs):
    try:
        os.makedirs("data", exist_ok=True)
        with open(LANG_FILE, "w") as f:
            json.dump(langs, f)
    except Exception:
        pass

def get_chat_lang(chat_id):
    return _load_langs().get(str(chat_id), "en")

def set_chat_lang(chat_id, lang):
    langs = _load_langs()
    langs[str(chat_id)] = lang
    _save_langs(langs)

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


async def build_signals_text(session: aiohttp.ClientSession, lang: str = "ru") -> list:
    """Build virtual bank + all trades text, returns list of message chunks."""
    log = load_breakout_log()
    bank = load_virtual_bank()

    if not log:
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
        pnl_dollar = bank["balance"] - bank["starting_balance"]
        pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
        if lang == "ru":
            return [f"🏦 *Виртуальный банк*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Плюс: {total_w} ({wr:.0f}%) | ❌ Минус: {total_l} ({100-wr:.0f}%)\n📈 Всего: {bank['total_trades']} сделок\n\n📭 Сегодня пока нет сигналов."]
        else:
            return [f"🏦 *Virtual Bank*\n\n💰 Start: `${bank['starting_balance']:,.2f}`\n💵 Current: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Wins: {total_w} ({wr:.0f}%) | ❌ Losses: {total_l} ({100-wr:.0f}%)\n📈 Total: {bank['total_trades']} trades\n\n📭 No signals today yet."]

    # Batch prices
    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                all_prices = await resp.json()
                for p in all_prices:
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass

    day_wins = 0
    day_losses = 0
    day_pending = 0
    day_pnl_dollar = 0.0
    trade_lines = []

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        ai_dir = entry.get("ai_direction", "")
        now_price = price_map.get(sym, entry.get("current_price", 0))

        # Use AI entry/SL/TP if available, otherwise fallback to breakout_price
        ai_entry = entry.get("ai_entry")
        ai_sl = entry.get("ai_sl")
        ai_tp = entry.get("ai_tp")
        entry_price = ai_entry if ai_entry else entry.get("breakout_price", 0)

        # Determine trade status: TP hit, SL hit, or still open
        status = "open"  # pending
        if ai_dir and entry_price > 0:
            if ai_dir == "LONG":
                if ai_tp and now_price >= ai_tp:
                    status = "tp"
                elif ai_sl and now_price <= ai_sl:
                    status = "sl"
            elif ai_dir == "SHORT":
                if ai_tp and now_price <= ai_tp:
                    status = "tp"
                elif ai_sl and now_price >= ai_sl:
                    status = "sl"

        # Calculate P&L based on direction and entry
        if entry_price > 0:
            if ai_dir == "SHORT":
                # SHORT: profit when price goes down
                if status == "tp" and ai_tp:
                    pnl_pct = ((entry_price - ai_tp) / entry_price) * 100
                elif status == "sl" and ai_sl:
                    pnl_pct = ((entry_price - ai_sl) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - now_price) / entry_price) * 100
            else:
                # LONG: profit when price goes up
                if status == "tp" and ai_tp:
                    pnl_pct = ((ai_tp - entry_price) / entry_price) * 100
                elif status == "sl" and ai_sl:
                    pnl_pct = ((ai_sl - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((now_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0

        pnl_dollar = (pnl_pct / 100) * VIRTUAL_BANK_POSITION_SIZE

        if status == "tp":
            day_wins += 1
            icon = "🟢"
            status_tag = " ✅TP"
        elif status == "sl":
            day_losses += 1
            icon = "🔴"
            status_tag = " 🚫SL"
        else:
            day_pending += 1
            icon = "🟡" if pnl_pct >= 0 else "🟠"
            status_tag = " ⏳"

        day_pnl_dollar += pnl_dollar

        short_sym = sym.replace("USDT", "")
        dir_tag = f" {ai_dir}" if ai_dir else ""
        trade_lines.append(
            f"{icon} `{short_sym}` {tf}{dir_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){status_tag}"
        )

    total_w = bank["total_wins"] + day_wins
    total_l = bank["total_losses"] + day_losses
    total_t = bank["total_trades"] + day_wins + day_losses
    wr_all = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
    projected_balance = bank["balance"] + day_pnl_dollar
    total_pnl_dollar = projected_balance - bank["starting_balance"]
    total_pnl_pct = (total_pnl_dollar / bank["starting_balance"]) * 100

    day_closed = day_wins + day_losses
    day_total = day_closed + day_pending
    day_wr = (day_wins / day_closed * 100) if day_closed > 0 else 0

    if lang == "ru":
        pending_text = f" | ⏳ Открытых: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Виртуальный банк*\n"
            f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
            f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Сегодня ({day_total} сигналов):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}\n"
            f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *Всего за всё время:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n"
            f"{'─' * 30}\n"
        )
    else:
        pending_text = f" | ⏳ Open: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Virtual Bank*\n"
            f"💰 Start: `${bank['starting_balance']:,.2f}` | Current: `${projected_balance:,.2f}`\n"
            f"📊 Total P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Today ({day_total} signals):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}\n"
            f"💵 Day P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *All-time:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} trades\n"
            f"{'─' * 30}\n"
        )

    all_msgs = []
    current_chunk = header
    for line in trade_lines:
        if len(current_chunk) + len(line) + 2 > 3900:
            all_msgs.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk:
        all_msgs.append(current_chunk)

    return all_msgs


async def auto_trend_sender(session: aiohttp.ClientSession):
    """Background task: at 23:57 UTC daily — send daily summary to TG group, update virtual bank, clear log."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            target = now.replace(hour=23, minute=57, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            
            sleep_sec = (target - now).total_seconds()
            logging.info(f"📊 Daily summary sleeping {sleep_sec:.0f}s until {target.strftime('%Y-%m-%d %H:%M')} UTC")
            await asyncio.sleep(sleep_sec)

            log = load_breakout_log()
            if not log:
                logging.info("📭 No breakouts to report, skipping daily summary.")
                await asyncio.sleep(120)
                continue

            # 1. Calculate P&L and update bank BEFORE building text
            price_map = {}
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                    if resp.status == 200:
                        for p in await resp.json():
                            price_map[p["symbol"]] = float(p["price"])
            except Exception as e:
                logging.error(f"❌ Daily summary price fetch: {e}")

            trades_pnl = []
            for entry in log:
                sym = entry["symbol"]
                ai_dir = entry.get("ai_direction", "")
                ai_entry = entry.get("ai_entry")
                ai_sl = entry.get("ai_sl")
                ai_tp = entry.get("ai_tp")
                entry_price = ai_entry if ai_entry else entry.get("breakout_price", 0)
                now_price = price_map.get(sym, entry.get("current_price", 0))

                # Determine TP/SL hit
                status = "open"
                if ai_dir and entry_price > 0:
                    if ai_dir == "LONG":
                        if ai_tp and now_price >= ai_tp:
                            status = "tp"
                        elif ai_sl and now_price <= ai_sl:
                            status = "sl"
                    elif ai_dir == "SHORT":
                        if ai_tp and now_price <= ai_tp:
                            status = "tp"
                        elif ai_sl and now_price >= ai_sl:
                            status = "sl"

                # Calculate P&L based on direction
                if entry_price > 0:
                    if ai_dir == "SHORT":
                        if status == "tp" and ai_tp:
                            pnl_pct = ((entry_price - ai_tp) / entry_price) * 100
                        elif status == "sl" and ai_sl:
                            pnl_pct = ((entry_price - ai_sl) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - now_price) / entry_price) * 100
                    else:
                        if status == "tp" and ai_tp:
                            pnl_pct = ((ai_tp - entry_price) / entry_price) * 100
                        elif status == "sl" and ai_sl:
                            pnl_pct = ((ai_sl - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((now_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                pnl_dollar = (pnl_pct / 100) * VIRTUAL_BANK_POSITION_SIZE
                trades_pnl.append((sym, pnl_pct, pnl_dollar))

            # Update bank with today's results
            update_bank_with_trades(trades_pnl)

            # 2. Build signals text (now shows updated bank)
            chunks = await build_signals_text(session, lang="ru")

            # Prepend daily header to first chunk
            if chunks:
                chunks[0] = f"🕐 *Ежедневный итог (23:57 UTC)*\n\n{chunks[0]}"

            # 3. Send to group
            tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            for chunk in chunks:
                await session.post(tg_url, json={"chat_id": GROUP_CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                await asyncio.sleep(0.5)

            logging.info(f"✅ Daily summary sent to group.")

            # 4. Clear breakout log for next day
            from config import clear_breakout_log
            clear_breakout_log()
            logging.info("🧹 Breakout log cleared for next day.")

            await asyncio.sleep(120)
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
                            cb_lang = _load_langs().get(str(chat_id), "ru") if chat_id else "ru"
                            
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
                                    deny_msg = "⛔️ Only admins can post to Square!" if cb_lang == "en" else "⛔️ Только админы могут постить в Square!"
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": deny_msg, "show_alert": True}
                                    )
                                    continue
                                
                                post_id = cb_data.replace("sq_", "")
                                text_to_post = square_cache_get(post_id)
                                if text_to_post:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⏳ Publishing..." if cb_lang == "en" else "⏳ Публикую..."}
                                    )
                                    result_msg = await post_to_binance_square(text_to_post)
                                    await send_response(app_session, chat_id, result_msg)
                                    square_cache_delete(post_id)
                                else:
                                    await app_session.post(
                                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                        json={"callback_query_id": cq_id, "text": "⚠️ Text is outdated." if cb_lang == "en" else "⚠️ Текст устарел.", "show_alert": True}
                                    )
                                continue

                            # 2. Binance Web3 Skills Buttons
                            if cb_data.startswith("sk_"):
                                await app_session.post(
                                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                                    json={"callback_query_id": cq_id, "text": "⏳ Fetching Web3 data..." if cb_lang == "en" else "⏳ Загружаю Web3 данные..."}
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

                        # --- NEW MEMBER WELCOME ---
                        new_members = msg.get("new_chat_members", [])
                        if new_members:
                            chat_id = msg.get("chat", {}).get("id")
                            for member in new_members:
                                name = member.get("first_name", "User")
                                welcome = (
                                    f"👋 *Welcome, {name}!*\n\n"
                                    "I'm *AiAlisa CopilotClaw* — AI Trading Assistant powered by OpenClaw 🦞\n\n"
                                    "*📋 Commands / Команды:*\n\n"
                                    "🔍 `scan BTC` / `посмотри BTC` — _AI analysis / анализ_\n"
                                    "📚 `/learn BTC` _(any coin)_ — _education / обучение_\n"
                                    "🏆 `/signals` — _winrate / точность_\n"
                                    "💰 `margin 100 leverage 10` — _stop-loss calc_\n"
                                    "🛠 `/skills` — _Web3 Skills_\n"
                                    "📈 `/top gainers` · 📉 `/top losers`\n"
                                    "📊 `/trend` — _breakouts / пробития_\n"
                                    "🔔 `/alert BTC 69500` — _price alert_\n"
                                    "🌐 `/lang en` | `/lang ru` — _language_\n\n"
                                    "Type `/help` for full list! 🚀"
                                )
                                await send_response(app_session, chat_id, welcome, parse_mode="Markdown")
                            continue

                        original_text = msg.get("text", "")
                        text = original_text.lower()
                        chat_id = msg.get("chat", {}).get("id")
                        msg_id = msg.get("message_id")

                        if not text:
                            continue
                            
                        # LANGUAGE: saved preference OR auto-detect from text
                        saved_lang = get_chat_lang(chat_id)
                        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
                        lang_pref = "ru" if has_cyrillic else saved_lang

                        # ==========================================
                        # LANGUAGE SWITCH: /lang en | /lang ru
                        # ==========================================
                        if text.startswith("/lang"):
                            parts = text.split()
                            if len(parts) >= 2 and parts[1] in ("en", "ru"):
                                set_chat_lang(chat_id, parts[1])
                                if parts[1] == "en":
                                    await send_response(app_session, chat_id, "🌐 Language set to *English* 🇬🇧", msg_id, parse_mode="Markdown")
                                else:
                                    await send_response(app_session, chat_id, "🌐 Язык установлен: *Русский* 🇷🇺", msg_id, parse_mode="Markdown")
                            else:
                                await send_response(app_session, chat_id, "🌐 Usage: `/lang en` or `/lang ru`", msg_id, parse_mode="Markdown")
                            continue

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
                                "🤖 *AiAlisa CopilotClaw* — AI Trading Assistant\n\n"
                                "*📋 Commands / Команды:*\n\n"
                                "🔍 `scan BTC` / `посмотри BTC`\n"
                                "    _AI analysis + chart / AI анализ + график_\n\n"
                                "📚 `/learn BTC` _(any futures coin / любая фьючерсная монета)_\n"
                                "    _Education: indicators explained / Обучение: объяснение индикаторов_\n\n"
                                "🏆 `/signals`\n"
                                "    _Signal accuracy & winrate / Точность сигналов_\n\n"
                                "💰 `margin 100 leverage 10 max 20%`\n"
                                "    _Stop-loss calculator / Расчёт стоп-лосса_\n\n"
                                "🛠 `/skills`\n"
                                "    _Web3 Skills menu / Меню Web3 навыков_\n\n"
                                "📈 `/top gainers` · 📉 `/top losers`\n"
                                "    _Top 10 growth/drops 24h / Топ 10 рост/падение_\n\n"
                                "📊 `/trend`\n"
                                "    _All breakouts since scan / Все пробития_\n\n"
                                "🔔 `/alert BTC 69500`\n"
                                "    _Price alert / Алерт на цену_\n"
                                "🔔 `/alert list` — _active / активные_\n"
                                "🔔 `/alert clear` — _remove all / удалить все_\n\n"
                                "🌐 `/lang en` — English\n"
                                "🌐 `/lang ru` — Русский"
                            )
                            if is_admin(msg):
                                welcome_text += (
                                    "\n\n🔐 *Admin:*\n"
                                    "🧠 `/models` — AI engine\n"
                                    "⏰ `/time 18:30` — scan schedule\n"
                                    "📢 `/autopost on/off` — auto Square\n"
                                    "🪙 `/autopost SOL BTC` — coins\n"
                                    "⏰ `/autopost time 09:00 21:00` — post times\n"
                                    "✏️ `/post text` — post to Square\n"
                                    "✏️ reply `/post text` — AI + your opinion\n"
                                    "💼 `/paper BTC 74000 long 5x sl 73000 tp 75000`\n"
                                    "💼 `/paper` — portfolio + live P&L\n"
                                    "💼 `/paper close 1` — close position\n"
                                    "💼 `/paper history` — trade history + winrate\n"
                                    "💼 `/paper clear` — reset all"
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
                            has_text_arg = len(parts) >= 2 and parts[1].strip()
                            reply_msg_obj = msg.get("reply_to_message")

                            if reply_msg_obj and has_text_arg:
                                # Reply to message + /post <my opinion> → AI text + user opinion
                                replied_text = reply_msg_obj.get("text", "").strip()
                                replied_caption = reply_msg_obj.get("caption", "").strip()
                                ai_text = replied_text or replied_caption
                                user_opinion = parts[1].strip()

                                if not ai_text:
                                    no_text = "⚠️ Replied message has no text." if lang_pref == "en" else "⚠️ В ответном сообщении нет текста."
                                    await send_response(app_session, chat_id, no_text, msg_id)
                                else:
                                    post_content = f"🤖 AI-ALISA-COPILOTCLAW\n\n{ai_text}\n\n💬 {user_opinion}\n\n#AIBinance #BinanceSquare #Write2Earn"
                                    if len(post_content) > 1950:
                                        post_content = post_content[:1947] + "..."
                                    pub_msg = "⏳ Publishing to Binance Square..." if lang_pref == "en" else "⏳ Публикую в Binance Square..."
                                    await send_response(app_session, chat_id, pub_msg, msg_id)
                                    result = await post_to_binance_square(post_content)
                                    await send_response(app_session, chat_id, result, msg_id)
                            elif has_text_arg:
                                # /post <text> — post custom text
                                user_text = parts[1].strip()
                                pub_msg = "⏳ Publishing to Binance Square..." if lang_pref == "en" else "⏳ Публикую в Binance Square..."
                                await send_response(app_session, chat_id, pub_msg, msg_id)
                                result = await post_to_binance_square(user_text)
                                await send_response(app_session, chat_id, result, msg_id)
                            elif reply_msg_obj:
                                # Reply /post (no text) — just publish replied message
                                replied_text = reply_msg_obj.get("text", "").strip()
                                replied_caption = reply_msg_obj.get("caption", "").strip()
                                post_content = replied_text or replied_caption

                                if not post_content:
                                    no_text = "⚠️ Replied message has no text." if lang_pref == "en" else "⚠️ В ответном сообщении нет текста."
                                    await send_response(app_session, chat_id, no_text, msg_id)
                                else:
                                    if "#AIBinance" not in post_content:
                                        post_content = f"🤖 AI-ALISA-COPILOTCLAW\n\n{post_content}\n\n#AIBinance #BinanceSquare #Write2Earn"
                                    if len(post_content) > 1950:
                                        post_content = post_content[:1947] + "..."
                                    pub_msg = "⏳ Publishing to Binance Square..." if lang_pref == "en" else "⏳ Публикую в Binance Square..."
                                    await send_response(app_session, chat_id, pub_msg, msg_id)
                                    result = await post_to_binance_square(post_content)
                                    await send_response(app_session, chat_id, result, msg_id)
                            else:
                                if lang_pref == "en":
                                    post_help = ("✏️ *How to use:*\n"
                                        "`/post Your text for Binance Square`\n\n"
                                        "Or reply to any message with `/post` to publish it.\n\n"
                                        "Example:\n"
                                        "`/post Hello Binance! BTC looks bullish today 🚀`")
                                else:
                                    post_help = ("✏️ *Как использовать:*\n"
                                        "`/post Ваш текст для Binance Square`\n\n"
                                        "Или ответьте на любое сообщение командой `/post` чтобы опубликовать его.\n\n"
                                        "Пример:\n"
                                        "`/post Привет Бинанс! Сегодня BTC выглядит бычьим 🚀`")
                                await send_response(app_session, chat_id, post_help, msg_id, parse_mode="Markdown")
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
                                gain_load = "⏳ Loading top gainers (Futures)..." if lang_pref == "en" else "⏳ Загружаю топ растущих (Futures)..."
                                await send_response(app_session, chat_id, gain_load, msg_id)
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
                                loss_load = "⏳ Loading top losers (Futures)..." if lang_pref == "en" else "⏳ Загружаю топ падающих (Futures)..."
                                await send_response(app_session, chat_id, loss_load, msg_id)
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
                                top_usage = (
                                    "📊 *Usage:*\n"
                                    "`/top gainers` — Top 10 growth (24h)\n"
                                    "`/top losers` — Top 10 drops (24h)"
                                ) if lang_pref == "en" else (
                                    "📊 *Использование:*\n"
                                    "`/top gainers` — Топ 10 рост (24ч)\n"
                                    "`/top losers` — Топ 10 падение (24ч)"
                                )
                                await send_response(app_session, chat_id, top_usage, msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # LEARN MODE: /learn BTC — explain indicators
                        # ==========================================
                        if text.startswith("/learn"):
                            parts = original_text.split()
                            if len(parts) >= 2:
                                coin_raw = parts[1].upper().strip()
                                learn_symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
                                short_coin = learn_symbol.replace("USDT", "")

                                learn_load = f"📚 Analyzing {learn_symbol} on 4H + 1H + 15m..." if lang_pref == "en" else f"📚 Анализирую {learn_symbol} на 4Ч + 1Ч + 15м..."
                                await send_response(app_session, chat_id, learn_load, msg_id)

                                raw_4h = await fetch_klines(app_session, learn_symbol, "4h", 120)
                                raw_1h = await fetch_klines(app_session, learn_symbol, "1h", 120)
                                raw_15m = await fetch_klines(app_session, learn_symbol, "15m", 120)

                                if raw_4h:
                                    row_4h, _ = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")
                                    row_1h = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0] if raw_1h else None
                                    row_15m = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0] if raw_15m else None
                                    funding = await fetch_funding_rate(app_session, learn_symbol)

                                    def _fmt_tf_learn(row, tf_label, lang):
                                        price = row.get("close", 0)
                                        rsi = row.get("rsi14", 0)
                                        mfi = row.get("mfi", 0)
                                        adx = row.get("adx", 0)
                                        stoch = row.get("stoch_k", 0)
                                        macd_h = row.get("macd_hist", 0)
                                        obv = row.get("obv_status", "Unknown")
                                        ichimoku = row.get("ichimoku_status", "Unknown")
                                        supertrend = row.get("supertrend", "Unknown")
                                        cmf = row.get("cmf", 0)

                                        if lang == "ru":
                                            rsi_n = "перекупленность ⚠️" if rsi > 70 else "перепроданность 🟢" if rsi < 30 else "нейтрально"
                                            mfi_n = "перекупленность" if mfi > 80 else "перепроданность" if mfi < 20 else "нейтрально"
                                            adx_n = "сильный тренд 💪" if adx > 25 else "слабый/боковой"
                                            stoch_n = "перекупленность" if stoch > 80 else "перепроданность" if stoch < 20 else "нейтрально"
                                            macd_n = "бычий 📈" if macd_h > 0 else "медвежий 📉"
                                            return (
                                                f"⏱ *{tf_label}* | Цена: `${price:.6f}`\n"
                                                f"• RSI(14): `{rsi:.1f}` → {rsi_n}\n"
                                                f"• MFI: `{mfi:.1f}` → {mfi_n} | ADX: `{adx:.1f}` → {adx_n}\n"
                                                f"• StochRSI: `{stoch:.1f}` → {stoch_n} | MACD: {macd_n}\n"
                                                f"• SuperTrend: {supertrend} | Ichimoku: {ichimoku}\n"
                                                f"• OBV: {obv} | CMF: `{cmf:.4f}`\n"
                                                f"• BB: `{row.get('bb_lower',0):.4f}` / `{row.get('bb_mid',0):.4f}` / `{row.get('bb_upper',0):.4f}`\n"
                                            )
                                        else:
                                            rsi_n = "overbought ⚠️" if rsi > 70 else "oversold 🟢" if rsi < 30 else "neutral"
                                            mfi_n = "overbought" if mfi > 80 else "oversold" if mfi < 20 else "neutral"
                                            adx_n = "strong trend 💪" if adx > 25 else "weak/sideways"
                                            stoch_n = "overbought" if stoch > 80 else "oversold" if stoch < 20 else "neutral"
                                            macd_n = "bullish 📈" if macd_h > 0 else "bearish 📉"
                                            return (
                                                f"⏱ *{tf_label}* | Price: `${price:.6f}`\n"
                                                f"• RSI(14): `{rsi:.1f}` → {rsi_n}\n"
                                                f"• MFI: `{mfi:.1f}` → {mfi_n} | ADX: `{adx:.1f}` → {adx_n}\n"
                                                f"• StochRSI: `{stoch:.1f}` → {stoch_n} | MACD: {macd_n}\n"
                                                f"• SuperTrend: {supertrend} | Ichimoku: {ichimoku}\n"
                                                f"• OBV: {obv} | CMF: `{cmf:.4f}`\n"
                                                f"• BB: `{row.get('bb_lower',0):.4f}` / `{row.get('bb_mid',0):.4f}` / `{row.get('bb_upper',0):.4f}`\n"
                                            )

                                    header = f"📚 *{'Обучение' if lang_pref == 'ru' else 'Learn'}: {short_coin}*\n"
                                    header += f"💰 {'Funding Rate' if lang_pref == 'en' else 'Ставка финансирования'}: `{funding}`\n\n"

                                    # Explanations block
                                    if lang_pref == "ru":
                                        explain = (
                                            "📖 *Что означают индикаторы:*\n"
                                            "• *RSI(14)* — скорость изменения цены (>70 перекуплен, <30 перепродан)\n"
                                            "• *MFI* — RSI с объёмом, давление денег\n"
                                            "• *ADX* — сила тренда (>25 тренд, <20 флэт)\n"
                                            "• *StochRSI* — чувствительный RSI для разворотов\n"
                                            "• *MACD* — импульс тренда (гистограмма >0 = бычий)\n"
                                            "• *SuperTrend* — направление тренда по ATR\n"
                                            "• *Ichimoku* — облако (выше = бычий, ниже = медвежий)\n"
                                            "• *OBV* — баланс объёмов (накопление/распределение)\n"
                                            "• *CMF* — денежный поток (>0 покупатели, <0 продавцы)\n"
                                            "• *BB* — канал волатильности\n"
                                        )
                                    else:
                                        explain = (
                                            "📖 *Indicator Guide:*\n"
                                            "• *RSI(14)* — momentum (>70 overbought, <30 oversold)\n"
                                            "• *MFI* — RSI with volume, money pressure\n"
                                            "• *ADX* — trend strength (>25 trending, <20 ranging)\n"
                                            "• *StochRSI* — sensitive RSI for reversals\n"
                                            "• *MACD* — trend momentum (histogram >0 = bullish)\n"
                                            "• *SuperTrend* — trend direction via ATR\n"
                                            "• *Ichimoku* — cloud (above = bullish, below = bearish)\n"
                                            "• *OBV* — volume balance (accumulation/distribution)\n"
                                            "• *CMF* — money flow (>0 buyers, <0 sellers)\n"
                                            "• *BB* — volatility channel\n"
                                        )

                                    msg1 = header + _fmt_tf_learn(row_4h, "4H", lang_pref) + "\n"
                                    if row_1h:
                                        msg1 += _fmt_tf_learn(row_1h, "1H", lang_pref) + "\n"
                                    if row_15m:
                                        msg1 += _fmt_tf_learn(row_15m, "15m", lang_pref)

                                    # Send data first, then explanations
                                    await send_response(app_session, chat_id, msg1, msg_id, parse_mode="Markdown")
                                    await send_response(app_session, chat_id, explain, parse_mode="Markdown")
                                else:
                                    await send_response(app_session, chat_id, f"⚠️ Pair `{learn_symbol}` not found on Binance Futures.", msg_id, parse_mode="Markdown")
                            else:
                                hint = "📚 Usage: `/learn BTC` — explains all indicators for any coin" if lang_pref == "en" else "📚 Использование: `/learn BTC` — объяснит все индикаторы"
                                await send_response(app_session, chat_id, hint, msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # SIGNAL ACCURACY: /signals — winrate from breakout log
                        # ==========================================
                        if text.startswith("/signal"):
                            try:
                                chunks = await build_signals_text(app_session, lang=lang_pref)
                                for i, chunk in enumerate(chunks):
                                    rid = msg_id if i == 0 else None
                                    await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
                            except Exception as e:
                                logging.error(f"❌ /signals error: {e}")
                                await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
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
                                    empty_alert = "📭 No active alerts.\n\nUsage:\n`/alert BTC 69500`" if lang_pref == "en" else "📭 Нет активных алертов.\n\nИспользуйте:\n`/alert BTC 69500`"
                                    await send_response(app_session, chat_id, empty_alert, msg_id, parse_mode="Markdown")
                                else:
                                    hdr = "🔔 *Your alerts:*\n" if lang_pref == "en" else "🔔 *Ваши алерты:*\n"
                                    lines = [hdr]
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
                                clr_msg = "✅ All alerts cleared." if lang_pref == "en" else "✅ Все алерты удалены."
                                await send_response(app_session, chat_id, clr_msg, msg_id)
                                continue

                            # /alert BTC 69500 — set new alert
                            if len(parts) >= 3:
                                coin_raw = parts[1].upper().strip()
                                symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
                                try:
                                    target_price = float(parts[2].replace(",", "."))
                                except ValueError:
                                    err_price = "⚠️ Invalid price. Example: `/alert BTC 69500`" if lang_pref == "en" else "⚠️ Неверная цена. Пример: `/alert BTC 69500`"
                                    await send_response(app_session, chat_id, err_price, msg_id, parse_mode="Markdown")
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
                                    not_found = f"⚠️ Pair `{symbol}` not found on Binance Futures." if lang_pref == "en" else f"⚠️ Пара `{symbol}` не найдена на Binance Futures."
                                    await send_response(app_session, chat_id, not_found, msg_id, parse_mode="Markdown")
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
                                if lang_pref == "en":
                                    dir_text = "rises above" if direction == "above" else "drops below"
                                    await send_response(app_session, chat_id,
                                        f"✅ Alert set!\n\n"
                                        f"🪙 `${short}`\n"
                                        f"💰 Now: `${current_price:.6f}`\n"
                                        f"{arrow} Target: `${target_price:.6f}`\n"
                                        f"📩 I'll notify you when price {dir_text} target.",
                                        msg_id, parse_mode="Markdown")
                                else:
                                    dir_text = "поднимется" if direction == "above" else "опустится"
                                    await send_response(app_session, chat_id,
                                        f"✅ Алерт установлен!\n\n"
                                        f"🪙 `${short}`\n"
                                        f"💰 Сейчас: `${current_price:.6f}`\n"
                                        f"{arrow} Цель: `${target_price:.6f}`\n"
                                        f"📩 Уведомлю когда цена {dir_text} до цели.",
                                        msg_id, parse_mode="Markdown")
                                continue

                            # No args — show help
                            if lang_pref == "en":
                                alert_help = ("🔔 *Price Alert:*\n\n"
                                    "Set: `/alert BTC 69500`\n"
                                    "List: `/alert list`\n"
                                    "Clear all: `/alert clear`")
                            else:
                                alert_help = ("🔔 *Price Alert:*\n\n"
                                    "Установить: `/alert BTC 69500`\n"
                                    "Список: `/alert list`\n"
                                    "Удалить все: `/alert clear`")
                            await send_response(app_session, chat_id, alert_help, msg_id, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # TREND BREAKOUT LIST (/trend) — simple breakout list
                        # ==========================================
                        if text.startswith("/trend") or text in ["тренд", "тренды", "пробития"]:
                            log = load_breakout_log()
                            if not log:
                                no_brk = "📭 No breakouts since last scan." if lang_pref == "en" else "📭 Нет пробитий с последнего скана."
                                await send_response(app_session, chat_id, no_brk, msg_id)
                                continue

                            # Batch fetch prices
                            price_map = {}
                            try:
                                async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                                    if resp.status == 200:
                                        for p in await resp.json():
                                            price_map[p["symbol"]] = float(p["price"])
                            except Exception:
                                pass

                            hdr = "📊 *Trendline Breakouts:*\n" if lang_pref == "en" else "📊 *Пробития трендовых линий:*\n"
                            lines = [hdr]
                            for entry in log:
                                sym = entry["symbol"]
                                short = sym.replace("USDT", "")
                                tf = entry["tf"]
                                bp = entry["breakout_price"]
                                now_price = price_map.get(sym, entry.get("current_price", 0))
                                diff_pct = ((now_price / bp) - 1) * 100 if bp > 0 else 0
                                arrow = "🟢" if diff_pct >= 0 else "🔴"

                                ai_dir = entry.get("ai_direction", "")
                                ai_mark = ""
                                if ai_dir:
                                    ai_ok = (ai_dir == "LONG" and diff_pct >= 0) or (ai_dir == "SHORT" and diff_pct < 0)
                                    ai_mark = "✅" if ai_ok else "❌"

                                bp_lbl = "Breakout" if lang_pref == "en" else "Пробитие"
                                now_lbl = "Now" if lang_pref == "en" else "Сейчас"
                                lines.append(
                                    f"{arrow}{ai_mark} `${short}` ({tf})\n"
                                    f"    {bp_lbl}: `${bp:.6f}`\n"
                                    f"    {now_lbl}: `${now_price:.6f}` (*{diff_pct:+.2f}%*)"
                                )

                            total_lbl = "Total" if lang_pref == "en" else "Всего"
                            coins_lbl = "coins" if lang_pref == "en" else "монет"
                            lines.append(f"\n_{total_lbl}: {len(log)} {coins_lbl}_")

                            full = "\n".join(lines)
                            if len(full) <= 4000:
                                await send_response(app_session, chat_id, full, msg_id, parse_mode="Markdown")
                            else:
                                chunks = []
                                cur = ""
                                for line in lines:
                                    if len(cur) + len(line) + 1 > 3900:
                                        chunks.append(cur)
                                        cur = line
                                    else:
                                        cur += "\n" + line if cur else line
                                if cur:
                                    chunks.append(cur)
                                for i, chunk in enumerate(chunks):
                                    rid = msg_id if i == 0 else None
                                    await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
                            continue

                        # ==========================================
                        # PAPER TRADING: /paper — virtual portfolio (admin only)
                        # ==========================================
                        if text.startswith("/paper"):
                            if not is_admin(msg):
                                await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
                                continue

                            user_id = str(msg.get("from", {}).get("id", 0))
                            parts = original_text.split()
                            paper = _load_paper()
                            if user_id not in paper:
                                paper[user_id] = {"open": [], "closed": []}
                            # Migrate old format (list → dict)
                            if isinstance(paper[user_id], list):
                                paper[user_id] = {"open": paper[user_id], "closed": []}
                            user_data = paper[user_id]

                            # /paper clear — remove all positions
                            if len(parts) == 2 and parts[1].lower() in ("clear", "очистить", "reset"):
                                paper[user_id] = {"open": [], "closed": []}
                                _save_paper(paper)
                                clr = "✅ Paper portfolio reset. History cleared." if lang_pref == "en" else "✅ Портфель сброшен. История очищена."
                                await send_response(app_session, chat_id, clr, msg_id)
                                continue

                            # /paper close 1 — close position by number
                            if len(parts) >= 2 and parts[1].lower() in ("close", "закрыть"):
                                if not user_data["open"]:
                                    await send_response(app_session, chat_id, "📭 No open positions." if lang_pref == "en" else "📭 Нет открытых позиций.", msg_id)
                                    continue
                                idx = 0
                                if len(parts) >= 3 and parts[2].isdigit():
                                    idx = int(parts[2]) - 1
                                if idx < 0 or idx >= len(user_data["open"]):
                                    await send_response(app_session, chat_id, f"⚠️ Position #{idx+1} not found. Use `/paper` to see list.", msg_id, parse_mode="Markdown")
                                    continue

                                pos = user_data["open"].pop(idx)
                                sym = pos["symbol"]
                                entry = pos["entry"]
                                direction = pos["direction"]
                                lev = pos["leverage"]
                                short_sym = sym.replace("USDT", "")

                                # Fetch close price
                                close_price = entry
                                try:
                                    async with app_session.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}", timeout=5) as resp:
                                        if resp.status == 200:
                                            d = await resp.json()
                                            close_price = float(d["price"])
                                except Exception:
                                    pass

                                if direction == "long":
                                    pnl_pct = ((close_price - entry) / entry) * 100 * lev
                                else:
                                    pnl_pct = ((entry - close_price) / entry) * 100 * lev

                                pos["close_price"] = close_price
                                pos["close_time"] = datetime.now(timezone.utc).isoformat()[:16]
                                pos["pnl_pct"] = round(pnl_pct, 2)
                                user_data["closed"].append(pos)
                                _save_paper(paper)

                                icon = "🟢" if pnl_pct >= 0 else "🔴"
                                arrow = "LONG" if direction == "long" else "SHORT"
                                if lang_pref == "ru":
                                    await send_response(app_session, chat_id,
                                        f"{icon} Позиция закрыта!\n\n"
                                        f"🪙 `{short_sym}` {arrow} {lev}x\n"
                                        f"💰 Вход: `${entry:.4f}` → Выход: `${close_price:.4f}`\n"
                                        f"📊 P&L: `{pnl_pct:+.2f}%`",
                                        msg_id, parse_mode="Markdown")
                                else:
                                    await send_response(app_session, chat_id,
                                        f"{icon} Position closed!\n\n"
                                        f"🪙 `{short_sym}` {arrow} {lev}x\n"
                                        f"💰 Entry: `${entry:.4f}` → Exit: `${close_price:.4f}`\n"
                                        f"📊 P&L: `{pnl_pct:+.2f}%`",
                                        msg_id, parse_mode="Markdown")
                                continue

                            # /paper history — show closed trades
                            if len(parts) == 2 and parts[1].lower() in ("history", "история"):
                                closed = user_data.get("closed", [])
                                if not closed:
                                    await send_response(app_session, chat_id, "📭 No closed trades yet." if lang_pref == "en" else "📭 Нет закрытых сделок.", msg_id)
                                    continue
                                header = "📜 *Trade History*\n\n" if lang_pref == "en" else "📜 *История сделок*\n\n"
                                lines = [header]
                                total = 0
                                wins = 0
                                for i, c in enumerate(closed[-20:], 1):  # Last 20
                                    pnl = c.get("pnl_pct", 0)
                                    total += pnl
                                    if pnl > 0:
                                        wins += 1
                                    icon = "🟢" if pnl >= 0 else "🔴"
                                    short = c["symbol"].replace("USDT", "")
                                    arr = "L" if c["direction"] == "long" else "S"
                                    lines.append(f"{icon} `{short}` {arr} {c['leverage']}x | `{pnl:+.2f}%` | {c.get('close_time', '')}")
                                wr = (wins / len(closed) * 100) if closed else 0
                                lines.append(f"\n📊 *Trades: {len(closed)} | Winrate: {wr:.0f}% | Total P&L: {total:+.2f}%*")
                                await send_response(app_session, chat_id, "\n".join(lines), msg_id, parse_mode="Markdown")
                                continue

                            # /paper BTC 74000 long 5x sl 73000 tp 75000 — add position
                            if len(parts) >= 4:
                                coin_raw = parts[1].upper().strip()
                                p_symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
                                try:
                                    entry_price = float(parts[2].replace(",", "."))
                                except ValueError:
                                    await send_response(app_session, chat_id, "⚠️ `/paper BTC 74000 long 5x sl 73000 tp 75000`", msg_id, parse_mode="Markdown")
                                    continue

                                direction = "long"
                                if len(parts) >= 4 and parts[3].lower() in ("short", "шорт", "s"):
                                    direction = "short"

                                leverage = 1
                                sl_price = None
                                tp_price = None
                                for j, p in enumerate(parts):
                                    p_low = p.lower()
                                    # Parse leverage (5x, 10x)
                                    p_clean = p_low.replace("x", "").replace("х", "")
                                    if p_clean.isdigit() and 1 < int(p_clean) <= 125:
                                        leverage = int(p_clean)
                                    # Parse SL
                                    if p_low in ("sl", "стоп") and j + 1 < len(parts):
                                        try:
                                            sl_price = float(parts[j+1].replace(",", "."))
                                        except ValueError:
                                            pass
                                    # Parse TP
                                    if p_low in ("tp", "тейк") and j + 1 < len(parts):
                                        try:
                                            tp_price = float(parts[j+1].replace(",", "."))
                                        except ValueError:
                                            pass

                                position = {
                                    "symbol": p_symbol,
                                    "entry": entry_price,
                                    "direction": direction,
                                    "leverage": leverage,
                                    "sl": sl_price,
                                    "tp": tp_price,
                                    "time": datetime.now(timezone.utc).isoformat()[:16]
                                }
                                user_data["open"].append(position)
                                _save_paper(paper)

                                short_coin = p_symbol.replace("USDT", "")
                                arrow = "📈 LONG" if direction == "long" else "📉 SHORT"
                                sl_text = f"\n🚫 SL: `${sl_price:.4f}`" if sl_price else ""
                                tp_text = f"\n🎯 TP: `${tp_price:.4f}`" if tp_price else ""
                                if lang_pref == "ru":
                                    await send_response(app_session, chat_id,
                                        f"✅ Виртуальная позиция открыта!\n\n"
                                        f"🪙 `{short_coin}` {arrow} {leverage}x\n"
                                        f"💰 Вход: `${entry_price:.6f}`{sl_text}{tp_text}",
                                        msg_id, parse_mode="Markdown")
                                else:
                                    await send_response(app_session, chat_id,
                                        f"✅ Paper position opened!\n\n"
                                        f"🪙 `{short_coin}` {arrow} {leverage}x\n"
                                        f"💰 Entry: `${entry_price:.6f}`{sl_text}{tp_text}",
                                        msg_id, parse_mode="Markdown")
                                continue

                            # /paper — show portfolio with live P&L + SL/TP status
                            open_positions = user_data.get("open", [])
                            if not open_positions:
                                empty = "📭 No open positions.\n\n`/paper BTC 74000 long 5x sl 73000 tp 75000`" if lang_pref == "en" else "📭 Нет открытых позиций.\n\n`/paper BTC 74000 long 5x sl 73000 tp 75000`"
                                await send_response(app_session, chat_id, empty, msg_id, parse_mode="Markdown")
                                continue

                            header = "💼 *Paper Trading Portfolio*\n\n" if lang_pref == "en" else "💼 *Виртуальный портфель*\n\n"
                            lines = [header]
                            total_pnl = 0
                            auto_closed = []

                            for i, pos in enumerate(open_positions, 1):
                                sym = pos["symbol"]
                                entry = pos["entry"]
                                direction = pos["direction"]
                                lev = pos["leverage"]
                                sl = pos.get("sl")
                                tp = pos.get("tp")
                                short_sym = sym.replace("USDT", "")

                                now_price = entry
                                try:
                                    async with app_session.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}", timeout=5) as resp:
                                        if resp.status == 200:
                                            d = await resp.json()
                                            now_price = float(d["price"])
                                except Exception:
                                    pass

                                if direction == "long":
                                    pnl_pct = ((now_price - entry) / entry) * 100 * lev
                                else:
                                    pnl_pct = ((entry - now_price) / entry) * 100 * lev

                                # Check SL/TP hit
                                hit = ""
                                if sl and direction == "long" and now_price <= sl:
                                    hit = " 🚫 *SL HIT*"
                                    auto_closed.append(i - 1)
                                elif sl and direction == "short" and now_price >= sl:
                                    hit = " 🚫 *SL HIT*"
                                    auto_closed.append(i - 1)
                                elif tp and direction == "long" and now_price >= tp:
                                    hit = " 🎯 *TP HIT*"
                                    auto_closed.append(i - 1)
                                elif tp and direction == "short" and now_price <= tp:
                                    hit = " 🎯 *TP HIT*"
                                    auto_closed.append(i - 1)

                                total_pnl += pnl_pct
                                icon = "🟢" if pnl_pct >= 0 else "🔴"
                                arrow_txt = "LONG" if direction == "long" else "SHORT"
                                sl_line = f"   🚫 SL: `${sl:.4f}`" if sl else ""
                                tp_line = f" | 🎯 TP: `${tp:.4f}`" if tp else ""

                                lines.append(
                                    f"#{i} {icon} `{short_sym}` {arrow_txt} {lev}x{hit}\n"
                                    f"   Entry: `${entry:.4f}` → Now: `${now_price:.4f}`\n"
                                    f"   P&L: `{pnl_pct:+.2f}%`\n"
                                    f"{sl_line}{tp_line}\n" if (sl or tp) else
                                    f"#{i} {icon} `{short_sym}` {arrow_txt} {lev}x{hit}\n"
                                    f"   Entry: `${entry:.4f}` → Now: `${now_price:.4f}`\n"
                                    f"   P&L: `{pnl_pct:+.2f}%`\n"
                                )

                            # Auto-close SL/TP positions
                            for idx in sorted(auto_closed, reverse=True):
                                pos = open_positions.pop(idx)
                                try:
                                    async with app_session.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={pos['symbol']}", timeout=5) as resp:
                                        if resp.status == 200:
                                            d = await resp.json()
                                            cp = float(d["price"])
                                        else:
                                            cp = pos["entry"]
                                except Exception:
                                    cp = pos["entry"]
                                if pos["direction"] == "long":
                                    pnl = ((cp - pos["entry"]) / pos["entry"]) * 100 * pos["leverage"]
                                else:
                                    pnl = ((pos["entry"] - cp) / pos["entry"]) * 100 * pos["leverage"]
                                pos["close_price"] = cp
                                pos["close_time"] = datetime.now(timezone.utc).isoformat()[:16]
                                pos["pnl_pct"] = round(pnl, 2)
                                user_data["closed"].append(pos)

                            if auto_closed:
                                _save_paper(paper)

                            total_icon = "🟢" if total_pnl >= 0 else "🔴"
                            closed_count = len(user_data.get("closed", []))
                            lines.append(f"\n{total_icon} *Total P&L: {total_pnl:+.2f}%*")
                            if closed_count:
                                lines.append(f"📜 Closed trades: {closed_count} (`/paper history`)")

                            full_text = "\n".join(lines)
                            if len(full_text) > 4000:
                                full_text = full_text[:4000] + "..."
                            await send_response(app_session, chat_id, full_text, msg_id, parse_mode="Markdown")
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
                            fetch_msg = f"⏳ Fetching chart data + building trend line... ({symbol})" if lang_pref == "en" else f"⏳ Загружаю график + строю трендовую линию... ({symbol})"
                            stream_msg_id = await send_and_get_msg_id(
                                app_session, chat_id, fetch_msg, msg_id
                            )

                            # Fetch 199 candles for trend line construction (same as main scanner)
                            raw_df_full = await fetch_klines(app_session, symbol, "4h", 199)
                            # Fetch multi-timeframe: 4H (primary) + 1H + 15m + 1D
                            raw_df_4h = await fetch_klines(app_session, symbol, "4h", 120)
                            raw_df_1h = await fetch_klines(app_session, symbol, "1h", 120)
                            raw_df_15m = await fetch_klines(app_session, symbol, "15m", 120)
                            raw_df_1d = await fetch_klines(app_session, symbol, "1d", 120)

                            raw_df = raw_df_4h  # primary TF for compatibility

                            if raw_df:
                                df = pd.DataFrame(raw_df)
                                last_row, full_df = calculate_binance_indicators(df, "4H")
                                funding = await fetch_funding_rate(app_session, symbol)
                                last_row["funding_rate"] = funding

                                # Build multi-TF data
                                mtf_data = {}
                                if raw_df_1d:
                                    mtf_data["1D"] = calculate_binance_indicators(pd.DataFrame(raw_df_1d), "1D")[0]
                                if raw_df_1h:
                                    mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_df_1h), "1H")[0]
                                if raw_df_15m:
                                    mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_df_15m), "15m")[0]

                                # Build telegram_stream dict for live AI streaming
                                tg_stream = None
                                if stream_msg_id:
                                    tg_stream = {
                                        "session": app_session,
                                        "chat_id": chat_id,
                                        "message_id": stream_msg_id,
                                        "bot_token": BOT_TOKEN
                                    }

                                ai_msg = await ask_ai_analysis(symbol, "4H", last_row, lang=lang_pref, telegram_stream=tg_stream, extended=True, mtf_data=mtf_data)

                                # Schedule delayed deletion of streaming message (15s after chart sent)
                                async def _delayed_delete(sess, cid, mid, delay=15):
                                    await asyncio.sleep(delay)
                                    try:
                                        await sess.post(
                                            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage",
                                            json={"chat_id": cid, "message_id": mid}, timeout=5
                                        )
                                    except Exception:
                                        pass

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
                                # Use Part 2 (extended analysis) for Square, fall back to full text
                                import uuid
                                post_id = str(uuid.uuid4())[:8]
                                square_ai = ai_msg
                                if "---" in ai_msg:
                                    sq_parts = ai_msg.split("---", 1)
                                    if len(sq_parts) > 1 and sq_parts[1].strip():
                                        square_ai = sq_parts[1].strip()
                                short_sym = symbol.replace("USDT", "")
                                square_text = f"🤖 AI-ALISA-COPILOTCLAW Analysis: ${short_sym}\n\n{square_ai}\n\n#AIBinance #BinanceSquare #{short_sym} #Write2Earn"
                                if len(square_text) > 1950:
                                    square_text = square_text[:1947] + "..."
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

                                # --- SPLIT AI RESPONSE: Part 1 (caption) + Part 2 (extended) ---
                                ai_part1 = ai_msg
                                ai_part2 = None
                                if "---" in ai_msg:
                                    parts = ai_msg.split("---", 1)
                                    ai_part1 = parts[0].strip()
                                    ai_part2 = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

                                # --- SEND: chart + AI text, or just text if no line found ---
                                if chart_path:
                                    import os as _os
                                    safe_ai = ai_part1 if len(ai_part1) < 800 else ai_part1[:800] + "..."
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
                                    await send_response(app_session, chat_id, ai_part1, msg_id, reply_markup=scan_markup)

                                # Send extended analysis as second message with Square button
                                if ai_part2:
                                    # Prepare separate Square cache for extended part
                                    post_id2 = str(uuid.uuid4())[:8]
                                    short_sym2 = symbol.replace("USDT", "")
                                    sq_text2 = f"🤖 AI-ALISA-COPILOTCLAW Analysis: ${short_sym2}\n\n{ai_part2}\n\n#AIBinance #BinanceSquare #{short_sym2} #Write2Earn"
                                    if len(sq_text2) > 1950:
                                        sq_text2 = sq_text2[:1947] + "..."
                                    square_cache_put(post_id2, sq_text2)

                                    ext_markup = {
                                        "inline_keyboard": [
                                            [{"text": "📢 Post to Binance Square", "callback_data": f"sq_{post_id2}"}]
                                        ]
                                    }
                                    extended_text = f"🔬 *{symbol} — Extended Analysis*\n\n{ai_part2}"
                                    if len(extended_text) > 4000:
                                        extended_text = extended_text[:4000] + "..."
                                    await send_response(app_session, chat_id, extended_text, parse_mode="Markdown", reply_markup=ext_markup)

                                # Delete streaming message 15s after chart/text sent
                                if stream_msg_id:
                                    asyncio.create_task(_delayed_delete(app_session, chat_id, stream_msg_id, 15))

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

                                raw_4h = await fetch_klines(app_session, coin_to_analyze, "4h", 120)
                                raw_1h = await fetch_klines(app_session, coin_to_analyze, "1h", 120)
                                raw_15m = await fetch_klines(app_session, coin_to_analyze, "15m", 120)
                                raw_1d = await fetch_klines(app_session, coin_to_analyze, "1d", 120)
                                if raw_4h:
                                    last_row, _ = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")
                                    funding = await fetch_funding_rate(app_session, coin_to_analyze)
                                    last_row["funding_rate"] = funding
                                    mtf_data = {}
                                    if raw_1d:
                                        mtf_data["1D"] = calculate_binance_indicators(pd.DataFrame(raw_1d), "1D")[0]
                                    if raw_1h:
                                        mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                                    if raw_15m:
                                        mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]
                                    ai_msg = await ask_ai_analysis(coin_to_analyze, "4H", last_row, user_margin=margin_data, lang=lang_pref, mtf_data=mtf_data)
                                    await send_response(app_session, chat_id, ai_msg, msg_id)

        except Exception as e:
            logging.error(f"TG Polling Error: {e}")
            await asyncio.sleep(2)
