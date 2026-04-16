"""
tg_commands.py — All /command and text-trigger handlers.
Extracted from tg_listener.py during refactor.
"""
import asyncio
import aiohttp
import logging
import re
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import (
    BOT_TOKEN, load_breakout_log, load_price_alerts, save_price_alerts,
    load_virtual_bank, save_virtual_bank, reset_virtual_bank,
    VIRTUAL_BANK_POSITION_SIZE,
    OPENROUTER_API_KEY_MONITOR, OPENROUTER_MODEL_MONITOR, MONITOR_GROUP_CHAT_ID,
    load_monitor_breakout_log, load_monitor_virtual_bank,
    reset_monitor_virtual_bank,
)
import config as _cfg
import agent.analyzer
import agent.square_publisher
from agent.square_publisher import set_coins, set_times, get_coins, get_times, get_status_text, set_hashtags, get_hashtags
from agent.skills import post_to_binance_square
from core.binance_api import fetch_klines, fetch_funding_rate, fetch_funding_history, fetch_market_positioning, format_positioning_text
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
from agent.skills import (
    get_smart_money_signals,
    get_unified_token_rank,
    get_social_hype_leaderboard,
    get_smart_money_inflow_rank,
    get_meme_rank,
    get_address_pnl_rank,
)
from core.geometry_scanner import find_trend_line
from core.chart_drawer import draw_scan_chart, draw_simple_chart

from core.tg_state import (
    send_response, send_and_get_msg_id, get_chat_lang, set_chat_lang,
    is_allowed_chat, is_admin, ADMIN_ID,
    _load_paper, _save_paper,
    square_cache_put,
    SCAN_SCHEDULE, _save_scan_schedule,
    _fetch_or_free_models,
)
from core.tg_reports import (
    build_signals_text, build_signals_close_text,
    build_signals_text_monitor, build_signals_close_text_monitor,
)


async def handle_message(app_session, update):
    """Handle a regular message update (commands, text triggers, new members).

    Returns silently for messages that are ignored or handled.
    """
    msg = update.get("message") or {}

    # --- NEW MEMBER WELCOME ---
    new_members = msg.get("new_chat_members", [])
    if new_members:
        chat_id = msg.get("chat", {}).get("id")
        if not is_allowed_chat(chat_id):
            return
        for member in new_members:
            name = member.get("first_name", "User")
            welcome = (
                f"👋 *Welcome, {name}!*\n\n"
                "I'm *AiAlisa CopilotClaw* — AI Trading Assistant 🤖\n\n"
                "*📋 Commands / Команды:*\n\n"
                "🔍 `scan BTC` / `посмотри BTC` — _AI analysis / анализ_\n"
                "📚 `/learn BTC` _(any coin)_ — _education / обучение_\n"
                "🏆 `/signals` — _winrate (admin)_\n"
                "💰 `margin 100 leverage 10` — _stop-loss calc_\n"
                "🛠 `/skills` — _Web3 Skills_\n"
                "📈 `/top gainers` · 📉 `/top losers`\n"
                "📊 `/trend` — _breakouts / пробития_\n"
                "🔔 `/alert BTC 69500` — _price alert_\n"
                "🌐 `/lang en` | `/lang ru` — _language_\n\n"
                "Type `/help` for full list! 🚀"
            )
            await send_response(app_session, chat_id, welcome, parse_mode="Markdown")
        return

    original_text = msg.get("text", "")
    text = original_text.lower()
    chat_id = msg.get("chat", {}).get("id")
    msg_id = msg.get("message_id")

    if not text:
        return

    # --- CHAT FILTER ---
    if not is_allowed_chat(chat_id):
        return

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
            new_lang = parts[1]
            set_chat_lang(chat_id, new_lang)
            from config import GROUP_CHAT_ID as _gcid, MONITOR_GROUP_CHAT_ID as _mcid
            if _gcid and str(chat_id) != str(_gcid):
                set_chat_lang(_gcid, new_lang)
            if _mcid and str(chat_id) != str(_mcid):
                set_chat_lang(_mcid, new_lang)
            if new_lang == "en":
                await send_response(app_session, chat_id, "🌐 Language set to *English* 🇬🇧 (autopush + monitor)", msg_id, parse_mode="Markdown")
            else:
                await send_response(app_session, chat_id, "🌐 Язык: *Русский* 🇷🇺 (автопуш + монитор)", msg_id, parse_mode="Markdown")
        else:
            await send_response(app_session, chat_id, "🌐 Usage: `/lang en` or `/lang ru`", msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # API KEY COMMANDS (/key)
    # ==========================================
    if text.startswith("/key"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        parts = original_text.split(maxsplit=2)
        # /key — show current keys (masked)
        if len(parts) == 1:
            main_key = _cfg.OPENROUTER_API_KEY or ""
            mon_key = _cfg.OPENROUTER_API_KEY_MONITOR or ""
            main_masked = f"...{main_key[-8:]}" if len(main_key) > 8 else ("✅ set" if main_key else "❌ not set")
            mon_masked = f"...{mon_key[-8:]}" if len(mon_key) > 8 else ("❌ not set (using main)" if not mon_key else "✅ set")
            await send_response(app_session, chat_id,
                f"🔑 *API Keys:*\n\n"
                f"🧠 Main: `{main_masked}`\n"
                f"🔵 Monitor: `{mon_masked}`\n\n"
                f"Сменить: `/key <new-key>`\n"
                f"Monitor: `/key monitor <new-key>`",
                msg_id, parse_mode="Markdown")
            return

        # /key monitor <new-key>
        if parts[1].lower() == "monitor":
            if len(parts) >= 3 and parts[2].strip():
                new_key = parts[2].strip()
                _cfg.OPENROUTER_API_KEY_MONITOR = new_key
                _cfg.update_env_file("OPENROUTER_API_KEY_MONITOR", new_key)
                masked = f"...{new_key[-8:]}" if len(new_key) > 8 else new_key
                await send_response(app_session, chat_id,
                    f"✅ Monitor API key updated & saved: `{masked}`",
                    msg_id, parse_mode="Markdown")
                logging.info("🔑 Monitor API key changed by admin (persisted to .env)")
            else:
                await send_response(app_session, chat_id,
                    "Usage: `/key monitor <new-api-key>`",
                    msg_id, parse_mode="Markdown")
            return

        # /key <new-key>
        new_key = parts[1].strip()
        _cfg.OPENROUTER_API_KEY = new_key
        agent.analyzer.OPENROUTER_API_KEY = new_key
        _cfg.update_env_file("OPENROUTER_API_KEY", new_key)
        masked = f"...{new_key[-8:]}" if len(new_key) > 8 else new_key
        await send_response(app_session, chat_id,
            f"✅ Main API key updated & saved: `{masked}`",
            msg_id, parse_mode="Markdown")
        logging.info("🔑 Main API key changed by admin (persisted to .env)")
        return

    # ==========================================
    # AI MODEL COMMANDS (/models)
    # ==========================================
    if text.startswith("/models") or text.startswith("/model"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        parts = text.split(maxsplit=1)
        sub_cmd = parts[1].strip() if len(parts) > 1 else ""

        # /model monitor
        if sub_cmd.lower().startswith("monitor"):
            mon_parts = sub_cmd.split(maxsplit=1)
            if len(mon_parts) >= 2 and mon_parts[1].strip():
                new_mon_model = mon_parts[1].strip()
                _cfg.OPENROUTER_MODEL_MONITOR = new_mon_model
                _s = agent.analyzer._get_ai_settings()
                _s["monitor_model"] = new_mon_model
                agent.analyzer._save_and_cache_settings(_s)
                await send_response(app_session, chat_id,
                    f"✅ Monitor AI model switched to:\n`{new_mon_model}`\n\n"
                    f"🔑 Monitor API key: {'✅ set' if _cfg.OPENROUTER_API_KEY_MONITOR else '❌ not set (using main key)'}",
                    msg_id, parse_mode="Markdown")
            else:
                cur_mon = _cfg.OPENROUTER_MODEL_MONITOR or agent.analyzer.OPENROUTER_MODEL
                has_key = "✅ отдельный ключ" if _cfg.OPENROUTER_API_KEY_MONITOR else "⚠️ основной ключ"
                mon_group = _cfg.MONITOR_GROUP_CHAT_ID or "не задана (шлёт в основную)"
                await send_response(app_session, chat_id,
                    f"🔵 *Monitor настройки:*\n\n"
                    f"🧠 Модель: `{cur_mon}`\n"
                    f"🔑 API ключ: {has_key}\n"
                    f"💬 Группа: `{mon_group}`\n\n"
                    f"Сменить модель: `/model monitor <model-id>`",
                    msg_id, parse_mode="Markdown")
            return

        # /models all
        if sub_cmd.lower() == "all":
            await send_response(app_session, chat_id, "⏳ Fetching all models from OpenRouter...", msg_id)
            try:
                async with app_session.get("https://openrouter.ai/api/v1/models", timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        models.sort(key=lambda m: m.get("id", ""))
                        lines = [f"🧠 *All OpenRouter Models ({len(models)}):*\n"]
                        lines.append(f"Current: `{agent.analyzer.OPENROUTER_MODEL}`\n")
                        for m in models:
                            mid = m.get("id", "?")
                            pricing = m.get("pricing", {})
                            prompt_price = pricing.get("prompt", "?")
                            is_free = str(prompt_price) == "0"
                            free_tag = " 🆓" if is_free else ""
                            lines.append(f"`{mid}`{free_tag}")
                        chunks = []
                        current_chunk = ""
                        for line in lines:
                            if len(current_chunk) + len(line) + 2 > 3900:
                                chunks.append(current_chunk)
                                current_chunk = line
                            else:
                                current_chunk += "\n" + line
                        if current_chunk:
                            chunks.append(current_chunk)
                        for chunk in chunks:
                            await send_response(app_session, chat_id, chunk, parse_mode="Markdown")
                        await send_response(app_session, chat_id,
                            f"💡 To switch: `/models <full-model-id>`\nExample: `/models google/gemini-2.5-flash-preview-05-20`")
                    else:
                        await send_response(app_session, chat_id, f"❌ OpenRouter API error: {resp.status}", msg_id)
            except Exception as e:
                await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
            return

        # /models <model-name>
        if sub_cmd and sub_cmd.lower() not in ("all", "monitor") and "/" in sub_cmd:
            new_model = sub_cmd.strip()
            from agent.analyzer import set_active_provider
            set_active_provider("openrouter", model=new_model)
            _cfg.OPENROUTER_MODEL = new_model
            agent.analyzer.OPENROUTER_MODEL = new_model
            await send_response(app_session, chat_id,
                f"✅ Provider: *OpenRouter*\nModel: `{new_model}`", msg_id, parse_mode="Markdown")
            return

        # /models — show provider selection
        from agent.analyzer import get_active_provider_info
        prov, mdl, _ = get_active_provider_info()
        model_text = f"🧠 *AI Provider Selection*\nActive: `{prov}` / `{mdl}`\n\n💡 `/models all` — full OpenRouter list\n💡 `/models <model-id>` — switch OpenRouter manually"
        model_markup = {
            "inline_keyboard": [
                [{"text": "🌐 OpenRouter", "callback_data": "prov_or"},
                 {"text": "💎 Gemini", "callback_data": "prov_gm"},
                 {"text": "⚡ Groq", "callback_data": "prov_gq"}]
            ]
        }
        await send_response(app_session, chat_id, model_text, msg_id, reply_markup=model_markup, parse_mode="Markdown")
        return

    # ==========================================
    # BINANCE WEB3 SKILLS MENU
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
        return

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
        return

    # ==========================================
    # BASIC COMMANDS (/start, /help, /time, /autopost)
    # ==========================================
    if text.startswith("/start") or text.startswith("/help") or text in ["привет", "hello"]:
        welcome_text = (
            "🤖 *AiAlisa CopilotClaw* — AI Trading Assistant\n\n"
            "*📋 Commands / Команды:*\n\n"
            "🔍 `scan BTC` / `посмотри BTC`\n"
            "    _AI analysis + chart / AI анализ + график_\n\n"
            "📚 `/learn BTC` _(any futures coin / любая фьючерсная монета)_\n"
            "    _Education: indicators explained / Обучение: объяснение индикаторов_\n\n"
            "💰 `margin 100 leverage 10 max 20%`\n"
            "    _Stop-loss calculator / Расчёт стоп-лосса_\n\n"
            "🛠 `/skills`\n"
            "    _Web3 Skills menu / Меню Web3 навыков_\n\n"
            "📈 `/top gainers` · 📉 `/top losers`\n"
            "    _Top 10 growth/drops 24h / Топ 10 рост/падение_\n\n"
            "📊 `/trend`\n"
            "    _All breakouts since scan / Все пробития_\n\n"
            "👁 `/monitor`\n"
            "    _Coins in monitor / Монеты в мониторинге_\n\n"
            "📊 `/vol`\n"
            "    _Volume waitlist / Ожидание объёма_\n\n"
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
                "🏆 `/signals` — signal winrate & bank\n"
                "🔒 `/signals close` — snapshot: close all open now\n"
                "🔄 `/signals clear` — reset bank to $10k\n"
                "🔵 `/bankm` — monitor bank & trades\n"
                "🔒 `/bankm close` — close all monitor now\n"
                "🔄 `/bankm clear` — reset monitor bank to $10k\n"
                "🧠 `/models` — AI engine\n"
                "🧠 `/models all` — all OpenRouter models\n"
                "🧠 `/models <id>` — switch to any model\n"
                "🧠 `/model monitor` — monitor AI settings\n"
                "🧠 `/model monitor <id>` — switch monitor model\n"
                "⏰ `/time 18:30` — scan schedule\n"
                "📢 `/autopost on/off` — auto Square\n"
                "🪙 `/autopost SOL BTC` — coins\n"
                "⏰ `/autopost time 09:00 15:00 21:00` — post times\n"
                "🏷 `/autopost hashtags #tag1 #tag2` — hashtags\n"
                "✏️ `/post text` — post to Square\n"
                "✏️ reply `/post text` — AI + your opinion\n"
                "💼 `/paper BTC 74000 long 5x sl 73000 tp 75000`\n"
                "💼 `/paper` — portfolio + live P&L\n"
                "💼 `/paper close 1` — close position\n"
                "💼 `/paper history` — trade history + winrate\n"
                "💼 `/paper clear` — reset all\n"
                "🔑 `/testapi AIzaSy...` — test Gemini key"
            )
        await send_response(app_session, chat_id, welcome_text, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # TEST GEMINI API KEY (/testapi <key>)
    # ==========================================
    if text.startswith("/testapi"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return
        parts = original_text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await send_response(app_session, chat_id,
                "🔑 *Test Gemini API Key*\n\n"
                "Usage: `/testapi AIzaSy...`\n\n"
                "Tests key against all Gemini models via proxy.",
                msg_id, parse_mode="Markdown")
            return
        test_key = parts[1].strip()
        _test_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        await send_response(app_session, chat_id,
            f"🔑 Testing key against {len(_test_models)} models...", msg_id)
        _test_results = []
        _any_ok = False
        _key_invalid = False
        req_timeout = aiohttp.ClientTimeout(total=15)
        test_payload = {"contents": [{"parts": [{"text": "Say OK"}]}]}
        async with aiohttp.ClientSession() as test_session:
            for _tm in _test_models:
                try:
                    test_url = f"https://botgem.zhoriha.workers.dev/v1beta/models/{_tm}:generateContent?key={test_key}"
                    async with test_session.post(test_url, json=test_payload, timeout=req_timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            candidates = data.get("candidates", [])
                            if candidates:
                                _test_results.append(f"✅ `{_tm}`")
                                _any_ok = True
                            else:
                                _test_results.append(f"⚠️ `{_tm}` — empty response")
                        elif resp.status == 400:
                            body = await resp.text()
                            if "API_KEY_INVALID" in body:
                                _key_invalid = True
                                break
                            _test_results.append(f"❌ `{_tm}` — 400")
                        elif resp.status == 403:
                            _test_results.append(f"🚫 `{_tm}` — forbidden")
                        elif resp.status == 429:
                            body = await resp.text()
                            if "limit: 0" in body:
                                _test_results.append(f"⚠️ `{_tm}` — quota=0")
                            else:
                                _test_results.append(f"⏳ `{_tm}` — rate limited")
                                _any_ok = True
                        elif resp.status == 404:
                            _test_results.append(f"➖ `{_tm}` — not available")
                        else:
                            _test_results.append(f"❓ `{_tm}` — HTTP {resp.status}")
                except asyncio.TimeoutError:
                    _test_results.append(f"⏰ `{_tm}` — timeout")
                except Exception as e:
                    _test_results.append(f"❌ `{_tm}` — {str(e)[:50]}")
                await asyncio.sleep(0.5)

        if _key_invalid:
            result_text = "❌ *Invalid API key.* Key does not exist or was deleted."
        else:
            header = "✅ *Key works!*" if _any_ok else "⚠️ *Key valid but no model responded*"
            result_text = f"🔑 {header}\n\n" + "\n".join(_test_results)
        await send_response(app_session, chat_id, result_text, msg_id, parse_mode="Markdown")
        return

    if text.startswith("/time "):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return
        try:
            time_str = text.split(" ", 1)[1].replace(":", " ").split()
            new_h = int(time_str[0])
            new_m = int(time_str[1])
            if 0 <= new_h < 24 and 0 <= new_m < 60:
                SCAN_SCHEDULE["hour"] = new_h
                SCAN_SCHEDULE["minute"] = new_m
                SCAN_SCHEDULE["force_rescan"] = True
                _save_scan_schedule()
                msg_text = f"✅ Global scan time successfully changed to *{new_h:02d}:{new_m:02d}* (UTC+3)"
            else:
                msg_text = "⚠️ Invalid time format. Use: `/time 18:15` or `/time 18 15`"
        except Exception:
            msg_text = "⚠️ Error parsing time. Example: `/time 18 15`"
        await send_response(app_session, chat_id, msg_text, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # CUSTOM POST TO BINANCE SQUARE (/post)
    # ==========================================
    if text.startswith("/post"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return
        parts = original_text.split(maxsplit=1)
        has_text_arg = len(parts) >= 2 and parts[1].strip()
        reply_msg_obj = msg.get("reply_to_message")

        if reply_msg_obj and has_text_arg:
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
            user_text = parts[1].strip()
            pub_msg = "⏳ Publishing to Binance Square..." if lang_pref == "en" else "⏳ Публикую в Binance Square..."
            await send_response(app_session, chat_id, pub_msg, msg_id)
            result = await post_to_binance_square(user_text)
            await send_response(app_session, chat_id, result, msg_id)
        elif reply_msg_obj:
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
        return

    if text.startswith("/autopost"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return
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
            time_parts = arg.split()[1:]
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
        elif arg_lower.startswith("hashtags") or arg_lower.startswith("tags"):
            tag_parts = arg.split(maxsplit=1)
            if len(tag_parts) > 1 and tag_parts[1].strip():
                new_tags = tag_parts[1].strip()
                set_hashtags(new_tags)
                msg_text = f"✅ Hashtags updated!\n🏷 `{new_tags}`"
            else:
                current = get_hashtags()
                msg_text = f"🏷 Current hashtags: `{current}`\n\nTo change: `/autopost hashtags #tag1 #tag2 #tag3`"
        elif arg_lower in ("", "status"):
            msg_text = get_status_text()
        else:
            coin_args = arg.split()
            if len(coin_args) >= 1:
                set_coins(coin_args)
                coins_str = ", ".join(get_coins())
                msg_text = f"✅ Coins updated!\n🪙 Auto-post list: `{coins_str}`"
            else:
                msg_text = get_status_text()

        await send_response(app_session, chat_id, msg_text, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # TOP GAINERS / LOSERS
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
        return

    # ==========================================
    # LEARN MODE: /learn BTC
    # ==========================================
    if text.startswith("/learn"):
        parts = original_text.split()
        if len(parts) >= 2:
            coin_raw = parts[1].upper().strip()
            learn_symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
            short_coin = learn_symbol.replace("USDT", "")

            learn_load = f"📚 Analyzing {learn_symbol} on 4H + 1H + 15m..." if lang_pref == "en" else f"📚 Анализирую {learn_symbol} на 4Ч + 1Ч + 15м..."
            await send_response(app_session, chat_id, learn_load, msg_id)

            raw_4h = await fetch_klines(app_session, learn_symbol, "4h", 250)
            raw_1h = await fetch_klines(app_session, learn_symbol, "1h", 250)
            raw_15m = await fetch_klines(app_session, learn_symbol, "15m", 250)

            if raw_4h:
                row_4h, _ = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")
                row_1h = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0] if raw_1h else None
                row_15m = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0] if raw_15m else None
                funding = await fetch_funding_history(app_session, learn_symbol)

                def _fmt_tf_learn(row, tf_label, lang):
                    price = row.get("close", 0)
                    rsi = row.get("rsi14", 0)
                    mfi = row.get("mfi", 0)
                    adx = row.get("adx", 0)
                    stoch = row.get("stoch_k", 0)
                    macd_h = row.get("macd_hist", 0)
                    obv_val = row.get("obv", 0)
                    obv_sma = row.get("obv_sma20", 0)
                    obv = "Accumulation" if obv_val > obv_sma else "Distribution"
                    ichimoku = row.get("ichimoku_status", "Unknown")
                    st_dir = row.get("supertrend_dir", 1)
                    supertrend = "BULLISH" if st_dir == 1 else "BEARISH"
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

                await send_response(app_session, chat_id, msg1, msg_id, parse_mode="Markdown")
                await send_response(app_session, chat_id, explain, parse_mode="Markdown")
            else:
                await send_response(app_session, chat_id, f"⚠️ Pair `{learn_symbol}` not found on Binance Futures.", msg_id, parse_mode="Markdown")
        else:
            hint = "📚 Usage: `/learn BTC` — explains all indicators for any coin" if lang_pref == "en" else "📚 Использование: `/learn BTC` — объяснит все индикаторы"
            await send_response(app_session, chat_id, hint, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # SIGNALS (/signals)
    # ==========================================
    if text.startswith("/signal"):
        if not is_admin(msg):
            deny = "⛔️ Admin only" if lang_pref == "en" else "⛔️ Только для админа"
            await send_response(app_session, chat_id, deny, msg_id)
            return

        sig_parts = text.split()
        if len(sig_parts) >= 2 and sig_parts[1] in ("clear", "reset", "сброс"):
            reset_virtual_bank()
            if lang_pref == "ru":
                await send_response(app_session, chat_id,
                    "🔄 *Банк сброшен!*\n\n"
                    "💰 Баланс: `$10,000.00`\n"
                    "📊 All-time статистика обнулена\n"
                    "📋 Сегодняшние сигналы остались в списке",
                    msg_id, parse_mode="Markdown")
            else:
                await send_response(app_session, chat_id,
                    "🔄 *Bank reset!*\n\n"
                    "💰 Balance: `$10,000.00`\n"
                    "📊 All-time stats cleared\n"
                    "📋 Today's signals kept in the list",
                    msg_id, parse_mode="Markdown")
            return

        if len(sig_parts) >= 2 and sig_parts[1] in ("close", "закрыть"):
            try:
                chunks = await build_signals_close_text(app_session, lang=lang_pref)
                for i, chunk in enumerate(chunks):
                    rid = msg_id if i == 0 else None
                    await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ /signals close error: {e}")
                await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
            return

        try:
            chunks = await build_signals_text(app_session, lang=lang_pref)
            for i, chunk in enumerate(chunks):
                rid = msg_id if i == 0 else None
                await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"❌ /signals error: {e}")
            await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
        return

    # ==========================================
    # MONITOR BANK: /bankm
    # ==========================================
    if text.startswith("/bankm"):
        if not is_admin(msg):
            deny = "⛔️ Admin only" if lang_pref == "en" else "⛔️ Только для админа"
            await send_response(app_session, chat_id, deny, msg_id)
            return

        bm_parts = text.split()
        if len(bm_parts) >= 2 and bm_parts[1] in ("clear", "reset", "сброс"):
            reset_monitor_virtual_bank()
            if lang_pref == "ru":
                await send_response(app_session, chat_id,
                    "🔄 *Monitor банк сброшен!*\n\n"
                    "💰 Баланс: `$10,000.00`\n"
                    "📊 All-time статистика обнулена\n"
                    "📋 Сегодняшние сигналы остались в списке",
                    msg_id, parse_mode="Markdown")
            else:
                await send_response(app_session, chat_id,
                    "🔄 *Monitor bank reset!*\n\n"
                    "💰 Balance: `$10,000.00`\n"
                    "📊 All-time stats cleared\n"
                    "📋 Today's signals kept in the list",
                    msg_id, parse_mode="Markdown")
            return

        if len(bm_parts) >= 2 and bm_parts[1] in ("close", "закрыть"):
            try:
                chunks = await build_signals_close_text_monitor(app_session, lang=lang_pref)
                for i, chunk in enumerate(chunks):
                    rid = msg_id if i == 0 else None
                    await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ /bankm close error: {e}")
                await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
            return

        try:
            chunks = await build_signals_text_monitor(app_session, lang=lang_pref)
            for i, chunk in enumerate(chunks):
                rid = msg_id if i == 0 else None
                await send_response(app_session, chat_id, chunk, rid, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"❌ /bankm error: {e}")
            await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
        return

    # ==========================================
    # PRICE ALERTS (/alert)
    # ==========================================
    if text.startswith("/alert"):
        parts = original_text.split()
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
            return

        if len(parts) == 2 and parts[1].lower() in ("clear", "очистить"):
            alerts = load_price_alerts()
            remaining = [a for a in alerts if a["chat_id"] != chat_id]
            save_price_alerts(remaining)
            clr_msg = "✅ All alerts cleared." if lang_pref == "en" else "✅ Все алерты удалены."
            await send_response(app_session, chat_id, clr_msg, msg_id)
            return

        if len(parts) >= 3:
            coin_raw = parts[1].upper().strip()
            symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
            try:
                target_price = float(parts[2].replace(",", "."))
            except ValueError:
                err_price = "⚠️ Invalid price. Example: `/alert BTC 69500`" if lang_pref == "en" else "⚠️ Неверная цена. Пример: `/alert BTC 69500`"
                await send_response(app_session, chat_id, err_price, msg_id, parse_mode="Markdown")
                return

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
                return

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
            return

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
        return

    # ==========================================
    # /vol — show volume waitlist
    # ==========================================
    if text.startswith("/vol") or text in ["объём", "обьем", "объем"]:
        from core.signal_pipeline import get_volume_waitlist
        waitlist = get_volume_waitlist()
        if not waitlist:
            no_vol = "📭 Volume waitlist empty." if lang_pref == "en" else "📭 Список ожидания объёма пуст."
            await send_response(app_session, chat_id, no_vol, msg_id)
            return

        header = "📊 *Ожидание объёма*\n\n" if lang_pref == "ru" else "📊 *Volume Waitlist*\n\n"
        lines = []
        for vw in sorted(waitlist, key=lambda x: x.get("symbol", "")):
            sym = vw["symbol"]
            tf = vw.get("tf", "?")
            vol_12h = vw.get("vol_12h", 0)
            vol_1h = vw.get("vol_1h", 0)
            green = vw.get("candle_green", None)
            added = vw.get("added_at", "?")
            if "T" in str(added):
                added = str(added).split("T")[1][:8]

            candle_emoji = "🟢" if green else "🔴" if green is not None else "❓"
            lines.append(
                f"📉 *{sym}* ({tf})\n"
                f"   12h: `${vol_12h:,.0f}` | 1h: `${vol_1h:,.0f}` {candle_emoji}\n"
                f"   Добавлен: {added} UTC"
            )

        chunk = header
        for line in lines:
            if len(chunk) + len(line) + 2 > 3900:
                await send_response(app_session, chat_id, chunk, msg_id, parse_mode="Markdown")
                chunk = ""
            chunk += line + "\n\n"
        if chunk.strip():
            await send_response(app_session, chat_id, chunk, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # /monitor — show coins in monitor
    # ==========================================
    if text.startswith("/monitor") or text in ["монитор", "мониторинг"]:
        from core.signal_pipeline import load_monitors
        monitors = load_monitors()
        if not monitors:
            no_mon = "📭 No coins in monitor." if lang_pref == "en" else "📭 Нет монет в мониторинге."
            await send_response(app_session, chat_id, no_mon, msg_id)
            return

        mon_symbols = list(set(m["symbol"] for m in monitors))
        current_prices = {}
        try:
            async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                if resp.status == 200:
                    tickers = await resp.json()
                    price_map = {t["symbol"]: float(t["price"]) for t in tickers}
                    for s in mon_symbols:
                        if s in price_map:
                            current_prices[s] = price_map[s]
        except Exception as e:
            logging.error(f"❌ /monitor price fetch: {e}")

        header = "📊 *Мониторинг*\n\n" if lang_pref == "ru" else "📊 *Monitor*\n\n"
        lines = []
        for m in sorted(monitors, key=lambda x: x.get("symbol", "")):
            sym = m["symbol"]
            tf = m.get("tf", "?")
            direction = m.get("direction", "?").upper()
            entry = m.get("entry_price", 0)
            reason = m.get("reason", "")
            checks = m.get("check_count", 0)
            cur = current_prices.get(sym, 0)

            if entry and entry > 0 and cur > 0:
                pct = ((cur - entry) / entry) * 100
                pct_str = f"{pct:+.2f}%"
            else:
                pct_str = "—"

            emoji = "🟢" if "long" in direction.lower() else "🔴" if "short" in direction.lower() else "⚪"
            lines.append(
                f"{emoji} *{sym}* {tf} {direction}\n"
                f"   Вход: `{entry:.6f}` → Сейчас: `{cur:.6f}` ({pct_str})\n"
                f"   Причина: {reason} | Проверок: {checks}"
            )

        chunk = header
        for line in lines:
            if len(chunk) + len(line) + 2 > 3900:
                await send_response(app_session, chat_id, chunk, msg_id, parse_mode="Markdown")
                chunk = ""
            chunk += line + "\n\n"
        if chunk.strip():
            await send_response(app_session, chat_id, chunk, msg_id, parse_mode="Markdown")
        return

    # ==========================================
    # /trend — breakout list
    # ==========================================
    if text.startswith("/trend") or text in ["тренд", "тренды", "пробития"]:
        log = load_breakout_log()
        if not log:
            no_brk = "📭 No breakouts since last scan." if lang_pref == "en" else "📭 Нет пробитий с последнего скана."
            await send_response(app_session, chat_id, no_brk, msg_id)
            return

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
        return

    # ==========================================
    # PAPER TRADING: /paper
    # ==========================================
    if text.startswith("/paper"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        user_id = str(msg.get("from", {}).get("id", 0))
        parts = original_text.split()
        paper = _load_paper()
        if user_id not in paper:
            paper[user_id] = {"open": [], "closed": []}
        if isinstance(paper[user_id], list):
            paper[user_id] = {"open": paper[user_id], "closed": []}
        user_data = paper[user_id]

        # /paper clear
        if len(parts) == 2 and parts[1].lower() in ("clear", "очистить", "reset"):
            paper[user_id] = {"open": [], "closed": []}
            _save_paper(paper)
            clr = "✅ Paper portfolio reset. History cleared." if lang_pref == "en" else "✅ Портфель сброшен. История очищена."
            await send_response(app_session, chat_id, clr, msg_id)
            return

        # /paper close <n>
        if len(parts) >= 2 and parts[1].lower() in ("close", "закрыть"):
            if not user_data["open"]:
                await send_response(app_session, chat_id, "📭 No open positions." if lang_pref == "en" else "📭 Нет открытых позиций.", msg_id)
                return
            idx = 0
            if len(parts) >= 3 and parts[2].isdigit():
                idx = int(parts[2]) - 1
            if idx < 0 or idx >= len(user_data["open"]):
                await send_response(app_session, chat_id, f"⚠️ Position #{idx+1} not found. Use `/paper` to see list.", msg_id, parse_mode="Markdown")
                return

            pos = user_data["open"].pop(idx)
            sym = pos["symbol"]
            entry = pos["entry"]
            direction = pos["direction"]
            lev = pos["leverage"]
            short_sym = sym.replace("USDT", "")

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
            return

        # /paper history
        if len(parts) == 2 and parts[1].lower() in ("history", "история"):
            closed = user_data.get("closed", [])
            if not closed:
                await send_response(app_session, chat_id, "📭 No closed trades yet." if lang_pref == "en" else "📭 Нет закрытых сделок.", msg_id)
                return
            header = "📜 *Trade History*\n\n" if lang_pref == "en" else "📜 *История сделок*\n\n"
            lines = [header]
            total = 0
            wins = 0
            for i, c in enumerate(closed[-20:], 1):
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
            return

        # /paper BTC 74000 long 5x sl 73000 tp 75000
        if len(parts) >= 4:
            coin_raw = parts[1].upper().strip()
            p_symbol = coin_raw + "USDT" if not coin_raw.endswith("USDT") else coin_raw
            try:
                entry_price = float(parts[2].replace(",", "."))
            except ValueError:
                await send_response(app_session, chat_id, "⚠️ `/paper BTC 74000 long 5x sl 73000 tp 75000`", msg_id, parse_mode="Markdown")
                return

            direction = "long"
            if len(parts) >= 4 and parts[3].lower() in ("short", "шорт", "s"):
                direction = "short"

            leverage = 1
            sl_price = None
            tp_price = None
            for j, p in enumerate(parts):
                p_low = p.lower()
                p_clean = p_low.replace("x", "").replace("х", "")
                if p_clean.isdigit() and 1 < int(p_clean) <= 125:
                    leverage = int(p_clean)
                if p_low in ("sl", "стоп") and j + 1 < len(parts):
                    try:
                        sl_price = float(parts[j+1].replace(",", "."))
                    except ValueError:
                        pass
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
            return

        # /paper — show portfolio
        open_positions = user_data.get("open", [])
        if not open_positions:
            empty = "📭 No open positions.\n\n`/paper BTC 74000 long 5x sl 73000 tp 75000`" if lang_pref == "en" else "📭 Нет открытых позиций.\n\n`/paper BTC 74000 long 5x sl 73000 tp 75000`"
            await send_response(app_session, chat_id, empty, msg_id, parse_mode="Markdown")
            return

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
        return

    # ==========================================
    # CHART ANALYSIS (scan / check / посмотри …)
    # ==========================================
    analysis_prefixes = [
        "scan ", "check ", "look ", "analyze ",
        "посмотри ", "посмотри на ", "глянь ", "чекни ", "анализ "
    ]
    matched_prefix = next((p for p in analysis_prefixes if text.startswith(p)), None)

    if matched_prefix:
        symbol_raw = text.replace(matched_prefix, "").strip().split()[0].upper()
        symbol = symbol_raw + "USDT" if not symbol_raw.endswith("USDT") else symbol_raw

        fetch_msg = f"⏳ Fetching chart data + building trend line... ({symbol})" if lang_pref == "en" else f"⏳ Загружаю график + строю трендовую линию... ({symbol})"
        stream_msg_id = await send_and_get_msg_id(app_session, chat_id, fetch_msg, msg_id)

        raw_df_full = await fetch_klines(app_session, symbol, "4h", 199)
        raw_df_4h = await fetch_klines(app_session, symbol, "4h", 250)
        raw_df_1h = await fetch_klines(app_session, symbol, "1h", 250)
        raw_df_15m = await fetch_klines(app_session, symbol, "15m", 250)
        raw_df_1d = await fetch_klines(app_session, symbol, "1d", 250)

        raw_df = raw_df_4h

        if raw_df:
            df = pd.DataFrame(raw_df)
            last_row, full_df = calculate_binance_indicators(df, "4H")
            funding = await fetch_funding_history(app_session, symbol)
            last_row["funding_rate"] = funding
            positioning = await fetch_market_positioning(app_session, symbol)
            last_row["positioning"] = positioning

            mtf_data = {}
            if raw_df_1d:
                mtf_data["1D"] = calculate_binance_indicators(pd.DataFrame(raw_df_1d), "1D")[0]
            if raw_df_1h:
                mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_df_1h), "1H")[0]
            if raw_df_15m:
                mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_df_15m), "15m")[0]

            from core.smc import analyze_smc
            smc_data = {}
            try:
                # SMC needs 500 candles for proper swing structure (size=50)
                raw_smc_1d = await fetch_klines(app_session, symbol, "1d", 500) if raw_df_1d else None
                raw_smc_4h = await fetch_klines(app_session, symbol, "4h", 500) if raw_df_4h else None
                if raw_smc_1d:
                    smc_data["1D"] = analyze_smc(pd.DataFrame(raw_smc_1d), "1D")
                elif raw_df_1d:
                    smc_data["1D"] = analyze_smc(pd.DataFrame(raw_df_1d), "1D")
                if raw_smc_4h:
                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_smc_4h), "4H")
                elif raw_df_4h:
                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_df_4h), "4H")
                if raw_df_1h:
                    smc_data["1H"] = analyze_smc(pd.DataFrame(raw_df_1h), "1H")
                if raw_df_15m:
                    smc_data["15m"] = analyze_smc(pd.DataFrame(raw_df_15m), "15m")
            except Exception as e:
                logging.error(f"❌ SMC scan error: {e}")

            tg_stream = None
            if stream_msg_id:
                tg_stream = {
                    "session": app_session,
                    "chat_id": chat_id,
                    "message_id": stream_msg_id,
                    "bot_token": BOT_TOKEN
                }

            ai_msg = await ask_ai_analysis(symbol, "4H", last_row, lang=lang_pref, telegram_stream=tg_stream, extended=True, mode="extended", mtf_data=mtf_data, smc_data=smc_data)

            async def _delayed_delete(sess, cid, mid, delay=15):
                await asyncio.sleep(delay)
                try:
                    await sess.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage",
                        json={"chat_id": cid, "message_id": mid}, timeout=5
                    )
                except Exception:
                    pass

            chart_path = None
            if raw_df_full:
                df_full = pd.DataFrame(raw_df_full)
                line_data, _ = await find_trend_line(df_full, "4H", symbol)
                if line_data:
                    chart_path = await draw_scan_chart(symbol, df_full, line_data, "4H")
                else:
                    chart_path = await draw_simple_chart(symbol, df_full, "4H")

            import uuid
            import os as _os
            post_id = str(uuid.uuid4())[:8]
            short_sym = symbol.replace("USDT", "")
            square_text = f"🤖 AI-ALISA-COPILOTCLAW Analysis: ${short_sym}\n\n{ai_msg}\n\n#AIBinance #BinanceSquare #{short_sym} #Write2Earn"
            if len(square_text) > 2100:
                square_text = square_text[:2097] + "..."
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

            ai_brief = ai_msg
            ai_extended = None
            if "---" in ai_msg:
                _parts = ai_msg.split("---", 1)
                ai_brief = _parts[0].strip()
                ai_extended = _parts[1].strip() if len(_parts) > 1 and _parts[1].strip() else None

            import re as _re
            _part_hdr = _re.compile(r'^={2,}\s*PART\s*\d*.*?={2,}\s*\n?', _re.IGNORECASE | _re.MULTILINE)
            ai_brief = _part_hdr.sub('', ai_brief).strip()
            if ai_extended:
                ai_extended = _part_hdr.sub('', ai_extended).strip()

            if chart_path:
                safe_brief = ai_brief if len(ai_brief) <= 828 else ai_brief[:825] + "..."
                caption = safe_brief
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
                                await send_response(app_session, chat_id, ai_brief, msg_id, reply_markup=scan_markup)
                except Exception as e:
                    logging.error(f"❌ Error sending scan chart: {e}")
                    await send_response(app_session, chat_id, ai_brief, msg_id, reply_markup=scan_markup)
                finally:
                    try:
                        _os.remove(chart_path)
                    except Exception:
                        pass

                if ai_extended:
                    ext_text = ai_extended
                    if len(ext_text) > 4000:
                        cut = ext_text[:4000].rfind('\n')
                        ext_text = ext_text[:cut] if cut > 2000 else ext_text[:4000]
                    post_id2 = str(uuid.uuid4())[:8]
                    sq_text2 = f"🤖 AI-ALISA-COPILOTCLAW Analysis: ${short_sym}\n\n{ai_extended}\n\n#AIBinance #BinanceSquare #{short_sym} #Write2Earn"
                    if len(sq_text2) > 2100:
                        sq_text2 = sq_text2[:2097] + "..."
                    square_cache_put(post_id2, sq_text2)
                    ext_markup = {
                        "inline_keyboard": [
                            [{"text": "📢 Post to Binance Square", "callback_data": f"sq_{post_id2}"}]
                        ]
                    }
                    await send_response(app_session, chat_id, ext_text, parse_mode="Markdown", reply_markup=ext_markup)
            else:
                await send_response(app_session, chat_id, ai_brief, msg_id, reply_markup=scan_markup)
                if ai_extended:
                    ext_text = ai_extended
                    if len(ext_text) > 4000:
                        cut = ext_text[:4000].rfind('\n')
                        ext_text = ext_text[:cut] if cut > 2000 else ext_text[:4000]
                    await send_response(app_session, chat_id, ext_text, parse_mode="Markdown")

            if stream_msg_id:
                asyncio.create_task(_delayed_delete(app_session, chat_id, stream_msg_id, 15))

        return

    # ==========================================
    # RISK / MARGIN CALCULATOR (margin + leverage)
    # ==========================================
    is_margin_en = "margin" in text and "leverage" in text
    is_margin_ru = ("маржа" in text or "маржу" in text) and "плечо" in text

    if is_margin_en or is_margin_ru:
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2:
            margin = float(nums[0])
            leverage = float(nums[1])
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

            raw_4h = await fetch_klines(app_session, coin_to_analyze, "4h", 250)
            raw_1h = await fetch_klines(app_session, coin_to_analyze, "1h", 250)
            raw_15m = await fetch_klines(app_session, coin_to_analyze, "15m", 250)
            raw_1d = await fetch_klines(app_session, coin_to_analyze, "1d", 250)
            if raw_4h:
                last_row, _ = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")
                funding = await fetch_funding_history(app_session, coin_to_analyze)
                last_row["funding_rate"] = funding
                positioning = await fetch_market_positioning(app_session, coin_to_analyze)
                last_row["positioning"] = positioning
                mtf_data = {}
                if raw_1d:
                    mtf_data["1D"] = calculate_binance_indicators(pd.DataFrame(raw_1d), "1D")[0]
                if raw_1h:
                    mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                if raw_15m:
                    mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]

                smc_data = {}
                try:
                    from core.smc import analyze_smc
                    # SMC needs 500 candles for proper swing structure
                    raw_smc_1d = await fetch_klines(app_session, coin_to_analyze, "1d", 500) if raw_1d else None
                    raw_smc_4h = await fetch_klines(app_session, coin_to_analyze, "4h", 500) if raw_4h else None
                    if raw_smc_1d:
                        smc_data["1D"] = analyze_smc(pd.DataFrame(raw_smc_1d), "1D")
                    elif raw_1d:
                        smc_data["1D"] = analyze_smc(pd.DataFrame(raw_1d), "1D")
                    if raw_smc_4h:
                        smc_data["4H"] = analyze_smc(pd.DataFrame(raw_smc_4h), "4H")
                    elif raw_4h:
                        smc_data["4H"] = analyze_smc(pd.DataFrame(raw_4h), "4H")
                    if raw_1h:
                        smc_data["1H"] = analyze_smc(pd.DataFrame(raw_1h), "1H")
                    if raw_15m:
                        smc_data["15m"] = analyze_smc(pd.DataFrame(raw_15m), "15m")
                except Exception as e:
                    logging.error(f"❌ SMC look error: {e}")

                ai_msg = await ask_ai_analysis(coin_to_analyze, "4H", last_row, user_margin=margin_data, lang=lang_pref, mode="scan", mtf_data=mtf_data, smc_data=smc_data)
                await send_response(app_session, chat_id, ai_msg, msg_id)
