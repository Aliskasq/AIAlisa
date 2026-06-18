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
    load_manual_alerts, save_manual_alerts,
    get_user_tz_offset, set_user_tz_offset,
    load_trend_above_pct, save_trend_above_pct,
    load_line_4h_settings, save_line_4h_settings,
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
from core.chart_drawer import draw_scan_chart, draw_simple_chart, draw_alert_chart

from core.tg_state import (
    send_response, send_and_get_msg_id, get_chat_lang, set_chat_lang,
    is_allowed_chat, is_admin, ADMIN_ID,

    square_cache_put,
    SCAN_SCHEDULE, _save_scan_schedule,
    _fetch_or_free_models,
    get_manual_alert_state, set_manual_alert_state, clear_manual_alert_state,
    track_alert_msg, get_tracked_alert_msgs, clear_tracked_alert_msgs,
    schedule_alert_cleanup,
    get_model_menu_state, set_model_menu_state, clear_model_menu_state,
    build_model_category_kb,
)
from core.tg_reports import (
    build_signals_text, build_signals_close_text,
)


def _build_alert_caption(short_sym, tf, current_price, all_alerts_for_sym_tf, chat_id, view_limit=199):
    """Build unified caption showing ALL lines for a symbol+tf.
    all_alerts_for_sym_tf: list of alert dicts from manual_alerts.json"""
    import math as _m
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    _color_emojis = ['⚫', '🔵', '🟠', '🟤', '🟢', '🔴']
    tz_off = get_user_tz_offset(chat_id)

    lines_text = []
    for a in all_alerts_for_sym_tf:
        ci = a.get('color_idx', 0)
        ce = _color_emojis[ci % len(_color_emojis)]

        # Format A/B times in user's TZ
        tf_ms_map = {"15m": 900000, "1H": 3600000, "4H": 14400000, "1D": 86400000}
        a_tf_ms = a.get('tf_ms', tf_ms_map.get(tf, 14400000))
        base_time = a.get('base_open_time', 0)
        base_idx = a.get('base_idx', view_limit - 1)
        idx_a = a.get('index_a', 0)
        idx_b = a.get('index_b', 0)

        # Calculate candle times from indices
        time_a_ms = base_time - (base_idx - idx_a) * a_tf_ms
        time_b_ms = base_time - (base_idx - idx_b) * a_tf_ms
        dt_a = _dt.fromtimestamp(time_a_ms / 1000, tz=_tz.utc) + _td(hours=tz_off)
        dt_b = _dt.fromtimestamp(time_b_ms / 1000, tz=_tz.utc) + _td(hours=tz_off)
        time_a_str = dt_a.strftime("%d.%m %H:%M")
        time_b_str = dt_b.strftime("%d.%m %H:%M")

        price_a = a.get('price_a', 0)
        price_b = a.get('price_b', 0)

        # Line price now
        slope = a.get('slope', 0)
        intercept = a.get('intercept', 0)
        # Recalculate current math_x
        from datetime import datetime as _dt2
        now_ms = int(_dt2.now(_tz.utc).timestamp() * 1000)
        candles_passed = (now_ms - base_time) / a_tf_ms if a_tf_ms else 0
        current_x = base_idx + candles_passed
        line_now = _m.exp(slope * current_x + intercept) if slope != 0 or intercept != 0 else 0

        # % to line: how much price needs to move
        if line_now > 0 and current_price > 0:
            if current_price > line_now:
                # Line below price: how much % price must FALL (max ~100%)
                pct_to = ((current_price - line_now) / current_price) * 100
                pct_str = f"↓ {pct_to:.2f}%"
            else:
                # Line above price: how much % price must RISE (no limit)
                pct_to = ((line_now - current_price) / current_price) * 100
                pct_str = f"↑ {pct_to:.2f}%"
        else:
            pct_str = "—"

        lines_text.append(
            f"{ce}🅰️ `{price_a:.8g}` ({time_a_str})\n"
            f"{ce}🅱️ `{price_b:.8g}` ({time_b_str})\n"
            f"Линия: `{line_now:.8g}` | до линии: {pct_str}"
        )

    caption = f"📐 *${short_sym}* ({tf})\n💰 Цена: `{current_price:.8g}`\n\n"
    caption += "\n\n".join(lines_text)
    caption += f"\n\n📐 Всего линий: {len(all_alerts_for_sym_tf)}"
    return caption


def _parse_flexible_datetime(text, chat_id):
    """Parse date+time from flexible input formats.
    Accepts: '10.06.26 04:45', '10,06,26,04,45', '10.06.26.04.45',
             '10 06 26 04 45', '10,06.26 12:00', '10.06 04:45' (current year).
    Returns datetime in UTC (applies user timezone offset) or None."""
    import re as _re
    raw = text.strip()
    # Normalize: replace commas/dots/dashes/colons/spaces with single separator
    # First, keep the structure: extract all digit groups
    nums = _re.findall(r'\d+', raw)
    if len(nums) == 5:
        # DD MM YY HH MM
        day, month, year, hour, minute = [int(x) for x in nums]
    elif len(nums) == 4:
        # DD MM HH MM (no year → current year)
        day, month, hour, minute = [int(x) for x in nums]
        year = datetime.now(timezone.utc).year
    elif len(nums) == 3:
        # DD MM YY (no time → 00:00) — unlikely but handle
        day, month, year = [int(x) for x in nums]
        hour, minute = 0, 0
    else:
        return None
    try:
        if year < 100:
            year += 2000
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        # Apply timezone offset: user enters in their TZ, convert to UTC
        tz_off = get_user_tz_offset(chat_id)
        dt = dt - timedelta(hours=tz_off)
        return dt
    except (ValueError, OverflowError):
        return None


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

                "🏆 `/signals` — _winrate (admin)_\n"
                "💰 `margin 100 leverage 10` — _stop-loss calc_\n"
                "🛠 `/skills` — _Web3 Skills_\n"
                "📈 `/top gainers` · 📉 `/top losers`\n"
                "📊 `/trend` — _breakouts / пробития_\n"
                "🔔 `/alert BTC 69500` — _price alert_\n"
                "📐 `алерт BTC` — _ручная линия / trendline_\n"
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

    # DEBUG: log every incoming text message
    if "алерт" in text or "alert" in text:
        logging.info(f"🔍 DEBUG incoming alert-related: text={repr(text[:80])}, chat={chat_id}")

    # --- CHAT FILTER ---
    if not is_allowed_chat(chat_id):
        return

    # LANGUAGE: saved preference OR auto-detect from text
    saved_lang = get_chat_lang(chat_id)
    has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
    lang_pref = "ru" if has_cyrillic else saved_lang

    # ==========================================
    # LINE 4H SETTINGS INPUT (awaiting % from user)
    # ==========================================
    from core.tg_state import get_line4h_input_state, clear_line4h_input_state
    _l4h_state = get_line4h_input_state(chat_id)
    if _l4h_state:
        try:
            val = float(text.replace("%", "").replace(",", "."))
            if val < 0 or val > 100:
                await send_response(app_session, chat_id, "❌ Допустимый диапазон: 0–100%", msg_id)
                clear_line4h_input_state(chat_id)
                return

            s = load_line_4h_settings()
            if _l4h_state == "awaiting_range_pct":
                s["range_pct"] = val
                s["mode"] = "custom"
                save_line_4h_settings(s)
                await send_response(app_session, chat_id,
                    f"✅ Допуск поиска линий: `{val}%`", msg_id, parse_mode="Markdown")
            elif _l4h_state == "awaiting_point_b_pct":
                s["point_b_pct"] = val
                s["point_b_rule"] = "narrow_pct"
                s["mode"] = "custom"
                save_line_4h_settings(s)
                await send_response(app_session, chat_id,
                    f"✅ Допуск точки Б: `{val}%`", msg_id, parse_mode="Markdown")
        except ValueError:
            await send_response(app_session, chat_id,
                "❌ Введите число, например: `15` или `23.55`", msg_id, parse_mode="Markdown")
        clear_line4h_input_state(chat_id)
        return

    # ==========================================
    # MANUAL ALERT STATE MACHINE (multi-step flow)
    # ==========================================
    ma_state = get_manual_alert_state(chat_id)
    if ma_state:
        # --- AWAITING PRICES: user enters "69500 67200" or "0.215+2% 0.240-1.5%" ---
        if ma_state.get('step') == 'awaiting_prices':
            track_alert_msg(chat_id, msg_id)  # track user's price message
            import re as _re_price
            raw_text = original_text.replace(",", ".").strip()
            parts_raw = raw_text.split()
            try:
                prices = []       # base prices (for candle matching)
                pct_offsets = []   # percent offsets (applied to trendline AFTER matching)
                for p in parts_raw:
                    # Parse price with optional percent: "0.215+2%" or "0.215-1.33%"
                    pct_match = _re_price.match(r'^([\d.]+)([+-][\d.]+)%$', p)
                    if pct_match:
                        base_price = float(pct_match.group(1))
                        pct = float(pct_match.group(2))
                        prices.append(base_price)
                        pct_offsets.append(pct)
                    elif p.replace(".", "", 1).isdigit():
                        prices.append(float(p))
                        pct_offsets.append(0)
                if len(prices) < 2:
                    raise ValueError("need 2 prices")
            except (ValueError, IndexError):
                bot_msg = await send_response(app_session, chat_id,
                    "⚠️ Введи две цены через пробел, например:\n"
                    "`69500 67200`\n"
                    "Или с процентом: `0.215+2% 0.240-1.5%`",
                    msg_id, parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
                return
            ma_state['prices'] = prices
            ma_state['pct_offsets'] = pct_offsets
            # Mode was already selected before prices — process now
            mode = ma_state.get('mode', 'high')
            await _finalize_manual_alert(app_session, chat_id, msg_id, ma_state, mode)
            return

        # --- AWAITING DATE A: user enters date+time for point A ---
        if ma_state.get('step') == 'awaiting_date_a':
            track_alert_msg(chat_id, msg_id)  # track user's date A message
            dt_a = _parse_flexible_datetime(original_text, chat_id)
            if dt_a is None:
                bot_msg = await send_response(app_session, chat_id,
                    "⚠️ Не понял дату. Примеры:\n"
                    "`10.06.26 04:45` или `10 06 26 04 45`\n"
                    "или `10,06,26,04,45` или `10.06.26.04.45`",
                    msg_id, parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
                return
            ma_state['dates'] = [dt_a.isoformat()]
            ma_state['step'] = 'awaiting_date_b'
            set_manual_alert_state(chat_id, ma_state)
            tz_off = get_user_tz_offset(chat_id)
            tz_label = f"UTC{tz_off:+d}" if tz_off else "UTC"
            bot_msg = await send_response(app_session, chat_id,
                f"✅ Точка A: `{dt_a.strftime('%d.%m.%y %H:%M')}` ({tz_label}→UTC)\n\n"
                f"📍 Точка B — введи дату и время:",
                msg_id, parse_mode="Markdown")
            track_alert_msg(chat_id, bot_msg)
            return

        # --- AWAITING DATE B: user enters date+time for point B ---
        if ma_state.get('step') == 'awaiting_date_b':
            track_alert_msg(chat_id, msg_id)  # track user's date B message
            dt_b = _parse_flexible_datetime(original_text, chat_id)
            if dt_b is None:
                bot_msg = await send_response(app_session, chat_id,
                    "⚠️ Не понял дату. Примеры:\n"
                    "`10.06.26 04:45` или `10 06 26 04 45`",
                    msg_id, parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
                return
            ma_state['dates'].append(dt_b.isoformat())
            ma_state['step'] = 'processing_dates'
            set_manual_alert_state(chat_id, ma_state)

            # Process the date-based alert
            await _process_manual_alert_dates(app_session, chat_id, msg_id, ma_state)
            return

        # --- LEGACY: old single-line format still works ---
        if ma_state.get('step') == 'awaiting_dates':
            track_alert_msg(chat_id, msg_id)  # track user's date message
            date_text = original_text.strip()
            date_text = date_text.replace('–', '-').replace('—', '-').replace('−', '-')
            import re as _re_date
            m = _re_date.match(
                r'(\d{1,2}\.\d{1,2}\.\d{2,4})\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}\.\d{1,2}\.\d{2,4})\s+(\d{1,2}:\d{2})',
                date_text
            )
            if not m:
                logging.warning(f"⚠️ Manual alert date parse FAIL: {repr(date_text)}")
                bot_msg = await send_response(app_session, chat_id,
                    "⚠️ Формат: `15.05.26 18:15-16.05.26 21:45`", msg_id, parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
                return
            try:
                tz_off = get_user_tz_offset(chat_id)
                def _parse_dt(d, t):
                    parts_d = d.split(".")
                    day, month = int(parts_d[0]), int(parts_d[1])
                    year = int(parts_d[2])
                    if year < 100:
                        year += 2000
                    hour, minute = [int(x) for x in t.split(":")]
                    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                    return dt - timedelta(hours=tz_off)  # convert user TZ → UTC
                dt_a = _parse_dt(m.group(1), m.group(2))
                dt_b = _parse_dt(m.group(3), m.group(4))
            except (ValueError, IndexError):
                await send_response(app_session, chat_id,
                    "⚠️ Не удалось распознать даты. Формат: `15.05.26 18:15-16.05.26 21:45`",
                    msg_id, parse_mode="Markdown")
                return

            ma_state['dates'] = [dt_a.isoformat(), dt_b.isoformat()]
            ma_state['step'] = 'processing_dates'
            set_manual_alert_state(chat_id, ma_state)
            await _process_manual_alert_dates(app_session, chat_id, msg_id, ma_state)
            return

    # ==========================================
    # MANUAL TRENDLINE ALERT (алерт <COIN> / alert <COIN>)
    # ==========================================
    _malert_triggers = ["алерт ", "alert "]
    _malert_match = next((t_p for t_p in _malert_triggers if text.startswith(t_p)), None)
    logging.info(f"🔍 DEBUG malert: text={repr(text[:50])}, match={repr(_malert_match)}, ma_state={get_manual_alert_state(chat_id)}, l4h={get_line4h_input_state(chat_id)}")
    if _malert_match:
        remaining_ma = text[len(_malert_match):].strip()
        logging.info(f"🔍 DEBUG malert remaining: {repr(remaining_ma)}")

        # Sub-command: timezone
        if remaining_ma in ("пояс", "timezone", "tz", "часовой пояс"):
            current_tz = get_user_tz_offset(chat_id)
            tz_buttons = []
            for off in range(-2, 13):
                label = f"UTC{off:+d}" if off != 0 else "UTC±0"
                if off == current_tz:
                    label = f"✅ {label}"
                tz_buttons.append({"text": label, "callback_data": f"malert_tz_{off}"})
            # 3 buttons per row
            rows = [tz_buttons[i:i+3] for i in range(0, len(tz_buttons), 3)]
            await send_response(app_session, chat_id,
                f"🕐 *Часовой пояс для алертов*\n\n"
                f"Сейчас: *UTC{current_tz:+d}*\n"
                f"Выбери свой пояс (МСК = UTC+3):",
                msg_id, reply_markup={"inline_keyboard": rows}, parse_mode="Markdown")
            return

        # Sub-commands: list / clear
        if remaining_ma in ("список", "list"):
            try:
                alerts = load_manual_alerts()
                logging.info(f"🔍 DEBUG alert list: loaded {len(alerts)} total alerts, looking for chat_id={chat_id} (type={type(chat_id).__name__})")
                if alerts:
                    logging.info(f"🔍 DEBUG alert list: first alert chat_id={alerts[0].get('chat_id')} (type={type(alerts[0].get('chat_id')).__name__})")
                user_alerts = [(i, a) for i, a in enumerate(alerts) if a.get("chat_id") == chat_id]
                logging.info(f"🔍 DEBUG alert list: found {len(user_alerts)} user alerts")
                if not user_alerts:
                    await send_response(app_session, chat_id,
                        "📭 Нет активных ручных линий.\n\nИспользуй: `алерт BTC`",
                        msg_id, parse_mode="Markdown")
                else:
                    lines_txt = ["📐 *Ручные алерт-линии:*\n"]
                    delete_buttons = []
                    for num, (global_idx, a) in enumerate(user_alerts, 1):
                        short = a["symbol"].replace("USDT", "")
                        mode_display = a.get('mode', '?').replace('_', ' ')
                        # Created timestamp
                        created = a.get('created_at', '')
                        date_str = ""
                        if created:
                            try:
                                from datetime import datetime as _dt
                                ct = _dt.fromisoformat(created.replace('Z', '+00:00'))
                                tz_off = get_user_tz_offset(chat_id)
                                ct = ct + timedelta(hours=tz_off)
                                date_str = f"\n     📅 {ct.strftime('%d.%m.%y %H:%M')}"
                            except Exception:
                                pass
                        lines_txt.append(
                            f"{num}. `${short}` {a['tf']} — A=`{a['price_a']:.8g}` B=`{a['price_b']:.8g}` ({mode_display}){date_str}"
                        )
                        delete_buttons.append(
                            {"text": f"❌ {num}", "callback_data": f"malert_del_{global_idx}"}
                        )
                    # Group buttons in rows of 4
                    button_rows = [delete_buttons[i:i+4] for i in range(0, len(delete_buttons), 4)]
                    kb = {"inline_keyboard": button_rows} if button_rows else None
                    await send_response(app_session, chat_id, "\n".join(lines_txt), msg_id,
                                        reply_markup=kb, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ DEBUG alert list EXCEPTION: {repr(e)}")
                import traceback
                logging.error(traceback.format_exc())
            return

        if remaining_ma.startswith("удалить") or remaining_ma.startswith("delete"):
            # Parse coins after command: "удалить BTC, ETH" or "удалить SOL"
            del_cmd_word = "удалить" if remaining_ma.startswith("удалить") else "delete"
            del_coins_raw = remaining_ma[len(del_cmd_word):].strip()
            alerts = load_manual_alerts()

            if del_coins_raw:
                # Delete specific coins
                coins = [c.strip().upper().replace("USDT", "") for c in del_coins_raw.replace(",", " ").split() if c.strip()]
                symbols = [c + "USDT" for c in coins]
                to_delete = [a for a in alerts if a.get("chat_id") == chat_id and a.get("symbol") in symbols]
                if not to_delete:
                    coins_str = ", ".join(coins)
                    await send_response(app_session, chat_id,
                        f"📭 Нет линий для: {coins_str}", msg_id)
                    return
                remaining_alerts = [a for a in alerts if not (a.get("chat_id") == chat_id and a.get("symbol") in symbols)]
                save_manual_alerts(remaining_alerts)
                deleted_summary = {}
                for a in to_delete:
                    short = a["symbol"].replace("USDT", "")
                    deleted_summary[short] = deleted_summary.get(short, 0) + 1
                summary = ", ".join(f"{coin}: {cnt}" for coin, cnt in deleted_summary.items())
                await send_response(app_session, chat_id, f"✅ Удалено: {summary}", msg_id)
            else:
                # No coins specified — show help
                await send_response(app_session, chat_id,
                    "📐 *Удаление линий:*\n\n"
                    "По монете: `алерт удалить BTC`\n"
                    "Несколько: `алерт удалить BTC, ETH`\n"
                    "Все сразу: `алерт очистить`",
                    msg_id, parse_mode="Markdown")
            return

        if remaining_ma in ("очистить", "clear"):
            alerts = load_manual_alerts()
            user_count = sum(1 for a in alerts if a.get("chat_id") == chat_id)
            if user_count == 0:
                await send_response(app_session, chat_id, "📭 Список пуст.", msg_id)
                return
            remaining_alerts = [a for a in alerts if a.get("chat_id") != chat_id]
            save_manual_alerts(remaining_alerts)
            await send_response(app_session, chat_id, f"✅ Все линии удалены ({user_count} шт.)", msg_id)
            return

        # Sub-command: view chart (алерт просмотр BTC [tf])
        if remaining_ma.startswith("просмотр") or remaining_ma.startswith("view"):
            view_parts = remaining_ma.split()
            if len(view_parts) < 2:
                await send_response(app_session, chat_id,
                    "⚠️ Укажи монету: `алерт просмотр BTC`\nС таймфреймом: `алерт просмотр BTC 4H`",
                    msg_id, parse_mode="Markdown")
                return
            view_coin = view_parts[1].upper().replace("USDT", "").strip()
            view_symbol = view_coin + "USDT"
            view_tf_input = view_parts[2].upper() if len(view_parts) >= 3 else None

            alerts = load_manual_alerts()
            user_sym_alerts = [a for a in alerts if a.get("chat_id") == chat_id and a.get("symbol") == view_symbol]
            if not user_sym_alerts:
                await send_response(app_session, chat_id,
                    f"📭 Нет ручных линий для `${view_coin}`.", msg_id, parse_mode="Markdown")
                return

            # Determine which timeframes have alerts
            available_tfs = sorted(set(a['tf'] for a in user_sym_alerts))
            if view_tf_input:
                # Normalize: "4h" → "4H", "15m" → "15m" etc.
                tf_norm = {"15M": "15m", "1H": "1H", "4H": "4H", "1D": "1D"}
                view_tf = tf_norm.get(view_tf_input, view_tf_input)
                tfs_to_show = [view_tf] if view_tf in available_tfs else []
                if not tfs_to_show:
                    await send_response(app_session, chat_id,
                        f"📭 Нет линий для `${view_coin}` на `{view_tf_input}`.\nДоступные: {', '.join(available_tfs)}",
                        msg_id, parse_mode="Markdown")
                    return
            else:
                tfs_to_show = available_tfs

            tf_map = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
            for tf in tfs_to_show:
                binance_interval = tf_map.get(tf, "4h")
                try:
                    df = await fetch_klines(app_session, view_symbol, binance_interval, limit=199)
                    if df is None or df.empty:
                        await send_response(app_session, chat_id,
                            f"⚠️ Не удалось получить данные для `{view_symbol}` {tf}.", msg_id, parse_mode="Markdown")
                        continue

                    tf_alerts = [a for a in user_sym_alerts if a['tf'] == tf]
                    all_lines_for_chart = [
                        {'price_a': a['price_a'], 'price_b': a['price_b'],
                         'index_a': a['index_a'], 'index_b': a['index_b'],
                         'base_open_time': a.get('base_open_time', 0), 'base_idx': a.get('base_idx', 0),
                         'tf_ms': a.get('tf_ms', 0), 'color_idx': a.get('color_idx', 0)}
                        for a in tf_alerts
                    ]

                    # SMC overlay
                    _alert_smc = None
                    try:
                        from core.smc import analyze_smc
                        _alert_smc = analyze_smc(df, tf, symbol=view_symbol)
                    except Exception as _smc_e:
                        logging.error(f"❌ SMC error for view chart: {repr(_smc_e)}")

                    chart_path = await draw_alert_chart(view_symbol, df, all_lines_for_chart, tf, smc_overlay=_alert_smc)

                    # Caption
                    current_price = float(df['close'].iloc[-1])
                    all_alerts_sym_tf = [a for a in alerts if a['symbol'] == view_symbol and a['tf'] == tf]
                    caption = _build_alert_caption(view_coin, tf, current_price, all_alerts_sym_tf, chat_id)

                    if chart_path:
                        photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                        try:
                            with open(chart_path, 'rb') as f:
                                import aiohttp as _aio
                                data = _aio.FormData()
                                data.add_field('chat_id', str(chat_id))
                                data.add_field('caption', caption)
                                data.add_field('parse_mode', 'Markdown')
                                data.add_field('photo', f, filename=f"{view_symbol}_view.png", content_type='image/png')
                                async with app_session.post(photo_url, data=data, timeout=30) as resp:
                                    if resp.status != 200:
                                        resp_text = await resp.text()
                                        logging.error(f"❌ View chart send error: {resp.status} - {resp_text}")
                                        await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
                        except Exception as e:
                            logging.error(f"❌ View chart send error: {repr(e)}")
                            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
                    else:
                        await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
                except Exception as e:
                    logging.error(f"❌ View chart error for {view_symbol} {tf}: {repr(e)}")
                    await send_response(app_session, chat_id,
                        f"⚠️ Ошибка построения графика `${view_coin}` {tf}.", msg_id, parse_mode="Markdown")
            return

        # Start new manual alert: parse coin
        coin_raw = remaining_ma.upper().replace("USDT", "").strip()
        if not coin_raw:
            await send_response(app_session, chat_id,
                "📐 *Ручные алерт-линии:*\n\n"
                "Создать: `алерт BTC`\n"
                "Список: `алерт список`\n"
                "Просмотр: `алерт просмотр BTC`\n"
                "Удалить: `алерт удалить BTC`\n"
                "Удалить все: `алерт очистить`\n"
                "Часовой пояс: `алерт пояс`",
                msg_id, parse_mode="Markdown")
            return

        symbol_ma = coin_raw + "USDT"
        # Verify symbol exists
        try:
            async with app_session.get(
                f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol_ma}", timeout=5
            ) as resp:
                if resp.status != 200:
                    await send_response(app_session, chat_id,
                        f"⚠️ Пара `{symbol_ma}` не найдена на Binance Futures.", msg_id, parse_mode="Markdown")
                    return
        except Exception:
            await send_response(app_session, chat_id, "⚠️ Ошибка проверки символа.", msg_id)
            return

        # Show timeframe selection
        clear_tracked_alert_msgs(chat_id)  # start fresh tracking
        track_alert_msg(chat_id, msg_id)   # track user's "алерт btc" message
        set_manual_alert_state(chat_id, {'step': 'awaiting_tf', 'symbol': symbol_ma})
        tf_kb = {"inline_keyboard": [
            [{"text": "15m", "callback_data": "malert_tf_15m"},
             {"text": "1H", "callback_data": "malert_tf_1h"},
             {"text": "4H", "callback_data": "malert_tf_4h"},
             {"text": "1D", "callback_data": "malert_tf_1d"}],
        ]}
        short_sym = symbol_ma.replace("USDT", "")
        bot_msg = await send_response(app_session, chat_id,
            f"📐 Ручная линия для *${short_sym}*\n\nВыбери таймфрейм:",
            msg_id, reply_markup=tf_kb, parse_mode="Markdown")
        track_alert_msg(chat_id, bot_msg)
        return

    # ==========================================
    # LANGUAGE SWITCH: /lang en | /lang ru
    # ==========================================
    if text.startswith("/lang"):
        parts = text.split()
        if len(parts) >= 2 and parts[1] in ("en", "ru"):
            new_lang = parts[1]
            set_chat_lang(chat_id, new_lang)
            from config import GROUP_CHAT_ID as _gcid
            if _gcid and str(chat_id) != str(_gcid):
                set_chat_lang(_gcid, new_lang)
            if new_lang == "en":
                await send_response(app_session, chat_id, "🌐 Language set to *English* 🇬🇧", msg_id, parse_mode="Markdown")
            else:
                await send_response(app_session, chat_id, "🌐 Язык: *Русский* 🇷🇺", msg_id, parse_mode="Markdown")
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
            main_masked = f"...{main_key[-8:]}" if len(main_key) > 8 else ("✅ set" if main_key else "❌ not set")
            await send_response(app_session, chat_id,
                f"🔑 *API Keys:*\n\n"
                f"🧠 Main: `{main_masked}`\n\n"
                f"Сменить: `/key <new-key>`",
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
    # MODEL MENU TEXT INPUT (awaiting OpenRouter $$$ model-id)
    # ==========================================
    from core.tg_state import get_model_menu_state, set_model_menu_state, clear_model_menu_state
    _mm_state = get_model_menu_state(chat_id)
    if _mm_state and _mm_state.get("awaiting_text"):
        if text.startswith("/"):
            clear_model_menu_state(chat_id)
            # fall through to normal command processing
        else:
            mode = _mm_state["mode"]
            new_model = text.strip()
            from config import load_ai_settings, save_ai_settings
            from agent.analyzer import set_active_provider, get_active_provider_info
            s = load_ai_settings()

            if mode == "model":
                set_active_provider("openrouter", model=new_model)
                _cfg.OPENROUTER_MODEL = new_model
                agent.analyzer.OPENROUTER_MODEL = new_model
                clear_model_menu_state(chat_id)
                await send_response(app_session, chat_id,
                    f"✅ Модель: *OpenRouter* `{new_model}`", msg_id, parse_mode="Markdown")
            elif mode == "startday":
                s["daily_reset_provider"] = "openrouter"
                s["daily_reset_model"] = new_model
                s["daily_reset_key_index"] = 0
                save_ai_settings(s)
                clear_model_menu_state(chat_id)
                await send_response(app_session, chat_id,
                    f"✅ Старт дня → *OpenRouter* `{new_model}`", msg_id, parse_mode="Markdown")
            elif mode == "fallback":
                chain = _mm_state.get("fallback_chain", [])
                chain.append(new_model)
                _mm_state["fallback_chain"] = chain
                _mm_state["awaiting_text"] = False
                set_model_menu_state(chat_id, _mm_state)
                chain_text = " → ".join([f"`{m}`" for m in chain])
                await send_response(app_session, chat_id,
                    f"➕ Добавлено: `{new_model}`\nЦепочка: {chain_text}\n\nВыбирай следующую или нажми ✅ Готово",
                    msg_id, parse_mode="Markdown",
                    reply_markup=build_model_category_kb("fallback", chain))
            return

    # ==========================================
    # UNIFIED AI MODEL COMMAND (/model)
    # ==========================================
    if text.startswith("/models") or text.startswith("/model"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        parts = text.split(maxsplit=1)
        sub_cmd = parts[1].strip() if len(parts) > 1 else ""

        # /models all — full OpenRouter list (kept for copy-paste)
        if sub_cmd.lower() == "all":
            await send_response(app_session, chat_id, "⏳ Загружаю все модели OpenRouter...", msg_id)
            try:
                async with app_session.get("https://openrouter.ai/api/v1/models", timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        models.sort(key=lambda m: m.get("id", ""))
                        lines = [f"🧠 *Все модели OpenRouter ({len(models)}):*\n"]
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
                            "💡 Скопируй model-id и вставь через кнопку 💰 OpenRouter $$$")
                    else:
                        await send_response(app_session, chat_id, f"❌ OpenRouter API error: {resp.status}", msg_id)
            except Exception as e:
                await send_response(app_session, chat_id, f"❌ Error: {e}", msg_id)
            return

        # /model — main settings menu
        from agent.analyzer import get_active_provider_info
        from config import load_ai_settings
        s = load_ai_settings()
        prov, mdl, ki = get_active_provider_info()

        # Current model display
        if prov == "disabled":
            model_line = "🚫 AI отключён"
        elif prov == "gemini":
            model_line = f"💎 Gemini #{ki+1} / `{mdl}`"
        elif prov == "groq":
            model_line = f"⚡ Groq #{ki+1} / `{mdl}`"
        else:
            model_line = f"🌐 OpenRouter / `{mdl}`"

        # Fallback display
        chain = s.get("openrouter_fallback_chain", ["openai/gpt-oss-120b:free", "openrouter/free"])
        if s.get("fallback_disabled"):
            fb_line = "🚫 Без фоллбэка"
        elif chain:
            fb_line = " → ".join([f"`{m}`" for m in chain])
        else:
            fb_line = "🚫 Без фоллбэка"

        # Start of day display
        rp = s.get("daily_reset_provider", "gemini")
        rm = s.get("daily_reset_model", "")
        rk = s.get("daily_reset_key_index", 0)
        if s.get("daily_reset_disabled"):
            sd_line = "🚫 Без сброса"
        elif rp == "gemini":
            sd_line = f"💎 Gemini #{rk+1} / `{rm or s.get('gemini_model', 'gemini-2.5-flash')}`"
        elif rp == "groq":
            sd_line = f"⚡ Groq #{rk+1} / `{rm or s.get('groq_model', 'llama-3.3-70b-versatile')}`"
        else:
            sd_line = f"🌐 OpenRouter / `{rm or s.get('openrouter_model', 'openrouter/free')}`"

        menu_text = (
            f"🧠 *AI Settings*\n\n"
            f"*Текущая модель:* {model_line}\n"
            f"*Фоллбэк:* {fb_line}\n"
            f"*Старт дня (00:00 UTC):* {sd_line}"
        )
        menu_kb = {"inline_keyboard": [
            [{"text": "🔄 Изменить модель", "callback_data": "mdm_model"},
             {"text": "🔗 Изменить фоллбэк", "callback_data": "mdm_fallback"}],
            [{"text": "🌅 Изменить старт дня", "callback_data": "mdm_startday"},
             {"text": "📋 Все модели", "callback_data": "mdm_all"}],
        ]}
        await send_response(app_session, chat_id, menu_text, msg_id, reply_markup=menu_kb, parse_mode="Markdown")
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

            "💰 `margin 100 leverage 10 max 20%`\n"
            "    _Stop-loss calculator / Расчёт стоп-лосса_\n\n"
            "🛠 `/skills`\n"
            "    _Web3 Skills menu / Меню Web3 навыков_\n\n"
            "📈 `/top gainers` · 📉 `/top losers`\n"
            "    _Top 10 growth/drops 24h / Топ 10 рост/падение_\n\n"
            "📊 `/trend`\n"
            "    _All breakouts since scan / Все пробития_\n\n"

            "📊 `/vol`\n"
            "    _Volume waitlist / Ожидание объёма_\n\n"
            "🔔 `/alert BTC 69500`\n"
            "    _Price alert / Алерт на цену_\n"
            "🔔 `/alert list` — _active / активные_\n"
            "🔔 `/alert clear` — _remove all / удалить все_\n\n"
            "📐 `алерт BTC` / `alert BTC`\n"
            "    _Ручная линия + алерт касания_\n"
            "📐 `алерт список` — _активные линии_\n"
            "📐 `алерт просмотр BTC` — _график с линиями_\n"
            "📐 `алерт удалить BTC` — _удалить линии монеты_\n"
            "📐 `алерт очистить` — _удалить все линии_\n"
            "📐 `алерт пояс` — _часовой пояс (МСК и др.)_\n\n"
            "🌐 `/lang en` — English\n"
            "🌐 `/lang ru` — Русский"
        )
        if is_admin(msg):
            welcome_text += (
                "\n\n🔐 *Admin:*\n"
                "🏆 `/signals` — signal winrate & bank\n"
                "🔒 `/signals close` — snapshot: close all open now\n"
                "🔄 `/signals clear` — reset bank to $10k\n"
                "📐 `порог 5%` — порог пробития трендлайна\n"
                "📐 `расширенные настройки линии 4ч`\n"
                "⚙️ `/stoploss` — режим SL: StopAI / Trail\n"
                "📐 `/smc` — настройки SMC: режим, OB блоки\n"

                "🧠 `/model` — AI модель, фоллбэк, старт дня\n"

                "⏰ `/time 18:30` — scan schedule\n"
                "📢 `/autopost on/off` — auto Square\n"
                "🪙 `/autopost SOL BTC` — coins\n"
                "⏰ `/autopost time 09:00 15:00 21:00` — post times\n"
                "🏷 `/autopost hashtags #tag1 #tag2` — hashtags\n"
                "✏️ `/post text` — post to Square\n"
                "✏️ reply `/post text` — AI + your opinion\n"

                "🔑 `/testapi AIzaSy...` — test Gemini key\n"
                "🔑 `/testall` — test all AI keys\n\n"

                "📊 *Индикаторы на графике:*\n"
                "🟢 OBV — кумулятивный объём\n"
                "🔴 SMA пунктир — средняя OBV"
                " (10 для 15m/1h, 20 для 4h/1d)\n"
                "🔵 CVD — дельта покупок/продаж\n"
                "🟣 Цена — масштабирована под OBV\n\n"
                "OBV > SMA = накопление 🟢\n"
                "OBV < SMA = распределение 🔴\n"
                "Цена↑ OBV↓ = медвежья дивергенция ⚠️\n"
                "Цена↓ OBV↑ = бычья дивергенция ⚠️\n"

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

    # ==========================================
    # TEST ALL KEYS (/testall)
    # ==========================================
    if text.startswith("/testall"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        from config import GEMINI_API_KEYS, GROQ_API_KEYS, OPENROUTER_API_KEY, KEY_ACCOUNT_LABELS
        from agent.analyzer import get_active_provider_info

        prov, mdl, kidx = get_active_provider_info()
        await send_response(app_session, chat_id,
            f"🔑 Тестирую все ключи...\n"
            f"Активный: `{prov}` #{kidx+1} ({KEY_ACCOUNT_LABELS.get(kidx, '?')})\n"
            f"Модель: `{mdl}`", msg_id, parse_mode="Markdown")

        _results = []
        req_timeout = aiohttp.ClientTimeout(total=15)
        test_payload = {"contents": [{"parts": [{"text": "Say OK"}]}]}
        gemini_model = "gemini-2.5-flash"

        # Test Gemini keys
        async with aiohttp.ClientSession() as test_session:
            for i, key in enumerate(GEMINI_API_KEYS):
                label = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                active_mark = " 👈" if prov == "gemini" and kidx == i else ""
                try:
                    test_url = f"https://botgem.zhoriha.workers.dev/v1beta/models/{gemini_model}:generateContent?key={key}"
                    async with test_session.post(test_url, json=test_payload, timeout=req_timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            if data.get("candidates"):
                                _results.append(f"✅ Gemini #{i+1} ({label}){active_mark}")
                            else:
                                _results.append(f"⚠️ Gemini #{i+1} ({label}) — пустой ответ{active_mark}")
                        elif resp.status == 429:
                            _results.append(f"⏳ Gemini #{i+1} ({label}) — лимит{active_mark}")
                        elif resp.status == 400:
                            body = await resp.text()
                            if "API_KEY_INVALID" in body:
                                _results.append(f"❌ Gemini #{i+1} ({label}) — невалидный{active_mark}")
                            else:
                                _results.append(f"❌ Gemini #{i+1} ({label}) — 400{active_mark}")
                        else:
                            _results.append(f"❌ Gemini #{i+1} ({label}) — HTTP {resp.status}{active_mark}")
                except asyncio.TimeoutError:
                    _results.append(f"⏰ Gemini #{i+1} ({label}) — таймаут{active_mark}")
                except Exception as e:
                    _results.append(f"❌ Gemini #{i+1} ({label}) — {str(e)[:40]}{active_mark}")
                await asyncio.sleep(0.3)

            # Test Groq keys
            groq_headers = {"Content-Type": "application/json"}
            groq_payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Say OK"}], "temperature": 0.1}
            for i, key in enumerate(GROQ_API_KEYS):
                active_mark = " 👈" if prov == "groq" and kidx == i else ""
                try:
                    _h = {**groq_headers, "Authorization": f"Bearer {key}"}
                    async with test_session.post("https://api.groq.com/openai/v1/chat/completions",
                                                  json=groq_payload, headers=_h, timeout=req_timeout) as resp:
                        if resp.status == 200:
                            _results.append(f"✅ Groq #{i+1}{active_mark}")
                        elif resp.status == 429:
                            _results.append(f"⏳ Groq #{i+1} — лимит{active_mark}")
                        else:
                            _results.append(f"❌ Groq #{i+1} — HTTP {resp.status}{active_mark}")
                except asyncio.TimeoutError:
                    _results.append(f"⏰ Groq #{i+1} — таймаут{active_mark}")
                except Exception as e:
                    _results.append(f"❌ Groq #{i+1} — {str(e)[:40]}{active_mark}")
                await asyncio.sleep(0.3)

            # Test OpenRouter
            if OPENROUTER_API_KEY:
                for or_model in ["openai/gpt-oss-120b:free", "openrouter/free"]:
                    active_mark = " 👈" if prov == "openrouter" and mdl == or_model else ""
                    try:
                        _h = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                        _p = {"model": or_model, "messages": [{"role": "user", "content": "Say OK"}], "temperature": 0.1}
                        async with test_session.post("https://openrouter.ai/api/v1/chat/completions",
                                                      json=_p, headers=_h, timeout=req_timeout) as resp:
                            if resp.status == 200:
                                _results.append(f"✅ OpenRouter `{or_model}`{active_mark}")
                            else:
                                _results.append(f"❌ OpenRouter `{or_model}` — HTTP {resp.status}{active_mark}")
                    except asyncio.TimeoutError:
                        _results.append(f"⏰ OpenRouter `{or_model}` — таймаут{active_mark}")
                    except Exception as e:
                        _results.append(f"❌ OpenRouter `{or_model}` — {str(e)[:40]}{active_mark}")
                    await asyncio.sleep(0.3)

        result_text = "🔑 *Результаты тестирования:*\n\n" + "\n".join(_results)
        await send_response(app_session, chat_id, result_text, msg_id, parse_mode="Markdown")
        return

    if text == "/time":
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return
        import datetime as _dt
        _h = SCAN_SCHEDULE["hour"]
        _m = SCAN_SCHEDULE["minute"]
        _now_msk = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=3)))
        _scan_today = _now_msk.replace(hour=_h, minute=_m, second=0, microsecond=0)
        if _now_msk >= _scan_today:
            _next_scan = _scan_today + _dt.timedelta(days=1)
        else:
            _next_scan = _scan_today
        _delta = _next_scan - _now_msk
        _hours_left = int(_delta.total_seconds() // 3600)
        _mins_left = int((_delta.total_seconds() % 3600) // 60)
        msg_text = (
            f"⏰ Текущее время скана: *{_h:02d}:{_m:02d}* (UTC+3)\n"
            f"🕐 Следующий скан: *{_next_scan.strftime('%d.%m %H:%M')}* (через {_hours_left}ч {_mins_left}м)"
        )
        await send_response(app_session, chat_id, msg_text, msg_id, parse_mode="Markdown")
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
    # STOPLOSS MODE (/stoploss)
    # ==========================================
    # ==========================================
    # LINE 4H SETTINGS COMMAND
    # ==========================================
    if text.startswith("расширенные настройки линии 4ч") or text.startswith("расширенные настройки линий 4ч"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Только для админа", msg_id)
            return

        s = load_line_4h_settings()
        _mode = s["mode"]

        if _mode == "standard":
            text_resp = (
                "📐 *Настройки линий 4Ч*\n\n"
                "Режим: ✅ Стандарт\n"
                "_Якорь: ближняя линия, допуск 20%, "
                "точка Б >80 свечей → сужение до 10%_"
            )
            kb = {"inline_keyboard": [
                [{"text": "✅ Стандарт", "callback_data": "l4h_standard"},
                 {"text": "Пользовательский", "callback_data": "l4h_custom"}],
            ]}
        else:
            _anc = s.get("anchor", "nearest_line")
            _rpct = s.get("range_pct", 20.0)
            _pb = s.get("point_b_rule", "narrow_pct")
            _pb_pct = s.get("point_b_pct", 10.0)

            _anc_label = "Ближняя линия" if _anc == "nearest_line" else "Верх тела свечи"
            _pb_labels = {
                "no_change": "Без изменений",
                "nearest": "Ближняя к цене",
                "narrow_pct": f"Сужение до {_pb_pct}%",
            }
            _pb_label = _pb_labels.get(_pb, f"Сужение до {_pb_pct}%")

            text_resp = (
                f"📐 *Настройки линий 4Ч*\n\n"
                f"Режим: ✅ Пользовательский\n\n"
                f"1️⃣ Якорь: `{_anc_label}`\n"
                f"2️⃣ Допуск: `{_rpct}%`\n"
                f"3️⃣ Точка Б >80 свечей: `{_pb_label}`"
            )

            _anc_row = [
                {"text": ("✅ " if _anc == "nearest_line" else "") + "Ближняя линия",
                 "callback_data": "l4h_anc_line"},
                {"text": ("✅ " if _anc == "candle_top" else "") + "Верх тела свечи",
                 "callback_data": "l4h_anc_candle"},
            ]

            _pb_row = [
                {"text": ("✅ " if _pb == "no_change" else "") + "Без изменений",
                 "callback_data": "l4h_pb_nochange"},
                {"text": ("✅ " if _pb == "nearest" else "") + "Ближняя к цене",
                 "callback_data": "l4h_pb_nearest"},
                {"text": ("✅ " if _pb == "narrow_pct" else "") + f"Ввести % ({_pb_pct}%)",
                 "callback_data": "l4h_pb_pct"},
            ]

            kb = {"inline_keyboard": [
                [{"text": "Стандарт", "callback_data": "l4h_standard"},
                 {"text": "✅ Пользовательский", "callback_data": "l4h_custom"}],
                [{"text": "── Якорь ──", "callback_data": "l4h_noop"}],
                _anc_row,
                [{"text": "── Допуск ──", "callback_data": "l4h_noop"}],
                [{"text": f"📏 Изменить ({_rpct}%)", "callback_data": "l4h_range"}],
                [{"text": "── Точка Б (>80 свечей) ──", "callback_data": "l4h_noop"}],
                _pb_row,
            ]}

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        await app_session.post(url, json={
            "chat_id": chat_id, "text": text_resp,
            "parse_mode": "Markdown", "reply_markup": kb,
            "reply_to_message_id": msg_id,
        })
        return

    # ==========================================
    # TREND ABOVE THRESHOLD: порог N%
    # ==========================================
    if text.startswith("порог"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Только для админа", msg_id)
            return

        parts = text.split()
        if len(parts) >= 2:
            try:
                new_pct = float(parts[1].replace("%", "").replace(",", "."))
                if new_pct < 0 or new_pct > 50:
                    await send_response(app_session, chat_id, "❌ Допустимый диапазон: 0–50%", msg_id)
                    return
                save_trend_above_pct(new_pct)
                gcm_pct = new_pct + 1
                resp_text = (
                    f"✅ *Порог пробития обновлён*\n\n"
                    f"📐 PEAK-TO-PEAK: `{new_pct}%`\n"
                    f"📐 GROWING-CANDLE: `{gcm_pct}%` (+1%)"
                )
                await send_response(app_session, chat_id, resp_text, msg_id, parse_mode="Markdown")
            except ValueError:
                await send_response(app_session, chat_id, "❌ Укажите число, например: `порог 5`", msg_id, parse_mode="Markdown")
        else:
            current_pct = load_trend_above_pct()
            gcm_pct = current_pct + 1
            resp_text = (
                f"📐 *Текущий порог пробития*\n\n"
                f"📐 PEAK-TO-PEAK: `{current_pct}%`\n"
                f"📐 GROWING-CANDLE: `{gcm_pct}%` (+1%)\n\n"
                f"💡 Изменить: `порог 5` или `порог 2.53`"
            )
            await send_response(app_session, chat_id, resp_text, msg_id, parse_mode="Markdown")
        return

    if text.startswith("/stoploss"):
        if not is_admin(msg):
            deny = "⛔️ Admin only" if lang_pref == "en" else "⛔️ Только для админа"
            await send_response(app_session, chat_id, deny, msg_id)
            return

        from config import load_sl_settings
        s = load_sl_settings()
        sig_mode = s["signals"]["mode"]
        btc_shield = s.get("btc_shield", "off")

        _mode_names = {"stopai": "🎯 StopAI", "trailing": "🔄 Trailing",
                       "fixed": "📏 Fixed ATR", "ema": "📈 EMA SL"}
        _shield_names = {"off": "ВЫКЛ", "soft": "Soft"}

        msg_text = (
            f"⚙️ *Настройки стоп-лосса*\n\n"
            f"📊 *Signals:* {_mode_names.get(sig_mode, sig_mode)}\n"
            f"🅱️ *BTC Shield:* {_shield_names.get(btc_shield, btc_shield)}\n\n"
            f"Выберите банк для настройки:"
        )
        kb = {"inline_keyboard": [
            [
                {"text": f"📊 Signals ({_mode_names.get(sig_mode, '?')})", "callback_data": "slm_s"},
            ],
            [
                {"text": f"🅱️ BTC Shield: {_shield_names.get(btc_shield, btc_shield)}", "callback_data": "slm_btc"},
            ]
        ]}
        await send_response(app_session, chat_id, msg_text, msg_id,
                           reply_markup=kb, parse_mode="Markdown")
        return

    # ==========================================
    # SMC MODE (/smc)
    # ==========================================
    if text.startswith("/smc"):
        if not is_admin(msg):
            await send_response(app_session, chat_id, "⛔️ Admin only.", msg_id)
            return

        from config import load_smc_settings
        s = load_smc_settings()
        strict = s.get("strict_luxalgo", True)
        iob = s.get("internal_obs", 5)
        sob = s.get("swing_obs", 0)

        mode_name = "📺 TradingView (LuxAlgo)" if strict else "🤖 AIAlisa (ранний internal)"
        def _ob_label(val):
            return "OFF" if val == 0 else str(val)
        def _ck_v(current, val, label):
            return f"✅ {label}" if current == val else label

        msg_text = (
            f"📐 *Настройки SMC*\n\n"
            f"*Режим:* {mode_name}\n"
            f"*Internal OB:* {_ob_label(iob)}\n"
            f"*Swing OB:* {_ob_label(sob)}\n\n"
            f"_TView = точная копия LuxAlgo_\n"
            f"_AIAlisa = ранний internal structure_"
        )
        kb = {"inline_keyboard": [
            [
                {"text": _ck_v(strict, True, "📺 TView"), "callback_data": "smc_tview"},
                {"text": _ck_v(strict, False, "🤖 AIAlisa"), "callback_data": "smc_alisa"},
            ],
            [{"text": "── Internal блоки ──", "callback_data": "smc_noop"}],
            [
                {"text": _ck_v(iob, 0, "OFF"), "callback_data": "smc_iob_0"},
                {"text": _ck_v(iob, 3, "3"), "callback_data": "smc_iob_3"},
                {"text": _ck_v(iob, 5, "5"), "callback_data": "smc_iob_5"},
                {"text": _ck_v(iob, 10, "10"), "callback_data": "smc_iob_10"},
            ],
            [{"text": "── Swing блоки ──", "callback_data": "smc_noop"}],
            [
                {"text": _ck_v(sob, 0, "OFF"), "callback_data": "smc_sob_0"},
                {"text": _ck_v(sob, 3, "3"), "callback_data": "smc_sob_3"},
                {"text": _ck_v(sob, 5, "5"), "callback_data": "smc_sob_5"},
                {"text": _ck_v(sob, 10, "10"), "callback_data": "smc_sob_10"},
            ],
        ]}
        await send_response(app_session, chat_id, msg_text, msg_id,
                           reply_markup=kb, parse_mode="Markdown")
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
    # CHART ANALYSIS (scan / check / посмотри …)
    # ==========================================
    # COIN ANALYSIS (посмотри/scan/check...)
    # ==========================================
    analysis_prefixes = [
        "scan ", "check ", "look ", "analyze ",
        "посмотри ", "посмотри на ", "глянь ", "чекни ", "анализ "
    ]
    matched_prefix = next((p for p in analysis_prefixes if text.startswith(p)), None)

    if matched_prefix:
        # --- Parse optional timeframe: "посмотри 1ч BTC", "scan 15m BTC", etc. ---
        remaining = text.replace(matched_prefix, "").strip()
        tf_aliases = {
            "15m": "15m", "15м": "15m", "15min": "15m",
            "1h": "1h", "1ч": "1h", "1час": "1h",
            "4h": "4h", "4ч": "4h", "4час": "4h",
            "1d": "1d", "1д": "1d", "1день": "1d", "1day": "1d",
        }
        parts = remaining.split()
        scan_tf = "4h"  # default
        if len(parts) >= 2 and parts[0].lower() in tf_aliases:
            scan_tf = tf_aliases[parts[0].lower()]
            symbol_raw = parts[1].upper()
        elif len(parts) >= 1:
            symbol_raw = parts[0].upper()
        else:
            symbol_raw = remaining.upper()

        # Map scan_tf to Binance interval and label
        tf_map = {"15m": ("15m", "15m"), "1h": ("1h", "1H"), "4h": ("4h", "4H"), "1d": ("1d", "1D")}
        binance_interval, tf_label = tf_map.get(scan_tf, ("4h", "4H"))

        symbol = symbol_raw + "USDT" if not symbol_raw.endswith("USDT") else symbol_raw

        fetch_msg = f"⏳ Fetching chart data + building trend line... ({symbol} {tf_label})" if lang_pref == "en" else f"⏳ Загружаю график + строю трендовую линию... ({symbol} {tf_label})"
        stream_msg_id = await send_and_get_msg_id(app_session, chat_id, fetch_msg, msg_id)

        raw_df_full = await fetch_klines(app_session, symbol, binance_interval, 199)
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

            from core.smc import analyze_smc, get_smc_mode
            _smc_strict = get_smc_mode()
            smc_data = {}
            try:
                # Primary TF (scan_tf) gets 1500 candles, others get 999
                # 999 is the sweet spot: same Binance weight as 500 (weight=5), but more data
                smc_candles = {"1d": 999, "4h": 999, "1h": 999, "15m": 999}
                smc_candles[scan_tf] = 1500  # primary TF gets max

                raw_smc_1d = await fetch_klines(app_session, symbol, "1d", smc_candles["1d"]) if raw_df_1d else None
                raw_smc_4h = await fetch_klines(app_session, symbol, "4h", smc_candles["4h"]) if raw_df_4h else None
                raw_smc_1h = await fetch_klines(app_session, symbol, "1h", smc_candles["1h"]) if raw_df_1h else None
                raw_smc_15m = await fetch_klines(app_session, symbol, "15m", smc_candles["15m"]) if raw_df_15m else None

                if raw_smc_1d:
                    smc_data["1D"] = analyze_smc(pd.DataFrame(raw_smc_1d), "1D", symbol=symbol, strict_luxalgo=_smc_strict)
                elif raw_df_1d:
                    smc_data["1D"] = analyze_smc(pd.DataFrame(raw_df_1d), "1D", symbol=symbol, strict_luxalgo=_smc_strict)
                if raw_smc_4h:
                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_smc_4h), "4H", symbol=symbol, strict_luxalgo=_smc_strict)
                elif raw_df_4h:
                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_df_4h), "4H", symbol=symbol, strict_luxalgo=_smc_strict)
                if raw_smc_1h:
                    smc_data["1H"] = analyze_smc(pd.DataFrame(raw_smc_1h), "1H", symbol=symbol, strict_luxalgo=_smc_strict)
                elif raw_df_1h:
                    smc_data["1H"] = analyze_smc(pd.DataFrame(raw_df_1h), "1H", symbol=symbol, strict_luxalgo=_smc_strict)
                if raw_smc_15m:
                    smc_data["15m"] = analyze_smc(pd.DataFrame(raw_smc_15m), "15m", symbol=symbol, strict_luxalgo=_smc_strict)
                elif raw_df_15m:
                    smc_data["15m"] = analyze_smc(pd.DataFrame(raw_df_15m), "15m", symbol=symbol, strict_luxalgo=_smc_strict)
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

            ai_msg = await ask_ai_analysis(symbol, tf_label, last_row, lang=lang_pref, telegram_stream=tg_stream, extended=True, mode="extended", mtf_data=mtf_data, smc_data=smc_data)

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
                line_data, _ = await find_trend_line(df_full, tf_label, symbol, save_alert=False)
                # Pass SMC data for the selected timeframe to chart drawer
                chart_smc = smc_data.get(tf_label)
                if line_data:
                    chart_path = await draw_scan_chart(symbol, df_full, line_data, tf_label, smc_overlay=chart_smc)
                else:
                    chart_path = await draw_simple_chart(symbol, df_full, tf_label, smc_overlay=chart_smc)

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
                            if resp.status == 429:
                                resp_json = await resp.json()
                                retry_after = resp_json.get("parameters", {}).get("retry_after", 15)
                                logging.warning(f"⚠️ Scan photo 429 — waiting {retry_after}s then retrying...")
                                await asyncio.sleep(retry_after + 1)
                                # Retry: re-open file and resend
                                try:
                                    with open(chart_path, 'rb') as f2:
                                        data2 = aiohttp.FormData()
                                        data2.add_field('chat_id', str(chat_id))
                                        data2.add_field('caption', caption)
                                        data2.add_field('parse_mode', 'Markdown')
                                        data2.add_field('reply_to_message_id', str(msg_id))
                                        data2.add_field('reply_markup', json.dumps(scan_markup))
                                        data2.add_field('photo', f2, filename=f"{symbol}.png", content_type='image/png')
                                        async with app_session.post(photo_url, data=data2, timeout=30) as resp2:
                                            if resp2.status != 200:
                                                resp_text2 = await resp2.text()
                                                logging.error(f"❌ Scan photo retry failed: {resp2.status} - {resp_text2}")
                                                await send_response(app_session, chat_id, ai_brief, msg_id, reply_markup=scan_markup)
                                except Exception as e2:
                                    logging.error(f"❌ Scan photo retry error: {e2}")
                                    await send_response(app_session, chat_id, ai_brief, msg_id, reply_markup=scan_markup)
                            elif resp.status != 200:
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
                    from core.smc import analyze_smc, get_smc_mode
                    _smc_strict = get_smc_mode()
                    # Margin scan: primary TF is 4h → 1500 candles, others → 999
                    raw_smc_1d = await fetch_klines(app_session, coin_to_analyze, "1d", 999) if raw_1d else None
                    raw_smc_4h = await fetch_klines(app_session, coin_to_analyze, "4h", 1500) if raw_4h else None
                    raw_smc_1h = await fetch_klines(app_session, coin_to_analyze, "1h", 999) if raw_1h else None
                    raw_smc_15m = await fetch_klines(app_session, coin_to_analyze, "15m", 999) if raw_15m else None

                    if raw_smc_1d:
                        smc_data["1D"] = analyze_smc(pd.DataFrame(raw_smc_1d), "1D", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    elif raw_1d:
                        smc_data["1D"] = analyze_smc(pd.DataFrame(raw_1d), "1D", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    if raw_smc_4h:
                        smc_data["4H"] = analyze_smc(pd.DataFrame(raw_smc_4h), "4H", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    elif raw_4h:
                        smc_data["4H"] = analyze_smc(pd.DataFrame(raw_4h), "4H", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    if raw_smc_1h:
                        smc_data["1H"] = analyze_smc(pd.DataFrame(raw_smc_1h), "1H", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    elif raw_1h:
                        smc_data["1H"] = analyze_smc(pd.DataFrame(raw_1h), "1H", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    if raw_smc_15m:
                        smc_data["15m"] = analyze_smc(pd.DataFrame(raw_smc_15m), "15m", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                    elif raw_15m:
                        smc_data["15m"] = analyze_smc(pd.DataFrame(raw_15m), "15m", symbol=coin_to_analyze, strict_luxalgo=_smc_strict)
                except Exception as e:
                    logging.error(f"❌ SMC look error: {e}")

                ai_msg = await ask_ai_analysis(coin_to_analyze, "4H", last_row, user_margin=margin_data, lang=lang_pref, mode="scan", mtf_data=mtf_data, smc_data=smc_data)

                await send_response(app_session, chat_id, ai_msg, msg_id)


# ==========================================
# MANUAL ALERT HELPER FUNCTIONS
# ==========================================

async def _finalize_manual_alert(app_session, chat_id, msg_id, ma_state, mode):
    """
    Core logic: fetch candles, find two points, build trendline, draw chart, save alert.
    mode: 'high' | 'low' | 'body' | 'date_top' | 'date_bottom'
    """
    import math as _math
    import numpy as np

    symbol = ma_state['symbol']
    tf = ma_state['tf']
    prices = ma_state['prices']  # [price_a_base, price_b_base] — base prices for candle matching
    pct_offsets = ma_state.get('pct_offsets', [0, 0])  # percent offsets applied AFTER matching

    tf_map = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
    tf_ms_map = {"15m": 900000, "1H": 3600000, "4H": 14400000, "1D": 86400000}
    binance_interval = tf_map.get(tf, "4h")
    tf_ms = tf_ms_map.get(tf, 14400000)

    bot_msg = await send_response(app_session, chat_id, "⏳ Строю линию...", msg_id)
    track_alert_msg(chat_id, bot_msg)

    try:
        raw = await fetch_klines(app_session, symbol, binance_interval, 199)
        if not raw:
            await send_response(app_session, chat_id, "❌ Не удалось получить данные.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        df = pd.DataFrame(raw)
        df[['high', 'low', 'close', 'open']] = df[['high', 'low', 'close', 'open']].apply(pd.to_numeric)
        view_limit = min(len(df), 199)
        df_l = df.iloc[-view_limit:].copy().reset_index(drop=True)

        price_a_target, price_b_target = prices[0], prices[1]

        # Find candles matching target price
        def _get_candle_price(row, mode_str):
            """Get price from candle based on mode."""
            if mode_str == 'high':
                return float(row['high'])
            elif mode_str == 'low':
                return float(row['low'])
            elif mode_str in ('body', 'body_mid'):
                return (float(row['open']) + float(row['close'])) / 2
            elif mode_str in ('body_top', 'date_top'):
                return max(float(row['open']), float(row['close']))
            elif mode_str in ('body_bot', 'date_bottom'):
                return min(float(row['open']), float(row['close']))
            return float(row['high'])

        def _find_matches(df_local, target_price, mode_str):
            """Find candles matching target_price.
            Returns (matches_list, exact_bool).
            - Exact match (0.001%): use silently
            - Consecutive exact matches → use last one silently
            - Non-consecutive exact → show buttons
            - No exact → 5 above + 5 below target price as buttons"""
            exact = []
            all_entries = []  # (idx, candle_price, candle_time)
            for i in range(len(df_local)):
                row = df_local.iloc[i]
                candle_price = _get_candle_price(row, mode_str)
                diff_pct = abs(candle_price - target_price) / target_price * 100 if target_price > 0 else float('inf')
                candle_time = int(row['open_time'])
                all_entries.append((i, candle_price, candle_time))
                if diff_pct <= 0.001:
                    exact.append((i, candle_price, candle_time))

            if exact:
                indices = [m[0] for m in exact]
                is_consecutive = all(indices[j+1] - indices[j] == 1 for j in range(len(indices)-1))
                if len(exact) == 1 or is_consecutive:
                    return [exact[-1]], True
                else:
                    return exact, True

            # No exact match → 5 candles above + 5 below target price
            above = sorted([e for e in all_entries if e[1] >= target_price],
                          key=lambda x: x[1])[:5]
            below = sorted([e for e in all_entries if e[1] < target_price],
                          key=lambda x: -x[1])[:5]
            # Combine: below (descending) + above (ascending) → natural price order
            combined = sorted(below + above, key=lambda x: x[1])
            if not combined:
                combined = sorted(all_entries, key=lambda x: abs(x[1] - target_price))[:10]
            return combined, False

        def _format_candle_time(time_ms):
            """Format candle timestamp for button label (in user's timezone)."""
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            tz_off = get_user_tz_offset(chat_id)
            dt = _dt.fromtimestamp(time_ms / 1000, tz=_tz.utc) + _td(hours=tz_off)
            return dt.strftime("%d.%m.%y %H:%M")

        matches_a, exact_a = _find_matches(df_l, price_a_target, mode)
        matches_b, exact_b = _find_matches(df_l, price_b_target, mode)

        # If multiple matches for point A → ask user to choose
        if len(matches_a) > 1:
            buttons = []
            for idx, price, time_ms in matches_a:
                label = f"{_format_candle_time(time_ms)} — {price}"
                cb_data = f"malert_pick_a_{idx}_{price}"
                buttons.append([{"text": label, "callback_data": cb_data}])
            buttons.append([{"text": "⬅️ Назад", "callback_data": "malert_back"}])
            # Save state for picking
            ma_state['step'] = 'picking_point_a'
            ma_state['_df_cache_key'] = f"{symbol}_{tf}"
            ma_state['matches_b'] = [(m[0], m[1], m[2]) for m in matches_b]
            ma_state['exact_b'] = exact_b
            set_manual_alert_state(chat_id, ma_state)
            title = "точной" if exact_a else "ближайшей"
            pick_msg = await send_response(app_session, chat_id,
                f"🔍 Для точки A ({price_a_target:.8g}) найдено несколько совпадений {title} цены.\n"
                f"Выбери свечу:",
                msg_id, reply_markup={"inline_keyboard": buttons})
            track_alert_msg(chat_id, pick_msg)
            return

        # If multiple matches for point B → ask user to choose
        if len(matches_b) > 1:
            idx_a = matches_a[0][0]
            actual_price_a = matches_a[0][1]
            buttons = []
            for idx, price, time_ms in matches_b:
                label = f"{_format_candle_time(time_ms)} — {price}"
                cb_data = f"malert_pick_b_{idx}_{price}"
                buttons.append([{"text": label, "callback_data": cb_data}])
            buttons.append([{"text": "⬅️ Назад", "callback_data": "malert_back"}])
            ma_state['step'] = 'picking_point_b'
            ma_state['chosen_a_idx'] = idx_a
            ma_state['chosen_a_price'] = actual_price_a
            set_manual_alert_state(chat_id, ma_state)
            title = "точной" if exact_b else "ближайшей"
            pick_msg = await send_response(app_session, chat_id,
                f"🔍 Для точки B ({price_b_target:.8g}) найдено несколько совпадений {title} цены.\n"
                f"Выбери свечу:",
                msg_id, reply_markup={"inline_keyboard": buttons})
            track_alert_msg(chat_id, pick_msg)
            return

        # Single match for both — proceed
        idx_a = matches_a[0][0]
        actual_price_a = matches_a[0][1]
        idx_b = matches_b[0][0]
        actual_price_b = matches_b[0][1]

        # Ensure A and B are different candles
        if idx_a == idx_b:
            await send_response(app_session, chat_id,
                "⚠️ Обе цены попали на одну свечу. Попробуй другие цены.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        # Ensure A is before B (left to right)
        swapped = False
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a
            actual_price_a, actual_price_b = actual_price_b, actual_price_a
            swapped = True

        # Apply percent offsets AFTER candle matching (shift trendline up/down)
        pct_a = pct_offsets[0] if len(pct_offsets) > 0 else 0
        pct_b = pct_offsets[1] if len(pct_offsets) > 1 else 0
        if swapped:
            pct_a, pct_b = pct_b, pct_a
        line_price_a = actual_price_a * (1 + pct_a / 100) if pct_a else actual_price_a
        line_price_b = actual_price_b * (1 + pct_b / 100) if pct_b else actual_price_b

        # Calculate slope in log space (using offset prices for the trendline)
        log_a = _math.log(line_price_a)
        log_b = _math.log(line_price_b)
        log_slope = (log_b - log_a) / (idx_b - idx_a)
        log_intercept = log_a - log_slope * idx_a

        last_idx = view_limit - 1
        base_open_time = int(df_l['open_time'].iloc[-1])

        # Save alert FIRST so it appears on the chart
        # monitor_from_time = next candle after current last → don't trigger on existing candles
        monitor_from_time = base_open_time + tf_ms
        alerts = load_manual_alerts()
        # Assign persistent color index (next available for this symbol+tf)
        existing_colors = [a.get('color_idx', 0) for a in alerts
                          if a['symbol'] == symbol and a['tf'] == tf]
        color_idx = 0
        while color_idx in existing_colors:
            color_idx += 1
        alert_entry = {
            "symbol": symbol,
            "tf": tf,
            "chat_id": chat_id,
            "price_a": line_price_a,
            "price_b": line_price_b,
            "index_a": idx_a,
            "index_b": idx_b,
            "slope": log_slope,
            "intercept": log_intercept,
            "base_idx": last_idx,
            "base_open_time": base_open_time,
            "monitor_from_time": monitor_from_time,
            "tf_ms": tf_ms,
            "mode": mode,
            "color_idx": color_idx,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        alerts.append(alert_entry)
        save_manual_alerts(alerts)

        # Draw chart with ALL active lines for this symbol+tf
        all_lines_for_chart = [
            {'price_a': a['price_a'], 'price_b': a['price_b'],
             'index_a': a['index_a'], 'index_b': a['index_b'],
             'base_open_time': a.get('base_open_time', 0), 'base_idx': a.get('base_idx', 0), 'tf_ms': a.get('tf_ms', 0), 'color_idx': a.get('color_idx', 0)}
            for a in alerts if a['symbol'] == symbol and a['tf'] == tf
        ]
        # Get SMC data for manual alert chart
        _alert_smc = None
        try:
            from core.smc import analyze_smc
            _alert_smc = analyze_smc(df, tf, symbol=symbol)
        except Exception as _smc_e:
            logging.error(f"❌ SMC error for alert chart: {repr(_smc_e)}")
        chart_path = await draw_alert_chart(symbol, df, all_lines_for_chart, tf, smc_overlay=_alert_smc)

        clear_manual_alert_state(chat_id)

        # Send chart
        short_sym = symbol.replace("USDT", "")
        current_price = float(df_l['close'].iloc[-1])
        all_alerts_sym_tf = [a for a in alerts if a['symbol'] == symbol and a['tf'] == tf]
        caption = _build_alert_caption(short_sym, tf, current_price, all_alerts_sym_tf, chat_id)

        if chart_path:
            import os
            photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            try:
                with open(chart_path, 'rb') as f:
                    import aiohttp as _aio
                    data = _aio.FormData()
                    data.add_field('chat_id', str(chat_id))
                    data.add_field('caption', caption)
                    data.add_field('parse_mode', 'Markdown')
                    data.add_field('photo', f, filename=f"{symbol}_alert.png", content_type='image/png')
                    async with app_session.post(photo_url, data=data, timeout=30) as resp:
                        if resp.status != 200:
                            resp_text = await resp.text()
                            logging.error(f"❌ Alert chart send error: {resp.status} - {resp_text}")
                            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ Alert chart send error: {repr(e)}")
                await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            finally:
                try:
                    if os.path.exists(chart_path):
                        os.remove(chart_path)
                except:
                    pass
        else:
            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")

        # Schedule cleanup of intermediate messages after 60 seconds
        await schedule_alert_cleanup(app_session, chat_id, delay_seconds=60)

    except Exception as e:
        logging.error(f"❌ Manual alert error: {repr(e)}")
        await send_response(app_session, chat_id, f"❌ Ошибка: {e}", msg_id)
        clear_manual_alert_state(chat_id)
        clear_tracked_alert_msgs(chat_id)


async def _finalize_manual_alert_with_indices(app_session, chat_id, msg_id, ma_state):
    """Finalize manual alert when user has already picked point A and B indices via buttons."""
    import math as _math

    symbol = ma_state['symbol']
    tf = ma_state['tf']
    mode = ma_state.get('mode', 'high')
    idx_a = ma_state['chosen_a_idx']
    actual_price_a = ma_state['chosen_a_price']
    idx_b = ma_state['chosen_b_idx']
    actual_price_b = ma_state['chosen_b_price']
    pct_offsets = ma_state.get('pct_offsets', [0, 0])

    tf_map = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
    tf_ms_map = {"15m": 900000, "1H": 3600000, "4H": 14400000, "1D": 86400000}
    binance_interval = tf_map.get(tf, "4h")
    tf_ms = tf_ms_map.get(tf, 14400000)

    bot_msg = await send_response(app_session, chat_id, "⏳ Строю линию...", msg_id)
    track_alert_msg(chat_id, bot_msg)

    try:
        raw = await fetch_klines(app_session, symbol, binance_interval, 199)
        if not raw:
            await send_response(app_session, chat_id, "❌ Не удалось получить данные.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        df = pd.DataFrame(raw)
        df[['high', 'low', 'close', 'open']] = df[['high', 'low', 'close', 'open']].apply(pd.to_numeric)
        view_limit = min(len(df), 199)
        df_l = df.iloc[-view_limit:].copy().reset_index(drop=True)

        if idx_a == idx_b:
            await send_response(app_session, chat_id, "⚠️ Обе точки на одной свече.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        swapped = False
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a
            actual_price_a, actual_price_b = actual_price_b, actual_price_a
            swapped = True

        # Apply percent offsets AFTER candle matching
        pct_a = pct_offsets[0] if len(pct_offsets) > 0 else 0
        pct_b = pct_offsets[1] if len(pct_offsets) > 1 else 0
        if swapped:
            pct_a, pct_b = pct_b, pct_a
        line_price_a = actual_price_a * (1 + pct_a / 100) if pct_a else actual_price_a
        line_price_b = actual_price_b * (1 + pct_b / 100) if pct_b else actual_price_b

        log_a = _math.log(line_price_a)
        log_b = _math.log(line_price_b)
        log_slope = (log_b - log_a) / (idx_b - idx_a)
        log_intercept = log_a - log_slope * idx_a

        last_idx = view_limit - 1
        base_open_time = int(df_l['open_time'].iloc[-1])
        monitor_from_time = base_open_time + tf_ms

        alerts = load_manual_alerts()
        existing_colors = [a.get('color_idx', 0) for a in alerts
                          if a['symbol'] == symbol and a['tf'] == tf]
        color_idx = 0
        while color_idx in existing_colors:
            color_idx += 1
        alert_entry = {
            "symbol": symbol, "tf": tf, "chat_id": chat_id,
            "price_a": line_price_a, "price_b": line_price_b,
            "index_a": idx_a, "index_b": idx_b,
            "slope": log_slope, "intercept": log_intercept,
            "base_idx": last_idx, "base_open_time": base_open_time,
            "monitor_from_time": monitor_from_time,
            "tf_ms": tf_ms, "mode": mode, "color_idx": color_idx,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        alerts.append(alert_entry)
        save_manual_alerts(alerts)

        all_lines_for_chart = [
            {'price_a': a['price_a'], 'price_b': a['price_b'],
             'index_a': a['index_a'], 'index_b': a['index_b'],
             'base_open_time': a.get('base_open_time', 0),
             'base_idx': a.get('base_idx', 0), 'tf_ms': a.get('tf_ms', 0)}
            for a in alerts if a['symbol'] == symbol and a['tf'] == tf
        ]
        # Get SMC data for manual alert chart
        _alert_smc = None
        try:
            from core.smc import analyze_smc
            _alert_smc = analyze_smc(df, tf, symbol=symbol)
        except Exception as _smc_e:
            logging.error(f"❌ SMC error for alert chart: {repr(_smc_e)}")
        chart_path = await draw_alert_chart(symbol, df, all_lines_for_chart, tf, smc_overlay=_alert_smc)
        clear_manual_alert_state(chat_id)

        short_sym = symbol.replace("USDT", "")
        current_price = float(df_l['close'].iloc[-1])
        all_alerts_sym_tf = [a for a in alerts if a['symbol'] == symbol and a['tf'] == tf]
        caption = _build_alert_caption(short_sym, tf, current_price, all_alerts_sym_tf, chat_id)

        if chart_path:
            import os
            photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            try:
                with open(chart_path, 'rb') as f:
                    import aiohttp as _aio
                    data = _aio.FormData()
                    data.add_field('chat_id', str(chat_id))
                    data.add_field('caption', caption)
                    data.add_field('parse_mode', 'Markdown')
                    data.add_field('photo', f, filename=f"{symbol}_alert.png", content_type='image/png')
                    async with app_session.post(photo_url, data=data, timeout=30) as resp:
                        if resp.status != 200:
                            resp_text = await resp.text()
                            logging.error(f"❌ Alert chart send error: {resp.status} - {resp_text}")
                            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ Alert chart send error: {repr(e)}")
                await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            finally:
                try:
                    if os.path.exists(chart_path):
                        os.remove(chart_path)
                except:
                    pass
        else:
            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")

        # Schedule cleanup of intermediate messages after 60 seconds
        await schedule_alert_cleanup(app_session, chat_id, delay_seconds=60)

    except Exception as e:
        logging.error(f"❌ Manual alert with indices error: {repr(e)}")
        await send_response(app_session, chat_id, f"❌ Ошибка: {e}", msg_id)
        clear_manual_alert_state(chat_id)
        clear_tracked_alert_msgs(chat_id)


async def _process_manual_alert_dates(app_session, chat_id, msg_id, ma_state):
    """Process manual alert with user-specified dates."""
    import math as _math

    symbol = ma_state['symbol']
    tf = ma_state['tf']
    date_mode = ma_state.get('date_mode', 'date_top')  # date_top or date_bottom
    dates = ma_state.get('dates', [])

    if len(dates) < 2:
        await send_response(app_session, chat_id, "❌ Нет дат.", msg_id)
        clear_manual_alert_state(chat_id)
        return

    tf_map = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
    tf_ms_map = {"15m": 900000, "1H": 3600000, "4H": 14400000, "1D": 86400000}
    binance_interval = tf_map.get(tf, "4h")
    tf_ms = tf_ms_map.get(tf, 14400000)

    bot_msg = await send_response(app_session, chat_id, "⏳ Строю линию по датам...", msg_id)
    track_alert_msg(chat_id, bot_msg)

    try:
        raw = await fetch_klines(app_session, symbol, binance_interval, 199)
        if not raw:
            await send_response(app_session, chat_id, "❌ Не удалось получить данные.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        df = pd.DataFrame(raw)
        df[['high', 'low', 'close', 'open']] = df[['high', 'low', 'close', 'open']].apply(pd.to_numeric)
        view_limit = min(len(df), 199)
        df_l = df.iloc[-view_limit:].copy().reset_index(drop=True)

        # Parse dates and find nearest candles
        dt_a = datetime.fromisoformat(dates[0])
        dt_b = datetime.fromisoformat(dates[1])
        ts_a = int(dt_a.timestamp() * 1000)
        ts_b = int(dt_b.timestamp() * 1000)

        def _find_by_time(df_local, target_ts):
            best_idx = 0
            best_diff = float('inf')
            for i in range(len(df_local)):
                ot = int(df_local['open_time'].iloc[i])
                diff = abs(ot - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            return best_idx

        idx_a = _find_by_time(df_l, ts_a)
        idx_b = _find_by_time(df_l, ts_b)

        if idx_a == idx_b:
            await send_response(app_session, chat_id,
                "⚠️ Обе даты попали на одну свечу. Попробуй другие даты.", msg_id)
            clear_manual_alert_state(chat_id)
            clear_tracked_alert_msgs(chat_id)
            return

        # Ensure A before B
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a

        # Get prices from candle body
        row_a = df_l.iloc[idx_a]
        row_b = df_l.iloc[idx_b]
        if date_mode == 'date_top':
            actual_price_a = max(float(row_a['open']), float(row_a['close']))
            actual_price_b = max(float(row_b['open']), float(row_b['close']))
        else:  # date_bottom
            actual_price_a = min(float(row_a['open']), float(row_a['close']))
            actual_price_b = min(float(row_b['open']), float(row_b['close']))

        # Build state for finalization
        ma_state['prices'] = [actual_price_a, actual_price_b]

        # Calculate slope in log space
        log_a = _math.log(actual_price_a)
        log_b = _math.log(actual_price_b)
        log_slope = (log_b - log_a) / (idx_b - idx_a)
        log_intercept = log_a - log_slope * idx_a

        last_idx = view_limit - 1
        base_open_time = int(df_l['open_time'].iloc[-1])
        monitor_from_time = base_open_time + tf_ms

        # Save alert FIRST so it appears on the chart
        alerts = load_manual_alerts()
        existing_colors = [a.get('color_idx', 0) for a in alerts
                          if a['symbol'] == symbol and a['tf'] == tf]
        color_idx = 0
        while color_idx in existing_colors:
            color_idx += 1
        alert_entry = {
            "symbol": symbol,
            "tf": tf,
            "chat_id": chat_id,
            "price_a": actual_price_a,
            "price_b": actual_price_b,
            "index_a": idx_a,
            "index_b": idx_b,
            "slope": log_slope,
            "intercept": log_intercept,
            "base_idx": last_idx,
            "base_open_time": base_open_time,
            "monitor_from_time": monitor_from_time,
            "tf_ms": tf_ms,
            "mode": date_mode,
            "color_idx": color_idx,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        alerts.append(alert_entry)
        save_manual_alerts(alerts)

        # Draw chart with ALL active lines for this symbol+tf
        all_lines_for_chart = [
            {'price_a': a['price_a'], 'price_b': a['price_b'],
             'index_a': a['index_a'], 'index_b': a['index_b'],
             'base_open_time': a.get('base_open_time', 0), 'base_idx': a.get('base_idx', 0), 'tf_ms': a.get('tf_ms', 0), 'color_idx': a.get('color_idx', 0)}
            for a in alerts if a['symbol'] == symbol and a['tf'] == tf
        ]
        # Get SMC data for manual alert chart
        _alert_smc = None
        try:
            from core.smc import analyze_smc
            _alert_smc = analyze_smc(df, tf, symbol=symbol)
        except Exception as _smc_e:
            logging.error(f"❌ SMC error for alert chart: {repr(_smc_e)}")
        chart_path = await draw_alert_chart(symbol, df, all_lines_for_chart, tf, smc_overlay=_alert_smc)

        clear_manual_alert_state(chat_id)

        # Send chart
        short_sym = symbol.replace("USDT", "")
        current_price = float(df_l['close'].iloc[-1])
        all_alerts_sym_tf = [a for a in alerts if a['symbol'] == symbol and a['tf'] == tf]
        caption = _build_alert_caption(short_sym, tf, current_price, all_alerts_sym_tf, chat_id)

        if chart_path:
            import os
            photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            try:
                with open(chart_path, 'rb') as f:
                    import aiohttp as _aio
                    data = _aio.FormData()
                    data.add_field('chat_id', str(chat_id))
                    data.add_field('caption', caption)
                    data.add_field('parse_mode', 'Markdown')
                    data.add_field('photo', f, filename=f"{symbol}_alert.png", content_type='image/png')
                    async with app_session.post(photo_url, data=data, timeout=30) as resp:
                        if resp.status != 200:
                            resp_text = await resp.text()
                            logging.error(f"❌ Alert chart send (date) error: {resp.status} - {resp_text}")
                            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ Alert chart send (date) error: {repr(e)}")
                await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")
            finally:
                try:
                    if os.path.exists(chart_path):
                        os.remove(chart_path)
                except:
                    pass
        else:
            await send_response(app_session, chat_id, caption, msg_id, parse_mode="Markdown")

        # Schedule cleanup of intermediate messages after 60 seconds
        await schedule_alert_cleanup(app_session, chat_id, delay_seconds=60)

    except Exception as e:
        logging.error(f"❌ Manual alert dates error: {repr(e)}")
        await send_response(app_session, chat_id, f"❌ Ошибка: {e}", msg_id)
        clear_manual_alert_state(chat_id)
        clear_tracked_alert_msgs(chat_id)
