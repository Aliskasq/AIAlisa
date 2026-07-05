"""
tg_callbacks.py — Inline keyboard / callback_query handlers.
Extracted from tg_listener.py during refactor.
"""
import logging
import agent.analyzer
import agent.square_publisher
import config as _cfg
from config import BOT_TOKEN
from core.tg_state import (
    send_response, is_allowed_chat, ADMIN_ID, _load_langs,
    square_cache_get, square_cache_delete, _fetch_or_free_models,
    track_alert_msg,
    set_sector_state, clear_sector_state,
)
from core.categories import (
    ALL_SECTORS, SECTOR_SHORT, SECTOR_EMOJI,
    get_sectors, get_sector_counts, get_symbols_by_sector, get_unknown_symbols,
    add_sector, remove_sector,
    toggle_scan_sector, toggle_scan_unknown, load_scan_settings,
)
from agent.skills import (
    post_to_binance_square,
    get_smart_money_signals,
    get_unified_token_rank,
    get_social_hype_leaderboard,
    get_smart_money_inflow_rank,
    get_meme_rank,
    get_address_pnl_rank,
)


# ============================
# SLM MENU RENDERERS
# ============================

def _ck(val, current, label):
    """Checkbox helper: ✅ if val == current, else plain label."""
    return f"✅ {label}" if val == current else label


async def _slm_edit(session, chat_id, msg_id, text, kb, cq_id, toast=""):
    """Edit message + answer callback in one call."""
    await session.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
        json={"chat_id": chat_id, "message_id": msg_id,
              "text": text, "parse_mode": "Markdown", "reply_markup": kb},
    )
    await session.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
        json={"callback_query_id": cq_id, "text": toast or "✅"},
    )


def _slm_bank_prefix(bank_key):
    return "slm_s"


def _slm_render_bank_menu(bank_key, bank_label, bs, _mode_names):
    """Render mode selection for a bank."""
    p = _slm_bank_prefix(bank_key)
    mode = bs["mode"]

    modes = ["stopai", "trailing", "fixed", "ema"]
    rows = []
    row = []
    for m in modes:
        label = _ck(m, mode, _mode_names[m])
        row.append({"text": label, "callback_data": f"{p}_{m}"})
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([{"text": "⬅️ Назад", "callback_data": f"{p}_top"}])

    # Summary
    summary = _slm_mode_summary(bs)
    text = f"⚙️ *{bank_label} — Режим стоп-лосса*\n\nТекущий: {_mode_names.get(mode, mode)}\n{summary}"
    return text, {"inline_keyboard": rows}


def _slm_mode_summary(bs):
    """One-line summary of current mode settings."""
    mode = bs["mode"]
    if mode == "stopai":
        return "SL/TP от ИИ"
    elif mode == "trailing":
        t = bs["trailing"]
        anchor_names = {"high-1": "High-1", "low-1": "Low-1", "high-2": "High-2", "low-2": "Low-2"}
        return (f"📐 {anchor_names.get(t['anchor'], t['anchor'])} | "
                f"⏱ {t['activation_candles']} свечей | "
                f"🛡 {t['initial_sl_atr']} ATR | "
                f"🔄 {t['trail_atr']} ATR")
    elif mode == "fixed":
        f = bs["fixed"]
        return f"🛡 SL: {f['sl_atr']} ATR | 🎯 TP: {f['tp_atr']} ATR"
    elif mode == "ema":
        e = bs["ema"]
        return f"🛡 SL: {e['initial_sl_atr']} ATR | EMA{e['ema_period']} {e['ema_tf']}"
    return ""


def _slm_render_trailing(bank_key, bank_label, bs, _mode_names, _anchor_names):
    """Render trailing stop config page with all options."""
    p = _slm_bank_prefix(bank_key)
    t = bs["trailing"]

    text = (
        f"🔄 *{bank_label} — Trailing Stop*\n\n"
        f"📐 *Привязка:* {_anchor_names.get(t['anchor'], t['anchor'])}\n"
        f"⏱ *Активация:* {t['activation_candles']} свечей 5м\n"
        f"🛡 *Начальный SL:* {t['initial_sl_atr']} ATR\n"
        f"🔄 *Трейлинг буфер:* {t['trail_atr']} ATR\n\n"
        f"_Привязка = от какой закрытой 5м-свечи считаем стоп_"
    )

    anchor_map = {"high-1": "h1", "low-1": "l1", "high-2": "h2", "low-2": "l2"}
    rows = [
        # Anchor
        [{"text": _ck("high-1", t["anchor"], "High -1"), "callback_data": f"{p}_t_h1"},
         {"text": _ck("low-1", t["anchor"], "Low -1"), "callback_data": f"{p}_t_l1"},
         {"text": _ck("high-2", t["anchor"], "High -2"), "callback_data": f"{p}_t_h2"},
         {"text": _ck("low-2", t["anchor"], "Low -2"), "callback_data": f"{p}_t_l2"}],
        # Activation candles label
        [{"text": "── Активация (свечи 5м) ──", "callback_data": "noop"}],
    ]
    # Candles 3-10 in two rows
    candle_row1 = []
    candle_row2 = []
    for c in range(3, 11):
        btn = {"text": _ck(c, t["activation_candles"], str(c)), "callback_data": f"{p}_t_c{c}"}
        if c <= 6:
            candle_row1.append(btn)
        else:
            candle_row2.append(btn)
    rows.append(candle_row1)
    rows.append(candle_row2)

    # Initial SL ATR
    rows.append([{"text": "── Начальный SL (ATR) ──", "callback_data": "noop"}])
    sl_row = []
    for v in [10, 15, 20, 25, 30]:
        val = v / 10.0
        sl_row.append({"text": _ck(val, t["initial_sl_atr"], f"{val}"), "callback_data": f"{p}_t_is{v}"})
    rows.append(sl_row)

    # Trail buffer ATR
    rows.append([{"text": "── Трейлинг буфер (ATR) ──", "callback_data": "noop"}])
    tr_row = []
    for v in [5, 10, 15, 20]:
        val = v / 10.0
        tr_row.append({"text": _ck(val, t["trail_atr"], f"{val}"), "callback_data": f"{p}_t_ts{v}"})
    rows.append(tr_row)

    rows.append([{"text": "⬅️ Назад", "callback_data": f"{p}_back"}])
    return text, {"inline_keyboard": rows}


def _slm_render_fixed(bank_key, bank_label, bs, _mode_names):
    """Render fixed ATR SL/TP config page."""
    p = _slm_bank_prefix(bank_key)
    f = bs["fixed"]

    text = (
        f"📏 *{bank_label} — Fixed SL/TP*\n\n"
        f"🛡 *Stop-Loss:* {f['sl_atr']} ATR\n"
        f"🎯 *Take-Profit:* {f['tp_atr']} ATR\n\n"
        f"_ATR14 от таймфрейма сигнала (4H/1D)_"
    )

    rows = [
        [{"text": "── Stop-Loss (ATR) ──", "callback_data": "noop"}],
    ]
    sl_row = []
    for v in [10, 15, 20, 25, 30]:
        val = v / 10.0
        sl_row.append({"text": _ck(val, f["sl_atr"], f"{val}"), "callback_data": f"{p}_f_s{v}"})
    rows.append(sl_row)

    rows.append([{"text": "── Take-Profit (ATR) ──", "callback_data": "noop"}])
    tp_row = []
    for v in [20, 30, 40, 50, 60]:
        val = v / 10.0
        tp_row.append({"text": _ck(val, f["tp_atr"], f"{val}"), "callback_data": f"{p}_f_t{v}"})
    rows.append(tp_row)

    rows.append([{"text": "⬅️ Назад", "callback_data": f"{p}_back"}])
    return text, {"inline_keyboard": rows}


def _slm_render_ema(bank_key, bank_label, bs, _mode_names):
    """Render EMA SL config page."""
    p = _slm_bank_prefix(bank_key)
    e = bs["ema"]

    text = (
        f"📈 *{bank_label} — EMA Stop-Loss*\n\n"
        f"🛡 *Начальный SL:* {e['initial_sl_atr']} ATR\n"
        f"📊 *EMA период:* {e['ema_period']}\n"
        f"⏱ *Таймфрейм:* {e['ema_tf']}\n\n"
        f"_Закрытие по пересечению EMA. LONG: свеча ушла под EMA. SHORT: свеча ушла над EMA._"
    )

    rows = [
        [{"text": "── Начальный SL (ATR) ──", "callback_data": "noop"}],
    ]
    sl_row = []
    for v in [10, 15, 20, 25, 30]:
        val = v / 10.0
        sl_row.append({"text": _ck(val, e["initial_sl_atr"], f"{val}"), "callback_data": f"{p}_e_is{v}"})
    rows.append(sl_row)

    rows.append([{"text": "── EMA период ──", "callback_data": "noop"}])
    rows.append([
        {"text": _ck(25, e["ema_period"], "EMA 25"), "callback_data": f"{p}_e_p25"},
        {"text": _ck(50, e["ema_period"], "EMA 50"), "callback_data": f"{p}_e_p50"},
    ])

    rows.append([{"text": "── Таймфрейм ──", "callback_data": "noop"}])
    rows.append([
        {"text": _ck("5m", e["ema_tf"], "5 мин"), "callback_data": f"{p}_e_tf5m"},
        {"text": _ck("15m", e["ema_tf"], "15 мин"), "callback_data": f"{p}_e_tf15m"},
    ])

    rows.append([{"text": "⬅️ Назад", "callback_data": f"{p}_back"}])
    return text, {"inline_keyboard": rows}


async def handle_callback_query(app_session, update):
    """Handle all callback_query (inline button press) events.

    Returns silently if the chat is not allowed.
    """
    cq = update["callback_query"]
    cb_data = cq.get("data", "")
    cq_id = cq.get("id")
    chat_id = cq.get("message", {}).get("chat", {}).get("id")

    if not is_allowed_chat(chat_id):
        return

    cb_lang = _load_langs().get(str(chat_id), "ru") if chat_id else "ru"

    # ------------------------------------------------------------------ #
    # 0. Stop-Loss Settings (slm_ prefix)
    # ------------------------------------------------------------------ #
    if cb_data.startswith("slm_"):
        user_id = cq.get("from", {}).get("id", 0)
        if user_id != ADMIN_ID:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⛔️ Admin only.", "show_alert": True},
            )
            return

        from config import load_sl_settings, save_sl_settings
        settings = load_sl_settings()
        msg_id_cb = cq.get("message", {}).get("message_id")
        toast = ""

        # Helper maps
        _mode_names = {"stopai": "🎯 StopAI", "trailing": "🔄 Trailing",
                       "fixed": "📏 Fixed ATR", "ema": "📈 EMA SL"}
        _anchor_names = {"high-1": "High -1", "low-1": "Low -1",
                         "high-2": "High -2", "low-2": "Low -2"}

        # --- PARSE callback data ---
        parts = cb_data.split("_")  # slm_s, slm_m, slm_s_stopai, slm_s_t_h1, etc.

        # Determine bank
        bank_key = None
        bank_label = ""
        if len(parts) >= 2:
            if parts[1] == "s":
                bank_key = "signals"
                bank_label = "📊 Signals"

        # --- BTC SHIELD TOGGLE: slm_btc ---
        if len(parts) >= 2 and parts[1] == "btc":
            _shield_names = {"off": "ВЫКЛ", "soft": "Soft"}
            current = settings.get("btc_shield", "off")
            # Toggle: off → soft → off
            new_val = "soft" if current == "off" else "off"
            settings["btc_shield"] = new_val
            save_sl_settings(settings)
            toast = f"🅱️ BTC Shield: {_shield_names[new_val]}"

            sig_mode = settings["signals"]["mode"]
            text = (
                f"⚙️ *Настройки стоп-лосса*\n\n"
                f"📊 *Signals:* {_mode_names.get(sig_mode, sig_mode)}\n"
                f"🅱️ *BTC Shield:* {_shield_names[new_val]}\n\n"
                f"Выберите банк для настройки:"
            )
            kb = {"inline_keyboard": [
                [{"text": f"📊 Signals ({_mode_names.get(sig_mode, '?')})", "callback_data": "slm_s"}],
                [{"text": f"🅱️ BTC Shield: {_shield_names[new_val]}", "callback_data": "slm_btc"}]
            ]}
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id, toast)
            return

        if not bank_key:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        bs = settings[bank_key]

        # --- TOP LEVEL: slm_s or slm_m → show mode selection ---
        if len(parts) == 2:
            text, kb = _slm_render_bank_menu(bank_key, bank_label, bs, _mode_names)
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id)
            return

        # --- MODE SELECT: slm_s_stopai, slm_s_trailing, slm_s_fixed, slm_s_ema ---
        action = parts[2] if len(parts) > 2 else ""

        if action in ("stopai", "trailing", "fixed", "ema"):
            bs["mode"] = action
            save_sl_settings(settings)
            toast = f"✅ {bank_label}: {_mode_names[action]}"

            if action == "trailing":
                text, kb = _slm_render_trailing(bank_key, bank_label, bs, _mode_names, _anchor_names)
            elif action == "fixed":
                text, kb = _slm_render_fixed(bank_key, bank_label, bs, _mode_names)
            elif action == "ema":
                text, kb = _slm_render_ema(bank_key, bank_label, bs, _mode_names)
            else:
                text, kb = _slm_render_bank_menu(bank_key, bank_label, bs, _mode_names)

            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id, toast)
            return

        # --- BACK: slm_s_back → back to bank menu ---
        if action == "back":
            text, kb = _slm_render_bank_menu(bank_key, bank_label, bs, _mode_names)
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id)
            return

        # --- BACK TO TOP: slm_s_top ---
        if action == "top":
            sig_mode = settings["signals"]["mode"]
            btc_shield = settings.get("btc_shield", "off")
            _shield_names = {"off": "ВЫКЛ", "soft": "Soft"}
            text = (
                f"⚙️ *Настройки стоп-лосса*\n\n"
                f"📊 *Signals:* {_mode_names.get(sig_mode, sig_mode)}\n"
                f"🅱️ *BTC Shield:* {_shield_names.get(btc_shield, btc_shield)}\n\n"
                f"Выберите банк для настройки:"
            )
            kb = {"inline_keyboard": [
                [{"text": f"📊 Signals ({_mode_names.get(sig_mode, '?')})", "callback_data": "slm_s"}],
                [{"text": f"🅱️ BTC Shield: {_shield_names.get(btc_shield, btc_shield)}", "callback_data": "slm_btc"}]
            ]}
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id)
            return

        # --- TRAILING SUB-SETTINGS: slm_s_t_XX ---
        if action == "t" and len(parts) > 3:
            sub = parts[3]
            t = bs["trailing"]

            # Anchor: h1, l1, h2, l2
            if sub in ("h1", "l1", "h2", "l2"):
                anchor_map = {"h1": "high-1", "l1": "low-1", "h2": "high-2", "l2": "low-2"}
                t["anchor"] = anchor_map[sub]
                toast = f"Привязка: {_anchor_names[t['anchor']]}"

            # Activation candles: c3..c10
            elif sub.startswith("c") and sub[1:].isdigit():
                t["activation_candles"] = int(sub[1:])
                toast = f"Активация: {t['activation_candles']} свечей"

            # Initial SL ATR: is10 = 1.0, is15 = 1.5, is20 = 2.0, etc.
            elif sub.startswith("is"):
                val = int(sub[2:]) / 10.0
                t["initial_sl_atr"] = val
                toast = f"Начальный SL: {val} ATR"

            # Trail ATR: ts05 = 0.5, ts10 = 1.0, etc.
            elif sub.startswith("ts"):
                val = int(sub[2:]) / 10.0
                t["trail_atr"] = val
                toast = f"Трейлинг: {val} ATR"

            save_sl_settings(settings)
            text, kb = _slm_render_trailing(bank_key, bank_label, bs, _mode_names, _anchor_names)
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id, toast)
            return

        # --- FIXED SUB-SETTINGS: slm_s_f_XX ---
        if action == "f" and len(parts) > 3:
            sub = parts[3]
            f = bs["fixed"]

            # SL ATR: s10 = 1.0, s15 = 1.5, etc.
            if sub.startswith("s") and sub[1:].isdigit():
                val = int(sub[1:]) / 10.0
                f["sl_atr"] = val
                toast = f"SL: {val} ATR"

            # TP ATR: t20 = 2.0, t30 = 3.0, etc.
            elif sub.startswith("t") and sub[1:].isdigit():
                val = int(sub[1:]) / 10.0
                f["tp_atr"] = val
                toast = f"TP: {val} ATR"

            save_sl_settings(settings)
            text, kb = _slm_render_fixed(bank_key, bank_label, bs, _mode_names)
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id, toast)
            return

        # --- EMA SUB-SETTINGS: slm_s_e_XX ---
        if action == "e" and len(parts) > 3:
            sub = parts[3]
            e = bs["ema"]

            # Initial SL ATR
            if sub.startswith("is"):
                val = int(sub[2:]) / 10.0
                e["initial_sl_atr"] = val
                toast = f"Начальный SL: {val} ATR"

            # EMA period: p25, p50
            elif sub.startswith("p") and sub[1:].isdigit():
                e["ema_period"] = int(sub[1:])
                toast = f"EMA период: {e['ema_period']}"

            # EMA timeframe: tf5m, tf15m
            elif sub.startswith("tf"):
                e["ema_tf"] = sub[2:]
                toast = f"EMA таймфрейм: {e['ema_tf']}"

            save_sl_settings(settings)
            text, kb = _slm_render_ema(bank_key, bank_label, bs, _mode_names)
            await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id, toast)
            return

        # Fallback: just ack
        await app_session.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id},
        )
        return

    # ------------------------------------------------------------------ #
    # 0c. Line 4H Settings (l4h_ prefix)
    # ------------------------------------------------------------------ #
    if cb_data.startswith("l4h_"):
        user_id = cq.get("from", {}).get("id", 0)
        if user_id != ADMIN_ID:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⛔️ Admin only.", "show_alert": True},
            )
            return

        if cb_data == "l4h_noop":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        from config import load_line_4h_settings, save_line_4h_settings
        from core.tg_state import set_line4h_input_state

        s = load_line_4h_settings()

        # --- Mode switches ---
        if cb_data == "l4h_standard":
            s["mode"] = "standard"
            save_line_4h_settings(s)
        elif cb_data == "l4h_custom":
            s["mode"] = "custom"
            save_line_4h_settings(s)

        # --- Anchor switches ---
        elif cb_data == "l4h_anc_line":
            s["anchor"] = "nearest_line"
            s["mode"] = "custom"
            save_line_4h_settings(s)
        elif cb_data == "l4h_anc_candle":
            s["anchor"] = "candle_top"
            s["mode"] = "custom"
            save_line_4h_settings(s)

        # --- Range input ---
        elif cb_data == "l4h_range":
            set_line4h_input_state(chat_id, "awaiting_range_pct")
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "📐 *Допуск поиска линий от якоря*\n\n"
                "Введите процент (например: `10`, `20`, `23.55`):",
                {"inline_keyboard": []}, cq_id, toast="✏️ Введите %")
            return

        # --- Point B rules ---
        elif cb_data == "l4h_pb_nochange":
            s["point_b_rule"] = "no_change"
            s["mode"] = "custom"
            save_line_4h_settings(s)
        elif cb_data == "l4h_pb_nearest":
            s["point_b_rule"] = "nearest"
            s["mode"] = "custom"
            save_line_4h_settings(s)
        elif cb_data == "l4h_pb_pct":
            set_line4h_input_state(chat_id, "awaiting_point_b_pct")
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "📐 *Допуск для точки Б (>80 свечей)*\n\n"
                "Введите процент (например: `5`, `10`, `15.5`):",
                {"inline_keyboard": []}, cq_id, toast="✏️ Введите %")
            return

        # --- Render current state ---
        s = load_line_4h_settings()  # reload after changes
        _mode = s["mode"]
        if _mode == "standard":
            text = (
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

            text = (
                f"📐 *Настройки линий 4Ч*\n\n"
                f"Режим: ✅ Пользовательский\n\n"
                f"1️⃣ Якорь: `{_anc_label}`\n"
                f"2️⃣ Допуск: `{_rpct}%`\n"
                f"3️⃣ Точка Б >80 свечей: `{_pb_label}`"
            )

            # Anchor buttons
            _anc_row = [
                {"text": ("✅ " if _anc == "nearest_line" else "") + "Ближняя линия",
                 "callback_data": "l4h_anc_line"},
                {"text": ("✅ " if _anc == "candle_top" else "") + "Верх тела свечи",
                 "callback_data": "l4h_anc_candle"},
            ]

            # Point B buttons
            _pb_row1_text = "Строим самую длинную линию.\nЕсли точка Б >80 свечей от текущей:"
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

        await _slm_edit(app_session, chat_id, msg_id_cb, text, kb, cq_id)
        return

    # ------------------------------------------------------------------ #
    # 0b. SMC Settings (smc_ prefix)
    # ------------------------------------------------------------------ #
    if cb_data.startswith("smc_"):
        user_id = cq.get("from", {}).get("id", 0)
        if user_id != ADMIN_ID:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⛔️ Admin only.", "show_alert": True},
            )
            return

        from config import load_smc_settings, save_smc_settings
        s = load_smc_settings()
        msg_id_cb = cq.get("message", {}).get("message_id")
        toast = ""

        # --- Apply action ---
        if cb_data == "smc_tview":
            s["strict_luxalgo"] = True
            save_smc_settings(s)
            toast = "📺 TradingView"
        elif cb_data == "smc_alisa":
            s["strict_luxalgo"] = False
            save_smc_settings(s)
            toast = "🤖 AIAlisa"
        # Internal OB count
        elif cb_data.startswith("smc_iob_"):
            val = int(cb_data.split("_")[-1])
            s["internal_obs"] = val
            save_smc_settings(s)
            toast = f"Internal OB: {'OFF' if val == 0 else val}"
        # Swing OB count
        elif cb_data.startswith("smc_sob_"):
            val = int(cb_data.split("_")[-1])
            s["swing_obs"] = val
            save_smc_settings(s)
            toast = f"Swing OB: {'OFF' if val == 0 else val}"
        # Internal size (AIAlisa mode only)
        elif cb_data.startswith("smc_isz_"):
            val = int(cb_data.split("_")[-1])
            s["internal_size"] = val
            save_smc_settings(s)
            toast = f"Internal Size: {val}"

        # --- Render menu ---
        strict = s.get("strict_luxalgo", True)
        iob = s.get("internal_obs", 5)
        sob = s.get("swing_obs", 0)
        isz = s.get("internal_size", 5)

        mode_name = "📺 TradingView (LuxAlgo)" if strict else "🤖 AIAlisa (кастом)"

        def _ob_label(val):
            return "OFF" if val == 0 else str(val)

        _isz_hint = {3: "чувствительный", 5: "стандарт", 7: "крупные"}

        if strict:
            mode_desc = "_📺 Точная копия LuxAlgo (internal\\_size=5)_"
        else:
            mode_desc = f"_🤖 Кастомный (internal\\_size={isz} — {_isz_hint.get(isz, '')})_"

        msg_text = (
            f"📐 *Настройки SMC*\n\n"
            f"*Режим:* {mode_name}\n"
            f"*Internal OB:* {_ob_label(iob)}\n"
            f"*Swing OB:* {_ob_label(sob)}\n"
        )
        if not strict:
            msg_text += f"*Internal Size:* {isz} ({_isz_hint.get(isz, '')})\n"
        msg_text += f"\n{mode_desc}"

        def _ck_v(current, val, label):
            return f"✅ {label}" if current == val else label

        rows = [
            # Row 1: Mode
            [
                {"text": _ck_v(strict, True, "📺 TView"), "callback_data": "smc_tview"},
                {"text": _ck_v(strict, False, "🤖 AIAlisa"), "callback_data": "smc_alisa"},
            ],
            # Row 2: Internal OB label
            [{"text": "── Internal блоки ──", "callback_data": "smc_noop"}],
            # Row 3: Internal OB options
            [
                {"text": _ck_v(iob, 0, "OFF"), "callback_data": "smc_iob_0"},
                {"text": _ck_v(iob, 3, "3"), "callback_data": "smc_iob_3"},
                {"text": _ck_v(iob, 5, "5"), "callback_data": "smc_iob_5"},
                {"text": _ck_v(iob, 10, "10"), "callback_data": "smc_iob_10"},
            ],
            # Row 4: Swing OB label
            [{"text": "── Swing блоки ──", "callback_data": "smc_noop"}],
            # Row 5: Swing OB options
            [
                {"text": _ck_v(sob, 0, "OFF"), "callback_data": "smc_sob_0"},
                {"text": _ck_v(sob, 3, "3"), "callback_data": "smc_sob_3"},
                {"text": _ck_v(sob, 5, "5"), "callback_data": "smc_sob_5"},
                {"text": _ck_v(sob, 10, "10"), "callback_data": "smc_sob_10"},
            ],
        ]
        # Show internal_size selector only in AIAlisa mode
        if not strict:
            rows.append([{"text": "── Чувствительность ──", "callback_data": "smc_noop"}])
            rows.append([
                {"text": _ck_v(isz, 3, "3 🔬"), "callback_data": "smc_isz_3"},
                {"text": _ck_v(isz, 5, "5 ⚖️"), "callback_data": "smc_isz_5"},
                {"text": _ck_v(isz, 7, "7 🏗️"), "callback_data": "smc_isz_7"},
            ])
        kb = {"inline_keyboard": rows}
        await _slm_edit(app_session, chat_id, msg_id_cb, msg_text, kb, cq_id, toast)
        return

    # ------------------------------------------------------------------ #
    # MANUAL TRENDLINE ALERT CALLBACKS (malert_ prefix)
    # ------------------------------------------------------------------ #
    if cb_data.startswith("malert_"):
        from core.tg_state import get_manual_alert_state, set_manual_alert_state, clear_manual_alert_state

        msg_id = cq.get("message", {}).get("message_id")
        track_alert_msg(chat_id, msg_id)  # track the callback message for cleanup
        ma_state = get_manual_alert_state(chat_id)

        # Answer callback immediately
        await app_session.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id, "text": "✅"},
        )

        # --- Timezone selection ---
        if cb_data.startswith("malert_tz_"):
            from config import set_user_tz_offset
            tz_val = int(cb_data.replace("malert_tz_", ""))
            set_user_tz_offset(chat_id, tz_val)
            label = f"UTC{tz_val:+d}" if tz_val != 0 else "UTC±0"
            await send_response(app_session, chat_id,
                f"✅ Часовой пояс установлен: *{label}*\n"
                f"Теперь все даты в алертах вводятся по этому поясу.",
                parse_mode="Markdown")
            return

        # --- Back button → return to price input ---
        if cb_data == "malert_back":
            if ma_state:
                mode = ma_state.get('mode', 'high')
                ma_state['step'] = 'awaiting_prices'
                set_manual_alert_state(chat_id, ma_state)
                bot_msg = await send_response(app_session, chat_id,
                    "⬅️ Введи две цены заново:", parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
            else:
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
            return

        # --- Timeframe selection → show 5 mode buttons ---
        if cb_data.startswith("malert_tf_"):
            if not ma_state or ma_state.get('step') != 'awaiting_tf':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            tf_val = cb_data.replace("malert_tf_", "").upper()
            tf_map_cb = {"15M": "15m", "1H": "1H", "4H": "4H", "1D": "1D"}
            tf_label = tf_map_cb.get(tf_val, "4H")
            ma_state['tf'] = tf_label
            ma_state['step'] = 'awaiting_mode'
            set_manual_alert_state(chat_id, ma_state)
            short_sym = ma_state['symbol'].replace("USDT", "")
            mode_kb = {"inline_keyboard": [
                [{"text": "🕯 Верх тел", "callback_data": "malert_mode_body_top"},
                 {"text": "🕯 Низ тел", "callback_data": "malert_mode_body_bot"}],
                [{"text": "📅 По датам", "callback_data": "malert_mode_date"}],
                [{"text": "📈 High", "callback_data": "malert_mode_high"},
                 {"text": "📉 Low", "callback_data": "malert_mode_low"}],
            ]}
            bot_msg = await send_response(app_session, chat_id,
                f"⏱ *${short_sym}* — {tf_label}\n\nКак строить линию?",
                reply_markup=mode_kb, parse_mode="Markdown")
            track_alert_msg(chat_id, bot_msg)
            return

        # --- Mode selection ---
        if cb_data.startswith("malert_mode_"):
            if not ma_state or ma_state.get('step') != 'awaiting_mode':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            mode = cb_data.replace("malert_mode_", "")

            if mode == "date":
                # По датам → show body vs wick choice
                ma_state['step'] = 'awaiting_date_type'
                set_manual_alert_state(chat_id, ma_state)
                date_type_kb = {"inline_keyboard": [
                    [{"text": "🕯 По телам", "callback_data": "malert_datetype_body"},
                     {"text": "📊 По теням", "callback_data": "malert_datetype_wick"}],
                ]}
                bot_msg = await send_response(app_session, chat_id,
                    "📅 *По датам*\n\nПо телам или теням свечей?",
                    reply_markup=date_type_kb, parse_mode="Markdown")
                track_alert_msg(chat_id, bot_msg)
                return

            # high / low / body_top / body_bot → ask for prices
            ma_state['mode'] = mode
            ma_state['step'] = 'awaiting_prices'
            set_manual_alert_state(chat_id, ma_state)

            mode_names = {
                "high": "📈 High (тени вверх)",
                "low": "📉 Low (тени вниз)",
                "body_top": "🕯 Верх тел (max O/C)",
                "body_bot": "🕯 Низ тел (min O/C)",
            }
            bot_msg = await send_response(app_session, chat_id,
                f"✅ Режим: *{mode_names.get(mode, mode)}*\n\n"
                f"Введи две цены через пробел:\n"
                f"Например: `69500 67200`",
                parse_mode="Markdown")
            track_alert_msg(chat_id, bot_msg)
            return

        # --- Date type: body vs wick ---
        if cb_data.startswith("malert_datetype_"):
            if not ma_state or ma_state.get('step') != 'awaiting_date_type':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            date_type = cb_data.replace("malert_datetype_", "")  # "body" or "wick"
            ma_state['step'] = 'awaiting_date_mode'
            set_manual_alert_state(chat_id, ma_state)
            if date_type == "body":
                date_sub_kb = {"inline_keyboard": [
                    [{"text": "⬆️ Верх тел", "callback_data": "malert_date_body_top"},
                     {"text": "⬇️ Низ тел", "callback_data": "malert_date_body_bot"}],
                ]}
                bot_msg = await send_response(app_session, chat_id,
                    "🕯 *По телам свечей*\n\nВерх или низ тел?",
                    reply_markup=date_sub_kb, parse_mode="Markdown")
            else:  # wick
                date_sub_kb = {"inline_keyboard": [
                    [{"text": "📈 High", "callback_data": "malert_date_high"},
                     {"text": "📉 Low", "callback_data": "malert_date_low"}],
                ]}
                bot_msg = await send_response(app_session, chat_id,
                    "📊 *По теням свечей*\n\nHigh или Low?",
                    reply_markup=date_sub_kb, parse_mode="Markdown")
            track_alert_msg(chat_id, bot_msg)
            return

        # --- Date sub-mode (body top/bot, high, low) → ask for dates ---
        if cb_data.startswith("malert_date_"):
            if not ma_state or ma_state.get('step') != 'awaiting_date_mode':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            date_mode = cb_data.replace("malert_", "")  # "date_body_top", "date_body_bot", "date_high", "date_low"
            ma_state['date_mode'] = date_mode
            ma_state['step'] = 'awaiting_date_a'
            set_manual_alert_state(chat_id, ma_state)
            mode_labels = {
                "date_body_top": "верх тел",
                "date_body_bot": "низ тел",
                "date_high": "High (тени)",
                "date_low": "Low (тени)",
                "date_top": "верх тел",       # legacy
                "date_bottom": "низ тел",     # legacy
            }
            body_label = mode_labels.get(date_mode, date_mode)
            from config import get_user_tz_offset
            tz_off = get_user_tz_offset(chat_id)
            tz_label = f"UTC{tz_off:+d}" if tz_off else "UTC"
            bot_msg = await send_response(app_session, chat_id,
                f"📅 Режим: *{body_label}* ({tz_label})\n\n"
                f"📍 Точка A — введи дату и время:\n"
                f"Примеры: `10.06.26 04:45`\n"
                f"С процентом: `10.06.26 04:45 +2%` или `10.06.26 04:45+2%`",
                parse_mode="Markdown")
            track_alert_msg(chat_id, bot_msg)
            return

        # --- Pick point A (duplicate price resolution) ---
        if cb_data.startswith("malert_pick_a_"):
            if not ma_state or ma_state.get('step') != 'picking_point_a':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            # Parse: malert_pick_a_{idx}_{price}
            parts = cb_data.replace("malert_pick_a_", "").split("_", 1)
            chosen_idx = int(parts[0])
            chosen_price = float(parts[1])
            ma_state['chosen_a_idx'] = chosen_idx
            ma_state['chosen_a_price'] = chosen_price

            # Check if point B also needs picking
            matches_b = ma_state.get('matches_b', [])
            if len(matches_b) > 1:
                from core.tg_commands import _finalize_manual_alert
                buttons = []
                for m in matches_b:
                    idx, price, time_ms = m[0], m[1], m[2]
                    from datetime import datetime as _dt, timezone as _tz
                    dt = _dt.fromtimestamp(time_ms / 1000, tz=_tz.utc)
                    label = f"{dt.strftime('%d.%m.%y %H:%M')} — {price}"
                    buttons.append([{"text": label, "callback_data": f"malert_pick_b_{idx}_{price}"}])
                ma_state['step'] = 'picking_point_b'
                set_manual_alert_state(chat_id, ma_state)
                exact_b = ma_state.get('exact_b', True)
                title = "точной" if exact_b else "ближайшей"
                bot_msg = await send_response(app_session, chat_id,
                    f"✅ Точка A выбрана.\n\n"
                    f"🔍 Для точки B найдено несколько совпадений {title} цены.\nВыбери свечу:",
                    reply_markup={"inline_keyboard": buttons})
                track_alert_msg(chat_id, bot_msg)
                return
            else:
                # B is single — finalize
                if matches_b:
                    ma_state['chosen_b_idx'] = matches_b[0][0]
                    ma_state['chosen_b_price'] = matches_b[0][1]
                ma_state['step'] = 'awaiting_prices'  # reset for _finalize
                set_manual_alert_state(chat_id, ma_state)
                from core.tg_commands import _finalize_manual_alert_with_indices
                await _finalize_manual_alert_with_indices(app_session, chat_id, msg_id, ma_state)
                return

        # --- Pick point B (duplicate price resolution) ---
        if cb_data.startswith("malert_pick_b_"):
            if not ma_state or ma_state.get('step') != 'picking_point_b':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            parts = cb_data.replace("malert_pick_b_", "").split("_", 1)
            ma_state['chosen_b_idx'] = int(parts[0])
            ma_state['chosen_b_price'] = float(parts[1])
            set_manual_alert_state(chat_id, ma_state)
            from core.tg_commands import _finalize_manual_alert_with_indices
            await _finalize_manual_alert_with_indices(app_session, chat_id, msg_id, ma_state)
            return

        # --- Delete specific alert ---
        if cb_data.startswith("malert_del_"):
            try:
                del_idx = int(cb_data.replace("malert_del_", ""))
                from config import load_manual_alerts, save_manual_alerts
                alerts = load_manual_alerts()
                if 0 <= del_idx < len(alerts):
                    removed = alerts.pop(del_idx)
                    save_manual_alerts(alerts)
                    short = removed["symbol"].replace("USDT", "")
                    await send_response(app_session, chat_id,
                        f"✅ Линия удалена: `${short}` {removed['tf']}", parse_mode="Markdown")
                else:
                    await send_response(app_session, chat_id, "⚠️ Алерт не найден (возможно уже удалён).")
            except Exception as e:
                logging.error(f"❌ malert_del error: {repr(e)}")
                await send_response(app_session, chat_id, "❌ Ошибка при удалении.")
            return

        return  # unknown malert_ callback

    # ------------------------------------------------------------------ #
    # 0d. SIGNALS CALLBACKS (sig_)
    # ------------------------------------------------------------------ #
    if cb_data.startswith("sig_"):
        msg_id_cb = cq.get("message", {}).get("message_id")

        # --- sig_close: snapshot (close all) ---
        if cb_data == "sig_close":
            try:
                from core.tg_reports import build_signals_close_text
                chunks = await build_signals_close_text(app_session, lang="ru")
                # Edit first message with snapshot, send rest as new
                if chunks:
                    kb = {"inline_keyboard": [
                        [{"text": "⬅️ Назад к банку", "callback_data": "sig_back"}],
                    ]}
                    await _slm_edit(app_session, chat_id, msg_id_cb,
                        chunks[0], {"inline_keyboard": []}, cq_id, "🔒 Снапшот")
                    for chunk in chunks[1:]:
                        await send_response(app_session, chat_id, chunk, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ sig_close error: {e}")
                await app_session.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                    json={"callback_query_id": cq_id, "text": f"❌ {e}"})
            return

        # --- sig_clear_ask: confirm bank reset ---
        if cb_data == "sig_clear_ask":
            kb = {"inline_keyboard": [
                [
                    {"text": "✅ Да, сбросить", "callback_data": "sig_clear_yes"},
                    {"text": "❌ Нет", "callback_data": "sig_back"},
                ],
            ]}
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🔄 *Сбросить банк на $10,000?*\n\n"
                "⚠️ All-time статистика будет обнулена\n"
                "📋 Сегодняшние сигналы останутся в списке",
                kb, cq_id)
            return

        # --- sig_clear_yes: actually reset ---
        if cb_data == "sig_clear_yes":
            from config import reset_virtual_bank
            reset_virtual_bank()
            kb = {"inline_keyboard": [
                [{"text": "⬅️ Назад к банку", "callback_data": "sig_back"}],
            ]}
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🔄 *Банк сброшен!*\n\n"
                "💰 Баланс: `$10,000.00`\n"
                "📊 All-time статистика обнулена\n"
                "📋 Сегодняшние сигналы остались в списке",
                kb, cq_id, "🔄 Сброшено")
            return

        # --- sig_back: back to signals main ---
        if cb_data == "sig_back":
            try:
                from core.tg_reports import build_signals_text
                chunks = await build_signals_text(app_session, lang="ru")
                kb = {"inline_keyboard": [
                    [
                        {"text": "🔒 Снапшот", "callback_data": "sig_close"},
                        {"text": "🔄 Сбросить банк", "callback_data": "sig_clear_ask"},
                    ],
                    [
                        {"text": "📊 Все пробития", "callback_data": "sig_trend_all"},
                        {"text": "📈 Растущие", "callback_data": "sig_trend_up"},
                    ],
                ]}
                if chunks:
                    await _slm_edit(app_session, chat_id, msg_id_cb,
                        chunks[0], kb, cq_id)
                    for chunk in chunks[1:]:
                        await send_response(app_session, chat_id, chunk, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"❌ sig_back error: {e}")
                await app_session.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                    json={"callback_query_id": cq_id, "text": f"❌ {e}"})
            return

        # --- sig_trend_all: all breakouts ---
        # --- sig_trend_up: only growing (price > breakout price) ---
        if cb_data in ("sig_trend_all", "sig_trend_up"):
            only_up = (cb_data == "sig_trend_up")
            from config import load_breakout_log
            from core.categories import get_sector_emoji
            log = load_breakout_log()
            if not log:
                kb = {"inline_keyboard": [
                    [{"text": "⬅️ Назад к банку", "callback_data": "sig_back"}],
                ]}
                await _slm_edit(app_session, chat_id, msg_id_cb,
                    "📭 Нет пробитий с последнего скана.",
                    kb, cq_id)
                return

            price_map = {}
            try:
                async with app_session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                    if resp.status == 200:
                        for p in await resp.json():
                            price_map[p["symbol"]] = float(p["price"])
            except Exception:
                pass

            hdr = "📈 *Растущие пробития:*\n" if only_up else "📊 *Пробития трендовых линий:*\n"
            lines = [hdr]
            count = 0
            for entry in log:
                sym = entry["symbol"]
                short = sym.replace("USDT", "")
                _sec_em = get_sector_emoji(sym)
                short_display = f"{_sec_em}{short}"
                tf = entry["tf"]
                bp = entry["breakout_price"]
                now_price = price_map.get(sym, entry.get("current_price", 0))
                diff_pct = ((now_price / bp) - 1) * 100 if bp > 0 else 0

                if only_up and diff_pct < 0:
                    continue

                arrow = "🟢" if diff_pct >= 0 else "🔴"
                ai_dir = entry.get("ai_direction", "")
                ai_mark = ""
                if ai_dir:
                    ai_ok = (ai_dir == "LONG" and diff_pct >= 0) or (ai_dir == "SHORT" and diff_pct < 0)
                    ai_mark = "✅" if ai_ok else "❌"

                lines.append(
                    f"{arrow}{ai_mark} `${short_display}` ({tf})\n"
                    f"    Пробитие: `${bp:.6f}`\n"
                    f"    Сейчас: `${now_price:.6f}` (*{diff_pct:+.2f}%*)"
                )
                count += 1

            if count == 0:
                lines = ["📈 *Растущие пробития:*\n\n📭 Нет растущих монет"]

            lines.append(f"\n_Всего: {count} монет_")

            full = "\n".join(lines)
            kb = {"inline_keyboard": [
                [
                    {"text": "📊 Все пробития", "callback_data": "sig_trend_all"},
                    {"text": "📈 Растущие", "callback_data": "sig_trend_up"},
                ],
                [{"text": "⬅️ Назад к банку", "callback_data": "sig_back"}],
            ]}

            if len(full) <= 4000:
                await _slm_edit(app_session, chat_id, msg_id_cb,
                    full, kb, cq_id)
            else:
                # Too long for edit — send as new message, edit original to minimal
                await _slm_edit(app_session, chat_id, msg_id_cb,
                    hdr + f"\n_Всего: {count} монет (см. ниже)_",
                    kb, cq_id)
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
                for chunk in chunks:
                    await send_response(app_session, chat_id, chunk, parse_mode="Markdown")
            return

        # Fallback for unrecognized sig_ callbacks
        await app_session.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id})
        return

    # ------------------------------------------------------------------ #
    # 0e. SECTOR CALLBACKS (sec_, sflt_)
    # ------------------------------------------------------------------ #
    if cb_data.startswith(("sec_", "sflt_")):
        msg_id_cb = cq.get("message", {}).get("message_id")

        # --- Helper: build main menu keyboard ---
        async def _sec_main_kb():
            counts = get_sector_counts()
            try:
                from core.binance_api import get_usdt_futures_symbols
                _all_syms = await get_usdt_futures_symbols()
                _unk_cnt = len(get_unknown_symbols(_all_syms))
            except Exception:
                _unk_cnt = 0
            rows = []
            row = []
            for i, sector in enumerate(ALL_SECTORS):
                cnt = counts.get(sector, 0)
                short = SECTOR_SHORT.get(sector, sector)
                row.append({"text": f"{short} ({cnt})", "callback_data": f"sec_view_{i}"})
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([{"text": f"❓ Без сектора ({_unk_cnt})", "callback_data": "sec_view_unknown"}])
            rows.append([
                {"text": "➕ Добавить", "callback_data": "sec_btn_add"},
                {"text": "🔄 Переместить", "callback_data": "sec_btn_move"},
            ])
            rows.append([
                {"text": "🔍 Найти", "callback_data": "sec_btn_find"},
                {"text": "🔄 Сверить", "callback_data": "sec_btn_verify"},
            ])
            return rows

        # --- sec_main: back to main menu ---
        if cb_data == "sec_main" or cb_data == "sec_back":
            clear_sector_state(chat_id)
            rows = await _sec_main_kb()
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🏷 *Секторы монет*\n\nНажми на сектор — список монет:",
                {"inline_keyboard": rows}, cq_id)
            return

        # --- sec_view_X: view sector contents ---
        if cb_data.startswith("sec_view_"):
            sub = cb_data.replace("sec_view_", "")
            if sub == "unknown":
                try:
                    from core.binance_api import get_usdt_futures_symbols
                    _all_syms = await get_usdt_futures_symbols()
                    unknown = get_unknown_symbols(_all_syms)
                except Exception:
                    unknown = []
                if unknown:
                    _lines = [f"❓ *Монеты без сектора ({len(unknown)}):*\n"]
                    for s in unknown[:80]:
                        _lines.append(f"`{s.replace('USDT', '')}`")
                    if len(unknown) > 80:
                        _lines.append(f"\n_...и ещё {len(unknown) - 80}_")
                    text = "\n".join(_lines)
                else:
                    text = "✅ Все монеты категоризированы!"
            else:
                try:
                    idx = int(sub)
                    sector = ALL_SECTORS[idx]
                except (ValueError, IndexError):
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id})
                    return
                symbols = get_symbols_by_sector(sector)
                if symbols:
                    _lines = [f"🏷 *{sector} ({len(symbols)}):*\n"]
                    for s in symbols:
                        _lines.append(f"`{s.replace('USDT', '')}`")
                    text = "\n".join(_lines)
                else:
                    text = f"🏷 *{sector}*\n\n📭 Пусто"
            await _slm_edit(app_session, chat_id, msg_id_cb,
                text, {"inline_keyboard": [[{"text": "⬅️ Назад", "callback_data": "sec_main"}]]},
                cq_id)
            return

        # --- sec_btn_add: ask user for coin name to add sector ---
        if cb_data == "sec_btn_add":
            set_sector_state(chat_id, {"action": "add", "step": "awaiting_coin"})
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "➕ *Добавить монету в сектор*\n\nВведите тикер монеты (например `BTC`):",
                {"inline_keyboard": [[{"text": "⬅️ Назад", "callback_data": "sec_main"}]]},
                cq_id)
            return

        # --- sec_btn_move: ask user for coin name to move ---
        if cb_data == "sec_btn_move":
            set_sector_state(chat_id, {"action": "move", "step": "awaiting_coin"})
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🔄 *Переместить монету*\n\nВведите тикер монеты (например `BTC`):",
                {"inline_keyboard": [[{"text": "⬅️ Назад", "callback_data": "sec_main"}]]},
                cq_id)
            return

        # --- sec_btn_find: ask user for coin name to find ---
        if cb_data == "sec_btn_find":
            set_sector_state(chat_id, {"action": "find", "step": "awaiting_coin"})
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🔍 *Найти монету*\n\nВведите тикер монеты (например `SOL`):",
                {"inline_keyboard": [[{"text": "⬅️ Назад", "callback_data": "sec_main"}]]},
                cq_id)
            return

        # --- sec_btn_verify: verify sectors against Binance ---
        if cb_data == "sec_btn_verify":
            try:
                from core.binance_api import get_usdt_futures_symbols
                _all_syms = await get_usdt_futures_symbols()
                unknown = get_unknown_symbols(_all_syms)
            except Exception:
                unknown = []
                _all_syms = []

            categorized = len(_all_syms) - len(unknown) if _all_syms else 0
            if unknown:
                _lines = [f"🔄 *Сверка секторов:*\n"]
                _lines.append(f"⚠️ *Монеты без сектора ({len(unknown)}):*\n")
                for s in unknown[:60]:
                    _lines.append(f"`{s.replace('USDT', '')}`")
                if len(unknown) > 60:
                    _lines.append(f"\n_...и ещё {len(unknown) - 60}_")
                _lines.append(f"\n\n✅ Остальные {categorized} монет в секторах")
                _lines.append("\n_(Монеты в секторах, но не на Binance — не трогаем)_")
                text = "\n".join(_lines)
            else:
                text = f"🔄 *Сверка секторов:*\n\n✅ Все {categorized} монет категоризированы!\n\n_(Монеты в секторах, но не на Binance — не трогаем)_"

            await _slm_edit(app_session, chat_id, msg_id_cb,
                text, {"inline_keyboard": [[{"text": "⬅️ Назад", "callback_data": "sec_main"}]]},
                cq_id)
            return

        # --- sec_add_pick_COIN: redirect from find → add (when coin has no sectors) ---
        if cb_data.startswith("sec_add_pick_"):
            coin_short = cb_data.replace("sec_add_pick_", "")
            symbol = coin_short + "USDT"
            current = get_sectors(symbol)
            rows = []
            row = []
            for i, sector in enumerate(ALL_SECTORS):
                if sector in current:
                    continue
                short_s = SECTOR_SHORT.get(sector, sector)
                row.append({"text": short_s, "callback_data": f"sec_add_{coin_short}_{i}"})
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([{"text": "⬅️ Назад", "callback_data": "sec_main"}])
            current_text = " / ".join(current) if current else "❓ нет"
            await _slm_edit(app_session, chat_id, msg_id_cb,
                f"➕ *Добавить сектор для {coin_short}*\n\nТекущие: {current_text}\n\nВыбери сектор:",
                {"inline_keyboard": rows}, cq_id)
            return

        # --- sec_add_COIN_IDX: pick sector for coin (add flow) ---
        if cb_data.startswith("sec_add_"):
            parts = cb_data.split("_")
            # sec_add_COIN_IDX
            if len(parts) >= 4:
                coin_short = parts[2]
                try:
                    sector_idx = int(parts[3])
                    sector = ALL_SECTORS[sector_idx]
                except (ValueError, IndexError):
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id})
                    return
                symbol = coin_short + "USDT"
                added = add_sector(symbol, sector)
                toast = f"✅ {coin_short} → {sector}" if added else "ℹ️ Уже есть"

                current = get_sectors(symbol)
                current_text = " / ".join(current) if current else "❓ нет"

                # Show "add more" or "done"
                rows = []
                row = []
                for i, s in enumerate(ALL_SECTORS):
                    if s in current:
                        continue
                    short_s = SECTOR_SHORT.get(s, s)
                    row.append({"text": short_s, "callback_data": f"sec_add_{coin_short}_{i}"})
                    if len(row) == 3:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)

                if rows:
                    rows.append([{"text": "✅ Готово", "callback_data": "sec_done"}])
                    text = f"✅ *{coin_short}* → {sector}\n\nТекущие: {current_text}\n\nДобавить ещё сектор?"
                else:
                    text = f"✅ *{coin_short}*\n\nСекторы: {current_text}"
                    rows = [[{"text": "⬅️ В меню", "callback_data": "sec_main"}]]

                await _slm_edit(app_session, chat_id, msg_id_cb,
                    text, {"inline_keyboard": rows}, cq_id, toast)
            return

        # --- sec_done: done adding/moving ---
        if cb_data == "sec_done":
            clear_sector_state(chat_id)
            rows = await _sec_main_kb()
            await _slm_edit(app_session, chat_id, msg_id_cb,
                "🏷 *Секторы монет*\n\nНажми на сектор — список монет:",
                {"inline_keyboard": rows}, cq_id, "✅ Готово")
            return

        # --- sec_mv_rm_COIN_IDX: remove sector from coin (move flow) ---
        if cb_data.startswith("sec_mv_rm_"):
            parts = cb_data.split("_")
            # sec_mv_rm_COIN_IDX
            if len(parts) >= 5:
                coin_short = parts[3]
                try:
                    sector_idx = int(parts[4])
                    sector = ALL_SECTORS[sector_idx]
                except (ValueError, IndexError):
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id})
                    return
                symbol = coin_short + "USDT"
                removed = remove_sector(symbol, sector)
                toast = f"❌ {sector} удалён" if removed else "ℹ️"

                current = get_sectors(symbol)
                if current:
                    rows = []
                    for sec in current:
                        idx = ALL_SECTORS.index(sec) if sec in ALL_SECTORS else 0
                        rows.append([{"text": f"❌ {sec}", "callback_data": f"sec_mv_rm_{coin_short}_{idx}"}])
                    rows.append([{"text": "➕ Добавить новый сектор", "callback_data": f"sec_mv_add_{coin_short}"}])
                    rows.append([{"text": "✅ Готово", "callback_data": "sec_done"}])
                    text = f"🔄 *{coin_short}:*\n\n" + "\n".join([f"• {s}" for s in current])
                else:
                    rows = [
                        [{"text": "➕ Добавить сектор", "callback_data": f"sec_mv_add_{coin_short}"}],
                        [{"text": "✅ Готово", "callback_data": "sec_done"}],
                    ]
                    text = f"🔄 *{coin_short}:*\n\n❓ Нет секторов"
                await _slm_edit(app_session, chat_id, msg_id_cb,
                    text, {"inline_keyboard": rows}, cq_id, toast)
            return

        # --- sec_mv_add_COIN: show sectors to add (move flow) ---
        if cb_data.startswith("sec_mv_add_"):
            coin_short = cb_data.replace("sec_mv_add_", "")
            symbol = coin_short + "USDT"
            current = get_sectors(symbol)
            rows = []
            row = []
            for i, s in enumerate(ALL_SECTORS):
                if s in current:
                    continue
                short_s = SECTOR_SHORT.get(s, s)
                row.append({"text": short_s, "callback_data": f"sec_mv_pick_{coin_short}_{i}"})
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([{"text": "⬅️ Назад", "callback_data": f"sec_mv_back_{coin_short}"}])
            await _slm_edit(app_session, chat_id, msg_id_cb,
                f"➕ *Добавить сектор для {coin_short}:*",
                {"inline_keyboard": rows}, cq_id)
            return

        # --- sec_mv_pick_COIN_IDX: pick sector to add (move flow) ---
        if cb_data.startswith("sec_mv_pick_"):
            parts = cb_data.split("_")
            # sec_mv_pick_COIN_IDX
            if len(parts) >= 5:
                coin_short = parts[3]
                try:
                    sector_idx = int(parts[4])
                    sector = ALL_SECTORS[sector_idx]
                except (ValueError, IndexError):
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id})
                    return
                symbol = coin_short + "USDT"
                add_sector(symbol, sector)
                toast = f"✅ {sector}"

                # Show updated move menu with add-more/done
                current = get_sectors(symbol)
                rows = []
                for sec in current:
                    idx = ALL_SECTORS.index(sec) if sec in ALL_SECTORS else 0
                    rows.append([{"text": f"❌ {sec}", "callback_data": f"sec_mv_rm_{coin_short}_{idx}"}])
                rows.append([{"text": "➕ Добавить ещё сектор", "callback_data": f"sec_mv_add_{coin_short}"}])
                rows.append([{"text": "✅ Готово", "callback_data": "sec_done"}])
                text = f"🔄 *{coin_short}:*\n\n" + "\n".join([f"• {s}" for s in current])
                await _slm_edit(app_session, chat_id, msg_id_cb,
                    text, {"inline_keyboard": rows}, cq_id, toast)
            return

        # --- sec_mv_back_COIN: back to move menu ---
        if cb_data.startswith("sec_mv_back_"):
            coin_short = cb_data.replace("sec_mv_back_", "")
            symbol = coin_short + "USDT"
            current = get_sectors(symbol)
            if current:
                rows = []
                for sec in current:
                    idx = ALL_SECTORS.index(sec) if sec in ALL_SECTORS else 0
                    rows.append([{"text": f"❌ {sec}", "callback_data": f"sec_mv_rm_{coin_short}_{idx}"}])
                rows.append([{"text": "➕ Добавить новый сектор", "callback_data": f"sec_mv_add_{coin_short}"}])
                rows.append([{"text": "✅ Готово", "callback_data": "sec_done"}])
                text = f"🔄 *{coin_short}:*\n\n" + "\n".join([f"• {s}" for s in current])
            else:
                rows = [
                    [{"text": "➕ Добавить сектор", "callback_data": f"sec_mv_add_{coin_short}"}],
                    [{"text": "✅ Готово", "callback_data": "sec_done"}],
                ]
                text = f"🔄 *{coin_short}:*\n\n❓ Нет секторов"
            await _slm_edit(app_session, chat_id, msg_id_cb,
                text, {"inline_keyboard": rows}, cq_id)
            return

        # --- sflt_t_IDX: toggle sector in scan filter ---
        if cb_data.startswith("sflt_t_"):
            try:
                idx = int(cb_data.replace("sflt_t_", ""))
                sector = ALL_SECTORS[idx]
            except (ValueError, IndexError):
                await app_session.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                    json={"callback_query_id": cq_id})
                return
            now_enabled = toggle_scan_sector(sector)
            toast = f"{'✅' if now_enabled else '❌'} {SECTOR_SHORT.get(sector, sector)}"

            settings = load_scan_settings()
            enabled = set(settings.get("enabled_sectors", []))
            scan_unknown = settings.get("scan_unknown", True)
            rows = []
            row = []
            for i, s in enumerate(ALL_SECTORS):
                is_on = s in enabled
                icon = "✅" if is_on else "❌"
                short_s = SECTOR_SHORT.get(s, s)
                row.append({"text": f"{icon} {short_s}", "callback_data": f"sflt_t_{i}"})
                if len(row) == 2:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            unk_icon = "✅" if scan_unknown else "❌"
            rows.append([{"text": f"{unk_icon} ❓ Без сектора", "callback_data": "sflt_unk"}])

            await _slm_edit(app_session, chat_id, msg_id_cb,
                "⚙️ *Фильтр сканера по секторам*\n\n"
                "✅ = сканируется | ❌ = пропускается\n"
                "Нажми чтобы переключить:",
                {"inline_keyboard": rows}, cq_id, toast)
            return

        # --- sflt_unk: toggle scan unknown ---
        if cb_data == "sflt_unk":
            now_on = toggle_scan_unknown()
            toast = f"{'✅' if now_on else '❌'} Без сектора"

            settings = load_scan_settings()
            enabled = set(settings.get("enabled_sectors", []))
            rows = []
            row = []
            for i, s in enumerate(ALL_SECTORS):
                is_on = s in enabled
                icon = "✅" if is_on else "❌"
                short_s = SECTOR_SHORT.get(s, s)
                row.append({"text": f"{icon} {short_s}", "callback_data": f"sflt_t_{i}"})
                if len(row) == 2:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            unk_icon = "✅" if now_on else "❌"
            rows.append([{"text": f"{unk_icon} ❓ Без сектора", "callback_data": "sflt_unk"}])

            await _slm_edit(app_session, chat_id, msg_id_cb,
                "⚙️ *Фильтр сканера по секторам*\n\n"
                "✅ = сканируется | ❌ = пропускается\n"
                "Нажми чтобы переключить:",
                {"inline_keyboard": rows}, cq_id, toast)
            return

        # Fallback for unrecognized sector callbacks
        await app_session.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id})
        return

    # ------------------------------------------------------------------ #
    # 1. Square Integration (Admin check)
    # ------------------------------------------------------------------ #
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
                json={"callback_query_id": cq_id, "text": deny_msg, "show_alert": True},
            )
            return

        post_id = cb_data.replace("sq_", "")
        text_to_post = square_cache_get(post_id)
        if text_to_post:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⏳ Publishing..." if cb_lang == "en" else "⏳ Публикую..."},
            )
            result_msg = await post_to_binance_square(text_to_post)
            await send_response(app_session, chat_id, result_msg)
            square_cache_delete(post_id)
        else:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⚠️ Text is outdated." if cb_lang == "en" else "⚠️ Текст устарел.", "show_alert": True},
            )
        return

    # ------------------------------------------------------------------ #
    # 2. Binance Web3 Skills Buttons
    # ------------------------------------------------------------------ #
    if cb_data.startswith("sk_"):
        await app_session.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id, "text": "⏳ Fetching Web3 data..." if cb_lang == "en" else "⏳ Загружаю Web3 данные..."},
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
        return

    # ------------------------------------------------------------------ #
    # 3. Unified Model Menu (mdm_*) + Legacy Provider Buttons (Admin only)
    # ------------------------------------------------------------------ #
    if cb_data.startswith(("mdm_", "md_", "prov_", "or_md_", "gm_", "gq_", "test_", "back_")):
        if cb_data in ("md_noop", "noop"):
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        user_id = cq.get("from", {}).get("id", 0)
        if user_id != ADMIN_ID:
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⛔️ Admin only.", "show_alert": True},
            )
            return

        from agent.analyzer import get_active_provider_info, set_active_provider, test_provider_key
        from config import GROQ_API_KEYS, GEMINI_API_KEYS, KEY_ACCOUNT_LABELS, load_ai_settings, save_ai_settings
        from core.tg_state import get_model_menu_state, set_model_menu_state, clear_model_menu_state

        msg_id_cb = cq.get("message", {}).get("message_id")

        # === Gemini model map (shared) ===
        _gm_model_map = {
            "3.1pro": "gemini-3.1-pro-preview",
            "3pro": "gemini-3-pro-preview",
            "2.5pro": "gemini-2.5-pro",
            "2.5flash": "gemini-2.5-flash",
            "2.5flashlite": "gemini-2.5-flash-lite",
            "2.0flash": "gemini-2.0-flash",
        }
        _groq_model_map = {
            "l3.3-70b": "llama-3.3-70b-versatile",
            "l3.1-70b": "llama-3.1-70b-versatile",
            "gemma2-27b": "gemma2-27b-it",
            "qwq32b": "qwen-qwq-32b",
        }

        # ----------------------------------------------------------
        # Helper: build main /model menu text + keyboard
        # ----------------------------------------------------------
        def _build_main_menu():
            s = load_ai_settings()
            prov, mdl, ki = get_active_provider_info()
            # Current model
            if s.get("ai_disabled"):
                model_line = "🚫 AI отключён"
            elif prov == "gemini":
                model_line = f"💎 Gemini #{ki+1} / `{mdl}`"
            elif prov == "groq":
                model_line = f"⚡ Groq #{ki+1} / `{mdl}`"
            else:
                model_line = f"🌐 OpenRouter / `{mdl}`"
            # Fallback
            chain = s.get("openrouter_fallback_chain", ["openai/gpt-oss-120b:free", "openrouter/free"])
            if s.get("fallback_disabled"):
                fb_line = "🚫 Без фоллбэка"
            elif chain:
                fb_line = " → ".join([f"`{m}`" for m in chain])
            else:
                fb_line = "🚫 Без фоллбэка"
            # Start of day
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
            txt = (
                f"🧠 *AI Settings*\n\n"
                f"*Текущая модель:* {model_line}\n"
                f"*Фоллбэк:* {fb_line}\n"
                f"*Старт дня (00:00 UTC):* {sd_line}"
            )
            kb = {"inline_keyboard": [
                [{"text": "🔄 Изменить модель", "callback_data": "mdm_model"},
                 {"text": "🔗 Изменить фоллбэк", "callback_data": "mdm_fallback"}],
                [{"text": "🌅 Изменить старт дня", "callback_data": "mdm_startday"},
                 {"text": "📋 Все модели", "callback_data": "mdm_all"}],
            ]}
            return txt, kb

        # ----------------------------------------------------------
        # Helper: build category selection header + keyboard
        # ----------------------------------------------------------
        def _build_category_kb(mode, fallback_chain=None):
            if mode == "fallback":
                chain_text = " → ".join([f"`{m}`" for m in fallback_chain]) if fallback_chain else "_(пусто)_"
                header = f"🔗 *Фоллбэк цепочка:* {chain_text}\nВыбирай модели по очереди:"
            elif mode == "startday":
                header = "🌅 *Старт дня (00:00 UTC)*\nВыбери модель:"
            else:
                header = "🔄 *Изменить текущую модель*\nВыбери модель:"
            from core.tg_state import build_model_category_kb
            kb = build_model_category_kb(mode, fallback_chain)
            return header, kb

        # ----------------------------------------------------------
        # Helper: build Gemini model list keyboard
        # ----------------------------------------------------------
        def _build_gemini_kb(mode):
            s = load_ai_settings()
            ki = s.get("active_key_index", 0) if s.get("active_provider") == "gemini" else 0
            gm_model = s.get("gemini_model", "gemini-2.5-flash")
            key_rows = []
            for i in range(0, min(8, len(GEMINI_API_KEYS)), 2):
                row = []
                for j in [i, i + 1]:
                    if j < len(GEMINI_API_KEYS):
                        marker = "✅ " if j == ki and mode == "model" else ""
                        lbl = KEY_ACCOUNT_LABELS.get(j, f"#{j+1}")
                        row.append({"text": f"{marker}🔑 {lbl}", "callback_data": f"mdm_{mode}_gm_k{j}"})
                key_rows.append(row)
            model_rows = [
                [{"text": "── Модели ──", "callback_data": "noop"}],
                [{"text": "⭐ Gemini 3.1 Pro", "callback_data": f"mdm_{mode}_gm_m_3.1pro"},
                 {"text": "Gemini 3 Pro", "callback_data": f"mdm_{mode}_gm_m_3pro"}],
                [{"text": "Gemini 2.5 Pro", "callback_data": f"mdm_{mode}_gm_m_2.5pro"},
                 {"text": "Gemini 2.5 Flash", "callback_data": f"mdm_{mode}_gm_m_2.5flash"}],
                [{"text": "Gemini 2.5 Flash Lite", "callback_data": f"mdm_{mode}_gm_m_2.5flashlite"},
                 {"text": "Gemini 2.0 Flash", "callback_data": f"mdm_{mode}_gm_m_2.0flash"}],
                [{"text": "🧪 Тест всех ключей", "callback_data": "test_gm"}],
                [{"text": "⬅️ Назад", "callback_data": f"mdm_{mode}"}],
            ]
            txt = f"💎 *Gemini*\nАктивный ключ: #{ki+1}\nМодель: `{gm_model}`"
            return txt, {"inline_keyboard": key_rows + model_rows}

        # ----------------------------------------------------------
        # Helper: build Groq model list keyboard
        # ----------------------------------------------------------
        def _build_groq_kb(mode):
            s = load_ai_settings()
            ki = s.get("active_key_index", 0) if s.get("active_provider") == "groq" else 0
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            key_row = []
            for i in range(min(3, len(GROQ_API_KEYS))):
                marker = "✅ " if i == ki and mode == "model" else ""
                lbl = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                key_row.append({"text": f"{marker}🔑 {lbl}", "callback_data": f"mdm_{mode}_gq_k{i}"})
            kb = {"inline_keyboard": [
                key_row,
                [{"text": "── Модели ──", "callback_data": "noop"}],
                [{"text": "🦙 Llama 3.3 70B", "callback_data": f"mdm_{mode}_gq_m_l3.3-70b"},
                 {"text": "🦙 Llama 3.1 70B", "callback_data": f"mdm_{mode}_gq_m_l3.1-70b"}],
                [{"text": "💎 Gemma2 27B", "callback_data": f"mdm_{mode}_gq_m_gemma2-27b"},
                 {"text": "🔮 Qwen QWQ 32B", "callback_data": f"mdm_{mode}_gq_m_qwq32b"}],
                [{"text": "🧪 Тест ключей", "callback_data": "test_gq"}],
                [{"text": "⬅️ Назад", "callback_data": f"mdm_{mode}"}],
            ]}
            txt = f"⚡ *Groq*\nАктивный ключ: #{ki+1}\nМодель: `{gq_model}`"
            return txt, kb

        # ----------------------------------------------------------
        # Helper: apply model selection based on mode
        # ----------------------------------------------------------
        def _apply_model(mode, provider, model, key_index=0):
            """Apply selected model. Returns confirmation text."""
            s = load_ai_settings()
            if mode == "model":
                set_active_provider(provider, model=model, key_index=key_index)
                if provider == "openrouter":
                    _cfg.OPENROUTER_MODEL = model
                    agent.analyzer.OPENROUTER_MODEL = model
                return f"✅ Модель → `{provider}` / `{model}`"
            elif mode == "startday":
                s["daily_reset_provider"] = provider
                s["daily_reset_model"] = model
                s["daily_reset_key_index"] = key_index
                s["daily_reset_disabled"] = False
                save_ai_settings(s)
                return f"✅ Старт дня → `{provider}` / `{model}`"
            elif mode == "fallback":
                # For fallback, we add to chain (handled separately)
                return ""

        # ----------------------------------------------------------
        # Helper: add model to fallback chain
        # ----------------------------------------------------------
        def _add_to_fallback(chat_id, provider, model):
            st = get_model_menu_state(chat_id) or {"mode": "fallback", "fallback_chain": []}
            chain = st.get("fallback_chain", [])
            # Store as provider-qualified identifier
            if provider == "gemini":
                chain.append(f"gemini:{model}")
            elif provider == "groq":
                chain.append(f"groq:{model}")
            else:
                chain.append(model)  # OpenRouter models already have provider/ prefix
            st["fallback_chain"] = chain
            st["awaiting_text"] = False
            set_model_menu_state(chat_id, st)
            return chain

        # ==========================================================
        # BACK TO MAIN MENU
        # ==========================================================
        if cb_data == "mdm_back":
            clear_model_menu_state(chat_id)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
            return

        # ==========================================================
        # MAIN MENU BUTTONS: model / fallback / startday / all
        # ==========================================================
        if cb_data in ("mdm_model", "mdm_fallback", "mdm_startday"):
            mode = cb_data.replace("mdm_", "")
            if mode == "fallback":
                st = get_model_menu_state(chat_id) or {}
                st["mode"] = "fallback"
                st["fallback_chain"] = st.get("fallback_chain", [])
                st["awaiting_text"] = False
                set_model_menu_state(chat_id, st)
                header, kb = _build_category_kb("fallback", st["fallback_chain"])
            else:
                set_model_menu_state(chat_id, {"mode": mode})
                header, kb = _build_category_kb(mode)
            await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id)
            return

        # --- 📋 All models button ---
        if cb_data == "mdm_all":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "⏳ Загружаю..."},
            )
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
                            is_free = str(pricing.get("prompt", "?")) == "0"
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
            except Exception as e:
                await send_response(app_session, chat_id, f"❌ Ошибка: {e}")
            return

        # ==========================================================
        # CATEGORY BUTTONS: cat_gm / cat_gq / cat_or / paid / none
        # ==========================================================
        # Parse mode from callback: mdm_{mode}_cat_gm, mdm_{mode}_paid, etc.
        for _mode in ("model", "fallback", "startday"):
            prefix = f"mdm_{_mode}_"
            if cb_data.startswith(prefix):
                sub = cb_data[len(prefix):]

                # --- Gemini category ---
                if sub == "cat_gm":
                    txt, kb = _build_gemini_kb(_mode)
                    await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
                    return

                # --- Groq category ---
                if sub == "cat_gq":
                    txt, kb = _build_groq_kb(_mode)
                    await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
                    return

                # --- OpenRouter Free category ---
                if sub == "cat_or":
                    free_models = await _fetch_or_free_models()
                    txt = f"🌐 *OpenRouter Free* ({len(free_models)})"
                    rows = []
                    row = []
                    for fm in free_models:
                        short = fm.split("/")[-1].replace(":free", "")
                        cb_val = f"mdm_{_mode}_or_{fm}"
                        if len(cb_val) > 64:
                            cb_val = f"mdm_{_mode}_or_{fm[:64-len(f'mdm_{_mode}_or_')]}"
                        row.append({"text": short[:30], "callback_data": cb_val})
                        if len(row) == 2:
                            rows.append(row)
                            row = []
                    if row:
                        rows.append(row)
                    rows.append([{"text": "🧪 Тест Free моделей", "callback_data": "test_or"}])
                    rows.append([{"text": "⬅️ Назад", "callback_data": f"mdm_{_mode}"}])
                    await _slm_edit(app_session, chat_id, msg_id_cb, txt, {"inline_keyboard": rows}, cq_id)
                    return

                # --- OpenRouter $$$ (paid / custom input) ---
                if sub == "paid":
                    st = get_model_menu_state(chat_id) or {"mode": _mode}
                    st["awaiting_text"] = True
                    st["mode"] = _mode
                    set_model_menu_state(chat_id, st)
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id, "text": "✏️ Введи model-id"},
                    )
                    await send_response(app_session, chat_id,
                        "✏️ Введи model-id OpenRouter (например: `anthropic/claude-3.5-sonnet`).\n"
                        "Используй 📋 Все модели, чтобы найти нужный id.",
                        parse_mode="Markdown")
                    return

                # --- 🚫 Без модели ---
                if sub == "none":
                    s = load_ai_settings()
                    if _mode == "model":
                        s["ai_disabled"] = True
                        s["active_provider"] = "disabled"
                        save_ai_settings(s)
                        toast = "🚫 AI отключён"
                    elif _mode == "startday":
                        s["daily_reset_disabled"] = True
                        save_ai_settings(s)
                        toast = "🚫 Сброс на старт дня отключён"
                    elif _mode == "fallback":
                        s["fallback_disabled"] = True
                        s["openrouter_fallback_chain"] = []
                        save_ai_settings(s)
                        clear_model_menu_state(chat_id)
                        toast = "🚫 Фоллбэк отключён"
                    txt, kb = _build_main_menu()
                    await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, toast)
                    return

                # --- Gemini key select: gm_k{N} ---
                if sub.startswith("gm_k"):
                    try:
                        ki = int(sub[4:])
                        if 0 <= ki < len(GEMINI_API_KEYS):
                            s = load_ai_settings()
                            gm_model = s.get("gemini_model", "gemini-2.5-flash")
                            if _mode == "fallback":
                                chain = _add_to_fallback(chat_id, "gemini", gm_model)
                                chain_disp = " → ".join([f"`{m}`" for m in chain])
                                header, kb = _build_category_kb("fallback", chain)
                                await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, f"➕ Gemini #{ki+1}")
                            else:
                                confirm = _apply_model(_mode, "gemini", gm_model, ki)
                                s["ai_disabled"] = False
                                save_ai_settings(s)
                                txt, kb = _build_main_menu()
                                await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, confirm)
                    except (ValueError, IndexError):
                        pass
                    return

                # --- Gemini model select: gm_m_{key} ---
                if sub.startswith("gm_m_"):
                    model_key = sub[5:]
                    if model_key in _gm_model_map:
                        new_model = _gm_model_map[model_key]
                        s = load_ai_settings()
                        ki = s.get("active_key_index", 0)
                        if _mode == "fallback":
                            chain = _add_to_fallback(chat_id, "gemini", new_model)
                            header, kb = _build_category_kb("fallback", chain)
                            await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, f"➕ {new_model}")
                        else:
                            confirm = _apply_model(_mode, "gemini", new_model, ki)
                            s["ai_disabled"] = False
                            save_ai_settings(s)
                            txt, kb = _build_main_menu()
                            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, confirm)
                    return

                # --- Groq key select: gq_k{N} ---
                if sub.startswith("gq_k"):
                    try:
                        ki = int(sub[4:])
                        if 0 <= ki < len(GROQ_API_KEYS):
                            s = load_ai_settings()
                            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
                            if _mode == "fallback":
                                chain = _add_to_fallback(chat_id, "groq", gq_model)
                                header, kb = _build_category_kb("fallback", chain)
                                await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, f"➕ Groq #{ki+1}")
                            else:
                                confirm = _apply_model(_mode, "groq", gq_model, ki)
                                s["ai_disabled"] = False
                                save_ai_settings(s)
                                txt, kb = _build_main_menu()
                                await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, confirm)
                    except (ValueError, IndexError):
                        pass
                    return

                # --- Groq model select: gq_m_{key} ---
                if sub.startswith("gq_m_"):
                    model_key = sub[5:]
                    if model_key in _groq_model_map:
                        new_model = _groq_model_map[model_key]
                        s = load_ai_settings()
                        ki = s.get("active_key_index", 0)
                        if _mode == "fallback":
                            chain = _add_to_fallback(chat_id, "groq", new_model)
                            header, kb = _build_category_kb("fallback", chain)
                            await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, f"➕ {new_model}")
                        else:
                            confirm = _apply_model(_mode, "groq", new_model, ki)
                            s["ai_disabled"] = False
                            save_ai_settings(s)
                            txt, kb = _build_main_menu()
                            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, confirm)
                    return

                # --- OpenRouter Free model select: or_{model_id} ---
                if sub.startswith("or_"):
                    new_model = sub[3:]
                    if _mode == "fallback":
                        chain = _add_to_fallback(chat_id, "openrouter", new_model)
                        header, kb = _build_category_kb("fallback", chain)
                        short = new_model.split("/")[-1] if "/" in new_model else new_model
                        await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, f"➕ {short}")
                    else:
                        confirm = _apply_model(_mode, "openrouter", new_model)
                        s = load_ai_settings()
                        s["ai_disabled"] = False
                        save_ai_settings(s)
                        txt, kb = _build_main_menu()
                        await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, confirm)
                    return

        # ==========================================================
        # FALLBACK: Done / Clear
        # ==========================================================
        if cb_data == "mdm_fb_done":
            st = get_model_menu_state(chat_id) or {}
            chain = st.get("fallback_chain", [])
            s = load_ai_settings()
            s["openrouter_fallback_chain"] = chain
            s["fallback_disabled"] = False
            save_ai_settings(s)
            clear_model_menu_state(chat_id)
            txt, kb = _build_main_menu()
            chain_disp = " → ".join(chain) if chain else "пусто"
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ Фоллбэк: {chain_disp}")
            return

        if cb_data == "mdm_fb_clear":
            st = get_model_menu_state(chat_id) or {"mode": "fallback"}
            st["fallback_chain"] = []
            set_model_menu_state(chat_id, st)
            header, kb = _build_category_kb("fallback", [])
            await _slm_edit(app_session, chat_id, msg_id_cb, header, kb, cq_id, "🗑 Цепочка очищена")
            return

        # ==========================================================
        # TEST BUTTONS (shared, mode-independent)
        # ==========================================================
        if cb_data == "test_or":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Тестирую..."},
            )
            await send_response(app_session, chat_id, "🧪 Тестирую OpenRouter Free модели...")
            free_models = await _fetch_or_free_models()
            results = []
            for fm in free_models:
                ok = await test_provider_key("openrouter", _cfg.OPENROUTER_API_KEY, fm)
                status = "✅" if ok else "❌"
                short = fm.split("/")[-1] if "/" in fm else fm
                results.append(f"{status} `{short}`")
            await send_response(app_session, chat_id, "🧪 *OpenRouter результаты:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        if cb_data == "test_gm":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Тестирую Gemini ключи..."},
            )
            s = load_ai_settings()
            gm_model = s.get("gemini_model", "gemini-2.5-flash")
            active_ki = s.get("active_key_index", 0) if s.get("active_provider") == "gemini" else -1
            await send_response(app_session, chat_id, f"🧪 Тестирую {len(GEMINI_API_KEYS)} Gemini ключей (`{gm_model}`)...", parse_mode="Markdown")
            results = []
            for i in range(len(GEMINI_API_KEYS)):
                ok = await test_provider_key("gemini", GEMINI_API_KEYS[i], gm_model)
                status = "✅" if ok else "❌"
                lbl = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                active_mark = " 👈" if i == active_ki else ""
                results.append(f"{status} #{i+1} ({lbl}){active_mark}")
            await send_response(app_session, chat_id, "🧪 *Gemini результаты:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        if cb_data == "test_gq":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Тестирую Groq..."},
            )
            s = load_ai_settings()
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            await send_response(app_session, chat_id, f"🧪 Тестирую Groq ключи (`{gq_model}`)...", parse_mode="Markdown")
            results = []
            for i, key in enumerate(GROQ_API_KEYS):
                ok = await test_provider_key("groq", key, gq_model)
                status = "✅" if ok else "❌"
                lbl = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                results.append(f"{status} #{i+1} ({lbl})")
            await send_response(app_session, chat_id, "🧪 *Groq результаты:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        # ==========================================================
        # LEGACY CALLBACKS (backward compat)
        # ==========================================================
        if cb_data == "back_models":
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
            return

        if cb_data == "prov_or":
            set_model_menu_state(chat_id, {"mode": "model"})
            free_models = await _fetch_or_free_models()
            txt = f"🌐 *OpenRouter Free* ({len(free_models)})"
            rows = []
            row = []
            for fm in free_models:
                short = fm.split("/")[-1].replace(":free", "")
                cb_val = f"mdm_model_or_{fm}"
                if len(cb_val) > 64:
                    cb_val = f"mdm_model_or_{fm[:64-len('mdm_model_or_')]}"
                row.append({"text": short[:30], "callback_data": cb_val})
                if len(row) == 2:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([{"text": "⬅️ Назад", "callback_data": "mdm_back"}])
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, {"inline_keyboard": rows}, cq_id)
            return

        if cb_data == "prov_gm":
            set_model_menu_state(chat_id, {"mode": "model"})
            txt, kb = _build_gemini_kb("model")
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
            return

        if cb_data == "prov_gq":
            set_model_menu_state(chat_id, {"mode": "model"})
            txt, kb = _build_groq_kb("model")
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id)
            return

        if cb_data.startswith("or_md_"):
            new_model = cb_data[6:]
            _apply_model("model", "openrouter", new_model)
            s = load_ai_settings(); s["ai_disabled"] = False; save_ai_settings(s)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ → {new_model}")
            return

        if cb_data.startswith("gm_k"):
            try:
                ki = int(cb_data[4:])
                if 0 <= ki < len(GEMINI_API_KEYS):
                    s = load_ai_settings()
                    gm_model = s.get("gemini_model", "gemini-2.5-flash")
                    _apply_model("model", "gemini", gm_model, ki)
                    s["ai_disabled"] = False; save_ai_settings(s)
                    txt, kb = _build_main_menu()
                    await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ Gemini #{ki+1}")
            except (ValueError, IndexError):
                pass
            return

        _legacy_gm_map = {
            "gm_md_3.1pro": "gemini-3.1-pro-preview", "gm_md_3pro": "gemini-3-pro-preview",
            "gm_md_2.5pro": "gemini-2.5-pro", "gm_md_2.5flash": "gemini-2.5-flash",
            "gm_md_2.5flashlite": "gemini-2.5-flash-lite", "gm_md_2.0flash": "gemini-2.0-flash",
        }
        if cb_data in _legacy_gm_map:
            new_model = _legacy_gm_map[cb_data]
            s = load_ai_settings(); ki = s.get("active_key_index", 0)
            _apply_model("model", "gemini", new_model, ki)
            s["ai_disabled"] = False; save_ai_settings(s)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ {new_model}")
            return

        if cb_data.startswith("gq_k") and len(cb_data) <= 5:
            ki = int(cb_data[4:])
            s = load_ai_settings()
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            _apply_model("model", "groq", gq_model, ki)
            s["ai_disabled"] = False; save_ai_settings(s)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ Groq #{ki+1}")
            return

        _legacy_gq_map = {
            "gq_md_l3.3-70b": "llama-3.3-70b-versatile", "gq_md_l3.1-70b": "llama-3.1-70b-versatile",
            "gq_md_gemma2-27b": "gemma2-27b-it", "gq_md_qwq32b": "qwen-qwq-32b",
        }
        if cb_data in _legacy_gq_map:
            new_model = _legacy_gq_map[cb_data]
            s = load_ai_settings(); ki = s.get("active_key_index", 0)
            _apply_model("model", "groq", new_model, ki)
            s["ai_disabled"] = False; save_ai_settings(s)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ {new_model}")
            return

        if cb_data.startswith("md_"):
            new_model = cb_data[3:]
            _apply_model("model", "openrouter", new_model)
            s = load_ai_settings(); s["ai_disabled"] = False; save_ai_settings(s)
            txt, kb = _build_main_menu()
            await _slm_edit(app_session, chat_id, msg_id_cb, txt, kb, cq_id, f"✅ {new_model}")
            return

        return
