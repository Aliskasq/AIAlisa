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
            await send_response(app_session, chat_id,
                f"⏱ *${short_sym}* — {tf_label}\n\nКак строить линию?",
                reply_markup=mode_kb, parse_mode="Markdown")
            return

        # --- Mode selection ---
        if cb_data.startswith("malert_mode_"):
            if not ma_state or ma_state.get('step') != 'awaiting_mode':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            mode = cb_data.replace("malert_mode_", "")

            if mode == "date":
                # По датам → show body sub-buttons
                ma_state['step'] = 'awaiting_date_mode'
                set_manual_alert_state(chat_id, ma_state)
                date_kb = {"inline_keyboard": [
                    [{"text": "⬆️ Верх тела", "callback_data": "malert_date_top"},
                     {"text": "⬇️ Низ тела", "callback_data": "malert_date_bottom"}],
                ]}
                await send_response(app_session, chat_id,
                    "📅 *По датам*\n\nОт какой части тела свечи строить?",
                    reply_markup=date_kb, parse_mode="Markdown")
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
            await send_response(app_session, chat_id,
                f"✅ Режим: *{mode_names.get(mode, mode)}*\n\n"
                f"Введи две цены через пробел:\n"
                f"Например: `69500 67200`",
                parse_mode="Markdown")
            return

        # --- Date sub-mode (top/bottom body) → ask for dates ---
        if cb_data.startswith("malert_date_"):
            if not ma_state or ma_state.get('step') != 'awaiting_date_mode':
                await send_response(app_session, chat_id, "⚠️ Начни заново: `алерт BTC`", parse_mode="Markdown")
                return
            date_mode = cb_data.replace("malert_", "")  # "date_top" or "date_bottom"
            ma_state['date_mode'] = date_mode
            ma_state['step'] = 'awaiting_date_a'
            set_manual_alert_state(chat_id, ma_state)
            body_label = "верх тела" if date_mode == "date_top" else "низ тела"
            from config import get_user_tz_offset
            tz_off = get_user_tz_offset(chat_id)
            tz_label = f"UTC{tz_off:+d}" if tz_off else "UTC"
            await send_response(app_session, chat_id,
                f"📅 Режим: *{body_label}* ({tz_label})\n\n"
                f"📍 Точка A — введи дату и время:\n"
                f"Примеры: `10.06.26 04:45` или `10 06 26 04 45`",
                parse_mode="Markdown")
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
                await send_response(app_session, chat_id,
                    f"✅ Точка A выбрана.\n\n"
                    f"🔍 Для точки B найдено несколько совпадений {title} цены.\nВыбери свечу:",
                    reply_markup={"inline_keyboard": buttons})
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
    # 3. Model / Provider Selection Buttons (Admin only)
    # ------------------------------------------------------------------ #
    if cb_data.startswith(("md_", "prov_", "or_md_", "gm_", "gq_", "test_", "back_")):
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
        from config import GROQ_API_KEYS, GEMINI_API_KEYS, KEY_ACCOUNT_LABELS

        # --- Back to provider menu ---
        if cb_data == "back_models":
            prov, mdl, ki = get_active_provider_info()
            txt = f"🧠 *AI Provider Selection*\nActive: `{prov}` / `{mdl}`"
            kb = {"inline_keyboard": [
                [{"text": "🌐 OpenRouter", "callback_data": "prov_or"},
                 {"text": "💎 Gemini", "callback_data": "prov_gm"},
                 {"text": "⚡ Groq", "callback_data": "prov_gq"}]
            ]}
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
                json={"chat_id": chat_id, "message_id": cq.get("message", {}).get("message_id"),
                      "text": txt, "parse_mode": "Markdown", "reply_markup": kb},
            )
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        # --- Provider selection: OpenRouter ---
        if cb_data == "prov_or":
            prov, mdl, _ = get_active_provider_info()
            cur = mdl if prov == "openrouter" else agent.analyzer._get_ai_settings().get("openrouter_model", "")
            free_models = await _fetch_or_free_models()
            txt = f"🌐 *OpenRouter Free Models* ({len(free_models)})\nCurrent: `{cur}`"
            rows = [[{"text": "── FREE MODELS ──", "callback_data": "noop"}]]
            row = []
            for fm in free_models:
                short = fm.split("/")[-1].replace(":free", "")
                cb = f"or_md_{fm}" if len(f"or_md_{fm}") <= 64 else f"or_md_{fm[:58]}"
                row.append({"text": short[:30], "callback_data": cb})
                if len(row) == 2:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([{"text": "🧪 Test All Free Models", "callback_data": "test_or"}])
            rows.append([{"text": "⬅️ Back", "callback_data": "back_models"}])
            kb = {"inline_keyboard": rows}
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
                json={"chat_id": chat_id, "message_id": cq.get("message", {}).get("message_id"),
                      "text": txt, "parse_mode": "Markdown", "reply_markup": kb},
            )
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        # --- Provider selection: Gemini ---
        if cb_data == "prov_gm":
            s = agent.analyzer._get_ai_settings()
            ki = s.get("active_key_index", 0) if s.get("active_provider") == "gemini" else 0
            gm_model = s.get("gemini_model", "gemini-2.5-flash")
            lbl = KEY_ACCOUNT_LABELS.get(ki, f"#{ki+1}")
            txt = f"💎 *Gemini* (8 keys)\nActive key: #{ki+1} ({lbl})\nModel: `{gm_model}`"

            # Build key buttons (2 per row)
            key_rows = []
            for i in range(0, min(8, len(GEMINI_API_KEYS)), 2):
                row = []
                for j in [i, i+1]:
                    if j < len(GEMINI_API_KEYS):
                        marker = "✅ " if j == ki else ""
                        row.append({"text": f"{marker}🔑 {KEY_ACCOUNT_LABELS.get(j, f'#{j+1}')}", "callback_data": f"gm_k{j}"})
                key_rows.append(row)

            kb = {"inline_keyboard": key_rows + [
                [{"text": "── Models ──", "callback_data": "noop"}],
                [{"text": "⭐ Gemini 3.1 Pro", "callback_data": "gm_md_3.1pro"},
                 {"text": "Gemini 3 Pro", "callback_data": "gm_md_3pro"}],
                [{"text": "Gemini 2.5 Pro", "callback_data": "gm_md_2.5pro"},
                 {"text": "Gemini 2.5 Flash", "callback_data": "gm_md_2.5flash"}],
                [{"text": "Gemini 2.5 Flash Lite", "callback_data": "gm_md_2.5flashlite"},
                 {"text": "Gemini 2.0 Flash", "callback_data": "gm_md_2.0flash"}],
                [{"text": "🧪 Test All Keys", "callback_data": "test_gm"}],
                [{"text": "⬅️ Back", "callback_data": "back_models"}]
            ]}
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
                json={"chat_id": chat_id, "message_id": cq.get("message", {}).get("message_id"),
                      "text": txt, "parse_mode": "Markdown", "reply_markup": kb},
            )
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        # --- Provider selection: Groq ---
        if cb_data == "prov_gq":
            s = agent.analyzer._get_ai_settings()
            ki = s.get("active_key_index", 0) if s.get("active_provider") == "groq" else 0
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            lbl = KEY_ACCOUNT_LABELS.get(ki, f"#{ki+1}")
            txt = f"⚡ *Groq*\nActive key: #{ki+1} ({lbl})\nModel: `{gq_model}`"
            key_row = []
            for i in range(min(3, len(GROQ_API_KEYS))):
                marker = "✅ " if i == ki else ""
                key_row.append({"text": f"{marker}🔑 {KEY_ACCOUNT_LABELS.get(i, f'#{i+1}')}", "callback_data": f"gq_k{i}"})
            kb = {"inline_keyboard": [
                key_row,
                [{"text": "── Models ──", "callback_data": "noop"}],
                [{"text": "🦙 Llama 3.3 70B", "callback_data": "gq_md_l3.3-70b"},
                 {"text": "🦙 Llama 3.1 70B", "callback_data": "gq_md_l3.1-70b"}],
                [{"text": "💎 Gemma2 27B", "callback_data": "gq_md_gemma2-27b"},
                 {"text": "🔮 Qwen QWQ 32B", "callback_data": "gq_md_qwq32b"}],
                [{"text": "🧪 Test All Keys", "callback_data": "test_gq"}],
                [{"text": "⬅️ Back", "callback_data": "back_models"}]
            ]}
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
                json={"chat_id": chat_id, "message_id": cq.get("message", {}).get("message_id"),
                      "text": txt, "parse_mode": "Markdown", "reply_markup": kb},
            )
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id},
            )
            return

        # --- OpenRouter model select ---
        if cb_data.startswith("or_md_"):
            new_model = cb_data[6:]
            set_active_provider("openrouter", model=new_model)
            agent.analyzer.OPENROUTER_MODEL = new_model
            _cfg.OPENROUTER_MODEL = new_model
            short = new_model.split("/")[-1] if "/" in new_model else new_model
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": f"✅ OpenRouter → {short}"},
            )
            await send_response(app_session, chat_id, f"✅ Provider: *OpenRouter*\nModel: `{new_model}`", parse_mode="Markdown")
            return

        # --- Gemini key select (indices 0-7) ---
        if cb_data.startswith("gm_k"):
            try:
                ki = int(cb_data[3:])  # Support multi-digit indices (gm_k0 .. gm_k20)
                if 0 <= ki < len(GEMINI_API_KEYS):
                    s = agent.analyzer._get_ai_settings()
                    gm_model = s.get("gemini_model", "gemini-2.5-flash")
                    set_active_provider("gemini", model=gm_model, key_index=ki)
                    lbl = KEY_ACCOUNT_LABELS.get(ki, f"#{ki+1}")
                    await app_session.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                        json={"callback_query_id": cq_id, "text": f"✅ Gemini key #{ki+1} ({lbl})"},
                    )
                    await send_response(app_session, chat_id, f"✅ Provider: *Gemini*\nKey: #{ki+1} ({lbl})\nModel: `{gm_model}`", parse_mode="Markdown")
            except (ValueError, IndexError):
                pass
            return

        # --- Gemini model select ---
        _gm_model_map = {
            "gm_md_3.1pro": "gemini-3.1-pro-preview",
            "gm_md_3pro": "gemini-3-pro-preview",
            "gm_md_2.5pro": "gemini-2.5-pro",
            "gm_md_2.5flash": "gemini-2.5-flash",
            "gm_md_2.5flashlite": "gemini-2.5-flash-lite",
            "gm_md_2.0flash": "gemini-2.0-flash",
        }
        if cb_data in _gm_model_map:
            new_model = _gm_model_map[cb_data]
            s = agent.analyzer._get_ai_settings()
            ki = s.get("active_key_index", 0)
            set_active_provider("gemini", model=new_model, key_index=ki)
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": f"✅ Gemini → {new_model}"},
            )
            await send_response(app_session, chat_id, f"✅ Provider: *Gemini*\nModel: `{new_model}`", parse_mode="Markdown")
            return

        # --- Groq key select ---
        if cb_data.startswith("gq_k") and len(cb_data) == 4:
            ki = int(cb_data[3])
            s = agent.analyzer._get_ai_settings()
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            set_active_provider("groq", model=gq_model, key_index=ki)
            lbl = KEY_ACCOUNT_LABELS.get(ki, f"#{ki+1}")
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": f"✅ Groq key #{ki+1} ({lbl})"},
            )
            await send_response(app_session, chat_id, f"✅ Provider: *Groq*\nKey: #{ki+1} ({lbl})\nModel: `{gq_model}`", parse_mode="Markdown")
            return

        # --- Groq model select ---
        _gq_model_map = {
            "gq_md_l3.3-70b": "llama-3.3-70b-versatile",
            "gq_md_l3.1-70b": "llama-3.1-70b-versatile",
            "gq_md_gemma2-27b": "gemma2-27b-it",
            "gq_md_qwq32b": "qwen-qwq-32b",
        }
        if cb_data in _gq_model_map:
            new_model = _gq_model_map[cb_data]
            s = agent.analyzer._get_ai_settings()
            ki = s.get("active_key_index", 0)
            set_active_provider("groq", model=new_model, key_index=ki)
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": f"✅ Groq → {new_model}"},
            )
            await send_response(app_session, chat_id, f"✅ Provider: *Groq*\nModel: `{new_model}`", parse_mode="Markdown")
            return

        # --- Test buttons ---
        if cb_data == "test_or":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Testing free models..."},
            )
            await send_response(app_session, chat_id, "🧪 Testing OpenRouter free models...")
            free_models = await _fetch_or_free_models()
            results = []
            for fm in free_models:
                ok = await test_provider_key("openrouter", _cfg.OPENROUTER_API_KEY, fm)
                status = "✅" if ok else "❌"
                short = fm.split("/")[-1] if "/" in fm else fm
                results.append(f"{status} `{short}`")
            await send_response(app_session, chat_id, "🧪 *OpenRouter Test Results:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        if cb_data == "test_gm":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Testing all 8 Gemini keys..."},
            )
            s = agent.analyzer._get_ai_settings()
            gm_model = s.get("gemini_model", "gemini-2.5-flash")
            active_ki = s.get("active_key_index", 0) if s.get("active_provider") == "gemini" else -1
            await send_response(app_session, chat_id, f"🧪 Testing {len(GEMINI_API_KEYS)} Gemini keys (model: `{gm_model}`)...", parse_mode="Markdown")

            results = []
            for i in range(len(GEMINI_API_KEYS)):
                ok = await test_provider_key("gemini", GEMINI_API_KEYS[i], gm_model)
                status = "✅" if ok else "❌"
                lbl = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                active_mark = " 👈" if i == active_ki else ""
                results.append(f"{status} #{i+1} ({lbl}){active_mark}")

            await send_response(app_session, chat_id, "🧪 *Gemini Test Results:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        if cb_data == "test_gq":
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": "🧪 Testing Groq keys..."},
            )
            s = agent.analyzer._get_ai_settings()
            gq_model = s.get("groq_model", "llama-3.3-70b-versatile")
            await send_response(app_session, chat_id, f"🧪 Testing Groq keys (model: `{gq_model}`)...", parse_mode="Markdown")
            results = []
            for i, key in enumerate(GROQ_API_KEYS):
                ok = await test_provider_key("groq", key, gq_model)
                status = "✅" if ok else "❌"
                lbl = KEY_ACCOUNT_LABELS.get(i, f"#{i+1}")
                results.append(f"{status} Key #{i+1} ({lbl})")
            await send_response(app_session, chat_id, "🧪 *Groq Test Results:*\n\n" + "\n".join(results), parse_mode="Markdown")
            return

        # --- Legacy md_ prefix (backward compat for OpenRouter) ---
        if cb_data.startswith("md_"):
            new_model = cb_data[3:]
            set_active_provider("openrouter", model=new_model)
            agent.analyzer.OPENROUTER_MODEL = new_model
            short_name = new_model.split("/")[-1] if "/" in new_model else new_model
            await app_session.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                json={"callback_query_id": cq_id, "text": f"✅ Switched to {short_name}"},
            )
            await send_response(app_session, chat_id, f"✅ AI Engine changed to:\n`{new_model}`", parse_mode="Markdown")
            return

        return
