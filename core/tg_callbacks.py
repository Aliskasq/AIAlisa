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
            txt = f"💎 *Gemini* (21 keys)\nActive key: #{ki+1} ({lbl})\nModel: `{gm_model}`"

            kb = {"inline_keyboard": [
                [{"text": "🔑 zhoriha (10)", "callback_data": "gm_acc0"}],
                [{"text": "🔑 alisa (10)", "callback_data": "gm_acc1"}],
                [{"text": "🔑 sudani210 (1)", "callback_data": "gm_acc2"}],
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

        # --- Gemini account group select ---
        if cb_data in ["gm_acc0", "gm_acc1", "gm_acc2"]:
            acc_num = int(cb_data[-1])
            s = agent.analyzer._get_ai_settings()
            active_ki = s.get("active_key_index", 0) if s.get("active_provider") == "gemini" else 0
            gm_model = s.get("gemini_model", "gemini-2.5-flash")

            if acc_num == 0:  # zhoriha account (keys 0-9)
                txt = f"💎 *zhoriha@gmail.com* (10 keys)\nModel: `{gm_model}`"
                key_buttons = []
                for i in range(10):
                    marker = "✅ " if i == active_ki else ""
                    key_buttons.append({"text": f"{marker}Key #{i+1}", "callback_data": f"gm_k{i}"})
                kb_rows = [key_buttons[i:i+5] for i in range(0, len(key_buttons), 5)]
            elif acc_num == 1:  # alisa account (keys 10-19)
                txt = f"💎 *alisasudani@gmail.com* (10 keys)\nModel: `{gm_model}`"
                key_buttons = []
                for i in range(10, 20):
                    marker = "✅ " if i == active_ki else ""
                    key_buttons.append({"text": f"{marker}Key #{i-9}", "callback_data": f"gm_k{i}"})
                kb_rows = [key_buttons[i:i+5] for i in range(0, len(key_buttons), 5)]
            else:  # acc_num == 2, sudani210 account (key 20)
                txt = f"💎 *alisasudani210@gmail.com* (1 key)\nModel: `{gm_model}`"
                marker = "✅ " if 20 == active_ki else ""
                kb_rows = [[{"text": f"{marker}Key #1", "callback_data": "gm_k20"}]]

            kb_rows.append([{"text": "⬅️ Back to Accounts", "callback_data": "prov_gm"}])
            kb = {"inline_keyboard": kb_rows}

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

        # --- Gemini key select (indices 0-20) ---
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
                json={"callback_query_id": cq_id, "text": "🧪 Testing all 21 Gemini keys..."},
            )
            s = agent.analyzer._get_ai_settings()
            gm_model = s.get("gemini_model", "gemini-2.5-flash")
            await send_response(app_session, chat_id, f"🧪 Testing all 21 Gemini keys (model: `{gm_model}`)...", parse_mode="Markdown")

            results = []
            # Account 1: zhoriha (keys 0-9)
            results.append("*zhoriha@gmail.com:*")
            for i in range(10):
                if i < len(GEMINI_API_KEYS):
                    ok = await test_provider_key("gemini", GEMINI_API_KEYS[i], gm_model)
                    status = "✅" if ok else "❌"
                    results.append(f"{status} Key #{i+1}")
                else:
                    results.append(f"❌ Key #{i+1} (missing)")

            # Account 2: alisasudani (keys 10-19)
            results.append("\n*alisasudani@gmail.com:*")
            for i in range(10, 20):
                if i < len(GEMINI_API_KEYS):
                    ok = await test_provider_key("gemini", GEMINI_API_KEYS[i], gm_model)
                    status = "✅" if ok else "❌"
                    results.append(f"{status} Key #{i-9}")
                else:
                    results.append(f"❌ Key #{i-9} (missing)")

            # Account 3: alisasudani210 (key 20)
            results.append("\n*alisasudani210@gmail.com:*")
            if 20 < len(GEMINI_API_KEYS):
                ok = await test_provider_key("gemini", GEMINI_API_KEYS[20], gm_model)
                status = "✅" if ok else "❌"
                results.append(f"{status} Key #1")
            else:
                results.append(f"❌ Key #1 (missing)")

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
