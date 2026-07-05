"""
tg_state.py — State management, constants, auth, and Telegram send helpers.
Extracted from tg_listener.py during refactor.
"""
import asyncio
import aiohttp
import logging
import os
import json
from config import BOT_TOKEN, GROUP_CHAT_ID, CHAT_ID

# --- MANUAL ALERT CONVERSATION STATE ---
# {chat_id: {step, symbol, tf, prices, mode, date_mode}}
_manual_alert_state = {}

def get_manual_alert_state(chat_id):
    return _manual_alert_state.get(chat_id)

def set_manual_alert_state(chat_id, state):
    _manual_alert_state[chat_id] = state

def clear_manual_alert_state(chat_id):
    _manual_alert_state.pop(chat_id, None)

# --- LINE 4H SETTINGS INPUT STATE ---
_line4h_input_state = {}  # {chat_id: "awaiting_range_pct" | "awaiting_point_b_pct"}

def get_line4h_input_state(chat_id):
    return _line4h_input_state.get(chat_id)

def set_line4h_input_state(chat_id, state):
    _line4h_input_state[chat_id] = state

def clear_line4h_input_state(chat_id):
    _line4h_input_state.pop(chat_id, None)

# --- MODEL MENU STATE (for /model multi-step flow) ---
# {chat_id: {"mode": "model"|"fallback"|"startday", "fallback_chain": [...], "msg_id": int}}
_model_menu_state = {}

def get_model_menu_state(chat_id):
    return _model_menu_state.get(chat_id)

def set_model_menu_state(chat_id, state):
    _model_menu_state[chat_id] = state

def clear_model_menu_state(chat_id):
    _model_menu_state.pop(chat_id, None)

# --- SCANSETTING STATE (for scansetting multi-step flow) ---
# {chat_id: {"action": "time"|"threshold", "step": "awaiting_input"}}
_scansetting_state = {}

def get_scansetting_state(chat_id):
    return _scansetting_state.get(chat_id)

def set_scansetting_state(chat_id, state):
    _scansetting_state[chat_id] = state

def clear_scansetting_state(chat_id):
    _scansetting_state.pop(chat_id, None)


# --- AUTOPOST STATE (for autopost multi-step flow) ---
# {chat_id: {"action": "coins"|"time"|"hashtags", "step": "awaiting_input"}}
_autopost_state = {}

def get_autopost_state(chat_id):
    return _autopost_state.get(chat_id)

def set_autopost_state(chat_id, state):
    _autopost_state[chat_id] = state

def clear_autopost_state(chat_id):
    _autopost_state.pop(chat_id, None)


# --- SECTOR STATE (for /sec multi-step flow) ---
# {chat_id: {"action": "add"|"move"|"find"|"verify", "step": str, "symbol": str|None, "msg_id": int|None}}
_sector_state = {}

def get_sector_state(chat_id):
    return _sector_state.get(chat_id)

def set_sector_state(chat_id, state):
    _sector_state[chat_id] = state

def clear_sector_state(chat_id):
    _sector_state.pop(chat_id, None)


def build_model_category_kb(mode, fallback_chain=None):
    """Build category selection keyboard for model/fallback/startday modes.
    Shared between tg_commands.py and tg_callbacks.py."""
    rows = [
        [{"text": "💎 Gemini", "callback_data": f"mdm_{mode}_cat_gm"},
         {"text": "⚡ Groq", "callback_data": f"mdm_{mode}_cat_gq"}],
        [{"text": "🌐 OpenRouter Free", "callback_data": f"mdm_{mode}_cat_or"},
         {"text": "💰 OpenRouter $$$", "callback_data": f"mdm_{mode}_paid"}],
        [{"text": "🚫 Без модели", "callback_data": f"mdm_{mode}_none"}],
    ]
    if mode == "fallback":
        extra = []
        if fallback_chain:
            extra.append({"text": "➕ Добавить ещё", "callback_data": "mdm_fallback"})
            extra.append({"text": "✅ Готово", "callback_data": "mdm_fb_done"})
            extra.append({"text": "🗑 Очистить", "callback_data": "mdm_fb_clear"})
        if extra:
            rows.append(extra)
    rows.append([{"text": "⬅️ Назад", "callback_data": "mdm_back"}])
    return {"inline_keyboard": rows}

# --- ALERT MESSAGE TRACKING (for auto-cleanup) ---
_alert_msg_tracker = {}  # {chat_id: [msg_id, msg_id, ...]}

def track_alert_msg(chat_id, msg_id):
    """Track a message_id for later cleanup during alert flow."""
    if msg_id is None:
        return
    if chat_id not in _alert_msg_tracker:
        _alert_msg_tracker[chat_id] = []
    if msg_id not in _alert_msg_tracker[chat_id]:
        _alert_msg_tracker[chat_id].append(msg_id)

def get_tracked_alert_msgs(chat_id):
    """Get all tracked message_ids for a chat."""
    return _alert_msg_tracker.get(chat_id, [])

def clear_tracked_alert_msgs(chat_id):
    """Clear tracked message_ids for a chat."""
    _alert_msg_tracker.pop(chat_id, None)

# --- OPENROUTER FREE MODELS (dynamic) ---
_or_free_models_cache = {"models": [], "ts": 0}

async def _fetch_or_free_models(force=False):
    """Fetch free models from OpenRouter API, cache for 1 hour."""
    import time
    now = time.time()
    if not force and _or_free_models_cache["models"] and now - _or_free_models_cache["ts"] < 3600:
        return _or_free_models_cache["models"]
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession() as session:
            async with session.get("https://openrouter.ai/api/v1/models", timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    free = sorted(
                        [m["id"] for m in data.get("data", [])
                         if ":free" in m.get("id", "") or (m.get("id") == "openrouter/free")],
                        key=lambda x: (0 if x == "openrouter/free" else 1, x.split("/")[-1])
                    )
                    if free:
                        _or_free_models_cache["models"] = free
                        _or_free_models_cache["ts"] = now
                        return free
    except Exception as e:
        logging.warning(f"⚠️ Failed to fetch OpenRouter models: {e}")
    return _or_free_models_cache["models"] or []

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

# --- SCAN SCHEDULE ---
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
GROUP_ID = int(GROUP_CHAT_ID) if GROUP_CHAT_ID else 0
ALLOWED_CHATS = {ADMIN_ID, GROUP_ID} - {0}

def is_allowed_chat(chat_id: int) -> bool:
    """Check if the chat is allowed (admin DM or configured group)."""
    return chat_id in ALLOWED_CHATS

def is_admin(msg: dict) -> bool:
    """Check if the message sender is the bot admin."""
    user_id = msg.get("from", {}).get("id", 0)
    return user_id == ADMIN_ID

# --- TELEGRAM SEND HELPERS ---

async def send_response(session, chat_id, text, reply_to_msg_id=None, reply_markup=None, parse_mode=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_to_msg_id:
        payload["reply_to_message_id"] = reply_to_msg_id
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", {}).get("message_id")
            else:
                resp_text = await resp.text()
                logging.error(f"❌ send_response non-200: status={resp.status}, body={resp_text[:300]}, payload_text={str(payload.get('text',''))[:100]}")
    except Exception as e:
        logging.error(f"❌ send_response error: {e}")
    return None


async def schedule_alert_cleanup(session, chat_id, delay_seconds=60):
    """Delete all tracked alert messages after a delay."""
    msg_ids = get_tracked_alert_msgs(chat_id)
    if not msg_ids:
        return
    clear_tracked_alert_msgs(chat_id)

    async def _do_cleanup():
        await asyncio.sleep(delay_seconds)
        for mid in msg_ids:
            try:
                await session.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage",
                    json={"chat_id": chat_id, "message_id": mid},
                )
            except Exception as e:
                logging.warning(f"⚠️ Failed to delete msg {mid}: {e}")

    asyncio.create_task(_do_cleanup())


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
