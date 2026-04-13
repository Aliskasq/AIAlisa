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
