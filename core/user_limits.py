"""
user_limits.py — Per-user rate limiting for API-heavy commands.

Three independent limit types (all optional, per-user):
1. Time between messages (cooldown in minutes)
2. Daily message count (resets at 03:00 MSK / 00:00 UTC)
3. Per-ticker cooldown (same coin can't be requested within X minutes)
"""
import json
import logging
import os
from datetime import datetime, timezone, timedelta

LIMITS_FILE = "data/user_limits.json"
_state: dict = {}  # runtime tracking: {user_id: {last_msg, daily_count, daily_date, tickers: {SYM: ts}}}


def _load_settings() -> dict:
    """Load limit settings (admin-configured). {user_id: {cooldown_min, daily_max, ticker_cooldown_min}}"""
    if os.path.exists(LIMITS_FILE):
        try:
            with open(LIMITS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading user limits: {e}")
    return {}


def _save_settings(data: dict):
    try:
        os.makedirs(os.path.dirname(LIMITS_FILE), exist_ok=True)
        with open(LIMITS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing user limits: {e}")


def get_user_settings(user_id) -> dict | None:
    """Get limit settings for a user, or None if no limits set.
    user_id can be int (Telegram ID) or str (username)."""
    data = _load_settings()
    return data.get(str(user_id))


def set_user_setting(user_id, key: str, value):
    """Set a single limit setting for a user.
    user_id can be int (Telegram ID) or str (username)."""
    data = _load_settings()
    uid = str(user_id)
    if uid not in data:
        data[uid] = {}
    data[uid][key] = value
    _save_settings(data)


def remove_user_limits(user_id):
    """Remove all limits for a user.
    user_id can be int (Telegram ID) or str (username)."""
    data = _load_settings()
    data.pop(str(user_id), None)
    _save_settings(data)
    _state.pop(user_id, None)


def get_all_limited_users() -> dict:
    """Return all users with limits."""
    return _load_settings()


def _get_daily_date() -> str:
    """Current 'day' key — resets at 03:00 MSK (00:00 UTC)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def _ensure_state(user_id: int):
    if user_id not in _state:
        _state[user_id] = {
            "last_msg": 0,
            "daily_count": 0,
            "daily_date": "",
            "tickers": {},
        }


def check_limits(user_id: int, ticker: str | None = None) -> str | None:
    """Check all limits for a user. Returns error message string if blocked, None if OK.
    
    Args:
        user_id: Telegram user ID
        ticker: optional coin symbol (e.g. 'BTCUSDT') for ticker cooldown check
    """
    settings = get_user_settings(user_id)
    if not settings:
        return None  # No limits configured

    _ensure_state(user_id)
    st = _state[user_id]
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    # 1. Time cooldown between messages
    cooldown_min = settings.get("cooldown_min")
    if cooldown_min and cooldown_min > 0:
        elapsed = (now_ts - st["last_msg"]) / 60 if st["last_msg"] > 0 else 999999
        if elapsed < cooldown_min:
            remaining = int(cooldown_min - elapsed) + 1
            return f"⏱ Вы можете писать раз в {cooldown_min} мин. Подождите ещё {remaining} мин."

    # 2. Daily message count
    daily_max = settings.get("daily_max")
    if daily_max and daily_max > 0:
        today = _get_daily_date()
        if st["daily_date"] != today:
            st["daily_count"] = 0
            st["daily_date"] = today
        if st["daily_count"] >= daily_max:
            return f"📊 В сутки вы можете отправить только {daily_max} сообщений."

    # 3. Ticker cooldown
    ticker_cooldown_min = settings.get("ticker_cooldown_min")
    if ticker_cooldown_min and ticker_cooldown_min > 0 and ticker:
        ticker_upper = ticker.upper()
        last_ticker_ts = st["tickers"].get(ticker_upper, 0)
        if last_ticker_ts > 0:
            elapsed_ticker = (now_ts - last_ticker_ts) / 60
            if elapsed_ticker < ticker_cooldown_min:
                remaining = int(ticker_cooldown_min - elapsed_ticker) + 1
                # Clean ticker name for display
                display = ticker_upper.replace("USDT", "")
                return f"🪙 Вы уже смотрели {display} {int(elapsed_ticker)} мин. назад. Подождите ещё {remaining} мин."

    return None  # All checks passed


def record_usage(user_id: int, ticker: str | None = None):
    """Record that user sent a valid command (after limits check passed)."""
    _ensure_state(user_id)
    st = _state[user_id]
    now_ts = datetime.now(timezone.utc).timestamp()

    st["last_msg"] = now_ts

    today = _get_daily_date()
    if st["daily_date"] != today:
        st["daily_count"] = 0
        st["daily_date"] = today
    st["daily_count"] += 1

    if ticker:
        ticker_upper = ticker.upper()
        st["tickers"][ticker_upper] = now_ts
        # Cleanup old tickers (>3 hours)
        cutoff = now_ts - 3 * 3600
        st["tickers"] = {k: v for k, v in st["tickers"].items() if v > cutoff}
