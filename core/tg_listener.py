"""
tg_listener.py — Telegram polling loop & dispatcher.

This module used to be a 3400-line monolith.  It has been split into:
  • tg_state.py      — shared state, constants, caches, send helpers
  • tg_reports.py    — signal report builders (virtual bank, TP/SL)
  • tg_background.py — background tasks (daily summary, price alerts)
  • tg_callbacks.py  — inline-keyboard / callback_query handlers
  • tg_commands.py   — all /command & text-trigger handlers

This file now contains only the polling loop and re-exports so that
existing ``from core.tg_listener import X`` statements keep working.
"""

import asyncio
import logging
from config import BOT_TOKEN

# ── Re-exports from sub-modules (backward compatibility) ──────────────
from core.tg_state import (                        # noqa: F401
    send_response,
    send_and_get_msg_id,
    get_chat_lang,
    set_chat_lang,
    is_allowed_chat,
    is_admin,
    ADMIN_ID,
    GROUP_ID,
    ALLOWED_CHATS,
    SCAN_SCHEDULE,
    _save_scan_schedule,
    _load_scan_schedule,
    _fetch_or_free_models,
    _load_paper,
    _save_paper,
    _load_langs,
    _save_langs,
    square_cache_put,
    square_cache_get,
    square_cache_delete,
    PAPER_FILE,
    LANG_FILE,
    SQUARE_CACHE_FILE,
    SCAN_SCHEDULE_FILE,
)

from core.tg_reports import (                      # noqa: F401
    build_signals_text,
    build_signals_close_text,
    build_signals_text_monitor,
    build_signals_close_text_monitor,
    _check_tp_sl_from_candles,
    _batch_check_tp_sl,
)

from core.tg_background import (                   # noqa: F401
    auto_trend_sender,
    price_alert_monitor,
)

from core.tg_callbacks import handle_callback_query  # noqa: F401
from core.tg_commands import handle_message          # noqa: F401


# ── Polling loop ───────────────────────────────────────────────────────
async def telegram_polling_loop(app_session):
    """Listens for messages and button presses from the Telegram group/chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    offset = 0
    while True:
        try:
            async with app_session.get(f"{url}?offset={offset}&timeout=10", timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1

                        if "callback_query" in update:
                            await handle_callback_query(app_session, update)
                            continue

                        await handle_message(app_session, update)

        except Exception as e:
            logging.error(f"TG Polling Error: {e}")
            await asyncio.sleep(2)
