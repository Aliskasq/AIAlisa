import asyncio
import logging
import json
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import aiohttp

# Import required project modules
from core.binance_api import fetch_klines, fetch_funding_rate
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
from agent.skills import post_to_binance_square

# --- GLOBAL TOGGLE FOR TELEGRAM ---
# Default OFF — admin enables manually via /autopost on
AUTO_SQUARE_ENABLED = False

# --- SETTINGS FILE (persists across restarts) ---
AUTOPOST_SETTINGS_FILE = "data/autopost_settings.json"

def _load_settings():
    """Load saved autopost settings (coins + times + hashtags) from disk."""
    defaults = {
        "coins": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "times": [{"hour": 9, "minute": 9}, {"hour": 21, "minute": 9}],
        "hashtags": "#AIBinance #BinanceSquare #Write2Earn"
    }
    if os.path.exists(AUTOPOST_SETTINGS_FILE):
        try:
            with open(AUTOPOST_SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                return {
                    "coins": saved.get("coins", defaults["coins"]),
                    "times": saved.get("times", defaults["times"]),
                    "hashtags": saved.get("hashtags", defaults["hashtags"])
                }
        except Exception:
            pass
    return defaults

def _save_settings(settings):
    """Persist autopost settings to disk."""
    os.makedirs(os.path.dirname(AUTOPOST_SETTINGS_FILE), exist_ok=True)
    with open(AUTOPOST_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# Load on import so tg_listener can read them
AUTOPOST_SETTINGS = _load_settings()

def set_coins(coin_list: list):
    """Update coins for auto-posting. Called from tg_listener."""
    symbols = []
    for c in coin_list:
        c = c.upper().strip()
        if not c.endswith("USDT"):
            c += "USDT"
        symbols.append(c)
    AUTOPOST_SETTINGS["coins"] = symbols
    _save_settings(AUTOPOST_SETTINGS)

def set_times(time_list: list):
    """Update schedule times for auto-posting. Called from tg_listener.
    time_list: list of dicts like [{"hour": 13, "minute": 30}, ...]
    """
    AUTOPOST_SETTINGS["times"] = time_list
    _save_settings(AUTOPOST_SETTINGS)

def get_coins():
    return AUTOPOST_SETTINGS["coins"]

def get_times():
    return AUTOPOST_SETTINGS["times"]

def set_hashtags(hashtags_str: str):
    """Update hashtags for auto-posting. Called from tg_listener."""
    AUTOPOST_SETTINGS["hashtags"] = hashtags_str.strip()
    _save_settings(AUTOPOST_SETTINGS)

def get_hashtags():
    return AUTOPOST_SETTINGS.get("hashtags", "#AIBinance #BinanceSquare #Write2Earn")

def get_status_text():
    """Human-readable status string for Telegram."""
    coins_str = ", ".join(AUTOPOST_SETTINGS["coins"])
    times_str = ", ".join(f"{t['hour']:02d}:{t['minute']:02d} UTC" for t in AUTOPOST_SETTINGS["times"])
    hashtags_str = AUTOPOST_SETTINGS.get("hashtags", "#AIBinance #BinanceSquare #Write2Earn")
    state = "ON ✅" if AUTO_SQUARE_ENABLED else "OFF ⏸"
    return (
        f"📢 *Auto-Post Status:* {state}\n"
        f"🪙 *Coins:* `{coins_str}`\n"
        f"⏰ *Schedule:* `{times_str}`\n"
        f"🏷 *Hashtags:* `{hashtags_str}`"
    )


async def auto_square_poster(session: aiohttp.ClientSession):
    """Background task: publishes AI reports to Binance Square at configured times."""
    global AUTO_SQUARE_ENABLED
    logging.info("🕒 Square Publisher task started.")

    while True:
        now = datetime.now(timezone.utc)
        times = get_times()

        # Build list of candidate target datetimes (today + tomorrow morning fallback)
        candidates = []
        for t in times:
            candidate = now.replace(hour=t["hour"], minute=t["minute"], second=0, microsecond=0)
            candidates.append(candidate)
            # Also add tomorrow's version in case all today's times have passed
            candidates.append(candidate + timedelta(days=1))

        # Pick the nearest future time
        future_candidates = [c for c in candidates if c > now]
        if not future_candidates:
            # Shouldn't happen, but safeguard
            await asyncio.sleep(60)
            continue

        target_time = min(future_candidates)
        sleep_sec = (target_time - now).total_seconds()

        if sleep_sec <= 0:
            sleep_sec = 60

        logging.info(f"⏳ Publisher sleeping for {sleep_sec:.0f}s until {target_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        await asyncio.sleep(sleep_sec)

        # --- CHECK IF USER DISABLED AUTO-POSTING VIA TELEGRAM ---
        if not AUTO_SQUARE_ENABLED:
            logging.info("⏸ Auto-post is DISABLED. Woke up, but skipping publication.")
            await asyncio.sleep(120)
            continue

        # Woke up at target time — post for each configured coin
        coins_to_post = get_coins()
        for symbol in coins_to_post:
            try:
                short_coin = symbol.replace("USDT", "")
                logging.info(f"📢 Generating Square post for {symbol}...")

                # Fetch 250 candles per TF for SMC + indicators
                raw_4h = await fetch_klines(session, symbol, "4h", 250)
                if not raw_4h:
                    continue

                df = pd.DataFrame(raw_4h)
                last_row, _ = calculate_binance_indicators(df, "4H")

                # Fetch funding rate
                funding = await fetch_funding_rate(session, symbol)
                last_row["funding_rate"] = funding

                # Multi-TF indicators
                raw_1h = await fetch_klines(session, symbol, "1h", 250)
                raw_15m = await fetch_klines(session, symbol, "15m", 250)
                mtf_data = {}
                if raw_1h:
                    mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                if raw_15m:
                    mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]

                # SMC Indicator (Structure, Order Blocks, FVG)
                smc_data = {}
                try:
                    from core.smc import analyze_smc
                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_4h), "4H")
                    if raw_1h:
                        smc_data["1H"] = analyze_smc(pd.DataFrame(raw_1h), "1H")
                    if raw_15m:
                        smc_data["15m"] = analyze_smc(pd.DataFrame(raw_15m), "15m")
                except Exception as e:
                    logging.error(f"❌ SMC autopost error: {e}")

                # Square-optimized AI analysis (1300-1900 chars, plain text, no markdown)
                ai_text = await ask_ai_analysis(symbol, "4H", last_row, lang="en", square=True, mtf_data=mtf_data, smc_data=smc_data)

                # Build Square post — POST format (not article), 1500-2100 chars total
                tags = get_hashtags()
                header = f"🤖 AI-ALISA-COPILOTCLAW | Automated Analysis\n\n"
                footer = f"\n\n{tags}"
                max_ai_len = 2100 - len(header) - len(footer)
                if len(ai_text) > max_ai_len:
                    ai_text = ai_text[:max_ai_len - 3] + "..."
                square_text = f"{header}{ai_text}{footer}"

                if len(square_text) > 2100:
                    square_text = square_text[:2097] + "..."

                res = await post_to_binance_square(square_text)
                logging.info(f"✅ Square Auto-Post result for {symbol}: {res}")

                await asyncio.sleep(15)
            except Exception as e:
                logging.error(f"❌ Auto post error for {symbol}: {e}")

        # Prevent double-trigger
        await asyncio.sleep(120)
