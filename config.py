import json
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load environment variables from .env file (use explicit path so it works regardless of CWD)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Telegram Bot configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROUP_CHAT_ID = os.getenv("TELEGRAM_GROUP_CHAT_ID")
BOTTOM_GROUP_CHAT_ID = os.getenv("TELEGRAM_BOTTOM_GROUP_CHAT_ID")  # Second group: bottom-only signals

# AI Service configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")



# Groq API keys (3 accounts for rotation/fallback)
GROQ_API_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
] if k]

# Gemini API keys (8 keys, 1 per account)
GEMINI_API_KEYS = [k for k in [
    os.getenv(f"GEMINI_KEY_{i}", "") for i in range(1, 9)
] if k]

# Key account labels (for UI display)
KEY_ACCOUNT_LABELS = {
    0: "talalai5208",
    1: "dmitrijtalalaj27",
    2: "sudanialisa",
    3: "zhoriha",
    4: "alisasudani211",
    5: "alisasudani",
    6: "alasasudani210",
    7: "alisasudani1",
}

# AI provider settings persistence
AI_SETTINGS_FILE = "data/ai_settings.json"
SL_SETTINGS_FILE = "data/sl_settings.json"

_DEFAULT_SL_SETTINGS = {
    "btc_shield": "off",  # off | soft
    "signals": {
        "mode": "stopai",  # stopai | trailing | fixed | ema
        "trailing": {
            "anchor": "low-1",      # high-1 | low-1 | high-2 | low-2
            "activation_candles": 3, # 3-10
            "initial_sl_atr": 1.5,   # ATR multiplier for initial SL
            "trail_atr": 1.0         # ATR multiplier buffer from candle level
        },
        "fixed": {
            "sl_atr": 1.5,
            "tp_atr": 3.0
        },
        "ema": {
            "initial_sl_atr": 1.5,
            "ema_period": 25,    # 25 | 50
            "ema_tf": "15m"      # 5m | 15m
        }
    },
    "bankml": {
        "mode": "trailing",  # trailing | fixed | ema (no stopai)
        "trailing": {
            "anchor": "low-1",
            "activation_candles": 3,
            "initial_sl_atr": 1.5,
            "trail_atr": 1.0
        },
        "fixed": {
            "sl_atr": 1.5,
            "tp_atr": 3.0
        },
        "ema": {
            "initial_sl_atr": 1.5,
            "ema_period": 25,
            "ema_tf": "15m"
        }
    }
}

def load_sl_settings() -> dict:
    """Load full SL settings for both banks."""
    import copy
    defaults = copy.deepcopy(_DEFAULT_SL_SETTINGS)
    try:
        if os.path.exists(SL_SETTINGS_FILE):
            with open(SL_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                if "btc_shield" in saved:
                    defaults["btc_shield"] = saved["btc_shield"]
                for bank in ("signals", "bankml"):
                    if bank in saved:
                        defaults[bank]["mode"] = saved[bank].get("mode", defaults[bank]["mode"])
                        for sub in ("trailing", "fixed", "ema"):
                            if sub in saved[bank]:
                                defaults[bank][sub].update(saved[bank][sub])
    except Exception as e:
        logging.error(f"Error reading SL settings: {e}")
    return defaults

def save_sl_settings(settings: dict):
    """Save full SL settings to disk."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(SL_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing SL settings: {e}")

def load_sl_mode(bank: str = "signals") -> str:
    """Quick helper — get mode for a bank. Backward compatible."""
    s = load_sl_settings()
    return s.get(bank, {}).get("mode", "stopai" if bank == "signals" else "trailing")

def load_ai_settings():
    """Load persisted AI provider/model/key settings."""
    defaults = {
        "active_provider": "openrouter",
        "active_key_index": 0,
        "openrouter_model": OPENROUTER_MODEL or "openrouter/free",
        "gemini_model": "gemini-2.5-flash",
        "groq_model": "llama-3.3-70b-versatile",

    }
    if os.path.exists(AI_SETTINGS_FILE):
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception as e:
            logging.error(f"Error reading AI settings: {e}")
    return defaults

def save_ai_settings(settings):
    """Persist AI provider/model/key settings to disk."""
    try:
        os.makedirs(os.path.dirname(AI_SETTINGS_FILE), exist_ok=True)
        with open(AI_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing AI settings: {e}")

_startup_settings = load_ai_settings()

# Binance Square
SQUARE_OPENAPI_KEY = os.getenv("SQUARE_OPENAPI_KEY")

# Encryption setup for Binance API keys
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "").encode()
ENCRYPTED_API_KEY = os.getenv("ENCRYPTED_API_KEY", "").encode()
ENCRYPTED_SECRET_KEY = os.getenv("ENCRYPTED_SECRET_KEY", "").encode()

if ENCRYPTION_KEY and ENCRYPTED_API_KEY and ENCRYPTED_SECRET_KEY:
    fernet = Fernet(ENCRYPTION_KEY)
    BINANCE_API_KEY = fernet.decrypt(ENCRYPTED_API_KEY).decode()
    BINANCE_SECRET_KEY = fernet.decrypt(ENCRYPTED_SECRET_KEY).decode()
else:
    BINANCE_API_KEY = None
    BINANCE_SECRET_KEY = None

TREND_STATE_FILE = "data/trend_state.json"
ALERTS_FILE = "data/pending_alerts.json"
BREAKOUT_LOG_FILE = "data/breakout_log.json"
PRICE_ALERTS_FILE = "data/price_alerts.json"
VIRTUAL_BANK_FILE = "data/virtual_bank.json"

# --- ML virtual bank (tracks ML model predictions separately) ---
ML_BREAKOUT_LOG_FILE = "data/breakout_log_ml.json"
ML_VIRTUAL_BANK_FILE = "data/virtual_bank_ml.json"

# --- VIRTUAL BANK ($10,000 starting) ---
VIRTUAL_BANK_POSITION_SIZE = 100  # $ per trade

# === SIGNAL PIPELINE SETTINGS ===
SIGNAL_CONFIDENCE_FULL = 58.0    # % — full signal with entry
SIGNAL_ADX_TRENDING = 20         # ADX below this = flat market
SIGNAL_LEVERAGE = 1              # ALWAYS 1x, no leverage
SIGNAL_DEPOSIT_PCT = 2           # ALWAYS 2% of bank per trade
SIGNAL_MIN_VOLUME_12H = 2_000_000   # $2M — 12h volume pass threshold
SIGNAL_MIN_VOLUME_1H = 120_000     # $120K — 1h candle volume + green = pass
SIGNAL_SL_ATR_MULT = 2.0          # SL = 2 × ATR from entry
SIGNAL_TP_ATR_MULT = 3.0          # TP = 3 × ATR (R:R = 1:1.5)
SIGNAL_SL_MIN_PCT = 5.0           # SL floor — never less than 5% from entry
SIGNAL_SL_MAX_PCT = 10.0          # SL cap — never more than 10% from entry
TRAILING_STOP_PCT = 3.0            # Trailing stop distance from peak
BREAKEVEN_TRIGGER_PCT = 5.0       # Move SL to breakeven+profit when price moves this % in our favor
BREAKEVEN_PROFIT_PCT = 0.5        # Guaranteed profit % after breakeven trigger
BREAKEVEN_TIME_TRIGGER_PCT = 3.0  # Alternative: +3% in our favor...
BREAKEVEN_TIME_CANDLES = 20       # ...AND 20+ candles (5m) passed → also activate breakeven
ML_SL_ATR_MULT = 1.5              # ML SL = 1.5 × ATR, clamped 5-10%
ML_SL_MIN_PCT = 5.0               # ML SL minimum
ML_SL_MAX_PCT = 10.0              # ML SL maximum

def load_virtual_bank():
    if os.path.exists(VIRTUAL_BANK_FILE):
        try:
            with open(VIRTUAL_BANK_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading virtual bank: {e}")
    return {"starting_balance": 10000, "balance": 10000, "total_trades": 0, "total_wins": 0, "total_losses": 0, "history": []}

def save_virtual_bank(bank):
    try:
        os.makedirs(os.path.dirname(VIRTUAL_BANK_FILE), exist_ok=True)
        with open(VIRTUAL_BANK_FILE, "w") as f:
            json.dump(bank, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing virtual bank: {e}")

def reset_virtual_bank():
    """Reset virtual bank to starting state (balance=10000, all-time stats zeroed).
    Does NOT clear today's breakout log — only resets bank + cumulative counters."""
    bank = {
        "starting_balance": 10000,
        "balance": 10000,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "history": []
    }
    save_virtual_bank(bank)
    return bank

def update_bank_with_trades(trades_pnl):
    """Update bank balance with list of (symbol, pnl_pct, pnl_dollar) tuples.
    Returns updated bank dict."""
    bank = load_virtual_bank()
    for symbol, pnl_pct, pnl_dollar in trades_pnl:
        bank["balance"] += pnl_dollar
        bank["total_trades"] += 1
        if pnl_pct >= 0:
            bank["total_wins"] += 1
        else:
            bank["total_losses"] += 1
        bank["history"].append({
            "symbol": symbol,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollar": round(pnl_dollar, 2),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        })
    bank["balance"] = round(bank["balance"], 2)
    save_virtual_bank(bank)
    return bank

def update_env_file(key: str, value: str):
    """Update or add a key=value pair in the .env file (persists across restarts)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    logging.info(f"💾 .env updated: {key}=...{value[-6:]}" if len(value) > 6 else f"💾 .env updated: {key}")


def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading alerts file: {e}")
    return []

def save_alerts(alerts):
    try:
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing alerts file: {e}")

def load_breakout_log():
    if os.path.exists(BREAKOUT_LOG_FILE):
        try:
            with open(BREAKOUT_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_breakout_log(log):
    try:
        os.makedirs(os.path.dirname(BREAKOUT_LOG_FILE), exist_ok=True)
        with open(BREAKOUT_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing breakout log: {e}")

def parse_ai_trade_params(ai_text: str) -> dict:
    """Parse Entry, SL, TP, Leverage, Deposit% from AI verdict text."""
    import re
    result = {"ai_entry": None, "ai_sl": None, "ai_tp": None, "ai_leverage": None, "ai_deposit_pct": None}
    if not ai_text:
        return result
    # Match patterns: 💰 Entry: 0.1234 or Entry: $0.1234 or Вход: 0.1234
    # Be specific to avoid matching "💰 Current Price:" or "💰 Funding:"
    for key, patterns in [
        ("ai_entry", [r"💰\s*entry[:\s]*\$?([\d]+\.[\d]+)", r"(?:entry|вход)[:\s]*\$?([\d]+\.[\d]+)"]),
        ("ai_sl", [r"🚫\s*sl[:\s]*\$?([\d]+\.[\d]+)", r"(?:sl|stop\s*loss|стоп|сл)[:\s]*\$?([\d]+\.[\d]+)"]),
        ("ai_tp", [r"🎯\s*tp[:\s]*\$?([\d]+\.[\d]+)", r"(?:tp|take\s*profit|тейк|тп)[:\s]*\$?([\d]+\.[\d]+)"]),
    ]:
        for pat in patterns:
            m = re.search(pat, ai_text, re.IGNORECASE)
            if m:
                try:
                    result[key] = float(m.group(1))
                except ValueError:
                    pass
                break

    # Parse leverage: "💼 REC: 5x | 10%" or "Leverage: 10x" or just "5x"
    lev_match = re.search(r"(?:REC|leverage|плечо)[:\s]*(\d+)\s*x", ai_text, re.IGNORECASE)
    if lev_match:
        try:
            result["ai_leverage"] = int(lev_match.group(1))
        except ValueError:
            pass

    # Parse deposit %: "💼 REC: 5x | 10%" or "Deposit: 10%"
    dep_match = re.search(r"(?:REC[^|]*\|\s*|deposit|депозит)[:\s]*(\d+(?:\.\d+)?)\s*%", ai_text, re.IGNORECASE)
    if dep_match:
        try:
            result["ai_deposit_pct"] = float(dep_match.group(1))
        except ValueError:
            pass

    return result


def _validate_ai_prices(symbol, current_price, ai_entry, ai_sl, ai_tp, ai_direction, max_entry_deviation=0.30):
    """Validate AI-parsed prices against real market price.
    
    FIXES BUG: AI model sometimes returns prices from a DIFFERENT coin
    (e.g., SEI price in HANA analysis). This causes wrong entry in virtual bank,
    wrong SL/TP levels, and phantom losses.
    
    Checks:
    1. ai_entry vs current_price: if >30% off → use current_price
    2. ai_sl direction: LONG SL must be < entry, SHORT SL must be > entry
    3. ai_tp direction: LONG TP must be > entry, SHORT TP must be < entry
    
    Returns: (ai_entry, ai_sl, ai_tp) — corrected values
    """
    if current_price <= 0:
        return ai_entry, ai_sl, ai_tp
    
    # 1. Validate ai_entry — must be close to current_price
    if ai_entry is not None and ai_entry > 0:
        deviation = abs(ai_entry - current_price) / current_price
        if deviation > max_entry_deviation:
            logging.warning(
                f"⚠️ PRICE MISMATCH {symbol}: ai_entry={ai_entry} vs market={current_price} "
                f"(deviation {deviation*100:.1f}% > {max_entry_deviation*100:.0f}%) — using market price"
            )
            ai_entry = current_price
    
    # Use effective entry for SL/TP validation
    effective_entry = ai_entry if (ai_entry and ai_entry > 0) else current_price
    direction = (ai_direction or "").upper()
    
    # 2. Validate ai_sl direction
    if ai_sl is not None and ai_sl > 0 and direction in ("LONG", "SHORT"):
        if direction == "LONG" and ai_sl >= effective_entry:
            logging.warning(f"⚠️ SL WRONG SIDE {symbol}: LONG but SL={ai_sl} >= entry={effective_entry} — clearing SL")
            ai_sl = None
        elif direction == "SHORT" and ai_sl <= effective_entry:
            logging.warning(f"⚠️ SL WRONG SIDE {symbol}: SHORT but SL={ai_sl} <= entry={effective_entry} — clearing SL")
            ai_sl = None
        # Also check SL isn't from a different coin (>50% away from entry)
        elif abs(ai_sl - effective_entry) / effective_entry > 0.50:
            logging.warning(f"⚠️ SL TOO FAR {symbol}: SL={ai_sl} is >50% from entry={effective_entry} — clearing SL")
            ai_sl = None
    
    # 3. Validate ai_tp direction
    if ai_tp is not None and ai_tp > 0 and direction in ("LONG", "SHORT"):
        if direction == "LONG" and ai_tp <= effective_entry:
            logging.warning(f"⚠️ TP WRONG SIDE {symbol}: LONG but TP={ai_tp} <= entry={effective_entry} — clearing TP")
            ai_tp = None
        elif direction == "SHORT" and ai_tp >= effective_entry:
            logging.warning(f"⚠️ TP WRONG SIDE {symbol}: SHORT but TP={ai_tp} >= entry={effective_entry} — clearing TP")
            ai_tp = None
        elif abs(ai_tp - effective_entry) / effective_entry > 0.50:
            logging.warning(f"⚠️ TP TOO FAR {symbol}: TP={ai_tp} is >50% from entry={effective_entry} — clearing TP")
            ai_tp = None
    
    # 4. Clamp SL between SIGNAL_SL_MIN_PCT (5%) and SIGNAL_SL_MAX_PCT (10%) from entry
    if ai_sl is not None and ai_sl > 0 and effective_entry > 0:
        sl_pct = abs(ai_sl - effective_entry) / effective_entry * 100
        if sl_pct < SIGNAL_SL_MIN_PCT:
            if direction == "LONG":
                ai_sl = effective_entry * (1 - SIGNAL_SL_MIN_PCT / 100)
            elif direction == "SHORT":
                ai_sl = effective_entry * (1 + SIGNAL_SL_MIN_PCT / 100)
            logging.info(f"📏 SL FLOOR {symbol}: was {sl_pct:.1f}% → raised to {SIGNAL_SL_MIN_PCT}% = {ai_sl}")
        elif sl_pct > SIGNAL_SL_MAX_PCT:
            if direction == "LONG":
                ai_sl = effective_entry * (1 - SIGNAL_SL_MAX_PCT / 100)
            elif direction == "SHORT":
                ai_sl = effective_entry * (1 + SIGNAL_SL_MAX_PCT / 100)
            logging.warning(f"⚠️ SL CAPPED {symbol}: was {sl_pct:.1f}% → capped to {SIGNAL_SL_MAX_PCT}% = {ai_sl}")
    
    return ai_entry, ai_sl, ai_tp


def add_breakout_entry(symbol, tf, breakout_price, current_price, line_type="", ai_direction="", ai_entry=None, ai_sl=None, ai_tp=None, ai_leverage=None, ai_deposit_pct=None, is_pump_filter=False, atr_value=None):
    """Add a breakout event to the log (deduplicates by symbol+tf).
    If same symbol+tf exists but old verdict was SKIP/empty and new is LONG/SHORT, upgrade it.
    """
    log = load_breakout_log()
    # Check if same symbol+tf already exists
    existing = next((e for e in log if e["symbol"] == symbol and e["tf"] == tf), None)
    if existing:
        old_dir = existing.get("ai_direction", "").upper()
        new_dir = ai_direction.upper() if ai_direction else ""
        # Upgrade: old was SKIP/empty/info → new is a real trade (LONG/SHORT)
        if old_dir not in ("LONG", "SHORT") and new_dir in ("LONG", "SHORT"):
            log.remove(existing)
            logging.info(f"♻️ Upgrading {symbol} {tf}: {old_dir or 'NONE'} → {new_dir}")
        else:
            return  # already exists with equal or better verdict

    # === VALIDATE AI PRICES against real market price ===
    ai_entry, ai_sl, ai_tp = _validate_ai_prices(
        symbol, current_price, ai_entry, ai_sl, ai_tp, ai_direction
    )

    # Check if same symbol (any TF) already has an active trade (LONG/SHORT) — mark as duplicate
    symbol_has_active_trade = any(
        e["symbol"] == symbol and e.get("ai_direction", "").upper() in ("LONG", "SHORT")
        for e in log
    )
    entry = {
        "symbol": symbol,
        "tf": tf,
        "breakout_price": round(breakout_price, 8),
        "current_price": round(current_price, 8),
        "type": line_type,
        "ai_direction": ai_direction.upper() if ai_direction else "",
        "time": datetime.now(timezone.utc).isoformat()
    }
    if ai_entry is not None:
        entry["ai_entry"] = round(ai_entry, 8)
    if ai_sl is not None:
        entry["ai_sl"] = round(ai_sl, 8)
    if ai_tp is not None:
        entry["ai_tp"] = round(ai_tp, 8)
    if ai_leverage is not None:
        entry["ai_leverage"] = ai_leverage
    if ai_deposit_pct is not None:
        entry["ai_deposit_pct"] = ai_deposit_pct
    if atr_value is not None:
        entry["atr_value"] = round(atr_value, 8)
    if is_pump_filter:
        entry["is_pump_filter"] = True
    if symbol_has_active_trade:
        entry["is_duplicate"] = True
    log.append(entry)
    save_breakout_log(log)

def clear_breakout_log():
    save_breakout_log([])


# ============================
# ML VIRTUAL BANK (tracks ML model predictions)
# ============================

def load_ml_virtual_bank():
    if os.path.exists(ML_VIRTUAL_BANK_FILE):
        try:
            with open(ML_VIRTUAL_BANK_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading ML virtual bank: {e}")
    return {"starting_balance": 10000, "balance": 10000, "total_trades": 0, "total_wins": 0, "total_losses": 0, "history": []}

def save_ml_virtual_bank(bank):
    try:
        os.makedirs(os.path.dirname(ML_VIRTUAL_BANK_FILE), exist_ok=True)
        with open(ML_VIRTUAL_BANK_FILE, "w") as f:
            json.dump(bank, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing ML virtual bank: {e}")

def reset_ml_virtual_bank():
    bank = {
        "starting_balance": 10000,
        "balance": 10000,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "history": []
    }
    save_ml_virtual_bank(bank)
    return bank

def update_ml_bank_with_trades(trades_pnl):
    bank = load_ml_virtual_bank()
    for symbol, pnl_pct, pnl_dollar in trades_pnl:
        bank["balance"] += pnl_dollar
        bank["total_trades"] += 1
        if pnl_pct >= 0:
            bank["total_wins"] += 1
        else:
            bank["total_losses"] += 1
        bank["history"].append({
            "symbol": symbol,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollar": round(pnl_dollar, 2),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        })
    bank["balance"] = round(bank["balance"], 2)
    save_ml_virtual_bank(bank)
    return bank

def load_ml_breakout_log():
    if os.path.exists(ML_BREAKOUT_LOG_FILE):
        try:
            with open(ML_BREAKOUT_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_ml_breakout_log(log):
    try:
        os.makedirs(os.path.dirname(ML_BREAKOUT_LOG_FILE), exist_ok=True)
        with open(ML_BREAKOUT_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing ML breakout log: {e}")

def add_ml_breakout_entry(symbol, tf, current_price, ml_direction, ml_sl, indicators=None, atr_value=None):
    """Add ML prediction to ML breakout log (deduplicates by symbol+tf).
    If same symbol+tf exists but direction was empty, upgrade it.
    """
    log = load_ml_breakout_log()
    existing = next((e for e in log if e["symbol"] == symbol and e["tf"] == tf), None)
    if existing:
        old_dir = existing.get("ml_direction", "").upper()
        new_dir = ml_direction.upper() if ml_direction else ""
        if old_dir not in ("LONG", "SHORT") and new_dir in ("LONG", "SHORT"):
            log.remove(existing)
            logging.info(f"♻️ ML upgrade {symbol} {tf}: {old_dir or 'NONE'} → {new_dir}")
            save_ml_breakout_log(log)  # save after remove
        else:
            return  # already exists
    entry = {
        "symbol": symbol,
        "tf": tf,
        "current_price": round(current_price, 8),
        "ml_direction": ml_direction.upper() if ml_direction else "",
        "ml_sl": round(ml_sl, 8) if ml_sl else None,
        "time": datetime.now(timezone.utc).isoformat()
    }
    if atr_value is not None:
        entry["atr_value"] = round(atr_value, 8)
    log.append(entry)
    save_ml_breakout_log(log)

def clear_ml_breakout_log():
    save_ml_breakout_log([])


# --- PRICE ALERTS ---
def load_price_alerts():
    if os.path.exists(PRICE_ALERTS_FILE):
        try:
            with open(PRICE_ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_price_alerts(alerts):
    try:
        os.makedirs(os.path.dirname(PRICE_ALERTS_FILE), exist_ok=True)
        with open(PRICE_ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing price alerts: {e}")
