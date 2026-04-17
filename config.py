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

# AI Service configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# Monitor: separate API key + model (parallel AI calls, no rate limit conflict)
OPENROUTER_API_KEY_MONITOR = os.getenv("OPENROUTER_API_KEY_MONITOR", "")
OPENROUTER_MODEL_MONITOR = os.getenv("OPENROUTER_MODEL_MONITOR", "") or OPENROUTER_MODEL

# Monitor group (separate from main signal group)
MONITOR_GROUP_CHAT_ID = os.getenv("MONITOR_GROUP_CHAT_ID", "")

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

def load_ai_settings():
    """Load persisted AI provider/model/key settings."""
    defaults = {
        "active_provider": "openrouter",
        "active_key_index": 0,
        "openrouter_model": OPENROUTER_MODEL or "openrouter/free",
        "gemini_model": "gemini-2.5-flash",
        "groq_model": "llama-3.3-70b-versatile",
        "monitor_model": OPENROUTER_MODEL_MONITOR or OPENROUTER_MODEL or "openrouter/free",
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

# Restore monitor model from saved settings on startup
_startup_settings = load_ai_settings()
if _startup_settings.get("monitor_model"):
    OPENROUTER_MODEL_MONITOR = _startup_settings["monitor_model"]

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

# --- MONITOR virtual bank (separate from main signals) ---
MONITOR_BREAKOUT_LOG_FILE = "data/breakout_log_monitor.json"
MONITOR_VIRTUAL_BANK_FILE = "data/virtual_bank_monitor.json"

# --- VIRTUAL BANK ($10,000 starting) ---
VIRTUAL_BANK_POSITION_SIZE = 100  # $ per trade

# === SIGNAL PIPELINE SETTINGS ===
SIGNAL_CONFIDENCE_FULL = 65      # % — full signal with entry
SIGNAL_CONFIDENCE_MONITOR = 50   # % — put on 30-min watch
SIGNAL_ADX_TRENDING = 20         # ADX below this = flat market
SIGNAL_LEVERAGE = 1              # ALWAYS 1x, no leverage
SIGNAL_DEPOSIT_PCT = 2           # ALWAYS 2% of bank per trade
SIGNAL_MIN_VOLUME_12H = 2_000_000   # $2M — 12h volume pass threshold
SIGNAL_MIN_VOLUME_1H = 170_000     # $170K — 1h candle volume + green = pass
SIGNAL_SL_ATR_MULT = 2.0          # SL = 2 × ATR from entry
SIGNAL_TP_ATR_MULT = 3.0          # TP = 3 × ATR (R:R = 1:1.5)

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


def add_breakout_entry(symbol, tf, breakout_price, current_price, line_type="", ai_direction="", ai_entry=None, ai_sl=None, ai_tp=None, ai_leverage=None, ai_deposit_pct=None, is_monitor=False, is_pump_filter=False):
    """Add a breakout event to the log (deduplicates by symbol+tf)."""
    log = load_breakout_log()
    # Check if same symbol+tf already exists
    existing = next((e for e in log if e["symbol"] == symbol and e["tf"] == tf), None)
    if existing:
        # If new entry is monitor but existing isn't marked, update it
        if is_monitor and not existing.get("is_monitor", False):
            existing["is_monitor"] = True
            if ai_direction:
                existing["ai_direction"] = ai_direction.upper()
            save_breakout_log(log)
        return  # already exists, don't duplicate
    if True:  # new entry — always add
        # Check if same symbol (any TF) already exists — mark as duplicate
        symbol_already_exists = any(e["symbol"] == symbol for e in log)
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
        if is_monitor:
            entry["is_monitor"] = True
        if is_pump_filter:
            entry["is_pump_filter"] = True
        if symbol_already_exists:
            entry["is_duplicate"] = True
        log.append(entry)
        save_breakout_log(log)

def upgrade_breakout_entry(symbol, tf, ai_direction, ai_entry=None, ai_sl=None, ai_tp=None, ai_leverage=None, ai_deposit_pct=None):
    """Upgrade an existing SKIP entry to a real trade (called when monitor upgrades).
    If no existing entry found, adds a new one."""
    log = load_breakout_log()
    found = False
    for entry in log:
        if entry["symbol"] == symbol and entry["tf"] == tf:
            entry["ai_direction"] = ai_direction.upper() if ai_direction else ""
            if ai_entry is not None:
                entry["ai_entry"] = round(ai_entry, 8)
                entry["current_price"] = round(ai_entry, 8)  # update price to upgrade moment
            if ai_sl is not None:
                entry["ai_sl"] = round(ai_sl, 8)
            if ai_tp is not None:
                entry["ai_tp"] = round(ai_tp, 8)
            if ai_leverage is not None:
                entry["ai_leverage"] = ai_leverage
            if ai_deposit_pct is not None:
                entry["ai_deposit_pct"] = ai_deposit_pct
            entry["upgraded_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if found:
        save_breakout_log(log)
        logging.info(f"🔄 Breakout entry upgraded: {symbol} {tf} → {ai_direction}")
    else:
        # No existing entry — add new (shouldn't normally happen)
        add_breakout_entry(symbol, tf, 0, ai_entry or 0, "monitor_upgrade",
                          ai_direction, ai_entry, ai_sl, ai_tp, ai_leverage, ai_deposit_pct)

def clear_breakout_log():
    save_breakout_log([])


# ============================
# MONITOR VIRTUAL BANK (separate)
# ============================

def load_monitor_virtual_bank():
    if os.path.exists(MONITOR_VIRTUAL_BANK_FILE):
        try:
            with open(MONITOR_VIRTUAL_BANK_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading monitor virtual bank: {e}")
    return {"starting_balance": 10000, "balance": 10000, "total_trades": 0, "total_wins": 0, "total_losses": 0, "history": []}

def save_monitor_virtual_bank(bank):
    try:
        os.makedirs(os.path.dirname(MONITOR_VIRTUAL_BANK_FILE), exist_ok=True)
        with open(MONITOR_VIRTUAL_BANK_FILE, "w") as f:
            json.dump(bank, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing monitor virtual bank: {e}")

def reset_monitor_virtual_bank():
    bank = {
        "starting_balance": 10000,
        "balance": 10000,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "history": []
    }
    save_monitor_virtual_bank(bank)
    return bank

def update_monitor_bank_with_trades(trades_pnl):
    bank = load_monitor_virtual_bank()
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
    save_monitor_virtual_bank(bank)
    return bank

def load_monitor_breakout_log():
    if os.path.exists(MONITOR_BREAKOUT_LOG_FILE):
        try:
            with open(MONITOR_BREAKOUT_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_monitor_breakout_log(log):
    try:
        os.makedirs(os.path.dirname(MONITOR_BREAKOUT_LOG_FILE), exist_ok=True)
        with open(MONITOR_BREAKOUT_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing monitor breakout log: {e}")

def add_monitor_breakout_entry(symbol, tf, breakout_price, current_price, line_type="", ai_direction="", ai_entry=None, ai_sl=None, ai_tp=None, ai_leverage=None, ai_deposit_pct=None):
    """Add a monitor upgrade event to the monitor log (deduplicates by symbol+tf)."""
    log = load_monitor_breakout_log()
    if not any(e["symbol"] == symbol and e["tf"] == tf for e in log):
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
        log.append(entry)
        save_monitor_breakout_log(log)

def clear_monitor_breakout_log():
    save_monitor_breakout_log([])


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
