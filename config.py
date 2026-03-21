import json
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load environment variables from .env file
load_dotenv()

# Telegram Bot configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROUP_CHAT_ID = os.getenv("TELEGRAM_GROUP_CHAT_ID")

# AI Service configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")

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

def add_breakout_entry(symbol, tf, breakout_price, current_price, line_type="", ai_direction="",
                       ai_entry=0.0, ai_sl=0.0, ai_tp=0.0, ai_leverage="", ai_deposit_pct=""):
    """Add a breakout event to the log (deduplicates by symbol+tf).
    Stores AI trade parameters for /signals P&L tracking.
    """
    log = load_breakout_log()
    # Don't duplicate same symbol+tf
    if not any(e["symbol"] == symbol and e["tf"] == tf for e in log):
        log.append({
            "symbol": symbol,
            "tf": tf,
            "breakout_price": round(breakout_price, 8),
            "current_price": round(current_price, 8),
            "type": line_type,
            "ai_direction": ai_direction.upper() if ai_direction else "",
            "ai_entry": round(ai_entry, 8) if ai_entry else 0.0,
            "ai_sl": round(ai_sl, 8) if ai_sl else 0.0,
            "ai_tp": round(ai_tp, 8) if ai_tp else 0.0,
            "ai_leverage": ai_leverage,
            "ai_deposit_pct": ai_deposit_pct,
            "time": datetime.now(timezone.utc).isoformat()
        })
        save_breakout_log(log)

def clear_breakout_log():
    save_breakout_log([])

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
