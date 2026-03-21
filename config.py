import json
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import aiohttp

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROUP_CHAT_ID = os.getenv("TELEGRAM_GROUP_CHAT_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")
SQUARE_OPENAPI_KEY = os.getenv("SQUARE_OPENAPI_KEY")

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
VIRTUAL_BANK_START = 10000.0

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
    log = load_breakout_log()
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
        logging.error(f"Error reading price alerts: {e}")

# --- VIRTUAL BANK ---
VIRTUAL_BANK_FILE = "data/virtual_bank.json"
VIRTUAL_BANK_START = 10000.0

def load_virtual_bank():
    if os.path.exists(VIRTUAL_BANK_FILE):
        try:
            with open(VIRTUAL_BANK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"balance": VIRTUAL_BANK_START, "history": []}

def save_virtual_bank(bank):
    try:
        os.makedirs(os.path.dirname(VIRTUAL_BANK_FILE), exist_ok=True)
        with open(VIRTUAL_BANK_FILE, "w") as f:
            json.dump(bank, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing virtual bank: {e}")

def compute_daily_stats(day_log: list, starting_balance: float):
    """Compute daily stats from breakout log entries. Returns dict with keys:
    wins, losses, pending, total_pnl, new_balance, winrate, total_closed
    """
    if not day_log:
        return {
            "wins": 0, "losses": 0, "pending": 0, "total_pnl": 0.0,
            "new_balance": starting_balance, "winrate": 0.0, "total_closed": 0
        }

    wins = losses = pending = 0
    total_pnl = 0.0
    for entry in day_log:
        sym = entry["symbol"]
        ai_dir = entry.get("ai_direction", "")
        ai_entry = entry.get("ai_entry", 0.0)
        ai_sl = entry.get("ai_sl", 0.0)
        ai_tp = entry.get("ai_tp", 0.0)
        ai_lev_str = entry.get("ai_leverage", "")
        ai_dep_pct_str = entry.get("ai_deposit_pct", "")
        bp = entry.get("breakout_price", 0)

        leverage = 1
        if ai_lev_str:
            try:
                leverage = int(ai_lev_str.lower().replace("x", ""))
            except Exception:
                leverage = 1
        if leverage < 1:
            leverage = 1

        dep_pct = 0.10
        if ai_dep_pct_str:
            try:
                dep_pct = float(ai_dep_pct_str.replace("%", "")) / 100
            except Exception:
                dep_pct = 0.10

        margin_used = starting_balance * dep_pct

        if not ai_dir or ai_entry <= 0 or ai_sl <= 0 or ai_tp <= 0:
            now_price = entry.get("current_price", bp)
            pnl_pct = ((now_price - bp) / bp) * 100 if bp > 0 else 0
            pnl_usd = margin_used * (pnl_pct / 100)
        else:
            now_price = entry.get("current_price", ai_entry)
            if ai_dir == "LONG":
                pnl_pct = ((now_price - ai_entry) / ai_entry) * 100 * leverage
            else:
                pnl_pct = ((ai_entry - now_price) / ai_entry) * 100 * leverage
            pnl_usd = margin_used * (pnl_pct / 100)

        total_pnl += pnl_usd
        # Determine status: we don't have SL/TP hit for today's signals yet; treat as pending
        pending += 1

    total_closed = wins + losses
    winrate = (wins / total_closed * 100) if total_closed > 0 else 0
    new_balance = starting_balance + total_pnl

    return {
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "total_pnl": total_pnl,
        "new_balance": new_balance,
        "winrate": winrate,
        "total_closed": total_closed
    }
