"""
categories.py — Coin sector/category logic.

Loads coin_categories.json and scan_settings.json.
Provides helpers: get_sectors, should_scan, add/remove sector, etc.
"""
import json
import logging
import os
from datetime import datetime, timezone

CATEGORIES_FILE = "data/coin_categories.json"
SCAN_SETTINGS_FILE = "data/scan_settings.json"

# All known sectors (canonical list)
ALL_SECTORS = [
    "🤖 AI", "🏦 DeFi", "🏗 L1", "🔗 L2", "💎 Meme", "🎮 Gaming",
    "🖼 NFT", "💰 RWA", "🔮 Oracle", "💾 Infra", "📊 Stocks",
    "🥇 Metals/Commodities", "👑 Blue Chip", "📦 Other", "⚽ Fan Token",
    "💳 Payments", "📈 ETFs",
]

# Short labels for buttons (to fit Telegram callback limits)
SECTOR_SHORT = {
    "🤖 AI": "🤖 AI",
    "🏦 DeFi": "🏦 DeFi",
    "🏗 L1": "🏗 L1",
    "🔗 L2": "🔗 L2",
    "💎 Meme": "💎 Meme",
    "🎮 Gaming": "🎮 Gaming",
    "🖼 NFT": "🖼 NFT",
    "💰 RWA": "💰 RWA",
    "🔮 Oracle": "🔮 Oracle",
    "💾 Infra": "💾 Infra",
    "📊 Stocks": "📊 Stocks",
    "🥇 Metals/Commodities": "🥇 M/C",
    "👑 Blue Chip": "👑 Blue Chip",
    "📦 Other": "📦 Other",
    "⚽ Fan Token": "⚽ Fan",
    "💳 Payments": "💳 Pay",
    "📈 ETFs": "📈 ETFs",
}

# Emoji-only for compact display in reports (first emoji of sector)
SECTOR_EMOJI = {
    "🤖 AI": "🤖",
    "🏦 DeFi": "🏦",
    "🏗 L1": "🏗",
    "🔗 L2": "🔗",
    "💎 Meme": "💎",
    "🎮 Gaming": "🎮",
    "🖼 NFT": "🖼",
    "💰 RWA": "💰",
    "🔮 Oracle": "🔮",
    "💾 Infra": "💾",
    "📊 Stocks": "📊",
    "🥇 Metals/Commodities": "🥇",
    "👑 Blue Chip": "👑",
    "📦 Other": "📦",
    "⚽ Fan Token": "⚽",
    "💳 Payments": "💳",
    "📈 ETFs": "📈",
}


# ── Load / Save ──────────────────────────────────────────────────────────

def load_categories() -> dict:
    """Load coin_categories.json. Returns full dict (includes _updated, _sectors, and symbol→sectors)."""
    if os.path.exists(CATEGORIES_FILE):
        try:
            with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ Error reading {CATEGORIES_FILE}: {e}")
    return {"_updated": "", "_sectors": ALL_SECTORS[:]}


def save_categories(data: dict):
    """Save coin_categories.json."""
    data["_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(CATEGORIES_FILE), exist_ok=True)
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"❌ Error writing {CATEGORIES_FILE}: {e}")


def load_scan_settings() -> dict:
    """Load scan_settings.json."""
    defaults = {
        "enabled_sectors": [s for s in ALL_SECTORS if s not in ("📊 Stocks", "🥇 Metals/Commodities", "📈 ETFs")],
        "disabled_sectors": ["📊 Stocks", "🥇 Metals/Commodities", "📈 ETFs"],
        "scan_unknown": True,
    }
    if os.path.exists(SCAN_SETTINGS_FILE):
        try:
            with open(SCAN_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception as e:
            logging.error(f"❌ Error reading {SCAN_SETTINGS_FILE}: {e}")
    return defaults


def save_scan_settings(settings: dict):
    """Save scan_settings.json."""
    try:
        os.makedirs(os.path.dirname(SCAN_SETTINGS_FILE), exist_ok=True)
        with open(SCAN_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"❌ Error writing {SCAN_SETTINGS_FILE}: {e}")


# ── Query helpers ────────────────────────────────────────────────────────

def get_sectors(symbol: str) -> list:
    """Get list of sectors for a symbol (e.g. ['🤖 AI', '💎 Meme']).
    Returns empty list if symbol not found."""
    cats = load_categories()
    # Normalize: add USDT if missing
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    return cats.get(symbol, [])


def get_sector_emoji(symbol: str) -> str:
    """Get emoji of first sector for display. Returns '❓' if no sector."""
    sectors = get_sectors(symbol)
    if not sectors:
        return "❓"
    return SECTOR_EMOJI.get(sectors[0], "❓")


def get_sector_label(symbol: str) -> str:
    """Get formatted sector string for caption: '🤖 AI / 💎 Meme' or '❓'."""
    sectors = get_sectors(symbol)
    if not sectors:
        return "❓"
    return " / ".join(sectors)


def should_scan(symbol: str) -> bool:
    """Check if symbol should be scanned based on scan_settings.
    
    1. Symbol in coin_categories → check if ANY of its sectors is in enabled_sectors
       - Yes → scan
       - No (all sectors disabled) → skip
    2. Symbol NOT in coin_categories → check scan_unknown
       - True → scan
       - False → skip
    """
    settings = load_scan_settings()
    cats = load_categories()
    
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    
    sectors = cats.get(symbol)
    
    if sectors is None or len(sectors) == 0:
        # Unknown coin — use scan_unknown setting
        return settings.get("scan_unknown", True)
    
    # Check if any sector is enabled
    enabled = set(settings.get("enabled_sectors", []))
    return any(s in enabled for s in sectors)


def get_symbols_by_sector(sector: str) -> list:
    """Get all symbols in a given sector."""
    cats = load_categories()
    result = []
    for key, val in cats.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list) and sector in val:
            result.append(key)
    result.sort()
    return result


def get_unknown_symbols(all_symbols: list) -> list:
    """Get symbols that are NOT in coin_categories."""
    cats = load_categories()
    unknown = []
    for sym in all_symbols:
        if sym not in cats:
            unknown.append(sym)
    unknown.sort()
    return unknown


def get_sector_counts() -> dict:
    """Get {sector: count} for all sectors + '❓ Без сектора' placeholder."""
    cats = load_categories()
    counts = {s: 0 for s in ALL_SECTORS}
    for key, val in cats.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list):
            for s in val:
                if s in counts:
                    counts[s] += 1
    return counts


# ── Mutation helpers ─────────────────────────────────────────────────────

def add_sector(symbol: str, sector: str) -> bool:
    """Add sector to a symbol. Returns True if added, False if already present."""
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    cats = load_categories()
    if symbol not in cats:
        cats[symbol] = []
    if sector in cats[symbol]:
        return False
    cats[symbol].append(sector)
    save_categories(cats)
    return True


def remove_sector(symbol: str, sector: str) -> bool:
    """Remove sector from a symbol. Returns True if removed."""
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    cats = load_categories()
    if symbol not in cats:
        return False
    if sector not in cats[symbol]:
        return False
    cats[symbol].remove(sector)
    if not cats[symbol]:
        del cats[symbol]
    save_categories(cats)
    return True


def set_sectors(symbol: str, sectors: list):
    """Replace all sectors for a symbol."""
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    cats = load_categories()
    if sectors:
        cats[symbol] = sectors
    elif symbol in cats:
        del cats[symbol]
    save_categories(cats)


def toggle_scan_sector(sector: str) -> bool:
    """Toggle sector enabled/disabled in scan_settings.
    Returns True if now enabled, False if now disabled."""
    settings = load_scan_settings()
    enabled = settings.get("enabled_sectors", [])
    disabled = settings.get("disabled_sectors", [])
    
    if sector in enabled:
        enabled.remove(sector)
        if sector not in disabled:
            disabled.append(sector)
        save_scan_settings(settings)
        return False
    else:
        if sector not in enabled:
            enabled.append(sector)
        if sector in disabled:
            disabled.remove(sector)
        save_scan_settings(settings)
        return True


def toggle_scan_unknown() -> bool:
    """Toggle scan_unknown. Returns new value."""
    settings = load_scan_settings()
    settings["scan_unknown"] = not settings.get("scan_unknown", True)
    save_scan_settings(settings)
    return settings["scan_unknown"]
