import logging
import time
import asyncio
import aiohttp
from config import BOT_TOKEN, CHAT_ID

LAST_WEIGHT_WARNING = 0

async def send_status_msg(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Failed to send status message: {e}")

async def wait_for_weight(session, threshold=1800):
    """Wait until API weight drops below threshold. Checks every 5s."""
    weight = getattr(session, 'last_weight', 0)
    if isinstance(weight, str):
        weight = int(weight) if weight.isdigit() else 0
    if weight > threshold:
        wait_secs = 60 - (time.time() % 60) + 2  # Wait until next minute resets weight
        logging.info(f"⏸️ API weight {weight}/{threshold} — pausing {wait_secs:.0f}s for reset...")
        await asyncio.sleep(wait_secs)

async def fetch_klines(session, symbol, interval, limit=199):
    global LAST_WEIGHT_WARNING
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url, timeout=10) as resp:
            weight_str = resp.headers.get('X-MBX-USED-WEIGHT-1M', '0')
            weight = int(weight_str) if weight_str.isdigit() else 0
            session.last_weight = weight
            
            if weight > 2000:
                current_time = time.time()
                if current_time - LAST_WEIGHT_WARNING > 60:
                    LAST_WEIGHT_WARNING = current_time
                    logging.warning(f"🚨 API WEIGHT ALERT: {weight}/2400!")
                    asyncio.create_task(send_status_msg(
                        f"⚠️ **CRITICAL BINANCE API WEIGHT** ⚠️\n\n"
                        f"📈 Current weight: `{weight}/2400`\n"
                        f"Bot is automatically slowing down."
                    ))
                # Wait for next minute to reset weight
                wait_secs = 60 - (time.time() % 60) + 2
                logging.info(f"⏸️ Weight {weight} > 2000 — waiting {wait_secs:.0f}s for reset...")
                await asyncio.sleep(wait_secs)
            
            if resp.status == 200:
                raw = await resp.json()
                if not raw: return None
                return [
                    {
                        'open_time': int(c[0]), 'open': float(c[1]), 'high': float(c[2]),
                        'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])
                    } for c in raw
                ]
            elif resp.status == 429:
                logging.warning(f"⚠️ BAN 429! Weight: {weight}. Pausing for 30 sec...")
                await asyncio.sleep(30)
                return None
            else:
                return None
    except Exception as e:
        return None

async def get_usdt_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    usdt_symbols = [
                        s["symbol"]
                        for s in data["symbols"]
                        if s["quoteAsset"] == "USDT"
                        and s["contractType"] == "PERPETUAL"
                        and s["status"] == "TRADING"
                    ]
                    logging.info(f"🔍 Found {len(usdt_symbols)} USDT PERPETUAL symbols")
                    return usdt_symbols
                else:
                    logging.error(f"❌ Error fetching exchangeInfo: {response.status}")
                    return []
    except Exception as e:
        logging.error(f"❌ Network error fetching exchangeInfo: {e}")
        return []

# --- FUNDING ---
async def fetch_funding_rate(session, symbol):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    try:
        async with session.get(url, params=params, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                rate = float(data.get("lastFundingRate", 0)) * 100
                return f"{rate:.4f}%"
            return "Unknown"
    except Exception as e:
        return "Unknown"
