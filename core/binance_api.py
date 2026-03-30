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


async def fetch_funding_history(session, symbol):
    """Fetch last 3 funding rate epochs with dynamics: 0.0100% → 0.0150% → 0.0250%"""
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=3"
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                rates = [f"{float(item['fundingRate']) * 100:.4f}%" for item in data]
                if rates:
                    return " → ".join(rates)
    except Exception:
        pass
    return "Unknown"


# --- MARKET POSITIONING (OI, L/S Ratio, Taker Volume) ---
async def fetch_market_positioning(session, symbol):
    """
    Fetch Open Interest, Long/Short ratios, and Taker buy/sell volume.
    Returns pre-interpreted dict for AI prompt.
    API weight: ~5 total (1 per endpoint).
    """
    result = {}

    # 1. Open Interest (current + 24h change)
    try:
        # Current OI
        async with session.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol}, timeout=5
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                result["oi_current"] = float(data.get("openInterest", 0))

        # OI history (last 4 periods of 4h = ~16h for trend)
        async with session.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "4h", "limit": 4}, timeout=5
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if len(data) >= 2:
                    oi_old = float(data[0].get("sumOpenInterest", 0))
                    oi_new = float(data[-1].get("sumOpenInterest", 0))
                    if oi_old > 0:
                        result["oi_change_pct"] = ((oi_new - oi_old) / oi_old) * 100
    except Exception as e:
        logging.debug(f"OI fetch error {symbol}: {e}")

    # 2. Global Long/Short Account Ratio
    try:
        async with session.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": "4h", "limit": 1}, timeout=5
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    result["ls_account_long"] = float(data[-1].get("longAccount", 0.5))
                    result["ls_account_short"] = float(data[-1].get("shortAccount", 0.5))
    except Exception as e:
        logging.debug(f"L/S account ratio error {symbol}: {e}")

    # 3. Top Trader Long/Short Position Ratio
    try:
        async with session.get(
            "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            params={"symbol": symbol, "period": "4h", "limit": 1}, timeout=5
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    result["top_long"] = float(data[-1].get("longAccount", 0.5))
                    result["top_short"] = float(data[-1].get("shortAccount", 0.5))
    except Exception as e:
        logging.debug(f"Top trader ratio error {symbol}: {e}")

    # 4. Taker Buy/Sell Volume
    try:
        async with session.get(
            "https://fapi.binance.com/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "4h", "limit": 1}, timeout=5
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    result["taker_buy_vol"] = float(data[-1].get("buyVol", 0))
                    result["taker_sell_vol"] = float(data[-1].get("sellVol", 0))
    except Exception as e:
        logging.debug(f"Taker volume error {symbol}: {e}")

    return result


def format_positioning_text(pos: dict, current_price: float = 0) -> str:
    """Format positioning data into pre-interpreted text for AI prompt."""
    if not pos:
        return ""

    lines = ["[MARKET POSITIONING — OI, Long/Short Ratio, Taker Volume]"]

    # OI
    oi = pos.get("oi_current", 0)
    oi_change = pos.get("oi_change_pct")
    if oi > 0:
        oi_text = f"Open Interest: {oi:,.0f} contracts"
        if oi_change is not None:
            if oi_change > 5:
                oi_text += f" | 📈 OI RISING {oi_change:+.1f}% (new positions opening — volatile move coming)"
            elif oi_change < -5:
                oi_text += f" | 📉 OI FALLING {oi_change:+.1f}% (positions closing — trend weakening)"
            else:
                oi_text += f" | ⚪ OI stable ({oi_change:+.1f}%)"
        lines.append(oi_text)

    # L/S Account Ratio
    long_acct = pos.get("ls_account_long", 0)
    short_acct = pos.get("ls_account_short", 0)
    if long_acct > 0 or short_acct > 0:
        long_pct = long_acct * 100
        short_pct = short_acct * 100
        if long_pct > 60:
            crowd = f"⚠️ CROWD IS LONG ({long_pct:.0f}%) — squeeze risk DOWN"
        elif short_pct > 60:
            crowd = f"⚠️ CROWD IS SHORT ({short_pct:.0f}%) — squeeze risk UP"
        else:
            crowd = f"⚪ Balanced ({long_pct:.0f}%L / {short_pct:.0f}%S)"
        lines.append(f"Accounts L/S: {crowd}")

    # Top Trader Ratio
    top_long = pos.get("top_long", 0)
    top_short = pos.get("top_short", 0)
    if top_long > 0 or top_short > 0:
        tl_pct = top_long * 100
        ts_pct = top_short * 100
        if tl_pct > 60:
            top_text = f"🟢 TOP TRADERS LONG ({tl_pct:.0f}%) — smart money bullish"
        elif ts_pct > 60:
            top_text = f"🔴 TOP TRADERS SHORT ({ts_pct:.0f}%) — smart money bearish"
        else:
            top_text = f"⚪ Top traders balanced ({tl_pct:.0f}%L / {ts_pct:.0f}%S)"
        lines.append(f"Top Traders: {top_text}")

    # Taker Volume
    buy_vol = pos.get("taker_buy_vol", 0)
    sell_vol = pos.get("taker_sell_vol", 0)
    total_vol = buy_vol + sell_vol
    if total_vol > 0:
        buy_pct = (buy_vol / total_vol) * 100
        sell_pct = (sell_vol / total_vol) * 100
        if buy_pct > 55:
            taker = f"🟢 BUYERS AGGRESSIVE ({buy_pct:.0f}% buy volume)"
        elif sell_pct > 55:
            taker = f"🔴 SELLERS AGGRESSIVE ({sell_pct:.0f}% sell volume)"
        else:
            taker = f"⚪ Balanced ({buy_pct:.0f}% buy / {sell_pct:.0f}% sell)"
        lines.append(f"Taker Volume: {taker}")

    # Liquidation Zones (estimated based on common leverages)
    if current_price > 0:
        liq_lines = ["Estimated Liquidation Zones:"]
        for lev in [3, 5, 10, 25]:
            long_liq = current_price * (1 - 1/lev)  # long liquidation = price drops
            short_liq = current_price * (1 + 1/lev)  # short liquidation = price rises
            liq_lines.append(f"  {lev}x: Long liq ≈{long_liq:.6f} | Short liq ≈{short_liq:.6f}")
        lines.extend(liq_lines)

    return "\n".join(lines)
