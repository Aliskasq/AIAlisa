import os
import json
import logging
import aiohttp
from config import ALERTS_FILE, SQUARE_OPENAPI_KEY

# ---------------------------------------------------------
# BINANCE WEB3 SKILLS (Direct API)
# ---------------------------------------------------------

async def get_smart_money_signals(symbol: str) -> str:
    """Check Smart Money / Whale buy and sell signals for a specific token."""
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/web/signal/smart-money"
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity", "User-Agent": "binance-web3/1.0 (Skill)"}
    payload = {"page": 1, "pageSize": 50, "chainId": "56"} 
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    signals = data.get("data", [])
                    base_coin = symbol.upper().replace("USDT", "")
                    for sig in signals:
                        if base_coin in sig.get("ticker", "").upper():
                            direction = sig.get("direction", "unknown").upper()
                            count = sig.get("smartMoneyCount", 0)
                            return f"Found {count} Smart Money addresses showing {direction} signals for {symbol}."
    except Exception: pass
    return "No obvious Smart Money on-chain activity detected."

async def get_unified_token_rank(rank_type: int = 10) -> str:
    """
    Get market token rankings.
    rank_type mapping: 10=Trending, 11=Top Search, 20=Binance Alpha picks.
    Use this to understand macro market trends.
    """
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/unified/rank/list"
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity", "User-Agent": "binance-web3/2.0 (Skill)"}
    payload = {"rankType": rank_type, "chainId": "56", "period": 50, "size": 5}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=5) as resp:
                data = await resp.json()
                tokens = data.get("data", {}).get("tokens", [])
                trends = [f"{t.get('symbol')} (Change: {t.get('percentChange24h', '0')}%)" for t in tokens[:5]]
                return f"Top Ranked Tokens (Type {rank_type}): {', '.join(trends)}"
    except Exception: pass
    return "Could not fetch unified token rank."

async def get_social_hype_leaderboard() -> str:
    """Get the top tokens based on Social Hype and Sentiment."""
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/social/hype/rank/leaderboard?chainId=56&sentiment=All&targetLanguage=en&timeRange=1"
    headers = {"Accept-Encoding": "identity", "User-Agent": "binance-web3/2.0 (Skill)"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as resp:
                data = await resp.json()
                items = data.get("data", {}).get("leaderBoardList", [])
                results = [f"{i.get('metaInfo', {}).get('symbol')} ({i.get('socialHypeInfo', {}).get('sentiment')})" for i in items[:5]]
                return f"Current Social Hype Leaders: {', '.join(results)}"
    except Exception: pass
    return "Could not fetch social hype leaderboard."

async def get_smart_money_inflow_rank() -> str:
    """Discover which tokens Smart Money is buying the most right now (Net Inflow)."""
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/tracker/wallet/token/inflow/rank/query"
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity", "User-Agent": "binance-web3/2.0 (Skill)"}
    payload = {"chainId": "56", "period": "24h", "tagType": 2}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=5) as resp:
                data = await resp.json()
                items = data.get("data", [])
                results = [f"{i.get('tokenName')} (Inflow: ${float(i.get('inflow', 0)):.0f})" for i in items[:5]]
                return f"Top Smart Money Inflow Tokens: {', '.join(results)}"
    except Exception: pass
    return "Could not fetch smart money inflow ranks."

async def get_meme_rank() -> str:
    """Find the top meme tokens most likely to break out."""
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/market/token/pulse/exclusive/rank/list?chainId=56"
    headers = {"Accept-Encoding": "identity", "User-Agent": "binance-web3/2.0 (Skill)"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as resp:
                data = await resp.json()
                items = data.get("data", {}).get("tokens", [])
                results = [f"{i.get('symbol')} (Score: {i.get('score')})" for i in items[:5]]
                return f"Top Breakthrough Meme Tokens: {', '.join(results)}"
    except Exception: pass
    return "Could not fetch meme rank."

async def get_address_pnl_rank() -> str:
    """Get top performing trader addresses (PnL & Win Rate)."""
    url = "https://web3.binance.com/bapi/defi/v1/public/wallet-direct/market/leaderboard/query?tag=ALL&chainId=CT_501&period=30d&pageSize=3"
    headers = {"Accept-Encoding": "identity", "User-Agent": "binance-web3/2.0 (Skill)"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as resp:
                data = await resp.json()
                items = data.get("data", {}).get("data", [])
                results = [f"Trader {i.get('addressLabel', 'Unknown')} (WinRate: {i.get('winRate')}, PnL: ${i.get('realizedPnl')})" for i in items]
                return f"Top 30d Traders: {', '.join(results)}"
    except Exception: pass
    return "Could not fetch trader PnL leadership."

# ---------------------------------------------------------
# EXPORT AND PUBLICATION (Called directly by bot)
# ---------------------------------------------------------
async def post_to_binance_square(text: str) -> str:
    """Publications to Binance Square."""
    if not SQUARE_OPENAPI_KEY:
        return "❌ Error: SQUARE_OPENAPI_KEY is not set."
    url = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"
    headers = {"X-Square-OpenAPI-Key": SQUARE_OPENAPI_KEY, "Content-Type": "application/json", "clienttype": "binanceSkill"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"bodyTextOnly": text}, timeout=10) as resp:
                data = await resp.json()
                if data.get("code") == "000000":
                    post_id = data.get("data", {}).get("id", "unknown")
                    return f"✅ Posted to Square! URL: https://www.binance.com/square/post/{post_id}"
                return f"❌ POST Error: {data.get('message')}"
    except Exception as e: return f"❌ Connection Error: {e}"
