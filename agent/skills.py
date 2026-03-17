import os
import json
import logging
import aiohttp
from config import ALERTS_FILE, SQUARE_OPENAPI_KEY

# ---------------------------------------------------------
# OPENCLAW BINANCE WEB3 SKILLS
# ---------------------------------------------------------

# --- OpenClaw SDK Skill Routing (lazy singleton) ---
try:
    import cmdop
    _cmdop_available = True
except ImportError:
    cmdop = None
    _cmdop_available = False

_sdk_client = None

async def _get_sdk_client():
    """Get or create CMDOP SDK client for OpenClaw Skill routing."""
    global _sdk_client
    if _sdk_client is None and _cmdop_available:
        try:
            api_key = os.getenv("CMDOP_API_KEY")
            if api_key:
                _sdk_client = cmdop.AsyncCMDOPClient.remote(api_key=api_key)
        except Exception:
            pass
    return _sdk_client

async def _try_sdk_skill(skill_name: str, prompt: str) -> str | None:
    """Attempt to execute a skill via OpenClaw Skills SDK.
    
    Returns the skill result text if successful, None otherwise.
    Transparent fallback — caller proceeds to direct HTTP if None.
    """
    client = await _get_sdk_client()
    if client:
        try:
            result = await client.skills.run(
                skill_name, prompt,
                options=cmdop.SkillRunOptions(timeout_seconds=15)
            )
            if result.success and result.text:
                logging.info(f"✅ [OpenClaw SDK] Skill '{skill_name}' executed successfully")
                return result.text
        except Exception as e:
            logging.debug(f"⚙️ SDK skill '{skill_name}' → direct HTTP fallback: {e}")
    return None

async def get_smart_money_signals(symbol: str) -> str:
    """OpenClaw Skill: Check Smart Money / Whale buy and sell signals for a specific token."""
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("smart_money_signals", f"Analyze Smart Money whale activity for {symbol} on BSC (ChainId: 56)")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
    OpenClaw Skill: Get market token rankings.
    rank_type mapping: 10=Trending, 11=Top Search, 20=Binance Alpha picks.
    Use this to understand macro market trends.
    """
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("unified_token_rank", f"Get market token rankings (type={rank_type}, Trending/TopSearch) on BSC")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
    """OpenClaw Skill: Get the top tokens based on Social Hype and Sentiment."""
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("social_hype_leaderboard", "Get top tokens by social hype and community sentiment on BSC")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
    """OpenClaw Skill: Discover which tokens Smart Money is buying the most right now (Net Inflow)."""
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("smart_money_inflow_rank", "Discover top tokens by Smart Money net inflow on BSC (24h)")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
    """OpenClaw Skill: Find the top meme tokens most likely to break out."""
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("meme_rank", "Find top meme tokens with highest breakout probability on BSC")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
    """OpenClaw Skill: Get top performing trader addresses (PnL & Win Rate)."""
    # --- OpenClaw Skills SDK routing ---
    sdk_result = await _try_sdk_skill("address_pnl_rank", "Get top performing trader addresses by PnL and Win Rate (30d)")
    if sdk_result:
        return sdk_result
    # --- Fallback: Direct Binance Web3 API ---
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
