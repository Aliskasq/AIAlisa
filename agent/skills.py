import logging
import aiohttp
from config import SQUARE_OPENAPI_KEY

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
