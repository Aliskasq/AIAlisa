"""
tg_background.py — Background async tasks (daily summary, price alerts).
Extracted from tg_listener.py during refactor.
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from config import (BOT_TOKEN, CHAT_ID, load_breakout_log, load_virtual_bank,
                     update_bank_with_trades, VIRTUAL_BANK_POSITION_SIZE,
                     load_ml_breakout_log, load_ml_virtual_bank,
                     update_ml_bank_with_trades, load_price_alerts, save_price_alerts)
from core.tg_state import send_response
from core.tg_reports import (build_signals_close_text, build_ml_signals_close_text,
                              _batch_check_trailing, _fetch_all_prices, calc_trailing_pnl_for_daily)


async def _daily_ml_close(session: aiohttp.ClientSession):
    """23:58 — ML bank daily close: update ML bank, send report, clear log."""
    try:
        log = load_ml_breakout_log()
        if not log:
            logging.info("📭 No ML signals to report.")
            return

        price_map = await _fetch_all_prices(session)
        bank = load_ml_virtual_bank()
        trailing_results = await _batch_check_trailing(session, log, price_map,
                                                        direction_key="ml_direction", sl_key="ml_sl")

        trades_pnl = calc_trailing_pnl_for_daily(log, trailing_results, price_map, bank,
                                                   direction_key="ml_direction")
        update_ml_bank_with_trades(trades_pnl)

        chunks = await build_ml_signals_close_text(session, lang="ru", show_bank=True, bank_already_updated=True)
        if chunks:
            chunks[0] = f"🕐 *ML ежедневный итог (23:58 UTC)*\n\n{chunks[0]}"

        tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        for chunk in chunks:
            await session.post(tg_url, json={"chat_id": CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
            await asyncio.sleep(0.5)

        from config import clear_ml_breakout_log
        clear_ml_breakout_log()
        logging.info(f"✅ ML daily close: {len(trades_pnl)} trades processed, log cleared.")
    except Exception as e:
        logging.error(f"❌ ML daily close error: {e}")


async def auto_trend_sender(session: aiohttp.ClientSession):
    """Background task: 23:58 ML close, 23:59 AI close — daily summary to admin DM."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            # Target: 23:58 for ML close
            target_ml = now.replace(hour=23, minute=58, second=0, microsecond=0)
            if target_ml <= now:
                target_ml += timedelta(days=1)

            sleep_sec = (target_ml - now).total_seconds()
            logging.info(f"📊 Daily summary sleeping {sleep_sec:.0f}s until {target_ml.strftime('%Y-%m-%d %H:%M')} UTC")
            await asyncio.sleep(sleep_sec)

            # === 23:58 — ML BANK CLOSE ===
            await _daily_ml_close(session)

            # Wait 75 seconds for 23:59:15
            await asyncio.sleep(75)

            # === 23:59 — AI SIGNALS CLOSE ===
            log = load_breakout_log()
            if not log:
                logging.info("📭 No AI breakouts to report, skipping daily summary.")
                await asyncio.sleep(120)
                continue

            price_map = await _fetch_all_prices(session)
            bank = load_virtual_bank()
            trailing_results = await _batch_check_trailing(session, log, price_map,
                                                            direction_key="ai_direction", sl_key="ai_sl")

            trades_pnl = calc_trailing_pnl_for_daily(log, trailing_results, price_map, bank,
                                                       direction_key="ai_direction")
            update_bank_with_trades(trades_pnl)

            chunks = await build_signals_close_text(session, lang="ru", show_bank=True, bank_already_updated=True)
            if chunks:
                chunks[0] = f"🕐 *Ежедневный итог (23:59 UTC)*\n\n{chunks[0]}"

            tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            for chunk in chunks:
                await session.post(tg_url, json={"chat_id": CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                await asyncio.sleep(0.5)

            logging.info(f"✅ AI daily summary sent to admin DM.")

            from config import clear_breakout_log
            clear_breakout_log()
            logging.info("🧹 Breakout log cleared for next day.")

            await asyncio.sleep(120)
        except Exception as e:
            logging.error(f"❌ Auto trend sender error: {e}")
            await asyncio.sleep(60)


async def price_alert_monitor(session: aiohttp.ClientSession):
    """Background task: checks price alerts every 30 seconds and notifies users."""
    logging.info("🔔 Price alert monitor started.")
    while True:
        try:
            alerts = load_price_alerts()
            if not alerts:
                await asyncio.sleep(30)
                continue

            symbols = list(set(a["symbol"] for a in alerts))
            prices = {}

            try:
                async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                    if resp.status == 200:
                        tickers = await resp.json()
                        for t in tickers:
                            prices[t["symbol"]] = float(t["price"])
            except Exception:
                await asyncio.sleep(30)
                continue

            triggered = []
            remaining = []

            for alert in alerts:
                sym = alert["symbol"]
                target = alert["target_price"]
                direction = alert["direction"]
                current = prices.get(sym)

                if current is None:
                    remaining.append(alert)
                    continue

                hit = False
                if direction == "above" and current >= target:
                    hit = True
                elif direction == "below" and current <= target:
                    hit = True

                if hit:
                    triggered.append((alert, current))
                else:
                    remaining.append(alert)

            for alert, current in triggered:
                short_sym = alert["symbol"].replace("USDT", "")
                arrow = "🟢📈" if alert["direction"] == "above" else "🔴📉"
                notify_text = (
                    f"{arrow} *PRICE ALERT!*\n\n"
                    f"💰 `${short_sym}` reached `${current:.6f}`\n"
                    f"🎯 Your target: `${alert['target_price']:.6f}`"
                )
                try:
                    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": alert["chat_id"],
                        "text": notify_text,
                        "parse_mode": "Markdown"
                    }
                    await session.post(url, json=payload)
                except Exception as e:
                    logging.error(f"❌ Alert notification error: {e}")

            if len(triggered) > 0:
                save_price_alerts(remaining)
                logging.info(f"🔔 {len(triggered)} price alert(s) triggered, {len(remaining)} remaining.")

            await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"❌ Price alert monitor error: {e}")
            await asyncio.sleep(30)
