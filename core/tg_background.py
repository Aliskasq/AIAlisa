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
                     load_monitor_breakout_log, load_monitor_virtual_bank,
                     update_monitor_bank_with_trades, load_price_alerts, save_price_alerts)
from core.tg_state import send_response
from core.tg_reports import build_signals_close_text, _batch_check_tp_sl

async def auto_trend_sender(session: aiohttp.ClientSession):
    """Background task: at 23:59:15 UTC daily — send daily summary to admin DM, update virtual bank, clear log."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            target = now.replace(hour=23, minute=59, second=15, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            
            sleep_sec = (target - now).total_seconds()
            logging.info(f"📊 Daily summary sleeping {sleep_sec:.0f}s until {target.strftime('%Y-%m-%d %H:%M')} UTC")
            await asyncio.sleep(sleep_sec)

            log = load_breakout_log()
            if not log:
                logging.info("📭 No breakouts to report, skipping daily summary.")
                await asyncio.sleep(120)
                continue

            # 1. Calculate P&L and update bank BEFORE building text
            price_map = {}
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
                    if resp.status == 200:
                        for p in await resp.json():
                            price_map[p["symbol"]] = float(p["price"])
            except Exception as e:
                logging.error(f"❌ Daily summary price fetch: {e}")

            bank = load_virtual_bank()

            # Check TP/SL via 97 x 15m candles after entry (accurate historical check)
            candle_results = await _batch_check_tp_sl(session, log, price_map)

            trades_pnl = []
            for entry in log:
                sym = entry["symbol"]
                tf = entry.get("tf", "?")
                ai_dir = entry.get("ai_direction", "")

                # Skip signals without AI direction, SKIP, or monitor — they don't affect the bank
                if not ai_dir or ai_dir == "SKIP" or entry.get("is_monitor", False):
                    continue

                ai_sl = entry.get("ai_sl")
                ai_tp = entry.get("ai_tp")
                ai_leverage = entry.get("ai_leverage", 1) or 1
                ai_deposit_pct = entry.get("ai_deposit_pct")
                # Use same priority as build_signals_close_text: ai_entry → current_price → breakout_price
                entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)

                # Use candle-based TP/SL check
                key = f"{sym}_{tf}"
                candle_status, candle_close = candle_results.get(key, ("open", 0))
                status = candle_status

                if status == "tp" and ai_tp:
                    now_price = ai_tp
                elif status == "sl" and ai_sl:
                    now_price = ai_sl
                else:
                    now_price = price_map.get(sym, entry.get("current_price", 0))

                # Calculate P&L based on direction
                if entry_price > 0:
                    if ai_dir == "SHORT":
                        if status == "tp" and ai_tp:
                            pnl_pct = ((entry_price - ai_tp) / entry_price) * 100
                        elif status == "sl" and ai_sl:
                            pnl_pct = ((entry_price - ai_sl) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - now_price) / entry_price) * 100
                    else:
                        if status == "tp" and ai_tp:
                            pnl_pct = ((ai_tp - entry_price) / entry_price) * 100
                        elif status == "sl" and ai_sl:
                            pnl_pct = ((ai_sl - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((now_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                # Apply leverage and position size
                pnl_pct_leveraged = pnl_pct * ai_leverage
                if ai_deposit_pct:
                    position_size = bank["balance"] * (ai_deposit_pct / 100)
                else:
                    position_size = VIRTUAL_BANK_POSITION_SIZE
                pnl_dollar = (pnl_pct_leveraged / 100) * position_size
                trades_pnl.append((sym, pnl_pct_leveraged, pnl_dollar))

            # Update bank with today's results
            update_bank_with_trades(trades_pnl)

            # 2. Build signals close text (all positions closed at market/TP/SL)
            chunks = await build_signals_close_text(session, lang="ru", show_bank=True, bank_already_updated=True)

            # Prepend daily header to first chunk
            if chunks:
                chunks[0] = f"🕐 *Ежедневный итог (23:59 UTC)*\n\n{chunks[0]}"

            # 3. Send to admin DM (not group)
            tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            for chunk in chunks:
                await session.post(tg_url, json={"chat_id": CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                await asyncio.sleep(0.5)

            logging.info(f"✅ Daily summary sent to admin DM.")

            # 4. Clear breakout log for next day
            from config import clear_breakout_log, clear_monitor_breakout_log
            clear_breakout_log()
            logging.info("🧹 Breakout log cleared for next day.")

            # 4b. Update monitor bank with today's monitor trades, then clear monitor log
            try:
                mon_log = load_monitor_breakout_log()
                if mon_log:
                    mon_bank = load_monitor_virtual_bank()
                    mon_candle_results = await _batch_check_tp_sl(session, mon_log, price_map)
                    mon_trades_pnl = []
                    for entry in mon_log:
                        sym = entry["symbol"]
                        tf = entry.get("tf", "?")
                        ai_dir = entry.get("ai_direction", "")
                        if not ai_dir or ai_dir == "SKIP":
                            continue
                        ai_sl = entry.get("ai_sl")
                        ai_tp = entry.get("ai_tp")
                        ai_leverage = entry.get("ai_leverage", 1) or 1
                        ai_deposit_pct = entry.get("ai_deposit_pct")
                        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
                        key = f"{sym}_{tf}"
                        candle_status, _ = mon_candle_results.get(key, ("open", 0))
                        if candle_status == "tp" and ai_tp:
                            now_price = ai_tp
                        elif candle_status == "sl" and ai_sl:
                            now_price = ai_sl
                        else:
                            now_price = price_map.get(sym, entry.get("current_price", 0))
                        if entry_price > 0:
                            if ai_dir == "SHORT":
                                if candle_status == "tp" and ai_tp:
                                    pnl_pct = ((entry_price - ai_tp) / entry_price) * 100
                                elif candle_status == "sl" and ai_sl:
                                    pnl_pct = ((entry_price - ai_sl) / entry_price) * 100
                                else:
                                    pnl_pct = ((entry_price - now_price) / entry_price) * 100
                            else:
                                if candle_status == "tp" and ai_tp:
                                    pnl_pct = ((ai_tp - entry_price) / entry_price) * 100
                                elif candle_status == "sl" and ai_sl:
                                    pnl_pct = ((ai_sl - entry_price) / entry_price) * 100
                                else:
                                    pnl_pct = ((now_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = 0
                        pnl_pct_leveraged = pnl_pct * ai_leverage
                        if ai_deposit_pct:
                            position_size = mon_bank["balance"] * (ai_deposit_pct / 100)
                        else:
                            position_size = VIRTUAL_BANK_POSITION_SIZE
                        pnl_dollar = (pnl_pct_leveraged / 100) * position_size
                        mon_trades_pnl.append((sym, pnl_pct_leveraged, pnl_dollar))
                    update_monitor_bank_with_trades(mon_trades_pnl)
                    logging.info(f"✅ Monitor bank updated with {len(mon_trades_pnl)} trades.")
                clear_monitor_breakout_log()
                logging.info("🧹 Monitor breakout log cleared for next day.")
            except Exception as e:
                logging.error(f"❌ Monitor bank daily update error: {e}")

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

            # Get all unique symbols
            symbols = list(set(a["symbol"] for a in alerts))
            prices = {}

            # Fetch current prices (batch via futures ticker)
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
                direction = alert["direction"]  # "above" or "below"
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

            # Send notifications for triggered alerts
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

            # Save remaining alerts
            if len(triggered) > 0:
                save_price_alerts(remaining)
                logging.info(f"🔔 {len(triggered)} price alert(s) triggered, {len(remaining)} remaining.")

            await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"❌ Price alert monitor error: {e}")
            await asyncio.sleep(30)


