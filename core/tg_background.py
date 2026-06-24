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
                     load_manual_alerts, save_manual_alerts,
                     load_price_alerts, save_price_alerts)
from core.tg_state import send_response
from core.tg_reports import (build_signals_close_text,
                              _batch_check_trailing, _fetch_all_prices, calc_trailing_pnl_for_daily)


async def auto_trend_sender(session: aiohttp.ClientSession):
    """Background task: 23:59 AI close — daily summary to admin DM."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            target = now.replace(hour=23, minute=59, second=15, microsecond=0)
            if target <= now:
                target += timedelta(days=1)

            sleep_sec = (target - now).total_seconds()
            logging.info(f"📊 Daily summary sleeping {sleep_sec:.0f}s until {target.strftime('%Y-%m-%d %H:%M')} UTC")
            await asyncio.sleep(sleep_sec)

            # === 23:59 — AI SIGNALS CLOSE ===
            log = load_breakout_log()
            if not log:
                logging.info("📭 No AI breakouts to report, skipping daily summary.")
                await asyncio.sleep(120)
                continue

            price_map = await _fetch_all_prices(session)
            bank = load_virtual_bank()
            trailing_results = await _batch_check_trailing(session, log, price_map,
                                                            direction_key="ai_direction", sl_key="ai_sl",
                                                            bank_name="signals")

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


async def manual_alert_monitor(session: aiohttp.ClientSession):
    """
    Background task: checks manual trendline alerts every 5 minutes.
    Loads 50 candles on 5m TF (weight=1 per symbol) and checks if ANY candle
    touched the line (high/low vs line price). Any touch = immediate alert
    to admin DM (CHAT_ID) with chart. No AI analysis.
    """
    import math
    import os
    import numpy as np
    import pandas as pd
    from core.binance_api import fetch_klines
    from core.chart_drawer import draw_alert_chart

    logging.info("📐 Manual alert monitor started (5m candles, every 5 min).")

    while True:
        try:
            alerts = load_manual_alerts()
            if not alerts:
                await asyncio.sleep(300)
                continue

            triggered = []
            remaining = []

            for alert in alerts:
                sym = alert["symbol"]
                slope = alert.get("slope", 0)
                intercept = alert.get("intercept", 0)
                base_idx = alert.get("base_idx", 198)
                base_open_time = alert.get("base_open_time", 0)
                alert_tf_ms = alert.get("tf_ms", 14400000)  # original TF in ms

                try:
                    # Fetch 50 candles on 5m (weight = 1)
                    raw_5m = await fetch_klines(session, sym, "5m", 50)
                    if not raw_5m:
                        remaining.append(alert)
                        continue

                    hit = False
                    touch_price = 0
                    touch_type = ""  # "wick" or "body"
                    # Only check candles AFTER alert creation (don't trigger on existing data)
                    monitor_from = alert.get("monitor_from_time", 0)

                    for candle in raw_5m:
                        candle_open_time = int(candle['open_time'])

                        # Skip candles that existed when alert was created
                        if monitor_from and candle_open_time < monitor_from:
                            continue

                        # Calculate how many ORIGINAL TF candles passed since base
                        candles_passed = (candle_open_time - base_open_time) / alert_tf_ms
                        if candles_passed < 0:
                            continue
                        current_math_x = base_idx + candles_passed

                        # Line price at this 5m candle's time
                        line_price = math.exp(slope * current_math_x + intercept)

                        c_high = float(candle['high'])
                        c_low = float(candle['low'])
                        c_open = float(candle['open'])
                        c_close = float(candle['close'])
                        body_top = max(c_open, c_close)
                        body_bot = min(c_open, c_close)

                        # Touch detection based on MODE (not slope)
                        # 0.01% tolerance for near-touches; full crosses also count
                        tol = line_price * 0.0001  # 0.01%
                        alert_mode = alert.get("mode", "high")

                        if alert_mode in ("low", "body_bot", "date_bottom", "date_body_bot", "date_low"):
                            # SUPPORT line → alert when price breaks DOWN to line
                            near_touch = (line_price - tol <= c_low <= line_price + tol)
                            full_cross = (c_low <= line_price <= c_high)
                            if near_touch or full_cross:
                                hit = True
                                touch_price = c_close
                                touch_type = "body" if body_bot <= line_price else "wick"
                                break

                        elif alert_mode in ("high", "body_top", "date_top", "date_body_top", "date_high"):
                            # RESISTANCE line → alert when price breaks UP to line
                            near_touch = (line_price - tol <= c_high <= line_price + tol)
                            full_cross = (c_low <= line_price <= c_high)
                            if near_touch or full_cross:
                                hit = True
                                touch_price = c_close
                                touch_type = "body" if body_top >= line_price else "wick"
                                break

                    if hit:
                        triggered.append((alert, touch_price, line_price, touch_type))
                    else:
                        remaining.append(alert)

                except Exception as e:
                    logging.error(f"❌ Manual alert check error for {sym}: {repr(e)}")
                    remaining.append(alert)

            # Send alerts to admin DM (CHAT_ID)
            for alert, touch_price, line_price, touch_type in triggered:
                sym = alert["symbol"]
                tf = alert["tf"]
                short_sym = sym.replace("USDT", "")
                diff_pct = ((touch_price / line_price) - 1) * 100
                # logging.info(f"🔔 TRIGGER detected: {sym} ({tf}) touch={touch_price:.8g} line={line_price:.8g} type={touch_type}")

                # Draw chart on original TF with ALL remaining lines for this symbol+tf
                chart_path = None
                try:
                    tf_map = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
                    binance_interval = tf_map.get(tf, "4h")
                    raw = await fetch_klines(session, sym, binance_interval, 199)
                    if raw:
                        df = pd.DataFrame(raw)
                        # Include triggered line + all remaining lines for this symbol+tf
                        all_lines_for_chart = [
                            {'price_a': a['price_a'], 'price_b': a['price_b'],
                             'index_a': a['index_a'], 'index_b': a['index_b'],
                             'base_open_time': a.get('base_open_time', 0),
                             'base_idx': a.get('base_idx', 0),
                             'tf_ms': a.get('tf_ms', 0),
                             'color_idx': a.get('color_idx', 0)}
                            for a in remaining if a['symbol'] == sym and a['tf'] == tf
                        ]
                        # Also add the triggered line itself (shown as the one that fired)
                        all_lines_for_chart.append({
                            'price_a': alert['price_a'], 'price_b': alert['price_b'],
                            'index_a': alert['index_a'], 'index_b': alert['index_b'],
                            'base_open_time': alert.get('base_open_time', 0),
                            'base_idx': alert.get('base_idx', 0),
                            'tf_ms': alert.get('tf_ms', 0),
                            'color_idx': alert.get('color_idx', 0),
                        })
                        # Compute SMC overlay for the chart
                        _alert_smc = None
                        try:
                            from core.smc import analyze_smc
                            _alert_smc = analyze_smc(df, tf, symbol=sym)
                        except Exception as e:
                            logging.error(f"❌ SMC for manual alert {sym}: {repr(e)}")
                        chart_path = await draw_alert_chart(sym, df, all_lines_for_chart, tf, smc_overlay=_alert_smc)
                        # logging.info(f"📊 Chart drawn: {chart_path}")
                    else:
                        pass  # no kline data for chart
                except Exception as e:
                    logging.error(f"❌ Manual alert chart error for {sym}: {repr(e)}")

                # Color emoji matching chart line palette: ⚫🔵🟠🟤🟢🔴
                _color_emojis = ['⚫', '🔵', '🟠', '🟤', '🟢', '🔴']
                triggered_color_idx = alert.get('color_idx', 0)
                color_emoji = _color_emojis[triggered_color_idx % len(_color_emojis)]
                touch_label = "Тело пробило" if touch_type == "body" else "Тень коснулась"
                notify_text = (
                    f"{color_emoji} *КАСАНИЕ РУЧНОЙ ЛИНИИ!*\n\n"
                    f"🪙 `${short_sym}` ({tf})\n"
                    f"💰 Цена: `{touch_price:.8g}`\n"
                    f"📊 Линия: `{line_price:.8g}`\n"
                    f"📐 {touch_label} ({diff_pct:+.2f}%)"
                )

                # Send to admin DM — with robust fallback chain
                target_chat = str(CHAT_ID)
                sent_ok = False
                try:
                    if chart_path:
                        photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                        with open(chart_path, 'rb') as f:
                            data = aiohttp.FormData()
                            data.add_field('chat_id', target_chat)
                            data.add_field('caption', notify_text)
                            data.add_field('parse_mode', 'Markdown')
                            data.add_field('photo', f, filename=f"{sym}_malert.png", content_type='image/png')
                            async with session.post(photo_url, data=data, timeout=30) as resp:
                                if resp.status == 200:
                                    sent_ok = True
                                else:
                                    resp_text = await resp.text()
                                    logging.error(f"❌ Manual alert photo error: {resp.status} - {resp_text}")
                        try:
                            if os.path.exists(chart_path):
                                os.remove(chart_path)
                        except:
                            pass
                except Exception as e:
                    logging.error(f"❌ Manual alert photo send error: {e}")

                # Fallback 1: text with Markdown
                if not sent_ok:
                    try:
                        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                        async with session.post(url, json={
                            "chat_id": target_chat, "text": notify_text, "parse_mode": "Markdown"
                        }, timeout=15) as resp:
                            if resp.status == 200:
                                sent_ok = True
                            else:
                                resp_text = await resp.text()
                                logging.error(f"❌ Manual alert text+MD error: {resp.status} - {resp_text}")
                    except Exception as e:
                        logging.error(f"❌ Manual alert text+MD send error: {e}")

                # Fallback 2: plain text (no Markdown, no parse_mode)
                if not sent_ok:
                    try:
                        plain = notify_text.replace('*', '').replace('`', '')
                        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                        async with session.post(url, json={
                            "chat_id": target_chat, "text": plain
                        }, timeout=15) as resp:
                            if resp.status == 200:
                                sent_ok = True
                            else:
                                resp_text = await resp.text()
                                logging.error(f"❌ Manual alert plain text error: {resp.status} - {resp_text}")
                    except Exception as e:
                        logging.error(f"❌ Manual alert plain send error: {e}")

                if not sent_ok:
                    logging.error(f"❌ FAILED to send manual alert for {sym} ({tf}) — all 3 attempts failed!")

            if triggered:
                save_manual_alerts(remaining)

            await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logging.error(f"❌ Manual alert monitor error: {e}")
            await asyncio.sleep(300)
