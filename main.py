import asyncio
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Import configuration and shared functions
from config import (TREND_STATE_FILE, load_alerts, save_alerts, add_breakout_entry, clear_breakout_log,
                     parse_ai_trade_params, GROUP_CHAT_ID, MONITOR_GROUP_CHAT_ID,
                     OPENROUTER_API_KEY_MONITOR, OPENROUTER_MODEL_MONITOR)
from core.binance_api import fetch_klines, get_usdt_futures_symbols, send_status_msg, wait_for_weight, fetch_market_positioning, format_positioning_text, fetch_funding_history
from core.geometry_scanner import find_trend_line
from core.chart_drawer import send_breakout_notification, delete_telegram_message
import aiohttp

# --- AI AND INDICATOR IMPORTS ---
from core.tg_listener import telegram_polling_loop, auto_trend_sender, price_alert_monitor, get_chat_lang
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
from agent.square_publisher import auto_square_poster

from core.tg_listener import SCAN_SCHEDULE
from core.signal_pipeline import (
    classify_signal, parse_confidence_from_ai,
    check_volume_filter, get_volume_12h,
    add_monitor, add_volume_monitor, FIXED_LEVERAGE, FIXED_DEPOSIT_PCT,
    get_1d_emergency_warnings
)

# Setup logging
# File handler: only coin/trade-related messages (no sleep, progress, loop noise)
class TradeLogFilter(logging.Filter):
    """Pass only trade/coin-relevant log lines to bot.log."""
    KEYWORDS = (
        "MONITOR", "UPGRADED", "FULL:", "SIGNAL", "ALERT",
        "bank", "P&L", "breakout", "SKIP", "LONG", "SHORT",
        "entry added", "entry failed",
        "❌", "⚠️", "🟢", "🔵", "🔴", "🎯",
        "close", "TP", "SL", "upgrade",
        "EMERGENCY", "cleanup",
    )
    NOISE = (
        "Sleeping", "💤", "Waiting list is empty", "👀 Checking",
        "Analysis progress", "Monitoring loop started",
        "STARTING GLOBAL", "recalculation completed",
        "Data ready", "After volume filter", "Starting AI queue",
        "parallel data fetch", "price_alert_monitor",
        "Monitor recheck loop started", "Monitor: 0 due",
    )
    def filter(self, record):
        msg = record.getMessage()
        for n in self.NOISE:
            if n in msg:
                return False
        for kw in self.KEYWORDS:
            if kw in msg:
                return True
        if record.levelno >= logging.WARNING:
            return True
        return False

file_handler = logging.FileHandler("bot.log", mode='a', encoding='utf-8')
file_handler.addFilter(TradeLogFilter())
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Don't use basicConfig — it's a no-op if any library already configured root logger.
# Attach handlers directly to root logger instead.
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

async def log_cleanup_task():
    """Background task: every 6 hours, remove log lines older than 3 days from bot.log."""
    while True:
        try:
            await asyncio.sleep(6 * 3600)  # every 6 hours

            log_file = "bot.log"
            if not os.path.exists(log_file):
                continue

            cutoff = datetime.now(timezone.utc) - timedelta(days=3)
            kept = []
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Parse timestamp from log line: "2026-04-04 16:00:00,123 - ..."
                    try:
                        ts_str = line[:23]  # "2026-04-04 16:00:00,123"
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
                        if ts >= cutoff:
                            kept.append(line)
                    except (ValueError, IndexError):
                        # Can't parse timestamp — keep the line (continuation / traceback)
                        kept.append(line)

            with open(log_file, "w", encoding="utf-8") as f:
                f.writelines(kept)

            logging.info(f"🧹 Log cleanup: bot.log trimmed to {len(kept)} lines (removed entries older than 3 days)")

        except Exception as e:
            logging.error(f"❌ Log cleanup error: {e}")
            await asyncio.sleep(60)

async def volume_recheck_loop(session):
    """
    Background loop: every 60s checks volume monitors ($1M-$2M coins).
    If volume crosses $2M threshold → re-inject into breakout queue as full signal.
    No AI call here — just volume check (free kline fetch).
    """
    from core.signal_pipeline import (
        get_due_volume_monitors, update_volume_monitor_checked,
        remove_volume_monitor, cleanup_expired_volume_monitors,
        get_volume_12h, check_volume_filter
    )
    from config import SIGNAL_MIN_VOLUME_12H

    logging.info("📊 Volume monitor loop started (recheck $1M-$2M coins every 30min)")

    while True:
        try:
            await asyncio.sleep(60)
            cleanup_expired_volume_monitors()
            due = get_due_volume_monitors()
            if not due:
                continue

            logging.info(f"📊 Volume monitor: {len(due)} due for recheck")

            for m in due:
                try:
                    sym = m["symbol"]
                    new_vol = await get_volume_12h(session, sym)
                    vol_tier = check_volume_filter(new_vol)

                    if vol_tier == "full":
                        # Volume crossed $2M! Remove from volume monitor.
                        # It will be picked up on the next scan cycle naturally.
                        remove_volume_monitor(m["key"])
                        logging.info(
                            f"📊✅ VOL UPGRADED: {sym} vol ${new_vol:,.0f} crossed $2M "
                            f"(was ${m.get('initial_volume_12h', 0):,.0f})"
                        )
                    elif vol_tier == "skip":
                        # Volume dropped below $1M — remove
                        remove_volume_monitor(m["key"])
                        logging.info(f"📊❌ VOL DROPPED: {sym} vol ${new_vol:,.0f} < $1M, removed")
                    else:
                        # Still in $1M-$2M range — update and wait
                        update_volume_monitor_checked(m["key"], new_vol)
                        logging.info(
                            f"📊 VOL STILL LOW: {sym} vol ${new_vol:,.0f} "
                            f"(need ${SIGNAL_MIN_VOLUME_12H:,.0f})"
                        )

                    await asyncio.sleep(1)  # small delay between kline fetches

                except Exception as e:
                    logging.error(f"❌ Volume monitor {m.get('symbol', '?')}: {e}")
                    update_volume_monitor_checked(m["key"], m.get("last_volume_12h", 0))

        except Exception as e:
            logging.error(f"❌ Volume monitor loop error: {e}")
            await asyncio.sleep(60)


async def main():
    # Ensure data directory exists on fresh installs
    os.makedirs("data", exist_ok=True)
    logging.info("--- Monitoring loop started (199 CANDLES MODE | 4H & 1D ONLY) ---")

    # Load mathematical state of lines
    if os.path.exists(TREND_STATE_FILE):
        try:
            with open(TREND_STATE_FILE, 'r') as f:
                state = json.load(f)
                stored_lines = state.get("lines", {"1D": {}, "4H": {}})
        except Exception as e:
            logging.error(f"Error reading state file: {e}")
            stored_lines = {"1D": {}, "4H": {}}
    else:
        stored_lines = {"1D": {}, "4H": {}}

    last_full_calc_date = None

    async with aiohttp.ClientSession() as session:
        session.last_weight = "?"
        
        # Start telegram listener in background
        asyncio.create_task(telegram_polling_loop(session))

        # Start background automatic publisher for Binance Square
        asyncio.create_task(auto_square_poster(session))
        
        # Start daily trend summary sender (23:57 UTC)
        asyncio.create_task(auto_trend_sender(session))
        
        # Start price alert monitor (checks every 30s)
        asyncio.create_task(price_alert_monitor(session))

        # Start log cleanup task (23:55 UTC daily)
        asyncio.create_task(log_cleanup_task())

        # Start monitor recheck loop (re-evaluates MONITOR signals every 30 min)
        from core.signal_pipeline import monitor_recheck_loop
        asyncio.create_task(monitor_recheck_loop(session))

        # Start volume monitor loop (rechecks $1M-$2M coins every 30 min)
        asyncio.create_task(volume_recheck_loop(session))


        while True:
            now_utc = datetime.now(timezone.utc)
            now_msk = now_utc + timedelta(hours=3) # UTC+3 timezone for logging

            # BLOCK 1: GLOBAL RECALCULATION (1D, 4H)
            # =========================================================
            # Dynamic launch time controlled via Telegram /time command
            if last_full_calc_date != now_msk.date() and now_msk.hour == SCAN_SCHEDULE["hour"] and now_msk.minute == SCAN_SCHEDULE["minute"]:
                await asyncio.sleep(2)
                clear_breakout_log()
                logging.info("🚀 STARTING GLOBAL GEOMETRIC ANALYSIS AND DRAWING (1D, 4H)...")
                symbols = await get_usdt_futures_symbols()

                if symbols:
                    chunk_size = 6
                    for i in range(0, len(symbols), chunk_size):
                        chunk = symbols[i:i+chunk_size]
                        tasks = []

                        for s in chunk:
                            tasks.append(fetch_klines(session, s, '1d', 199))
                            tasks.append(fetch_klines(session, s, '4h', 199))

                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        res_idx = 0
                        for s in chunk:
                            d1_raw = results[res_idx] if not isinstance(results[res_idx], Exception) else None
                            d4_raw = results[res_idx+1] if not isinstance(results[res_idx+1], Exception) else None
                            res_idx += 2

                            df_1d = pd.DataFrame(d1_raw) if d1_raw else None
                            df_4h = pd.DataFrame(d4_raw) if d4_raw else None

                            line_1d, _ = await find_trend_line(df_1d, "1D", s) if df_1d is not None else (None, None)
                            line_4h, _ = await find_trend_line(df_4h, "4H", s) if df_4h is not None else (None, None)
                            
                            stored_lines["1D"][s] = line_1d
                            stored_lines["4H"][s] = line_4h
                            
                            # Log coins that already broke through during scan
                            for tf_label, line_res, raw_data in [("1D", line_1d, d1_raw), ("4H", line_4h, d4_raw)]:
                                if line_res and line_res.get("status") == "READY" and raw_data:
                                    cp = float(pd.DataFrame(raw_data)['close'].iloc[-1])
                                    add_breakout_entry(s, tf_label, line_res.get("trigger_price", 0), cp, line_res.get("type", ""))
                            
                        logging.info(f"📊 Analysis progress: {min(i + chunk_size, len(symbols))} / {len(symbols)}")
                        await wait_for_weight(session, 1800)
                        await asyncio.sleep(1.5)

                    last_full_calc_date = now_msk.date()
                    with open(TREND_STATE_FILE, 'w') as f:
                        json.dump({"lines": stored_lines}, f)
                    logging.info("✅ Global recalculation completed successfully!")
                else:
                    logging.error("❌ Symbol list is empty, retrying in 60 seconds...")
                    await asyncio.sleep(60)
                    continue

            # =========================================================
            # BLOCK 2: MONITORING (EVERY 5 MINUTES DYNAMIC TRIGGERS)
            # =========================================================
            alerts = load_alerts()
            alerts_to_remove = []
            processed_symbols = set()  # Prevent duplicate processing of same symbol in one cycle
            breakout_queue = []  # Collect all breakouts, then prep data in parallel, AI in queue

            if alerts:
                logging.info(f"👀 Checking {len(alerts)} coins wating for breakout... Time (UTC+3): {now_msk.strftime('%H:%M:%S')}")
                check_chunk_size = 36
                for i in range(0, len(alerts), check_chunk_size):
                    chunk = alerts[i:i+check_chunk_size]
                    tasks = []
                    
                    for alert in chunk:
                        interval = '1d' if alert['tf'] == "1D" else '4h'
                        tasks.append(fetch_klines(session, alert['symbol'], interval, 2))

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for alert, curr in zip(chunk, results):
                        if isinstance(curr, Exception) or not curr or len(curr) < 2:
                            continue
                            
                        symbol = alert['symbol']
                        tf_key = alert['tf']
                        status = alert['status']
                        alert_type = alert['type']
                        
                        prev_candle = curr[-2]
                        curr_candle = curr[-1]
                        current_price = float(curr_candle['close'])
                        curr_open_time = int(curr_candle['open_time'])
                        
                        base_open_time = alert.get('base_open_time', curr_open_time)

                        # Timeframe milliseconds (1D = 86400000, 4H = 14400000)
                        tf_ms = 86400000 if tf_key == "1D" else 14400000
                        candles_passed = int((curr_open_time - base_open_time) / tf_ms)
                        
                        if candles_passed < 0: candles_passed = 0

                        base_idx_val = alert.get('base_idx', 198)
                        dynamic_idx = base_idx_val + candles_passed
                        slope = alert.get('slope', 0.0)
                        intercept = alert.get('intercept', 0.0)

                        if slope != 0.0:
                            dynamic_line_price = np.exp(slope * dynamic_idx + intercept)
                        else:
                            dynamic_line_price = alert['line_price']

                        trigger_hit = False
                        
                        if alert_type in ["GROWING-CANDLE-MODE", "DROP-ONGOING", "DROP-FLAT-HORIZ"]:
                            dynamic_trigger = alert['trigger_price']
                        else:
                            if status == "WAITING_RED_CLOSE":
                                dynamic_trigger = dynamic_line_price * 1.0001
                            else:
                                dynamic_trigger = dynamic_line_price * 1.02
                                
                        if status == "WAITING_RED_CLOSE":
                            prev_is_red = prev_candle['close'] < prev_candle['open']
                            if prev_is_red:
                                if current_price >= dynamic_trigger:
                                    trigger_hit = True
                            else:
                                fallback_trigger = dynamic_line_price * 1.02
                                if current_price >= fallback_trigger:
                                    trigger_hit = True
                                    
                        elif status == "WAITING_2_PERCENT":
                            if current_price >= dynamic_trigger:
                                trigger_hit = True

                        # --- COLLECT BREAKOUTS FOR PARALLEL PREP ---
                        if trigger_hit and symbol not in processed_symbols:
                            processed_symbols.add(symbol)
                            logging.info(f"🎯 SIGNAL ALERT! {symbol} {tf_key} broke dynamic trigger (Price: {current_price})")
                            line_data = stored_lines.get(tf_key, {}).get(symbol)
                            if line_data:
                                breakout_queue.append({
                                    "symbol": symbol,
                                    "tf_key": tf_key,
                                    "alert_type": alert_type,
                                    "alert": alert,
                                    "line_data": line_data,
                                    "dynamic_trigger": dynamic_trigger,
                                    "current_price": current_price,
                                })
                            else:
                                # Line not in memory cache (bug). Delete to avoid infinite loop.
                                alerts_to_remove.append(alert)

                    await wait_for_weight(session, 1800)
                    await asyncio.sleep(3.0)

                # =========================================================
                # PARALLEL DATA PREP + SEQUENTIAL AI QUEUE
                # =========================================================
                if breakout_queue:
                    logging.info(f"📋 Processing {len(breakout_queue)} breakouts: parallel data fetch → sequential AI queue")

                    async def _prepare_breakout_data(item, http_session):
                        """Fetch all data for a breakout (klines, MTF, funding, positioning, SMC).
                        This runs in parallel for all breakouts — NO AI calls here."""
                        sym = item["symbol"]
                        tf = item["tf_key"]
                        try:
                            interval_fetch = '1d' if tf == "1D" else '4h'
                            full_raw = await fetch_klines(http_session, sym, interval_fetch, 250)
                            if not full_raw:
                                item["error"] = "no_klines"
                                return item

                            full_df = pd.DataFrame(full_raw)
                            last_indic_row, _ = calculate_binance_indicators(full_df, tf)

                            # Parallel fetch: funding + positioning + MTF klines
                            mtf_tasks = [
                                fetch_funding_history(http_session, sym),
                                fetch_market_positioning(http_session, sym),
                                fetch_klines(http_session, sym, "1h", 250),
                                fetch_klines(http_session, sym, "15m", 250),
                            ]
                            if tf == "1D":
                                mtf_tasks.append(fetch_klines(http_session, sym, "4h", 250))

                            mtf_results = await asyncio.gather(*mtf_tasks, return_exceptions=True)

                            funding = mtf_results[0] if not isinstance(mtf_results[0], Exception) else "Unknown"
                            positioning = mtf_results[1] if not isinstance(mtf_results[1], Exception) else {}
                            raw_1h = mtf_results[2] if not isinstance(mtf_results[2], Exception) else None
                            raw_15m = mtf_results[3] if not isinstance(mtf_results[3], Exception) else None
                            raw_4h = mtf_results[4] if len(mtf_results) > 4 and not isinstance(mtf_results[4], Exception) else None

                            last_indic_row["funding_rate"] = funding
                            last_indic_row["positioning"] = positioning

                            mtf_data = {}
                            if raw_4h:
                                mtf_data["4H"] = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")[0]
                            if raw_1h:
                                mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                            if raw_15m:
                                mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]

                            smc_data = {}
                            try:
                                from core.smc import analyze_smc
                                smc_data[tf] = analyze_smc(pd.DataFrame(full_raw), tf)
                                if raw_4h and tf == "1D":
                                    smc_data["4H"] = analyze_smc(pd.DataFrame(raw_4h), "4H")
                                if raw_1h:
                                    smc_data["1H"] = analyze_smc(pd.DataFrame(raw_1h), "1H")
                                if raw_15m:
                                    smc_data["15m"] = analyze_smc(pd.DataFrame(raw_15m), "15m")
                            except Exception as e:
                                logging.error(f"❌ SMC error for {sym}: {e}")

                            item["full_df"] = full_df
                            item["last_indic_row"] = last_indic_row
                            item["mtf_data"] = mtf_data
                            item["smc_data"] = smc_data
                            item["ready"] = True
                        except Exception as e:
                            logging.error(f"❌ Data prep error for {sym}: {e}")
                            item["error"] = str(e)
                        return item

                    # Phase 1: Fetch all data in parallel (no AI, no 429 risk)
                    prep_tasks = [_prepare_breakout_data(item, session) for item in breakout_queue]
                    await asyncio.gather(*prep_tasks)

                    ready_count = sum(1 for item in breakout_queue if item.get("ready"))
                    logging.info(f"✅ Data ready for {ready_count}/{len(breakout_queue)} breakouts.")

                    # Phase 1.5: Volume filter — skip low-volume coins before wasting AI calls
                    filtered_queue = []
                    for item in breakout_queue:
                        if not item.get("ready"):
                            filtered_queue.append(item)  # keep for error handling
                            continue
                        try:
                            volume_12h = await get_volume_12h(session, item["symbol"])
                            vol_tier = check_volume_filter(volume_12h)
                            if vol_tier == "skip":
                                logging.info(f"⏭ {item['symbol']}: 12h vol ${volume_12h:,.0f} < $1M, skip")
                                alerts_to_remove.append(item["alert"])
                                continue
                            elif vol_tier == "monitor":
                                logging.info(f"📊 {item['symbol']}: 12h vol ${volume_12h:,.0f} ($1M-$2M), adding to volume monitor")
                                add_volume_monitor(
                                    item["symbol"], item.get("tf", "4H"), volume_12h,
                                    item.get("price", 0), item.get("breakout_pct", 0)
                                )
                                alerts_to_remove.append(item["alert"])
                                continue
                            item["volume_12h"] = volume_12h
                        except Exception as e:
                            logging.warning(f"⚠️ Volume check failed for {item['symbol']}: {e}, allowing through")
                        filtered_queue.append(item)
                    breakout_queue = filtered_queue
                    logging.info(f"📋 After volume filter: {len([i for i in breakout_queue if i.get('ready')])} breakouts. Starting AI queue...")

                    # Phase 2: Sequential AI calls (through lock + cooldown = no 429)
                    def _is_ai_error(text):
                        """Check if AI response is an error, not real analysis."""
                        if not text:
                            return True
                        if text.startswith("❌"):
                            return True
                        error_markers = ["429", "rate limit", "network error", "api error", "timeout"]
                        return any(m in text.lower() for m in error_markers)

                    for idx, item in enumerate(breakout_queue):
                        if not item.get("ready"):
                            if item.get("error") == "no_klines":
                                logging.error(f"❌ No klines for {item['symbol']}. Will retry next cycle.")
                            continue

                        sym = item["symbol"]
                        tf = item["tf_key"]
                        alert_type = item["alert_type"]
                        dynamic_trigger = item["dynamic_trigger"]
                        full_df = item["full_df"]
                        line_data = item["line_data"]
                        last_indic_row = item["last_indic_row"]

                        logging.info(f"🤖 AI queue [{idx+1}/{len(breakout_queue)}]: {sym} {tf}")

                        # For 1D breakouts: use 4H as primary indicators (not 1D)
                        # 1D geometry detects trendline break, but trade decision = 4H+1H
                        ai_indic = last_indic_row
                        ai_tf = tf
                        if tf == "1D" and "4H" in item.get("mtf_data", {}):
                            ai_indic = item["mtf_data"]["4H"]
                            ai_tf = "4H"
                            logging.info(f"📊 1D breakout {sym}: using 4H indicators for AI (1D = geometry only)")

                        # AI call — mode="scan" for full multi-TF analysis
                        ai_verdict_full = await ask_ai_analysis(
                            sym, ai_tf, ai_indic, item.get("dynamic_trigger"),
                            mode="scan", lang=get_chat_lang(GROUP_CHAT_ID), mtf_data=item["mtf_data"], smc_data=item["smc_data"]
                        )

                        # Extract Part 1 only (before ---) for chart caption
                        if ai_verdict_full and "---" in ai_verdict_full:
                            ai_verdict = ai_verdict_full.split("---", 1)[0].strip()
                        else:
                            ai_verdict = ai_verdict_full

                        # Check for AI errors — retry once
                        ai_has_error = False
                        error_msg_id = None

                        if ai_verdict and _is_ai_error(ai_verdict):
                            error_type = "429/rate limit" if ("429" in ai_verdict or "rate limit" in ai_verdict.lower()) else "network/API error"
                            logging.warning(f"⚠️ AI {error_type} for {sym}, sending chart with placeholder, will retry...")
                            _sent_err, error_msg_id = await send_breakout_notification(
                                sym, full_df, line_data, tf, alert_type, session,
                                dynamic_trigger, f"⏳ AI {error_type} — retrying..."
                            )
                            await asyncio.sleep(15)
                            ai_verdict_full = await ask_ai_analysis(
                                sym, tf, last_indic_row, dynamic_trigger,
                                mode="scan", lang=get_chat_lang(GROUP_CHAT_ID), mtf_data=item["mtf_data"], smc_data=item["smc_data"]
                            )
                            if ai_verdict_full and "---" in ai_verdict_full:
                                ai_verdict = ai_verdict_full.split("---", 1)[0].strip()
                            else:
                                ai_verdict = ai_verdict_full
                            if ai_verdict and _is_ai_error(ai_verdict):
                                ai_has_error = True
                                logging.error(f"❌ AI retry failed for {sym} ({error_type}), keeping error chart")
                                ai_verdict = ""
                            else:
                                logging.info(f"✅ AI retry succeeded for {sym}, will replace error chart after new one sent")
                                # Don't delete error_msg_id here — delete AFTER new message is sent successfully

                        # === SIGNAL PIPELINE CLASSIFICATION ===
                        _ai_dir = ""
                        _ai_params = parse_ai_trade_params(ai_verdict) if ai_verdict else {}
                        if ai_verdict and not ai_has_error:
                            import re as _re
                            # Parse verdict: English (LONG/SHORT/SKIP) or Russian (ЛОНГ/ШОРТ/ПРОПУСК)
                            _verdict_match = _re.search(r"(?:VERDICT|ВЕРДИКТ)[:\s]*(LONG|SHORT|SKIP|ЛОНГ|ШОРТ|ПРОПУСК)", ai_verdict, _re.IGNORECASE)
                            if _verdict_match:
                                _raw_dir = _verdict_match.group(1).upper()
                                _ai_dir = {"ЛОНГ": "LONG", "ШОРТ": "SHORT", "ПРОПУСК": "SKIP"}.get(_raw_dir, _raw_dir)
                            else:
                                _all_dirs = _re.findall(r"\b(LONG|SHORT|ЛОНГ|ШОРТ)\b", ai_verdict.upper())
                                if _all_dirs:
                                    _raw = _all_dirs[-1]
                                    _ai_dir = {"ЛОНГ": "LONG", "ШОРТ": "SHORT"}.get(_raw, _raw)

                        # Parse confidence from AI and classify signal tier
                        long_pct, short_pct = parse_confidence_from_ai(ai_verdict or "")
                        adx_value = last_indic_row.get("adx", 0)
                        adx_trend = last_indic_row.get("adx_trend", "stable")
                        adx_avg_50 = last_indic_row.get("adx_avg_50", 0)
                        tier = classify_signal(long_pct, short_pct, adx_value,
                                              adx_trend=adx_trend, adx_avg_50=adx_avg_50,
                                              mtf_data=item.get("mtf_data"))

                        if tier == "monitor" and not ai_has_error:
                            # 🔵 MONITOR — add to monitor queue + send full signal to monitor group
                            reason = "flat_market" if adx_value < 20 else "low_confidence"
                            direction = "LONG" if long_pct > short_pct else "SHORT"
                            add_monitor(sym, tf, direction, long_pct, short_pct,
                                       item["current_price"], reason)
                            logging.info(f"🔵 MONITOR: {sym} ({reason}, conf {max(long_pct,short_pct)}%, ADX {adx_value:.0f})")

                            # Send full breakout with SKIP verdict to MAIN group (all breakouts go to main)
                            try:
                                _skip_caption = ai_verdict or ""
                                if not _skip_caption:
                                    # Build fallback with per-TF scorecard data
                                    _fb_lines = []
                                    _mtf = item.get("mtf_data", {})
                                    for _tf_name in ["4H", "1H", "15m"]:
                                        _tf_ind = _mtf.get(_tf_name, {})
                                        if not _tf_ind:
                                            if _tf_name == tf:
                                                _tf_ind = last_indic_row
                                            else:
                                                continue
                                        _rsi = _tf_ind.get("rsi14", 0)
                                        _adx_tf = _tf_ind.get("adx", 0)
                                        _macd_h = _tf_ind.get("macd_hist", 0)
                                        _ema7 = _tf_ind.get("ema7", 0)
                                        _ema25 = _tf_ind.get("ema25", 0)
                                        _ema99 = _tf_ind.get("ema99", 0)
                                        # Count bull/bear signals
                                        _b, _r = 0, 0
                                        if _ema7 > _ema25 > _ema99: _b += 1
                                        elif _ema7 < _ema25 < _ema99: _r += 1
                                        if _macd_h > 0: _b += 1
                                        else: _r += 1
                                        if _rsi > 55: _b += 1
                                        elif _rsi < 45: _r += 1
                                        if _adx_tf > 25:
                                            if _tf_ind.get("di_plus", 0) > _tf_ind.get("di_minus", 0): _b += 1
                                            else: _r += 1
                                        _st = _tf_ind.get("supertrend", "")
                                        if "Bullish" in str(_st): _b += 1
                                        elif "Bearish" in str(_st): _r += 1
                                        _total = _b + _r if (_b + _r) > 0 else 1
                                        _lpct = round(_b / _total * 100)
                                        _fb_lines.append(f"⏱ {_tf_name}: LONG {_lpct}% / SHORT {100-_lpct}% (RSI {_rsi:.0f}, ADX {_adx_tf:.0f})")
                                    _fb_tf_text = "\n".join(_fb_lines)
                                    _reason_ru = "флэт" if reason == "flat_market" else "низкая уверенность"
                                    _skip_caption = (
                                        f"🏆 ВЕРДИКТ: SKIP ({direction})\n"
                                        f"📊 Общий: LONG {long_pct:.0f}% / SHORT {short_pct:.0f}%\n\n"
                                        f"{_fb_tf_text}\n\n"
                                        f"⚠️ {_reason_ru} | ADX: {adx_value:.0f}"
                                    )
                                _skip_caption = f"🔵 MONITOR (SKIP)\n{_skip_caption}"
                                _mon_sent, _ = await send_breakout_notification(
                                    sym, item["full_df"], item["line_data"], tf, alert_type, session,
                                    dynamic_trigger, _skip_caption
                                )
                                if _mon_sent:
                                    # Delete old error chart after new monitor signal sent
                                    if error_msg_id:
                                        await delete_telegram_message(session, error_msg_id)
                                        error_msg_id = None
                                    logging.info(f"📤 MONITOR signal sent to main group: {sym}")
                            except Exception as _me:
                                logging.error(f"❌ Monitor send error for {sym}: {_me}")

                            # Add to breakout log as info-only (no P&L impact)
                            # Save with ACTUAL AI verdict:
                            # - "SKIP" if AI said SKIP → ⚪ white dot, no P&L
                            # - "LONG"/"SHORT" if AI gave direction but low confidence → 🟢/🔴 dot, tracked in P&L
                            # - "" if AI didn't respond → ⚫ black dot
                            _monitor_save_dir = _ai_dir  # actual AI verdict (LONG/SHORT/SKIP/"")
                            _monitor_entry = _ai_params.get("ai_entry") if (_ai_params and _ai_dir in ("LONG", "SHORT")) else None
                            _monitor_sl = _ai_params.get("ai_sl") if (_ai_params and _ai_dir in ("LONG", "SHORT")) else None
                            _monitor_tp = _ai_params.get("ai_tp") if (_ai_params and _ai_dir in ("LONG", "SHORT")) else None
                            add_breakout_entry(sym, tf, dynamic_trigger, item["current_price"], alert_type,
                                               ai_direction=_monitor_save_dir,
                                               ai_entry=_monitor_entry,
                                               ai_sl=_monitor_sl,
                                               ai_tp=_monitor_tp,
                                               ai_leverage=1, ai_deposit_pct=2.0)
                            alerts_to_remove.append(item["alert"])
                            continue

                        # 🟢 FULL SIGNAL — fetch 1D emergency warnings before sending
                        emergency_warnings = []
                        try:
                            raw_1d = await fetch_klines(session, sym, "1d", 250)
                            if raw_1d:
                                from core.indicators import calculate_binance_indicators as _calc_indic
                                indic_1d, _ = _calc_indic(pd.DataFrame(raw_1d), "1D")
                                emergency_warnings = get_1d_emergency_warnings(indic_1d)
                                if emergency_warnings:
                                    logging.warning(f"⚠️ 1D EMERGENCY for {sym}: {emergency_warnings}")
                                    # Append warnings to AI verdict caption
                                    warn_text = "\n".join(emergency_warnings)
                                    if ai_verdict:
                                        ai_verdict = ai_verdict + f"\n\n📊 1D MACRO:\n{warn_text}"
                                    
                                    # For 1D breakout: use 4H indicators as primary (not 1D)
                                    # 1D geometry detects the line, but trade decision uses 4H+1H
                                    if tf == "1D" and "4H" not in item.get("mtf_data", {}):
                                        try:
                                            raw_4h_for_1d = await fetch_klines(session, sym, "4h", 250)
                                            if raw_4h_for_1d:
                                                item["mtf_data"]["4H"] = _calc_indic(pd.DataFrame(raw_4h_for_1d), "4H")[0]
                                        except Exception:
                                            pass
                        except Exception as e:
                            logging.error(f"❌ 1D emergency check for {sym}: {e}")

                        is_sent = False
                        if ai_has_error:
                            is_sent = True
                        else:
                            is_sent, _ = await send_breakout_notification(
                                sym, full_df, line_data, tf, alert_type, session,
                                dynamic_trigger, ai_verdict or ""
                            )
                            # Delete old error chart ONLY after new one sent successfully
                            if is_sent and error_msg_id:
                                await delete_telegram_message(session, error_msg_id)
                                error_msg_id = None
                            logging.info(f"🟢 FULL: {sym} {_ai_dir} (conf {max(long_pct,short_pct)}% ADX {adx_value:.0f})")

                        if is_sent:
                            add_breakout_entry(sym, tf, dynamic_trigger, item["current_price"], alert_type,
                                               ai_direction=_ai_dir,
                                               ai_entry=_ai_params.get("ai_entry") if _ai_params else None,
                                               ai_sl=_ai_params.get("ai_sl") if _ai_params else None,
                                               ai_tp=_ai_params.get("ai_tp") if _ai_params else None,
                                               ai_leverage=FIXED_LEVERAGE,       # ALWAYS 1x
                                               ai_deposit_pct=FIXED_DEPOSIT_PCT) # ALWAYS 2%
                            alerts_to_remove.append(item["alert"])
                        else:
                            logging.warning(f"🔄 Signal {sym} ({tf}) LEFT IN QUEUE. Will retry in 5 minutes.")

                # Clean up processed alerts
                if alerts_to_remove:
                    alerts = [a for a in alerts if a not in alerts_to_remove]
                    save_alerts(alerts)
            else:
                logging.info(f"👀 Waiting list is empty. Time (UTC+3): {now_msk.strftime('%H:%M:%S')}")

            # =========================================================
            # BLOCK 3: SMART TIMER (Sleep until exact time XX:05:02, XX:10:02)
            # =========================================================
            now = datetime.now(timezone.utc)
            seconds_since_hour = now.minute * 60 + now.second

            next_trigger_seconds = ((seconds_since_hour // 300) + 1) * 300 + 2
            sleep_time = next_trigger_seconds - seconds_since_hour - (now.microsecond / 1000000.0)

            if sleep_time <= 0:
                sleep_time += 300

            next_run_msk = (now + timedelta(seconds=sleep_time) + timedelta(hours=3)).strftime('%H:%M:%S')
            logging.info(f"💤 Sleeping for {sleep_time:.2f} sec. Next check exactly at {next_run_msk} (UTC+3)")
            await asyncio.sleep(sleep_time)

# --- 🏁 ENTRY POINT ---
if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(send_status_msg("🚀 **Bot successfully initialized!**\nStarting geometric non-linear analysis for Binance Futures..."))
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logging.info("Shutting down bot gracefully...")
        loop.run_until_complete(send_status_msg("🛑 **Bot was stopped manually.**"))
    except Exception as e:
        logging.exception("Fatal crash:")
        loop.run_until_complete(send_status_msg(f"💥 **Bot crashed!**\n`{str(e)[:100]}`"))
    finally:
        loop.close()
