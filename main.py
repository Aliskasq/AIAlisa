import asyncio
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Import configuration and shared functions
from config import TREND_STATE_FILE, load_alerts, save_alerts, add_breakout_entry, clear_breakout_log, parse_ai_trade_params
from core.binance_api import fetch_klines, get_usdt_futures_symbols, send_status_msg, wait_for_weight
from core.geometry_scanner import find_trend_line
from core.chart_drawer import send_breakout_notification, delete_telegram_message
import aiohttp

# --- AI AND INDICATOR IMPORTS ---
from core.tg_listener import telegram_polling_loop, auto_trend_sender, price_alert_monitor
from core.indicators import calculate_binance_indicators
from agent.analyzer import ask_ai_analysis
from agent.square_publisher import auto_square_poster

from core.tg_listener import SCAN_SCHEDULE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log", mode='a', encoding='utf-8'), logging.StreamHandler()]
)

async def fetch_funding_history(session: aiohttp.ClientSession, symbol: str) -> str:
    """Requests historical funding rate dynamics (last 3 epochs) from Binance."""
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=3"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                # Convert data to a readable string: "0.0100% -> 0.0150% -> 0.0250%"
                rates = [f"{float(item['fundingRate']) * 100:.4f}%" for item in data]
                if rates:
                    return " -> ".join(rates)
    except Exception as e:
        pass
    return "Unknown"

async def log_cleanup_task():
    """Background task: at 23:55 UTC daily, truncate bot.log and remove log files older than 3 days."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 55:
                # Truncate main bot.log (keep last 1000 lines)
                log_file = "bot.log"
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    if len(lines) > 1000:
                        with open(log_file, "w", encoding="utf-8") as f:
                            f.writelines(lines[-1000:])
                        logging.info(f"🧹 Log cleanup: bot.log trimmed from {len(lines)} to 1000 lines")

                # Remove old log files from logs/ directory (older than 3 days)
                logs_dir = "logs"
                if os.path.isdir(logs_dir):
                    cutoff = now.timestamp() - (3 * 86400)
                    for fname in os.listdir(logs_dir):
                        fpath = os.path.join(logs_dir, fname)
                        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            logging.info(f"🧹 Log cleanup: removed old log {fname}")

                await asyncio.sleep(120)  # Skip rest of this minute window
            else:
                await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"❌ Log cleanup error: {e}")
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

                        # --- TRIGGER ALERT TO TELEGRAM ---
                        if trigger_hit and symbol not in processed_symbols:
                            processed_symbols.add(symbol)
                            logging.info(f"🎯 SIGNAL ALERT! {symbol} {tf_key} broke dynamic trigger (Price: {current_price})")
                            line_data = stored_lines.get(tf_key, {}).get(symbol)
                            is_sent = False # Default to not sent

                            if line_data:
                                # 1. Breakout confirmed! Download history for chart + SMC
                                limit_k = 250
                                interval_fetch = '1d' if tf_key == "1D" else '4h'
                                full_raw = await fetch_klines(session, symbol, interval_fetch, limit_k)

                                if full_raw:
                                    full_df = pd.DataFrame(full_raw)

                                    # 2. Calculate SMC (Order Blocks, FVG) and standard indicators
                                    last_indic_row, _ = calculate_binance_indicators(full_df, tf_key)

                                    # 3. Fetch Funding History and add to AI data
                                    funding_history = await fetch_funding_history(session, symbol)
                                    last_indic_row["funding_rate"] = funding_history

                                    # 3b. Multi-timeframe data for better AI accuracy (250 candles for SMC)
                                    mtf_data = {}
                                    smc_data = {}
                                    if tf_key == "1D":
                                        raw_4h = await fetch_klines(session, symbol, "4h", 250)
                                        raw_1h = await fetch_klines(session, symbol, "1h", 250)
                                        raw_15m = await fetch_klines(session, symbol, "15m", 250)
                                        if raw_4h:
                                            mtf_data["4H"] = calculate_binance_indicators(pd.DataFrame(raw_4h), "4H")[0]
                                        if raw_1h:
                                            mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                                        if raw_15m:
                                            mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]
                                    else:
                                        raw_1h = await fetch_klines(session, symbol, "1h", 250)
                                        raw_15m = await fetch_klines(session, symbol, "15m", 250)
                                        if raw_1h:
                                            mtf_data["1H"] = calculate_binance_indicators(pd.DataFrame(raw_1h), "1H")[0]
                                        if raw_15m:
                                            mtf_data["15m"] = calculate_binance_indicators(pd.DataFrame(raw_15m), "15m")[0]

                                    # 3c. SMC analysis (Smart Money Concepts)
                                    try:
                                        from core.smc import analyze_smc
                                        if tf_key == "4H" and full_raw:
                                            smc_data["4H"] = analyze_smc(pd.DataFrame(full_raw), "4H")
                                        if tf_key == "1D" and raw_4h:
                                            smc_data["4H"] = analyze_smc(pd.DataFrame(raw_4h), "4H")
                                        if raw_1h:
                                            smc_data["1H"] = analyze_smc(pd.DataFrame(raw_1h), "1H")
                                        if raw_15m:
                                            smc_data["15m"] = analyze_smc(pd.DataFrame(raw_15m), "15m")
                                    except Exception as e:
                                        logging.error(f"❌ SMC auto-scan error: {e}")

                                    # 4. Request AI verdict with 429 retry logic
                                    ai_verdict = await ask_ai_analysis(symbol, tf_key, last_indic_row, dynamic_line_price, mtf_data=mtf_data, smc_data=smc_data)

                                    # Check for 429 error — send chart with error, then retry AI
                                    ai_got_429 = False
                                    error_msg_id = None
                                    if ai_verdict and ("429" in ai_verdict or "rate limit" in ai_verdict.lower()):
                                        logging.warning(f"⚠️ AI 429 for {symbol}, sending chart with error, will retry...")
                                        # Send chart immediately with 429 error text
                                        _sent_err, error_msg_id = await send_breakout_notification(
                                            symbol, full_df, line_data, tf_key, alert_type, session,
                                            dynamic_trigger, "⏳ AI rate limit (429) — retrying..."
                                        )
                                        await asyncio.sleep(30)
                                        ai_verdict = await ask_ai_analysis(symbol, tf_key, last_indic_row, dynamic_line_price, mtf_data=mtf_data, smc_data=smc_data)
                                        if ai_verdict and ("429" in ai_verdict or "rate limit" in ai_verdict.lower()):
                                            ai_got_429 = True
                                            logging.error(f"❌ AI 429 retry failed for {symbol}, keeping error chart")
                                            ai_verdict = ""
                                        else:
                                            # AI responded! Delete old error message from TG
                                            logging.info(f"✅ AI retry succeeded for {symbol}, replacing error chart")
                                            await delete_telegram_message(session, error_msg_id)
                                            error_msg_id = None

                                    # 5. Send chart (with AI text if available; skip if 429 error chart already sent and retry failed)
                                    if ai_got_429:
                                        # Error chart already in TG, just mark as sent
                                        is_sent = True
                                    else:
                                        is_sent, _ = await send_breakout_notification(
                                            symbol, full_df, line_data, tf_key, alert_type, session,
                                            dynamic_trigger, ai_verdict
                                        )
                                else:
                                    logging.error(f"❌ Failed to download chart for {symbol}. Will retry next cycle.")
                            else:
                                # Line not in memory cache (bug). Delete to avoid infinite loop.
                                is_sent = True

                            if is_sent:
                                # Parse AI direction and trade params from verdict text
                                _ai_dir = ""
                                _ai_params = parse_ai_trade_params(ai_verdict) if ai_verdict else {}
                                if ai_verdict:
                                    # Check for SKIP verdict first
                                    import re as _re
                                    _skip_match = _re.search(r"VERDICT[:\s]*SKIP", ai_verdict, _re.IGNORECASE)
                                    if _skip_match:
                                        logging.info(f"🚫 {symbol} ({tf_key}): AI returned VERDICT: SKIP — not pushing signal")
                                        alerts_to_remove.append(alert)
                                        continue
                                    # Parse direction from VERDICT line specifically, not from TF lines
                                    _verdict_match = _re.search(r"VERDICT[:\s]*(LONG|SHORT)", ai_verdict, _re.IGNORECASE)
                                    if _verdict_match:
                                        _ai_dir = _verdict_match.group(1).upper()
                                    else:
                                        # Fallback: last LONG/SHORT in text (verdict is usually at the end of analysis)
                                        _all_dirs = _re.findall(r"\b(LONG|SHORT)\b", ai_verdict.upper())
                                        if _all_dirs:
                                            _ai_dir = _all_dirs[-1]
                                # ── HARD CAPS: leverage ≤ 3x, deposit ≤ 2% ──
                                _raw_lev = _ai_params.get("ai_leverage") if _ai_params else None
                                _raw_dep = _ai_params.get("ai_deposit_pct") if _ai_params else None
                                _capped_lev = min(_raw_lev, 3) if _raw_lev else None
                                _capped_dep = min(_raw_dep, 2.0) if _raw_dep else 2.0

                                # ── R:R VALIDATION: TP distance must be ≥ SL distance (1:1 min) ──
                                _entry = _ai_params.get("ai_entry") if _ai_params else None
                                _sl = _ai_params.get("ai_sl") if _ai_params else None
                                _tp = _ai_params.get("ai_tp") if _ai_params else None
                                _skip_rr = False
                                if _entry and _sl and _tp and _entry > 0:
                                    _sl_dist = abs(_entry - _sl)
                                    _tp_dist = abs(_tp - _entry)
                                    if _sl_dist > 0 and _tp_dist < _sl_dist:
                                        logging.warning(
                                            f"⚠️ SKIP {symbol} ({tf_key}): bad R:R — "
                                            f"SL dist={_sl_dist:.6f} > TP dist={_tp_dist:.6f} "
                                            f"(ratio {_tp_dist/_sl_dist:.2f}:1, need ≥1:1)"
                                        )
                                        _skip_rr = True

                                if _skip_rr:
                                    logging.info(f"🚫 {symbol} ({tf_key}) NOT logged — failed R:R check")
                                else:
                                    # Log breakout with AI entry/SL/TP (only if AI responded)
                                    add_breakout_entry(symbol, tf_key, dynamic_trigger, current_price, alert_type,
                                                       ai_direction=_ai_dir,
                                                       ai_entry=_ai_params.get("ai_entry") if _ai_params else None,
                                                       ai_sl=_sl,
                                                       ai_tp=_tp,
                                                       ai_leverage=_capped_lev,
                                                       ai_deposit_pct=_capped_dep)
                                # REMOVE FROM QUEUE — delivered or 429 exhausted (even if R:R failed)
                                alerts_to_remove.append(alert)
                            else:
                                logging.warning(f"🔄 Signal {symbol} ({tf_key}) LEFT IN QUEUE. Will retry in 5 minutes.")

                    await wait_for_weight(session, 1800)
                    await asyncio.sleep(3.0)

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
