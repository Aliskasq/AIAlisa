"""
tg_reports.py — Signal report builders (virtual bank, TP/SL checks).
Extracted from tg_listener.py during refactor.
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from config import (load_breakout_log, load_virtual_bank, VIRTUAL_BANK_POSITION_SIZE,
                     load_monitor_breakout_log, load_monitor_virtual_bank)

async def _check_tp_sl_from_candles(session: aiohttp.ClientSession, symbol: str,
                                      ai_dir: str, entry_price: float,
                                      ai_tp: float, ai_sl: float,
                                      entry_time_iso: str) -> tuple:
    """
    Fetch 97 x 15m candles after entry time and walk high/low to find first TP/SL hit.
    Returns (status, close_price) where status is 'tp', 'sl', or 'open'.
    close_price = TP/SL level if hit, or last candle close if open.
    """
    try:
        # Parse entry time → ms for Binance API startTime
        entry_dt = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
        start_ms = int(entry_dt.timestamp() * 1000)

        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval=15m&limit=97&startTime={start_ms}")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                logging.warning(f"⚠️ Candle check {symbol}: HTTP {resp.status}")
                return ("open", 0)
            raw = await resp.json()
            if not raw:
                logging.warning(f"⚠️ Candle check {symbol}: empty response")
                return ("open", 0)

        logging.info(f"🕯️ {symbol}: loaded {len(raw)} candles (15m) from {entry_time_iso}, dir={ai_dir} entry={entry_price} TP={ai_tp} SL={ai_sl}")

        # Walk candles chronologically — first hit wins
        last_close = 0
        for c in raw:
            high = float(c[2])
            low = float(c[3])
            last_close = float(c[4])

            if ai_dir == "LONG":
                # SL: price drops to or below SL
                if ai_sl and ai_sl < entry_price and low <= ai_sl:
                    logging.info(f"🕯️ {symbol}: SL HIT (LONG) — candle low {low} <= SL {ai_sl}")
                    return ("sl", ai_sl)
                if ai_tp and ai_tp > entry_price and high >= ai_tp:
                    logging.info(f"🕯️ {symbol}: TP HIT (LONG) — candle high {high} >= TP {ai_tp}")
                    return ("tp", ai_tp)
            elif ai_dir == "SHORT":
                # SL: price rises to or above SL
                if ai_sl and ai_sl > entry_price and high >= ai_sl:
                    logging.info(f"🕯️ {symbol}: SL HIT (SHORT) — candle high {high} >= SL {ai_sl}")
                    return ("sl", ai_sl)
                if ai_tp and ai_tp < entry_price and low <= ai_tp:
                    logging.info(f"🕯️ {symbol}: TP HIT (SHORT) — candle low {low} <= TP {ai_tp}")
                    return ("tp", ai_tp)

        logging.info(f"🕯️ {symbol}: OPEN (no TP/SL hit in {len(raw)} candles)")
        return ("open", last_close if last_close else 0)
    except Exception as e:
        logging.error(f"❌ _check_tp_sl_from_candles {symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return ("open", 0)


async def _batch_check_tp_sl(session: aiohttp.ClientSession, log: list, price_map: dict) -> dict:
    """
    For all entries in breakout log, check TP/SL via 15m candles.
    Returns dict: symbol+tf → (status, close_price_from_candles).
    Limits concurrency to avoid API weight spikes.
    """
    sem = asyncio.Semaphore(35)  # max 35 concurrent kline requests (~35 weight/sec, limit 2400/min)
    results = {}

    async def check_one(entry):
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        key = f"{sym}_{tf}"
        ai_dir = entry.get("ai_direction", "")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        ai_tp = entry.get("ai_tp")
        ai_sl = entry.get("ai_sl")
        entry_time = entry.get("time", "")

        if not ai_dir or entry_price <= 0 or not entry_time:
            logging.warning(f"⚠️ Candle check SKIP {sym}: ai_dir={ai_dir} entry_price={entry_price} time={entry_time}")
            results[key] = ("open", price_map.get(sym, entry.get("current_price", 0)))
            return

        async with sem:
            status, close_price = await _check_tp_sl_from_candles(
                session, sym, ai_dir, entry_price, ai_tp, ai_sl, entry_time
            )
        # If candle check returned open, use current market price for display
        if status == "open":
            close_price = price_map.get(sym, entry.get("current_price", 0))
        results[key] = (status, close_price)

    logging.info(f"🕯️ Batch candle check: {len(log)} coins to verify TP/SL...")
    await asyncio.gather(*(check_one(e) for e in log))
    tp_count = sum(1 for v in results.values() if v[0] == "tp")
    sl_count = sum(1 for v in results.values() if v[0] == "sl")
    open_count = sum(1 for v in results.values() if v[0] == "open")
    logging.info(f"🕯️ Batch candle check done: TP={tp_count} SL={sl_count} OPEN={open_count}")
    return results


async def build_signals_text(session: aiohttp.ClientSession, lang: str = "ru") -> list:
    """Build virtual bank + all trades text, returns list of message chunks."""
    log = load_breakout_log()
    bank = load_virtual_bank()

    if not log:
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
        pnl_dollar = bank["balance"] - bank["starting_balance"]
        pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
        if lang == "ru":
            return [f"🏦 *Виртуальный банк*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Плюс: {total_w} ({wr:.0f}%) | ❌ Минус: {total_l} ({100-wr:.0f}%)\n📈 Всего: {bank['total_trades']} сделок\n\n📭 Сегодня пока нет сигналов."]
        else:
            return [f"🏦 *Virtual Bank*\n\n💰 Start: `${bank['starting_balance']:,.2f}`\n💵 Current: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Wins: {total_w} ({wr:.0f}%) | ❌ Losses: {total_l} ({100-wr:.0f}%)\n📈 Total: {bank['total_trades']} trades\n\n📭 No signals today yet."]

    # Batch current prices (for display of open trades)
    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                all_prices = await resp.json()
                for p in all_prices:
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass

    # Check TP/SL via 97 x 15m candles after entry (accurate historical check)
    candle_results = await _batch_check_tp_sl(session, log, price_map)

    day_wins = 0
    day_losses = 0
    day_pending = 0
    day_pnl_dollar = 0.0
    trade_lines = []

    day_skipped = 0

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        ai_dir = entry.get("ai_direction", "")

        # Entry = AI entry price (actual price at signal time), fallback to current_price, then breakout_price
        ai_sl = entry.get("ai_sl")
        ai_tp = entry.get("ai_tp")
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)

        short_sym = sym.replace("USDT", "")
        dir_tag = f" {ai_dir}" if ai_dir else ""
        is_monitor = entry.get("is_monitor", False)

        # === Non-bank signals: Ⓜ️ (monitor), ⚪ (skip), ⚫ (no AI) — NOT counted in PnL ===

        # SKIP signals — ⚪
        if ai_dir == "SKIP":
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(
                f"{price_icon}⚪ `{short_sym}` {tf} SKIP | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ⏭"
            )
            continue

        # No AI verdict — ⚫ (show prices + % but don't count in bank)
        if not ai_dir:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(
                f"{price_icon}⚫ `{short_sym}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️"
            )
            continue

        # Monitor signals with LONG/SHORT — Ⓜ️ (verdict + % but don't count in bank)
        if is_monitor:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            if ai_dir == "SHORT":
                pct = -pct
            price_icon = "🟢" if pct >= 0 else "🔴"
            trade_lines.append(
                f"{price_icon}Ⓜ️ `{short_sym}` {tf} {ai_dir} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️"
            )
            continue

        # Use candle-based TP/SL check (97 x 15m candles after entry)
        key = f"{sym}_{tf}"
        candle_status, candle_close = candle_results.get(key, ("open", 0))
        status = candle_status  # 'tp', 'sl', or 'open'

        # now_price = TP/SL level if hit, or current market price if open
        if status == "tp" and ai_tp:
            now_price = ai_tp
        elif status == "sl" and ai_sl:
            now_price = ai_sl
        else:
            now_price = price_map.get(sym, entry.get("current_price", 0))

        # Calculate P&L based on direction and entry
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

        # Apply leverage to P&L percentage
        pnl_pct_leveraged = pnl_pct * ai_leverage

        # Position size: deposit_pct% of bank, or fallback to VIRTUAL_BANK_POSITION_SIZE
        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE

        pnl_dollar = (pnl_pct_leveraged / 100) * position_size

        # Icon: 🟢 price went UP, 🔴 price went DOWN (raw price direction)
        price_up = now_price >= entry_price
        # AI prediction match: ✅ if AI direction matches actual price movement, ❌ if not
        ai_match = ""
        if ai_dir:
            ai_correct = (ai_dir == "LONG" and price_up) or (ai_dir == "SHORT" and not price_up)
            ai_match = "✅" if ai_correct else "❌"

        if status == "tp":
            day_wins += 1
            icon = "🟢" if price_up else "🔴"
            status_tag = " ✅TP"
        elif status == "sl":
            day_losses += 1
            icon = "🟢" if price_up else "🔴"
            status_tag = " 🚫SL"
        else:
            day_pending += 1
            icon = "🟡" if pnl_pct >= 0 else "🟠"
            status_tag = " ⏳"

        day_pnl_dollar += pnl_dollar

        lev_tag = f" {ai_leverage}x" if ai_leverage > 1 else ""
        trade_lines.append(
            f"{icon}{ai_match} `{short_sym}` {tf}{dir_tag}{lev_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct_leveraged:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){status_tag}"
        )

    total_w = bank["total_wins"] + day_wins
    total_l = bank["total_losses"] + day_losses
    total_t = bank["total_trades"] + day_wins + day_losses
    wr_all = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
    projected_balance = bank["balance"] + day_pnl_dollar
    total_pnl_dollar = projected_balance - bank["starting_balance"]
    total_pnl_pct = (total_pnl_dollar / bank["starting_balance"]) * 100

    day_closed = day_wins + day_losses
    day_total = day_closed + day_pending
    day_wr = (day_wins / day_closed * 100) if day_closed > 0 else 0
    skip_text_ru = f" | ⏭ Пропуск: {day_skipped}" if day_skipped > 0 else ""
    skip_text_en = f" | ⏭ Skip: {day_skipped}" if day_skipped > 0 else ""

    if lang == "ru":
        pending_text = f" | ⏳ Открытых: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Виртуальный банк*\n"
            f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
            f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Сегодня ({day_total} сигналов):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}{skip_text_ru}\n"
            f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *Всего за всё время:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n"
            f"{'─' * 30}\n"
        )
    else:
        pending_text = f" | ⏳ Open: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Virtual Bank*\n"
            f"💰 Start: `${bank['starting_balance']:,.2f}` | Current: `${projected_balance:,.2f}`\n"
            f"📊 Total P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Today ({day_total} signals):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}{skip_text_en}\n"
            f"💵 Day P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *All-time:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} trades\n"
            f"{'─' * 30}\n"
        )

    all_msgs = []
    current_chunk = header
    for line in trade_lines:
        if len(current_chunk) + len(line) + 2 > 3900:
            all_msgs.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk:
        all_msgs.append(current_chunk)

    return all_msgs


async def build_signals_close_text(session: aiohttp.ClientSession, lang: str = "ru", show_bank: bool = True, bank_already_updated: bool = False) -> list:
    """Snapshot view: close all pending trades at current price, show day P&L summary.
    bank_already_updated=True means bank balance already includes today's P&L (used in daily report after update_bank_with_trades)."""
    log = load_breakout_log()
    bank = load_virtual_bank()

    if not log:
        if show_bank:
            total_w = bank["total_wins"]
            total_l = bank["total_losses"]
            wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
            pnl_dollar = bank["balance"] - bank["starting_balance"]
            pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
            if lang == "ru":
                return [f"🏦 *Виртуальный банк*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Плюс: {total_w} ({wr:.0f}%) | ❌ Минус: {total_l} ({100-wr:.0f}%)\n📈 Всего: {bank['total_trades']} сделок\n\n📭 Сегодня пока нет сигналов."]
            else:
                return [f"🏦 *Virtual Bank*\n\n💰 Start: `${bank['starting_balance']:,.2f}`\n💵 Current: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Wins: {total_w} ({wr:.0f}%) | ❌ Losses: {total_l} ({100-wr:.0f}%)\n📈 Total: {bank['total_trades']} trades\n\n📭 No signals today yet."]
        if lang == "ru":
            return ["📭 Сегодня пока нет сигналов."]
        else:
            return ["📭 No signals today yet."]

    # Batch current prices
    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                for p in await resp.json():
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass

    # Check TP/SL via 97 x 15m candles after entry (accurate historical check)
    candle_results = await _batch_check_tp_sl(session, log, price_map)

    day_wins = 0
    day_losses = 0
    day_skipped = 0
    day_pnl_dollar = 0.0
    trade_lines = []

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        ai_dir = entry.get("ai_direction", "")

        # Entry = AI entry price (actual price at signal time), fallback to current_price, then breakout_price
        ai_sl = entry.get("ai_sl")
        ai_tp = entry.get("ai_tp")
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)

        short_sym = sym.replace("USDT", "")
        dir_tag = f" {ai_dir}" if ai_dir else ""
        is_monitor = entry.get("is_monitor", False)

        # === Non-bank signals: Ⓜ️ (monitor), ⚪ (skip), ⚫ (no AI) — NOT counted in PnL ===

        # SKIP signals — ⚪
        if ai_dir == "SKIP":
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(
                f"{price_icon}⚪ `{short_sym}` {tf} SKIP | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ⏭"
            )
            continue

        # No AI verdict — ⚫ (show prices + % but don't count in bank)
        if not ai_dir:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(
                f"{price_icon}⚫ `{short_sym}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️"
            )
            continue

        # Monitor signals with LONG/SHORT — Ⓜ️ (verdict + % but don't count in bank)
        if is_monitor:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            if ai_dir == "SHORT":
                pct = -pct
            price_icon = "🟢" if pct >= 0 else "🔴"
            trade_lines.append(
                f"{price_icon}Ⓜ️ `{short_sym}` {tf} {ai_dir} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️"
            )
            continue

        # Use candle-based TP/SL check (97 x 15m candles after entry)
        key = f"{sym}_{tf}"
        candle_status, candle_close = candle_results.get(key, ("open", 0))
        status = candle_status

        # now_price = TP/SL level if hit, or current market price if open
        if status == "tp" and ai_tp:
            now_price = ai_tp
        elif status == "sl" and ai_sl:
            now_price = ai_sl
        else:
            now_price = price_map.get(sym, entry.get("current_price", 0))

        # P&L — for TP/SL use their prices, for open use current (= "close now")
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

        # Apply leverage
        pnl_pct_leveraged = pnl_pct * ai_leverage

        # Position size
        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE

        pnl_dollar = (pnl_pct_leveraged / 100) * position_size

        # Icon: 🟢 price went UP, 🔴 price went DOWN (raw price direction)
        price_up = now_price >= entry_price
        # AI prediction match: ✅ if AI direction matches actual price movement, ❌ if not
        ai_match = ""
        if ai_dir:
            ai_correct = (ai_dir == "LONG" and price_up) or (ai_dir == "SHORT" and not price_up)
            ai_match = "✅" if ai_correct else "❌"

        # Everything counts as closed — plus = TP, minus = SL
        if pnl_pct >= 0:
            day_wins += 1
            icon = "🟢" if price_up else "🔴"
        else:
            day_losses += 1
            icon = "🟢" if price_up else "🔴"

        day_pnl_dollar += pnl_dollar

        lev_tag = f" {ai_leverage}x" if ai_leverage > 1 else ""
        if status == "tp":
            closed_tag = " ✅TP"
        elif status == "sl":
            closed_tag = " 🚫SL"
        else:
            # No TP/SL hit — closed at market price
            closed_tag = " 📊MKT"
        trade_lines.append(
            f"{icon}{ai_match} `{short_sym}` {tf}{dir_tag}{lev_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct_leveraged:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){closed_tag}"
        )

    day_total = day_wins + day_losses
    day_wr = (day_wins / day_total * 100) if day_total > 0 else 0
    skip_text_ru = f" | ⏭ Пропуск: {day_skipped}" if day_skipped > 0 else ""
    skip_text_en = f" | ⏭ Skip: {day_skipped}" if day_skipped > 0 else ""

    # Virtual bank info for close view
    if bank_already_updated:
        projected_balance = bank["balance"]  # Bank already includes today's P&L
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        total_t = bank["total_trades"]
    else:
        projected_balance = bank["balance"] + day_pnl_dollar  # Preview: add today's P&L
        total_w = bank["total_wins"] + day_wins
        total_l = bank["total_losses"] + day_losses
        total_t = bank["total_trades"] + day_wins + day_losses
    wr_all = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
    total_pnl_dollar = projected_balance - bank["starting_balance"]
    total_pnl_pct = (total_pnl_dollar / bank["starting_balance"]) * 100

    if show_bank:
        if lang == "ru":
            bank_block = (
                f"🏦 *Виртуальный банк*\n"
                f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
                f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n"
                f"📈 Всего: ✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n\n"
            )
        else:
            bank_block = (
                f"🏦 *Virtual Bank*\n"
                f"💰 Start: `${bank['starting_balance']:,.2f}` | Current: `${projected_balance:,.2f}`\n"
                f"📊 Total P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n"
                f"📈 All-time: ✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} trades\n\n"
            )
    else:
        bank_block = ""

    if lang == "ru":
        header = (
            f"🔒 *Закрытие всех позиций (снимок)*\n\n"
            f"{bank_block}"
            f"📅 *Сегодня ({day_total} сделок):*\n"
            f"✅ Плюс: {day_wins} ({day_wr:.0f}%) | ❌ Минус: {day_losses} ({100-day_wr:.0f}%){skip_text_ru}\n"
            f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n"
            f"{'─' * 30}\n"
        )
    else:
        header = (
            f"🔒 *Close all positions (snapshot)*\n\n"
            f"{bank_block}"
            f"📅 *Today ({day_total} trades):*\n"
            f"✅ Wins: {day_wins} ({day_wr:.0f}%) | ❌ Losses: {day_losses} ({100-day_wr:.0f}%){skip_text_en}\n"
            f"💵 Day P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n"
            f"{'─' * 30}\n"
        )

    all_msgs = []
    current_chunk = header
    for line in trade_lines:
        if len(current_chunk) + len(line) + 2 > 3900:
            all_msgs.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk:
        all_msgs.append(current_chunk)

    return all_msgs


# ============================
# MONITOR BANK — /bankm (same logic, separate log/bank)
# ============================
async def build_signals_text_monitor(session: aiohttp.ClientSession, lang: str = "ru") -> list:
    """Build monitor virtual bank + all monitor trades text, returns list of message chunks."""
    log = load_monitor_breakout_log()
    bank = load_monitor_virtual_bank()

    if not log:
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
        pnl_dollar = bank["balance"] - bank["starting_balance"]
        pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
        if lang == "ru":
            return [f"🏦 *Виртуальный банк (Monitor)*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Плюс: {total_w} ({wr:.0f}%) | ❌ Минус: {total_l} ({100-wr:.0f}%)\n📈 Всего: {bank['total_trades']} сделок\n\n📭 Сегодня пока нет сигналов из монитора."]
        else:
            return [f"🏦 *Virtual Bank (Monitor)*\n\n💰 Start: `${bank['starting_balance']:,.2f}`\n💵 Current: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Wins: {total_w} ({wr:.0f}%) | ❌ Losses: {total_l} ({100-wr:.0f}%)\n📈 Total: {bank['total_trades']} trades\n\n📭 No monitor signals today yet."]

    # Batch current prices
    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                all_prices = await resp.json()
                for p in all_prices:
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass

    candle_results = await _batch_check_tp_sl(session, log, price_map)

    day_wins = 0
    day_losses = 0
    day_pending = 0
    day_pnl_dollar = 0.0
    trade_lines = []
    day_skipped = 0

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        ai_dir = entry.get("ai_direction", "")
        ai_sl = entry.get("ai_sl")
        ai_tp = entry.get("ai_tp")
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        short_sym = sym.replace("USDT", "")
        dir_tag = f" {ai_dir}" if ai_dir else ""
        is_monitor = entry.get("is_monitor", False)

        if ai_dir == "SKIP":
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚪ `{short_sym}` {tf} SKIP | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ⏭")
            continue

        if not ai_dir:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚫ `{short_sym}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️")
            continue

        key = f"{sym}_{tf}"
        candle_status, candle_close = candle_results.get(key, ("open", 0))
        status = candle_status

        if status == "tp" and ai_tp:
            now_price = ai_tp
        elif status == "sl" and ai_sl:
            now_price = ai_sl
        else:
            now_price = price_map.get(sym, entry.get("current_price", 0))

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

        pnl_pct_leveraged = pnl_pct * ai_leverage
        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE
        pnl_dollar = (pnl_pct_leveraged / 100) * position_size

        price_up = now_price >= entry_price
        ai_match = ""
        if ai_dir:
            ai_correct = (ai_dir == "LONG" and price_up) or (ai_dir == "SHORT" and not price_up)
            ai_match = "✅" if ai_correct else "❌"

        if status == "tp":
            day_wins += 1
            icon = "🟢" if price_up else "🔴"
            status_tag = " ✅TP"
        elif status == "sl":
            day_losses += 1
            icon = "🟢" if price_up else "🔴"
            status_tag = " 🚫SL"
        else:
            day_pending += 1
            icon = "🟡" if pnl_pct >= 0 else "🟠"
            status_tag = " ⏳"

        day_pnl_dollar += pnl_dollar
        lev_tag = f" {ai_leverage}x" if ai_leverage > 1 else ""
        trade_lines.append(
            f"{icon}{ai_match} `{short_sym}` {tf}{dir_tag}{lev_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct_leveraged:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){status_tag}"
        )

    total_w = bank["total_wins"] + day_wins
    total_l = bank["total_losses"] + day_losses
    total_t = bank["total_trades"] + day_wins + day_losses
    wr_all = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
    projected_balance = bank["balance"] + day_pnl_dollar
    total_pnl_dollar = projected_balance - bank["starting_balance"]
    total_pnl_pct = (total_pnl_dollar / bank["starting_balance"]) * 100

    day_closed = day_wins + day_losses
    day_total = day_closed + day_pending
    day_wr = (day_wins / day_closed * 100) if day_closed > 0 else 0
    skip_text_ru = f" | ⏭ Пропуск: {day_skipped}" if day_skipped > 0 else ""
    skip_text_en = f" | ⏭ Skip: {day_skipped}" if day_skipped > 0 else ""

    if lang == "ru":
        pending_text = f" | ⏳ Открытых: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Виртуальный банк (Monitor)*\n"
            f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
            f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Сегодня ({day_total} сигналов из монитора):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}{skip_text_ru}\n"
            f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *Всего за всё время:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n"
            f"{'─' * 30}\n"
        )
    else:
        pending_text = f" | ⏳ Open: {day_pending}" if day_pending > 0 else ""
        header = (
            f"🏦 *Virtual Bank (Monitor)*\n"
            f"💰 Start: `${bank['starting_balance']:,.2f}` | Current: `${projected_balance:,.2f}`\n"
            f"📊 Total P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
            f"📅 *Today ({day_total} monitor signals):*\n"
            f"✅ TP: {day_wins} | ❌ SL: {day_losses} | WR: {day_wr:.0f}%{pending_text}{skip_text_en}\n"
            f"💵 Day P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
            f"📈 *All-time:*\n"
            f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} trades\n"
            f"{'─' * 30}\n"
        )

    all_msgs = []
    current_chunk = header
    for line in trade_lines:
        if len(current_chunk) + len(line) + 2 > 3900:
            all_msgs.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk:
        all_msgs.append(current_chunk)

    return all_msgs


async def build_signals_close_text_monitor(session: aiohttp.ClientSession, lang: str = "ru", show_bank: bool = True, bank_already_updated: bool = False) -> list:
    """Snapshot view for monitor: close all pending trades at current price, show day P&L summary."""
    log = load_monitor_breakout_log()
    bank = load_monitor_virtual_bank()

    if not log:
        if show_bank:
            total_w = bank["total_wins"]
            total_l = bank["total_losses"]
            wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
            pnl_dollar = bank["balance"] - bank["starting_balance"]
            pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
            if lang == "ru":
                return [f"🏦 *Виртуальный банк (Monitor)*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Плюс: {total_w} ({wr:.0f}%) | ❌ Минус: {total_l} ({100-wr:.0f}%)\n📈 Всего: {bank['total_trades']} сделок\n\n📭 Сегодня пока нет сигналов из монитора."]
            else:
                return [f"🏦 *Virtual Bank (Monitor)*\n\n💰 Start: `${bank['starting_balance']:,.2f}`\n💵 Current: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ Wins: {total_w} ({wr:.0f}%) | ❌ Losses: {total_l} ({100-wr:.0f}%)\n📈 Total: {bank['total_trades']} trades\n\n📭 No monitor signals today yet."]
        if lang == "ru":
            return ["📭 Сегодня пока нет сигналов из монитора."]
        else:
            return ["📭 No monitor signals today yet."]

    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                for p in await resp.json():
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass

    candle_results = await _batch_check_tp_sl(session, log, price_map)

    day_wins = 0
    day_losses = 0
    day_skipped = 0
    day_pnl_dollar = 0.0
    trade_lines = []

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        ai_dir = entry.get("ai_direction", "")
        ai_sl = entry.get("ai_sl")
        ai_tp = entry.get("ai_tp")
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        short_sym = sym.replace("USDT", "")
        dir_tag = f" {ai_dir}" if ai_dir else ""
        is_monitor = entry.get("is_monitor", False)

        if ai_dir == "SKIP":
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚪ `{short_sym}` {tf} SKIP | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ⏭")
            continue

        if not ai_dir:
            day_skipped += 1
            now_price = price_map.get(sym, entry.get("current_price", 0))
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚫ `{short_sym}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️")
            continue

        key = f"{sym}_{tf}"
        candle_status, candle_close = candle_results.get(key, ("open", 0))
        status = candle_status

        if status == "tp" and ai_tp:
            now_price = ai_tp
        elif status == "sl" and ai_sl:
            now_price = ai_sl
        else:
            now_price = price_map.get(sym, entry.get("current_price", 0))

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

        pnl_pct_leveraged = pnl_pct * ai_leverage
        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE
        pnl_dollar = (pnl_pct_leveraged / 100) * position_size

        price_up = now_price >= entry_price
        ai_match = ""
        if ai_dir:
            ai_correct = (ai_dir == "LONG" and price_up) or (ai_dir == "SHORT" and not price_up)
            ai_match = "✅" if ai_correct else "❌"

        if pnl_pct >= 0:
            day_wins += 1
            icon = "🟢" if price_up else "🔴"
        else:
            day_losses += 1
            icon = "🟢" if price_up else "🔴"

        day_pnl_dollar += pnl_dollar
        lev_tag = f" {ai_leverage}x" if ai_leverage > 1 else ""
        if status == "tp":
            closed_tag = " ✅TP"
        elif status == "sl":
            closed_tag = " 🚫SL"
        else:
            closed_tag = " 📊MKT"
        trade_lines.append(
            f"{icon}{ai_match} `{short_sym}` {tf}{dir_tag}{lev_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct_leveraged:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){closed_tag}"
        )

    day_total = day_wins + day_losses
    day_wr = (day_wins / day_total * 100) if day_total > 0 else 0
    skip_text_ru = f" | ⏭ Пропуск: {day_skipped}" if day_skipped > 0 else ""
    skip_text_en = f" | ⏭ Skip: {day_skipped}" if day_skipped > 0 else ""

    if bank_already_updated:
        projected_balance = bank["balance"]
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        total_t = bank["total_trades"]
    else:
        projected_balance = bank["balance"] + day_pnl_dollar
        total_w = bank["total_wins"] + day_wins
        total_l = bank["total_losses"] + day_losses
        total_t = bank["total_trades"] + day_wins + day_losses
    wr_all = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
    total_pnl_dollar = projected_balance - bank["starting_balance"]
    total_pnl_pct = (total_pnl_dollar / bank["starting_balance"]) * 100

    if show_bank:
        if lang == "ru":
            bank_block = (
                f"🏦 *Виртуальный банк (Monitor)*\n"
                f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
                f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n"
                f"📈 Всего: ✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n\n"
            )
        else:
            bank_block = (
                f"🏦 *Virtual Bank (Monitor)*\n"
                f"💰 Start: `${bank['starting_balance']:,.2f}` | Current: `${projected_balance:,.2f}`\n"
                f"📊 Total P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n"
                f"📈 All-time: ✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} trades\n\n"
            )
    else:
        bank_block = ""

    if lang == "ru":
        header = (
            f"🔒 *Закрытие позиций Monitor (снимок)*\n\n"
            f"{bank_block}"
            f"📅 *Сегодня ({day_total} сделок):*\n"
            f"✅ Плюс: {day_wins} ({day_wr:.0f}%) | ❌ Минус: {day_losses} ({100-day_wr:.0f}%){skip_text_ru}\n"
            f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n"
            f"{'─' * 30}\n"
        )
    else:
        header = (
            f"🔒 *Close all Monitor positions (snapshot)*\n\n"
            f"{bank_block}"
            f"📅 *Today ({day_total} trades):*\n"
            f"✅ Wins: {day_wins} ({day_wr:.0f}%) | ❌ Losses: {day_losses} ({100-day_wr:.0f}%){skip_text_en}\n"
            f"💵 Day P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n"
            f"{'─' * 30}\n"
        )

    all_msgs = []
    current_chunk = header
    for line in trade_lines:
        if len(current_chunk) + len(line) + 2 > 3900:
            all_msgs.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    if current_chunk:
        all_msgs.append(current_chunk)

    return all_msgs


