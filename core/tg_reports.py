"""
tg_reports.py — Signal report builders (virtual bank, trailing stop checks).
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from config import (load_breakout_log, load_virtual_bank, VIRTUAL_BANK_POSITION_SIZE,
                     TRAILING_STOP_PCT, load_sl_settings)
from core.signal_pipeline import (check_trailing_stop_from_candles, check_fixed_sl_tp_from_candles,
                                   check_fixed_atr_sl_tp, check_candle_trailing, check_ema_sl,
                                   check_btc_shield)
from core.categories import get_sector_emoji


async def _fetch_5m_candles_after(session: aiohttp.ClientSession, symbol: str,
                                   entry_time_iso: str) -> list:
    """Fetch 5m candles from entry time to now (up to 288 = 24h)."""
    try:
        entry_dt = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
        start_ms = int(entry_dt.timestamp() * 1000)
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval=5m&limit=288&startTime={start_ms}")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    except Exception as e:
        logging.error(f"❌ _fetch_5m_candles_after {symbol}: {e}")
        return []


async def _fetch_ema_candles(session: aiohttp.ClientSession, symbol: str,
                             ema_tf: str, limit: int = 100) -> list:
    """Fetch candles for EMA calculation (15m or 5m)."""
    try:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval={ema_tf}&limit={limit}")
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    except Exception as e:
        logging.error(f"❌ _fetch_ema_candles {symbol} {ema_tf}: {e}")
        return []


async def _check_trailing_for_entry(session: aiohttp.ClientSession, symbol: str,
                                     direction: str, entry_price: float,
                                     initial_sl: float, entry_time_iso: str,
                                     trail_pct: float = None,
                                     tp_price: float = None,
                                     atr_value: float = None,
                                     bank_settings: dict = None) -> tuple:
    """
    Fetch 5m candles and check stop based on bank_settings mode.

    Modes:
        stopai   — AI's fixed SL/TP
        trailing — candle-based trailing with ATR buffer
        fixed    — ATR-based fixed SL/TP
        ema      — initial SL + EMA crossover exit

    Returns: (status, close_price, peak_price, sl_used)
    status: 'sl', 'tp', 'trail', 'ema', or 'open'
    """
    if trail_pct is None:
        trail_pct = TRAILING_STOP_PCT

    if not direction or direction == "SKIP" or entry_price <= 0 or not entry_time_iso:
        return ("open", 0, entry_price, initial_sl)

    # Fallback SL
    if not initial_sl or initial_sl <= 0:
        initial_sl = entry_price * (0.90 if direction == "LONG" else 1.10)
    if direction == "LONG" and initial_sl >= entry_price:
        initial_sl = entry_price * 0.90
    elif direction == "SHORT" and initial_sl <= entry_price:
        initial_sl = entry_price * 1.10

    # Fallback ATR
    if not atr_value or atr_value <= 0:
        atr_value = entry_price * 0.03

    candles = await _fetch_5m_candles_after(session, symbol, entry_time_iso)
    if not candles:
        return ("open", 0, entry_price, initial_sl)

    mode = (bank_settings or {}).get("mode", "stopai")

    if mode == "stopai":
        return check_fixed_sl_tp_from_candles(candles, direction, entry_price, initial_sl, tp_price)

    elif mode == "trailing":
        t = (bank_settings or {}).get("trailing", {})
        return check_candle_trailing(
            candles, direction, entry_price, atr_value,
            initial_sl_atr=t.get("initial_sl_atr", 1.5),
            trail_atr=t.get("trail_atr", 1.0),
            anchor=t.get("anchor", "low-1"),
            activation_candles=t.get("activation_candles", 3)
        )

    elif mode == "fixed":
        f = (bank_settings or {}).get("fixed", {})
        return check_fixed_atr_sl_tp(
            candles, direction, entry_price, atr_value,
            sl_atr=f.get("sl_atr", 1.5),
            tp_atr=f.get("tp_atr", 3.0)
        )

    elif mode == "ema":
        e = (bank_settings or {}).get("ema", {})
        ema_tf = e.get("ema_tf", "15m")
        ema_period = e.get("ema_period", 25)
        ema_candles = await _fetch_ema_candles(session, symbol, ema_tf, ema_period + 10)
        return check_ema_sl(
            candles, ema_candles, direction, entry_price, atr_value,
            initial_sl_atr=e.get("initial_sl_atr", 1.5),
            ema_period=ema_period
        )

    else:
        # Fallback: old trailing stop
        return check_trailing_stop_from_candles(candles, direction, entry_price, initial_sl, trail_pct)


async def _batch_check_trailing(session: aiohttp.ClientSession, log: list,
                                 price_map: dict, direction_key: str = "ai_direction",
                                 sl_key: str = "ai_sl",
                                 bank_name: str = "signals") -> dict:
    """
    For all entries in a breakout log, check stop via 5m candles.
    bank_name: 'signals' — selects settings.
    Returns dict: symbol_tf → (status, close_price, peak_price, trailing_sl)
    """
    sem = asyncio.Semaphore(35)
    results = {}
    all_settings = load_sl_settings()
    bank_settings = all_settings.get(bank_name, all_settings.get("signals", {}))

    # BTC Shield check (once for all entries)
    btc_shield_mode = all_settings.get("btc_shield", "off")
    btc_bearish = False
    btc_bullish = False
    if btc_shield_mode == "soft":
        btc_data = await check_btc_shield(session, ema_period=25)
        btc_bearish = btc_data.get("bearish", False)
        btc_bullish = btc_data.get("bullish", False)
        if btc_bearish:
            logging.info(f"🅱️ BTC Shield BEARISH: 15m candle high {btc_data.get('candle_high')} < EMA25 {btc_data.get('ema_value')}")
        if btc_bullish:
            logging.info(f"🅱️ BTC Shield BULLISH: 15m candle low {btc_data.get('candle_low')} > EMA25 {btc_data.get('ema_value')}")

    import time as _time
    _t0 = _time.monotonic()
    _api_count = 0

    async def check_one(entry):
        nonlocal _api_count
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        key = f"{sym}_{tf}"
        direction = entry.get(direction_key, "")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        initial_sl = entry.get(sl_key)
        tp_price = entry.get("ai_tp")
        atr_value = entry.get("atr_value")
        entry_time = entry.get("time", "")

        if not direction or direction == "SKIP" or entry_price <= 0 or not entry_time:
            results[key] = ("open", price_map.get(sym, entry_price), entry_price, 0)
            return

        _api_count += 1
        async with sem:
            status, close_price, peak, trail_sl = await _check_trailing_for_entry(
                session, sym, direction, entry_price, initial_sl, entry_time,
                tp_price=tp_price, atr_value=atr_value, bank_settings=bank_settings
            )

        if status == "open":
            close_price = price_map.get(sym, entry_price)

            # BTC Shield: close losing LONG positions when BTC 15m candle fully below EMA
            if btc_bearish and direction == "LONG":
                pnl_pct = ((close_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                if pnl_pct < -1.0:
                    results[key] = ("btc", close_price, peak, trail_sl)
                    return

            # BTC Shield: close losing SHORT positions when BTC 15m candle fully above EMA
            if btc_bullish and direction == "SHORT":
                pnl_pct = ((entry_price - close_price) / entry_price) * 100 if entry_price > 0 else 0
                if pnl_pct < -1.0:
                    results[key] = ("btc", close_price, peak, trail_sl)
                    return

        results[key] = (status, close_price, peak, trail_sl)

    await asyncio.gather(*(check_one(e) for e in log))
    _elapsed = _time.monotonic() - _t0
    _skipped = len(log) - _api_count
    logging.info(f"📊 _batch_check_trailing [{bank_name}]: {len(log)} entries, {_api_count} API calls, {_skipped} skipped, {_elapsed:.2f}s")
    return results


def _calc_pnl(direction: str, entry_price: float, close_price: float) -> float:
    """Calculate P&L % for a trade."""
    if entry_price <= 0:
        return 0
    if direction == "SHORT":
        return ((entry_price - close_price) / entry_price) * 100
    else:
        return ((close_price - entry_price) / entry_price) * 100


def _build_bank_report(log: list, bank: dict, trailing_results: dict, price_map: dict,
                        lang: str, bank_title: str, direction_key: str = "ai_direction",
                        sl_key: str = "ai_sl", is_close: bool = False,
                        bank_already_updated: bool = False) -> list:
    """
    Universal report builder for both AI signals and ML bank.
    Returns list of message chunks.
    """
    if not log:
        total_w = bank["total_wins"]
        total_l = bank["total_losses"]
        wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) > 0 else 0
        pnl_dollar = bank["balance"] - bank["starting_balance"]
        pnl_pct = (pnl_dollar / bank["starting_balance"]) * 100
        empty_msg = "📭 Сегодня пока нет сигналов." if lang == "ru" else "📭 No signals today yet."
        return [f"🏦 *{bank_title}*\n\n💰 Старт: `${bank['starting_balance']:,.2f}`\n💵 Текущий: `${bank['balance']:,.2f}`\n📊 P&L: `{'+' if pnl_dollar >= 0 else ''}{pnl_dollar:,.2f}$` (`{pnl_pct:+.2f}%`)\n\n✅ {total_w} ({wr:.0f}%) | ❌ {total_l} ({100-wr:.0f}%)\n📈 {bank['total_trades']} сделок\n\n{empty_msg}"]

    day_wins = 0
    day_losses = 0
    day_pending = 0
    day_skipped = 0
    day_pnl_dollar = 0.0
    trade_lines = []

    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        direction = entry.get(direction_key, "")
        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")
        short_sym = sym.replace("USDT", "")
        _sec_emoji = get_sector_emoji(sym)
        short_sym_display = f"{_sec_emoji}{short_sym}"
        dir_tag = f" {direction}" if direction else ""

        # SKIP signals — ⚪
        if direction == "SKIP":
            day_skipped += 1
            now_price = price_map.get(sym, entry_price)
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚪ `{short_sym_display}` {tf} SKIP | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ⏭")
            continue

        # No verdict — ⚫
        if not direction:
            day_skipped += 1
            now_price = price_map.get(sym, entry_price)
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}⚫ `{short_sym_display}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️")
            continue

        # Duplicate symbol — ©️
        if entry.get("is_duplicate", False):
            day_skipped += 1
            now_price = price_map.get(sym, entry_price)
            pct = _calc_pnl(direction, entry_price, now_price)
            price_icon = "🟢" if pct >= 0 else "🔴"
            trade_lines.append(f"{price_icon}©️ `{short_sym_display}` {tf}{dir_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️")
            continue

        # 24h pump filter — 💯
        if entry.get("is_pump_filter", False):
            day_skipped += 1
            now_price = price_map.get(sym, entry_price)
            pct = ((now_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            price_icon = "🟢" if now_price >= entry_price else "🔴"
            trade_lines.append(f"{price_icon}💯 `{short_sym_display}` {tf} | `{entry_price:.6f}` → `{now_price:.6f}` ({pct:+.2f}%) ℹ️")
            continue

        # === Trailing stop check ===
        key = f"{sym}_{tf}"
        status, close_price, peak, trail_sl = trailing_results.get(key, ("open", 0, entry_price, 0))

        if status == "open" and not is_close:
            # Live view: show current price
            now_price = price_map.get(sym, entry_price)
        elif status == "open" and is_close:
            # Close view: close at current market
            now_price = price_map.get(sym, entry_price)
        else:
            # SL or trail hit: use the close_price from trailing check
            now_price = close_price

        pnl_pct = _calc_pnl(direction, entry_price, now_price)
        pnl_pct_leveraged = pnl_pct * ai_leverage

        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE
        pnl_dollar = (pnl_pct_leveraged / 100) * position_size

        # First circle: price went up (🟢) or down (🔴) — always based on price movement
        price_up = now_price >= entry_price
        icon = "🟢" if price_up else "🔴"

        # Second circle: AI direction match
        ai_correct = (direction == "LONG" and price_up) or (direction == "SHORT" and not price_up)
        ai_match = "✅" if ai_correct else "❌"

        if status == "sl":
            day_losses += 1
            status_tag = " 🚫SL"
        elif status == "tp":
            day_wins += 1
            status_tag = " ✅TP"
        elif status == "trail":
            if pnl_pct >= 0:
                day_wins += 1
            else:
                day_losses += 1
            status_tag = " 🔄TRAIL"
        elif status == "ema":
            if pnl_pct >= 0:
                day_wins += 1
            else:
                day_losses += 1
            status_tag = " 📈EMA"
        elif status == "btc":
            day_losses += 1
            status_tag = " 🅱️"
        elif is_close:
            # Close view: everything closed at market
            if pnl_pct >= 0:
                day_wins += 1
            else:
                day_losses += 1
            status_tag = " 📊MKT"
        else:
            # Live view: still open
            day_pending += 1
            status_tag = " ⏳"

        day_pnl_dollar += pnl_dollar
        lev_tag = f" {ai_leverage}x" if ai_leverage > 1 else ""
        trade_lines.append(
            f"{icon}{ai_match} `{short_sym_display}` {tf}{dir_tag}{lev_tag} | `{entry_price:.6f}` → `{now_price:.6f}` ({pnl_pct_leveraged:+.2f}% | {'+' if pnl_dollar >= 0 else ''}{pnl_dollar:.2f}$){status_tag}"
        )

    # Stats
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

    day_closed = day_wins + day_losses
    day_total = day_closed + day_pending
    day_wr = (day_wins / day_closed * 100) if day_closed > 0 else 0
    skip_text = f" | ⏭ Пропуск: {day_skipped}" if day_skipped > 0 else ""

    pending_text = f" | ⏳ Открытых: {day_pending}" if day_pending > 0 and not is_close else ""

    if is_close:
        title_line = f"🔒 *Закрытие позиций {bank_title} (снимок)*\n\n"
    else:
        title_line = ""

    # Count BTC shield closures
    btc_count = sum(1 for key, (st, *_) in trailing_results.items() if st == "btc")
    btc_line = f"\n🅱️ BTC Shield: {btc_count} закрыто" if btc_count > 0 else ""

    header = (
        f"{title_line}"
        f"🏦 *{bank_title}*\n"
        f"💰 Старт: `${bank['starting_balance']:,.2f}` | Текущий: `${projected_balance:,.2f}`\n"
        f"📊 Общий P&L: `{'+' if total_pnl_dollar >= 0 else ''}{total_pnl_dollar:,.2f}$` (`{total_pnl_pct:+.2f}%`)\n\n"
        f"📅 *Сегодня ({day_total} сигналов):*\n"
        f"✅ {day_wins} | ❌ {day_losses} | WR: {day_wr:.0f}%{pending_text}{skip_text}{btc_line}\n"
        f"💵 Дневной P&L: `{'+' if day_pnl_dollar >= 0 else ''}{day_pnl_dollar:.2f}$`\n\n"
        f"📈 *Всего:*\n"
        f"✅ {total_w} ({wr_all:.0f}%) | ❌ {total_l} ({100-wr_all:.0f}%) | 🔢 {total_t} сделок\n"
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
# AI SIGNALS BANK
# ============================

async def build_signals_text(session: aiohttp.ClientSession, lang: str = "ru") -> list:
    """Build AI signals bank — live view."""
    log = load_breakout_log()
    bank = load_virtual_bank()
    price_map = await _fetch_all_prices(session)
    trailing_results = await _batch_check_trailing(session, log, price_map,
                                                    direction_key="ai_direction", sl_key="ai_sl",
                                                    bank_name="signals")
    return _build_bank_report(log, bank, trailing_results, price_map, lang,
                               "Виртуальный банк (AI)", direction_key="ai_direction",
                               sl_key="ai_sl", is_close=False)


async def build_signals_close_text(session: aiohttp.ClientSession, lang: str = "ru",
                                    show_bank: bool = True, bank_already_updated: bool = False) -> list:
    """Build AI signals bank — close snapshot."""
    log = load_breakout_log()
    bank = load_virtual_bank()
    price_map = await _fetch_all_prices(session)
    trailing_results = await _batch_check_trailing(session, log, price_map,
                                                    direction_key="ai_direction", sl_key="ai_sl",
                                                    bank_name="signals")
    return _build_bank_report(log, bank, trailing_results, price_map, lang,
                               "Виртуальный банк (AI)", direction_key="ai_direction",
                               sl_key="ai_sl", is_close=True, bank_already_updated=bank_already_updated)


# ============================
# HELPERS
# ============================

async def _fetch_all_prices(session: aiohttp.ClientSession) -> dict:
    """Fetch all Binance Futures prices in one call."""
    price_map = {}
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=10) as resp:
            if resp.status == 200:
                for p in await resp.json():
                    price_map[p["symbol"]] = float(p["price"])
    except Exception:
        pass
    return price_map


def calc_trailing_pnl_for_daily(log: list, trailing_results: dict, price_map: dict,
                                 bank: dict, direction_key: str = "ai_direction") -> list:
    """
    Calculate P&L tuples for daily bank update.
    Returns: list of (symbol, pnl_pct, pnl_dollar)
    """
    trades_pnl = []
    for entry in log:
        sym = entry["symbol"]
        tf = entry.get("tf", "?")
        direction = entry.get(direction_key, "")
        if not direction or direction == "SKIP":
            continue
        if entry.get("is_duplicate", False) or entry.get("is_pump_filter", False):
            continue

        entry_price = entry.get("ai_entry") or entry.get("current_price", 0) or entry.get("breakout_price", 0)
        ai_leverage = entry.get("ai_leverage", 1) or 1
        ai_deposit_pct = entry.get("ai_deposit_pct")

        key = f"{sym}_{tf}"
        status, close_price, peak, trail_sl = trailing_results.get(key, ("open", 0, entry_price, 0))

        if status == "open":
            close_price = price_map.get(sym, entry_price)

        pnl_pct = _calc_pnl(direction, entry_price, close_price)
        pnl_pct_leveraged = pnl_pct * ai_leverage

        if ai_deposit_pct:
            position_size = bank["balance"] * (ai_deposit_pct / 100)
        else:
            position_size = VIRTUAL_BANK_POSITION_SIZE
        pnl_dollar = (pnl_pct_leveraged / 100) * position_size
        trades_pnl.append((sym, pnl_pct_leveraged, pnl_dollar))

    return trades_pnl
