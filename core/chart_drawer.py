import asyncio
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import gc
import json
import numpy as np
import aiohttp
import pandas as pd
import logging
import uuid
from config import BOT_TOKEN, GROUP_CHAT_ID, BOTTOM_GROUP_CHAT_ID

SQUARE_CACHE_FILE = "data/square_cache.json"

async def send_breakout_notification(symbol, df, line, tf, line_type, session, trigger_price=0.0, ai_text="", target_chat_id=None, smc_overlay=None, from_vol_wait=False):
    # FIX: Sync view limit perfectly with scanner (199)
    view_limit = min(len(df), 199)
    custom_style = 'charles'

    plot_df = df.iloc[-view_limit:].copy().reset_index(drop=True)
    plot_df['ds'] = pd.to_datetime(plot_df['open_time'], unit='ms')
    plot_df.set_index('ds', inplace=True)

    # --- PERFECT TIME SYNC LOGIC FOR ALL HISTORIES ---
    tf_ms = 14400000 if tf == "4H" else 86400000
    curr_last_time = int(plot_df['open_time'].iloc[-1])
    base_open_time = line.get('base_open_time', curr_last_time)

    shift = round((curr_last_time - base_open_time) / tf_ms)
    if shift < 0: shift = 0

    # Dynamic fallback for base_idx
    base_idx = line.get('base_idx', view_limit - 1)
    last_math_x = base_idx + shift

    # Calculate accurate offset connecting local plot to global math
    offset = last_math_x - (view_limit - 1)

    line_vals = []
    all_peaks_series = [float('nan')] * view_limit
    point_a_series = [float('nan')] * view_limit
    point_b_series = [float('nan')] * view_limit

    idx_a_view = -1
    idx_b_view = -1
    all_peaks_set = set(line.get('all_peaks', []))

    # 🛡️ ПАТЧ: Предохранители для Matplotlib LogLocator (чтобы не было RecursionError)
    min_chart_price = plot_df['low'].min() * 0.001   # Режем невидимый низ линии
    max_chart_price = plot_df['high'].max() * 1000.0 # Режем невидимый верх линии

    for i in range(view_limit):
        math_x = i + offset
        val_log = line['slope'] * math_x + line['intercept']
        val = np.exp(val_log)
        
        # Защита от бесконечного цикла масштабной сетки!
        val_safe = max(min(val, max_chart_price), min_chart_price)
        line_vals.append(val_safe)

        if math_x in all_peaks_set:
            all_peaks_series[i] = plot_df['high'].iloc[i]

        if math_x == line['index_A']:
            point_a_series[i] = line['price_A']
            idx_a_view = i

        if math_x == line['index_B']:
            point_b_series[i] = line['price_B']
            idx_b_view = i

    addplots = []
    # Безопасная проверка на наличие точек
    if not all(np.isnan(x) for x in all_peaks_series):
        addplots.append(mpf.make_addplot(all_peaks_series, type='scatter', markersize=50, marker='o', color='lime', alpha=0.8))
    if idx_a_view != -1:
        addplots.append(mpf.make_addplot(point_a_series, type='scatter', markersize=120, marker='o', color='blue'))
    if idx_b_view != -1:
        addplots.append(mpf.make_addplot(point_b_series, type='scatter', markersize=120, marker='o', color='red'))

    file_path = f"break_{symbol}_{tf}.png"
    fig = None

    try:
        # Create figure with 4 panels: main chart (6) + OBV (1) + RSI (1) + MACD (1)
        _dummy_panels = [
            mpf.make_addplot([float('nan')] * view_limit, panel=1, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=2, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=3, color='gray'),
        ]

        fig, axlist = mpf.plot(
            plot_df, type='candle', style=custom_style,
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots + _dummy_panels, yscale='log',
            title=f"\n{symbol} {line_type} (LOG-MODE)",
            figsize=(14, 13), returnfig=True, tight_layout=True,
            panel_ratios=(6, 1, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        if idx_a_view != -1:
            ax.text(idx_a_view, line['price_A'], f"{line['price_A']:.4f}", color='blue', fontsize=11, fontweight='bold', ha='center', va='bottom')
        if idx_b_view != -1:
            ax.text(idx_b_view, line['price_B'], f"{line['price_B']:.4f}", color='red', fontsize=11, fontweight='bold', ha='center', va='bottom')

        ax.text(0.5, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='center', va='bottom', alpha=0.9)

        # SMC overlay on breakout chart
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, offset)
            except Exception as e:
                logging.error(f"❌ SMC overlay error (breakout): {repr(e)}")

        # Custom grid + date labels + right margin SMC
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_right_margin_smc(ax, smc_overlay, view_limit, plot_df)

        # Indicator panels (OBV, RSI, MACD)
        try:
            _add_indicator_panels(fig, axlist, plot_df, view_limit)
        except Exception as e:
            logging.error(f"❌ Indicator panels error: {repr(e)}")

        # Date labels on bottom panel
        _apply_date_labels_bottom(axlist[6], plot_df, view_limit)

        fig.savefig(file_path, dpi=120, bbox_inches='tight')

    except Exception as e:
        logging.error(f"❌ Error generating chart {symbol}: {repr(e)}")
    finally:
        if fig:
            fig.clf()
        plt.close('all')
        gc.collect()

    if not os.path.exists(file_path):
        return False

    # FIX: Negative percentage protection
    current_price = plot_df['close'].iloc[-1]
    target_price_now = np.exp(line['slope'] * last_math_x + line['intercept'])

    if current_price <= target_price_now:
        current_price = plot_df['high'].iloc[-1]
        if current_price <= target_price_now:
            current_price = target_price_now * 1.0005 # Force display slightly above

    diff_pct = ((current_price / target_price_now) - 1) * 100

    # --- PREPARE BINANCE SQUARE PUBLICATION ---
    post_id = str(uuid.uuid4())[:8]
    square_text = f"🚀 ${symbol} Technical Breakout Alert!\n\nTimeframe: {tf}\nCurrent Price: ${current_price:.4f}\n\n{ai_text}"
    # Save to file-based cache (no circular import needed)
    try:
        cache = {}
        if os.path.exists(SQUARE_CACHE_FILE):
            with open(SQUARE_CACHE_FILE, 'r') as f:
                cache = json.load(f)
        cache[post_id] = square_text
        with open(SQUARE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"❌ Failed to save square cache: {e}")

    # --- TELEGRAM MESSAGE DESIGN ---
    short_symbol = symbol.replace('USDT', '')
    
    # AI text — no hard truncation here; the split logic below handles caption limits
    ai_text = ai_text or ""
    safe_ai_text = ai_text
    # Sanitize markdown: escape unpaired * and _ to prevent Telegram parse errors
    import re as _re
    def _sanitize_tg_markdown(txt):
        # Remove ** (not supported in Telegram Markdown v1)
        txt = txt.replace("**", "")
        # Count * — if odd number, escape the last one
        if txt.count("*") % 2 != 0:
            # Find last * and escape it
            idx = txt.rfind("*")
            txt = txt[:idx] + txt[idx+1:]
        # Same for _
        if txt.count("_") % 2 != 0:
            idx = txt.rfind("_")
            txt = txt[:idx] + txt[idx+1:]
        return txt
    safe_ai_text = _sanitize_tg_markdown(safe_ai_text)
    
    # Build header for photo caption
    header = (
        f"${short_symbol} {'📈VOL UP. TREND BREAKOUT' if from_vol_wait else '🎯 TREND BREAKOUT'}\n"
        f"⏳ TF: {tf} | 💰 Price: {current_price:.6f}\n"
        f"💡 Above trendline by {diff_pct:.2f}%\n\n"
        f"🤖 AI-Alisa-CopilotClow:\n"
    )

    # AI text limit for photo caption (header + 813 ≈ 943, within Telegram's 1024 caption limit)
    AI_TEXT_LIMIT = 813
    overflow_text = ""

    if len(safe_ai_text) <= AI_TEXT_LIMIT:
        # Everything fits in caption
        photo_caption = header + safe_ai_text
    else:
        # Split: AI text that fits in caption, rest as separate message
        cut_text = safe_ai_text[:AI_TEXT_LIMIT]
        # Try to split at last newline for clean break
        last_nl = cut_text.rfind('\n')
        if last_nl > AI_TEXT_LIMIT // 2:
            cut_point = last_nl
        else:
            # No good newline — split at last space to avoid breaking words
            last_space = cut_text.rfind(' ')
            if last_space > AI_TEXT_LIMIT // 2:
                cut_point = last_space
            else:
                cut_point = AI_TEXT_LIMIT
        photo_caption = header + safe_ai_text[:cut_point]
        overflow_text = safe_ai_text[cut_point:].strip()

    app_link = f"https://app.binance.com/en/futures/{symbol.upper()}"
    web_link = f"https://www.binance.com/en/futures/{symbol.upper()}"
    
    reply_markup = {
        "inline_keyboard": [
            [{"text": "📱 Open BINANCE App", "url": app_link}],
            [{"text": f"🖥 Open {symbol} Chart on Web", "url": web_link}],
            [{"text": "📢 Post to Binance Square", "callback_data": f"sq_{post_id}"}]
        ]
    }

    photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    msg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    send_success = False
    sent_message_id = None
    _send_chat_id = str(target_chat_id) if target_chat_id else str(GROUP_CHAT_ID)

    # Send chart with caption (+ buttons if no overflow)
    _photo_reply_markup = reply_markup if not overflow_text else None
    for attempt in range(1, 6):
        try:
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('chat_id', _send_chat_id)
                data.add_field('caption', photo_caption)
                data.add_field('parse_mode', 'Markdown')
                if _photo_reply_markup:
                    data.add_field('reply_markup', json.dumps(_photo_reply_markup))
                data.add_field('photo', f, filename=f"{symbol}.png", content_type='image/png')
                
                async with session.post(photo_url, data=data, timeout=30) as resp:
                    if resp.status == 200:
                        logging.info(f"✅ Signal sent to GROUP ({tf}): {symbol}")
                        send_success = True
                        try:
                            resp_json = await resp.json()
                            sent_message_id = resp_json.get('result', {}).get('message_id')
                        except:
                            pass
                        break
                    elif resp.status == 429:
                        retry_after = 5
                        try:
                            resp_json = await resp.json()
                            retry_after = resp_json.get('parameters', {}).get('retry_after', 5)
                        except: pass
                        logging.warning(f"⚠️ Telegram rate limit. Waiting {retry_after}s for {symbol}...")
                        await asyncio.sleep(retry_after + 1)
                    else:
                        resp_text = await resp.text()
                        logging.error(f"❌ Telegram send error (Attempt {attempt}): {resp.status} - {resp_text}")
                        await asyncio.sleep(3)
        except asyncio.TimeoutError:
            logging.error(f"❌ Timeout sending {symbol} (Attempt {attempt}).")
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"❌ System error sending {symbol} (Attempt {attempt}): {repr(e)}")
            await asyncio.sleep(2)
            
    # Send overflow text as a separate message (with buttons)
    if send_success and overflow_text:
        await asyncio.sleep(0.5)
        for attempt in range(1, 4):
            try:
                payload = {
                    'chat_id': _send_chat_id,
                    'text': overflow_text,
                    'parse_mode': 'Markdown',
                    'reply_markup': json.dumps(reply_markup)
                }
                async with session.post(msg_url, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        logging.info(f"✅ Overflow text sent ({tf}): {symbol}")
                        break
                    elif resp.status == 429:
                        retry_after = 5
                        try:
                            resp_json = await resp.json()
                            retry_after = resp_json.get('parameters', {}).get('retry_after', 5)
                        except: pass
                        await asyncio.sleep(retry_after + 1)
                    else:
                        resp_text = await resp.text()
                        logging.error(f"❌ Overflow text error (Attempt {attempt}): {resp.status} - {resp_text}")
                        if "can't parse entities" in resp_text.lower():
                            payload.pop('parse_mode', None)
                            async with session.post(msg_url, json=payload, timeout=30) as resp2:
                                if resp2.status == 200:
                                    logging.info(f"✅ Overflow text sent plain ({tf}): {symbol}")
                                    break
                        await asyncio.sleep(2)
            except Exception as e:
                logging.error(f"❌ Error sending overflow (Attempt {attempt}): {repr(e)}")
                await asyncio.sleep(2)

    # === BOTTOM FILTER: duplicate to second group if signal is from the bottom ===
    if send_success and BOTTOM_GROUP_CHAT_ID and not target_chat_id:
        try:
            _view = min(len(df), 199)
            _plot = df.iloc[-_view:].copy().reset_index(drop=True)
            _chart_high = float(_plot['high'].max())
            _chart_low = float(_plot['low'].min())
            # Calculate bottom third in LOG scale (matches visual chart rendering)
            import math
            if _chart_low > 0 and _chart_high > _chart_low:
                _log_low = math.log(_chart_low)
                _log_high = math.log(_chart_high)
                _lower_third_ceiling = math.exp(_log_low + (_log_high - _log_low) / 3.0)
            else:
                _range = _chart_high - _chart_low
                _lower_third_ceiling = _chart_low + _range / 3.0
            _current_close = float(_plot['close'].iloc[-1])

            # Point B position relative to end of chart
            _last_idx = _view - 1

            # Calculate index_B in view coordinates (same sync as chart drawing)
            _tf_ms = 14400000 if tf == "4H" else 86400000
            _curr_last_time = int(_plot['open_time'].iloc[-1])
            _base_open_time = line.get('base_open_time', _curr_last_time)
            _shift = round((_curr_last_time - _base_open_time) / _tf_ms)
            if _shift < 0: _shift = 0
            _base_idx = line.get('base_idx', _last_idx)
            _last_math_x = _base_idx + _shift
            _offset = _last_math_x - _last_idx
            _idx_b_math = line.get('index_B', _last_idx)
            _idx_b_view = _idx_b_math - _offset

            _b_not_recent = _idx_b_view <= (_last_idx - 5)
            _price_in_bottom_third = _current_close <= _lower_third_ceiling
            # Lower quarter ceiling in log scale
            if _chart_low > 0 and _chart_high > _chart_low:
                _lower_quarter_ceiling = math.exp(_log_low + (_log_high - _log_low) / 4.0)
            else:
                _lower_quarter_ceiling = _chart_low + (_chart_high - _chart_low) / 4.0
            _price_in_bottom_quarter = _current_close <= _lower_quarter_ceiling

            # Two conditions for bottom group:
            # 1) Price in bottom THIRD + point B not on last 5 candles
            # 2) Price in bottom QUARTER (any B position)
            _pass_filter = (_price_in_bottom_third and _b_not_recent) or _price_in_bottom_quarter

            if _pass_filter:
                _reason = "QUARTER (any B)" if _price_in_bottom_quarter else "THIRD + B not recent"
                logging.info(f"📡 BOTTOM FILTER PASS [{_reason}]: {symbol} ({tf}) — price {_current_close:.6f}, 1/3={_lower_third_ceiling:.6f}, 1/4={_lower_quarter_ceiling:.6f}, B view idx {_idx_b_view}/{_last_idx}")
                # Re-send chart to bottom group
                if os.path.exists(file_path):
                    _bottom_caption = photo_caption
                    for _attempt in range(1, 4):
                        try:
                            with open(file_path, 'rb') as _f:
                                _data = aiohttp.FormData()
                                _data.add_field('chat_id', str(BOTTOM_GROUP_CHAT_ID))
                                _data.add_field('caption', _bottom_caption)
                                _data.add_field('parse_mode', 'Markdown')
                                _data.add_field('photo', _f, filename=f"{symbol}_bottom.png", content_type='image/png')
                                async with session.post(photo_url, data=_data, timeout=30) as _resp:
                                    if _resp.status == 200:
                                        logging.info(f"✅ Bottom signal sent: {symbol} ({tf})")
                                        break
                                    elif _resp.status == 429:
                                        _ra = 5
                                        try:
                                            _rj = await _resp.json()
                                            _ra = _rj.get('parameters', {}).get('retry_after', 5)
                                        except: pass
                                        await asyncio.sleep(_ra + 1)
                                    else:
                                        _rt = await _resp.text()
                                        logging.error(f"❌ Bottom group send error ({_attempt}): {_resp.status} - {_rt}")
                                        await asyncio.sleep(2)
                        except Exception as _e:
                            logging.error(f"❌ Bottom group error ({_attempt}): {repr(_e)}")
                            await asyncio.sleep(2)

                    # Send overflow to bottom group too
                    if overflow_text:
                        await asyncio.sleep(0.5)
                        try:
                            async with session.post(msg_url, json={
                                'chat_id': str(BOTTOM_GROUP_CHAT_ID),
                                'text': overflow_text,
                                'parse_mode': 'Markdown'
                            }, timeout=30) as _resp:
                                if _resp.status == 200:
                                    logging.info(f"✅ Bottom overflow sent: {symbol} ({tf})")
                        except Exception as _e:
                            logging.error(f"❌ Bottom overflow error: {repr(_e)}")
            else:
                logging.info(f"⏭️ BOTTOM FILTER SKIP: {symbol} ({tf}) — third={_price_in_bottom_third} quarter={_price_in_bottom_quarter} b_not_recent={_b_not_recent} (B view idx: {_idx_b_view}/{_last_idx}, price: {_current_close:.6f}, 1/3={_lower_third_ceiling:.6f}, 1/4={_lower_quarter_ceiling:.6f})")
        except Exception as _e:
            logging.error(f"❌ Bottom filter error for {symbol}: {repr(_e)}")

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except: pass

    return send_success, sent_message_id


async def delete_telegram_message(session, message_id):
    """Delete a message from Telegram group chat by message_id."""
    if not message_id:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
    try:
        async with session.post(url, json={
            'chat_id': str(GROUP_CHAT_ID),
            'message_id': message_id
        }, timeout=10) as resp:
            if resp.status == 200:
                logging.info(f"🗑️ Deleted TG message {message_id}")
                return True
            else:
                resp_text = await resp.text()
                logging.error(f"❌ Failed to delete TG message {message_id}: {resp.status} - {resp_text}")
    except Exception as e:
        logging.error(f"❌ Error deleting TG message {message_id}: {repr(e)}")
    return False


def _apply_custom_grid(ax, plot_df, view_limit):
    """
    Custom grid: vertical lines every 20 candles, horizontal every 10% price.
    Date labels horizontal in 2 lines (Apr 16 / 12:00) under each vertical line.
    Removes default matplotlib grid and x/y tick labels.
    """
    import matplotlib.ticker as mticker
    from matplotlib.dates import num2date

    # Remove default grid and tick labels
    ax.grid(False)
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', which='both', labelright=False, labelleft=False, right=False, left=False)

    # --- Vertical lines every 20 candles ---
    for i in range(0, view_limit, 20):
        ax.axvline(x=i, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)

    # Date labels are drawn by _apply_date_labels_bottom on the lowest panel

    # --- Horizontal lines every 10% price ---
    y_low, y_high = ax.get_ylim()
    if y_low > 0 and y_high > 0:
        import math
        # Log scale: find nice 10% steps
        log_low = math.log10(y_low)
        log_high = math.log10(y_high)
        # Step through prices: start from a round base, go up in 10% steps
        # Find starting price (round down to nearest power of 10)
        base = 10 ** math.floor(log_low)
        price = base
        while price < y_low:
            price *= 1.1
        while price <= y_high:
            ax.axhline(y=price, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
            price *= 1.1


def _apply_date_labels_bottom(ax_bottom, plot_df, view_limit):
    """
    Add date labels at the bottom of the figure (below all panels).
    Called on the lowest axis. Labels at every 20-candle mark.
    """
    ax_bottom.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax_bottom.transData, ax_bottom.transAxes)
    for i in range(0, view_limit, 20):
        if i < len(plot_df):
            dt = plot_df.index[i]
            line1 = dt.strftime('%b %d')
            line2 = dt.strftime('%H:%M')
            label = f"{line1}\n{line2}"
            ax_bottom.text(i, -0.3, label,
                    color='#888888', fontsize=6, ha='center', va='top',
                    transform=trans, clip_on=False,
                    zorder=5)


def _draw_right_margin_smc(ax, smc_data, view_limit, plot_df):
    """
    Draw SMC annotations on right margin:
    - Strong High / Weak Low with price
    - Visible OB prices (top & bottom of each block)
    - Arrows for off-screen OBs (max 2 up, max 2 down)
    Replaces the standard Y-axis scale.
    """
    if not smc_data or "error" in str(smc_data.get("summary", "")).lower():
        return

    y_low, y_high = ax.get_ylim()
    trailing = smc_data.get("trailing", {})
    right_x = view_limit + 1  # X position for right margin text

    # --- Strong High / Weak Low ---
    t_high = trailing.get("trailing_high")
    t_low = trailing.get("trailing_low")
    high_label = trailing.get("high_label", "High")
    low_label = trailing.get("low_label", "Low")

    if t_high is not None:
        price_str = f"{t_high:.4f}" if t_high >= 0.01 else f"{t_high:.6f}"
        ax.text(right_x, t_high, f"{high_label}\n{price_str}",
                color='#F23645', fontsize=7, fontweight='bold',
                ha='left', va='center', zorder=5)

    if t_low is not None:
        price_str = f"{t_low:.4f}" if t_low >= 0.01 else f"{t_low:.6f}"
        ax.text(right_x, t_low, f"{low_label}\n{price_str}",
                color='#089981', fontsize=7, fontweight='bold',
                ha='left', va='center', zorder=5)

    # --- Collect all OBs ---
    all_obs = []
    for ob in smc_data.get("swing_order_blocks", []):
        all_obs.append(ob)
    for ob in smc_data.get("internal_order_blocks", []):
        all_obs.append(ob)

    # Classify: visible vs above vs below chart area
    visible_obs = []
    above_obs = []
    below_obs = []

    for ob in all_obs:
        ob_mid = (ob["high"] + ob["low"]) / 2
        if ob["low"] > y_high:
            above_obs.append(ob)
        elif ob["high"] < y_low:
            below_obs.append(ob)
        else:
            visible_obs.append(ob)

    # Sort by price for clean layout
    above_obs.sort(key=lambda o: o["low"])
    below_obs.sort(key=lambda o: -o["high"])

    # --- Visible OB prices on right margin ---
    for ob in visible_obs:
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        # Top price of block
        ax.text(right_x, ob["high"], hi_str,
                color=color, fontsize=6, ha='left', va='bottom', zorder=5)
        # Bottom price of block
        ax.text(right_x, ob["low"], lo_str,
                color=color, fontsize=6, ha='left', va='top', zorder=5)

    # --- Arrows for off-screen OBs (max 2 each direction) ---
    arrow_x = right_x + 0.5

    # Above (↑ arrows near top)
    for i, ob in enumerate(above_obs[:2]):
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        y_pos = y_high * (0.97 - i * 0.06)  # stack from top
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        ax.text(arrow_x, y_pos, f"↑ {lo_str}-{hi_str}",
                color=color, fontsize=6, fontweight='bold',
                ha='left', va='top', zorder=5)

    # Below (↓ arrows near bottom)
    for i, ob in enumerate(below_obs[:2]):
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        y_pos = y_low * (1.03 + i * 0.06)  # stack from bottom
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        ax.text(arrow_x, y_pos, f"↓ {lo_str}-{hi_str}",
                color=color, fontsize=6, fontweight='bold',
                ha='left', va='bottom', zorder=5)


def _add_indicator_panels(fig, axlist, plot_df, view_limit):
    """
    Add OBV, RSI(6,12,24), MACD panels below the main chart.
    Expects fig created with panel_ratios that include space for 3 extra panels.
    mplfinance returns 2 axes per panel (main + secondary/volume).
    axlist[0]=panel0_main, axlist[1]=panel0_vol, axlist[2]=panel1_main, ...
    So: panel1=axlist[2], panel2=axlist[4], panel3=axlist[6]
    """
    from core.indicators import rma

    # Recalculate indicators on chart data
    calc_df = plot_df.copy()
    calc_df = calc_df.reset_index()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in calc_df.columns:
            calc_df[col] = calc_df[col].astype(float)

    close = calc_df['close']
    volume = calc_df['volume']

    # --- OBV ---
    obv_change = np.sign(close.diff()) * volume
    obv = obv_change.fillna(0).cumsum()
    obv_sma20 = obv.rolling(20).mean()

    # --- RSI 6, 12, 24 ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    def _rsi(g, l, period):
        alpha = 1.0 / period
        avg_g = np.zeros(len(g))
        avg_l = np.zeros(len(g))
        avg_g[0] = g.iloc[0]
        avg_l[0] = l.iloc[0]
        for i in range(1, len(g)):
            avg_g[i] = alpha * g.iloc[i] + (1 - alpha) * avg_g[i - 1]
            avg_l[i] = alpha * l.iloc[i] + (1 - alpha) * avg_l[i - 1]
        rs = np.where(avg_l == 0, 100, avg_g / avg_l)
        return 100 - (100 / (1 + rs))

    rsi6 = _rsi(gain, loss, 6)
    rsi12 = _rsi(gain, loss, 12)
    rsi24 = _rsi(gain, loss, 24)

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    x = np.arange(view_limit)

    # --- Panel 1: OBV (green) ---
    # mplfinance: 2 axes per panel → panel1=axlist[2], panel2=axlist[4], panel3=axlist[6]
    ax_obv = axlist[2]
    ax_obv.fill_between(x, obv.values, alpha=0.3, color='#26a69a')
    ax_obv.plot(x, obv.values, color='#26a69a', linewidth=1.0)
    ax_obv.plot(x, obv_sma20.values, color='#ef5350', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_obv.set_xlim(-0.5, view_limit - 0.5)
    ax_obv.set_ylabel('OBV', fontsize=7, color='#26a69a')
    ax_obv.tick_params(axis='both', labelsize=5, colors='#888888')
    ax_obv.grid(False)
    ax_obv.tick_params(axis='x', labelbottom=False)

    # --- Panel 2: RSI 6, 12, 24 (blue tones) ---
    ax_rsi = axlist[4]
    ax_rsi.plot(x, rsi6, color='#2196F3', linewidth=1.0, label='RSI 6')
    ax_rsi.plot(x, rsi12, color='#1565C0', linewidth=0.8, label='RSI 12')
    ax_rsi.plot(x, rsi24, color='#0D47A1', linewidth=0.8, label='RSI 24')
    ax_rsi.axhline(y=70, color='#F23645', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_rsi.axhline(y=30, color='#089981', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_xlim(-0.5, view_limit - 0.5)
    ax_rsi.set_ylabel('RSI', fontsize=7, color='#2196F3')
    ax_rsi.tick_params(axis='both', labelsize=5, colors='#888888')
    ax_rsi.grid(False)
    ax_rsi.tick_params(axis='x', labelbottom=False)

    # --- Panel 3: MACD (yellow/gold) ---
    ax_macd = axlist[6]
    colors_hist = ['#26a69a' if v >= 0 else '#ef5350' for v in macd_hist.values]
    ax_macd.bar(x, macd_hist.values, color=colors_hist, width=0.8, alpha=0.6)
    ax_macd.plot(x, macd_line.values, color='#F0B90B', linewidth=1.0, label='MACD')
    ax_macd.plot(x, macd_signal.values, color='#E040FB', linewidth=0.8, label='Signal')
    ax_macd.axhline(y=0, color='#888888', linewidth=0.3)
    ax_macd.set_xlim(-0.5, view_limit - 0.5)
    ax_macd.set_ylabel('MACD', fontsize=7, color='#F0B90B')
    ax_macd.tick_params(axis='both', labelsize=5, colors='#888888')
    ax_macd.grid(False)
    ax_macd.tick_params(axis='x', labelbottom=False)


def _draw_smc_overlay(ax, plot_df, smc_data, view_limit, global_offset=0):
    """
    Draw Smart Money Concepts overlay on chart — matching LuxAlgo TradingView visuals.
    Elements: Order Blocks, BOS/CHoCH lines, Strong/Weak High/Low.
    """
    from matplotlib.patches import Rectangle

    if not smc_data or "error" in str(smc_data.get("summary", "")).lower():
        return

    # Calculate index offset: SMC uses N candles (e.g. 500), chart shows last view_limit (199).
    # chart_x = smc_global_index - idx_offset
    smc_n = smc_data.get("n", 500)  # exact candle count from analyze_smc()
    idx_offset = smc_n - view_limit
    trailing = smc_data.get("trailing", {})

    # ─── 1. ORDER BLOCKS ────────────────────────────────────────────────
    # Deduplicate: skip internal OBs that overlap >50% with a swing OB
    swing_obs_list = smc_data.get("swing_order_blocks", [])
    internal_obs_list = smc_data.get("internal_order_blocks", [])

    def _obs_overlap(a, b):
        """Return overlap ratio (0-1) of two OBs by price range."""
        overlap_lo = max(a["low"], b["low"])
        overlap_hi = min(a["high"], b["high"])
        if overlap_hi <= overlap_lo:
            return 0.0
        overlap = overlap_hi - overlap_lo
        smaller = min(a["high"] - a["low"], b["high"] - b["low"])
        return overlap / smaller if smaller > 0 else 0.0

    filtered_internal = []
    for iob in internal_obs_list:
        dominated = False
        for sob in swing_obs_list:
            if iob["bias"] == sob["bias"] and _obs_overlap(iob, sob) > 0.5:
                dominated = True
                break
        if not dominated:
            filtered_internal.append(iob)

    # Also deduplicate internal-vs-internal (keep more recent on overlap)
    deduped_internal = []
    for iob in filtered_internal:
        merged = False
        for j in range(len(deduped_internal) - 1, -1, -1):
            existing = deduped_internal[j]
            if iob["bias"] == existing["bias"] and _obs_overlap(iob, existing) > 0.5:
                if iob["index"] > existing["index"]:
                    deduped_internal[j] = iob
                merged = True
                break
        if not merged:
            deduped_internal.append(iob)

    for ob_list, is_internal in [
        (swing_obs_list, False),
        (deduped_internal, True),
    ]:
        for ob in ob_list:
            ob_x_start = ob["index"] - idx_offset
            ob_x_end = view_limit - 1

            if ob_x_end < 0 or ob_x_start >= view_limit:
                continue
            ob_x_start = max(ob_x_start, -0.5)

            ob_low = ob["low"]
            ob_high = ob["high"]

            if ob["bias"] == 1:  # BULLISH
                fc = (0.19, 0.47, 0.96, 0.20) if is_internal else (0.09, 0.28, 0.80, 0.20)
            else:  # BEARISH
                fc = (0.97, 0.49, 0.50, 0.20) if is_internal else (0.70, 0.16, 0.20, 0.20)

            width = ob_x_end - ob_x_start + 0.5
            rect = Rectangle(
                (ob_x_start, ob_low), width, ob_high - ob_low,
                linewidth=0 if is_internal else 0.5,
                edgecolor=fc[:3] + (0.5,) if not is_internal else None,
                facecolor=fc, zorder=1
            )
            ax.add_patch(rect)

    # ─── 2. BOS / CHoCH LINES ──────────────────────────────────────────
    for struct_list, is_internal in [
        (smc_data.get("swing_structures", []), False),
        (smc_data.get("internal_structures", []), True),
    ]:
        for s in struct_list[-15:]:
            pivot_x = s["pivot_index"] - idx_offset
            break_x = s["break_index"] - idx_offset

            if break_x < 0 or pivot_x >= view_limit:
                continue

            price = s["price"]
            is_bullish = s["bias"] == 1
            tag = s["type"]

            color = '#089981' if is_bullish else '#F23645'
            line_style = '--' if is_internal else '-'

            x_start = max(pivot_x, -0.5)
            x_end = min(break_x, view_limit - 0.5)

            ax.hlines(y=price, xmin=x_start, xmax=x_end,
                      colors=color, linestyles=line_style,
                      linewidths=1.0 if is_internal else 1.5,
                      zorder=3, alpha=0.8)

            label_x = (x_start + x_end) / 2
            label_va = 'bottom' if is_bullish else 'top'
            fontsize = 7 if is_internal else 8

            ax.text(label_x, price, tag,
                    color=color, fontsize=fontsize, fontweight='bold',
                    ha='center', va=label_va, zorder=4, alpha=0.9)

    # ─── 3. STRONG/WEAK HIGH/LOW ───────────────────────────────────────
    if trailing:
        t_high = trailing.get("trailing_high")
        t_low = trailing.get("trailing_low")
        high_label = trailing.get("high_label", "High")
        low_label = trailing.get("low_label", "Low")
        t_high_idx = trailing.get("trailing_high_index", 0) - idx_offset
        t_low_idx = trailing.get("trailing_low_index", 0) - idx_offset

        if t_high is not None:
            x_start_h = max(t_high_idx, -0.5)
            h_color = '#F23645'
            ax.hlines(y=t_high, xmin=x_start_h, xmax=view_limit + 2,
                      colors=h_color, linestyles='-', linewidths=1.0,
                      zorder=3, alpha=0.7)
            # Text label moved to _draw_right_margin_smc

        if t_low is not None:
            x_start_l = max(t_low_idx, -0.5)
            l_color = '#089981'
            ax.hlines(y=t_low, xmin=x_start_l, xmax=view_limit + 2,
                      colors=l_color, linestyles='-', linewidths=1.0,
                      zorder=3, alpha=0.7)
            # Text label moved to _draw_right_margin_smc


async def draw_scan_chart(symbol: str, df: pd.DataFrame, line: dict, tf: str, smc_overlay: dict = None) -> str | None:
    """
    Draw a logarithmic chart with trend line for /scan commands.
    Returns the file path to the PNG image, or None on failure.
    Does NOT send to Telegram — caller handles that.
    """
    view_limit = min(len(df), 199)

    plot_df = df.iloc[-view_limit:].copy().reset_index(drop=True)
    plot_df['ds'] = pd.to_datetime(plot_df['open_time'], unit='ms')
    plot_df.set_index('ds', inplace=True)

    # --- Time sync (same logic as breakout notification) ---
    tf_ms_map = {"15m": 900000, "1H": 3600000, "4H": 14400000, "1D": 86400000}
    tf_ms = tf_ms_map.get(tf, 14400000)
    curr_last_time = int(plot_df['open_time'].iloc[-1])
    base_open_time = line.get('base_open_time', curr_last_time)

    shift = round((curr_last_time - base_open_time) / tf_ms)
    if shift < 0:
        shift = 0

    base_idx = line.get('base_idx', view_limit - 1)
    last_math_x = base_idx + shift
    offset = last_math_x - (view_limit - 1)

    # --- Build trend line values and peak markers ---
    line_vals = []
    all_peaks_series = [float('nan')] * view_limit
    point_a_series = [float('nan')] * view_limit
    point_b_series = [float('nan')] * view_limit

    idx_a_view = -1
    idx_b_view = -1
    all_peaks_set = set(line.get('all_peaks', []))

    min_chart_price = plot_df['low'].min() * 0.001
    max_chart_price = plot_df['high'].max() * 1000.0

    for i in range(view_limit):
        math_x = i + offset
        val_log = line['slope'] * math_x + line['intercept']
        val = np.exp(val_log)
        val_safe = max(min(val, max_chart_price), min_chart_price)
        line_vals.append(val_safe)

        if math_x in all_peaks_set:
            all_peaks_series[i] = plot_df['high'].iloc[i]

        if math_x == line.get('index_A'):
            point_a_series[i] = line['price_A']
            idx_a_view = i

        if math_x == line.get('index_B'):
            point_b_series[i] = line['price_B']
            idx_b_view = i

    addplots = []
    if not all(np.isnan(x) for x in all_peaks_series):
        addplots.append(mpf.make_addplot(all_peaks_series, type='scatter', markersize=50, marker='o', color='lime', alpha=0.8))
    if idx_a_view != -1:
        addplots.append(mpf.make_addplot(point_a_series, type='scatter', markersize=120, marker='o', color='blue'))
    if idx_b_view != -1:
        addplots.append(mpf.make_addplot(point_b_series, type='scatter', markersize=120, marker='o', color='red'))

    # --- Trend line status label ---
    current_price = float(plot_df['close'].iloc[-1])
    line_price_now = np.exp(line['slope'] * last_math_x + line['intercept'])
    diff_pct = ((current_price / line_price_now) - 1) * 100

    if diff_pct >= 0:
        price_label = f"Price is ABOVE trendline by {diff_pct:.2f}%"
    else:
        price_label = f"Price is {abs(diff_pct):.2f}% BELOW trendline"

    line_type = line.get('type', 'SCAN')
    file_path = f"scan_{symbol}_{tf}.png"
    fig = None

    try:
        # Create figure with 4 panels: main chart (6) + OBV (1) + RSI (1) + MACD (1)
        _dummy_panels = [
            mpf.make_addplot([float('nan')] * view_limit, panel=1, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=2, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=3, color='gray'),
        ]

        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots + _dummy_panels, yscale='log',
            title=f"\n{symbol} {tf} | {line_type} (LOG-MODE)",
            figsize=(14, 13), returnfig=True, tight_layout=True,
            panel_ratios=(6, 1, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        if idx_a_view != -1:
            ax.text(idx_a_view, line['price_A'], f"{line['price_A']:.4f}", color='blue', fontsize=11, fontweight='bold', ha='center', va='bottom')
        if idx_b_view != -1:
            ax.text(idx_b_view, line['price_B'], f"{line['price_B']:.4f}", color='red', fontsize=11, fontweight='bold', ha='center', va='bottom')

        # Watermark
        ax.text(0.5, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='center', va='bottom', alpha=0.9)

        # Price vs trendline info
        ax.text(0.5, 0.97, price_label, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green' if diff_pct >= 0 else 'red', alpha=0.7))

        # SMC overlay (Order Blocks, BOS/CHoCH, Strong/Weak High/Low)
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, offset)
            except Exception as e:
                logging.error(f"❌ SMC overlay error: {repr(e)}")

        # Custom grid + date labels + right margin SMC
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_right_margin_smc(ax, smc_overlay, view_limit, plot_df)

        # Indicator panels (OBV, RSI, MACD)
        try:
            _add_indicator_panels(fig, axlist, plot_df, view_limit)
        except Exception as e:
            logging.error(f"❌ Indicator panels error (scan): {repr(e)}")

        # Date labels on bottom panel
        _apply_date_labels_bottom(axlist[6], plot_df, view_limit)

        fig.savefig(file_path, dpi=120, bbox_inches='tight')

    except Exception as e:
        logging.error(f"❌ Error generating scan chart {symbol}: {repr(e)}")
        return None
    finally:
        if fig:
            fig.clf()
        plt.close('all')
        gc.collect()

    if os.path.exists(file_path):
        return file_path
    return None


async def draw_simple_chart(symbol: str, df: pd.DataFrame, tf: str, smc_overlay: dict = None) -> str | None:
    """
    Draw a simple candlestick chart WITHOUT trend line.
    Used when trend line construction fails.
    Returns file path to PNG or None on failure.
    """
    view_limit = min(len(df), 199)

    plot_df = df.iloc[-view_limit:].copy().reset_index(drop=True)
    plot_df['ds'] = pd.to_datetime(plot_df['open_time'], unit='ms')
    plot_df.set_index('ds', inplace=True)

    file_path = f"scan_{symbol}_{tf}.png"
    fig = None

    try:
        # Create figure with 4 panels: main chart (6) + OBV (1) + RSI (1) + MACD (1)
        _dummy_panels = [
            mpf.make_addplot([float('nan')] * view_limit, panel=1, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=2, color='gray'),
            mpf.make_addplot([float('nan')] * view_limit, panel=3, color='gray'),
        ]

        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            addplot=_dummy_panels, yscale='log',
            title=f"\n{symbol} {tf} | SCAN (LOG-MODE)",
            figsize=(14, 13), returnfig=True, tight_layout=True,
            panel_ratios=(6, 1, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Watermark
        ax.text(0.5, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='center', va='bottom', alpha=0.9)

        # No trendline label
        ax.text(0.5, 0.97, "No trendline detected", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.7))

        # SMC overlay
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, 0)
            except Exception as e:
                logging.error(f"❌ SMC overlay error (simple): {repr(e)}")

        # Custom grid + date labels + right margin SMC
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_right_margin_smc(ax, smc_overlay, view_limit, plot_df)

        # Indicator panels (OBV, RSI, MACD)
        try:
            _add_indicator_panels(fig, axlist, plot_df, view_limit)
        except Exception as e:
            logging.error(f"❌ Indicator panels error (simple): {repr(e)}")

        # Date labels on bottom panel
        _apply_date_labels_bottom(axlist[6], plot_df, view_limit)

        fig.savefig(file_path, dpi=120, bbox_inches='tight')

    except Exception as e:
        logging.error(f"❌ Error generating simple chart {symbol}: {repr(e)}")
        return None
    finally:
        if fig:
            fig.clf()
        plt.close('all')
        gc.collect()

    if os.path.exists(file_path):
        return file_path
    return None
