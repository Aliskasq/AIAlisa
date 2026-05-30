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
        # Compute indicator addplots for panels (RSI only, no MACD)
        _ind_addplots, _rsi_vals = _compute_indicator_addplots(plot_df, view_limit)

        fig, axlist = mpf.plot(
            plot_df, type='candle', style=custom_style,
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots + _ind_addplots, yscale='log',
            title=f"\n{symbol} {line_type}",
            figsize=(14, 12), returnfig=True, tight_layout=False,
            panel_ratios=(6, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Tight Y-axis: candle range ± 5% padding (log-safe)
        import math as _math
        _candle_low = float(plot_df['low'].min())
        _candle_high = float(plot_df['high'].max())
        if _candle_low > 0 and _candle_high > _candle_low:
            _log_lo = _math.log(_candle_low)
            _log_hi = _math.log(_candle_high)
            _log_pad = (_log_hi - _log_lo) * 0.05
            ax.set_ylim(_math.exp(_log_lo - _log_pad), _math.exp(_log_hi + _log_pad))

        if idx_a_view != -1:
            ax.text(idx_a_view, line['price_A'], f"{line['price_A']:.4f}", color='blue', fontsize=11, fontweight='bold', ha='center', va='bottom')
        if idx_b_view != -1:
            ax.text(idx_b_view, line['price_B'], f"{line['price_B']:.4f}", color='red', fontsize=11, fontweight='bold', ha='center', va='bottom')

        # Watermark: end at second-to-last candle, right-aligned
        _wm_x = (view_limit - 2) / max(view_limit - 1, 1)  # normalized 0-1
        ax.text(_wm_x, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='right', va='bottom', alpha=0.9)

        # Clamp Y-axis BEFORE drawing overlay (so lines for far-away values are skipped)
        clamp_info = {}
        if smc_overlay:
            clamp_info = _prepare_chart_ylim(ax, smc_overlay, plot_df)

        # SMC overlay on breakout chart
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, offset, clamp_info)
            except Exception as e:
                logging.error(f"❌ SMC overlay error (breakout): {repr(e)}")

        # Custom grid + SMC annotations
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_smc_annotations(ax, fig, smc_overlay, view_limit, plot_df, clamp_info)

        # Style indicator panels (RSI levels + value labels)
        try:
            _style_indicator_panels(axlist, rsi_values=_rsi_vals, fig=fig)
        except Exception as e:
            logging.error(f"❌ Indicator panels style error: {repr(e)}")

        # Date labels between main chart and indicators (fig-level text)
        _apply_date_labels_main(ax, fig, plot_df, view_limit, axlist=axlist)

        # Nuke "1e7" offset text: set offset=False on ScalarFormatter axes,
        # and blank the text on all axes so bbox_inches='tight' can't bring it back
        from matplotlib.ticker import ScalarFormatter
        for _ax in axlist:
            fmt = _ax.yaxis.get_major_formatter()
            if isinstance(fmt, ScalarFormatter):
                fmt.set_useOffset(False)
            ot = _ax.yaxis.get_offset_text()
            ot.set_visible(False)
            ot.set_text("")

        fig.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0.1)

    except Exception as e:
        logging.error(f"❌ Error generating chart {symbol}: {repr(e)}")
    finally:
        if fig:
            fig.clf()
        plt.close('all')
        gc.collect()

    if not os.path.exists(file_path):
        return False, None

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
    Removes default matplotlib grid, x/y tick labels, and "Price" ylabel.
    """
    import matplotlib.ticker as mticker

    # Remove default grid, tick labels, and "Price" ylabel
    ax.grid(False)
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', which='both', labelright=False, labelleft=False, right=False, left=False)
    ax.set_ylabel('')  # Remove "Price" label

    # --- Vertical lines every 20 candles ---
    for i in range(0, view_limit, 20):
        ax.axvline(x=i, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)

    # --- Horizontal lines every 10% price ---
    y_low, y_high = ax.get_ylim()
    if y_low > 0 and y_high > 0:
        import math
        log_low = math.log10(y_low)
        log_high = math.log10(y_high)
        base = 10 ** math.floor(log_low)
        price = base
        while price < y_low:
            price *= 1.1
        while price <= y_high:
            ax.axhline(y=price, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
            price *= 1.1


def _apply_date_labels_main(ax_main, fig, plot_df, view_limit, axlist=None):
    """
    Draw date labels between main chart and indicators using fig.text().
    Uses figure-level coordinates so text is never hidden behind indicator panels.
    Bold black font, 2-line format (date + time), every 20 candles.
    NOTE: panel spacing is handled by _style_indicator_panels — do NOT call
    subplots_adjust here as it overrides manual panel positioning.
    """
    fig.canvas.draw()

    bbox_main = ax_main.get_position()
    main_bottom = bbox_main.y0

    # Find OBV panel top to place labels in the middle of the gap
    obv_top = main_bottom - 0.05  # fallback
    if axlist is not None:
        panels = {}
        for ax in axlist:
            pnum = getattr(ax, '_panel_num', None)
            if pnum is not None and pnum not in panels:
                panels[pnum] = ax
        if not panels.get(1) and len(axlist) >= 6:
            panels[1] = axlist[2]
        elif not panels.get(1) and len(axlist) >= 3:
            panels[1] = axlist[1]
        if panels.get(1):
            obv_top = panels[1].get_position().y1

    # Place date labels in the middle of the gap between main chart and OBV
    y_fig = (main_bottom + obv_top) / 2

    for i in range(0, view_limit, 20):
        if i < len(plot_df):
            dt = plot_df.index[i]
            label = f"{dt.strftime('%b %d')}\n{dt.strftime('%H:%M')}"
            disp = ax_main.transData.transform((i, 0))
            fig_x = fig.transFigure.inverted().transform(disp)[0]
            fig.text(fig_x, y_fig, label,
                     color='black', fontsize=7, fontweight='bold',
                     ha='center', va='center')


def _prepare_chart_ylim(ax, smc_data, plot_df):
    """
    Clamp Y-axis if Strong High/Low is too far (>30%) from visible candles.
    Must be called BEFORE _draw_smc_overlay so lines for clamped values are skipped.
    Returns dict with clamping info.
    """
    info = {"high_clamped": False, "low_clamped": False}
    if not smc_data or "error" in str(smc_data.get("summary", "")).lower():
        return info

    trailing = smc_data.get("trailing", {})
    t_high = trailing.get("trailing_high")
    t_low = trailing.get("trailing_low")
    chart_high = float(plot_df['high'].max())
    chart_low = float(plot_df['low'].min())

    if t_high is not None and chart_high > 0 and t_high / chart_high > 1.3:
        info["high_clamped"] = True
    if t_low is not None and chart_low > 0 and chart_low / t_low > 1.3:
        info["low_clamped"] = True

    if info["high_clamped"] or info["low_clamped"]:
        cur_ylow, cur_yhigh = ax.get_ylim()
        new_top = chart_high * 1.10 if info["high_clamped"] else cur_yhigh
        new_bot = chart_low * 0.90 if info["low_clamped"] else cur_ylow
        ax.set_ylim(new_bot, new_top)

    return info


def _draw_smc_annotations(ax, fig, smc_data, view_limit, plot_df, clamp_info=None):
    """
    Draw SMC annotations on the chart:
    - Strong High label ABOVE the high line, Strong Low BELOW the low line
    - If high/low clamped: show "+XX%" at top/bottom of chart
    - Visible OB prices at block boundaries (INSIDE the block, near edges)
    - Off-screen OBs: ↑ arrows in top strip, ↓ arrows in bottom strip
    """
    import matplotlib.transforms as mtransforms

    if not smc_data or "error" in str(smc_data.get("summary", "")).lower():
        return

    if clamp_info is None:
        clamp_info = {}

    trailing = smc_data.get("trailing", {})
    t_high = trailing.get("trailing_high")
    t_low = trailing.get("trailing_low")
    high_label = trailing.get("high_label", "High")
    low_label = trailing.get("low_label", "Low")
    current_price = float(plot_df['close'].iloc[-1])

    high_clamped = clamp_info.get("high_clamped", False)
    low_clamped = clamp_info.get("low_clamped", False)

    y_low, y_high = ax.get_ylim()

    # Normalized X for second-to-last candle (right-aligned anchor)
    _wm_x = (view_limit - 2) / max(view_limit - 1, 1)

    # --- Strong High / Weak High label ---
    if t_high is not None:
        price_str = f"{t_high:.4f}" if t_high >= 0.01 else f"{t_high:.6f}"
        if high_clamped:
            pct = ((t_high / current_price) - 1) * 100
            # Clamped high: show on the TITLE line (top of chart, right side)
            high_text = f"  {high_label}  {price_str}  (+{pct:.0f}%)"
            ax.text(0.99, 1.02, high_text,
                    color='#FF0000', fontsize=14, fontweight='bold',
                    ha='right', va='bottom',
                    transform=ax.transAxes, clip_on=False, zorder=6)
        else:
            # Label ABOVE the high line (offset up by ~1 line width so it doesn't sit on the line)
            y_lo, y_hi = ax.get_ylim()
            h_offset = (y_hi - y_lo) * 0.012
            ax.text(view_limit - 3, t_high + h_offset, f"{high_label}  {price_str}",
                    color='#FF0000', fontsize=12, fontweight='bold',
                    ha='right', va='bottom', zorder=6)

    # --- Strong Low / Weak Low label ---
    if t_low is not None:
        price_str = f"{t_low:.4f}" if t_low >= 0.01 else f"{t_low:.6f}"
        if low_clamped:
            pct = ((current_price - t_low) / current_price) * 100
            # Clamped low: show on the WATERMARK line (Alisa row, after the watermark text)
            low_text = f"  {low_label}  {price_str}  (-{pct:.0f}%)"
            # Position just to the left of watermark end (watermark is right-aligned at _wm_x)
            ax.text(_wm_x, 0.06, low_text,
                    color='#089981', fontsize=14, fontweight='bold',
                    ha='right', va='bottom',
                    transform=ax.transAxes, clip_on=False, zorder=6)
        else:
            # Label BELOW the low line (offset down by ~1 line width)
            y_lo, y_hi = ax.get_ylim()
            l_offset = (y_hi - y_lo) * 0.012
            ax.text(view_limit - 3, t_low - l_offset, f"{low_label}  {price_str}",
                    color='#089981', fontsize=12, fontweight='bold',
                    ha='right', va='top', zorder=6)

    # --- Collect all OBs for annotation (both internal and swing) ---
    all_obs = list(smc_data.get("swing_order_blocks", []))
    all_obs += list(smc_data.get("internal_order_blocks", []))

    # Classify: visible vs above vs below chart area
    visible_obs = []
    above_obs = []
    below_obs = []

    for ob in all_obs:
        if ob["low"] > y_high:
            above_obs.append(ob)
        elif ob["high"] < y_low:
            below_obs.append(ob)
        else:
            visible_obs.append(ob)

    above_obs.sort(key=lambda o: o["low"])
    below_obs.sort(key=lambda o: -o["high"])

    # --- Visible OB prices at block boundaries (INSIDE the block, near top/bottom) ---
    # va='top' for high price → text drops DOWN from top boundary (stays inside)
    # va='bottom' for low price → text rises UP from bottom boundary (stays inside)
    right_x = view_limit + 0.5
    for ob in visible_obs:
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        ax.text(right_x, ob["high"], hi_str,
                color=color, fontsize=6, ha='left', va='top', zorder=5, clip_on=False)
        ax.text(right_x, ob["low"], lo_str,
                color=color, fontsize=6, ha='left', va='bottom', zorder=5, clip_on=False)

    # --- Off-screen OBs: ↑ arrows in TOP strip (above chart) ---
    for i, ob in enumerate(above_obs[:2]):
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        x_frac = 0.75 + i * 0.12
        ax.text(x_frac, 1.01 + i * 0.025, f"↑ {lo_str}-{hi_str}",
                color=color, fontsize=6, fontweight='bold',
                ha='center', va='bottom',
                transform=ax.transAxes, clip_on=False, zorder=5)

    # --- Off-screen OBs: ↓ arrows in BOTTOM strip (below chart, near dates) ---
    for i, ob in enumerate(below_obs[:2]):
        hi_str = f"{ob['high']:.4f}" if ob['high'] >= 0.01 else f"{ob['high']:.6f}"
        lo_str = f"{ob['low']:.4f}" if ob['low'] >= 0.01 else f"{ob['low']:.6f}"
        color = '#F23645' if ob["bias"] == -1 else '#089981'
        x_frac = 0.75 + i * 0.12
        ax.text(x_frac, -0.03 - i * 0.025, f"↓ {lo_str}-{hi_str}",
                color=color, fontsize=6, fontweight='bold',
                ha='center', va='top',
                transform=ax.transAxes, clip_on=False, zorder=5)


def _compute_indicator_addplots(plot_df, view_limit):
    """
    Compute OBV, RSI(6,12,24), MACD from plot_df and return list of
    mpf.make_addplot() objects for panels 1, 2, 3.
    Also returns the MACD histogram colors for bar chart (drawn post-render).
    """
    close = plot_df['close'].astype(float)
    volume = plot_df['volume'].astype(float)
    n = len(close)

    # --- OBV ---
    obv_change = np.sign(close.diff()) * volume
    obv = obv_change.fillna(0).cumsum()
    obv_sma20 = obv.rolling(20).mean().fillna(obv.iloc[0] if n > 0 else 0)

    # --- RSI 6, 12, 24 ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = -delta.where(delta < 0, 0.0)

    def _rsi(g, l, period):
        alpha = 1.0 / period
        avg_g = np.zeros(len(g))
        avg_l = np.zeros(len(g))
        avg_g[0] = g.iloc[0] if len(g) > 0 else 0
        avg_l[0] = l.iloc[0] if len(l) > 0 else 0
        for i in range(1, len(g)):
            avg_g[i] = alpha * g.iloc[i] + (1 - alpha) * avg_g[i - 1]
            avg_l[i] = alpha * l.iloc[i] + (1 - alpha) * avg_l[i - 1]
        rs = np.where(avg_l == 0, 100, avg_g / avg_l)
        return pd.Series(100 - (100 / (1 + rs)), index=g.index)

    rsi6 = _rsi(gain, loss_s, 6)
    rsi12 = _rsi(gain, loss_s, 12)
    rsi24 = _rsi(gain, loss_s, 24)

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Build addplots: OBV (panel 1) + RSI (panel 2), no MACD
    # RSI colors match Binance: RSI(6) yellow, RSI(12) pink, RSI(24) dark purple
    addplots = [
        # Panel 1: OBV
        mpf.make_addplot(obv, panel=1, color='#26a69a', width=1.0, ylabel='OBV'),
        mpf.make_addplot(obv_sma20, panel=1, color='#ef5350', width=0.8, linestyle='--'),
        # Panel 2: RSI 6, 12, 24 (Binance-style colors)
        mpf.make_addplot(rsi6, panel=2, color='#F0B90B', width=1.2, ylabel='RSI'),
        mpf.make_addplot(rsi12, panel=2, color='#E040FB', width=1.0),
        mpf.make_addplot(rsi24, panel=2, color='#7B1FA2', width=1.0),
    ]

    # Store RSI values for labels (used by _style_indicator_panels)
    _last_rsi = {
        'rsi6': float(rsi6.iloc[-1]) if len(rsi6) > 0 else 0,
        'rsi12': float(rsi12.iloc[-1]) if len(rsi12) > 0 else 0,
        'rsi24': float(rsi24.iloc[-1]) if len(rsi24) > 0 else 0,
    }

    return addplots, _last_rsi


def _style_indicator_panels(axlist, rsi_values=None, fig=None):
    """
    Style the indicator panels after rendering:
    - OBV panel 1: disable scientific notation on Y axis
    - RSI panel 2: 70/30 lines + Binance-style value labels
    """
    from matplotlib.ticker import FuncFormatter

    # Disable tight_layout so manual set_position() calls below
    # are not overridden by fig.canvas.draw() / savefig
    if fig is not None:
        fig.set_tight_layout(False)

    # Collect all unique panels via _panel_num attribute
    panels = {}
    for ax in axlist:
        pnum = getattr(ax, '_panel_num', None)
        if pnum is not None and pnum not in panels:
            panels[pnum] = ax

    # Robust fallback: mplfinance with 3 panels creates 6 axes (main+twin per panel)
    if len(panels) < 3:
        if len(axlist) >= 6:
            panels = {0: axlist[0], 1: axlist[2], 2: axlist[4]}
        elif len(axlist) >= 3:
            panels = {0: axlist[0], 1: axlist[1], 2: axlist[2]}
    
    logging.info(f"📊 Panel detection: {len(panels)} panels from {len(axlist)} axes")

    for pnum in [1, 2]:
        ax = panels.get(pnum)
        if not ax:
            continue
        ax.grid(False)
        ax.tick_params(axis='both', labelsize=5, colors='black')
        ax.tick_params(axis='x', labelbottom=False)

    # OBV panel: disable scientific notation (the "1e9" yellow circle)
    ax_obv = panels.get(1)
    if ax_obv:
        def _fmt_obv(x, pos):
            if abs(x) >= 1e9:
                return f"{x/1e9:.1f}B"
            elif abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            elif abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
        ax_obv.yaxis.set_major_formatter(FuncFormatter(_fmt_obv))
        ax_obv.yaxis.get_offset_text().set_visible(False)

        # Horizontal grid lines for OBV (same style as main chart grid)
        obv_low, obv_high = ax_obv.get_ylim()
        obv_range = obv_high - obv_low
        logging.info(f"📊 OBV ylim: {obv_low:.0f} — {obv_high:.0f}, range={obv_range:.0f}")
        if obv_range > 0:
            # Pick a nice step: ~4-6 lines across the panel
            import math
            raw_step = obv_range / 5
            magnitude = 10 ** math.floor(math.log10(abs(raw_step))) if raw_step != 0 else 1
            nice_steps = [1, 2, 2.5, 5, 10]
            step = magnitude * min(nice_steps, key=lambda s: abs(s * magnitude - raw_step))
            level = math.ceil(obv_low / step) * step
            _obv_grid_count = 0
            while level <= obv_high:
                ax_obv.axhline(y=level, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
                level += step
                _obv_grid_count += 1
            logging.info(f"📊 OBV grid: {_obv_grid_count} lines, step={step:.0f}")

    # === FULL MANUAL LAYOUT: main chart + date gap + OBV + sep + RSI ===
    # This replaces mplfinance's default positioning entirely.
    ax_main = panels.get(0)
    ax_rsi = panels.get(2)
    if ax_main and ax_obv and ax_rsi and fig is not None:
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Bbox

        # Get the x-extent from the main chart (left/right margins stay the same)
        bbox0 = ax_main.get_position()
        x0, x1 = bbox0.x0, bbox0.x1

        # Layout parameters (fraction of figure height)
        top_margin = 0.06     # space above main chart (for title)
        bottom_margin = 0.04  # space below RSI
        date_gap = 0.05       # gap between main chart and OBV (for date labels)
        obv_rsi_gap = 0.025   # gap between OBV and RSI (separator line)

        # Panel height ratios: main=6, obv=1, rsi=1 → total 8 parts
        usable = 1.0 - top_margin - bottom_margin - date_gap - obv_rsi_gap
        main_h = usable * (6.0 / 8.0)
        obv_h = usable * (1.0 / 8.0)
        rsi_h = usable * (1.0 / 8.0)

        # Position from top to bottom
        main_top = 1.0 - top_margin
        main_bot = main_top - main_h
        obv_top = main_bot - date_gap
        obv_bot = obv_top - obv_h
        rsi_top = obv_bot - obv_rsi_gap
        rsi_bot = rsi_top - rsi_h

        ax_main.set_position(Bbox([[x0, main_bot], [x1, main_top]]))
        ax_obv.set_position(Bbox([[x0, obv_bot], [x1, obv_top]]))
        ax_rsi.set_position(Bbox([[x0, rsi_bot], [x1, rsi_top]]))

        # Also reposition twin axes (mplfinance creates ax+twin per panel)
        for _ax in axlist:
            pnum = getattr(_ax, '_panel_num', None)
            if pnum == 0 and _ax is not ax_main:
                _ax.set_position(Bbox([[x0, main_bot], [x1, main_top]]))
            elif pnum == 1 and _ax is not ax_obv:
                _ax.set_position(Bbox([[x0, obv_bot], [x1, obv_top]]))
            elif pnum == 2 and _ax is not ax_rsi:
                _ax.set_position(Bbox([[x0, rsi_bot], [x1, rsi_top]]))

        # Separator line between OBV and RSI
        sep_y = (obv_bot + rsi_top) / 2
        sep_line = Line2D([x0, x1 + 0.02], [sep_y, sep_y], transform=fig.transFigure,
                          color='black', linewidth=1.0, zorder=10, clip_on=False)
        fig.add_artist(sep_line)

    # RSI panel: add 70/30 levels + Binance-style value labels
    if not ax_rsi:
        ax_rsi = panels.get(2)
    if ax_rsi:
        ax_rsi.set_ylim(0, 100)
        # Grid lines at 20, 40, 60, 80, 100 (same style as main chart grid)
        for level in [20, 40, 60, 80, 100]:
            ax_rsi.axhline(y=level, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
        logging.info(f"📊 RSI grid: 5 lines drawn (20-100), ylim={ax_rsi.get_ylim()}")
        # Overbought/oversold dashed lines on top of grid
        ax_rsi.axhline(y=70, color='#F23645', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
        ax_rsi.axhline(y=30, color='#089981', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)

        # Binance-style RSI labels at top of panel
        if rsi_values:
            r6 = rsi_values.get('rsi6', 0)
            r12 = rsi_values.get('rsi12', 0)
            r24 = rsi_values.get('rsi24', 0)
            ax_rsi.text(0.01, 0.95, f"RSI(6): {r6:.2f}", color='#F0B90B',
                       fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                       va='top', ha='left')
            ax_rsi.text(0.18, 0.95, f"RSI(12): {r12:.2f}", color='#E040FB',
                       fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                       va='top', ha='left')
            ax_rsi.text(0.38, 0.95, f"RSI(24): {r24:.2f}", color='#7B1FA2',
                       fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                       va='top', ha='left')


def _draw_smc_overlay(ax, plot_df, smc_data, view_limit, global_offset=0, clamp_info=None):
    """
    Draw Smart Money Concepts overlay on chart — matching LuxAlgo TradingView visuals.
    Elements: Order Blocks, BOS/CHoCH lines, Strong/Weak High/Low.
    """
    from matplotlib.patches import Rectangle

    if not smc_data or "error" in str(smc_data.get("summary", "")).lower():
        return

    if clamp_info is None:
        clamp_info = {}

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

    # Both internal and swing OBs displayed.
    # Swing OBs: thin black border to distinguish from internal.
    # Internal OBs: no border (LuxAlgo default style).
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
            # Swing OBs: thin black border; Internal OBs: no border
            rect = Rectangle(
                (ob_x_start, ob_low), width, ob_high - ob_low,
                linewidth=0.8 if not is_internal else 0,
                edgecolor='black' if not is_internal else None,
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

    # ─── 3. STRONG/WEAK HIGH/LOW LINES ─────────────────────────────────
    # Lines extend from the pivot candle all the way to the right margin edge.
    # Using ax.plot() with clip_on=False for reliable edge-to-edge rendering.
    # Colors: red (#F23645) for high, teal/sea-green (#089981) for low.
    # Skip drawing if the value is clamped (off-screen, shown as +XX% text).
    if trailing:
        t_high = trailing.get("trailing_high")
        t_low = trailing.get("trailing_low")
        t_high_idx = trailing.get("trailing_high_index", 0) - idx_offset
        t_low_idx = trailing.get("trailing_low_index", 0) - idx_offset

        if t_high is not None and not clamp_info.get("high_clamped", False):
            x_start_h = max(t_high_idx, -0.5)
            # Line from high candle to right edge + beyond margin
            ax.plot([x_start_h, view_limit + 10], [t_high, t_high],
                    color='#FF0000', linewidth=2.4, alpha=1.0,
                    zorder=3, clip_on=False, solid_capstyle='butt')
            # Small diamond marker on the actual high candle
            if 0 <= t_high_idx < view_limit:
                ax.plot(t_high_idx, t_high, marker='D', color='#FF0000',
                        markersize=5, zorder=4)

        if t_low is not None and not clamp_info.get("low_clamped", False):
            x_start_l = max(t_low_idx, -0.5)
            # Line from low candle to right edge + beyond margin
            ax.plot([x_start_l, view_limit + 10], [t_low, t_low],
                    color='#089981', linewidth=2.4, alpha=1.0,
                    zorder=3, clip_on=False, solid_capstyle='butt')
            # Small diamond marker on the actual low candle
            if 0 <= t_low_idx < view_limit:
                ax.plot(t_low_idx, t_low, marker='D', color='#089981',
                        markersize=5, zorder=4)


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
        # Compute indicator addplots (RSI only)
        _ind_addplots, _rsi_vals = _compute_indicator_addplots(plot_df, view_limit)

        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots + _ind_addplots, yscale='log',
            title=f"\n{symbol} {tf} | {line_type}",
            figsize=(14, 12), returnfig=True, tight_layout=False,
            panel_ratios=(6, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Tight Y-axis: candle range ± 5% padding (log-safe)
        import math as _math
        _candle_low = float(plot_df['low'].min())
        _candle_high = float(plot_df['high'].max())
        if _candle_low > 0 and _candle_high > _candle_low:
            _log_lo = _math.log(_candle_low)
            _log_hi = _math.log(_candle_high)
            _log_pad = (_log_hi - _log_lo) * 0.05
            ax.set_ylim(_math.exp(_log_lo - _log_pad), _math.exp(_log_hi + _log_pad))

        if idx_a_view != -1:
            ax.text(idx_a_view, line['price_A'], f"{line['price_A']:.4f}", color='blue', fontsize=11, fontweight='bold', ha='center', va='bottom')
        if idx_b_view != -1:
            ax.text(idx_b_view, line['price_B'], f"{line['price_B']:.4f}", color='red', fontsize=11, fontweight='bold', ha='center', va='bottom')

        # Watermark: end at second-to-last candle, right-aligned
        _wm_x = (view_limit - 2) / max(view_limit - 1, 1)
        ax.text(_wm_x, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='right', va='bottom', alpha=0.9)

        # Price vs trendline info
        ax.text(0.5, 0.97, price_label, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green' if diff_pct >= 0 else 'red', alpha=0.7))

        # Clamp Y-axis BEFORE overlay
        clamp_info = {}
        if smc_overlay:
            clamp_info = _prepare_chart_ylim(ax, smc_overlay, plot_df)

        # SMC overlay
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, offset, clamp_info)
            except Exception as e:
                logging.error(f"❌ SMC overlay error: {repr(e)}")

        # Custom grid + SMC annotations
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_smc_annotations(ax, fig, smc_overlay, view_limit, plot_df, clamp_info)

        # Style indicator panels (RSI + value labels)
        try:
            _style_indicator_panels(axlist, rsi_values=_rsi_vals, fig=fig)
        except Exception as e:
            logging.error(f"❌ Indicator panels style error (scan): {repr(e)}")

        # Date labels between main chart and indicators
        _apply_date_labels_main(ax, fig, plot_df, view_limit, axlist=axlist)

        # Nuke "1e7" offset text on all axes before save
        from matplotlib.ticker import ScalarFormatter as _SF
        for _ax in axlist:
            _fmt = _ax.yaxis.get_major_formatter()
            if isinstance(_fmt, _SF):
                _fmt.set_useOffset(False)
            _ot = _ax.yaxis.get_offset_text()
            _ot.set_visible(False)
            _ot.set_text("")

        fig.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0.1)

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
        # Compute indicator addplots (RSI only)
        _ind_addplots, _rsi_vals = _compute_indicator_addplots(plot_df, view_limit)

        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            addplot=_ind_addplots, yscale='log',
            title=f"\n{symbol} {tf} | SCAN",
            figsize=(14, 12), returnfig=True, tight_layout=False,
            panel_ratios=(6, 1, 1)
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Tight Y-axis: candle range ± 5% padding (log-safe)
        import math as _math
        _candle_low = float(plot_df['low'].min())
        _candle_high = float(plot_df['high'].max())
        if _candle_low > 0 and _candle_high > _candle_low:
            _log_lo = _math.log(_candle_low)
            _log_hi = _math.log(_candle_high)
            _log_pad = (_log_hi - _log_lo) * 0.05
            ax.set_ylim(_math.exp(_log_lo - _log_pad), _math.exp(_log_hi + _log_pad))

        # Watermark
        # Watermark: end at second-to-last candle, right-aligned
        _wm_x = (view_limit - 2) / max(view_limit - 1, 1)
        ax.text(_wm_x, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='right', va='bottom', alpha=0.9)

        # No trendline label
        ax.text(0.5, 0.97, "No trendline detected", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.7))

        # Clamp Y-axis BEFORE overlay
        clamp_info = {}
        if smc_overlay:
            clamp_info = _prepare_chart_ylim(ax, smc_overlay, plot_df)

        # SMC overlay
        if smc_overlay:
            try:
                _draw_smc_overlay(ax, plot_df, smc_overlay, view_limit, 0, clamp_info)
            except Exception as e:
                logging.error(f"❌ SMC overlay error (simple): {repr(e)}")

        # Custom grid + SMC annotations
        _apply_custom_grid(ax, plot_df, view_limit)
        if smc_overlay:
            _draw_smc_annotations(ax, fig, smc_overlay, view_limit, plot_df, clamp_info)

        # Style indicator panels (RSI + value labels)
        try:
            _style_indicator_panels(axlist, rsi_values=_rsi_vals, fig=fig)
        except Exception as e:
            logging.error(f"❌ Indicator panels style error (simple): {repr(e)}")

        # Date labels between main chart and indicators
        _apply_date_labels_main(ax, fig, plot_df, view_limit, axlist=axlist)

        # Nuke "1e7" offset text on all axes before save
        from matplotlib.ticker import ScalarFormatter as _SF
        for _ax in axlist:
            _fmt = _ax.yaxis.get_major_formatter()
            if isinstance(_fmt, _SF):
                _fmt.set_useOffset(False)
            _ot = _ax.yaxis.get_offset_text()
            _ot.set_visible(False)
            _ot.set_text("")

        fig.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0.1)

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
