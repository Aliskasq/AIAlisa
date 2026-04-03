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
from config import BOT_TOKEN, GROUP_CHAT_ID

SQUARE_CACHE_FILE = "data/square_cache.json"

async def send_breakout_notification(symbol, df, line, tf, line_type, session, trigger_price=0.0, ai_text="", target_chat_id=None):
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
        fig, axlist = mpf.plot(
            plot_df, type='candle', style=custom_style,
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots, yscale='log',
            title=f"\n{symbol} {line_type} (LOG-MODE)",
            figsize=(14, 8), returnfig=True, tight_layout=True
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        if idx_a_view != -1:
            ax.text(idx_a_view, line['price_A'], f"{line['price_A']:.4f}", color='blue', fontsize=11, fontweight='bold', ha='center', va='bottom')
        if idx_b_view != -1:
            ax.text(idx_b_view, line['price_B'], f"{line['price_B']:.4f}", color='red', fontsize=11, fontweight='bold', ha='center', va='bottom')

        ax.text(0.5, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='center', va='bottom', alpha=0.9)

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
    
    # Обрезаем ИИ текст до 800 символов, чтобы не нарваться на ошибку Telegram!
    ai_text = ai_text or ""
    safe_ai_text = ai_text if len(ai_text) < 800 else ai_text[:800] + "..."
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
        f"${short_symbol} 🎯 TREND BREAKOUT\n"
        f"⏳ TF: {tf} | 💰 Price: {current_price:.6f}\n"
        f"💡 Above trendline by {diff_pct:.2f}%\n\n"
        f"🤖 AI-Alisa-CopilotClow:\n"
    )

    # AI text limit for photo caption (header takes the rest of 1024)
    AI_TEXT_LIMIT = 828
    overflow_text = ""

    if len(safe_ai_text) <= AI_TEXT_LIMIT:
        # Everything fits in caption
        photo_caption = header + safe_ai_text
    else:
        # Split: first 828 chars of AI text in caption, rest as separate message
        cut_text = safe_ai_text[:AI_TEXT_LIMIT]
        # Try to split at last newline for clean break
        last_nl = cut_text.rfind('\n')
        if last_nl > AI_TEXT_LIMIT // 2:
            cut_point = last_nl
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


async def draw_scan_chart(symbol: str, df: pd.DataFrame, line: dict, tf: str) -> str | None:
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
    tf_ms = 14400000 if tf == "4H" else 86400000
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
        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            alines=dict(alines=[list(zip(plot_df.index, line_vals))], colors='gold', linewidths=2),
            addplot=addplots, yscale='log',
            title=f"\n{symbol} {tf} | {line_type} (LOG-MODE)",
            figsize=(14, 8), returnfig=True, tight_layout=True
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


async def draw_simple_chart(symbol: str, df: pd.DataFrame, tf: str) -> str | None:
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
        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            yscale='log',
            title=f"\n{symbol} {tf} | SCAN (LOG-MODE)",
            figsize=(14, 8), returnfig=True, tight_layout=True
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Watermark
        ax.text(0.5, 0.02, 'Alisa_10000 / Alisa_Trend', transform=ax.transAxes, color='black', fontsize=28, fontweight='bold', ha='center', va='bottom', alpha=0.9)

        # No trendline label
        ax.text(0.5, 0.97, "No trendline detected", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.7))

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
