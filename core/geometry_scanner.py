import logging
import pandas as pd
import numpy as np

try:
    from config import load_alerts, save_alerts
except ImportError:
    pass

def get_body_max(row):
    return max(row['open'], row['close'])

def is_peak_flexible(df, idx, radius=6):
    if idx < 1 or idx >= len(df) - 1:
        return False

    target_val = get_body_max(df.iloc[idx])
    left_rad = radius
    right_rad = min(radius, len(df) - 1 - idx)

    for i in range(max(0, idx - left_rad), idx + right_rad + 1):
        if get_body_max(df.iloc[i]) > target_val:
            return False

    has_left_slope = any(get_body_max(df.iloc[i]) < target_val for i in range(max(0, idx - left_rad), idx))
    has_right_slope = any(get_body_max(df.iloc[i]) < target_val for i in range(idx + 1, idx + right_rad + 1))

    if not has_left_slope: return False
    if right_rad > 0 and not has_right_slope: return False

    if idx + 1 < len(df) and get_body_max(df.iloc[idx + 1]) == target_val:
        return False

    return True

def is_line_valid_advanced(df_full, idx_A, idx_B, price_A, price_B, is_fallback=False):
    try:
        if idx_B <= idx_A or float(price_A) <= float(price_B):
            return False, 0.0, False

        log_A = np.log(float(price_A))
        log_B = np.log(float(price_B))

        m = (log_B - log_A) / (idx_B - idx_A)
        c = log_A - m * idx_A

        eps = 1e-10
        last_index = len(df_full) - 1
        dist_B_C = last_index - idx_B

        closes_lin = df_full['close'].values.astype(float)
        opens_lin = df_full['open'].values.astype(float)
        closes_log = np.log(closes_lin)
        opens_log = np.log(opens_lin)

        is_intersecting_current = False
        mid_AB_breaches = 0
        mid_BC_breaches = 0

        for i in range(idx_A + 1, last_index + 1):
            if i == idx_B: continue

            line_p_log = m * i + c
            body_max_log = max(opens_log[i], closes_log[i])
            body_min_log = min(opens_log[i], closes_log[i])
            body_size_log = body_max_log - body_min_log

            # Strict rule: Current or previous candle cannot be pierced by standard line
            if i >= last_index - 1:
                if body_max_log > line_p_log + eps:
                    is_intersecting_current = True
                    if not is_fallback:
                        return False, m, False
                continue

            dist_to_A = abs(i - idx_A)
            dist_to_B = abs(i - idx_B)
            dist_to_anchor = min(dist_to_A, dist_to_B)

            body_size_lin = abs(closes_lin[i] - opens_lin[i])
            is_doji = body_size_lin <= (opens_lin[i] * 0.01) + 1e-10
            if dist_to_anchor <= 6 and is_doji: continue

            if idx_B < i < last_index - 1:
                if body_min_log > line_p_log + eps: return False, m, False

            if dist_to_anchor <= 6:
                if dist_to_anchor in [1, 2, 3]:
                    threshold_log = body_max_log - body_size_log * 0.5
                else:
                    threshold_log = body_max_log - body_size_log * (1.0 / 3.0)
                if line_p_log < threshold_log - eps: return False, m, False
            else:
                if body_max_log > line_p_log + eps:
                    if is_fallback:
                        return False, m, False # Fallback must be 100% clean
                    
                    if i < idx_B:
                        if (idx_B - idx_A) >= 70:
                            threshold_log = body_max_log - body_size_log * (1.0 / 3.0)
                            if line_p_log >= threshold_log - eps:
                                mid_AB_breaches += 1
                                if mid_AB_breaches > 3: return False, m, False
                            else: return False, m, False
                        else: return False, m, False
                    else:
                        if dist_B_C >= 15:
                            threshold_log = body_max_log - body_size_log * (1.0 / 3.0)
                            if line_p_log >= threshold_log - eps:
                                mid_BC_breaches += 1
                                if mid_BC_breaches > 3: return False, m, False
                            else: return False, m, False
                        else:
                            # CRITICAL FIX (Photo 2 issue solved): 
                            # If distance is < 15, NO PIERCING ALLOWED
                            return False, m, False

        return True, m, is_intersecting_current
    except Exception:
        logging.exception("Error in is_line_valid_advanced")
        return False, 0.0, False

async def find_trend_line(df, tf_name, symbol, mode="ROOF"):
    if df is None or df.empty:
        logging.warning(f"⚠️ {symbol:10} {tf_name}: Data is empty.")
        return None, {"error": "no_data"}

    try:
        current_alerts = load_alerts() or []
        filtered_alerts = [a for a in current_alerts if not (a['symbol'] == symbol and a['tf'] == tf_name)]
        if len(current_alerts) != len(filtered_alerts):
            save_alerts(filtered_alerts)
    except Exception:
        pass

    df[['high', 'low', 'close', 'open']] = df[['high', 'low', 'close', 'open']].apply(pd.to_numeric)
    
    # Strict slice to ensure local 0 to 198 coordinates
    view_limit = min(199, len(df))
    df_l = df.iloc[-view_limit:].copy().reset_index(drop=True)
    
    last_idx = len(df_l) - 1
    current_price = float(df_l['close'].iloc[-1])

    best_line = None
    stats = {"peaks": 0, "break_err": 0, "fallback_err": 0}

    raw_peaks =[]
    for i in range(7, last_idx):
        if any(is_peak_flexible(df_l, i, radius=r) for r in range(1, 7)):
            raw_peaks.append(i)

    valid_peaks =[]
    last_height = 0
    for p_idx in sorted(raw_peaks, reverse=True):
        p_price = get_body_max(df_l.iloc[p_idx])
        if p_price > last_height:
            if not valid_peaks or abs(p_idx - valid_peaks[-1]) >= 3:
                valid_peaks.append(p_idx)
                last_height = p_price
            else:
                if p_price > get_body_max(df_l.iloc[valid_peaks[-1]]):
                    valid_peaks[-1] = p_idx
                    last_height = p_price
    valid_peaks.sort()

    if len(valid_peaks) < 2 and len(raw_peaks) >= 2:
        valid_peaks =[]
        last_height = 0
        for p_idx in sorted(raw_peaks, reverse=True):
            p_price = get_body_max(df_l.iloc[p_idx])
            if p_price > last_height:
                valid_peaks.append(p_idx)
                last_height = p_price
        valid_peaks.sort()

    stats["peaks"] = len(valid_peaks)

    clean_candidates = []
    dirty_candidates =[]

    for i in range(len(valid_peaks) - 1, -1, -1):
        idx_B = valid_peaks[i]
        price_B = get_body_max(df_l.iloc[idx_B])

        for j in range(0, i):
            idx_A = valid_peaks[j]
            price_A = get_body_max(df_l.iloc[idx_A])

            if price_A <= price_B: continue
            
            dist_B_C = last_idx - idx_B
            if dist_B_C < 5 and (idx_B - idx_A) < 5:
                continue

            ok, m, is_inter = is_line_valid_advanced(df_l, idx_A, idx_B, price_A, price_B)

            if ok:
                line_price_now = np.exp(m * last_idx + (np.log(price_A) - m * idx_A))
                data = {
                    'm': m, 'is_inter': is_inter, 'idx_A': idx_A, 'idx_B': idx_B,
                    'price_A': price_A, 'price_B': price_B,
                    'line_price_now': line_price_now, 'dist': idx_B - idx_A
                }
                if is_inter:
                    dirty_candidates.append(data)
                else:
                    if line_price_now >= current_price * 0.998:
                        clean_candidates.append(data)

    best_clean = None
    if clean_candidates:
        if str(tf_name).upper() == "1D":
            # 1D Logic: Get nearest line that is not shorter than 30% of the longest valid line
            max_dist = max(c['dist'] for c in clean_candidates)
            valid_length_cands =[c for c in clean_candidates if c['dist'] >= (max_dist * 0.70)]
            best_clean = min(valid_length_cands, key=lambda x: x['line_price_now'])
        else:
            # 4H Logic with >80 candles rule
            anchor_clean = min(clean_candidates, key=lambda x: x['line_price_now'])
            ceiling_price_20 = anchor_clean['line_price_now'] * 1.20
            
            valid_roofs = [c for c in clean_candidates if c['line_price_now'] <= ceiling_price_20]
            if valid_roofs:
                longest_roof = max(valid_roofs, key=lambda x: (x['dist'], x['line_price_now']))
                
                # Check if Point B of the longest line is more than 80 candles away
                dist_B_to_end = last_idx - longest_roof['idx_B']
                if dist_B_to_end > 80:
                    ceiling_price_10 = anchor_clean['line_price_now'] * 1.10
                    valid_roofs_10 =[c for c in clean_candidates if c['line_price_now'] <= ceiling_price_10]
                    if valid_roofs_10:
                        best_clean = max(valid_roofs_10, key=lambda x: (x['dist'], x['line_price_now']))
                    else:
                        best_clean = anchor_clean # Fallback to nearest if 10% fails
                else:
                    best_clean = longest_roof

    best_dirty = None
    if dirty_candidates:
        best_dirty = min(dirty_candidates, key=lambda x: x['line_price_now'])

    fallback_choice = None
    fallback_error_msg = ""

    if len(valid_peaks) > 0:
        # Try Current Candle
        cand_B = last_idx
        price_B = get_body_max(df_l.iloc[cand_B]) * 1.02

        for idx_A in reversed(valid_peaks):
            if (last_idx - idx_A) < 4:
                fallback_error_msg = f"dist {(last_idx - idx_A)} < 4"
                continue
                
            price_A = get_body_max(df_l.iloc[idx_A])
            if price_A <= price_B: continue

            ok, m, is_inter = is_line_valid_advanced(df_l, idx_A, cand_B, price_A, price_B, is_fallback=True)
            if ok:
                line_price_now = np.exp(m * last_idx + (np.log(price_A) - m * idx_A))
                fallback_choice = {
                    'm': m, 'is_inter': True, 'idx_A': idx_A, 'idx_B': cand_B,
                    'price_A': price_A, 'price_B': price_B,
                    'line_price_now': line_price_now, 'dist': cand_B - idx_A, 'fb_type': 'CURRENT'
                }
                break

        # Try Previous Candle
        if not fallback_choice:
            cand_B = last_idx - 1
            price_B = get_body_max(df_l.iloc[cand_B]) * 1.02

            for idx_A in reversed(valid_peaks):
                if (last_idx - idx_A) < 4:
                    fallback_error_msg = f"dist {(last_idx - idx_A)} < 4"
                    continue

                price_A = get_body_max(df_l.iloc[idx_A])
                if price_A <= price_B: continue

                ok, m, is_inter = is_line_valid_advanced(df_l, idx_A, cand_B, price_A, price_B, is_fallback=True)
                if ok:
                    line_price_now = np.exp(m * last_idx + (np.log(price_A) - m * idx_A))
                    if get_body_max(df_l.iloc[last_idx]) > line_price_now:
                        continue 
                        
                    fallback_choice = {
                        'm': m, 'is_inter': False, 'idx_A': idx_A, 'idx_B': cand_B,
                        'price_A': price_A, 'price_B': price_B,
                        'line_price_now': line_price_now, 'dist': cand_B - idx_A, 'fb_type': 'PREV'
                    }
                    break

    final_choice = None
    baseline_dirty = fallback_choice if fallback_choice else best_dirty
    bypass_used = False

    if best_clean and baseline_dirty:
        diff_pct = (best_clean['line_price_now'] - baseline_dirty['line_price_now']) / baseline_dirty['line_price_now']
        if diff_pct <= 0.20 or current_price > baseline_dirty['line_price_now'] or str(tf_name).upper() == "1D":
            final_choice = best_clean
        else:
            final_choice = baseline_dirty
    elif best_clean:
        final_choice = best_clean
        if not baseline_dirty and fallback_error_msg:
            bypass_used = True
    elif baseline_dirty:
        final_choice = baseline_dirty

    if bypass_used:
        logging.info(f"⚠️ {symbol} ({tf_name}): Fallback failed ({fallback_error_msg}). Selected clean line >20%.")

    if final_choice:
        m = final_choice['m']
        idx_A = final_choice['idx_A']
        idx_B = final_choice['idx_B']
        price_A = final_choice['price_A']
        price_B = final_choice['price_B']
        line_price_now = final_choice['line_price_now']
        fb_type = final_choice.get('fb_type')

        is_fallback_peak = (fb_type is not None)
        
        prev_is_green = float(df_l['close'].iloc[last_idx-1]) >= float(df_l['open'].iloc[last_idx-1])
        curr_is_red = float(df_l['close'].iloc[last_idx]) < float(df_l['open'].iloc[last_idx])
        green_then_red = prev_is_green and curr_is_red

        if is_fallback_peak:
            if fb_type == 'CURRENT':
                line_type = "GROWING-CANDLE-MODE"
                trigger_price = current_price * 1.02
                status = "WAITING_2_PERCENT"
            else:
                line_type = "PEAK-TO-PEAK"
                trigger_price = line_price_now * 1.02
                status = "WAITING_2_PERCENT"
        else:
            if final_choice.get('is_inter', False) or current_price >= line_price_now:
                line_type = "GROWING-CANDLE-MODE"
                trigger_price = current_price * 1.02
                status = "WAITING_2_PERCENT"
            else:
                line_type = "PEAK-TO-PEAK"
                trigger_price = line_price_now * 1.02
                status = "WAITING_2_PERCENT"

        if status == "WAITING_2_PERCENT" and current_price >= trigger_price:
            status = "READY"

        base_open_time = int(df_l['open_time'].iloc[-1])
        intercept = float(np.log(price_A) - float(m) * idx_A)

        if status in ["WAITING_2_PERCENT", "WAITING_RED_CLOSE"]:
            try:
                current_alerts = load_alerts() or []
                if not any(a['symbol'] == symbol and a['tf'] == tf_name for a in current_alerts):
                    tf_ms = 14400000 if tf_name == "4H" else 86400000
                    current_alerts.append({
                        'symbol': symbol, 'tf': tf_name,
                        'trigger_price': round(trigger_price, 8),
                        'line_price': round(line_price_now, 8),
                        'base_price': round(current_price, 8),
                        'type': line_type, 'status': status,
                        'added_at': str(pd.Timestamp.now()),
                        'slope': float(m), 'intercept': intercept,
                        'base_idx': int(last_idx),
                        'base_open_time': base_open_time,
                        'tf_ms': tf_ms
                    })
                    save_alerts(current_alerts)
            except Exception as e:
                logging.error(f"❌ JSON Error: {e}")

        best_line = {
            'tf': tf_name, 'index_A': int(idx_A), 'index_B': int(idx_B),
            'price_A': float(price_A), 'price_B': float(price_B), 'slope': float(m),
            'intercept': intercept, 'type': line_type, 
            'all_peaks': [int(p) for p in valid_peaks],
            'trigger_price': float(trigger_price), 'line_price': float(line_price_now),
            'status': status, 'is_active': False, 'base_open_time': base_open_time
        }

    if best_line:
        logging.info(f"✅ {symbol:10} | {tf_name:3} | {best_line['type']:19} | Trig: {best_line.get('trigger_price', 0):.8f} | A={price_A:.8f} B={price_B:.8f} Current={current_price:.8f}")
    else:
        stats["fallback_err"] += 1
        reason = "Pierced by bodies" if stats["peaks"] >= 2 else f"Not enough peaks ({stats['peaks']})"
        if fallback_error_msg: reason += f" (Fallback failed: {fallback_error_msg})"
        logging.info(f"❌ {symbol:10} | {tf_name:3} | Not built: {reason}")

    return best_line, stats
