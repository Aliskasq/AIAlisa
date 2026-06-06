#!/usr/bin/env python3
"""Debug: dump leg changes for PTBUSDT 1D to find missing swing pivots."""
import asyncio
import aiohttp
import numpy as np

SWING_SIZE = 50

async def fetch_klines(session, symbol, interval, limit):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        raw = await resp.json()
        return [{"open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4])} for c in raw]

def compute_legs_debug(highs, lows, size):
    """Compute legs with verbose output on every change."""
    n = len(highs)
    legs = np.zeros(n, dtype=int)
    current_leg = 0
    changes = []

    for i in range(size, n):
        bar_high = highs[i - size]
        bar_low = lows[i - size]
        window_high = np.max(highs[i - size + 1: i + 1])
        window_low = np.min(lows[i - size + 1: i + 1])

        old_leg = current_leg
        newLegHigh = bar_high > window_high
        newLegLow = bar_low < window_low

        if newLegHigh:
            current_leg = 0  # BEARISH_LEG
        elif newLegLow:
            current_leg = 1  # BULLISH_LEG

        legs[i] = current_leg

        if current_leg != old_leg:
            pivot_bar = i - size
            pivot_type = "HIGH" if current_leg == 0 else "LOW"
            pivot_price = highs[pivot_bar] if pivot_type == "HIGH" else lows[pivot_bar]
            changes.append({
                "detection_bar": i,
                "pivot_bar": pivot_bar,
                "type": pivot_type,
                "price": pivot_price,
                "old_leg": old_leg,
                "new_leg": current_leg,
                "newLegHigh": newLegHigh,
                "newLegLow": newLegLow,
                "bar_high": bar_high,
                "window_high": window_high,
                "bar_low": bar_low,
                "window_low": window_low,
            })
    return legs, changes

async def main():
    async with aiohttp.ClientSession() as session:
        candles = await fetch_klines(session, "PTBUSDT", "1d", 300)
    
    print(f"Total candles: {len(candles)}")
    print(f"First: O={candles[0]['open']} H={candles[0]['high']} L={candles[0]['low']} C={candles[0]['close']}")
    print(f"Last:  O={candles[-1]['open']} H={candles[-1]['high']} L={candles[-1]['low']} C={candles[-1]['close']}")
    
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    
    print(f"\n=== LEG CHANGES (swing_size={SWING_SIZE}) ===")
    legs, changes = compute_legs_debug(highs, lows, SWING_SIZE)
    
    for c in changes:
        print(f"  Bar {c['detection_bar']:3d} → Pivot {c['type']:4s} @ bar {c['pivot_bar']:3d}  "
              f"price={c['price']:.6f}  "
              f"(newHigh={c['newLegHigh']} newLow={c['newLegLow']} "
              f"barH={c['bar_high']:.6f} winH={c['window_high']:.6f} "
              f"barL={c['bar_low']:.6f} winL={c['window_low']:.6f})")
    
    print(f"\nTotal leg changes: {len(changes)}")
    
    # Also dump bars 0-10 OHLC for verification with TradingView
    print(f"\n=== FIRST 10 BARS (verify matches TradingView) ===")
    for i in range(min(10, len(candles))):
        print(f"  Bar {i}: O={candles[i]['open']:.6f} H={candles[i]['high']:.6f} "
              f"L={candles[i]['low']:.6f} C={candles[i]['close']:.6f}")

    # Bars around where TradingView might detect pivots (bar 50-100 area)
    print(f"\n=== BARS 50-70 (where early structure might form) ===")
    for i in range(50, min(71, len(candles))):
        print(f"  Bar {i}: O={candles[i]['open']:.6f} H={candles[i]['high']:.6f} "
              f"L={candles[i]['low']:.6f} C={candles[i]['close']:.6f} leg={legs[i]}")

asyncio.run(main())
