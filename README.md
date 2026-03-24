# 🦞 AiAlisa Copilot: The Ultimate OpenClaw Trading & Influencer Agent

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![OpenClaw](https://img.shields.io/badge/Powered_by-OpenClaw_SDK-F3BA2F.svg?logo=binance)
![Binance API](https://img.shields.io/badge/Binance-Web3_Skills-F3BA2F.svg?logo=binance)

**AiAlisa CopilotClaw** is a fully autonomous, enterprise-grade AI trading assistant and content publisher built natively on the **Binance OpenClaw Architecture**. 

Designed to enhance the Binance ecosystem, Alisa solves four major challenges:
1. **Trading Strategy:** Scans **540+ Binance Futures pairs** simultaneously using proprietary logarithmic geometry and Smart Money Concepts.
2. **Risk Management:** Interactively calculates exact Stop-Loss limits based on user-defined margin and leverage.
3. **Education:** `/learn` mode explains all 16+ indicators in plain language — making crypto accessible to beginners.
4. **Community Marketing:** Empowers Crypto-Influencers with automated AI publications to **Binance Square** with extended analysis.

---

### ⚡ Key Features

| Feature | Description |
|---|---|
| 🔴 **Live AI Streaming** | Real-time SSE token streaming — watch AI think live in Telegram |
| 📐 **SMC Indicator** | Full Smart Money Concepts engine: BOS/CHoCH structure, Order Blocks, Fair Value Gaps, EQH/EQL, Premium/Discount zones — ported from LuxAlgo TradingView. Runs on 4H, 1H, 15m with 250 candles |
| 📚 **Education Mode** | `/learn BTC` — explains every indicator in plain language for beginners |
| 🏆 **Signal Tracker** | `/signals` — virtual bank with live P&L, leverage, deposit%, winrate. Tracks TP/SL hits, AI prediction accuracy (✅/❌) |
| 🔒 **Signals Close** | `/signals close` — snapshot: what if you closed all open positions right now? |
| 🔄 **Signals Clear** | `/signals clear` — reset virtual bank to $10k, keep today's signals |
| 🔬 **Extended Analysis** | `scan BTC` returns deep breakdown of all 16+ indicators + SMC Indicator + 7 Web3 skills |
| 🌐 **Bilingual** | `/lang en` / `/lang ru` — full English & Russian support, per-chat persistent |
| 🔔 **Smart Alerts** | `/alert BTC 75000` — persistent price alerts that survive restarts |
| 📢 **Smart Auto-Post** | AI analysis auto-published to Binance Square (post format, ~1000 chars) with SMC data |
| 💼 **Paper Trading** | `/paper BTC 74000 long 5x` — virtual portfolio with live P&L from real Binance prices |
| 👋 **Auto-Welcome** | New group members receive full command list automatically |

---

### 📐 SMC Indicator (Smart Money Concepts) — NEW

Full port of the **LuxAlgo Smart Money Concepts** TradingView indicator into Python (`core/smc.py`).

Analyzes **250 candles** on each timeframe (4H, 1H, 15m) and detects:

| Component | Description |
|---|---|
| **BOS** (Break of Structure) | Trend continuation — price broke the last swing high/low |
| **CHoCH** (Change of Character) | Trend reversal — price broke in the opposite direction |
| **Order Blocks** (OB) | Key candles at structure breaks — real support/resistance levels |
| **Fair Value Gaps** (FVG) | 3-candle imbalances — price magnets, unfilled gaps |
| **EQH / EQL** | Equal Highs/Lows — liquidity pools where stops cluster |
| **Premium / Discount** | Current zone relative to swing range — overbought vs oversold |

**Internal structure** (size=5) for short-term moves, **Swing structure** (size=50) for major trends.  
ATR-based volatility filter for Order Blocks (high-volatility bars parsed separately, same as LuxAlgo).

SMC data is fed directly into the AI prompt so that **SL/TP are placed on real Order Blocks, FVG levels, and structure breaks** — not random numbers.

> ⚠️ **Not to confuse with** the Binance Web3 "Smart Money" skill (`/skills smart money`) which tracks whale wallets via Binance DeFi API. The SMC Indicator is a technical analysis tool based on price action.

---

### 🏆 Virtual Bank & Signal Tracking

The `/signals` system tracks every breakout signal with a virtual bank:

- **Entry price** = breakout price (price at the moment the signal was pushed)
- **Leverage** and **deposit %** from AI recommendation are applied to P&L
- **SL/TP** sanity check: TP must be above entry for LONG (below for SHORT), otherwise ignored
- **R:R minimum 1:1.5** — AI must give TP distance ≥ 1.5× SL distance
- **AI prediction match**: ✅ if AI direction (LONG/SHORT) matches actual P&L direction, ❌ if not
- **SL/TP must be justified** by real technical levels (EMA, OB, support, ATR, Fibonacci) with reason in parentheses

Commands:
- `/signals` — full view: virtual bank + today's signals + all-time stats
- `/signals close` — snapshot: close all open positions at current price, show day P&L
- `/signals clear` — reset bank to $10,000, clear all-time stats (keep today's signals)

---

### 🦞 Deep OpenClaw SDK Integration (4 Services)

| SDK Service | Purpose | Fallback |
|---|---|---|
| **`client.extract.run(model=TradeVerdict)`** | Returns **typed Pydantic models** (entry, SL, TP as floats) for programmatic signal validation | `agent.run()` → OpenRouter |
| **`client.skills.run(skill_name, prompt)`** | All 7 Binance Web3 Skills routed through **OpenClaw Skills API** | Direct HTTP to Binance Web3 API |
| **`client.agent.run(prompt)`** | Core AI inference for trading verdicts via CMDOP Cloud relay | OpenRouter aiohttp failsafe |
| **SSE Streaming** | Real-time token-by-token streaming to Telegram via `stream=true` | Progressive display fallback |

The 3-tier fallback chain (`Extract → Agent → OpenRouter`) ensures **zero downtime**.

---

## 📸 Proof of Concept & Killer Features

### 1. Autonomous Global Scanning & Safe 1-Click Publishing
At the configured scan time, Alisa analyzes over 540 futures pairs. Using a custom logarithmic algorithm, she builds geometric trendlines. When a true breakout occurs, the OpenClaw Agent analyzes the chart with **SMC Indicator + 16+ technical indicators** and sends a fully generated push notification to the Telegram group.

<img width="1600" height="1327" alt="frames-export-1773344230773_edit_462820069716103" src="https://github.com/user-attachments/assets/bad291f2-d21c-44d6-97d7-e6b3611c3c3c" />

### 2. Interactive Analysis & Dynamic Risk Management
Type `scan BTC` or `посмотри BTC` for full AI analysis with SMC Indicator data. Reply to any signal with margin/leverage for exact Stop-Loss calculation.

<img width="1600" height="1331" alt="frames-export-1773344720986_edit_463360898442876" src="https://github.com/user-attachments/assets/9f24d943-5d69-4a49-9d84-309c4ab77e67" />

### 3. Automated Influencer Mode
The Square Publisher generates and posts AI market updates to Binance Square automatically. Post format (~1000 chars, not article) with `#AIBinance #BinanceSquare #Write2Earn`. Supports **unlimited schedule times** via `/autopost time 09:00 12:30 15:00 18:00 21:00`.

![7469](https://github.com/user-attachments/assets/ef30b84b-ae3f-4341-a16e-62cb03315da3)

### 4. Real-Time AI Streaming (🔴 LIVE)
`scan BTC` streams the AI's reasoning **token by token** directly into Telegram via SSE. The message updates live every 1.5s with a blinking cursor (▌) and 🔴 LIVE indicator.

### 5. Education Mode
`/learn BTC` explains all indicators with real-time values in plain language. Covers RSI, MFI, ADX, StochRSI, MACD, OBV, Ichimoku, SuperTrend, Volume Decay, Funding Rate, and more.

### 6. Signal Accuracy Tracker
`/signals` provides full transparency on bot performance with virtual bank P&L, leverage-adjusted returns, AI prediction accuracy (✅/❌), and winrate stats.

### 7. Binance Web3 Skills Dashboard
`/skills` provides interactive access to on-chain data: Smart Money signals, Social Hype, Meme Rank, Inflow, Top Traders PnL, Token Rank.

![Screenshot_20260312_221445_org_telegram_messenger_LaunchActivity](https://github.com/user-attachments/assets/d077ac2b-7295-438b-a960-9d1744dcd74a)

<details>
<summary><b>🔌 OpenClaw Web3 Skills Arsenal (Click to Expand)</b></summary>
<br>

1. **🐋 Smart Money Signals** — whale wallet tracking via Binance DeFi API
2. **🔥 Social Hype Leaderboard** — community sentiment and retail FOMO
3. **💸 Smart Money Net Inflow** — 24h capital inflow ranking across chains
4. **📊 Unified Token Rank** — Binance trending + top search rankings
5. **🐶 Meme Rank** — explosive breakout probability scoring
6. **👨‍💻 Top Traders PnL** — 30-day win rate of best on-chain traders
7. **📢 Binance Square OpenAPI** — auto-publish AI reports to Square
</details>

<details>
<summary><b>🧠 Technical Analysis Arsenal (Click to Expand)</b></summary>
<br>

**Custom Mathematics:**
- **Logarithmic Trendline Formula:** $Price = e^{(m \cdot x + c)}$ for geometric slope filtering
- **SMC Indicator:** BOS/CHoCH, Order Blocks, FVG, EQH/EQL, Premium/Discount zones (ported from LuxAlgo TradingView)

**16+ Technical Indicators:**
- **Trend & Momentum:** EMA (7, 25, 99), RSI (6, 12, 24), MACD, ADX, SuperTrend, Ichimoku Cloud, StochRSI
- **Volume & Money Flow:** OBV, MFI, CMF, Volume Decay, VWAP, Volume Block Analysis
- **Volatility:** Bollinger Bands (20, 2), ATR (14)
- **Grid:** Dynamic Fibonacci retracements
- **Funding Rate:** Real-time from Binance Futures API

**250 candles** per timeframe for accurate SMC detection on 4H, 1H, 15m.
</details>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TELEGRAM USER                      │
│    scan BTC / /learn ETH / /signals / /alert SOL     │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        BINANCE FUTURES API (540+ pairs)               │
│    250 candles × 4H/1H/15m/1D                        │
│    Logarithmic Geometry Scanner + API Weight Limiter  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        SMC INDICATOR (core/smc.py)                    │
│  BOS/CHoCH │ Order Blocks │ FVG │ EQH/EQL │ Zones   │
│  Internal (size=5) + Swing (size=50) structure        │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        OPENCLAW SKILLS + EXTRACT + STREAMING          │
│  Web3 Skills │ TradeVerdict │ SSE Live Streaming      │
│  3-tier fallback: Extract → Agent → OpenRouter        │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        OUTPUT: TELEGRAM + BINANCE SQUARE              │
│  📊 Chart + AI Verdict + SMC │ 📢 Auto-Post          │
│  🏆 Signal Tracker │ 📚 Education │ 🔔 Alerts        │
└─────────────────────────────────────────────────────┘
```

---

## 🎙️ Command Reference

**👥 Public Commands:**

| Command | Description |
|---|---|
| `scan BTC` / `look BTC` / `посмотри BTC` | 🔍 AI analysis with SMC + streaming |
| `/learn BTC` | 📚 Education mode |
| `margin 100 leverage 10 max 20%` | 💰 Risk calculator |
| `/skills` | 🛠 Web3 Skills dashboard |
| `/top gainers` / `/top losers` | 📈📉 Top 10 (24h) |
| `/trend` | 📊 All breakouts + AI accuracy |
| `/alert BTC 69500` | 🔔 Price alert |
| `/alert list` / `/alert clear` | 🔔 Manage alerts |
| `/lang en` / `/lang ru` | 🌐 Language switch |

**🔐 Admin Commands:**

| Command | Description |
|---|---|
| `/signals` | 🏆 Virtual bank + signal winrate |
| `/signals close` | 🔒 Close all open — day P&L snapshot |
| `/signals clear` | 🔄 Reset bank to $10k |
| `/models` | 🧠 AI engine selector |
| `/time HH:MM` | ⏰ Set scan schedule |
| `/autopost on` / `off` | 📢 Toggle Square publisher |
| `/autopost SOL BTC ETH` | 🪙 Set coins |
| `/autopost time 09:00 12:00 15:00 18:00 21:00` | ⏰ Set times (unlimited) |
| `/post [text]` | ✏️ Manual post to Square |
| `/paper BTC 74000 long 5x sl 73000 tp 75000` | 💼 Open paper position |
| `/paper` / `/paper close 1` / `/paper history` / `/paper clear` | 💼 Paper trading |

---

## 🚀 Installation

### Prerequisites
- Ubuntu/Linux server
- Python 3.11
- Telegram Bot Token, Binance API keys, OpenRouter API key

### Quick Setup
```bash
git clone https://github.com/Aliskasq/AIAlisa.git
cd AIAlisa
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install openclaw cmdop
pip install -r requirements.txt

# OpenClaw SDK hotfix
echo "TimeoutError = TimeoutError" >> venv/lib/python3.11/site-packages/cmdop/exceptions.py

# Configure
cp .env.example .env
nano .env  # Fill in API keys

# Create systemd service
sudo tee /etc/systemd/system/AI.service > /dev/null << 'EOF'
[Unit]
Description=AI Alisa Copilot Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/AIAlisa
ExecStart=/root/AIAlisa/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable AI.service
sudo systemctl start AI.service

# Watch logs
journalctl -u AI.service -f
```

### Verify
```bash
source venv/bin/activate
python -c 'import openclaw; print("✅ OPENCLAW SDK READY")'
python -c 'from agent.analyzer import TradeVerdict; print("✅ TradeVerdict model:", TradeVerdict.model_fields.keys())'
python -c 'from core.smc import analyze_smc; print("✅ SMC Indicator READY")'
```

---

## 📁 Project Structure

```
AIAlisa/
├── main.py                  # Entry point: geometry scanner + breakout engine
├── config.py                # Virtual bank, breakout log, alerts, AI param parser
├── agent/
│   ├── analyzer.py          # AI prompt builder + OpenClaw SDK integration
│   └── square_publisher.py  # Binance Square auto-publisher
├── core/
│   ├── smc.py               # SMC Indicator (BOS/CHoCH, OB, FVG, EQH/EQL)
│   ├── indicators.py        # 16+ technical indicators calculator
│   ├── tg_listener.py       # Telegram command handler (scan, signals, paper, etc.)
│   ├── geometry_scanner.py  # Logarithmic trendline builder
│   ├── chart_drawer.py      # Chart image generator with trendlines
│   └── binance_api.py       # Binance API wrapper with rate limiting
└── data/                    # Runtime data (breakout log, virtual bank, alerts, etc.)
```

---

*Built with ❤️ on [OpenClaw](https://github.com/openclaw) for the Binance ecosystem.*
