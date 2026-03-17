# 🦞 AiAlisa Copilot: The Ultimate OpenClaw Trading & Influencer Agent

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![OpenClaw](https://img.shields.io/badge/Powered_by-OpenClaw_SDK-F3BA2F.svg?logo=binance)
![Binance API](https://img.shields.io/badge/Binance-Web3_Skills-F3BA2F.svg?logo=binance)

**AiAlisa Copilot** is a fully autonomous, enterprise-grade AI trading assistant and content publisher built natively on the **Binance OpenClaw Architecture**. 

Designed to enhance the Binance ecosystem, Alisa solves three major challenges:
1. **Trading Strategy:** Scans all 540+ Binance Futures pairs simultaneously using proprietary logarithmic geometry and Smart Money Concepts.
2. **Risk Management:** Interactively calculates exact Stop-Loss limits based on user-defined margin and leverage.
3. **Community Marketing:** Empowers Crypto-Influencers by providing 1-click & fully automated AI publications directly to **Binance Square**.

### 🦞 Deep OpenClaw SDK Integration (3 Services)

AiAlisa is not just "using OpenClaw for chat" — the entire AI pipeline is built on **three distinct CMDOP/OpenClaw SDK services**:

| SDK Service | Purpose | Fallback |
|---|---|---|
| **`client.extract.run(model=TradeVerdict)`** | Returns **typed Pydantic models** (entry, SL, TP as floats) instead of raw text. Enables programmatic signal validation. | `agent.run()` → OpenRouter |
| **`client.skills.run(skill_name, prompt)`** | All 7 Binance Web3 Skills are routed through the **OpenClaw Skills API** for orchestrated execution. | Direct HTTP to Binance Web3 API |
| **`client.agent.run(prompt)`** | Core AI inference for trading verdicts via CMDOP Cloud relay. | OpenRouter aiohttp failsafe |

**Why this matters:** Every AI decision passes through OpenClaw's typed extraction pipeline. The `TradeVerdict` Pydantic model ensures the agent returns **validated floats** for Entry/SL/TP — not just text that might hallucinate wrong numbers. This is critical for real trading risk management.

---

## 📸 Proof of Concept & Killer Features

### 1. Autonomous Global Scanning & Safe 1-Click Publishing
Exactly at `00:00 UTC` (the start of a new daily candle), Alisa analyzes over 540 futures pairs. Using a custom logarithmic algorithm, she builds geometric trendlines. When a true breakout occurs, the OpenClaw Agent analyzes the chart and sends a fully generated push notification to a dedicated Telegram group.
* **Binance Square Integration:** The signal features an inline button to instantly publish the AI analysis to Binance Square. 
* **Zero-Trust Security:** To prevent sabotage, the button verifies Telegram roles. You can invite friends to your signal group—only the Admin can trigger the Binance Square publishing. Routine commands are kept in a separate private DM to keep the main group clean.

<img width="1600" height="1327" alt="frames-export-1773344230773_edit_462820069716103" src="https://github.com/user-attachments/assets/bad291f2-d21c-44d6-97d7-e6b3611c3c3c" />

### 2. Interactive Analysis & Dynamic Risk Management
Traders can manually request an analysis by typing `look [coin]` (e.g., `look btc`). The OpenClaw agent immediately processes the chart, applies Web3 skills, and delivers a verdict.
* **Smart Risk Manager:** If a user replies to an AI signal with text like: *"Margin 100, leverage 5x. What stop-loss should I set to avoid losing more than 10%?"* — the OpenClaw LLM strictly calculates the exact asset price for the Stop-Loss to defensively protect the portfolio.
  
<img width="1600" height="1331" alt="frames-export-1773344720986_edit_463360898442876" src="https://github.com/user-attachments/assets/9f24d943-5d69-4a49-9d84-309c4ab77e67" />


### 3. Fully Automated "Hands-Free" Influencer Mode & Scheduling
Alisa is not just a responder; she operates in the background. The bot features an internal `Square Publisher` task that utilizes OpenClaw to automatically generate and post bi-daily market updates to Binance Square without any human intervention.
* **Dynamic Time Configuration:** You don't need to modify code or restart the server. Admins can dynamically change the global scanning or posting schedule directly via Telegram (e.g., `/time 03 00` shifts the internal chronometer on the fly).


![Screenshot_20260312_221919_com_termux_TermuxActivity_edit_461923476867955](https://github.com/user-attachments/assets/5ad82289-4531-477f-9287-38c35bc2eb6a)


### 4. Interactive Binance Web3 Skills Dashboard
Powered by the new Binance OpenClaw Web3 API integration, Alisa provides an interactive `/skills` dashboard. Instead of browsing multiple websites, users can instantly retrieve on-chain data directly inside the chat to validate their trading bias.

![Screenshot_20260312_221445_org_telegram_messenger_LaunchActivity](https://github.com/user-attachments/assets/d077ac2b-7295-438b-a960-9d1744dcd74a)


<details>
<summary><b>🔌 The Core: OpenClaw Web3 Skills Arsenal (Click to Expand)</b></summary>
<br>
To give the AI true market awareness, Alisa deeply integrates with the newly released **Binance OpenClaw Web3 API**. Instead of relying purely on mathematical indicators, the agent requests real-time on-chain data and social sentiment before making a final verdict. 

Through the intuitive Telegram `/skills` dashboard, users and the AI can execute the following proprietary skills:

1. **🐋 Smart Money Signals (`get_smart_money_signals`):** Queries Binance Defi API to track massive whale wallets. It filters the on-chain noise to detect distinct buy/sell directions in real-time for specific base assets (e.g., BTC, ETH).
2. **🔥 Social Hype Leaderboard (`get_social_hype_leaderboard`):** Analyzes global community sentiment. The LLM uses this to measure retail FOMO (Fear Of Missing Out) and identify overcrowded trades.
3. **💸 Smart Money Net Inflow (`get_smart_money_inflow_rank`):** Scans multi-chain networks (ChainId: 56) to rank tokens by pure 24h capital inflow. Alisa uses this to find hidden accumulation phases before the price breaks out.
4. **📊 Unified Token Rank (`get_unified_token_rank`):** Fetches Binance's internal macro pulse (Trending and Top Search rankings) to align the AI's bias with the broader market direction.
5. **🐶 Breakthrough Meme Rank (`get_meme_rank`):** Utilizes Binance's exclusive pulse scoring system to algorithmically identify meme coins with the highest probability of an explosive breakout.
6. **👨‍💻 Top Traders PnL (`get_address_pnl_rank`):** Tracks the 30-day Win Rate and Realized PnL of the absolute best on-chain traders, allowing the user to follow "smart money" wallet behavior.
7. **📢 Binance Square OpenAPI (`post_to_binance_square`):** A custom write-skill that bypasses manual entry, allowing the AI to format, hashtag, and push the final financial report directly to the millions of users on the Binance Square social network in under 1 second.
</details>

---

<details>
<summary><b>🧠 The Brain: Deep Technical Analysis Arsenal (Click to Expand)</b></summary>
<br>
To ensure the OpenClaw Agent delivers professional-grade verdicts, we feed it highly structured mathematical data. The prompt injection includes real-time values from:

**Custom Mathematics:**
*   **Logarithmic Trendline Formula:** Calculates geometric slopes using $Price = e^{(m \cdot x + c)}$ to filter market noise.
*   **Smart Money Concepts (SMC):** Real-time mapping of Order Blocks (OB) and Fair Value Gaps (FVG).

**Market Sentiment:**
*   Live Binance Futures **Funding Rates** are aggressively monitored to predict short squeezes.

**10+ Technical Indicators:**
*   **Trend & Oscillators:** EMA (7, 25, 99), MACD Histogram, Ichimoku Cloud, SuperTrend, ADX, RSI (6), StochRSI, MFI (Money Flow Index).
*   **Volume Metrics:** OBV (On-Balance Volume) and custom Volume Decay algorithms.
*   **Grid:** Dynamic Fibonacci retracements based on local extremums.
</details>

---

## 🏗️ OpenClaw SDK Architecture: How It Works

```
┌─────────────────────────────────────────────────────┐
│                   TELEGRAM USER                      │
│         scan BTC / margin 100 leverage 5x            │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│          BINANCE FUTURES API (540+ pairs)            │
│    fetch_klines() → Logarithmic Geometry Scanner     │
│    199 candles × 4H/1D → find_trend_line()           │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        OPENCLAW SKILLS SDK (client.skills.run)       │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │ smart_money_     │  │ social_hype_leaderboard  │  │
│  │ signals          │  │ meme_rank                │  │
│  │ inflow_rank      │  │ unified_token_rank       │  │
│  │ address_pnl_rank │  │ post_to_binance_square   │  │
│  └─────────────────┘  └──────────────────────────┘  │
│          ↓ fallback: Direct Binance Web3 HTTP        │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│      OPENCLAW EXTRACT (client.extract.run)           │
│                                                      │
│   Prompt + Indicators + Skills Data                  │
│        ↓                                             │
│   TradeVerdict(BaseModel):                           │
│     direction: "LONG"                                │
│     entry_price: 98245.50    ← typed float           │
│     stop_loss: 96100.00      ← typed float           │
│     take_profit: 103500.00   ← typed float           │
│     risk_percent: 4.37       ← typed float           │
│     logic: "Funding negative, Smart Money buying..." │
│        ↓ fallback: agent.run() → OpenRouter          │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│        TELEGRAM + BINANCE SQUARE OUTPUT              │
│  📊 Chart PNG + AI Verdict + [Post to Square] btn   │
└─────────────────────────────────────────────────────┘
```

The 3-tier fallback chain (`Extract → Agent → OpenRouter`) ensures **zero downtime** — if any OpenClaw service is temporarily unavailable, the bot seamlessly degrades to the next layer without the user noticing.

---

## 🛡️ Enterprise-Grade Engine (Respecting Binance Infrastructure)

A great bot doesn't just trade; it protects the ecosystem it lives in. 
We implemented a strict **Binance API Rate Limit Manager** (`core/binance_api.py`). The bot actively listens to the `X-MBX-USED-WEIGHT-1M` headers from Binance APIs. 
If the API weight exceeds `2350 / 2400`, the bot automatically triggers a safe throttle, pauses its asynchronous gather loops, and broadcasts a warning message to the administrator, completely eliminating the risk of receiving an IP ban (HTTP 429).

---

## 🎙️ Telegram Command Center

Alisa provides a comprehensive control panel via Telegram:

**👥 Public Commands (all users):**
*   `/start` - Initializes the dashboard.
*   `scan [coin]` / `look [coin]` / `посмотри [coin]` - Forces an immediate OpenClaw analysis of a specific asset.
*   `margin 100 leverage 10 max 20%` - Reply to a signal for exact Stop-Loss math.
*   `/skills` - Opens the interactive Binance Web3 Skills keyboard.
*   `/top gainers` - Top 10 Futures growth (24h).
*   `/top losers` - Top 10 Futures drops (24h).
*   `/trend` - Lists all coins that broke through trendlines since the last global scan (with breakout price & current price).
*   `/alert [coin] [price]` - Set a price alert (e.g., `/alert BTC 69500`). Get notified when the target price is reached.
*   `/alert list` - View your active alerts.
*   `/alert clear` - Remove all your alerts.

**🔐 Admin Commands (bot owner only):**
*   `/model` - Interactive AI engine selector (inline buttons: Free / GPT / Gemini models via OpenRouter).
*   `/time HH:MM` - Dynamically sets the time for the global geometric scan.
*   `/autopost` - Shows auto-posting status.
*   `/autopost on / off` - Toggles the background Binance Square automatic publisher task.
*   `/autopost SOL BTC ETH` - Sets coins for auto-posting.
*   `/autopost time 13:30 22:50` - Sets auto-post schedule.
*   `/post [text]` - Manually publishes text to Binance Square.
*   `📢 Post to Binance Square` button under signals - Admin-verified inline publishing.

**🤖 Automated Tasks:**
*   **Global Geometric Scan** - Runs at the configured `/time` schedule (default: 03:00 UTC+3).
*   **Daily Trend Summary** - Auto-sends all breakout coins to the group at 23:57 UTC (02:57 UTC+3).
*   **Square Auto-Publisher** - Disabled by default. Enable via `/autopost on`.

---

## 🚀 Easy Installation Guide (Production Ready)

This project is built for Ubuntu/Linux servers. To avoid dependency conflicts and ensure complete compatibility with the **OpenClaw SDK** (`openclaw` + `cmdop`), we strictly use **Python 3.11** in an isolated environment. Follow these exact steps to deploy the agent 24/7.

### Step 1: Install Python 3.11
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev build-essential -y
```

### Step 2: Clone and Setup Environment
```bash
git clone https://github.com/Aliskasq/AIAlisa.git
cd AIAlisa

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install openclaw cmdop
pip install -r requirements.txt
```

### Step 4: OpenClaw Library Hotfix (Crucial)
*Note: Currently, the beta release of the `cmdop` package (an OpenClaw dependency) is missing the `TimeoutError` import. This causes the SDK to throw an OS dependency error (`WARNING ⚠️ OpenClaw core detected missing OS dependencies`) on boot and forces the agent into the failsafe routing mode (bypassing `client.extract.run` and `client.skills.run`).* 

Apply this 1-line patch to fix the library code directly via terminal:
```bash
echo "TimeoutError = TimeoutError" >> venv/lib/python3.11/site-packages/cmdop/exceptions.py
```

### Step 5: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
nano .env
```
Fill in the following values:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your personal Telegram user ID (admin) |
| `TELEGRAM_GROUP_CHAT_ID` | Signal group chat ID |
| `OPENROUTER_API_KEY` | OpenRouter API key (failsafe AI routing) |
| `CMDOP_API_KEY` | CMDOP/OpenClaw API key (primary AI routing) |
| `SQUARE_OPENAPI_KEY` | Binance Square OpenAPI key |
| `ENCRYPTION_KEY` | Fernet key for Binance API encryption |
| `ENCRYPTED_API_KEY` | Encrypted Binance API key |
| `ENCRYPTED_SECRET_KEY` | Encrypted Binance Secret key |

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 6: Create a Background Service (systemd)
To keep the bot running 24/7, create a system service:
```bash
sudo tee /etc/systemd/system/alisa.service > /dev/null << 'EOF'
[Unit]
Description=AI Alisa Copilot Bot (OpenClaw SDK)
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/AIAlisa
ExecStart=/root/AIAlisa/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=alisa-bot
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start the daemon:
```bash
sudo systemctl daemon-reload
sudo systemctl enable alisa.service
sudo systemctl start alisa.service
```

### Step 7: Verify Installation
Verify that the OpenClaw SDK is successfully compiled and patched:
```bash
source venv/bin/activate
python -c 'import openclaw; print("✅ OPENCLAW SDK READY")'
```

Verify the new Structured Output model loads:
```bash
python -c 'from agent.analyzer import TradeVerdict; print("✅ TradeVerdict model:", TradeVerdict.model_fields.keys())'
```

Watch the live logs to ensure the bot is online:
```bash
journalctl -u alisa.service -f
```

You should see these lines confirming the 3-service OpenClaw integration:
```
⚙️ [OpenClaw Architecture] Executing Agentic Binance Skills...
✅ [OpenClaw SDK] Skill 'smart_money_signals' executed successfully
📊 [OpenClaw Extract] Requesting structured TradeVerdict...
✅ OpenClaw Extract: Structured verdict received → LONG (Entry: ..., SL: ..., TP: ...)
```

### Step 8: Trigger the Geometric Scan (Bootstrapping)
By default, the global logarithmic mathematical scan is heavily resource-intensive and is scheduled to trigger exactly at `00:00 UTC` (the start of a new daily candle). 
If you want to test the breakout detection **immediately**, enter your Telegram bot chat and dynamically change the schedule using the `/time` command.

*(Note: The server time zone logs operate in UTC+3)*. 
Example: If your local server time is `17:54`, type this to the bot:
**`/time 17 55`**

The bot will reply: `✅ Global scan time successfully changed to 17:55 (UTC+3)`. Once the minute hits, the scanning sequence will execute, and you will begin receiving real-time AI analytical signals and charts!
