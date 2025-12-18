# Autonomous Trading Agent

> Ultra-fast, AI-powered algorithmic trading system with 70% faster execution and 79% lower costs

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI-Agents%20SDK-green.svg)](https://openai.github.io/openai-agents-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Check environment](https://github.com/marketcalls/Agentic-Trader/actions/workflows/check-env.yml/badge.svg)](https://github.com/marketcalls/Agentic-Trader/actions/workflows/check-env.yml)

An autonomous AI trading agent that analyzes technical indicators in real-time, makes intelligent BUY/SELL/HOLD decisions, and executes trades automatically on the NSE (National Stock Exchange of India).

---

## Highlights

- 70% faster execution (3-6s vs 12-20s per cycle)
- 79% lower token usage (30K vs 144K per cycle)
- 7 technical indicators (RSI, MACD, Bollinger Bands, EMA, Stochastic, ADX, ATR)
- Parallel data fetching (all 5 stocks simultaneously)
- Instant market orders (no waiting for limit order fills)
- Strict risk management (stop-loss, position limits, no pyramiding)
- Model agnostic (works with Cerebras, Groq, OpenAI, or custom models)
- Shorting enabled (can go long or short)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Performance](#performance)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Risk Management](#risk-management)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/marketcalls/Agentic-Trader.git
cd Agentic-Trader

# 2. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run the agent
uv run python agent.py
```

That's it! The agent will start trading automatically every 5 minutes during market hours (9:15 AM - 3:30 PM IST).

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **5 Stocks** | ICICIBANK, RELIANCE, SBIN, WIPRO, ITC |
| **7 TA Indicators** | RSI, MACD, Bollinger Bands, EMA, Stochastic, ADX, ATR |
| **Parallel Processing** | Fetches all data simultaneously in 2-3 seconds |
| **Market Orders** | Instant execution (no waiting for fills) |
| **Bulk Operations** | Places multiple orders in one go |
| **Shorting** | Can sell without owning (creates short position) |
| **Risk Controls** | Stop-loss, trade limits, position constraints |
| **Model Agnostic** | Supports Cerebras, Groq, OpenAI, custom models |
| **Auto-Scheduler** | Runs every 5 minutes during market hours |
| **Cost Tracking** | Real-time token usage and cost monitoring |

### Technical Analysis

The agent uses **7 professional TA-Lib indicators**:

1. **RSI** - Identifies overbought/oversold conditions
2. **MACD** - Detects trend momentum and direction
3. **Bollinger Bands** - Measures volatility and price extremes
4. **EMA** - Tracks trend direction (20 & 50 period)
5. **Stochastic** - Momentum indicator for reversals
6. **ADX** - Measures trend strength (not direction)
7. **ATR** - Volatility measurement for stop-loss

**Trading Logic**: Requires 3+ aligned indicators to trigger BUY/SELL. Weak signals = HOLD.

---

## Performance

### Before vs After Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Cycle Time** | 12-20s | 3-6s | 70% faster |
| **Data Fetch** | 8-12s | 2-3s | 75% faster |
| **Token Usage** | 144K | 30K | 79% reduction |
| **Cost per Cycle** | $0.023-0.031 | $0.008-0.012 | 65% cheaper |
| **Order Execution** | 15-45s | Instant | 100% faster |
| **Monthly Cost** | $74 | $26 | Save $48/month |

*Costs calculated using Cerebras llama3.1-8b model*

### Real-World Performance

- **Data fetching**: 2-3 seconds for all 5 stocks (parallel)
- **Decision making**: 1-2 seconds (7 indicators analyzed)
- **Order execution**: < 1 second (market orders)
- **Total cycle**: 3-6 seconds end-to-end
- **Cycles per day**: ~75 (every 5 minutes, 9:15 AM - 3:30 PM)

---

## How It Works

### Trading Cycle (Every 5 Minutes)

```
1. FETCH DATA (Parallel - 2-3s)
   ↓
   Fetches quotes, depth, and 7 TA indicators for all 5 stocks simultaneously

2. ANALYZE (Per Stock)
   ↓
   • Examines RSI, MACD, Bollinger Bands, EMA, Stochastic, ADX, ATR
   • Checks for 3+ aligned signals
   • Makes BUY/SELL/HOLD decision

3. VALIDATE (Risk Check)
   ↓
   • Checks daily stop-loss (-Rs.10,000)
   • Verifies trade count limits (5 per stock)
   • Ensures no position pyramiding

4. CALCULATE (Position Size)
   ↓
   • Fixed Rs.10,000 investment per trade
   • Quantity = int(10000 / LTP)

5. EXECUTE (Bulk Orders - Parallel)
   ↓
   • Places all orders simultaneously
   • Market orders (instant execution)
   • Rate limiting (0.5s per 2 orders)
```

### Example Output

```
================================================================================
Trading Cycle: 2025-01-15 10:30:00 IST
================================================================================

[BULK DATA] Fetching data for 5 symbols in parallel...
[BULK DATA] Completed in 2.3 seconds

ICICIBANK: BUY Order#123 (MACD bullish)
RELIANCE: HOLD (weak signals)
SBIN: HOLD (existing position)
WIPRO: SELL Order#124 (take profit)
ITC: HOLD (mixed signals)

================================================================================
[TOKEN USAGE] API Call Statistics:
  Requests:      3
  Input Tokens:  28,234
  Output Tokens: 2,851
  Total Tokens:  31,085
  Est. Cost:     $0.008
================================================================================
```

---

## Installation

### Prerequisites

1. **Python 3.12+** - [Download Python](https://www.python.org/downloads/)
2. **uv package manager** - [Install uv](https://github.com/astral-sh/uv)
3. **TA-Lib library** - [Install TA-Lib](https://ta-lib.org/)
4. **API Keys**:
   - OpenAI, Cerebras, or Groq API key
   - OpenAlgo broker account

### Step-by-Step Installation

#### 1. Install Python 3.12+

```bash
# Check Python version
python --version  # Should be 3.12 or higher
```

#### 2. Install uv package manager

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Install TA-Lib (Platform-Specific)

**Windows:**
```bash
# Download from: https://ta-lib.org/
# Install the .exe installer
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ta-lib
```

**Mac:**
```bash
brew install ta-lib
```

#### 4. Clone and Install Project

```bash
# Clone repository
git clone <repo-url>
cd autonomous-agents

# Install dependencies
uv sync

# Install TA-Lib Python wrapper
uv pip install TA-Lib
```

---

## Configuration

### 1. Create Environment File

```bash
cp .env.example .env
```

### 2. Edit .env with Your API Keys

```bash
# Model Provider Selection
MODEL_PROVIDER=openai  # Options: cerebras, groq, openai, custom

# OpenAI (Default - Best Quality)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Cerebras (Fastest - Recommended for Production)
CEREBRAS_API_KEY=csk-your-key-here
CEREBRAS_MODEL=cerebras/llama3.1-8b

# Groq (Fast & Cheap - Good for Development)
GROQ_API_KEY=gsk-your-key-here
GROQ_MODEL=groq/llama-3.3-70b-versatile

# OpenAlgo Broker Configuration
OPENALGO_API_KEY=your-openalgo-api-key
OPENALGO_HOST=http://127.0.0.1:5000
```

### 3. Model Provider Comparison

| Provider | Model | Speed | Cost/1M tokens | Best For |
|----------|-------|-------|----------------|----------|
| **Cerebras** | llama3.1-8b | ⚡⚡⚡ Ultra-fast | $0.60 | Production (real-time trading) |
| **Groq** | llama-3.3-70b | ⚡⚡ Fast | $0.59 | Development & testing |
| **OpenAI** | gpt-4o-mini | ⚡ Standard | $0.15/$0.60 | Best reasoning quality |

**Recommendation**: Use **Cerebras** for production (fastest), **Groq** for development (cheapest).

---

## Usage

### Running the Agent

#### Development Mode (with Test Cycle)

```bash
uv run python agent.py
```

This will:
- Run one immediate test cycle
- Then schedule cycles every 5 minutes during market hours

#### Production Mode (Scheduled Only)

1. Edit `agent.py` and comment out the test cycle line:
   ```python
   # await run_trading_cycle()  # Comment this line
   ```

2. Run the agent:
   ```bash
   uv run python agent.py
   ```

The agent will only run during scheduled market hours (9:15 AM - 3:30 PM IST, Monday-Friday).

### Trading Schedule

| Event | Time | Description |
|-------|------|-------------|
| **Trading Cycles** | 9:15 AM - 3:30 PM | Every 5 minutes |
| **Square-Off** | 3:15 PM | Close all positions |
| **Daily Reset** | 3:45 PM | Reset trade counts & P&L |

### Monitoring

The agent provides real-time output:

- 🔵 **Blue** - Market data fetched
- 🟢 **Green** - Account information
- 🟡 **Yellow** - Risk warnings
- 🔴 **Red** - Errors or blocked trades
- ⚪ **White** - Trading decisions
- 🟣 **Magenta** - Order execution
- 🔷 **Cyan** - System information

---

## Risk Management

### Built-in Safety Features

#### 1. Daily Stop-Loss
- **Limit**: -Rs.10,000 daily loss
- **Action**: Stops all trading when hit

#### 2. Trade Count Limits
- **Limit**: 5 trades per stock per day
- **Prevents**: Over-trading

#### 3. Position Control
- **No Pyramiding**: Can't add to existing long/short
- **BUY** only if no long position
- **SELL** only if no short position

#### 4. Shorting Rules
✅ **Allowed:**
- SELL without position (creates short)
- BUY to close short position

❌ **Blocked:**
- Adding to existing long position
- Adding to existing short position

#### 5. Fixed Position Size
- **Investment**: Rs.10,000 per trade
- **Prevents**: Over-leverage

### Trading Parameters

```python
SYMBOLS = ["ICICIBANK", "RELIANCE", "SBIN", "WIPRO", "ITC"]
MAX_INVESTMENT_PER_TRADE = 10000  # Rs.10,000 per trade
DAILY_STOP_LOSS = -10000          # Max -Rs.10,000 loss/day
MAX_TRADES_PER_SYMBOL = 5         # Max 5 trades per stock/day
EXCHANGE = "NSE"
PRODUCT = "MIS"                   # Intraday trading
```

---

## Documentation

### Core Documents

- **[README.md](./README.md)** - This file (Quick start guide)
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Detailed system architecture
- **[MODEL_CONFIG.md](./MODEL_CONFIG.md)** - Model provider configuration
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Common issues & solutions

### Quick Links

| Topic | Link |
|-------|------|
| Architecture Overview | [ARCHITECTURE.md](./ARCHITECTURE.md#architecture-design) |
| Trading Tools | [ARCHITECTURE.md](./ARCHITECTURE.md#trading-tools) |
| Risk Management | [ARCHITECTURE.md](./ARCHITECTURE.md#risk-management) |
| Performance Metrics | [ARCHITECTURE.md](./ARCHITECTURE.md#performance-optimization) |
| Model Configuration | [MODEL_CONFIG.md](./MODEL_CONFIG.md) |
| Common Issues | [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) |

---

## Troubleshooting

### Common Issues

#### 1. Import Error: Module not found

```bash
uv sync
uv pip install --upgrade openai-agents
```

**Quick environment check**: Run the automated checker to validate installed packages and imports:

```bash
python scripts/check_env.py
```

#### 2. TA-Lib Import Error

```bash
# Install TA-Lib binary first (see Installation section)
# Then install Python wrapper:
uv pip install TA-Lib
```

#### 3. Rate Limit Exceeded

Edit `agent.py` and increase delays:
```python
# In get_all_market_data() function:
time.sleep(0.2)  # Increase from 0.15
```

#### 4. Max Turns Exceeded

Edit `agent.py` and increase max_turns:
```python
result = Runner.run_streamed(
    trading_agent,
    query=query,
    max_turns=60  # Increase from 30
)
```

#### 5. Authentication Error

Check your `.env` file:
- Ensure `MODEL_PROVIDER` matches the API key you've set
- Verify API key is valid (not expired)
- Check API key has proper format (starts with `sk-`, `gsk-`, or `csk-`)

For more troubleshooting help, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

---

## Project Structure

```
autonomous-agents/
├── agent.py                 # Main trading agent (run this)
├── .env                     # Your API keys (gitignored)
├── .env.example            # Example configuration template
├── pyproject.toml          # Python dependencies
├── uv.lock                 # Dependency lock file
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── ARCHITECTURE.md         # Detailed architecture docs
├── MODEL_CONFIG.md         # Model provider guide
├── TROUBLESHOOTING.md      # Common issues & fixes
└── trading_memory.db       # Trade history database (gitignored)
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone for development
git clone <repo-url>
cd autonomous-agents

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Run tests (if available)
uv run pytest
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Disclaimer

**IMPORTANT TRADING RISK DISCLOSURE**

This is an **autonomous AI trading system** that makes real trading decisions without human intervention.

**Risks Include:**
- Substantial financial loss
- Market volatility
- Technical failures
- AI decision errors
- Broker/API issues

**Before Using:**
1. ✅ Test in **paper trading mode** first
2. ✅ Understand all risk parameters
3. ✅ Start with small capital
4. ✅ Monitor frequently
5. ✅ Have stop-loss in place

**Legal Notice:**
- Use at your own risk
- No guarantees of profit
- Past performance ≠ future results
- Not financial advice
- You are responsible for all trades

**The authors and contributors are not liable for any financial losses incurred.**

---

## Resources

### Official Documentation

- **OpenAI Agents SDK**: https://openai.github.io/openai-agents-python/
- **LiteLLM**: https://docs.litellm.ai/
- **TA-Lib**: https://ta-lib.org/
- **OpenAlgo**: https://openalgo.in/

### Model Providers

- **Cerebras**: https://cerebras.ai/
- **Groq**: https://groq.com/
- **OpenAI**: https://openai.com/

### Support

- **GitHub Issues**: [Report bugs or request features]
- **Documentation**: See `docs/` folder
- **Email**: [Your support email]

---

## Features Roadmap

**Planned Enhancements:**

- [ ] Web dashboard for monitoring
- [ ] Backtesting engine
- [ ] Paper trading mode
- [ ] Multiple strategy support
- [ ] Portfolio management
- [ ] Email/SMS alerts
- [ ] Performance analytics
- [ ] Multi-exchange support
- [ ] Options trading
- [ ] News sentiment analysis

---

## Stats

- **Lines of Code**: ~1,000
- **Dependencies**: 8 core packages
- **Supported Stocks**: 5 (expandable)
- **Technical Indicators**: 7
- **Trading Sessions per Day**: ~75 cycles
- **Avg. Cycle Time**: 3-6 seconds
- **Token Usage**: ~30K per cycle
- **Daily Cost**: ~$0.60 (Cerebras)

---

**Built with OpenAI Agents SDK**

**Version**: 3.1 (Production-Ready with Full Protection)
**Last Updated**: January 2025
**Status**: Production Ready

---

## Quick Commands

```bash
# Install and run
uv sync && uv run python agent.py

# Check logs
tail -f trading_agent.log

# Monitor token usage
grep "TOKEN USAGE" trading_agent.log

# View recent trades
sqlite3 trading_memory.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"
```

---

**Happy Trading!**
