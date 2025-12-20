"""
OpenAlgo Autonomous AI Trading Agent
Self-learning agent that makes data-driven trading decisions
"""

import io
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Keep oneDNN optimizations but suppress message

# Force UTF-8 encoding for console output (Windows compatibility)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules with fallback chain
function_tool = None
Agent = None
Runner = None
LitellmModel = None

try:
    # Try relative imports first (when running as a package)
    from agent import Agent
    from run import Runner # type: ignore
    
    from tool import function_tool # type: ignore
except ImportError:
    pass

try:
    from extensions.models.litellm_model import LitellmModel # type: ignore
except ImportError:
    try:
        from .extensions.models.litellm_model import LitellmModel # type: ignore
    except ImportError:
        try:
            # Fallback to absolute imports when running as a top-level script
            from agents.agent import Agent
            from agents.run import Runner
            from agents.tool import function_tool
            from agents.extensions.models.litellm_model import LitellmModel
        except ImportError:
            try:
                # Try importing from current directory as standalone modules
                import importlib.util
                tool_spec = importlib.util.spec_from_file_location("tool", str(Path(__file__).parent / "tool.py"))
                if tool_spec and tool_spec.loader:
                    tool_module = importlib.util.module_from_spec(tool_spec)
                    tool_spec.loader.exec_module(tool_module)
                    function_tool = tool_module.function_tool
                else:
                    raise ImportError("Could not load tool module")
            except ImportError as e:
                print(f"Warning: Could not import core modules: {e}")
                raise

# Verify critical imports
if function_tool is None:
    raise ImportError("Failed to import function_tool from tool module")
if Agent is None:
    raise ImportError("Failed to import Agent from agent module")
if Runner is None:
    raise ImportError("Failed to import Runner from run module")
if LitellmModel is None:
    raise ImportError("Failed to import LitellmModel from extensions.models.litellm_model")
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pytz
import talib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from colorama import Fore, Style, init
from dotenv import load_dotenv
from openalgo import api

# Try to import new risk utilities (best-effort)
try:
    import risk
except Exception:
    try:
        from . import risk
    except Exception:
        risk = None

# Initialize colorama for Windows compatibility
init(autoreset=True)

load_dotenv(override=True)

# ============================================================================
# MODEL CONFIGURATION (Model Agnostic - OpenAI, Groq, Cerebras, or Custom)
# ============================================================================

# Choose model provider: "cerebras", "groq", "openai", or "custom"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()

if MODEL_PROVIDER == "cerebras":
    # Cerebras Configuration (ULTRA-FAST inference, excellent for real-time trading)
    # gpt-oss-120b is the fastest Cerebras model (1800+ tokens/sec)
    model_name = os.getenv("CEREBRAS_MODEL", "cerebras/gpt-oss-120b")
    trading_model = LitellmModel(
        model=model_name,
        api_key=os.getenv("CEREBRAS_API_KEY")
    )
    print(f"{Fore.CYAN}[MODEL] Using Cerebras ({model_name}) - ULTRA-FAST (1800+ tokens/sec){Style.RESET_ALL}")

elif MODEL_PROVIDER == "groq":
    # Groq Configuration (Fast inference, cost-effective)
    model_name = os.getenv("GROQ_MODEL", "groq/qwen/qwen3-32b")
    trading_model = LitellmModel(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY")
    )
    print(f"{Fore.CYAN}[MODEL] Using Groq ({model_name}){Style.RESET_ALL}")

elif MODEL_PROVIDER == "custom":
    # Custom Model Configuration (Advanced users)
    model_name = os.getenv("CUSTOM_MODEL", "openai/gpt-4o-mini")
    api_key = os.getenv("CUSTOM_API_KEY", os.getenv("OPENAI_API_KEY"))
    api_base = os.getenv("CUSTOM_API_BASE", None)

    # Set API base via environment variable if provided
    if api_base:
        # Extract provider name from model (e.g., "anthropic" from "anthropic/claude-3")
        provider = model_name.split("/")[0].upper()
        os.environ[f"{provider}_API_BASE"] = api_base

    trading_model = LitellmModel(
        model=model_name,
        api_key=api_key
    )
    print(f"{Fore.CYAN}[MODEL] Using Custom ({model_name}){Style.RESET_ALL}")

else:
    # OpenAI Configuration (Default - High quality, reliable)
    model_name = os.getenv("OPENAI_MODEL", "openai/gpt-5-mini")
    trading_model = LitellmModel(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print(f"{Fore.CYAN}[MODEL] Using OpenAI ({model_name}){Style.RESET_ALL}")

# Initialize OpenAlgo client
client = api(
    api_key=os.getenv("OPENALGO_API_KEY"),
    host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
)

# Trading Universe
SYMBOLS = ["ICICIBANK", "RELIANCE", "SBIN", "WIPRO", "ITC"]
EXCHANGE = "NSE"
PRODUCT = "MIS"
MAX_INVESTMENT_PER_TRADE = 10000
DAILY_STOP_LOSS = -10000
MAX_TRADES_PER_SYMBOL = 5

IST = pytz.timezone('Asia/Kolkata')

# Trading Session Times (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MINUTE = 15
DAILY_RESET_HOUR = 15
DAILY_RESET_MINUTE = 45

# Initialize Memory Session for tracking past trades and decisions
# Memory persists across trading cycles to learn from past performance
# Using SQLite database to store conversation history
trading_session = None  # Will be initialized in async context

# State Management
trade_state = {
    "daily_pnl": 0.0,
    "trade_counts": {symbol: 0 for symbol in SYMBOLS},
    "trade_history": [],
    "active_positions": {},
    "stop_loss_hit": False,
    "squared_off_today": False
}


# ============================================================================
# OPENALGO TOOLS
# ============================================================================

@function_tool
def get_all_market_data() -> Dict[str, Any]:
    """Fetch ALL market data (quotes, depth, history) for ALL symbols with rate limiting.

    Uses threading to fetch all symbols in parallel while respecting 10 req/sec limit.
    Much faster than sequential calls.
    """
    import threading
    import time
    from datetime import datetime, timedelta

    def fetch_symbol_data(symbol: str, results: dict, index: int):
        """Fetch all data for one symbol (runs in separate thread)."""
        try:
            # Rate limiting delay based on symbol index
            time.sleep(index * 0.2)  # Stagger starts by 0.2s

            # Fetch quotes
            quotes_response = client.quotes(symbol=symbol, exchange=EXCHANGE)
            quotes_data = quotes_response.get("data", {}) if quotes_response.get("status") == "success" else {}

            time.sleep(0.15)  # Rate limit delay

            # Fetch depth
            depth_response = client.depth(symbol=symbol, exchange=EXCHANGE)
            depth_data = depth_response.get("data", {}) if depth_response.get("status") == "success" else {}

            # Calculate bid/ask ratio
            total_bid = sum([b["quantity"] for b in depth_data.get("bids", [])])
            total_ask = sum([a["quantity"] for a in depth_data.get("asks", [])])
            bid_ask_ratio = round(total_bid / total_ask, 2) if total_ask > 0 else 0

            time.sleep(0.15)  # Rate limit delay

            # Fetch historical data
            end_date = datetime.now(IST).strftime("%Y-%m-%d")
            start_date = (datetime.now(IST) - timedelta(days=3)).strftime("%Y-%m-%d")
            history_response = client.history(symbol=symbol, exchange=EXCHANGE, interval="5m", start_date=start_date, end_date=end_date)

            # Calculate technical indicators
            if isinstance(history_response, dict) and history_response.get("status") == "error":
                rsi_current = 50.0
                macd_trend = "neutral"
                ema_trend = "neutral"
            else:
                close_prices = history_response['close'].values

                rsi = talib.RSI(close_prices, timeperiod=14)
                macd, macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                ema_20 = talib.EMA(close_prices, timeperiod=20)
                ema_50 = talib.EMA(close_prices, timeperiod=50)

                rsi_current = round(float(rsi[~np.isnan(rsi)][-1]), 2) if len(rsi[~np.isnan(rsi)]) > 0 else 50.0
                macd_current = float(macd[~np.isnan(macd)][-1]) if len(macd[~np.isnan(macd)]) > 0 else 0
                macd_signal_current = float(macd_signal[~np.isnan(macd_signal)][-1]) if len(macd_signal[~np.isnan(macd_signal)]) > 0 else 0
                macd_trend = "bullish" if macd_current > macd_signal_current else "bearish"

                ema_20_current = float(ema_20[~np.isnan(ema_20)][-1]) if len(ema_20[~np.isnan(ema_20)]) > 0 else 0
                ema_50_current = float(ema_50[~np.isnan(ema_50)][-1]) if len(ema_50[~np.isnan(ema_50)]) > 0 else 0
                ema_trend = "bullish" if ema_20_current > ema_50_current else "bearish"

            results[symbol] = {
                "symbol": symbol,
                "ltp": quotes_data.get("ltp", 0),
                "volume": quotes_data.get("volume", 0),
                "bid_ask_ratio": bid_ask_ratio,
                "rsi": rsi_current,
                "macd_trend": macd_trend,
                "ema_trend": ema_trend
            }
        except Exception as e:
            results[symbol] = {"symbol": symbol, "error": str(e)}

    # Fetch all symbols using threads (no asyncio event loop conflict)
    print(f"{Fore.BLUE}[BULK FETCH] Fetching ALL market data (5 symbols in parallel)...{Style.RESET_ALL}", flush=True)
    start_time = time.time()

    results = {}
    threads = []

    # Create and start threads
    for i, symbol in enumerate(SYMBOLS):
        thread = threading.Thread(target=fetch_symbol_data, args=(symbol, results, i))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    elapsed = time.time() - start_time
    print(f"{Fore.GREEN}[BULK FETCH] ✓ All data fetched in {elapsed:.1f}s (15 API calls){Style.RESET_ALL}", flush=True)

    return {"status": "success", "data": results, "elapsed_seconds": round(elapsed, 1)}


@function_tool
def get_market_quotes(symbol: str) -> Dict[str, Any]:
    """Get current market quotes for a symbol."""
    try:
        print(f"{Fore.BLUE}[FETCHING] Market quotes for {symbol}...{Style.RESET_ALL}", flush=True)
        response = client.quotes(symbol=symbol, exchange=EXCHANGE)
        if response.get("status") == "success":
            data = response["data"]
            print(f"{Fore.BLUE}[DATA] {symbol} Quote: LTP={data['ltp']}, Volume={data['volume']:,}{Style.RESET_ALL}", flush=True)
            return {
                "symbol": symbol,
                "ltp": data["ltp"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "volume": data["volume"],
                "prev_close": data["prev_close"]
            }
        print(f"{Fore.RED}[ERROR] Failed to fetch quotes for {symbol}{Style.RESET_ALL}", flush=True)
        return {"error": response.get("message", "Failed to fetch quotes")}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"error": str(e)}


@function_tool
def get_market_depth(symbol: str) -> Dict[str, Any]:
    """Get market depth (bid/ask levels) for a symbol."""
    try:
        print(f"{Fore.BLUE}[FETCHING] Market depth for {symbol}...{Style.RESET_ALL}", flush=True)
        response = client.depth(symbol=symbol, exchange=EXCHANGE)
        if response.get("status") == "success":
            data = response["data"]
            total_bid = sum([b["quantity"] for b in data["bids"]])
            total_ask = sum([a["quantity"] for a in data["asks"]])
            bid_ask_ratio = total_bid / total_ask if total_ask > 0 else 0

            print(f"{Fore.BLUE}[DATA] {symbol} Depth: Bids={total_bid:,}, Asks={total_ask:,}, Ratio={bid_ask_ratio:.2f}{Style.RESET_ALL}", flush=True)

            return {
                "symbol": symbol,
                "total_bid_qty": total_bid,
                "total_ask_qty": total_ask,
                "bid_ask_ratio": round(bid_ask_ratio, 2),
                "best_bid": data["bids"][0]["price"] if data["bids"] else 0,
                "best_ask": data["asks"][0]["price"] if data["asks"] else 0
            }
        print(f"{Fore.RED}[ERROR] Failed to fetch depth for {symbol}{Style.RESET_ALL}", flush=True)
        return {"error": response.get("message", "Failed to fetch depth")}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"error": str(e)}


@function_tool
def get_historical_data(symbol: str, lookback_bars: int = 5) -> Dict[str, Any]:
    """Get 5-minute historical data and calculate TA-Lib technical indicators for last N bars.

    Args:
        symbol: Stock symbol to fetch
        lookback_bars: Number of recent bars to return (default: 5, min: 1, max: 20)
    """
    try:
        # Validate lookback_bars (reduced max for speed)
        lookback_bars = max(1, min(20, lookback_bars))

        print(f"{Fore.BLUE}[FETCHING] Historical data for {symbol} (last {lookback_bars} bars)...{Style.RESET_ALL}", flush=True)
        end_date = datetime.now(IST).strftime("%Y-%m-%d")
        start_date = (datetime.now(IST) - timedelta(days=3)).strftime("%Y-%m-%d")

        response = client.history(
            symbol=symbol,
            exchange=EXCHANGE,
            interval="5m",
            start_date=start_date,
            end_date=end_date
        )

        if isinstance(response, dict) and response.get("status") == "error":
            print(f"{Fore.RED}[ERROR] Failed to fetch history for {symbol}{Style.RESET_ALL}", flush=True)
            return {"error": response.get("message")}

        # Extract price data as numpy arrays
        close_prices = response['close'].values
        high_prices = response['high'].values
        low_prices = response['low'].values

        # Calculate Technical Indicators using TA-Lib (full series first)

        # 1. RSI (Relative Strength Index)
        rsi = talib.RSI(close_prices, timeperiod=14)

        # 2. MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

        # 3. Bollinger Bands
        upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=20)

        # 4. EMA (Exponential Moving Averages)
        ema_20 = talib.EMA(close_prices, timeperiod=20)
        ema_50 = talib.EMA(close_prices, timeperiod=50)

        # 5. ATR (Average True Range) - Volatility
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        # 6. Stochastic Oscillator
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)

        # 7. ADX (Average Directional Index) - Trend Strength
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)

        # Extract last N bars of data
        def get_last_n_bars(arr, n):
            """Get last N valid (non-NaN) values from array."""
            valid_arr = arr[~np.isnan(arr)]
            if len(valid_arr) == 0:
                return []
            return [round(float(x), 2) for x in valid_arr[-n:]]

        # Get last N bars for each indicator
        rsi_bars = get_last_n_bars(rsi, lookback_bars)
        macd_bars = get_last_n_bars(macd, lookback_bars)
        macd_signal_bars = get_last_n_bars(macd_signal, lookback_bars)
        macd_hist_bars = get_last_n_bars(macd_hist, lookback_bars)
        bb_upper_bars = get_last_n_bars(upper_band, lookback_bars)
        bb_middle_bars = get_last_n_bars(middle_band, lookback_bars)
        bb_lower_bars = get_last_n_bars(lower_band, lookback_bars)
        ema_20_bars = get_last_n_bars(ema_20, lookback_bars)
        ema_50_bars = get_last_n_bars(ema_50, lookback_bars)
        atr_bars = get_last_n_bars(atr, lookback_bars)
        stoch_bars = get_last_n_bars(slowk, lookback_bars)
        adx_bars = get_last_n_bars(adx, lookback_bars)
        close_bars = [round(float(x), 2) for x in close_prices[-lookback_bars:]]

        # Current values (most recent bar)
        current_rsi = rsi_bars[-1] if rsi_bars else 50.0
        current_macd = macd_bars[-1] if macd_bars else 0.0
        current_macd_signal = macd_signal_bars[-1] if macd_signal_bars else 0.0
        current_price = close_bars[-1] if close_bars else 0.0
        current_ema_20 = ema_20_bars[-1] if ema_20_bars else current_price
        current_ema_50 = ema_50_bars[-1] if ema_50_bars else current_price
        current_atr = atr_bars[-1] if atr_bars else 0.0
        current_stoch = stoch_bars[-1] if stoch_bars else 50.0
        current_adx = adx_bars[-1] if adx_bars else 0.0

        # Signal interpretations
        rsi_signal = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
        macd_trend = "bullish" if current_macd > current_macd_signal else "bearish"
        bb_upper = bb_upper_bars[-1] if bb_upper_bars else current_price
        bb_lower = bb_lower_bars[-1] if bb_lower_bars else current_price
        bb_position = "overbought" if current_price > bb_upper else "oversold" if current_price < bb_lower else "neutral"
        ema_trend = "bullish" if current_ema_20 > current_ema_50 else "bearish"
        stoch_signal = "overbought" if current_stoch > 80 else "oversold" if current_stoch < 20 else "neutral"
        trend_strength = "strong" if current_adx > 25 else "weak"

        # Recent volatility
        recent_data = response.tail(12)  # Last hour
        volatility = recent_data['close'].std()
        avg_volume = recent_data['volume'].mean()

        print(f"{Fore.BLUE}[DATA] {symbol} TA Indicators (last {lookback_bars} bars): RSI={current_rsi}, MACD={macd_trend}, BB={bb_position}, EMA={ema_trend}, ADX={current_adx}{Style.RESET_ALL}", flush=True)

        return {
            "symbol": symbol,
            "lookback_bars": lookback_bars,

            # Current values (most recent bar)
            "current": {
                "price": current_price,
                "rsi": current_rsi,
                "rsi_signal": rsi_signal,
                "macd": current_macd,
                "macd_signal": current_macd_signal,
                "macd_trend": macd_trend,
                "bb_position": bb_position,
                "ema_20": current_ema_20,
                "ema_50": current_ema_50,
                "ema_trend": ema_trend,
                "atr": current_atr,
                "stoch": current_stoch,
                "stoch_signal": stoch_signal,
                "adx": current_adx,
                "trend_strength": trend_strength,
            },

            # Historical bars (time series data)
            "bars": {
                "close": close_bars,
                "rsi": rsi_bars,
                "macd": macd_bars,
                "macd_signal": macd_signal_bars,
                "macd_hist": macd_hist_bars,
                "bb_upper": bb_upper_bars,
                "bb_middle": bb_middle_bars,
                "bb_lower": bb_lower_bars,
                "ema_20": ema_20_bars,
                "ema_50": ema_50_bars,
                "atr": atr_bars,
                "stoch": stoch_bars,
                "adx": adx_bars,
            },

            # Additional metrics
            "volatility_1h": round(volatility, 2),
            "avg_volume_1h": int(avg_volume)
        }
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"error": str(e)}


@function_tool
def check_all_risk_constraints(trades: str) -> Dict[str, Any]:
    """Validate multiple trades at once (bulk risk check).

    Args:
        trades: JSON string with format: [{"symbol": "ICICIBANK", "action": "BUY"}, ...]

    Returns:
        Risk check results for all trades
    """
    import json

    try:
        trades_list = json.loads(trades)

        if not isinstance(trades_list, list):
            return {"success": False, "error": "trades must be a JSON array"}

        print(f"{Fore.YELLOW}[BULK RISK CHECK] Checking {len(trades_list)} trades...{Style.RESET_ALL}", flush=True)

        results = []
        for trade in trades_list:
            symbol = trade["symbol"]
            action = trade["action"]
            risk_result = check_risk_constraints(symbol, action)
            results.append({
                "symbol": symbol,
                "action": action,
                "allowed": risk_result.get("allowed", False),
                "reason": risk_result.get("reason", "")
            })

        allowed_count = sum(1 for r in results if r["allowed"])
        print(f"{Fore.GREEN}[BULK RISK CHECK] ✓ {allowed_count}/{len(trades_list)} trades allowed{Style.RESET_ALL}", flush=True)

        return {"success": True, "results": results}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def calculate_all_position_sizes(positions: str) -> Dict[str, Any]:
    """Calculate position sizes for multiple symbols at once (bulk calculation).

    Args:
        positions: JSON string with format: [{"symbol": "ICICIBANK", "ltp": 1350.0}, ...]

    Returns:
        Position size calculations for all symbols
    """
    import json

    try:
        positions_list = json.loads(positions)

        if not isinstance(positions_list, list):
            return {"success": False, "error": "positions must be a JSON array"}

        print(f"{Fore.CYAN}[BULK POSITION CALC] Calculating sizes for {len(positions_list)} symbols...{Style.RESET_ALL}", flush=True)

        results = []
        for pos in positions_list:
            symbol = pos["symbol"]
            ltp = pos["ltp"]
            max_investment = pos.get("max_investment", 10000.0)

            quantity = int(max_investment / ltp) if ltp > 0 else 0
            actual_investment = quantity * ltp

            results.append({
                "symbol": symbol,
                "ltp": round(ltp, 2),
                "quantity": quantity,
                "actual_investment": round(actual_investment, 2)
            })

            print(
                f"{Fore.CYAN}  {symbol}: Qty={quantity}, Investment=Rs.{actual_investment:.2f}{Style.RESET_ALL}",
                flush=True
            )

        return {"success": True, "results": results}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def calculate_position_size(symbol: str, ltp: float, max_investment: float = 10000.0) -> Dict[str, Any]:
    """Calculate the correct quantity of shares to buy based on LTP and max investment per trade.

    This tool MUST be used before placing any order to ensure correct position sizing.

    Behavior:
      - If environment variable `RISK_PER_TRADE_PERCENT` > 0, the function will compute an ATR-based stop-loss
        and size the position using percent-of-equity risk sizing (preferred).
      - Otherwise it falls back to simple `int(max_investment / ltp)` behavior to remain backwards compatible.

    Args:
        symbol: Stock symbol (e.g., "ICICIBANK", "WIPRO")
        ltp: Last Traded Price (current market price)
        max_investment: Maximum investment per trade in Rs. (default: 10,000)

    Returns:
        Dictionary with calculated quantity and investment details
    """
    try:
        if ltp <= 0:
            return {
                "error": f"Invalid LTP {ltp}. LTP must be positive.",
                "symbol": symbol,
                "quantity": 0
            }

        # If risk-based sizing is enabled, compute ATR stop and size by risk-percent
        risk_percent_env = float(os.getenv("RISK_PER_TRADE_PERCENT", "0"))
        # Only use risk-based sizing if module is available
        if risk_percent_env > 0 and (globals().get("risk") is not None):
            # Get account snapshot
            snapshot = get_account_snapshot()
            available_cash = float(snapshot.get("available_cash", 0))

            # Try to fetch history for ATR calculation (best-effort)
            try:
                end_date = datetime.now(IST).strftime("%Y-%m-%d")
                start_date = (datetime.now(IST) - timedelta(days=3)).strftime("%Y-%m-%d")
                history = client.history(symbol=symbol, exchange=EXCHANGE, interval="5m", start_date=start_date, end_date=end_date)
                if isinstance(history, dict) and history.get("status") == "error":
                    raise ValueError("history fetch error")

                highs = history["high"].values
                lows = history["low"].values
                closes = history["close"].values

                # Compute stop using ATR
                stop_price_buy = None
                try:
                    stop_price_buy = risk.compute_atr_stoploss(highs, lows, closes, ltp, action="BUY")
                except Exception:
                    stop_price_buy = round(ltp * 0.98, 2)

                # Compute quantity by risk
                sizing = risk.compute_position_size_by_risk(ltp=ltp, stop_price=stop_price_buy, account_equity=available_cash, risk_percent=risk_percent_env, max_investment=max_investment)

                if sizing.get("quantity", 0) == 0:
                    return {
                        "error": "Risk-based sizing resulted in zero quantity",
                        "symbol": symbol,
                        "quantity": 0
                    }

                quantity = int(sizing["quantity"])
                actual_investment = quantity * ltp

                print(
                    f"{Fore.CYAN}{AGENT_ICONS['calculator']} [RISK SIZER] {symbol}: LTP={ltp}, Stop={stop_price_buy}, Quantity={quantity}, Investment=Rs.{actual_investment:.2f}{Style.RESET_ALL}",
                    flush=True
                )

                return {
                    "symbol": symbol,
                    "ltp": round(ltp, 2),
                    "quantity": quantity,
                    "max_investment": max_investment,
                    "actual_investment": round(actual_investment, 2),
                    "stop_loss": stop_price_buy,
                    "risk_amount": round(sizing.get("risk_amount", 0), 2),
                    "per_share_risk": round(sizing.get("per_share_risk", 0), 2),
                    "formula": f"risk_percent={risk_percent_env}, risk_amount={round(sizing.get('risk_amount',0),2)}",
                    "success": True
                }

            except Exception as e:
                print(f"{Fore.YELLOW}[RISK] Could not fetch history/compute ATR: {str(e)}. Falling back to simple sizing.{Style.RESET_ALL}", flush=True)

        # Fallback: simple max_investment / ltp
        quantity = int(max_investment / ltp)

        if quantity == 0:
            return {
                "error": f"LTP {ltp} too high for max_investment {max_investment}. Quantity would be 0.",
                "symbol": symbol,
                "quantity": 0,
                "ltp": ltp
            }

        # Calculate actual investment
        actual_investment = quantity * ltp

        print(
            f"{Fore.CYAN}{AGENT_ICONS['calculator']} [POSITION CALCULATOR] {symbol}: LTP={ltp}, Quantity={quantity}, "
            f"Investment=Rs.{actual_investment:.2f}{Style.RESET_ALL}",
            flush=True
        )

        return {
            "symbol": symbol,
            "ltp": round(ltp, 2),
            "quantity": quantity,
            "max_investment": max_investment,
            "actual_investment": round(actual_investment, 2),
            "formula": f"int({max_investment} / {ltp}) = {quantity}",
            "success": True
        }

    except Exception as e:
        print(f"{Fore.RED}[ERROR] Position calculation failed: {str(e)}{Style.RESET_ALL}", flush=True)
        return {
            "error": str(e),
            "symbol": symbol,
            "quantity": 0
        }


@function_tool
def get_account_snapshot() -> Dict[str, Any]:
    """Get current account funds and margin."""
    try:
        print(f"{Fore.GREEN}[FETCHING] Account funds...{Style.RESET_ALL}", flush=True)
        response = client.funds()
        if response.get("status") == "success":
            data = response["data"]
            cash = float(data.get("availablecash", 0))
            print(f"{Fore.GREEN}[DATA] Available Cash: Rs.{cash:,.2f}{Style.RESET_ALL}", flush=True)
            return {
                "available_cash": cash,
                "m2m_unrealized": float(data.get("m2munrealized", 0)),
                "m2m_realized": float(data.get("m2mrealized", 0))
            }
        print(f"{Fore.RED}[ERROR] Failed to fetch funds{Style.RESET_ALL}", flush=True)
        return {"error": "Failed to fetch funds"}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"error": str(e)}


def update_daily_pnl():
    """Update daily P&L from position book."""
    try:
        response = client.positionbook()
        if response.get("status") == "success":
            positions = response.get("data", [])
            total_pnl = sum(float(pos.get("pnl", 0)) for pos in positions)
            trade_state["daily_pnl"] = total_pnl
            return total_pnl
        return 0.0
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to update P&L: {str(e)}{Style.RESET_ALL}", flush=True)
        return 0.0


@function_tool
def get_current_positions() -> Dict[str, Any]:
    """Get all open positions."""
    try:
        print(f"{Fore.GREEN}[FETCHING] Open positions...{Style.RESET_ALL}", flush=True)
        response = client.positionbook()
        if response.get("status") == "success":
            positions = {}
            for pos in response["data"]:
                if int(pos["quantity"]) != 0:
                    positions[pos["symbol"]] = {
                        "quantity": int(pos["quantity"]),
                        "avg_price": float(pos["average_price"]),
                        "ltp": float(pos["ltp"]),
                        "pnl": float(pos["pnl"])
                    }
            print(f"{Fore.GREEN}[DATA] Open Positions: {len(positions)}{Style.RESET_ALL}", flush=True)
            return {"positions": positions, "count": len(positions)}
        print(f"{Fore.RED}[ERROR] Failed to fetch positions{Style.RESET_ALL}", flush=True)
        return {"error": "Failed to fetch positions"}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"error": str(e)}


@function_tool
def analyze_past_trades(symbol: str) -> Dict[str, Any]:
    """Analyze historical trade performance for a symbol."""
    symbol_trades = [t for t in trade_state["trade_history"] if t["symbol"] == symbol]

    if len(symbol_trades) < 3:
        return {
            "symbol": symbol,
            "confidence": "low",
            "reason": "Insufficient trade history (<3 trades)",
            "win_rate": 0,
            "avg_profit": 0
        }

    wins = [t for t in symbol_trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(symbol_trades) if symbol_trades else 0
    avg_profit = sum([t.get("pnl", 0) for t in symbol_trades]) / len(symbol_trades)

    # Find patterns
    recent_3 = symbol_trades[-3:]
    recent_success = sum([1 for t in recent_3 if t.get("pnl", 0) > 0])

    confidence = "high" if win_rate > 0.6 else "medium" if win_rate > 0.4 else "low"

    return {
        "symbol": symbol,
        "total_trades": len(symbol_trades),
        "win_rate": round(win_rate, 2),
        "avg_profit": round(avg_profit, 2),
        "recent_success": recent_success,
        "confidence": confidence,
        "reason": f"Win rate {win_rate:.0%}, avg profit ₹{avg_profit:.2f}, recent 3 trades: {recent_success}/3 wins"
    }


@function_tool
def check_risk_constraints(symbol: str, action: str) -> Dict[str, Any]:
    """Validate trade against all risk management rules.

    SHORTING ALLOWED:
    - Can SELL without owning (creates short position)
    - Can BUY to cover short position
    - Cannot add to existing long (must close first)
    - Cannot add to existing short (must close first)
    """
    print(f"{Fore.YELLOW}[RISK CHECK] Validating {action} for {symbol}...{Style.RESET_ALL}", flush=True)

    # Daily stop-loss check
    if trade_state["stop_loss_hit"]:
        print(f"{Fore.RED}[RISK BLOCKED] Daily loss limit reached{Style.RESET_ALL}", flush=True)
        return {
            "allowed": False,
            "reason": "Daily loss limit reached — no new trades"
        }

    if trade_state["daily_pnl"] <= DAILY_STOP_LOSS:
        trade_state["stop_loss_hit"] = True
        print(f"{Fore.RED}[RISK BLOCKED] Daily stop-loss hit: Rs.{trade_state['daily_pnl']:.2f}{Style.RESET_ALL}", flush=True)
        return {
            "allowed": False,
            "reason": f"Daily stop-loss hit (Rs.{trade_state['daily_pnl']:.2f})"
        }

    # Per-symbol trade cap
    if trade_state["trade_counts"][symbol] >= MAX_TRADES_PER_SYMBOL:
        print(f"{Fore.RED}[RISK BLOCKED] Max trades reached for {symbol}{Style.RESET_ALL}", flush=True)
        return {
            "allowed": False,
            "reason": f"Max trades reached for {symbol} ({MAX_TRADES_PER_SYMBOL}/day)"
        }

    # Get current positions from OpenAlgo
    # Shorting is ALLOWED - can SELL without BUY (creates short position)
    try:
        print(f"{Fore.YELLOW}[CHECKING] Current positions for {symbol}...{Style.RESET_ALL}", flush=True)
        response = client.positionbook()
        has_position = False
        position_qty = 0

        if response.get("status") == "success":
            for pos in response["data"]:
                if pos["symbol"] == symbol and int(pos["quantity"]) != 0:
                    has_position = True
                    position_qty = int(pos["quantity"])
                    break

        # Check if trying to add to existing position (not allowed)
        if action == "BUY" and has_position and position_qty > 0:
            # Already long, cannot add more longs
            print(f"{Fore.RED}[RISK BLOCKED] Cannot BUY {symbol} - Already have long position (Qty: {position_qty}){Style.RESET_ALL}", flush=True)
            return {
                "allowed": False,
                "reason": f"Already have long position in {symbol} (Qty: {position_qty}). Must close first."
            }
        elif action == "SELL" and has_position and position_qty < 0:
            # Already short, cannot add more shorts
            print(f"{Fore.RED}[RISK BLOCKED] Cannot SELL {symbol} - Already have short position (Qty: {position_qty}){Style.RESET_ALL}", flush=True)
            return {
                "allowed": False,
                "reason": f"Already have short position in {symbol} (Qty: {position_qty}). Must close first."
            }

        # Shorting allowed: SELL without position creates short position
        # Covering allowed: BUY to close short position
        # All other cases allowed

        # Additional exposure checks (per-symbol exposure and portfolio exposure)
        if globals().get("risk") is not None:
            try:
                # Build a current positions map for exposure checks
                current_positions = {}
                for pos in response.get("data", []):
                    if int(pos.get("quantity", 0)) != 0:
                        current_positions[pos["symbol"]] = {"quantity": int(pos.get("quantity", 0)), "ltp": float(pos.get("ltp", 0)), "avg_price": float(pos.get("average_price", 0))}

                # Fetch LTP for symbol to estimate proposed exposure
                quotes = client.quotes(symbol=symbol, exchange=EXCHANGE)
                ltp_val = float(quotes.get("data", {}).get("ltp", 0)) if quotes.get("status") == "success" else 0
                if ltp_val <= 0:
                    ltp_val = 1.0  # guard

                # Proposed max quantity (conservative): based on max investment per trade
                proposed_qty = int(MAX_INVESTMENT_PER_TRADE / ltp_val)

                exposure_check = risk.enforce_exposure_limits(current_positions=current_positions, symbol=symbol, proposed_quantity=proposed_qty, ltp=ltp_val, max_per_symbol_investment=MAX_INVESTMENT_PER_TRADE)
                if not exposure_check.get("allowed", True):
                    print(f"{Fore.RED}[RISK BLOCKED] {exposure_check.get('reason')}{Style.RESET_ALL}", flush=True)
                    return {"allowed": False, "reason": exposure_check.get("reason")}

                # Portfolio-level exposure: ensure total invested doesn't exceed a percent of account equity
                snapshot = get_account_snapshot()
                equity = float(snapshot.get("available_cash", 0)) + float(snapshot.get("m2m_unrealized", 0))
                if equity > 0:
                    current_total = sum(int(p.get("quantity",0)) * float(p.get("ltp",0)) for p in current_positions.values())
                    if (current_total + (proposed_qty * ltp_val)) > (equity * float(os.getenv("MAX_PORTFOLIO_EXPOSURE_PERCENT", "0.2"))):
                        reason = "Portfolio exposure limit exceeded"
                        print(f"{Fore.RED}[RISK BLOCKED] {reason}{Style.RESET_ALL}", flush=True)
                        return {"allowed": False, "reason": reason}

                # Trailing stops: compute suggested trailing stop (no automatic order placement yet)
                try:
                    trail_enabled = os.getenv("TRAILING_STOP_ENABLED", "false").lower() in ("1", "true", "yes")
                    trail_pct = float(os.getenv("TRAILING_STOP_PERCENT", "0.03"))
                    if trail_enabled:
                        # If we have an open position, compute trailing stop based on peak price (best-effort fetch)
                        pos = current_positions.get(symbol)
                        if pos:
                            entry = float(pos.get("avg_price", ltp_val))
                            # For demonstration, use latest LTP as peak if no history present
                            peak = max(ltp_val, entry)
                            candidate_trail = risk.compute_trailing_stop(entry_price=entry, peak_price=peak, trail_percent=trail_pct, action=action)
                            # Attach suggested trailing stop to response via a debug print
                            print(f"{Fore.YELLOW}[TRAILING STOP] Suggested for {symbol}: {candidate_trail}{Style.RESET_ALL}", flush=True)

                except Exception as e:
                    print(f"{Fore.YELLOW}[RISK] Trailing stop check failed: {str(e)} (continuing){Style.RESET_ALL}", flush=True)

            except Exception as e:
                print(f"{Fore.YELLOW}[RISK] Exposure checks failed: {str(e)} — continuing with other checks{Style.RESET_ALL}", flush=True)

    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to check positions: {str(e)}{Style.RESET_ALL}", flush=True)

    # Time check (no trades after 3:15 PM)
    now = datetime.now(IST)
    if now.hour >= 15 and now.minute >= 15:
        print(f"{Fore.RED}[RISK BLOCKED] Market closing time{Style.RESET_ALL}", flush=True)
        return {
            "allowed": False,
            "reason": "Market closing time — no new trades after 3:15 PM"
        }

    print(f"{Fore.YELLOW}[RISK PASSED] {action} allowed for {symbol}{Style.RESET_ALL}", flush=True)
    return {"allowed": True, "reason": "All risk checks passed"}


@function_tool
def place_bulk_orders(orders: str) -> Dict[str, Any]:
    """Place multiple market orders at once (batch processing).

    Args:
        orders: JSON string with format: [{"symbol": "ICICIBANK", "action": "BUY", "quantity": 7, "reason": "bullish"}, ...]

    Returns:
        Results for all orders placed
    """
    import json
    import threading

    try:
        # Parse orders JSON
        orders_list = json.loads(orders)

        if not isinstance(orders_list, list):
            return {"success": False, "error": "orders must be a JSON array"}

        print(f"{Fore.MAGENTA}[BULK ORDER] Placing {len(orders_list)} orders in parallel...{Style.RESET_ALL}", flush=True)

        def place_single_order(order_data: dict, results: dict, index: int):
            """Place one order in a thread."""
            try:
                symbol = order_data["symbol"]
                action = order_data["action"]
                quantity = order_data["quantity"]
                reason = order_data.get("reason", "bulk order")

                if quantity <= 0:
                    results[index] = {"symbol": symbol, "success": False, "error": "Invalid quantity"}
                    return

                # Place market order
                response = client.placeorder(
                    strategy="AI Agent",
                    symbol=symbol,
                    action=action,
                    exchange=EXCHANGE,
                    price_type="MARKET",
                    product=PRODUCT,
                    quantity=quantity
                )

                if response.get("status") != "success":
                    print(f"{Fore.RED}[ORDER FAILED] {symbol} {action}: {response.get('message')}{Style.RESET_ALL}", flush=True)
                    results[index] = {"symbol": symbol, "success": False, "error": response.get("message")}
                    return

                order_id = response["orderid"]
                print(f"{Fore.MAGENTA}[ORDER {index+1}] {symbol} {action} {quantity} → Order #{order_id} ({reason}){Style.RESET_ALL}", flush=True)

                # Update state
                trade_state["trade_counts"][symbol] += 1
                trade_state["trade_history"].append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_id": order_id,
                    "timestamp": datetime.now(IST).isoformat(),
                    "reason": reason,
                    "status": "completed"
                })

                results[index] = {
                    "symbol": symbol,
                    "success": True,
                    "order_id": order_id,
                    "action": action,
                    "quantity": quantity,
                    "reason": reason
                }

            except Exception as e:
                results[index] = {"symbol": order_data.get("symbol", "unknown"), "success": False, "error": str(e)}

        # Place orders with rate limiting: 0.5s delay between every 2 orders
        results = {}
        threads = []

        for i, order_data in enumerate(orders_list):
            thread = threading.Thread(target=place_single_order, args=(order_data, results, i))
            thread.start()
            threads.append(thread)

            # Add 0.5s delay after every 2 orders (to respect rate limits)
            if (i + 1) % 2 == 0 and (i + 1) < len(orders_list):
                import time
                time.sleep(0.5)

        # Wait for all orders to complete
        for thread in threads:
            thread.join()

        # Format results
        success_count = sum(1 for r in results.values() if r.get("success"))
        print(f"{Fore.GREEN}[BULK ORDER] ✓ {success_count}/{len(orders_list)} orders placed successfully{Style.RESET_ALL}", flush=True)

        return {
            "success": True,
            "total_orders": len(orders_list),
            "successful": success_count,
            "results": [results[i] for i in sorted(results.keys())]
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def place_market_order(symbol: str, action: str, quantity: int, reason: str) -> Dict[str, Any]:
    """Place a MARKET order for immediate execution.

    Market orders execute immediately at the current market price.
    No need for price parameter or status checks - fills instantly.

    QUANTITY VALIDATION: Automatically validates quantity.
    """
    try:
        if quantity <= 0:
            return {"success": False, "error": f"Invalid quantity: {quantity}"}

        # Place market order
        response = client.placeorder(
            strategy="AI Agent",
            symbol=symbol,
            action=action,
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=quantity
        )

        if response.get("status") != "success":
            print(f"{Fore.RED}[ORDER FAILED] {symbol} {action}: {response.get('message')}{Style.RESET_ALL}", flush=True)
            return {"success": False, "error": response.get("message")}

        order_id = response["orderid"]
        print(f"{Fore.MAGENTA}[ORDER PLACED] {symbol} {action} {quantity} → Order #{order_id} ({reason}){Style.RESET_ALL}", flush=True)

        # Update state (market orders fill immediately)
        trade_state["trade_counts"][symbol] += 1
        trade_state["trade_history"].append({
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_id": order_id,
            "timestamp": datetime.now(IST).isoformat(),
            "reason": reason,
            "status": "completed"
        })

        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "reason": reason,
            "status": "completed"
        }

    except Exception as e:
        print(f"{Fore.RED}[ERROR] {str(e)}{Style.RESET_ALL}", flush=True)
        return {"success": False, "error": str(e)}


# Helper functions for market closing (not tools, directly callable)
def _square_off_all_positions_direct():
    """Close all open positions at 3:20 PM by placing opposite orders."""
    try:
        print(f"\n{Fore.YELLOW}[SQUARE OFF] Checking open positions to close...{Style.RESET_ALL}", flush=True)

        # Get current positions
        positions_response = client.positionbook()

        if positions_response.get("status") != "success":
            print(
                f"{Fore.RED}[SQUARE OFF] Failed to get positions: {positions_response}"
                f"{Style.RESET_ALL}",
                flush=True
            )
            return {"success": False, "error": "Failed to get positions"}

        positions = positions_response.get("data", [])
        print(
            f"{Fore.CYAN}[SQUARE OFF] Found {len(positions)} positions{Style.RESET_ALL}",
            flush=True
        )

        closed_positions = []
        failed_positions = []

        for position in positions:
            symbol = position.get("symbol")
            quantity = int(float(position.get("quantity", 0)))
            exchange = position.get("exchange", EXCHANGE)
            product = position.get("product", PRODUCT)

            # Skip if no position (quantity = 0)
            if quantity == 0:
                print(
                    f"{Fore.CYAN}[SQUARE OFF] {symbol}: No position to close (qty=0)"
                    f"{Style.RESET_ALL}",
                    flush=True
                )
                continue

            # Determine action (opposite of current position)
            if quantity > 0:
                # Long position -> SELL to close
                action = "SELL"
                close_qty = quantity
            else:
                # Short position -> BUY to close
                action = "BUY"
                close_qty = abs(quantity)

            print(
                f"{Fore.YELLOW}[SQUARE OFF] {symbol}: Closing {quantity} qty with {action} {close_qty}"
                f"{Style.RESET_ALL}",
                flush=True
            )

            try:
                # Place market order to close position
                order_response = client.placeorder(
                    strategy="AI Agent",
                    symbol=symbol,
                    action=action,
                    exchange=exchange,
                    price_type="MARKET",
                    product=product,
                    quantity=close_qty
                )

                if order_response.get("status") == "success":
                    order_id = order_response.get("orderid")
                    print(
                        f"{Fore.GREEN}[SQUARE OFF] {symbol}: Order #{order_id} placed to close position{Style.RESET_ALL}",
                        flush=True
                    )
                    closed_positions.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "action": action,
                        "order_id": order_id
                    })
                else:
                    print(
                        f"{Fore.RED}[SQUARE OFF] {symbol}: Failed to place order - {order_response}"
                        f"{Style.RESET_ALL}",
                        flush=True
                    )
                    failed_positions.append({"symbol": symbol, "error": order_response})

            except Exception as e:
                print(
                    f"{Fore.RED}[SQUARE OFF] {symbol}: Exception - {str(e)}{Style.RESET_ALL}",
                    flush=True
                )
                failed_positions.append({"symbol": symbol, "error": str(e)})

        # Summary
        print(
            f"\n{Fore.GREEN}[SQUARE OFF] Summary: Closed {len(closed_positions)} positions, "
            f"Failed {len(failed_positions)} positions{Style.RESET_ALL}",
            flush=True
        )

        # Log to trade history
        trade_state["trade_history"].append({
            "action": "SQUARE_OFF_ALL",
            "timestamp": datetime.now(IST).isoformat(),
            "reason": "Market closing - square off all positions",
            "status": "completed",
            "closed_positions": closed_positions,
            "failed_positions": failed_positions
        })

        return {
            "success": True,
            "action": "square_off",
            "closed_count": len(closed_positions),
            "failed_count": len(failed_positions),
            "closed_positions": closed_positions,
            "failed_positions": failed_positions
        }

    except Exception as e:
        print(f"{Fore.RED}[ERROR] Square off failed: {str(e)}{Style.RESET_ALL}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _cancel_all_pending_orders_direct():
    """Cancel all pending orders (direct call)."""
    try:
        print(f"\n{Fore.YELLOW}[CANCEL] Canceling all pending orders...{Style.RESET_ALL}", flush=True)
        response = client.cancelallorder(strategy="AI Agent")

        # Print the actual API response
        print(f"{Fore.GREEN}[CANCEL] Response: {response}{Style.RESET_ALL}", flush=True)

        reason = "Market closing - cancel pending orders"

        # Log to trade history
        trade_state["trade_history"].append({
            "action": "CANCEL_ALL_ORDERS",
            "timestamp": datetime.now(IST).isoformat(),
            "reason": reason,
            "status": "completed",
            "response": response
        })

        return {"success": True, "action": "cancel_orders", "reason": reason, "response": response}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Cancel orders failed: {str(e)}{Style.RESET_ALL}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Tool wrappers for agent to use
@function_tool
def square_off_all_positions() -> Dict[str, Any]:
    """Close all open positions by placing opposite orders for exact quantities."""
    return _square_off_all_positions_direct()


@function_tool
def cancel_all_pending_orders() -> Dict[str, Any]:
    """Cancel all pending orders."""
    return _cancel_all_pending_orders_direct()


# ============================================================================
# AI AGENT - Single Agent Architecture (Optimized for Speed)
# ============================================================================

# Agent Icons for Visual Identification
AGENT_ICONS = {
    "coordinator": "🎯",       # Trading Agent - Target/precision
    "bot": "🤖",               # Trading Bot - Robot
    "streaming": "🌊",         # Streaming - Wave
    "calculator": "🧮"         # Position Calculator - Abacus
}

# Single Autonomous Trading Agent (calls all tools directly for speed)
trading_agent = Agent(
    name=f"{AGENT_ICONS['coordinator']} Autonomous Trading Agent",
    instructions="""Process 5 symbols. Plain text. NO MARKDOWN.

STEP 1: get_all_market_data() ONCE → all data for all 5 symbols
STEP 2: For each symbol, decide BUY/SELL/HOLD (RSI, MACD, EMA signals)
STEP 3: For each trade decision:
  - check_risk_constraints(symbol, action)
  - calculate_position_size(symbol, ltp)
  - Add to orders list if allowed
STEP 4: place_bulk_orders() ONCE with complete JSON array

CRITICAL: Keep reasons SHORT (2-4 words)

OUTPUT FORMAT (plain text, one line per symbol):
ICICIBANK: BUY Order#123 (MACD bullish)
RELIANCE: HOLD (weak signals)
SBIN: HOLD (existing position)
WIPRO: SELL Order#124 (take profit)
ITC: HOLD (mixed signals)""",
    tools=[
        # Bulk operations (FAST - use these)
        get_all_market_data,
        check_all_risk_constraints,
        calculate_all_position_sizes,
        place_bulk_orders,
        # Account tools
        get_account_snapshot,
        get_current_positions,
        # Legacy individual tools (fallback only - avoid)
        check_risk_constraints,
        calculate_position_size,
        place_market_order,
        get_market_quotes,
        get_market_depth,
        get_historical_data
    ],
    model=trading_model
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_trading_cycle():
    """Execute one complete trading cycle."""
    now = datetime.now(IST)
    print(f"\n{'='*80}", flush=True)
    print(f"Trading Cycle: {now.strftime('%Y-%m-%d %H:%M:%S IST')}", flush=True)
    print(f"{'='*80}", flush=True)

    # Check if outside trading hours (before market open or after square-off time)
    if now.hour < MARKET_OPEN_HOUR or (now.hour == MARKET_OPEN_HOUR and now.minute < MARKET_OPEN_MINUTE):
        print(f"\n{Fore.YELLOW}[INFO] Market not open yet. Trading starts at {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} AM IST.{Style.RESET_ALL}", flush=True)
        return

    if now.hour > SQUARE_OFF_HOUR or (now.hour == SQUARE_OFF_HOUR and now.minute >= SQUARE_OFF_MINUTE):
        # After square-off time
        if not trade_state.get("squared_off_today", False):
            print("\nMarket Closing Time - Squaring Off All Positions", flush=True)
            _square_off_all_positions_direct()
            _cancel_all_pending_orders_direct()
            trade_state["squared_off_today"] = True
            print(f"\n{Fore.GREEN}[DONE] Square-off completed. No more trading today.{Style.RESET_ALL}", flush=True)
        else:
            print(f"\n{Fore.CYAN}[INFO] Market closed. Trading resumes at {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} AM IST tomorrow.{Style.RESET_ALL}", flush=True)
        return

    print("\nStarting autonomous trading workflow...", flush=True)

    # Update daily P&L from current positions
    current_pnl = update_daily_pnl()
    print(f"{Fore.CYAN}[P&L] Current Daily P&L: Rs.{current_pnl:.2f}{Style.RESET_ALL}", flush=True)

    # Run main decision cycle - ultra-brief query to minimize tokens
    query = f"""Trade cycle {now.strftime('%H:%M')}. Process all 5 symbols. P&L: Rs.{trade_state['daily_pnl']:.0f}. Stop-loss: {trade_state['stop_loss_hit']}."""

    print(f"{Fore.CYAN}{AGENT_ICONS['coordinator']} Executing MULTI-AGENT workflow with Master Coordinator...{Style.RESET_ALL}", flush=True)
    print(f"{Fore.WHITE}{AGENT_ICONS['coordinator']} [COORDINATOR] Starting delegation to specialist agents...{Style.RESET_ALL}\n", flush=True)

    # Run agent WITHOUT memory session to avoid token accumulation
    # Each trading cycle is independent and doesn't need conversation history
    print(f"{Fore.CYAN}{AGENT_ICONS['streaming']} [PROCESSING] Agent analyzing markets...{Style.RESET_ALL}", flush=True)

    try:
        final_result = await Runner.run(
            trading_agent,
            input=query,
            max_turns=60  # 5 symbols × ~8 tools per symbol (quotes, depth, history, risk, calc, order) + retries
            # No session parameter - each cycle is stateless to prevent token bloat
        )
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Trading cycle failed: {str(e)}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}[INFO] Skipping this cycle, will retry in next scheduled run{Style.RESET_ALL}", flush=True)
        return

    # Print usage statistics
    if hasattr(final_result, 'context_wrapper') and hasattr(final_result.context_wrapper, 'usage'):
        usage = final_result.context_wrapper.usage
        print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}[TOKEN USAGE] API Call Statistics:{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}  Requests:      {usage.requests if hasattr(usage, 'requests') else 'N/A'}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}  Input Tokens:  {usage.input_tokens:,}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}  Output Tokens: {usage.output_tokens:,}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}  Total Tokens:  {usage.total_tokens:,}{Style.RESET_ALL}", flush=True)

        # Calculate cost estimates (approximate based on model)
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        if MODEL_PROVIDER == "cerebras":
            cost = (input_tokens * 0.60 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
        elif MODEL_PROVIDER == "groq":
            cost = (input_tokens + output_tokens) * 0.05 / 1_000_000
        else:  # OpenAI
            cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        print(f"{Fore.YELLOW}  Est. Cost:     ${cost:.6f}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}\n", flush=True)

    print(
        f"\n{Fore.WHITE}{AGENT_ICONS['coordinator']} [RESULT] "
        f"{final_result.final_output}{Style.RESET_ALL}",
        flush=True
    )

    print(f"\n{'='*80}", flush=True)
    print("Trading cycle completed.", flush=True)
    print(f"{'='*80}\n", flush=True)


async def initialize_trading_state():
    """Initialize trading state by fetching current funds, positions, and P&L."""
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}", flush=True)
    print(f"{Fore.CYAN}[INIT] Initializing Trading State...{Style.RESET_ALL}", flush=True)
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}", flush=True)

    try:
        # 1. Fetch account funds
        print(f"{Fore.YELLOW}[INIT] Fetching account funds...{Style.RESET_ALL}", flush=True)
        funds_response = client.funds()
        if funds_response.get("status") == "success":
            funds_data = funds_response["data"]
            available_cash = float(funds_data.get("availablecash", 0))
            m2m_realized = float(funds_data.get("m2mrealized", 0))
            m2m_unrealized = float(funds_data.get("m2munrealized", 0))
            print(f"{Fore.GREEN}[INIT] ✓ Available Cash: Rs.{available_cash:,.2f}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.GREEN}[INIT] ✓ M2M Realized: Rs.{m2m_realized:,.2f}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.GREEN}[INIT] ✓ M2M Unrealized: Rs.{m2m_unrealized:,.2f}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.RED}[INIT] ✗ Failed to fetch funds{Style.RESET_ALL}", flush=True)

        # 2. Fetch open positions
        print(f"\n{Fore.YELLOW}[INIT] Fetching open positions...{Style.RESET_ALL}", flush=True)
        positions_response = client.positionbook()
        if positions_response.get("status") == "success":
            positions = positions_response.get("data", [])
            open_positions = [pos for pos in positions if float(pos.get("quantity", 0)) != 0]
            print(f"{Fore.GREEN}[INIT] ✓ Open Positions: {len(open_positions)}{Style.RESET_ALL}", flush=True)

            if open_positions:
                print(f"{Fore.CYAN}[INIT] Current Positions:{Style.RESET_ALL}", flush=True)
                for pos in open_positions:
                    symbol = pos.get("symbol")
                    quantity = float(pos.get("quantity", 0))
                    avg_price = float(pos.get("average_price", 0))
                    ltp = float(pos.get("ltp", 0))
                    pnl = float(pos.get("pnl", 0))
                    pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                    print(f"{Fore.CYAN}  • {symbol}: Qty={quantity:.0f}, Avg={avg_price:.2f}, LTP={ltp:.2f}, P&L={pnl_color}Rs.{pnl:.2f}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.RED}[INIT] ✗ Failed to fetch positions{Style.RESET_ALL}", flush=True)

        # 3. Calculate current P&L
        print(f"\n{Fore.YELLOW}[INIT] Calculating daily P&L...{Style.RESET_ALL}", flush=True)
        current_pnl = update_daily_pnl()
        pnl_color = Fore.GREEN if current_pnl >= 0 else Fore.RED
        print(f"{pnl_color}[INIT] ✓ Daily P&L: Rs.{current_pnl:,.2f}{Style.RESET_ALL}", flush=True)

        # 4. Check stop-loss status
        if current_pnl <= DAILY_STOP_LOSS:
            trade_state["stop_loss_hit"] = True
            print(f"{Fore.RED}[INIT] ⚠ STOP-LOSS HIT! Trading will be blocked.{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.GREEN}[INIT] ✓ Stop-loss check: OK (limit: Rs.{DAILY_STOP_LOSS:,}){Style.RESET_ALL}", flush=True)

        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.GREEN}[INIT] ✓ Initialization Complete{Style.RESET_ALL}", flush=True)
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n", flush=True)

    except Exception as e:
        print(f"{Fore.RED}[INIT] ✗ Initialization failed: {str(e)}{Style.RESET_ALL}", flush=True)
        import traceback
        traceback.print_exc()


async def start_autonomous_agent():
    """Start the autonomous trading agent."""
    print("\n" + "="*80, flush=True)
    print(f"{AGENT_ICONS['bot']} OpenAlgo Python Bot is running.", flush=True)
    print(f"{AGENT_ICONS['bot']} Autonomous AI Trading Agent - Self-Learning System", flush=True)
    print("="*80, flush=True)
    print(f"\nTrading Universe: {', '.join(SYMBOLS)}", flush=True)
    print(f"Max Investment/Trade: Rs.{MAX_INVESTMENT_PER_TRADE:,}", flush=True)
    print(f"Daily Stop-Loss: Rs.{DAILY_STOP_LOSS:,}", flush=True)
    print(f"Max Trade Per Symbol for the Day: {MAX_TRADES_PER_SYMBOL}", flush=True)
    print(f"Square-Off Time: {SQUARE_OFF_HOUR}:{SQUARE_OFF_MINUTE:02d} PM IST", flush=True)
    print("\n" + "="*80 + "\n", flush=True)

    # Initialize trading state
    await initialize_trading_state()

    # Setup scheduler
    scheduler = AsyncIOScheduler(timezone=IST)

    # Run every 5 minutes during market hours (9:15 AM - 3:30 PM)
    scheduler.add_job(
        run_trading_cycle,
        'cron',
        day_of_week='mon-fri',
        hour='9-15',
        minute='*/5',
        id='trading_cycle'
    )

    # Reset state at end of day
    scheduler.add_job(
        reset_daily_state,
        'cron',
        day_of_week='mon-fri',
        hour=DAILY_RESET_HOUR,
        minute=DAILY_RESET_MINUTE,
        id='daily_reset'
    )

    scheduler.start()
    print("Scheduler started - Agent will run every 5 minutes during market hours\n", flush=True)

    # Run immediately for testing if during market hours
    now = datetime.now(IST)
    if MARKET_OPEN_HOUR <= now.hour < SQUARE_OFF_HOUR or (now.hour == SQUARE_OFF_HOUR and now.minute < SQUARE_OFF_MINUTE):
        print("Running initial test cycle...\n", flush=True)
        await run_trading_cycle()
    else:
        print(
            f"{Fore.YELLOW}Outside market hours ({MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} AM - "
            f"{SQUARE_OFF_HOUR}:{SQUARE_OFF_MINUTE:02d} PM). Waiting for next scheduled run.{Style.RESET_ALL}\n",
            flush=True
        )

    # Keep running
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        print("\n\nShutting down agent...")
        scheduler.shutdown()


def reset_daily_state():
    """Reset state at end of trading day."""
    print("\n" + "="*80)
    print("End of Day - Resetting State")
    print(f"Final Daily P&L: Rs.{trade_state['daily_pnl']:.2f}")
    print(f"Total Trades: {sum(trade_state['trade_counts'].values())}")
    print("="*80 + "\n")

    # Save today's performance for learning
    with open(f"trade_history_{datetime.now(IST).strftime('%Y%m%d')}.json", "w") as f:
        json.dump(trade_state, f, indent=2)

    # Reset for next day
    trade_state["daily_pnl"] = 0.0
    trade_state["trade_counts"] = {symbol: 0 for symbol in SYMBOLS}
    trade_state["stop_loss_hit"] = False
    trade_state["squared_off_today"] = False  # Reset square-off flag


if __name__ == "__main__":
    asyncio.run(start_autonomous_agent())
