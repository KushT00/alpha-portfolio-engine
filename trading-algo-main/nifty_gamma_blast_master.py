"""
Nifty Gamma Blast & 0DTE OI/IV Spike Engine — OpenAlgo Platform Master Version
================================================================================
Institutional Option Accumulation Detector & 0-DTE Momentum Option Buyer:
  - Price + OI + IV Matrix: Classifies Long Buildup, Short Buildup, Short Covering.
  - Only trades on CONFIRMED Long Buildup (Price UP + OI UP + IV UP) or Short Covering.
  - Monitors ATM ±2 Strikes on both CE and PE sides (5 strikes each, 10 total).
  - Multi-Timeframe Rolling Buffers: 3m, 5m, 15m, 30m OI & IV & Price snapshots.
  - Execution: NRML product type with 50% Hard Stop Loss.
  - Anti-IV-Crush: Exits if option price fails to move +15% within 5 minutes of entry.
  - Trailing Profit Lock: Breakeven trail at +25%, targets at +50%, +100%, +200%.
  - Day-Type Modes: PRIMARY ACTIVE on 0-DTE Tuesday, STANDBY on regular days.
  - Hard exit by 3:00 PM on 0-DTE Expiry Day.
"""

import os
import re
import json
import time
import math
import csv
import logging
from datetime import datetime, date, timedelta
from collections import deque
from openalgo import api

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GammaBlast")

# ============================================================
# 1. PARAMETERS & CONFIG
# ============================================================
CONFIG = {
    # OpenAlgo API Configuration
    "api_key": "ff1d258f2c5d48bf87292b3f055b39772ca76f4afe69ff7c7c5fa121a32334d8",
    "host": "http://127.0.0.1:5000",

    # Core Instrument Parameters
    "underlying": "NIFTY",
    "exchange_index": "NSE_INDEX",
    "exchange_deriv": "NFO",
    "lot_size": 65,                # Nifty Lot Size is 65
    "strike_step": 50,             # Nifty Strike Step is 50

    # Strike Monitoring: ATM +/- this many steps on each side
    "atm_strike_range": 2,         # Monitor ATM-2, ATM-1, ATM, ATM+1, ATM+2

    # OI Change Thresholds for Accumulation Detection
    "oi_threshold_3m": 10.0,       # 3-min OI surge threshold (%)
    "oi_threshold_5m": 12.0,       # 5-min OI surge threshold (%)
    "oi_threshold_15m": 30.0,      # 15-min OI surge threshold (%)
    "oi_alert_cells_pct": 30.0,    # Alert when 30%+ matrix cells breach threshold

    # Spot Lag Condition: Spot must be nearly flat for accumulation to be "pre-move"
    "spot_lag_threshold_pct": 0.15, # Spot movement < 0.15% while options are loading

    # IV Change Threshold for Long Buildup confirmation
    "iv_surge_threshold_pct": 5.0, # IV must rise by 5%+ to confirm Long Buildup

    # Order Parameters
    "quantity": 65,                # 1 Lot = 65 (NRML NIFTY option)
    "order_type": "MARKET",
    "product_type": "NRML",        # NRML — holds against naked short legs as hedge
    "tag": "GammaBlast",

    # Risk Management
    "hard_sl_pct": 0.50,           # Hard Stop Loss: 50% premium decay
    "trail_be_at_pct": 0.25,       # Trail stop to breakeven when +25% gain reached
    "target_1_pct": 0.50,          # First profit target: +50%
    "target_2_pct": 1.00,          # Second profit target: +100%
    "target_3_pct": 2.00,          # Third profit target: +200%
    "anti_crush_min_move_pct": 0.15, # Option must move +15% within 5 mins or exit (anti-IV-crush)
    "anti_crush_window_secs": 300,  # 5-minute anti-crush window

    # Day-Type Thresholds (STANDBY on regular days, lower trigger required)
    # On regular days, use double threshold to only trade extreme events as hedges
    "standby_oi_multiplier": 2.0,  # Multiply OI thresholds by this on non-expiry days

    # Stochastic RSI (14, 14, 3, 3) 3-Minute Candle Signal Engine
    "use_stoch_rsi_filter": True,   # Enables 3-min Stoch RSI confirmation
    "stoch_rsi_rsi_len": 14,        # RSI Length = 14
    "stoch_rsi_stoch_len": 14,      # Stochastic Length = 14
    "stoch_rsi_k_smooth": 3,        # K Smoothing = 3
    "stoch_rsi_d_smooth": 3,        # D Smoothing = 3
    "stoch_rsi_oversold": 25.0,     # Oversold threshold for CE Buy (%K crosses above %D < 25)
    "stoch_rsi_overbought": 75.0,   # Overbought threshold for PE Buy (%K crosses below %D > 75)

    # 0-DTE Day Hard Square-Off Time (3:00 PM IST)
    "zero_dte_squareoff_hour": 15,
    "zero_dte_squareoff_minute": 0,

    # Loop Interval (seconds)
    "poll_interval": 30,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(SCRIPT_DIR, "nifty_gamma_blast_state.json")
TRADE_LOG   = os.path.join(SCRIPT_DIR, "nifty_gamma_blast_trades.csv")
AUDIT_LOG   = os.path.join(SCRIPT_DIR, "nifty_gamma_blast_audit.csv")


# ============================================================
# 2. ROLLING BUFFER ENGINE
#    Stores time-stamped OI, IV, and Price snapshots for each
#    monitored strike. Calculates % change over rolling windows.
# ============================================================
class StrikeBuffer:
    """Rolling time-series buffer for a single option strike"""
    def __init__(self, symbol):
        self.symbol = symbol
        # Each entry: {"ts": datetime, "ltp": float, "oi": float, "iv": float}
        self.history = deque(maxlen=200)

    def push(self, ltp, oi, iv):
        self.history.append({
            "ts": datetime.now(),
            "ltp": ltp,
            "oi": oi,
            "iv": iv,
        })

    def _get_reading_at_window(self, window_secs):
        """Find the oldest reading within the rolling window"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_secs)
        for entry in reversed(self.history):
            if entry["ts"] <= cutoff:
                return entry
        # If no reading old enough, use oldest available
        if self.history:
            return self.history[0]
        return None

    def pct_change_oi(self, window_secs):
        """Return OI % change over the given window in seconds"""
        ref = self._get_reading_at_window(window_secs)
        if not ref or len(self.history) == 0:
            return 0.0
        current_oi = self.history[-1]["oi"]
        ref_oi = ref["oi"]
        if ref_oi <= 0:
            return 0.0
        return ((current_oi - ref_oi) / ref_oi) * 100.0

    def pct_change_ltp(self, window_secs):
        """Return LTP % change over the given window"""
        ref = self._get_reading_at_window(window_secs)
        if not ref or len(self.history) == 0:
            return 0.0
        current = self.history[-1]["ltp"]
        ref_val = ref["ltp"]
        if ref_val <= 0:
            return 0.0
        return ((current - ref_val) / ref_val) * 100.0

    def pct_change_iv(self, window_secs):
        """Return IV % change over the given window"""
        ref = self._get_reading_at_window(window_secs)
        if not ref or len(self.history) == 0:
            return 0.0
        current = self.history[-1]["iv"]
        ref_val = ref["iv"]
        if ref_val <= 0:
            return 0.0
        return ((current - ref_val) / ref_val) * 100.0

    @property
    def current_ltp(self):
        return self.history[-1]["ltp"] if self.history else 0.0

    @property
    def current_oi(self):
        return self.history[-1]["oi"] if self.history else 0.0

    @property
    def current_iv(self):
        return self.history[-1]["iv"] if self.history else 0.0


# ============================================================
# 3. BLACK-SCHOLES IMPLIED VOLATILITY ESTIMATOR
#    Simple iterative BS IV solver (Newton-Raphson)
# ============================================================
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(spot, strike, T, sigma, r=0.07, is_call=True):
    """Calculate Black-Scholes option price"""
    if T <= 0 or sigma <= 0:
        return max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return strike * math.exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

def estimate_iv(spot, strike, dte_days, ltp, is_call=True, r=0.07):
    """Estimate Implied Volatility via Newton-Raphson iteration"""
    T = max(dte_days, 0.5) / 365.0
    sigma = 0.15  # Initial guess
    for _ in range(50):
        price = bs_price(spot, strike, T, sigma, r, is_call)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = spot * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break
        diff = price - ltp
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))
        if abs(diff) < 0.001:
            break
    return round(sigma * 100.0, 2)  # Return as percentage


# ============================================================
# 3. STOCHASTIC RSI (14, 14, 3, 3) 3-MINUTE CANDLE ENGINE
# ============================================================
class StochRSICandleAggregator:
    """Aggregates 3-minute OHLC candles and computes Stochastic RSI (14, 14, 3, 3)."""
    def __init__(self, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
        self.rsi_len = rsi_len
        self.stoch_len = stoch_len
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.candles = []
        self.current_candle = None
        self.stoch_k = 50.0
        self.stoch_d = 50.0
        self.prev_k = 50.0
        self.prev_d = 50.0

    def update_tick(self, spot):
        now = datetime.now()
        minute = now.minute - (now.minute % 3)
        bucket = now.replace(minute=minute, second=0, microsecond=0)

        if self.current_candle is None or self.current_candle["time"] != bucket:
            if self.current_candle is not None:
                self.candles.append(self.current_candle.copy())
                if len(self.candles) > 100:
                    self.candles.pop(0)
                self._recalculate_stoch_rsi()

            self.current_candle = {
                "time": bucket,
                "open": spot,
                "high": spot,
                "low": spot,
                "close": spot,
            }
        else:
            self.current_candle["high"] = max(self.current_candle["high"], spot)
            self.current_candle["low"]  = min(self.current_candle["low"], spot)
            self.current_candle["close"] = spot

    def _recalculate_stoch_rsi(self):
        if len(self.candles) < self.rsi_len + self.stoch_len + 5:
            return

        closes = [c["close"] for c in self.candles]
        gains, losses = [], []

        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))

        # RSI(14)
        rsi_series = []
        avg_gain = sum(gains[:self.rsi_len]) / self.rsi_len
        avg_loss = sum(losses[:self.rsi_len]) / self.rsi_len

        for i in range(self.rsi_len, len(gains)):
            avg_gain = (avg_gain * 13 + gains[i]) / 14.0
            avg_loss = (avg_loss * 13 + losses[i]) / 14.0
            rs = avg_gain / max(avg_loss, 1e-10)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_series.append(rsi)

        if len(rsi_series) < self.stoch_len:
            return

        # Stoch RSI(14)
        stoch_raw = []
        for i in range(self.stoch_len - 1, len(rsi_series)):
            window = rsi_series[i - self.stoch_len + 1 : i + 1]
            min_r = min(window)
            max_r = max(window)
            rng = max_r - min_r
            val = ((rsi_series[i] - min_r) / max(rng, 1e-10)) * 100.0
            stoch_raw.append(val)

        if len(stoch_raw) < self.k_smooth:
            return

        # %K (3-period SMA)
        k_series = []
        for i in range(self.k_smooth - 1, len(stoch_raw)):
            k_val = sum(stoch_raw[i - self.k_smooth + 1 : i + 1]) / float(self.k_smooth)
            k_series.append(k_val)

        if len(k_series) < self.d_smooth:
            return

        # %D (3-period SMA)
        d_series = []
        for i in range(self.d_smooth - 1, len(k_series)):
            d_val = sum(k_series[i - self.d_smooth + 1 : i + 1]) / float(self.d_smooth)
            d_series.append(d_val)

        if k_series and d_series:
            self.prev_k = self.stoch_k
            self.prev_d = self.stoch_d
            self.stoch_k = round(k_series[-1], 2)
            self.stoch_d = round(d_series[-1], 2)

    def get_signal(self, oversold=25.0, overbought=75.0):
        """Returns 'CE_CONFIRMED', 'PE_CONFIRMED', or 'NEUTRAL' based on 3-min Stoch RSI crossover."""
        if self.prev_k <= self.prev_d and self.stoch_k > self.stoch_d and self.stoch_k <= oversold + 15:
            return "CE_CONFIRMED"
        if self.prev_k >= self.prev_d and self.stoch_k < self.stoch_d and self.stoch_k >= overbought - 15:
            return "PE_CONFIRMED"
        return "NEUTRAL"


# ============================================================
# 4. BUILDUP CLASSIFIER
#    Implements the Price + OI + IV Matrix Logic
# ============================================================
class BuildupClassifier:
    """Classifies option activity using Price + OI + IV Matrix"""

    LONG_BUILDUP    = "LONG_BUILDUP"    # Price UP + OI UP + IV UP  → Aggressive Buyers
    SHORT_BUILDUP   = "SHORT_BUILDUP"   # Price DOWN + OI UP         → Option Writers
    SHORT_COVERING  = "SHORT_COVERING"  # Price UP + OI DOWN + IV UP → Short Squeeze
    LONG_UNWINDING  = "LONG_UNWINDING"  # Price DOWN + OI DOWN       → Longs Exiting
    NEUTRAL         = "NEUTRAL"         # No conclusive pattern

    @staticmethod
    def classify(delta_ltp_pct, delta_oi_pct, delta_iv_pct):
        """
        Classify buildup from % changes in LTP, OI, and IV.

        Returns (classification, confidence_score)
        confidence_score: 0-3 (how many criteria are strongly met)
        """
        price_up  = delta_ltp_pct > 1.0
        price_dn  = delta_ltp_pct < -1.0
        oi_up     = delta_oi_pct  > 5.0
        oi_dn     = delta_oi_pct  < -5.0
        iv_up     = delta_iv_pct  > 3.0

        if price_up and oi_up and iv_up:
            confidence = sum([delta_ltp_pct > 3.0, delta_oi_pct > 10.0, delta_iv_pct > 5.0])
            return BuildupClassifier.LONG_BUILDUP, confidence

        if price_up and oi_dn and iv_up:
            confidence = sum([delta_ltp_pct > 3.0, delta_oi_pct < -10.0, delta_iv_pct > 5.0])
            return BuildupClassifier.SHORT_COVERING, confidence

        if price_dn and oi_up:
            return BuildupClassifier.SHORT_BUILDUP, 0  # Option writing — filter out

        if price_dn and oi_dn:
            return BuildupClassifier.LONG_UNWINDING, 0

        return BuildupClassifier.NEUTRAL, 0


# ============================================================
# 5. GAMMA BLAST ENGINE
# ============================================================
class GammaBlastEngine:

    def __init__(self, config=CONFIG):
        self.config = config
        self.client = api(api_key=config["api_key"], host=config["host"])

        # Open Positions Tracking: symbol → {entry_price, entry_time, opt_type, qty, trail_sl, be_triggered}
        self.open_positions = {}

        # Strike Buffers: symbol → StrikeBuffer
        self.buffers = {}

        # Spot history for lag condition
        self.spot_history = deque(maxlen=50)

        # 3-Minute Stochastic RSI (14, 14, 3, 3) Candle Aggregator
        self.stoch_rsi_agg = StochRSICandleAggregator(
            rsi_len=self.config.get("stoch_rsi_rsi_len", 14),
            stoch_len=self.config.get("stoch_rsi_stoch_len", 14),
            k_smooth=self.config.get("stoch_rsi_k_smooth", 3),
            d_smooth=self.config.get("stoch_rsi_d_smooth", 3),
        )

        # Load saved state
        self.state = self._load_state()
        self.open_positions = self.state.get("open_positions", {})

        logger.info("🚀 Gamma Blast Engine Initialized with 3-min Stoch RSI Filter.")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"State load error: {e}")
        return {}

    def _save_state(self):
        self.state["open_positions"] = self.open_positions
        self.state["last_eval"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"State save error: {e}")

    def _log_trade(self, order_id, symbol, action, qty, price, status):
        try:
            exists = os.path.exists(TRADE_LOG)
            with open(TRADE_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["timestamp", "order_id", "symbol", "action", "qty", "price", "status"])
                writer.writerow([datetime.now().isoformat(), order_id, symbol, action, qty, price, status])
        except Exception as e:
            logger.error(f"Trade log error: {e}")

    def _log_decision(self, action, reason, details=None):
        logger.info(f"[{action}] {reason} | {details or ''}")
        try:
            exists = os.path.exists(AUDIT_LOG)
            with open(AUDIT_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["timestamp", "action", "reason", "details"])
                writer.writerow([datetime.now().isoformat(), action, reason, json.dumps(details or {})])
        except Exception:
            pass

    def _is_zero_dte_day(self):
        """Returns True if today is Tuesday (NIFTY weekly 0-DTE expiry day)"""
        return datetime.now().weekday() == 1  # Monday=0, Tuesday=1

    def _get_active_thresholds(self):
        """Returns OI thresholds based on day type (Primary vs Standby mode)"""
        mult = 1.0 if self._is_zero_dte_day() else self.config["standby_oi_multiplier"]
        return {
            "3m":  self.config["oi_threshold_3m"]  * mult,
            "5m":  self.config["oi_threshold_5m"]  * mult,
            "15m": self.config["oi_threshold_15m"] * mult,
        }

    def _get_spot_price(self):
        try:
            res = self.client.quotes(exchange=self.config["exchange_index"], symbol=self.config["underlying"])
            if res and isinstance(res, dict):
                data = res.get("data", res)
                return float(data.get("ltp") or data.get("last_price", 0.0))
        except Exception as e:
            logger.error(f"Spot fetch error: {e}")
        return 0.0

    def _get_option_quote(self, symbol):
        """Returns (ltp, oi) for a given option symbol"""
        try:
            res = self.client.quotes(exchange=self.config["exchange_deriv"], symbol=symbol)
            if res and isinstance(res, dict):
                data = res.get("data", res)
                ltp = float(data.get("ltp") or data.get("last_price") or 0.0)
                oi  = float(data.get("oi") or data.get("open_interest") or 0.0)
                return ltp, oi
        except Exception as e:
            logger.warning(f"Quote fetch error for {symbol}: {e}")
        return 0.0, 0.0

    def _get_expiry(self):
        """Fetch nearest active options expiry code (e.g. 04AUG26)"""
        try:
            res = self.client.expiry(
                symbol=self.config["underlying"],
                exchange=self.config["exchange_deriv"],
                instrumenttype="options"
            )
            data = res if isinstance(res, list) else res.get("data", [])
            if data:
                return data[0].replace("-", "")
        except Exception as e:
            logger.error(f"Expiry fetch error: {e}")
        return ""

    def _build_symbol(self, expiry_code, strike, opt_type):
        return f"{self.config['underlying']}{expiry_code}{int(strike)}{opt_type}"

    def _get_atm_strike(self, spot):
        step = self.config["strike_step"]
        return int(round(spot / step) * step)

    def _get_dte(self, expiry_code):
        """Calculate Days to Expiry from expiry code like '04AUG26'"""
        try:
            exp_date = datetime.strptime(expiry_code, "%d%b%y").date()
            return max(0, (exp_date - date.today()).days)
        except Exception:
            return 0

    def _get_spot_pct_change(self, window_secs=300):
        """Returns spot % change over the last N seconds"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_secs)
        for entry in reversed(self.spot_history):
            if entry["ts"] <= cutoff:
                ref_spot = entry["spot"]
                if ref_spot > 0:
                    current_spot = self.spot_history[-1]["spot"] if self.spot_history else 0
                    return abs((current_spot - ref_spot) / ref_spot * 100.0)
        return 0.0

    # ─── Strike Buffer Update ─────────────────────────────────────────────────

    def update_strike_buffers(self, spot, expiry_code):
        """Fetch quotes for ATM ±2 strikes and push to rolling buffers"""
        atm = self._get_atm_strike(spot)
        step = self.config["strike_step"]
        dte = self._get_dte(expiry_code)

        for offset_steps in range(-self.config["atm_strike_range"], self.config["atm_strike_range"] + 1):
            strike = atm + (offset_steps * step)
            for opt_type in ["CE", "PE"]:
                symbol = self._build_symbol(expiry_code, strike, opt_type)
                ltp, oi = self._get_option_quote(symbol)

                if ltp > 0:
                    is_call = (opt_type == "CE")
                    iv = estimate_iv(spot, strike, dte, ltp, is_call=is_call)
                else:
                    iv = 0.0

                if symbol not in self.buffers:
                    self.buffers[symbol] = StrikeBuffer(symbol)
                self.buffers[symbol].push(ltp, oi, iv)

    # ─── Buildup Signal Detection ─────────────────────────────────────────────

    def scan_for_signals(self, spot):
        """
        Scan all monitored buffers for Long Buildup or Short Covering signals.
        Returns list of (symbol, opt_type, classification, confidence, ltp)
        """
        thresholds = self._get_active_thresholds()
        spot_change_pct = self._get_spot_pct_change(window_secs=300)

        signals = []

        for symbol, buf in self.buffers.items():
            if len(buf.history) < 2:
                continue

            ltp = buf.current_ltp
            if ltp <= 0:
                continue

            # Get changes over 5-min window for classification
            delta_ltp_5m = buf.pct_change_ltp(300)
            delta_oi_5m  = buf.pct_change_oi(300)
            delta_iv_5m  = buf.pct_change_iv(300)

            # Check OI thresholds across timeframes
            oi_3m  = buf.pct_change_oi(180)
            oi_5m  = delta_oi_5m
            oi_15m = buf.pct_change_oi(900)

            breaches = sum([
                oi_3m  >= thresholds["3m"],
                oi_5m  >= thresholds["5m"],
                oi_15m >= thresholds["15m"],
            ])

            if breaches < 1:
                continue  # Not enough OI surge

            # Classify buildup
            classification, confidence = BuildupClassifier.classify(
                delta_ltp_5m, delta_oi_5m, delta_iv_5m
            )

            # Only trade Long Buildup and Short Covering — reject Short Buildup
            if classification not in (BuildupClassifier.LONG_BUILDUP, BuildupClassifier.SHORT_COVERING):
                self._log_decision(
                    "SIGNAL_FILTERED",
                    f"{symbol} → {classification} | OI {oi_5m:.1f}% | Price {delta_ltp_5m:.1f}% — Rejected",
                    {"spot_change": spot_change_pct}
                )
                continue

            # Spot Lag Condition: Spot must not have moved too much already
            if spot_change_pct > self.config["spot_lag_threshold_pct"] and classification == BuildupClassifier.LONG_BUILDUP:
                # Spot already moved — less reliable entry, log but skip
                self._log_decision(
                    "SPOT_LAG_FAIL",
                    f"{symbol} → Spot already moved {spot_change_pct:.2f}% — Pre-move window missed.",
                )
                continue

            opt_type = "CE" if symbol.endswith("CE") else "PE"

            # Stochastic RSI (14, 14, 3, 3) 3-Minute Candle Double-Confirmation Filter
            if self.config.get("use_stoch_rsi_filter", True):
                stoch_signal = self.stoch_rsi_agg.get_signal(
                    oversold=self.config["stoch_rsi_oversold"],
                    overbought=self.config["stoch_rsi_overbought"]
                )
                stoch_k = self.stoch_rsi_agg.stoch_k

                # For CE Long Buildup: require Stoch RSI to confirm oversold bounce (%K <= 35 or CE_CONFIRMED)
                if opt_type == "CE" and not (stoch_signal == "CE_CONFIRMED" or stoch_k <= 35.0):
                    self._log_decision(
                        "STOCH_RSI_CE_REJECTED",
                        f"{symbol} → Stoch RSI %K ({stoch_k:.1f}) not oversold (target <= 35.0). Skipping CE Buy."
                    )
                    continue

                # For PE Long Buildup: require Stoch RSI to confirm overbought reversal (%K >= 65 or PE_CONFIRMED)
                if opt_type == "PE" and not (stoch_signal == "PE_CONFIRMED" or stoch_k >= 65.0):
                    self._log_decision(
                        "STOCH_RSI_PE_REJECTED",
                        f"{symbol} → Stoch RSI %K ({stoch_k:.1f}) not overbought (target >= 65.0). Skipping PE Buy."
                    )
                    continue

            signals.append((symbol, opt_type, classification, confidence, ltp, breaches))
            logger.info(
                f"🔔 [{classification}] {symbol} | OI 3m:{oi_3m:.1f}% 5m:{oi_5m:.1f}% 15m:{oi_15m:.1f}% "
                f"| ΔPrice:{delta_ltp_5m:.1f}% | ΔIV:{delta_iv_5m:.1f}% | StochRSI %K:{self.stoch_rsi_agg.stoch_k:.1f} | Confidence:{confidence}"
            )

        # Sort by confidence + breaches descending
        signals.sort(key=lambda x: (x[3], x[5]), reverse=True)
        return signals

    # ─── Order Execution ──────────────────────────────────────────────────────

    def execute_buy(self, symbol, opt_type, ltp, classification):
        """Execute NRML BUY order for a confirmed Long Buildup or Short Covering"""
        if symbol in self.open_positions:
            logger.info(f"Already in position for {symbol}. Skipping.")
            return

        qty = self.config["quantity"]
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="BUY",
                exchange=self.config["exchange_deriv"],
                price=0,
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty,
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            logger.info(f"✅ [BUY EXECUTED] {symbol} x {qty} @ Rs.{ltp:.2f} | Signal: {classification}")
            self._log_trade(order_id, symbol, "BUY", qty, ltp, res.get("status", "SUCCESS"))
            self._log_decision("BUY_EXECUTED", f"{classification} → Bought {symbol} x {qty} @ Rs.{ltp:.2f}")

            self.open_positions[symbol] = {
                "qty": qty,
                "entry_price": ltp,
                "entry_time": datetime.now().isoformat(),
                "opt_type": opt_type,
                "signal": classification,
                "trail_sl": ltp * (1.0 - self.config["hard_sl_pct"]),  # Initial SL at -50%
                "be_triggered": False,
                "anti_crush_deadline": (datetime.now() + timedelta(seconds=self.config["anti_crush_window_secs"])).isoformat(),
            }
            self._save_state()

        except Exception as e:
            logger.error(f"BUY order error for {symbol}: {e}")

    def execute_sell(self, symbol, pos, reason):
        """Square off a NRML long option position"""
        ltp, _ = self._get_option_quote(symbol)
        if ltp <= 0:
            ltp = pos["entry_price"] * 0.5  # Fallback if quote unavailable
        qty = pos["qty"]
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="SELL",
                exchange=self.config["exchange_deriv"],
                price=0,
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty,
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            pnl = (ltp - pos["entry_price"]) * qty
            logger.info(f"🔴 [{reason}] Sold {symbol} x {qty} @ Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
            self._log_trade(order_id, symbol, "SELL", qty, ltp, res.get("status", "SUCCESS"))
            self._log_decision(reason, f"Closed {symbol} x {qty} @ Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
        except Exception as e:
            logger.error(f"SELL order error for {symbol}: {e}")

    # ─── Position Management ──────────────────────────────────────────────────

    def manage_open_positions(self):
        """
        For each open long option position:
        1. Check 50% Hard Stop Loss
        2. Anti-IV Crush time exit (fails to move +15% in 5 mins)
        3. Trail SL to breakeven at +25% gain
        4. Scale out at +50%, +100%, +200% targets
        """
        to_remove = []

        for symbol, pos in list(self.open_positions.items()):
            ltp, _ = self._get_option_quote(symbol)
            if ltp <= 0:
                continue

            entry = pos["entry_price"]
            current_gain_pct = (ltp - entry) / entry

            # ── 1. Hard Stop Loss (50% decay) ──────────────────────────────
            hard_sl_price = entry * (1.0 - self.config["hard_sl_pct"])
            if ltp <= hard_sl_price:
                logger.warning(f"🛑 [HARD_SL] {symbol} | Entry: Rs.{entry:.2f} | LTP: Rs.{ltp:.2f} | -50% Hit!")
                self.execute_sell(symbol, pos, "HARD_STOP_LOSS")
                to_remove.append(symbol)
                continue

            # ── 2. Anti-IV Crush Exit ──────────────────────────────────────
            deadline = datetime.fromisoformat(pos["anti_crush_deadline"])
            if datetime.now() > deadline and current_gain_pct < self.config["anti_crush_min_move_pct"]:
                logger.warning(f"⏰ [ANTI_CRUSH_EXIT] {symbol} | {self.config['anti_crush_window_secs']//60}m elapsed, only +{current_gain_pct*100:.1f}% — IV Crush risk. Exiting.")
                self.execute_sell(symbol, pos, "ANTI_IV_CRUSH_EXIT")
                to_remove.append(symbol)
                continue

            # ── 3. Trail SL to Breakeven at +25% ──────────────────────────
            if not pos["be_triggered"] and current_gain_pct >= self.config["trail_be_at_pct"]:
                pos["trail_sl"] = entry  # Trail to breakeven
                pos["be_triggered"] = True
                logger.info(f"🔒 [TRAIL_BE] {symbol} | +{current_gain_pct*100:.1f}% reached → SL moved to Breakeven Rs.{entry:.2f}")
                self._save_state()

            if pos["be_triggered"] and ltp <= pos["trail_sl"]:
                logger.info(f"🔒 [TRAIL_SL_HIT] {symbol} | LTP Rs.{ltp:.2f} ≤ Trail SL Rs.{pos['trail_sl']:.2f}")
                self.execute_sell(symbol, pos, "TRAIL_SL_HIT")
                to_remove.append(symbol)
                continue

            # ── 4. Profit Target Exits ─────────────────────────────────────
            if current_gain_pct >= self.config["target_3_pct"]:
                logger.info(f"🎯 [TARGET_3] {symbol} | +{current_gain_pct*100:.1f}% ≥ +200% target!")
                self.execute_sell(symbol, pos, "TARGET_3_HIT")
                to_remove.append(symbol)
            elif current_gain_pct >= self.config["target_2_pct"]:
                logger.info(f"🎯 [TARGET_2] {symbol} | +{current_gain_pct*100:.1f}% ≥ +100% target!")
                self.execute_sell(symbol, pos, "TARGET_2_HIT")
                to_remove.append(symbol)
            elif current_gain_pct >= self.config["target_1_pct"]:
                logger.info(f"🎯 [TARGET_1] {symbol} | +{current_gain_pct*100:.1f}% ≥ +50% target!")
                self.execute_sell(symbol, pos, "TARGET_1_HIT")
                to_remove.append(symbol)
            else:
                logger.info(
                    f"[GAMMA POS] {symbol} | Entry: Rs.{entry:.2f} | LTP: Rs.{ltp:.2f} | "
                    f"Gain: {current_gain_pct*100:+.1f}% | SL: Rs.{pos['trail_sl']:.2f}"
                )

        for sym in to_remove:
            self.open_positions.pop(sym, None)
        if to_remove:
            self._save_state()

    # ─── 0-DTE Hard Square-Off ────────────────────────────────────────────────

    def check_zero_dte_squareoff(self):
        """Force exit all open positions at 3:00 PM on 0-DTE expiry day"""
        if not self._is_zero_dte_day():
            return False

        now = datetime.now()
        sq_hour = self.config["zero_dte_squareoff_hour"]
        sq_min  = self.config["zero_dte_squareoff_minute"]

        if now.hour > sq_hour or (now.hour == sq_hour and now.minute >= sq_min):
            if self.open_positions:
                logger.warning("⚡ [0-DTE SQUAREOFF] 3:00 PM reached — Hard closing all Gamma Blast positions!")
                for symbol, pos in list(self.open_positions.items()):
                    self.execute_sell(symbol, pos, "ZERO_DTE_SQUAREOFF")
                self.open_positions.clear()
                self._save_state()
            return True
        return False

    # ─── Main Tick ────────────────────────────────────────────────────────────

    def execute_tick(self):
        """Main strategy tick"""
        now = datetime.now()
        mode = "PRIMARY (0-DTE Day)" if self._is_zero_dte_day() else "STANDBY (Regular Day)"
        logger.info(f"[GAMMA BLAST TICK] {now.strftime('%H:%M:%S')} | Mode: {mode} | Open: {len(self.open_positions)}")

        # 0-DTE Hard Square-Off Check
        if self.check_zero_dte_squareoff():
            # Still manage trailing SLs up to 3 PM
            return

        # Get live spot price
        spot = self._get_spot_price()
        if spot <= 0:
            logger.warning("Invalid spot. Retrying.")
            return

        # Store spot for lag condition calculation and update 3-min Stoch RSI candles
        self.spot_history.append({"ts": datetime.now(), "spot": spot})
        self.stoch_rsi_agg.update_tick(spot)

        # Fetch current expiry
        expiry_code = self._get_expiry()
        if not expiry_code:
            logger.warning("Could not fetch expiry. Skipping.")
            return

        # Step 1: Update all strike rolling buffers
        self.update_strike_buffers(spot, expiry_code)

        # Step 2: Manage existing open positions (SL, targets, trail)
        self.manage_open_positions()

        # Step 3: Scan for new entry signals
        signals = self.scan_for_signals(spot)

        # Step 4: Execute top-confidence signal if no position in that strike/type
        for symbol, opt_type, classification, confidence, ltp, breaches in signals:
            if symbol not in self.open_positions:
                self.execute_buy(symbol, opt_type, ltp, classification)
                break  # One new trade per tick to avoid overtrading


# ============================================================
# 6. RUN LOOP
# ============================================================
def run():
    engine = GammaBlastEngine(CONFIG)
    logger.info("🚀 Starting Gamma Blast & 0DTE OI/IV Spike Engine Loop...")
    while True:
        try:
            engine.execute_tick()
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in Gamma Blast loop: {e}", exc_info=True)

        time.sleep(CONFIG["poll_interval"])


if __name__ == "__main__":
    run()
