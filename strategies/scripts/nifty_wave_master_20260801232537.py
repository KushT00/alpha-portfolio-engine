"""
Nifty Wave Scraper Strategy — OpenAlgo Platform Master Version
================================================================
Non-Directional Options Wave Scraper Strategy (Mean Reversion):
  - Dynamic VIX Auto-Config: Fetches INDIA VIX on startup and sets gaps, offsets, profit targets, and position caps.
  - Positionbook Sync: Syncs local open positions with live broker positionbook on startup and each tick.
  - Reference Price: NIFTY SPOT (NSE_INDEX:NIFTY).
  - CE Selling Trigger: When Spot rises by ce_gap (+20 pts) above ce_reference, sells OTM Call (Spot + ce_otm_offset).
  - PE Selling Trigger: When Spot falls by pe_gap (-20 pts) below pe_reference, sells OTM Put (Spot - pe_otm_offset).
  - Level Cap Safeguard: Limits max consecutive sell levels per side to 2 before requiring profit booking or reset.
  - Profit Booking & Leg SL: Buys back option on 40% decay target or on leg stop loss breach (2.0x entry price).
  - Portfolio Delta Gate: Calculates Black-Scholes portfolio delta; pauses CE/PE sells if delta breaches thresholds (-100 / +100).
  - Reference Reset: Adjusts ce_reference/pe_reference on strong market reversals to maintain wave responsiveness.
  - Expiry Safety: Auto-squares off positions by Monday 3:15 PM EOD.
  - 0-DTE Dual Mode (Tuesday): Manages + hard-closes today's expiry by 3:00 PM AND executes new sells on NEXT expiry at 2x OTM offset.
"""

import os
import json
import time
import csv
import math
import logging
from datetime import datetime, date
from openalgo import api

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WaveMaster")

# ============================================================
# 1. PARAMETERS & CONFIG
# ============================================================
CONFIG = {
    # OpenAlgo API Configuration
    "api_key": "ff1d258f2c5d48bf87292b3f055b39772ca76f4afe69ff7c7c5fa121a32334d8",
    "host": "http://127.0.0.1:5000",
    
    # Dynamic VIX Auto-Config Toggle
    "auto_vix_config": True,
    
    # Core Instrument Parameters
    "underlying": "NIFTY",
    "exchange_index": "NSE_INDEX",
    "exchange_deriv": "NFO",
    "lot_size": 65,                # Nifty Lot Size is 65
    "strike_step": 50,             # Nifty Strike Step is 50
    
    # Default Wave Trigger Thresholds (Overridden dynamically by VIX)
    "ce_gap": 20.0,
    "pe_gap": 20.0,
    
    # OTM Offset Parameters
    "ce_otm_offset": 100,
    "pe_otm_offset": 100,
    
    # Position Quantities & Caps
    "ce_quantity": 65,             # 1 Lot = 65
    "pe_quantity": 65,             # 1 Lot = 65
    "max_active_positions": 4,     # Max concurrent open short positions per side
    "max_consecutive_sells": 2,    # Max consecutive sell levels per side before retrace
    
    # Profit Booking & Risk Filters
    "min_price_to_sell": 15.0,     # Minimum required option premium (Rs.)
    "profit_booking_pct": 0.40,    # Book profit when option LTP decays by 40%
    "max_leg_loss_pct": 1.0,       # Leg Stop Loss: 100% loss (2.0x entry price cap)
    
    # Reference Reset Parameters
    "ce_reset_gap": 25.0,
    "pe_reset_gap": 25.0,
    
    # Starting Reference Points (0.0 = use live LTP on first run)
    "ce_start_point": 0.0,
    "pe_start_point": 0.0,

    # Risk Management Parameters (Delta Limits)
    "min_nifty_delta": -100.0,
    "max_nifty_delta": 100.0,
    
    # Black-Scholes Greeks Parameters
    "interest_rate": 10.0,
    "todays_volatility": 14.0,     # Dynamically updated by VIX
    
    # Order Parameters
    "order_type": "MARKET",
    "product_type": "NRML",
    "tag": "WaveScraper",
    
    # Expiry Safety Settings
    "monday_eod_exit": True,         # Auto exit on Monday 3:15 PM before 0-DTE Tuesday
    "zero_dte_dual_mode": True,      # 0-DTE Dual Mode: manage today + sell next expiry at 2x OTM
    "zero_dte_otm_multiplier": 2,    # OTM offset multiplier on 0-DTE day (e.g. 100pt → 200pt)
    "zero_dte_max_consec_sells": 1,  # Reduced consecutive sells on 0-DTE day (extra caution)
    "zero_dte_squareoff_hour": 15,   # Hard square-off hour (15 = 3 PM IST)
    "zero_dte_squareoff_minute": 0,  # Hard square-off minute
    
    # Loop Interval (seconds)
    "poll_interval": 30,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "nifty_wave_state.json")
TRADE_LOG  = os.path.join(SCRIPT_DIR, "nifty_wave_trades.csv")
AUDIT_LOG  = os.path.join(SCRIPT_DIR, "nifty_wave_audit.csv")

# ============================================================
# 2. BLACK-SCHOLES GREEKS ENGINE
# ============================================================
def _norm_cdf(x):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_delta(spot, strike, tte_days, iv_pct, r_pct=10.0, is_call=True):
    """Calculate Black-Scholes Delta for an option leg"""
    if tte_days <= 0 or iv_pct <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if (is_call and spot > strike) else (0.0 if is_call else -1.0)
    
    t = tte_days / 365.0
    sigma = iv_pct / 100.0
    r = r_pct / 100.0
    sqrt_t = math.sqrt(t)
    
    d1 = (math.log(spot / strike) + (r + sigma**2 / 2) * t) / (sigma * sqrt_t)
    return _norm_cdf(d1) if is_call else (_norm_cdf(d1) - 1.0)

# ============================================================
# 3. WAVE SCRAPER STRATEGY ENGINE
# ============================================================
class OpenAlgoWaveEngine:
    def __init__(self, config=CONFIG):
        self.config = config
        self.client = api(api_key=config["api_key"], host=config["host"])
        self.state = self.load_state()
        self.expiry_code = self.get_expiry()
        
        self.open_positions = self.state.get("open_positions", {})
        self.consecutive_ce_sells = self.state.get("consecutive_ce_sells", 0)
        self.consecutive_pe_sells = self.state.get("consecutive_pe_sells", 0)
        
        # 1. Fetch live VIX and dynamically configure parameters if enabled
        if self.config.get("auto_vix_config", True):
            self.fetch_vix_and_apply_config()

        # 2. Initialize reference points
        self._initialize_reference_points()

        # 3. Sync open positions with live broker positionbook
        self.sync_open_positions_with_broker()
        
        logger.info(
            f"Initialized Wave Scraper Engine | Expiry: {self.expiry_code} | "
            f"CE Ref: {self.ce_reference:.2f} | PE Ref: {self.pe_reference:.2f} | "
            f"CE Gap: {self.config['ce_gap']} | PE Gap: {self.config['pe_gap']} | "
            f"Open Positions: {len(self.open_positions)}"
        )

    def fetch_vix_and_apply_config(self):
        """Fetch live INDIA VIX and dynamically set parameters based on VIX matrix"""
        vix = 14.0  # Default fallback VIX
        try:
            res = self.client.quotes(exchange="NSE_INDEX", symbol="INDIA VIX")
            if res and isinstance(res, dict):
                data = res.get("data", res)
                vix_val = float(data.get("ltp") or data.get("last_price", 0.0))
                if vix_val > 0:
                    vix = vix_val
        except Exception as e:
            logger.warning(f"Could not fetch INDIA VIX quote ({e}). Using default VIX 14.0.")

        # VIX Matrix Allocation
        if vix < 12.0:
            # Tier 1: Ultra Low VIX (< 12.0)
            self.config["ce_gap"] = 15.0
            self.config["pe_gap"] = 15.0
            self.config["ce_otm_offset"] = 100
            self.config["pe_otm_offset"] = 100
            self.config["ce_reset_gap"] = 18.0
            self.config["pe_reset_gap"] = 18.0
            self.config["profit_booking_pct"] = 0.50
            self.config["max_active_positions"] = 6
            self.config["todays_volatility"] = 12.0
        elif 12.0 <= vix < 14.5:
            # Tier 2: Normal VIX (12.0 - 14.5)
            self.config["ce_gap"] = 20.0
            self.config["pe_gap"] = 20.0
            self.config["ce_otm_offset"] = 100
            self.config["pe_otm_offset"] = 100
            self.config["ce_reset_gap"] = 25.0
            self.config["pe_reset_gap"] = 25.0
            self.config["profit_booking_pct"] = 0.40
            self.config["max_active_positions"] = 4
            self.config["todays_volatility"] = 13.5
        elif 14.5 <= vix < 17.5:
            # Tier 3: Moderate VIX (14.5 - 17.5)
            self.config["ce_gap"] = 30.0
            self.config["pe_gap"] = 30.0
            self.config["ce_otm_offset"] = 150
            self.config["pe_otm_offset"] = 150
            self.config["ce_reset_gap"] = 35.0
            self.config["pe_reset_gap"] = 35.0
            self.config["profit_booking_pct"] = 0.35
            self.config["max_active_positions"] = 3
            self.config["todays_volatility"] = 16.0
        else:
            # Tier 4: High / Extreme VIX (>= 17.5)
            self.config["ce_gap"] = 40.0
            self.config["pe_gap"] = 40.0
            self.config["ce_otm_offset"] = 200
            self.config["pe_otm_offset"] = 200
            self.config["ce_reset_gap"] = 45.0
            self.config["pe_reset_gap"] = 45.0
            self.config["profit_booking_pct"] = 0.30
            self.config["max_active_positions"] = 2
            self.config["todays_volatility"] = 20.0

        logger.info(
            f"⚡ [DYNAMIC VIX CONFIG] INDIA VIX: {vix:.2f} | "
            f"Gaps: {self.config['ce_gap']}pt | Offsets: {self.config['ce_otm_offset']}pt | "
            f"Profit Tgt: {self.config['profit_booking_pct']*100}% | Max Pos: {self.config['max_active_positions']}"
        )

    def sync_open_positions_with_broker(self):
        """Reconcile local tracked positions with live broker positionbook to purge closed/expired legs"""
        try:
            pos_res = self.client.positionbook()
            positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", []) if isinstance(pos_res, dict) else []
            
            active_broker_symbols = set()
            for pos in positions:
                sym = pos.get("symbol", "")
                qty = int(pos.get("quantity") or pos.get("quantity_total", 0))
                if qty != 0:
                    active_broker_symbols.add(sym)
            
            purged = []
            for sym in list(self.open_positions.keys()):
                if sym not in active_broker_symbols:
                    purged.append(sym)
                    pos_info = self.open_positions.pop(sym, None)
                    if isinstance(pos_info, dict):
                        opt_type = pos_info.get("opt_type")
                        if opt_type == "CE":
                            self.consecutive_ce_sells = max(0, self.consecutive_ce_sells - 1)
                        elif opt_type == "PE":
                            self.consecutive_pe_sells = max(0, self.consecutive_pe_sells - 1)
            
            if purged:
                logger.info(f"Synced with broker positionbook. Purged {len(purged)} closed/expired legs: {purged}")
                self.save_state()
        except Exception as e:
            logger.error(f"Error syncing open positions with broker positionbook: {e}")

    def get_expiry(self):
        """Fetch nearest active options expiry (e.g. 04AUG26)"""
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

    def get_next_expiry(self):
        """Fetch NEXT week's options expiry (index [1] in expiry list, e.g. 11AUG26)"""
        try:
            res = self.client.expiry(
                symbol=self.config["underlying"],
                exchange=self.config["exchange_deriv"],
                instrumenttype="options"
            )
            data = res if isinstance(res, list) else res.get("data", [])
            if len(data) >= 2:
                return data[1].replace("-", "")
            elif len(data) == 1:
                # Fallback: only one expiry available, use it
                logger.warning("Only one expiry available — using current for next expiry fallback.")
                return data[0].replace("-", "")
        except Exception as e:
            logger.error(f"Next expiry fetch error: {e}")
        return ""

    def get_otm_offset(self, base_offset):
        """Return OTM offset, doubled on 0-DTE day for extra safety when rolling to next expiry"""
        if self._is_zero_dte_day():
            return base_offset * self.config.get("zero_dte_otm_multiplier", 2)
        return base_offset

    def _is_zero_dte_day(self):
        """Returns True if today is Tuesday (NIFTY weekly 0-DTE expiry day)"""
        return datetime.now().weekday() == 1  # Monday=0, Tuesday=1

    def get_spot_price(self):
        """Fetch current NIFTY index spot price (NSE_INDEX)"""
        try:
            res = self.client.quotes(exchange=self.config["exchange_index"], symbol=self.config["underlying"])
            if res and isinstance(res, dict):
                data = res.get("data", res)
                return float(data.get("ltp") or data.get("last_price", 0.0))
        except Exception as e:
            logger.error(f"Spot quote fetch error: {e}")
        return 0.0

    def get_option_quote(self, symbol):
        """Fetch option LTP via OpenAlgo quotes API"""
        try:
            res = self.client.quotes(exchange=self.config["exchange_deriv"], symbol=symbol)
            if res and isinstance(res, dict):
                data = res.get("data", res)
                ltp = data.get("ltp") or data.get("last_price")
                if ltp:
                    return float(ltp)
        except Exception as e:
            logger.warning(f"Option quote fetch error for {symbol}: {e}")
        return 0.0

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return {}

    def save_state(self):
        self.state["ce_reference"] = self.ce_reference
        self.state["pe_reference"] = self.pe_reference
        self.state["open_positions"] = self.open_positions
        self.state["consecutive_ce_sells"] = self.consecutive_ce_sells
        self.state["consecutive_pe_sells"] = self.consecutive_pe_sells
        self.state["last_eval"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def log_decision(self, action, reason, details=None):
        logger.info(f"[{action}] {reason} | {details or ''}")
        try:
            file_exists = os.path.exists(AUDIT_LOG)
            with open(AUDIT_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "action", "reason", "details"])
                writer.writerow([datetime.now().isoformat(), action, reason, json.dumps(details or {})])
        except Exception:
            pass

    def log_trade(self, order_id, symbol, action, qty, price, status):
        try:
            file_exists = os.path.exists(TRADE_LOG)
            with open(TRADE_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "order_id", "symbol", "action", "qty", "price", "status"])
                writer.writerow([datetime.now().isoformat(), order_id, symbol, action, qty, price, status])
        except Exception as e:
            logger.error(f"Trade log write error: {e}")

    def _initialize_reference_points(self):
        """Initialize CE and PE tracking reference points"""
        spot = self.get_spot_price()
        
        if self.state.get("ce_reference", 0) > 0:
            self.ce_reference = float(self.state["ce_reference"])
        elif self.config.get("ce_start_point", 0) > 0:
            self.ce_reference = float(self.config["ce_start_point"])
        else:
            self.ce_reference = spot

        if self.state.get("pe_reference", 0) > 0:
            self.pe_reference = float(self.state["pe_reference"])
        elif self.config.get("pe_start_point", 0) > 0:
            self.pe_reference = float(self.config["pe_start_point"])
        else:
            self.pe_reference = spot

    def build_option_symbol(self, strike, opt_type, expiry_code_override=None):
        """Build standard option symbol format (e.g. NIFTY04AUG2624500CE)"""
        expiry = expiry_code_override or self.expiry_code
        if not expiry:
            expiry = self.get_expiry()
            self.expiry_code = expiry
        return f"{self.config['underlying']}{expiry}{strike}{opt_type}"

    def calculate_portfolio_delta(self, spot_price):
        """Calculate aggregate portfolio delta across open positions using Black-Scholes"""
        if spot_price <= 0:
            return 0.0
            
        total_delta = 0.0
        try:
            pos_res = self.client.positionbook()
            positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", []) if isinstance(pos_res, dict) else []
            
            for pos in positions:
                sym = pos.get("symbol", "")
                qty = int(pos.get("quantity") or pos.get("quantity_total", 0))
                if not sym.startswith("NIFTY") or qty == 0:
                    continue
                
                if sym.endswith("FUT"):
                    total_delta += qty
                    continue
                
                import re
                match = re.search(r'(\d+)(CE|PE)$', sym)
                if match:
                    strike = float(match.group(1))
                    is_call = (match.group(2) == "CE")
                    d_val = bs_delta(spot_price, strike, tte_days=7, iv_pct=self.config["todays_volatility"],
                                     r_pct=self.config["interest_rate"], is_call=is_call)
                    total_delta += d_val * qty
        except Exception as e:
            logger.error(f"Error calculating portfolio delta: {e}")
            
        return round(total_delta, 2)

    def find_sellable_option(self, spot, base_offset, opt_type, expiry_code_override=None):
        """Walk strikes outward from base OTM offset towards ATM until finding premium >= min_price_to_sell."""
        min_prem = self.config["min_price_to_sell"]
        current_offset = base_offset
        
        while current_offset >= 0:
            if opt_type == "CE":
                target_strike = int(round((spot + current_offset) / self.config["strike_step"]) * self.config["strike_step"])
            else:
                target_strike = int(round((spot - current_offset) / self.config["strike_step"]) * self.config["strike_step"])
                
            symbol = self.build_option_symbol(target_strike, opt_type, expiry_code_override)
            ltp = self.get_option_quote(symbol)
            
            if ltp >= min_prem:
                return symbol, ltp, target_strike
                
            logger.info(f"Symbol {symbol} premium Rs.{ltp:.1f} < min Rs.{min_prem}. Trying closer strike.")
            current_offset -= self.config["strike_step"]
            
        # Fallback to 1 strike OTM
        if opt_type == "CE":
            target_strike = int(round((spot + 50) / self.config["strike_step"]) * self.config["strike_step"])
        else:
            target_strike = int(round((spot - 50) / self.config["strike_step"]) * self.config["strike_step"])
            
        symbol = self.build_option_symbol(target_strike, opt_type, expiry_code_override)
        ltp = self.get_option_quote(symbol)
        return symbol, ltp, target_strike

    def handle_ce_sell(self, spot, port_delta, expiry_code_override=None, max_consec_override=None):
        """Evaluate selling OTM CE when spot rises above ce_reference + ce_gap.
        On 0-DTE day: uses next-expiry code and doubled OTM offset."""
        if spot - self.ce_reference < self.config["ce_gap"]:
            return

        # Max consecutive sell level cap (reduced on 0-DTE day)
        max_consec = max_consec_override if max_consec_override is not None else self.config["max_consecutive_sells"]
        if self.consecutive_ce_sells >= max_consec:
            self.log_decision("CE_SKIPPED", f"Max consecutive CE sell levels ({max_consec}) reached.")
            return

        # Count active CE positions
        active_ce_count = sum(1 for p in self.open_positions.values() if p.get("opt_type") == "CE")
        if active_ce_count >= self.config["max_active_positions"]:
            self.log_decision("CE_SKIPPED", f"Max active CE positions ({self.config['max_active_positions']}) reached.")
            return

        if port_delta > self.config["max_nifty_delta"]:
            self.log_decision("CE_SKIPPED", f"Delta limit breached ({port_delta} > {self.config['max_nifty_delta']})")
            return

        # Get OTM offset (doubled on 0-DTE day)
        effective_offset = self.get_otm_offset(self.config["ce_otm_offset"])
        symbol, ltp, strike = self.find_sellable_option(spot, effective_offset, "CE", expiry_code_override)
        qty = self.config["ce_quantity"]

        # Shift CE reference point UP
        self.ce_reference += self.config["ce_gap"]
        self.consecutive_ce_sells += 1

        expiry_label = expiry_code_override or self.expiry_code
        mode_tag = "WAVE_CE_SELL_NEXT_EXPIRY" if expiry_code_override else "WAVE_CE_SELL"

        # Execute SELL order
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="SELL",
                exchange=self.config["exchange_deriv"],
                price=0,
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            logger.info(f"Executed {mode_tag}: {symbol} x {qty} @ Rs.{ltp:.2f} | Expiry: {expiry_label} | OTM Offset: {effective_offset}pt")
            self.log_trade(order_id, symbol, "SELL", qty, ltp, res.get("status", "SUCCESS"))
            self.log_decision("CE_EXECUTED", f"Sold {symbol} x {qty} @ Rs.{ltp:.2f}", {"spot": spot, "ce_ref": self.ce_reference, "expiry": expiry_label})
            
            # Store in open positions
            self.open_positions[symbol] = {
                "qty": qty,
                "entry_price": ltp,
                "entry_time": datetime.now().isoformat(),
                "opt_type": "CE",
                "strike": strike,
                "expiry": expiry_label,
            }
        except Exception as e:
            logger.error(f"CE Order placement error: {e}")

    def handle_pe_sell(self, spot, port_delta, expiry_code_override=None, max_consec_override=None):
        """Evaluate selling OTM PE when spot drops below pe_reference - pe_gap.
        On 0-DTE day: uses next-expiry code and doubled OTM offset."""
        if self.pe_reference - spot < self.config["pe_gap"]:
            return

        # Max consecutive sell level cap (reduced on 0-DTE day)
        max_consec = max_consec_override if max_consec_override is not None else self.config["max_consecutive_sells"]
        if self.consecutive_pe_sells >= max_consec:
            self.log_decision("PE_SKIPPED", f"Max consecutive PE sell levels ({max_consec}) reached.")
            return

        # Count active PE positions
        active_pe_count = sum(1 for p in self.open_positions.values() if p.get("opt_type") == "PE")
        if active_pe_count >= self.config["max_active_positions"]:
            self.log_decision("PE_SKIPPED", f"Max active PE positions ({self.config['max_active_positions']}) reached.")
            return

        if port_delta < self.config["min_nifty_delta"]:
            self.log_decision("PE_SKIPPED", f"Delta limit breached ({port_delta} < {self.config['min_nifty_delta']})")
            return

        # Get OTM offset (doubled on 0-DTE day)
        effective_offset = self.get_otm_offset(self.config["pe_otm_offset"])
        symbol, ltp, strike = self.find_sellable_option(spot, effective_offset, "PE", expiry_code_override)
        qty = self.config["pe_quantity"]

        # Shift PE reference point DOWN
        self.pe_reference -= self.config["pe_gap"]
        self.consecutive_pe_sells += 1

        expiry_label = expiry_code_override or self.expiry_code
        mode_tag = "WAVE_PE_SELL_NEXT_EXPIRY" if expiry_code_override else "WAVE_PE_SELL"

        # Execute SELL order
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="SELL",
                exchange=self.config["exchange_deriv"],
                price=0,
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            logger.info(f"Executed {mode_tag}: {symbol} x {qty} @ Rs.{ltp:.2f} | Expiry: {expiry_label} | OTM Offset: {effective_offset}pt")
            self.log_trade(order_id, symbol, "SELL", qty, ltp, res.get("status", "SUCCESS"))
            self.log_decision("PE_EXECUTED", f"Sold {symbol} x {qty} @ Rs.{ltp:.2f}", {"spot": spot, "pe_ref": self.pe_reference, "expiry": expiry_label})
            
            # Store in open positions
            self.open_positions[symbol] = {
                "qty": qty,
                "entry_price": ltp,
                "entry_time": datetime.now().isoformat(),
                "opt_type": "PE",
                "strike": strike,
                "expiry": expiry_label,
            }
        except Exception as e:
            logger.error(f"PE Order placement error: {e}")

    def handle_profit_booking_and_sl(self, spot):
        """Check open positions and BUY back to square off on profit target (decay %) or leg stop-loss breach"""
        to_remove = []
        for symbol, pos in list(self.open_positions.items()):
            ltp = self.get_option_quote(symbol)
            if ltp <= 0:
                continue
                
            entry_price = pos["entry_price"]
            decay_target = entry_price * (1.0 - self.config["profit_booking_pct"])
            sl_target = entry_price * (1.0 + self.config.get("max_leg_loss_pct", 1.0))
            
            is_profit = (ltp <= decay_target)
            is_stoploss = (ltp >= sl_target)
            
            if is_profit or is_stoploss:
                pnl = (entry_price - ltp) * pos["qty"]
                action_tag = "PROFIT_BOOKED" if is_profit else "LEG_STOP_LOSS"
                logger.info(f"[{action_tag}] Triggered for {symbol}! Entry: Rs.{entry_price:.2f} | Current: Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
                try:
                    res = self.client.placeorder(
                        strategy=self.config["tag"],
                        symbol=symbol,
                        action="BUY",
                        exchange=self.config["exchange_deriv"],
                        price=0,
                        pricetype=self.config["order_type"],
                        product=self.config["product_type"],
                        quantity=pos["qty"]
                    )
                    order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
                    self.log_trade(order_id, symbol, "BUY", pos["qty"], ltp, res.get("status", "SUCCESS"))
                    self.log_decision(action_tag, f"Bought back {symbol} x {pos['qty']} @ Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
                    to_remove.append((symbol, pos.get("opt_type")))
                except Exception as e:
                    logger.error(f"Error executing buyback for {symbol}: {e}")
                    
        for sym, opt_type in to_remove:
            self.open_positions.pop(sym, None)
            if opt_type == "CE":
                self.consecutive_ce_sells = max(0, self.consecutive_ce_sells - 1)
            elif opt_type == "PE":
                self.consecutive_pe_sells = max(0, self.consecutive_pe_sells - 1)

    def reset_reference_values(self, spot):
        """Reset reference values on strong market reversals to maintain wave tracking"""
        # CE Reset: Spot drops significantly below CE reference
        if (self.ce_reference - spot) > self.config["ce_reset_gap"]:
            new_val = spot + self.config["ce_gap"]
            logger.info(f"Resetting CE reference from {self.ce_reference:.2f} to {new_val:.2f}")
            self.log_decision("CE_REF_RESET", f"CE reference reset from {self.ce_reference:.2f} to {new_val:.2f}")
            self.ce_reference = new_val
            self.consecutive_ce_sells = 0

        # PE Reset: Spot rises significantly above PE reference
        if (spot - self.pe_reference) > self.config["pe_reset_gap"]:
            new_val = spot - self.config["pe_gap"]
            logger.info(f"Resetting PE reference from {self.pe_reference:.2f} to {new_val:.2f}")
            self.log_decision("PE_REF_RESET", f"PE reference reset from {self.pe_reference:.2f} to {new_val:.2f}")
            self.pe_reference = new_val
            self.consecutive_pe_sells = 0

    def check_monday_eod_and_tuesday_expiry(self):
        """
        Expiry Rule: Tuesday is Nifty 0-DTE Expiry Day.

        Monday after 3:15 PM:
          → Square off ALL open positions (current expiry cleanup before 0-DTE day).
          → Pause new entries.

        Tuesday (0-DTE Day) — DUAL MODE:
          MODE 1 (MANAGE): Continue managing current-expiry open positions
                           (profit booking, leg SL, reference resets).
          MODE 2 (BUILD):  Accept new sell entries ONLY on NEXT expiry at 2x OTM offset.
          At 3:00 PM:       Hard square-off ALL current-expiry positions.

        Returns: (block_new_entries_current_expiry, next_expiry_code_for_new_entries)
        """
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Tuesday=1

        # ── Monday EOD Exit (3:15 PM onwards) ─────────────────────────────────
        if weekday == 0 and now.hour == 15 and now.minute >= 15 and self.config.get("monday_eod_exit", True):
            if self.open_positions:
                logger.info("⚡ Monday 3:15 PM EOD Exit Triggered — Squaring off all open positions before Tuesday 0-DTE.")
                for symbol, pos in list(self.open_positions.items()):
                    try:
                        ltp = self.get_option_quote(symbol)
                        res = self.client.placeorder(
                            strategy=self.config["tag"],
                            symbol=symbol,
                            action="BUY",
                            exchange=self.config["exchange_deriv"],
                            price=0,
                            pricetype=self.config["order_type"],
                            product=self.config["product_type"],
                            quantity=pos["qty"]
                        )
                        order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
                        self.log_trade(order_id, symbol, "BUY", pos["qty"], ltp, res.get("status", "SUCCESS"))
                        self.log_decision("MONDAY_EOD_EXIT", f"Closed {symbol} x {pos['qty']} @ Rs.{ltp:.2f}")
                    except Exception as e:
                        logger.error(f"Monday EOD exit error for {symbol}: {e}")
                self.open_positions.clear()
                self.consecutive_ce_sells = 0
                self.consecutive_pe_sells = 0
                self.save_state()
            # Block all entries on Monday EOD
            return True, None

        # ── Tuesday 0-DTE Dual Mode ────────────────────────────────────────────
        if weekday == 1 and self.config.get("zero_dte_dual_mode", True):
            sq_hour = self.config.get("zero_dte_squareoff_hour", 15)
            sq_min  = self.config.get("zero_dte_squareoff_minute", 0)

            # Hard Square-Off at 3:00 PM — current expiry only
            if now.hour > sq_hour or (now.hour == sq_hour and now.minute >= sq_min):
                current_expiry_positions = {
                    sym: pos for sym, pos in self.open_positions.items()
                    if pos.get("expiry", self.expiry_code) == self.expiry_code
                }
                if current_expiry_positions:
                    logger.warning("⚡ [0-DTE SQUAREOFF] 3:00 PM reached — Hard closing all current-expiry positions!")
                    for symbol, pos in list(current_expiry_positions.items()):
                        try:
                            ltp = self.get_option_quote(symbol)
                            res = self.client.placeorder(
                                strategy=self.config["tag"],
                                symbol=symbol,
                                action="BUY",
                                exchange=self.config["exchange_deriv"],
                                price=0,
                                pricetype=self.config["order_type"],
                                product=self.config["product_type"],
                                quantity=pos["qty"]
                            )
                            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
                            self.log_trade(order_id, symbol, "BUY", pos["qty"], ltp, res.get("status", "SUCCESS"))
                            self.log_decision("ZERO_DTE_SQUAREOFF", f"Closed {symbol} x {pos['qty']} @ Rs.{ltp:.2f}")
                            self.open_positions.pop(symbol, None)
                        except Exception as e:
                            logger.error(f"0-DTE square-off error for {symbol}: {e}")
                    self.save_state()
                # After 3 PM on Tuesday: block all new entries, next expiry sells also stop
                return True, None

            # Before 3 PM on Tuesday — Dual Mode Active
            next_expiry = self.get_next_expiry()
            if not next_expiry:
                logger.warning("Could not fetch next expiry. Falling back to standard mode.")
                return False, None

            logger.info(
                f"⚡ [0-DTE DUAL MODE] Managing today ({self.expiry_code}) + Building next expiry ({next_expiry}) "
                f"at {self.config.get('zero_dte_otm_multiplier', 2)}x OTM offset."
            )
            # Block entries for today's expiry, allow entries for next expiry
            return True, next_expiry

        return False, None

    def execute_tick(self):
        """Main strategy tick loop.

        On regular days: Standard wave selling on current expiry.
        On Tuesday 0-DTE day (Dual Mode):
          - Current expiry: Only manage + profit-book + leg SL. No new entries.
          - Next expiry: New sells allowed at 2x OTM offset with reduced consecutive sells cap.
          - At 3:00 PM: Hard square-off all current expiry positions.
        """
        spot = self.get_spot_price()
        if spot <= 0:
            logger.warning("Invalid spot price. Retrying...")
            return

        # Always sync with live broker positionbook
        self.sync_open_positions_with_broker()

        # Check Monday EOD exit / Tuesday 0-DTE Dual Mode
        block_current_entries, next_expiry_code = self.check_monday_eod_and_tuesday_expiry()

        port_delta = self.calculate_portfolio_delta(spot)
        mode_str = (
            f"0-DTE DUAL MODE — Next Expiry: {next_expiry_code}" if next_expiry_code
            else ("RESTRICTED" if block_current_entries else "NORMAL")
        )
        logger.info(
            f"[WAVE TICK] Spot: {spot:.2f} | CE Ref: {self.ce_reference:.2f} | PE Ref: {self.pe_reference:.2f} | "
            f"Delta: {port_delta:.2f} | Open Positions: {len(self.open_positions)} | Mode: {mode_str}"
        )

        if not block_current_entries:
            # NORMAL DAY: sell entries on current expiry at standard OTM offset
            self.handle_ce_sell(spot, port_delta)
            self.handle_pe_sell(spot, port_delta)

        elif next_expiry_code:
            # 0-DTE DUAL MODE: sell entries on NEXT expiry at 2x OTM, reduced consecutive sells
            zero_dte_max_consec = self.config.get("zero_dte_max_consec_sells", 1)
            self.handle_ce_sell(spot, port_delta, expiry_code_override=next_expiry_code, max_consec_override=zero_dte_max_consec)
            self.handle_pe_sell(spot, port_delta, expiry_code_override=next_expiry_code, max_consec_override=zero_dte_max_consec)

        # 2. Profit booking and leg stop-loss (always runs — manages all open positions)
        self.handle_profit_booking_and_sl(spot)

        # 3. Reference resets (always runs)
        self.reset_reference_values(spot)

        # 4. Save state
        self.save_state()


def run():
    engine = OpenAlgoWaveEngine(CONFIG)
    logger.info("Starting Wave Scraper Options Master Strategy Loop...")
    while True:
        try:
            engine.execute_tick()
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in strategy loop: {e}", exc_info=True)
        
        time.sleep(CONFIG["poll_interval"])


if __name__ == "__main__":
    run()
