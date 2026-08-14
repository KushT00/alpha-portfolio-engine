"""
Nifty Survivor Options Strategy — OpenAlgo Platform Master Version
===================================================================
Systematic Index-Movement Option Selling Strategy for OpenAlgo SDK:
  - Dynamic VIX Auto-Config: Fetches INDIA VIX on startup and sets gaps, offsets, and position multiplier limits.
  - Put (PE) Trading: Triggered when NIFTY moves UP beyond pe_gap threshold (+20 pts). Sells OTM Puts (~200pt OTM).
  - Call (CE) Trading: Triggered when NIFTY moves DOWN beyond ce_gap threshold (-20 pts). Sells OTM Calls (~200pt OTM).
  - Dynamic Position Scaling: Scales position quantity based on price movement magnitude (sell_multiplier).
  - Risk Multiplier Safeguard: Rejects trade if sell_multiplier > sell_multiplier_threshold (5).
  - Dynamic Premium Filter: Ensures sold option premium >= min_price_to_sell (Rs. 15). Steps closer to ATM if lower.
  - Reference Reset Mechanism: Adjusts PE/CE reference values on favorable market reversals to prevent reference drift.
  - Expiry Safety: Auto-squares off positions by Monday 3:15 PM EOD.
  - 0-DTE Dual Mode (Tuesday): Manages today's expiry + hard-closes by 3:00 PM AND executes new sells on NEXT expiry at 2x OTM offset.
"""

import os
import json
import time
import csv
import logging
from datetime import datetime, date
from openalgo import api

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SurvivorMaster")

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
    
    # Default Gap Parameters (Overridden dynamically by VIX)
    "pe_gap": 20.0,
    "ce_gap": 20.0,
    
    # Position Quantities (1 Lot = 65)
    "pe_quantity": 65,
    "ce_quantity": 65,
    
    # Strike Distance Parameters (OTM Strike Offset in points)
    "pe_symbol_gap": 200,
    "ce_symbol_gap": 200,
    
    # Risk Management Parameters
    "min_price_to_sell": 15.0,        # Minimum required option premium (Rs.)
    "sell_multiplier_threshold": 5,   # Max allowed position scaling multiplier
    
    # Reference Reset Parameters
    "pe_reset_gap": 30.0,
    "ce_reset_gap": 30.0,
    
    # Starting Reference Points (0 = use live LTP)
    "pe_start_point": 0.0,
    "ce_start_point": 0.0,
    
    # Order Parameters
    "trans_type": "SELL",
    "order_type": "MARKET",
    "product_type": "NRML",
    "tag": "Survivor",
    
    # Expiry Safety Settings
    "monday_eod_exit": True,         # Auto exit on Monday 3:15 PM before 0-DTE Tuesday
    "zero_dte_dual_mode": True,      # 0-DTE Dual Mode: manage today + sell next expiry at 2x OTM
    "zero_dte_otm_multiplier": 2,    # OTM offset multiplier on 0-DTE day (e.g. 200pt → 400pt)
    "zero_dte_squareoff_hour": 15,   # Hard square-off hour (15 = 3 PM IST)
    "zero_dte_squareoff_minute": 0,  # Hard square-off minute
    
    # Loop Polling Interval (seconds)
    "poll_interval": 30,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "nifty_survivor_state.json")
TRADE_LOG  = os.path.join(SCRIPT_DIR, "nifty_survivor_trades.csv")
AUDIT_LOG  = os.path.join(SCRIPT_DIR, "nifty_survivor_audit.csv")

# ============================================================
# 2. SURVIVOR STRATEGY ENGINE
# ============================================================
class OpenAlgoSurvivorEngine:
    def __init__(self, config=CONFIG):
        self.config = config
        self.client = api(api_key=config["api_key"], host=config["host"])
        self.state = self.load_state()
        self.expiry_code = self.get_expiry()
        
        self.pe_reset_gap_flag = self.state.get("pe_reset_gap_flag", 0)
        self.ce_reset_gap_flag = self.state.get("ce_reset_gap_flag", 0)
        
        # 1. Fetch live VIX and dynamically configure parameters if enabled
        if self.config.get("auto_vix_config", True):
            self.fetch_vix_and_apply_config()

        # 2. Initialize reference points
        self._initialize_reference_points()
        
        logger.info(
            f"Initialized Survivor Engine | Expiry: {self.expiry_code} | "
            f"PE Ref: {self.nifty_pe_last_value:.2f} | CE Ref: {self.nifty_ce_last_value:.2f} | "
            f"PE Gap: {self.config['pe_gap']} | CE Gap: {self.config['ce_gap']}"
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

        # VIX Matrix Allocation for Survivor
        if vix < 12.0:
            self.config["pe_gap"] = 15.0
            self.config["ce_gap"] = 15.0
            self.config["pe_symbol_gap"] = 150
            self.config["ce_symbol_gap"] = 150
            self.config["pe_reset_gap"] = 20.0
            self.config["ce_reset_gap"] = 20.0
        elif 12.0 <= vix < 14.5:
            self.config["pe_gap"] = 20.0
            self.config["ce_gap"] = 20.0
            self.config["pe_symbol_gap"] = 200
            self.config["ce_symbol_gap"] = 200
            self.config["pe_reset_gap"] = 30.0
            self.config["ce_reset_gap"] = 30.0
        elif 14.5 <= vix < 17.5:
            self.config["pe_gap"] = 30.0
            self.config["ce_gap"] = 30.0
            self.config["pe_symbol_gap"] = 250
            self.config["ce_symbol_gap"] = 250
            self.config["pe_reset_gap"] = 40.0
            self.config["ce_reset_gap"] = 40.0
        else:
            self.config["pe_gap"] = 40.0
            self.config["ce_gap"] = 40.0
            self.config["pe_symbol_gap"] = 300
            self.config["ce_symbol_gap"] = 300
            self.config["pe_reset_gap"] = 50.0
            self.config["ce_reset_gap"] = 50.0

        logger.info(
            f"⚡ [SURVIVOR VIX CONFIG] INDIA VIX: {vix:.2f} | "
            f"Gaps: {self.config['pe_gap']}pt | Symbol Offsets: {self.config['pe_symbol_gap']}pt"
        )

    def get_expiry(self):
        """Fetch nearest active weekly/monthly options expiry (e.g. 04AUG26)"""
        try:
            res = self.client.expiry(symbol=self.config["underlying"], exchange=self.config["exchange_deriv"], instrumenttype="options")
            data = res if isinstance(res, list) else res.get("data", [])
            if data:
                return data[0].replace("-", "")
        except Exception as e:
            logger.error(f"Expiry fetch error: {e}")
        return ""

    def get_next_expiry(self):
        """Fetch NEXT week's options expiry (index [1] in expiry list, e.g. 11AUG26)"""
        try:
            res = self.client.expiry(symbol=self.config["underlying"], exchange=self.config["exchange_deriv"], instrumenttype="options")
            data = res if isinstance(res, list) else res.get("data", [])
            if len(data) >= 2:
                return data[1].replace("-", "")
            elif len(data) == 1:
                logger.warning("Only one expiry available — using current for next expiry fallback.")
                return data[0].replace("-", "")
        except Exception as e:
            logger.error(f"Next expiry fetch error: {e}")
        return ""

    def get_otm_offset(self, base_offset):
        """Return OTM offset, doubled on 0-DTE day for extra safety when rolling to next expiry"""
        if self._is_zero_dte_day():
            return int(base_offset * self.config.get("zero_dte_otm_multiplier", 2))
        return base_offset

    def _is_zero_dte_day(self):
        """Returns True if today is Tuesday (NIFTY weekly 0-DTE expiry day)"""
        return datetime.now().weekday() == 1  # Monday=0, Tuesday=1

    def get_spot_price(self):
        """Fetch current NIFTY index spot price"""
        try:
            res = self.client.quotes(exchange=self.config["exchange_index"], symbol=self.config["underlying"])
            if res and isinstance(res, dict):
                data = res.get("data", res)
                return float(data.get("ltp") or data.get("last_price", 0.0))
        except Exception as e:
            logger.error(f"Spot quote fetch error: {e}")
        return 0.0

    def get_option_quote(self, symbol):
        """Fetch option LTP via OpenAlgo API"""
        try:
            res = self.client.quotes(exchange=self.config["exchange_deriv"], symbol=symbol)
            if res and isinstance(res, dict):
                data = res.get("data", res)
                ltp = data.get("ltp") or data.get("last_price")
                if ltp:
                    return float(ltp)
        except Exception:
            pass
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
        self.state["nifty_pe_last_value"] = self.nifty_pe_last_value
        self.state["nifty_ce_last_value"] = self.nifty_ce_last_value
        self.state["pe_reset_gap_flag"] = self.pe_reset_gap_flag
        self.state["ce_reset_gap_flag"] = self.ce_reset_gap_flag
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
        """Initialize PE and CE tracking reference points"""
        spot = self.get_spot_price()
        
        # PE Reference
        if self.state.get("nifty_pe_last_value", 0) > 0:
            self.nifty_pe_last_value = float(self.state["nifty_pe_last_value"])
        elif self.config["pe_start_point"] > 0:
            self.nifty_pe_last_value = float(self.config["pe_start_point"])
        else:
            self.nifty_pe_last_value = spot

        # CE Reference
        if self.state.get("nifty_ce_last_value", 0) > 0:
            self.nifty_ce_last_value = float(self.state["nifty_ce_last_value"])
        elif self.config["ce_start_point"] > 0:
            self.nifty_ce_last_value = float(self.config["ce_start_point"])
        else:
            self.nifty_ce_last_value = spot

    def build_option_symbol(self, strike, opt_type, expiry_code_override=None):
        """Build standard option symbol format: NIFTY04AUG2624500CE"""
        expiry = expiry_code_override or self.expiry_code
        if not expiry:
            expiry = self.get_expiry()
            self.expiry_code = expiry
        return f"{self.config['underlying']}{expiry}{int(strike)}{opt_type}"

    def check_sell_multiplier_breach(self, sell_multiplier):
        """Safeguard: check if position multiplier breaches risk threshold"""
        if sell_multiplier > self.config["sell_multiplier_threshold"]:
            logger.warning(f"Multiplier breach! {sell_multiplier} > max ({self.config['sell_multiplier_threshold']})")
            return True
        return False

    # ─── Delta Rebalancing Engine (Survivor Binary Search Anchor Adjustment) ───

    def compute_expiry_delta_at_spot(self, positions, hypothetical_spot):
        """Sum delta of all open positions at a hypothetical spot price."""
        total_delta = 0.0
        for pos in positions:
            sym = pos.get("symbol", "")
            qty = int(pos.get("quantity") or pos.get("quantity_total", 0))
            if not sym.startswith("NIFTY") or qty == 0 or sym.endswith("FUT"):
                continue

            m = re.search(r'(\d{4,5})(CE|PE)$', sym)
            if m:
                strike = float(m.group(1))
                is_call = (m.group(2) == "CE")
                t_days = 7.0
                sigma = 14.0 / 100.0
                r = 0.10
                t = t_days / 365.0
                sqrt_t = math.sqrt(t)
                d1 = (math.log(hypothetical_spot / strike) + (r + sigma**2 / 2.0) * t) / (sigma * sqrt_t)
                leg_delta = (0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))) if is_call else (0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0)
                total_delta += leg_delta * qty

        return total_delta

    def find_spot_for_target_delta(self, positions, target_delta, search_center, search_half_width=3000.0, tolerance=1.0):
        """Binary search (Bisection method) for spot price where portfolio delta equals target_delta."""
        low = search_center - search_half_width
        high = search_center + search_half_width

        d_low = self.compute_expiry_delta_at_spot(positions, low)
        d_high = self.compute_expiry_delta_at_spot(positions, high)

        if not (min(d_low, d_high) <= target_delta <= max(d_low, d_high)):
            return None

        for _ in range(60):
            mid = (low + high) / 2.0
            d_mid = self.compute_expiry_delta_at_spot(positions, mid)
            if abs(d_mid - target_delta) < tolerance:
                return mid
            if d_mid > target_delta:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

    def _rebalance_anchor_after_sell(self, sell_side, trigger_spot):
        """Background thread: 3 seconds after a sell, update opposite anchor via binary search delta targeting."""
        time.sleep(3)
        try:
            pos_res = self.client.positionbook()
            positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", []) if isinstance(pos_res, dict) else []

            if sell_side == "PE":
                target_ce_delta = self.compute_expiry_delta_at_spot(positions, self.nifty_ce_last_value)
                computed_new_ce = self.find_spot_for_target_delta(positions, target_ce_delta, trigger_spot)
                if computed_new_ce is not None:
                    capped = min(computed_new_ce, trigger_spot - self.config["ce_gap"])
                    old_ce = self.nifty_ce_last_value
                    shift = abs(capped - old_ce)
                    if shift >= 5.0 and abs(capped - trigger_spot) < abs(old_ce - trigger_spot):
                        self.nifty_ce_last_value = capped
                        logger.info(f"⚡ [DELTA REBALANCE] PE Sell → CE Anchor shifted: {old_ce:.0f} → {capped:.0f} (Spot: {trigger_spot:.0f})")
                        self.save_state()

            else:  # "CE"
                target_pe_delta = self.compute_expiry_delta_at_spot(positions, self.nifty_pe_last_value)
                computed_new_pe = self.find_spot_for_target_delta(positions, target_pe_delta, trigger_spot)
                if computed_new_pe is not None:
                    capped = max(computed_new_pe, trigger_spot + self.config["pe_gap"])
                    old_pe = self.nifty_pe_last_value
                    shift = abs(capped - old_pe)
                    if shift >= 5.0 and abs(capped - trigger_spot) < abs(old_pe - trigger_spot):
                        self.nifty_pe_last_value = capped
                        logger.info(f"⚡ [DELTA REBALANCE] CE Sell → PE Anchor shifted: {old_pe:.0f} → {capped:.0f} (Spot: {trigger_spot:.0f})")
                        self.save_state()

        except Exception as e:
            logger.error(f"Delta rebalance thread error after {sell_side} sell: {e}")

    def handle_pe_trade(self, current_price, expiry_code_override=None):
        """Evaluate Put (PE) option selling on upward spot move.
        On 0-DTE day: uses next-expiry code and doubled OTM offset."""
        if current_price <= self.nifty_pe_last_value:
            return

        price_diff = round(current_price - self.nifty_pe_last_value, 0)
        if price_diff > self.config["pe_gap"]:
            sell_multiplier = int(price_diff / self.config["pe_gap"])
            
            if self.check_sell_multiplier_breach(sell_multiplier):
                self.log_decision("PE_SKIPPED", f"Multiplier breach: {sell_multiplier} > {self.config['sell_multiplier_threshold']}")
                return

            total_qty = sell_multiplier * self.config["pe_quantity"]
            # Get OTM offset (doubled on 0-DTE day when selling next expiry)
            effective_gap = self.get_otm_offset(self.config["pe_symbol_gap"])
            temp_gap = effective_gap
            expiry_label = expiry_code_override or self.expiry_code
            
            # Strike selection with premium filter
            symbol_to_sell = None
            price_found = 0.0
            
            while temp_gap >= 0:
                target_strike = int(round((current_price - temp_gap) / self.config["strike_step"]) * self.config["strike_step"])
                candidate_sym = self.build_option_symbol(target_strike, "PE", expiry_code_override)
                ltp = self.get_option_quote(candidate_sym)
                
                if ltp >= self.config["min_price_to_sell"]:
                    symbol_to_sell = candidate_sym
                    price_found = ltp
                    break
                
                logger.info(f"Symbol {candidate_sym} premium Rs.{ltp:.1f} < min Rs.{self.config['min_price_to_sell']}. Trying closer strike.")
                temp_gap -= self.config["strike_step"]

            if not symbol_to_sell:
                # Fallback to nearest ATM
                target_strike = int(round((current_price - 50) / self.config["strike_step"]) * self.config["strike_step"])
                symbol_to_sell = self.build_option_symbol(target_strike, "PE", expiry_code_override)
                price_found = self.get_option_quote(symbol_to_sell)

            # Update reference point
            self.nifty_pe_last_value += self.config["pe_gap"] * sell_multiplier
            self.pe_reset_gap_flag = 1

            mode_tag = "SURVIVOR_PE_SELL_NEXT_EXPIRY" if expiry_code_override else "SURVIVOR_PE_SELL"

            # Execute Order via OpenAlgo
            try:
                res = self.client.placeorder(
                    strategy=self.config["tag"],
                    symbol=symbol_to_sell,
                    action="SELL",
                    exchange=self.config["exchange_deriv"],
                    price=0,  # Market order
                    pricetype=self.config["order_type"],
                    product=self.config["product_type"],
                    quantity=total_qty
                )
                order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
                logger.info(f"Executed {mode_tag}: {symbol_to_sell} x {total_qty} @ Rs.{price_found:.2f} | Expiry: {expiry_label} | OTM: {effective_gap}pt")
                self.log_trade(order_id, symbol_to_sell, "SELL", total_qty, price_found, res.get("status", "SUCCESS"))
                self.log_decision("PE_EXECUTED", f"Sold {symbol_to_sell} x {total_qty} @ Rs.{price_found:.2f}", {"spot": current_price, "mult": sell_multiplier, "expiry": expiry_label})

                # Trigger background thread: 3 seconds post-sell delta rebalance of CE anchor
                threading.Thread(
                    target=self._rebalance_anchor_after_sell,
                    args=("PE", current_price),
                    daemon=True,
                    name="delta_rebalance_PE"
                ).start()
            except Exception as e:
                logger.error(f"PE Order placement error: {e}")

    def handle_ce_trade(self, current_price, expiry_code_override=None):
        """Evaluate Call (CE) option selling on downward spot move.
        On 0-DTE day: uses next-expiry code and doubled OTM offset."""
        if current_price >= self.nifty_ce_last_value:
            return

        price_diff = round(self.nifty_ce_last_value - current_price, 0)
        if price_diff > self.config["ce_gap"]:
            sell_multiplier = int(price_diff / self.config["ce_gap"])
            
            if self.check_sell_multiplier_breach(sell_multiplier):
                self.log_decision("CE_SKIPPED", f"Multiplier breach: {sell_multiplier} > {self.config['sell_multiplier_threshold']}")
                return

            total_qty = sell_multiplier * self.config["ce_quantity"]
            # Get OTM offset (doubled on 0-DTE day when selling next expiry)
            effective_gap = self.get_otm_offset(self.config["ce_symbol_gap"])
            temp_gap = effective_gap
            expiry_label = expiry_code_override or self.expiry_code
            
            # Strike selection with premium filter
            symbol_to_sell = None
            price_found = 0.0
            
            while temp_gap >= 0:
                target_strike = int(round((current_price + temp_gap) / self.config["strike_step"]) * self.config["strike_step"])
                candidate_sym = self.build_option_symbol(target_strike, "CE", expiry_code_override)
                ltp = self.get_option_quote(candidate_sym)
                
                if ltp >= self.config["min_price_to_sell"]:
                    symbol_to_sell = candidate_sym
                    price_found = ltp
                    break
                
                logger.info(f"Symbol {candidate_sym} premium Rs.{ltp:.1f} < min Rs.{self.config['min_price_to_sell']}. Trying closer strike.")
                temp_gap -= self.config["strike_step"]

            if not symbol_to_sell:
                # Fallback to nearest ATM
                target_strike = int(round((current_price + 50) / self.config["strike_step"]) * self.config["strike_step"])
                symbol_to_sell = self.build_option_symbol(target_strike, "CE", expiry_code_override)
                price_found = self.get_option_quote(symbol_to_sell)

            # Update reference point
            self.nifty_ce_last_value -= self.config["ce_gap"] * sell_multiplier
            self.ce_reset_gap_flag = 1

            mode_tag = "SURVIVOR_CE_SELL_NEXT_EXPIRY" if expiry_code_override else "SURVIVOR_CE_SELL"

            # Execute Order via OpenAlgo
            try:
                res = self.client.placeorder(
                    strategy=self.config["tag"],
                    symbol=symbol_to_sell,
                    action="SELL",
                    exchange=self.config["exchange_deriv"],
                    price=0,  # Market order
                    pricetype=self.config["order_type"],
                    product=self.config["product_type"],
                    quantity=total_qty
                )
                order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
                logger.info(f"Executed {mode_tag}: {symbol_to_sell} x {total_qty} @ Rs.{price_found:.2f} | Expiry: {expiry_label} | OTM: {effective_gap}pt")
                self.log_trade(order_id, symbol_to_sell, "SELL", total_qty, price_found, res.get("status", "SUCCESS"))
                self.log_decision("CE_EXECUTED", f"Sold {symbol_to_sell} x {total_qty} @ Rs.{price_found:.2f}", {"spot": current_price, "mult": sell_multiplier, "expiry": expiry_label})

                # Trigger background thread: 3 seconds post-sell delta rebalance of PE anchor
                threading.Thread(
                    target=self._rebalance_anchor_after_sell,
                    args=("CE", current_price),
                    daemon=True,
                    name="delta_rebalance_CE"
                ).start()
            except Exception as e:
                logger.error(f"CE Order placement error: {e}")

    def reset_reference_values(self, current_price):
        """Reset PE/CE reference values when market moves favorably in reverse"""
        # PE Reset: spot drops significantly below PE reference
        if (self.nifty_pe_last_value - current_price) > self.config["pe_reset_gap"] and self.pe_reset_gap_flag:
            new_val = current_price + self.config["pe_reset_gap"]
            logger.info(f"Resetting PE reference from {self.nifty_pe_last_value} to {new_val}")
            self.log_decision("PE_RESET", f"PE reference adjusted from {self.nifty_pe_last_value} to {new_val}")
            self.nifty_pe_last_value = new_val

        # CE Reset: spot rises significantly above CE reference
        if (current_price - self.nifty_ce_last_value) > self.config["ce_reset_gap"] and self.ce_reset_gap_flag:
            new_val = current_price - self.config["ce_reset_gap"]
            logger.info(f"Resetting CE reference from {self.nifty_ce_last_value} to {new_val}")
            self.log_decision("CE_RESET", f"CE reference adjusted from {self.nifty_ce_last_value} to {new_val}")
            self.nifty_ce_last_value = new_val

    def check_monday_eod_and_tuesday_expiry(self):
        """
        Expiry Rule: Tuesday is Nifty 0-DTE Expiry Day.

        Monday after 3:15 PM:
          → Pause new entries (log condition).

        Tuesday (0-DTE Day) — DUAL MODE:
          MODE 1 (MANAGE): Existing positions continue getting profit-booked and reference-reset.
          MODE 2 (BUILD):  New sell entries ONLY on NEXT expiry at 2x OTM offset.
          At 3:00 PM:      Hard square-off no longer needed for Survivor (positions are far OTM),
                           but new entries are blocked after 3 PM.

        Returns: (block_current_entries, next_expiry_code)
        """
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Tuesday=1

        # ── Monday EOD Exit (3:15 PM onwards) ─────────────────────────────────
        if weekday == 0 and now.hour == 15 and now.minute >= 15 and self.config.get("monday_eod_exit", True):
            logger.info("⚡ Monday 3:15 PM EOD Triggered — Survivor Strategy Pausing New Entries for Tuesday 0-DTE.")
            return True, None

        # ── Tuesday 0-DTE Dual Mode ────────────────────────────────────────────
        if weekday == 1 and self.config.get("zero_dte_dual_mode", True):
            sq_hour = self.config.get("zero_dte_squareoff_hour", 15)
            sq_min  = self.config.get("zero_dte_squareoff_minute", 0)

            if now.hour > sq_hour or (now.hour == sq_hour and now.minute >= sq_min):
                logger.info("⚡ [0-DTE POST 3PM] Survivor blocking all new entries. Managing open positions only.")
                return True, None

            # Before 3 PM — Dual Mode Active
            next_expiry = self.get_next_expiry()
            if not next_expiry:
                logger.warning("Could not fetch next expiry. Falling back to standard mode.")
                return False, None

            logger.info(
                f"⚡ [0-DTE DUAL MODE] Survivor managing today ({self.expiry_code}) + Building next expiry ({next_expiry}) "
                f"at {self.config.get('zero_dte_otm_multiplier', 2)}x OTM offset."
            )
            return True, next_expiry

        return False, None

    def execute_tick(self):
        """Main tick iteration function.

        On regular days: Standard survivor selling on current expiry.
        On Tuesday 0-DTE day (Dual Mode):
          - Current expiry: Only manage existing positions + reference resets. No new entries.
          - Next expiry: New sells allowed at 2x OTM offset.
          - After 3:00 PM: Block all new entries.
        """
        current_price = self.get_spot_price()
        if current_price <= 0:
            logger.warning("Invalid spot price. Retrying...")
            return

        block_current_entries, next_expiry_code = self.check_monday_eod_and_tuesday_expiry()

        mode_str = (
            f"0-DTE DUAL MODE — Next Expiry: {next_expiry_code}" if next_expiry_code
            else ("RESTRICTED" if block_current_entries else "NORMAL")
        )
        logger.info(f"[SURVIVOR TICK] Spot: {current_price:.2f} | PE Ref: {self.nifty_pe_last_value:.2f} | CE Ref: {self.nifty_ce_last_value:.2f} | Mode: {mode_str}")
        
        if not block_current_entries:
            # NORMAL DAY: sell entries on current expiry
            self.handle_pe_trade(current_price)
            self.handle_ce_trade(current_price)
            self.reset_reference_values(current_price)
            self.save_state()

        elif next_expiry_code:
            # 0-DTE DUAL MODE: sell entries on NEXT expiry at 2x OTM
            self.handle_pe_trade(current_price, expiry_code_override=next_expiry_code)
            self.handle_ce_trade(current_price, expiry_code_override=next_expiry_code)
            self.reset_reference_values(current_price)
            self.save_state()


def run():
    engine = OpenAlgoSurvivorEngine(CONFIG)
    logger.info("Starting Survivor Options Strategy Master Loop...")
    while True:
        try:
            engine.execute_tick()
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in survivor strategy loop: {e}", exc_info=True)
        
        time.sleep(CONFIG["poll_interval"])


if __name__ == "__main__":
    run()
