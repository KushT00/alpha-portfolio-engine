"""
Nifty Wave Scraper Strategy — OpenAlgo Platform Version
========================================================
Non-Directional Options Wave Scraper Strategy (Mean Reversion):
  - Reference Price: NIFTY SPOT (NSE_INDEX:NIFTY).
  - CE Selling Trigger: When Spot rises by ce_gap (+20 pts) above ce_reference, sells OTM Call (Spot + ce_otm_offset).
  - PE Selling Trigger: When Spot falls by pe_gap (-20 pts) below pe_reference, sells OTM Put (Spot - pe_otm_offset).
  - Profit Booking (Square-Off): Automatically buys back short options when option LTP decays by profit_booking_pct (e.g. 40% decay) or on market retracement.
  - Premium Filter: Ensures sold option premium >= min_price_to_sell (Rs. 15). Steps closer to ATM if lower.
  - Portfolio Delta Gate: Calculates Black-Scholes portfolio delta; pauses CE/PE sells if delta breaches thresholds (-100 / +100).
  - Reference Reset: Adjusts ce_reference/pe_reference on strong market reversals to maintain wave responsiveness.
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
    
    # Core Instrument Parameters
    "underlying": "NIFTY",
    "exchange_index": "NSE_INDEX",
    "exchange_deriv": "NFO",
    "lot_size": 65,                # Nifty Lot Size is 65
    "strike_step": 50,             # Nifty Strike Step is 50
    
    # Wave Trigger Thresholds (Spot movement in points to trigger entry)
    "ce_gap": 20.0,                # Spot rise threshold to sell OTM CE (20 pts)
    "pe_gap": 20.0,                # Spot fall threshold to sell OTM PE (20 pts)
    
    # OTM Offset Parameters (Distance from Spot to target strike in points)
    "ce_otm_offset": 100,          # Sell CE strike ~100pts above Spot
    "pe_otm_offset": 100,          # Sell PE strike ~100pts below Spot
    
    # Position Quantities
    "ce_quantity": 65,             # 1 Lot = 65
    "pe_quantity": 65,             # 1 Lot = 65
    "max_active_positions": 4,     # Max concurrent open short positions per side
    
    # Profit Booking & Risk Filters
    "min_price_to_sell": 15.0,     # Minimum required option premium (Rs.)
    "profit_booking_pct": 0.40,    # Book profit when option LTP decays by 40%
    
    # Reference Reset Parameters (Adjust reference on large opposite moves)
    "ce_reset_gap": 25.0,          # Reset CE ref if spot drops 25pts below ref
    "pe_reset_gap": 25.0,          # Reset PE ref if spot rises 25pts above ref
    
    # Starting Reference Points (0.0 = use live LTP on first run)
    "ce_start_point": 0.0,
    "pe_start_point": 0.0,

    # Risk Management Parameters (Delta Limits)
    "min_nifty_delta": -100.0,
    "max_nifty_delta": 100.0,
    
    # Black-Scholes Greeks Parameters
    "interest_rate": 10.0,         # 10% annual
    "todays_volatility": 20.0,     # 20% IV assumption
    
    # Order Parameters
    "order_type": "MARKET",        # MARKET order execution
    "product_type": "NRML",
    "tag": "WaveScraper",
    
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
        self._initialize_reference_points()
        
        logger.info(
            f"Initialized Wave Scraper Engine | Expiry: {self.expiry_code} | "
            f"CE Ref: {self.ce_reference:.2f} | PE Ref: {self.pe_reference:.2f} | "
            f"CE Gap: {self.config['ce_gap']} | PE Gap: {self.config['pe_gap']} | "
            f"Open Positions: {len(self.open_positions)}"
        )

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

    def build_option_symbol(self, strike, opt_type):
        """Build standard option symbol format (e.g. NIFTY04AUG2624500CE)"""
        if not self.expiry_code:
            self.expiry_code = self.get_expiry()
        return f"{self.config['underlying']}{self.expiry_code}{strike}{opt_type}"

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

    def find_sellable_option(self, spot, base_offset, opt_type):
        """
        Walk strikes outward from base OTM offset towards ATM until finding premium >= min_price_to_sell.
        """
        min_prem = self.config["min_price_to_sell"]
        current_offset = base_offset
        
        while current_offset >= 0:
            if opt_type == "CE":
                target_strike = int(round((spot + current_offset) / self.config["strike_step"]) * self.config["strike_step"])
            else:
                target_strike = int(round((spot - current_offset) / self.config["strike_step"]) * self.config["strike_step"])
                
            symbol = self.build_option_symbol(target_strike, opt_type)
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
            
        symbol = self.build_option_symbol(target_strike, opt_type)
        ltp = self.get_option_quote(symbol)
        return symbol, ltp, target_strike

    def handle_ce_sell(self, spot, port_delta):
        """Evaluate selling OTM CE when spot rises above ce_reference + ce_gap"""
        if spot - self.ce_reference < self.config["ce_gap"]:
            return

        # Count active CE positions
        active_ce_count = sum(1 for p in self.open_positions.values() if p.get("opt_type") == "CE")
        if active_ce_count >= self.config["max_active_positions"]:
            self.log_decision("CE_SKIPPED", f"Max active CE positions ({self.config['max_active_positions']}) reached.")
            return

        if port_delta > self.config["max_nifty_delta"]:
            self.log_decision("CE_SKIPPED", f"Delta limit breached ({port_delta} > {self.config['max_nifty_delta']})")
            return

        symbol, ltp, strike = self.find_sellable_option(spot, self.config["ce_otm_offset"], "CE")
        qty = self.config["ce_quantity"]

        # Shift CE reference point UP
        self.ce_reference += self.config["ce_gap"]

        # Execute SELL order
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="SELL",
                exchange=self.config["exchange_deriv"],
                price=0,  # Market order
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            logger.info(f"Executed WAVE CE SELL: {symbol} x {qty} @ Rs.{ltp:.2f}")
            self.log_trade(order_id, symbol, "SELL", qty, ltp, res.get("status", "SUCCESS"))
            self.log_decision("CE_EXECUTED", f"Sold {symbol} x {qty} @ Rs.{ltp:.2f}", {"spot": spot, "ce_ref": self.ce_reference})
            
            # Store in open positions
            self.open_positions[symbol] = {
                "qty": qty,
                "entry_price": ltp,
                "entry_time": datetime.now().isoformat(),
                "opt_type": "CE",
                "strike": strike
            }
        except Exception as e:
            logger.error(f"CE Order placement error: {e}")

    def handle_pe_sell(self, spot, port_delta):
        """Evaluate selling OTM PE when spot drops below pe_reference - pe_gap"""
        if self.pe_reference - spot < self.config["pe_gap"]:
            return

        # Count active PE positions
        active_pe_count = sum(1 for p in self.open_positions.values() if p.get("opt_type") == "PE")
        if active_pe_count >= self.config["max_active_positions"]:
            self.log_decision("PE_SKIPPED", f"Max active PE positions ({self.config['max_active_positions']}) reached.")
            return

        if port_delta < self.config["min_nifty_delta"]:
            self.log_decision("PE_SKIPPED", f"Delta limit breached ({port_delta} < {self.config['min_nifty_delta']})")
            return

        symbol, ltp, strike = self.find_sellable_option(spot, self.config["pe_otm_offset"], "PE")
        qty = self.config["pe_quantity"]

        # Shift PE reference point DOWN
        self.pe_reference -= self.config["pe_gap"]

        # Execute SELL order
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action="SELL",
                exchange=self.config["exchange_deriv"],
                price=0,  # Market order
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            logger.info(f"Executed WAVE PE SELL: {symbol} x {qty} @ Rs.{ltp:.2f}")
            self.log_trade(order_id, symbol, "SELL", qty, ltp, res.get("status", "SUCCESS"))
            self.log_decision("PE_EXECUTED", f"Sold {symbol} x {qty} @ Rs.{ltp:.2f}", {"spot": spot, "pe_ref": self.pe_reference})
            
            # Store in open positions
            self.open_positions[symbol] = {
                "qty": qty,
                "entry_price": ltp,
                "entry_time": datetime.now().isoformat(),
                "opt_type": "PE",
                "strike": strike
            }
        except Exception as e:
            logger.error(f"PE Order placement error: {e}")

    def handle_profit_booking(self, spot):
        """Check open positions and BUY back to square off when profit target (decay %) is hit"""
        to_remove = []
        for symbol, pos in list(self.open_positions.items()):
            ltp = self.get_option_quote(symbol)
            if ltp <= 0:
                continue
                
            entry_price = pos["entry_price"]
            decay_target = entry_price * (1.0 - self.config["profit_booking_pct"])
            
            # Trigger profit booking if option price decayed by profit_booking_pct
            if ltp <= decay_target:
                pnl = (entry_price - ltp) * pos["qty"]
                logger.info(f"Profit Target Hit for {symbol}! Entry: Rs.{entry_price:.2f} | Current: Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
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
                    self.log_decision("PROFIT_BOOKED", f"Bought back {symbol} x {pos['qty']} @ Rs.{ltp:.2f} | PnL: Rs.{pnl:.2f}")
                    to_remove.append(symbol)
                except Exception as e:
                    logger.error(f"Error booking profit for {symbol}: {e}")
                    
        for sym in to_remove:
            self.open_positions.pop(sym, None)

    def reset_reference_values(self, spot):
        """Reset reference values on strong market reversals to maintain wave tracking"""
        # CE Reset: Spot drops significantly below CE reference
        if (self.ce_reference - spot) > self.config["ce_reset_gap"]:
            new_val = spot + self.config["ce_gap"]
            logger.info(f"Resetting CE reference from {self.ce_reference:.2f} to {new_val:.2f}")
            self.log_decision("CE_REF_RESET", f"CE reference reset from {self.ce_reference:.2f} to {new_val:.2f}")
            self.ce_reference = new_val

        # PE Reset: Spot rises significantly above PE reference
        if (spot - self.pe_reference) > self.config["pe_reset_gap"]:
            new_val = spot - self.config["pe_gap"]
            logger.info(f"Resetting PE reference from {self.pe_reference:.2f} to {new_val:.2f}")
            self.log_decision("PE_REF_RESET", f"PE reference reset from {self.pe_reference:.2f} to {new_val:.2f}")
            self.pe_reference = new_val

    def execute_tick(self):
        """Main strategy tick loop"""
        spot = self.get_spot_price()
        if spot <= 0:
            logger.warning("Invalid spot price. Retrying...")
            return

        port_delta = self.calculate_portfolio_delta(spot)
        logger.info(
            f"[WAVE TICK] Spot: {spot:.2f} | CE Ref: {self.ce_reference:.2f} | PE Ref: {self.pe_reference:.2f} | "
            f"Delta: {port_delta:.2f} | Open Positions: {len(self.open_positions)}"
        )

        # 1. Check entries
        self.handle_ce_sell(spot, port_delta)
        self.handle_pe_sell(spot, port_delta)

        # 2. Check profit booking (buy-backs)
        self.handle_profit_booking(spot)

        # 3. Check reference resets
        self.reset_reference_values(spot)

        # 4. Save state
        self.save_state()


def run():
    engine = OpenAlgoWaveEngine(CONFIG)
    logger.info("Starting Wave Scraper Options Strategy Loop...")
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
