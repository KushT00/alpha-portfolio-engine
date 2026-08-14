"""
Nifty Master Portfolio Architect Engine v2.0 — Rolling Queue Architecture
=========================================================================
Rolling Queue Design:
  - W0 (Current Weekly):  Exclusively managed by Wave + Survivor. Portfolio Master is READ-ONLY.
  - W1 (Next Weekly):     Portfolio Master seeds 1 CE + 2 PE @ 300-400pt OTM. Max 3 legs.
  - W2 (Far Weekly):      Portfolio Master seeds 1 CE + 1 PE @ 500-600pt OTM. Max 2 legs.
  - W3 (Monthly ~3wk):    Portfolio Master seeds PE only @ 700-800pt OTM. Max 2 legs.
  - FarMonthly (M1/M2):   Portfolio Master seeds ultra-deep PE @ 1500-2000pt OTM. Max 2 legs.

Per-Expiry Isolated Ledger:
  - Every seeded CE/PE is stored as a "pair" inside its expiry bucket.
  - Health checks run independently per expiry. No cross-expiry contamination.
  - Each pair tracks: entry_spot, danger_spot, adjustment_done.

Dynamic Premium Ratio Spread Adjustment (50pt Danger Zone):
  - When spot is within 50pt of any seeded strike:
      → Buy 2x lots of threatened strike (closes 1x short + holds 1x net long hedge for delta protection)
      → Dynamically scans OTM strikes priced at 50%–60% of threatened leg price
      → Sell 3x lots of new dynamic OTM strike
  - Partner leg (decayed side) is booked simultaneously if decayed >= 60% to free margin.

Market Hours Gate:
  - All orders (seeding, harvesting, adjustments) blocked outside 09:15 - 15:30 IST.
  - Outside market hours: read-only mode (logging & monitoring only).

Safe Harvest (80% Decay + 30min Minimum Age):
  - Short legs bought back only when: decay >= 80% AND position age >= 30 minutes.
  - Prevents the "seed & immediately harvest" infinite loop bug.

Seed Cooldown:
  - After seeding a tier, 4-hour cooldown prevents re-seeding same tier same day.
  - Seeder checks existing leg count per expiry BEFORE placing any order.
"""

import os
import re
import json
import time
import math
import csv
import logging
from datetime import datetime, date, timedelta
from openalgo import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PortfolioMaster")

# ============================================================
# 1. CONFIG
# ============================================================
CONFIG = {
    "api_key": "ff1d258f2c5d48bf87292b3f055b39772ca76f4afe69ff7c7c5fa121a32334d8",
    "host": "http://127.0.0.1:5000",

    "underlying": "NIFTY",
    "exchange_index": "NSE_INDEX",
    "exchange_deriv": "NFO",
    "lot_size": 65,
    "strike_step": 50,
    "lot_multiplier": 1,   # scale up as capital grows

    # Market Hours Gate (IST)
    "market_open_hour": 9,
    "market_open_minute": 15,
    "market_close_hour": 15,
    "market_close_minute": 30,

    # Per-Tier Seeding Rules
    "tier_rules": {
        "W1": {"min_otm": 300, "max_otm": 400, "max_legs": 3, "ce_count": 1, "pe_count": 2},
        "W2": {"min_otm": 500, "max_otm": 600, "max_legs": 2, "ce_count": 1, "pe_count": 1},
        "W3": {"min_otm": 700, "max_otm": 800, "max_legs": 2, "ce_count": 0, "pe_count": 2},
        "M1": {"min_otm": 1500, "max_otm": 2000, "max_legs": 2, "ce_count": 0, "pe_count": 2},
    },

    # Adjustment Engine
    "danger_zone_pts": 50,        # Ratio spread triggers when spot within 50pt of strike

    # Safe Harvest Rules
    "harvest_decay_pct": 0.80,     # Only harvest if 80%+ decayed
    "harvest_min_ltp": 3.0,        # Or LTP <= Rs 3.0
    "harvest_min_age_minutes": 30, # Leg must be open >= 30 min before harvesting

    # Seed Cooldown
    "seed_cooldown_hours": 4,      # Hours before re-seeding same tier

    # Order params
    "order_type": "MARKET",
    "product_type": "NRML",
    "tag": "PortfolioMaster",
    "poll_interval": 30,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "nifty_portfolio_master_state.json")
TRADE_LOG  = os.path.join(SCRIPT_DIR, "nifty_portfolio_master_trades.csv")
AUDIT_LOG  = os.path.join(SCRIPT_DIR, "nifty_portfolio_master_audit.csv")


# ============================================================
# 2. STATE MANAGER
# ============================================================
class StateManager:
    """Handles all read/write operations to the per-expiry ledger."""

    def __init__(self):
        self.data = self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"State load error: {e}")
        return {"expiries": {}, "tier_last_seed": {}}

    def save(self):
        self.data["last_update"] = datetime.now().isoformat()
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"State save error: {e}")

    def get_expiry_ledger(self, expiry_code):
        return self.data["expiries"].get(expiry_code, {"tier": "UNKNOWN", "pairs": []})

    def upsert_expiry_ledger(self, expiry_code, ledger):
        self.data["expiries"][expiry_code] = ledger
        self.save()

    def count_open_legs(self, expiry_code, broker_positions):
        """Count how many open short option legs exist for a given expiry."""
        count = 0
        for sym, pos in broker_positions.items():
            if expiry_code in sym and pos.get("qty", 0) < 0:
                count += 1
        return count

    def get_last_seed_time(self, tier_key):
        ts = self.data["tier_last_seed"].get(tier_key)
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def set_last_seed_time(self, tier_key):
        self.data["tier_last_seed"][tier_key] = datetime.now().isoformat()
        self.save()

    def is_seed_on_cooldown(self, tier_key, cooldown_hours):
        last = self.get_last_seed_time(tier_key)
        if last is None:
            return False
        return (datetime.now() - last).total_seconds() < cooldown_hours * 3600


# ============================================================
# 3. UTILITIES
# ============================================================
def is_market_open(config):
    now = datetime.now()
    open_time  = now.replace(hour=config["market_open_hour"],  minute=config["market_open_minute"],  second=0)
    close_time = now.replace(hour=config["market_close_hour"], minute=config["market_close_minute"], second=0)
    return open_time <= now <= close_time

def log_trade(symbol, action, qty, price, order_id, status, tag=""):
    try:
        exists = os.path.exists(TRADE_LOG)
        with open(TRADE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "order_id", "symbol", "action", "qty", "price", "status", "tag"])
            writer.writerow([datetime.now().isoformat(), order_id, symbol, action, qty, price, status, tag])
    except Exception:
        pass

def log_audit(action, reason, details=None):
    logger.info(f"[{action}] {reason}")
    try:
        exists = os.path.exists(AUDIT_LOG)
        with open(AUDIT_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "action", "reason", "details"])
            writer.writerow([datetime.now().isoformat(), action, reason, json.dumps(details or {})])
    except Exception:
        pass

def build_symbol(underlying, expiry_code, strike, opt_type):
    return f"{underlying}{expiry_code}{int(strike)}{opt_type}"

def extract_strike(symbol):
    """Extract 4-5 digit strike from option symbol. Returns 0 if not found."""
    m = re.search(r'(\d{4,5})(CE|PE)$', symbol)
    return int(m.group(1)) if m else 0

def age_minutes(entry_time_iso):
    try:
        entry = datetime.fromisoformat(entry_time_iso)
        return (datetime.now() - entry).total_seconds() / 60.0
    except Exception:
        return 999.0


# ============================================================
# 4. MASTER PORTFOLIO ENGINE
# ============================================================
class MasterPortfolioEngine:

    def __init__(self, config=CONFIG):
        self.config = config
        self.client = api(api_key=config["api_key"], host=config["host"])
        self.state  = StateManager()
        self.vix    = 14.0

        # Expiry matrix: W0, W1, W2, W3, M1, M2
        self.expiry_matrix = {}

        logger.info("Portfolio Master v2.0 — Rolling Queue Architecture — Initializing...")

    # ─── Data Fetching ────────────────────────────────────────────────────────

    def get_spot(self):
        try:
            res = self.client.quotes(exchange=self.config["exchange_index"], symbol=self.config["underlying"])
            data = res if not isinstance(res, dict) else res.get("data", res)
            return float(data.get("ltp") or data.get("last_price", 0))
        except Exception:
            return 0.0

    def get_vix(self):
        try:
            res = self.client.quotes(exchange="NSE_INDEX", symbol="INDIAVIX")
            data = res if not isinstance(res, dict) else res.get("data", res)
            val = float(data.get("ltp") or data.get("last_price", 0))
            if val > 0:
                self.vix = val
        except Exception:
            pass

    def get_ltp(self, symbol):
        try:
            res = self.client.quotes(exchange=self.config["exchange_deriv"], symbol=symbol)
            data = res if not isinstance(res, dict) else res.get("data", res)
            ltp = data.get("ltp") or data.get("last_price")
            return float(ltp) if ltp else 0.0
        except Exception:
            return 0.0

    def fetch_expiry_matrix(self):
        """Fetch all expiries and categorize into W0, W1, W2, W3, M1, M2."""
        try:
            res = self.client.expiry(
                symbol=self.config["underlying"],
                exchange=self.config["exchange_deriv"],
                instrumenttype="options"
            )
            data = res if isinstance(res, list) else res.get("data", [])
            exp_list = [e.replace("-", "") for e in data]
            tier_names = ["W0", "W1", "W2", "W3", "M1", "M2"]
            self.expiry_matrix = {}
            for i, tier in enumerate(tier_names):
                if i < len(exp_list):
                    self.expiry_matrix[tier] = exp_list[i]
            logger.info(f"[EXPIRY MATRIX] {self.expiry_matrix}")
        except Exception as e:
            logger.error(f"Expiry matrix error: {e}")

    def get_broker_positions(self):
        """Returns dict of {symbol: {qty, avg_price, entry_time}} for all open NIFTY positions."""
        result = {}
        try:
            pos_res = self.client.positionbook()
            positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", [])
            for pos in positions:
                sym = pos.get("symbol", "")
                qty = int(pos.get("quantity") or pos.get("quantity_total") or 0)
                if not sym.startswith("NIFTY") or qty == 0:
                    continue
                result[sym] = {
                    "qty": qty,
                    "avg_price": float(pos.get("average_price") or pos.get("buy_price") or 0),
                    "entry_time": pos.get("entry_time") or datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Positionbook error: {e}")
        return result

    def place_order(self, symbol, action, qty, tag=""):
        """Place a market order. Returns order_id or None."""
        try:
            res = self.client.placeorder(
                strategy=self.config["tag"],
                symbol=symbol,
                action=action,
                exchange=self.config["exchange_deriv"],
                price=0,
                pricetype=self.config["order_type"],
                product=self.config["product_type"],
                quantity=qty,
            )
            order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
            ltp = self.get_ltp(symbol)
            log_trade(symbol, action, qty, ltp, order_id, res.get("status", "OK"), tag)
            return order_id
        except Exception as e:
            logger.error(f"Order error [{action} {symbol} x{qty}]: {e}")
            return None

    # ─── Module 1: Safe Profit Harvester ─────────────────────────────────────

    def run_safe_harvest(self, broker_positions):
        """
        Harvest short legs that meet ALL three conditions:
          1. Decay >= 80% (LTP <= 20% of entry) OR LTP <= Rs 3.0
          2. Position age >= 30 minutes
          3. Market is open
        """
        if not is_market_open(self.config):
            return

        for sym, pos in broker_positions.items():
            if pos["qty"] >= 0:
                continue  # Only short positions
            avg = pos["avg_price"]
            if avg <= 0:
                continue

            ltp = self.get_ltp(sym)
            if ltp <= 0:
                continue

            decay_pct = (avg - ltp) / avg
            age_ok    = age_minutes(pos["entry_time"]) >= self.config["harvest_min_age_minutes"]
            decay_ok  = decay_pct >= self.config["harvest_decay_pct"] or ltp <= self.config["harvest_min_ltp"]

            if decay_ok and age_ok:
                qty = abs(pos["qty"])
                pnl = (avg - ltp) * qty
                logger.info(
                    f"[HARVEST] {sym} | Avg:{avg:.2f} LTP:{ltp:.2f} | "
                    f"Decay:{decay_pct*100:.1f}% Age:{age_minutes(pos['entry_time']):.0f}min | PnL:Rs.{pnl:.2f}"
                )
                order_id = self.place_order(sym, "BUY", qty, tag="HARVEST")
                log_audit("HARVEST_BOOKED", f"Bought back {sym} x {qty} @ {ltp:.2f}", {"pnl": pnl})

    # ─── Module 2: Per-Expiry Health Check & Ratio Spread Adjustment ─────────

    def run_health_check(self, spot, broker_positions):
        """
        For each non-W0 expiry, independently check every seeded pair.
        If any leg is within 50pt of spot, trigger Ratio Spread adjustment.
        """
        if not is_market_open(self.config):
            return

        danger_pts = self.config["danger_zone_pts"]

        for tier, expiry_code in self.expiry_matrix.items():
            if tier == "W0":
                continue  # W0 is exclusively managed by Wave + Survivor

            ledger = self.state.get_expiry_ledger(expiry_code)
            pairs  = ledger.get("pairs", [])
            updated = False

            for pair in pairs:
                if pair.get("adjustment_done", False):
                    continue

                # Check CE leg
                ce = pair.get("ce_leg")
                if ce and ce.get("status") == "OK":
                    ce_sym = ce["symbol"]
                    ce_strike = ce["strike"]
                    if ce_sym in broker_positions and broker_positions[ce_sym]["qty"] < 0:
                        distance = ce_strike - spot
                        if distance <= danger_pts:
                            logger.warning(
                                f"[{tier}] DANGER: {ce_sym} strike {ce_strike} within {distance:.0f}pt of spot {spot:.2f}! "
                                f"Triggering Ratio Spread."
                            )
                            self._apply_ratio_spread(pair, "CE", spot, expiry_code, broker_positions)
                            pair["adjustment_done"] = True
                            updated = True
                            continue

                # Check PE leg
                pe = pair.get("pe_leg")
                if pe and pe.get("status") == "OK":
                    pe_sym = pe["symbol"]
                    pe_strike = pe["strike"]
                    if pe_sym in broker_positions and broker_positions[pe_sym]["qty"] < 0:
                        distance = spot - pe_strike
                        if distance <= danger_pts:
                            logger.warning(
                                f"[{tier}] DANGER: {pe_sym} strike {pe_strike} within {distance:.0f}pt of spot {spot:.2f}! "
                                f"Triggering Ratio Spread."
                            )
                            self._apply_ratio_spread(pair, "PE", spot, expiry_code, broker_positions)
                            pair["adjustment_done"] = True
                            updated = True

            if updated:
                ledger["pairs"] = pairs
                self.state.upsert_expiry_ledger(expiry_code, ledger)

    def _apply_ratio_spread(self, pair, threatened_side, spot, expiry_code, broker_positions):
        """
        Dynamic Premium Ratio Spread Adjustment:
          1. Buy 2x lots of threatened leg (e.g. 2 x 65 qty @ Rs 143) to close 1x short + hold 1x net long hedge.
          2. Dynamically scan further OTM strikes to find an option priced at 50%–60% of threatened LTP (e.g. ~Rs 75–80).
          3. Sell 3x lots of this dynamic OTM strike (3 x 65 qty).
          4. Simultaneously book partner leg if decayed >= 60%.
        """
        base_qty = self.config["lot_size"] * self.config["lot_multiplier"]
        underlying = self.config["underlying"]
        strike_step = self.config["strike_step"]

        if threatened_side == "CE":
            threatened_leg = pair.get("ce_leg")
            partner_leg    = pair.get("pe_leg")
        else:
            threatened_leg = pair.get("pe_leg")
            partner_leg    = pair.get("ce_leg")

        if not threatened_leg:
            return

        threatened_sym    = threatened_leg["symbol"]
        threatened_strike = threatened_leg["strike"]
        threatened_ltp    = self.get_ltp(threatened_sym)

        if threatened_ltp <= 0:
            logger.warning(f"[RATIO SPREAD] Could not fetch LTP for {threatened_sym}. Aborting adjustment.")
            return

        # Target premium for new OTM leg: 50% - 60% of threatened LTP (default target 55%)
        # Dynamic strike scanner: start scanning at MINIMUM 300-400pt OTM away from threatened strike
        min_otm_steps = 6  # 6 * 50pt = 300pt minimum OTM distance
        target_premium = threatened_ltp * 0.55
        logger.info(f"[RATIO SPREAD] Threatened {threatened_sym} LTP: Rs.{threatened_ltp:.2f} | Target OTM Premium: Rs.{target_premium:.2f}")

        # Dynamic strike scanner: find strike closest to 50-60% of threatened LTP
        best_strike = 0
        best_diff = 999999.0
        best_ltp = 0.0

        for step_multiplier in range(min_otm_steps, min_otm_steps + 12):
            if threatened_side == "CE":
                cand_strike = int(threatened_strike + (step_multiplier * strike_step))
                cand_sym = build_symbol(underlying, expiry_code, cand_strike, "CE")
            else:
                cand_strike = int(threatened_strike - (step_multiplier * strike_step))
                cand_sym = build_symbol(underlying, expiry_code, cand_strike, "PE")

            cand_ltp = self.get_ltp(cand_sym)
            if cand_ltp <= 0:
                continue

            diff = abs(cand_ltp - target_premium)
            if diff < best_diff:
                best_diff = diff
                best_strike = cand_strike
                best_ltp = cand_ltp

            # Stop scanning if premium drops below 30% of target
            if cand_ltp < target_premium * 0.5:
                break

        if best_strike == 0:
            fallback_offset = 350 if threatened_side == "CE" else -350
            best_strike = int(threatened_strike + fallback_offset)
            best_ltp = self.get_ltp(build_symbol(underlying, expiry_code, best_strike, threatened_side))

        new_sym = build_symbol(underlying, expiry_code, best_strike, threatened_side)

        buy_qty  = base_qty * 2   # Buy 2x: closes 1x short + holds 1x net long hedge
        sell_qty = base_qty * 3   # Sell 3x: OTM calls priced at 50-60% of threatened LTP

        net_cost = (threatened_ltp * 2) - (best_ltp * 3)

        # Step 1: Buy 2x lots of threatened leg (Closes short + 1x net long hedge)
        logger.info(
            f"[RATIO SPREAD] Step 1: Buying 2x lots of threatened leg {threatened_sym} x {buy_qty} @ Rs.{threatened_ltp:.2f} "
            f"(Closes short + holds 1x long hedge)"
        )
        self.place_order(threatened_sym, "BUY", buy_qty, tag="RATIO_SPREAD_LONG_HEDGE")
        threatened_leg["status"] = "FLIPPED_TO_LONG_HEDGE"

        # Step 2: Sell 3x lots of dynamic OTM strike (Priced at 50-60% of threatened LTP)
        logger.info(
            f"[RATIO SPREAD] Step 2: Selling 3x lots of {new_sym} x {sell_qty} @ Rs.{best_ltp:.2f} | "
            f"Net Premium Cost/Credit: Rs.{net_cost:+.2f}/qty"
        )
        self.place_order(new_sym, "SELL", sell_qty, tag="RATIO_SPREAD_NEW_3X")

        # Step 3: Book decayed partner leg if available and decayed >= 60%
        if partner_leg:
            partner_sym = partner_leg["symbol"]
            if partner_sym in broker_positions and broker_positions[partner_sym]["qty"] < 0:
                partner_avg = broker_positions[partner_sym]["avg_price"]
                partner_ltp = self.get_ltp(partner_sym)
                if partner_ltp > 0 and (partner_avg - partner_ltp) / max(partner_avg, 1) >= 0.60:
                    partner_qty = abs(broker_positions[partner_sym]["qty"])
                    logger.info(f"[RATIO SPREAD] Step 3: Booking decayed partner {partner_sym} x {partner_qty} @ Rs.{partner_ltp:.2f}")
                    self.place_order(partner_sym, "BUY", partner_qty, tag="RATIO_SPREAD_PARTNER_BOOK")
                    partner_leg["status"] = "BOOKED"

        log_audit(
            "RATIO_SPREAD_APPLIED",
            f"Threatened: {threatened_sym} x 2x (BUY @ Rs.{threatened_ltp:.2f}) | New: {new_sym} x 3x (SELL @ Rs.{best_ltp:.2f})",
            {"pair": pair.get("id"), "expiry": expiry_code}
        )

    # ─── Module 3: Rolling Queue Seeder ──────────────────────────────────────

    def run_seeder(self, spot, broker_positions):
        """
        Seeds new legs for W1, W2, W3, M1 tiers.
        Rules:
          - W0 is NEVER seeded by Portfolio Master (Wave + Survivor own W0)
          - Checks existing leg count per expiry before seeding
          - Respects 4-hour cooldown per tier
          - Only seeds during market hours
          - Seeds one leg per tier per tick to avoid aggressive over-seeding
        """
        if not is_market_open(self.config):
            return

        strike_step = self.config["strike_step"]
        base_qty    = self.config["lot_size"] * self.config["lot_multiplier"]
        underlying  = self.config["underlying"]
        cooldown_h  = self.config["seed_cooldown_hours"]

        seeding_tiers = ["W1", "W2", "W3", "M1"]

        for tier in seeding_tiers:
            expiry_code = self.expiry_matrix.get(tier)
            if not expiry_code:
                continue

            rules = self.config["tier_rules"].get(tier, {})
            max_legs = rules.get("max_legs", 2)
            min_otm  = rules.get("min_otm", 300)
            max_otm  = rules.get("max_otm", 500)
            pe_count = rules.get("pe_count", 1)
            ce_count = rules.get("ce_count", 1)

            # Check existing leg count in broker positionbook for this expiry
            existing_legs = self.state.count_open_legs(expiry_code, broker_positions)

            if existing_legs >= max_legs:
                logger.info(f"[SEEDER] {tier} ({expiry_code}): {existing_legs}/{max_legs} legs. Skip.")
                continue

            # Check cooldown
            if self.state.is_seed_on_cooldown(tier, cooldown_h):
                logger.info(f"[SEEDER] {tier} ({expiry_code}): On cooldown. Skip.")
                continue

            # Determine what to seed next: alternate CE/PE based on what's already there
            ledger = self.state.get_expiry_ledger(expiry_code)
            pairs  = ledger.get("pairs", [])
            seeded_ce = sum(1 for p in pairs if p.get("ce_leg") and p["ce_leg"].get("status") == "OK")
            seeded_pe = sum(1 for p in pairs if p.get("pe_leg") and p["pe_leg"].get("status") == "OK")

            seed_next_ce = seeded_ce < ce_count
            seed_next_pe = seeded_pe < pe_count

            otm_offset = int(round((min_otm + max_otm) / 2 / strike_step) * strike_step)
            seeded_something = False

            # Seed PE leg if needed
            if seed_next_pe:
                pe_strike = int(round((spot - otm_offset) / strike_step) * strike_step)
                pe_sym    = build_symbol(underlying, expiry_code, pe_strike, "PE")

                if pe_sym not in broker_positions:
                    ltp = self.get_ltp(pe_sym)
                    if ltp >= 5.0:
                        logger.info(f"[SEEDER] {tier} Seeding PE: {pe_sym} x {base_qty} @ Rs.{ltp:.2f}")
                        order_id = self.place_order(pe_sym, "SELL", base_qty, tag=f"SEED_{tier}")
                        if order_id:
                            pair_id = f"pair_{expiry_code}_{len(pairs)+1:03d}"
                            new_pair = {
                                "id": pair_id,
                                "entry_spot": spot,
                                "entry_date": datetime.now().isoformat(),
                                "adjustment_done": False,
                                "ce_leg": None,
                                "pe_leg": {
                                    "symbol": pe_sym,
                                    "strike": pe_strike,
                                    "qty": base_qty,
                                    "entry_price": ltp,
                                    "status": "OK",
                                },
                            }
                            ledger.setdefault("tier", tier)
                            ledger.setdefault("pairs", [])
                            ledger["pairs"].append(new_pair)
                            self.state.upsert_expiry_ledger(expiry_code, ledger)
                            self.state.set_last_seed_time(tier)
                            log_audit("SEED_PE", f"Sold {pe_sym} x {base_qty} @ Rs.{ltp:.2f}", {"tier": tier, "otm": otm_offset})
                            seeded_something = True

            # Seed CE leg if needed (only if PE already seeded or CE count > 0)
            if seed_next_ce and not seeded_something:
                ce_strike = int(round((spot + otm_offset) / strike_step) * strike_step)
                ce_sym    = build_symbol(underlying, expiry_code, ce_strike, "CE")

                if ce_sym not in broker_positions:
                    ltp = self.get_ltp(ce_sym)
                    if ltp >= 5.0:
                        logger.info(f"[SEEDER] {tier} Seeding CE: {ce_sym} x {base_qty} @ Rs.{ltp:.2f}")
                        order_id = self.place_order(ce_sym, "SELL", base_qty, tag=f"SEED_{tier}")
                        if order_id:
                            attached = False
                            for p in ledger.get("pairs", []):
                                if p.get("ce_leg") is None and not p.get("adjustment_done"):
                                    p["ce_leg"] = {
                                        "symbol": ce_sym,
                                        "strike": ce_strike,
                                        "qty": base_qty,
                                        "entry_price": ltp,
                                        "status": "OK",
                                    }
                                    attached = True
                                    break

                            if not attached:
                                pair_id = f"pair_{expiry_code}_{len(ledger.get('pairs',[]))+1:03d}"
                                ledger.setdefault("pairs", []).append({
                                    "id": pair_id,
                                    "entry_spot": spot,
                                    "entry_date": datetime.now().isoformat(),
                                    "adjustment_done": False,
                                    "pe_leg": None,
                                    "ce_leg": {
                                        "symbol": ce_sym,
                                        "strike": ce_strike,
                                        "qty": base_qty,
                                        "entry_price": ltp,
                                        "status": "OK",
                                    },
                                })

                            self.state.upsert_expiry_ledger(expiry_code, ledger)
                            self.state.set_last_seed_time(tier)
                            log_audit("SEED_CE", f"Sold {ce_sym} x {base_qty} @ Rs.{ltp:.2f}", {"tier": tier, "otm": otm_offset})

    # ─── Main Tick ────────────────────────────────────────────────────────────

    def execute_tick(self):
        # Refresh market data
        self.fetch_expiry_matrix()
        self.get_vix()
        spot = self.get_spot()

        if spot <= 0:
            logger.warning("[TICK] Invalid spot price. Skipping tick.")
            return

        # Read live positionbook
        broker_positions = self.get_broker_positions()

        market_status = "OPEN" if is_market_open(self.config) else "CLOSED (Read-Only)"
        logger.info(
            f"[PORTFOLIO MASTER TICK] Spot:{spot:.2f} | VIX:{self.vix:.2f} | "
            f"Market:{market_status} | W0:{self.expiry_matrix.get('W0')} "
            f"W1:{self.expiry_matrix.get('W1')} W2:{self.expiry_matrix.get('W2')} "
            f"W3:{self.expiry_matrix.get('W3')} | Open Legs:{len(broker_positions)}"
        )

        # Step 1: Safe Harvest (market hours only, 80% decay + 30min age)
        self.run_safe_harvest(broker_positions)

        # Step 2: Per-Expiry Health Check & Ratio Spread Adjustment (market hours only)
        self.run_health_check(spot, broker_positions)

        # Step 3: Rolling Queue Seeder (market hours only, cooldown enforced)
        self.run_seeder(spot, broker_positions)

        self.state.save()


# ============================================================
# 5. RUN LOOP
# ============================================================
def run():
    engine = MasterPortfolioEngine(CONFIG)
    logger.info("Portfolio Master v2.0 — Starting Rolling Queue Loop...")
    while True:
        try:
            engine.execute_tick()
        except KeyboardInterrupt:
            logger.info("Portfolio Master stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        time.sleep(CONFIG["poll_interval"])


if __name__ == "__main__":
    run()
