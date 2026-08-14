"""
NIFTY Portfolio Analytics & Greeks Dashboard - Flask Application
===================================================================
Fetches live positions & quotes from OpenAlgo API (http://127.0.0.1:5000),
calculates Black-Scholes Greeks (Delta, Theta, Gamma, Vega) using live IV back-solver,
aggregates portfolio sentiment & stress metrics, time value decomposition,
per-expiry Greeks breakdown, today's trade impact, and cross-strategy duplicate detector.

New Features (v2.0):
  - #2 Live IV Back-Solver: Newton-Raphson solver from LTP — replaces fixed VIX Greeks.
  - #3 Time Value vs Intrinsic: Splits short premiums into safe decay vs ITM risk.
  - #4 Per-Expiry Greeks: Delta + Theta breakdown per expiry date.
  - #5 Today's Trade Impact: Delta/Theta added specifically by today's orders.
  - #1 Position Guard: /api/position-guard endpoint flags unintended net long exposure.
"""

import os
import re
import math
import logging
from datetime import datetime, date
import requests
from flask import Flask, render_template, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PortfolioDashboard")

app = Flask(__name__)

# OpenAlgo Configuration
OPENALGO_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
OPENALGO_API_KEY = os.getenv("OPENALGO_API_KEY", "ff1d258f2c5d48bf87292b3f055b39772ca76f4afe69ff7c7c5fa121a32334d8")
NIFTY_LOT_SIZE = 65

# Regex pattern for Indian Option Symbols
# e.g., NIFTY04AUG2624800CE, NIFTY11AUG2624450PE
SYMBOL_PATTERN = re.compile(r"^(NIFTY|BANKNIFTY|FINNIFTY)?(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$", re.IGNORECASE)

# ============================================================
# MATH HELPERS — Standard Normal Distribution
# ============================================================
def norm_cdf(x):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    """Probability density function for standard normal distribution"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _bs_call_price(spot, strike, T, sigma, r=0.07):
    """Black-Scholes call price"""
    if T <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)

def _bs_put_price(spot, strike, T, sigma, r=0.07):
    """Black-Scholes put price"""
    if T <= 0 or sigma <= 0:
        return max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return strike * math.exp(-r * T) * norm_cdf(-d2) - spot * norm_cdf(-d1)


# ============================================================
# #2 — LIVE IV BACK-SOLVER (Newton-Raphson from market LTP)
# ============================================================
def estimate_live_iv(spot, strike, dte_days, ltp, opt_type="CE", fallback=15.0, r=0.07):
    """
    Back-solve Implied Volatility from live option LTP using Newton-Raphson.
    Returns IV as percentage (e.g. 14.5 for 14.5% IV).
    Falls back to `fallback` (live VIX) if LTP too low or solver fails.
    """
    if ltp < 0.5 or spot <= 0 or strike <= 0 or dte_days <= 0:
        return fallback
    T = max(dte_days, 0.5) / 365.0
    sigma = fallback / 100.0  # Start from VIX as initial guess
    is_call = opt_type.upper() in ("CE", "CALL")
    try:
        for _ in range(50):
            price = _bs_call_price(spot, strike, T, sigma, r) if is_call else _bs_put_price(spot, strike, T, sigma, r)
            d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = spot * norm_pdf(d1) * math.sqrt(T)
            if vega < 1e-10:
                break
            diff = price - ltp
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))
            if abs(diff) < 0.001:
                break
        iv_pct = sigma * 100.0
        if 1.0 < iv_pct < 200.0:
            return round(iv_pct, 2)
    except Exception:
        pass
    return fallback


# ============================================================
# BLACK-SCHOLES GREEKS ENGINE
# ============================================================
def calculate_greeks(spot, strike, dte_days, iv_pct, opt_type="CE", rate=0.07):
    """
    Calculate Black-Scholes Option Greeks:
    Delta, Gamma, Theta (per day), Vega (per 1% IV)
    """
    if spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    # Minimum DTE is 0.5 days for 0-DTE expiry to prevent divide by zero
    T = max(dte_days, 0.5) / 365.0
    sigma = max(iv_pct, 1.0) / 100.0
    r = rate

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        pdf_d1 = norm_pdf(d1)
        cdf_d1 = norm_cdf(d1)
        cdf_d2 = norm_cdf(d2)
        cdf_neg_d2 = norm_cdf(-d2)

        gamma = pdf_d1 / (spot * sigma * math.sqrt(T))
        vega = (spot * pdf_d1 * math.sqrt(T)) / 100.0

        if opt_type.upper() in ["CE", "CALL"]:
            delta = cdf_d1
            theta_annual = -(spot * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * strike * math.exp(-r * T) * cdf_d2
        else:
            delta = cdf_d1 - 1.0
            theta_annual = -(spot * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * strike * math.exp(-r * T) * cdf_neg_d2

        theta_daily = theta_annual / 365.0

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta_daily, 4),
            "vega": round(vega, 4)
        }
    except Exception as e:
        logger.error(f"Greeks calculation error for Strike {strike}: {e}")
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def parse_option_symbol(symbol):
    """
    Parse option symbol to extract Underlying, Strike, Option Type, and Expiry Date
    """
    sym = symbol.strip().upper()
    match = SYMBOL_PATTERN.match(sym)
    if match:
        underlying = match.group(1) or "NIFTY"
        day_str = match.group(2)
        month_str = match.group(3)
        year_str = match.group(4)
        strike = float(match.group(5))
        opt_type = match.group(6)

        date_str = f"{day_str}{month_str}20{year_str}"
        try:
            exp_date = datetime.strptime(date_str, "%d%b%Y").date()
            today = datetime.now().date()
            dte = max(0, (exp_date - today).days)
            display_expiry = exp_date.strftime("%d %b")
        except Exception:
            exp_date = None
            dte = 0
            display_expiry = f"{day_str} {month_str}"

        return {
            "underlying": underlying,
            "strike": strike,
            "opt_type": opt_type,
            "expiry_date": str(exp_date) if exp_date else "",
            "display_expiry": display_expiry,
            "dte": dte
        }

    # Generic fallback parser
    opt_type = "CE" if "CE" in sym else "PE" if "PE" in sym else "FUT"
    nums = re.findall(r"\d+", sym)
    strike = float(nums[-1]) if nums else 0.0
    return {
        "underlying": "NIFTY",
        "strike": strike,
        "opt_type": opt_type,
        "expiry_date": "",
        "display_expiry": "Active",
        "dte": 0
    }


# In-memory cache for persistence when OpenAlgo API times out or fails
_cache = {
    "spot": 24600.0,
    "vix": 12.3,
    "positionbook": [],
    "last_successful_fetch": None,
    "is_cached": False,
    "quotes": {}
}


def fetch_live_quotes():
    """Fetch NIFTY Spot price & INDIA VIX from OpenAlgo API with fallback to persistent cache"""
    global _cache
    spot = _cache["spot"]
    vix = _cache["vix"]

    try:
        # Fetch NIFTY Spot
        resp = requests.post(
            f"{OPENALGO_HOST}/api/v1/quotes",
            json={"apikey": OPENALGO_API_KEY, "exchange": "NSE_INDEX", "symbol": "NIFTY"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            quote = data.get("data", data) if isinstance(data, dict) else {}
            if isinstance(quote, dict):
                val = float(quote.get("ltp") or quote.get("last_price") or 0.0)
                if val > 0:
                    spot = val
                    _cache["spot"] = spot
    except Exception as e:
        logger.warning(f"Could not fetch NIFTY spot quote: {e} (using cached: {spot})")

    try:
        # Fetch INDIA VIX
        resp = requests.post(
            f"{OPENALGO_HOST}/api/v1/quotes",
            json={"apikey": OPENALGO_API_KEY, "exchange": "NSE_INDEX", "symbol": "INDIAVIX"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            quote = data.get("data", data) if isinstance(data, dict) else {}
            if isinstance(quote, dict):
                val = float(quote.get("ltp") or quote.get("last_price") or 0.0)
                if val > 0:
                    vix = val
                    _cache["vix"] = vix
    except Exception as e:
        logger.warning(f"Could not fetch INDIAVIX quote: {e} (using cached: {vix})")

    return spot, vix


def fetch_positionbook():
    """Fetch live positionbook from OpenAlgo API with fallback to persistent cache"""
    global _cache
    try:
        resp = requests.post(
            f"{OPENALGO_HOST}/api/v1/positionbook",
            json={"apikey": OPENALGO_API_KEY},
            timeout=4
        )
        if resp.status_code == 200:
            res = resp.json()
            data = []
            if isinstance(res, dict):
                data = res.get("data", [])
            elif isinstance(res, list):
                data = res
            
            _cache["positionbook"] = data
            _cache["last_successful_fetch"] = datetime.now().strftime("%H:%M:%S")
            _cache["is_cached"] = False
            return {"data": data, "is_cached": False, "last_successful_fetch": _cache["last_successful_fetch"]}
    except Exception as e:
        logger.error(f"Error fetching positionbook from OpenAlgo: {e}. Retaining persistent cached positions.")

    _cache["is_cached"] = True
    return {
        "data": _cache["positionbook"],
        "is_cached": True,
        "last_successful_fetch": _cache["last_successful_fetch"] or datetime.now().strftime("%H:%M:%S")
    }


@app.route("/")
def index():
    """Render main HTML dashboard"""
    return render_template("index.html")


@app.route("/api/portfolio")
def get_portfolio_data():
    """
    API Endpoint: Calculates portfolio metrics, Greeks (with live IV per position),
    time value decomposition, per-expiry Greeks map, and today's trade impact.
    """
    min_premium = float(request.args.get("min_premium", 0.0))
    min_days = int(request.args.get("min_days", 0))

    spot, vix = fetch_live_quotes()
    pos_res = fetch_positionbook()
    positions = pos_res.get("data", [])

    processed_positions = []

    total_puts_sold_qty  = 0
    total_calls_sold_qty = 0

    overall_delta = 0.0
    overall_theta = 0.0
    overall_gamma = 0.0
    overall_vega  = 0.0

    unrealized_pnl = 0.0
    realized_pnl   = 0.0

    # #3 — Time Value vs Intrinsic Decomposition
    total_time_value = 0.0   # Safe premium that decays to zero
    total_itm_risk   = 0.0   # Intrinsic value at risk if held to expiry

    # #4 — Per-Expiry Greeks Map
    expiry_groups = {}
    expiry_greeks = {}  # display_expiry -> {delta, theta, ce_qty, pe_qty, expiry_date}

    # #5 — Today's Trade Impact
    today_delta = 0.0
    today_theta = 0.0

    for pos in positions:
        symbol = pos.get("symbol", "")
        qty = int(pos.get("quantity") or pos.get("quantity_total") or pos.get("netqty") or 0)

        avg_price    = float(pos.get("average_price") or pos.get("avg_price") or pos.get("price") or 0.0)
        ltp          = float(pos.get("last_price") or pos.get("ltp") or pos.get("close_price") or 0.0)
        pos_pnl      = float(pos.get("pnl") or 0.0)
        pos_realized = float(pos.get("today_realized_pnl") or 0.0)
        pos_unrealized = float(pos.get("unrealized_pnl") or 0.0)

        realized_pnl += pos_realized

        if qty == 0:
            processed_positions.append({
                "symbol":   symbol,
                "opt_type": "CE" if "CE" in symbol else "PE",
                "strike":   parse_option_symbol(symbol)["strike"],
                "expiry":   parse_option_symbol(symbol)["display_expiry"],
                "dte":      parse_option_symbol(symbol)["dte"],
                "qty": 0, "lots": 0.0,
                "avg_price": round(avg_price, 2),
                "ltp":       round(ltp, 2),
                "pnl":       round(pos_pnl, 2),
                "pnl_pct":   0.0,
                "status":    "CLOSED",
                "delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0, "iv": 0.0,
            })
            continue

        unrealized_pnl += pos_unrealized

        if ltp < min_premium:
            continue

        parsed = parse_option_symbol(symbol)

        if parsed["dte"] < min_days:
            continue

        pnl_pct = float(pos.get("pnlpercent") or 0.0)

        # --- #2: Live IV Back-Solver (Newton-Raphson from LTP, VIX as seed) ---
        live_iv = estimate_live_iv(
            spot, parsed["strike"], parsed["dte"], ltp, parsed["opt_type"], fallback=vix
        )

        # Calculate Greeks using live IV instead of fixed VIX
        greeks = calculate_greeks(spot, parsed["strike"], parsed["dte"], live_iv, parsed["opt_type"])

        leg_delta = round(qty * greeks["delta"], 2)
        leg_theta = round(qty * greeks["theta"], 2)
        leg_gamma = round(qty * greeks["gamma"], 6)
        leg_vega  = round(qty * greeks["vega"],  2)

        overall_delta += leg_delta
        overall_theta += leg_theta
        overall_gamma += leg_gamma
        overall_vega  += leg_vega

        # --- #5: Today's Trade Impact ---
        today_qty = int(pos.get("day_buy_quantity", 0)) - int(pos.get("day_sell_quantity", 0))
        if today_qty != 0:
            today_delta += round(today_qty * greeks["delta"], 2)
            today_theta += round(today_qty * greeks["theta"], 2)

        # --- #3: Time Value vs Intrinsic (short positions only, qty < 0) ---
        if qty < 0 and ltp > 0:
            if parsed["opt_type"] == "CE":
                intrinsic = max(spot - parsed["strike"], 0.0)
            else:
                intrinsic = max(parsed["strike"] - spot, 0.0)
            time_val = max(ltp - intrinsic, 0.0)
            abs_qty = abs(qty)
            total_time_value += abs_qty * time_val
            total_itm_risk   += abs_qty * intrinsic

        # Track sold qty
        if parsed["opt_type"] == "PE" and qty < 0:
            total_puts_sold_qty += abs(qty)
        elif parsed["opt_type"] == "CE" and qty < 0:
            total_calls_sold_qty += abs(qty)

        # --- #4: Per-Expiry Greeks Breakdown ---
        exp_key = parsed["display_expiry"]
        if exp_key not in expiry_groups:
            expiry_groups[exp_key] = {"puts_qty": 0, "calls_qty": 0, "puts_lots": 0.0, "calls_lots": 0.0}

        if parsed["opt_type"] == "PE":
            expiry_groups[exp_key]["puts_qty"]  += abs(qty) if qty < 0 else -abs(qty)
        elif parsed["opt_type"] == "CE":
            expiry_groups[exp_key]["calls_qty"] += abs(qty) if qty < 0 else -abs(qty)

        if exp_key not in expiry_greeks:
            expiry_greeks[exp_key] = {
                "delta": 0.0, "theta": 0.0,
                "ce_qty": 0, "pe_qty": 0,
                "expiry_date": parsed.get("expiry_date", exp_key),
            }
        expiry_greeks[exp_key]["delta"] = round(expiry_greeks[exp_key]["delta"] + leg_delta, 2)
        expiry_greeks[exp_key]["theta"] = round(expiry_greeks[exp_key]["theta"] + leg_theta, 2)
        if parsed["opt_type"] == "CE":
            expiry_greeks[exp_key]["ce_qty"] += qty
        else:
            expiry_greeks[exp_key]["pe_qty"] += qty

        processed_positions.append({
            "symbol":   symbol,
            "opt_type": parsed["opt_type"],
            "strike":   parsed["strike"],
            "expiry":   parsed["display_expiry"],
            "dte":      parsed["dte"],
            "qty":      qty,
            "lots":     round(qty / float(NIFTY_LOT_SIZE), 2),
            "avg_price": round(avg_price, 2),
            "ltp":       round(ltp, 2),
            "pnl":       round(pos_pnl, 2),
            "pnl_pct":   round(pnl_pct, 2),
            "status":    "OPEN",
            "delta": leg_delta,
            "theta": leg_theta,
            "gamma": leg_gamma,
            "vega":  leg_vega,
            "iv":    live_iv,
        })

    total_pnl = float(pos_res.get("total_pnl") or (realized_pnl + unrealized_pnl))

    for key, data in expiry_groups.items():
        data["puts_lots"]  = round(data["puts_qty"]  / float(NIFTY_LOT_SIZE), 1)
        data["calls_lots"] = round(data["calls_qty"] / float(NIFTY_LOT_SIZE), 1)

    puts_sold_lots  = round(total_puts_sold_qty  / float(NIFTY_LOT_SIZE), 1)
    calls_sold_lots = round(total_calls_sold_qty / float(NIFTY_LOT_SIZE), 1)

    total_short_lots = puts_sold_lots + calls_sold_lots
    estimated_margin = round(total_short_lots * 120000.0, 2)

    if overall_delta > 100.0:
        sentiment = "STRONGLY BULLISH";  sentiment_color = "#10b981"
    elif 25.0 < overall_delta <= 100.0:
        sentiment = "MILDLY BULLISH";    sentiment_color = "#34d399"
    elif -25.0 <= overall_delta <= 25.0:
        sentiment = "DELTA NEUTRAL";     sentiment_color = "#3b82f6"
    elif -100.0 <= overall_delta < -25.0:
        sentiment = "MILDLY BEARISH";   sentiment_color = "#f87171"
    else:
        sentiment = "STRONGLY BEARISH"; sentiment_color = "#ef4444"

    pnl_plus_100  = round(overall_delta *  100.0 + 0.5 * overall_gamma * (100.0**2),  2)
    pnl_minus_100 = round(overall_delta * -100.0 + 0.5 * overall_gamma * (100.0**2),  2)

    return jsonify({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "spot": round(spot, 2),
        "vix":  round(vix, 2),
        "is_cached": pos_res.get("is_cached", False),
        "last_successful_fetch": pos_res.get("last_successful_fetch"),
        "summary": {
            "puts_sold":        total_puts_sold_qty,
            "calls_sold":       total_calls_sold_qty,
            "puts_sold_lots":   puts_sold_lots,
            "calls_sold_lots":  calls_sold_lots,
            "overall_delta":    round(overall_delta, 2),
            "overall_theta":    round(overall_theta, 2),
            "overall_gamma":    round(overall_gamma, 6),
            "overall_vega":     round(overall_vega,  2),
            "unrealized_pnl":   round(unrealized_pnl, 2),
            "realized_pnl":     round(realized_pnl,   2),
            "total_pnl":        round(total_pnl,       2),
            "margin_required":  estimated_margin,
            "sentiment":        sentiment,
            "sentiment_color":  sentiment_color,
            "pnl_plus_100":     pnl_plus_100,
            "pnl_minus_100":    pnl_minus_100,
            # #3 Time Value vs Intrinsic
            "time_value_total": round(total_time_value, 2),
            "itm_risk_total":   round(total_itm_risk,   2),
            # #5 Today's Trade Impact
            "today_delta":      round(today_delta, 2),
            "today_theta":      round(today_theta, 2),
        },
        "expiry_groups": expiry_groups,
        "expiry_greeks":  expiry_greeks,  # #4 Per-Expiry Greeks
        "positions":      processed_positions,
    })


@app.route("/api/payoff")
def payoff_graph():
    """
    Returns Portfolio Payoff Curve (At Expiry PnL and T+0 BS Live PnL)
    across a range of spot prices (Spot - 2000 to Spot + 2000 in 25pt steps).
    Supports optional query param `expiry` to filter by specific expiry date.
    """
    expiry_filter = request.args.get("expiry", "ALL")
    spot, vix = fetch_live_quotes()
    pos_res = fetch_positionbook()
    positions = pos_res.get("data", [])

    if spot <= 0:
        spot = 24400.0

    active_legs = []
    for pos in positions:
        sym = pos.get("symbol", "")
        qty = int(pos.get("quantity") or pos.get("netqty") or 0)
        if not sym.startswith("NIFTY") or qty == 0:
            continue

        parsed = parse_option_symbol(sym)
        if not parsed:
            continue

        if expiry_filter != "ALL" and parsed["display_expiry"] != expiry_filter and parsed.get("expiry_date") != expiry_filter:
            continue

        avg_price = float(pos.get("sell_price") or pos.get("average_sell_price") or pos.get("average_price") or pos.get("buy_price") or pos.get("price") or 0.0)
        ltp = float(pos.get("last_price") or pos.get("ltp") or 0.0)
        live_iv = estimate_live_iv(spot, parsed["strike"], parsed["dte"], ltp, parsed["opt_type"], fallback=vix)

        active_legs.append({
            "symbol": sym,
            "opt_type": parsed["opt_type"],
            "strike": parsed["strike"],
            "qty": qty,
            "avg_price": avg_price,
            "ltp": ltp,
            "dte": parsed["dte"],
            "iv": live_iv,
        })

    # Range of spots: Spot - 2000 to Spot + 2000 in 25pt steps
    min_spot = int(math.floor((spot - 2000) / 50.0) * 50)
    max_spot = int(math.ceil((spot + 2000) / 50.0) * 50)
    step = 25

    spot_range = list(range(min_spot, max_spot + 1, step))
    payoff_expiry = []
    payoff_today = []

    for s in spot_range:
        pnl_exp = 0.0
        pnl_t0 = 0.0

        for leg in active_legs:
            K = leg["strike"]
            Q = leg["qty"]
            P_avg = leg["avg_price"]
            opt_type = leg["opt_type"]
            T = max(leg["dte"], 0.5) / 365.0
            sigma = max(leg["iv"], 1.0) / 100.0

            # Expiry Payoff
            if opt_type == "CE":
                intrinsic_exp = max(0.0, s - K)
                price_t0 = _bs_call_price(s, K, T, sigma)
            else:
                intrinsic_exp = max(0.0, K - s)
                price_t0 = _bs_put_price(s, K, T, sigma)

            pnl_exp += Q * (intrinsic_exp - P_avg)
            pnl_t0  += Q * (price_t0 - P_avg)

        payoff_expiry.append(round(pnl_exp, 2))
        payoff_today.append(round(pnl_t0, 2))

    # Breakeven points (where payoff_expiry crosses 0)
    breakevens = []
    for i in range(len(spot_range) - 1):
        y1 = payoff_expiry[i]
        y2 = payoff_expiry[i+1]
        if (y1 <= 0 and y2 >= 0) or (y1 >= 0 and y2 <= 0):
            x1, x2 = spot_range[i], spot_range[i+1]
            if y2 != y1:
                be = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                breakevens.append(round(be, 1))

    max_pnl = max(payoff_expiry) if payoff_expiry else 0.0
    min_pnl = min(payoff_expiry) if payoff_expiry else 0.0

    # Calculate Standard Deviation Levels (-2SD, -1SD, +1SD, +2SD)
    avg_dte = (sum(leg["dte"] for leg in active_legs) / len(active_legs)) if active_legs else 7.0
    avg_dte = max(avg_dte, 1.0)
    sd_pts = spot * (vix / 100.0) * math.sqrt(avg_dte / 365.0)

    sd_levels = {
        "m2sd": round(spot - 2.0 * sd_pts),
        "m1sd": round(spot - 1.0 * sd_pts),
        "p1sd": round(spot + 1.0 * sd_pts),
        "p2sd": round(spot + 2.0 * sd_pts),
        "sd_pts": round(sd_pts, 1),
        "avg_dte": round(avg_dte, 1),
    }

    # Projected PnL at current spot
    current_pnl_exp = 0.0
    current_pnl_today = 0.0
    for leg in active_legs:
        K = leg["strike"]
        Q = leg["qty"]
        P_avg = leg["avg_price"]
        opt_type = leg["opt_type"]
        T = max(leg["dte"], 0.5) / 365.0
        sigma = max(leg["iv"], 1.0) / 100.0

        if opt_type == "CE":
            intrinsic_exp = max(0.0, spot - K)
            price_t0 = _bs_call_price(spot, K, T, sigma)
        else:
            intrinsic_exp = max(0.0, K - spot)
            price_t0 = _bs_put_price(spot, K, T, sigma)

        current_pnl_exp += Q * (intrinsic_exp - P_avg)
        current_pnl_today += Q * (price_t0 - P_avg)

    return jsonify({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "spot": round(spot, 2),
        "vix": round(vix, 2),
        "expiry_filter": expiry_filter,
        "spot_range": spot_range,
        "payoff_expiry": payoff_expiry,
        "payoff_today": payoff_today,
        "breakevens": breakevens,
        "max_profit": round(max_pnl, 2),
        "max_loss": round(min_pnl, 2),
        "current_pnl_exp": round(current_pnl_exp, 2),
        "current_pnl_today": round(current_pnl_today, 2),
        "sd_levels": sd_levels,
        "leg_count": len(active_legs),
    })


@app.route("/api/position-guard")
def position_guard():
    """
    #1 — Cross-Strategy Duplicate Long Detector.
    Scans all NIFTY NFO positions for unintended net long exposure.
    Flags any CE/PE symbol with net qty > 0 — unexpected for a short-selling system.
    HIGH severity: >= 1 full lot. WARN: fractional long.
    """
    pos_res   = fetch_positionbook()
    positions = pos_res.get("data", [])

    alerts = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        qty    = int(pos.get("quantity") or pos.get("netqty") or 0)

        if qty > 0 and ("CE" in symbol or "PE" in symbol) and symbol.startswith("NIFTY"):
            parsed = parse_option_symbol(symbol)
            ltp    = float(pos.get("last_price") or pos.get("ltp") or 0.0)
            alerts.append({
                "symbol":   symbol,
                "qty":      qty,
                "lots":     round(qty / float(NIFTY_LOT_SIZE), 2),
                "opt_type": parsed["opt_type"],
                "strike":   parsed["strike"],
                "expiry":   parsed["display_expiry"],
                "ltp":      round(ltp, 2),
                "severity": "HIGH" if qty >= NIFTY_LOT_SIZE else "WARN",
            })

    return jsonify({
        "alerts":      alerts,
        "alert_count": len(alerts),
        "checked_at":  datetime.now().strftime("%H:%M:%S"),
    })


# ============================================================
# 0-DTE GAMMA BLAST OI ACCUMULATION MATRIX TRACKER
# Jane Street Detector: Monitors ATM ± 2 Strikes across 3m, 5m, 15m, 30m
# ============================================================
from collections import deque

class GammaMatrixBuffer:
    def __init__(self):
        self.snapshots = deque(maxlen=200)

    def push(self, spot, vix, quotes):
        self.snapshots.append({
            "time": datetime.now(),
            "spot": spot,
            "vix": vix,
            "quotes": quotes  # dict: symbol -> {ltp, oi}
        })

    def _get_entry_at(self, window_secs):
        now = datetime.now()
        cutoff = now - datetime.timedelta(seconds=window_secs) if hasattr(datetime, "timedelta") else now
        from datetime import timedelta
        cutoff = now - timedelta(seconds=window_secs)
        for entry in reversed(self.snapshots):
            if entry["time"] <= cutoff:
                return entry
        return self.snapshots[0] if self.snapshots else None

    def get_oi_change_pct(self, symbol, window_secs):
        if not self.snapshots:
            return 0.0
        latest = self.snapshots[-1]["quotes"].get(symbol, {})
        latest_oi = float(latest.get("oi") or 0.0)
        ref = self._get_entry_at(window_secs)
        if not ref:
            return 0.0
        ref_oi = float(ref["quotes"].get(symbol, {}).get("oi") or 0.0)
        if ref_oi <= 0:
            return 0.0
        return round(((latest_oi - ref_oi) / ref_oi) * 100.0, 1)

    def get_spot_change_pct(self, window_secs):
        if not self.snapshots:
            return 0.0
        latest_spot = self.snapshots[-1]["spot"]
        ref = self._get_entry_at(window_secs)
        if not ref or ref["spot"] <= 0:
            return 0.0
        return round(((latest_spot - ref["spot"]) / ref["spot"]) * 100.0, 2)

gamma_tracker = GammaMatrixBuffer()

@app.route("/api/gamma-blast-matrix")
def gamma_blast_matrix():
    """
    Returns 0-DTE Gamma Blast OI Accumulation Matrix (Jane Street Detector).
    3 Live Tables: Calls OI Matrix, Puts OI Matrix, NIFTY Spot Lag Table.
    Cell ratings: 3m >= 10%, 5m >= 12%, 15m >= 30%.
    """
    spot, vix = fetch_live_quotes()
    step = 50
    atm = int(round(spot / step) * step)

    # Monitored strikes: ATM-2, ATM-1, ATM, ATM+1, ATM+2
    strike_offsets = [-2, -1, 0, 1, 2]
    strikes = [atm + (o * step) for o in strike_offsets]
    strike_labels = ["ATM-2", "ATM-1", "ATM", "ATM+1", "ATM+2"]

    # Fetch live quotes for these strikes
    live_quotes = {}
    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            sym = f"NIFTY{int(strike)}{opt_type}"
            try:
                resp = requests.post(
                    f"{OPENALGO_HOST}/api/v1/quotes",
                    json={"apikey": OPENALGO_API_KEY, "exchange": "NFO", "symbol": sym},
                    timeout=1.5
                )
                if resp.status_code == 200:
                    q = resp.json()
                    quote_data = q.get("data", q) if isinstance(q, dict) else {}
                    if isinstance(quote_data, dict):
                        live_quotes[sym] = {
                            "ltp": float(quote_data.get("ltp") or quote_data.get("last_price") or 0.0),
                            "oi":  float(quote_data.get("oi")  or quote_data.get("open_interest") or 0.0)
                        }
            except Exception:
                live_quotes[sym] = {"ltp": 0.0, "oi": 0.0}

    gamma_tracker.push(spot, vix, live_quotes)

    windows = [("3m", 180), ("5m", 300), ("15m", 900), ("30m", 1800)]

    call_matrix = []
    put_matrix = []
    breached_cells = 0

    for idx, strike in enumerate(strikes):
        label = f"{strike_labels[idx]} ({strike})"

        # Call Row
        ce_sym = f"NIFTY{int(strike)}CE"
        ce_row = {"label": label, "strike": strike, "sym": ce_sym, "changes": {}}
        for win_name, win_secs in windows:
            oi_change = gamma_tracker.get_oi_change_pct(ce_sym, win_secs)
            breached = (win_name == "3m" and oi_change >= 10.0) or \
                       (win_name == "5m" and oi_change >= 12.0) or \
                       (win_name == "15m" and oi_change >= 30.0)
            if breached:
                breached_cells += 1
            ce_row["changes"][win_name] = {"pct": oi_change, "breached": breached}
        call_matrix.append(ce_row)

        # Put Row
        pe_sym = f"NIFTY{int(strike)}PE"
        pe_row = {"label": label, "strike": strike, "sym": pe_sym, "changes": {}}
        for win_name, win_secs in windows:
            oi_change = gamma_tracker.get_oi_change_pct(pe_sym, win_secs)
            breached = (win_name == "3m" and oi_change >= 10.0) or \
                       (win_name == "5m" and oi_change >= 12.0) or \
                       (win_name == "15m" and oi_change >= 30.0)
            if breached:
                breached_cells += 1
            pe_row["changes"][win_name] = {"pct": oi_change, "breached": breached}
        put_matrix.append(pe_row)

    # Spot Lag Matrix
    spot_changes = {}
    for win_name, win_secs in windows:
        spot_changes[win_name] = gamma_tracker.get_spot_change_pct(win_secs)

    alert_active = (breached_cells >= 3)  # 30%+ of cells breach threshold

    return jsonify({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "spot": spot,
        "atm": atm,
        "call_matrix": call_matrix,
        "put_matrix": put_matrix,
        "spot_changes": spot_changes,
        "breached_cells": breached_cells,
        "alert_active": alert_active,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    logger.info(f"🚀 Starting NIFTY Portfolio Analytics Dashboard on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
