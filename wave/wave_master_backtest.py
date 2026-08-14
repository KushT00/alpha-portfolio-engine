"""
Nifty Wave Master - Option Selling Backtest Engine & Heatmap Report
=====================================================================
Backtests the Wave Scraper strategy (nifty_wave_master.py) using Option Selling mode
on 1-minute Nifty spot data and India VIX for dynamic IV.

Features:
  - Option Selling PnL & Delta Gating.
  - Daily Margin Requirements (SPAN + Exposure).
  - HTML Interactive Report with Daily PnL Calendar Heatmap (matching image).
"""

import os
import calendar
import json
import pandas as pd
import numpy as np
from math import log, sqrt, exp, erf
from datetime import datetime, date, time, timedelta

# ============================================================
# STRATEGY PARAMETERS
# ============================================================
BUY_GAP             = 25.0
SELL_GAP            = 25.0
LOT_SIZE            = 65
COOL_OFF_BARS       = 10
STRIKE_INTERVAL     = 50
MAX_NIFTY_DELTA     = 100.0
MIN_NIFTY_DELTA     = -100.0
RISK_FREE_RATE      = 0.07
TARGET_PREMIUM      = 200.0
MARGIN_PCT          = 0.11          # ~11% SPAN + Exposure Margin for Short Nifty Option

ANALYSIS_START_DATE = "2026-07-01"
ANALYSIS_END_DATE   = "2026-07-31"

MULTIPLIER_SCALE = {
     0: [1.0,  1.0],
     1: [1.3,  1.0],  2: [1.7,  1.0],  3: [2.5,  1.0],
     4: [3.0,  1.0],  5: [10.0, 1.0],  6: [10.0, 1.0],
     7: [10.0, 1.0],  8: [15.0, 1.0],  9: [15.0, 1.0],  10: [15.0, 1.0],
    -1: [1.0,  1.3], -2: [1.0,  1.7], -3: [1.0,  2.5],
    -4: [1.0,  3.0], -5: [1.0, 10.0], -6: [1.0, 10.0],
    -7: [1.0, 10.0], -8: [1.0, 15.0], -9: [1.0, 15.0], -10: [1.0, 15.0],
}

# ============================================================
# BLACK-SCHOLES
# ============================================================
def norm_cdf(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def bs_d2(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * sqrt(T)

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K)
    d1, d2 = bs_d1(S, K, T, r, sigma), bs_d2(S, K, T, r, sigma)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, K - S)
    d1, d2 = bs_d1(S, K, T, r, sigma), bs_d2(S, K, T, r, sigma)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm_cdf(bs_d1(S, K, T, r, sigma))

def bs_put_delta(S, K, T, r, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm_cdf(bs_d1(S, K, T, r, sigma)) - 1.0

# ============================================================
# EXPIRY & STRIKE HELPERS
# ============================================================
def get_next_thursday_expiry(dt):
    days_ahead = 3 - dt.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        expiry_time = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        if dt < expiry_time:
            return expiry_time
        days_ahead = 7
    nxt = dt + timedelta(days=days_ahead)
    return nxt.replace(hour=15, minute=30, second=0, microsecond=0)

def time_to_expiry_years(current_dt, expiry_dt):
    diff = (expiry_dt - current_dt).total_seconds()
    return max(0.0, diff / (365.25 * 24 * 3600))

def find_strike_near_premium(S, current_dt, option_type,
                              target_premium=200.0, r=RISK_FREE_RATE, sigma=0.14):
    expiry_dt = get_next_thursday_expiry(current_dt)
    T = time_to_expiry_years(current_dt, expiry_dt)
    if T <= 0:
        expiry_dt = get_next_thursday_expiry(current_dt + timedelta(days=1))
        T = time_to_expiry_years(current_dt, expiry_dt)
    atm = round(S / STRIKE_INTERVAL) * STRIKE_INTERVAL
    best_strike, best_premium, best_delta, min_diff = atm, 0.0, 0.0, float("inf")
    for offset in range(-40, 41):
        K = atm + offset * STRIKE_INTERVAL
        if K <= 0:
            continue
        if option_type == "CE":
            premium = bs_call(S, K, T, r, sigma)
            delta   = bs_call_delta(S, K, T, r, sigma)
        else:
            premium = bs_put(S, K, T, r, sigma)
            delta   = abs(bs_put_delta(S, K, T, r, sigma))
        diff = abs(premium - target_premium)
        if diff < min_diff:
            min_diff = diff
            best_strike  = K
            best_premium = premium
            best_delta   = delta
    return best_strike, best_premium, best_delta, expiry_dt

def compute_option_price(S, K, expiry_dt, current_dt, option_type,
                          r=RISK_FREE_RATE, sigma=0.14):
    T = time_to_expiry_years(current_dt, expiry_dt)
    if option_type == "CE":
        return bs_call(S, K, T, r, sigma)
    return bs_put(S, K, T, r, sigma)

def compute_option_delta(S, K, expiry_dt, current_dt, option_type,
                          r=RISK_FREE_RATE, sigma=0.14):
    T = time_to_expiry_years(current_dt, expiry_dt)
    if option_type == "CE":
        return bs_call_delta(S, K, T, r, sigma)
    return abs(bs_put_delta(S, K, T, r, sigma))

def get_scaled_gaps(diff_scale):
    key  = max(-10, min(10, int(round(diff_scale))))
    mult = MULTIPLIER_SCALE.get(key, [10.0, 10.0])
    return round(BUY_GAP * mult[0], 1), round(SELL_GAP * mult[1], 1)

def compute_margin_required(spot_price, num_lots):
    if num_lots <= 0:
        return 0.0
    return spot_price * LOT_SIZE * MARGIN_PCT * num_lots

# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def run_backtest():
    print("=" * 80)
    print("  Nifty Wave Master - Option Selling & Heatmap Engine")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\nLoading 1-min Nifty data...")
    df = pd.read_csv(os.path.join(base_dir, "nifty.csv"))
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  Loaded {len(df):,} bars  |  {df['datetime'].iloc[0]}  to  {df['datetime'].iloc[-1]}")

    print("Loading India VIX data...")
    vix_df = pd.read_csv(os.path.join(base_dir, "vix.csv"))
    vix_df["vix_dt"] = pd.to_datetime(vix_df["date"] + " " + vix_df["time"])
    vix_df = vix_df.sort_values("vix_dt").reset_index(drop=True)
    vix_daily = (
        vix_df.groupby(vix_df["vix_dt"].dt.normalize())["close"]
        .last().reset_index()
        .rename(columns={"vix_dt": "vix_date", "close": "vix_close"})
    )
    vix_daily["iv"] = vix_daily["vix_close"] / 100.0
    df["trade_date"] = df["datetime"].dt.normalize()
    df = df.merge(vix_daily[["vix_date", "iv"]], left_on="trade_date", right_on="vix_date", how="left")
    df["iv"] = df["iv"].ffill().bfill()
    df.drop(columns=["vix_date"], inplace=True, errors="ignore")

    # Strategy State
    initial_position   = 0
    current_position   = 0
    portfolio_delta    = 0.0
    pending_buy_price  = None
    pending_sell_price = None
    cycle_cooloff      = 0
    trades             = []
    open_short_trades  = []
    last_day           = None

    daily_margin_records = []
    current_day_max_margin = 0.0

    print("\nRunning Option Selling simulation on 1-min bars...\n")

    for _idx, row in df.iterrows():
        current_dt    = row["datetime"]
        current_high  = row["high"]
        current_low   = row["low"]
        current_close = row["close"]
        current_iv    = row["iv"]
        market_time   = current_dt.time()

        in_session = (current_dt.weekday() < 5 and time(9, 15) <= market_time < time(15, 30))
        if not in_session:
            continue

        today = current_dt.date()
        if today != last_day:
            if last_day is not None:
                daily_margin_records.append({
                    "date": last_day,
                    "max_margin": current_day_max_margin
                })
            initial_position       = current_position
            last_day               = today
            current_day_max_margin = 0.0
            pending_buy_price      = None
            pending_sell_price     = None
            cycle_cooloff          = 0

        open_lots = len(open_short_trades)
        current_bar_margin = compute_margin_required(current_close, open_lots)
        if current_bar_margin > current_day_max_margin:
            current_day_max_margin = current_bar_margin

        # Step 1: Fills
        fills_this_bar = []
        if pending_sell_price is not None and current_high >= pending_sell_price:
            fills_this_bar.append(("SELL_SHORT", pending_sell_price, current_iv, current_dt))
        if pending_buy_price is not None and current_low <= pending_buy_price:
            fills_this_bar.append(("BUY_COVER", pending_buy_price, current_iv, current_dt))

        for fill_side, fill_price, fill_iv, fill_dt in fills_this_bar:
            if fill_side == "SELL_SHORT":
                current_position -= 1
                portfolio_delta   = float(current_position)
                strike, entry_prem, entry_delta, expiry = find_strike_near_premium(
                    fill_price, fill_dt, "CE", target_premium=TARGET_PREMIUM, sigma=fill_iv
                )
                open_short_trades.append({
                    "entry_time":       fill_dt,
                    "entry_price":      fill_price,
                    "opt_type":         "CE",
                    "opt_strike":       strike,
                    "opt_entry_prem":   entry_prem,
                    "opt_entry_delta":  entry_delta,
                    "opt_expiry_dt":    expiry,
                })
                pending_sell_price = None

            elif fill_side == "BUY_COVER":
                current_position += 1
                portfolio_delta   = float(current_position)
                if open_short_trades:
                    leg      = open_short_trades.pop(0)
                    spot_pnl = leg["entry_price"] - fill_price
                    opt_exit_prem  = compute_option_price(
                        fill_price, leg["opt_strike"], leg["opt_expiry_dt"], fill_dt, leg["opt_type"], sigma=fill_iv
                    )
                    opt_exit_delta = compute_option_delta(
                        fill_price, leg["opt_strike"], leg["opt_expiry_dt"], fill_dt, leg["opt_type"], sigma=fill_iv
                    )
                    opt_pnl     = leg["opt_entry_prem"] - opt_exit_prem
                    opt_pnl_pct = (opt_pnl / leg["opt_entry_prem"] * 100 if leg["opt_entry_prem"] > 0 else 0.0)
                    hold_hours  = (fill_dt - leg["entry_time"]).total_seconds() / 3600
                    trades.append({
                        "type":              "SHORT_OPTION",
                        "entry_reason":      "SELL_PREMIUM",
                        "entry_time":        leg["entry_time"],
                        "entry_price":       leg["entry_price"],
                        "exit_time":         fill_dt,
                        "exit_price":        fill_price,
                        "exit_reason":       "BUY_COVER",
                        "pnl_points":        spot_pnl,
                        "pnl_pct":           spot_pnl / leg["entry_price"] * 100,
                        "opt_strike":        leg["opt_strike"],
                        "opt_entry_premium": leg["opt_entry_prem"],
                        "opt_exit_premium":  opt_exit_prem,
                        "opt_pnl":           opt_pnl,
                        "opt_pnl_pct":       opt_pnl_pct,
                        "opt_entry_delta":   leg["opt_entry_delta"],
                        "opt_exit_delta":    opt_exit_delta,
                        "opt_pnl_per_lot":   opt_pnl * LOT_SIZE,
                        "hold_hours":        hold_hours,
                    })
                pending_buy_price = None

        # Step 2: Orders
        if cycle_cooloff > 0:
            cycle_cooloff -= 1
        else:
            diff_scale = current_position - initial_position
            scaled_buy_gap, scaled_sell_gap = get_scaled_gaps(diff_scale)

            restrict_buy  = portfolio_delta >= MAX_NIFTY_DELTA
            restrict_sell = portfolio_delta <= MIN_NIFTY_DELTA

            if not restrict_sell:
                pending_sell_price = round(current_close + scaled_sell_gap, 2)
            if not restrict_buy:
                pending_buy_price  = round(current_close - scaled_buy_gap, 2)

            cycle_cooloff = COOL_OFF_BARS

        # Step 3: End of Session
        if market_time >= time(15, 29) and open_short_trades:
            for leg in open_short_trades:
                spot_pnl = leg["entry_price"] - current_close
                opt_exit_prem  = compute_option_price(
                    current_close, leg["opt_strike"], leg["opt_expiry_dt"], current_dt, leg["opt_type"], sigma=current_iv
                )
                opt_exit_delta = compute_option_delta(
                    current_close, leg["opt_strike"], leg["opt_expiry_dt"], current_dt, leg["opt_type"], sigma=current_iv
                )
                opt_pnl     = leg["opt_entry_prem"] - opt_exit_prem
                opt_pnl_pct = (opt_pnl / leg["opt_entry_prem"] * 100 if leg["opt_entry_prem"] > 0 else 0.0)
                hold_hours  = (current_dt - leg["entry_time"]).total_seconds() / 3600
                trades.append({
                    "type":              "SHORT_OPTION",
                    "entry_reason":      "SELL_PREMIUM",
                    "entry_time":        leg["entry_time"],
                    "entry_price":       leg["entry_price"],
                    "exit_time":         current_dt,
                    "exit_price":        current_close,
                    "exit_reason":       "EOD_CLOSE",
                    "pnl_points":        spot_pnl,
                    "pnl_pct":           spot_pnl / leg["entry_price"] * 100,
                    "opt_strike":        leg["opt_strike"],
                    "opt_entry_premium": leg["opt_entry_prem"],
                    "opt_exit_premium":  opt_exit_prem,
                    "opt_pnl":           opt_pnl,
                    "opt_pnl_pct":       opt_pnl_pct,
                    "opt_entry_delta":   leg["opt_entry_delta"],
                    "opt_exit_delta":    opt_exit_delta,
                    "opt_pnl_per_lot":   opt_pnl * LOT_SIZE,
                    "hold_hours":        hold_hours,
                })
            open_short_trades.clear()
            current_position   = 0
            portfolio_delta    = 0.0
            pending_buy_price  = None
            pending_sell_price = None

    if last_day is not None:
        daily_margin_records.append({
            "date": last_day,
            "max_margin": current_day_max_margin
        })

    margin_df = pd.DataFrame(daily_margin_records)
    avg_daily_margin = margin_df["max_margin"].mean() if not margin_df.empty else 0.0
    max_peak_margin  = margin_df["max_margin"].max()  if not margin_df.empty else 0.0
    single_lot_margin= df["close"].mean() * LOT_SIZE * MARGIN_PCT

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades completed!")
        return

    full_n         = len(trades_df)
    full_opt_total = trades_df["opt_pnl"].sum()
    full_opt_wins  = (trades_df["opt_pnl"] > 0).sum()
    full_opt_wr    = full_opt_wins / full_n * 100
    full_opt_lot   = trades_df["opt_pnl_per_lot"].sum()

    print("=" * 80)
    print("  OPTION SELLING BACKTEST RESULTS & MARGIN REQUIREMENTS")
    print("=" * 80)
    print(f"  * Mode                  : OPTION SELLING (Short Options)")
    print(f"  * Total Trades          : {full_n}")
    print(f"  * Option Win Rate       : {full_opt_wr:.1f}%")
    print(f"  * Total PnL (Points)    : {full_opt_total:+.1f} pts per unit")
    print(f"  * Total PnL (Rs / Lot)  : Rs. {full_opt_lot:+,.0f}")
    print("-" * 80)
    print(f"  * Single Lot Margin Est  : ~Rs. {single_lot_margin:,.0f}")
    print(f"  * Avg Daily Margin Req   : ~Rs. {avg_daily_margin:,.0f}")
    print(f"  * Max Peak Margin Req    : ~Rs. {max_peak_margin:,.0f}")
    print("=" * 80)

    # Generate HTML Report with Heatmap
    generate_html_report(trades_df, avg_daily_margin, max_peak_margin, single_lot_margin, df)

# ============================================================
# HEATMAP GENERATOR FUNCTION
# ============================================================
def generate_html_report(trades_df, avg_margin, max_margin, single_lot_margin, raw_df):
    trades_df['exit_day'] = trades_df['exit_time'].dt.normalize()
    daily_pnl = trades_df.groupby('exit_day')['opt_pnl_per_lot'].sum().reset_index()

    start_date = trades_df['entry_time'].min().normalize()
    end_date   = trades_df['exit_time'].max().normalize()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    daily_df = pd.DataFrame({'date': date_range})
    daily_df = daily_df.merge(daily_pnl, left_on='date', right_on='exit_day', how='left').fillna(0)

    # Build Heatmap Data structure grouped by Month
    daily_pnl_map = {r['date'].strftime("%Y-%m-%d"): r['opt_pnl_per_lot'] for _, r in daily_df.iterrows()}

    months_list = []
    # Loop over year-months present in data
    current_m = start_date.date().replace(day=1)
    end_date_val = end_date.date()
    while current_m <= end_date_val:
        year = current_m.year
        month = current_m.month
        month_name = calendar.month_name[month]
        num_days = calendar.monthrange(year, month)[1]

        # Day cells
        days = []
        # Pre-padding empty cells for day of week offset (Mon=0, Sun=6)
        first_weekday = date(year, month, 1).weekday()
        for _ in range(first_weekday):
            days.append({"day": 0, "pnl": None, "type": "pad"})

        for d in range(1, num_days + 1):
            date_str = f"{year}-{month:02d}-{d:02d}"
            dt_obj = date(year, month, d)
            is_weekend = dt_obj.weekday() >= 5
            pnl_val = daily_pnl_map.get(date_str, 0.0)

            if is_weekend:
                days.append({"day": d, "date": date_str, "pnl": 0.0, "type": "weekend"})
            else:
                days.append({"day": d, "date": date_str, "pnl": pnl_val, "type": "trade"})

        months_list.append({"name": month_name, "year": year, "days": days})
        # Increment month
        if month == 12:
            current_m = date(year + 1, 1, 1)
        else:
            current_m = date(year, month + 1, 1)

    # Summary Numbers
    full_opt_lot = trades_df['opt_pnl_per_lot'].sum()
    full_opt_wr  = (trades_df['opt_pnl'] > 0).sum() / len(trades_df) * 100
    total_trades = len(trades_df)

    months_json = json.dumps(months_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty Wave Master - Daily PnL Heatmap Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --cyan: #06b6d4;
            --purple: #8b5cf6;
            --amber: #f59e0b;
            --red: #ef4444;
            --green: #10b981;
            --txt: #f3f4f6;
            --muted: #9ca3af;
            --dim: #6b7280;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--bg);
            color: var(--txt);
            font-family: 'Inter', sans-serif;
            padding: 40px 5%;
            min-height: 100vh;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }}
        header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(to right, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        header p {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 6px;
            background: rgba(6, 182, 212, 0.15);
            color: #06b6d4;
            border: 1px solid rgba(6, 182, 212, 0.3);
        }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 35px;
        }}
        .card-stat {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 22px;
            backdrop-filter: blur(16px);
        }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }}
        .stat-value {{ font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
        .stat-sub {{ font-size: 11px; color: var(--dim); }}
        .text-green {{ color: #10b981 !important; }}
        .text-red {{ color: #ef4444 !important; }}

        /* HEATMAP SECTION */
        .heatmap-container-box {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(16px);
            margin-bottom: 35px;
        }}
        .heatmap-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #f3f4f6;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .heatmap-subtitle {{
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 25px;
        }}

        .months-flex {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            justify-content: flex-start;
        }}

        .month-block {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}

        .month-grid {{
            display: grid;
            grid-template-columns: repeat(7, 16px);
            gap: 5px;
            margin-bottom: 10px;
        }}

        .day-tile {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            position: relative;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .day-tile:hover {{
            transform: scale(1.35);
            z-index: 20;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}

        .tile-pad {{ background: transparent; pointer-events: none; }}
        .tile-weekend {{ background: rgba(255, 255, 255, 0.04); }}
        .tile-zero {{ background: rgba(255, 255, 255, 0.07); }}

        /* Profit Greens */
        .tile-g1 {{ background: #86efac; }}
        .tile-g2 {{ background: #4ade80; }}
        .tile-g3 {{ background: #22c55e; }}
        .tile-g4 {{ background: #15803d; }}

        /* Loss Reds */
        .tile-r1 {{ background: #fca5a5; }}
        .tile-r2 {{ background: #f87171; }}
        .tile-r3 {{ background: #ef4444; }}
        .tile-r4 {{ background: #b91c1c; }}

        .month-name {{
            font-size: 13px;
            font-weight: 600;
            color: var(--muted);
            margin-top: 4px;
        }}

        /* Tooltip */
        #tooltip {{
            position: absolute;
            display: none;
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}

        .legend-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 25px;
            font-size: 12px;
            color: var(--muted);
        }}
        .legend-box {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>NIFTY WAVE MASTER — DAILY PnL HEATMAP</h1>
            <p>Option Selling Mode · Jan 2026 – Jul 2026 · Daily Profit & Loss Calendar Breakdown</p>
        </div>
        <div class="badge">Option Selling Mode | Lot Size: 65</div>
    </header>

    <div class="grid-stats">
        <div class="card-stat">
            <div class="stat-label">Total Option PnL</div>
            <div class="stat-value text-green">Rs. {full_opt_lot:+,.0f}</div>
            <div class="stat-sub">{total_trades} Round Trips | {full_opt_wr:.1f}% Win Rate</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Single Lot Margin</div>
            <div class="stat-value">Rs. {single_lot_margin:,.0f}</div>
            <div class="stat-sub">SPAN + Exposure Margin per Short Lot</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Avg Daily Margin Req</div>
            <div class="stat-value">Rs. {avg_margin:,.0f}</div>
            <div class="stat-sub">Average Capital Held Daily</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Peak Max Margin Req</div>
            <div class="stat-value text-red">Rs. {max_margin:,.0f}</div>
            <div class="stat-sub">Max Peak Capital Scaling Required</div>
        </div>
    </div>

    <!-- HEATMAP CONTAINER -->
    <div class="heatmap-container-box">
        <div class="heatmap-title">
            <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            Daily Performance Heatmap
        </div>
        <div class="heatmap-subtitle">Visual daily breakdown of profit (green) vs loss (red) across all calendar months</div>

        <div class="months-flex" id="heatmapGrid"></div>

        <div class="legend-row">
            <span>Loss:</span>
            <div class="legend-box tile-r4"></div>
            <div class="legend-box tile-r3"></div>
            <div class="legend-box tile-r2"></div>
            <div class="legend-box tile-r1"></div>
            <span style="margin: 0 10px;">Zero / Weekend:</span>
            <div class="legend-box tile-weekend"></div>
            <span style="margin: 0 10px;">Profit:</span>
            <div class="legend-box tile-g1"></div>
            <div class="legend-box tile-g2"></div>
            <div class="legend-box tile-g3"></div>
            <div class="legend-box tile-g4"></div>
        </div>
    </div>

    <div id="tooltip"></div>

    <script>
        const monthsData = {months_json};
        const container = document.getElementById('heatmapGrid');
        const tooltip = document.getElementById('tooltip');

        monthsData.forEach(m => {{
            const monthBlock = document.createElement('div');
            monthBlock.className = 'month-block';

            const grid = document.createElement('div');
            grid.className = 'month-grid';

            m.days.forEach(d => {{
                const tile = document.createElement('div');
                tile.className = 'day-tile';

                if (d.type === 'pad') {{
                    tile.classList.add('tile-pad');
                }} else if (d.type === 'weekend') {{
                    tile.classList.add('tile-weekend');
                }} else {{
                    const pnl = d.pnl;
                    if (pnl === 0) {{
                        tile.classList.add('tile-zero');
                    }} else if (pnl > 0) {{
                        if (pnl > 100000) tile.classList.add('tile-g4');
                        else if (pnl > 50000) tile.classList.add('tile-g3');
                        else if (pnl > 20000) tile.classList.add('tile-g2');
                        else tile.classList.add('tile-g1');
                    }} else {{
                        const loss = Math.abs(pnl);
                        if (loss > 100000) tile.classList.add('tile-r4');
                        else if (loss > 50000) tile.classList.add('tile-r3');
                        else if (loss > 20000) tile.classList.add('tile-r2');
                        else tile.classList.add('tile-r1');
                    }}

                    // Hover tooltip
                    tile.addEventListener('mouseenter', (e) => {{
                        const formattedPnL = pnl >= 0 ? '+' + pnl.toLocaleString('en-IN', {{style: 'currency', currency: 'INR', maximumFractionDigits:0}}) : '-' + Math.abs(pnl).toLocaleString('en-IN', {{style: 'currency', currency: 'INR', maximumFractionDigits:0}});
                        tooltip.innerHTML = `<strong>${{d.date}}</strong><br>PnL: <span style="color:${{pnl >= 0 ? '#4ade80' : '#f87171'}}">${{formattedPnL}}</span>`;
                        tooltip.style.display = 'block';
                    }});

                    tile.addEventListener('mousemove', (e) => {{
                        tooltip.style.left = (e.pageX + 12) + 'px';
                        tooltip.style.top = (e.pageY + 12) + 'px';
                    }});

                    tile.addEventListener('mouseleave', () => {{
                        tooltip.style.display = 'none';
                    }});
                }}

                grid.appendChild(tile);
            }});

            const monthLabel = document.createElement('div');
            monthLabel.className = 'month-name';
            monthLabel.innerText = m.name;

            monthBlock.appendChild(grid);
            monthBlock.appendChild(monthLabel);
            container.appendChild(monthBlock);
        }});
    </script>

</body>
</html>"""

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "wave_master_backtest_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML Heatmap Report saved -> {html_path}")
    print("  Open in browser to view the daily PnL calendar heatmap.\n")


if __name__ == "__main__":
    run_backtest()