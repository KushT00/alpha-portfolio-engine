"""
Nifty Combined Strategy Backtest — Wave Master + Survivor Master
==================================================================
Strictly backtests the two core strategies in the repository:
  1. Wave Master (nifty_wave_master.py): Grid range scraper.
  2. Survivor Master (nifty_survivor_master.py): Systematic trend option selling.

Note:
  - 0DTE Gamma Blast code removed completely (no historical 1-min option chain OI data).
  - Monday EOD square-off enforced before weekly Tuesday expiry.
"""

import os
import calendar
import json
import pandas as pd
import numpy as np
from math import log, sqrt, exp, erf
from datetime import datetime, date, time, timedelta

# ============================================================
# STRATEGY PARAMETERS  (Matching repository strategy files)
# ============================================================
LOT_SIZE            = 65
STRIKE_INTERVAL     = 50
RISK_FREE_RATE      = 0.07
MARGIN_PCT          = 0.11          # ~11% SPAN + Exposure Margin for Short Nifty Option
SLIPPAGE_PTS        = 0.20          # 0.20 pt realistic option execution slippage
BROKERAGE_PER_TRADE = 40.0          # Rs. 40 round-trip brokerage & statutory fees

# WAVE MASTER PARAMETERS (nifty_wave_master.py)
WAVE_BUY_GAP        = 25.0
WAVE_SELL_GAP       = 25.0
WAVE_COOL_OFF_BARS  = 10
WAVE_TARGET_PREMIUM = 200.0
WAVE_MAX_OPEN_LOTS  = 3

WAVE_MULTIPLIER_SCALE = {
     0: [1.0,  1.0],  1: [1.3,  1.0],  2: [1.7,  1.0],  3: [2.5,  1.0],
    -1: [1.0,  1.3], -2: [1.0,  1.7], -3: [1.0,  2.5],
}

# SURVIVOR MASTER PARAMETERS (nifty_survivor_master.py)
SURVIVOR_PE_GAP        = 20.0        # Upward trigger threshold (pts)
SURVIVOR_CE_GAP        = 20.0        # Downward trigger threshold (pts)
SURVIVOR_PE_SYMBOL_GAP = 200.0       # OTM Put offset (~200pt OTM)
SURVIVOR_CE_SYMBOL_GAP = 200.0       # OTM Call offset (~200pt OTM)
SURVIVOR_MIN_PREMIUM   = 15.0        # Min option premium filter (Rs)
SURVIVOR_MAX_MULT      = 2
SURVIVOR_MAX_OPEN_LOTS = 4
SURVIVOR_PE_RESET_GAP  = 30.0        # Reversal reset threshold (pts)
SURVIVOR_CE_RESET_GAP  = 30.0        # Reversal reset threshold (pts)
SURVIVOR_ROLLING_TRIGGER_TRADES = 3  # Close oldest decayed leg after 3 active trades

# ============================================================
# BLACK-SCHOLES (Standard Option Pricing)
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
# EXPIRY HELPERS (Weekly Tuesday Expiry)
# ============================================================
def get_next_tuesday_expiry(dt):
    days_ahead = 1 - dt.weekday()
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

def find_strike_near_premium(S, current_dt, option_type, target_premium=200.0, r=RISK_FREE_RATE, sigma=0.14):
    expiry_dt = get_next_tuesday_expiry(current_dt)
    T = time_to_expiry_years(current_dt, expiry_dt)
    if T <= 0:
        expiry_dt = get_next_tuesday_expiry(current_dt + timedelta(days=1))
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

def find_survivor_strike(S, current_dt, option_type, target_offset_pts=200.0, min_premium=15.0, r=RISK_FREE_RATE, sigma=0.14):
    expiry_dt = get_next_tuesday_expiry(current_dt)
    T = time_to_expiry_years(current_dt, expiry_dt)
    if T <= 0:
        expiry_dt = get_next_tuesday_expiry(current_dt + timedelta(days=1))
        T = time_to_expiry_years(current_dt, expiry_dt)

    temp_gap = target_offset_pts
    while temp_gap >= 0:
        if option_type == "PE":
            target_strike = int(round((S - temp_gap) / STRIKE_INTERVAL) * STRIKE_INTERVAL)
            premium = bs_put(S, target_strike, T, r, sigma)
            delta   = abs(bs_put_delta(S, target_strike, T, r, sigma))
        else:
            target_strike = int(round((S + temp_gap) / STRIKE_INTERVAL) * STRIKE_INTERVAL)
            premium = bs_call(S, target_strike, T, r, sigma)
            delta   = bs_call_delta(S, target_strike, T, r, sigma)

        if premium >= min_premium:
            return target_strike, premium, delta, expiry_dt
        temp_gap -= STRIKE_INTERVAL

    if option_type == "PE":
        target_strike = int(round((S - 50) / STRIKE_INTERVAL) * STRIKE_INTERVAL)
        premium = bs_put(S, target_strike, T, r, sigma)
        delta   = abs(bs_put_delta(S, target_strike, T, r, sigma))
    else:
        target_strike = int(round((S + 50) / STRIKE_INTERVAL) * STRIKE_INTERVAL)
        premium = bs_call(S, target_strike, T, r, sigma)
        delta   = bs_call_delta(S, target_strike, T, r, sigma)

    return target_strike, premium, delta, expiry_dt

def compute_option_price(S, K, expiry_dt, current_dt, option_type, r=RISK_FREE_RATE, sigma=0.14):
    T = time_to_expiry_years(current_dt, expiry_dt)
    if option_type == "CE":
        return bs_call(S, K, T, r, sigma)
    return bs_put(S, K, T, r, sigma)

def compute_margin_required(spot_price, num_lots):
    if num_lots <= 0:
        return 0.0
    return spot_price * LOT_SIZE * MARGIN_PCT * num_lots

# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def run_backtest():
    print("=" * 80)
    print("  COMBINED BACKTEST: WAVE MASTER + SURVIVOR MASTER")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    print("\nLoading 1-min Nifty data...")
    df = pd.read_csv(os.path.join(base_dir, "nifty.csv"))
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)

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

    # State
    wave_initial_pos   = 0
    wave_current_pos   = 0
    wave_pending_buy   = None
    wave_pending_sell  = None
    wave_cooloff       = 0
    wave_open_shorts   = []
    wave_trades        = []

    survivor_pe_last_val  = 0.0
    survivor_ce_last_val  = 0.0
    survivor_pe_reset_flag = 0
    survivor_ce_reset_flag = 0
    survivor_open_trades   = []
    survivor_trades        = []

    last_day = None
    daily_margin_records = []
    current_day_max_margin = 0.0

    print("\nSimulating Wave Master + Survivor Master...\n")

    for _idx, row in df.iterrows():
        current_dt    = row["datetime"]
        current_open  = row["open"]
        current_high  = row["high"]
        current_low   = row["low"]
        current_close = row["close"]
        current_iv    = row["iv"]
        market_time   = current_dt.time()
        weekday       = current_dt.weekday()   # Mon=0, Tue=1 (0DTE), Wed=2, Thu=3, Fri=4
        is_tuesday    = (weekday == 1)

        in_session = (time(9, 15) <= market_time < time(15, 30))
        if not in_session:
            continue

        today = current_dt.date()
        if today != last_day:
            if last_day is not None:
                daily_margin_records.append({
                    "date": last_day,
                    "max_margin": current_day_max_margin
                })
            last_day               = today
            current_day_max_margin = 0.0

            wave_initial_pos  = wave_current_pos
            wave_pending_buy  = None
            wave_pending_sell = None
            wave_cooloff      = 0

            if survivor_pe_last_val == 0.0:
                survivor_pe_last_val = current_open
            if survivor_ce_last_val == 0.0:
                survivor_ce_last_val = current_open

        # Daily Margin Calculation
        total_open_short_lots = len(wave_open_shorts) + sum(t["qty_lots"] for t in survivor_open_trades)
        current_bar_margin = compute_margin_required(current_close, total_open_short_lots)
        if current_bar_margin > current_day_max_margin:
            current_day_max_margin = current_bar_margin

        # =========================================================================
        # 1. SURVIVOR MASTER ROLLING PROFIT BOOKING
        # =========================================================================
        if len(survivor_open_trades) >= SURVIVOR_ROLLING_TRIGGER_TRADES:
            survivor_open_trades.sort(key=lambda x: x["entry_time"])
            oldest_leg = survivor_open_trades[0]

            curr_prem = compute_option_price(
                current_close, oldest_leg["opt_strike"], oldest_leg["opt_expiry_dt"], current_dt, oldest_leg["opt_type"], sigma=current_iv
            )
            entry_prem = oldest_leg["opt_entry_prem"]
            decay_ratio = (entry_prem - curr_prem) / entry_prem if entry_prem > 0 else 0.0

            if decay_ratio >= 0.50 or curr_prem <= 5.0:
                survivor_open_trades.pop(0)
                opt_exit_prem = curr_prem + SLIPPAGE_PTS
                opt_pnl = (entry_prem - opt_exit_prem) * oldest_leg["qty_lots"]
                hold_hours = (current_dt - oldest_leg["entry_time"]).total_seconds() / 3600
                survivor_trades.append({
                    "strategy":        "SURVIVOR",
                    "entry_time":      oldest_leg["entry_time"],
                    "exit_time":       current_dt,
                    "opt_type":        oldest_leg["opt_type"],
                    "opt_pnl":         opt_pnl,
                    "opt_pnl_per_lot": (opt_pnl * LOT_SIZE) - BROKERAGE_PER_TRADE,
                    "qty_lots":        oldest_leg["qty_lots"],
                    "hold_hours":      hold_hours,
                    "exit_reason":     "ROLLING_PROFIT_BOOKED",
                })

        # =========================================================================
        # 2. WAVE MASTER & SURVIVOR MASTER TICKS (NON-TUESDAY DAYS)
        # =========================================================================
        if not is_tuesday:
            # WAVE MASTER
            wave_fills = []
            if wave_pending_sell is not None and current_high >= wave_pending_sell:
                wave_fills.append(("SELL_SHORT", wave_pending_sell, current_iv, current_dt))
            if wave_pending_buy is not None and current_low <= wave_pending_buy:
                wave_fills.append(("BUY_COVER", wave_pending_buy, current_iv, current_dt))

            for fill_side, fill_price, fill_iv, fill_dt in wave_fills:
                if fill_side == "SELL_SHORT" and len(wave_open_shorts) < WAVE_MAX_OPEN_LOTS:
                    wave_current_pos -= 1
                    strike, entry_prem, entry_delta, expiry = find_strike_near_premium(
                        fill_price, fill_dt, "CE", target_premium=WAVE_TARGET_PREMIUM, sigma=fill_iv
                    )
                    entry_prem_after_slip = max(0.0, entry_prem - SLIPPAGE_PTS)
                    wave_open_shorts.append({
                        "entry_time":      fill_dt,
                        "entry_price":     fill_price,
                        "opt_type":        "CE",
                        "opt_strike":      strike,
                        "opt_entry_prem":  entry_prem_after_slip,
                        "opt_expiry_dt":   expiry,
                    })
                    wave_pending_sell = None

                elif fill_side == "BUY_COVER":
                    wave_current_pos += 1
                    if wave_open_shorts:
                        leg      = wave_open_shorts.pop(0)
                        spot_pnl = leg["entry_price"] - fill_price
                        opt_exit_prem  = compute_option_price(
                            fill_price, leg["opt_strike"], leg["opt_expiry_dt"], fill_dt, leg["opt_type"], sigma=fill_iv
                        ) + SLIPPAGE_PTS
                        opt_pnl     = leg["opt_entry_prem"] - opt_exit_prem
                        hold_hours  = (fill_dt - leg["entry_time"]).total_seconds() / 3600
                        wave_trades.append({
                            "strategy":        "WAVE",
                            "entry_time":      leg["entry_time"],
                            "exit_time":       fill_dt,
                            "spot_pnl":        spot_pnl,
                            "opt_pnl":         opt_pnl,
                            "opt_pnl_per_lot": (opt_pnl * LOT_SIZE) - BROKERAGE_PER_TRADE,
                            "hold_hours":      hold_hours,
                            "exit_reason":     "WAVE_COVER",
                        })
                    wave_pending_buy = None

            if wave_cooloff > 0:
                wave_cooloff -= 1
            else:
                diff_scale = wave_current_pos - wave_initial_pos
                key = max(-3, min(3, int(round(diff_scale))))
                mult = WAVE_MULTIPLIER_SCALE.get(key, [1.0, 1.0])
                s_buy_gap  = round(WAVE_BUY_GAP * mult[0], 1)
                s_sell_gap = round(WAVE_SELL_GAP * mult[1], 1)

                if len(wave_open_shorts) < WAVE_MAX_OPEN_LOTS:
                    wave_pending_sell = round(current_close + s_sell_gap, 2)
                wave_pending_buy = round(current_close - s_buy_gap, 2)
                wave_cooloff = WAVE_COOL_OFF_BARS

            # SURVIVOR MASTER
            current_survivor_lots = sum(t["qty_lots"] for t in survivor_open_trades)

            if current_close > survivor_pe_last_val:
                price_diff = round(current_close - survivor_pe_last_val, 0)
                if price_diff >= SURVIVOR_PE_GAP:
                    sell_mult = min(SURVIVOR_MAX_MULT, int(price_diff / SURVIVOR_PE_GAP))
                    if current_survivor_lots + sell_mult <= SURVIVOR_MAX_OPEN_LOTS:
                        strike, entry_prem, entry_delta, expiry = find_survivor_strike(
                            current_close, current_dt, "PE", target_offset_pts=SURVIVOR_PE_SYMBOL_GAP,
                            min_premium=SURVIVOR_MIN_PREMIUM, sigma=current_iv
                        )
                        entry_prem_after_slip = max(0.0, entry_prem - SLIPPAGE_PTS)
                        survivor_open_trades.append({
                            "type":            "SURVIVOR_PE",
                            "entry_time":      current_dt,
                            "entry_price":     current_close,
                            "opt_type":        "PE",
                            "opt_strike":      strike,
                            "opt_entry_prem":  entry_prem_after_slip,
                            "opt_expiry_dt":   expiry,
                            "qty_lots":        sell_mult,
                        })
                        survivor_pe_last_val += SURVIVOR_PE_GAP * sell_mult
                        survivor_pe_reset_flag = 1

            if current_close < survivor_ce_last_val:
                price_diff = round(survivor_ce_last_val - current_close, 0)
                if price_diff >= SURVIVOR_CE_GAP:
                    sell_mult = min(SURVIVOR_MAX_MULT, int(price_diff / SURVIVOR_CE_GAP))
                    if current_survivor_lots + sell_mult <= SURVIVOR_MAX_OPEN_LOTS:
                        strike, entry_prem, entry_delta, expiry = find_survivor_strike(
                            current_close, current_dt, "CE", target_offset_pts=SURVIVOR_CE_SYMBOL_GAP,
                            min_premium=SURVIVOR_MIN_PREMIUM, sigma=current_iv
                        )
                        entry_prem_after_slip = max(0.0, entry_prem - SLIPPAGE_PTS)
                        survivor_open_trades.append({
                            "type":            "SURVIVOR_CE",
                            "entry_time":      current_dt,
                            "entry_price":     current_close,
                            "opt_type":        "CE",
                            "opt_strike":      strike,
                            "opt_entry_prem":  entry_prem_after_slip,
                            "opt_expiry_dt":   expiry,
                            "qty_lots":        sell_mult,
                        })
                        survivor_ce_last_val -= SURVIVOR_CE_GAP * sell_mult
                        survivor_ce_reset_flag = 1

            if (survivor_pe_last_val - current_close) > SURVIVOR_PE_RESET_GAP and survivor_pe_reset_flag:
                survivor_pe_last_val = current_close + SURVIVOR_PE_RESET_GAP
            if (current_close - survivor_ce_last_val) > SURVIVOR_CE_RESET_GAP and survivor_ce_reset_flag:
                survivor_ce_last_val = current_close - SURVIVOR_CE_RESET_GAP

        # =========================================================================
        # 3. MONDAY EOD SQUARE OFF
        # =========================================================================
        if weekday == 0 and market_time >= time(15, 25):
            if wave_open_shorts:
                for leg in wave_open_shorts:
                    spot_pnl = leg["entry_price"] - current_close
                    opt_exit_prem  = compute_option_price(
                        current_close, leg["opt_strike"], leg["opt_expiry_dt"], current_dt, leg["opt_type"], sigma=current_iv
                    ) + SLIPPAGE_PTS
                    opt_pnl    = leg["opt_entry_prem"] - opt_exit_prem
                    hold_hours = (current_dt - leg["entry_time"]).total_seconds() / 3600
                    wave_trades.append({
                        "strategy":        "WAVE",
                        "entry_time":      leg["entry_time"],
                        "exit_time":       current_dt,
                        "spot_pnl":        spot_pnl,
                        "opt_pnl":         opt_pnl,
                        "opt_pnl_per_lot": (opt_pnl * LOT_SIZE) - BROKERAGE_PER_TRADE,
                        "hold_hours":      hold_hours,
                        "exit_reason":     "MON_EOD_CLOSE",
                    })
                wave_open_shorts.clear()
                wave_current_pos   = 0
                wave_pending_buy   = None
                wave_pending_sell  = None

            if survivor_open_trades:
                for leg in survivor_open_trades:
                    opt_exit_prem  = compute_option_price(
                        current_close, leg["opt_strike"], leg["opt_expiry_dt"], current_dt, leg["opt_type"], sigma=current_iv
                    ) + SLIPPAGE_PTS
                    opt_pnl    = (leg["opt_entry_prem"] - opt_exit_prem) * leg["qty_lots"]
                    hold_hours = (current_dt - leg["entry_time"]).total_seconds() / 3600
                    survivor_trades.append({
                        "strategy":        "SURVIVOR",
                        "entry_time":      leg["entry_time"],
                        "exit_time":       current_dt,
                        "opt_type":        leg["opt_type"],
                        "opt_pnl":         opt_pnl,
                        "opt_pnl_per_lot": (opt_pnl * LOT_SIZE) - BROKERAGE_PER_TRADE,
                        "qty_lots":        leg["qty_lots"],
                        "hold_hours":      hold_hours,
                        "exit_reason":     "MON_EOD_CLOSE",
                    })
                survivor_open_trades.clear()

    if last_day is not None:
        daily_margin_records.append({
            "date": last_day,
            "max_margin": current_day_max_margin
        })

    wave_df     = pd.DataFrame(wave_trades)
    survivor_df = pd.DataFrame(survivor_trades)
    margin_df   = pd.DataFrame(daily_margin_records)

    wave_tot_pnl     = wave_df['opt_pnl_per_lot'].sum()     if not wave_df.empty else 0.0
    survivor_tot_pnl = survivor_df['opt_pnl_per_lot'].sum() if not survivor_df.empty else 0.0
    combined_tot_pnl = wave_tot_pnl + survivor_tot_pnl

    avg_margin = margin_df['max_margin'].mean() if not margin_df.empty else 0.0
    max_margin = margin_df['max_margin'].max()  if not margin_df.empty else 0.0
    single_lot_margin = df['close'].mean() * LOT_SIZE * MARGIN_PCT

    print("=" * 80)
    print("  COMBINED BACKTEST RESULTS: WAVE MASTER + SURVIVOR MASTER")
    print("=" * 80)
    print(f"  1. Wave Master Total Trades      : {len(wave_df)}")
    print(f"  1. Wave Master Total PnL         : Rs. {wave_tot_pnl:+,.0f}")
    print("-" * 80)
    print(f"  2. Survivor Master Total Trades  : {len(survivor_df)}")
    print(f"  2. Survivor Master Total PnL     : Rs. {survivor_tot_pnl:+,.0f}")
    print("=" * 80)
    print(f"  >>> COMBINED NET REALIZED PnL (1-Lot Base): Rs. {combined_tot_pnl:+,.0f} <<<")
    print("=" * 80)
    print("  MARGIN & CAPITAL ANALYSIS:")
    print(f"  * Single Lot Base Margin Est     : ~Rs. {single_lot_margin:,.0f}")
    print(f"  * Avg Daily Margin Req           : ~Rs. {avg_margin:,.0f}")
    print(f"  * Peak Max Margin Req            : ~Rs. {max_margin:,.0f}")
    print("=" * 80)

    generate_html_report(wave_df, survivor_df, combined_tot_pnl, avg_margin, max_margin, single_lot_margin, df)

# ============================================================
# HEATMAP GENERATOR
# ============================================================
def generate_html_report(wave_df, survivor_df, combined_tot_pnl, avg_margin, max_margin, single_lot_margin, raw_df):
    if not wave_df.empty: wave_df['exit_day'] = wave_df['exit_time'].dt.normalize()
    if not survivor_df.empty: survivor_df['exit_day'] = survivor_df['exit_time'].dt.normalize()

    wave_daily     = wave_df.groupby('exit_day')['opt_pnl_per_lot'].sum().reset_index() if not wave_df.empty else pd.DataFrame(columns=['exit_day', 'opt_pnl_per_lot'])
    survivor_daily = survivor_df.groupby('exit_day')['opt_pnl_per_lot'].sum().reset_index() if not survivor_df.empty else pd.DataFrame(columns=['exit_day', 'opt_pnl_per_lot'])

    min_dates = [df['entry_time'].min() for df in [wave_df, survivor_df] if not df.empty]
    max_dates = [df['exit_time'].max() for df in [wave_df, survivor_df] if not df.empty]

    start_date = min(min_dates).normalize()
    end_date   = max(max_dates).normalize()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    daily_df = pd.DataFrame({'date': date_range})
    daily_df = daily_df.merge(wave_daily, left_on='date', right_on='exit_day', how='left').rename(columns={'opt_pnl_per_lot': 'wave_pnl'}).fillna(0)
    daily_df = daily_df.merge(survivor_daily, left_on='date', right_on='exit_day', how='left').rename(columns={'opt_pnl_per_lot': 'survivor_pnl'}).fillna(0)

    daily_df['combined_pnl'] = daily_df['wave_pnl'] + daily_df['survivor_pnl']

    daily_pnl_map = {r['date'].strftime("%Y-%m-%d"): r['combined_pnl'] for _, r in daily_df.iterrows()}
    wave_pnl_map  = {r['date'].strftime("%Y-%m-%d"): r['wave_pnl'] for _, r in daily_df.iterrows()}
    surv_pnl_map  = {r['date'].strftime("%Y-%m-%d"): r['survivor_pnl'] for _, r in daily_df.iterrows()}

    months_list = []
    current_m = start_date.date().replace(day=1)
    end_date_val = end_date.date()

    while current_m <= end_date_val:
        year = current_m.year
        month = current_m.month
        month_name = calendar.month_name[month]
        num_days = calendar.monthrange(year, month)[1]

        days = []
        first_weekday = date(year, month, 1).weekday()
        for _ in range(first_weekday):
            days.append({"day": 0, "pnl": None, "type": "pad"})

        for d in range(1, num_days + 1):
            date_str = f"{year}-{month:02d}-{d:02d}"
            dt_obj = date(year, month, d)
            is_weekend = dt_obj.weekday() >= 5
            is_tuesday = dt_obj.weekday() == 1

            c_pnl = daily_pnl_map.get(date_str, 0.0)
            w_pnl = wave_pnl_map.get(date_str, 0.0)
            s_pnl = surv_pnl_map.get(date_str, 0.0)

            if is_weekend:
                days.append({"day": d, "date": date_str, "pnl": 0.0, "type": "weekend"})
            elif is_tuesday:
                days.append({"day": d, "date": date_str, "pnl": 0.0, "type": "tuesday_0dte"})
            else:
                days.append({
                    "day": d, "date": date_str,
                    "pnl": c_pnl, "wave_pnl": w_pnl, "survivor_pnl": s_pnl,
                    "type": "trade"
                })

        months_list.append({"name": month_name, "year": year, "days": days})
        if month == 12:
            current_m = date(year + 1, 1, 1)
        else:
            current_m = date(year, month + 1, 1)

    wave_tot     = wave_df['opt_pnl_per_lot'].sum() if not wave_df.empty else 0.0
    survivor_tot = survivor_df['opt_pnl_per_lot'].sum() if not survivor_df.empty else 0.0
    months_json  = json.dumps(months_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wave Master + Survivor Master — Daily PnL Heatmap</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #080d16;
            --card-bg: rgba(14, 20, 35, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --cyan: #06b6d4;
            --purple: #8b5cf6;
            --amber: #f59e0b;
            --red: #ef4444;
            --green: #10b981;
            --txt: #f1f5f9;
            --muted: #94a3b8;
            --dim: #64748b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--bg); color: var(--txt); font-family: 'Inter', sans-serif; padding: 40px 5%; min-height: 100vh;
        }}
        header {{
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--card-border);
        }}
        header h1 {{
            font-family: 'Outfit', sans-serif; font-size: 24px; font-weight: 800;
            background: linear-gradient(to right, #06b6d4, #8b5cf6, #10b981);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        header p {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
        .badge {{
            display: inline-block; padding: 6px 12px; font-size: 12px; font-weight: 600; border-radius: 6px; background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .grid-stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px; margin-bottom: 35px;
        }}
        .card-stat {{
            background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 20px; backdrop-filter: blur(16px);
        }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }}
        .stat-value {{ font-family: 'Outfit', sans-serif; font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
        .stat-sub {{ font-size: 11px; color: var(--dim); }}
        .text-green {{ color: #10b981 !important; }}
        .text-cyan {{ color: #06b6d4 !important; }}
        .text-purple {{ color: #8b5cf6 !important; }}

        .heatmap-box {{
            background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 20px; padding: 30px; backdrop-filter: blur(16px); margin-bottom: 35px;
        }}
        .heatmap-title {{ font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 700; margin-bottom: 8px; color: #f3f4f6; }}
        .heatmap-subtitle {{ font-size: 13px; color: var(--muted); margin-bottom: 25px; }}

        .months-flex {{ display: flex; flex-wrap: wrap; gap: 24px; }}
        .month-block {{ display: flex; flex-direction: column; align-items: flex-start; }}
        .month-grid {{ display: grid; grid-template-columns: repeat(7, 16px); gap: 5px; margin-bottom: 8px; }}

        .day-tile {{ width: 16px; height: 16px; border-radius: 3px; transition: transform 0.15s ease; }}
        .day-tile:hover {{ transform: scale(1.4); z-index: 20; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}

        .tile-pad {{ background: transparent; pointer-events: none; }}
        .tile-weekend {{ background: rgba(255, 255, 255, 0.04); }}
        .tile-0dte {{ background: rgba(148, 163, 184, 0.15); border: 1px dashed rgba(148, 163, 184, 0.3); }}
        .tile-zero {{ background: rgba(255, 255, 255, 0.07); }}

        .tile-g1 {{ background: #86efac; }}
        .tile-g2 {{ background: #4ade80; }}
        .tile-g3 {{ background: #22c55e; }}
        .tile-g4 {{ background: #15803d; }}

        .tile-r1 {{ background: #fca5a5; }}
        .tile-r2 {{ background: #f87171; }}
        .tile-r3 {{ background: #ef4444; }}
        .tile-r4 {{ background: #b91c1c; }}

        .month-name {{ font-size: 13px; font-weight: 600; color: var(--muted); }}

        #tooltip {{
            position: absolute; display: none; background: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; font-size: 12px; pointer-events: none; z-index: 1000; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>WAVE MASTER + SURVIVOR MASTER BACKTEST REPORT</h1>
            <p>Strict Repository Strategy Execution · Rolling Profit Booking · Mon EOD Exit</p>
        </div>
        <div class="badge">Pure Code Backtest</div>
    </header>

    <div class="grid-stats">
        <div class="card-stat">
            <div class="stat-label">Wave Strategy PnL</div>
            <div class="stat-value text-cyan">Rs. {wave_tot:+,.0f}</div>
            <div class="stat-sub">Grid Scraper Range System</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Survivor Strategy PnL</div>
            <div class="stat-value text-purple">Rs. {survivor_tot:+,.0f}</div>
            <div class="stat-sub">Trend Option Selling System</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Combined Net PnL</div>
            <div class="stat-value text-green">Rs. {combined_tot_pnl:+,.0f}</div>
            <div class="stat-sub">7-Month Realized PnL</div>
        </div>
        <div class="card-stat">
            <div class="stat-label">Avg Daily Margin</div>
            <div class="stat-value">Rs. {avg_margin:,.0f}</div>
            <div class="stat-sub">Peak Max: Rs. {max_margin:,.0f}</div>
        </div>
    </div>

    <div class="heatmap-box">
        <div class="heatmap-title">Combined Portfolio Performance Heatmap</div>
        <div class="heatmap-subtitle">Daily breakdown of Wave Master and Survivor Master strategy performance</div>

        <div class="months-flex" id="heatmapGrid"></div>
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
                }} else if (d.type === 'tuesday_0dte') {{
                    tile.classList.add('tile-0dte');
                    tile.addEventListener('mouseenter', () => {{
                        tooltip.innerHTML = `<strong>${{d.date}} (Tuesday)</strong><br><span style="color:#94a3b8">0-DTE Expiry Day — Skipped</span>`;
                        tooltip.style.display = 'block';
                    }});
                    tile.addEventListener('mousemove', (e) => {{
                        tooltip.style.left = (e.pageX + 12) + 'px';
                        tooltip.style.top = (e.pageY + 12) + 'px';
                    }});
                    tile.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
                }} else {{
                    const pnl = d.pnl;
                    if (pnl === 0) {{
                        tile.classList.add('tile-zero');
                    }} else if (pnl > 0) {{
                        if (pnl > 80000) tile.classList.add('tile-g4');
                        else if (pnl > 40000) tile.classList.add('tile-g3');
                        else if (pnl > 15000) tile.classList.add('tile-g2');
                        else tile.classList.add('tile-g1');
                    }} else {{
                        const loss = Math.abs(pnl);
                        if (loss > 80000) tile.classList.add('tile-r4');
                        else if (loss > 40000) tile.classList.add('tile-r3');
                        else if (loss > 15000) tile.classList.add('tile-r2');
                        else tile.classList.add('tile-r1');
                    }}

                    tile.addEventListener('mouseenter', (e) => {{
                        const fmtC = pnl >= 0 ? '+' + pnl.toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}}) : '-' + Math.abs(pnl).toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}});
                        const fmtW = (d.wave_pnl || 0) >= 0 ? '+' + (d.wave_pnl || 0).toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}}) : '-' + Math.abs(d.wave_pnl || 0).toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}});
                        const fmtS = (d.survivor_pnl || 0) >= 0 ? '+' + (d.survivor_pnl || 0).toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}}) : '-' + Math.abs(d.survivor_pnl || 0).toLocaleString('en-IN', {{style:'currency', currency:'INR', maximumFractionDigits:0}});

                        tooltip.innerHTML = `<strong>${{d.date}}</strong><br>` +
                            `Combined: <span style="color:${{pnl >= 0 ? '#4ade80' : '#f87171'}}">${{fmtC}}</span><br>` +
                            `<span style="color:#06b6d4">Wave: ${{fmtW}}</span> | <span style="color:#8b5cf6">Survivor: ${{fmtS}}</span>`;
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

    base_dir  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    html_path = os.path.join(base_dir, "wave_survivor_combined_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Clean Backtest Report saved -> {html_path}")
    print("  Open in browser to view the heatmap.\n")


if __name__ == "__main__":
    run_backtest()
