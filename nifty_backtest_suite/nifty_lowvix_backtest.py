import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta

# ============================================================
# 1. PARAMETERS & CONFIG
# ============================================================
INITIAL_CAPITAL    = 500000.0   # ₹5 Lakhs (3x deployment ~4.5L + 50k safety buffer)
LOT_SIZE           = 65         # Nifty lot size from strategy
LOTS               = 6          # 6 lots per leg (3x original 2 lots)
QTY                = LOTS * LOT_SIZE # 390 contracts
STRIKE_INTERVAL    = 50
RISK_FREE_RATE     = 0.07       # 7%
SLIPPAGE_PER_LEG   = 0.10       # ₹0.10 per contract per leg
VIX_THRESHOLD      = 13.0
TARGET_MTM_PROFIT  = 2000.0     # ₹1,200 - Monday adjustment trigger
TUESDAY_TRAIL_LOCK = 1500.0     # ₹2,000 - Max allowed drawback from Tuesday peak before exit
PUT_IV_MULT        = 0.98       # To reflect Put weekly option pricing in low-VIX
CALL_IV_MULT       = 0.94       # To reflect Call weekly option pricing in low-VIX

# Indian Transaction Charges Rates
BROKERAGE_PER_ORDER = 20.0      # ₹20 flat per leg order
EXCHANGE_TAX_RATE   = 0.00053   # 0.053% NSE exchange transaction charge
STT_RATE            = 0.000625  # 0.0625% on sell premium
STAMP_DUTY_RATE     = 0.00003   # 0.003% on buy premium
SEBI_TURNOVER_RATE  = 0.000001  # 0.0001% (₹10 per crore)
GST_RATE            = 0.18      # 18% of (Brokerage + Exchange charges)

# ============================================================
# 2. BLACK-SCHOLES OPTION PRICING
# ============================================================
def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def bs_call(S, K, T, r, sigma):
    """Black-Scholes Call Option Price."""
    if T <= 0:
        return max(0.0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Black-Scholes Put Option Price."""
    if T <= 0:
        return max(0.0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def time_to_expiry_years(current_dt, expiry_dt):
    diff_days = (expiry_dt - current_dt).total_seconds() / (24 * 3600)
    if diff_days <= 0:
        return 0.0
    # Weekend adjustment: if starting on Friday (4) and ending Tuesday (1), subtract 1.6 days
    if current_dt.weekday() == 4 and expiry_dt.weekday() == 1:
        diff_days = max(diff_days - 1.6, 0.05)
    return diff_days / 252.0

# ============================================================
# 3. CHARGES CALCULATION
# ============================================================
def calculate_trade_charges(buys_premium, sells_premium, num_orders):
    """
    Calculate complete transaction costs (brokerage, exchange charges, GST, STT, stamp duty, slippage)
    """
    # 1. Brokerage
    brokerage = num_orders * BROKERAGE_PER_ORDER
    
    # 2. Exchange Transaction Charges
    exchange_charges = (buys_premium + sells_premium) * QTY * EXCHANGE_TAX_RATE
    
    # 3. GST (18% on Brokerage + Exchange charges)
    gst = (brokerage + exchange_charges) * GST_RATE
    
    # 4. STT (0.0625% on Sell side)
    stt = sells_premium * QTY * STT_RATE
    
    # 5. Stamp Duty (0.003% on Buy side)
    stamp_duty = buys_premium * QTY * STAMP_DUTY_RATE
    
    # 6. SEBI Turnover Fee
    sebi_fee = (buys_premium + sells_premium) * QTY * SEBI_TURNOVER_RATE
    
    # 7. Slippage
    slippage = num_orders * SLIPPAGE_PER_LEG * QTY
    
    total_charges = brokerage + exchange_charges + gst + stt + stamp_duty + sebi_fee + slippage
    return {
        "brokerage": brokerage,
        "exchange_charges": exchange_charges,
        "gst": gst,
        "stt": stt,
        "stamp_duty": stamp_duty,
        "sebi_fee": sebi_fee,
        "slippage": slippage,
        "total": total_charges
    }

# ============================================================
# 4. MAIN SIMULATION ENGINE
# ============================================================
def run_backtest():
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load spot data
    df = pd.read_csv(os.path.join(base_dir, "NIFTY_NSE_INDEX_5m.csv"))
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Load 5m VIX data
    vix_df = pd.read_csv(os.path.join(base_dir, "vix_5m.csv"))
    vix_df['datetime'] = pd.to_datetime(vix_df['date'] + ' ' + vix_df['time'])
    vix_df = vix_df.sort_values('datetime').reset_index(drop=True)
    vix_clean = vix_df[['datetime', 'close']].copy()
    vix_clean.rename(columns={'close': 'vix'}, inplace=True)
    vix_clean['iv'] = vix_clean['vix'] / 100.0
    
    # Merge datasets on exact datetime
    df = df.merge(vix_clean, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Filter starting from Jan 1, 2022
    df = df[df['datetime'] >= '2022-01-01 00:00:00'].reset_index(drop=True)
    
    print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Nifty Spot range: {df['close'].min():.1f} to {df['close'].max():.1f}")
    print(f"VIX range: {df['vix'].min():.2f} to {df['vix'].max():.2f}")
    
    # Find Friday entries
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['time_str'] = df['datetime'].dt.time
    df['trade_date'] = df['datetime'].dt.normalize()
    
    # Find Friday entries — scan 14:00 to 15:25, take FIRST bar where VIX < threshold
    # This handles cases where VIX only drops below 13 at 2:30, 3:00, or 3:15 PM
    fridays_window = df[
        (df['dayofweek'] == 4) &
        (df['time_str'] >= time(14, 0, 0)) &
        (df['time_str'] <= time(15, 25, 0))
    ].copy()

    # For each Friday, find the first bar where VIX is below the threshold
    low_vix_bars = fridays_window[fridays_window['vix'] < VIX_THRESHOLD].copy()
    friday_entries = low_vix_bars.sort_values('datetime').drop_duplicates(subset=['trade_date'], keep='first').reset_index(drop=True)

    print(f"Total Fridays in dataset: {fridays_window['trade_date'].nunique()}")
    print(f"Fridays with VIX < {VIX_THRESHOLD} at some point 14:00-15:25: {len(friday_entries)}")
    
    trades = []
    
    for _, entry_row in friday_entries.iterrows():
        entry_time = entry_row['datetime']
        entry_vix = entry_row['vix']
        
        entry_spot = entry_row['close']
        entry_iv = entry_row['iv']
        
        # Expiry is Tuesday 15:30 (4 days later)
        expiry_date = entry_time.date() + timedelta(days=4)
        expiry_dt = datetime.combine(expiry_date, time(15, 30, 0))
        
        # Options strikes
        atm = round(entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        
        short_put = atm - 6 * STRIKE_INTERVAL   # OTM 6
        long_put = atm - 9 * STRIKE_INTERVAL    # OTM 9
        short_call = atm + 6 * STRIKE_INTERVAL  # OTM 6
        long_call = atm + 9 * STRIKE_INTERVAL   # OTM 9
        
        # Hedges OTM 5 (for adjustment)
        long_put5 = atm - 5 * STRIKE_INTERVAL   # OTM 5
        long_call5 = atm + 5 * STRIKE_INTERVAL  # OTM 5
        
        # Time to expiry at entry
        T_entry = time_to_expiry_years(entry_time, expiry_dt)
        
        # Calculate option prices at entry
        sp_entry = bs_put(entry_spot, short_put, T_entry, RISK_FREE_RATE, entry_iv * PUT_IV_MULT)
        lp_entry = bs_put(entry_spot, long_put, T_entry, RISK_FREE_RATE, entry_iv * PUT_IV_MULT)
        sc_entry = bs_call(entry_spot, short_call, T_entry, RISK_FREE_RATE, entry_iv * CALL_IV_MULT)
        lc_entry = bs_call(entry_spot, long_call, T_entry, RISK_FREE_RATE, entry_iv * CALL_IV_MULT)
        
        # Credits collected (sells - buys)
        initial_net_credit = (sp_entry + sc_entry) - (lp_entry + lc_entry)
        
        # Get subsequent bars for this trade cycle
        trade_bars = df[(df['datetime'] > entry_time) & (df['datetime'] <= expiry_dt)].copy()
        if trade_bars.empty:
            continue
            
        # Variables to track trade state
        adjusted = False
        adj_time = None
        adj_spot = 0.0
        adj_cost_premium = 0.0
        adj_mtm = 0.0
        
        exit_time = None
        exit_spot = 0.0
        exit_reason = ""
        exit_pnl_gross = 0.0
        
        # Tuesday trailing stop tracker
        tuesday_peak_mtm = None  # None until we first see a positive MTM on Tuesday
        
        # Option premium components for transaction charges tracking
        buys_premium_total = lp_entry + lc_entry
        sells_premium_total = sp_entry + sc_entry
        num_orders = 4 # Entry has 4 legs
        
        # Monday opening gap analysis
        monday_bars = trade_bars[trade_bars['datetime'].dt.dayofweek == 0]
        monday_open_spot = np.nan
        monday_gap_pct = 0.0
        if not monday_bars.empty:
            monday_open_row = monday_bars.iloc[0]
            monday_open_spot = monday_open_row['open']
            friday_close_spot = entry_row['close'] # Or Friday EOD close. Let's use Friday entry close as baseline
            monday_gap_pct = (monday_open_spot - friday_close_spot) / friday_close_spot * 100
        
        # Iterate bar by bar
        for idx, bar in trade_bars.iterrows():
            bar_time = bar['datetime']
            bar_spot = bar['close']
            bar_high = bar['high']
            bar_low = bar['low']
            bar_open = bar['open']
            bar_iv = bar['iv']
            bar_vix = bar['vix']
            
            is_monday_onwards = bar_time.weekday() in [0, 1] # Monday=0, Tuesday=1
            
            T_current = time_to_expiry_years(bar_time, expiry_dt)
            
            # Monday onwards checks
            if is_monday_onwards:
                # 1. Breach check
                # Check if first bar of Monday opens in a breach
                is_first_monday_bar = (bar_time.weekday() == 0) and (bar_time.time() == time(9, 15, 0))
                
                if is_first_monday_bar:
                    breach_c = bar_open >= short_call
                    breach_p = bar_open <= short_put
                    check_spot = bar_open
                else:
                    breach_c = bar_high >= short_call
                    breach_p = bar_low <= short_put
                    check_spot = bar_spot
                
                if (breach_c or breach_p) and not adjusted:
                    # Breach exit!
                    exit_time = bar_time
                    exit_spot = check_spot
                    exit_reason = "BREACH_EXIT"
                    
                    # Compute option prices at exit
                    sp_exit = bs_put(exit_spot, short_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    lp_exit = bs_put(exit_spot, long_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    sc_exit = bs_call(exit_spot, short_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    lc_exit = bs_call(exit_spot, long_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    
                    if adjusted:
                        # Price OTM 5 hedges instead
                        lp5_exit = bs_put(exit_spot, long_put5, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                        lc5_exit = bs_call(exit_spot, long_call5, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                        
                        exit_debit = (sp_exit + sc_exit) - (lp5_exit + lc5_exit)
                        exit_pnl_gross = (initial_net_credit - exit_debit - adj_cost_premium) * QTY
                        
                        buys_premium_total += sp_exit + sc_exit
                        sells_premium_total += lp5_exit + lc5_exit
                    else:
                        exit_debit = (sp_exit + sc_exit) - (lp_exit + lc_exit)
                        exit_pnl_gross = (initial_net_credit - exit_debit) * QTY
                        
                        buys_premium_total += sp_exit + sc_exit
                        sells_premium_total += lp_exit + lc_exit
                        
                    num_orders += 4
                    break
                
                # 2. Adjustment check (if not adjusted yet and not breached)
                if not adjusted:
                    # Calculate current MTM
                    sp_curr = bs_put(bar_spot, short_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    lp_curr = bs_put(bar_spot, long_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    sc_curr = bs_call(bar_spot, short_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    lc_curr = bs_call(bar_spot, long_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    
                    current_net_cost = (sp_curr + sc_curr) - (lp_curr + lc_curr)
                    mtm = (initial_net_credit - current_net_cost) * QTY
                    
                    if mtm >= TARGET_MTM_PROFIT:
                        # Trigger Adjustment!
                        adjusted = True
                        adj_time = bar_time
                        adj_spot = bar_spot
                        adj_mtm = mtm
                        
                        # Cost of new OTM 5 hedges
                        lp5_curr = bs_put(bar_spot, long_put5, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                        lc5_curr = bs_call(bar_spot, long_call5, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                        
                        # Buy new hedges, sell old hedges
                        # Old hedges squared off (sells): lp_curr + lc_curr
                        # New hedges bought (buys): lp5_curr + lc5_curr
                        adj_cost_premium = (lp5_curr + lc5_curr) - (lp_curr + lc_curr)
                        
                        # Add to premium calculations
                        buys_premium_total += lp5_curr + lc5_curr
                        sells_premium_total += lp_curr + lc_curr
                        num_orders += 4 # 4 transactions for adjustment (sell 2 old, buy 2 new)
            
            # 3. Tuesday trailing profit-lock stop
            # Once we have been in profit on Tuesday, don't give back more than TUESDAY_TRAIL_LOCK
            is_tuesday_bar = (bar_time.weekday() == 1)
            if is_tuesday_bar:
                # Compute current live MTM
                sp_curr_t = bs_put(bar_spot, short_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                lp_curr_t = bs_put(bar_spot, long_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                sc_curr_t = bs_call(bar_spot, short_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                lc_curr_t = bs_call(bar_spot, long_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                if adjusted:
                    lp5_curr_t = bs_put(bar_spot, long_put5, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    lc5_curr_t = bs_call(bar_spot, long_call5, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    current_net_cost_t = (sp_curr_t + sc_curr_t) - (lp5_curr_t + lc5_curr_t)
                else:
                    current_net_cost_t = (sp_curr_t + sc_curr_t) - (lp_curr_t + lc_curr_t)
                if adjusted:
                    live_mtm = (initial_net_credit - current_net_cost_t - adj_cost_premium) * QTY
                else:
                    live_mtm = (initial_net_credit - current_net_cost_t) * QTY

                # Update Tuesday peak once we first turn profitable
                if live_mtm > 0:
                    if tuesday_peak_mtm is None or live_mtm > tuesday_peak_mtm:
                        tuesday_peak_mtm = live_mtm

                # Trigger trailing stop if peak was set and MTM dropped by TUESDAY_TRAIL_LOCK
                if tuesday_peak_mtm is not None and live_mtm < (tuesday_peak_mtm - TUESDAY_TRAIL_LOCK):
                    exit_time = bar_time
                    exit_spot = bar_spot
                    exit_reason = "TUESDAY_TRAIL_STOP"
                    if adjusted:
                        lp5_curr_t2 = bs_put(bar_spot, long_put5, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                        lc5_curr_t2 = bs_call(bar_spot, long_call5, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                        exit_debit = (sp_curr_t + sc_curr_t) - (lp5_curr_t2 + lc5_curr_t2)
                        exit_pnl_gross = (initial_net_credit - exit_debit - adj_cost_premium) * QTY
                        buys_premium_total += sp_curr_t + sc_curr_t
                        sells_premium_total += lp5_curr_t2 + lc5_curr_t2
                    else:
                        exit_debit = (sp_curr_t + sc_curr_t) - (lp_curr_t + lc_curr_t)
                        exit_pnl_gross = (initial_net_credit - exit_debit) * QTY
                        buys_premium_total += sp_curr_t + sc_curr_t
                        sells_premium_total += lp_curr_t + lc_curr_t
                    num_orders += 4
                    break

            # 4. Hard exit at Tuesday 15:10
            is_expiry = (bar_time.weekday() == 1) and (bar_time.time() >= time(15, 10, 0))
            is_last_bar = (idx == len(trade_bars) - 1)
            
            if is_expiry or is_last_bar:
                exit_time = bar_time
                exit_spot = bar_spot
                exit_reason = "EXPIRY_EXIT" if is_expiry else "END_OF_DATA"
                
                sp_exit = bs_put(exit_spot, short_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                lp_exit = bs_put(exit_spot, long_put, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                sc_exit = bs_call(exit_spot, short_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                lc_exit = bs_call(exit_spot, long_call, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                
                if adjusted:
                    lp5_exit = bs_put(exit_spot, long_put5, T_current, RISK_FREE_RATE, bar_iv * PUT_IV_MULT)
                    lc5_exit = bs_call(exit_spot, long_call5, T_current, RISK_FREE_RATE, bar_iv * CALL_IV_MULT)
                    
                    exit_debit = (sp_exit + sc_exit) - (lp5_exit + lc5_exit)
                    exit_pnl_gross = (initial_net_credit - exit_debit - adj_cost_premium) * QTY
                    
                    buys_premium_total += sp_exit + sc_exit
                    sells_premium_total += lp5_exit + lc5_exit
                else:
                    exit_debit = (sp_exit + sc_exit) - (lp_exit + lc_exit)
                    exit_pnl_gross = (initial_net_credit - exit_debit) * QTY
                    
                    buys_premium_total += sp_exit + sc_exit
                    sells_premium_total += lp_exit + lc_exit
                    
                num_orders += 4
                break
                
        # Skip trades where no exit was found (e.g. holiday weeks with no Tuesday data)
        if not exit_reason:
            continue

        # Calculate charges
        charges = calculate_trade_charges(buys_premium_total, sells_premium_total, num_orders)
        exit_pnl_net = exit_pnl_gross - charges["total"]
        
        trades.append({
            "entry_time": entry_time,
            "entry_vix": entry_vix,
            "entry_spot": entry_spot,
            "expiry_dt": expiry_dt,
            "atm": atm,
            "short_call": short_call,
            "short_put": short_put,
            "long_call": long_call,
            "long_put": long_put,
            "monday_open_spot": monday_open_spot,
            "monday_gap_pct": monday_gap_pct,
            "adjusted": adjusted,
            "adj_time": adj_time,
            "adj_spot": adj_spot,
            "adj_cost_premium": adj_cost_premium * QTY,
            "adj_mtm": adj_mtm,
            "exit_time": exit_time,
            "exit_spot": exit_spot,
            "exit_reason": exit_reason,
            "gross_pnl": exit_pnl_gross,
            "charges": charges["total"],
            "net_pnl": exit_pnl_net,
            "initial_credit": initial_net_credit * QTY
        })
        
    return pd.DataFrame(trades)

# ============================================================
# 5. METRICS & ANALYSIS
# ============================================================
def analyze_results(trades_df):
    if trades_df.empty:
        print("No trades triggered.")
        return
        
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    
    # Calculate cumulative capital
    trades_df["cumulative_net_pnl"] = trades_df["net_pnl"].cumsum()
    trades_df["capital"] = INITIAL_CAPITAL + trades_df["cumulative_net_pnl"]
    
    # Drawdown
    capital_series = pd.Series([INITIAL_CAPITAL] + trades_df["capital"].tolist())
    running_max = capital_series.cummax()
    drawdowns = (running_max - capital_series) / running_max * 100
    max_drawdown = drawdowns.max()
    max_drawdown_amount = (running_max - capital_series).max()
    
    # Win rate
    total_trades = len(trades_df)
    winning_trades = (trades_df["net_pnl"] > 0).sum()
    win_rate = winning_trades / total_trades * 100
    
    # PnL metrics
    net_profit = trades_df["net_pnl"].sum()
    gross_profit = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss = abs(trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    average_trade = trades_df["net_pnl"].mean()
    expectancy = average_trade / trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].std() if trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].std() > 0 else 0.0
    
    # Monthly returns
    trades_df["year_month"] = trades_df["entry_time"].dt.to_period("M")
    monthly_pnl = trades_df.groupby("year_month")["net_pnl"].sum().reset_index()
    monthly_pnl["year"] = monthly_pnl["year_month"].dt.year
    monthly_pnl["month"] = monthly_pnl["year_month"].dt.month
    
    pivot_table = monthly_pnl.pivot(index="year", columns="month", values="net_pnl").fillna(0.0)
    
    # Friday VIX Level breakdown
    vix_bins = [0, 11, 12, 13]
    vix_labels = ["<11", "11-12", "12-13"]
    trades_df["vix_group"] = pd.cut(trades_df["entry_vix"], bins=vix_bins, labels=vix_labels)
    vix_stats = trades_df.groupby("vix_group", observed=False).agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        net_profit=("net_pnl", "sum"),
        avg_profit=("net_pnl", "mean")
    ).reset_index()
    
    # Monday Gaps breakdown
    gap_bins = [-100, -1.0, -0.5, 0, 0.5, 1.0, 100]
    gap_labels = ["<-1.0%", "-1.0% to -0.5%", "-0.5% to 0%", "0% to 0.5%", "0.5% to 1.0%", ">1.0%"]
    trades_df["gap_group"] = pd.cut(trades_df["monday_gap_pct"], bins=gap_bins, labels=gap_labels)
    gap_stats = trades_df.groupby("gap_group", observed=False).agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        net_profit=("net_pnl", "sum"),
        avg_profit=("net_pnl", "mean")
    ).reset_index()
    
    # Breach, Trail-Stop and Adjustment counts
    num_breach = (trades_df["exit_reason"] == "BREACH_EXIT").sum()
    num_trail_stop = (trades_df["exit_reason"] == "TUESDAY_TRAIL_STOP").sum()
    num_adjusted = trades_df["adjusted"].sum()
    avg_mtm_before_adj = trades_df[trades_df["adjusted"]]["adj_mtm"].mean()
    pnl_after_adj = trades_df[trades_df["adjusted"]]["net_pnl"].mean()
    
    # Print results to console
    print("\n" + "="*80)
    print(" BACKTEST RESULTS: Friday Low-VIX Weekend Carry Iron Condor")
    print("="*80)
    print(f"Initial Capital            : Rs. {INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital              : Rs. {trades_df['capital'].iloc[-1]:,.2f}")
    print(f"Net Profit                 : Rs. {net_profit:,.2f} ({net_profit/INITIAL_CAPITAL*100:.2f}%)")
    print(f"Total Trades               : {total_trades}")
    print(f"Winning Trades             : {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades              : {total_trades - winning_trades}")
    print(f"Profit Factor              : {profit_factor:.2f}")
    print(f"Expectancy (Net Avg/Risk)  : {expectancy:.2f}")
    print(f"Max Drawdown %             : {max_drawdown:.2f}%")
    print(f"Max Drawdown Rs.           : Rs. {max_drawdown_amount:,.2f}")
    print("-"*80)
    print(f"Breach Exits (Mon/Tue)     : {num_breach} trades")
    print(f"Tuesday Trail-Stop Exits   : {num_trail_stop} trades")
    print(f"Successful Adjustments     : {num_adjusted} trades")
    print(f"Avg Monday MTM Before Adj  : Rs. {avg_mtm_before_adj:.2f}")
    print(f"Avg Net PnL of Adjusted    : Rs. {pnl_after_adj:.2f}")
    print("="*80)

    # Walk-Forward Split (Overfitting check)
    # In-Sample: 2022-2023, Out-of-Sample: 2024-onwards
    IS_CUTOFF = pd.Timestamp("2024-01-01")
    is_df  = trades_df[trades_df["entry_time"] < IS_CUTOFF]
    oos_df = trades_df[trades_df["entry_time"] >= IS_CUTOFF]

    def period_stats(df, label):
        if df.empty:
            print(f"\n{label}: No trades")
            return
        n = len(df)
        wins = (df["net_pnl"] > 0).sum()
        pf_g = df[df["net_pnl"] > 0]["net_pnl"].sum()
        pf_l = abs(df[df["net_pnl"] <= 0]["net_pnl"].sum())
        pf = pf_g / pf_l if pf_l > 0 else float("inf")
        cap_series = pd.Series([INITIAL_CAPITAL] + (INITIAL_CAPITAL + df["net_pnl"].cumsum()).tolist())
        dd = ((cap_series.cummax() - cap_series) / cap_series.cummax() * 100).max()
        print(f"\n{label} ({df['entry_time'].dt.year.min()}-{df['entry_time'].dt.year.max()})")
        print(f"  Trades      : {n}  |  Win Rate  : {wins/n*100:.1f}%  |  Profit Factor : {pf:.2f}")
        print(f"  Net P&L     : Rs. {df['net_pnl'].sum():,.0f}  |  Max Drawdown : {dd:.2f}%")

    print("\n" + "="*80)
    print(" WALK-FORWARD OVERFITTING CHECK")
    print("="*80)
    period_stats(is_df,  "IN-SAMPLE  (Training  2022-2023)")
    period_stats(oos_df, "OUT-OF-SAMPLE (Live   2024-2026)")
    print("\n  Healthy if OOS Win Rate and Profit Factor are in the same ballpark as IS.")
    print("  A sharp drop in OOS metrics signals overfitting.")
    print("="*80)

    # Save Equity Curve Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Equity curve
    ax1.plot(trades_df["entry_time"], trades_df["capital"], label="Net Equity", color="#2ca02c", lw=2)
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Strategy Performance: Equity Curve")
    ax1.set_ylabel("Capital (Rs.)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Drawdowns
    ax2.fill_between(trades_df["entry_time"], -drawdowns[1:], color="#d62728", alpha=0.3, label="Drawdown %")
    ax2.set_title("Drawdown Curve")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    chart_path = "equity_drawdown.png"
    plt.savefig(chart_path)
    plt.close()
    
    # Generate HTML Report
    generate_html_report(trades_df, pivot_table, vix_stats, gap_stats, max_drawdown, profit_factor, expectancy, num_breach, num_adjusted, avg_mtm_before_adj, pnl_after_adj)
    print(f"\nHTML Performance Report generated: [nifty_lowvix_report.html](file:///{os.path.abspath('nifty_lowvix_report.html')})")

# ============================================================
# 6. HTML REPORT GENERATOR
# ============================================================
def generate_html_report(trades_df, pivot_table, vix_stats, gap_stats, max_drawdown, profit_factor, expectancy, num_breach, num_adjusted, avg_mtm_before_adj, pnl_after_adj):
    net_profit = trades_df["net_pnl"].sum()
    total_trades = len(trades_df)
    win_rate = (trades_df["net_pnl"] > 0).mean() * 100
    
    # Generate HTML table for monthly returns
    monthly_html = "<table class='monthly-table'><thead><tr><th>Year</th>"
    for m in range(1, 13):
        monthly_html += f"<th>Month {m}</th>"
    monthly_html += "<th>Total</th></tr></thead><tbody>"
    
    for year, row in pivot_table.iterrows():
        monthly_html += f"<tr><td><strong>{year}</strong></td>"
        year_total = 0.0
        for m in range(1, 13):
            val = row.get(m, 0.0)
            year_total += val
            color_class = "pos-return" if val > 0 else "neg-return" if val < 0 else "zero-return"
            monthly_html += f"<td class='{color_class}'>₹{val:,.0f}</td>"
        color_class = "pos-return" if year_total > 0 else "neg-return" if year_total < 0 else "zero-return"
        monthly_html += f"<td class='{color_class}'><strong>₹{year_total:,.0f}</strong></td></tr>"
    monthly_html += "</tbody></table>"
    
    # Generate VIX stats table
    vix_html = "<table class='stats-table'><thead><tr><th>VIX Range</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th><th>Avg Profit/Trade</th></tr></thead><tbody>"
    for _, row in vix_stats.iterrows():
        vix_html += f"<tr><td>{row['vix_group']}</td><td>{row['trades']}</td><td>{row['win_rate']:.1f}%</td><td>₹{row['net_profit']:,.2f}</td><td>₹{row['avg_profit']:,.2f}</td></tr>"
    vix_html += "</tbody></table>"
    
    # Generate Gap stats table
    gap_html = "<table class='stats-table'><thead><tr><th>Monday Opening Gap</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th><th>Avg Profit/Trade</th></tr></thead><tbody>"
    for _, row in gap_stats.iterrows():
        gap_html += f"<tr><td>{row['gap_group']}</td><td>{row['trades']}</td><td>{row['win_rate']:.1f}%</td><td>₹{row['net_profit']:,.2f}</td><td>₹{row['avg_profit']:,.2f}</td></tr>"
    gap_html += "</tbody></table>"
    
    # Generate trade logs table
    trade_logs_html = "<table class='logs-table'><thead><tr><th>Date</th><th>Entry Spot</th><th>Entry VIX</th><th>Adjusted</th><th>Adj Spot</th><th>Adj PnL</th><th>Exit Reason</th><th>Exit Spot</th><th>Gross PnL</th><th>Charges</th><th>Net PnL</th></tr></thead><tbody>"
    
    # Show last 30 trades in HTML log for brevity
    recent_trades = trades_df.iloc[::-1].head(30)
    for _, r in recent_trades.iterrows():
        adj_str = "Yes" if r["adjusted"] else "No"
        adj_spot_str = f"₹{r['adj_spot']:,.1f}" if r["adjusted"] else "-"
        adj_pnl_str = f"₹{r['adj_mtm']:,.1f}" if r["adjusted"] else "-"
        color_class = "pos-return" if r["net_pnl"] > 0 else "neg-return"
        gross_color = "pos-return" if r["gross_pnl"] > 0 else "neg-return"
        trade_logs_html += f"<tr><td>{r['entry_time'].strftime('%Y-%m-%d')}</td><td>₹{r['entry_spot']:,.1f}</td><td>{r['entry_vix']:.2f}</td><td>{adj_str}</td><td>{adj_spot_str}</td><td>{adj_pnl_str}</td><td>{r['exit_reason']}</td><td>₹{r['exit_spot']:,.1f}</td><td class='{gross_color}'>₹{r['gross_pnl']:,.2f}</td><td style='color: #7f8c8d;'>₹{r['charges']:,.2f}</td><td class='{color_class}'>₹{r['net_pnl']:,.2f}</td></tr>"
    trade_logs_html += "</tbody></table>"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Friday Low-VIX Carry Backtest Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fb;
            color: #333;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 15px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            text-align: center;
        }}
        .metric-card.profit {{
            border-left-color: #2ecc71;
        }}
        .metric-card.drawdown {{
            border-left-color: #e74c3c;
        }}
        .metric-title {{
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .chart-section {{
            margin-bottom: 40px;
            text-align: center;
        }}
        .chart-img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .table-section {{
            margin-bottom: 45px;
        }}
        .table-section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
        }}
        .pos-return {{
            color: #2ecc71;
            font-weight: 500;
        }}
        .neg-return {{
            color: #e74c3c;
            font-weight: 500;
        }}
        .zero-return {{
            color: #94a3b8;
        }}
        .monthly-table td {{
            font-size: 12px;
        }}
        .monthly-table th {{
            font-size: 12px;
        }}
        .stats-table, .logs-table {{
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            border-radius: 6px;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Nifty Low-VIX Friday Carry Iron Condor Performance Report</h1>
        
        <div class="metrics-grid">
            <div class="metric-card profit">
                <div class="metric-title">Net Profit</div>
                <div class="metric-value">₹{net_profit:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value">{win_rate:.2f}%</div>
            </div>
            <div class="metric-card profit">
                <div class="metric-title">Profit Factor</div>
                <div class="metric-value">{profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Expectancy</div>
                <div class="metric-value">{expectancy:.2f}</div>
            </div>
            <div class="metric-card drawdown">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value">{max_drawdown:.2f}%</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Equity Curve & Drawdown Profile</h2>
            <img class="chart-img" src="equity_drawdown.png" alt="Equity & Drawdown Curve" />
        </div>
        
        <div class="table-section">
            <h2>Monthly P&L Matrix</h2>
            {monthly_html}
        </div>
        
        <div class="table-section">
            <h2>Performance by Friday VIX Level</h2>
            {vix_html}
        </div>
        
        <div class="table-section">
            <h2>Performance by Monday Opening Gap</h2>
            {gap_html}
        </div>
        
        <div class="table-section">
            <h2>Execution & Adjustment Analysis</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Breach Exits (Monday/Tuesday)</th>
                        <th>Successful Adjustments (MTM >= ₹1,200)</th>
                        <th>Avg Monday MTM Before Adj</th>
                        <th>Avg Net PnL After Adjustment</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="neg-return">{num_breach} trades</td>
                        <td class="pos-return">{num_adjusted} trades</td>
                        <td>₹{avg_mtm_before_adj:.2f}</td>
                        <td class="pos-return">₹{pnl_after_adj:.2f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="table-section">
            <h2>Recent Trade Log (Last 30 Trades)</h2>
            {trade_logs_html}
        </div>
    </div>
</body>
</html>
"""
    with open("nifty_lowvix_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# ============================================================
# 7. EXECUTION
# ============================================================
if __name__ == "__main__":
    trades_df = run_backtest()
    analyze_results(trades_df)
