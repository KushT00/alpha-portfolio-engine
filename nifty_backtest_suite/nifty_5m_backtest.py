"""
nifty_5m_backtest.py
====================
Backtest for the Nifty 5m Triple Confirm option-buying strategy.
Matches nifty_5m_triple_confirm.py logic exactly.

Now upgraded to include:
1. Pure Black-Scholes option pricing model.
2. Dynamic Implied Volatility (IV) mapped from India VIX daily (vix.csv).
3. Exact strike selection (weekly Tuesday expiry closest to Rs. 200 premium).
4. Real-world intra-bar stop-loss triggers and Tuesday 15:15 rolls.
"""
import os
import pandas as pd
from datetime import time, datetime, timedelta
from math import log, sqrt, exp, erf

# ============================================================
# PARAMETERS
# ============================================================
SMA_FAST_LEN     = 68
SMA_SLOW_LEN     = 90
EMA_TREND_LEN    = 340
SL_PCT           = 0.75       # % on Nifty Spot
TRAIL_PCT        = 0.75       # % on Nifty Spot
TARGET_PREMIUM   = 200.0      # Rs. per unit option premium at entry
LOT_SIZE         = 65         # Nifty lot size
BROKERAGE        = 60.0       # flat Rs. per round-trip trade
STRIKE_INTERVAL  = 50         # Nifty strike interval
RISK_FREE_RATE   = 0.07       # 7%

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NIFTY_NSE_INDEX_5m.csv")
_VIX_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vix.csv")

# ============================================================
# BLACK-SCHOLES OPTION PRICING
# ============================================================
def norm_cdf(x):
    """Standard normal CDF using built-in math.erf."""
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
    """Black-Scholes Call option price."""
    if T <= 0:
        return max(0.0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Black-Scholes Put option price."""
    if T <= 0:
        return max(0.0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def get_next_tuesday_expiry(dt):
    """Get the next Tuesday 15:30 from the given datetime."""
    days_ahead = 1 - dt.weekday()  # Tuesday = 1
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        expiry_time = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        if dt < expiry_time:
            return expiry_time
        else:
            days_ahead = 7
    next_tue = dt + timedelta(days=days_ahead)
    return next_tue.replace(hour=15, minute=30, second=0, microsecond=0)

def time_to_expiry_years(current_dt, expiry_dt):
    """Calculate T in years."""
    diff = (expiry_dt - current_dt).total_seconds()
    if diff <= 0:
        return 0.0
    return diff / (365.25 * 24 * 3600)

def find_strike_near_premium(S, current_dt, option_type, target_premium=200.0, r=RISK_FREE_RATE, sigma=0.14):
    """Find the Nifty strike whose BS premium is closest to target_premium."""
    expiry_dt = get_next_tuesday_expiry(current_dt)
    T = time_to_expiry_years(current_dt, expiry_dt)
    if T <= 0:
        expiry_dt = get_next_tuesday_expiry(current_dt + timedelta(days=1))
        T = time_to_expiry_years(current_dt, expiry_dt)
    
    atm = round(S / STRIKE_INTERVAL) * STRIKE_INTERVAL
    best_strike = atm
    min_diff = float('inf')
    
    for offset in range(-40, 41):  # ±2000 points
        K = atm + offset * STRIKE_INTERVAL
        if K <= 0:
            continue
        if option_type == "CE":
            premium = bs_call(S, K, T, r, sigma)
        else:
            premium = bs_put(S, K, T, r, sigma)
        diff = abs(premium - target_premium)
        if diff < min_diff:
            min_diff = diff
            best_strike = K
            
    return best_strike, expiry_dt

def compute_option_price(S, K, expiry_dt, current_dt, option_type, r=RISK_FREE_RATE, sigma=0.14):
    """Compute option price at a given time."""
    T = time_to_expiry_years(current_dt, expiry_dt)
    if option_type == "CE":
        return bs_call(S, K, T, r, sigma)
    else:
        return bs_put(S, K, T, r, sigma)

# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def run_5m_backtest(start_date: str = "2022-01-01", data_path: str = None) -> list:
    """
    Run 5m Triple Confirm backtest and return list of closed trade dicts.
    """
    path = data_path or _DATA_PATH
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    start_dt = pd.to_datetime(start_date)

    # Load India VIX daily data for dynamic IV
    vix_df = pd.read_csv(_VIX_PATH)
    vix_df['vix_date'] = pd.to_datetime(vix_df['date'])
    vix_df = vix_df.sort_values('vix_date').reset_index(drop=True)
    vix_df['iv'] = vix_df['close'] / 100.0  # VIX 16.45 -> 0.1645 annualized vol
    vix_daily = vix_df[['vix_date', 'iv']].copy()
    
    # Map VIX to Nifty 5m bars by date
    df['trade_date'] = df['datetime'].dt.normalize()
    df = df.merge(vix_daily, left_on='trade_date', right_on='vix_date', how='left')
    df['iv'] = df['iv'].ffill().bfill()  # Forward-fill then back-fill any gaps
    df.drop(columns=['vix_date', 'trade_date'], inplace=True)

    close_buf = []
    ema340 = None
    ema_initialized = False
    ema_mult = 2.0 / (EMA_TREND_LEN + 1.0)

    # Strategy state
    bull_trend         = False
    bear_trend         = False
    long_pullback_rdy  = False
    short_pullback_rdy = False
    last_signal        = 0
    opposite_seen      = True

    # Position state
    pos_active         = False
    pos_opt_type       = ""
    pos_entry_spot     = 0.0
    pos_entry_time     = None
    pos_sl             = 0.0
    pos_trail_high     = 0.0
    pos_trail_low      = 0.0
    pos_trail_active   = False
    pos_strike         = 0
    pos_entry_premium  = 0.0
    pos_expiry_dt      = None

    all_trades = []

    def _close(exit_time, exit_spot, exit_reason, bar_iv):
        nonlocal pos_active, last_signal, opposite_seen
        nonlocal long_pullback_rdy, short_pullback_rdy
        nonlocal pos_strike, pos_entry_premium, pos_expiry_dt

        # Compute option price at exit using Black-Scholes and bar_iv
        ep = compute_option_price(
            exit_spot, pos_strike, pos_expiry_dt, exit_time, pos_opt_type, r=RISK_FREE_RATE, sigma=bar_iv
        )
        gpnl = (ep - pos_entry_premium) * LOT_SIZE
        npnl = gpnl - BROKERAGE
        
        all_trades.append({
            "entry_time":         pos_entry_time,
            "exit_time":          exit_time,
            "option_type":        pos_opt_type,
            "direction":          "LONG" if pos_opt_type == "CE" else "SHORT",
            "entry_spot":         pos_entry_spot,
            "exit_spot":          exit_spot,
            "entry_option_price": round(pos_entry_premium, 2),
            "exit_option_price":  round(ep, 2),
            "gross_pnl":          round(gpnl, 2),
            "charges":            BROKERAGE,
            "net_pnl":            round(npnl, 2),
            "exit_reason":        exit_reason,
        })
        pos_active    = False
        last_signal   = 0
        opposite_seen = True
        if bull_trend:  long_pullback_rdy  = False
        if bear_trend:  short_pullback_rdy = False

    def _open(dt, close, high, low, opt_type, bar_iv):
        nonlocal pos_active, pos_opt_type, pos_entry_spot, pos_entry_time
        nonlocal pos_sl, pos_trail_high, pos_trail_low, pos_trail_active
        nonlocal last_signal, opposite_seen, long_pullback_rdy, short_pullback_rdy
        nonlocal pos_strike, pos_entry_premium, pos_expiry_dt

        pos_active       = True
        pos_opt_type     = opt_type
        pos_entry_spot   = close
        pos_entry_time   = dt
        pos_trail_active = False

        # Select option strike and compute entry premium via Black-Scholes
        pos_strike, pos_expiry_dt = find_strike_near_premium(
            close, dt, opt_type, target_premium=TARGET_PREMIUM, r=RISK_FREE_RATE, sigma=bar_iv
        )
        pos_entry_premium = compute_option_price(
            close, pos_strike, pos_expiry_dt, dt, opt_type, r=RISK_FREE_RATE, sigma=bar_iv
        )

        if opt_type == "CE":
            pos_sl            = close * (1.0 - SL_PCT / 100)
            pos_trail_high    = high
            pos_trail_low     = 0.0
            last_signal       = 1
            long_pullback_rdy = False
        else:
            pos_sl             = close * (1.0 + SL_PCT / 100)
            pos_trail_high     = 0.0
            pos_trail_low      = low
            last_signal        = -1
            short_pullback_rdy = False
        opposite_seen = False

    for _, row in df.iterrows():
        dt    = row["datetime"]
        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        bar_iv = float(row["iv"])
        t     = dt.time()

        # EMA340 incremental maintenance
        close_buf.append(close)
        if not ema_initialized:
            if len(close_buf) == EMA_TREND_LEN:
                ema340 = sum(close_buf) / EMA_TREND_LEN
                ema_initialized = True
        else:
            ema340 = (close * ema_mult) + (ema340 * (1.0 - ema_mult))

        if not ema_initialized:
            continue
        if dt < start_dt:
            continue
        if len(close_buf) < SMA_SLOW_LEN + 1:
            continue

        # SMA calculations
        sma68 = sum(close_buf[-SMA_FAST_LEN:]) / SMA_FAST_LEN
        sma90 = sum(close_buf[-SMA_SLOW_LEN:]) / SMA_SLOW_LEN
        p = close_buf[:-1]
        if len(p) < SMA_SLOW_LEN:
            continue
        prev_sma68 = sum(p[-SMA_FAST_LEN:]) / SMA_FAST_LEN
        prev_sma90 = sum(p[-SMA_SLOW_LEN:]) / SMA_SLOW_LEN

        bull_cross = (prev_sma68 <= prev_sma90) and (sma68 > sma90)
        bear_cross = (prev_sma68 >= prev_sma90) and (sma68 < sma90)
        above_ema  = close > ema340
        below_ema  = close < ema340

        # Trend memory
        if bull_cross:
            bull_trend = True;  bear_trend = False
        elif bear_cross:
            bear_trend = True;  bull_trend = False

        # Debounce
        if last_signal == 1  and bear_cross: opposite_seen = True
        if last_signal == -1 and bull_cross: opposite_seen = True

        # Pullback memory
        if bull_trend and below_ema: long_pullback_rdy  = True
        if bear_trend and above_ema: short_pullback_rdy = True
        if bear_cross: long_pullback_rdy  = False
        if bull_cross: short_pullback_rdy = False

        # ── TUESDAY 15:15 EXPIRY ROLLOVER ─────────────────────────────
        is_tuesday = (dt.weekday() == 1)
        if is_tuesday and time(15, 15) <= t <= time(15, 20) and pos_active:
            prev_type = pos_opt_type
            _close(dt, close, "EXPIRY_ROLLOVER", bar_iv)
            # Re-open same direction in next expiry at Rs.200
            _open(dt, close, high, low, prev_type, bar_iv)

        # ── EXIT LOGIC ───────────────────────────────────────────────
        if pos_active:
            if pos_opt_type == "CE":
                if not pos_trail_active and (high - pos_entry_spot) / pos_entry_spot * 100 >= TRAIL_PCT:
                    pos_trail_active = True
                    pos_trail_high   = high
                if pos_trail_active:
                    pos_trail_high = max(pos_trail_high, high)
                    pos_sl = max(pos_sl, pos_trail_high * (1 - TRAIL_PCT / 100))
                else:
                    pos_sl = pos_entry_spot * (1 - SL_PCT / 100)
                if low <= pos_sl:
                    _close(dt, pos_sl, "TRAIL_EXIT" if pos_trail_active else "INITIAL_SL_EXIT", bar_iv)

            elif pos_opt_type == "PE":
                if not pos_trail_active and (pos_entry_spot - low) / pos_entry_spot * 100 >= TRAIL_PCT:
                    pos_trail_active = True
                    pos_trail_low    = low
                if pos_trail_active:
                    pos_trail_low = min(pos_trail_low, low)
                    pos_sl = min(pos_sl, pos_trail_low * (1 + TRAIL_PCT / 100))
                else:
                    pos_sl = pos_entry_spot * (1 + SL_PCT / 100)
                if high >= pos_sl:
                    _close(dt, pos_sl, "TRAIL_EXIT" if pos_trail_active else "INITIAL_SL_EXIT", bar_iv)

        # ── ENTRY LOGIC ──────────────────────────────────────────────
        if not pos_active and time(9, 15) <= t <= time(15, 25):
            long_base    = bull_cross and above_ema and (last_signal != 1 or opposite_seen)
            long_reentry = bull_trend and long_pullback_rdy and above_ema
            short_base    = bear_cross and below_ema and (last_signal != -1 or opposite_seen)
            short_reentry = bear_trend and short_pullback_rdy and below_ema

            if long_base or long_reentry:
                _open(dt, close, high, low, "CE", bar_iv)
            elif short_base or short_reentry:
                _open(dt, close, high, low, "PE", bar_iv)

    return all_trades

# ============================================================
# METRICS & REPORTING
# ============================================================
def aggregate_5m_metrics(trades: list, initial_capital: float = 150000.0) -> dict:
    """Compute summary stats and monthly breakdown for option-buying trades."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "net_pnl": 0.0, "gross_pnl": 0.0, "charges": 0.0,
            "max_drawdown_pct": 0.0, "monthly_data": {}, "equity_curve": [initial_capital]
        }

    wins   = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in trades if t["net_pnl"] <= 0]
    tw, tl = sum(wins), abs(sum(losses))
    pf = tw / tl if tl > 0 else float("inf")

    equity = initial_capital
    peak   = initial_capital
    max_dd = 0.0
    equity_curve = [initial_capital]
    for t in trades:
        equity += t["net_pnl"]
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak * 100)
        equity_curve.append(round(equity, 2))

    monthly_data = {}
    for t in trades:
        ym = t["exit_time"].strftime("%Y-%m")
        if ym not in monthly_data:
            monthly_data[ym] = {"trades": 0, "gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0}
        monthly_data[ym]["trades"]    += 1
        monthly_data[ym]["gross_pnl"] += t["gross_pnl"]
        monthly_data[ym]["charges"]   += t["charges"]
        monthly_data[ym]["net_pnl"]   += t["net_pnl"]

    return {
        "total_trades":     len(trades),
        "win_rate":         len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor":    pf,
        "net_pnl":          sum(t["net_pnl"]   for t in trades),
        "gross_pnl":        sum(t["gross_pnl"] for t in trades),
        "charges":          sum(t["charges"]   for t in trades),
        "max_drawdown_pct": max_dd,
        "monthly_data":     monthly_data,
        "equity_curve":     equity_curve,
    }


if __name__ == "__main__":
    print("Running 5m Triple Confirm backtest...")
    trades = run_5m_backtest(start_date="2022-01-01")
    stats  = aggregate_5m_metrics(trades)
    print(f"\n=== 5m Triple Confirm Backtest Results ===")
    print(f"Total Trades    : {stats['total_trades']}")
    print(f"Win Rate        : {stats['win_rate']:.1f}%")
    print(f"Profit Factor   : {stats['profit_factor']:.2f}")
    print(f"Net P&L         : Rs. {stats['net_pnl']:,.2f}")
    print(f"Gross P&L       : Rs. {stats['gross_pnl']:,.2f}")
    print(f"Total Charges   : Rs. {stats['charges']:,.2f}")
    print(f"Max Drawdown    : {stats['max_drawdown_pct']:.2f}%")
    print(f"\nMonthly Breakdown (last 12):")
    for k, v in sorted(stats['monthly_data'].items())[-12:]:
        print(f"  {k}: Trades={v['trades']} Gross=Rs.{v['gross_pnl']:,.0f} Charges=Rs.{v['charges']:,.0f} Net=Rs.{v['net_pnl']:,.0f}")
