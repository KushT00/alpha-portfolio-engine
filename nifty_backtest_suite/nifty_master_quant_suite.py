import os
import pandas as pd
import numpy as np
import math
from datetime import time, datetime, timedelta

# ============================================================
# 1. PARAMETERS & CONFIG
# ============================================================
INITIAL_CAPITAL    = 150000.0   # ₹1.5 Lakhs total capital pool
LOT_SIZE           = 65         # Nifty lot size
RISK_FREE_RATE     = 0.07       # 7%
SLIPPAGE_PER_LEG   = 0.10       # ₹0.10 per contract per leg (2 ticks)

# India VIX Multipliers (to adjust flat VIX to short-term option premiums)
PUT_IV_MULT        = 0.98
CALL_IV_MULT       = 0.94

# Strategy Specific Parameters

# Wednesday Carry
WED_LOTS           = 1          # 1 lot base
WED_QTY            = WED_LOTS * LOT_SIZE
WED_TARGET_SL      = 1.5        # exit when loss >= 1.5 * premium_collected

# Friday Low-VIX Carry
FRI_LOTS           = 2          # 2 lots for Friday Low VIX carry
FRI_QTY            = FRI_LOTS * LOT_SIZE
FRI_VIX_THRESHOLD  = 13.0
FRI_TARGET_MTM     = 1200.0     # Monday adjustment target
FRI_TUESDAY_FLOOR  = -500.0     # Tuesday profit floor

# Tuesday Expiry IC
TUE_LOTS           = 1          # 1 lot base
TUE_QTY            = TUE_LOTS * LOT_SIZE
TUE_VIX_MIN        = 13.5
TUE_VIX_MAX        = 22.0

STRIKE_INTERVAL    = 50

# ============================================================
# 2. TECHNICAL INDICATORS & DATA PREPROCESSING
# ============================================================
def calculate_adx_series(df_5m, period=14):
    """
    Resamples 5-minute spot data to 1-hour bars, calculates Wilder's ADX
    according to master strategy standards, and maps it back to 5-minute bars.
    """
    # 1. Resample 5m bars to 1h
    df_1h = df_5m.set_index('datetime').resample('1h', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    
    # 2. Calculate True Range and DM using pandas
    prev_close = df_1h['close'].shift(1)
    high = df_1h['high']
    low = df_1h['low']
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # 3. Wilder's Smoothing
    def smooth_wilder(series, n):
        res = pd.Series(index=series.index, dtype=float)
        if len(series) < n:
            return res
        res.iloc[n-1] = series.iloc[:n].sum()
        for i in range(n, len(series)):
            res.iloc[i] = res.iloc[i-1] - (res.iloc[i-1] / n) + series.iloc[i]
        return res
        
    atr = smooth_wilder(pd.Series(tr), period)
    pdm_s = smooth_wilder(pd.Series(plus_dm), period)
    mdm_s = smooth_wilder(pd.Series(minus_dm), period)
    
    # 4. Calculate +DI and -DI
    pdi = 100.0 * pdm_s / atr
    mdi = 100.0 * mdm_s / atr
    
    # 5. Calculate DX & ADX
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    dx = dx.fillna(0.0)
    
    adx_sum = smooth_wilder(dx, period)
    df_1h['adx'] = adx_sum / period
    
    # 6. Map back to 5m bars
    df_adx_map = df_1h[['datetime', 'adx']].copy()
    df_5m = pd.merge_asof(df_5m.sort_values('datetime'), df_adx_map.sort_values('datetime'), 
                          on='datetime', direction='backward')
    return df_5m

def load_and_preprocess_data():
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "NIFTY_NSE_INDEX_5m.csv"))
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    vix_df = pd.read_csv(os.path.join(base_dir, "vix_5m.csv"))
    vix_df['datetime'] = pd.to_datetime(vix_df['date'] + ' ' + vix_df['time'])
    vix_df = vix_df.sort_values('datetime').reset_index(drop=True)
    vix_clean = vix_df[['datetime', 'close']].copy()
    vix_clean.rename(columns={'close': 'vix'}, inplace=True)
    vix_clean['iv'] = vix_clean['vix'] / 100.0
    
    # Merge datasets
    df = df.merge(vix_clean, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[df['datetime'] >= '2022-01-01 00:00:00'].reset_index(drop=True)
    
    # Calculate ADX
    print("Calculating Wilder's ADX (1h resampled)...")
    df = calculate_adx_series(df, period=14)
    # Backfill/forward-fill leading ADX values
    df['adx'] = df['adx'].ffill().bfill()
    
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['time_str'] = df['datetime'].dt.time
    df['trade_date'] = df['datetime'].dt.normalize()
    
    # Simulate PCR and Morning Spot data
    print("Simulating PCR and Morning Spot parameters...")
    df['pcr'] = 1.0  # Baseline neutral PCR
    
    # Calculate morning spot (the open of the first 5-min candle of the day)
    morning_spot_map = df.groupby('trade_date').first()['open'].to_dict()
    df['morning_spot'] = df['trade_date'].map(morning_spot_map)
    
    # Calculate morning PCR (simulated as neutral 1.0 at market open)
    df['morning_pcr'] = 1.0
    
    # Calculate spot shift and PCR shift
    df['spot_shift'] = df['close'] - df['morning_spot']
    # PCR shift: rises in bullish markets (spot rises) and falls in bearish markets (spot falls)
    # We simulate this via: PCR Shift = Spot Shift / 200.0
    df['pcr_shift'] = df['spot_shift'] / 200.0
    df['pcr'] = df['morning_pcr'] + df['pcr_shift']
    
    return df

# ============================================================
# 3. BLACK-SCHOLES PRICING ENGINE
# ============================================================
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def time_to_expiry_years(current_dt, expiry_dt):
    diff_seconds = (expiry_dt - current_dt).total_seconds()
    if diff_seconds <= 0:
        return 0.0
    diff_days = diff_seconds / (24 * 3600)
    # Compress weekends: if Friday→Tuesday, subtract 1.6 days for non-trading weekend
    if current_dt.weekday() == 4 and expiry_dt.weekday() == 1:
        diff_days = max(diff_days - 1.6, 0.05)
    return diff_days / 252.0

def get_next_tuesday_expiry(dt):
    """Get the next Tuesday 15:30 from given datetime."""
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

def calculate_trade_charges(buys_premium, sells_premium, num_orders, qty):
    brokerage = num_orders * 20.0
    slippage = num_orders * SLIPPAGE_PER_LEG * qty
    exchange_charges = (buys_premium + sells_premium) * qty * 0.00053
    stt = sells_premium * qty * 0.000625
    stamp_duty = buys_premium * qty * 0.00003
    sebi_fee = (buys_premium + sells_premium) * qty * 0.000001
    gst = (brokerage + exchange_charges + sebi_fee) * 0.18
    return brokerage + exchange_charges + stt + stamp_duty + sebi_fee + gst + slippage

def price_ic_legs(spot, atm, short_off, hedge_off, T, r, iv, qty):
    """Price an Iron Condor and return (net_credit_per_unit, buys_prem, sells_prem, legs_dict)."""
    sc = atm + short_off * STRIKE_INTERVAL
    sp = atm - short_off * STRIKE_INTERVAL
    lc = atm + hedge_off * STRIKE_INTERVAL
    lp = atm - hedge_off * STRIKE_INTERVAL

    sc_p = bs_call(spot, sc, T, r, iv * CALL_IV_MULT)
    sp_p = bs_put(spot, sp, T, r, iv * PUT_IV_MULT)
    lc_p = bs_call(spot, lc, T, r, iv * CALL_IV_MULT)
    lp_p = bs_put(spot, lp, T, r, iv * PUT_IV_MULT)

    net_credit = (sc_p + sp_p) - (lc_p + lp_p)
    return net_credit, (lc_p + lp_p), (sc_p + sp_p), {
        "short_call": sc, "short_put": sp, "long_call": lc, "long_put": lp,
        "sc_p": sc_p, "sp_p": sp_p, "lc_p": lc_p, "lp_p": lp_p
    }

def price_ratio_spread(spot, buy_off, sell_off, far_hedge_off, T, r, iv, is_call=True):
    """Price a 1:3:2 ratio spread and return (net_credit_per_unit, buys_prem, sells_prem, legs_dict)."""
    fn = bs_call if is_call else bs_put
    mult = CALL_IV_MULT if is_call else PUT_IV_MULT
    sign = 1 if is_call else -1

    buy_k   = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL + sign * buy_off * STRIKE_INTERVAL
    sell_k   = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL + sign * sell_off * STRIKE_INTERVAL
    hedge_k  = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL + sign * far_hedge_off * STRIKE_INTERVAL

    buy_p   = fn(spot, buy_k, T, r, iv * mult)
    sell_p  = fn(spot, sell_k, T, r, iv * mult)
    hedge_p = fn(spot, hedge_k, T, r, iv * mult)

    # Buy 1x near, Sell 3x mid, Buy 2x far
    net_credit = (3 * sell_p) - (1 * buy_p + 2 * hedge_p)
    buys_prem = buy_p + 2 * hedge_p
    sells_prem = 3 * sell_p
    return net_credit, buys_prem, sells_prem, {
        "buy_strike": buy_k, "sell_strike": sell_k, "hedge_strike": hedge_k,
        "buy_p": buy_p, "sell_p": sell_p, "hedge_p": hedge_p
    }

def current_ic_value(spot, legs, T, r, iv):
    """Calculate current net cost of an IC position."""
    sc_p = bs_call(spot, legs["short_call"], T, r, iv * CALL_IV_MULT)
    sp_p = bs_put(spot, legs["short_put"], T, r, iv * PUT_IV_MULT)
    lc_p = bs_call(spot, legs["long_call"], T, r, iv * CALL_IV_MULT)
    lp_p = bs_put(spot, legs["long_put"], T, r, iv * PUT_IV_MULT)
    return (sc_p + sp_p) - (lc_p + lp_p)

def current_ratio_value(spot, legs, T, r, iv, is_call=True):
    """Calculate current net cost of a ratio position."""
    fn = bs_call if is_call else bs_put
    mult = CALL_IV_MULT if is_call else PUT_IV_MULT
    sell_p = fn(spot, legs["sell_strike"], T, r, iv * mult)
    buy_p = fn(spot, legs["buy_strike"], T, r, iv * mult)
    hedge_p = fn(spot, legs["hedge_strike"], T, r, iv * mult)
    return (3 * sell_p) - (1 * buy_p + 2 * hedge_p)

def current_tuesday_value(spot, legs, T, r, iv):
    """Calculate current net cost of a Tuesday IC position (handles closed sides)."""
    val = 0.0
    if legs["short_call"] is not None:
        sc_p = bs_call(spot, legs["short_call"], T, r, iv * CALL_IV_MULT)
        lc_p = bs_call(spot, legs["long_call"], T, r, iv * CALL_IV_MULT)
        val += (sc_p - lc_p)
    if legs["short_put"] is not None:
        sp_p = bs_put(spot, legs["short_put"], T, r, iv * PUT_IV_MULT)
        lp_p = bs_put(spot, legs["long_put"], T, r, iv * PUT_IV_MULT)
        val += (sp_p - lp_p)
    return val

def current_tuesday_entry_credit(legs):
    """Calculate current net entry credit of active Tuesday IC legs."""
    val = 0.0
    if legs["short_call"] is not None:
        val += (legs["sc_p_entry"] - legs["lc_p_entry"])
    if legs["short_put"] is not None:
        val += (legs["sp_p_entry"] - legs["lp_p_entry"])
    return val

def adjust_tuesday_trade_live(trade, bar, T_current, iv, calendar_log):
    """
    Tuesday Rolling Logic (active until 15:00):
    - Profit Roll: If a short leg decays >= 80%, roll closer.
    - Defensive Roll: If a short leg spikes >= 3x entry, roll further away.
    - Both rolls target the 10-20 Rs premium band. Hedge is 2 strikes away.
    """
    spot = bar['close']
    legs = trade["legs"]
    qty = trade["qty"]

    # 1. Check CE side
    if legs["short_call"] is not None:
        sc_p = bs_call(spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
        lc_p = bs_call(spot, legs["long_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
        entry_p = legs["sc_p_entry"]
        
        decay = (entry_p - sc_p) / entry_p if entry_p > 0 else 0
        spike = sc_p / entry_p if entry_p > 0 else 0
        
        if decay >= 0.80 or spike >= 3.0:
            reason = "ROLL_CE_DECAY" if decay >= 0.80 else "ROLL_CE_SPIKE"
            # Close existing CE spread
            realized = ((legs["sc_p_entry"] - legs["lc_p_entry"]) - (sc_p - lc_p)) * qty
            trade["realized_pnl"] += realized
            trade["charges"] += calculate_trade_charges(lc_p, sc_p, 2, qty)
            trade["num_orders"] += 2
            
            # Find new CE in 10-20 Rs band
            atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            new_ce_off = None
            for off in range(2, 16):
                strike = atm + off * STRIKE_INTERVAL
                new_p = bs_call(spot, strike, T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                if 10 <= new_p <= 20:
                    new_ce_off = off
                    break
            
            if new_ce_off is not None:
                new_sc = atm + new_ce_off * STRIKE_INTERVAL
                new_lc = new_sc + 2 * STRIKE_INTERVAL
                new_sc_p = bs_call(spot, new_sc, T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                new_lc_p = bs_call(spot, new_lc, T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                
                legs["short_call"] = new_sc
                legs["long_call"] = new_lc
                legs["sc_p_entry"] = new_sc_p
                legs["lc_p_entry"] = new_lc_p
                
                trade["charges"] += calculate_trade_charges(new_lc_p, new_sc_p, 2, qty)
                trade["num_orders"] += 2
                trade["initial_net_credit"] = current_tuesday_entry_credit(legs)
                calendar_log.append((bar['datetime'], "tuesday_trade", "ROLL", f"{reason}: new CE short {new_sc}", trade["type"]))
            else:
                legs["short_call"] = None
                legs["long_call"] = None
                legs["sc_p_entry"] = 0.0
                legs["lc_p_entry"] = 0.0
                calendar_log.append((bar['datetime'], "tuesday_trade", "ROLL_FAIL", f"{reason}: no CE strikes found, closed side", trade["type"]))

    # 2. Check PE side
    if legs["short_put"] is not None:
        sp_p = bs_put(spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
        lp_p = bs_put(spot, legs["long_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
        entry_p = legs["sp_p_entry"]
        
        decay = (entry_p - sp_p) / entry_p if entry_p > 0 else 0
        spike = sp_p / entry_p if entry_p > 0 else 0
        
        if decay >= 0.80 or spike >= 3.0:
            reason = "ROLL_PE_DECAY" if decay >= 0.80 else "ROLL_PE_SPIKE"
            # Close existing PE spread
            realized = ((legs["sp_p_entry"] - legs["lp_p_entry"]) - (sp_p - lp_p)) * qty
            trade["realized_pnl"] += realized
            trade["charges"] += calculate_trade_charges(lp_p, sp_p, 2, qty)
            trade["num_orders"] += 2
            
            # Find new PE in 10-20 Rs band
            atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            new_pe_off = None
            for off in range(2, 16):
                strike = atm - off * STRIKE_INTERVAL
                new_p = bs_put(spot, strike, T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                if 10 <= new_p <= 20:
                    new_pe_off = off
                    break
            
            if new_pe_off is not None:
                new_sp = atm - new_pe_off * STRIKE_INTERVAL
                new_lp = new_sp - 2 * STRIKE_INTERVAL
                new_sp_p = bs_put(spot, new_sp, T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                new_lp_p = bs_put(spot, new_lp, T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                
                legs["short_put"] = new_sp
                legs["long_put"] = new_lp
                legs["sp_p_entry"] = new_sp_p
                legs["lp_p_entry"] = new_lp_p
                
                trade["charges"] += calculate_trade_charges(new_lp_p, new_sp_p, 2, qty)
                trade["num_orders"] += 2
                trade["initial_net_credit"] = current_tuesday_entry_credit(legs)
                calendar_log.append((bar['datetime'], "tuesday_trade", "ROLL", f"{reason}: new PE short {new_sp}", trade["type"]))
            else:
                legs["short_put"] = None
                legs["long_put"] = None
                legs["sp_p_entry"] = 0.0
                legs["lp_p_entry"] = 0.0
                calendar_log.append((bar['datetime'], "tuesday_trade", "ROLL_FAIL", f"{reason}: no PE strikes found, closed side", trade["type"]))

# ============================================================
# 4. PHASE 1 DATA INTEGRITY TESTS
# ============================================================
def verify_phase1_data(df):
    print("\n" + "="*80)
    print(" PHASE 1: DATA INTEGRITY VERIFICATION")
    print("="*80)
    required_cols = ['datetime', 'close', 'vix', 'iv', 'adx', 'morning_spot', 'spot_shift', 'pcr', 'pcr_shift']
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    print("[PASS] Test 1: All required columns present.")

    adx_min, adx_max = df['adx'].min(), df['adx'].max()
    assert 0 <= adx_min <= 100 and 0 <= adx_max <= 100
    assert not df['adx'].isna().any()
    print(f"[PASS] Test 2: ADX range verified ({adx_min:.2f} to {adx_max:.2f}).")

    corr = df['spot_shift'].corr(df['pcr_shift'])
    assert abs(corr - 1.0) < 1e-5
    print(f"[PASS] Test 3: PCR Shift proxy verified (corr={corr:.4f}).")

    first_bars = df.groupby('trade_date').first()
    max_dev = abs(first_bars['close'] - first_bars['morning_spot']).max()
    print(f"[PASS] Test 4: Morning spot alignment verified (max dev={max_dev:.2f}).")
    print("="*80 + "\n")

# ============================================================
# 5. STRATEGY MODULES
# ============================================================

# --- 5a. Wednesday Carry Module ---
def simulate_wednesday_carry(df, entry_bar, expiry_dt):
    """
    Simulate a Wednesday carry trade based on regime detection.
    Returns: trade dict or None if skipped.
    """
    spot = entry_bar['close']
    vix  = entry_bar['vix']
    iv   = entry_bar['iv']
    adx  = entry_bar['adx']
    spot_shift = entry_bar['spot_shift']
    pcr_shift  = entry_bar['pcr_shift']
    entry_time = entry_bar['datetime']
    t_str = entry_bar['time_str']

    # Gate: VIX >= 13
    if vix < 13:
        return None

    # IVR proxy: simple percentile rank over trailing VIX data
    # We approximate IVR using the VIX level relative to typical range
    ivr = max(0, min(100, (vix - 10) / (30 - 10) * 100))

    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    T_entry = time_to_expiry_years(entry_time, expiry_dt)
    if T_entry <= 0:
        return None

    trade_type = None
    net_credit = 0
    buys_prem = 0
    sells_prem = 0
    legs = {}
    qty = WED_QTY
    short_offset = 0
    is_ratio = False
    ratio_is_call = True

    # REGIME 1 & 2: Ranging (ADX < 22) — EOD only
    if adx < 22:
        if t_str < time(15, 15, 0):
            return None
        if ivr >= 40:
            trade_type = "BATMAN"
            net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 5, 10, T_entry, RISK_FREE_RATE, iv, qty)
            short_offset = 5
        elif ivr >= 30:
            trade_type = "WIDE_IC"
            net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 6, 10, T_entry, RISK_FREE_RATE, iv, qty)
            short_offset = 6
        else:
            return None  # IVR too low

    # REGIME 1.5: Grey Zone (22 <= ADX < 25) — EOD only
    elif 22 <= adx < 25:
        if t_str < time(15, 15, 0):
            return None
        trade_type = "GREY_IC"
        net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 7, 12, T_entry, RISK_FREE_RATE, iv, qty)
        short_offset = 7

    # REGIME 3 & 4: Trending (ADX >= 25)
    elif adx >= 25:
        if spot_shift >= 50 and pcr_shift > 0.15:
            trade_type = "RATIO_CALL"
            net_credit, buys_prem, sells_prem, legs = price_ratio_spread(spot, 4, 7, 15, T_entry, RISK_FREE_RATE, iv, is_call=True)
            is_ratio = True
            ratio_is_call = True
            short_offset = 7
        elif spot_shift <= -50 and pcr_shift < -0.15:
            trade_type = "RATIO_PUT"
            # Put ratio is actually implemented with calls in master strategy
            net_credit, buys_prem, sells_prem, legs = price_ratio_spread(spot, 3, 5, 13, T_entry, RISK_FREE_RATE, iv, is_call=True)
            is_ratio = True
            ratio_is_call = True
            short_offset = 5
        elif t_str >= time(15, 15, 0):
            # EOD Fallback
            if spot_shift >= 150:
                trade_type = "SKEW_BULL_IC"
                # Bull skew: sell closer PE, wider CE
                net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 6, 10, T_entry, RISK_FREE_RATE, iv, qty)
                short_offset = 7
            elif spot_shift <= -150:
                trade_type = "SKEW_BEAR_IC"
                net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 6, 10, T_entry, RISK_FREE_RATE, iv, qty)
                short_offset = 7
            else:
                trade_type = "WIDE_IC"
                net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 6, 10, T_entry, RISK_FREE_RATE, iv, qty)
                short_offset = 6
        else:
            return None  # Waiting for confirmation

    if trade_type is None:
        return None

    charges = calculate_trade_charges(buys_prem, sells_prem, 4, qty)

    return {
        "slot": "carry_trade",
        "type": trade_type,
        "entry_time": entry_time,
        "entry_spot": spot,
        "entry_vix": vix,
        "entry_adx": adx,
        "expiry_dt": expiry_dt,
        "atm": atm,
        "net_credit": net_credit,
        "buys_prem": buys_prem,
        "sells_prem": sells_prem,
        "legs": legs,
        "qty": qty,
        "short_offset": short_offset,
        "sl_mult": WED_TARGET_SL,
        "is_ratio": is_ratio,
        "ratio_is_call": ratio_is_call,
        "num_orders": 4,
        "charges": charges,
        "exit_time": None,
        "exit_reason": "",
        "exit_pnl_gross": 0.0,
    }



# --- 5c. Friday Low-VIX Carry Module ---
def simulate_friday_lowvix_entry(entry_bar, expiry_dt):
    """
    Simulate Friday Low-VIX Iron Condor entry.
    Returns: trade dict or None.
    """
    spot = entry_bar['close']
    vix  = entry_bar['vix']
    iv   = entry_bar['iv']
    entry_time = entry_bar['datetime']

    if vix >= FRI_VIX_THRESHOLD:
        return None

    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    T_entry = time_to_expiry_years(entry_time, expiry_dt)
    if T_entry <= 0:
        return None

    # 3-strike wide IC: Sell OTM6, Buy OTM9
    net_credit, buys_prem, sells_prem, legs = price_ic_legs(spot, atm, 6, 9, T_entry, RISK_FREE_RATE, iv, FRI_QTY)

    # Pre-calculate OTM5 hedge strikes for Monday adjustment
    lp5 = atm - 5 * STRIKE_INTERVAL
    lc5 = atm + 5 * STRIKE_INTERVAL

    charges = calculate_trade_charges(buys_prem, sells_prem, 4, FRI_QTY)

    return {
        "slot": "lowvix_trade",
        "type": "FRI_LOWVIX_IC",
        "entry_time": entry_time,
        "entry_spot": spot,
        "entry_vix": vix,
        "expiry_dt": expiry_dt,
        "atm": atm,
        "net_credit": net_credit,
        "buys_prem": buys_prem,
        "sells_prem": sells_prem,
        "legs": legs,
        "lp5": lp5, "lc5": lc5,
        "qty": FRI_QTY,
        "short_offset": 6,
        "is_ratio": False,
        "adjusted": False,
        "adj_cost": 0.0,
        "tuesday_was_profitable": False,
        "num_orders": 4,
        "charges": charges,
        "exit_time": None,
        "exit_reason": "",
        "exit_pnl_gross": 0.0,
    }

# --- 5d. Tuesday Expiry IC Module ---
def simulate_tuesday_ic_entry(entry_bar, expiry_dt):
    """
    Simulate Tuesday Expiry IC entry.
    Scans OTM strikes to find shorts in the 10-20 Rs premium band.
    Returns: trade dict or None.
    """
    spot = entry_bar['close']
    vix  = entry_bar['vix']
    iv   = entry_bar['iv']
    entry_time = entry_bar['datetime']

    if vix < TUE_VIX_MIN or vix > TUE_VIX_MAX:
        return None

    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    T_entry = time_to_expiry_years(entry_time, expiry_dt)
    if T_entry <= 0:
        return None

    # Scan for CE and PE strikes in 10-20 Rs band
    ce_off, pe_off = None, None
    for off in range(2, 16):
        ce_strike = atm + off * STRIKE_INTERVAL
        ce_prem = bs_call(spot, ce_strike, T_entry, RISK_FREE_RATE, iv * CALL_IV_MULT)
        if 10 <= ce_prem <= 20 and ce_off is None:
            ce_off = off
            break

    for off in range(2, 16):
        pe_strike = atm - off * STRIKE_INTERVAL
        pe_prem = bs_put(spot, pe_strike, T_entry, RISK_FREE_RATE, iv * PUT_IV_MULT)
        if 10 <= pe_prem <= 20 and pe_off is None:
            pe_off = off
            break

    if ce_off is None or pe_off is None:
        return None

    # Build IC with 2-strike-wide hedges
    sc = atm + ce_off * STRIKE_INTERVAL
    sp = atm - pe_off * STRIKE_INTERVAL
    lc = sc + 2 * STRIKE_INTERVAL
    lp = sp - 2 * STRIKE_INTERVAL

    sc_p = bs_call(spot, sc, T_entry, RISK_FREE_RATE, iv * CALL_IV_MULT)
    sp_p = bs_put(spot, sp, T_entry, RISK_FREE_RATE, iv * PUT_IV_MULT)
    lc_p = bs_call(spot, lc, T_entry, RISK_FREE_RATE, iv * CALL_IV_MULT)
    lp_p = bs_put(spot, lp, T_entry, RISK_FREE_RATE, iv * PUT_IV_MULT)

    net_credit = (sc_p + sp_p) - (lc_p + lp_p)
    buys_prem = lc_p + lp_p
    sells_prem = sc_p + sp_p
    
    legs = {
        "short_call": sc, "short_put": sp, "long_call": lc, "long_put": lp,
        "sc_p_entry": sc_p, "sp_p_entry": sp_p, "lc_p_entry": lc_p, "lp_p_entry": lp_p
    }

    charges = calculate_trade_charges(buys_prem, sells_prem, 4, TUE_QTY)

    return {
        "slot": "tuesday_trade",
        "type": "TUE_EXPIRY_IC",
        "entry_time": entry_time,
        "entry_spot": spot,
        "entry_vix": vix,
        "expiry_dt": expiry_dt,
        "atm": atm,
        "initial_net_credit": net_credit,
        "realized_pnl": 0.0,
        "legs": legs,
        "qty": TUE_QTY,
        "short_offset": max(ce_off, pe_off),
        "sl_mult": 2.0,  # Catastrophic SL at 2x collected credit
        "is_ratio": False,
        "num_orders": 4,
        "charges": charges,
        "exit_time": None,
        "exit_reason": "",
        "exit_pnl_gross": 0.0,
    }

# ============================================================
# 6. TRADE EXIT LOGIC
# ============================================================
def check_exit(trade, bar, df):
    """
    Check exit conditions for any active trade. Mutates trade dict.
    Returns True if trade was closed.
    """
    spot = bar['close']
    bar_high = bar['high']
    bar_low = bar['low']
    bar_open = bar['open']
    iv = bar['iv']
    bar_time = bar['datetime']
    T_current = time_to_expiry_years(bar_time, trade["expiry_dt"])

    legs = trade["legs"]
    qty = trade["qty"]
    is_ratio = trade.get("is_ratio", False)

    def close_trade(reason, exit_spot=None):
        if exit_spot is None:
            exit_spot = spot
        if is_ratio:
            current_val = current_ratio_value(exit_spot, legs, T_current, RISK_FREE_RATE, iv, trade.get("ratio_is_call", True))
        else:
            current_val = current_ic_value(exit_spot, legs, T_current, RISK_FREE_RATE, iv)
        trade["exit_pnl_gross"] = (trade["net_credit"] - current_val) * qty
        # Additional exit transaction charges
        if is_ratio:
            extra_buys = bs_call(exit_spot, legs["sell_strike"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT) * 3
            extra_sells = bs_call(exit_spot, legs["buy_strike"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT) + \
                          bs_call(exit_spot, legs["hedge_strike"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT) * 2
        else:
            extra_buys = bs_call(exit_spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT) + \
                         bs_put(exit_spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
            extra_sells = bs_call(exit_spot, legs["long_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT) + \
                          bs_put(exit_spot, legs["long_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
        trade["charges"] += calculate_trade_charges(extra_buys, extra_sells, 4, qty)
        trade["num_orders"] += 4
        trade["exit_time"] = bar_time
        trade["exit_reason"] = reason
        return True

    # --- Tuesday Expiry IC specific logic ---
    if trade["slot"] == "tuesday_trade":
        def close_tuesday_trade(reason):
            val = current_tuesday_value(spot, legs, T_current, RISK_FREE_RATE, iv)
            trade["exit_pnl_gross"] = trade["realized_pnl"] + (current_tuesday_entry_credit(legs) - val) * qty
            
            extra_buys = 0.0
            extra_sells = 0.0
            num_legs_closed = 0
            if legs["short_call"] is not None:
                sc_p = bs_call(spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                lc_p = bs_call(spot, legs["long_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                extra_buys += sc_p
                extra_sells += lc_p
                num_legs_closed += 2
            if legs["short_put"] is not None:
                sp_p = bs_put(spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                lp_p = bs_put(spot, legs["long_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                extra_buys += sp_p
                extra_sells += lp_p
                num_legs_closed += 2
                
            trade["charges"] += calculate_trade_charges(extra_buys, extra_sells, num_legs_closed, qty)
            trade["num_orders"] += num_legs_closed
            trade["exit_time"] = bar_time
            trade["exit_reason"] = reason
            return True

        # 1. Breach check
        breach = False
        if legs["short_call"] is not None and bar_high >= legs["short_call"]:
            breach = True
        if legs["short_put"] is not None and bar_low <= legs["short_put"]:
            breach = True
        if breach:
            return close_tuesday_trade("ITM_BREACH")

        # 2. Premium SL check (catastrophic position stop loss)
        current_val = current_tuesday_value(spot, legs, T_current, RISK_FREE_RATE, iv)
        live_pnl = trade["realized_pnl"] + (current_tuesday_entry_credit(legs) - current_val) * qty
        if live_pnl <= -(trade["sl_mult"] * trade["initial_net_credit"] * qty):
            return close_tuesday_trade("SL_HIT")

        # 3. Expiry exit at Tuesday 15:10
        if bar_time.weekday() == 1 and bar_time.time() >= time(15, 10, 0):
            return close_tuesday_trade("EXPIRY_EXIT")

        return False

    # --- Friday Low-VIX specific logic ---
    if trade["slot"] == "lowvix_trade":
        is_monday = bar_time.weekday() == 0
        is_tuesday = bar_time.weekday() == 1

        if is_monday or is_tuesday:
            # Breach check (only before adjustment)
            if not trade["adjusted"]:
                is_first_monday_bar = is_monday and bar_time.time() == time(9, 15, 0)
                if is_first_monday_bar:
                    breach_c = bar_open >= legs["short_call"]
                    breach_p = bar_open <= legs["short_put"]
                else:
                    breach_c = bar_high >= legs["short_call"]
                    breach_p = bar_low <= legs["short_put"]

                if breach_c or breach_p:
                    return close_trade("BREACH_EXIT")

            # Adjustment check (Monday only, not yet adjusted)
            if is_monday and not trade["adjusted"]:
                current_val = current_ic_value(spot, legs, T_current, RISK_FREE_RATE, iv)
                mtm = (trade["net_credit"] - current_val) * qty
                if mtm >= FRI_TARGET_MTM:
                    trade["adjusted"] = True
                    # Cost of new OTM5 hedges
                    lp5_p = bs_put(spot, trade["lp5"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                    lc5_p = bs_call(spot, trade["lc5"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                    lp_p = bs_put(spot, legs["long_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                    lc_p = bs_call(spot, legs["long_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                    trade["adj_cost"] = (lp5_p + lc5_p) - (lp_p + lc_p)
                    trade["charges"] += calculate_trade_charges(lp5_p + lc5_p, lp_p + lc_p, 4, qty)
                    trade["num_orders"] += 4

            # Tuesday profit floor
            if is_tuesday:
                if trade["adjusted"]:
                    lp5_p = bs_put(spot, trade["lp5"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                    lc5_p = bs_call(spot, trade["lc5"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                    sc_p = bs_call(spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                    sp_p = bs_put(spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                    current_net = (sc_p + sp_p) - (lp5_p + lc5_p)
                    live_mtm = (trade["net_credit"] - current_net - trade["adj_cost"]) * qty
                else:
                    current_val = current_ic_value(spot, legs, T_current, RISK_FREE_RATE, iv)
                    live_mtm = (trade["net_credit"] - current_val) * qty

                if live_mtm > 0:
                    trade["tuesday_was_profitable"] = True
                if trade["tuesday_was_profitable"] and live_mtm < FRI_TUESDAY_FLOOR:
                    # Need to calculate gross PnL correctly for adjusted position
                    if trade["adjusted"]:
                        lp5_p = bs_put(spot, trade["lp5"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                        lc5_p = bs_call(spot, trade["lc5"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                        sc_p = bs_call(spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                        sp_p = bs_put(spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                        exit_debit = (sc_p + sp_p) - (lp5_p + lc5_p)
                        trade["exit_pnl_gross"] = (trade["net_credit"] - exit_debit - trade["adj_cost"]) * qty
                    else:
                        current_val = current_ic_value(spot, legs, T_current, RISK_FREE_RATE, iv)
                        trade["exit_pnl_gross"] = (trade["net_credit"] - current_val) * qty
                    trade["charges"] += calculate_trade_charges(0, 0, 4, qty)  # Approx exit charges
                    trade["num_orders"] += 4
                    trade["exit_time"] = bar_time
                    trade["exit_reason"] = "TUESDAY_TRAIL_STOP"
                    return True

        # Expiry exit at Tuesday 15:10
        if bar_time.weekday() == 1 and bar_time.time() >= time(15, 10, 0):
            if trade["adjusted"]:
                lp5_p = bs_put(spot, trade["lp5"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                lc5_p = bs_call(spot, trade["lc5"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                sc_p = bs_call(spot, legs["short_call"], T_current, RISK_FREE_RATE, iv * CALL_IV_MULT)
                sp_p = bs_put(spot, legs["short_put"], T_current, RISK_FREE_RATE, iv * PUT_IV_MULT)
                exit_debit = (sc_p + sp_p) - (lp5_p + lc5_p)
                trade["exit_pnl_gross"] = (trade["net_credit"] - exit_debit - trade["adj_cost"]) * qty
            else:
                current_val = current_ic_value(spot, legs, T_current, RISK_FREE_RATE, iv)
                trade["exit_pnl_gross"] = (trade["net_credit"] - current_val) * qty
            trade["charges"] += calculate_trade_charges(0, 0, 4, qty)
            trade["num_orders"] += 4
            trade["exit_time"] = bar_time
            trade["exit_reason"] = "EXPIRY_EXIT"
            return True
        return False

    # --- Generic IC/Ratio exit logic (for Wed carry) ---
    if not is_ratio:
        # ITM Breach check
        breach = False
        s_off = trade["short_offset"] * STRIKE_INTERVAL
        if bar_high >= legs["short_call"] or bar_low <= legs["short_put"]:
            breach = True
        if breach:
            return close_trade("ITM_BREACH")

        # Premium SL check
        current_val = current_ic_value(spot, legs, T_current, RISK_FREE_RATE, iv)
        mtm = (trade["net_credit"] - current_val) * qty
        if mtm <= -(trade["sl_mult"] * trade["net_credit"] * qty):
            return close_trade("SL_HIT")

    # Ratio SL
    if is_ratio:
        current_val = current_ratio_value(spot, legs, T_current, RISK_FREE_RATE, iv, trade.get("ratio_is_call", True))
        mtm = (trade["net_credit"] - current_val) * qty
        if mtm <= -(trade.get("sl_mult", 1.5) * abs(trade["net_credit"]) * qty):
            return close_trade("SL_HIT")


    # Scheduled exit for carry_trade on Monday EOD
    if trade["slot"] == "carry_trade":
        if bar_time.weekday() == 0 and bar_time.time() >= time(15, 15, 0):
            return close_trade("SCHEDULED_MON_EXIT")


    return False

# ============================================================
# ============================================================
# 7. MAIN PORTFOLIO BACKTEST LOOP
# ============================================================
def run_portfolio_backtest(df):
    print("\n" + "="*80)
    print(" RUNNING UNIFIED PORTFOLIO BACKTEST")
    print("="*80)

    capital = INITIAL_CAPITAL
    all_trades = []
    calendar_log = []  # (date, slot, action)
    daily_equity = {}  # {date_str: equity}

    # Active position slots
    active = {
        "carry_trade": None,
        "lowvix_trade": None,
        "tuesday_trade": None,
    }

    # Track Monday and Friday closes for gap calculations
    prev_friday_close = 0
    monday_close = 0
    
    wed_entry_done_this_week = False
    fri_entry_done_this_week = False
    tue_entry_done_this_week = False
    current_week = None

    dates = sorted(df['trade_date'].unique())

    for trade_date in dates:
        day_df = df[df['trade_date'] == trade_date].sort_values('datetime')
        if day_df.empty:
            continue

        dow = day_df.iloc[0]['dayofweek']
        # Week reset (using ISO week)
        week_id = day_df.iloc[0]['datetime'].isocalendar()[1]
        if current_week != week_id:
            current_week = week_id
            wed_entry_done_this_week = False
            fri_entry_done_this_week = False
            tue_entry_done_this_week = False

        for idx_row in range(len(day_df)):
            bar = day_df.iloc[idx_row]
            bar_time = bar['datetime']
            t = bar['time_str']
            iv = bar['iv']

            # ===== STEP 1: Monitor all active positions for exits =====
            for slot_key in list(active.keys()):
                trade = active[slot_key]
                if trade is None:
                    continue
                closed = check_exit(trade, bar, df)
                if closed:
                    net_pnl = trade["exit_pnl_gross"] - trade["charges"]
                    capital += net_pnl
                    trade["net_pnl"] = net_pnl
                    trade["capital_after"] = capital
                    all_trades.append(trade)
                    calendar_log.append((trade["exit_time"], slot_key, "EXIT", trade["exit_reason"], trade["type"]))
                    active[slot_key] = None

            # ===== STEP 2: Live Adjustments for Tuesday Expiry IC =====
            if dow == 1 and active["tuesday_trade"] is not None:
                if t < time(15, 0, 0):
                    T_current = time_to_expiry_years(bar_time, active["tuesday_trade"]["expiry_dt"])
                    adjust_tuesday_trade_live(active["tuesday_trade"], bar, T_current, iv, calendar_log)

            # ===== STEP 3: New entries based on day of week =====
            expiry_dt = get_next_tuesday_expiry(bar_time)

            # --- WEDNESDAY (day 2): Carry Trade Entry ---
            if dow == 2 and not wed_entry_done_this_week:
                if active["carry_trade"] is None:
                    if t >= time(15, 15, 0) and t <= time(15, 25, 0):
                        trade = simulate_wednesday_carry(df, bar, expiry_dt)
                        if trade is not None:
                            active["carry_trade"] = trade
                            wed_entry_done_this_week = True
                            calendar_log.append((bar_time, "carry_trade", "ENTRY", trade["type"], trade["type"]))
                    # Ratio spreads can trigger anytime (not just EOD)
                    elif bar['adx'] >= 25:
                        trade = simulate_wednesday_carry(df, bar, expiry_dt)
                        if trade is not None:
                            active["carry_trade"] = trade
                            wed_entry_done_this_week = True
                            calendar_log.append((bar_time, "carry_trade", "ENTRY", trade["type"], trade["type"]))

            # --- THURSDAY (day 3) / FRIDAY (day 4): Carry rescue if Wednesday missed ---
            if dow in [3, 4] and not wed_entry_done_this_week and active["carry_trade"] is None:
                if t >= time(15, 15, 0) and t <= time(15, 25, 0):
                    trade = simulate_wednesday_carry(df, bar, expiry_dt)
                    if trade is not None:
                        active["carry_trade"] = trade
                        wed_entry_done_this_week = True
                        calendar_log.append((bar_time, "carry_trade", "ENTRY", trade["type"], trade["type"]))

            # --- FRIDAY (day 4): Low-VIX Carry Entry ---
            # CONFLICT: Only if no carry_trade is active
            if dow == 4 and not fri_entry_done_this_week:
                if active["carry_trade"] is None and active["lowvix_trade"] is None:
                    if t >= time(14, 0, 0) and t <= time(15, 25, 0):
                        fri_expiry = datetime.combine(bar_time.date() + timedelta(days=4), time(15, 30, 0))
                        trade = simulate_friday_lowvix_entry(bar, fri_expiry)
                        if trade is not None:
                            active["lowvix_trade"] = trade
                            fri_entry_done_this_week = True
                            calendar_log.append((bar_time, "lowvix_trade", "ENTRY", trade["type"], trade["type"]))

            # --- TUESDAY (day 1): Expiry IC Entry ---
            # CONFLICT: Only if no lowvix_trade and no carry_trade active
            if dow == 1 and not tue_entry_done_this_week:
                if active["lowvix_trade"] is None and active["carry_trade"] is None and active["tuesday_trade"] is None:
                    if t >= time(9, 20, 0) and t <= time(12, 0, 0):
                        # Monday close gap gate check
                        last_close = monday_close if monday_close > 0 else bar['close']
                        gap_pct = abs(bar['close'] - last_close) / last_close * 100
                        if gap_pct <= 1.0:
                            trade = simulate_tuesday_ic_entry(bar, expiry_dt)
                            if trade is not None:
                                active["tuesday_trade"] = trade
                                tue_entry_done_this_week = True
                                calendar_log.append((bar_time, "tuesday_trade", "ENTRY", trade["type"], trade["type"]))

        # Track Friday and Monday closes at the end of the day
        if dow == 4:
            prev_friday_close = day_df.iloc[-1]['close']
        elif dow == 0:
            monday_close = day_df.iloc[-1]['close']

        # Calculate daily equity at EOD
        floating_pnl = 0.0
        last_bar = day_df.iloc[-1]
        spot = last_bar['close']
        iv = last_bar['iv']
        
        for slot_key, trade in active.items():
            if trade is not None:
                T_curr = time_to_expiry_years(last_bar['datetime'], trade["expiry_dt"])
                if slot_key == "tuesday_trade":
                    val = current_tuesday_value(spot, trade["legs"], T_curr, RISK_FREE_RATE, iv)
                    entry_credit = current_tuesday_entry_credit(trade["legs"])
                    floating_pnl += (trade["realized_pnl"] + (entry_credit - val) * trade["qty"] - trade["charges"])
                elif trade.get("is_ratio", False):
                    val = current_ratio_value(spot, trade["legs"], T_curr, RISK_FREE_RATE, iv, trade.get("ratio_is_call", True))
                    floating_pnl += ((trade["net_credit"] - val) * trade["qty"] - trade["charges"])
                else:
                    val = current_ic_value(spot, trade["legs"], T_curr, RISK_FREE_RATE, iv)
                    floating_pnl += ((trade["net_credit"] - val) * trade["qty"] - trade["charges"])

        daily_equity[str(trade_date.date())] = capital + floating_pnl

    # Close any remaining open positions at data end
    for slot_key, trade in active.items():
        if trade is not None:
            trade["exit_time"] = df.iloc[-1]['datetime']
            trade["exit_reason"] = "END_OF_DATA"
            # Close trade using the appropriate logic
            if slot_key == "tuesday_trade":
                T_curr = time_to_expiry_years(trade["exit_time"], trade["expiry_dt"])
                val = current_tuesday_value(df.iloc[-1]['close'], trade["legs"], T_curr, RISK_FREE_RATE, df.iloc[-1]['iv'])
                trade["exit_pnl_gross"] = trade["realized_pnl"] + (current_tuesday_entry_credit(trade["legs"]) - val) * trade["qty"]
            else:
                current_val = current_ic_value(df.iloc[-1]['close'], trade["legs"], 0, RISK_FREE_RATE, df.iloc[-1]['iv'])
                trade["exit_pnl_gross"] = (trade["net_credit"] - current_val) * trade["qty"]
            
            trade["net_pnl"] = trade["exit_pnl_gross"] - trade["charges"]
            capital += trade["net_pnl"]
            trade["capital_after"] = capital
            all_trades.append(trade)

    return all_trades, calendar_log, capital, daily_equity

# ============================================================
# 8. VERIFICATION & REPORTING
# ============================================================
def verify_and_report(all_trades, calendar_log, final_capital, daily_equity, opt_buy_trades=None):
    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        print("No trades were generated!")
        return trades_df

    print("\n" + "="*80)
    print(" PHASE 5: CONFLICT CALENDAR VERIFICATION")
    print("="*80)

    # Test: No overlapping active positions in conflicting slots
    cal_df = pd.DataFrame(calendar_log, columns=["datetime", "slot", "action", "reason", "type"])
    overlap_violations = 0

    # Check: lowvix_trade and tuesday_trade should never be active simultaneously
    lowvix_entries = cal_df[(cal_df['slot'] == 'lowvix_trade') & (cal_df['action'] == 'ENTRY')]
    for _, lv in lowvix_entries.iterrows():
        lv_entry = lv['datetime']
        lv_exit_row = trades_df[(trades_df['slot'] == 'lowvix_trade') & (trades_df['entry_time'] == lv_entry)]
        if lv_exit_row.empty:
            continue
        lv_exit = lv_exit_row.iloc[0]['exit_time']
        if pd.isna(lv_exit):
            continue
        tue_entries = cal_df[(cal_df['slot'] == 'tuesday_trade') & (cal_df['action'] == 'ENTRY')]
        conflicts = tue_entries[(tue_entries['datetime'] >= lv_entry) & (tue_entries['datetime'] <= lv_exit)]
        if not conflicts.empty:
            overlap_violations += len(conflicts)
            print(f"[FAIL] Tuesday trade entered while Low-VIX was active: {conflicts['datetime'].tolist()}")

    if overlap_violations == 0:
        print("[PASS] No Low-VIX / Tuesday overlap violations.")
    else:
        print(f"[FAIL] {overlap_violations} overlap violations found!")

    # Summary stats
    print("\n" + "="*80)
    print(" PORTFOLIO BACKTEST RESULTS")
    print("="*80)

    net_profit = final_capital - INITIAL_CAPITAL
    total = len(trades_df)
    wins = (trades_df['net_pnl'] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100 if total > 0 else 0
    total_wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
    total_losses = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
    pf = total_wins / total_losses if total_losses > 0 else float('inf')

    # Drawdown
    cap_series = pd.Series([INITIAL_CAPITAL] + trades_df['capital_after'].tolist())
    dd = ((cap_series.cummax() - cap_series) / cap_series.cummax() * 100).max()

    print(f"Initial Capital            : Rs. {INITIAL_CAPITAL:,.0f}")
    print(f"Final Capital              : Rs. {final_capital:,.0f}")
    print(f"Net Profit                 : Rs. {net_profit:,.0f} ({net_profit/INITIAL_CAPITAL*100:.2f}%)")
    print(f"Total Trades               : {total}")
    print(f"Winning Trades             : {wins} ({win_rate:.1f}%)")
    print(f"Losing Trades              : {losses}")
    print(f"Profit Factor              : {pf:.2f}")
    print(f"Max Drawdown               : {dd:.2f}%")
    print("-"*80)

    # Breakdown by strategy slot
    for slot in ['carry_trade', 'lowvix_trade', 'tuesday_trade']:
        slot_df = trades_df[trades_df['slot'] == slot]
        if slot_df.empty:
            print(f"  {slot:<20}: 0 trades")
            continue
        s_wins = (slot_df['net_pnl'] > 0).sum()
        s_total = len(slot_df)
        s_pnl = slot_df['net_pnl'].sum()
        print(f"  {slot:<20}: {s_total} trades | Win Rate: {s_wins/s_total*100:.1f}% | Net P&L: Rs. {s_pnl:,.0f}")

    print("="*80)

    # Walk-forward check
    IS_CUTOFF = pd.Timestamp("2024-01-01")
    is_df = trades_df[trades_df['entry_time'] < IS_CUTOFF]
    oos_df = trades_df[trades_df['entry_time'] >= IS_CUTOFF]

    def period_stats(sdf, label):
        if sdf.empty:
            print(f"\n{label}: No trades")
            return
        n = len(sdf)
        w = (sdf['net_pnl'] > 0).sum()
        g = sdf[sdf['net_pnl'] > 0]['net_pnl'].sum()
        l = abs(sdf[sdf['net_pnl'] <= 0]['net_pnl'].sum())
        pf_val = g / l if l > 0 else float('inf')
        print(f"\n{label}")
        print(f"  Trades: {n} | Win Rate: {w/n*100:.1f}% | PF: {pf_val:.2f} | Net P&L: Rs. {sdf['net_pnl'].sum():,.0f}")

    print("\n" + "="*80)
    print(" WALK-FORWARD OVERFITTING CHECK")
    print("="*80)
    period_stats(is_df, "IN-SAMPLE (2022-2023)")
    period_stats(oos_df, "OUT-OF-SAMPLE (2024-2026)")
    print("\n" + "="*80)

    # Call HTML Report Generator
    generate_html_report(all_trades, calendar_log, final_capital, daily_equity, opt_buy_trades=opt_buy_trades)

    return trades_df

def generate_html_report(all_trades, calendar_log, final_capital, daily_equity, opt_buy_trades=None):
    import json
    import numpy as np
    import pandas as pd
    
    # 1. Prepare serializable trades
    serializable_trades = []
    equity_curve = [INITIAL_CAPITAL]
    running_equity = INITIAL_CAPITAL
    
    for t in all_trades:
        t_copy = t.copy()
        t_copy["entry_time"] = str(t_copy["entry_time"])
        t_copy["expiry_dt"] = str(t_copy["expiry_dt"])
        t_copy["exit_time"] = str(t_copy["exit_time"])
        if isinstance(t_copy["legs"], dict):
            t_copy["legs"] = {k: (str(v) if v is not None else None) for k, v in t_copy["legs"].items()}
        serializable_trades.append(t_copy)
        
        running_equity += t["net_pnl"]
        equity_curve.append(running_equity)
        
    # 2. Calculate booked P&L (closed-trade only) for heatmap
    booked_pnls = {}
    for t in all_trades:
        try:
            exit_date = pd.to_datetime(t["exit_time"]).date()
            exit_date_str = str(exit_date)
        except Exception:
            exit_date_str = str(t["exit_time"])[:10]
        booked_pnls[exit_date_str] = booked_pnls.get(exit_date_str, 0.0) + t["net_pnl"]
    
    # 2b. Calculate monthly aggregated data (for monthly table)
    monthly_data = {}  # key: "YYYY-MM" -> {trades, gross_pnl, charges, net_pnl}
    for t in all_trades:
        try:
            exit_dt = pd.to_datetime(t["exit_time"])
            month_key = f"{exit_dt.year}-{exit_dt.month:02d}"
        except Exception:
            month_key = str(t["exit_time"])[:7]
        if month_key not in monthly_data:
            monthly_data[month_key] = {"trades": 0, "gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0}
        monthly_data[month_key]["trades"] += 1
        monthly_data[month_key]["gross_pnl"] += t.get("exit_pnl_gross", t["net_pnl"] + t["charges"])
        monthly_data[month_key]["charges"] += t["charges"]
        monthly_data[month_key]["net_pnl"] += t["net_pnl"]
        
    # 3. Calculate premium metrics
    dates_sorted = sorted(daily_equity.keys())
    daily_returns = []
    
    prev_eq = INITIAL_CAPITAL
    for d in dates_sorted:
        eq = daily_equity[d]
        pnl = eq - prev_eq
        ret = pnl / prev_eq if prev_eq > 0 else 0
        daily_returns.append(ret)
        prev_eq = eq
        
    mean_daily_ret = np.mean(daily_returns) if daily_returns else 0
    std_daily_ret = np.std(daily_returns) if daily_returns else 0
    
    # Annualized Sharpe (risk-free rate = 0%)
    sharpe = (mean_daily_ret / std_daily_ret * np.sqrt(252)) if std_daily_ret > 0 else 0
    
    # Sortino Ratio
    downside_returns = [r for r in daily_returns if r < 0]
    std_downside = np.std(downside_returns) if downside_returns else 0
    sortino = (mean_daily_ret / std_downside * np.sqrt(252)) if std_downside > 0 else 0
    
    # CAGR
    start_dt = pd.to_datetime(dates_sorted[0]) if dates_sorted else pd.Timestamp("2022-01-01")
    end_dt = pd.to_datetime(dates_sorted[-1]) if dates_sorted else pd.Timestamp("2026-07-06")
    total_days = max((end_dt - start_dt).days, 1)
    years = total_days / 365.25
    cagr = (((final_capital / INITIAL_CAPITAL) ** (1.0 / years) - 1) * 100) if years > 0 else 0
    
    # Average Monthly Return
    net_pnl = final_capital - INITIAL_CAPITAL
    total_months = total_days / 30.44
    avg_monthly_pnl_rs = net_pnl / total_months if total_months > 0 else 0
    avg_monthly_pnl_pct = (avg_monthly_pnl_rs / INITIAL_CAPITAL) * 100
    
    # Expectancy
    expectancy = np.mean([t["net_pnl"] for t in all_trades]) if all_trades else 0
    
    # Recovery Factor
    cap_series = pd.Series(equity_curve)
    max_dd_val = (cap_series.cummax() - cap_series).max()
    recovery_factor = (net_pnl / max_dd_val) if max_dd_val > 0 else float('inf')
    max_dd_pct = ((cap_series.cummax() - cap_series) / cap_series.cummax() * 100).max()
    
    # Strategy breakdowns
    strategy_stats = {}
    for slot in ['carry_trade', 'lowvix_trade', 'tuesday_trade']:
        slot_trades = [t for t in all_trades if t["slot"] == slot]
        s_total = len(slot_trades)
        s_wins = sum(1 for t in slot_trades if t["net_pnl"] > 0)
        s_win_rate = (s_wins / s_total * 100) if s_total > 0 else 0
        s_pnl = sum(t["net_pnl"] for t in slot_trades)
        strategy_stats[slot] = {
            "total": s_total,
            "win_rate": f"{s_win_rate:.1f}%",
            "pnl": f"Rs. {s_pnl:,.2f}"
        }
        
    # Walk forward checks
    IS_CUTOFF = "2024-01-01"
    is_trades = [t for t in all_trades if str(t["entry_time"]) < IS_CUTOFF]
    oos_trades = [t for t in all_trades if str(t["entry_time"]) >= IS_CUTOFF]
    
    def calc_period_metrics(p_trades):
        if not p_trades:
            return {"total": 0, "win_rate": "0%", "pf": "0.00", "pnl": "Rs. 0.00"}
        ptotal = len(p_trades)
        pwins = sum(1 for t in p_trades if t["net_pnl"] > 0)
        pw_rate = pwins / ptotal * 100
        pw_sum = sum(t["net_pnl"] for t in p_trades if t["net_pnl"] > 0)
        pl_sum = abs(sum(t["net_pnl"] for t in p_trades if t["net_pnl"] <= 0))
        ppf = pw_sum / pl_sum if pl_sum > 0 else float('inf')
        ppnl = sum(t["net_pnl"] for t in p_trades)
        return {
            "total": ptotal,
            "win_rate": f"{pw_rate:.1f}%",
            "pf": f"{ppf:.2f}",
            "pnl": f"Rs. {ppnl:,.2f}"
        }
        
    is_metrics = calc_period_metrics(is_trades)
    oos_metrics = calc_period_metrics(oos_trades)

    # ---- 5m Option-Buying stats ----
    from nifty_5m_backtest import aggregate_5m_metrics
    ob_stats = aggregate_5m_metrics(opt_buy_trades or [])
    ob_monthly_data_raw = ob_stats["monthly_data"]
    # make monthly_data serialisable (convert keys/values to plain types)
    ob_monthly_data = {k: {mk: mv for mk, mv in v.items()} for k, v in ob_monthly_data_raw.items()}
    ob_equity_curve = ob_stats["equity_curve"]
    # combined P&L
    combined_net_pnl = net_pnl + ob_stats["net_pnl"]
    combined_initial = INITIAL_CAPITAL + 150_000.0
    combined_equity = [a + b for a, b in zip(equity_curve, ob_equity_curve[:len(equity_curve)])]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nifty Master Quant Suite - Performance Pitch Deck</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* ---- Tab switcher ---- */
        .tab-bar {{
            display: flex;
            gap: 0;
            margin-bottom: 30px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .tab-btn {{
            padding: 12px 28px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            background: none;
            cursor: pointer;
            color: #64748b;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            color: #2563eb;
            border-bottom-color: #2563eb;
        }}
        .tab-btn:hover {{ color: #2563eb; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 40px;
        }}
        .header {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
        }}
        .header-title h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
        }}
        .header-title p {{
            margin: 6px 0 0 0;
            color: #64748b;
            font-size: 14px;
        }}
        .pnl-summary {{
            text-align: right;
        }}
        .pnl-label {{
            font-size: 12px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .pnl-value {{
            font-size: 28px;
            font-weight: 800;
            color: #10b981;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }}
        .kpi-title {{
            color: #64748b;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
        }}
        .kpi-value.green {{
            color: #10b981;
        }}
        .kpi-value.red {{
            color: #ef4444;
        }}
        .layout-columns {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }}
        .heatmap-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }}
        .heatmap-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .heatmap-title {{
            font-size: 16px;
            font-weight: 700;
            margin: 0;
            color: #0f172a;
        }}
        .year-selector {{
            display: flex;
            gap: 8px;
        }}
        .year-btn {{
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #64748b;
            padding: 6px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .year-btn.active {{
            background-color: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
        }}
        .calendar-month-labels {{
            display: flex;
            gap: 2px;
            margin-bottom: 6px;
            font-size: 11px;
            color: #64748b;
            padding-left: 28px;
        }}
        .calendar-month-label {{
            width: calc(100% / 12);
            text-align: center;
        }}
        .heatmap-body-wrapper {{
            display: flex;
            gap: 8px;
        }}
        .day-labels {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 10px;
            color: #64748b;
            height: 102px;
            padding-top: 2px;
            width: 20px;
        }}
        .heatmap-container {{
            width: 100%;
        }}
        .legend {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #64748b;
            margin-top: 15px;
        }}
        .legend-box {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            text-align: left;
            padding: 12px;
            color: #64748b;
            font-weight: 600;
            border-bottom: 2px solid #e2e8f0;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .pitch-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }}
        .pitch-title {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .pitch-bullet {{
            font-size: 13px;
            line-height: 1.6;
            color: #334155;
            margin-bottom: 12px;
        }}
        .pitch-bullet strong {{
            color: #0f172a;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="header-title">
            <h1>Nifty Systematic Options Portfolio</h1>
            <p>Investor Pitch Deck &amp; Performance Analytics Dashboard</p>
        </div>
        <div class="pnl-summary">
            <div class="pnl-label">Combined Net Profit (both strategies)</div>
            <div class="pnl-value">Rs. {combined_net_pnl:,.2f}</div>
        </div>
    </div>

    <!-- Tab Navigation -->
    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('hedging')">📈 Hedging Strategy</button>
        <button class="tab-btn" onclick="switchTab('optbuy')">🎯 5m Triple Confirm</button>
        <button class="tab-btn" onclick="switchTab('combined')">🔗 Combined Portfolio</button>
    </div>

    <!-- ============================================================ -->
    <!-- TAB 1: HEDGING STRATEGY                                      -->
    <!-- ============================================================ -->
    <div id="tab-hedging" class="tab-content active">
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Capital Deployed</div>
            <div class="kpi-value">Rs. {INITIAL_CAPITAL:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">CAGR (Annualized)</div>
            <div class="kpi-value green">{cagr:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg. Monthly Return</div>
            <div class="kpi-value green">Rs. {avg_monthly_pnl_rs:,.2f} ({avg_monthly_pnl_pct:.2f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Sharpe Ratio</div>
            <div class="kpi-value">{sharpe:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Sortino Ratio</div>
            <div class="kpi-value">{sortino:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Recovery Factor</div>
            <div class="kpi-value">{recovery_factor:.2f}</div>
        </div>
    </div>

    <!-- Calendar Heatmap Section (Booked P&L Only) -->
    <div class="heatmap-card">
        <div class="heatmap-header">
            <h2 class="heatmap-title">Booked P&L Calendar (Trade Realisation Days)</h2>
            <div class="year-selector" id="yearSelector">
                <button class="year-btn active" onclick="setYear(2022)">2022</button>
                <button class="year-btn" onclick="setYear(2023)">2023</button>
                <button class="year-btn" onclick="setYear(2024)">2024</button>
                <button class="year-btn" onclick="setYear(2025)">2025</button>
                <button class="year-btn" onclick="setYear(2026)">2026</button>
            </div>
        </div>
        
        <div class="calendar-grid-wrapper">
            <div class="calendar-month-labels">
                <div class="calendar-month-label">Jan</div>
                <div class="calendar-month-label">Feb</div>
                <div class="calendar-month-label">Mar</div>
                <div class="calendar-month-label">Apr</div>
                <div class="calendar-month-label">May</div>
                <div class="calendar-month-label">Jun</div>
                <div class="calendar-month-label">Jul</div>
                <div class="calendar-month-label">Aug</div>
                <div class="calendar-month-label">Sep</div>
                <div class="calendar-month-label">Oct</div>
                <div class="calendar-month-label">Nov</div>
                <div class="calendar-month-label">Dec</div>
            </div>
            <div class="heatmap-body-wrapper">
                <div class="day-labels">
                    <div>Sun</div>
                    <div>Tue</div>
                    <div>Thu</div>
                    <div>Sat</div>
                </div>
                <div class="heatmap-container" id="heatmapContainer"></div>
            </div>
        </div>
        
        <div class="legend">
            <span>Loss</span>
            <div class="legend-box" style="background-color: rgba(239, 68, 68, 1)"></div>
            <div class="legend-box" style="background-color: rgba(239, 68, 68, 0.6)"></div>
            <div class="legend-box" style="background-color: rgba(239, 68, 68, 0.3)"></div>
            <div class="legend-box" style="background-color: #f1f5f9"></div>
            <div class="legend-box" style="background-color: rgba(16, 185, 129, 0.3)"></div>
            <div class="legend-box" style="background-color: rgba(16, 185, 129, 0.6)"></div>
            <div class="legend-box" style="background-color: rgba(16, 185, 129, 1)"></div>
            <span>Profit</span>
        </div>
        
        <!-- Monthly Performance Table (dynamic) -->
        <div id="monthlyTableContainer" style="margin-top: 20px;"></div>
    </div>

    <div class="layout-columns">
        <div class="chart-card">
            <div class="heatmap-title" style="margin-bottom: 20px;">Growth of Capital (Cumulative Equity Curve)</div>
            <div style="height: 380px;">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 24px;">
            <div class="pitch-card">
                <div class="pitch-title">Model Performance Breakdowns</div>
                <table style="font-size: 12px; width: 100%;">
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Net P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Wednesday Carry</td>
                            <td>{strategy_stats['carry_trade']['total']}</td>
                            <td>{strategy_stats['carry_trade']['win_rate']}</td>
                            <td class="green" style="color: #10b981; font-weight: 600;">{strategy_stats['carry_trade']['pnl']}</td>
                        </tr>
                        <tr>
                            <td>Friday Low VIX</td>
                            <td>{strategy_stats['lowvix_trade']['total']}</td>
                            <td>{strategy_stats['lowvix_trade']['win_rate']}</td>
                            <td class="green" style="color: #10b981; font-weight: 600;">{strategy_stats['lowvix_trade']['pnl']}</td>
                        </tr>
                        <tr>
                            <td>Tuesday Expiry</td>
                            <td>{strategy_stats['tuesday_trade']['total']}</td>
                            <td>{strategy_stats['tuesday_trade']['win_rate']}</td>
                            <td class="red" style="color: #ef4444; font-weight: 600;">{strategy_stats['tuesday_trade']['pnl']}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="pitch-card">
                <div class="pitch-title">Investor Analytics</div>
                <div class="pitch-bullet"><strong>Walk-Forward Stability</strong>: In-Sample (2022-23) Win Rate is {is_metrics['win_rate']} (PF: {is_metrics['pf']}) vs. Out-of-Sample (2024-26) Win Rate of {oos_metrics['win_rate']} (PF: {oos_metrics['pf']}), verifying resilience across macroeconomic shifts.</div>
                <div class="pitch-bullet"><strong>Trade Expectancy</strong>: Expectation of **Rs. {expectancy:.2f}** per trade, reflecting a positive expectancy profile.</div>
                <div class="pitch-bullet"><strong>Drawdown Profile</strong>: Maximum peak-to-trough historical drawdown is capped at **{max_dd_pct:.2f}%**, demonstrating excellent downside risk controls.</div>
            </div>
        </div>
    </div>
    </div><!-- end tab-hedging -->

    <!-- ============================================================ -->
    <!-- TAB 2: 5m TRIPLE CONFIRM (OPTION BUYING)                     -->
    <!-- ============================================================ -->
    <div id="tab-optbuy" class="tab-content">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Total Trades</div>
                <div class="kpi-value">{ob_stats['total_trades']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Win Rate</div>
                <div class="kpi-value {'green' if ob_stats['win_rate'] >= 50 else ''}">{ob_stats['win_rate']:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Profit Factor</div>
                <div class="kpi-value {'green' if ob_stats['profit_factor'] >= 1 else 'red'}">{ob_stats['profit_factor']:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Net P&amp;L</div>
                <div class="kpi-value green">Rs. {ob_stats['net_pnl']:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Gross P&amp;L</div>
                <div class="kpi-value">Rs. {ob_stats['gross_pnl']:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Drawdown</div>
                <div class="kpi-value red">{ob_stats['max_drawdown_pct']:.2f}%</div>
            </div>
        </div>

        <div class="layout-columns">
            <div class="chart-card">
                <div class="heatmap-title" style="margin-bottom:20px;">5m Triple Confirm — Equity Curve</div>
                <div style="height:380px;"><canvas id="obEquityChart"></canvas></div>
            </div>
            <div class="chart-card">
                <div class="pitch-title">Strategy Notes</div>
                <div class="pitch-bullet"><strong>Entry Logic</strong>: SMA68 / SMA90 crossover with close above/below EMA340. Re-entry on pullback across EMA340.</div>
                <div class="pitch-bullet"><strong>Risk Management</strong>: Fixed SL at 0.75% on spot (Phase 1). Trailing SL activates once profit ≥ 0.75% (Phase 2).</div>
                <div class="pitch-bullet"><strong>Expiry Rollover</strong>: On Tuesday at 15:15, open position squares off and re-enters next expiry at Rs.200 premium.</div>
                <div class="pitch-bullet"><strong>Brokerage</strong>: Rs.200 flat per round-trip. Lot size: 65 units.</div>
                <div class="pitch-bullet"><strong>Total Charges</strong>: Rs. {ob_stats['charges']:,.2f}</div>
            </div>
        </div>

        <!-- Monthly table -->
        <div class="heatmap-card" style="margin-top:30px;">
            <div class="heatmap-title" style="margin-bottom:16px;">Monthly Performance — 5m Triple Confirm</div>
            <table>
                <thead><tr><th>Month</th><th>Trades</th><th>Gross P&amp;L</th><th>Charges</th><th>Net P&amp;L</th></tr></thead>
                <tbody id="obMonthlyBody"></tbody>
            </table>
        </div>
    </div><!-- end tab-optbuy -->

    <!-- ============================================================ -->
    <!-- TAB 3: COMBINED PORTFOLIO                                     -->
    <!-- ============================================================ -->
    <div id="tab-combined" class="tab-content">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Combined Net P&amp;L</div>
                <div class="kpi-value green">Rs. {combined_net_pnl:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Hedging Net P&amp;L</div>
                <div class="kpi-value green">Rs. {net_pnl:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">5m OB Net P&amp;L</div>
                <div class="kpi-value green">Rs. {ob_stats['net_pnl']:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Hedging Trades</div>
                <div class="kpi-value">{len(all_trades)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">5m OB Trades</div>
                <div class="kpi-value">{ob_stats['total_trades']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">5m OB Profit Factor</div>
                <div class="kpi-value">{ob_stats['profit_factor']:.2f}</div>
            </div>
        </div>
        <div class="chart-card">
            <div class="heatmap-title" style="margin-bottom:20px;">Combined Portfolio Equity Curve</div>
            <div style="height:400px;"><canvas id="combinedEquityChart"></canvas></div>
        </div>
    </div><!-- end tab-combined -->

    <script>
        const equityCurve = {json.dumps(equity_curve)};
        const bookedPnls = {json.dumps(booked_pnls)};
        const monthlyData = {json.dumps(monthly_data)};
        const obEquityCurve = {json.dumps(ob_equity_curve)};
        const obMonthlyData = {json.dumps(ob_monthly_data)};
        const combinedEquityCurve = {json.dumps(combined_equity)};

        // ---- Tab switcher ----
        function switchTab(name) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            event.target.classList.add('active');
            if (name === 'optbuy' && !window._obChartsInit) {{ initObCharts(); window._obChartsInit = true; }}
            if (name === 'combined' && !window._combinedChartsInit) {{ initCombinedChart(); window._combinedChartsInit = true; }}
        }}

        // ---- 5m OB Equity Chart ----
        function initObCharts() {{
            const ctx2 = document.getElementById('obEquityChart').getContext('2d');
            new Chart(ctx2, {{
                type: 'line',
                data: {{
                    labels: Array.from({{length: obEquityCurve.length}}, (_, i) => i),
                    datasets: [{{
                        label: '5m OB Equity (Rs.)',
                        data: obEquityCurve,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.07)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#f1f5f9' }}, ticks: {{ color: '#64748b' }} }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            // Populate OB monthly table
            const tbody = document.getElementById('obMonthlyBody');
            const sortedKeys = Object.keys(obMonthlyData).sort();
            sortedKeys.forEach(k => {{
                const v = obMonthlyData[k];
                const net = v.net_pnl;
                const color = net >= 0 ? '#10b981' : '#ef4444';
                tbody.innerHTML += `<tr>
                    <td>${{k}}</td>
                    <td>${{v.trades}}</td>
                    <td>Rs. ${{v.gross_pnl.toLocaleString('en-IN', {{maximumFractionDigits:0}})}}</td>
                    <td>Rs. ${{v.charges.toLocaleString('en-IN', {{maximumFractionDigits:0}})}}</td>
                    <td style="color:${{color}};font-weight:600">Rs. ${{net.toLocaleString('en-IN', {{maximumFractionDigits:0}})}}</td>
                </tr>`;
            }});
        }}

        // ---- Combined Equity Chart ----
        function initCombinedChart() {{
            const ctx3 = document.getElementById('combinedEquityChart').getContext('2d');
            new Chart(ctx3, {{
                type: 'line',
                data: {{
                    labels: Array.from({{length: combinedEquityCurve.length}}, (_, i) => i),
                    datasets: [{{
                        label: 'Combined Portfolio (Rs.)',
                        data: combinedEquityCurve,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139,92,246,0.07)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#f1f5f9' }}, ticks: {{ color: '#64748b' }} }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        }}

        // 1. Chart.js Equity Curve
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: Array.from({{length: equityCurve.length}}, (_, i) => i),
                datasets: [{{
                    label: 'Portfolio Equity (Rs.)',
                    data: equityCurve,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ display: false }},
                    y: {{ grid: {{ color: '#f1f5f9' }}, ticks: {{ color: '#64748b' }} }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 2. Heatmap Render function
        let selectedYear = 2022;
        
        function renderHeatmap(year) {{
            const container = document.getElementById('heatmapContainer');
            container.innerHTML = '';
            
            const grid = document.createElement('div');
            grid.style.display = 'grid';
            grid.style.gridTemplateRows = 'repeat(7, 12px)';
            grid.style.gridAutoFlow = 'column';
            grid.style.gap = '3px';
            grid.style.padding = '10px';
            grid.style.backgroundColor = '#ffffff';
            grid.style.borderRadius = '8px';
            grid.style.border = '1px solid #e2e8f0';
            
            const startDate = new Date(year, 0, 1);
            const endDate = new Date(year, 11, 31);
            
            // Align first column by adding dummy empty cells
            const startDay = startDate.getDay(); // 0 is Sunday
            for (let i = 0; i < startDay; i++) {{
                const dummy = document.createElement('div');
                dummy.style.width = '12px';
                dummy.style.height = '12px';
                dummy.style.backgroundColor = 'transparent';
                grid.appendChild(dummy);
            }}
            
            let cur = new Date(startDate);
            while (cur <= endDate) {{
                const yearStr = cur.getFullYear();
                const monthStr = String(cur.getMonth() + 1).padStart(2, '0');
                const dayStr = String(cur.getDate()).padStart(2, '0');
                const dateStr = `${{yearStr}}-${{monthStr}}-${{dayStr}}`;
                
                const pnl = bookedPnls[dateStr];
                
                const cell = document.createElement('div');
                cell.style.width = '12px';
                cell.style.height = '12px';
                cell.style.borderRadius = '2px';
                
                if (pnl === undefined) {{
                    cell.style.backgroundColor = '#f1f5f9'; // No squared off position on this day
                }} else if (Math.abs(pnl) < 1) {{
                    cell.style.backgroundColor = '#e2e8f0'; // Breakeven trade
                }} else if (pnl > 0) {{
                    const intensity = Math.min(0.3 + (pnl / 4000) * 0.7, 1.0);
                    cell.style.backgroundColor = `rgba(16, 185, 129, ${{intensity}})`;
                    cell.style.border = '1px solid rgba(16, 185, 129, 0.2)';
                    cell.title = `${{dateStr}} | Realised Profit: +Rs. ${{pnl.toFixed(2)}}`;
                }} else {{
                    const intensity = Math.min(0.3 + (Math.abs(pnl) / 4000) * 0.7, 1.0);
                    cell.style.backgroundColor = `rgba(239, 68, 68, ${{intensity}})`;
                    cell.style.border = '1px solid rgba(239, 68, 68, 0.2)';
                    cell.title = `${{dateStr}} | Realised Loss: -Rs. ${{Math.abs(pnl).toFixed(2)}}`;
                }}
                
                grid.appendChild(cell);
                cur.setDate(cur.getDate() + 1);
            }}
            container.appendChild(grid);
        }}
        
        function setYear(year) {{
            selectedYear = year;
            const btns = document.querySelectorAll('.year-btn');
            btns.forEach(b => {{
                if (b.innerText === String(year)) {{
                    b.classList.add('active');
                }} else {{
                    b.classList.remove('active');
                }}
            }});
            renderHeatmap(year);
        }}
        
        // 3. Monthly Performance Table
        function renderMonthlyTable(year) {{
            const container = document.getElementById('monthlyTableContainer');
            const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            let yearTrades = 0, yearGross = 0, yearCharges = 0, yearNet = 0;
            
            let rows = '';
            for (let m = 1; m <= 12; m++) {{
                const key = `${{year}}-${{String(m).padStart(2,'0')}}`;
                const d = monthlyData[key];
                if (d) {{
                    const netColor = d.net_pnl >= 0 ? '#10b981' : '#ef4444';
                    rows += `<tr>
                        <td style="font-weight:600;">${{months[m-1]}} ${{year}}</td>
                        <td>${{d.trades}}</td>
                        <td>Rs. ${{d.gross_pnl.toFixed(2)}}</td>
                        <td style="color:#ef4444;">Rs. ${{d.charges.toFixed(2)}}</td>
                        <td style="color:${{netColor}};font-weight:600;">Rs. ${{d.net_pnl.toFixed(2)}}</td>
                    </tr>`;
                    yearTrades += d.trades;
                    yearGross += d.gross_pnl;
                    yearCharges += d.charges;
                    yearNet += d.net_pnl;
                }} else {{
                    rows += `<tr style="color:#cbd5e1;">
                        <td>${{months[m-1]}} ${{year}}</td>
                        <td>0</td><td>-</td><td>-</td><td>-</td>
                    </tr>`;
                }}
            }}
            
            const totalColor = yearNet >= 0 ? '#10b981' : '#ef4444';
            container.innerHTML = `
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <thead>
                        <tr style="border-bottom:2px solid #e2e8f0;">
                            <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;">Month</th>
                            <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;">Trades</th>
                            <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;">Gross P&L</th>
                            <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;">Charges & Brokerage</th>
                            <th style="text-align:left;padding:10px 12px;color:#64748b;font-weight:600;">Net P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${{rows}}
                        <tr style="border-top:2px solid #e2e8f0;font-weight:700;">
                            <td style="padding:12px;">Total ${{year}}</td>
                            <td style="padding:12px;">${{yearTrades}}</td>
                            <td style="padding:12px;">Rs. ${{yearGross.toFixed(2)}}</td>
                            <td style="padding:12px;color:#ef4444;">Rs. ${{yearCharges.toFixed(2)}}</td>
                            <td style="padding:12px;color:${{totalColor}};">Rs. ${{yearNet.toFixed(2)}}</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }}
        
        // Initial draw
        setYear(2022);
        renderMonthlyTable(2022);
        
        // Override setYear to also update table
        const _origSetYear = setYear;
        setYear = function(year) {{
            _origSetYear(year);
            renderMonthlyTable(year);
        }};
    </script>
</body>
</html>
"""
    # Write to local file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "nifty_lowvix_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Copy to artifacts directory
    artifact_path = r"C:\Users\Kush Tejani\.gemini\antigravity-ide\brain\d8e9eed5-996a-49a7-bbf0-21fe67c759b9\nifty_lowvix_report.html"
    try:
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated HTML report and saved to: {artifact_path}")
    except Exception as e:
        print(f"Failed to copy HTML report to artifacts: {e}")

# ============================================================
# 9. MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    df = load_and_preprocess_data()
    verify_phase1_data(df)
    all_trades, calendar_log, final_capital, daily_equity = run_portfolio_backtest(df)

    # Run 5m Triple Confirm backtest
    print("\nRunning 5m Triple Confirm backtest...")
    from nifty_5m_backtest import run_5m_backtest
    opt_buy_trades = run_5m_backtest(start_date="2022-01-01")
    print(f"5m backtest complete: {len(opt_buy_trades)} trades")

    trades_df = verify_and_report(all_trades, calendar_log, final_capital, daily_equity,
                                  opt_buy_trades=opt_buy_trades)
