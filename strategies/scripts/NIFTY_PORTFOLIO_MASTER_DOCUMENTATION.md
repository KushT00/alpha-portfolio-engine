# Nifty Portfolio Master Architect Engine v3.0 — Complete Operational & Technical Guide

---

## 1. Executive Summary & Core Philosophy

**Nifty Portfolio Master v3.0** is an automated, non-directional option selling strategy built for the **OpenAlgo Platform**. It implements a **Perpetual Rolling Queue Architecture** across multiple weekly and monthly option expiries.

### Core Objectives:
1. **Continuous Premium Harvesting**: Systematically seed far-OTM option legs across future expiries (W1 to M1).
2. **Autonomous Profit Realization**: Automatically lock in profits when short options decay by 80%+ and instantly free margin.
3. **Perpetual Capital Redeployment**: When a leg is harvested, its tier cooldown resets automatically, allowing the seeder to immediately deploy a fresh OTM leg on subsequent ticks.
4. **Dynamic Danger Adjustment**: Protect threatened legs using a net-credit Ratio Spread (Buying 2x threatened strike, selling 3x further OTM).
5. **Margin Capacity Protection**: Automatically monitor open margin usage and execute Emergency Margin Relief if open legs reach capacity.

---

## 2. Rolling Queue Expiry Architecture

The engine categorizes NIFTY option expiries into isolated buckets (tiers):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ROLLING QUEUE EXPIRY MATRIX                          │
├─────────┬───────────────────┬───────────────────────────────────────────────┤
│ Tier    │ Expiry Bucket     │ Strategy Ownership & Seeding Rules            │
├─────────┼───────────────────┼───────────────────────────────────────────────┤
│ W0      │ Current Weekly    │ READ-ONLY (Exclusively owned by Wave/Survivor)│
│ W1      │ Next Weekly       │ Seeds 1 CE + 2 PE @ 300–400pt OTM (Max 3 legs)│
│ W2      │ Far Weekly        │ Seeds 1 CE + 1 PE @ 500–600pt OTM (Max 2 legs)│
│ W3      │ Monthly (~3 wk)   │ Seeds PE Only @ 700–800pt OTM (Max 2 legs)    │
│ M1/M2   │ Far Monthly       │ Seeds PE Only @ 1500–2000pt OTM (Max 2 legs)  │
└─────────┴───────────────────┴───────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **W0 Isolation Safeguard**: Portfolio Master **never seeds W0** positions. W0 is reserved for intraday/short-term strategies (Wave Scraper and Survivor). Portfolio Master only manages long-dated tiers (W1, W2, W3, M1).

---

## 3. Module-by-Module Feature Breakdown

```
                         ┌─────────────────────────────────┐
                         │   EACH 30-SECOND TICK ITERATION │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │  1. Refresh Matrix, Spot & VIX  │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │  2. Expiry Rollover Archiving   │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │  3. Read Broker Positionbook    │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │ 4. Module 1: Safe Harvest (80%) │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │ 5. Module 1.5: Margin Relief    │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │ 6. Module 2: Ratio Spread Check │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │ 7. Module 3: Rolling Seeder     │
                         └─────────────────────────────────┘
```

---

### Module 1: Safe Profit Harvester

#### Logic & Rules:
- **Condition 1 (Decay %)**: Short option LTP must have decayed by **$\ge 80\%$** relative to entry price:
  $$\text{Decay \%} = \frac{\text{Average Entry Price} - \text{LTP}}{\text{Average Entry Price}} \ge 0.80$$
- **Condition 2 (Age Gate)**: Position age must be **$\ge 30$ minutes** (prevents high-frequency churn on initial entry).
- **Condition 3 (Market Gate)**: Market must be open (09:15 – 15:30 IST).

#### Optimizations:
- **Zero-Latency LTP**: Reads `ltp` directly from the single broker `positionbook()` payload instead of making individual serial quote API calls.
- **Ledger Fallback**: If the broker API returns `0` for average sell price, the engine automatically cross-references `nifty_portfolio_master_state.json` to obtain the exact recorded entry price.

---

### Module 1.5: Emergency Margin Relief Harvester

#### Purpose:
Prevents margin exhaustion when the portfolio holds multiple positions during quiet or consolidating market phases.

#### Logic & Rules:
- **Trigger**: Activates automatically if **total open short legs $\ge 12$** (`max_open_legs_threshold`).
- **Qualification**: Scans all short legs decayed **$\ge 60\%$** (`margin_relief_min_decay`).
- **Selection**: Sorts qualifying legs by decay percentage descending (highest decay first).
- **Execution**: Books the **top 2 most decayed positions** (`margin_relief_top_n`).

#### Benefit:
Captures $60\% - 75\%$ of maximum profit early, instantly freeing $\approx ₹2.4$ Lakhs in margin capacity so fresh entries are never blocked.

---

### Module 2: Health Check & Ratio Spread Adjustment Engine

#### Danger Zone Detection:
On every tick, for all active non-W0 expiries, the engine checks every open pair:
- **CE Danger**: If $\text{CE Strike} - \text{Nifty Spot} \le 50\text{ pt}$
- **PE Danger**: If $\text{Nifty Spot} - \text{PE Strike} \le 50\text{ pt}$

#### Dynamic Premium Ratio Spread Adjustment:
When a strike enters the 50pt Danger Zone:
1. **Step 1 (Buy Back Threatened Leg)**: Buy 2 lots of the threatened strike (closes the short position and leaves 1 lot net long hedge).
2. **Step 2 (Sell OTM Net Credit)**: Dynamically scan OTM strikes for an option priced at $50\% - 60\%$ of the threatened LTP, and sell 3 lots.
   $$\text{Net Credit} = (3 \times \text{New OTM LTP}) - (2 \times \text{Threatened LTP})$$
3. **Step 3 (Partner Leg Booking)**: Simultaneously buy back the decaying partner leg if it has decayed $\ge 60\%$.

---

### Module 3: Rolling Queue Seeder & Re-seeding Engine

#### Seeding Rules:
- **W1**: Seeds up to 3 legs (1 CE + 2 PE) at 300–400pt OTM.
- **W2**: Seeds up to 2 legs (1 CE + 1 PE) at 500–600pt OTM.
- **W3**: Seeds up to 2 legs (PE only) at 700–800pt OTM.
- **M1**: Seeds up to 2 legs (PE only) at 1500–2000pt OTM.

#### Re-Seeding Mechanism:
- When a leg is harvested (Module 1 or 1.5), `mark_leg_harvested()` sets the leg status to `"HARVESTED"` in the state file and clears the 4-hour tier cooldown.
- On the next tick, the Seeder observes:
  1. `existing_legs < max_legs`
  2. Cooldown active = `False`
- It places a market order for a fresh OTM leg, maintaining a continuous flow of premium collection.

---

### Module 4: Expiry Rollover Ledger Archiving

#### Rollover Protocol:
When a weekly expiry passes (e.g. after Tuesday 3:30 PM):
- The expiry matrix shifts (`W1 → W0`, `W2 → W1`, `W3 → W2`).
- `archive_expired_ledgers()` compares active matrix expiry codes against state file keys.
- Any expired ledger entry (e.g. `"11AUG26"`) is automatically moved to `archived_expiries` in `nifty_portfolio_master_state.json`.

---

### Module 5: Market Hours Gate

- **Market Hours**: 09:15 to 15:30 IST.
- **Outside Market Hours**: The script runs in **READ-ONLY** mode. Order placement, harvesting, and adjustments are suppressed; logging and monitoring remain active.

---

## 4. Operational Scenarios & Use-Cases

### Scenario A: Normal Market (Decay → Harvest → Re-seed Loop)
1. Tier W1 has a `NIFTY25AUG2625100CE` sold at ₹71.10.
2. Market consolidates over 3 days; LTP drops to ₹8.25 (88.4% decay).
3. **Tick Execution**: Module 1 detects decay $\ge 80\%$ and age $> 30$ min.
4. **Order**: Places BUY order to buy back 65 qty at ₹8.25 (Locks ₹4,085 PnL).
5. **State Update**: Marks leg `"HARVESTED"`, resets W1 tier cooldown.
6. **Next Tick**: Seeder sees W1 has 2/3 legs and zero cooldown. Placed SELL order for a fresh 350pt OTM CE.

---

### Scenario B: Market Rally (Danger Zone & Ratio Spread)
1. Spot rises rapidly to 24,760, approaching `NIFTY18AUG2624800CE` (Distance = 40pt $\le$ 50pt).
2. **Tick Execution**: Module 2 flags Danger Zone breach.
3. **Adjustment**:
   - Buys 2 lots of `24800 CE` (closes short, holds 1 lot long call hedge).
   - Scans OTM: Finds `25150 CE` @ ₹38 (55% of 24800 CE's ₹70 LTP).
   - Sells 3 lots of `25150 CE`.
   - Books decayed partner `23900 PE` if decayed $\ge 60\%$.
4. Position converted into a wider, credit-positive ratio spread.

---

### Scenario C: High Margin Utilization (Emergency Relief)
1. Portfolio holds 14 active short legs across W1, W2, W3, M1.
2. Market remains quiet, but margin available is low.
3. **Tick Execution**: Module 1.5 detects Open Legs ($14$) $\ge$ Threshold ($12$).
4. Scans short legs: Finds `22800 PE` (59.6% decay) and `24200 PE` (64.5% decay).
5. **Action**: Immediately books `24200 PE` (highest decay $\ge 60\%$).
6. Frees ₹1.2 Lakhs in margin capacity instantly.

---

## 5. Configuration Guide (`CONFIG`)

```python
CONFIG = {
    "api_key": "your_openalgo_api_key",
    "host": "http://127.0.0.1:5000",
    "underlying": "NIFTY",
    "exchange_index": "NSE_INDEX",
    "exchange_deriv": "NFO",
    "lot_size": 65,
    "strike_step": 50,
    "lot_multiplier": 1,

    # Market Hours Gate (IST)
    "market_open_hour": 9,
    "market_open_minute": 15,
    "market_close_hour": 15,
    "market_close_minute": 30,

    # Tier Rules: [min_otm, max_otm, max_legs, ce_count, pe_count]
    "tier_rules": {
        "W1": {"min_otm": 300, "max_otm": 400, "max_legs": 3, "ce_count": 1, "pe_count": 2},
        "W2": {"min_otm": 500, "max_otm": 600, "max_legs": 2, "ce_count": 1, "pe_count": 1},
        "W3": {"min_otm": 700, "max_otm": 800, "max_legs": 2, "ce_count": 0, "pe_count": 2},
        "M1": {"min_otm": 1500, "max_otm": 2000, "max_legs": 2, "ce_count": 0, "pe_count": 2},
    },

    # Adjustment Engine
    "danger_zone_pts": 50,
    "ratio_spread_ce_offset": 350,
    "ratio_spread_pe_offset": 350,

    # Safe Harvest Rules
    "harvest_decay_pct": 0.80,      # 80%+ decay required
    "harvest_min_age_minutes": 30,  # 30-min minimum position age

    # Emergency Margin Relief Rules
    "margin_relief_enabled": True,
    "max_open_legs_threshold": 12,  # Triggered if open legs >= 12
    "margin_relief_top_n": 2,        # Books top 2 most decayed
    "margin_relief_min_decay": 0.60, # Minimum 60% decay to qualify

    # Targeted Heavy-Leg Margin Compressor Rules
    "total_capital_budget": 3000000.0,  # ₹30,000,000 (30 Lakhs capital budget)
    "margin_alert_pct": 0.80,           # Trigger targeted compression at 80% utilization (₹24L)
    "heavy_leg_min_lots": 3,            # Leg with >= 3 lots (195+ qty) qualifies as Heavy
    "hedge_otm_min_pts": 350,           # Buy hedge at least 350pt further OTM
    "hedge_max_ltp": 10.0,              # Max premium to pay for cheap hedge (₹10.0)
    "hedge_min_ltp": 2.0,               # Min premium (₹2.0)

    # Seeder Settings
    "seed_cooldown_hours": 4,
    "poll_interval": 30,
}
```

---

## 6. How to Run & Monitor

### Run Command:
```powershell
python strategies/scripts/nifty_portfolio_master_20260807014943.py
```

### Generated Files:
- **`nifty_portfolio_master_state.json`**: Persisted isolated ledger tracking pairs, entry prices, timestamps, and active tier cooldowns.
- **`nifty_portfolio_master_trades.csv`**: Record of executed order IDs, symbols, prices, quantities, and tags.
- **`nifty_portfolio_master_audit.csv`**: High-level audit log of strategy decisions (`HARVEST_BOOKED`, `MARGIN_RELIEF_HARVEST`, `RATIO_SPREAD_APPLIED`, `SEED_PE`, `SEED_CE`).
