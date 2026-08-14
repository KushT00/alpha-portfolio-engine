"""
EMERGENCY POSITION RESTORE SCRIPT
===================================
Re-sells the 5 positions that nifty_portfolio_master.py incorrectly bought back.
Run once. Exits automatically after all orders are placed.
"""
from openalgo import api
import time

client = api(
    api_key="ff1d258f2c5d48bf87292b3f055b39772ca76f4afe69ff7c7c5fa121a32334d8",
    host="http://127.0.0.1:5000"
)

# Positions to restore as SHORT (SELL) — from screenshot
POSITIONS_TO_RESTORE = [
    {"symbol": "NIFTY11AUG2624750CE", "qty": 65},
    {"symbol": "NIFTY11AUG2624700CE", "qty": 260},
    {"symbol": "NIFTY11AUG2624800CE", "qty": 455},
    {"symbol": "NIFTY11AUG2624500PE", "qty": 260},
    {"symbol": "NIFTY11AUG2624600PE", "qty": 65},
]

print("\n*** EMERGENCY POSITION RESTORE — Re-selling incorrectly closed positions ***\n")

for pos in POSITIONS_TO_RESTORE:
    sym = pos["symbol"]
    qty = pos["qty"]
    try:
        res = client.placeorder(
            strategy="RestorePositions",
            symbol=sym,
            action="SELL",
            exchange="NFO",
            price=0,
            pricetype="MARKET",
            product="NRML",
            quantity=qty,
        )
        order_id = res.get("orderid") or res.get("order_id", "UNKNOWN")
        status   = res.get("status", "SENT")
        print(f"  [OK] SELL {sym} x {qty} | Order ID: {order_id} | Status: {status}")
    except Exception as e:
        print(f"  [FAILED] {sym} x {qty} | Error: {e}")
    time.sleep(0.5)

print("\n[DONE] All restore orders placed. Check your broker orderbook to confirm fills.")
