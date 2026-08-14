
import sqlite3
import json

def debug_snapshot():
    conn = sqlite3.connect('sensibull.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 1. Get profile
    slug = 'latered-garage'
    profile = c.execute("SELECT * FROM profiles WHERE slug = ?", (slug,)).fetchone()
    if not profile:
        print(f"Profile {slug} not found")
        return

    # 2. Get latest snapshot for this profile that has data
    date = '2025-12-15'
    changes = c.execute("""
        SELECT * FROM position_changes 
        WHERE profile_id = ? AND date(timestamp) = ?
        ORDER BY timestamp ASC
    """, (profile['id'], date)).fetchall()
    
    print(f"Found {len(changes)} changes for {date}")
    for ch in changes:
        snap = c.execute("SELECT * FROM snapshots WHERE id = ?", (ch['snapshot_id'],)).fetchone()
        raw = json.loads(snap['raw_data'])
        data = raw.get('data', [])
        
        # quick sum
        total = 0
        for item in data:
            for trade in item.get('trades', []):
                total += (trade.get('unbooked_pnl', 0) + trade.get('booked_profit_loss', 0))
                
        print(f"Change ID: {ch['id']}, Time: {ch['timestamp']}, Snap ID: {ch['snapshot_id']}, Calc Total: {total:,.2f}")

    if not changes:
        return

    last_change = changes[-1]

    
    if not last_change:
        print("No changes found")
        return
        
    print(f"Latest Change Date: {last_change['timestamp']}")
    
    snapshot = c.execute("SELECT * FROM snapshots WHERE id = ?", (last_change['snapshot_id'],)).fetchone()
    raw = json.loads(snapshot['raw_data'])
    data = raw.get('data', [])
    
    print(f"\n--- Analysis for Snapshot ID: {snapshot['id']} ---")
    
    total_agg_profit = 0
    calculated_unbooked = 0
    calculated_booked = 0
    
    print(f"\n{'Symbol':<30} | {'Qty':<10} | {'Unbooked':<15} | {'Booked':<15} | {'Total (Calc)':<15}")
    print("-" * 100)
    
    for item in data:
        # 'total_profit' is often at the group level (item)
        agg_profit = item.get('total_profit', 0)
        total_agg_profit += agg_profit
        
        # Now sum trades
        for trade in item.get('trades', []):
            unbooked = trade.get('unbooked_pnl', 0)
            booked = trade.get('booked_profit_loss', 0)
            
            calculated_unbooked += unbooked
            calculated_booked += booked
            
            # Print row
            t_pnl = unbooked + booked
            print(f"{trade.get('trading_symbol'):<30} | {trade.get('quantity'):<10} | {unbooked:<15.2f} | {booked:<15.2f} | {t_pnl:<15.2f}")
            
    print("-" * 100)
    print(f"Aggregate 'total_profit' from JSON: {total_agg_profit:,.2f}")
    print(f"Sum of Unbooked P&L:              {calculated_unbooked:,.2f}")
    print(f"Sum of Booked P&L:                {calculated_booked:,.2f}")
    print(f"Total Calculated (Unbooked+Booked): {calculated_unbooked + calculated_booked:,.2f}")
    
    diff = total_agg_profit - (calculated_unbooked + calculated_booked)
    print(f"\nDifference (Agg - Calc): {diff:,.2f}")

    conn.close()

if __name__ == '__main__':
    debug_snapshot()
