"""Run ONCE (needs internet, may take a few minutes):  python refresh_crw_cache.py
Downloads official NOAA CRW (DHW, SST, HotSpot, SSTA) with long timeouts and saves it to data/crw_cache,
so the app can read it instantly. Prints the real error if NOAA cannot be reached."""
import time, pandas as pd
import data.crw_client as c
c.FAIL_MARKER.unlink(missing_ok=True)
c.TIMEOUT = (60, 180); c.RETRIES = 2; c.BUDGET_S = 900
site = "Saint Martin"
end = (pd.Timestamp.today() - pd.Timedelta(days=1)).date()
start = pd.Timestamp("2026-01-01").date()
t = time.time()
df = c.fetch_crw_for_site(site, str(start), str(end))
print(f"took {time.time()-t:.0f}s")
if df is None or df.empty:
    print("FAILED. Reasons per server:"); [print("  ",k,"->",v) for k,v in c.ERRORS.items()]
else:
    print("columns:", list(df.columns)); print("range:", df['date'].min().date(), "->", df['date'].max().date())
    print(df.tail(5).to_string(index=False))

print("Source used:", c.ERRORS.get("used"))
