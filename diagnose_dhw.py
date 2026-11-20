"""Run:  python diagnose_dhw.py   (needs internet)
Shows where the high DHW comes from: Open-Meteo SST vs MMM, and the latest official NOAA CRW values."""
import pandas as pd
from data.openmeteo import SITE_COORDS, SITE_MMM, fetch_atmospheric_archive, fetch_marine_sst_forecast
from data.crw_client import fetch_crw_series, SITE_CRW_GRID
site = "Saint Martin"
lat, lon = SITE_COORDS[site]; mmm = SITE_MMM[site]
end = (pd.Timestamp.today() - pd.Timedelta(days=1)).date(); start = end - pd.Timedelta(days=120)
a = fetch_atmospheric_archive(lat, lon, str(start), str(end))
a["HS_from_OpenMeteo"] = (a["SST"] - mmm).clip(lower=0)
print("Open-Meteo SST (last 120 d) monthly mean / max:"); print(a.set_index("date")["SST"].resample("MS").agg(["mean","max"]).round(2))
print(f"days with SST-MMM >= 1 C: {(a['SST']-mmm>=1).sum()} of {len(a)}   (MMM={mmm})")
import data.crw_client as _cc; _cc.FAIL_MARKER.unlink(missing_ok=True)
c = fetch_crw_series(*SITE_CRW_GRID[site], str(start), str(end))
import data.crw_client as _c
print("\\nOfficial NOAA CRW tail:"); print(c.tail(8) if c is not None else "CRW fetch FAILED (this is why the app falls back to the proxy)")

if c is None: print("Reasons per server:"); [print("  ",k,"->",v) for k,v in _c.ERRORS.items()] or print("   (none recorded)")

if c is not None: print("Source used:", _c.ERRORS.get("used"))
