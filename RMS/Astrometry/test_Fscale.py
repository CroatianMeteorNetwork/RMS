#!/usr/bin/env python3
"""
plot_fscale_over_night.py
Track the per-frame F_scale through the night and show a trend plot.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- config ----------------------------------------------------------

JSON_FILE = Path("/Users/lucbusquin/Projects/RMS_data/ArchivedFiles/US9999_20231210_054940_924175_detected/platepars_all_recalibrated.json")   # update if you renamed it

# ----------------------------------------------------------------------

# 1) load
with JSON_FILE.open() as f:
    pp_dict = json.load(f)

# 2) pull time stamp & F_scale out of each FF key
ts_re = re.compile(r"FF_[A-Z0-9]+_(\d{8})_(\d{6})_(\d{3})")
rows = []
for ff_name, pp in pp_dict.items():
    m = ts_re.search(ff_name)
    if not m:
        continue
    yyyymmdd, hhmmss, ms = m.groups()
    dt = datetime.strptime(yyyymmdd + hhmmss, "%Y%m%d%H%M%S").replace(
        microsecond=int(ms) * 1000
    )
    rows.append((dt, float(pp["F_scale"])))

# 3) sort chronologically
rows.sort()
times, fscale = zip(*rows)

# 4) quick-look stats
fscale = np.array(fscale)
print(f"Frames: {len(fscale)}")
print(f"F_scale  mean: {fscale.mean():.6f}")
print(f"         std : {fscale.std():.6f}")
print(f"         span: {fscale.min():.6f} … {fscale.max():.6f}")

# 5) plot
plt.plot(times, fscale, ".-")
plt.xlabel("UTC time")
plt.ylabel("F_scale (px / rad)")
plt.title("Per-frame plate-scale through the night")
plt.tight_layout()
plt.show()