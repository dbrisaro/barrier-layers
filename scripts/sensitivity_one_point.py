"""
sensitivity_one_point.py

Downloads one day of temperature/salinity data at a single lat/lon point and
runs all 6 ILD detection configurations (3 gradient + 3 difference thresholds).
Produces a plot with the same aesthetic as the single-profile notebooks,
overlaying all 6 ILD estimates so the methods can be compared visually.

Output: displays/sensitivity_one_point_{date}_{lat}_{lon}.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import copernicusmarine as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

# ── Settings ──────────────────────────────────────────────────────────────────
DATE      = "2023-12-30"
LAT       =  -10.0          # Indian Ocean – known barrier layer region
LON       =   55.0
DEPTH_MAX = 300.0           # plot depth limit

CONFIGS = [
    # (label,             method,       threshold,  color,        linestyle)
    ("Gradient −0.015°C/m", "gradient",   -0.015,  "#1a6faf",    "--"),
    ("Gradient −0.025°C/m", "gradient",   -0.025,  "#4dacd6",    "--"),
    ("Gradient −0.1°C/m",   "gradient",   -0.1,    "#91d4f5",    "--"),
    ("Diff 0.2°C",          "difference",  0.2,    "#2a7a3b",    "-."),
    ("Diff 0.5°C",          "difference",  0.5,    "#5dbf6e",    "-."),
    ("Diff 0.8°C",          "difference",  0.8,    "#a8dbb0",    "-."),
]

# ── Download ──────────────────────────────────────────────────────────────────
print(f"Downloading data — {DATE}  lat={LAT}  lon={LON} ...")
ds = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1D-m",
    dataset_version="202311",
    variables=["mlotst_glor", "thetao_glor", "so_glor"],
    minimum_longitude=LON,   maximum_longitude=LON,
    minimum_latitude=LAT,    maximum_latitude=LAT,
    start_datetime=DATE + "T00:00:00",
    end_datetime=DATE   + "T23:59:59",
    minimum_depth=0.5057600140571594,
    maximum_depth=508.639892578125,
    coordinates_selection_method="nearest",
)

pt = ds.isel(time=0).squeeze()
depth  = pt["depth"].values
temp   = pt["thetao_glor"].values
sal    = pt["so_glor"].values
mld    = float(pt["mlotst_glor"].values)

# Gradient (for right panel)
valid      = ~np.isnan(temp)
d_v, t_v   = depth[valid], temp[valid]
grad       = np.diff(t_v) / np.diff(d_v)
depth_mid  = (d_v[:-1] + d_v[1:]) / 2.0

print(f"MLD = {mld:.1f} m")

# ── Compute ILD for all 6 configs ─────────────────────────────────────────────
results = []
for label, method, thr, color, ls in CONFIGS:
    ild = ild_from_temp_profile(temp, depth, method=method, threshold=thr)
    bld = (ild - mld) if not np.isnan(ild) else np.nan
    results.append((label, method, thr, color, ls, ild, bld))
    print(f"  {label:26s}  ILD={ild:6.1f} m   BLD={bld:6.1f} m")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("white")

# Left panel: T + S profile
ax = plt.axes([0.05, 0.06, 0.38, 0.86])
bx = ax.twiny()

ax.plot(temp, -depth, ".", color="salmon",  markersize=3, lw=0.5, ls="-")
bx.plot(sal,  -depth, ".", color="black",   markersize=3, lw=0.5, ls="-")

# MLD
ax.axhline(-mld, color="orange", lw=1.5, ls="-", zorder=5)
ax.text(np.nanmin(temp) + 0.05, -mld - 6, f"MLD: {mld:.1f} m",
        color="orange", fontsize=8, va="top")

# 6 ILD lines
for label, method, thr, color, ls, ild, bld in results:
    if not np.isnan(ild):
        ax.axhline(-ild, color=color, lw=1.4, ls=ls, zorder=4)

# Axes styling
ax.set_xlabel("Temperature (°C)", color="salmon", fontsize=9)
bx.set_xlabel("Salinity (psu)",   color="black",  fontsize=9)
ax.set_ylabel("Depth (m)", fontsize=9)
ax.set_ylim([-DEPTH_MAX, 0])
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
ax.spines["bottom"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", colors="salmon", labelsize=8)
ax.spines["top"].set_color("salmon")
bx.xaxis.set_ticks_position("top"); bx.xaxis.set_label_position("top")
bx.spines["bottom"].set_visible(False); bx.spines["right"].set_visible(False)
bx.spines["top"].set_position(("outward", 38))
bx.spines["top"].set_color("black")
bx.tick_params(axis="x", colors="black", labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_title("", pad=0)

# Right panel: gradient profile
cx = plt.axes([0.48, 0.06, 0.26, 0.86])
cx.plot(grad, -depth_mid, ".", color="darkgrey", markersize=3, lw=0.5, ls="-")
cx.set_title("dT/dz (°C/m)", fontsize=10, pad=8)
cx.set_ylim([-DEPTH_MAX, 0])
cx.set_xlim([-0.35, 0.05])
cx.xaxis.set_ticks_position("top"); cx.xaxis.set_label_position("top")
cx.spines["bottom"].set_visible(False); cx.spines["right"].set_visible(False)
cx.tick_params(axis="both", labelsize=8)

# gradient threshold lines (same colors as those configs)
for label, method, thr, color, ls, ild, bld in results:
    if method == "gradient":
        cx.axvline(thr, color=color, lw=1.2, ls=":", zorder=4)

# ILD lines on gradient panel too
for label, method, thr, color, ls, ild, bld in results:
    if not np.isnan(ild):
        cx.axhline(-ild, color=color, lw=1.2, ls=ls, zorder=4)

# Legend panel
lx = plt.axes([0.77, 0.06, 0.21, 0.86])
lx.set_axis_off()

lx.text(0.0, 0.97, "ILD methods", fontsize=9, fontweight="bold", va="top", transform=lx.transAxes)
lx.text(0.0, 0.91, "— Gradient", fontsize=8, color="#1a6faf", va="top", transform=lx.transAxes)
lx.text(0.0, 0.86, "— Difference", fontsize=8, color="#2a7a3b", va="top", transform=lx.transAxes)
lx.axhline(0.83, color="orange",  lw=1.5, xmin=0, xmax=0.25)
lx.text(0.0, 0.80, "MLD", fontsize=8, color="orange", va="top", transform=lx.transAxes)

y0 = 0.72
for label, method, thr, color, ls, ild, bld in results:
    # draw line sample
    lx.axhline(y0, color=color, lw=1.4, ls=ls, xmin=0.0, xmax=0.28)
    tag = f"{label}"
    val = f"ILD={ild:.0f}m  BLD={bld:.0f}m" if not np.isnan(ild) else "ILD=—"
    lx.text(0.32, y0 + 0.01, tag, fontsize=7.5, color=color, va="center", transform=lx.transAxes)
    lx.text(0.32, y0 - 0.03, val, fontsize=7,   color="gray",  va="center", transform=lx.transAxes)
    y0 -= 0.12

lx.set_ylim([0, 1])

out = ROOT / "displays" / f"sensitivity_one_point_{DATE}_lat{LAT}_lon{LON}.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
