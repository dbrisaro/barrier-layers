"""
plot_amazon_plume_profiles.py

Monthly mean T/S profiles for the Amazon Plume region (lat -5–10°N, lon -55–-30°W),
showing interannual spread (individual years as thin gray lines) + climatological
mean + MLD/ILD markers for all 6 detection methods.

Aesthetic follows sensitivity_one_point.py:
  left panel  — T (salmon) + S (black) on twin top axis
  right panel — dT/dz gradient
  legend panel — method colors / linestyles

Outputs:
  displays/amazon_plume_monthly_profiles.png   — 4×3 grid (Jan–Dec)

Data is cached to data/025deg/amazon_plume_monthly_1993_2024.zarr on first run.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

# ── Settings ──────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -5,  10
LON_MIN, LON_MAX = -55, -30
DEPTH_MAX        = 300.0

CACHE_PATH = ROOT / "data" / "025deg" / "amazon_plume_monthly_1993_2024.zarr"

CONFIGS = [
    ("Gradient −0.015°C/m", "gradient",   -0.015, "#1a6faf", "--"),
    ("Gradient −0.025°C/m", "gradient",   -0.025, "#4dacd6", "--"),
    ("Gradient −0.1°C/m",   "gradient",   -0.1,   "#91d4f5", "--"),
    ("Diff 0.2°C",          "difference",  0.2,   "#2a7a3b", "-."),
    ("Diff 0.5°C",          "difference",  0.5,   "#5dbf6e", "-."),
    ("Diff 0.8°C",          "difference",  0.8,   "#a8dbb0", "-."),
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── 1. Download / load cache ──────────────────────────────────────────────────

if not CACHE_PATH.exists():
    print("Downloading Amazon Plume profiles from CMEMS (monthly, 1993–2024) ...")
    try:
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
            variables=["thetao_glor", "so_glor", "mlotst_glor"],
            minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN,  maximum_latitude=LAT_MAX,
            minimum_depth=0.5057600140571594,
            maximum_depth=508.639892578125,
            coordinates_selection_method="strict-inside",
        )
        has_sal = True
        print("  Salinity (so_glor) available ✓")
    except Exception:
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
            variables=["thetao_glor", "mlotst_glor"],
            minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN,  maximum_latitude=LAT_MAX,
            minimum_depth=0.5057600140571594,
            maximum_depth=508.639892578125,
            coordinates_selection_method="strict-inside",
        )
        has_sal = False
        print("  Salinity not in monthly dataset — temperature-only mode")

    print("  Computing spatial mean over Amazon Plume box ...")
    out_vars = {
        "temp": ds_r["thetao_glor"].mean(dim=["latitude", "longitude"]),
        "mld":  ds_r["mlotst_glor"].mean(dim=["latitude", "longitude"]),
    }
    if has_sal:
        out_vars["sal"] = ds_r["so_glor"].mean(dim=["latitude", "longitude"])

    ds_out = xr.Dataset(out_vars).compute()
    ds_out.attrs["has_sal"] = int(has_sal)
    ds_out.to_zarr(CACHE_PATH, mode="w")
    print(f"  Cached → {CACHE_PATH.name}")
else:
    print(f"Loading cache: {CACHE_PATH.name}")

ds      = xr.open_zarr(CACHE_PATH, consolidated=False)
temp_da = ds["temp"]         # (time, depth)
mld_da  = ds["mld"]          # (time,)
depth   = ds["depth"].values
times   = pd.DatetimeIndex(ds["time"].values)
HAS_SAL = "sal" in ds.data_vars
sal_da  = ds["sal"] if HAS_SAL else None

dmask = depth <= DEPTH_MAX
d_plt = depth[dmask]


# ── 2. Group by calendar month ─────────────────────────────────────────────────

print("Computing monthly statistics and ILD ...")

month_temp = {}   # m → (n_yrs, n_depths)  [depth-truncated]
month_sal  = {}
month_mld  = {}   # m → (n_yrs,)
month_ild  = {cfg[0]: {} for cfg in CONFIGS}  # label → m → array

for m in range(1, 13):
    mask = times.month == m
    t_arr = temp_da.isel(time=mask).values[:, dmask]
    m_arr = mld_da.isel(time=mask).values
    month_temp[m] = t_arr
    month_mld[m]  = m_arr
    if HAS_SAL:
        month_sal[m] = sal_da.isel(time=mask).values[:, dmask]

    # ILD for each method on each year's spatial-mean profile
    # (use full-depth profile, not truncated, for correct threshold search)
    t_full = temp_da.isel(time=mask).values   # (n_yrs, n_depths_full)
    for label, method, thr, color, ls in CONFIGS:
        ilds = [
            ild_from_temp_profile(t_full[yr], depth, method=method, threshold=thr)
            for yr in range(t_full.shape[0])
        ]
        month_ild[label][m] = np.array(ilds)

print("  Done.")


# ── 3. Figure ─────────────────────────────────────────────────────────────────
print("Plotting ...")

# Layout: 4 rows × (3 months × [T/S wide | gradient narrow | spacer])
# Col widths pattern: [3, 1, 0.15] × 3 = 9 cols
ncol = 3
width_ratios = [3, 1, 0.12] * ncol

fig = plt.figure(figsize=(26, 30))
fig.patch.set_facecolor("white")
# Leave right 18 % of figure for the locator map
gs = gridspec.GridSpec(
    4, len(width_ratios),
    width_ratios=width_ratios,
    hspace=0.42, wspace=0.0,
    left=0.04, right=0.78,
    top=0.93, bottom=0.06,
)

# Map month index → (row, ts_col, gr_col)
def col_indices(i):
    col_group = i % ncol
    return col_group * 3, col_group * 3 + 1   # ts_col, gr_col

for i, m in enumerate(range(1, 13)):
    row = i // ncol
    ts_col, gr_col = col_indices(i)

    ax_ts = fig.add_subplot(gs[row, ts_col])
    ax_gr = fig.add_subplot(gs[row, gr_col])

    t_arr   = month_temp[m]
    t_mean  = np.nanmean(t_arr, axis=0)
    mld_arr = month_mld[m]
    mld_m   = float(np.nanmean(mld_arr))
    mld_s   = float(np.nanstd(mld_arr))

    # ── Individual year profiles (gray) ──
    for t_yr in t_arr:
        ax_ts.plot(t_yr, -d_plt, color="#c8c8c8", lw=0.55, alpha=0.7, zorder=1)

    # ── Mean temperature profile ──
    ax_ts.plot(t_mean, -d_plt, color="salmon", lw=2.0, zorder=3)

    # ── Salinity on twin axis ──
    if HAS_SAL:
        ax_s = ax_ts.twiny()
        s_arr  = month_sal[m]
        s_mean = np.nanmean(s_arr, axis=0)
        for s_yr in s_arr:
            ax_s.plot(s_yr, -d_plt, color="#d8d8d8", lw=0.45, alpha=0.45, zorder=1)
        ax_s.plot(s_mean, -d_plt, color="black", lw=2.0, zorder=3)

        ax_s.xaxis.set_ticks_position("top")
        ax_s.xaxis.set_label_position("top")
        ax_s.set_xlabel("Salinity (psu)", color="black", fontsize=7)
        ax_s.tick_params(axis="x", colors="black", labelsize=6)
        ax_s.spines["bottom"].set_visible(False)
        ax_s.spines["right"].set_visible(False)
        ax_s.spines["top"].set_position(("outward", 28))
        ax_s.spines["top"].set_color("black")
        ax_s.set_ylim([-DEPTH_MAX, 0])

    # ── MLD ──
    ax_ts.axhline(-mld_m, color="orange", lw=1.8, zorder=5)
    ax_ts.axhspan(-mld_m - mld_s, -mld_m + mld_s,
                  color="orange", alpha=0.13, zorder=2)

    # ── 6 ILD lines + bands ──
    for label, method, thr, color, ls in CONFIGS:
        ild_arr = month_ild[label][m]
        ild_m   = float(np.nanmean(ild_arr))
        ild_s   = float(np.nanstd(ild_arr))
        if not np.isnan(ild_m):
            ax_ts.axhline(-ild_m, color=color, lw=1.4, ls=ls, zorder=4)
            ax_ts.axhspan(-ild_m - ild_s, -ild_m + ild_s,
                          color=color, alpha=0.09, zorder=2)

    # ── T/S axes styling ──
    ax_ts.set_ylim([-DEPTH_MAX, 0])
    ax_ts.xaxis.set_ticks_position("top")
    ax_ts.xaxis.set_label_position("top")
    ax_ts.set_xlabel("Temperature (°C)", color="salmon", fontsize=7)
    ax_ts.tick_params(axis="x", colors="salmon", labelsize=6)
    ax_ts.tick_params(axis="y", labelsize=6)
    ax_ts.spines["bottom"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)
    ax_ts.spines["top"].set_color("salmon")
    ax_ts.set_ylabel("Depth (m)", fontsize=7)
    pad_top = 42 if HAS_SAL else 14
    ax_ts.set_title(MONTH_NAMES[i], fontsize=10, fontweight="bold", pad=pad_top)

    # ── Gradient panel ──
    t_full_mean = np.nanmean(temp_da.isel(time=(times.month == m)).values, axis=0)
    valid  = ~np.isnan(t_full_mean)
    d_v    = depth[valid]
    t_v    = t_full_mean[valid]
    grad   = np.diff(t_v) / np.diff(d_v)
    d_mid  = (d_v[:-1] + d_v[1:]) / 2.0
    gm     = d_mid <= DEPTH_MAX

    ax_gr.plot(grad[gm], -d_mid[gm], ".", color="darkgray",
               markersize=2.5, lw=0.5, ls="-", zorder=2)
    ax_gr.axhline(-mld_m, color="orange", lw=1.4, zorder=5)
    for label, method, thr, color, ls in CONFIGS:
        if method == "gradient":
            ax_gr.axvline(thr, color=color, lw=1.0, ls=":", zorder=3, alpha=0.9)
        ild_m_val = float(np.nanmean(month_ild[label][m]))
        if not np.isnan(ild_m_val):
            ax_gr.axhline(-ild_m_val, color=color, lw=1.0, ls=ls, zorder=4)

    ax_gr.set_ylim([-DEPTH_MAX, 0])
    ax_gr.set_xlim([-0.36, 0.06])
    ax_gr.xaxis.set_ticks_position("top")
    ax_gr.xaxis.set_label_position("top")
    ax_gr.set_xlabel("dT/dz (°C/m)", fontsize=6)
    ax_gr.tick_params(axis="both", labelsize=5)
    ax_gr.tick_params(axis="x", rotation=45)
    ax_gr.spines["bottom"].set_visible(False)
    ax_gr.spines["right"].set_visible(False)
    ax_gr.yaxis.set_ticklabels([])
    ax_gr.set_ylabel("")


# ── 4. Locator map (right side, spanning full height) ─────────────────────────
# Regional view: tropical Atlantic + NE South America
map_proj = ccrs.PlateCarree()
ax_map = fig.add_axes([0.80, 0.10, 0.18, 0.80], projection=map_proj)

ax_map.set_extent([-85, 20, -35, 35], crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
ax_map.add_feature(cfeature.LAND,      facecolor="#e8e8e8", zorder=1)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5,        zorder=2)
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3,        zorder=2,
                   linestyle=":")
ax_map.add_feature(cfeature.RIVERS,    linewidth=0.4,        zorder=2,
                   edgecolor="#90b4d4")

# Highlight the Amazon Plume box
box = mpatches.Rectangle(
    (LON_MIN, LAT_MIN),
    LON_MAX - LON_MIN,
    LAT_MAX - LAT_MIN,
    linewidth=2.0, edgecolor="#c0392b", facecolor="#e74c3c",
    alpha=0.30, transform=ccrs.PlateCarree(), zorder=3,
)
ax_map.add_patch(box)
# Red outline on top (no fill) for crisp border
box_outline = mpatches.Rectangle(
    (LON_MIN, LAT_MIN),
    LON_MAX - LON_MIN,
    LAT_MAX - LAT_MIN,
    linewidth=2.0, edgecolor="#c0392b", facecolor="none",
    transform=ccrs.PlateCarree(), zorder=4,
)
ax_map.add_patch(box_outline)

# Equator and grid lines
ax_map.plot([-85, 20], [0, 0], color="gray", lw=0.6, ls="--",
            transform=ccrs.PlateCarree(), zorder=3)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False
gl.xlabel_style = {"size": 7}
gl.ylabel_style = {"size": 7}

ax_map.set_title("Study region", fontsize=9, fontweight="bold", pad=6)

# Annotation inside the box
ax_map.text(
    (LON_MIN + LON_MAX) / 2, (LAT_MIN + LAT_MAX) / 2,
    "Amazon\nPlume", fontsize=7.5, color="#7b241c",
    ha="center", va="center", fontweight="bold",
    transform=ccrs.PlateCarree(), zorder=5,
)

# ── 5. Shared legend (bottom strip, narrowed to fit map) ──────────────────────
lx = fig.add_axes([0.04, 0.015, 0.74, 0.038])
lx.set_axis_off()

x0 = 0.0
# MLD
lx.plot([x0, x0 + 0.035], [0.55, 0.55], color="orange", lw=2.0,
        transform=lx.transAxes, clip_on=False)
lx.text(x0 + 0.040, 0.55, "MLD", fontsize=8, color="orange",
        va="center", transform=lx.transAxes)
x0 += 0.13

for label, method, thr, color, ls in CONFIGS:
    lx.plot([x0, x0 + 0.035], [0.55, 0.55], color=color, lw=1.8, ls=ls,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0 + 0.040, 0.55, label, fontsize=7.5, color=color,
            va="center", transform=lx.transAxes)
    x0 += 0.155

# Gray lines explanation
lx.plot([x0, x0 + 0.035], [0.55, 0.55], color="#c8c8c8", lw=1.5,
        transform=lx.transAxes, clip_on=False)
lx.text(x0 + 0.040, 0.55, "Individual years (1993–2024)",
        fontsize=7.5, color="gray", va="center", transform=lx.transAxes)
lx.text(0.85, 0.05, "Shaded bands = ±1 std across years",
        fontsize=7, color="gray", va="bottom", transform=lx.transAxes)

fig.suptitle(
    f"Monthly T/S profiles — Amazon Plume  "
    f"(lat {LAT_MIN}°–{LAT_MAX}°N, lon {abs(LON_MAX)}°–{abs(LON_MIN)}°W)  |  1993–2024",
    fontsize=13, y=0.965,
)

out = ROOT / "displays" / "amazon_plume_monthly_profiles.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: displays/{out.name}")
