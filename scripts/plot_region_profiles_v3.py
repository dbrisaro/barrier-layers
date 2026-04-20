"""
plot_region_profiles_v3.py

Monthly T/S profiles (Jan–Dec) with all 6 ILD threshold methods for:
  1. Tropical Atlantic   (lat  -5–10°N,  lon -55–-30°W)
  2. N. Argentine Shelf  (lat -45–-30°S, lon -65–-50°W)

Each panel: individual year T profiles (gray), climatological mean T (salmon),
climatological mean S (black, twin top axis), MLD (orange), 6 ILD lines.
Right column: dT/dz gradient panel.
Right side:   locator map (pyshp, no cartopy).

Font: Helvetica Neue. Aspect ratio never broken (set_aspect("equal") on locator).
Outputs:
  displays/profiles_tropical_atlantic.png
  displays/profiles_arg_sea_north.png
"""
import sys, warnings
from pathlib import Path

import numpy as np
import gsw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import shapefile as shp
import xarray as xr
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Users/daniela/Documents/barrier_layers")
sys.path.insert(0, str(ROOT))
from src.barrier_layers import ild_from_temp_profile

# Use absolute values for T inversions (Southern Ocean, etc.)
USE_ABS = True


def sigma0_from_ts(temp_1d, sal_1d, depth_1d):
    """Potential density anomaly σ₀ (kg/m³) via TEOS-10."""
    p   = gsw.p_from_z(-depth_1d, 0.0)          # pressure (dbar), lat≈0 ok for sigma0
    SA  = gsw.SA_from_SP(sal_1d, p, 0.0, 0.0)   # Absolute Salinity
    CT  = gsw.CT_from_t(SA, temp_1d, p)          # Conservative Temperature
    return gsw.sigma0(SA, CT)                    # σ₀ = ρ(S,T,0) − 1000


def mld_from_density(sigma0_1d, depth_1d, delta_sigma=0.03):
    """
    MLD from density criterion: first depth where σ₀(z) > σ₀(surface) + delta_sigma.
    delta_sigma = 0.03 kg/m³ is the CMEMS/de Boyer Montégut standard.
    """
    sig = np.asarray(sigma0_1d, dtype=float)
    dep = np.asarray(depth_1d,  dtype=float)
    valid = np.isfinite(sig)
    if valid.sum() < 2:
        return np.nan
    sig_surf = sig[valid][0]
    crossed  = (sig[valid] > sig_surf + delta_sigma)
    if not np.any(crossed):
        return np.nan
    return float(dep[valid][np.argmax(crossed)])

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
})

CACHE_DIR = ROOT / "data" / "025deg"
OUT_DIR   = ROOT / "displays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NE_COAST = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"
NE_LAND  = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"

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

REGIONS = {
    "tropical_atlantic": dict(
        label      = "Tropical Atlantic",
        lat_min    = -5,   lat_max = 10,
        lon_min    = -55,  lon_max = -30,
        depth_max  = 300,
        max_depth_dl = 600.0,
        # Reuse existing amazon_plume cache — same bounding box
        cache_zarr = CACHE_DIR / "amazon_plume_monthly_1993_2024.zarr",
        map_extent = [-68, -15, -18, 22],
        map_lpos   = (-42.5, 2.5),
        box_color  = "#e74c3c",
    ),
    "arg_sea_north": dict(
        label      = "N. Argentine Shelf",
        lat_min    = -45,  lat_max = -30,
        lon_min    = -65,  lon_max = -50,
        depth_max  = 200,
        max_depth_dl = 600.0,
        cache_zarr = CACHE_DIR / "arg_sea_north_monthly_1993_2024.zarr",
        map_extent = [-72, -40, -58, -20],
        map_lpos   = (-57.5, -37.5),
        box_color  = "#16a085",
    ),
    "southern_ocean": dict(
        label      = "Southern Ocean (Indian sector)",
        lat_min    = -65,  lat_max = -45,
        lon_min    = 20,   lon_max = 100,
        depth_max  = 600,
        max_depth_dl = 1000.0,
        cache_zarr = CACHE_DIR / "profiles_southern_ocean_1993_2024.zarr",
        map_extent = [5, 115, -75, -38],
        map_lpos   = (60, -55),
        box_color  = "#2980b9",
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_land_coast(ax, land_color="#e8e8e8", coast_lw=0.5):
    if NE_LAND.exists():
        sf = shp.Reader(str(NE_LAND))
        patches = []
        for shape in sf.shapes():
            if shape.shapeType == 0:
                continue
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                patches.append(plt.Polygon(seg, closed=True))
        ax.add_collection(mcollections.PatchCollection(
            patches, facecolor=land_color, edgecolor="none", zorder=2))
    if NE_COAST.exists():
        sf = shp.Reader(str(NE_COAST))
        for shape in sf.shapes():
            pts   = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for k in range(len(shape.parts)):
                seg = pts[parts[k]:parts[k+1]]
                ax.plot(seg[:, 0], seg[:, 1],
                        lw=coast_lw, color="#555555", zorder=3)


def ensure_cache(rcfg):
    """Download T/S profiles from CMEMS and cache as zarr if not present."""
    zpath = rcfg["cache_zarr"]
    if zpath.exists():
        print(f"  Cache found: {zpath.name}")
        return
    print(f"  Downloading {rcfg['label']} from CMEMS ...")
    import copernicusmarine as cm
    try:
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
            variables=["thetao_glor", "so_glor", "mlotst_glor"],
            minimum_longitude=rcfg["lon_min"], maximum_longitude=rcfg["lon_max"],
            minimum_latitude =rcfg["lat_min"], maximum_latitude =rcfg["lat_max"],
            minimum_depth=0.5057600140571594,
            maximum_depth=rcfg.get("max_depth_dl", 600.0),
            coordinates_selection_method="strict-inside",
        )
        has_sal = True
    except Exception as e:
        print(f"  Salinity unavailable ({e}), temp-only mode")
        ds_r = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1M-m",
            variables=["thetao_glor", "mlotst_glor"],
            minimum_longitude=rcfg["lon_min"], maximum_longitude=rcfg["lon_max"],
            minimum_latitude =rcfg["lat_min"], maximum_latitude =rcfg["lat_max"],
            minimum_depth=0.5057600140571594,
            maximum_depth=rcfg.get("max_depth_dl", 600.0),
            coordinates_selection_method="strict-inside",
        )
        has_sal = False

    out_vars = {
        "temp": ds_r["thetao_glor"].mean(dim=["latitude", "longitude"]),
        "mld":  ds_r["mlotst_glor"].mean(dim=["latitude", "longitude"]),
    }
    if has_sal:
        out_vars["sal"] = ds_r["so_glor"].mean(dim=["latitude", "longitude"])
    ds_out = xr.Dataset(out_vars).compute()
    ds_out.attrs["has_sal"] = int(has_sal)
    ds_out.to_zarr(zpath, mode="w", zarr_format=2)
    print(f"  Cached → {zpath.name}")


# ── Profile figure ────────────────────────────────────────────────────────────

def plot_region(rkey, rcfg):
    print(f"\n── {rcfg['label']} ──")
    ensure_cache(rcfg)

    ds      = xr.open_zarr(rcfg["cache_zarr"], consolidated=False)
    temp_da = ds["temp"]
    mld_da  = ds["mld"]  if "mld"  in ds.data_vars else None
    sal_da  = ds["sal"]  if "sal"  in ds.data_vars else None
    depth   = ds["depth"].values
    times   = pd.DatetimeIndex(ds["time"].values)
    HAS_SAL = sal_da is not None
    depth_max = rcfg["depth_max"]
    dmask = depth <= depth_max
    d_plt = depth[dmask]

    # ── Group by calendar month & compute ILD + density ──────────────────
    month_temp  = {}; month_sal = {}; month_mld = {}
    month_sig0  = {}   # σ₀ mean profiles
    month_mld_rho = {}  # MLD from density criterion
    month_ild   = {cfg[0]: {} for cfg in CONFIGS}

    for m in range(1, 13):
        mask_m = times.month == m
        month_temp[m] = temp_da.isel(time=mask_m).values[:, dmask]
        if HAS_SAL:
            month_sal[m] = sal_da.isel(time=mask_m).values[:, dmask]
        if mld_da is not None:
            month_mld[m] = mld_da.isel(time=mask_m).values

        # σ₀ from climatological mean T and S
        if HAS_SAL:
            t_clim = np.nanmean(month_temp[m], axis=0)
            s_clim = np.nanmean(month_sal[m],  axis=0)
            valid  = np.isfinite(t_clim) & np.isfinite(s_clim)
            sig = np.full(len(d_plt), np.nan)
            if valid.sum() >= 2:
                sig[valid] = sigma0_from_ts(t_clim[valid], s_clim[valid], d_plt[valid])
            month_sig0[m] = sig
            month_mld_rho[m] = mld_from_density(sig, d_plt)

        t_full = temp_da.isel(time=mask_m).values   # full depth for ILD
        for lbl, method, thr, color, ls in CONFIGS:
            ilds = [ild_from_temp_profile(t_full[yr], depth, method=method,
                                          threshold=thr, use_abs=USE_ABS)
                    for yr in range(t_full.shape[0])]
            month_ild[lbl][m] = np.array(ilds)

    # ── Global axis limits (fixed across all months) ──────────────────────
    all_t = np.concatenate([month_temp[m].ravel() for m in range(1,13)])
    t_min = np.nanmin(all_t); t_max = np.nanmax(all_t)
    # 5 % padding
    t_pad = (t_max - t_min) * 0.05
    T_LIM = (t_min - t_pad, t_max + t_pad)

    if HAS_SAL:
        all_s = np.concatenate([month_sal[m].ravel() for m in range(1,13)])
        s_min = np.nanmin(all_s); s_max = np.nanmax(all_s)
        s_pad = (s_max - s_min) * 0.05
        S_LIM = (s_min - s_pad, s_max + s_pad)

    # Gradient limits from all monthly mean profiles
    all_grad = []
    for m in range(1, 13):
        t_fm = np.nanmean(temp_da.isel(time=(times.month == m)).values, axis=0)
        vld  = ~np.isnan(t_fm)
        if vld.sum() > 2:
            d_v = depth[vld]; t_v = t_fm[vld]
            gr  = np.diff(t_v) / np.diff(d_v)
            dm  = (d_v[:-1] + d_v[1:]) / 2.0
            all_grad.extend(gr[dm <= depth_max].tolist())
    if all_grad:
        g_min = max(np.nanpercentile(all_grad, 1), -1.0)
        g_max = min(np.nanpercentile(all_grad, 99),  0.2)
        g_pad = (g_max - g_min) * 0.05
        G_LIM = (g_min - g_pad, g_max + g_pad)
    else:
        G_LIM = (-0.36, 0.06)

    # σ₀ limits
    if HAS_SAL and month_sig0:
        all_sig = np.concatenate([month_sig0[m] for m in range(1,13)])
        all_sig = all_sig[np.isfinite(all_sig)]
        if all_sig.size > 0:
            sig_pad = (all_sig.max() - all_sig.min()) * 0.05
            SIG_LIM = (all_sig.min() - sig_pad, all_sig.max() + sig_pad)
        else:
            SIG_LIM = None
    else:
        SIG_LIM = None

    # ── Figure layout ─────────────────────────────────────────────────────
    # Each month: [T/S wide | dT/dz narrow | σ₀ narrow | spacer]
    ncol         = 3
    width_ratios = [3, 1, 1, 0.08] * ncol

    fig = plt.figure(figsize=(30, 28))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        4, len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.50, wspace=0.0,
        left=0.04, right=0.76,
        top=0.96, bottom=0.07,
    )

    for i, m in enumerate(range(1, 13)):
        row    = i // ncol
        g      = i % ncol
        ts_col = g * 4        # 4 columns per month group: T/S | grad | σ₀ | spacer
        gr_col = g * 4 + 1
        sg_col = g * 4 + 2

        ax_ts = fig.add_subplot(gs[row, ts_col])
        ax_gr = fig.add_subplot(gs[row, gr_col])
        ax_sg = fig.add_subplot(gs[row, sg_col])

        t_arr  = month_temp[m]
        t_mean = np.nanmean(t_arr, axis=0)
        mld_m  = float(np.nanmean(month_mld[m])) if m in month_mld else np.nan

        # Individual year T profiles (gray)
        for t_yr in t_arr:
            ax_ts.plot(t_yr, -d_plt, color="#c8c8c8", lw=0.55, alpha=0.7, zorder=1)
        # Climatological mean T
        ax_ts.plot(t_mean, -d_plt, color="salmon", lw=2.0, zorder=3)

        # Salinity on twin top axis
        if HAS_SAL and m in month_sal:
            ax_s   = ax_ts.twiny()
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
            ax_s.spines["top"].set_position(("outward", 32))
            ax_s.spines["top"].set_color("black")
            ax_s.set_xlim(S_LIM)
            ax_s.set_ylim([-depth_max, 0])

        # MLD line (model, orange)
        if np.isfinite(mld_m):
            ax_ts.axhline(-mld_m, color="orange", lw=1.8, zorder=5)

        # 6 ILD lines (mean only)
        for lbl, method, thr, color, ls in CONFIGS:
            ild_arr = month_ild[lbl][m]
            ild_m_v = float(np.nanmean(ild_arr))
            if not np.isnan(ild_m_v):
                ax_ts.axhline(-ild_m_v, color=color, lw=1.4, ls=ls, zorder=4)

        # T axis styling
        ax_ts.set_xlim(T_LIM)
        ax_ts.set_ylim([-depth_max, 0])
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

        # ── dT/dz gradient panel ──────────────────────────────────────────
        t_full_mean = np.nanmean(temp_da.isel(time=(times.month == m)).values, axis=0)
        valid = ~np.isnan(t_full_mean)
        if valid.sum() > 2:
            d_v   = depth[valid]; t_v = t_full_mean[valid]
            grad  = np.diff(t_v) / np.diff(d_v)
            d_mid = (d_v[:-1] + d_v[1:]) / 2.0
            gm    = d_mid <= depth_max
            ax_gr.plot(grad[gm], -d_mid[gm], ".", color="darkgray",
                       markersize=2.5, lw=0.5, ls="-", zorder=2)
        if np.isfinite(mld_m):
            ax_gr.axhline(-mld_m, color="orange", lw=1.4, zorder=5)
        for lbl, method, thr, color, ls in CONFIGS:
            if method == "gradient":
                ax_gr.axvline(thr, color=color, lw=1.0, ls=":", zorder=3, alpha=0.9)
                if USE_ABS:
                    ax_gr.axvline(-thr, color=color, lw=0.7, ls=":", zorder=3, alpha=0.5)
            ild_v = float(np.nanmean(month_ild[lbl][m]))
            if not np.isnan(ild_v):
                ax_gr.axhline(-ild_v, color=color, lw=1.0, ls=ls, zorder=4)

        ax_gr.set_ylim([-depth_max, 0])
        ax_gr.set_xlim(G_LIM)
        ax_gr.xaxis.set_ticks_position("top")
        ax_gr.xaxis.set_label_position("top")
        ax_gr.set_xlabel("dT/dz (°C/m)", fontsize=6)
        ax_gr.tick_params(axis="both", labelsize=5)
        ax_gr.tick_params(axis="x", rotation=45)
        ax_gr.spines["bottom"].set_visible(False)
        ax_gr.spines["right"].set_visible(False)
        ax_gr.yaxis.set_ticklabels([])

        # ── σ₀ density panel ──────────────────────────────────────────────
        if HAS_SAL and m in month_sig0:
            sig = month_sig0[m]
            mld_rho = month_mld_rho.get(m, np.nan)
            valid_s = np.isfinite(sig)
            if valid_s.sum() > 1:
                ax_sg.plot(sig[valid_s], -d_plt[valid_s],
                           color="#6c3483", lw=2.0, zorder=3)
            # Model MLD (orange) — from mlotst
            if np.isfinite(mld_m):
                ax_sg.axhline(-mld_m, color="orange", lw=1.8, zorder=5,
                              label="MLD model")
            # Density-criterion MLD (dashed purple)
            if np.isfinite(mld_rho):
                ax_sg.axhline(-mld_rho, color="#6c3483", lw=1.4,
                              ls="--", zorder=5, label="MLD Δσ₀=0.03")

        ax_sg.set_ylim([-depth_max, 0])
        if SIG_LIM:
            ax_sg.set_xlim(SIG_LIM)
        ax_sg.xaxis.set_ticks_position("top")
        ax_sg.xaxis.set_label_position("top")
        ax_sg.set_xlabel("σ₀ (kg/m³)", fontsize=6, color="#6c3483")
        ax_sg.tick_params(axis="both", labelsize=5)
        ax_sg.tick_params(axis="x", rotation=45, colors="#6c3483")
        ax_sg.spines["bottom"].set_visible(False)
        ax_sg.spines["right"].set_visible(False)
        ax_sg.spines["top"].set_color("#6c3483")
        ax_sg.yaxis.set_ticklabels([])

    # ── Locator map (right side) ───────────────────────────────────────────
    me = rcfg["map_extent"]  # [lon0, lon1, lat0, lat1]
    ax_map = fig.add_axes([0.80, 0.12, 0.18, 0.78])
    ax_map.set_facecolor("#d6eaf8")
    draw_land_coast(ax_map, land_color="#e8e8e8")
    ax_map.set_xlim(me[0], me[1])
    ax_map.set_ylim(me[2], me[3])

    # Correct aspect for the map centre latitude
    lat_ctr = (rcfg["lat_min"] + rcfg["lat_max"]) / 2.0
    ax_map.set_aspect(1.0 / max(np.cos(np.deg2rad(lat_ctr)), 0.1))

    col = rcfg["box_color"]
    # Filled region box
    ax_map.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor=col, facecolor=col, alpha=0.25, zorder=3
    ))
    ax_map.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor=col, facecolor="none", zorder=4
    ))
    # Equator dashed line (if applicable)
    if me[2] < 0 < me[3]:
        ax_map.axhline(0, color="gray", lw=0.6, ls="--", zorder=2)

    ax_map.text(
        rcfg["map_lpos"][0], rcfg["map_lpos"][1], rcfg["label"],
        fontsize=8, color=col, ha="center", va="center",
        fontweight="bold", zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5)
    )
    ax_map.tick_params(labelsize=7)
    ax_map.set_xlabel("Longitude", fontsize=7)
    ax_map.set_ylabel("Latitude",  fontsize=7)

    # ── Legend strip at bottom ─────────────────────────────────────────────
    lx = fig.add_axes([0.04, 0.015, 0.74, 0.04])
    lx.set_axis_off()
    x0 = 0.0
    lx.plot([x0, x0+0.032], [0.55, 0.55], color="orange", lw=2.0,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.036, 0.55, "MLD", fontsize=8, color="orange",
            va="center", transform=lx.transAxes)
    x0 += 0.09
    for lbl, method, thr, color, ls in CONFIGS:
        lx.plot([x0, x0+0.032], [0.55, 0.55], color=color, lw=1.8, ls=ls,
                transform=lx.transAxes, clip_on=False)
        lx.text(x0+0.036, 0.55, lbl, fontsize=7.5, color=color,
                va="center", transform=lx.transAxes)
        x0 += 0.148
    lx.plot([x0, x0+0.032], [0.55, 0.55], color="#c8c8c8", lw=1.5,
            transform=lx.transAxes, clip_on=False)
    lx.text(x0+0.036, 0.55, "Individual years (1993–2024)",
            fontsize=7.5, color="gray", va="center", transform=lx.transAxes)
    lx.text(0.86, 0.05, "Shaded bands = ±1 std across years",
            fontsize=7, color="gray", va="bottom", transform=lx.transAxes)

    out = OUT_DIR / f"profiles_{rkey}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → displays/{out.name}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for rkey, rcfg in REGIONS.items():
        plot_region(rkey, rcfg)
    print("\nDone.")
