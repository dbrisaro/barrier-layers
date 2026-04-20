"""
scatter_lr_hr.py

Scatter plots comparing LR (0.25°) vs HR (0.083°) CMEMS products for 2024.

Figure 1 — T/S profile scatter per region and depth level:
    2 rows (T, S) × 3 cols (amazon_plume, bay_of_bengal, wpwp)
    Each point = one (month, depth) pair; color = depth level
    Shows bias and RMSE in raw profiles.

Figure 2 — Global MLD/ILD/BLD scatter per basin:
    3 cols (MLD, ILD, BLD); color = ocean basin
    HR regridded to LR 0.25° grid (annual mean, 2024)
    Shows whether global differences are basin-dependent.

Usage:
    python scripts/scatter_lr_hr.py
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LR_CACHE_DIR = ROOT / "data" / "025deg"
HR_CACHE_DIR = ROOT / "data" / "083deg"
DISP_DIR     = ROOT / "displays"
DISP_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = {
    "amazon_plume":  "Amazon Plume\n(W. Atlantic)",
    "bay_of_bengal": "Bay of Bengal\n(Indian)",
    "wpwp":          "W. Pacific Warm Pool\n(Pacific)",
}

LR_PATHS = {
    "amazon_plume":  LR_CACHE_DIR / "amazon_plume_monthly_1993_2024.zarr",
    "bay_of_bengal": LR_CACHE_DIR / "profiles_bay_of_bengal_1993_2024.zarr",
    "wpwp":          LR_CACHE_DIR / "profiles_wpwp_lr_2024.zarr",
}

HR_PATHS = {
    "amazon_plume":  HR_CACHE_DIR / "profiles_amazon_plume_hr_2024.zarr",
    "bay_of_bengal": HR_CACHE_DIR / "profiles_bay_of_bengal_hr_2024.zarr",
    "wpwp":          HR_CACHE_DIR / "profiles_wpwp_hr_2024.zarr",
}

MAX_DEPTH = 300.0  # only compare upper 300 m (where BLD is detected)

# ── Ocean basin mask ────────────────────────────────────────────────────────────
BASIN_COLORS = {
    "Southern": "#4575b4",
    "Atlantic": "#d73027",
    "Indian":   "#fe9929",
    "Pacific":  "#41b6c4",
}

def basin_label(lat, lon):
    lon360 = float(lon) % 360
    if lat < -45:
        return "Southern"
    elif 20 < lon360 < 120:
        return "Indian"
    elif (lon360 > 240) or (lon360 <= 20):
        return "Atlantic"
    else:
        return "Pacific"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: T/S profile scatter per region (depth as color)
# ─────────────────────────────────────────────────────────────────────────────

def load_profiles_2024(region):
    lr = xr.open_zarr(LR_PATHS[region])
    hr = xr.open_zarr(HR_PATHS[region])

    lr_2024 = lr.sel(time=lr.time.dt.year == 2024)
    lr_2024 = lr_2024.sel(depth=lr_2024.depth <= MAX_DEPTH)
    hr      = hr.sel(depth=hr.depth <= MAX_DEPTH)

    return lr_2024, hr


def interp_lr_to_depths(lr_vals, lr_depth, target_depth):
    """
    Interpolate LR profile (n_time × n_depth_lr) onto target depth grid.
    Returns (n_time × n_target).
    """
    out = np.full((lr_vals.shape[0], len(target_depth)), np.nan)
    for t in range(lr_vals.shape[0]):
        mask = np.isfinite(lr_vals[t])
        if mask.sum() < 2:
            continue
        out[t] = np.interp(target_depth, lr_depth[mask], lr_vals[t][mask],
                           left=np.nan, right=np.nan)
    return out


def plot_profile_scatter():
    print("Building T/S profile scatter...")

    depth_cmap = plt.cm.plasma_r
    depth_norm = mcolors.Normalize(vmin=5, vmax=MAX_DEPTH)

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.08, wspace=0.10,
                            left=0.07, right=0.93,
                            top=0.96, bottom=0.09,
                            width_ratios=[1, 1, 1, 0.06])

    for col, (region, region_label) in enumerate(REGIONS.items()):
        lr_2024, hr = load_profiles_2024(region)

        lr_depth = lr_2024.depth.values.astype(float)
        hr_depth = hr.depth.values.astype(float)

        for row, var in enumerate(["temp", "sal"]):
            ax = fig.add_subplot(gs[row, col])

            lr_vals = lr_2024[var].values  # (12, n_lr_depth)
            hr_vals = hr[var].values       # (12, n_hr_depth)

            # Interpolate LR to HR depth grid for fair comparison
            lr_interp = interp_lr_to_depths(lr_vals, lr_depth, hr_depth)

            # Flatten to 1D
            depths_flat = np.tile(hr_depth, 12)
            lr_flat     = lr_interp.ravel()
            hr_flat     = hr_vals.ravel()

            valid   = np.isfinite(lr_flat) & np.isfinite(hr_flat)
            lr_plot = lr_flat[valid]
            hr_plot = hr_flat[valid]
            d_plot  = depths_flat[valid]

            # Sort by depth for colorbar continuity
            order   = np.argsort(d_plot)
            sc = ax.scatter(lr_plot[order], hr_plot[order],
                            c=d_plot[order], cmap=depth_cmap, norm=depth_norm,
                            s=10, alpha=0.65, linewidths=0, rasterized=True)

            # 1:1 line
            lo = min(lr_plot.min(), hr_plot.min())
            hi = max(lr_plot.max(), hr_plot.max())
            ax.plot([lo, hi], [lo, hi], "k-", lw=1.2, zorder=5)

            # Stats
            bias = float(np.mean(hr_plot - lr_plot))
            rmse = float(np.sqrt(np.mean((hr_plot - lr_plot) ** 2)))
            r, _ = pearsonr(lr_plot, hr_plot)
            unit = "°C" if var == "temp" else "PSU"

            ax.text(0.04, 0.97,
                    f"bias = {bias:+.4f} {unit}\nRMSE = {rmse:.4f} {unit}\nR = {r:.5f}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="none", alpha=0.7))

            # Labels
            var_label = "Temperature (°C)" if var == "temp" else "Salinity (PSU)"
            # region label removed — kept in slide title/caption
            if col == 0:
                ax.set_ylabel(f"HR 0.083°\n{var_label}", fontsize=8.5)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("LR 0.25°", fontsize=8.5)
            ax.tick_params(labelsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Shared colorbar
    cax = fig.add_subplot(gs[:, 3])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(norm=depth_norm, cmap=depth_cmap),
        cax=cax, orientation="vertical"
    )
    cb.set_label("Depth (m)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = DISP_DIR / "scatter_profiles_lr_hr.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Global MLD / ILD / BLD scatter per ocean basin
# ─────────────────────────────────────────────────────────────────────────────

def plot_bld_scatter():
    print("Loading global BLD data...")
    lr_bld = xr.open_zarr(LR_CACHE_DIR / "bld_025deg_monthly_1993_2024.zarr")
    hr_bld = xr.open_zarr(HR_CACHE_DIR / "bld_083deg_monthly_2024.zarr")

    # 2024 annual mean
    lr_2024 = lr_bld.sel(time=lr_bld.time.dt.year == 2024).mean("time")
    hr_mean = hr_bld.mean("time")

    print("Regridding HR → LR grid (nearest)...")
    hr_on_lr = hr_mean.interp(
        latitude=lr_2024.latitude,
        longitude=lr_2024.longitude,
        method="nearest"
    )

    # Basin labels per grid cell
    lat_vals = lr_2024.latitude.values
    lon_vals = lr_2024.longitude.values
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    basins = np.vectorize(basin_label)(lat2d, lon2d)  # (lat, lon)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.13, wspace=0.32)

    variables  = ["mld",         "ild",         "bld"]
    var_labels = {"mld": "MLD (m)", "ild": "ILD (m)", "bld": "BLD (m)"}
    var_lims   = {"mld": (0, 200),  "ild": (0, 200),  "bld": (-20, 120)}

    rng = np.random.default_rng(42)

    for ax, var in zip(axes, variables):
        lr_vals = lr_2024[var].values.ravel()   # (n_lat × n_lon,)
        hr_vals = hr_on_lr[var].values.ravel()
        bas     = basins.ravel()

        vmin, vmax = var_lims[var]

        all_lr, all_hr = [], []

        for basin_name, color in BASIN_COLORS.items():
            mask = (bas == basin_name) & np.isfinite(lr_vals) & np.isfinite(hr_vals)
            if mask.sum() < 10:
                continue
            lr_b = lr_vals[mask]
            hr_b = hr_b_full = hr_vals[mask]

            # Keep only within plotting range
            inrange = (lr_b >= vmin) & (lr_b <= vmax) & (hr_b >= vmin) & (hr_b <= vmax)
            lr_b = lr_b[inrange]
            hr_b = hr_b[inrange]

            all_lr.append(lr_b)
            all_hr.append(hr_b)

            # Subsample to avoid overplotting (max 6000 pts per basin)
            n = min(6000, len(lr_b))
            idx = rng.choice(len(lr_b), n, replace=False)

            ax.scatter(lr_b[idx], hr_b[idx], c=color, s=5, alpha=0.20,
                       linewidths=0, label=basin_name, rasterized=True)

        # 1:1 line
        ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=1.5, zorder=6)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("LR 0.25°", fontsize=9.5)
        ax.set_ylabel("HR 0.083°", fontsize=9.5)
        ax.set_title(var_labels[var], fontsize=11, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8.5)

        # Global stats
        all_lr_arr = np.concatenate(all_lr)
        all_hr_arr = np.concatenate(all_hr)
        bias_g = float(np.nanmean(all_hr_arr - all_lr_arr))
        rmse_g = float(np.sqrt(np.nanmean((all_hr_arr - all_lr_arr) ** 2)))
        r_g, _ = pearsonr(all_lr_arr, all_hr_arr)
        ax.text(0.04, 0.97,
                f"bias = {bias_g:+.2f} m\nRMSE = {rmse_g:.2f} m\nR = {r_g:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # Per-basin legend (opaque markers)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=b)
        for b, c in BASIN_COLORS.items()
    ]
    axes[-1].legend(handles=legend_handles, fontsize=8.5, loc="lower right",
                    title="Basin", title_fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "LR 0.25° vs HR 0.083°  —  Global MLD / ILD / BLD scatter  —  2024 annual mean\n"
        "HR regridded to LR 0.25° grid;  each point = one 0.25° grid cell;  color = ocean basin",
        fontsize=10.5, fontweight="bold"
    )

    out = DISP_DIR / "scatter_bld_lr_hr.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_profile_scatter()
    plot_bld_scatter()
