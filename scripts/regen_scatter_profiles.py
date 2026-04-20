"""Regenerate scatter_profiles_lr_hr.png with style fixes."""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DISP_DIR = ROOT / "displays"
LR_DIR   = ROOT / "data/025deg"
HR_DIR   = ROOT / "data/083deg"
MAX_DEPTH = 300.0

REGIONS = {
    "amazon_plume":  "Amazon Plume",
    "bay_of_bengal": "Bahia de Bengala",
    "wpwp":          "W. Pacific Warm Pool",
}
LR_PATHS = {
    "amazon_plume":  LR_DIR / "amazon_plume_monthly_1993_2024.zarr",
    "bay_of_bengal": LR_DIR / "profiles_bay_of_bengal_1993_2024.zarr",
    "wpwp":          LR_DIR / "profiles_wpwp_lr_2024.zarr",
}
HR_PATHS = {
    "amazon_plume":  HR_DIR / "profiles_amazon_plume_hr_2024.zarr",
    "bay_of_bengal": HR_DIR / "profiles_bay_of_bengal_hr_2024.zarr",
    "wpwp":          HR_DIR / "profiles_wpwp_hr_2024.zarr",
}

def interp_lr(vals, lr_d, tgt_d):
    out = np.full((vals.shape[0], len(tgt_d)), np.nan)
    for t in range(vals.shape[0]):
        m = np.isfinite(vals[t])
        if m.sum() < 2: continue
        out[t] = np.interp(tgt_d, lr_d[m], vals[t][m], left=np.nan, right=np.nan)
    return out

depth_cmap = plt.cm.plasma_r
depth_norm = mcolors.Normalize(vmin=5, vmax=MAX_DEPTH)

fig = plt.figure(figsize=(17, 10), facecolor="white")
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.20, wspace=0.10,
                        left=0.07, right=0.93,
                        top=0.96, bottom=0.09,
                        width_ratios=[1, 1, 1, 0.06])

for col, (region, label) in enumerate(REGIONS.items()):
    print(f"  {region}...", flush=True)
    lr      = xr.open_zarr(LR_PATHS[region])
    hr      = xr.open_zarr(HR_PATHS[region])
    lr_2024 = lr.sel(time=lr.time.dt.year == 2024)
    lr_2024 = lr_2024.sel(depth=lr_2024.depth <= MAX_DEPTH)
    hr      = hr.sel(depth=hr.depth <= MAX_DEPTH)
    lr_d    = lr_2024.depth.values.astype(float)
    hr_d    = hr.depth.values.astype(float)

    for row, var in enumerate(["temp", "sal"]):
        ax          = fig.add_subplot(gs[row, col])
        lr_interp   = interp_lr(lr_2024[var].values, lr_d, hr_d)
        depths_flat = np.tile(hr_d, 12)
        lr_flat     = lr_interp.ravel()
        hr_flat     = hr[var].values.ravel()
        valid       = np.isfinite(lr_flat) & np.isfinite(hr_flat)
        lp, hp, dp  = lr_flat[valid], hr_flat[valid], depths_flat[valid]
        order       = np.argsort(dp)

        ax.scatter(lp[order], hp[order], c=dp[order],
                   cmap=depth_cmap, norm=depth_norm,
                   s=10, alpha=0.65, linewidths=0, rasterized=True)
        lo, hi = min(lp.min(), hp.min()), max(lp.max(), hp.max())
        ax.plot([lo, hi], [lo, hi], "k-", lw=1.2, zorder=5)

        bias = float(np.mean(hp - lp))
        rmse = float(np.sqrt(np.mean((hp - lp) ** 2)))
        r, _ = pearsonr(lp, hp)
        unit = "C" if var == "temp" else "PSU"
        ax.text(0.04, 0.97,
                f"bias = {bias:+.4f} {unit}\nRMSE = {rmse:.4f} {unit}\nR = {r:.5f}",
                transform=ax.transAxes, fontsize=7.5, va="top")

        var_label = "Temperatura (C)" if var == "temp" else "Salinidad (PSU)"
        if row == 0:
            ax.set_title(label, fontsize=9.5, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"HR 0.083\n{var_label}", fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("LR 0.25", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

cax = fig.add_subplot(gs[:, 3])
cb  = plt.colorbar(plt.cm.ScalarMappable(norm=depth_norm, cmap=depth_cmap),
                   cax=cax, orientation="vertical")
cb.set_label("Profundidad (m)", fontsize=9)
cb.ax.tick_params(labelsize=8)

out = DISP_DIR / "scatter_profiles_lr_hr.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out.name}  ({out.stat().st_size // 1024} KB)")
