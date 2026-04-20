from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


ROOT = Path(__file__).resolve().parent.parent

log("Loading datasets ...")
ds_025 = xr.open_dataset(ROOT / "data" / "025deg" / "mld_ild_bld_202312_prof_mayores_5m.nc")
ds_083 = xr.open_dataset(ROOT / "data" / "083deg" / "mld_ild_bld_083deg_202312.nc")

mean_025 = ds_025.mean(dim="time")
mean_083 = ds_083.mean(dim="time")

variables = [
    {"name": "mld", "label": "MLD (m)",  "vmin": 0,   "vmax": 100, "cmap": "Blues"},
    {"name": "ild", "label": "ILD (m)",  "vmin": 0,   "vmax": 150, "cmap": "Blues"},
    {"name": "bld", "label": "BLD (m)",  "vmin": -20, "vmax": 100, "cmap": "RdBu_r"},
]

projection = ccrs.Robinson()
transform  = ccrs.PlateCarree()

log("Plotting ...")
fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(18, 14),
    subplot_kw={"projection": projection},
)

col_titles = ["0.25°", "0.083°"]
datasets   = [mean_025, mean_083]

for row, var in enumerate(variables):
    for col, (ds, col_title) in enumerate(zip(datasets, col_titles)):
        ax = axes[row, col]

        data = ds[var["name"]].values
        lons = ds["longitude"].values
        lats = ds["latitude"].values

        img = ax.pcolormesh(
            lons, lats, data,
            vmin=var["vmin"], vmax=var["vmax"],
            cmap=var["cmap"],
            transform=transform,
            rasterized=True,
        )

        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
        ax.set_global()

        if row == 0:
            ax.set_title(col_title, fontsize=13, fontweight="bold", pad=8)

        if col == 0:
            ax.text(
                -0.04, 0.5, var["label"],
                va="center", ha="right",
                fontsize=12, rotation=90,
                transform=ax.transAxes,
            )

        cbar = fig.colorbar(img, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, aspect=30)
        cbar.ax.tick_params(labelsize=8)

fig.suptitle("Dec 21–30 2023 mean  |  0.25° vs 0.083°", fontsize=14, y=1.01)
plt.tight_layout()

output_path = ROOT / "displays" / "global_maps_bld_comparison_025_vs_083_dec2023.png"
fig.savefig(output_path, bbox_inches="tight", dpi=200)
log(f"Saved: {output_path}")
