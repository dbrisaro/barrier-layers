"""
plot_enso_composite.py
ENSO composite BLD anomaly maps — El Niño vs La Niña years (1993–2024).
Uses gradient −0.025°C/m config. Shows anomaly relative to 1993–2024 mean.
Outputs: displays/enso_composite_gradient_025.png
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "data" / "025deg" / "sensitivity"
OUT      = ROOT / "displays"

# NOAA ONI-based classification (calendar year of peak activity)
ELNINO_YEARS = [1994, 1997, 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2019, 2023]
LANINA_YEARS = [1995, 1999, 2000, 2007, 2008, 2010, 2011, 2012, 2017, 2020, 2021, 2022]

CONFIG = "gradient_025"
zarr_f = next(f for f in SENS_DIR.iterdir() if CONFIG in f.name)
ds     = xr.open_zarr(zarr_f)
bld    = ds["bld"].where(ds["bld"] > 0)
ann    = bld.resample(time="YE").mean("time").compute()

clim   = ann.mean("time")          # 1993–2024 mean
years  = ann.time.dt.year.values

def composite(yr_list):
    idx = [i for i, y in enumerate(years) if y in yr_list]
    return ann.isel(time=idx).mean("time") - clim

nino_anom = composite(ELNINO_YEARS)
nina_anom = composite(LANINA_YEARS)
diff_anom = nina_anom - nino_anom   # La Nina minus El Nino

lats = ann.latitude.values
lons = ann.longitude.values
LON2, LAT2 = np.meshgrid(lons, lats)

proj = ccrs.Robinson(); pc = ccrs.PlateCarree()

PANELS = [
    (nino_anom.values, "Anomalía El Niño",   8),
    (nina_anom.values, "Anomalía La Niña",   8),
    (diff_anom.values, "La Niña - El Niño", 12),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                         subplot_kw={"projection": proj})
fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.88, bottom=0.12)

ims = []
for ax, (data, label, vlim) in zip(axes, PANELS):
    ax.set_global()
    ax.add_feature(cfeature.LAND,      color="#d4d4d4", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4,   zorder=3)

    # Discrete colorbar: bins of 2 m
    bounds = np.arange(-vlim, vlim + 2, 2)
    cmap   = plt.cm.RdBu_r
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(LON2, LAT2, data, cmap=cmap, norm=norm,
                       transform=pc, zorder=1)
    ax.set_title(label, fontsize=12, pad=6)
    ims.append((im, bounds, vlim))
    for sp in ax.spines.values(): sp.set_visible(False)

# One colorbar per panel at bottom
for idx, (ax, (im, bounds, vlim)) in enumerate(zip(axes, ims)):
    cax = ax.inset_axes([0.1, -0.10, 0.8, 0.04])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("BLD anomalia (m)", fontsize=9)
    cb.set_ticks(bounds[::2])
    cb.ax.tick_params(labelsize=8)

# El Nino / La Nina year list as caption
nino_str = ", ".join(str(y) for y in sorted(ELNINO_YEARS))
nina_str = ", ".join(str(y) for y in sorted(LANINA_YEARS))
fig.text(0.5, 0.01,
         f"El Nino: {nino_str}\nLa Nina: {nina_str}",
         ha="center", va="bottom", fontsize=7.5, color="#444444")

out = OUT / f"enso_composite_{CONFIG}.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
