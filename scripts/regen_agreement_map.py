"""
regen_agreement_map_v2.py
Regenerates displays/method_agreement_map.png from the cached agreement_map_cache.npz.
Changes vs previous version:
  - "Amazon Plume" → "Tropical Atlantic"
  - Added N. Argentine Shelf box
  - All labels in English
  - Helvetica Neue font
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import shapefile as shp

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
})

ROOT    = Path("/Users/daniela/Documents/barrier_layers")
OUT_DIR = ROOT / "displays"
CACHE   = ROOT / "data" / "025deg" / "agreement_map_cache.npz"

NE_COAST = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"
NE_LAND  = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"

# Load cached lat/lon from a reference zarr
import xarray as xr
_ref = xr.open_zarr(ROOT / "data/025deg/sensitivity/bld_gradient_025_1993_2024.zarr",
                    consolidated=False)
lat = _ref["latitude"].values
lon = _ref["longitude"].values
del _ref

cache_data    = np.load(CACHE)
agreement_map = cache_data["agreement_map"]
print(f"Loaded agreement_map {agreement_map.shape}  "
      f"median={np.nanmedian(agreement_map):.1f}%")

# ── Region boxes ──────────────────────────────────────────────────────────────
REGIONS = {
    "tropical_atlantic": dict(
        label   = "Tropical\nAtlantic",
        lat_min = -5,  lat_max = 10,
        lon_min = -55, lon_max = -30,
        color   = "#e74c3c",
        lpos    = (-42.5, -7.5),   # label below box
    ),
    "bay_of_bengal": dict(
        label   = "Bay of\nBengal",
        lat_min = 5,   lat_max = 25,
        lon_min = 80,  lon_max = 100,
        color   = "#8e44ad",
        lpos    = (90, 3),
    ),
    "southern_ocean": dict(
        label   = "Southern Ocean\n(Indian sector)",
        lat_min = -65, lat_max = -45,
        lon_min = 20,  lon_max = 100,
        color   = "#2980b9",
        lpos    = (60, -70),
    ),
    "wpwp": dict(
        label   = "W. Pacific\nWarm Pool",
        lat_min = -10, lat_max = 10,
        lon_min = 130, lon_max = 180,
        color   = "#e67e22",
        lpos    = (155, -14),
    ),
    "arg_sea_north": dict(
        label   = "N. Argentine\nShelf",
        lat_min = -45, lat_max = -30,
        lon_min = -65, lon_max = -50,
        color   = "#16a085",
        lpos    = (-57.5, -48),
    ),
}

# ── Coastlines helper ─────────────────────────────────────────────────────────
def draw_coast(ax, land_color="#e0e0e0", coast_lw=0.4):
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
                ax.plot(seg[:, 0], seg[:, 1], lw=coast_lw, color="#555555", zorder=3)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")

img = ax.pcolormesh(lon, lat, agreement_map,
                    vmin=0, vmax=100, cmap="RdYlGn", rasterized=True)
draw_coast(ax)

cb = fig.colorbar(img, ax=ax, pad=0.01, shrink=0.85, aspect=30)
cb.set_label("Method agreement  (% of methods detecting BLD > 0)", fontsize=9)
cb.ax.tick_params(labelsize=8)

for rkey, rcfg in REGIONS.items():
    col = rcfg["color"]
    # Filled rectangle (semi-transparent)
    ax.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor=col, facecolor=col, alpha=0.18, zorder=4
    ))
    # Outline only (crisp)
    ax.add_patch(mpatches.Rectangle(
        (rcfg["lon_min"], rcfg["lat_min"]),
        rcfg["lon_max"] - rcfg["lon_min"],
        rcfg["lat_max"] - rcfg["lat_min"],
        lw=2, edgecolor=col, facecolor="none", zorder=5
    ))
    # Label
    ax.text(rcfg["lpos"][0], rcfg["lpos"][1], rcfg["label"],
            fontsize=7, color=col, ha="center", va="top",
            fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1))

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_aspect("equal")
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude",  fontsize=9)
ax.tick_params(labelsize=8)

out = OUT_DIR / "method_agreement_map.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
