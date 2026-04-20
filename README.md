# Barrier Layers

Analysis of **Barrier Layer Depth (BLD)** in the global ocean using CMEMS reanalysis data (1993–2024), comparing two horizontal resolutions (0.25° and 0.083°) and six ILD detection methods.

---

## Physics

The **Barrier Layer** is a salinity-stratified layer that sits between the surface Mixed Layer and the deeper Isothermal Layer, inhibiting heat exchange between the ocean surface and the thermocline.

```
surface ─────────────────────────────────
           Mixed Layer (MLD)              ← turbulent, density-driven mixing
 ─────────────────────────────────────── ← MLD base
           Barrier Layer (BLD)            ← fresh + warm, salinity-stratified
 ─────────────────────────────────────── ← ILD base (thermocline top)
           Thermocline
```

**BLD = ILD − MLD**

A positive BLD means a barrier layer is present. Negative values (ILD < MLD) indicate no barrier layer.

---

## ILD Detection Methods

Two families of methods implemented in `src/barrier_layers.py`:

| Method | Definition | Thresholds used |
|---|---|---|
| **Gradient** | first depth z where dT/dz ≤ threshold | −0.015, −0.025, −0.1 °C/m |
| **Temperature difference** | first depth z where T(z) < T(ref) − ΔT | ΔT = 0.2, 0.5, 0.8 °C |

The gradient method is more sensitive to sharp thermoclines; the difference method is more robust in noisy profiles. Depths shallower than 5 m are excluded from ILD detection. The reference depth for the difference method is 10 m.

**Preferred methods for display:** gradient −0.025 °C/m and difference 0.2 °C.

---

## Data Sources (CMEMS)

| Dataset ID | Resolution | Period | Key variables |
|---|---|---|---|
| `cmems_mod_glo_phy-all_my_0.25deg_P1M-m` | **0.25° monthly** | 1993–2024 | `thetao_glor`, `so_glor`, `mlotst_glor` |
| `cmems_mod_glo_phy_my_0.083deg_P1M-m` | **0.083° monthly** | 1993–2024 | `thetao`, `so`, `mlotst` |
| `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` | 0.083° NRT | 2021–present | `thetao` |
| `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | 0.083° NRT | 2021–present | `mlotst`, `so` |

**Vertical grids:**
- LR (0.25°): 74 levels total, 34 in the 0–300 m range
- HR (0.083°): 32 levels total, 28 in the 0–300 m range
- Near the surface (0–50 m) the two grids are nearly identical; below 50 m LR is slightly finer than HR.

---

## Project Structure

```
barrier_layers/
├── src/
│   └── barrier_layers.py                   # core computation module
├── scripts/                                # all runnable analysis scripts
├── displays/                               # output figures (~78 PNGs)
├── data/
│   ├── 025deg/                             # LR global BLD + regional profiles + sensitivity
│   └── 083deg/                             # HR global BLD + regional profiles
├── reports/
│   ├── barrier_layers_beamer.tex           # LaTeX Beamer presentation (main deliverable)
│   ├── barrier_layers_beamer.pdf           # compiled presentation (45 slides)
│   ├── barrier_layers_results.pdf          # earlier presentation (reference)
│   ├── sensitivity_summary.md             # sensitivity analysis narrative
│   ├── lr_hr_comparison.md               # LR vs HR comparison narrative
│   └── trend_variability.md              # trends and variability narrative
├── notebooks/                              # exploratory Jupyter notebooks
├── docs/                                   # reference literature (PDFs)
└── README.md
```

---

## Module: `src/barrier_layers.py`

| Function | Description |
|---|---|
| `ild_from_temp_profile(temp_1d, depth_1d, method, threshold)` | ILD detection on a single 1D profile; Dask-compatible via `xr.apply_ufunc` |
| `compute_global_bld(ds, temp_var, mld_var, method, threshold)` | Applies ILD detection globally; returns `xr.Dataset` with `mld`, `ild`, `bld` |
| `compute_segment_gradient(temp, depth)` | dT/dz per depth interval; returns `(depth_mid, gradient)` |
| `first_depth_below_threshold(gradient, depth_mid, threshold)` | First depth where gradient ≤ threshold |

---

## Scripts

### Computation

| Script | What it does | Output |
|---|---|---|
| `compute_all_years_025deg.py` | Global monthly BLD at 0.25°, 1993–2024 | `data/025deg/bld_025deg_monthly_1993_2024.zarr` |
| `compute_bld_083deg_2024.py` | Global monthly BLD at 0.083° for 2024 | `data/083deg/bld_083deg_monthly_2024.zarr` |
| `compute_year_083deg.py` | Global monthly BLD at 0.083° for any year | `data/083deg/bld_083deg_monthly_{year}.zarr` |
| `sensitivity_analysis_025deg.py` | BLD for all 6 ILD configurations, LR 1993–2024 | `data/025deg/sensitivity/bld_{config}_1993_2024.zarr` (×6) |
| `compute_global_025deg_dec2023.py` | Daily BLD at 0.25° for Dec 2023 (diagnostic) | `data/*.nc` |
| `compute_global_083deg_dec2023.py` | Daily BLD at 0.083° for Dec 2023 (diagnostic) | `data/083deg/*.nc` |

### Visualisation — LR climatology and sensitivity

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_bld_summary.py` | Global BLD maps, time series, coverage stats | `summary_bld_025deg_monthly_1993_2024.png` |
| `plot_monthly_climatology.py` | Monthly climatology maps (Jan–Dec) for each ILD method | `monthly_clim_{method}.png` (×6) + `monthly_clim_seasonal_comparison.png` |
| `regen_clim_shared_vmax.py` | Regenerate climatology with shared vmax across methods | `monthly_clim_difference_02.png`, `monthly_clim_gradient_025.png` |
| `compare_sensitivity.py` | Ensemble statistics: mean/spread/agreement maps + time series | `sensitivity_ensemble_maps.png`, `sensitivity_global_timeseries.png`, `sensitivity_regional_timeseries.png`, `sensitivity_stats_table.png` |
| `sensitivity_one_point.py` | All 6 ILD methods on a single downloaded profile | `sensitivity_one_point_{date}_lat{lat}_lon{lon}.png` |
| `plot_method_disagreement_profiles.py` | Agreement map (% methods detecting BLD > 0) + T/S profiles for intermediate-agreement zones by region | `method_agreement_map.png`, `method_agreement_{region}.png` (×4) |

### Visualisation — trends, variability, ENSO

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_bld_trends.py` | Per-pixel linear trend (m/decade), regional trend series, ensemble spread | `bld_trend_map.png`, `bld_trend_regions.png`, `bld_trend_ensemble.png` |
| `plot_trend_maps_v2.py` | Trend maps with discrete colorbars; "all" and significance-masked versions | `bld_trend_map_{method}_all.png`, `bld_trend_map_{method}_sig.png` (×2 methods) |
| `plot_bld_variability.py` | Total std, seasonal amplitude, interannual std, annual anomaly maps | `bld_variability_maps.png`, `bld_anomaly_maps.png` |
| `plot_enso_composite.py` | BLD anomaly composites for El Niño and La Niña years | `enso_composite_gradient_025.png` |

### Visualisation — regional profiles

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_region_profiles.py` | Monthly T/S profiles for Amazon Plume, Bay of Bengal, Southern Ocean; individual years + climatology + 6 ILD methods | `profiles_{region}.png` (×3) |
| `plot_amazon_plume_profiles.py` | Amazon Plume monthly T/S profile time series (4×3 grid) | `amazon_plume_monthly_profiles.png` |

### Visualisation — LR vs HR comparison

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `compare_lr_hr.py` | Global annual mean maps (LR / HR / difference), 4 regional zooms, monthly time series | `compare_lr_hr_annual_mean.png`, `compare_lr_hr_zoom_regions.png`, `compare_lr_hr_monthly_stats.png` |
| `regen_zoom_regions.py` | Regenerate zoom comparison (2 rows LR/HR × 4 regions) | `compare_lr_hr_zoom_regions.png` |
| `compare_lr_hr_profiles.py` | T/S profile overlay LR vs HR — 3 regions × 6 ILD methods | `compare_profiles_{region}_{method}.png` (×18) |
| `compare_lr_hr_resolvability.py` | Heat-map ΔILD/grid-spacing per method × month; profiles with ±½ spacing bands | `resolvability_{region}.png` (×3), `compare_profiles_{region}_{method}_with_uncertainty.png` (×9) |
| `scatter_lr_hr.py` | Scatter LR vs HR: T/S profiles per depth + global MLD/ILD/BLD per basin | `scatter_profiles_lr_hr.png`, `scatter_bld_lr_hr.png` |

---

## Data Cache

### LR 0.25° — `data/025deg/`

| File | Contents | Period |
|---|---|---|
| `bld_025deg_monthly_1993_2024.zarr` | Global `mld`, `ild`, `bld` — (384 × 681 × 1440) | 1993–2024 |
| `amazon_plume_monthly_1993_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — Amazon Plume | 1993–2024 |
| `profiles_bay_of_bengal_1993_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — Bay of Bengal | 1993–2024 |
| `profiles_southern_ocean_1993_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — Southern Ocean | 1993–2024 |
| `profiles_wpwp_lr_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — W. Pacific Warm Pool | 2024 only |
| `sensitivity/bld_{config}_1993_2024.zarr` | `ild`, `bld`, `mld` for each of 6 configurations | 1993–2024 |
| `agreement_map_cache.npz` | Pre-computed agreement map (% methods with BLD > 0) — cached for speed | 1993–2024 |

### HR 0.083° — `data/083deg/`

| File | Contents | Period |
|---|---|---|
| `bld_083deg_monthly_2024.zarr` | Global `mld`, `ild`, `bld` — (12 × 2041 × 4320) | 2024 |
| `bld_083deg_monthly_2025.zarr` | Global `mld`, `ild`, `bld` — (12 × 2041 × 4320) | 2025 |
| `profiles_amazon_plume_hr_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — Amazon Plume | 2024 |
| `profiles_bay_of_bengal_hr_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — Bay of Bengal | 2024 |
| `profiles_wpwp_hr_2024.zarr` | Spatially-averaged `temp`, `sal`, `mld` — W. Pacific Warm Pool | 2024 |

### Analysis regions (bounding boxes)

| Region | Latitude | Longitude | Depth max |
|---|---|---|---|
| Amazon Plume | 5°S – 10°N | 55°W – 30°W | 300 m |
| Bay of Bengal | 5°N – 25°N | 80°E – 100°E | 300 m |
| W. Pacific Warm Pool (WPWP) | 10°S – 10°N | 130°E – 180°E | 300 m |
| Southern Ocean (Indian sector) | 65°S – 45°S | 20°E – 100°E | 600 m |
| Tropical Indian Ocean | 20°S – 20°N | 40°E – 100°E | 300 m |

---

## Key Findings

### LR vs HR comparison (2024)

1. **Raw T/S profiles agree very well**: R > 0.997 for temperature, R > 0.978 for salinity across all regions. HR is slightly warmer than LR (bias < 0.3 °C) with no strong depth dependence.

2. **Vertical grids are nearly identical**: Near the surface (0–50 m) both resolutions have the same spacing. Below 50 m, LR (74 levels, 34 in 0–300 m) is actually slightly finer than HR (32 levels, 28 in 0–300 m). Finer HR vertical resolution is **not** an explanation for BLD differences.

3. **Most ILD differences are within grid quantization**: For most month/region/method combinations, |ILD_HR − ILD_LR| ≤ maximum grid spacing at that depth. Only in specific cases (Amazon Plume Aug–Oct with gradient −0.1; Bay of Bengal Apr–May) is the difference genuinely above the grid-spacing limit. See `resolvability_{region}.png`.

4. **Global scatter (MLD/ILD/BLD)**: HR MLD is slightly deeper than LR (bias +3.0 m, RMSE 13.4 m, R = 0.90). The Southern Ocean shows the largest disagreement (MLD bias +8.8 m, BLD R = 0.57); the Tropical Indian Ocean shows the best agreement (BLD R = 0.88).

### Trends and variability (LR 1993–2024)

5. **Significant positive BLD trends** exist in parts of the western Pacific and Indian Ocean warm pool; negative trends appear in the eastern Pacific. Trend magnitudes are typically 1–5 m/decade.

6. **Inter-method spread** in trends is large: the sign of the trend is not consistent across all 6 methods, especially in regions with weak climatological BLD. Results should be interpreted with caution outside the high-agreement zones.

### Sensitivity and inter-method agreement

7. **Agreement map** (% of 6 methods detecting BLD > 0): the W. Pacific Warm Pool and Bay of Bengal show very high agreement (>95%); the Southern Ocean is the most heterogeneous (median ~37%). The Amazon Plume shows the widest intra-regional spread.

8. **In intermediate-agreement zones**, the 6 ILD methods diverge most strongly in months with weak thermoclines or noisy T profiles (visible in the dT/dz panel of `method_agreement_{region}.png`).

### ENSO signal

9. **El Niño** composites show positive BLD anomalies in the central/eastern Pacific (deepened ILD, shoaled MLD) and negative anomalies in the Maritime Continent. **La Niña** shows the reverse pattern. The La Niña − El Niño difference reaches ±30 m in the western Pacific. See `enso_composite_gradient_025.png`.

---

## How to Run

```bash
# Activate the project environment
conda activate /Users/daniela/Documents/barrier_layers/.conda_arm   # ARM64 nativo (sin Rosetta)

# Compute LR BLD for all years (long — run in tmux)
python scripts/compute_all_years_025deg.py

# Sensitivity analysis — 6 ILD configurations, LR 1993–2024
python scripts/sensitivity_analysis_025deg.py

# Figures — climatology, trends, variability
python scripts/plot_monthly_climatology.py
python scripts/plot_bld_trends.py
python scripts/plot_trend_maps_v2.py
python scripts/plot_bld_variability.py
python scripts/plot_enso_composite.py

# Regional T/S profiles
python scripts/plot_region_profiles.py           # all regions
python scripts/plot_region_profiles.py bay_of_bengal  # single region

# Sensitivity and inter-method agreement
python scripts/compare_sensitivity.py
python scripts/plot_method_disagreement_profiles.py  # uses cached agreement_map_cache.npz

# LR vs HR comparison (requires HR data)
python scripts/compare_lr_hr.py
python scripts/compare_lr_hr_profiles.py
python scripts/scatter_lr_hr.py

# Compile the Beamer presentation
cd reports
pdflatex -interaction=nonstopmode barrier_layers_beamer.tex
pdflatex -interaction=nonstopmode barrier_layers_beamer.tex  # second pass for references
```

---

## Dependencies

```
xarray         zarr           numpy          pandas
matplotlib     scipy          shapefile      copernicusmarine
```

> **Note:** `cartopy` is installed in the project conda environment but is not called directly in most scripts; coastlines are rendered via `pyshp` (`shapefile`) which avoids Rosetta compatibility issues on Apple Silicon.

Install with:
```bash
pip install xarray zarr numpy pandas matplotlib scipy pyshp copernicusmarine
```
