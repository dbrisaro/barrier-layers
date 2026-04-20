# Barrier Layers

Analysis of **Barrier Layer Depth (BLD)** in the global ocean using CMEMS GLORYS12 reanalysis (1993–2024), comparing six ILD detection methods with a sensitivity analysis across threshold choices.

---

## Physics

The **Barrier Layer** is a salinity-stratified layer between the surface Mixed Layer and the deeper Isothermal Layer, inhibiting heat exchange between the ocean surface and the thermocline.

```
surface ─────────────────────────────────
           Mixed Layer (MLD)              ← turbulent, density-driven mixing
 ─────────────────────────────────────── ← MLD base
           Barrier Layer (BLD)            ← fresh + warm, salinity-stratified
 ─────────────────────────────────────── ← ILD base (thermocline top)
           Thermocline
```

**BLD = ILD − MLD**

A positive BLD indicates a barrier layer is present. Negative values (ILD < MLD) indicate no barrier layer.

---

## ILD Detection Methods

Two families of methods implemented in `src/barrier_layers.py`, both using **absolute values** (`use_abs=True`) to handle temperature inversions (important in the Southern Ocean):

| Method | Definition | Thresholds used |
|---|---|---|
| **Gradient** | first depth z > 5 m where \|dT/dz\| ≥ \|threshold\| | −0.015, −0.025, −0.1 °C/m |
| **Temperature difference** | first depth z > 10 m where \|T(z) − T(10m)\| > ΔT | ΔT = 0.2, 0.5, 0.8 °C |

Using absolute values instead of signed differences ensures that temperature inversions (e.g. cold fresh water overlying warmer saltier water, common in the Southern Ocean in winter) are correctly detected rather than returning NaN.

**MLD source:** `mlotst_glor` from GLORYS12, computed by the model using a density criterion Δσ₀ = 0.03 kg/m³ referenced to 10 m depth (TEOS-10).

---

## Data Sources (CMEMS)

| Dataset ID | Resolution | Period | Key variables |
|---|---|---|---|
| `cmems_mod_glo_phy-all_my_0.25deg_P1M-m` | **0.25° monthly** | 1993–2024 | `thetao_glor`, `so_glor`, `mlotst_glor` |
| `cmems_mod_glo_phy_my_0.083deg_P1M-m` | **0.083° monthly** | 1993–2024 | `thetao`, `so`, `mlotst` |

**Vertical grids:**
- LR (0.25°): 74 levels total, 34 in the 0–300 m range
- HR (0.083°): 32 levels total, 28 in the 0–300 m range

---

## Project Structure

```
barrier_layers/
├── src/
│   └── barrier_layers.py                   # core computation module
├── scripts/                                # all runnable analysis scripts
├── data/                                   # not tracked (12 GB zarr stores)
│   ├── 025deg/
│   │   ├── sensitivity/                    # 6 × bld_{config}_1993_2024.zarr
│   │   └── *.zarr                          # regional profile caches
│   └── 083deg/
├── displays/                               # not tracked (regenerable from scripts)
├── reports/
│   ├── barrier_layers_beamer.tex           # LaTeX Beamer presentation
│   ├── barrier_layers_beamer_part2.tex     # part 2 (sensitivity + trends)
│   ├── sensitivity_summary.md
│   ├── lr_hr_comparison.md
│   └── trend_variability.md
├── notebooks/                              # exploratory Jupyter notebooks
└── README.md
```

---

## Module: `src/barrier_layers.py`

| Function | Description |
|---|---|
| `ild_from_temp_profile(temp_1d, depth_1d, method, threshold, use_abs)` | ILD detection on a single 1D profile |
| `compute_global_bld(ds, temp_var, mld_var, method, threshold, use_abs)` | Applies ILD detection globally via `xr.apply_ufunc`; returns `xr.Dataset` with `mld`, `ild`, `bld` |
| `compute_segment_gradient(temp, depth)` | dT/dz per depth interval |
| `first_depth_below_threshold(gradient, depth_mid, threshold)` | First depth where gradient ≤ threshold |

---

## Scripts

### Computation

| Script | What it does | Output |
|---|---|---|
| `sensitivity_analysis_025deg.py` | Global BLD for all 6 ILD configs, 1993–2024 (`use_abs=True`) | `data/025deg/sensitivity/bld_{config}_1993_2024.zarr` (×6) |
| `compute_all_years_025deg.py` | Global monthly BLD at 0.25° (default config) | `data/025deg/bld_025deg_monthly_1993_2024.zarr` |
| `compute_bld_083deg_2024.py` | Global monthly BLD at 0.083° | `data/083deg/bld_083deg_monthly_{year}.zarr` |

### Visualisation — sensitivity and agreement

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_method_disagreement_profiles.py` | Agreement map (% methods detecting BLD > 0) with 5 region boxes; T/S profiles for intermediate-agreement zones | `method_agreement_map.png` |
| `compare_sensitivity.py` | Ensemble statistics: mean/spread/agreement maps + time series | `sensitivity_ensemble_maps.png`, `sensitivity_*_timeseries.png` |
| `sensitivity_one_point.py` | All 6 ILD methods on a single profile | `sensitivity_one_point_*.png` |

### Visualisation — trends (ILD / MLD / BLD decomposition)

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_ild_mld_bld_trends.py` | ILD, MLD and BLD trend maps (1993–2024); regional time series; ILD trend per method (2×3 panel); ensemble mean + spread | `trends_ild_mld_bld_global.png`, `trends_ild_mld_bld_regional.png`, `trends_ild_per_method.png`, `trends_ild_ensemble.png` |
| `plot_bld_variability.py` | Total std, seasonal amplitude, interannual std | `bld_variability_maps.png` |
| `plot_enso_composite.py` | BLD anomaly composites for El Niño / La Niña | `enso_composite_gradient_025.png` |

### Visualisation — regional profiles

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_region_profiles_v3.py` | Monthly T/S/density profiles (12 panels × region); model MLD vs density-criterion MLD; 6 ILD method lines; fixed axis limits; `use_abs=True` | `profiles_{region}.png` |

### Visualisation — climatology and summary

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `plot_monthly_climatology.py` | Monthly climatology maps (Jan–Dec) per ILD method | `monthly_clim_{method}.png` (×6) |
| `plot_bld_summary.py` | Global BLD maps, time series, coverage stats | `summary_bld_025deg_*.png` |

### Visualisation — LR vs HR comparison

| Script | What it does | Key outputs in `displays/` |
|---|---|---|
| `compare_lr_hr.py` | Global annual mean maps (LR / HR / difference), regional zooms, monthly time series | `compare_lr_hr_annual_mean.png`, `compare_lr_hr_zoom_regions.png` |
| `compare_lr_hr_profiles.py` | T/S profile overlay LR vs HR — 3 regions × 6 ILD methods | `compare_profiles_{region}_{method}.png` |
| `scatter_lr_hr.py` | Scatter LR vs HR: T/S profiles per depth + global MLD/ILD/BLD per basin | `scatter_bld_lr_hr.png` |

---

## Analysis Regions

| Region | Latitude | Longitude | Notes |
|---|---|---|---|
| Tropical Atlantic | 5°S – 10°N | 55°W – 30°W | Salinity-driven BL from river discharge |
| Bay of Bengal | 5°N – 25°N | 80°E – 100°E | Strong freshwater forcing, high BLD |
| W. Pacific Warm Pool (WPWP) | 10°S – 10°N | 130°E – 180°E | Highest BLD globally |
| N. Argentine Shelf | 45°S – 30°S | 65°W – 50°W | Shelf pixels only (depth < 200 m) |
| Southern Ocean (Indian sector) | 65°S – 45°S | 20°E – 100°E | Temperature inversions; `use_abs` critical here |

---

## Key Findings

### Sensitivity and inter-method agreement

1. **Agreement map** (% of 6 methods detecting BLD > 0): WPWP and Bay of Bengal show very high agreement (>95%). The Southern Ocean is the most heterogeneous.

2. **Absolute-value criterion is essential in the Southern Ocean**: using a signed gradient/difference criterion leaves ~83% of Southern Ocean pixels as NaN in winter due to temperature inversions. `use_abs=True` resolves this.

3. **ILD spread (std across 6 methods)** is highest in the Southern Ocean and in regions with weak thermoclines. The coefficient of variation (CV = std/mean) reveals where method choice matters most.

### Trend decomposition: BLD = ILD − MLD

4. **BLD is increasing (positive trend) in most tropical and subtropical regions** (1993–2024). This is explained by two concurrent signals:
   - **MLD is shoaling** (negative trend): surface ocean warming strengthens near-surface stratification, trapping mixing in a shallower layer.
   - **ILD is deepening** (positive trend): the thermocline descends in warm pool regions.

5. **Regional trends (gradient −0.1°C/m method):**

   | Region | ΔILD | ΔMLD | ΔBLD | Driver |
   |---|---|---|---|---|
   | Tropical Atlantic | weak | −1.1 m/dec* | +1.1 m/dec* | MLD shoaling |
   | W. Pacific Warm Pool | strong + | negative | +7 m/dec* | both |
   | N. Argentine Shelf | + | negative* | +1.1 m/dec* | both |
   | Southern Ocean | variable | negative | +1 m/dec* | MLD shoaling |

6. **Method robustness of ILD trends**: all 6 methods agree on a positive ILD trend in the WPWP and Bay of Bengal. In the Southern Ocean and eastern boundary currents, the sign is not consistent across methods.

### ENSO signal

7. **El Niño** composites show positive BLD anomalies in the central/eastern Pacific and negative anomalies in the Maritime Continent. **La Niña** shows the reverse. The La Niña − El Niño difference reaches ±30 m in the western Pacific.

---

## How to Run

```bash
# Activate the project environment
conda activate /Users/daniela/Documents/barrier_layers/.conda_arm

# Recompute all 6 sensitivity zarrs (use_abs=True) — long run, use tmux
python scripts/sensitivity_analysis_025deg.py

# Trend maps: ILD / MLD / BLD decomposition + per-method ILD
python scripts/plot_ild_mld_bld_trends.py

# Regional T/S/density profiles
python scripts/plot_region_profiles_v3.py

# Inter-method agreement map
python scripts/plot_method_disagreement_profiles.py

# Climatology and variability
python scripts/plot_monthly_climatology.py
python scripts/plot_bld_variability.py

# ENSO composites
python scripts/plot_enso_composite.py

# Compile Beamer presentation
cd reports
pdflatex -interaction=nonstopmode barrier_layers_beamer_part2.tex
pdflatex -interaction=nonstopmode barrier_layers_beamer_part2.tex
```

---

## Dependencies

```
xarray       zarr         numpy        pandas
matplotlib   scipy        pyshp        gsw
copernicusmarine
```

Install:
```bash
pip install xarray zarr numpy pandas matplotlib scipy pyshp gsw copernicusmarine
```

> Coastlines are rendered via `pyshp` (`import shapefile`) rather than `cartopy` to avoid Rosetta compatibility issues on Apple Silicon.
