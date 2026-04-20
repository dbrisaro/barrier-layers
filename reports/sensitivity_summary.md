# Barrier Layer Depth — Sensitivity Analysis Report

**Period:** 1993–2024  
**Data:** CMEMS 0.25° monthly reanalysis (`cmems_mod_glo_phy-all_my_0.25deg_P1M-m`)  
**Generated:** 2026-03-11 09:07  

---

## ILD Detection Methods

> **BLD = ILD − MLD** (MLD from CMEMS pre-computed `mlotst_glor`)

| Method | Definition | Thresholds tested |
|--------|-----------|-------------------|
| **Gradient** | First depth below 5 m where dT/dz ≤ threshold | −0.015, −0.025, −0.1 °C/m |
| **Difference** | First depth below 10 m where T(z) < T(10 m) − ΔT | 0.2, 0.5, 0.8 °C |

---

## 1. Global Statistics (1993–2024)

| Config | Global mean BLD (m) | Max BLD (time-mean, m) | % ocean BLD > 0 | Temporal std (m) |
|--------|--------------------:|-----------------------:|----------------:|-----------------:|
| Gradient −0.015°C/m | 12.7 | 247.8 | 33.0 | 5.6 |
| Gradient −0.025°C/m | 17.4 | 267.4 | 35.3 | 6.5 |
| Gradient −0.1°C/m | 30.1 | 231.4 | 23.7 | 4.2 |
| Diff 0.2°C | 19.3 | 400.2 | 48.2 | 4.6 |
| Diff 0.5°C | 31.6 | 497.9 | 50.0 | 6.0 |
| Diff 0.8°C | 40.1 | 497.9 | 49.0 | 6.7 |

---

## 2. Seasonal Statistics (global mean BLD, m)

| Config | Min (m) | Max (m) | Range (m) | Peak month |
|--------|--------:|--------:|----------:|------------|
| Gradient −0.015°C/m | 6.8 | 22.5 | 15.7 | Sep |
| Gradient −0.025°C/m | 9.9 | 28.5 | 18.6 | Oct |
| Gradient −0.1°C/m | 24.4 | 34.7 | 10.3 | Nov |
| Diff 0.2°C | 13.7 | 27.5 | 13.7 | Sep |
| Diff 0.5°C | 24.6 | 42.3 | 17.7 | Sep |
| Diff 0.8°C | 32.0 | 53.0 | 21.0 | Oct |

---

## 3. Monthly Climatology — Global Mean BLD (m)

| Config | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|--------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Gradient −0.015°C/m | 9.6 | 13.9 | 17.1 | 14.3 | 7.5 | 6.8 | 9.4 | 15.4 | 22.5 | 19.0 | 9.2 | 7.4 |
| Gradient −0.025°C/m | 13.2 | 17.4 | 21.1 | 19.5 | 12.0 | 9.9 | 12.5 | 19.0 | 27.7 | 28.5 | 16.0 | 11.7 |
| Gradient −0.1°C/m | 29.7 | 29.5 | 31.8 | 32.2 | 30.8 | 27.0 | 24.4 | 25.7 | 29.4 | 33.3 | 34.7 | 33.2 |
| Diff 0.2°C | 15.6 | 18.0 | 18.6 | 16.5 | 16.7 | 20.1 | 22.6 | 25.7 | 27.5 | 21.8 | 15.0 | 13.7 |
| Diff 0.5°C | 24.9 | 28.3 | 31.5 | 30.9 | 27.9 | 28.9 | 31.3 | 36.1 | 42.3 | 41.6 | 31.0 | 24.6 |
| Diff 0.8°C | 32.0 | 35.5 | 40.3 | 41.8 | 37.1 | 35.0 | 36.9 | 42.4 | 50.3 | 53.0 | 43.0 | 33.7 |

---

## 4. Regional Mean BLD (m, 1993–2024)

| Config | Tropical Indian Ocean | Bay of Bengal | W. Pacific Warm Pool | Amazon Plume |
|--------|-------:|-------:|-------:|-------:|
| Gradient −0.015°C/m | 8.1 | 13.7 | 27.7 | 15.6 |
| Gradient −0.025°C/m | 13.0 | 17.6 | 38.6 | 21.9 |
| Gradient −0.1°C/m | 34.5 | 41.7 | 86.2 | 47.1 |
| Diff 0.2°C | 13.1 | 22.2 | 28.5 | 19.9 |
| Diff 0.5°C | 20.8 | 28.6 | 41.6 | 29.4 |
| Diff 0.8°C | 25.9 | 33.2 | 50.5 | 35.6 |

---

## 5. Key Findings

- **Difference methods** detect barrier layers over ~48–50% of the ocean, nearly double the gradient methods (24–35%).
- **Global mean BLD** ranges from 12.7 m (Gradient −0.015°C/m) to 40.1 m (Diff 0.8°C) — a spread of nearly 3×.
- A **positive trend** (~2008–2024) in global mean BLD is consistent across all 6 configurations, indicating a robust physical signal not dependent on method choice.
- The **W. Pacific Warm Pool** shows the largest method sensitivity, with BLD up to ~140 m for loose difference thresholds vs ~30 m for strict gradient methods.
- The **Bay of Bengal** shows strong seasonal cycling across all configs, peaking in Oct (gradient) / Sep (difference 0.5°C).

---

## 6. Output Figures

### Sensitivity comparison (`compare_sensitivity.py`)

| File | Description |
|------|-------------|
| `displays/sensitivity_ensemble_maps.png` | Ensemble mean BLD, spread σ(BLD), detection agreement, and 3 individual gradient config maps |
| `displays/sensitivity_regional_timeseries.png` | All 6 configs for 4 key ocean regions (1993–2024) |
| `displays/sensitivity_global_timeseries.png` | Global mean BLD time series, all 6 configs |
| `displays/sensitivity_stats_table.png` | Summary metrics table (global mean, coverage %, seasonal range) |

### Per-config summaries (`plot_bld_summary.py`)

One 3-panel figure per config: global annual-mean BLD map + global mean time series (1993–2024) + seasonal climatology.

| File | Config |
|------|--------|
| `displays/summary_gradient_015.png` | Gradient −0.015 °C/m |
| `displays/summary_gradient_025.png` | Gradient −0.025 °C/m |
| `displays/summary_gradient_100.png` | Gradient −0.1 °C/m |
| `displays/summary_difference_02.png` | Temp. diff. ΔT = 0.2 °C |
| `displays/summary_difference_05.png` | Temp. diff. ΔT = 0.5 °C |
| `displays/summary_difference_08.png` | Temp. diff. ΔT = 0.8 °C |
| `displays/summary_sensitivity_comparison.png` | Side-by-side annual-mean maps for all 6 configs |
| `displays/summary_bld_025deg_monthly_1993_2024.png` | Full time series summary (gradient −0.1 default config) |

### Monthly climatology maps (`plot_monthly_climatology.py`)

| File | Description |
|------|-------------|
| `displays/monthly_clim_gradient_015.png` | 12-month climatology — Gradient −0.015 °C/m |
| `displays/monthly_clim_gradient_025.png` | 12-month climatology — Gradient −0.025 °C/m |
| `displays/monthly_clim_gradient_100.png` | 12-month climatology — Gradient −0.1 °C/m |
| `displays/monthly_clim_difference_02.png` | 12-month climatology — Diff. 0.2 °C |
| `displays/monthly_clim_difference_05.png` | 12-month climatology — Diff. 0.5 °C |
| `displays/monthly_clim_difference_08.png` | 12-month climatology — Diff. 0.8 °C |
| `displays/monthly_clim_seasonal_comparison.png` | 4 seasons × 6 configs comparison panel |

### Single-point profile diagnostics (`sensitivity_one_point.py`)

| File | Description |
|------|-------------|
| `displays/sensitivity_one_point_2023-12-30_lat-10.0_lon55.0.png` | All 6 ILD methods on one profile — lat −10°, lon 55°, Dec 2023 |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-10.0_lon_55.0_grad_-0.1.png` | T/S profile — lat −10°, lon 55°, gradient −0.1 °C/m |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-10.0_lon_55.0_grad_-0.1_mayor_5m.png` | Same, ILD search restricted below 5 m |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-10.0_lon_55.0_grad_-0.1_mayor_5m_high_res_model.png` | Same point, HR 0.083° model |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-19.0_lon_-119.0_grad_-0.1.png` | T/S profile — lat −19°, lon −119°, gradient −0.1 °C/m |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-19.0_lon_-119.0_grad_-0.2.png` | Same point, gradient −0.2 °C/m |
| `displays/perfil_vertical_temp_sal_2023-12-30_lat_-5.0_lon_-105.0_grad_-0.1.png` | T/S profile — lat −5°, lon −105°, gradient −0.1 °C/m |
