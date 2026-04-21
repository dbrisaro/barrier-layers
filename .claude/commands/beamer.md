# Beamer presentation manager

You are helping manage the Barrier Layer Depth Beamer presentation.

## Key files
- **Source**: `reports/barrier_layers_beamer_part2.tex`
- **Figures folder**: `displays/` (all PNGs live here)
- **Compile**: `cd reports && pdflatex -interaction=nonstopmode barrier_layers_beamer_part2.tex` (run twice for references)

## Presentation conventions

### LaTeX helpers defined in the preamble
```latex
\fullimg{../displays/filename.png}   % full-width image, keeps aspect ratio
\captext{Caption text here.}         % tiny gray caption below image
\sectionframe{Section title}         % dark navy full-slide section divider
```

### Typical image slide
```latex
\begin{frame}{Slide title}
\begin{center}
\fullimg{../displays/filename.png}
\end{center}
\captext{Short caption explaining what the figure shows.}
\end{frame}
```

### Two-image slide (side by side)
```latex
\begin{frame}{Slide title}
\begin{columns}[c]
\column{0.50\textwidth}
\includegraphics[width=\textwidth]{../displays/left.png}
\column{0.50\textwidth}
\includegraphics[width=\textwidth]{../displays/right.png}
\end{columns}
\captext{Caption.}
\end{frame}
```

### Color palette
- `oceanblue` = #065A82 (frame title background, block titles, structure)
- `deepnavy`  = #1C2951 (section dividers, title slide)
- `iceblue`   = #CADCFC (subtitle text)
- `teal`      = #1C7293 (BLD annotations)

## Available figures and what they show

### Sensitivity / agreement
- `summary_sensitivity_comparison.png` — 2×3 grid, BLD annual mean for each of the 6 ILD methods
- `method_agreement_map.png` — global map, % of methods detecting BLD > 0; 5 region boxes
- `sensitivity_stats_table.png` — table of global statistics per method
- `ild_spread_maps.png` — ILD std dev and range across 6 methods (2 panels)
- `ild_spread_cv_scatter.png` — ILD coefficient of variation map + scatter by latitude band

### Trends
- `trends_ild_mld_bld_global.png` — 3-panel: ILD / MLD / BLD global trend maps (m/decade)
- `trends_ild_mld_bld_regional.png` — 5 regions × 3 variables, annual mean + trend line
- `trends_ild_per_method.png` — 2×3 grid, ILD trend map one panel per method (shared colorscale)
- `trends_ild_ensemble.png` — ILD ensemble mean trend + uncertainty (std across 6 methods)
- `bld_trend_global.png` — BLD global trend map (default method: gradient −0.1°C/m)
- `bld_trend_regional.png` — BLD regional time series (5 regions)
- `bld_trend_ensemble.png` — BLD ensemble mean trend + spread

### Regional profiles
- `profiles_tropical_atlantic.png` — monthly T/S/density profiles, Tropical Atlantic
- `profiles_arg_sea_north.png` — monthly T/S/density profiles, N. Argentine Shelf (shelf pixels only)
- `profiles_southern_ocean.png` — monthly T/S/density profiles, Southern Ocean

### Climatology
- `monthly_clim_gradient_025.png` — monthly BLD climatology, gradient −0.025°C/m
- `monthly_clim_difference_02.png` — monthly BLD climatology, difference 0.2°C

### LR vs HR comparison
- `compare_lr_hr_annual_mean.png` — LR / HR / difference global annual mean
- `compare_lr_hr_zoom_regions.png` — 4 regional zooms LR vs HR

## Current slide order
1. Title slide
2. ¿Qué es la Barrier Layer? (diagram + equation)
3. Productos CMEMS utilizados (table)
4. Grillas verticales: LR vs HR
5. 6 métodos de detección de ILD (table)
6. Los 6 métodos sobre un mismo perfil
7. BLD media anual: los 6 métodos comparados
8. Acuerdo entre los 6 métodos
9. Estadísticas globales: sensibilidad al método

## Your task

$ARGUMENTS

If no specific task is given, suggest what slides could be added or updated based on the figures available in `displays/` that are not yet in the presentation. Focus on the trend decomposition (ILD / MLD / BLD) and the per-method ILD trend comparison, which are the newest results.

Always read the current .tex file before making any edits. After editing, compile and report any LaTeX errors.
