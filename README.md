# Regime-conditional NO₂ forecasting and adaptive conformal prediction (Almaty & Astana)

Code for the paper **"When the Regime Picks the Model: A Skill-Based Benchmark and Regime-Conditional Conformal Prediction for Hourly NO₂ Forecasting in Two Climatically Opposite Cities"** (submitted to *Sensors*, MDPI).

We benchmark ten forecasters for hourly NO₂ in **Almaty** (mountain-basin, accumulation regime) and **Astana** (open-steppe, ventilation regime), score them by **skill over persistence** with significance tests, study **high-NO₂ episode detection**, and introduce a **regime-conditional, recency-adaptive conformal** scheme for trustworthy intervals.

## Key findings
- **No universal winner.** Persistence is unbeatable at +1 h; an LSTM is most skilful at +6/+24 h, but mainly in the ventilation regime (Astana). Gradient boosting is never best and is often worse than persistence.
- **Task matters too.** For detecting high-NO₂ episodes the ranking reverses — persistence and linear models catch episodes that the peak-smoothing recurrent networks miss.
- **Trustworthy uncertainty.** Marginal conformal prediction is mis-calibrated within regimes and drifts year-to-year; the regime-conditional recency-adaptive scheme restores near-nominal coverage (mean |PICP−0.90|: 0.057 → 0.004) and, as an alarm rule, raises episode detection (POD 0.72 → 0.98 in Almaty).

## Repository structure
```
src/                analysis scripts (Python, run in Google Colab with a GPU)
figures/            figures used in the paper
results_in_paper.csv   exact numbers reported in the manuscript, with the producing script
requirements.txt    exact library versions
CITATION.cff        citation metadata
LICENSE             MIT
```

## Data and city-level aggregation
Hourly NO₂ from the **Kazhydromet** network (2021–2025), **ERA5** meteorology, **CAMS** reanalysis (Almaty 2021–2025; Astana 2020–2025) and **Sentinel-5P/TROPOMI** NO₂ columns.

The **city-level NO₂ series is the hourly arithmetic mean of NO₂ across all stations reporting in that hour** (see `src/build_model_table_clean.py`, lines ~42–45). Hours with no reporting station are dropped (no zero-filling); the `n_stations` field records how many stations contributed to each city-hour. Typical (median) station counts per hour: Almaty 11–15 (peak 20 in 2023), Astana ~4 (occasionally up to 10 in 2023–2025). ERA5 and CAMS fields are spatial means over the city grid cells, resampled to 1 h.

The raw monitoring data are subject to Kazhydromet terms and are **not redistributed here**; the processed city-level table is available from the authors on request. Scripts read a single table `model_table_NO2_full.csv` (city-level hourly, 88,274 rows).

## Environment
Python 3.12 (Google Colab). Install dependencies:
```bash
pip install -r requirements.txt
```
A GPU runtime (e.g., Colab T4) is recommended for the deep-learning baselines.

## How to reproduce
Place `model_table_NO2_full.csv` in your working folder, then run (in Colab, after mounting Google Drive). Each script reads from / writes results to a Google Drive folder set at the top of the file; change the path to your own. Random seed is fixed (42).

1. **`src/build_model_table_clean.py`** — builds the clean city-level table from the raw feature table (city = hourly mean over reporting stations; CAMS/ERA5 merge). *Run only if rebuilding from raw data.*
2. **`src/core_conformal.py`** — mini benchmark (+6 h) and the **core novelty**: marginal vs regime vs recency-adaptive conformal (`conformal_adaptive.csv`, coverage figure).
3. **`src/deep_baselines.py`** — LSTM / GRU / CNN-LSTM for +1/+6/+24 h (`deep_baselines.csv`).
4. **`src/stats_and_peaks.py`** — Diebold–Mariano tests + block-bootstrap skill CIs (`significance_tests.csv`).
5. **`src/exceedance_v2.py`** — episode detection (POD/FAR/CSI/F1), point vs adaptive-conformal alarms (`exceedance_metrics_v2.csv`).
6. **`src/pcmb_conf_experiments.py`** — full experiment suite (all baselines + conformal + ablations).
7. *(optional)* **`src/add_s5p_to_clean.py`** — adds extra Sentinel-5P features (CO, SO₂, aerosol index).

## Reproducing the exact numbers in the paper
The authoritative numbers reported in the manuscript are listed in **`results_in_paper.csv`** together with the script that produces each one. All accuracy metrics in Table 2/3 are computed on a **single unified chronological test set (2025)** so that tabular, classical and deep models are directly comparable.

**Note on the +1 h persistence value.** The manuscript reports persistence R² = 0.911 at +1 h, computed on the unified 2025 test set (the same set used for every other model and horizon). An earlier exploratory file, `accuracy_y_t1.csv`, reports 0.770 for the same model because it was evaluated on a smaller, lower-variance validation subset; R² is normalised by the subset variance, so the two values describe different test sets, not different errors (the +1 h persistence RMSE is ≈ 7.1 µg/m³ in both). Use the unified-test-set value (0.911) and `results_in_paper.csv`. The +6 h and +24 h numbers are identical across all files.

## Citation
> [AUTHORS]. When the Regime Picks the Model: A Skill-Based Benchmark and Regime-Conditional Conformal Prediction for Hourly NO₂ Forecasting in Two Climatically Opposite Cities. *Sensors* (under review), 2025.

(Edit `CITATION.cff`, `LICENSE` and this line to add your name once finalised.)

## License
MIT — see `LICENSE`.
