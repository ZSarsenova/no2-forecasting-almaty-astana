# Regime-conditional NO₂ forecasting and adaptive conformal prediction (Almaty & Astana)

Code for the paper **"When the Regime Picks the Model: A Skill-Based Benchmark and Regime-Conditional Conformal Prediction for Hourly NO₂ Forecasting in Two Climatically Opposite Cities"** (submitted to *Sensors*, MDPI).

We benchmark ten forecasters for hourly NO₂ in **Almaty** (mountain-basin, accumulation regime) and **Astana** (open-steppe, ventilation regime), score them by **skill over persistence** with significance tests, study **high-NO₂ episode detection**, and introduce a **regime-conditional, recency-adaptive conformal** scheme for trustworthy intervals.

## Key findings
- **No universal winner.** Persistence is unbeatable at +1 h; an LSTM is most skilful at +6/+24 h, but mainly in the ventilation regime (Astana). Gradient boosting is never best and is often worse than persistence.
- **Task matters too.** For detecting high-NO₂ episodes the ranking reverses — persistence and linear models catch episodes that the peak-smoothing recurrent networks miss.
- **Trustworthy uncertainty.** Marginal conformal prediction is mis-calibrated within regimes and drifts year-to-year; the regime-conditional recency-adaptive scheme restores near-nominal coverage (mean |PICP−0.90|: 0.057 → 0.004) and, as an alarm rule, raises episode detection (POD 0.72 → 0.98 in Almaty).

## Repository structure
```
src/        analysis scripts (Python, run in Google Colab with a GPU)
figures/    figures used in the paper
requirements.txt   exact library versions
LICENSE     MIT
```

## Data
Hourly NO₂ from the **Kazhydromet** network (2021–2025), **ERA5** meteorology, **CAMS** reanalysis (Almaty 2021–2025; Astana 2020–2025) and **Sentinel-5P/TROPOMI** NO₂ columns. The raw monitoring data are subject to Kazhydromet terms and are **not redistributed here**; the processed city-level table is available from the authors on request. Scripts read a single table `model_table_NO2_full.csv` (city-level hourly, 88,274 rows).

## Environment
Python 3.12 (Google Colab). Install dependencies:
```bash
pip install -r requirements.txt
```
A GPU runtime (e.g., Colab T4) is recommended for the deep-learning baselines.

## How to reproduce
Place `model_table_NO2_full.csv` in your working folder, then run (in Colab, after mounting Google Drive):

1. **`src/build_model_table_clean.py`** — builds the clean city-level modelling table from the raw feature table (fixes per-year aggregation and CAMS merge). *Run only if rebuilding the table from raw data.*
2. **`src/core_conformal.py`** — mini benchmark (+6 h) and the **core novelty**: marginal vs regime vs recency-adaptive conformal (`conformal_adaptive.csv`, coverage figure).
3. **`src/deep_baselines.py`** — LSTM / GRU / CNN-LSTM for +1/+6/+24 h (`deep_baselines.csv`).
4. **`src/stats_and_peaks.py`** — Diebold–Mariano tests + block-bootstrap skill CIs (`significance_tests.csv`, `skill_ci.csv`).
5. **`src/exceedance_v2.py`** — high-NO₂ episode detection (POD/FAR/CSI/F1) for point-forecast vs adaptive-conformal alarms, multiple thresholds (`exceedance_metrics_v2.csv`).
6. **`src/pcmb_conf_experiments.py`** — full experiment suite (all baselines + conformal schemes + ablations).
7. *(optional)* **`src/add_s5p_to_clean.py`** — adds extra Sentinel-5P features (CO, SO₂, aerosol index) to the clean table.

Each script reads from / writes results to a Google Drive folder (paths are set at the top of each file); fix the folder path to your own. Random seed is fixed (42); library versions are pinned in `requirements.txt`.

## Citation
> [AUTHORS]. When the Regime Picks the Model: A Skill-Based Benchmark and Regime-Conditional Conformal Prediction for Hourly NO₂ Forecasting in Two Climatically Opposite Cities. *Sensors* (under review), 2025.

## License
MIT — see `LICENSE`.
