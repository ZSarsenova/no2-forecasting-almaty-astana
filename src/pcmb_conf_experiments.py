#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  PCMB-Conf : Physics-Constrained Monotone Boosting + Regime-Conditional
#              (Mondrian) Conformal Prediction  —  NO2 forecasting, KZ
#  Almaty (basin / inversions) vs Astana (ventilation) = opposite regimes.
#
#  NOVELTY (3 pillars):
#    (1) monotone-constrained boosting for AQ  (literature uses UNconstrained);
#    (2) regime-conditional / Mondrian conformal (AQ work uses marginal only);
#    (3) Almaty<->Astana natural experiment of opposite pollution regimes.
#  Plain LightGBM is now just ONE BASELINE: "monotone > plain" answers the
#  reviewer who called the original single-model draft insufficient.
#
#  Runs top-to-bottom in Google Colab. Only edit DRIVE_DIR if your path differs.
#  Designed for input file: model_table_NO2_full.csv  (city-level modeling table
#  rebuilt with the new CAMS 2024-2025). Schema: city,timestamp,no2,n_stations,
#  lags/rolling/calendar/era5/cams/derived/s5p, persistence, y_t1/y_t6/y_t24.
#
#  Tested library versions (Colab, 2026): numpy 1.26+, pandas 2.x,
#    scikit-learn 1.4+, lightgbm 4.x, xgboost 2.x, scipy 1.11+, shap 0.45+,
#    statsmodels 0.14+, tensorflow 2.15+ (optional, for LSTM/GRU/CNN-LSTM),
#    matplotlib 3.8+.
# =============================================================================

# %% ===== CELL 0 : install + imports + global config ========================
# In Colab uncomment:
# !pip -q install lightgbm xgboost shap statsmodels scikit-learn matplotlib
import os, json, time, warnings, platform
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# --- paths ---
USE_DRIVE   = True                       # set False to run on a local CSV
DRIVE_DIR   = "/content/drive/MyDrive/Air Quality Cloud 2805"
LOCAL_DIR   = "."
INPUT_CSV   = "model_table_NO2_full.csv"
OUT_DIR     = "pcmb_conf_outputs"        # results + figures saved here

# --- experiment config ---
HORIZONS    = ["y_t6", "y_t1", "y_t24"]  # primary first
PRIMARY_H   = "y_t6"
TRAIN_END   = 2023                       # train year <=
CALIB_YEAR  = 2024                       # conformal calibration (no leakage)
TEST_YEAR   = 2025
ALPHA       = 0.10                       # nominal miscoverage -> 90% intervals
RUN_DEEP    = True                       # LSTM/GRU/CNN-LSTM (needs TensorFlow)
RUN_ARIMA   = True                       # statsmodels AR baseline
DEEP_EPOCHS = 25
SEQ_LEN     = 12                         # window length for sequence models

# --- feature groups (explicit order; monotone vector aligns to FEATURES) ----
AUTOCORR = ["no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12",
            "no2_lag_24","no2_lag_48","no2_lag_168","no2_rollmean_3",
            "no2_rollmean_6","no2_rollmean_12","no2_rollmean_24","no2_rollstd_3",
            "no2_rollstd_6","no2_rollstd_24","no2_rollmin_24","no2_rollmax_24",
            "no2_diff_1","no2_diff_24"]
CALENDAR = ["hour","dayofweek","month","dayofyear","is_weekend","heating_season",
            "hour_sin","hour_cos","doy_sin","doy_cos"]
ERA5     = ["era5_u10","era5_v10","era5_d2m","era5_t2m","era5_blh","era5_sp",
            "era5_tp","era5_ssrd","era5_tcc","era5_wind_speed","era5_wind_dir"]
CAMS     = ["cams_no2_surf","cams_o3","cams_so2","cams_co"]
DERIVED  = ["ventilation_coefficient","stagnation_indicator","atmospheric_dryness",
            "winter_inversion_indicator","cams_bias_correction"]
S5P      = ["s5p_no2_trop_column"]
FEATURES = AUTOCORR + CALENDAR + ERA5 + CAMS + DERIVED + S5P
REGIME_FEATS = ["heating_season","winter_inversion_indicator","stagnation_indicator",
                "ventilation_coefficient"]  # for ablation (c)

# --- physics monotone directions: +1 raises NO2, -1 lowers NO2, 0 = free -----
MONO_DIR = {
    "era5_blh": -1,                  # deeper mixing layer -> lower NO2
    "era5_wind_speed": -1,           # stronger wind -> dispersion -> lower NO2
    "ventilation_coefficient": -1,   # better ventilation -> lower NO2
    "winter_inversion_indicator": +1,
    "stagnation_indicator": +1,
    "s5p_no2_trop_column": +1,       # higher tropospheric column -> higher NO2
    "cams_no2_surf": +1,             # CAMS surface NO2 prior -> higher NO2
}
MONO_VEC = [MONO_DIR.get(c, 0) for c in FEATURES]

LGB_PARAMS = dict(n_estimators=2000, num_leaves=63, max_depth=10,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=40, n_jobs=-1, random_state=SEED, verbose=-1)


def data_dir():
    return DRIVE_DIR if USE_DRIVE else LOCAL_DIR

def output_dir():
    # Write to Colab LOCAL disk (/content), which does NOT drop like the Drive
    # FUSE mount. Outputs are copied to Drive at the end (best-effort).
    root = "/content" if os.path.isdir("/content") else "."
    d = os.path.join(root, OUT_DIR); os.makedirs(d, exist_ok=True); return d


# %% ===== CELL 1 : mount Drive + load =======================================
def mount_and_load():
    global USE_DRIVE
    if USE_DRIVE:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("[drive] not in Colab or mount failed -> local mode:", e)
            USE_DRIVE = False
    path = os.path.join(data_dir(), INPUT_CSV)
    if not os.path.exists(path):
        path = os.path.join(LOCAL_DIR, INPUT_CSV)
    print("[load] reading", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["year"] = df["timestamp"].dt.year
    # regime label (PRIMARY = city; FINE = city x heating_season)
    df["regime"] = df["city"].astype(str)
    hs = df["heating_season"].fillna(0).astype(int) if "heating_season" in df else 0
    df["regime_fine"] = df["city"].astype(str) + "|" + np.where(
        (hs == 1) if np.ndim(hs) else False, "heat", "warm")
    os.makedirs(output_dir(), exist_ok=True)
    te = df[df.year == TEST_YEAR]
    cov = 100 * te["cams_no2_surf"].notna().mean() if len(te) else 0
    print(f"[load] rows={len(df)} cities={df['city'].unique().tolist()} "
          f"| CAMS coverage in test {TEST_YEAR}: {cov:.0f}% (must be >0)")
    return df


# %% ===== CELL 2 : split (chronological, leakage-free) ======================
def split(df, horizon):
    d = df.dropna(subset=[horizon]).copy()
    feats = [c for c in FEATURES if c in d.columns]
    tr  = d[d.year <= TRAIN_END]
    cal = d[d.year == CALIB_YEAR]
    te  = d[d.year == TEST_YEAR]
    return d, feats, tr, cal, te


# %% ===== CELL 3 : metrics helpers ==========================================
def acc_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae  = float(np.mean(np.abs(y - p)))
    nz   = np.abs(y) > 1e-6
    mape = float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100) if nz.any() else np.nan
    ss   = np.sum((y - y.mean()) ** 2)
    r2   = float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else np.nan
    ia   = float(1 - np.sum((y - p) ** 2) /
                 np.sum((np.abs(p - y.mean()) + np.abs(y - y.mean())) ** 2 + 1e-12))
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, IA=ia)


def diebold_mariano(y, p1, p2, h=6):
    y, p1, p2 = map(lambda x: np.asarray(x, float), (y, p1, p2))
    d = (y - p1) ** 2 - (y - p2) ** 2
    T = len(d); dbar = d.mean()
    g0 = np.mean((d - dbar) ** 2); s = g0
    for k in range(1, h):
        s += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    var = s / T
    if var <= 0:
        return float(dbar), np.nan
    from scipy import stats
    dm = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    return float(dbar), float(2 * (1 - stats.t.cdf(abs(dm), df=T - 1)))


# %% ===== CELL 4 : sequence builder for deep baselines ======================
def make_sequences(frame, feats, horizon, L=SEQ_LEN):
    """Per-city sliding windows + positional index of each target row in `frame`
       (so predictions map back exactly). Reduced feature set for memory."""
    keep = [c for c in (AUTOCORR[:8] + ["era5_blh", "era5_wind_speed",
            "era5_t2m", "hour_sin", "hour_cos"]) if c in feats]
    f = frame.reset_index(drop=True)                 # positional index 0..n-1
    f = f.loc[f.sort_values(["city", "timestamp"]).index]   # time-sorted within city
    Xs, ys, pos = [], [], []
    for _, g in f.groupby("city", sort=False):
        M = g[keep].fillna(g[keep].median()).to_numpy("float32")
        y = g[horizon].to_numpy("float32")
        gpos = g.index.to_numpy()                    # positions in `frame`
        for i in range(L, len(g)):
            Xs.append(M[i - L:i]); ys.append(y[i]); pos.append(gpos[i])
    if not Xs:
        return None, None, None, keep
    return (np.asarray(Xs, "float32"), np.asarray(ys, "float32"),
            np.asarray(pos, dtype=int), keep)


# %% ===== CELL 5 : train method + all baselines =============================
def train_all(feats, tr, cal, te, horizon):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb

    # CRITICAL (no leakage): point models train on TRAIN ONLY (<=2023).
    # Calibration year (2024) is held out and used ONLY by the conformal step,
    # so conformal residuals are out-of-sample and coverage is valid.
    trf = tr.copy()
    Xtr, ytr = trf[feats], trf[horizon].values
    Xte, yte = te[feats], te[horizon].values
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc  = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = sc.transform(imp.transform(Xtr)); Xte_s = sc.transform(imp.transform(Xte))

    preds, timings = {}, {}
    def run(name, fn):
        t0 = time.time(); preds[name] = fn(); timings[name] = round(time.time()-t0, 1)
        print(f"  [{name}] done in {timings[name]}s")

    if "persistence" in te.columns:
        preds["Persistence"] = te["persistence"].values
    run("Ridge(AR)",  lambda: Ridge(alpha=1.0).fit(Xtr_s, ytr).predict(Xte_s))
    run("RandomForest", lambda: RandomForestRegressor(
        n_estimators=150, max_depth=18, min_samples_leaf=20, max_samples=0.6,
        n_jobs=-1, random_state=SEED
    ).fit(imp.transform(Xtr), ytr).predict(imp.transform(Xte)))
    try:
        from xgboost import XGBRegressor
        run("XGBoost", lambda: XGBRegressor(
            n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=SEED
        ).fit(Xtr, ytr).predict(Xte))
    except Exception as e:
        print("  XGBoost skipped:", e)
    # plain LightGBM (KEY BASELINE) and the METHOD share params except constraints
    plain = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, ytr)
    preds["LightGBM(plain)"] = plain.predict(Xte)
    method = lgb.LGBMRegressor(monotone_constraints=MONO_VEC, **LGB_PARAMS).fit(Xtr, ytr)
    preds["PCMB(monotone)"] = method.predict(Xte)
    print("  [PCMB(monotone)] trained with monotone_constraints")

    if RUN_ARIMA:
        try:
            from statsmodels.tsa.ar_model import AutoReg
            pr = np.full(len(te), np.nan)
            for city in te["city"].unique():
                s_tr = trf[trf.city == city].sort_values("timestamp")["no2"].astype(float)
                m = AutoReg(s_tr.values, lags=[1,2,3,24,48,168], old_names=False).fit()
                params = m.params
                # direct h-step: use lag features already in te (approx via Ridge on lags)
                from sklearn.linear_model import LinearRegression
                lagcols = [c for c in AUTOCORR if c.startswith("no2_lag")]
                lr = LinearRegression().fit(trf[trf.city==city][lagcols].fillna(0),
                                            trf[trf.city==city][horizon])
                idx = te["city"] == city
                pr[idx.values] = lr.predict(te[idx][lagcols].fillna(0))
            preds["ARIMA(AR)"] = pr
            print("  [ARIMA(AR)] done")
        except Exception as e:
            print("  ARIMA skipped:", e)

    if RUN_DEEP:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks
            tf.random.set_seed(SEED)
            Xtr_seq, ytr_seq, _,     keep = make_sequences(trf, feats, horizon)
            Xte_seq, yte_seq, te_pos, _   = make_sequences(te,  feats, horizon)
            if Xtr_seq is not None and Xte_seq is not None and len(Xte_seq) > 0:
                mu, sd = Xtr_seq.mean((0, 1)), Xtr_seq.std((0, 1)) + 1e-6
                Xtr_seq = (Xtr_seq - mu) / sd; Xte_seq = (Xte_seq - mu) / sd
                nf = Xtr_seq.shape[-1]
                def build(kind):
                    inp = layers.Input((SEQ_LEN, nf))
                    if kind == "LSTM":      x = layers.LSTM(64)(inp)
                    elif kind == "GRU":     x = layers.GRU(64)(inp)
                    else:                                   # CNN-LSTM
                        x = layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
                        x = layers.LSTM(64)(x)
                    x = layers.Dense(32, activation="relu")(x)
                    out = layers.Dense(1)(x)
                    mdl = models.Model(inp, out); mdl.compile("adam", "mse"); return mdl
                es = callbacks.EarlyStopping(patience=4, restore_best_weights=True)
                for kind in ["LSTM", "GRU", "CNN-LSTM"]:
                    mdl = build(kind)
                    mdl.fit(Xtr_seq, ytr_seq, validation_split=0.1, epochs=DEEP_EPOCHS,
                            batch_size=512, callbacks=[es], verbose=0)
                    p = mdl.predict(Xte_seq, verbose=0).ravel()
                    out = np.full(len(te), np.nan)
                    out[te_pos] = p                          # exact positional mapping
                    preds[kind] = out
                    print(f"  [{kind}] done (covered {np.isfinite(out).sum()}/{len(te)} rows)")
            else:
                print("  deep baselines: not enough sequence data, skipped")
        except Exception as e:
            print("  deep baselines skipped:", repr(e))

    return preds, timings, dict(plain=plain, method=method, imp=imp, sc=sc,
                                Xtr=Xtr, ytr=ytr)


# %% ===== CELL 6 : accuracy table (overall + per city) ======================
def accuracy_table(preds, te, horizon):
    rows = []
    for name, p in preds.items():
        m = acc_metrics(te[horizon].values, p); m["model"] = name; m["scope"] = "ALL"
        rows.append(m)
        for city in te["city"].unique():
            idx = (te["city"] == city).values
            if np.isfinite(p[idx]).sum() > 10:
                mc = acc_metrics(te[horizon].values[idx], p[idx])
                mc["model"] = name; mc["scope"] = city; rows.append(mc)
    res = pd.DataFrame(rows)[["model","scope","RMSE","MAE","MAPE","R2","IA"]].round(3)
    return res.sort_values(["scope","RMSE"]).reset_index(drop=True)


# %% ===== CELL 7 : CONFORMAL  (global vs Mondrian)  — core experiment =======
def split_conformal(model, imp, cal, te, feats, horizon, alpha, by=None):
    """Split conformal. by=None -> global/marginal. by='regime' -> Mondrian."""
    rc = np.abs(cal[horizon].values - model.predict(cal[feats]))
    yhat = model.predict(te[feats]); yte = te[horizon].values
    lo = np.empty(len(te)); hi = np.empty(len(te))
    if by is None:
        n = len(rc); k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q = np.sort(rc)[k - 1]
        lo, hi = yhat - q, yhat + q
        qmap = {"GLOBAL": q}
    else:
        qmap = {}
        for r in te[by].unique():
            cmask = (cal[by] == r).values; tmask = (te[by] == r).values
            rr = rc[cmask]
            if len(rr) < 30:                       # fallback to global if tiny
                n = len(rc); k = min(int(np.ceil((n+1)*(1-alpha))), n); q = np.sort(rc)[k-1]
            else:
                n = len(rr); k = min(int(np.ceil((n+1)*(1-alpha))), n); q = np.sort(rr)[k-1]
            lo[tmask] = yhat[tmask] - q; hi[tmask] = yhat[tmask] + q; qmap[str(r)] = float(q)
    return yhat, lo, hi, yte, qmap


def coverage_report(te, lo, hi, yte, by="city", tag=""):
    rows = []
    def picp_mpiw(mask):
        m = mask & np.isfinite(lo) & np.isfinite(hi)
        if m.sum() == 0: return np.nan, np.nan, 0
        cov = float(np.mean((yte[m] >= lo[m]) & (yte[m] <= hi[m])))
        wid = float(np.mean(hi[m] - lo[m])); return cov, wid, int(m.sum())
    cov, wid, n = picp_mpiw(np.ones(len(te), bool))
    rows.append(dict(method=tag, scope="ALL", PICP=round(cov,3), MPIW=round(wid,1), n=n))
    for r in te[by].unique():
        cov, wid, n = picp_mpiw((te[by] == r).values)
        rows.append(dict(method=tag, scope=str(r), PICP=round(cov,3), MPIW=round(wid,1), n=n))
    return pd.DataFrame(rows)


def conformal_experiment(art, cal, te, feats, horizon, alpha):
    model, imp = art["method"], art["imp"]
    yh, loG, hiG, yte, qG = split_conformal(model, imp, cal, te, feats, horizon, alpha, by=None)
    _,  loM, hiM, _,   qM = split_conformal(model, imp, cal, te, feats, horizon, alpha, by="regime")
    repG = coverage_report(te, loG, hiG, yte, by="city", tag="Global/marginal")
    repM = coverage_report(te, loM, hiM, yte, by="city", tag="Mondrian/regime")
    rep = pd.concat([repG, repM], ignore_index=True)
    print("\n  CONFORMAL coverage (nominal = %.0f%%):" % (100*(1-alpha)))
    print(rep.to_string(index=False))
    skew = repG[repG.scope != "ALL"]["PICP"]
    print(f"  -> global coverage spread across cities: "
          f"{skew.min():.3f}..{skew.max():.3f} (Mondrian should be near nominal in both)")
    # reliability sweep (Mondrian) over several nominal levels
    noms = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; emp = []
    for a in [1 - x for x in noms]:
        _, lo, hi, _, _ = split_conformal(model, imp, cal, te, feats, horizon, a, by="regime")
        m = np.isfinite(lo) & np.isfinite(hi)
        emp.append(float(np.mean((yte[m] >= lo[m]) & (yte[m] <= hi[m]))))
    reliab = pd.DataFrame({"nominal": noms, "empirical_mondrian": emp})
    return rep, dict(yhat=yh, loG=loG, hiG=hiG, loM=loM, hiM=hiM, yte=yte, reliab=reliab)


# %% == CORE NOVELTY: marginal vs regime-conditional vs recency-adaptive conformal ==
def conformal_adaptive_experiment(art, cal, te, feats, horizon, alpha, window=336):
    """Decisive experiment. Within FINER regimes (city x heating-season), compare:
        (1) Static-2024 MARGINAL  (single global quantile) - what AQ literature uses;
        (2) Static-2024 REGIME-conditional (per finer regime);
        (3) RECENCY-ADAPTIVE regime-conditional (online rolling window, seeded by 2024).
       Shows (1) is mis-calibrated within regimes & under 2024->2025 shift, and (3)
       restores nominal coverage in each regime. Quality = mean |PICP - nominal|."""
    from collections import deque
    model = art["method"]
    REG = "regime_fine" if "regime_fine" in te.columns else "regime"
    cal = cal.copy(); te = te.sort_values("timestamp").copy()
    rc = np.abs(cal[horizon].values - model.predict(cal[feats]))
    reg_cal = cal[REG].values
    yhat = model.predict(te[feats]); yte = te[horizon].values
    reg_te = te[REG].values; n_t = len(te); resid_te = np.abs(yte - yhat)
    def quant(arr, a):
        arr = np.sort(np.asarray(arr, float))
        if len(arr) == 0: return np.nan
        return arr[min(int(np.ceil((len(arr) + 1) * (1 - a))), len(arr)) - 1]

    # (1) static marginal
    qg = quant(rc, alpha); lo1, hi1 = yhat - qg, yhat + qg
    # (2) static regime-conditional
    lo2, hi2 = np.empty(n_t), np.empty(n_t)
    for r in np.unique(reg_te):
        rr = rc[reg_cal == r]; q = quant(rr, alpha) if (reg_cal == r).sum() >= 30 else qg
        m = reg_te == r; lo2[m] = yhat[m] - q; hi2[m] = yhat[m] + q
    # (3) recency-adaptive regime-conditional (online; residual seen AFTER prediction)
    lo3, hi3 = np.empty(n_t), np.empty(n_t)
    buf = {r: deque((np.sort(rc[reg_cal == r])[-window:] if (reg_cal == r).any()
                     else np.sort(rc)[-window:]).tolist(), maxlen=window)
           for r in np.unique(reg_te)}
    for i in range(n_t):
        b = buf[reg_te[i]]
        q = quant(np.fromiter(b, float) if len(b) else rc, alpha)
        lo3[i] = yhat[i] - q; hi3[i] = yhat[i] + q
        b.append(resid_te[i])

    def by_regime(lo, hi, tag):
        rows = [dict(scheme=tag, regime="ALL", PICP=round(float(np.mean((yte>=lo)&(yte<=hi))),3),
                     MPIW=round(float(np.mean(hi-lo)),1))]
        for r in np.unique(reg_te):
            m = reg_te == r
            rows.append(dict(scheme=tag, regime=str(r),
                PICP=round(float(np.mean((yte[m]>=lo[m])&(yte[m]<=hi[m]))),3),
                MPIW=round(float(np.mean(hi[m]-lo[m])),1)))
        return pd.DataFrame(rows)
    res = pd.concat([by_regime(lo1,hi1,"1.Static-marginal"),
                     by_regime(lo2,hi2,"2.Static-regime"),
                     by_regime(lo3,hi3,"3.Adaptive-regime")], ignore_index=True)
    nom = 1 - alpha
    print("\n  CORE: PICP within finer regimes (nominal %.0f%%):" % (100*nom))
    print(res.pivot(index="regime", columns="scheme", values="PICP").to_string())
    print("  calibration quality (mean |PICP - nominal| across regimes, lower=better):")
    for tag in ["1.Static-marginal","2.Static-regime","3.Adaptive-regime"]:
        sub = res[(res.scheme==tag)&(res.regime!="ALL")]
        print(f"    {tag}: {np.mean(np.abs(sub['PICP']-nom)):.3f}")
    return res


def fig_adaptive(res, outdir, alpha):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    piv = res[res.regime != "ALL"].pivot(index="regime", columns="scheme", values="PICP")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax); ax.axhline(1-alpha, ls="--", c="k", label="nominal")
    ax.set_ylabel("PICP (coverage)"); ax.set_ylim(0.6, 1.02)
    ax.set_title("Within-regime coverage: marginal vs regime vs recency-adaptive")
    ax.legend(fontsize=7); fig.tight_layout()
    p = f"{outdir}/F7_adaptive_regime_coverage.png"; fig.savefig(p, dpi=200); plt.close(fig)
    print("  figure:", p); return p


# %% ===== CELL 8 : ablation (a/b/c) =========================================
def ablation(df, horizon, alpha):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    d, feats, tr, cal, te = split(df, horizon)
    trf = tr.copy()                                   # train-only; cal held out for conformal
    imp = SimpleImputer(strategy="median").fit(trf[feats])
    rows = []
    # (a) plain vs monotone (accuracy)
    for tag, mc in [("plain", None), ("monotone", MONO_VEC)]:
        m = lgb.LGBMRegressor(monotone_constraints=mc, **LGB_PARAMS).fit(trf[feats], trf[horizon])
        a = acc_metrics(te[horizon].values, m.predict(te[feats]))
        rows.append(dict(ablation="(a) boosting", variant=tag, RMSE=round(a["RMSE"],3),
                         R2=round(a["R2"],3)))
    # (b) global vs Mondrian conformal coverage spread (method model)
    method = lgb.LGBMRegressor(monotone_constraints=MONO_VEC, **LGB_PARAMS).fit(trf[feats], trf[horizon])
    for tag, by in [("global", None), ("Mondrian", "regime")]:
        _, lo, hi, yte, _ = split_conformal(method, imp, cal, te, feats, horizon, alpha, by=by)
        rep = coverage_report(te, lo, hi, yte, by="city", tag=tag)
        sub = rep[rep.scope != "ALL"]["PICP"]
        rows.append(dict(ablation="(b) conformal", variant=tag,
                         RMSE=np.nan, R2=np.nan,
                         cov_spread=f"{sub.min():.3f}-{sub.max():.3f}"))
    # (c) with vs without regime features (accuracy)
    for tag, fl in [("with_regime", feats),
                    ("no_regime", [c for c in feats if c not in REGIME_FEATS])]:
        mc = [MONO_DIR.get(c,0) for c in fl]
        m = lgb.LGBMRegressor(monotone_constraints=mc, **LGB_PARAMS).fit(trf[fl], trf[horizon])
        a = acc_metrics(te[horizon].values, m.predict(te[fl]))
        rows.append(dict(ablation="(c) regime feats", variant=tag, RMSE=round(a["RMSE"],3),
                         R2=round(a["R2"],3)))
    return pd.DataFrame(rows)


# %% ===== CELL 9 : significance (DM + Wilcoxon vs plain LightGBM) ===========
def significance(preds, te, horizon):
    from scipy import stats
    y = te[horizon].values
    base = preds.get("LightGBM(plain)"); method = preds.get("PCMB(monotone)")
    rows = []
    if base is not None and method is not None:
        dbar, p_dm = diebold_mariano(y, method, base)
        e1 = np.abs(y - method); e2 = np.abs(y - base)
        try:    w, p_w = stats.wilcoxon(e1, e2)
        except Exception: p_w = np.nan
        rows.append(dict(comparison="PCMB(monotone) vs LightGBM(plain)",
                         DM_d=round(dbar,4), DM_p=round(p_dm,4) if p_dm==p_dm else None,
                         Wilcoxon_p=round(float(p_w),4) if p_w==p_w else None))
    return pd.DataFrame(rows)


# %% ===== CELL 10 : SHAP + physics-direction check ==========================
def shap_and_physics(art, feats):
    import shap, lightgbm as lgb
    method = art["method"]
    Xs = art["Xtr"].sample(min(4000, len(art["Xtr"])), random_state=SEED)
    expl = shap.TreeExplainer(method); sv = expl.shap_values(Xs)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    # physics check: sign of correlation(feature, shap) vs intended monotone dir
    checks = []
    for c, d in MONO_DIR.items():
        if c in feats:
            j = feats.index(c)
            corr = np.corrcoef(Xs[c].values, sv[:, j])[0, 1]
            if not np.isfinite(corr):
                checks.append(dict(feature=c, intended=d, shap_corr=None,
                                   consistent="n/a")); continue
            ok = (np.sign(corr) == np.sign(d)) or abs(corr) < 0.02
            checks.append(dict(feature=c, intended=d, shap_corr=round(float(corr),3),
                               consistent=bool(ok)))
    return imp, pd.DataFrame(checks), (expl, sv, Xs)


# %% ===== CELL 11/12 : figures ==============================================
def make_figures(outdir, acc, conf_rep, conf_arrays, te, shap_pack, imp_tbl, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_paths = []
    # F1 coverage by regime: global vs Mondrian
    fig, ax = plt.subplots(figsize=(6,4))
    piv = conf_rep[conf_rep.scope != "ALL"].pivot(index="scope", columns="method", values="PICP")
    piv.plot(kind="bar", ax=ax); ax.axhline(1-ALPHA, ls="--", c="k", label="nominal")
    ax.set_ylabel("PICP (coverage)"); ax.set_title("Coverage by regime: global vs Mondrian")
    ax.legend(); fig.tight_layout(); p = f"{outdir}/F1_coverage_by_regime.png"
    fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    # F2 interval width by regime
    fig, ax = plt.subplots(figsize=(6,4))
    piv2 = conf_rep[conf_rep.scope != "ALL"].pivot(index="scope", columns="method", values="MPIW")
    piv2.plot(kind="bar", ax=ax); ax.set_ylabel("MPIW (interval width)")
    ax.set_title("Interval width by regime"); fig.tight_layout()
    p = f"{outdir}/F2_interval_width.png"; fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    # F3 pred vs actual with Mondrian intervals (sample window, per city)
    fig, axes = plt.subplots(2, 1, figsize=(9,6), sharex=False)
    ca = conf_arrays
    for ax, city in zip(axes, te["city"].unique()[:2]):
        idx = np.where((te["city"] == city).values)[0][:300]
        ax.plot(te[horizon].values[idx], label="observed", lw=1)
        ax.plot(ca["yhat"][idx], label="forecast", lw=1)
        ax.fill_between(range(len(idx)), ca["loM"][idx], ca["hiM"][idx], alpha=.25,
                        label="Mondrian 90% PI")
        ax.set_title(f"{city}: forecast vs observed (+ PI)"); ax.legend(fontsize=7)
    fig.tight_layout(); p = f"{outdir}/F3_forecast_intervals.png"
    fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    # F4 reliability diagram (empirical vs nominal coverage, Mondrian sweep)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1],"k--", label="ideal")
    rel = conf_arrays.get("reliab")
    if rel is not None:
        ax.plot(rel["nominal"], rel["empirical_mondrian"], "o-", c="C1", label="Mondrian")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability (Mondrian)"); ax.legend(); fig.tight_layout()
    p = f"{outdir}/F4_reliability.png"; fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    # F5 SHAP importance (top 15)
    fig, ax = plt.subplots(figsize=(6,5))
    top = imp_tbl.head(15)[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"]); ax.set_title("SHAP feature importance")
    fig.tight_layout(); p = f"{outdir}/F5_shap_importance.png"
    fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    # F6 residuals
    fig, ax = plt.subplots(figsize=(6,4))
    resid = ca["yte"] - ca["yhat"]
    ax.hist(resid[np.isfinite(resid)], bins=60); ax.set_title("Residuals (test)")
    ax.set_xlabel("observed - forecast"); fig.tight_layout()
    p = f"{outdir}/F6_residuals.png"; fig.savefig(p, dpi=200); plt.close(fig); fig_paths.append(p)
    print("  figures:", [os.path.basename(x) for x in fig_paths])
    return fig_paths


# %% ===== CELL 13 : secondary SCF degradation (appendix, NOT novelty) =======
def scf_degradation(df, horizon):
    """If station-level n_stations exists, simulate sparser networks by capping
       contributing stations is not possible from city-level table; instead we
       degrade temporally (keep fraction of training rows) as a robustness proxy."""
    import lightgbm as lgb
    d, feats, tr, cal, te = split(df, horizon)
    trf = pd.concat([tr, cal]); rows = []
    ground = [c for c in feats if c.startswith("no2_") or c in CALENDAR]
    for frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        sub = trf.sample(frac=frac, random_state=SEED)
        mg = lgb.LGBMRegressor(**LGB_PARAMS).fit(sub[ground], sub[horizon])
        mf = lgb.LGBMRegressor(**LGB_PARAMS).fit(sub[feats], sub[horizon])
        rows.append(dict(keep_frac=frac,
                         rmse_ground=round(acc_metrics(te[horizon].values, mg.predict(te[ground]))["RMSE"],2),
                         rmse_fusion=round(acc_metrics(te[horizon].values, mf.predict(te[feats]))["RMSE"],2)))
    res = pd.DataFrame(rows); res["compensation"] = (res.rmse_ground - res.rmse_fusion).round(2)
    return res


# %% ===== CELL 14 : run everything + save ===================================
def main():
    df = mount_and_load()
    outdir = output_dir()
    all_acc = {}
    for horizon in HORIZONS:
        print("\n" + "#"*72 + f"\n# HORIZON = {horizon}\n" + "#"*72)
        d, feats, tr, cal, te = split(df, horizon)
        preds, timings, art = train_all(feats, tr, cal, te, horizon)
        acc = accuracy_table(preds, te, horizon)
        print("\n  ACCURACY:\n", acc.to_string(index=False))
        acc.to_csv(f"{outdir}/accuracy_{horizon}.csv", index=False); all_acc[horizon] = acc
        if horizon == PRIMARY_H:
            conf_rep, conf_arr = conformal_experiment(art, cal, te, feats, horizon, ALPHA)
            conf_rep.to_csv(f"{outdir}/conformal_coverage.csv", index=False)
            # CORE NOVELTY: marginal vs regime vs recency-adaptive conformal
            adaptive = conformal_adaptive_experiment(art, cal, te, feats, horizon, ALPHA)
            adaptive.to_csv(f"{outdir}/conformal_adaptive.csv", index=False)
            fig_adaptive(adaptive, outdir, ALPHA)
            abl = ablation(df, horizon, ALPHA); abl.to_csv(f"{outdir}/ablation.csv", index=False)
            print("\n  ABLATION:\n", abl.to_string(index=False))
            sig = significance(preds, te, horizon); sig.to_csv(f"{outdir}/significance.csv", index=False)
            print("\n  SIGNIFICANCE:\n", sig.to_string(index=False))
            imp_tbl, phys, shap_pack = shap_and_physics(art, feats)
            imp_tbl.to_csv(f"{outdir}/shap_importance.csv", index=False)
            phys.to_csv(f"{outdir}/physics_check.csv", index=False)
            print("\n  PHYSICS-DIRECTION CHECK:\n", phys.to_string(index=False))
            make_figures(outdir, acc, conf_rep, conf_arr, te, shap_pack, imp_tbl, horizon)
            pd.DataFrame([dict(model=k, seconds=v) for k,v in timings.items()]
                         ).to_csv(f"{outdir}/timings.csv", index=False)
            scf = scf_degradation(df, horizon); scf.to_csv(f"{outdir}/scf_degradation_appendix.csv", index=False)
            print("\n  SCF (appendix):\n", scf.to_string(index=False))
    # reproducibility stamp
    env = dict(python=platform.python_version(), seed=SEED, alpha=ALPHA,
               horizons=HORIZONS, monotone=MONO_DIR, lgb_params=LGB_PARAMS)
    with open(f"{outdir}/run_config.json", "w") as f: json.dump(env, f, indent=2, default=str)
    print("\nALL DONE. Outputs in:", outdir)
    # best-effort copy to Google Drive (survives if Drive is alive; else download from /content)
    try:
        import shutil
        if os.path.isdir(DRIVE_DIR):
            dst = os.path.join(DRIVE_DIR, OUT_DIR)
            shutil.copytree(outdir, dst, dirs_exist_ok=True)
            print("Copied outputs to Drive:", dst)
    except Exception as e:
        print(f"Could not copy to Drive ({e}). Download the folder from {outdir} "
              f"in the Colab file browser (left panel) instead.")
    print("Send me: conformal_adaptive.csv (CORE), accuracy_y_t6/t1/t24.csv, "
          "conformal_coverage.csv, ablation.csv, significance.csv + figure F7.")


if __name__ == "__main__":
    main()
