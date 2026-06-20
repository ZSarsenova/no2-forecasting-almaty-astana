#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Closes the two remaining result gaps in ONE reproducible run:
#   (1) STATISTICAL SIGNIFICANCE of the headline differences
#       - Diebold-Mariano test (Newey-West + Harvey small-sample correction)
#       - moving-block paired bootstrap -> skill-vs-persistence with 95% CI
#   (2) PEAK / EXCEEDANCE metrics for high-NO2 episodes (POD, FAR, CSI, F1)
#  Reads model_table_NO2_full.csv from Google Drive and saves all CSVs + figures
#  back into the SAME Drive folder under pcmb_conf_outputs/.
#  Run in a notebook (GPU recommended):
#     from google.colab import drive; drive.mount('/content/drive')
#     import os; os.chdir('/content'); exec(open('/content/stats_and_peaks.py').read())
# =============================================================================
import os, time, json, platform, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
SEED = 42; np.random.seed(SEED)

# ---- fixed Drive locations (user-confirmed folder name) ----
DRIVE_DIR  = "/content/drive/MyDrive/air_quality_claude_28_05"   # correct folder
ALT_DIR    = "/content/drive/MyDrive/Air Quality Cloud 2805"     # fallback only
CSV_NAME   = "model_table_NO2_full.csv"

HORIZONS   = ["y_t6", "y_t1", "y_t24"]     # primary first
TRAIN_END, TEST_YEAR = 2023, 2025
SEQ_LEN, EPOCHS, PATIENCE, BATCH, MAX_TRAIN = 12, 40, 8, 256, 60000
RUN_DEEP   = True                          # set False to skip LSTM/GRU/CNN-LSTM
B_BOOT     = 1000                          # bootstrap replications
BLOCK      = 24                            # moving-block length (hours) for autocorrelation
PEAK_PCTL  = 90                            # high-NO2 episode threshold = this percentile of TRAIN

AUTOCORR=["no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12","no2_lag_24","no2_lag_48","no2_lag_168",
 "no2_rollmean_3","no2_rollmean_6","no2_rollmean_12","no2_rollmean_24","no2_rollstd_3","no2_rollstd_6",
 "no2_rollstd_24","no2_rollmin_24","no2_rollmax_24","no2_diff_1","no2_diff_24"]
CALENDAR=["hour","dayofweek","month","dayofyear","is_weekend","heating_season","hour_sin","hour_cos","doy_sin","doy_cos"]
ERA5=["era5_u10","era5_v10","era5_d2m","era5_t2m","era5_blh","era5_sp","era5_tp","era5_ssrd","era5_tcc","era5_wind_speed","era5_wind_dir"]
CAMS=["cams_no2_surf","cams_o3","cams_so2","cams_co"]
DERIVED=["ventilation_coefficient","stagnation_indicator","atmospheric_dryness","winter_inversion_indicator","cams_bias_correction"]
S5P=["s5p_no2_trop_column"]
FEATURES=AUTOCORR+CALENDAR+ERA5+CAMS+DERIVED+S5P
MONO={"era5_blh":-1,"era5_wind_speed":-1,"ventilation_coefficient":-1,"winter_inversion_indicator":1,
      "stagnation_indicator":1,"s5p_no2_trop_column":1,"cams_no2_surf":1}
LGB=dict(n_estimators=2000,num_leaves=63,max_depth=10,learning_rate=0.05,subsample=0.8,
         colsample_bytree=0.8,min_child_samples=40,n_jobs=-1,random_state=SEED,verbose=-1)
SEQ_FEATS=AUTOCORR[:8]+["era5_blh","era5_wind_speed","era5_t2m","hour_sin","hour_cos"]

def drive_dir():
    if not (os.path.isdir(DRIVE_DIR) or os.path.isdir(ALT_DIR)):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e: print("[drive] mount skipped:", e)
    d = DRIVE_DIR if os.path.isdir(DRIVE_DIR) else (ALT_DIR if os.path.isdir(ALT_DIR) else None)
    if d is None: raise FileNotFoundError(f"Drive folder not found ({DRIVE_DIR}). Mount Drive first.")
    return d

# ---------- metrics ----------
def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))
def skill(y,pm,pr): return float(1 - np.mean((y-pm)**2)/np.mean((y-pr)**2))

def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error loss differential d=e1^2-e2^2 (mean>0 => model2 better).
       Newey-West variance with lag h-1 + Harvey-Leybourne-Newbold small-sample correction."""
    from scipy.stats import t
    d = e1**2 - e2**2; n = len(d); dbar = d.mean()
    g0 = np.mean((d-dbar)**2); s = g0
    for k in range(1, h):
        s += 2*(1-k/h)*np.mean((d[k:]-dbar)*(d[:-k]-dbar))
    var = s/n
    if var <= 0: return np.nan, np.nan
    DM = dbar/np.sqrt(var)
    DM *= np.sqrt(max((n+1-2*h+h*(h-1)/n)/n, 1e-9))
    p = 2*(1 - t.cdf(abs(DM), df=n-1))
    return float(DM), float(p)

def block_boot_skill(y, pm, pr, B=B_BOOT, block=BLOCK, seed=SEED):
    """Moving-block paired bootstrap of skill(model vs ref). Returns point, CI95, two-sided p vs 0."""
    rng = np.random.default_rng(seed); n = len(y); nb = int(np.ceil(n/block))
    starts = np.arange(0, n-block+1) if n>block else np.array([0])
    em = (y-pm)**2; er = (y-pr)**2
    out = np.empty(B)
    for b in range(B):
        s = rng.choice(starts, nb, replace=True)
        idx = np.concatenate([np.arange(x, x+block) for x in s])[:n]
        out[b] = 1 - em[idx].mean()/er[idx].mean()
    point = 1 - em.mean()/er.mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2*min(np.mean(out <= 0), np.mean(out >= 0)); p = min(p, 1.0)
    return float(point), float(lo), float(hi), float(p)

def exceedance(y, p, thr):
    """POD (hit rate), FAR (false-alarm ratio), CSI, F1 for predicting y>=thr."""
    yt = y >= thr; pt = p >= thr
    TP = int(np.sum(yt & pt)); FP = int(np.sum(~yt & pt)); FN = int(np.sum(yt & ~pt))
    POD = TP/(TP+FN) if (TP+FN) else np.nan
    FAR = FP/(TP+FP) if (TP+FP) else np.nan
    CSI = TP/(TP+FP+FN) if (TP+FP+FN) else np.nan
    F1  = 2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) else np.nan
    return dict(base_rate=round(float(np.mean(yt)),3), POD=round(POD,3) if POD==POD else np.nan,
                FAR=round(FAR,3) if FAR==FAR else np.nan, CSI=round(CSI,3) if CSI==CSI else np.nan,
                F1=round(F1,3) if F1==F1 else np.nan, n_events=int(np.sum(yt)))

# ---------- sequence builder (positional mapping) ----------
def make_seq(frame, keep, H, L):
    f = frame.reset_index(drop=True); f = f.loc[f.sort_values(["city","timestamp"]).index]
    Xs, ys, pos = [], [], []
    for _, g in f.groupby("city", sort=False):
        M = g[keep].fillna(g[keep].median()).to_numpy("float32"); y = g[H].to_numpy("float32")
        gp = g.index.to_numpy()
        for i in range(L, len(g)): Xs.append(M[i-L:i]); ys.append(y[i]); pos.append(gp[i])
    if not Xs: return None, None, None
    return np.asarray(Xs,"float32"), np.asarray(ys,"float32"), np.asarray(pos,int)

# ---------- per-horizon predictions ----------
def predict_all(df, H):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    d = df.dropna(subset=[H]).reset_index(drop=True); d["year"]=d["timestamp"].dt.year
    tr = d[d.year<=TRAIN_END]; te = d[d.year==TEST_YEAR].reset_index(drop=True)
    feats=[c for c in FEATURES if c in d.columns]
    imp=SimpleImputer(strategy="median").fit(tr[feats]); sc=StandardScaler().fit(imp.transform(tr[feats]))
    Xtr,Xte=sc.transform(imp.transform(tr[feats])),sc.transform(imp.transform(te[feats]))
    P={}
    P["Persistence"]   = te["persistence"].values if "persistence" in te else te[H].values
    P["Ridge(AR)"]     = Ridge(alpha=1.0).fit(Xtr,tr[H]).predict(Xte)
    P["RandomForest"]  = RandomForestRegressor(n_estimators=150,max_depth=18,min_samples_leaf=20,
                            max_samples=0.6,n_jobs=-1,random_state=SEED).fit(imp.transform(tr[feats]),tr[H]).predict(imp.transform(te[feats]))
    P["XGBoost"]       = XGBRegressor(n_estimators=600,max_depth=8,learning_rate=0.05,subsample=0.8,
                            colsample_bytree=0.8,n_jobs=-1,random_state=SEED,verbosity=0).fit(tr[feats],tr[H]).predict(te[feats])
    P["LightGBM(plain)"]   = lgb.LGBMRegressor(**LGB).fit(tr[feats],tr[H]).predict(te[feats])
    P["LightGBM(monotone)"]= lgb.LGBMRegressor(monotone_constraints=[MONO.get(c,0) for c in feats],**LGB).fit(tr[feats],tr[H]).predict(te[feats])
    covered = np.ones(len(te), bool)
    if RUN_DEEP:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks, optimizers
        tf.random.set_seed(SEED)
        keep=[c for c in SEQ_FEATS if c in d.columns]
        Xtr_s,ytr_s,_   = make_seq(tr, keep, H, SEQ_LEN)
        Xte_s,yte_s,pos = make_seq(te, keep, H, SEQ_LEN)
        mu,sd=Xtr_s.mean((0,1)),Xtr_s.std((0,1))+1e-6; Xtr_s,Xte_s=(Xtr_s-mu)/sd,(Xte_s-mu)/sd
        ym,ys_=ytr_s.mean(),ytr_s.std()+1e-6; ytr_z=(ytr_s-ym)/ys_; nf=Xtr_s.shape[-1]
        def build(kind):
            inp=layers.Input((SEQ_LEN,nf))
            if kind=="LSTM": x=layers.LSTM(64)(inp)
            elif kind=="GRU": x=layers.GRU(64)(inp)
            else:
                x=layers.Conv1D(32,3,padding="same")(inp); x=layers.BatchNormalization()(x)
                x=layers.Activation("relu")(x); x=layers.LSTM(64)(x)
            x=layers.Dense(32,activation="relu")(x)
            m=models.Model(inp,layers.Dense(1)(x)); m.compile(optimizers.Adam(1e-3,clipnorm=1.0),"mse"); return m
        es=callbacks.EarlyStopping(patience=PATIENCE,restore_best_weights=True,monitor="val_loss")
        rl=callbacks.ReduceLROnPlateau(patience=4,factor=0.5,monitor="val_loss")
        for kind in ["LSTM","GRU","CNN-LSTM"]:
            m=build(kind); m.fit(Xtr_s,ytr_z,validation_split=0.1,epochs=EPOCHS,batch_size=BATCH,callbacks=[es,rl],verbose=0)
            out=np.full(len(te),np.nan); out[pos]=m.predict(Xte_s,verbose=0).ravel()*ys_+ym; P[kind]=out
            covered &= np.isfinite(out)
    y=te[H].values; city=te["city"].values
    thr={c: float(np.percentile(tr[tr.city==c][H], PEAK_PCTL)) for c in np.unique(city)}
    return P, y, city, covered, thr

def main():
    t0=time.time()
    import sklearn, scipy, lightgbm, xgboost
    dd=drive_dir(); OUT=os.path.join(dd,"pcmb_conf_outputs"); os.makedirs(OUT,exist_ok=True)
    vers=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
              sklearn=sklearn.__version__, scipy=scipy.__version__, lightgbm=lightgbm.__version__,
              xgboost=xgboost.__version__, seed=SEED, B_boot=B_BOOT, block=BLOCK, peak_pctl=PEAK_PCTL)
    if RUN_DEEP:
        import tensorflow as tf; vers["tensorflow"]=tf.__version__; vers["GPU"]=bool(tf.config.list_physical_devices("GPU"))
    print("[env]", json.dumps(vers)); json.dump(vers, open(os.path.join(OUT,"stats_run_config.json"),"w"), indent=2)
    df=pd.read_csv(os.path.join(dd,CSV_NAME), parse_dates=["timestamp"])
    a25=df[(df.city=="Almaty")&(df.timestamp.dt.year==2025)]["no2"].median()
    print(f"[data] rows={len(df)} | Almaty 2025 NO2 median={a25:.1f} (clean ~30-40)")

    scopes=lambda city: [("ALL", np.ones(len(city),bool))]+[(c,(city==c)) for c in np.unique(city)]
    PAIRS=[("LSTM","Persistence"),("Ridge(AR)","Persistence"),("LightGBM(monotone)","Persistence"),
           ("GRU","Persistence"),("LSTM","Ridge(AR)")]
    sig_rows, skill_rows, peak_rows = [], [], []
    for H in HORIZONS:
        print(f"\n### {H}: training models ...")
        P, y, city, cov, thr = predict_all(df, H)
        models_avail=[m for m in P]
        print(f"   models={models_avail} | covered rows={cov.sum()}/{len(y)} | thresholds={ {k:round(v,1) for k,v in thr.items()} }")
        for sc, m_all in scopes(city):
            m = m_all & cov
            if m.sum() < 50: continue
            ys=y[m]; href=int(H.split("t")[-1])
            # significance: pairs
            for A,Bm in PAIRS:
                if A in P and Bm in P:
                    eA=ys-P[A][m]; eB=ys-P[Bm][m]
                    DM,pdm=dm_test(eA,eB,h=max(href,1))
                    sk,lo,hi,pb=block_boot_skill(ys,P[A][m],P[Bm][m])
                    better = A if np.mean(eA**2)<np.mean(eB**2) else Bm
                    sig_rows.append(dict(horizon=H,scope=sc,model_A=A,ref_B=Bm,
                        rmse_A=round(rmse(ys,P[A][m]),2),rmse_B=round(rmse(ys,P[Bm][m]),2),
                        skill_A_vs_B=round(sk,3),ci_lo=round(lo,3),ci_hi=round(hi,3),
                        boot_p=round(pb,4),DM_stat=round(DM,2) if DM==DM else np.nan,
                        DM_p=round(pdm,4) if pdm==pdm else np.nan,better=better,
                        sig_05=bool((pb<0.05) and (pdm<0.05)) ))
            # skill vs persistence + CI for every model
            for mdl in models_avail:
                if mdl=="Persistence": continue
                sk,lo,hi,pb=block_boot_skill(ys,P[mdl][m],P["Persistence"][m])
                skill_rows.append(dict(horizon=H,scope=sc,model=mdl,skill_vs_persistence=round(sk,3),
                    ci_lo=round(lo,3),ci_hi=round(hi,3),boot_p=round(pb,4),beats_persistence=bool(lo>0)))
            # peak / exceedance per model (city-specific threshold; for ALL use union mask per row)
            for mdl in models_avail:
                if sc=="ALL":
                    thr_vec=np.array([thr[c] for c in city[m]]); ex=exceedance(ys,P[mdl][m],thr_vec)
                else:
                    ex=exceedance(ys,P[mdl][m],thr[sc])
                peak_rows.append(dict(horizon=H,scope=sc,model=mdl,**ex))

    sig=pd.DataFrame(sig_rows); skl=pd.DataFrame(skill_rows); pk=pd.DataFrame(peak_rows)
    sig.to_csv(os.path.join(OUT,"significance_tests.csv"),index=False)
    skl.to_csv(os.path.join(OUT,"skill_ci.csv"),index=False)
    pk.to_csv(os.path.join(OUT,"exceedance_metrics.csv"),index=False)
    print("\n==== SIGNIFICANCE (+6h, key pairs) ====")
    print(sig[sig.horizon=="y_t6"][["scope","model_A","ref_B","skill_A_vs_B","ci_lo","ci_hi","boot_p","DM_p","better","sig_05"]].to_string(index=False))
    print("\n==== EXCEEDANCE F1 (+6h) ====")
    print(pk[pk.horizon=="y_t6"].pivot_table(index="model",columns="scope",values="F1").to_string())

    # figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        s6=skl[skl.horizon=="y_t6"]
        for scope in ["Almaty","Astana"]:
            sub=s6[s6.scope==scope].sort_values("skill_vs_persistence")
            fig,ax=plt.subplots(figsize=(7,4))
            err=[sub["skill_vs_persistence"]-sub["ci_lo"], sub["ci_hi"]-sub["skill_vs_persistence"]]
            ax.barh(sub["model"], sub["skill_vs_persistence"], xerr=err, capsize=3)
            ax.axvline(0,ls="--",c="k"); ax.set_xlabel("skill vs persistence (95% block-bootstrap CI)")
            ax.set_title(f"+6h skill over persistence — {scope}"); fig.tight_layout()
            fig.savefig(os.path.join(OUT,f"skill_CI_{scope}.png"),dpi=200); plt.close(fig)
        p6=pk[pk.horizon=="y_t6"]; piv=p6[p6.scope!="ALL"].pivot(index="model",columns="scope",values="F1")
        fig,ax=plt.subplots(figsize=(7,4)); piv.plot(kind="bar",ax=ax)
        ax.set_ylabel("F1 (high-NO2 episode detection)"); ax.set_title("+6h exceedance F1 by city"); fig.tight_layout()
        fig.savefig(os.path.join(OUT,"exceedance_F1.png"),dpi=200); plt.close(fig)
        print("[fig] saved skill_CI_Almaty.png, skill_CI_Astana.png, exceedance_F1.png")
    except Exception as e: print("[fig] skipped:", e)

    print(f"\nSAVED TO GOOGLE DRIVE: {OUT}")
    print("Files:", [f for f in os.listdir(OUT) if f.startswith(('significance','skill_','exceedance','stats_'))])
    print(f"[done] {round(time.time()-t0,1)}s")

if __name__=="__main__":
    main()
