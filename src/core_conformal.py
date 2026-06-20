#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  CORE result only (fast, robust). Computes the novelty experiment:
#  marginal vs regime-conditional vs recency-adaptive conformal, + a mini
#  benchmark (persistence / Ridge / plain LGBM / monotone LGBM).
#  No RandomForest / XGBoost / TensorFlow / SHAP -> runs in ~1-2 min and
#  cannot hang. PRINTS everything to console (so you can paste numbers even
#  if file saving fails) and also saves to /content/pcmb_conf_outputs.
#
#  Run:  !python core_conformal.py
#  It searches common locations for model_table_NO2_full.csv automatically.
# =============================================================================
import os, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from collections import deque
SEED = 42

HORIZON, TRAIN_END, CALIB_YEAR, TEST_YEAR, ALPHA, WINDOW = "y_t6", 2023, 2024, 2025, 0.10, 336

AUTOCORR=["no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12","no2_lag_24","no2_lag_48",
 "no2_lag_168","no2_rollmean_3","no2_rollmean_6","no2_rollmean_12","no2_rollmean_24","no2_rollstd_3",
 "no2_rollstd_6","no2_rollstd_24","no2_rollmin_24","no2_rollmax_24","no2_diff_1","no2_diff_24"]
CALENDAR=["hour","dayofweek","month","dayofyear","is_weekend","heating_season","hour_sin","hour_cos","doy_sin","doy_cos"]
ERA5=["era5_u10","era5_v10","era5_d2m","era5_t2m","era5_blh","era5_sp","era5_tp","era5_ssrd","era5_tcc","era5_wind_speed","era5_wind_dir"]
CAMS=["cams_no2_surf","cams_o3","cams_so2","cams_co"]
DERIVED=["ventilation_coefficient","stagnation_indicator","atmospheric_dryness","winter_inversion_indicator","cams_bias_correction"]
S5P=["s5p_no2_trop_column"]
FEATURES=AUTOCORR+CALENDAR+ERA5+CAMS+DERIVED+S5P
MONO_DIR={"era5_blh":-1,"era5_wind_speed":-1,"ventilation_coefficient":-1,"winter_inversion_indicator":+1,
          "stagnation_indicator":+1,"s5p_no2_trop_column":+1,"cams_no2_surf":+1}
LGB=dict(n_estimators=2000,num_leaves=63,max_depth=10,learning_rate=0.05,subsample=0.8,
         colsample_bytree=0.8,min_child_samples=40,n_jobs=-1,random_state=SEED,verbose=-1)

def find_csv():
    for p in ["model_table_NO2_full.csv","/content/work/model_table_NO2_full.csv",
              "/content/model_table_NO2_full.csv"]:
        if os.path.exists(p): return p
    hits = glob.glob("/content/**/model_table_NO2_full.csv", recursive=True)
    if hits: return hits[0]
    raise FileNotFoundError("model_table_NO2_full.csv not found - upload it to /content")

def outdir():
    d = "/content/pcmb_conf_outputs" if os.path.isdir("/content") else "./pcmb_conf_outputs"
    os.makedirs(d, exist_ok=True); return d

def r2(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); ss=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/ss) if ss>0 else np.nan
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def quant(a,al):
    a=np.sort(np.asarray(a,float)); 
    return np.nan if len(a)==0 else a[min(int(np.ceil((len(a)+1)*(1-al))),len(a))-1]

def main():
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    OUT=outdir()
    df=pd.read_csv(find_csv(), parse_dates=["timestamp"]); df["year"]=df["timestamp"].dt.year
    df["regime_fine"]=df["city"].astype(str)+"|"+np.where(df["heating_season"].fillna(0).astype(int).eq(1),"heat","warm")
    df=df.dropna(subset=[HORIZON]).reset_index(drop=True)
    feats=[c for c in FEATURES if c in df.columns]
    tr=df[df.year<=TRAIN_END]; cal=df[df.year==CALIB_YEAR]; te=df[df.year==TEST_YEAR].sort_values("timestamp")
    cov=100*te["cams_no2_surf"].notna().mean()
    print(f"rows={len(df)} | CAMS in test {TEST_YEAR}: {cov:.0f}% | train={len(tr)} calib={len(cal)} test={len(te)}")

    # ---- mini benchmark (point models trained on train-only, no leakage) ----
    imp=SimpleImputer(strategy="median").fit(tr[feats]); sc=StandardScaler().fit(imp.transform(tr[feats]))
    Xtr,Xte=sc.transform(imp.transform(tr[feats])),sc.transform(imp.transform(te[feats]))
    plain =lgb.LGBMRegressor(**LGB).fit(tr[feats],tr[HORIZON])
    method=lgb.LGBMRegressor(monotone_constraints=[MONO_DIR.get(c,0) for c in feats],**LGB).fit(tr[feats],tr[HORIZON])
    preds={"Persistence":te["persistence"].values if "persistence" in te else te[HORIZON].values,
           "Ridge(AR)":Ridge(alpha=1.0).fit(Xtr,tr[HORIZON]).predict(Xte),
           "LightGBM(plain)":plain.predict(te[feats]),
           "PCMB(monotone)":method.predict(te[feats])}
    rows=[]
    for nm,p in preds.items():
        for sc_,mask in [("ALL",np.ones(len(te),bool))]+[(c,(te.city==c).values) for c in te.city.unique()]:
            rows.append(dict(model=nm,scope=sc_,RMSE=round(rmse(te[HORIZON].values[mask],p[mask]),2),
                             R2=round(r2(te[HORIZON].values[mask],p[mask]),3)))
    acc=pd.DataFrame(rows).sort_values(["scope","RMSE"])
    print("\n=== MINI BENCHMARK (+6h) ===\n", acc.to_string(index=False))
    acc.to_csv(f"{OUT}/mini_benchmark.csv",index=False)

    # ---- CORE: marginal vs regime vs recency-adaptive conformal ----
    rc=np.abs(cal[HORIZON].values-method.predict(cal[feats])); reg_cal=cal["regime_fine"].values
    yhat=method.predict(te[feats]); yte=te[HORIZON].values; reg=te["regime_fine"].values; n=len(te)
    resid=np.abs(yte-yhat)
    qg=quant(rc,ALPHA); lo1,hi1=yhat-qg,yhat+qg                                  # (1) marginal
    lo2,hi2=np.empty(n),np.empty(n)                                              # (2) regime static
    for r in np.unique(reg):
        q=quant(rc[reg_cal==r],ALPHA) if (reg_cal==r).sum()>=30 else qg
        m=reg==r; lo2[m],hi2[m]=yhat[m]-q,yhat[m]+q
    lo3,hi3=np.empty(n),np.empty(n)                                              # (3) adaptive regime
    buf={r:deque((np.sort(rc[reg_cal==r])[-WINDOW:] if (reg_cal==r).any() else np.sort(rc)[-WINDOW:]).tolist(),maxlen=WINDOW) for r in np.unique(reg)}
    for i in range(n):
        b=buf[reg[i]]; q=quant(np.fromiter(b,float) if len(b) else rc,ALPHA)
        lo3[i],hi3[i]=yhat[i]-q,yhat[i]+q; b.append(resid[i])
    def rep(lo,hi,tag):
        out=[dict(scheme=tag,regime="ALL",PICP=round(float(np.mean((yte>=lo)&(yte<=hi))),3),MPIW=round(float(np.mean(hi-lo)),1))]
        for r in np.unique(reg):
            m=reg==r; out.append(dict(scheme=tag,regime=str(r),
                PICP=round(float(np.mean((yte[m]>=lo[m])&(yte[m]<=hi[m]))),3),MPIW=round(float(np.mean(hi[m]-lo[m])),1)))
        return pd.DataFrame(out)
    res=pd.concat([rep(lo1,hi1,"1.Static-marginal"),rep(lo2,hi2,"2.Static-regime"),rep(lo3,hi3,"3.Adaptive-regime")],ignore_index=True)
    nom=1-ALPHA
    print("\n=== CORE: PICP within finer regimes (nominal %.0f%%) ===" % (100*nom))
    print(res.pivot(index="regime",columns="scheme",values="PICP").to_string())
    print("\ncalibration quality  mean|PICP-%.2f| across regimes (LOWER = BETTER):" % nom)
    for tag in ["1.Static-marginal","2.Static-regime","3.Adaptive-regime"]:
        sub=res[(res.scheme==tag)&(res.regime!="ALL")]
        print(f"   {tag}: {np.mean(np.abs(sub['PICP']-nom)):.3f}")
    res.to_csv(f"{OUT}/conformal_adaptive.csv",index=False)

    # figure F7
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        piv=res[res.regime!="ALL"].pivot(index="regime",columns="scheme",values="PICP")
        fig,ax=plt.subplots(figsize=(7,4)); piv.plot(kind="bar",ax=ax)
        ax.axhline(nom,ls="--",c="k",label="nominal"); ax.set_ylim(0.6,1.02); ax.set_ylabel("PICP")
        ax.set_title("Within-regime coverage: marginal vs regime vs adaptive"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/F7_adaptive_regime_coverage.png",dpi=200); plt.close(fig)
    except Exception as e: print("figure skipped:",e)

    print("\nSAVED to:",OUT,"->",os.listdir(OUT))
    print("If the folder is empty in the file browser, the numbers above are still valid - paste them to me.")

if __name__=="__main__":
    main()
