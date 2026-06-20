#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Peak / exceedance metrics, REDONE properly.
#  Thresholds (3): per-city 75th pct, COMMON 75th pct (pooled train), and
#  regulatory (WHO daily NO2 = 25 ug/m3 'elevated'; EU 1-h = 200 ug/m3 'severe').
#  Alarm rules (2): (A) point forecast yhat>=thr; (B) upper bound of the
#  recency-adaptive REGIME-conditional conformal interval (yhat+q_regime)>=thr.
#  -> ties the peak analysis to the paper's novelty (adaptive conformal).
#  Metrics: POD, FAR, CSI, F1, base_rate, n_events.
#  Reads model_table from Drive, saves CSV+figures to Drive/pcmb_conf_outputs.
#  Run:  from google.colab import drive; drive.mount('/content/drive')
#        import os; os.chdir('/content'); exec(open('/content/exceedance_v2.py').read())
# =============================================================================
import os, time, json, platform, warnings, numpy as np, pandas as pd
from collections import deque
warnings.filterwarnings("ignore"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
SEED=42; np.random.seed(SEED)

DRIVE_DIR="/content/drive/MyDrive/air_quality_claude_28_05"
ALT_DIR  ="/content/drive/MyDrive/Air Quality Cloud 2805"
CSV_NAME ="model_table_NO2_full.csv"
HORIZONS=["y_t6","y_t1","y_t24"]; TRAIN_END,CALIB_YEAR,TEST_YEAR=2023,2024,2025
ALPHA=0.10; WINDOW=336; SEQ_LEN,EPOCHS,PATIENCE,BATCH=12,40,8,256
WHO_DAILY=25.0; EU_HOURLY=200.0; PCTL=75
RUN_DEEP=True

AUTOCORR=["no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12","no2_lag_24","no2_lag_48","no2_lag_168",
 "no2_rollmean_3","no2_rollmean_6","no2_rollmean_12","no2_rollmean_24","no2_rollstd_3","no2_rollstd_6",
 "no2_rollstd_24","no2_rollmin_24","no2_rollmax_24","no2_diff_1","no2_diff_24"]
CALENDAR=["hour","dayofweek","month","dayofyear","is_weekend","heating_season","hour_sin","hour_cos","doy_sin","doy_cos"]
ERA5=["era5_u10","era5_v10","era5_d2m","era5_t2m","era5_blh","era5_sp","era5_tp","era5_ssrd","era5_tcc","era5_wind_speed","era5_wind_dir"]
CAMS=["cams_no2_surf","cams_o3","cams_so2","cams_co"]; DERIVED=["ventilation_coefficient","stagnation_indicator","atmospheric_dryness","winter_inversion_indicator","cams_bias_correction"]; S5P=["s5p_no2_trop_column"]
FEATURES=AUTOCORR+CALENDAR+ERA5+CAMS+DERIVED+S5P
MONO={"era5_blh":-1,"era5_wind_speed":-1,"ventilation_coefficient":-1,"winter_inversion_indicator":1,"stagnation_indicator":1,"s5p_no2_trop_column":1,"cams_no2_surf":1}
LGB=dict(n_estimators=2000,num_leaves=63,max_depth=10,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_samples=40,n_jobs=-1,random_state=SEED,verbose=-1)
SEQ_FEATS=AUTOCORR[:8]+["era5_blh","era5_wind_speed","era5_t2m","hour_sin","hour_cos"]

def drive_dir():
    if not (os.path.isdir(DRIVE_DIR) or os.path.isdir(ALT_DIR)):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e: print("[drive] mount skipped:",e)
    d=DRIVE_DIR if os.path.isdir(DRIVE_DIR) else (ALT_DIR if os.path.isdir(ALT_DIR) else None)
    if d is None: raise FileNotFoundError("Drive folder not found")
    return d

def quant(a,al):
    a=np.sort(np.asarray(a,float))
    return np.nan if len(a)==0 else a[min(int(np.ceil((len(a)+1)*(1-al))),len(a))-1]

def exceed(y,score,thr):
    """score>=thr is the alarm; y>=thr is the true event. thr may be scalar or per-row vector."""
    yt=y>=thr; pt=score>=thr
    TP=int(np.sum(yt&pt)); FP=int(np.sum(~yt&pt)); FN=int(np.sum(yt&~pt))
    POD=TP/(TP+FN) if (TP+FN) else np.nan
    FAR=FP/(TP+FP) if (TP+FP) else np.nan
    CSI=TP/(TP+FP+FN) if (TP+FP+FN) else np.nan
    F1=2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) else np.nan
    return dict(base_rate=round(float(np.mean(yt)),3),n_events=int(np.sum(yt)),
        POD=round(POD,3) if POD==POD else np.nan, FAR=round(FAR,3) if FAR==FAR else np.nan,
        CSI=round(CSI,3) if CSI==CSI else np.nan, F1=round(F1,3) if F1==F1 else np.nan)

def make_seq(frame,keep,H,L):
    f=frame.reset_index(drop=True); f=f.loc[f.sort_values(["city","timestamp"]).index]
    Xs,ys,pos=[],[],[]
    for _,g in f.groupby("city",sort=False):
        M=g[keep].fillna(g[keep].median()).to_numpy("float32"); y=g[H].to_numpy("float32"); gp=g.index.to_numpy()
        for i in range(L,len(g)): Xs.append(M[i-L:i]); ys.append(y[i]); pos.append(gp[i])
    if not Xs: return None,None,None
    return np.asarray(Xs,"float32"),np.asarray(ys,"float32"),np.asarray(pos,int)

def adaptive_q_per_row(model, cal, te, feats, H, regfield):
    """Recency-adaptive regime-conditional conformal radius q for EACH test row (online)."""
    rc=np.abs(cal[H].values-model.predict(cal[feats])); reg_cal=cal[regfield].values
    yhat=model.predict(te[feats]); reg=te[regfield].values; resid=np.abs(te[H].values-yhat); n=len(te)
    buf={r:deque((np.sort(rc[reg_cal==r])[-WINDOW:] if (reg_cal==r).any() else np.sort(rc)[-WINDOW:]).tolist(),maxlen=WINDOW) for r in np.unique(reg)}
    q=np.empty(n)
    for i in range(n):
        b=buf[reg[i]]; q[i]=quant(np.fromiter(b,float) if len(b) else rc,ALPHA); b.append(resid[i])
    return yhat, q

def main():
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    t0=time.time(); dd=drive_dir(); OUT=os.path.join(dd,"pcmb_conf_outputs"); os.makedirs(OUT,exist_ok=True)
    df=pd.read_csv(os.path.join(dd,CSV_NAME),parse_dates=["timestamp"]); df["year"]=df["timestamp"].dt.year
    df["regime_fine"]=df["city"].astype(str)+"|"+np.where(df["heating_season"].fillna(0).astype(int).eq(1),"heat","warm")
    feats=[c for c in FEATURES if c in df.columns]
    print("[data] rows=",len(df),"| thresholds: WHO_daily",WHO_DAILY,"EU_hourly",EU_HOURLY,"pct",PCTL)

    rows=[]
    for H in HORIZONS:
        d=df.dropna(subset=[H]).reset_index(drop=True)
        tr=d[d.year<=TRAIN_END]; cal=d[d.year==CALIB_YEAR]; te=d[d.year==TEST_YEAR].sort_values("timestamp").reset_index(drop=True)
        imp=SimpleImputer(strategy="median").fit(tr[feats]); sc=StandardScaler().fit(imp.transform(tr[feats]))
        Xtr,Xte=sc.transform(imp.transform(tr[feats])),sc.transform(imp.transform(te[feats]))
        # point models
        P={}
        P["Persistence"]=te["persistence"].values if "persistence" in te else te[H].values
        P["Ridge(AR)"]=Ridge(1.0).fit(Xtr,tr[H]).predict(Xte)
        P["RandomForest"]=RandomForestRegressor(n_estimators=150,max_depth=18,min_samples_leaf=20,max_samples=0.6,n_jobs=-1,random_state=SEED).fit(imp.transform(tr[feats]),tr[H]).predict(imp.transform(te[feats]))
        P["XGBoost"]=XGBRegressor(n_estimators=600,max_depth=8,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,random_state=SEED,verbosity=0).fit(tr[feats],tr[H]).predict(te[feats])
        P["LightGBM(plain)"]=lgb.LGBMRegressor(**LGB).fit(tr[feats],tr[H]).predict(te[feats])
        mono=lgb.LGBMRegressor(monotone_constraints=[MONO.get(c,0) for c in feats],**LGB).fit(tr[feats],tr[H])
        P["LightGBM(monotone)"]=mono.predict(te[feats])
        if RUN_DEEP:
            import tensorflow as tf
            from tensorflow.keras import layers,models,callbacks,optimizers
            tf.random.set_seed(SEED); keep=[c for c in SEQ_FEATS if c in d.columns]
            Xs,ys,_=make_seq(tr,keep,H,SEQ_LEN); Xv,yv,pos=make_seq(te,keep,H,SEQ_LEN)
            mu,sd=Xs.mean((0,1)),Xs.std((0,1))+1e-6; Xs,Xv=(Xs-mu)/sd,(Xv-mu)/sd
            ym,ys_=ys.mean(),ys.std()+1e-6; yz=(ys-ym)/ys_; nf=Xs.shape[-1]
            def build(k):
                inp=layers.Input((SEQ_LEN,nf))
                if k=="LSTM": x=layers.LSTM(64)(inp)
                elif k=="GRU": x=layers.GRU(64)(inp)
                else:
                    x=layers.Conv1D(32,3,padding="same")(inp); x=layers.BatchNormalization()(x); x=layers.Activation("relu")(x); x=layers.LSTM(64)(x)
                x=layers.Dense(32,activation="relu")(x); m=models.Model(inp,layers.Dense(1)(x)); m.compile(optimizers.Adam(1e-3,clipnorm=1.0),"mse"); return m
            es=callbacks.EarlyStopping(patience=PATIENCE,restore_best_weights=True,monitor="val_loss"); rl=callbacks.ReduceLROnPlateau(patience=4,factor=0.5,monitor="val_loss")
            for k in ["LSTM","GRU","CNN-LSTM"]:
                m=build(k); m.fit(Xs,yz,validation_split=0.1,epochs=EPOCHS,batch_size=BATCH,callbacks=[es,rl],verbose=0)
                out=np.full(len(te),np.nan); out[pos]=m.predict(Xv,verbose=0).ravel()*ys_+ym; P[k]=out
        # conformal upper bound for the boosting model (novelty link), per test row
        yhat_m,q_row=adaptive_q_per_row(mono,cal,te,feats,H,"regime_fine")
        upper_conf=yhat_m+q_row
        y=te[H].values; city=te["city"].values
        # thresholds
        thr_city={c:float(np.percentile(tr[tr.city==c][H],PCTL)) for c in np.unique(city)}
        thr_common=float(np.percentile(tr[H],PCTL))
        thr_sets={"pct75_city":("city",thr_city),"pct75_common":("scalar",thr_common),
                  "WHO_daily_25":("scalar",WHO_DAILY),"EU_hourly_200":("scalar",EU_HOURLY)}
        scopes=[("ALL",np.ones(len(city),bool))]+[(c,(city==c)) for c in np.unique(city)]
        for tname,(kind,tv) in thr_sets.items():
            for sc,mask in scopes:
                if mask.sum()<30: continue
                thr=(np.array([tv[c] for c in city[mask]]) if kind=="city" else tv)
                # (A) point-forecast alarm, every model
                for mdl in P:
                    rows.append(dict(horizon=H,threshold=tname,alarm="A_point",scope=sc,model=mdl,
                                     **exceed(y[mask],P[mdl][mask],thr)))
                # (B) conformal-upper-bound alarm, boosting model (novelty link)
                rows.append(dict(horizon=H,threshold=tname,alarm="B_conformal_upper",scope=sc,model="LightGBM(monotone)+adaptiveConf",
                                 **exceed(y[mask],upper_conf[mask],thr)))
        print(f"  [{H}] done; city thr(p75)={ {k:round(v,1) for k,v in thr_city.items()} } common={thr_common:.1f}")

    res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"exceedance_metrics_v2.csv"),index=False)
    print("\n==== +6h, threshold=pct75_city, alarm=A_point (F1) ====")
    a=res[(res.horizon=="y_t6")&(res.threshold=="pct75_city")&(res.alarm=="A_point")]
    print(a.pivot(index="model",columns="scope",values="F1").to_string())
    print("\n==== +6h point vs conformal-upper (POD / FAR), threshold=pct75_city ====")
    b=res[(res.horizon=="y_t6")&(res.threshold=="pct75_city")&(res.scope!="ALL")&
          (res.model.isin(["LightGBM(monotone)","LightGBM(monotone)+adaptiveConf"]))]
    print(b[["scope","model","alarm","POD","FAR","F1","n_events"]].to_string(index=False))
    # figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        a2=a[a.scope!="ALL"].pivot(index="model",columns="scope",values="F1")
        fig,ax=plt.subplots(figsize=(7,4)); a2.plot(kind="bar",ax=ax); ax.set_ylabel("F1 (episode detection)")
        ax.set_title("+6h exceedance F1 (threshold = city 75th pct)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT,"exceedance_F1_p75.png"),dpi=200); plt.close(fig)
        # point vs conformal POD/FAR for both cities
        sub=res[(res.horizon=="y_t6")&(res.threshold=="pct75_city")&(res.scope!="ALL")&
                (res.model.isin(["LightGBM(monotone)","LightGBM(monotone)+adaptiveConf"]))]
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,metric in zip(axs,["POD","FAR"]):
            piv=sub.pivot_table(index="scope",columns="alarm",values=metric)
            piv.plot(kind="bar",ax=ax); ax.set_title(metric); ax.set_ylabel(metric)
        fig.suptitle("+6h alarms: point forecast vs adaptive-conformal upper bound"); fig.tight_layout()
        fig.savefig(os.path.join(OUT,"exceedance_pointVSconformal.png"),dpi=200); plt.close(fig)
        print("[fig] exceedance_F1_p75.png, exceedance_pointVSconformal.png")
    except Exception as e: print("[fig] skipped:",e)
    print("\nSAVED:",OUT,"| files:",[f for f in os.listdir(OUT) if "exceedance" in f])
    print(f"[done] {round(time.time()-t0,1)}s")

if __name__=="__main__":
    main()
