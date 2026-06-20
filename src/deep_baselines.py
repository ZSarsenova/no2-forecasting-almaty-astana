#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  DEEP BASELINES (v4): LSTM / GRU / CNN-LSTM for +1/+6/+24 h, per city.
#  READS  model_table_NO2_full.csv FROM Google Drive:
#         /content/drive/MyDrive/air_quality_claude_28_05/
#  SAVES  the result INTO Google Drive subfolder:
#         /content/drive/MyDrive/air_quality_claude_28_05/pcmb_conf_outputs/
#  If deep_baselines.csv already exists there, a timestamped name is used.
#  Fixes GRU/CNN-LSTM non-convergence (standardized target, more epochs +
#  patience, BatchNorm CNN-LSTM, gradient clipping, LR scheduling).
#
#  In a notebook: mount Drive once, then run this file:
#     from google.colab import drive; drive.mount('/content/drive')
#     import os; os.chdir('/content'); exec(open('/content/deep_baselines.py').read())
#  (GPU runtime recommended.)
# =============================================================================
import os, time, datetime, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
SEED = 42; np.random.seed(SEED)

# ---- fixed Google Drive locations ----
DRIVE_DIR  = "/content/drive/MyDrive/air_quality_claude_28_05"
OUT_DIR    = os.path.join(DRIVE_DIR, "pcmb_conf_outputs")   # <- results saved HERE (on Drive)
CSV_PATH   = os.path.join(DRIVE_DIR, "model_table_NO2_full.csv")
RESULT_CSV = "deep_baselines.csv"

HORIZONS = ["y_t6", "y_t1", "y_t24"]
TRAIN_END, TEST_YEAR = 2023, 2025
SEQ_LEN, EPOCHS, PATIENCE, BATCH, MAX_TRAIN = 12, 40, 8, 256, 60000
AUTOCORR = ["no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12","no2_lag_24","no2_lag_48","no2_lag_168"]
SEQ_FEATS = AUTOCORR + ["era5_blh","era5_wind_speed","era5_t2m","hour_sin","hour_cos"]

def ensure_drive():
    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e:
            print("[drive] mount failed:", e)
    if not os.path.isdir(DRIVE_DIR):
        raise FileNotFoundError(f"Drive folder not found: {DRIVE_DIR}  (mount Drive first)")
    os.makedirs(OUT_DIR, exist_ok=True)   # create pcmb_conf_outputs on Drive if missing
    print("[drive] data folder :", DRIVE_DIR)
    print("[drive] output folder:", OUT_DIR)

def r2(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); ss=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/ss) if ss>0 else np.nan
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def mae(y,p):  return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))

def make_seq(frame, keep, horizon, L):
    Xs, ys, cs = [], [], []
    for city, g in frame.sort_values("timestamp").groupby("city"):
        M = g[keep].fillna(g[keep].median()).to_numpy("float32")
        y = g[horizon].to_numpy("float32")
        for i in range(L, len(g)):
            Xs.append(M[i-L:i]); ys.append(y[i]); cs.append(city)
    if not Xs: return None, None, None
    return np.asarray(Xs,"float32"), np.asarray(ys,"float32"), np.array(cs)

def save_result(res):
    target = os.path.join(OUT_DIR, RESULT_CSV)
    if os.path.exists(target):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(OUT_DIR, f"deep_baselines_{stamp}.csv")
        print(f"[save] {RESULT_CSV} already exists -> new name: {os.path.basename(target)}")
    res.to_csv(target, index=False)
    print(f"\n[save] RESULT SAVED TO GOOGLE DRIVE:\n       {target}")
    print("[save] folder now contains:", [f for f in os.listdir(OUT_DIR) if f.startswith('deep_baselines')])

def main():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    tf.random.set_seed(SEED)
    print("TensorFlow", tf.__version__, "| GPU:", bool(tf.config.list_physical_devices("GPU")))
    ensure_drive()
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]); df["year"]=df["timestamp"].dt.year
    a25 = df[(df.city=="Almaty") & (df.year==2025)]["no2"].median()
    print(f"[data] read {CSV_PATH}\n[data] rows={len(df)} | Almaty 2025 NO2 median={a25:.1f} (clean ~30-40, buggy ~8.7)")
    keep = [c for c in SEQ_FEATS if c in df.columns]; print("seq features:", keep)

    all_rows = []
    for H in HORIZONS:
        d = df.dropna(subset=[H]); tr = d[d.year <= TRAIN_END]; te = d[d.year == TEST_YEAR]
        Xtr, ytr, ctr = make_seq(tr, keep, H, SEQ_LEN); Xte, yte, cte = make_seq(te, keep, H, SEQ_LEN)
        if Xtr is None or Xte is None: print(f"[{H}] not enough data, skip"); continue
        if len(Xtr) > MAX_TRAIN:
            idx = np.random.RandomState(SEED).choice(len(Xtr), MAX_TRAIN, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
        mu, sd = Xtr.mean((0,1)), Xtr.std((0,1)) + 1e-6; Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
        ym, ys_ = ytr.mean(), ytr.std() + 1e-6; ytr_z = (ytr - ym)/ys_
        nf = Xtr.shape[-1]; print(f"\n### HORIZON {H}: train_seq={len(Xtr)} test_seq={len(Xte)} feat={nf}")
        def build(kind):
            inp = layers.Input((SEQ_LEN, nf))
            if   kind == "LSTM": x = layers.LSTM(64)(inp)
            elif kind == "GRU":  x = layers.GRU(64)(inp)
            else:
                x = layers.Conv1D(32, 3, padding="same")(inp); x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x); x = layers.LSTM(64)(x)
            x = layers.Dense(32, activation="relu")(x); out = layers.Dense(1)(x)
            m = models.Model(inp, out); m.compile(optimizer=optimizers.Adam(1e-3, clipnorm=1.0), loss="mse"); return m
        es = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")
        rl = callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss")
        for kind in ["LSTM","GRU","CNN-LSTM"]:
            t0 = time.time(); m = build(kind)
            m.fit(Xtr, ytr_z, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, callbacks=[es, rl], verbose=0)
            p = m.predict(Xte, verbose=0).ravel() * ys_ + ym; dt = round(time.time()-t0, 1)
            for sc, mask in [("ALL", np.ones(len(yte),bool))] + [(c, cte==c) for c in np.unique(cte)]:
                all_rows.append(dict(horizon=H, model=kind, scope=str(sc),
                    RMSE=round(rmse(yte[mask],p[mask]),2), MAE=round(mae(yte[mask],p[mask]),2), R2=round(r2(yte[mask],p[mask]),3)))
            print(f"  [{kind}] {dt}s  R2(ALL)={r2(yte,p):.3f}")

    res = pd.DataFrame(all_rows)
    print("\n================ DEEP BASELINES v4 (paste this to me) ================")
    for H in HORIZONS:
        sub = res[res.horizon==H]
        if len(sub):
            print(f"\n--- {H} ---"); print(sub[sub.scope=="ALL"].to_string(index=False)); print(sub[sub.scope!="ALL"].to_string(index=False))
    save_result(res)

if __name__ == "__main__":
    main()
