#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  CLEAN model_table builder  (replaces the buggy feature_table -> model_table cell)
#  Fixes: (1) 2024-2025 NO2 dilution (aggregate over PRESENT stations only, no
#         zero/NaN fill); (2) consistent unit conversion mg/m3 -> ug/m3 for ALL
#         years; (3) CAMS coalesce Almaty(short names from .nc) + Astana(long
#         names from CSV) so CAMS is present in 2024-2025.
#  Reads the SAME raw sources as build_feature_table.py and writes
#  model_table_NO2_full.csv with the exact 59-column schema.
#
#  Run in Colab AFTER the three downloaders + the new CAMS for 2024-2025.
#  Requires: pandas, numpy, xarray, netCDF4.
# =============================================================================
import glob, os, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings("ignore")

# ---- paths (edit if needed) ----
USE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/Air Quality Cloud 2805"
GND   = "kazhydromet_hourly_canonical.csv"
NO2_UNIT_FACTOR = 1000.0          # raw mg/m3 -> ug/m3 (raw ~0.05 -> ~50 ug/m3)
HEATING_MONTHS  = [10, 11, 12, 1, 2, 3]

def base(p):
    if USE_DRIVE and os.path.exists(os.path.join(DRIVE_DIR, p)):
        return os.path.join(DRIVE_DIR, p)
    return p

if USE_DRIVE:
    try: from google.colab import drive; drive.mount("/content/drive"); os.chdir(DRIVE_DIR)
    except Exception as e: print("[drive] local mode:", e)

# =============================================================================
# 1) GROUND -> city-level hourly NO2 (PRESENT stations only; correct units)
# =============================================================================
g = pd.read_csv(base(GND), parse_dates=["hour"]).rename(columns={"hour": "timestamp"})
g["NO2"] = pd.to_numeric(g["NO2"], errors="coerce")
# city-hour aggregation: mean over stations that actually reported (skipna),
# plus the count of contributing stations. NO zero-fill -> no dilution.
grp = g.groupby(["city", "timestamp"])
city = grp["NO2"].mean().reset_index().rename(columns={"NO2": "no2_mg_m3"})
city["n_stations"] = grp["NO2"].apply(lambda s: s.notna().sum()).values
city = city[city["n_stations"] > 0].copy()                # drop empty hours
city["no2"] = city["no2_mg_m3"] * NO2_UNIT_FACTOR         # ug/m3, ALL years same
print(f"[ground] city-hours={len(city)}")
for c in city["city"].unique():
    s = city[city.city == c]; s = s.assign(y=s.timestamp.dt.year)
    print(f"  {c} NO2(ug/m3) median by year:",
          {int(y): round(v, 1) for y, v in s.groupby("y")["no2"].median().items()})

# =============================================================================
# 2) ERA5 (city, timestamp)
# =============================================================================
def era5_city(cty):
    files = sorted(glob.glob(base(f"era5/era5_{cty}_*.csv")))
    if files:
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        ncs = sorted(glob.glob(base(f"era5/era5_{cty}_*.nc")))
        if not ncs: return None
        frames = []
        for f in ncs:
            ds = xr.open_dataset(f)
            frames.append(ds.mean(dim=[x for x in ds.dims if x in ("latitude","longitude")]).to_dataframe().reset_index())
            ds.close()
        d = pd.concat(frames, ignore_index=True)
    tcol = next((x for x in ("valid_time","time","date","datetime") if x in d.columns), d.columns[0])
    d = d.rename(columns={tcol: "timestamp"}); d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    ren = {"2m_temperature":"t2m","2m_dewpoint_temperature":"d2m","10m_u_component_of_wind":"u10",
           "10m_v_component_of_wind":"v10","boundary_layer_height":"blh","surface_pressure":"sp",
           "total_precipitation":"tp","surface_solar_radiation_downwards":"ssrd","total_cloud_cover":"tcc"}
    d = d.rename(columns={k:v for k,v in ren.items() if k in d.columns})
    if {"u10","v10"}.issubset(d.columns):
        d["wind_speed"] = np.hypot(d["u10"], d["v10"])
        d["wind_dir"]   = np.degrees(np.arctan2(-d["u10"], -d["v10"])) % 360
    keep = ["timestamp"] + [c for c in ["u10","v10","d2m","t2m","blh","sp","tp","ssrd","tcc",
            "wind_speed","wind_dir"] if c in d.columns]
    d = d[keep].copy(); d["city"] = cty
    d.columns = [("era5_"+c if c not in ("timestamp","city") else c) for c in d.columns]
    return d
era5 = pd.concat([x for x in (era5_city("Almaty"), era5_city("Astana")) if x is not None], ignore_index=True)
era5 = era5.groupby(["city","timestamp"], as_index=False).mean(numeric_only=True)

# =============================================================================
# 3) CAMS coalesce: Almaty(.nc short names) + Astana(CSV long names) -> unified
# =============================================================================
NC2LONG = {"no2":"nitrogen_dioxide","go3":"ozone","o3":"ozone","so2":"sulphur_dioxide","co":"carbon_monoxide"}
def cams_astana():
    fs = sorted(glob.glob(base("cams_astana_pollution_*.csv")))
    if not fs: return None
    d = pd.concat([pd.read_csv(f, parse_dates=["datetime"]) for f in fs], ignore_index=True)
    d = d.rename(columns={"datetime":"timestamp"}); d["city"] = "Astana"; return d
def cams_almaty():
    fs = sorted(glob.glob(base("cams_almaty/*.nc")))
    if not fs: return None
    frames = []
    for f in fs:
        ds = xr.open_dataset(f)
        dd = ds.mean(dim=[x for x in ds.dims if x in ("latitude","longitude")]).to_dataframe().reset_index()
        ds.close(); frames.append(dd)
    d = pd.concat(frames, ignore_index=True)
    tcol = "valid_time" if "valid_time" in d.columns else "time"
    d = d.rename(columns={tcol:"timestamp"})
    d = d.rename(columns={k:v for k,v in NC2LONG.items() if k in d.columns})  # short->long
    d["city"] = "Almaty"; return d

cam = pd.concat([x for x in (cams_astana(), cams_almaty()) if x is not None], ignore_index=True)
# keep only the 4 gases we use, by their LONG names; coalesce + resample to 1h per city
GAS = {"nitrogen_dioxide":"cams_no2_surf","ozone":"cams_o3","sulphur_dioxide":"cams_so2","carbon_monoxide":"cams_co"}
cam = cam[["city","timestamp"] + [c for c in GAS if c in cam.columns]].rename(columns=GAS)
out = []
for c, gg in cam.groupby("city"):
    gg = gg.set_index("timestamp").sort_index()
    gg = gg[[x for x in GAS.values() if x in gg.columns]].resample("1h").interpolate("time")
    gg["city"] = c; out.append(gg.reset_index())
cam = pd.concat(out, ignore_index=True)
print("[cams] coverage by city x year (%):")
cc = cam.assign(y=cam.timestamp.dt.year)
print(cc.groupby(["city","y"]).apply(lambda d: round(100*d["cams_no2_surf"].notna().mean(),0)).unstack(fill_value=0))

# =============================================================================
# 4) S5P (city, day) -> hourly broadcast
# =============================================================================
def s5p():
    p = base("s5p_no2_daily.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p, parse_dates=["date"]); d["day"] = d["date"].dt.normalize()
    return d[["city","day","s5p_no2_trop"]].rename(columns={"s5p_no2_trop":"s5p_no2_trop_column"})
s5 = s5p()

# =============================================================================
# 5) MERGE everything onto (city, timestamp)
# =============================================================================
df = city.merge(era5, on=["city","timestamp"], how="left")
df = df.merge(cam,  on=["city","timestamp"], how="left")
if s5 is not None:
    df["day"] = df["timestamp"].dt.normalize()
    df = df.merge(s5, on=["city","day"], how="left").drop(columns=["day"])
df = df.sort_values(["city","timestamp"]).reset_index(drop=True)

# =============================================================================
# 6) CALENDAR + DERIVED + LAGS/ROLLING + TARGETS  (per city, leakage-safe)
# =============================================================================
ts = df["timestamp"]
df["hour"]=ts.dt.hour; df["dayofweek"]=ts.dt.dayofweek; df["month"]=ts.dt.month
df["dayofyear"]=ts.dt.dayofyear; df["is_weekend"]=(df["dayofweek"]>=5).astype(int)
df["heating_season"]=df["month"].isin(HEATING_MONTHS).astype(int)
df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
df["doy_sin"]=np.sin(2*np.pi*df["dayofyear"]/365); df["doy_cos"]=np.cos(2*np.pi*df["dayofyear"]/365)

# physically-motivated derived features (documented definitions)
df["ventilation_coefficient"]  = df.get("era5_blh") * df.get("era5_wind_speed")           # BLH * wind
df["stagnation_indicator"]     = (df.get("era5_wind_speed") < 2.0).astype(float)          # low-wind flag
df["atmospheric_dryness"]      = df.get("era5_t2m") - df.get("era5_d2m")                  # dewpoint depression
df["winter_inversion_indicator"]= ((df["heating_season"]==1) & (df.get("era5_blh") < 300)).astype(float)
df["cams_bias_correction"]     = df.get("cams_no2_surf") - df["no2"]                       # CAMS - obs (origin t)

g2 = df.groupby("city", group_keys=False)
for L in [1,2,3,6,12,24,48,168]:
    df[f"no2_lag_{L}"] = g2["no2"].shift(L)
for W in [3,6,12,24]:
    df[f"no2_rollmean_{W}"] = g2["no2"].apply(lambda s: s.shift(1).rolling(W, min_periods=1).mean())
for W in [3,6,24]:
    df[f"no2_rollstd_{W}"]  = g2["no2"].apply(lambda s: s.shift(1).rolling(W, min_periods=1).std())
df["no2_rollmin_24"]=g2["no2"].apply(lambda s: s.shift(1).rolling(24, min_periods=1).min())
df["no2_rollmax_24"]=g2["no2"].apply(lambda s: s.shift(1).rolling(24, min_periods=1).max())
df["no2_diff_1"] =g2["no2"].diff(1)
df["no2_diff_24"]=g2["no2"].diff(24)
df["persistence"]=df["no2"]                                  # value at origin t
df["y_t1"] =g2["no2"].shift(-1)
df["y_t6"] =g2["no2"].shift(-6)
df["y_t24"]=g2["no2"].shift(-24)

# =============================================================================
# 7) order columns to the exact 59-col schema and save
# =============================================================================
SCHEMA = ["city","timestamp","no2","no2_mg_m3","n_stations",
 "no2_lag_1","no2_lag_2","no2_lag_3","no2_lag_6","no2_lag_12","no2_lag_24","no2_lag_48","no2_lag_168",
 "no2_rollmean_3","no2_rollmean_6","no2_rollmean_12","no2_rollmean_24","no2_rollstd_3","no2_rollstd_6",
 "no2_rollstd_24","no2_rollmin_24","no2_rollmax_24","no2_diff_1","no2_diff_24","hour","dayofweek","month",
 "dayofyear","is_weekend","heating_season","hour_sin","hour_cos","doy_sin","doy_cos","era5_u10","era5_v10",
 "era5_d2m","era5_t2m","era5_blh","era5_sp","era5_tp","era5_ssrd","era5_tcc","cams_no2_surf","cams_o3",
 "cams_so2","cams_co","era5_wind_speed","era5_wind_dir","ventilation_coefficient","stagnation_indicator",
 "atmospheric_dryness","winter_inversion_indicator","cams_bias_correction","s5p_no2_trop_column",
 "persistence","y_t1","y_t6","y_t24"]
for c in SCHEMA:
    if c not in df.columns: df[c] = np.nan
df = df[SCHEMA]
df.to_csv(base("model_table_NO2_full.csv"), index=False)
print(f"\n[done] model_table_NO2_full.csv  rows={len(df)} cols={df.shape[1]}")
dd = df.assign(y=pd.to_datetime(df.timestamp).dt.year)
print("NO2 median by city x year (should be STABLE ~30-60, no collapse):")
print(dd.groupby(["city","y"])["no2"].median().round(1).unstack(fill_value=np.nan))
print("CAMS coverage by city x year (should be >0 in 2024-2025):")
print(dd.groupby(["city","y"]).apply(lambda d: round(100*d["cams_no2_surf"].notna().mean(),0)).unstack(fill_value=0))
