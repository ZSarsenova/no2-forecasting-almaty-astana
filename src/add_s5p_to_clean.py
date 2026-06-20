#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add s5p_co / s5p_so2 / s5p_aer_ai from the (buggy) plusS5P file onto the CLEAN
# model_table. Only the 3 S5P columns are taken (they come from satellite and are
# NOT affected by the ground-aggregation bug); NO2/CAMS stay from the CLEAN table.
# Run:  import os; os.chdir('/content'); exec(open('/content/add_s5p_to_clean.py').read())
import os, pandas as pd
CLEAN = "model_table_NO2_full.csv"                 # the FIXED table (Almaty 2025 ~35)
PLUS  = "model_table_NO2_full_plusS5P.csv"         # the old file with extra S5P cols
EXTRA = ["s5p_co", "s5p_so2", "s5p_aer_ai"]

clean = pd.read_csv(CLEAN, parse_dates=["timestamp"])
plus  = pd.read_csv(PLUS,  parse_dates=["timestamp"])
have  = [c for c in EXTRA if c in plus.columns]
print("merging columns:", have)

# sanity: confirm CLEAN is really the fixed one
alm25 = clean[(clean.city=="Almaty") & (clean.timestamp.dt.year==2025)]["no2"].median()
print(f"CLEAN Almaty 2025 NO2 median = {alm25:.1f}  (must be ~30-40, NOT 8.7)")
assert alm25 > 20, "CLEAN table looks buggy (Almaty 2025 too low) - use the FIXED table!"

m = clean.merge(plus[["city","timestamp"]+have], on=["city","timestamp"], how="left")
m.to_csv(CLEAN, index=False)                        # overwrite clean table, now with extra S5P
print(f"done: {CLEAN} now has {m.shape[1]} cols, rows={len(m)}")
d = m.assign(y=m.timestamp.dt.year)
for c in have:
    print(c, "coverage % by year:",
          {int(y): round(100*g[c].notna().mean()) for y,g in d.groupby("y")})
print("Almaty 2025 NO2 median still:", round(d[(d.city=='Almaty')&(d.y==2025)]['no2'].median(),1))
