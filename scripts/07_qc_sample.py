import os, random, pandas as pd

IN = "outputs/target_policy_hits.csv"
OUT = "outputs/qc_sample.csv"
N = 15

df = pd.read_csv(IN)
if df.empty:
    raise SystemExit("No hits found. Run 03_target_analysis.py first.")

sample = df.sample(min(N, len(df)), random_state=42)[
    ["file","year","page","theme","term","quote","path"]
].sort_values(["year","file","page"])

os.makedirs("outputs", exist_ok=True)
sample.to_csv(OUT, index=False)
print(f"✅ Wrote {OUT} ({len(sample)} rows)")
print("Next: open each PDF and confirm the quote appears on the reported page.")
