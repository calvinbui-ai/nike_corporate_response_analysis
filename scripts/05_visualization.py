import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/visualizations", exist_ok=True)

df = pd.read_csv("outputs/target_policy_hits.csv")
df["year"] = df["year"].astype(str)

# 1) Theme frequency
by_theme = df.groupby("theme")["file"].count().reset_index(name="hits")
plt.figure()
plt.bar(by_theme["theme"], by_theme["hits"])
plt.title("Policy/Target Hits by Theme")
plt.xlabel("Theme")
plt.ylabel("Hits")
plt.tight_layout()
plt.savefig("outputs/visualizations/10_hits_by_theme.png")

# 2) Year × theme trend
by_year_theme = df.groupby(["year","theme"])["file"].count().reset_index(name="hits")
for t in sorted(by_year_theme["theme"].unique()):
    sub = by_year_theme[by_year_theme["theme"] == t]
    plt.figure()
    plt.plot(sub["year"], sub["hits"], marker="o")
    plt.title(f"Hits by Year — {t}")
    plt.xlabel("Year")
    plt.ylabel("Hits")
    plt.tight_layout()
    plt.savefig(f"outputs/visualizations/11_hits_by_year_{t}.png")

print("✅ Saved charts in outputs/visualizations/")
