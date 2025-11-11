import re, pandas as pd, os

IN_CSV  = "outputs/target_policy_hits.csv"
OUT_XLS = "outputs/citations_index.xlsx"

def sanitize(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Remove Excel-illegal control chars (0x00–0x1F except tab/newline/carriage)
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", s)
    # Collapse weird whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    if not os.path.exists(IN_CSV):
        raise SystemExit(f"Missing {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    # Keep the core columns in a friendly order if they exist
    cols = ["file","year","page","theme","term","quote","path"]
    out_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[out_cols]

    # Sanitize all string cells
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(sanitize)

    # Write to Excel
    df.to_excel(OUT_XLS, index=False)
    print(f"✅ Wrote {OUT_XLS} ({len(df)} rows)")

if __name__ == "__main__":
    main()
