import os, re, argparse
import pandas as pd
from PyPDF2 import PdfReader
import yaml

def load_lexicon(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def iter_pdfs(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for r, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".pdf"):
                    yield os.path.join(r, name)

def page_excerpt(text, hit_start, window_chars=320, max_words=40):
    text = text or ""
    start = max(0, hit_start - window_chars)
    end = min(len(text), hit_start + window_chars)
    snippet = " ".join(text[start:end].split())
    words = snippet.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--include-sec", default="true")
    ap.add_argument("--lexicon", default="data/policy_lexicon.yaml")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    roots = years[:]
    if args.include_sec.lower() == "true":
        roots.append("SEC_filings")

    lex = load_lexicon(args.lexicon)
    compiled = {theme: [re.compile(pat, re.IGNORECASE) for pat in pats]
                for theme, pats in lex.items()}

    hits = []
    for pdf in iter_pdfs(roots):
        year_guess = next((y for y in years if f"/{y}/" in pdf or pdf.startswith(y + "/")), "")
        try:
            reader = PdfReader(pdf)
        except Exception as e:
            print(f"[WARN] Cannot open {pdf}: {e}")
            continue

        for pno, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            for theme, patterns in compiled.items():
                for pat in patterns:
                    for m in pat.finditer(text):
                        hits.append({
                            "file": os.path.basename(pdf),
                            "path": pdf,
                            "year": year_guess,
                            "page": pno,
                            "theme": theme,
                            "term": pat.pattern,
                            "quote": page_excerpt(text, m.start())
                        })

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(hits)
    df.to_csv("outputs/target_policy_hits.csv", index=False)

    # citations index (Excel)
    cols = ["file","year","page","theme","term","quote","path"]
    try:
        df[cols].to_excel("outputs/citations_index.xlsx", index=False)
    except Exception as e:
        print(f"[WARN] Could not write Excel: {e}")

    print(f"✅ Wrote outputs/target_policy_hits.csv ({len(df)} rows)")
if __name__ == "__main__":
    main()