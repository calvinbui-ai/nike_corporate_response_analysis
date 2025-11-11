import os
import sys
import argparse
import pandas as pd
from PyPDF2 import PdfReader
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def iter_pdf_paths(root_dirs, recursive=True):
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        if recursive:
            for r, _, files in os.walk(root):
                for name in files:
                    if name.lower().endswith(".pdf"):
                        yield os.path.join(r, name)
        else:
            for name in os.listdir(root):
                if name.lower().endswith(".pdf"):
                    yield os.path.join(root, name)

def read_text_quick(path, max_pages=None):
    """Extract text from the first max_pages pages to keep it fast."""
    reader = PdfReader(path)
    pages = reader.pages
    n = len(pages)
    limit = n if max_pages is None else min(max_pages, n)
    chunks = []
    for i in range(limit):
        try:
            chunks.append(pages[i].extract_text() or "")
        except Exception:
            chunks.append("")
    return " ".join(chunks), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025",
                    help="Comma-separated years to scan")
    ap.add_argument("--include-sec", default="true",
                    help="true/false to include SEC_filings folder")
    ap.add_argument("--fast", default="true",
                    help="If true, limit to first N pages for speed")
    ap.add_argument("--pages", type=int, default=15,
                    help="When --fast true, number of pages to scan per PDF")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    roots = years[:]
    if args.include_sec.lower() == "true":
        roots.append("SEC_filings")

    fast = args.fast.lower() == "true"
    max_pages = args.pages if fast else None

    analyzer = SentimentIntensityAnalyzer()
    records = []

    pdfs = list(iter_pdf_paths(roots, recursive=True))
    if not pdfs:
        print("[INFO] No PDFs found in", roots)
        os.makedirs("outputs", exist_ok=True)
        pd.DataFrame(records).to_csv("outputs/master_dataset.csv", index=False)
        print("✅ Done. Wrote outputs/master_dataset.csv (0 rows)")
        return

    print(f"[INFO] Found {len(pdfs)} PDFs. Fast mode: {fast} (pages={max_pages})")
    for idx, p in enumerate(sorted(pdfs), start=1):
        try:
            text, total_pages = read_text_quick(p, max_pages=max_pages)
            s = analyzer.polarity_scores(text or "")
            year_guess = ""
            for y in years:
                if f"/{y}/" in p or p.startswith(y + "/"):
                    year_guess = y
                    break
            records.append({
                "file": os.path.basename(p),
                "path": p,
                "year": year_guess,
                "total_pages": total_pages,
                "scanned_pages": (max_pages if max_pages is not None else total_pages),
                "word_count_scanned": len((text or "").split()),
                "sent_pos": s.get("pos", 0.0),
                "sent_neg": s.get("neg", 0.0),
                "sent_neu": s.get("neu", 0.0),
                "sent_compound": s.get("compound", 0.0),
            })
            if idx % 3 == 0 or idx == len(pdfs):
                print(f"[{idx}/{len(pdfs)}] Processed: {p}")
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user. Writing partial results…", file=sys.stderr)
            break
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}", file=sys.stderr)

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(records)
    out = "outputs/master_dataset.csv"
    df.to_csv(out, index=False)
    print(f"✅ Done. Wrote {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
