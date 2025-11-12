"""
Nike ESG Analysis - PDF Text Extraction Script
==============================================

This script extracts text from all Nike annual reports (2020-2024) while preserving
page numbers for citation and quality assurance purposes.

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import os
import json
import re
from pathlib import Path
import PyPDF2
import pandas as pd
from datetime import datetime


class PDFExtractor:
    """Extract text from PDFs with page-level tracking"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / 'data' / 'raw_extractions'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_with_pypdf2(self, pdf_path):
        """
        Primary extraction method using PyPDF2
        Returns: List of dictionaries with page-level content
        """
        pages_data = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    pages_data.append({
                        'page_number': page_num + 1,  # 1-indexed for human readability
                        'text': text,
                        'char_count': len(text),
                        'extraction_method': 'PyPDF2'
                    })

        except Exception as e:
            print(f"PyPDF2 error on {pdf_path.name}: {e}")

        return pages_data

    def extract_pdf(self, pdf_path):
        """
        Extract text from a single PDF using PyPDF2

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extraction metadata and page-level content
        """
        pdf_path = Path(pdf_path)

        print(f"\nExtracting: {pdf_path.name}")
        print(f"  Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Extraction with PyPDF2
        pypdf2_data = self.extract_with_pypdf2(pdf_path)

        pypdf2_total = sum(p['char_count'] for p in pypdf2_data)

        result = {
            'document_name': pdf_path.name,
            'file_path': str(pdf_path),
            'extraction_date': datetime.now().isoformat(),
            'total_pages': len(pypdf2_data),
            'file_size_mb': pdf_path.stat().st_size / 1024 / 1024,
            'total_chars': pypdf2_total,
            'pages': pypdf2_data
        }

        print(f"  Extracted {result['total_pages']} pages ({pypdf2_total:,} characters)")

        return result

    def save_extraction(self, extraction_data, output_format='json'):
        """Save extraction results to file"""

        doc_name = extraction_data['document_name'].replace('.pdf', '')

        # Save as JSON for full structured data
        if output_format in ['json', 'both']:
            json_path = self.output_dir / f"{doc_name}_extraction.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved JSON: {json_path.name}")

        # Save as TXT for easy reading with page markers
        if output_format in ['txt', 'both']:
            txt_path = self.output_dir / f"{doc_name}_extraction.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"DOCUMENT: {extraction_data['document_name']}\n")
                f.write(f"EXTRACTED: {extraction_data['extraction_date']}\n")
                f.write(f"TOTAL PAGES: {extraction_data['total_pages']}\n")
                f.write("=" * 80 + "\n\n")

                for page_data in extraction_data['pages']:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"PAGE {page_data['page_number']}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(page_data['text'])
                    f.write("\n\n")

            print(f"  Saved TXT: {txt_path.name}")

    def extract_all_reports(self, reports_dir):
        """
        Extract all PDF reports in the directory

        Args:
            reports_dir: Path to directory containing PDF files
        """
        reports_dir = Path(reports_dir)
        pdf_files = sorted(reports_dir.glob('**/*.pdf'))

        # Exclude macOS metadata files
        pdf_files = [f for f in pdf_files if '__MACOSX' not in str(f)]

        print(f"\nFound {len(pdf_files)} PDF files to process")
        print("=" * 80)

        extraction_summary = []

        for pdf_file in pdf_files:
            try:
                # Extract the PDF
                extraction_data = self.extract_pdf(pdf_file)

                # Save in both formats
                self.save_extraction(extraction_data, output_format='both')

                # Track summary
                extraction_summary.append({
                    'document': extraction_data['document_name'],
                    'year': self.extract_year_from_filename(pdf_file.name),
                    'pages': extraction_data['total_pages'],
                    'file_size_mb': extraction_data['file_size_mb'],
                    'total_chars': extraction_data['total_chars'],
                    'status': 'Success'
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                extraction_summary.append({
                    'document': pdf_file.name,
                    'year': self.extract_year_from_filename(pdf_file.name),
                    'pages': 0,
                    'file_size_mb': 0,
                    'total_chars': 0,
                    'status': f'Failed: {str(e)[:100]}'
                })

        # Save extraction summary
        summary_df = pd.DataFrame(extraction_summary)
        summary_path = self.output_dir / 'extraction_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\nProcessed: {len(extraction_summary)} documents")
        print(f"Successful: {len([s for s in extraction_summary if s['status'] == 'Success'])}")
        print(f"Failed: {len([s for s in extraction_summary if 'Failed' in str(s['status'])])}")
        print(f"\nSummary saved to: {summary_path}")

        return summary_df

    @staticmethod
    def extract_year_from_filename(filename):
        """Extract year from filename (e.g., 'Nike_2020_Impact_Report.pdf' -> 2020)"""
        match = re.search(r'20(2[0-4])', filename)
        return int(match.group(0)) if match else None


def main():
    """Main execution function"""

    # Set up paths
    base_dir = Path(__file__).parent.parent
    reports_dir = base_dir / 'Nike_disclosures'

    # Initialize extractor
    extractor = PDFExtractor(base_dir)

    # Extract all reports from subdirectories
    print("Extracting Nike ESG reports...")
    summary = extractor.extract_all_reports(reports_dir)

    print("\n\n✓ PDF EXTRACTION PIPELINE COMPLETE")
    print(f"✓ All extractions saved to: {extractor.output_dir}")
    print(f"✓ Review extraction_summary.csv for quality metrics")


if __name__ == "__main__":
    main()
