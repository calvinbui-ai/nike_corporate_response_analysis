"""
Nike ESG Analysis - Master Pipeline Runner
==========================================

This script orchestrates the complete analysis pipeline:
1. PDF Extraction
2. Target Analysis
3. Language Analysis
4. Visualization Generation
5. Master Dataset Compilation

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import sys
from pathlib import Path
import subprocess
from datetime import datetime

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_script(script_name, description):
    """Run a Python script and report status"""

    print_banner(f"STEP: {description}")

    script_path = scripts_dir / script_name

    if not script_path.exists():
        print(f"  ✗ ERROR: Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per script
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode == 0:
            print(f"\n  ✓ {description} completed successfully")
            return True
        else:
            print(f"\n  ✗ {description} failed with error:")
            if result.stderr:
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"\n  ✗ {description} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n  ✗ {description} failed with exception: {e}")
        return False


def main():
    """Run the complete analysis pipeline"""

    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("  NIKE ESG POLICY RESPONSE ANALYSIS")
    print("  Complete Analysis Pipeline")
    print("  " + "=" * 76)
    print(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Define pipeline steps
    pipeline = [
        ('01_pdf_extraction.py', 'PDF Text Extraction with Page Tracking'),
        ('03_target_analysis.py', 'Target Setting & Goals Analysis'),
        ('04_language_analysis.py', 'Language & Tone Analysis'),
        ('05_visualization.py', 'Data Visualization Generation'),
        ('06_create_master_dataset.py', 'Master Dataset Compilation'),
    ]

    # Track results
    results = {}

    # Execute pipeline
    for script_name, description in pipeline:
        success = run_script(script_name, description)
        results[description] = success

        if not success:
            print(f"\n⚠ WARNING: {description} failed but continuing with pipeline...")

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_banner("PIPELINE EXECUTION SUMMARY")

    print("Script Execution Results:")
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {step}")

    successful = sum(1 for s in results.values() if s)
    total = len(results)

    print(f"\nOverall: {successful}/{total} steps completed successfully")
    print(f"Total execution time: {duration}")

    print("\n" + "=" * 80)
    print("  OUTPUT LOCATIONS:")
    print("=" * 80)

    base_dir = Path(__file__).parent.parent

    output_locations = {
        'Raw Extractions': base_dir / 'data' / 'raw_extractions',
        'Processed Data': base_dir / 'data' / 'processed',
        'Visualizations': base_dir / 'outputs' / 'visualizations',
        'Master Dataset': base_dir / 'outputs' / 'master_dataset.csv',
        'Citations Index': base_dir / 'outputs' / 'citations_index.xlsx'
    }

    for name, path in output_locations.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}:")
        print(f"     {path}")

    if successful == total:
        print("\n🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    else:
        print("\n⚠ ANALYSIS PIPELINE COMPLETED WITH WARNINGS ⚠")

    print("\nNext Steps:")
    print("  1. Review extraction_summary.csv for quality metrics")
    print("  2. Examine processed datasets in data/processed/")
    print("  3. Review visualizations in outputs/visualizations/")
    print("  4. Validate findings against original PDFs using page references")

    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
