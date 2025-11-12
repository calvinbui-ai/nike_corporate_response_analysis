"""
Nike ESG Analysis - Master Dataset Generator
============================================

This script consolidates all analysis outputs into a single master dataset
with complete page references for quality checking and client presentation.

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class MasterDatasetGenerator:
    """Generate consolidated master dataset"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data' / 'processed'
        self.output_dir = self.base_dir / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all_datasets(self):
        """Load all processed datasets"""

        print("Loading processed datasets...")

        datasets = {}

        # Targets data
        targets_file = self.data_dir / 'targets_detailed.csv'
        if targets_file.exists():
            datasets['targets'] = pd.read_csv(targets_file)
            print(f"  ✓ Loaded {len(datasets['targets'])} target records")
        else:
            print(f"  ⚠ Targets dataset not found")

        # Language analysis
        language_file = self.data_dir / 'language_analysis.csv'
        if language_file.exists():
            datasets['language'] = pd.read_csv(language_file)
            print(f"  ✓ Loaded {len(datasets['language'])} language records")
        else:
            print(f"  ⚠ Language dataset not found")

        # Target changes
        changes_file = self.data_dir / 'target_changes.csv'
        if changes_file.exists():
            datasets['changes'] = pd.read_csv(changes_file)
            print(f"  ✓ Loaded {len(datasets['changes'])} change records")
        else:
            print(f"  ⚠ Changes dataset not found (may not exist if no changes detected)")

        return datasets

    def create_master_dataset(self, datasets):
        """Create consolidated master dataset"""

        print("\n" + "=" * 80)
        print("CREATING MASTER DATASET")
        print("=" * 80)

        # Start with targets as the base
        if 'targets' in datasets and len(datasets['targets']) > 0:
            master_df = datasets['targets'].copy()

            # Add source column
            master_df['data_source'] = 'target_analysis'

            # Rename columns for consistency
            master_df['finding'] = master_df['sentence']
            master_df['finding_type'] = 'target'

            # Select key columns
            key_cols = ['year', 'document', 'page', 'category', 'finding',
                       'commitment_strength', 'contains_timeline', 'contains_percentage',
                       'data_source', 'finding_type']

            master_df = master_df[[col for col in key_cols if col in master_df.columns]]

        else:
            # Create empty master dataset
            master_df = pd.DataFrame()

        # Save master dataset
        output_path = self.output_dir / 'master_dataset.csv'
        master_df.to_csv(output_path, index=False)

        print(f"\n✓ Created master dataset with {len(master_df)} records")
        print(f"✓ Saved to: {output_path}")

        return master_df

    def create_citations_index(self, datasets):
        """Create comprehensive citations index"""

        print("\nCreating citations index...")

        citations = []

        # Extract citations from targets
        if 'targets' in datasets:
            for _, row in datasets['targets'].iterrows():
                # Clean text for Excel compatibility
                excerpt = str(row.get('sentence', ''))[:200]
                # Remove illegal characters for Excel
                excerpt = ''.join(char for char in excerpt if ord(char) >= 32 or char in '\t\n\r')
                if len(str(row.get('sentence', ''))) > 200:
                    excerpt += '...'

                citations.append({
                    'document': row.get('document', ''),
                    'year': row.get('year', ''),
                    'page': row.get('page', ''),
                    'category': row.get('category', ''),
                    'finding_type': 'target',
                    'excerpt': excerpt,
                    'full_citation': f"{row.get('document', '')}, Page {row.get('page', '')}"
                })

        citations_df = pd.DataFrame(citations)

        # Sort by year and page
        citations_df = citations_df.sort_values(['year', 'document', 'page'])

        output_path = self.output_dir / 'citations_index.xlsx'

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            citations_df.to_excel(writer, sheet_name='All Citations', index=False)

            # Create separate sheets by category if we have targets
            if 'targets' in datasets:
                for category in citations_df['category'].unique():
                    if pd.notna(category):
                        category_data = citations_df[citations_df['category'] == category]
                        sheet_name = str(category)[:31]  # Excel sheet name limit
                        category_data.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✓ Created citations index with {len(citations_df)} citations")
        print(f"✓ Saved to: {output_path}")

        return citations_df

    def create_summary_statistics(self, datasets):
        """Create summary statistics report"""

        print("\nCreating summary statistics...")

        stats = {
            'analysis_date': datetime.now().isoformat(),
            'total_documents_analyzed': 0,
            'total_pages_analyzed': 0,
            'date_range': '2020-2024',
        }

        # Language stats
        if 'language' in datasets and len(datasets['language']) > 0:
            stats['total_documents_analyzed'] = len(datasets['language'])
            stats['total_pages_analyzed'] = datasets['language']['page_count'].sum() if 'page_count' in datasets['language'].columns else 0
            stats['total_words_analyzed'] = datasets['language']['word_count'].sum() if 'word_count' in datasets['language'].columns else 0

        # Target stats
        if 'targets' in datasets and len(datasets['targets']) > 0:
            stats['total_target_mentions'] = len(datasets['targets'])
            stats['unique_target_categories'] = datasets['targets']['category'].nunique() if 'category' in datasets['targets'].columns else 0
            stats['strong_commitments'] = len(datasets['targets'][datasets['targets']['commitment_strength'] == 'strong']) if 'commitment_strength' in datasets['targets'].columns else 0

        # Changes stats
        if 'changes' in datasets and len(datasets['changes']) > 0:
            stats['total_changes_detected'] = len(datasets['changes'])
            stats['strengthened_targets'] = len(datasets['changes'][datasets['changes']['change_type'] == 'strengthened']) if 'change_type' in datasets['changes'].columns else 0
            stats['weakened_targets'] = len(datasets['changes'][datasets['changes']['change_type'] == 'weakened']) if 'change_type' in datasets['changes'].columns else 0

        # Save stats
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']

        output_path = self.output_dir / 'summary_statistics.csv'
        stats_df.to_csv(output_path)

        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\n✓ Saved to: {output_path}")

        return stats


def main():
    """Main execution"""

    base_dir = Path(__file__).parent.parent
    generator = MasterDatasetGenerator(base_dir)

    # Load all datasets
    datasets = generator.load_all_datasets()

    # Create master dataset
    master_df = generator.create_master_dataset(datasets)

    # Create citations index
    citations_df = generator.create_citations_index(datasets)

    # Create summary statistics
    stats = generator.create_summary_statistics(datasets)

    print("\n✓ MASTER DATASET GENERATION COMPLETE")


if __name__ == "__main__":
    main()
