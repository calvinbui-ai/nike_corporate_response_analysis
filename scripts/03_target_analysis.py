"""
Nike ESG Analysis - Target Setting & Goals Analysis
===================================================

This script analyzes Nike's quantitative and qualitative targets across years,
identifying changes, strengthening, weakening, or removal of commitments.

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


class TargetAnalyzer:
    """Analyze target setting and goals evolution"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.extractions_dir = self.base_dir / 'data' / 'raw_extractions'
        self.output_dir = self.base_dir / 'data' / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Keywords and patterns for target identification
        self.target_keywords = {
            'climate': [
                'net zero', 'carbon neutral', 'emissions reduction', 'greenhouse gas',
                'GHG', 'climate target', 'decarbonization', 'scope 1', 'scope 2', 'scope 3',
                'climate positive', 'science-based target', 'SBT', 'paris agreement',
                '1.5°C', '2°C', 'carbon footprint', 'renewable energy'
            ],
            'circular': [
                'circular', 'recycled material', 'recycling', 'waste reduction',
                'reuse', 'repair', 'refurbished', 'circularity',
                'closed loop', 'textile-to-textile', 'polyester', 'sustainable materials',
                'Nike Grind', 'Move to Zero'
            ],
            'water': [
                'water reduction', 'water footprint', 'water stewardship', 'water consumption',
                'water efficiency', 'wastewater', 'clean water', 'water recycling'
            ],
            'chemicals': [
                'hazardous chemicals', 'chemical management', 'ZDHC', 'restricted substances',
                'chemical elimination', 'safer chemistry'
            ],
            'social': [
                'living wage', 'fair wage', 'worker rights', 'freedom of association',
                'collective bargaining', 'health and safety', 'diversity', 'inclusion',
                'gender equality', 'modern slavery', 'forced labor', 'child labor',
                'human rights'
            ],
            'transparency': [
                'supplier disclosure', 'supply chain transparency', 'traceability',
                'supplier list', 'tier 1', 'tier 2', 'manufacturing disclosure',
                'transparency report'
            ],
            'product': [
                'sustainable product', 'eco-friendly', 'low-carbon', 'sustainable innovation',
                'sustainable design', 'product sustainability'
            ]
        }

        # Patterns for extracting numeric targets
        self.numeric_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # XX%
            r'reduce.*?by\s*(\d+(?:\.\d+)?)\s*%',  # reduce by XX%
            r'(\d+(?:\.\d+)?)\s*percent',  # XX percent
            r'by\s*(20\d{2})',  # by 2025
            r'target.*?(20\d{2})',  # target 2025
            r'goal.*?(20\d{2})',  # goal 2030
            r'achieve.*?(\d+(?:\.\d+)?)\s*%.*?(20\d{2})',  # achieve XX% by 2025
        ]

        # Commitment language strength indicators
        self.strong_commitment = [
            'will', 'committed to', 'commits to', 'pledge', 'guarantee',
            'ensure', 'must', 'required to', 'obligated', 'achieve'
        ]

        self.weak_commitment = [
            'aim to', 'aspire', 'hope to', 'intend to', 'seek to',
            'strive', 'plan to', 'working towards', 'ambition', 'explore'
        ]

    def load_extraction(self, json_path):
        """Load JSON extraction file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_targets_from_text(self, text, page_num, doc_name, year):
        """
        Extract target-related information from text

        Returns: List of target dictionaries
        """
        targets = []

        # Split into sentences for better context
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short fragments
                continue

            # Check if sentence contains target-related content
            for category, keywords in self.target_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        # Extract numeric values
                        numbers = self.extract_numbers_from_sentence(sentence)

                        # Assess commitment strength
                        commitment_strength = self.assess_commitment_strength(sentence)

                        target = {
                            'year': year,
                            'document': doc_name,
                            'page': page_num,
                            'category': category,
                            'keyword': keyword,
                            'sentence': sentence,
                            'extracted_numbers': numbers,
                            'commitment_strength': commitment_strength,
                            'contains_timeline': bool(re.search(r'20\d{2}', sentence)),
                            'contains_percentage': '%' in sentence or 'percent' in sentence.lower()
                        }

                        targets.append(target)
                        break  # Only count once per category per sentence

        return targets

    def extract_numbers_from_sentence(self, sentence):
        """Extract all numeric values and years from a sentence"""
        numbers = {
            'percentages': [],
            'years': [],
            'other_numbers': []
        }

        # Extract percentages
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', sentence)
        numbers['percentages'] = [float(x) for x in pct_matches]

        # Extract years
        year_matches = re.findall(r'20\d{2}', sentence)
        numbers['years'] = [int(x) for x in year_matches]

        # Extract other numbers
        other_matches = re.findall(r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b', sentence)
        numbers['other_numbers'] = [x.replace(',', '') for x in other_matches]

        return numbers

    def assess_commitment_strength(self, sentence):
        """
        Assess the strength of commitment language

        Returns: 'strong', 'moderate', or 'weak'
        """
        sentence_lower = sentence.lower()

        strong_count = sum(1 for phrase in self.strong_commitment if phrase in sentence_lower)
        weak_count = sum(1 for phrase in self.weak_commitment if phrase in sentence_lower)

        if strong_count > weak_count:
            return 'strong'
        elif weak_count > strong_count:
            return 'weak'
        else:
            return 'moderate'

    def analyze_document(self, json_path):
        """Analyze a single document for targets"""

        extraction = self.load_extraction(json_path)
        doc_name = extraction['document_name']
        year = self.extract_year_from_filename(doc_name)

        print(f"Analyzing: {doc_name} (Year: {year})")

        all_targets = []

        # Process each page
        for page_data in extraction['pages']:
            page_num = page_data['page_number']
            text = page_data['text']

            targets = self.extract_targets_from_text(text, page_num, doc_name, year)
            all_targets.extend(targets)

        print(f"  Found {len(all_targets)} target-related mentions")

        return all_targets

    def analyze_all_documents(self):
        """Analyze all extracted documents"""

        json_files = sorted(self.extractions_dir.glob('*_extraction.json'))

        print(f"\nAnalyzing {len(json_files)} documents for target setting...")
        print("=" * 80)

        all_targets_data = []

        for json_file in json_files:
            try:
                targets = self.analyze_document(json_file)
                all_targets_data.extend(targets)
            except Exception as e:
                print(f"  ERROR processing {json_file.name}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(all_targets_data)

        # Save detailed results
        output_path = self.output_dir / 'targets_detailed.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved detailed targets to: {output_path}")

        # Create summary analysis
        self.create_target_summary(df)

        return df

    def create_target_summary(self, df):
        """Create summary statistics and insights"""

        print("\n" + "=" * 80)
        print("TARGET ANALYSIS SUMMARY")
        print("=" * 80)

        # By year
        print("\n1. Targets by Year:")
        year_summary = df.groupby('year').agg({
            'sentence': 'count',
            'commitment_strength': lambda x: (x == 'strong').sum()
        }).rename(columns={'sentence': 'total_mentions', 'commitment_strength': 'strong_commitments'})
        print(year_summary)

        # By category
        print("\n2. Targets by Category:")
        category_summary = df.groupby(['year', 'category']).size().unstack(fill_value=0)
        print(category_summary)

        # Commitment strength evolution
        print("\n3. Commitment Strength Over Time:")
        strength_summary = df.groupby(['year', 'commitment_strength']).size().unstack(fill_value=0)
        print(strength_summary)

        # Percentage targets
        print("\n4. Percentage-Based Targets:")
        pct_targets = df[df['contains_percentage']]
        print(f"   Total percentage targets: {len(pct_targets)}")
        print(f"   By year:\n{pct_targets.groupby('year').size()}")

        # Timeline targets
        print("\n5. Timeline-Based Targets:")
        timeline_targets = df[df['contains_timeline']]
        print(f"   Total timeline targets: {len(timeline_targets)}")
        print(f"   By year:\n{timeline_targets.groupby('year').size()}")

        # Save summaries
        summary_data = {
            'year_summary': year_summary,
            'category_summary': category_summary,
            'strength_summary': strength_summary
        }

        # Save to Excel with multiple sheets
        output_excel = self.output_dir / 'targets_summary.xlsx'
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            year_summary.to_excel(writer, sheet_name='By Year')
            category_summary.to_excel(writer, sheet_name='By Category')
            strength_summary.to_excel(writer, sheet_name='Commitment Strength')

        print(f"\n✓ Saved summary to: {output_excel}")

    def identify_target_changes(self, df):
        """
        Identify specific targets that changed across years

        This requires more sophisticated analysis to match similar targets across years
        """
        print("\n" + "=" * 80)
        print("IDENTIFYING TARGET CHANGES")
        print("=" * 80)

        changes = []

        # Group by category and keyword
        for category in df['category'].unique():
            for keyword in df[df['category'] == category]['keyword'].unique():
                subset = df[(df['category'] == category) & (df['keyword'] == keyword)]

                if len(subset) > 1:
                    # Targets appear in multiple years
                    years = sorted(subset['year'].unique())

                    for i in range(len(years) - 1):
                        prev_year = years[i]
                        curr_year = years[i + 1]

                        prev_data = subset[subset['year'] == prev_year]
                        curr_data = subset[subset['year'] == curr_year]

                        # Compare commitment strength
                        prev_strength = prev_data['commitment_strength'].mode()[0] if len(prev_data) > 0 else None
                        curr_strength = curr_data['commitment_strength'].mode()[0] if len(curr_data) > 0 else None

                        if prev_strength != curr_strength:
                            change = {
                                'category': category,
                                'keyword': keyword,
                                'from_year': prev_year,
                                'to_year': curr_year,
                                'previous_strength': prev_strength,
                                'current_strength': curr_strength,
                                'change_type': self.classify_change(prev_strength, curr_strength)
                            }
                            changes.append(change)

        if changes:
            changes_df = pd.DataFrame(changes)
            output_path = self.output_dir / 'target_changes.csv'
            changes_df.to_csv(output_path, index=False)
            print(f"\n✓ Identified {len(changes)} target changes")
            print(f"✓ Saved to: {output_path}")
        else:
            print("\nNo significant target changes detected")

    @staticmethod
    def classify_change(prev_strength, curr_strength):
        """Classify the type of change in commitment"""
        strength_order = {'weak': 1, 'moderate': 2, 'strong': 3}

        if prev_strength is None or curr_strength is None:
            return 'unclear'

        prev_val = strength_order.get(prev_strength, 0)
        curr_val = strength_order.get(curr_strength, 0)

        if curr_val > prev_val:
            return 'strengthened'
        elif curr_val < prev_val:
            return 'weakened'
        else:
            return 'maintained'

    @staticmethod
    def extract_year_from_filename(filename):
        """Extract year from filename"""
        match = re.search(r'20(2[0-4])', filename)
        return int(match.group(0)) if match else None


def main():
    """Main execution"""

    base_dir = Path(__file__).parent.parent
    analyzer = TargetAnalyzer(base_dir)

    # Analyze all documents
    targets_df = analyzer.analyze_all_documents()

    # Identify changes over time
    analyzer.identify_target_changes(targets_df)

    print("\n✓ TARGET ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
