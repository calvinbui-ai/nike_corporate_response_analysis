"""
Nike ESG Analysis - Data Visualization
======================================

This script generates comprehensive visualizations of Nike's ESG response
to policy shifts over time.

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


class ESGVisualizer:
    """Create visualizations for ESG analysis"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data' / 'processed'
        self.output_dir = self.base_dir / 'outputs' / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_data()

    def load_data(self):
        """Load processed data files"""

        print("Loading processed data...")

        # Language analysis
        lang_file = self.data_dir / 'language_analysis.csv'
        self.language_df = pd.read_csv(lang_file) if lang_file.exists() else None

        # Targets analysis
        targets_file = self.data_dir / 'targets_detailed.csv'
        self.targets_df = pd.read_csv(targets_file) if targets_file.exists() else None

        # Target changes
        changes_file = self.data_dir / 'target_changes.csv'
        self.changes_df = pd.read_csv(changes_file) if changes_file.exists() else None

        print(f"  ✓ Language data: {len(self.language_df) if self.language_df is not None else 0} rows")
        print(f"  ✓ Targets data: {len(self.targets_df) if self.targets_df is not None else 0} rows")
        print(f"  ✓ Changes data: {len(self.changes_df) if self.changes_df is not None else 0} rows")

    def create_all_visualizations(self):
        """Generate all visualizations"""

        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        viz_functions = [
            self.plot_sentiment_evolution,
            self.plot_commitment_ratio,
            self.plot_esg_terminology_trends,
            self.plot_target_categories_heatmap,
            self.plot_commitment_strength_stacked,
            self.plot_compliance_vs_leadership,
            self.plot_target_changes_timeline,
            self.plot_overall_dashboard
        ]

        for viz_func in viz_functions:
            try:
                viz_func()
                plt.close('all')  # Clean up
            except Exception as e:
                print(f"  ERROR in {viz_func.__name__}: {e}")

        print(f"\n✓ All visualizations saved to: {self.output_dir}")

    def plot_sentiment_evolution(self):
        """Plot sentiment scores over time"""

        if self.language_df is None:
            return

        print("\n1. Creating sentiment evolution chart...")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by year and average
        yearly_sentiment = self.language_df.groupby('year')[
            ['sentiment_positive', 'sentiment_neutral', 'sentiment_negative', 'sentiment_compound']
        ].mean()

        # Plot compound sentiment
        ax.plot(yearly_sentiment.index, yearly_sentiment['sentiment_compound'],
                marker='o', linewidth=2.5, markersize=8, label='Compound Sentiment')

        # Add trend line
        z = np.polyfit(yearly_sentiment.index, yearly_sentiment['sentiment_compound'], 1)
        p = np.poly1d(z)
        ax.plot(yearly_sentiment.index, p(yearly_sentiment.index),
                "--", alpha=0.5, color='gray', label='Trend')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Compound Sentiment Score', fontsize=12)
        ax.set_title('Nike ESG Communication Sentiment Evolution (2020-2024)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '01_sentiment_evolution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 01_sentiment_evolution.png")

    def plot_commitment_ratio(self):
        """Plot commitment vs hedging language ratio"""

        if self.language_df is None:
            return

        print("2. Creating commitment ratio chart...")

        fig, ax = plt.subplots(figsize=(12, 6))

        yearly_commitment = self.language_df.groupby('year')['commitment_ratio'].mean()

        bars = ax.bar(yearly_commitment.index, yearly_commitment.values,
                      color=sns.color_palette("RdYlGn", len(yearly_commitment)))

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)

        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Commitment Ratio', fontsize=12)
        ax.set_title('Commitment Language Strength Over Time\n(Higher = Stronger Commitment)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '02_commitment_ratio.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 02_commitment_ratio.png")

    def plot_esg_terminology_trends(self):
        """Plot ESG terminology frequency trends"""

        if self.language_df is None:
            return

        print("3. Creating ESG terminology trends...")

        # Select key term categories (per 1000 words)
        term_cols = [col for col in self.language_df.columns if '_per_1000_words' in col]

        if not term_cols:
            return

        # Get top 8 most frequent term categories
        term_totals = self.language_df[term_cols].sum().sort_values(ascending=False)
        top_terms = term_totals.head(8).index.tolist()

        yearly_terms = self.language_df.groupby('year')[top_terms].mean()

        # Clean names for display
        yearly_terms.columns = [col.replace('_per_1000_words', '').replace('_', ' ').title()
                               for col in yearly_terms.columns]

        fig, ax = plt.subplots(figsize=(14, 7))

        yearly_terms.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=6)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Frequency (per 1000 words)', fontsize=12)
        ax.set_title('ESG Terminology Frequency Evolution (2020-2024)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Term Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '03_esg_terminology_trends.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 03_esg_terminology_trends.png")

    def plot_target_categories_heatmap(self):
        """Create heatmap of target mentions by category and year"""

        if self.targets_df is None:
            return

        print("4. Creating target categories heatmap...")

        # Count targets by year and category
        target_matrix = self.targets_df.groupby(['year', 'category']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(target_matrix.T, annot=True, fmt='d', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Number of Target Mentions'})

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Target Category', fontsize=12)
        ax.set_title('Nike Target Setting Activity by Category (2020-2024)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / '04_target_categories_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 04_target_categories_heatmap.png")

    def plot_commitment_strength_stacked(self):
        """Stacked bar chart of commitment strength over time"""

        if self.targets_df is None:
            return

        print("5. Creating commitment strength chart...")

        strength_data = self.targets_df.groupby(['year', 'commitment_strength']).size().unstack(fill_value=0)

        # Reorder columns
        desired_order = ['strong', 'moderate', 'weak']
        strength_data = strength_data[[col for col in desired_order if col in strength_data.columns]]

        fig, ax = plt.subplots(figsize=(12, 6))

        strength_data.plot(kind='bar', stacked=True, ax=ax,
                          color=['#2ecc71', '#f39c12', '#e74c3c'])

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Targets', fontsize=12)
        ax.set_title('Target Commitment Strength Distribution Over Time',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Commitment Strength', labels=['Strong', 'Moderate', 'Weak'])
        ax.set_xticklabels(strength_data.index, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '05_commitment_strength_stacked.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 05_commitment_strength_stacked.png")

    def plot_compliance_vs_leadership(self):
        """Plot compliance vs voluntary leadership language"""

        if self.language_df is None:
            return

        if 'compliance_language' not in self.language_df.columns:
            return

        print("6. Creating compliance vs leadership chart...")

        yearly_framing = self.language_df.groupby('year')[
            ['compliance_language', 'voluntary_leadership']
        ].mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(yearly_framing.index))
        width = 0.35

        bars1 = ax.bar(x - width/2, yearly_framing['compliance_language'],
                      width, label='Compliance Language', color='#3498db')
        bars2 = ax.bar(x + width/2, yearly_framing['voluntary_leadership'],
                      width, label='Leadership Language', color='#e74c3c')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Compliance vs. Voluntary Leadership Framing',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(yearly_framing.index)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '06_compliance_vs_leadership.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 06_compliance_vs_leadership.png")

    def plot_target_changes_timeline(self):
        """Timeline of target changes (strengthened/weakened)"""

        if self.changes_df is None or len(self.changes_df) == 0:
            print("7. Skipping target changes (no data)")
            return

        print("7. Creating target changes timeline...")

        fig, ax = plt.subplots(figsize=(14, 8))

        change_counts = self.changes_df.groupby(['to_year', 'change_type']).size().unstack(fill_value=0)

        change_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#95a5a6', '#e74c3c'])

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Changes', fontsize=12)
        ax.set_title('Target Commitment Changes Over Time',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Change Type')
        ax.set_xticklabels(change_counts.index, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '07_target_changes_timeline.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 07_target_changes_timeline.png")

    def plot_overall_dashboard(self):
        """Create comprehensive dashboard"""

        print("8. Creating overall dashboard...")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Sentiment
        if self.language_df is not None:
            ax1 = fig.add_subplot(gs[0, 0])
            yearly_sentiment = self.language_df.groupby('year')['sentiment_compound'].mean()
            ax1.plot(yearly_sentiment.index, yearly_sentiment.values, marker='o', linewidth=2.5, color='#3498db')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax1.set_title('Sentiment Evolution', fontweight='bold')
            ax1.set_ylabel('Compound Sentiment')
            ax1.grid(True, alpha=0.3)

        # 2. Commitment Ratio
        if self.language_df is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            yearly_commitment = self.language_df.groupby('year')['commitment_ratio'].mean()
            ax2.bar(yearly_commitment.index, yearly_commitment.values, color='#2ecc71')
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Commitment Ratio', fontweight='bold')
            ax2.set_ylabel('Ratio')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Target Categories
        if self.targets_df is not None:
            ax3 = fig.add_subplot(gs[1, :])
            category_counts = self.targets_df.groupby(['year', 'category']).size().unstack(fill_value=0)
            category_counts.plot(kind='bar', stacked=True, ax=ax3)
            ax3.set_title('Target Setting by Category', fontweight='bold')
            ax3.set_ylabel('Number of Targets')
            ax3.set_xlabel('Year')
            ax3.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Word Count Evolution
        if self.language_df is not None:
            ax4 = fig.add_subplot(gs[2, 0])
            yearly_words = self.language_df.groupby('year')['word_count'].sum()
            ax4.bar(yearly_words.index, yearly_words.values, color='#9b59b6')
            ax4.set_title('Total Word Count', fontweight='bold')
            ax4.set_ylabel('Words')
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Compliance vs Leadership
        if self.language_df is not None and 'compliance_language' in self.language_df.columns:
            ax5 = fig.add_subplot(gs[2, 1])
            yearly_framing = self.language_df.groupby('year')[
                ['compliance_language', 'voluntary_leadership']
            ].mean()

            x = np.arange(len(yearly_framing.index))
            width = 0.35
            ax5.bar(x - width/2, yearly_framing['compliance_language'], width, label='Compliance', color='#3498db')
            ax5.bar(x + width/2, yearly_framing['voluntary_leadership'], width, label='Leadership', color='#e74c3c')
            ax5.set_title('Compliance vs Leadership', fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(yearly_framing.index)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Nike ESG Policy Response Analysis Dashboard (2020-2024)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.output_dir / '08_overall_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 08_overall_dashboard.png")


def main():
    """Main execution"""

    base_dir = Path(__file__).parent.parent

    visualizer = ESGVisualizer(base_dir)
    visualizer.create_all_visualizations()

    print("\n✓ VISUALIZATION COMPLETE")
    print(f"✓ All charts saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
