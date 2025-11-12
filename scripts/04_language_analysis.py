"""
Nike ESG Analysis - Language and Tone Analysis
==============================================

This script analyzes language patterns, tone shifts, and sentiment
in Nike's ESG communications over time.

Author: Adapted from Coco Zhang's H&M analysis
Date: 2025-11-11
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class LanguageAnalyzer:
    """Analyze language patterns and tone in ESG reports"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.extractions_dir = self.base_dir / 'data' / 'raw_extractions'
        self.output_dir = self.base_dir / 'data' / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()

        # ESG terminology to track
        self.esg_terms = {
            'climate_action': [
                'climate change', 'climate action', 'climate crisis', 'climate emergency',
                'global warming', 'climate positive', 'climate neutral'
            ],
            'carbon_terminology': [
                'net zero', 'carbon neutral', 'carbon negative', 'carbon footprint',
                'decarbonization', 'low carbon', 'carbon emissions'
            ],
            'circular_economy': [
                'circular economy', 'circular fashion', 'circular design',
                'recycling', 'recycled materials', 'Nike Grind', 'circularity'
            ],
            'sustainability_framing': [
                'sustainability', 'sustainable development', 'sustainable',
                'responsible', 'ethical', 'Move to Zero'
            ],
            'compliance_language': [
                'compliance', 'regulatory', 'mandatory', 'requirement',
                'regulation', 'legislation', 'directive', 'law'
            ],
            'voluntary_leadership': [
                'leadership', 'pioneering', 'leading', 'first to',
                'innovation', 'ambitious', 'bold', 'transformative'
            ],
            'science_based': [
                'science-based', 'evidence-based', 'scientific',
                'paris agreement', '1.5°C', 'IPCC'
            ],
            'stakeholder_engagement': [
                'stakeholder', 'collaboration', 'partnership', 'engagement',
                'consultation', 'dialogue', 'transparency'
            ],
            'supply_chain': [
                'supply chain', 'supplier', 'value chain', 'upstream',
                'downstream', 'tier 1', 'tier 2', 'traceability'
            ],
            'social_justice': [
                'living wage', 'fair wage', 'human rights', 'worker rights',
                'gender equality', 'diversity', 'inclusion', 'equity'
            ]
        }

        # Hedging/qualifying language (indicates uncertainty or retreat)
        self.hedging_terms = [
            'may', 'might', 'could', 'would', 'should',
            'potentially', 'possibly', 'perhaps', 'approximately',
            'around', 'estimate', 'project', 'expect',
            'aim to', 'strive', 'aspire', 'seek to'
        ]

        # Strong commitment language
        self.commitment_terms = [
            'will', 'commit', 'committed to', 'pledge',
            'guarantee', 'ensure', 'achieve', 'deliver',
            'must', 'required', 'obligated', 'dedicated to'
        ]

    def load_extraction(self, json_path):
        """Load JSON extraction file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER

        Returns: Dictionary with sentiment scores
        """
        scores = self.sia.polarity_scores(text)
        return {
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'positive': scores['pos'],
            'compound': scores['compound']
        }

    def analyze_sentiment_chunked(self, text, chunk_size=5000):
        """
        Analyze sentiment of long text by splitting into chunks
        VADER has issues with very long texts, so we analyze in chunks and average

        Returns: Dictionary with averaged sentiment scores
        """
        # Split text into sentences first
        import re
        sentences = re.split(r'[.!?]+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence.split())

            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Analyze each chunk
        if not chunks:
            return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'compound': 0.0}

        neg_scores = []
        neu_scores = []
        pos_scores = []
        compound_scores = []

        for chunk in chunks:
            scores = self.sia.polarity_scores(chunk)
            neg_scores.append(scores['neg'])
            neu_scores.append(scores['neu'])
            pos_scores.append(scores['pos'])
            compound_scores.append(scores['compound'])

        # Average the scores
        return {
            'negative': sum(neg_scores) / len(neg_scores),
            'neutral': sum(neu_scores) / len(neu_scores),
            'positive': sum(pos_scores) / len(pos_scores),
            'compound': sum(compound_scores) / len(compound_scores)
        }

    def count_term_frequency(self, text, term_list):
        """Count frequency of terms in text (case-insensitive)"""
        text_lower = text.lower()
        count = 0
        matches = []

        for term in term_list:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            found = re.findall(pattern, text_lower)
            count += len(found)
            if found:
                matches.extend([term] * len(found))

        return count, matches

    def analyze_document(self, json_path):
        """Analyze language patterns in a single document"""

        extraction = self.load_extraction(json_path)
        doc_name = extraction['document_name']
        year = self.extract_year_from_filename(doc_name)

        print(f"Analyzing language: {doc_name} (Year: {year})")

        # Combine all text
        full_text = ' '.join([page['text'] for page in extraction['pages']])
        word_count = len(full_text.split())

        # Overall sentiment - analyze in chunks to avoid VADER max length
        sentiment = self.analyze_sentiment_chunked(full_text)

        # Count ESG term categories
        term_frequencies = {}
        term_details = {}

        for category, terms in self.esg_terms.items():
            count, matches = self.count_term_frequency(full_text, terms)
            term_frequencies[category] = count
            term_frequencies[f'{category}_per_1000_words'] = (count / word_count * 1000) if word_count > 0 else 0
            term_details[category] = Counter(matches)

        # Hedging vs commitment language
        hedging_count, _ = self.count_term_frequency(full_text, self.hedging_terms)
        commitment_count, _ = self.count_term_frequency(full_text, self.commitment_terms)

        # Calculate commitment ratio
        total_commitment_language = hedging_count + commitment_count
        commitment_ratio = commitment_count / total_commitment_language if total_commitment_language > 0 else 0.5

        result = {
            'year': year,
            'document': doc_name,
            'word_count': word_count,
            'page_count': len(extraction['pages']),
            'sentiment_negative': sentiment['negative'],
            'sentiment_neutral': sentiment['neutral'],
            'sentiment_positive': sentiment['positive'],
            'sentiment_compound': sentiment['compound'],
            'hedging_count': hedging_count,
            'commitment_count': commitment_count,
            'commitment_ratio': commitment_ratio,
            **term_frequencies
        }

        return result, term_details

    def analyze_all_documents(self):
        """Analyze all documents"""

        json_files = sorted(self.extractions_dir.glob('*_extraction.json'))

        print(f"\nAnalyzing language in {len(json_files)} documents...")
        print("=" * 80)

        all_results = []
        all_term_details = defaultdict(list)

        for json_file in json_files:
            try:
                result, term_details = self.analyze_document(json_file)
                all_results.append(result)

                # Store term details
                for category, counter in term_details.items():
                    all_term_details[category].append({
                        'year': result['year'],
                        'document': result['document'],
                        'terms': dict(counter)
                    })

            except Exception as e:
                print(f"  ERROR: {e}")

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Sort by year
        df = df.sort_values('year')

        # Save detailed results
        output_path = self.output_dir / 'language_analysis.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved language analysis to: {output_path}")

        # Save term details
        term_details_path = self.output_dir / 'esg_term_details.json'
        with open(term_details_path, 'w') as f:
            json.dump(dict(all_term_details), f, indent=2)
        print(f"✓ Saved term details to: {term_details_path}")

        # Generate insights
        self.generate_language_insights(df)

        return df

    def generate_language_insights(self, df):
        """Generate insights from language analysis"""

        print("\n" + "=" * 80)
        print("LANGUAGE ANALYSIS INSIGHTS")
        print("=" * 80)

        # 1. Sentiment evolution
        print("\n1. Sentiment Evolution:")
        sentiment_cols = ['year', 'sentiment_negative', 'sentiment_neutral',
                          'sentiment_positive', 'sentiment_compound']
        if all(col in df.columns for col in sentiment_cols):
            sentiment_df = df[sentiment_cols].groupby('year').mean()
            print(sentiment_df.round(3))

        # 2. Commitment vs Hedging
        print("\n2. Commitment Language Ratio (higher = stronger commitment):")
        if 'commitment_ratio' in df.columns:
            commitment_df = df[['year', 'commitment_ratio']].groupby('year').mean()
            print(commitment_df.round(3))

        # 3. ESG terminology trends
        print("\n3. ESG Terminology Frequency (per 1000 words):")

        term_cols = [col for col in df.columns if '_per_1000_words' in col]
        if term_cols:
            trends_df = df.groupby('year')[term_cols].mean()

            # Rename columns for readability
            trends_df.columns = [col.replace('_per_1000_words', '') for col in trends_df.columns]

            print(trends_df.round(2))

            # Calculate year-over-year changes
            print("\n4. Year-over-Year Changes in Key Terms:")
            pct_change = trends_df.pct_change() * 100

            # Show significant changes
            for col in pct_change.columns:
                recent_change = pct_change[col].iloc[-1] if len(pct_change) > 0 else 0
                if abs(recent_change) > 20:  # More than 20% change
                    direction = "increased" if recent_change > 0 else "decreased"
                    print(f"   • {col}: {direction} by {abs(recent_change):.1f}%")

        # 5. Compliance vs Leadership language
        print("\n5. Compliance vs. Leadership Framing:")
        if 'compliance_language' in df.columns and 'voluntary_leadership' in df.columns:
            framing_df = df.groupby('year')[['compliance_language', 'voluntary_leadership']].mean()
            print(framing_df.round(2))

        # Save summary
        summary_path = self.output_dir / 'language_summary.xlsx'
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            if all(col in df.columns for col in sentiment_cols):
                sentiment_df.to_excel(writer, sheet_name='Sentiment')
            if term_cols:
                trends_df.to_excel(writer, sheet_name='ESG Terms')
            if 'commitment_ratio' in df.columns:
                commitment_df.to_excel(writer, sheet_name='Commitment Ratio')

        print(f"\n✓ Saved summary to: {summary_path}")

    @staticmethod
    def extract_year_from_filename(filename):
        """Extract year from filename"""
        match = re.search(r'20(2[0-4])', filename)
        return int(match.group(0)) if match else None


def main():
    """Main execution"""

    base_dir = Path(__file__).parent.parent
    analyzer = LanguageAnalyzer(base_dir)

    # Analyze all documents
    results_df = analyzer.analyze_all_documents()

    print("\n✓ LANGUAGE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
