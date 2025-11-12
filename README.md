# Nike Corporate Response Analysis

Comprehensive ESG (Environmental, Social, and Governance) policy response analysis of Nike's corporate disclosures from 2020-2024.

## Project Overview

This project analyzes Nike's ESG communications to identify:
- **Target Setting Evolution**: Quantitative and qualitative goals across climate, circular economy, water, social justice, and transparency
- **Language & Tone Shifts**: Sentiment analysis and commitment strength tracking
- **Policy Response Patterns**: How Nike's communications change in response to policy shifts
- **Commitment Tracking**: Identification of strengthened, weakened, or removed commitments

## Project Structure

```
nike_corporate_response_analysis/
├── Nike_disclosures/          # Original PDF reports organized by year
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
├── scripts/                   # Analysis scripts
│   ├── 00_run_full_analysis.py          # Master pipeline runner
│   ├── 01_pdf_extraction.py             # PDF text extraction
│   ├── 03_target_analysis.py            # Target setting analysis
│   ├── 04_language_analysis.py          # Language & sentiment analysis
│   ├── 05_visualization.py              # Visualization generation
│   └── 06_create_master_dataset.py      # Master dataset compilation
├── data/                      # Data storage
│   ├── raw_extractions/      # Extracted PDF text with page numbers
│   └── processed/            # Processed analytical datasets
├── outputs/                   # Analysis outputs
│   ├── visualizations/       # Charts and graphs
│   ├── master_dataset.csv    # Consolidated findings
│   ├── citations_index.xlsx  # Page-referenced citations
│   └── summary_statistics.csv # Analysis summary
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data** (for sentiment analysis)
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Usage

### Running the Complete Analysis

To run the entire analysis pipeline:

```bash
python scripts/00_run_full_analysis.py
```

This will execute all steps in sequence:
1. Extract text from all PDF reports
2. Analyze target setting and goals
3. Perform language and sentiment analysis
4. Generate visualizations
5. Compile master dataset with citations

### Running Individual Scripts

You can also run individual analysis steps:

```bash
# Extract PDFs only
python scripts/01_pdf_extraction.py

# Analyze targets only
python scripts/03_target_analysis.py

# Language analysis only
python scripts/04_language_analysis.py

# Generate visualizations only
python scripts/05_visualization.py

# Create master dataset only
python scripts/06_create_master_dataset.py
```

## Analysis Components

### 1. PDF Extraction (`01_pdf_extraction.py`)
- Extracts text from all Nike disclosure PDFs
- Preserves page numbers for citation tracking
- Outputs: JSON and TXT formats with page-level granularity

### 2. Target Analysis (`03_target_analysis.py`)
- Identifies quantitative and qualitative targets
- Categories: Climate, Circular Economy, Water, Chemicals, Social, Transparency, Product
- Assesses commitment strength (strong/moderate/weak)
- Tracks targets with timelines and percentages
- Identifies year-over-year changes

### 3. Language Analysis (`04_language_analysis.py`)
- Sentiment analysis using VADER
- ESG terminology frequency tracking
- Commitment vs. hedging language ratio
- Compliance vs. voluntary leadership framing

### 4. Visualization (`05_visualization.py`)
Generates 8 comprehensive charts:
- Sentiment evolution over time
- Commitment ratio trends
- ESG terminology frequency
- Target categories heatmap
- Commitment strength distribution
- Compliance vs. leadership language
- Target changes timeline
- Overall dashboard

### 5. Master Dataset (`06_create_master_dataset.py`)
- Consolidates all findings with page references
- Creates Excel citations index organized by category
- Generates summary statistics

## Key Outputs

### Processed Data Files
- `data/processed/targets_detailed.csv` - All target mentions with metadata
- `data/processed/targets_summary.xlsx` - Summary tables by year and category
- `data/processed/language_analysis.csv` - Language metrics by document
- `data/processed/language_summary.xlsx` - Language trends summary
- `data/processed/target_changes.csv` - Identified commitment changes

### Visualizations
All charts saved as high-resolution PNG files (300 DPI) in `outputs/visualizations/`:
1. `01_sentiment_evolution.png`
2. `02_commitment_ratio.png`
3. `03_esg_terminology_trends.png`
4. `04_target_categories_heatmap.png`
5. `05_commitment_strength_stacked.png`
6. `06_compliance_vs_leadership.png`
7. `07_target_changes_timeline.png`
8. `08_overall_dashboard.png`

### Final Outputs
- `outputs/master_dataset.csv` - Complete dataset with all findings
- `outputs/citations_index.xlsx` - Page-referenced citations by category
- `outputs/summary_statistics.csv` - Analysis overview metrics

## Target Categories

The analysis tracks targets across these categories:

| Category | Keywords Tracked |
|----------|-----------------|
| **Climate** | Net zero, carbon neutral, emissions reduction, GHG, renewable energy, SBT |
| **Circular** | Circular economy, recycled materials, waste reduction, Nike Grind, Move to Zero |
| **Water** | Water reduction, water stewardship, clean water, wastewater |
| **Chemicals** | ZDHC, hazardous chemicals, safer chemistry, restricted substances |
| **Social** | Living wage, worker rights, diversity, human rights, gender equality |
| **Transparency** | Supply chain transparency, traceability, supplier disclosure |
| **Product** | Sustainable product, sustainable innovation, eco-friendly design |

## Quality Assurance

All findings include:
- **Document Name**: Source file
- **Year**: Report year
- **Page Number**: Exact page reference
- **Full Text**: Complete sentence/paragraph context

This enables cross-referencing with original PDFs for validation.

## Analysis Workflow

Based on Coco Zhang's corporate response analysis framework:

1. ✅ Download all public reports (2020-2024)
2. ✅ Organize by year with consistent naming
3. ✅ Extract text while preserving page references
4. ✅ Analyze target setting and commitment language
5. ✅ Track language patterns and sentiment
6. ✅ Generate visualizations
7. ✅ Compile master dataset with citations
8. ⏳ Quality check against original reports
9. ⏳ Present findings

## Technical Notes

- **Text Extraction**: Uses PyPDF2 for reliable PDF processing
- **Sentiment Analysis**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Visualization**: Matplotlib & Seaborn with high-DPI output
- **Data Format**: CSV for analysis, Excel for citations (better for page references)

## Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements.txt
```

**NLTK data missing**
```python
import nltk
nltk.download('vader_lexicon')
```

**PDF extraction errors**
- Ensure PDFs are not corrupted
- Check that Nike_disclosures folder contains year subfolders
- Verify PDF filenames follow pattern: `Nike_YYYY_*.pdf`

## Author

Adapted from Coco Zhang's H&M Corporate Response Analysis framework

## License

For educational and research purposes.

## Next Steps

1. Run the complete analysis pipeline
2. Review extraction quality in `data/raw_extractions/extraction_summary.csv`
3. Examine processed datasets for insights
4. Validate key findings against original PDFs using page references
5. Use visualizations for presentation and reporting