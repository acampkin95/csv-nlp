# Unified 15-Pass Pipeline - Quick Start Guide

## Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install vaderSentiment textblob nrclex nltk pandas psycopg2-binary
```

### Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## Basic Usage

### Run 15-Pass Unified Pipeline
```bash
python message_processor.py your_data.csv --unified -o Reports/
```

### Run Legacy 10-Pass Pipeline
```bash
python message_processor.py your_data.csv -o Reports/
```

### Using SQLite Backend (Local)
```bash
python message_processor.py your_data.csv --unified --use-sqlite -o Reports/
```

### With Verbose Output
```bash
python message_processor.py your_data.csv --unified -v -o Reports/
```

---

## Input File Format

Your CSV file should contain these columns:
```
sender,text,timestamp
Alice,"Hello, how are you?",2024-01-01 10:00:00
Bob,"I'm fine thanks",2024-01-01 10:05:00
Alice,"Good to hear",2024-01-01 10:10:00
```

### Supported Column Names
- **Sender:** `sender`, `Sender`, `Sender Name`, `From`
- **Message:** `text`, `Text`, `Message`, `Content`
- **Timestamp:** `timestamp`, `Timestamp`, `date`, `Date`, `time`, `Time`

Additional columns are preserved in analysis.

---

## Output Files

After running the pipeline, you'll find:

1. **JSON Report:** `Reports/unified_analysis_[ID]_[TIME].json`
   - Complete analysis results
   - All 15 passes detailed
   - Machine-readable format

2. **CSV Summary:** `Reports/unified_analysis_[ID]_[TIME]_summary.csv`
   - Quick overview of all passes
   - Key metrics
   - Spreadsheet-friendly

---

## Key Results to Look For

### Overall Risk Level
```
risk_level: "low" | "moderate" | "high" | "critical"
```

### Persons Identified (Pass 11)
```python
person_identification: {
    'total_speakers': 2,
    'persons': [
        {
            'name': 'Alice',
            'role': 'primary_initiator',
            'message_count': 150
        },
        {
            'name': 'Bob',
            'role': 'primary_responder',
            'message_count': 140
        }
    ]
}
```

### Gaslighting Detection (Pass 13)
```python
gaslighting_detection: {
    'gaslighting_risk': 'high',  # low, moderate, high, critical
    'total_indicators': 12,
    'perpetrators': [('Alice', 10)],
    'victims': [('Bob', 8)]
}
```

### Relationship Dynamics (Pass 14)
```python
relationship_analysis: {
    'power_imbalance': True,
    'power_imbalance_severity': 'high',
    'relationship_type': 'controlling',  # controlling, imbalanced, balanced
    'relationship_quality': 'poor'       # poor, unhealthy, fair
}
```

### Intervention Recommendations (Pass 15)
```python
intervention_recommendations: {
    'intervention_priority': 'urgent',   # routine, urgent
    'recommendations': [
        'Seek immediate professional mental health support',
        'Consult with a trauma-informed therapist',
        ...
    ]
}
```

---

## Python API Usage

### Basic Example

```python
from src.pipeline.unified_processor import UnifiedProcessor
from src.config.config_manager import ConfigManager

# Initialize
config = ConfigManager().load_config()
processor = UnifiedProcessor(config)

# Process file
result = processor.process_file("conversations.csv", output_dir="Reports")

# Access results
print(f"Risk Level: {result.overall_risk_level}")
print(f"Persons: {result.person_identification['total_speakers']}")
print(f"Gaslighting Risk: {result.gaslighting_detection['gaslighting_risk']}")
print(f"Recommendations: {len(result.recommendations)} items")
```

### With PostgreSQL Backend

```python
from message_processor import UnifiedEnhancedMessageProcessor
from src.config.config_manager import ConfigManager

config = ConfigManager().load_config()
processor = UnifiedEnhancedMessageProcessor(config, use_postgresql=True)

result = processor.process_csv_file("conversations.csv", "Reports")
print(f"Stored with Analysis ID: {result['analysis_run_id']}")
```

### Advanced: Select Specific Passes

```python
from src.config.config_manager import ConfigManager

config = ConfigManager().load_config()

# Disable specific analyses
config.nlp.enable_grooming_detection = False
config.nlp.enable_deception_markers = False

processor = UnifiedProcessor(config)
result = processor.process_file("conversations.csv")
```

---

## Interpreting Results

### Risk Levels
- **LOW:** Minimal concern, healthy communication patterns
- **MODERATE:** Some concerning patterns, monitor situation
- **HIGH:** Significant concerns, professional support recommended
- **CRITICAL:** Severe patterns, immediate intervention needed

### Gaslighting Risk
- **LOW:** 0-5% indicator density
- **MODERATE:** 5-10% indicator density
- **HIGH:** 10-20% indicator density
- **CRITICAL:** >20% indicator density

### Power Imbalance Severity
- **NONE:** Balanced communication
- **HIGH:** One person dominates significantly
- **SEVERE:** Severe control/dominance patterns

### Relationship Types
- **BALANCED:** Equal power, healthy dynamics
- **IMBALANCED:** Unequal but not controlling
- **CONTROLLING:** Significant control patterns present

---

## Common Issues & Solutions

### Issue: "Module not found" error
```bash
# Solution: Install in editable mode from Dev-Root
cd "/Users/alex/Projects/Dev/Projects/Message Processor/Dev-Root"
pip install -e .
```

### Issue: PostgreSQL connection fails
```bash
# Solution: Use SQLite backend temporarily
python message_processor.py input.csv --unified --use-sqlite
```

### Issue: NLTK data missing
```bash
# Solution: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Issue: Large file processing slow
```bash
# Solution: Disable unnecessary passes
python message_processor.py input.csv --unified --no-grooming --no-deception
```

---

## Configuration Presets

### Deep Analysis (Recommended)
```bash
python message_processor.py input.csv --unified -c deep_analysis
```
- All 15 passes enabled
- Comprehensive analysis
- Detailed output

### Quick Analysis
```bash
python message_processor.py input.csv --unified -c quick_analysis
```
- Essential passes only
- Fast processing
- Basic risk assessment

### Clinical Report
```bash
python message_processor.py input.csv --unified -c clinical_report
```
- Therapeutic focus
- Clinical language
- Professional recommendations

### Legal Report
```bash
python message_processor.py input.csv --unified -c legal_report
```
- Forensic focus
- Evidence-based
- Timeline reconstruction

---

## Example Workflow

### Step 1: Prepare Your Data
```bash
# Ensure CSV has columns: sender, text, timestamp
# Export from chat application or prepare manually
```

### Step 2: Run Analysis
```bash
python message_processor.py messages.csv --unified -o analysis_output/
```

### Step 3: Review Results
```bash
# Open JSON report in text editor or IDE
# Review CSV summary in spreadsheet application
# Check console output for key findings
```

### Step 4: Share Results
```bash
# JSON: For archival and re-analysis
# CSV: For stakeholder review
# PostgreSQL: For longitudinal studies
```

---

## Best Practices

### Data Preparation
- ✓ Ensure consistent sender names (standardize spelling)
- ✓ Include timestamps when possible
- ✓ Remove metadata rows (column headers in data)
- ✓ Use UTF-8 encoding

### Analysis Selection
- ✓ Use `--unified` for person-centric analysis
- ✓ Use legacy mode only if 10-pass sufficient
- ✓ Disable unused passes for speed
- ✓ Use appropriate preset for use case

### Result Interpretation
- ✓ Review all 15 pass results, not just risk level
- ✓ Consider clinical context
- ✓ Don't over-interpret low-sample conversations
- ✓ Consult with professionals for clinical decisions

### Data Privacy
- ✓ Use SQLite for sensitive local data
- ✓ Encrypt PostgreSQL connections in production
- ✓ Store credentials in environment variables
- ✓ Implement access controls

---

## Advanced Features

### Batch Processing Multiple Files
```bash
for file in *.csv; do
    python message_processor.py "$file" --unified -o "Reports/${file%.csv}/"
done
```

### Custom Configuration
```python
from src.config.config_manager import ConfigManager, Configuration

config = Configuration()
config.nlp.enable_grooming_detection = True
config.nlp.risk_weight_grooming = 0.3
config.database.enable_caching = True

processor = UnifiedProcessor(config)
result = processor.process_file("input.csv")
```

### Result Re-analysis
```python
import json

# Load previous results
with open('analysis_output/unified_analysis_123_20240101.json') as f:
    previous_result = json.load(f)

# Compare with new analysis
# Generate change report
# Track progression
```

---

## Performance Tips

### For Large Files (>500 messages)
```bash
# Disable non-essential passes
python message_processor.py large_file.csv --unified \
    --no-grooming --no-deception

# Use PostgreSQL for persistence
python message_processor.py large_file.csv --unified
# (assumes PostgreSQL configured)
```

### For Repeated Analysis
```bash
# Use SQLite caching
python message_processor.py input.csv --unified --use-sqlite
# Results cached in local database
```

### Parallel Processing
```python
# Configuration supports workers parameter
config.analysis.workers = 4  # Use 4 CPU cores
processor = UnifiedProcessor(config)
```

---

## Getting Help

### Check Documentation
1. **PIPELINE_DOCUMENTATION.md** - Complete reference
2. **ARCHITECTURE.md** - Technical details
3. **Module docstrings** - In source code

### Common Commands Reference
```bash
# Full analysis with details
python message_processor.py input.csv --unified -v -o output/

# Fast analysis with SQLite
python message_processor.py input.csv --unified --use-sqlite

# Legacy 10-pass analysis
python message_processor.py input.csv -o output/

# Show help
python message_processor.py --help
```

### Debugging
```bash
# Enable verbose logging
python message_processor.py input.csv --unified -v

# Check for Python syntax errors
python -m py_compile src/pipeline/unified_processor.py

# Test import
python -c "from src.pipeline.unified_processor import UnifiedProcessor"
```

---

## Next Steps

1. **Review PIPELINE_DOCUMENTATION.md** for detailed pass descriptions
2. **Review ARCHITECTURE.md** for technical implementation
3. **Run example:** `python message_processor.py sample_data.csv --unified`
4. **Examine output** to understand result structure
5. **Customize config** for your specific use case
6. **Deploy** to production with appropriate backend

---

## Support Resources

- **Configuration:** `src/config/config_manager.py`
- **Main Entry:** `message_processor.py`
- **Pipeline Code:** `src/pipeline/unified_processor.py`
- **Person Analysis:** `src/nlp/person_analyzer.py`
- **Full Docs:** `PIPELINE_DOCUMENTATION.md`
- **Architecture:** `ARCHITECTURE.md`

Good luck with your analysis!
