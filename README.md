# Message Processor - Psychological Analysis System

A comprehensive chat analysis system that detects behavioral patterns, assesses psychological risks, and provides actionable safety recommendations. Built with research-backed NLP techniques and enterprise-grade PostgreSQL backend.

## 🚀 Quick Start

```bash
# Analyze a CSV file
./analyze.sh your_messages.csv

# Quick analysis (faster, basic features)
./analyze.sh your_messages.csv -c quick_analysis

# Deep analysis (thorough, all features)
./analyze.sh your_messages.csv -c deep_analysis

# Generate clinical report
./analyze.sh your_messages.csv -c clinical_report -o ClinicalReports/
```

## 🎯 Features

### Advanced Pattern Detection
- **Grooming Detection**: 6 categories, 20+ patterns based on academic research
- **Manipulation & Gaslighting**: 25+ linguistic markers across 6 tactics
- **Deception Analysis**: Forensic linguistics approach with credibility assessment
- **Intent Classification**: 5 intent categories with communication style analysis
- **Sentiment Analysis**: Multi-engine (VADER, TextBlob, NRCLex) with emotion detection

### Risk Assessment
- **4-Component Risk Scoring**: Grooming, manipulation, deception, hostility
- **Risk Levels**: Low, Moderate, High, Critical
- **Escalation Detection**: Identifies increasing risk patterns
- **Intervention Priorities**: Emergency, urgent, priority, routine
- **Actionable Recommendations**: Context-aware safety suggestions

### Database Backend
- **PostgreSQL Integration**: Remote database at acdev.host
- **JSONB Optimization**: Flexible schema with high performance
- **Dedicated CSV Tables**: Each import gets its own table for data integrity
- **Complete Audit Trail**: Full provenance tracking
- **Timeline Aggregations**: Pre-computed analytics for performance

## 📋 Requirements

### System Requirements
- Python 3.8+
- PostgreSQL access (provided: acdev.host)
- 4GB RAM recommended for large datasets

### Python Dependencies
```bash
pip install psycopg2-binary pandas chardet
```

Optional for full features:
```bash
pip install vaderSentiment textblob nrclex nltk
```

## 📊 Input Format

The system accepts CSV files with flexible column naming. Common formats:

| Required Columns | Accepted Names |
|-----------------|----------------|
| Date | Date, date, DATE, Message Date |
| Time | Time, time, TIME, Message Time |
| Sender | Sender Name, Sender, From, Speaker |
| Message | Text, Message, Content, Body |

Optional columns: Sender Number, Recipients, Type, Service, Attachment

## 🔍 Analysis Pipeline

1. **CSV Validation**: Encoding detection, data quality checks
2. **Database Import**: Creates dedicated table, preserves all data
3. **Sentiment Analysis**: Emotional tone, trajectory, volatility
4. **Pattern Detection**: Grooming, manipulation, deception markers
5. **Intent Classification**: Communication dynamics
6. **Risk Assessment**: Comprehensive behavioral risk scoring
7. **Report Generation**: JSON, CSV, and PDF-ready outputs

## 📈 Output

### Reports Generated
- `analysis_TIMESTAMP.json` - Complete analysis data
- `analysis_TIMESTAMP_summary.csv` - Summary statistics
- Console output with key findings and recommendations

### Risk Assessment Includes
- Overall risk level (Low/Moderate/High/Critical)
- Primary concerns identified
- Behavioral pattern summary
- Safety recommendations
- Resource suggestions

## ⚙️ Configuration

### Presets Available
- `quick_analysis` - Fast, basic features only
- `deep_analysis` - Thorough, all features enabled
- `clinical_report` - Optimized for clinical use
- `legal_report` - Includes full evidence trail
- `research` - Academic research configuration

### Custom Configuration
Edit `config/default.json` to customize:
- NLP feature toggles
- Risk weight adjustments
- Analysis parameters
- Database settings
- Output preferences

## 🗄️ Database Schema

### PostgreSQL Tables
- `csv_import_sessions` - Tracks every CSV import
- `messages_master` - Normalized message storage
- `speakers` - Speaker profiles and statistics
- `analysis_runs` - Complete analysis history
- `detected_patterns` - All identified patterns
- `risk_assessments` - Comprehensive risk profiles
- `timeline_aggregations` - Performance optimizations

## 🛡️ Privacy & Security

- All data stored in secure PostgreSQL database
- Complete audit trail for legal/clinical use
- No external API calls for analysis
- Local processing option available with `--use-sqlite`

## 🔧 Advanced Usage

### Command Line Options
```bash
# Use local SQLite instead of PostgreSQL
./analyze.sh messages.csv --use-sqlite

# Disable specific detectors
./analyze.sh messages.csv --no-grooming --no-manipulation

# Verbose output for debugging
./analyze.sh messages.csv -v

# Custom output directory
./analyze.sh messages.csv -o /path/to/reports/
```

### Python API
```python
from message_processor import EnhancedMessageProcessor
from src.config.config_manager import get_config

# Load configuration
config = get_config("deep_analysis")

# Create processor
processor = EnhancedMessageProcessor(config)

# Process file
results = processor.process_csv_file("messages.csv")
```

## 📊 Performance

- Processes 10,000 messages in <60 seconds
- Supports datasets up to 1M+ messages
- Connection pooling for database efficiency
- Parallel processing with multiprocessing
- Feature caching for repeated analyses

## 🚨 Safety Resources

The system may suggest contacting:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- National Domestic Violence Hotline: 1-800-799-7233
- RAINN National Sexual Assault Hotline: 1-800-656-4673

## 📚 Research Foundation

Pattern detection based on:
- VADER Sentiment Analysis (85-90% human agreement)
- Forensic linguistics research for deception detection
- Clinical psychology grooming identification patterns
- Manipulation and gaslighting academic studies

## 🤝 Contributing

This is an active research project. Contributions welcome for:
- Additional pattern definitions
- Language support beyond English
- Visualization improvements
- Clinical validation studies

## 📝 License

Proprietary - For authorized use only

## ⚠️ Disclaimer

This system is a tool for analysis and should not replace professional judgment. Always consult qualified professionals for clinical or legal decisions.

---

**Version**: 2.0
**Database**: PostgreSQL on acdev.host
**Last Updated**: November 2024