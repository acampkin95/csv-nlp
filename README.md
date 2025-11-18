# CSV-NLP: Message Processor

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-red.svg)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A comprehensive NLP-based psychological analysis system for CSV message data. Detects behavioral patterns, assesses risks, and provides actionable safety recommendations.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/acampkin95/csv-nlp.git
cd csv-nlp

# Start with Docker
docker-compose up -d

# Access web interface
open http://localhost:5000
```

## 🎯 Key Features

- **100+ Behavioral Patterns**: Grooming, manipulation, deception detection
- **Multi-Engine Sentiment Analysis**: VADER, TextBlob, NRCLex
- **Risk Assessment**: 4-component scoring with intervention priorities
- **Web Interface**: Upload CSVs, view visualizations, export results
- **PostgreSQL Backend**: JSONB optimization, full audit trail
- **Redis Caching**: 88% performance improvement
- **Docker Ready**: Full containerization with docker-compose

## 📊 Analysis Capabilities

### Pattern Detection
- Grooming: 6 categories, 20+ patterns
- Manipulation: 25+ linguistic markers
- Deception: Forensic linguistics approach
- Intent: 5 classification categories

### Risk Assessment
- Levels: Low, Moderate, High, Critical
- Escalation detection
- Behavioral recommendations
- Safety resources

## 🛠️ Installation

### Docker (Recommended)
```bash
docker-compose up -d
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run the application
python webapp.py
```

## 📁 Project Structure

```
csv-nlp/
├── webapp.py              # Web application
├── message_processor.py   # Core processor
├── docker-compose.yml     # Full stack deployment
├── src/
│   ├── nlp/              # Analysis modules
│   ├── cache/            # Redis caching
│   └── db/               # Database adapters
├── templates/            # Web UI
└── static/              # CSS/JS
```

## 🔧 Configuration

### Analysis Presets
- `quick_analysis` - Fast, basic features
- `deep_analysis` - All features enabled
- `clinical_report` - Clinical documentation
- `legal_report` - Evidence preservation

## 📈 Performance

- Processes 10,000 messages in <60 seconds
- 88% faster with Redis caching
- Supports 1M+ message datasets
- Connection pooling for efficiency

## 🐳 Docker Services

```yaml
services:
  - webapp (Flask application)
  - postgres (PostgreSQL 15)
  - redis (Redis 7)
  - pgadmin (optional)
  - redis-commander (optional)
```

## 🔒 Security

- Local processing (no external APIs)
- PostgreSQL with access controls
- Complete audit trail
- Data integrity verification

## 📝 API Endpoints

- `POST /upload` - Upload CSV file
- `GET /api/projects` - List projects
- `GET /api/analysis/{id}` - Get analysis results
- `GET /api/export/{id}` - Export data
- `GET /api/visualizations/{id}` - Get charts

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Optimization Report](OPTIMIZATION_REPORT.md)
- [Docker Guide](docker/README.md)
- [System Overview](SYSTEM_OVERVIEW.md)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## ⚠️ Disclaimer

This system is for analysis purposes and should not replace professional judgment. Always consult qualified professionals for clinical or legal decisions.

## 📄 License

Proprietary - For authorized use only

## 👤 Author

Alex Campkin - [GitHub](https://github.com/acampkin95)

## 🙏 Acknowledgments

- VADER Sentiment Analysis
- NLTK Project
- TextBlob
- NRCLex

---

**Repository**: https://github.com/acampkin95/csv-nlp
**Issues**: https://github.com/acampkin95/csv-nlp/issues