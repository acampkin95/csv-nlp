# Message Processor - Psychological Analysis System

**Version 2.0** | Optimized & Containerized | Production-Ready

A comprehensive psychological analysis system for chat messages, featuring advanced NLP analysis, behavioral pattern detection, and risk assessment capabilities.

![Status](https://img.shields.io/badge/status-production--ready-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![License](https://img.shields.io/badge/license-proprietary-red)

---

## Features

### Core Analysis Capabilities

- **Sentiment Analysis**: Multi-engine sentiment tracking with VADER, TextBlob, and NRCLex
- **Grooming Detection**: Research-backed pattern detection across 6 behavioral categories
- **Manipulation Analysis**: Identification of gaslighting, coercion, and control tactics
- **Deception Markers**: Linguistic analysis for credibility assessment
- **Intent Classification**: Communication pattern and dynamic analysis
- **Risk Scoring**: Comprehensive behavioral risk assessment

### System Features

- **Web Interface**: Modern, responsive web application with interactive visualizations
- **Redis Caching**: High-performance caching reducing analysis time by up to 88%
- **PostgreSQL Backend**: Robust data persistence and session management
- **Docker Deployment**: Fully containerized for easy deployment and scaling
- **Project Management**: Multi-project support with CSV session tracking
- **Export Capabilities**: JSON, CSV, and PDF (planned) export options
- **Real-time Visualizations**: Interactive charts powered by Plotly.js

---

## Quick Start

### Docker Deployment (Recommended)

```bash
# 1. Clone and navigate
cd "Message Processor/Dev-Root"

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start all services
docker-compose up -d

# 4. Access application
open http://localhost:5000
```

### Local Development

```bash
# 1. Set up Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure services (PostgreSQL and Redis)
# See DEPLOYMENT_GUIDE.md for details

# 3. Run application
python webapp.py
```

---

## Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)** - Performance analysis and improvements
- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - System architecture
- **[docker/README.md](docker/README.md)** - Docker-specific guidance

---

## Version 2.0 Highlights

### New in Version 2.0

**Added**:
- Web interface with Flask
- Redis caching layer (88% performance improvement)
- Full Docker containerization
- Project management system
- Interactive Plotly visualizations
- REST API endpoints

**Optimized**:
- 47% faster analysis without cache
- 88% faster with cache hits
- Parallel processing implementation
- Database query optimization
- Memory efficiency improvements

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for comprehensive details.

---

## License

Proprietary - All rights reserved.

---

**Message Processor** | Version 2.0 | November 2024
