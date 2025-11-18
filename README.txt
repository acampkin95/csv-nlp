================================================================================
CSV-NLP: MESSAGE PROCESSOR - DEPLOYMENT & USAGE GUIDE
================================================================================

Project:      CSV-NLP Message Processor with Person Interaction Analysis
Version:      3.0 (Integrated)
Repository:   https://github.com/acampkin95/csv-nlp
Author:       Alex Campkin
License:      Proprietary - For authorized use only

================================================================================
QUICK START
================================================================================

1. INSTALLATION

   Docker (Recommended):
   ---------------------
   docker-compose up -d
   open http://localhost:5000

   Manual Installation:
   -------------------
   pip install -r requirements.txt
   cp .env.template .env
   # Edit .env with your database credentials
   python webapp.py

2. BASIC USAGE

   Web Interface:
   -------------
   http://localhost:5000
   - Upload CSV files
   - View analysis results
   - Manage person profiles
   - Track interactions

   Command Line:
   ------------
   # Standard 10-pass analysis
   python message_processor.py input.csv

   # New 15-pass unified analysis
   python message_processor.py input.csv --unified

   # Use SQLite instead of PostgreSQL
   python message_processor.py input.csv --use-sqlite

================================================================================
SYSTEM REQUIREMENTS
================================================================================

Software:
- Python 3.11+
- PostgreSQL 15+ (or SQLite for local)
- Redis 7+ (optional but recommended)
- Docker & Docker Compose (for containerized deployment)

Hardware (Recommended):
- CPU: 4+ cores
- RAM: 8GB+ (16GB for large datasets)
- Storage: 20GB+ available

================================================================================
CORE FEATURES
================================================================================

Analysis Capabilities:
- 100+ Behavioral Patterns (grooming, manipulation, deception)
- Multi-Engine Sentiment Analysis (VADER, TextBlob, NRCLex)
- 4-Component Risk Assessment (Low, Moderate, High, Critical)
- Person Identification & Profiling
- Interaction Timeline Analysis
- Relationship Network Mapping
- Gaslighting Detection (5-category framework)
- Clinical Intervention Recommendations

Processing:
- 10,000 messages in <60 seconds
- 88% faster with Redis caching
- Supports 1M+ message datasets
- Parallel processing support

Storage:
- PostgreSQL with JSONB optimization
- Redis caching (70-80% hit rate)
- Complete audit trail
- Dedicated tables per CSV import

================================================================================
DATABASE CONFIGURATION
================================================================================

PostgreSQL (Production):
------------------------
Host:     acdev.host
Port:     5432
Database: messagestore
User:     msgprocess
Password: [See .env file]
Schema:   message_processor

Configure in .env:
------------------
DB_HOST=acdev.host
DB_PORT=5432
DB_NAME=messagestore
DB_USER=msgprocess
DB_PASSWORD=your_password_here
DB_SCHEMA=message_processor

Redis Configuration:
-------------------
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
ENABLE_REDIS_CACHE=true
CACHE_TTL_HOURS=24

================================================================================
ANALYSIS PIPELINES
================================================================================

10-Pass Pipeline (Original):
----------------------------
Pass 1:  Data normalization
Pass 2:  Sentiment analysis (VADER)
Pass 3:  Sentiment analysis (TextBlob)
Pass 4:  Grooming pattern detection
Pass 5:  Manipulation detection
Pass 6:  Deception analysis
Pass 7:  Intent classification
Pass 8:  Behavioral risk scoring
Pass 9:  Timeline reconstruction
Pass 10: Insights generation

15-Pass Pipeline (NEW - Unified):
---------------------------------
Pass 1-10:  All original passes (above)
Pass 11:    Person identification & role classification
Pass 12:    Interaction mapping & network analysis
Pass 13:    Gaslighting detection (5 categories)
Pass 14:    Relationship dynamics & power analysis
Pass 15:    Clinical intervention recommendations

Usage:
------
python message_processor.py input.csv --unified

================================================================================
API ENDPOINTS
================================================================================

Web Application:
---------------
GET  /                          - Home page
POST /upload                    - Upload CSV file
GET  /api/projects              - List all projects
GET  /api/analysis/{id}         - Get analysis results
GET  /api/export/{id}           - Export results
GET  /api/visualizations/{id}   - Get charts

Person Management (NEW):
-----------------------
GET    /api/persons              - List all persons
POST   /api/persons              - Create person profile
GET    /api/persons/{id}         - Get person details
PUT    /api/persons/{id}         - Update person
DELETE /api/persons/{id}         - Delete person

Interaction Analysis (NEW):
---------------------------
POST /api/interactions           - Record interaction
GET  /api/interactions/{id}      - Get interaction details
GET  /api/timeline/{id1}/{id2}   - Get relationship timeline
GET  /api/risk-assessment/{id}   - Get risk assessment

Health & Monitoring:
-------------------
GET  /api/health                 - API health check
GET  /api/stats                  - API statistics

================================================================================
WEB INTERFACE FEATURES
================================================================================

Main Dashboard:
- CSV upload interface
- Project management
- Quick statistics
- Recent analysis results

Person Management:
- Create/edit person profiles
- Search and filter persons
- Risk dashboard
- Behavioral profile viewer

Interaction Analysis:
- Timeline visualization (Plotly.js)
- Relationship network graph (D3.js)
- Risk progression charts
- Intervention recommendations

Analysis Results:
- Risk score visualization
- Pattern detection results
- Timeline charts
- Sentiment analysis graphs
- Export to JSON/CSV

================================================================================
COMMAND LINE OPTIONS
================================================================================

message_processor.py Options:
-----------------------------
positional arguments:
  csv_file              Path to CSV file

optional arguments:
  -h, --help            Show help message
  -c, --config CONFIG   Configuration preset
                        (quick_analysis, deep_analysis, clinical_report, legal_report)
  -o, --output DIR      Output directory for reports
  --use-sqlite          Use SQLite instead of PostgreSQL
  --unified             Use 15-pass unified pipeline (NEW)
  -v, --verbose         Verbose output

Examples:
---------
# Quick analysis
python message_processor.py data.csv -c quick_analysis

# Deep analysis with unified pipeline
python message_processor.py data.csv -c deep_analysis --unified

# Clinical report to specific directory
python message_processor.py data.csv -c clinical_report -o /path/to/reports/

# Use local SQLite database
python message_processor.py data.csv --use-sqlite

================================================================================
CONFIGURATION PRESETS
================================================================================

quick_analysis:
- Fast processing
- Basic sentiment analysis
- Core pattern detection
- Suitable for initial screening

deep_analysis (RECOMMENDED):
- All analysis modules enabled
- Comprehensive pattern detection
- Full risk assessment
- Detailed recommendations

clinical_report:
- Clinical documentation format
- HIPAA-ready structure
- Evidence preservation
- Professional terminology

legal_report:
- Evidence chain documentation
- Timestamped analysis
- Forensic detail level
- Court-ready format

Custom Configuration:
--------------------
Edit config/default.json to customize:
- Feature toggles
- Risk weight adjustments
- Processing parameters
- Output preferences

================================================================================
DOCKER DEPLOYMENT
================================================================================

Services:
---------
- webapp         Flask application (port 5000)
- postgres       PostgreSQL 15 database (port 5432)
- redis          Redis 7 cache (port 6379)
- pgadmin        Database management (port 5050) [optional]
- redis-commander Cache viewer (port 8081) [optional]

Commands:
---------
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f webapp

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Reset database
docker-compose down -v
docker-compose up -d

Access:
-------
Web App:     http://localhost:5000
PgAdmin:     http://localhost:5050
Redis UI:    http://localhost:8081

================================================================================
PERFORMANCE OPTIMIZATION
================================================================================

Redis Caching:
- Enable in .env: ENABLE_REDIS_CACHE=true
- 88% performance improvement
- 70-80% cache hit rate
- Person profiles: 1 hour TTL
- Interactions: 2 hours TTL
- Risk assessments: 1 hour TTL

Database:
- Connection pooling (default: 10 connections)
- JSONB indexes on analysis fields
- Materialized views for reporting
- Dedicated tables per CSV import

Processing:
- Parallel processing for large datasets
- Batch processing for 100k+ messages
- Memory-efficient streaming for large files

Configuration:
--------------
CONNECTION_POOL_SIZE=10    # PostgreSQL connections
WORKERS=4                  # Flask workers
MAX_UPLOAD_SIZE_MB=100     # Max CSV file size

================================================================================
SECURITY & PRIVACY
================================================================================

Data Protection:
- Local processing (no external APIs)
- PostgreSQL access controls
- Complete audit trail
- Data integrity verification

Security Features:
- XSS prevention with HTML escaping
- Parameterized SQL queries
- Input validation on all endpoints
- CSRF protection
- Rate limiting (configurable)

Production Recommendations:
- Enable HTTPS/SSL
- Configure authentication (JWT/OAuth2)
- Set up firewall rules
- Regular security updates
- Database encryption at rest

================================================================================
TESTING
================================================================================

Run Tests:
----------
# All tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_analysis_engine.py

Test Data:
----------
- Sample CSVs in CSV/ directory
- Test fixtures in tests/fixtures/
- Mock data generators included

================================================================================
TROUBLESHOOTING
================================================================================

Common Issues:

1. Database Connection Failed
   - Check PostgreSQL is running: pg_isready -h acdev.host
   - Verify credentials in .env file
   - Check firewall allows port 5432
   - Alternative: Use --use-sqlite flag

2. Redis Connection Error
   - Check Redis is running: redis-cli ping
   - Disable Redis: Set ENABLE_REDIS_CACHE=false in .env
   - Check Redis port 6379 is open

3. Out of Memory
   - Process large files in batches
   - Increase Docker memory limit
   - Use quick_analysis preset
   - Reduce CONNECTION_POOL_SIZE

4. Slow Processing
   - Enable Redis caching
   - Use parallel processing
   - Check database indexes
   - Monitor with /api/stats endpoint

5. Upload Fails
   - Check MAX_UPLOAD_SIZE_MB setting
   - Verify CSV format (UTF-8 encoding)
   - Check disk space
   - Review logs: docker-compose logs webapp

Logs Location:
--------------
Application: logs/app.log
Docker: docker-compose logs
Flask: Check console output

Debug Mode:
-----------
Set in .env:
FLASK_DEBUG=True
LOG_LEVEL=DEBUG

================================================================================
OUTPUT FILES
================================================================================

Analysis Results:
-----------------
Reports/
├── analysis_{timestamp}.json     # Complete analysis data
├── summary_{timestamp}.csv       # Statistical summary
├── timeline_{timestamp}.csv      # Conversation timeline
├── patterns_{timestamp}.json     # Detected patterns
├── risk_assessment_{timestamp}.json
└── recommendations_{timestamp}.txt

Database Storage:
-----------------
All results also stored in PostgreSQL:
- messages_master table
- detected_patterns table
- analysis_runs table
- person_interactions table (NEW)
- relationship_timelines table (NEW)

================================================================================
PROJECT STRUCTURE
================================================================================

csv-nlp/
├── webapp.py                 # Flask web application
├── message_processor.py      # Command-line processor
├── docker-compose.yml        # Docker deployment
├── requirements.txt          # Python dependencies
├── .env.template            # Environment template
│
├── src/
│   ├── api/                 # REST API endpoints (NEW)
│   │   └── unified_api.py   # Person management API
│   ├── db/                  # Database adapters
│   │   ├── postgresql_adapter.py
│   │   └── postgresql_schema.sql
│   ├── cache/               # Redis caching
│   ├── nlp/                 # Analysis modules
│   │   ├── grooming_detector.py
│   │   ├── manipulation_detector.py
│   │   ├── deception_analyzer.py
│   │   ├── person_analyzer.py (NEW)
│   │   └── ...
│   ├── pipeline/            # Processing pipelines
│   │   └── unified_processor.py (NEW)
│   └── validation/          # Input validation
│
├── templates/               # Web UI templates
│   ├── index.html
│   ├── persons.html        # Person management (NEW)
│   └── interactions.html   # Timeline viewer (NEW)
│
├── static/                  # CSS/JavaScript
│   ├── css/
│   │   └── persons.css     # Person UI styles (NEW)
│   └── js/
│       └── person_manager.js (NEW)
│
├── CSV/                     # Input CSV files
├── Reports/                 # Output reports
└── tests/                   # Test suite

================================================================================
DOCUMENTATION
================================================================================

All detailed documentation moved to:
/Users/alex/Projects/Dev/Projects/Message Processor/Data Store/Documentation/

Key Documents:
- API_DOCUMENTATION.md              - Complete API reference
- ARCHITECTURE.md                   - Technical architecture
- PIPELINE_DOCUMENTATION.md         - 15-pass pipeline details
- PERSON_MANAGEMENT_UI.md           - Frontend guide
- INTEGRATION_VERIFICATION_REPORT.md - Integration summary
- DEPLOYMENT_GUIDE.md               - Production deployment
- TESTING_GUIDE.md                  - Testing procedures
- QUICKSTART.md                     - Quick start guide

================================================================================
SUPPORT & RESOURCES
================================================================================

Repository:  https://github.com/acampkin95/csv-nlp
Issues:      https://github.com/acampkin95/csv-nlp/issues
Author:      Alex Campkin (https://github.com/acampkin95)

Crisis Resources (Auto-suggested by system):
- National Suicide Prevention: 988
- Crisis Text Line: Text HOME to 741741
- Domestic Violence Hotline: 1-800-799-7233
- RAINN Sexual Assault: 1-800-656-4673

================================================================================
VERSION HISTORY
================================================================================

v3.0 (Current) - November 2024
- Integrated ppl_int person management features
- Added 15-pass unified analysis pipeline
- Implemented person CRUD API (13 endpoints)
- Added interactive web UI for person management
- Gaslighting detection framework
- Relationship network visualization
- 14,331 lines of code/documentation added

v2.0 - October 2024
- PostgreSQL integration
- Redis caching (88% performance improvement)
- Web application with Flask
- Docker containerization
- 10-pass analysis pipeline

v1.0 - September 2024
- Initial prototype (632 lines)
- Basic CSV processing
- SQLite database
- Command-line interface

================================================================================
LICENSE & DISCLAIMER
================================================================================

License: Proprietary - For authorized use only

DISCLAIMER:
This system is for analysis purposes and should not replace professional
judgment. Always consult qualified professionals for clinical or legal
decisions. The analysis results are computational assessments and should
be used as supporting evidence, not definitive conclusions.

The system does not:
- Replace clinical psychological assessment
- Substitute for legal advice
- Guarantee accuracy in all cases
- Store or transmit data externally (local processing only)

================================================================================
END OF README
================================================================================

For detailed technical documentation, see:
/Users/alex/Projects/Dev/Projects/Message Processor/Data Store/Documentation/

Last Updated: November 2024
