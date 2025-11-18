# Message Processor - Deployment Guide

Complete deployment guide for the Message Processor system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB available space

### Software Requirements

#### Option 1: Docker Deployment (Recommended)

- Docker Engine 20.10+
- Docker Compose 2.0+

#### Option 2: Local Development

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Git

---

## Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd "Message Processor/Dev-Root"

# 2. Copy environment configuration
cp .env.example .env

# 3. Edit .env with your settings
nano .env  # or your preferred editor

# 4. Start all services
docker-compose up -d

# 5. Check service status
docker-compose ps

# 6. Access application
open http://localhost:5000
```

The application will be available at:
- **Web Interface**: http://localhost:5000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

---

## Development Setup

### Local Python Environment

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')"

# 4. Set up PostgreSQL database
createdb messagestore
psql messagestore < src/db/postgresql_schema.sql

# 5. Start Redis (in separate terminal)
redis-server

# 6. Configure environment
export POSTGRES_HOST=localhost
export POSTGRES_DB=messagestore
export POSTGRES_USER=msgprocess
export POSTGRES_PASSWORD=your_password
export REDIS_HOST=localhost
export FLASK_ENV=development

# 7. Run application
python webapp.py
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python
- Pylance
- Docker
- PostgreSQL

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Set Python interpreter to virtual environment
2. Enable "Django support" for template editing
3. Configure database connection to PostgreSQL
4. Set up run configurations for webapp.py

---

## Production Deployment

### Option 1: Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml msgproc

# Scale services
docker service scale msgproc_webapp=3

# View services
docker service ls

# View logs
docker service logs -f msgproc_webapp
```

### Option 2: Kubernetes

```bash
# Create namespace
kubectl create namespace msgproc

# Apply configurations
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment webapp --replicas=3 -n msgproc

# View status
kubectl get all -n msgproc
```

### Option 3: Traditional Server

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv postgresql-15 redis-server nginx

# Create application directory
sudo mkdir -p /opt/messageprocessor
sudo chown $USER:$USER /opt/messageprocessor
cd /opt/messageprocessor

# Clone and setup
git clone <repository-url> .
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure systemd service
sudo cp deployment/systemd/messageprocessor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable messageprocessor
sudo systemctl start messageprocessor

# Configure nginx reverse proxy
sudo cp deployment/nginx/messageprocessor.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/messageprocessor.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Configuration

### Environment Variables

Create `.env` file with the following variables:

```bash
# Flask Configuration
FLASK_APP=webapp.py
FLASK_ENV=production
SECRET_KEY=<generate_secure_random_key>

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=messagestore
POSTGRES_USER=msgprocess
POSTGRES_PASSWORD=<secure_password>

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<optional_password>

# Application Settings
MAX_WORKERS=4
UPLOAD_MAX_SIZE=52428800  # 50MB
LOG_LEVEL=INFO

# Security (Production)
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
PERMANENT_SESSION_LIFETIME=3600
```

### Database Configuration

#### PostgreSQL Tuning

Edit `postgresql.conf`:

```
# Connection Settings
max_connections = 100

# Memory Settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 16MB

# Query Planning
random_page_cost = 1.1  # For SSD

# Write Performance
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

#### Redis Tuning

Edit `redis.conf`:

```
# Memory Management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec

# Performance
tcp-backlog 511
```

### Application Configuration

Configuration presets in `src/config/`:

- `quick_analysis`: Fast analysis with basic features
- `deep_analysis`: Comprehensive analysis (default)
- `clinical_report`: Detailed clinical assessment
- `legal_report`: Forensic-quality documentation

---

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_sentiment.py

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark
```

### Manual Testing

```bash
# Test API endpoints
curl -X POST http://localhost:5000/api/upload \
  -F "file=@test.csv" \
  -F "project_name=Test Project"

# Test analysis
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"csv_session_id": "abc-123", "config": "quick_analysis"}'

# Check cache stats
curl http://localhost:5000/api/cache/stats
```

---

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:5000/health

# PostgreSQL
docker exec msgproc-postgres pg_isready -U msgprocess

# Redis
docker exec msgproc-redis redis-cli ping
```

### Logging

```bash
# Application logs
docker-compose logs -f webapp

# PostgreSQL logs
docker-compose logs -f postgres

# Redis logs
docker-compose logs -f redis

# All services
docker-compose logs -f
```

### Performance Monitoring

#### Prometheus + Grafana Setup

```bash
# Start monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000

# Default credentials: admin/admin
```

#### Built-in Monitoring

```bash
# Cache statistics
curl http://localhost:5000/api/cache/stats

# Database statistics
docker exec msgproc-postgres psql -U msgprocess -d messagestore \
  -c "SELECT * FROM pg_stat_database WHERE datname='messagestore';"

# Container resource usage
docker stats
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed

**Symptom**: "Could not connect to PostgreSQL"

**Solutions**:
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection settings
docker-compose exec postgres psql -U msgprocess -d messagestore

# Restart PostgreSQL
docker-compose restart postgres
```

#### 2. Redis Connection Failed

**Symptom**: "Redis connection refused"

**Solutions**:
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping

# Clear cache if corrupted
docker-compose exec redis redis-cli FLUSHDB
```

#### 3. Upload Fails

**Symptom**: "File upload error"

**Solutions**:
```bash
# Check file permissions
chmod 755 uploads/

# Check disk space
df -h

# Increase upload size limit in .env
UPLOAD_MAX_SIZE=104857600  # 100MB
```

#### 4. Slow Analysis

**Symptom**: Analysis takes very long

**Solutions**:
```bash
# Increase workers
export MAX_WORKERS=8

# Enable caching
# Check Redis is available

# Use quick_analysis preset
curl -X POST .../api/analyze -d '{"config": "quick_analysis"}'
```

### Debug Mode

Enable debug mode for detailed error messages:

```bash
# In .env
FLASK_ENV=development
LOG_LEVEL=DEBUG

# Restart application
docker-compose restart webapp
```

### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove all data (WARNING: destructive)
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build
```

---

## Backup and Restore

### Database Backup

```bash
# Create backup
docker exec msgproc-postgres pg_dump -U msgprocess messagestore > backup_$(date +%Y%m%d).sql

# Automated daily backups
echo "0 2 * * * docker exec msgproc-postgres pg_dump -U msgprocess messagestore > /backups/db_\$(date +\%Y\%m\%d).sql" | crontab -
```

### Restore Database

```bash
# Restore from backup
docker exec -i msgproc-postgres psql -U msgprocess messagestore < backup_20241118.sql
```

### Redis Backup

```bash
# Manual save
docker exec msgproc-redis redis-cli SAVE

# Copy RDB file
docker cp msgproc-redis:/data/dump.rdb ./redis_backup.rdb
```

---

## Security Hardening

### Production Checklist

- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up fail2ban
- [ ] Enable database SSL connections
- [ ] Implement rate limiting
- [ ] Set up security monitoring
- [ ] Regular security updates
- [ ] Backup encryption

### SSL/TLS Setup

```bash
# Generate certificate (or use Let's Encrypt)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Update nginx configuration
# Add SSL configuration to nginx conf
```

---

## Support

For issues and questions:

1. Check this guide and `OPTIMIZATION_REPORT.md`
2. Review logs: `docker-compose logs -f`
3. Check documentation in `docs/`
4. Contact system administrator

---

**Last Updated**: November 18, 2024
**Version**: 2.0
