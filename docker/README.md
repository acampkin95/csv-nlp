# Docker Deployment Guide

This directory contains Docker configuration for the Message Processor system.

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Start all services (PostgreSQL + Redis + Web App)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### 2. Access the Application

- **Web Application**: http://localhost:5000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 3. Optional Management Tools

Start with management tools (pgAdmin & Redis Commander):

```bash
docker-compose --profile tools up -d
```

- **pgAdmin**: http://localhost:5050 (admin@msgprocessor.local / admin)
- **Redis Commander**: http://localhost:8081

## Service Architecture

```
┌─────────────────┐
│   Web Browser   │
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Flask WebApp   │────>│  PostgreSQL  │     │    Redis     │
│   (Port 5000)   │     │  (Port 5432) │     │ (Port 6379)  │
└─────────────────┘     └──────────────┘     └──────────────┘
         │
         └─> Uploads, Results, Logs (Volumes)
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
SECRET_KEY=your_secure_random_key
POSTGRES_PASSWORD=your_secure_db_password
```

## Development Mode

For development with hot-reload:

```bash
# Set Flask environment to development
export FLASK_ENV=development

# Run with docker-compose in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Production Deployment

### 1. Build Production Image

```bash
docker build -t message-processor:latest .
```

### 2. Security Hardening

- Change all default passwords in `.env`
- Set strong `SECRET_KEY`
- Use environment-specific configurations
- Enable SSL/TLS for production
- Configure firewall rules
- Use secrets management (Docker Swarm secrets, Kubernetes secrets, etc.)

### 3. Scaling

Scale the web application:

```bash
docker-compose up -d --scale webapp=3
```

## Data Persistence

All data is stored in named Docker volumes:

- `postgres_data`: PostgreSQL database
- `redis_data`: Redis cache
- `pgadmin_data`: pgAdmin configuration

### Backup

```bash
# Backup PostgreSQL
docker exec msgproc-postgres pg_dump -U msgprocess messagestore > backup.sql

# Backup Redis
docker exec msgproc-redis redis-cli SAVE
docker cp msgproc-redis:/data/dump.rdb ./redis-backup.rdb
```

### Restore

```bash
# Restore PostgreSQL
docker exec -i msgproc-postgres psql -U msgprocess messagestore < backup.sql

# Restore Redis
docker cp ./redis-backup.rdb msgproc-redis:/data/dump.rdb
docker-compose restart redis
```

## Monitoring

### View Container Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f webapp
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Health Checks

```bash
# Check service health
docker-compose ps

# Inspect specific container
docker inspect msgproc-webapp
```

### Resource Usage

```bash
# View resource usage
docker stats
```

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Connect to PostgreSQL directly
docker exec -it msgproc-postgres psql -U msgprocess -d messagestore
```

### Redis Connection Issues

```bash
# Test Redis connection
docker exec -it msgproc-redis redis-cli ping

# Monitor Redis
docker exec -it msgproc-redis redis-cli monitor
```

### Web Application Issues

```bash
# Check webapp logs
docker-compose logs webapp

# Restart webapp
docker-compose restart webapp

# Rebuild webapp
docker-compose up -d --build webapp
```

### Clear All Data

```bash
# WARNING: This will delete all data
docker-compose down -v
docker-compose up -d
```

## Network Configuration

The services communicate over a dedicated bridge network (`msgproc-network`).

Internal DNS resolution:
- `postgres` -> PostgreSQL container
- `redis` -> Redis container
- `webapp` -> Web application container

## Volume Mounts

Local directories mounted in containers:

- `./uploads` -> `/app/uploads` (CSV uploads)
- `./results` -> `/app/results` (Analysis results)
- `./logs` -> `/app/logs` (Application logs)

## Security Best Practices

1. **Never commit `.env` file to version control**
2. Use strong, unique passwords
3. Keep Docker images updated
4. Limit container privileges
5. Use read-only file systems where possible
6. Enable Docker Content Trust
7. Scan images for vulnerabilities
8. Use non-root users in containers

## Performance Tuning

### PostgreSQL

Edit `docker-compose.yml` to add PostgreSQL tuning:

```yaml
postgres:
  command:
    - postgres
    - -c
    - shared_buffers=256MB
    - -c
    - max_connections=100
```

### Redis

Adjust Redis memory limit in `docker-compose.yml`:

```yaml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Web Application

Scale workers:

```yaml
webapp:
  environment:
    MAX_WORKERS: 8
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Image](https://hub.docker.com/_/postgres)
- [Redis Docker Image](https://hub.docker.com/_/redis)
