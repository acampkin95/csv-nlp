# Immediate Security Fixes Required
**Priority**: CRITICAL - Implement before next deployment
**Estimated Time**: 3.5 hours
**Risk Level**: Production deployment unsafe without these fixes

---

## 🔴 Critical Fix #1: Remove Hardcoded Credentials (2 hours)

### Files to Modify:
1. `src/db/postgresql_adapter.py`
2. `message_processor.py`
3. `webapp.py`

### Implementation:

#### Step 1: Create `.env` file (DO NOT COMMIT)
```bash
# .env (add to .gitignore!)
POSTGRES_HOST=acdev.host
POSTGRES_DB=messagestore
POSTGRES_USER=msgprocess
POSTGRES_PASSWORD=<GENERATE_NEW_STRONG_PASSWORD>
SECRET_KEY=<GENERATE_NEW_SECRET_KEY>
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Step 2: Generate Strong Secrets
```bash
# Generate new database password
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate new Flask secret key
python3 -c "import secrets; print(secrets.token_hex(32))"
```

#### Step 3: Update `src/db/postgresql_adapter.py`
```python
# BEFORE (Lines 22-30):
@dataclass
class DatabaseConfig:
    host: str = "acdev.host"
    port: int = 5432
    database: str = "messagestore"
    user: str = "msgprocess"
    password: str = "DHifde93jes9dk"  # ❌ REMOVE THIS
    schema: str = "message_processor"
    min_connections: int = 2
    max_connections: int = 10

# AFTER:
import os
from dataclasses import dataclass, field

@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.environ['POSTGRES_HOST'])
    port: int = field(default_factory=lambda: int(os.environ.get('POSTGRES_PORT', '5432')))
    database: str = field(default_factory=lambda: os.environ['POSTGRES_DB'])
    user: str = field(default_factory=lambda: os.environ['POSTGRES_USER'])
    password: str = field(default_factory=lambda: os.environ['POSTGRES_PASSWORD'])
    schema: str = field(default_factory=lambda: os.environ.get('POSTGRES_SCHEMA', 'message_processor'))
    min_connections: int = field(default_factory=lambda: int(os.environ.get('POSTGRES_MIN_CONN', '2')))
    max_connections: int = field(default_factory=lambda: int(os.environ.get('POSTGRES_MAX_CONN', '10')))
```

#### Step 4: Update `webapp.py`
```python
# BEFORE (Line 41):
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')

# AFTER:
# Validate required environment variables at startup
required_env_vars = ['SECRET_KEY', 'POSTGRES_PASSWORD', 'POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

app.secret_key = os.environ['SECRET_KEY']

# BEFORE (Lines 56-61):
db_config = DatabaseConfig(
    host=os.environ.get('POSTGRES_HOST', 'acdev.host'),
    database=os.environ.get('POSTGRES_DB', 'messagestore'),
    user=os.environ.get('POSTGRES_USER', 'msgprocess'),
    password=os.environ.get('POSTGRES_PASSWORD', 'DHifde93jes9dk')  # ❌ REMOVE FALLBACK
)

# AFTER:
db_config = DatabaseConfig()  # Uses environment variables from dataclass defaults
```

#### Step 5: Update `message_processor.py`
```python
# Remove hardcoded credentials on lines 54 and 117
# BEFORE:
db_config = DatabaseConfig(
    host="acdev.host",
    database="messagestore",
    user="msgprocess",
    password="DHifde93jes9dk"  # ❌ REMOVE
)

# AFTER:
db_config = DatabaseConfig()  # Uses environment variables
```

#### Step 6: Update `.gitignore`
```bash
# Add to .gitignore if not already present
.env
.env.local
.env.*.local
*.env
```

#### Step 7: Create `.env.example`
```bash
# .env.example (safe to commit)
# Copy this file to .env and fill in values
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name
POSTGRES_USER=your_database_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_SCHEMA=message_processor
SECRET_KEY=your_secret_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Step 8: Update Documentation
Add to README.md:
```markdown
## Environment Variables

The application requires the following environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_HOST` | PostgreSQL server hostname | Yes |
| `POSTGRES_PORT` | PostgreSQL server port | No (default: 5432) |
| `POSTGRES_DB` | Database name | Yes |
| `POSTGRES_USER` | Database username | Yes |
| `POSTGRES_PASSWORD` | Database password | Yes |
| `SECRET_KEY` | Flask secret key for sessions | Yes |
| `REDIS_HOST` | Redis server hostname | No (default: localhost) |
| `REDIS_PORT` | Redis server port | No (default: 6379) |

### Setup

1. Copy `.env.example` to `.env`
2. Fill in all required values
3. NEVER commit `.env` to version control
```

---

## 🔴 Critical Fix #2: SQL Injection in Timeline Aggregation (30 minutes)

### File: `src/db/postgresql_adapter.py`

### Implementation:

```python
# BEFORE (Lines 611-651):
def create_timeline_aggregation(self, csv_session_id: str, window_size: str = 'day'):
    trunc_map = {
        'hour': 'hour',
        'day': 'day',
        'week': 'week',
        'month': 'month'
    }

    trunc = trunc_map.get(window_size, 'day')

    cursor.execute(f"""
        INSERT INTO timeline_aggregations (csv_session_id, window_start, window_end, window_size, metrics)
        SELECT
            %s as csv_session_id,
            date_trunc('{trunc}', timestamp) as window_start,
            date_trunc('{trunc}', timestamp) + interval '1 {window_size}' as window_end,  # ❌ VULNERABLE
            %s as window_size,
            ...
    """, (csv_session_id, window_size, csv_session_id))

# AFTER:
def create_timeline_aggregation(self, csv_session_id: str, window_size: str = 'day'):
    """Create timeline aggregations for performance

    Args:
        csv_session_id: CSV session ID
        window_size: Aggregation window (hour, day, week, month)

    Raises:
        ValueError: If window_size is not valid
    """
    # ✅ SECURE: Whitelist validation
    valid_windows = {'hour', 'day', 'week', 'month'}
    if window_size not in valid_windows:
        raise ValueError(
            f"Invalid window_size: '{window_size}'. "
            f"Must be one of: {', '.join(sorted(valid_windows))}"
        )

    # Safe to use after validation
    trunc_map = {
        'hour': 'hour',
        'day': 'day',
        'week': 'week',
        'month': 'month'
    }

    trunc = trunc_map[window_size]

    with self.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO timeline_aggregations (csv_session_id, window_start, window_end, window_size, metrics)
                SELECT
                    %s as csv_session_id,
                    date_trunc('{trunc}', timestamp) as window_start,
                    date_trunc('{trunc}', timestamp) + interval '1 {window_size}' as window_end,
                    %s as window_size,
                    jsonb_build_object(
                        'message_count', COUNT(*),
                        'unique_speakers', COUNT(DISTINCT sender_name),
                        'sentiment', jsonb_build_object(
                            'mean', AVG((sentiment_analysis->>'compound')::numeric),
                            'min', MIN((sentiment_analysis->>'compound')::numeric),
                            'max', MAX((sentiment_analysis->>'compound')::numeric),
                            'variance', VAR_POP((sentiment_analysis->>'compound')::numeric)
                        ),
                        'risk_events', COUNT(*) FILTER (WHERE (risk_analysis->>'risk_level')::text IN ('high', 'critical'))
                    ) as metrics
                FROM messages_master
                WHERE csv_session_id = %s
                GROUP BY date_trunc('{trunc}', timestamp)
            """, (csv_session_id, window_size, csv_session_id))

            conn.commit()
```

### Add Unit Test:
```python
# tests/test_security.py
import pytest
from src.db.postgresql_adapter import PostgreSQLAdapter

def test_timeline_aggregation_sql_injection_prevention():
    """Ensure SQL injection attempts in window_size are blocked"""
    adapter = PostgreSQLAdapter()

    # Test malicious input
    with pytest.raises(ValueError, match="Invalid window_size"):
        adapter.create_timeline_aggregation(
            csv_session_id="test-id",
            window_size="day'; DROP TABLE messages_master; --"
        )

    # Test valid inputs
    valid_windows = ['hour', 'day', 'week', 'month']
    for window in valid_windows:
        # Should not raise
        try:
            adapter.create_timeline_aggregation("test-id", window)
        except ValueError as e:
            if "Invalid window_size" in str(e):
                pytest.fail(f"Valid window_size '{window}' was rejected")
```

---

## 🔴 Critical Fix #3: Generic Error Messages (1 hour)

### File: `webapp.py`

### Implementation:

```python
# Add at top of file
import logging

# Create separate loggers
app_logger = logging.getLogger('app')
security_logger = logging.getLogger('security')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Add security log file
security_handler = logging.FileHandler('security.log')
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | IP=%(client_ip)s | %(message)s'
))
security_logger.addHandler(security_handler)

# Create error response helper
def safe_error_response(error: Exception, user_message: str, status_code: int = 500):
    """Return generic error to client, log details internally

    Args:
        error: Original exception
        user_message: Generic message for user
        status_code: HTTP status code

    Returns:
        JSON response with generic error
    """
    # Log full details (for debugging)
    app_logger.error(f"{user_message}: {str(error)}", exc_info=True, extra={
        'client_ip': request.remote_addr,
        'endpoint': request.endpoint,
        'method': request.method
    })

    # Return generic message to client
    return jsonify({
        'error': user_message,
        'status': status_code
    }), status_code

# UPDATE ALL EXCEPTION HANDLERS:

# BEFORE (Line 290-292):
except Exception as e:
    logger.error(f"Upload error: {e}")
    return jsonify({'error': str(e)}), 500

# AFTER:
except FileNotFoundError as e:
    return safe_error_response(e, 'File not found', 404)
except ValueError as e:
    return safe_error_response(e, 'Invalid file format or content', 400)
except psycopg2.Error as e:
    return safe_error_response(e, 'Database operation failed', 500)
except Exception as e:
    return safe_error_response(e, 'Upload failed', 500)

# Apply same pattern to all endpoints:
# - /api/analyze (lines 339-341)
# - /api/analysis/<id>/results (lines 382-383)
# - /api/analysis/<id>/timeline (lines 408-409)
# - /api/visualizations/<id>/sentiment (lines 438-439)
# - /api/export/pdf/<id> (lines 457-458)
# - /api/export/json/<id> (lines 485-486)
# - /api/cache/stats (lines 496-497)

# UPDATE FLASK ERROR HANDLERS:

@app.errorhandler(404)
def not_found(error):
    app_logger.warning(f"404 error: {request.url}", extra={'client_ip': request.remote_addr})
    return jsonify({'error': 'Resource not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    app_logger.error(f"500 error: {str(error)}", exc_info=True, extra={'client_ip': request.remote_addr})
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

@app.errorhandler(403)
def forbidden(error):
    security_logger.warning(f"403 Forbidden: {request.url}", extra={'client_ip': request.remote_addr})
    return jsonify({'error': 'Access forbidden', 'status': 403}), 403

@app.errorhandler(401)
def unauthorized(error):
    security_logger.warning(f"401 Unauthorized: {request.url}", extra={'client_ip': request.remote_addr})
    return jsonify({'error': 'Authentication required', 'status': 401}), 401

# Ensure debug mode is NEVER enabled in production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    # CRITICAL: Never use debug in production
    flask_env = os.environ.get('FLASK_ENV', 'production')
    debug = flask_env == 'development'

    if debug:
        app_logger.warning("⚠️  Running in DEBUG mode - NOT suitable for production!")

    app.run(host='0.0.0.0', port=port, debug=debug)
```

### Update `.env.example`:
```bash
FLASK_ENV=production  # Set to 'development' only for local dev
PORT=5000
```

---

## Verification Checklist

After implementing all fixes:

### ✅ Credential Removal Verification
```bash
# Search for hardcoded credentials (should return nothing)
grep -r "DHifde93jes9dk" --exclude-dir=.git --exclude-dir=__pycache__
grep -r "dev_secret_key_change_in_production" --exclude-dir=.git

# Verify .env is in .gitignore
grep "^\.env$" .gitignore

# Verify no .env in git history
git log --all --full-history -- .env
```

### ✅ SQL Injection Fix Verification
```bash
# Run unit tests
pytest tests/test_security.py -v

# Manual test with malicious input
python3 -c "
from src.db.postgresql_adapter import PostgreSQLAdapter
adapter = PostgreSQLAdapter()
try:
    adapter.create_timeline_aggregation('test', \"day'; DROP TABLE messages; --\")
    print('❌ VULNERABLE: SQL injection not blocked')
except ValueError as e:
    print('✅ SECURE: SQL injection blocked')
"
```

### ✅ Error Message Verification
```bash
# Start application
FLASK_ENV=production python webapp.py

# Test endpoint with error condition (in another terminal)
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"csv_session_id": "invalid-id"}'

# Verify response does NOT contain:
# - Internal file paths
# - Database connection details
# - Stack traces
# - Library versions
```

### ✅ Environment Variables Verification
```bash
# Test startup without environment variables (should fail fast)
unset POSTGRES_PASSWORD SECRET_KEY
python webapp.py  # Should exit with error about missing vars

# Test with variables set (should start successfully)
export POSTGRES_PASSWORD="test123"
export SECRET_KEY="testsecret"
export POSTGRES_HOST="localhost"
export POSTGRES_DB="testdb"
export POSTGRES_USER="testuser"
python webapp.py  # Should start without errors
```

---

## Deployment Instructions

### Pre-Deployment
1. ✅ Complete all three critical fixes
2. ✅ Run verification checklist
3. ✅ Update documentation
4. ✅ Create `.env.example`
5. ✅ Add `.env` to `.gitignore`
6. ✅ Run dependency security scan: `pip install safety && safety check`

### Production Deployment
```bash
# 1. Generate new production secrets
PROD_DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
PROD_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# 2. Update production .env file
cat > /path/to/production/.env <<EOF
POSTGRES_HOST=your-production-db-host
POSTGRES_PORT=5432
POSTGRES_DB=messagestore_prod
POSTGRES_USER=msgprocess_prod
POSTGRES_PASSWORD=${PROD_DB_PASSWORD}
SECRET_KEY=${PROD_SECRET_KEY}
FLASK_ENV=production
REDIS_HOST=your-redis-host
REDIS_PORT=6379
EOF

# 3. Secure .env file permissions
chmod 600 /path/to/production/.env
chown app-user:app-user /path/to/production/.env

# 4. Update database password
psql -h your-production-db-host -U postgres -c "ALTER USER msgprocess_prod PASSWORD '${PROD_DB_PASSWORD}';"

# 5. Test application startup
cd /path/to/production
source venv/bin/activate
python webapp.py  # Should start without errors

# 6. Verify no debug mode
curl http://localhost:5000/nonexistent  # Should return generic 404, not debug trace
```

### Post-Deployment Verification
```bash
# Check logs for startup errors
tail -f app.log security.log

# Verify secure headers
curl -I http://your-production-url/

# Run security scan
safety check
bandit -r src/ -f json -o bandit_report.json
```

---

## Rollback Plan

If issues occur after deployment:

1. **Immediate**: Restore previous version
```bash
git checkout previous-stable-tag
systemctl restart webapp
```

2. **Database**: No schema changes in security fixes, no rollback needed

3. **Environment Variables**: Keep new secure values, don't revert to hardcoded

4. **Investigate**: Check `app.log` and `security.log` for errors

---

## Next Steps (After Critical Fixes)

1. **Week 1**: Address HIGH severity issues (SQL injection in other locations)
2. **Week 2**: Add CSRF protection (Flask-WTF)
3. **Week 3**: Add rate limiting (Flask-Limiter)
4. **Week 4**: Implement authentication (Flask-Login)
5. **Month 2**: Full security testing (OWASP ZAP, penetration testing)

---

## Support

If you encounter issues during implementation:

1. Check logs: `tail -f app.log security.log`
2. Verify environment variables: `env | grep POSTGRES`
3. Test database connection: `psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB`
4. Contact security team or create issue with:
   - Error message
   - Logs (sanitized)
   - Steps to reproduce

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: READY FOR IMPLEMENTATION
