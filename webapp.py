"""
Message Processor Web Application
Flask-based web interface for CSV upload, project management, and interactive visualizations
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import pandas as pd
import json
import uuid
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.db.postgresql_adapter import PostgreSQLAdapter, DatabaseConfig
from src.validation.csv_validator import CSVValidator
from src.config.config_manager import ConfigManager
from src.cache.redis_cache import RedisCache

# Import main processor
from message_processor import EnhancedMessageProcessor

# Import unified API
from src.api.unified_api import create_api_blueprint

# Import security extensions
try:
    from flask_wtf.csrf import CSRFProtect
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    SECURITY_EXTENSIONS_AVAILABLE = True
except ImportError:
    logger.warning("Security extensions not available. Install: pip install Flask-WTF Flask-Limiter")
    SECURITY_EXTENSIONS_AVAILABLE = False

# Configure logging with separate security log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)
security_logger = logging.getLogger('security')

# Add security log file handler
security_handler = logging.FileHandler('security.log')
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)

# Validate required environment variables at startup
required_env_vars = ['SECRET_KEY', 'POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.critical("Please set these environment variables or create a .env file from .env.example")
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']  # Fail fast if not set
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_SIZE_MB', '10')) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Initialize components
redis_cache = RedisCache(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379))
)

# Use secure environment-based configuration (validated above)
db_config = DatabaseConfig()
db = PostgreSQLAdapter(db_config)
config_manager = ConfigManager()

# Initialize CSRF protection if available
if SECURITY_EXTENSIONS_AVAILABLE and os.environ.get('ENABLE_CSRF', 'true').lower() == 'true':
    csrf = CSRFProtect(app)
    logger.info("CSRF protection enabled")
else:
    csrf = None
    logger.warning("CSRF protection DISABLED - not recommended for production")

# Initialize rate limiting if available
if SECURITY_EXTENSIONS_AVAILABLE and os.environ.get('ENABLE_RATE_LIMITING', 'true').lower() == 'true':
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=f"redis://{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', 6379)}"
    )
    logger.info("Rate limiting enabled")
else:
    limiter = None
    logger.warning("Rate limiting DISABLED - not recommended for production")


# ==========================================
# Security Helper Functions
# ==========================================

def safe_error_response(error: Exception, user_message: str, status_code: int = 500):
    """Return generic error to client, log details internally

    Args:
        error: Original exception
        user_message: Generic message for user
        status_code: HTTP status code

    Returns:
        JSON response with generic error
    """
    # Log full details for debugging
    logger.error(f"{user_message}: {str(error)}", exc_info=True, extra={
        'client_ip': request.remote_addr,
        'endpoint': request.endpoint,
        'method': request.method,
        'url': request.url
    })

    # Log security-relevant errors
    if status_code in [401, 403, 429]:
        security_logger.warning(
            f"{status_code} | {request.remote_addr} | {request.method} {request.path} | {user_message}"
        )

    # Return generic message to client (no internal details)
    return jsonify({
        'error': user_message,
        'status': status_code
    }), status_code


# ==========================================
# Project Management
# ==========================================

class ProjectManager:
    """Manage analysis projects"""

    def __init__(self, db: PostgreSQLAdapter, cache: RedisCache):
        self.db = db
        self.cache = cache
        self.persons = {}  # Store persons in project context

    def create_project(self, name: str, description: str, user_id: str = 'default') -> str:
        """Create a new project

        Args:
            name: Project name
            description: Project description
            user_id: User identifier

        Returns:
            str: Project ID
        """
        project_id = str(uuid.uuid4())
        project_data = {
            'id': project_id,
            'name': name,
            'description': description,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'csv_sessions': [],
            'analysis_runs': [],
            'persons': [],
            'interactions': []
        }

        # Store in cache
        self.cache.create_session(f'project:{project_id}', project_data)

        logger.info(f"Created project: {project_id}")
        return project_id

    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project data"""
        return self.cache.get_session(f'project:{project_id}')

    def add_csv_to_project(self, project_id: str, csv_session_id: str, filename: str):
        """Add CSV session to project"""
        project = self.get_project(project_id)
        if project:
            project['csv_sessions'].append({
                'id': csv_session_id,
                'filename': filename,
                'uploaded_at': datetime.now().isoformat()
            })
            self.cache.create_session(f'project:{project_id}', project)

    def add_analysis_to_project(self, project_id: str, analysis_run_id: str):
        """Add analysis run to project"""
        project = self.get_project(project_id)
        if project:
            project['analysis_runs'].append({
                'id': analysis_run_id,
                'created_at': datetime.now().isoformat()
            })
            self.cache.create_session(f'project:{project_id}', project)

    def add_person_to_project(self, project_id: str, person_id: str, person_data: Dict):
        """Add person to project"""
        project = self.get_project(project_id)
        if project:
            project['persons'].append({
                'id': person_id,
                'data': person_data,
                'added_at': datetime.now().isoformat()
            })
            self.cache.create_session(f'project:{project_id}', project)

    def add_interaction_to_project(self, project_id: str, interaction_id: str, interaction_data: Dict):
        """Add interaction to project"""
        project = self.get_project(project_id)
        if project:
            project['interactions'].append({
                'id': interaction_id,
                'data': interaction_data,
                'recorded_at': datetime.now().isoformat()
            })
            self.cache.create_session(f'project:{project_id}', project)

    def list_projects(self, user_id: str = 'default') -> List[Dict]:
        """List all projects for a user"""
        # For now, get from cache
        # In production, this would query database
        return []


project_manager = ProjectManager(db, redis_cache)

# ==========================================
# Unified API Integration
# ==========================================

# Create and register unified API blueprint
api_blueprint, api_person_manager, api_interaction_tracker, relationship_analyzer, risk_engine = create_api_blueprint(
    db, redis_cache
)
app.register_blueprint(api_blueprint)

logger.info("Registered unified API blueprint with ppl_int features")


# ==========================================
# Routes - Main Pages
# ==========================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard with project overview"""
    user_id = session.get('user_id', 'default')
    projects = project_manager.list_projects(user_id)
    return render_template('dashboard.html', projects=projects)


@app.route('/upload')
def upload_page():
    """CSV upload page"""
    return render_template('upload.html')


@app.route('/project/<project_id>')
def project_view(project_id):
    """View project details"""
    project = project_manager.get_project(project_id)
    if not project:
        return "Project not found", 404

    return render_template('project.html', project=project)


@app.route('/analysis/<analysis_run_id>')
def analysis_view(analysis_run_id):
    """View analysis results"""
    # Get analysis results from database
    with db.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM analysis_runs WHERE id = %s",
                (analysis_run_id,)
            )
            analysis = cursor.fetchone()

    if not analysis:
        return "Analysis not found", 404

    return render_template('analysis.html', analysis=analysis, analysis_id=analysis_run_id)


# ==========================================
# API Endpoints
# ==========================================

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload CSV file with rate limiting and security validation"""
    # Apply rate limiting if available
    if limiter:
        try:
            limiter.check()
        except Exception:
            return safe_error_response(Exception("Rate limit exceeded"), "Too many requests", 429)

    security_logger.info(f"File upload attempt from {request.remote_addr}")

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        security_logger.warning(f"Invalid file type upload attempt from {request.remote_addr}: {file.filename}")
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        # SECURITY: Sanitize filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"

        # SECURITY: Validate upload folder exists and is directory
        upload_folder = Path(app.config['UPLOAD_FOLDER']).resolve()
        if not upload_folder.exists():
            upload_folder.mkdir(parents=True, exist_ok=True)
        if not upload_folder.is_dir():
            return safe_error_response(Exception("Upload folder invalid"), "Upload failed", 500)

        filepath = upload_folder / saved_filename
        file.save(str(filepath))

        # Validate CSV
        validator = CSVValidator()
        validation_result, df = validator.validate_file(str(filepath))

        if not validation_result.is_valid:
            # Clean up invalid file
            filepath.unlink(missing_ok=True)
            return jsonify({
                'error': 'CSV validation failed',
                'details': validation_result.errors
            }), 400

        # Sanitize project name (prevent XSS)
        import bleach
        project_name = bleach.clean(request.form.get('project_name', filename))
        project_id = request.form.get('project_id')

        if not project_id:
            project_id = project_manager.create_project(
                name=project_name,
                description=f"Analysis of {filename}"
            )

        # Import to PostgreSQL
        csv_session_id = db.create_csv_import_session(filename, df)

        # Add to project
        project_manager.add_csv_to_project(project_id, csv_session_id, filename)

        security_logger.info(f"File uploaded successfully from {request.remote_addr}: {saved_filename}")

        return jsonify({
            'success': True,
            'project_id': project_id,
            'csv_session_id': csv_session_id,
            'filename': saved_filename,
            'rows': len(df),
            'columns': len(df.columns)
        })

    except FileNotFoundError as e:
        return safe_error_response(e, 'File not found', 404)
    except ValueError as e:
        return safe_error_response(e, 'Invalid file format or content', 400)
    except psycopg2.Error as e:
        return safe_error_response(e, 'Database operation failed', 500)
    except Exception as e:
        return safe_error_response(e, 'Upload failed', 500)


@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Start analysis on uploaded CSV"""
    data = request.get_json()
    csv_session_id = data.get('csv_session_id')
    project_id = data.get('project_id')
    config_preset = data.get('config', 'deep_analysis')

    if not csv_session_id:
        return jsonify({'error': 'CSV session ID required'}), 400

    try:
        # Load configuration
        config = config_manager.load_config(config_preset)

        # Create processor
        processor = EnhancedMessageProcessor(config, use_postgresql=True)

        # Get messages from database
        messages = db.get_messages(csv_session_id=csv_session_id)

        # Create analysis run
        analysis_run_id = db.create_analysis_run(
            csv_session_id=csv_session_id,
            config=config.to_dict()
        )

        # Add to project
        if project_id:
            project_manager.add_analysis_to_project(project_id, analysis_run_id)

        # Start analysis (in production, this would be async/background task)
        # For now, run synchronously
        result = processor.process_csv_file(
            input_file=f"CSV Session: {csv_session_id}",
            output_dir=app.config['RESULTS_FOLDER']
        )

        return jsonify({
            'success': True,
            'analysis_run_id': analysis_run_id,
            'status': 'completed'
        })

    except psycopg2.Error as e:
        return safe_error_response(e, 'Database operation failed', 500)
    except ValueError as e:
        return safe_error_response(e, 'Invalid analysis configuration', 400)
    except Exception as e:
        return safe_error_response(e, 'Analysis failed', 500)


@app.route('/api/analysis/<analysis_run_id>/results')
def get_analysis_results(analysis_run_id):
    """Get analysis results"""
    try:
        # Get from database
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get analysis run
                cursor.execute(
                    "SELECT * FROM analysis_runs WHERE id = %s",
                    (analysis_run_id,)
                )
                analysis = cursor.fetchone()

                if not analysis:
                    return jsonify({'error': 'Analysis not found'}), 404

                # Get patterns
                cursor.execute(
                    "SELECT * FROM patterns WHERE analysis_run_id = %s",
                    (analysis_run_id,)
                )
                patterns = cursor.fetchall()

                # Get risk assessment
                cursor.execute(
                    "SELECT * FROM risk_assessments WHERE analysis_run_id = %s",
                    (analysis_run_id,)
                )
                risk = cursor.fetchone()

        return jsonify({
            'analysis': dict(analysis) if analysis else {},
            'patterns': [dict(p) for p in patterns] if patterns else [],
            'risk': dict(risk) if risk else {}
        })

    except psycopg2.Error as e:
        return safe_error_response(e, 'Database operation failed', 500)
    except Exception as e:
        return safe_error_response(e, 'Failed to fetch results', 500)


@app.route('/api/analysis/<analysis_run_id>/timeline')
def get_timeline_data(analysis_run_id):
    """Get timeline visualization data"""
    try:
        # Get messages and their sentiment/risk over time
        messages = db.get_messages(analysis_run_id=analysis_run_id)

        timeline_data = []
        for i, msg in enumerate(messages):
            timeline_data.append({
                'index': i,
                'timestamp': msg.get('timestamp', ''),
                'sender': msg.get('sender', ''),
                'text_preview': msg.get('text', '')[:50],
                # These would come from cached analysis results
                'sentiment': 0,
                'risk': 0
            })

        return jsonify({'timeline': timeline_data})

    except psycopg2.Error as e:
        return safe_error_response(e, 'Database operation failed', 500)
    except Exception as e:
        return safe_error_response(e, 'Failed to fetch timeline', 500)


@app.route('/api/visualizations/<analysis_run_id>/sentiment')
def get_sentiment_viz(analysis_run_id):
    """Generate sentiment visualization"""
    try:
        # Get analysis results
        # This is a simplified version - in production would fetch from database

        # Create sample visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(10)),
            y=[0.2, 0.3, -0.1, 0.5, -0.3, 0.1, 0.4, -0.2, 0.0, 0.3],
            mode='lines+markers',
            name='Sentiment Over Time'
        ))

        fig.update_layout(
            title='Sentiment Trajectory',
            xaxis_title='Message Index',
            yaxis_title='Sentiment Score',
            template='plotly_white'
        )

        return json.dumps(fig, cls=PlotlyJSONEncoder)

    except Exception as e:
        return safe_error_response(e, 'Failed to generate visualization', 500)


@app.route('/api/export/pdf/<analysis_run_id>')
def export_pdf(analysis_run_id):
    """Export analysis results to PDF"""
    try:
        # Generate PDF report
        # This would use ReportLab to create comprehensive PDF
        pdf_path = Path(app.config['RESULTS_FOLDER']) / f"analysis_{analysis_run_id}.pdf"

        # For now, return placeholder
        return jsonify({
            'message': 'PDF generation not yet implemented',
            'analysis_id': analysis_run_id
        })

    except Exception as e:
        return safe_error_response(e, 'Failed to export PDF', 500)


@app.route('/api/export/json/<analysis_run_id>')
def export_json(analysis_run_id):
    """Export analysis results to JSON"""
    try:
        # Get all results
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM analysis_runs WHERE id = %s",
                    (analysis_run_id,)
                )
                analysis = cursor.fetchone()

        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404

        # Export as JSON file
        json_path = Path(app.config['RESULTS_FOLDER']) / f"analysis_{analysis_run_id}.json"
        with open(json_path, 'w') as f:
            json.dump(dict(analysis), f, indent=2, default=str)

        return send_file(json_path, as_attachment=True)

    except FileNotFoundError as e:
        return safe_error_response(e, 'Analysis not found', 404)
    except Exception as e:
        return safe_error_response(e, 'Failed to export JSON', 500)


@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    try:
        stats = redis_cache.get_stats()
        return jsonify(stats)
    except Exception as e:
        return safe_error_response(e, 'Failed to fetch cache statistics', 500)


# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(404)
def not_found(error):
    security_logger.info(f"404 | {request.remote_addr} | {request.method} {request.path}")
    return jsonify({'error': 'Resource not found', 'status': 404}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}", exc_info=True, extra={'client_ip': request.remote_addr})
    return jsonify({'error': 'Internal server error', 'status': 500}), 500


@app.errorhandler(403)
def forbidden(error):
    security_logger.warning(f"403 | {request.remote_addr} | {request.method} {request.path}")
    return jsonify({'error': 'Access forbidden', 'status': 403}), 403


@app.errorhandler(401)
def unauthorized(error):
    security_logger.warning(f"401 | {request.remote_addr} | {request.method} {request.path}")
    return jsonify({'error': 'Authentication required', 'status': 401}), 401


@app.errorhandler(429)
def ratelimit_handler(error):
    security_logger.warning(f"429 Rate Limit | {request.remote_addr} | {request.method} {request.path}")
    return jsonify({'error': 'Too many requests', 'status': 429}), 429


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    flask_env = os.environ.get('FLASK_ENV', 'production')
    debug = flask_env == 'development'

    if debug:
        logger.warning("⚠️  Running in DEBUG mode - NOT suitable for production!")
        logger.warning("⚠️  Detailed error messages will be exposed to clients")
    else:
        logger.info("Running in PRODUCTION mode with security hardening")

    logger.info(f"Starting Message Processor Web Application on port {port}")
    logger.info(f"CSRF Protection: {'Enabled' if csrf else 'Disabled'}")
    logger.info(f"Rate Limiting: {'Enabled' if limiter else 'Disabled'}")

    app.run(host='0.0.0.0', port=port, debug=debug)
