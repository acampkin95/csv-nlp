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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Initialize components
redis_cache = RedisCache(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379))
)

db_config = DatabaseConfig(
    host=os.environ.get('POSTGRES_HOST', 'acdev.host'),
    database=os.environ.get('POSTGRES_DB', 'messagestore'),
    user=os.environ.get('POSTGRES_USER', 'msgprocess'),
    password=os.environ.get('POSTGRES_PASSWORD', 'DHifde93jes9dk')
)

db = PostgreSQLAdapter(db_config)
config_manager = ConfigManager()


# ==========================================
# Project Management
# ==========================================

class ProjectManager:
    """Manage analysis projects"""

    def __init__(self, db: PostgreSQLAdapter, cache: RedisCache):
        self.db = db
        self.cache = cache

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
            'analysis_runs': []
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

    def list_projects(self, user_id: str = 'default') -> List[Dict]:
        """List all projects for a user"""
        # For now, get from cache
        # In production, this would query database
        return []


project_manager = ProjectManager(db, redis_cache)


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
    """Upload CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

        # Validate CSV
        validator = CSVValidator()
        validation_result, df = validator.validate_file(filepath)

        if not validation_result.is_valid:
            return jsonify({
                'error': 'CSV validation failed',
                'details': validation_result.errors
            }), 400

        # Create or get project
        project_name = request.form.get('project_name', filename)
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

        return jsonify({
            'success': True,
            'project_id': project_id,
            'csv_session_id': csv_session_id,
            'filename': saved_filename,
            'rows': len(df),
            'columns': len(df.columns)
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


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

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


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

    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return jsonify({'error': str(e)}), 500


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

    except Exception as e:
        logger.error(f"Error fetching timeline: {e}")
        return jsonify({'error': str(e)}), 500


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
        logger.error(f"Error generating visualization: {e}")
        return jsonify({'error': str(e)}), 500


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
        logger.error(f"Error exporting PDF: {e}")
        return jsonify({'error': str(e)}), 500


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

    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    try:
        stats = redis_cache.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Message Processor Web Application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
