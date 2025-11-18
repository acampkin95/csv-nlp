-- Message Processor Database Schema
-- SQLite database for persistent storage of messages, analysis results, and patterns

-- Messages table (core data)
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    csv_index INTEGER,
    timestamp DATETIME,
    date TEXT,
    time TEXT,
    speaker_id INTEGER,
    text TEXT,
    attachment TEXT,
    service TEXT,
    type TEXT,
    recipients TEXT,
    features_json TEXT, -- cached VADER, NRC, TextBlob features
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_speaker ON messages(speaker_id);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);

-- Speakers table
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    phone TEXT,
    aggregate_stats_json TEXT, -- cached per-speaker statistics
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name);

-- Analysis runs table (provenance and audit trail)
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_file_path TEXT,
    input_file_hash TEXT,
    config_json TEXT,
    library_versions_json TEXT,
    duration_seconds REAL,
    message_count INTEGER,
    speaker_count INTEGER,
    results_json TEXT,
    user_notes TEXT,
    status TEXT DEFAULT 'started', -- started, completed, failed
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON analysis_runs(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_status ON analysis_runs(status);

-- Patterns table (detected behavioral patterns)
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER,
    message_id INTEGER,
    pattern_type TEXT, -- 'grooming', 'manipulation', 'deception', 'control', 'conflict'
    pattern_subtype TEXT, -- 'trust_building', 'gaslighting', 'blame_shifting', etc.
    severity REAL, -- 0-1 scale
    confidence REAL, -- 0-1 confidence in detection
    matched_text TEXT,
    context_before TEXT,
    context_after TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_severity ON patterns(severity);
CREATE INDEX IF NOT EXISTS idx_patterns_run ON patterns(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_patterns_message ON patterns(message_id);

-- Timeline bins table (aggregated statistics)
CREATE TABLE IF NOT EXISTS timeline_bins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER,
    bin_start DATETIME,
    bin_end DATETIME,
    bin_size TEXT, -- 'hour', 'day', 'week', 'month'
    speaker_id INTEGER,
    message_count INTEGER,
    mean_compound REAL,
    mean_positive REAL,
    mean_negative REAL,
    mean_neutral REAL,
    conflict_score REAL,
    grooming_score REAL,
    manipulation_score REAL,
    risk_score REAL,
    dominant_emotion TEXT,
    dominant_intent TEXT,
    features_json TEXT, -- additional aggregated features
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
    FOREIGN KEY (speaker_id) REFERENCES speakers(id)
);

CREATE INDEX IF NOT EXISTS idx_timeline_start ON timeline_bins(bin_start);
CREATE INDEX IF NOT EXISTS idx_timeline_run ON timeline_bins(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_timeline_speaker ON timeline_bins(speaker_id);

-- Risk assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER,
    speaker_id INTEGER,
    assessment_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    grooming_risk REAL, -- 0-1
    manipulation_risk REAL, -- 0-1
    deception_risk REAL, -- 0-1
    hostility_risk REAL, -- 0-1
    overall_risk REAL, -- 0-1
    risk_level TEXT, -- 'low', 'moderate', 'high', 'critical'
    primary_concern TEXT,
    recommendations_json TEXT,
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
    FOREIGN KEY (speaker_id) REFERENCES speakers(id)
);

CREATE INDEX IF NOT EXISTS idx_risk_run ON risk_assessments(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_risk_speaker ON risk_assessments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_risk_level ON risk_assessments(risk_level);

-- Configuration templates table
CREATE TABLE IF NOT EXISTS config_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    config_json TEXT NOT NULL,
    is_default BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Feature cache table (for performance optimization)
CREATE TABLE IF NOT EXISTS feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT UNIQUE NOT NULL,
    vader_json TEXT,
    nrclex_json TEXT,
    textblob_json TEXT,
    patterns_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_hash ON feature_cache(text_hash);

-- Database metadata
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial metadata
INSERT OR REPLACE INTO db_metadata (key, value) VALUES
    ('schema_version', '1.0'),
    ('created_date', datetime('now')),
    ('last_updated', datetime('now'));