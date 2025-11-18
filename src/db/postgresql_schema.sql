-- PostgreSQL Schema for Message Processor
-- Host: acdev.host
-- Database: messagestore
-- User: msgprocess

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For GIN indexes on multiple columns
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- For encryption if needed

-- Drop existing tables if needed (careful in production)
-- DROP SCHEMA IF EXISTS message_processor CASCADE;
CREATE SCHEMA IF NOT EXISTS message_processor;
SET search_path TO message_processor, public;

-- =====================================================
-- Core Tables
-- =====================================================

-- CSV Import Sessions (tracks each CSV file imported)
CREATE TABLE IF NOT EXISTS csv_import_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,  -- SHA256 hash for deduplication
    imported_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    table_name TEXT NOT NULL,  -- Dynamic table created for this CSV
    row_count INTEGER,
    column_mapping JSONB,  -- Maps CSV columns to standard fields
    validation_results JSONB,  -- Validation warnings, errors, stats
    metadata JSONB,  -- Additional file metadata
    UNIQUE(file_hash)
);

CREATE INDEX idx_csv_sessions_filename ON csv_import_sessions(filename);
CREATE INDEX idx_csv_sessions_imported_at ON csv_import_sessions(imported_at DESC);

-- Master Messages Table (normalized view of all messages)
CREATE TABLE IF NOT EXISTS messages_master (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    csv_session_id UUID REFERENCES csv_import_sessions(id) ON DELETE CASCADE,
    original_row_id INTEGER,  -- Row number in original CSV

    -- Standard message fields
    timestamp TIMESTAMP WITH TIME ZONE,
    date DATE,
    time TIME,
    sender_name TEXT,
    sender_number TEXT,
    recipients TEXT[],
    message_text TEXT,
    message_type TEXT,
    service TEXT,
    attachment_info JSONB,

    -- Computed fields
    text_vector tsvector,  -- For full-text search
    word_count INTEGER,
    char_count INTEGER,

    -- Analysis cache (JSONB for flexibility)
    sentiment_analysis JSONB,
    pattern_analysis JSONB,
    risk_analysis JSONB,
    features_cache JSONB,  -- All extracted features

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_messages_timestamp ON messages_master(timestamp DESC);
CREATE INDEX idx_messages_sender ON messages_master(sender_name);
CREATE INDEX idx_messages_date ON messages_master(date DESC);
CREATE GIN INDEX idx_messages_text_search ON messages_master(text_vector);
CREATE GIN INDEX idx_messages_features ON messages_master(features_cache);
CREATE INDEX idx_messages_csv_session ON messages_master(csv_session_id);

-- Speakers/Participants Table
CREATE TABLE IF NOT EXISTS speakers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    phone_numbers TEXT[] DEFAULT '{}',  -- Array of associated numbers
    email_addresses TEXT[] DEFAULT '{}',

    -- Aggregate statistics (JSONB for flexibility)
    statistics JSONB DEFAULT '{}',
    behavioral_profile JSONB DEFAULT '{}',
    risk_profile JSONB DEFAULT '{}',

    -- Metadata
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    total_messages INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(name)
);

CREATE INDEX idx_speakers_name ON speakers(name);
CREATE GIN INDEX idx_speakers_phones ON speakers(phone_numbers);

-- Analysis Runs Table
CREATE TABLE IF NOT EXISTS analysis_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    csv_session_id UUID REFERENCES csv_import_sessions(id),

    -- Run configuration
    config JSONB NOT NULL,
    analysis_type TEXT,  -- 'full', 'quick', 'custom'

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds NUMERIC(10,2),

    -- Results
    status TEXT DEFAULT 'started',  -- started, running, completed, failed
    progress NUMERIC(5,2) DEFAULT 0,  -- Percentage
    results JSONB,  -- Complete analysis results
    insights JSONB,  -- Key insights and findings
    recommendations JSONB,  -- Generated recommendations

    -- Statistics
    messages_analyzed INTEGER,
    patterns_detected INTEGER,
    risk_score NUMERIC(3,2),  -- 0-1 scale

    -- Error handling
    error_message TEXT,
    error_details JSONB
);

CREATE INDEX idx_analysis_runs_session ON analysis_runs(csv_session_id);
CREATE INDEX idx_analysis_runs_started ON analysis_runs(started_at DESC);
CREATE INDEX idx_analysis_runs_status ON analysis_runs(status);

-- Detected Patterns Table
CREATE TABLE IF NOT EXISTS detected_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_run_id UUID REFERENCES analysis_runs(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages_master(id) ON DELETE CASCADE,

    -- Pattern details
    pattern_category TEXT,  -- grooming, manipulation, deception, etc.
    pattern_type TEXT,  -- Specific pattern type
    pattern_subtype TEXT,
    matched_text TEXT,

    -- Context
    context_before TEXT,
    context_after TEXT,

    -- Scoring
    severity NUMERIC(3,2),  -- 0-1 scale
    confidence NUMERIC(3,2),  -- 0-1 scale

    -- Metadata
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB  -- Additional pattern-specific details
);

CREATE INDEX idx_patterns_analysis ON detected_patterns(analysis_run_id);
CREATE INDEX idx_patterns_message ON detected_patterns(message_id);
CREATE INDEX idx_patterns_category ON detected_patterns(pattern_category);
CREATE INDEX idx_patterns_severity ON detected_patterns(severity DESC);

-- Risk Assessments Table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_run_id UUID REFERENCES analysis_runs(id) ON DELETE CASCADE,
    speaker_id UUID REFERENCES speakers(id),

    -- Risk components
    grooming_risk NUMERIC(3,2),
    manipulation_risk NUMERIC(3,2),
    deception_risk NUMERIC(3,2),
    hostility_risk NUMERIC(3,2),
    overall_risk NUMERIC(3,2),

    -- Risk metadata
    risk_level TEXT,  -- low, moderate, high, critical
    primary_concern TEXT,
    escalation_risk NUMERIC(3,2),

    -- Detailed assessment
    assessment_details JSONB,
    behavioral_indicators JSONB,
    recommendations TEXT[],

    -- Temporal
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_analysis ON risk_assessments(analysis_run_id);
CREATE INDEX idx_risk_speaker ON risk_assessments(speaker_id);
CREATE INDEX idx_risk_level ON risk_assessments(risk_level);

-- Timeline Aggregations Table (for performance)
CREATE TABLE IF NOT EXISTS timeline_aggregations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    csv_session_id UUID REFERENCES csv_import_sessions(id),

    -- Time window
    window_start TIMESTAMP WITH TIME ZONE,
    window_end TIMESTAMP WITH TIME ZONE,
    window_size TEXT,  -- hour, day, week, month

    -- Aggregated metrics (JSONB for flexibility)
    metrics JSONB,
    /*
    Example metrics structure:
    {
        "message_count": 100,
        "unique_speakers": 5,
        "sentiment": {
            "mean": 0.3,
            "min": -0.8,
            "max": 0.9,
            "variance": 0.2
        },
        "emotions": {
            "joy": 45,
            "anger": 10,
            ...
        },
        "risk_events": 3,
        "pattern_counts": {...}
    }
    */

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_timeline_session ON timeline_aggregations(csv_session_id);
CREATE INDEX idx_timeline_window ON timeline_aggregations(window_start, window_end);

-- =====================================================
-- Dynamic CSV Tables
-- =====================================================

-- Function to create a dedicated table for each CSV import
CREATE OR REPLACE FUNCTION create_csv_table(
    table_name TEXT,
    columns JSONB
) RETURNS VOID AS $$
DECLARE
    create_sql TEXT;
    col_name TEXT;
    col_type TEXT;
BEGIN
    -- Build CREATE TABLE statement
    create_sql := 'CREATE TABLE IF NOT EXISTS ' || quote_ident(table_name) || ' (
        id SERIAL PRIMARY KEY,
        import_session_id UUID,';

    -- Add columns from JSONB
    FOR col_name, col_type IN SELECT * FROM jsonb_each_text(columns)
    LOOP
        create_sql := create_sql || quote_ident(col_name) || ' ' || col_type || ',';
    END LOOP;

    -- Add metadata columns
    create_sql := create_sql || '
        raw_data JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )';

    -- Execute the CREATE TABLE
    EXECUTE create_sql;

    -- Create indexes
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_' || table_name || '_session
             ON ' || quote_ident(table_name) || '(import_session_id)';
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Audit and Logging
-- =====================================================

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT,
    operation TEXT,  -- INSERT, UPDATE, DELETE, ANALYSIS
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    user_id TEXT,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_table ON audit_log(table_name);
CREATE INDEX idx_audit_operation ON audit_log(operation);

-- =====================================================
-- Performance Optimization Views
-- =====================================================

-- Materialized view for speaker statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS speaker_stats AS
SELECT
    s.id,
    s.name,
    COUNT(DISTINCT m.id) as total_messages,
    COUNT(DISTINCT DATE(m.timestamp)) as active_days,
    AVG((m.sentiment_analysis->>'compound')::numeric) as avg_sentiment,
    MAX((m.risk_analysis->>'overall_risk')::numeric) as max_risk_score,
    MIN(m.timestamp) as first_message,
    MAX(m.timestamp) as last_message
FROM speakers s
LEFT JOIN messages_master m ON m.sender_name = s.name
GROUP BY s.id, s.name;

CREATE UNIQUE INDEX idx_speaker_stats_id ON speaker_stats(id);

-- =====================================================
-- Functions and Triggers
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER update_messages_updated_at
    BEFORE UPDATE ON messages_master
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_speakers_updated_at
    BEFORE UPDATE ON speakers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to update text search vector
CREATE OR REPLACE FUNCTION update_text_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.text_vector := to_tsvector('english', COALESCE(NEW.message_text, ''));
    NEW.word_count := array_length(string_to_array(NEW.message_text, ' '), 1);
    NEW.char_count := length(NEW.message_text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_message_search_vector
    BEFORE INSERT OR UPDATE OF message_text ON messages_master
    FOR EACH ROW
    EXECUTE FUNCTION update_text_search_vector();

-- =====================================================
-- Performance Settings
-- =====================================================

-- Optimize for JSONB operations
ALTER DATABASE messagestore SET random_page_cost = 1.1;
ALTER DATABASE messagestore SET effective_cache_size = '4GB';
ALTER DATABASE messagestore SET shared_buffers = '1GB';
ALTER DATABASE messagestore SET work_mem = '50MB';

-- Create statistics for better query planning
CREATE STATISTICS messages_stats ON timestamp, sender_name FROM messages_master;

-- =====================================================
-- Security
-- =====================================================

-- Row Level Security (optional, uncomment if needed)
-- ALTER TABLE messages_master ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE analysis_runs ENABLE ROW LEVEL SECURITY;

-- Grant appropriate permissions
GRANT ALL PRIVILEGES ON SCHEMA message_processor TO msgprocess;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA message_processor TO msgprocess;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA message_processor TO msgprocess;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA message_processor TO msgprocess;