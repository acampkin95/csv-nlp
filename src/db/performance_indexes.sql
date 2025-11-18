-- Performance Indexes for CSV-NLP Message Processor
-- Creates comprehensive indexes for 10-100x faster queries on large datasets

-- ============================================================================
-- MESSAGE QUERIES (Used in every analysis)
-- ============================================================================

-- Sender name lookup (used for speaker analysis)
CREATE INDEX IF NOT EXISTS idx_messages_sender
ON messages_master(sender_name);

-- Timestamp ordering (used for timeline views)
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
ON messages_master(timestamp);

-- Combined session + timestamp (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
ON messages_master(csv_session_id, timestamp)
WHERE timestamp IS NOT NULL;

-- Sender within session (speaker-specific queries)
CREATE INDEX IF NOT EXISTS idx_messages_session_sender
ON messages_master(csv_session_id, sender_name);

-- ============================================================================
-- PATTERN SEARCHES (Used in visualizations)
-- ============================================================================

-- Pattern severity lookup (find high-risk patterns)
CREATE INDEX IF NOT EXISTS idx_patterns_severity
ON detected_patterns(severity DESC)
WHERE severity >= 0.6;

-- Pattern category filtering
CREATE INDEX IF NOT EXISTS idx_patterns_category
ON detected_patterns(pattern_category, pattern_type);

-- Patterns by analysis run and severity (dashboard queries)
CREATE INDEX IF NOT EXISTS idx_patterns_run_severity
ON detected_patterns(analysis_run_id, severity DESC);

-- Message-specific patterns (show patterns for message)
CREATE INDEX IF NOT EXISTS idx_patterns_message
ON detected_patterns(message_id);

-- ============================================================================
-- ANALYSIS RUN QUERIES
-- ============================================================================

-- Status filtering (find running/completed analyses)
CREATE INDEX IF NOT EXISTS idx_analysis_status
ON analysis_runs(status);

-- Recent analyses (ordered by creation time)
CREATE INDEX IF NOT EXISTS idx_analysis_created
ON analysis_runs(created_at DESC);

-- Session-specific analyses
CREATE INDEX IF NOT EXISTS idx_analysis_session
ON analysis_runs(csv_session_id);

-- ============================================================================
-- JSONB INDEXES (Fast filtering on analysis results)
-- ============================================================================

-- Sentiment analysis GIN index (enables fast JSON queries)
CREATE INDEX IF NOT EXISTS idx_messages_sentiment_gin
ON messages_master USING GIN (sentiment_analysis);

-- Risk analysis GIN index
CREATE INDEX IF NOT EXISTS idx_messages_risk_gin
ON messages_master USING GIN (risk_analysis);

-- Pattern analysis GIN index
CREATE INDEX IF NOT EXISTS idx_messages_pattern_gin
ON messages_master USING GIN (pattern_analysis);

-- ============================================================================
-- PARTIAL INDEXES (High-risk messages only - saves space)
-- ============================================================================

-- High-risk messages (critical/high only)
CREATE INDEX IF NOT EXISTS idx_messages_high_risk
ON messages_master(csv_session_id, timestamp, sender_name)
WHERE (risk_analysis->>'risk_level')::text IN ('high', 'critical');

-- Messages with detected patterns
CREATE INDEX IF NOT EXISTS idx_messages_with_patterns
ON messages_master(csv_session_id, sender_name)
WHERE pattern_analysis IS NOT NULL;

-- ============================================================================
-- CSV IMPORT SESSION INDEXES
-- ============================================================================

-- File hash lookup (detect duplicate imports)
CREATE INDEX IF NOT EXISTS idx_csv_sessions_hash
ON csv_import_sessions(file_hash);

-- Session lookup by filename
CREATE INDEX IF NOT EXISTS idx_csv_sessions_filename
ON csv_import_sessions(filename);

-- Recent imports
CREATE INDEX IF NOT EXISTS idx_csv_sessions_created
ON csv_import_sessions(created_at DESC);

-- ============================================================================
-- SPEAKER INDEXES
-- ============================================================================

-- Speaker name lookup (primary key already indexed, but explicit for clarity)
CREATE INDEX IF NOT EXISTS idx_speakers_name
ON speakers(name);

-- Recent activity
CREATE INDEX IF NOT EXISTS idx_speakers_last_seen
ON speakers(last_seen DESC);

-- High-volume speakers
CREATE INDEX IF NOT EXISTS idx_speakers_message_count
ON speakers(total_messages DESC)
WHERE total_messages > 10;

-- ============================================================================
-- RISK ASSESSMENT INDEXES
-- ============================================================================

-- Analysis run lookup
CREATE INDEX IF NOT EXISTS idx_risk_analysis_run
ON risk_assessments(analysis_run_id);

-- Speaker risk lookup
CREATE INDEX IF NOT EXISTS idx_risk_speaker
ON risk_assessments(speaker_id);

-- High-risk assessments
CREATE INDEX IF NOT EXISTS idx_risk_high_risk
ON risk_assessments(overall_risk DESC, risk_level)
WHERE risk_level IN ('high', 'critical');

-- ============================================================================
-- TIMELINE AGGREGATION INDEXES
-- ============================================================================

-- Session + window lookup (fast timeline queries)
CREATE INDEX IF NOT EXISTS idx_timeline_session_window
ON timeline_aggregations(csv_session_id, window_start);

-- Window size filtering
CREATE INDEX IF NOT EXISTS idx_timeline_window_size
ON timeline_aggregations(window_size, csv_session_id);

-- ============================================================================
-- COMPOUND INDEXES FOR COMPLEX QUERIES
-- ============================================================================

-- Message analysis combo (session, sender, timestamp, risk)
CREATE INDEX IF NOT EXISTS idx_messages_analysis_combo
ON messages_master(csv_session_id, sender_name, timestamp)
INCLUDE (sentiment_analysis, risk_analysis)
WHERE timestamp IS NOT NULL;

-- ============================================================================
-- TEXT SEARCH INDEXES (Optional - for message content search)
-- ============================================================================

-- Full-text search on message text (using GIN)
-- Uncomment if you need full-text search capability
-- CREATE INDEX IF NOT EXISTS idx_messages_text_search
-- ON messages_master USING GIN (to_tsvector('english', message_text));

-- ============================================================================
-- MAINTENANCE NOTES
-- ============================================================================

-- To rebuild indexes after bulk data changes:
-- REINDEX INDEX CONCURRENTLY idx_messages_session_timestamp;

-- To check index usage:
-- SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
-- FROM pg_stat_user_indexes
-- WHERE schemaname = 'message_processor'
-- ORDER BY idx_scan;

-- To find missing indexes:
-- SELECT schemaname, tablename, attname, n_distinct, correlation
-- FROM pg_stats
-- WHERE schemaname = 'message_processor'
-- AND tablename = 'messages_master'
-- ORDER BY abs(correlation) DESC;
