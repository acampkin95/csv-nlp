"""
PostgreSQL Database Adapter for Message Processor
Provides high-performance interface to PostgreSQL with JSONB optimization
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import logging
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration"""
    host: str = "acdev.host"
    port: int = 5432
    database: str = "messagestore"
    user: str = "msgprocess"
    password: str = "DHifde93jes9dk"
    schema: str = "message_processor"
    min_connections: int = 2
    max_connections: int = 10


class PostgreSQLAdapter:
    """High-performance PostgreSQL adapter with connection pooling"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize PostgreSQL adapter with connection pool

        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        self.pool = None
        self._init_connection_pool()
        self._init_schema()

    def _init_connection_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                options=f'-c search_path={self.config.schema},public'
            )
            logger.info(f"Connected to PostgreSQL at {self.config.host}")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _init_schema(self):
        """Initialize database schema"""
        schema_path = Path(__file__).parent / "postgresql_schema.sql"
        if schema_path.exists():
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Create schema if not exists
                        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.config.schema}")
                        cursor.execute(f"SET search_path TO {self.config.schema}, public")

                        # Read and execute schema
                        with open(schema_path, 'r') as f:
                            schema_sql = f.read()
                            cursor.execute(schema_sql)

                        conn.commit()
                        logger.info("Database schema initialized")
            except Exception as e:
                logger.error(f"Failed to initialize schema: {e}")

    @contextmanager
    def get_connection(self):
        """Get database connection from pool

        Yields:
            psycopg2 connection
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)

    def close(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()

    # ==========================================
    # CSV Import Management
    # ==========================================

    def create_csv_import_session(self, filename: str, df) -> str:
        """Create a new CSV import session and dedicated table

        Args:
            filename: CSV filename
            df: Pandas DataFrame with CSV data

        Returns:
            str: Session ID
        """
        # Calculate file hash from data
        file_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()

        # Check if already imported
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id FROM csv_import_sessions WHERE file_hash = %s",
                    (file_hash,)
                )
                existing = cursor.fetchone()

                if existing:
                    logger.info(f"CSV already imported with session ID: {existing['id']}")
                    return str(existing['id'])

                # Create new session
                session_id = str(uuid.uuid4())
                table_name = f"csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"

                # Create dedicated table for this CSV
                self._create_csv_table(conn, table_name, df)

                # Insert CSV data into dedicated table
                self._insert_csv_data(conn, table_name, session_id, df)

                # Create import session record
                cursor.execute("""
                    INSERT INTO csv_import_sessions
                    (id, filename, file_hash, table_name, row_count, column_mapping)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    session_id,
                    filename,
                    file_hash,
                    table_name,
                    len(df),
                    Json(dict(df.dtypes.astype(str)))
                ))

                # Also populate master messages table
                self._populate_master_messages(conn, session_id, df)

                conn.commit()
                logger.info(f"Created CSV import session: {session_id}")
                return session_id

    def _create_csv_table(self, conn, table_name: str, df):
        """Create dedicated table for CSV data

        Args:
            conn: Database connection
            table_name: Table name
            df: DataFrame with data
        """
        with conn.cursor() as cursor:
            # Build CREATE TABLE statement
            columns = []
            for col in df.columns:
                # Determine PostgreSQL type
                dtype = str(df[col].dtype)
                if 'int' in dtype:
                    pg_type = 'BIGINT'
                elif 'float' in dtype:
                    pg_type = 'NUMERIC'
                elif 'datetime' in dtype:
                    pg_type = 'TIMESTAMP WITH TIME ZONE'
                elif 'date' in dtype:
                    pg_type = 'DATE'
                else:
                    pg_type = 'TEXT'

                columns.append(f"{self._sanitize_column_name(col)} {pg_type}")

            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    import_session_id UUID,
                    {', '.join(columns)},
                    raw_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """

            cursor.execute(create_sql)

            # Create indexes
            cursor.execute(f"CREATE INDEX idx_{table_name}_session ON {table_name}(import_session_id)")

    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for PostgreSQL

        Args:
            name: Original column name

        Returns:
            str: Sanitized name
        """
        # Replace spaces and special characters
        sanitized = name.lower().replace(' ', '_').replace('-', '_')
        # Remove non-alphanumeric characters
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"col_{sanitized}"
        return sanitized or "column"

    def _insert_csv_data(self, conn, table_name: str, session_id: str, df):
        """Insert CSV data into dedicated table using batch operations

        Args:
            conn: Database connection
            table_name: Table name
            session_id: Import session ID
            df: DataFrame with data
        """
        import psycopg2.extras

        with conn.cursor() as cursor:
            # Prepare columns
            columns = [self._sanitize_column_name(col) for col in df.columns]
            columns_str = ', '.join(columns)

            insert_sql = f"""
                INSERT INTO {table_name}
                (import_session_id, {columns_str}, raw_data)
                VALUES (%s, {', '.join(['%s'] * len(columns))}, %s)
            """

            # Prepare all rows for batch insert (100x faster than per-row)
            rows = []
            for _, row in df.iterrows():
                values = [session_id]
                values.extend(row.values)
                values.append(Json(row.to_dict()))
                rows.append(tuple(values))

            # Batch insert with optimal page size
            psycopg2.extras.execute_batch(cursor, insert_sql, rows, page_size=1000)
            logger.info(f"Batch inserted {len(rows)} rows into {table_name}")

    def _populate_master_messages(self, conn, session_id: str, df):
        """Populate master messages table from CSV data using batch operations

        Args:
            conn: Database connection
            session_id: Import session ID
            df: DataFrame with CSV data
        """
        import psycopg2.extras

        with conn.cursor() as cursor:
            # Map common column names
            column_mapping = {
                'date': ['Date', 'date', 'DATE', 'Message Date'],
                'time': ['Time', 'time', 'TIME', 'Message Time'],
                'sender_name': ['Sender Name', 'Sender', 'From', 'Speaker'],
                'sender_number': ['Sender Number', 'Phone', 'Number'],
                'text': ['Text', 'Message', 'Content', 'Body'],
                'recipients': ['Recipients', 'To', 'Recipient'],
                'type': ['Type', 'Message Type'],
                'service': ['Service', 'Platform'],
                'attachment': ['Attachment', 'Attachments', 'Media']
            }

            # Find actual columns in DataFrame
            actual_columns = {}
            for target, possibilities in column_mapping.items():
                for possible in possibilities:
                    if possible in df.columns:
                        actual_columns[target] = possible
                        break

            # Prepare all message rows for batch insert
            message_rows = []
            speakers_to_update = {}

            for idx, row in df.iterrows():
                # Extract standard fields
                date_val = row.get(actual_columns.get('date'))
                time_val = row.get(actual_columns.get('time'))

                # Try to combine date and time into timestamp
                timestamp = None
                if date_val and time_val:
                    try:
                        timestamp = pd.to_datetime(f"{date_val} {time_val}")
                    except:
                        pass

                # Handle recipients as array
                recipients_raw = row.get(actual_columns.get('recipients'), '')
                if isinstance(recipients_raw, str):
                    recipients = recipients_raw.split(',') if recipients_raw else []
                else:
                    recipients = []

                sender_name = row.get(actual_columns.get('sender_name'))
                sender_number = row.get(actual_columns.get('sender_number'))

                message_rows.append((
                    session_id,
                    idx,
                    timestamp,
                    date_val,
                    time_val,
                    sender_name,
                    sender_number,
                    recipients,
                    row.get(actual_columns.get('text')),
                    row.get(actual_columns.get('type')),
                    row.get(actual_columns.get('service'))
                ))

                # Collect unique speakers for batch update
                if sender_name:
                    if sender_name not in speakers_to_update:
                        speakers_to_update[sender_name] = sender_number
                    elif sender_number and speakers_to_update[sender_name] != sender_number:
                        # Multiple numbers for same name - keep first one
                        pass

            # Batch insert messages (100x faster)
            insert_sql = """
                INSERT INTO messages_master
                (csv_session_id, original_row_id, timestamp, date, time,
                 sender_name, sender_number, recipients, message_text,
                 message_type, service)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            psycopg2.extras.execute_batch(cursor, insert_sql, message_rows, page_size=1000)
            logger.info(f"Batch inserted {len(message_rows)} messages into messages_master")

            # Batch update speakers
            for sender_name, sender_number in speakers_to_update.items():
                self._update_speaker(cursor, sender_name, sender_number)

    def _update_speaker(self, cursor, name: str, phone: Optional[str] = None):
        """Update or create speaker record

        Args:
            cursor: Database cursor
            name: Speaker name
            phone: Optional phone number
        """
        cursor.execute("""
            INSERT INTO speakers (name, phone_numbers, first_seen, last_seen, total_messages)
            VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            ON CONFLICT (name) DO UPDATE SET
                phone_numbers = CASE
                    WHEN %s IS NOT NULL AND NOT (%s = ANY(speakers.phone_numbers))
                    THEN array_append(speakers.phone_numbers, %s)
                    ELSE speakers.phone_numbers
                END,
                last_seen = CURRENT_TIMESTAMP,
                total_messages = speakers.total_messages + 1
        """, (
            name,
            [phone] if phone else [],
            phone, phone, phone
        ))

    # ==========================================
    # Analysis Management
    # ==========================================

    def create_analysis_run(self, csv_session_id: str, config: Dict) -> str:
        """Create new analysis run

        Args:
            csv_session_id: CSV import session ID
            config: Analysis configuration

        Returns:
            str: Analysis run ID
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                run_id = str(uuid.uuid4())

                cursor.execute("""
                    INSERT INTO analysis_runs
                    (id, csv_session_id, config, analysis_type, status)
                    VALUES (%s, %s, %s, %s, 'started')
                    RETURNING id
                """, (
                    run_id,
                    csv_session_id,
                    Json(config),
                    config.get('analysis_type', 'full')
                ))

                conn.commit()
                return run_id

    def update_analysis_run(self, run_id: str, **kwargs):
        """Update analysis run

        Args:
            run_id: Analysis run ID
            **kwargs: Fields to update
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Build UPDATE statement
                updates = []
                values = []

                for key, value in kwargs.items():
                    if key in ['results', 'insights', 'recommendations', 'error_details']:
                        updates.append(f"{key} = %s")
                        values.append(Json(value))
                    elif key == 'status':
                        updates.append("status = %s")
                        values.append(value)
                        if value == 'completed':
                            updates.append("completed_at = CURRENT_TIMESTAMP")
                    elif key == 'progress':
                        updates.append("progress = %s")
                        values.append(value)
                    elif key == 'duration':
                        updates.append("duration_seconds = %s")
                        values.append(value)
                    else:
                        updates.append(f"{key} = %s")
                        values.append(value)

                if updates:
                    values.append(run_id)
                    cursor.execute(
                        f"UPDATE analysis_runs SET {', '.join(updates)} WHERE id = %s",
                        values
                    )
                    conn.commit()

    # ==========================================
    # Message Operations
    # ==========================================

    def get_messages(self, csv_session_id: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Dict]:
        """Get messages from master table

        Args:
            csv_session_id: Optional CSV session filter
            limit: Maximum messages to return

        Returns:
            List of message dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = "SELECT * FROM messages_master WHERE 1=1"
                params = []

                if csv_session_id:
                    query += " AND csv_session_id = %s"
                    params.append(csv_session_id)

                query += " ORDER BY timestamp"

                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query, params)
                return cursor.fetchall()

    def update_message_analysis(self, message_id: str, analysis_type: str, analysis_data: Dict):
        """Update message with analysis results

        Args:
            message_id: Message ID
            analysis_type: Type of analysis (sentiment, pattern, risk)
            analysis_data: Analysis results
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                field_map = {
                    'sentiment': 'sentiment_analysis',
                    'pattern': 'pattern_analysis',
                    'risk': 'risk_analysis',
                    'features': 'features_cache'
                }

                field = field_map.get(analysis_type)
                if field:
                    cursor.execute(
                        f"UPDATE messages_master SET {field} = %s WHERE id = %s",
                        (Json(analysis_data), message_id)
                    )
                    conn.commit()

    # ==========================================
    # Pattern Storage
    # ==========================================

    def insert_patterns_batch(self, patterns: List[Dict]):
        """Insert detected patterns in batch

        Args:
            patterns: List of pattern dictionaries
        """
        if not patterns:
            return

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Use COPY for maximum performance
                from io import StringIO
                import csv

                # Prepare data for COPY
                output = StringIO()
                writer = csv.writer(output)

                for pattern in patterns:
                    writer.writerow([
                        str(uuid.uuid4()),  # Generate ID
                        pattern.get('analysis_run_id'),
                        pattern.get('message_id'),
                        pattern.get('pattern_category'),
                        pattern.get('pattern_type'),
                        pattern.get('pattern_subtype'),
                        pattern.get('matched_text'),
                        pattern.get('context_before'),
                        pattern.get('context_after'),
                        pattern.get('severity', 0),
                        pattern.get('confidence', 0),
                        json.dumps(pattern.get('details', {}))
                    ])

                output.seek(0)

                # Use COPY for bulk insert
                cursor.copy_from(
                    output,
                    'detected_patterns',
                    columns=['id', 'analysis_run_id', 'message_id', 'pattern_category',
                            'pattern_type', 'pattern_subtype', 'matched_text',
                            'context_before', 'context_after', 'severity',
                            'confidence', 'details'],
                    sep=',',
                    null=''
                )

                conn.commit()
                logger.info(f"Inserted {len(patterns)} patterns")

    # ==========================================
    # Risk Assessment
    # ==========================================

    def save_risk_assessment(self, assessment: Dict):
        """Save risk assessment results

        Args:
            assessment: Risk assessment dictionary
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO risk_assessments
                    (analysis_run_id, speaker_id, grooming_risk, manipulation_risk,
                     deception_risk, hostility_risk, overall_risk, risk_level,
                     primary_concern, escalation_risk, assessment_details,
                     behavioral_indicators, recommendations)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    assessment.get('analysis_run_id'),
                    assessment.get('speaker_id'),
                    assessment.get('grooming_risk', 0),
                    assessment.get('manipulation_risk', 0),
                    assessment.get('deception_risk', 0),
                    assessment.get('hostility_risk', 0),
                    assessment.get('overall_risk', 0),
                    assessment.get('risk_level'),
                    assessment.get('primary_concern'),
                    assessment.get('escalation_risk', 0),
                    Json(assessment.get('assessment_details', {})),
                    Json(assessment.get('behavioral_indicators', {})),
                    assessment.get('recommendations', [])
                ))
                conn.commit()

    # ==========================================
    # Analytics and Aggregation
    # ==========================================

    def create_timeline_aggregation(self, csv_session_id: str, window_size: str = 'day'):
        """Create timeline aggregations for performance

        Args:
            csv_session_id: CSV session ID
            window_size: Aggregation window (hour, day, week, month)
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Define window truncation based on size
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

    def get_speaker_statistics(self, speaker_name: str) -> Dict:
        """Get comprehensive speaker statistics

        Args:
            speaker_name: Speaker name

        Returns:
            Dict: Speaker statistics
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        s.*,
                        COUNT(m.id) as message_count,
                        AVG((m.sentiment_analysis->>'compound')::numeric) as avg_sentiment,
                        MAX((m.risk_analysis->>'overall_risk')::numeric) as max_risk,
                        MIN(m.timestamp) as first_message,
                        MAX(m.timestamp) as last_message,
                        jsonb_agg(DISTINCT m.pattern_analysis->'primary_concern') as concerns
                    FROM speakers s
                    LEFT JOIN messages_master m ON m.sender_name = s.name
                    WHERE s.name = %s
                    GROUP BY s.id
                """, (speaker_name,))

                return cursor.fetchone()

    # ==========================================
    # Maintenance and Optimization
    # ==========================================

    def create_performance_indexes(self):
        """Create all performance indexes for 10-100x faster queries

        This method applies comprehensive indexing strategy from performance_indexes.sql
        Safe to run multiple times (uses IF NOT EXISTS)
        """
        indexes_path = Path(__file__).parent / "performance_indexes.sql"

        if not indexes_path.exists():
            logger.warning(f"Performance indexes SQL file not found: {indexes_path}")
            return

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Read and execute index creation script
                    with open(indexes_path, 'r') as f:
                        indexes_sql = f.read()

                    # Execute the entire script
                    cursor.execute(indexes_sql)
                    conn.commit()

                    logger.info("✅ Performance indexes created successfully")
                    logger.info("Expected improvement: 10-100x faster queries on large datasets")
        except Exception as e:
            logger.error(f"Failed to create performance indexes: {e}")
            raise

    def refresh_materialized_views(self):
        """Refresh all materialized views"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY speaker_stats")
                conn.commit()
                logger.info("Refreshed materialized views")

    def vacuum_analyze(self):
        """Run VACUUM ANALYZE for optimization"""
        with self.get_connection() as conn:
            # Need autocommit for VACUUM
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)
            try:
                with conn.cursor() as cursor:
                    cursor.execute("VACUUM ANALYZE")
                    logger.info("Completed VACUUM ANALYZE")
            finally:
                conn.set_isolation_level(old_isolation)

    def get_database_stats(self) -> Dict:
        """Get database statistics

        Returns:
            Dict: Database statistics
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                stats = {}

                # Table sizes
                cursor.execute("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = %s
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """, (self.config.schema,))
                stats['table_sizes'] = cursor.fetchall()

                # Row counts
                for table in ['messages_master', 'csv_import_sessions', 'analysis_runs', 'detected_patterns']:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()['count']

                # Database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(%s)) as db_size
                """, (self.config.database,))
                stats['total_size'] = cursor.fetchone()['db_size']

                return stats


# Convenience functions
def get_postgresql_adapter(config: Optional[DatabaseConfig] = None) -> PostgreSQLAdapter:
    """Get PostgreSQL adapter instance

    Args:
        config: Optional database configuration

    Returns:
        PostgreSQLAdapter instance
    """
    return PostgreSQLAdapter(config)


import pandas as pd  # Add this import at the top