"""
Database adapter layer for Message Processor
Provides high-level interface to SQLite database with connection pooling,
transaction management, and error handling.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Speaker:
    """Speaker entity"""
    id: Optional[int] = None
    name: str = ""
    phone: Optional[str] = None
    aggregate_stats_json: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Message:
    """Message entity"""
    id: Optional[int] = None
    csv_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    date: Optional[str] = None
    time: Optional[str] = None
    speaker_id: Optional[int] = None
    text: str = ""
    attachment: Optional[str] = None
    service: Optional[str] = None
    type: Optional[str] = None
    recipients: Optional[str] = None
    features_json: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class AnalysisRun:
    """Analysis run entity for provenance tracking"""
    id: Optional[int] = None
    run_timestamp: Optional[datetime] = None
    input_file_path: Optional[str] = None
    input_file_hash: Optional[str] = None
    config_json: Optional[str] = None
    library_versions_json: Optional[str] = None
    duration_seconds: Optional[float] = None
    message_count: Optional[int] = None
    speaker_count: Optional[int] = None
    results_json: Optional[str] = None
    user_notes: Optional[str] = None
    status: str = "started"
    error_message: Optional[str] = None


@dataclass
class Pattern:
    """Detected pattern entity"""
    id: Optional[int] = None
    analysis_run_id: int = 0
    message_id: int = 0
    pattern_type: str = ""
    pattern_subtype: Optional[str] = None
    severity: float = 0.0
    confidence: float = 0.0
    matched_text: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    detected_at: Optional[datetime] = None


class DatabaseAdapter:
    """High-level database interface with connection management"""

    def __init__(self, db_path: str = "data/analysis.db"):
        """Initialize database adapter

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._init_database()

    def _init_database(self):
        """Initialize database with schema"""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                logger.info(f"Database initialized at {self.db_path}")
        else:
            logger.warning(f"Schema file not found at {schema_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    # === Speaker Operations ===

    def create_speaker(self, name: str, phone: Optional[str] = None) -> int:
        """Create or get speaker by name

        Args:
            name: Speaker name
            phone: Optional phone number

        Returns:
            int: Speaker ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Try to get existing speaker
            cursor.execute("SELECT id FROM speakers WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                return row['id']

            # Create new speaker
            cursor.execute(
                "INSERT INTO speakers (name, phone) VALUES (?, ?)",
                (name, phone)
            )
            return cursor.lastrowid

    def get_speaker(self, speaker_id: int) -> Optional[Speaker]:
        """Get speaker by ID

        Args:
            speaker_id: Speaker ID

        Returns:
            Optional[Speaker]: Speaker entity or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,))
            row = cursor.fetchone()

            if row:
                return Speaker(**dict(row))
            return None

    def update_speaker_stats(self, speaker_id: int, stats: Dict[str, Any]):
        """Update speaker aggregate statistics

        Args:
            speaker_id: Speaker ID
            stats: Statistics dictionary
        """
        stats_json = json.dumps(stats)
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE speakers
                   SET aggregate_stats_json = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (stats_json, speaker_id)
            )

    # === Message Operations ===

    def insert_messages_batch(self, messages: List[Dict[str, Any]]) -> int:
        """Insert messages in batch

        Args:
            messages: List of message dictionaries

        Returns:
            int: Number of messages inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Prepare data for batch insert
            rows = []
            for msg in messages:
                # Get or create speaker
                speaker_id = self.create_speaker(
                    msg.get('Sender Name', 'Unknown'),
                    msg.get('Sender Number')
                )

                rows.append((
                    msg.get('csv_index'),
                    msg.get('timestamp'),
                    msg.get('Date'),
                    msg.get('Time'),
                    speaker_id,
                    msg.get('Text', ''),
                    msg.get('Attachment'),
                    msg.get('Service'),
                    msg.get('Type'),
                    msg.get('Recipients'),
                    json.dumps(msg.get('features', {})) if msg.get('features') else None
                ))

            cursor.executemany(
                """INSERT INTO messages
                   (csv_index, timestamp, date, time, speaker_id, text,
                    attachment, service, type, recipients, features_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )

            return cursor.rowcount

    def get_messages(self,
                     speaker_id: Optional[int] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: Optional[int] = None) -> List[Message]:
        """Get messages with optional filters

        Args:
            speaker_id: Filter by speaker
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of messages

        Returns:
            List[Message]: List of message entities
        """
        query = "SELECT * FROM messages WHERE 1=1"
        params = []

        if speaker_id:
            query += " AND speaker_id = ?"
            params.append(speaker_id)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        if limit:
            query += f" LIMIT {limit}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [Message(**dict(row)) for row in rows]

    def update_message_features(self, message_id: int, features: Dict[str, Any]):
        """Update cached features for a message

        Args:
            message_id: Message ID
            features: Features dictionary
        """
        features_json = json.dumps(features)
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE messages SET features_json = ? WHERE id = ?",
                (features_json, message_id)
            )

    # === Analysis Run Operations ===

    def create_analysis_run(self,
                           input_file_path: str,
                           config: Dict[str, Any]) -> int:
        """Create new analysis run

        Args:
            input_file_path: Path to input file
            config: Configuration dictionary

        Returns:
            int: Analysis run ID
        """
        # Calculate file hash
        file_hash = self._calculate_file_hash(input_file_path)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO analysis_runs
                   (input_file_path, input_file_hash, config_json, status)
                   VALUES (?, ?, ?, ?)""",
                (input_file_path, file_hash, json.dumps(config), 'started')
            )
            return cursor.lastrowid

    def update_analysis_run(self,
                           run_id: int,
                           status: str = None,
                           duration: float = None,
                           results: Dict[str, Any] = None,
                           error_message: str = None):
        """Update analysis run

        Args:
            run_id: Analysis run ID
            status: Run status
            duration: Duration in seconds
            results: Results dictionary
            error_message: Error message if failed
        """
        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)

        if duration is not None:
            updates.append("duration_seconds = ?")
            params.append(duration)

        if results:
            updates.append("results_json = ?")
            params.append(json.dumps(results))

        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(run_id)

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE analysis_runs SET {', '.join(updates)} WHERE id = ?",
                params
            )

    def get_analysis_run(self, run_id: int) -> Optional[AnalysisRun]:
        """Get analysis run by ID

        Args:
            run_id: Analysis run ID

        Returns:
            Optional[AnalysisRun]: Analysis run entity or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()

            if row:
                return AnalysisRun(**dict(row))
            return None

    # === Pattern Operations ===

    def insert_pattern(self, pattern: Pattern) -> int:
        """Insert detected pattern

        Args:
            pattern: Pattern entity

        Returns:
            int: Pattern ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO patterns
                   (analysis_run_id, message_id, pattern_type, pattern_subtype,
                    severity, confidence, matched_text, context_before, context_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pattern.analysis_run_id, pattern.message_id, pattern.pattern_type,
                 pattern.pattern_subtype, pattern.severity, pattern.confidence,
                 pattern.matched_text, pattern.context_before, pattern.context_after)
            )
            return cursor.lastrowid

    def insert_patterns_batch(self, patterns: List[Pattern]) -> int:
        """Insert patterns in batch

        Args:
            patterns: List of pattern entities

        Returns:
            int: Number of patterns inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            rows = [
                (p.analysis_run_id, p.message_id, p.pattern_type, p.pattern_subtype,
                 p.severity, p.confidence, p.matched_text, p.context_before, p.context_after)
                for p in patterns
            ]

            cursor.executemany(
                """INSERT INTO patterns
                   (analysis_run_id, message_id, pattern_type, pattern_subtype,
                    severity, confidence, matched_text, context_before, context_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )

            return cursor.rowcount

    def get_patterns(self,
                    analysis_run_id: Optional[int] = None,
                    pattern_type: Optional[str] = None,
                    min_severity: Optional[float] = None) -> List[Pattern]:
        """Get patterns with optional filters

        Args:
            analysis_run_id: Filter by analysis run
            pattern_type: Filter by pattern type
            min_severity: Minimum severity threshold

        Returns:
            List[Pattern]: List of pattern entities
        """
        query = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if analysis_run_id:
            query += " AND analysis_run_id = ?"
            params.append(analysis_run_id)

        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)

        if min_severity is not None:
            query += " AND severity >= ?"
            params.append(min_severity)

        query += " ORDER BY severity DESC, detected_at DESC"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [Pattern(**dict(row)) for row in rows]

    # === Feature Cache Operations ===

    def get_cached_features(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached features for text

        Args:
            text: Message text

        Returns:
            Optional[Dict]: Cached features or None
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM feature_cache WHERE text_hash = ?",
                (text_hash,)
            )
            row = cursor.fetchone()

            if row:
                features = {}
                if row['vader_json']:
                    features['vader'] = json.loads(row['vader_json'])
                if row['nrclex_json']:
                    features['nrclex'] = json.loads(row['nrclex_json'])
                if row['textblob_json']:
                    features['textblob'] = json.loads(row['textblob_json'])
                if row['patterns_json']:
                    features['patterns'] = json.loads(row['patterns_json'])
                return features

        return None

    def cache_features(self, text: str, features: Dict[str, Any]):
        """Cache features for text

        Args:
            text: Message text
            features: Features dictionary
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO feature_cache
                   (text_hash, vader_json, nrclex_json, textblob_json, patterns_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (text_hash,
                 json.dumps(features.get('vader')) if 'vader' in features else None,
                 json.dumps(features.get('nrclex')) if 'nrclex' in features else None,
                 json.dumps(features.get('textblob')) if 'textblob' in features else None,
                 json.dumps(features.get('patterns')) if 'patterns' in features else None)
            )

    # === Utility Methods ===

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file

        Args:
            file_path: Path to file

        Returns:
            str: File hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics

        Returns:
            Dict: Statistics dictionary
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Count tables
            for table in ['messages', 'speakers', 'analysis_runs', 'patterns']:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()['count']

            # Database file size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

            # Last analysis run
            cursor.execute(
                "SELECT run_timestamp FROM analysis_runs ORDER BY run_timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                stats['last_analysis'] = row['run_timestamp']

            return stats

    def clear_all_data(self):
        """Clear all data from database (use with caution)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete in correct order due to foreign keys
            tables = ['patterns', 'timeline_bins', 'risk_assessments',
                     'messages', 'speakers', 'analysis_runs', 'feature_cache']

            for table in tables:
                cursor.execute(f"DELETE FROM {table}")

            # Reset autoincrement counters - use parameterized query for security
            for table in tables:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))

            logger.warning("All data cleared from database")