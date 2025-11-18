"""
Timestamp Integrity Validator
Validates timestamp consistency and format for temporal analysis
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TimestampValidationResult:
    """Results of timestamp validation"""
    is_valid: bool = False
    total_messages: int = 0
    messages_with_timestamps: int = 0
    coverage_percentage: float = 0.0
    timestamp_formats: List[str] = field(default_factory=list)
    dominant_format: Optional[str] = None
    format_consistency: float = 0.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    timespan_days: float = 0.0
    gaps_detected: int = 0
    large_gaps: List[Dict] = field(default_factory=list)  # Gaps > 7 days
    can_use_temporal_analysis: bool = False


class TimestampValidator:
    """Validates timestamp integrity for temporal analysis"""

    # Minimum coverage required for temporal analysis
    MIN_COVERAGE_THRESHOLD = 0.90  # 90% of messages must have timestamps

    # Minimum format consistency required
    MIN_FORMAT_CONSISTENCY = 0.95  # 95% of timestamps must use same format

    # Maximum gap (in days) before flagging as suspicious
    LARGE_GAP_THRESHOLD_DAYS = 7

    def __init__(self):
        """Initialize timestamp validator"""
        pass

    def validate_timestamps(self, messages: List[Dict[str, Any]]) -> TimestampValidationResult:
        """Validate timestamp integrity across messages

        Args:
            messages: List of messages with potential 'timestamp', 'date', 'time' fields

        Returns:
            TimestampValidationResult: Validation results
        """
        result = TimestampValidationResult()

        if not messages:
            result.issues.append("No messages provided")
            return result

        result.total_messages = len(messages)

        # Extract and analyze timestamps
        timestamps = []
        timestamp_formats = []
        missing_count = 0

        for i, msg in enumerate(messages):
            timestamp = self._extract_timestamp(msg)

            if timestamp:
                timestamps.append((i, timestamp))

                # Determine format
                fmt = self._detect_timestamp_format(timestamp)
                if fmt:
                    timestamp_formats.append(fmt)
            else:
                missing_count += 1

        result.messages_with_timestamps = len(timestamps)
        result.coverage_percentage = (len(timestamps) / result.total_messages) * 100

        # Check coverage
        if result.coverage_percentage < self.MIN_COVERAGE_THRESHOLD * 100:
            result.issues.append(
                f"Insufficient timestamp coverage: {result.coverage_percentage:.1f}% "
                f"(minimum: {self.MIN_COVERAGE_THRESHOLD * 100}%)"
            )
            result.issues.append(f"{missing_count} messages missing timestamps")
        elif missing_count > 0:
            result.warnings.append(f"{missing_count} messages have missing timestamps")

        # If we have no timestamps at all, we're done
        if not timestamps:
            result.issues.append("No valid timestamps found in any message")
            return result

        # Analyze timestamp formats
        if timestamp_formats:
            format_counts = Counter(timestamp_formats)
            result.timestamp_formats = list(format_counts.keys())
            result.dominant_format = format_counts.most_common(1)[0][0]

            # Calculate format consistency
            dominant_count = format_counts[result.dominant_format]
            result.format_consistency = dominant_count / len(timestamp_formats)

            if result.format_consistency < self.MIN_FORMAT_CONSISTENCY:
                result.issues.append(
                    f"Inconsistent timestamp formats: {result.format_consistency:.1%} consistency "
                    f"(minimum: {self.MIN_FORMAT_CONSISTENCY:.0%})"
                )
                result.issues.append(f"Formats detected: {result.timestamp_formats}")
            elif len(result.timestamp_formats) > 1:
                result.warnings.append(
                    f"Multiple timestamp formats detected: {result.timestamp_formats}"
                )

        # Analyze timestamp range and gaps
        sorted_timestamps = sorted(timestamps, key=lambda x: x[1])
        result.earliest_timestamp = sorted_timestamps[0][1]
        result.latest_timestamp = sorted_timestamps[-1][1]
        result.timespan_days = (result.latest_timestamp - result.earliest_timestamp).total_seconds() / 86400

        # Detect gaps
        gaps = self._detect_gaps(sorted_timestamps)
        result.gaps_detected = len(gaps)
        result.large_gaps = [g for g in gaps if g['days'] > self.LARGE_GAP_THRESHOLD_DAYS]

        if result.large_gaps:
            result.warnings.append(
                f"Found {len(result.large_gaps)} large gaps (>{self.LARGE_GAP_THRESHOLD_DAYS} days) in conversation"
            )

        # Determine if temporal analysis can be used
        result.can_use_temporal_analysis = (
            len(result.issues) == 0 and
            result.coverage_percentage >= self.MIN_COVERAGE_THRESHOLD * 100 and
            result.format_consistency >= self.MIN_FORMAT_CONSISTENCY
        )

        result.is_valid = result.can_use_temporal_analysis

        return result

    def _extract_timestamp(self, msg: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from message

        Tries multiple fields: 'timestamp', 'datetime', 'date'+'time'

        Args:
            msg: Message dictionary

        Returns:
            datetime or None
        """
        # Try direct timestamp field
        if 'timestamp' in msg and msg['timestamp']:
            return self._parse_timestamp(msg['timestamp'])

        # Try datetime field
        if 'datetime' in msg and msg['datetime']:
            return self._parse_timestamp(msg['datetime'])

        # Try combining date and time fields
        if 'date' in msg and 'time' in msg:
            date_val = msg.get('date')
            time_val = msg.get('time')

            if date_val and time_val:
                try:
                    combined = f"{date_val} {time_val}"
                    return self._parse_timestamp(combined)
                except:
                    pass

        # Try date field alone (set time to midnight)
        if 'date' in msg and msg['date']:
            return self._parse_timestamp(msg['date'])

        return None

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse timestamp from various formats

        Args:
            timestamp: Timestamp value (datetime, string, etc.)

        Returns:
            datetime or None
        """
        # Already a datetime object
        if isinstance(timestamp, datetime):
            return timestamp

        # String parsing
        if isinstance(timestamp, str):
            # Common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue

        return None

    def _detect_timestamp_format(self, timestamp: datetime) -> str:
        """Detect format category of timestamp

        Args:
            timestamp: datetime object

        Returns:
            Format category string
        """
        # For datetime objects, we categorize based on precision
        if timestamp.microsecond > 0:
            return "microsecond_precision"
        elif timestamp.second > 0:
            return "second_precision"
        elif timestamp.hour > 0 or timestamp.minute > 0:
            return "minute_precision"
        else:
            return "date_only"

    def _detect_gaps(self, sorted_timestamps: List[tuple]) -> List[Dict]:
        """Detect gaps between consecutive timestamps

        Args:
            sorted_timestamps: List of (index, datetime) tuples, sorted by datetime

        Returns:
            List of gap dictionaries
        """
        gaps = []

        for i in range(len(sorted_timestamps) - 1):
            idx1, ts1 = sorted_timestamps[i]
            idx2, ts2 = sorted_timestamps[i + 1]

            gap_seconds = (ts2 - ts1).total_seconds()
            gap_days = gap_seconds / 86400

            # Record any gap > 1 hour
            if gap_seconds > 3600:
                gaps.append({
                    'from_index': idx1,
                    'to_index': idx2,
                    'from_time': ts1,
                    'to_time': ts2,
                    'gap_seconds': gap_seconds,
                    'days': gap_days,
                    'description': self._format_gap_duration(gap_seconds)
                })

        return gaps

    def _format_gap_duration(self, seconds: float) -> str:
        """Format gap duration in human-readable form

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string
        """
        if seconds < 3600:
            return f"{seconds / 60:.0f} minutes"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f} hours"
        elif seconds < 604800:  # 1 week
            return f"{seconds / 86400:.1f} days"
        else:
            return f"{seconds / 604800:.1f} weeks"

    def generate_report(self, result: TimestampValidationResult) -> str:
        """Generate human-readable validation report

        Args:
            result: Validation result

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("TIMESTAMP VALIDATION REPORT")
        lines.append("=" * 70)

        # Summary
        status = "✅ PASS" if result.is_valid else "❌ FAIL"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Can use temporal analysis: {'YES' if result.can_use_temporal_analysis else 'NO'}")

        # Coverage
        lines.append(f"\n📊 Coverage:")
        lines.append(f"  Total messages: {result.total_messages}")
        lines.append(f"  Messages with timestamps: {result.messages_with_timestamps}")
        lines.append(f"  Coverage: {result.coverage_percentage:.1f}%")

        # Format
        if result.timestamp_formats:
            lines.append(f"\n🕐 Format:")
            lines.append(f"  Dominant format: {result.dominant_format}")
            lines.append(f"  Format consistency: {result.format_consistency:.1%}")
            if len(result.timestamp_formats) > 1:
                lines.append(f"  All formats: {', '.join(result.timestamp_formats)}")

        # Timespan
        if result.earliest_timestamp and result.latest_timestamp:
            lines.append(f"\n📅 Timespan:")
            lines.append(f"  Earliest: {result.earliest_timestamp}")
            lines.append(f"  Latest: {result.latest_timestamp}")
            lines.append(f"  Duration: {result.timespan_days:.1f} days")

        # Gaps
        if result.gaps_detected > 0:
            lines.append(f"\n⚠️  Gaps:")
            lines.append(f"  Total gaps > 1 hour: {result.gaps_detected}")
            if result.large_gaps:
                lines.append(f"  Large gaps (>{self.LARGE_GAP_THRESHOLD_DAYS} days): {len(result.large_gaps)}")
                for gap in result.large_gaps[:5]:  # Show first 5
                    lines.append(f"    - {gap['description']} (msgs {gap['from_index']} → {gap['to_index']})")

        # Issues
        if result.issues:
            lines.append(f"\n❌ Issues ({len(result.issues)}):")
            for issue in result.issues:
                lines.append(f"  - {issue}")

        # Warnings
        if result.warnings:
            lines.append(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
