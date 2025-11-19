"""
Digital Footprint Analyzer Module
Analyzes messaging patterns, frequency analysis, response time patterns,
platform switching, privacy-seeking behavior, and evidence hiding indicators.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MessagingPattern:
    """Container for messaging pattern analysis"""
    pattern_type: str
    description: str
    frequency: float
    intensity: float  # 0-1 scale
    risk_score: float


@dataclass
class DigitalFootprintAnalysis:
    """Complete digital footprint analysis results"""
    messaging_frequency: Dict[str, float] = field(default_factory=dict)
    response_time_analysis: Dict[str, float] = field(default_factory=dict)
    platform_patterns: Dict[str, float] = field(default_factory=dict)
    privacy_seeking_indicators: bool = False
    evidence_hiding_indicators: bool = False
    timestamp_patterns: List[str] = field(default_factory=list)
    conversation_intensity: float = 0.0
    deletion_frequency_markers: List[str] = field(default_factory=list)
    pattern_anomalies: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_level: str = "low"
    behavioral_insights: Dict[str, Any] = field(default_factory=dict)


class DigitalFootprintAnalyzer:
    """Analyzes digital behavior patterns and footprint"""

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.20,
        "moderate": 0.40,
        "high": 0.65,
        "critical": 0.85
    }

    # Privacy-seeking keywords
    PRIVACY_SEEKING_KEYWORDS = [
        "delete", "remove", "erase", "clear", "unrecoverable",
        "burn after reading", "disappear", "gone", "permanent",
        "proof", "evidence", "trace", "screenshot", "saved",
        "recorded", "secret", "hidden", "private", "no one knows"
    ]

    # Evidence hiding patterns
    EVIDENCE_HIDING_PATTERNS = [
        r"\b(screenshot|save|record|capture)\b",
        r"\b(delete (it|this|everything)|erase|remove|clear)\b",
        r"\b(don't (tell|show|save|send|share)|please delete)\b",
        r"\b(disappear|self-destruct|burn|gone|unrecoverable)\b",
        r"\b(no one (can|should) (know|see|find))\b"
    ]

    def __init__(self):
        """Initialize digital footprint analyzer"""
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for efficiency

        Returns:
            List: Compiled patterns
        """
        compiled = []
        for pattern_str in self.EVIDENCE_HIDING_PATTERNS:
            try:
                compiled.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                logger.error(f"Invalid regex pattern: {pattern_str} - {e}")
        return compiled

    def analyze_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> DigitalFootprintAnalysis:
        """Analyze digital footprint from message collection

        Args:
            messages: List of message dictionaries with text, timestamp, sender

        Returns:
            DigitalFootprintAnalysis: Analysis results
        """
        analysis = DigitalFootprintAnalysis()

        if not messages:
            return analysis

        # Analyze messaging frequency
        analysis.messaging_frequency = self._analyze_messaging_frequency(messages)

        # Analyze response times
        analysis.response_time_analysis = self._analyze_response_times(messages)

        # Analyze platform patterns
        analysis.platform_patterns = self._analyze_platform_usage(messages)

        # Detect privacy-seeking behavior
        analysis.privacy_seeking_indicators = self._detect_privacy_seeking(messages)

        # Detect evidence hiding
        analysis.evidence_hiding_indicators = self._detect_evidence_hiding(messages)

        # Analyze timestamp patterns
        analysis.timestamp_patterns = self._analyze_timestamp_patterns(messages)

        # Calculate conversation intensity
        analysis.conversation_intensity = self._calculate_conversation_intensity(messages)

        # Identify deletion frequency markers
        analysis.deletion_frequency_markers = self._identify_deletion_markers(messages)

        # Identify pattern anomalies
        analysis.pattern_anomalies = self._identify_anomalies(messages)

        # Calculate behavioral insights
        analysis.behavioral_insights = self._generate_behavioral_insights(
            analysis, messages
        )

        # Calculate overall risk score
        analysis.risk_score = self._calculate_risk_score(analysis)
        analysis.risk_level = self._determine_risk_level(analysis.risk_score)

        return analysis

    def _analyze_messaging_frequency(self, messages: List[Dict]) -> Dict[str, float]:
        """Analyze messaging frequency patterns

        Args:
            messages: List of messages

        Returns:
            Dict: Frequency metrics
        """
        if not messages:
            return {}

        # Group by sender
        by_sender = {}
        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            if sender not in by_sender:
                by_sender[sender] = []
            by_sender[sender].append(msg)

        frequency_metrics = {}

        for sender, sender_messages in by_sender.items():
            total_messages = len(sender_messages)

            # Calculate message frequency
            if total_messages == 0:
                continue

            # Average per message metric
            frequency_metrics[f"{sender}_total_messages"] = total_messages
            frequency_metrics[f"{sender}_avg_message_length"] = (
                sum(len(m.get('text', '')) for m in sender_messages) / total_messages
            )

            # Calculate if messages are clustered (sent in rapid succession)
            timestamps = []
            for msg in sender_messages:
                ts = msg.get('timestamp')
                if ts:
                    if isinstance(ts, str):
                        try:
                            timestamps.append(datetime.fromisoformat(ts))
                        except ValueError:
                            pass
                    elif isinstance(ts, (int, float)):
                        timestamps.append(datetime.fromtimestamp(ts))

            if len(timestamps) > 1:
                timestamps.sort()
                time_diffs = [
                    (timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]

                if time_diffs:
                    avg_interval = statistics.mean(time_diffs)
                    frequency_metrics[f"{sender}_avg_interval_seconds"] = avg_interval

                    # Detect clustering (rapid succession)
                    rapid_intervals = sum(1 for t in time_diffs if t < 60)
                    frequency_metrics[f"{sender}_rapid_succession_count"] = rapid_intervals

        return frequency_metrics

    def _analyze_response_times(self, messages: List[Dict]) -> Dict[str, float]:
        """Analyze response time patterns between parties

        Args:
            messages: List of messages

        Returns:
            Dict: Response time metrics
        """
        response_metrics = {}

        if not messages or len(messages) < 2:
            return response_metrics

        # Get unique senders
        senders = list(set(m.get('sender', 'Unknown') for m in messages))

        if len(senders) < 2:
            return response_metrics

        # Analyze response times between alternating speakers
        response_times = []

        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            # Only count if different senders
            if current_msg.get('sender') != next_msg.get('sender'):
                current_ts = current_msg.get('timestamp')
                next_ts = next_msg.get('timestamp')

                if current_ts and next_ts:
                    try:
                        if isinstance(current_ts, str):
                            current_ts = datetime.fromisoformat(current_ts)
                        elif isinstance(current_ts, (int, float)):
                            current_ts = datetime.fromtimestamp(current_ts)

                        if isinstance(next_ts, str):
                            next_ts = datetime.fromisoformat(next_ts)
                        elif isinstance(next_ts, (int, float)):
                            next_ts = datetime.fromtimestamp(next_ts)

                        response_time = (next_ts - current_ts).total_seconds()
                        if response_time >= 0:
                            response_times.append(response_time)
                    except (ValueError, TypeError):
                        pass

        if response_times:
            response_metrics["average_response_time_seconds"] = statistics.mean(response_times)
            response_metrics["median_response_time_seconds"] = statistics.median(response_times)
            response_metrics["min_response_time_seconds"] = min(response_times)
            response_metrics["max_response_time_seconds"] = max(response_times)

            # Detect very fast responses (potential indicator of close proximity or monitoring)
            fast_responses = sum(1 for t in response_times if t < 30)
            response_metrics["rapid_response_count"] = fast_responses

            # Detect delayed responses (potential evasion)
            delayed_responses = sum(1 for t in response_times if t > 3600)
            response_metrics["delayed_response_count"] = delayed_responses

        return response_metrics

    def _analyze_platform_usage(self, messages: List[Dict]) -> Dict[str, float]:
        """Analyze platform switching patterns

        Args:
            messages: List of messages

        Returns:
            Dict: Platform usage metrics
        """
        platform_metrics = {}

        platforms = {}
        platform_switches = 0

        for i, msg in enumerate(messages):
            platform = msg.get('platform', 'unknown')

            platforms[platform] = platforms.get(platform, 0) + 1

            # Count platform switches
            if i > 0:
                prev_platform = messages[i-1].get('platform', 'unknown')
                if platform != prev_platform:
                    platform_switches += 1

        # Calculate platform distribution
        total_messages = len(messages)
        for platform, count in platforms.items():
            platform_metrics[f"{platform}_percentage"] = count / total_messages if total_messages > 0 else 0

        platform_metrics["platform_switch_count"] = platform_switches
        platform_metrics["unique_platforms"] = len(platforms)

        # Detect suspicious platform switching
        if len(platforms) > 2 and platform_switches > (total_messages * 0.2):
            platform_metrics["suspicious_switching"] = True
        else:
            platform_metrics["suspicious_switching"] = False

        return platform_metrics

    def _detect_privacy_seeking(self, messages: List[Dict]) -> bool:
        """Detect privacy-seeking behavior

        Args:
            messages: List of messages

        Returns:
            bool: Whether privacy-seeking indicators detected
        """
        for msg in messages:
            text = msg.get('text', '').lower()

            for keyword in self.PRIVACY_SEEKING_KEYWORDS:
                if keyword in text:
                    return True

        return False

    def _detect_evidence_hiding(self, messages: List[Dict]) -> bool:
        """Detect evidence hiding indicators

        Args:
            messages: List of messages

        Returns:
            bool: Whether evidence hiding detected
        """
        for msg in messages:
            text = msg.get('text', '')

            for pattern in self.compiled_patterns:
                if pattern.search(text):
                    return True

        return False

    def _analyze_timestamp_patterns(self, messages: List[Dict]) -> List[str]:
        """Analyze timestamp patterns

        Args:
            messages: List of messages

        Returns:
            List: Timestamp pattern observations
        """
        patterns = []

        if not messages:
            return patterns

        # Extract hours from timestamps
        hours = []
        for msg in messages:
            ts = msg.get('timestamp')
            if ts:
                try:
                    dt = None
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts)
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts)
                    if dt:
                        hours.append(dt.hour)
                except (ValueError, TypeError):
                    pass

        if hours:
            # Detect nocturnal activity (late night messages)
            late_night_messages = sum(1 for h in hours if h >= 23 or h < 5)
            if late_night_messages > len(hours) * 0.3:
                patterns.append("High frequency of late-night messages")

            # Detect daytime dominance
            daytime_messages = sum(1 for h in hours if 9 <= h < 17)
            if daytime_messages > len(hours) * 0.6:
                patterns.append("Predominantly daytime communication")

            # Detect odd hour patterns
            if set(hours) and min(hours) == max(hours):
                patterns.append("All messages sent within same hour")

        return patterns

    def _calculate_conversation_intensity(self, messages: List[Dict]) -> float:
        """Calculate conversation intensity (0-1 scale)

        Args:
            messages: List of messages

        Returns:
            float: Intensity score
        """
        if not messages or len(messages) < 2:
            return 0.0

        # Extract timestamps
        timestamps = []
        for msg in messages:
            ts = msg.get('timestamp')
            if ts:
                try:
                    if isinstance(ts, str):
                        timestamps.append(datetime.fromisoformat(ts))
                    elif isinstance(ts, (int, float)):
                        timestamps.append(datetime.fromtimestamp(ts))
                except (ValueError, TypeError):
                    pass

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort()

        # Calculate time span
        time_span = (timestamps[-1] - timestamps[0]).total_seconds()

        if time_span == 0:
            # All messages in same second = extremely intense
            return 1.0

        # Messages per hour
        hours = time_span / 3600
        messages_per_hour = len(messages) / max(hours, 0.1)

        # Normalize to 0-1 scale (assuming 50 messages/hour is intense)
        intensity = min(1.0, messages_per_hour / 50)

        return intensity

    def _identify_deletion_markers(self, messages: List[Dict]) -> List[str]:
        """Identify markers of message deletion or hiding

        Args:
            messages: List of messages

        Returns:
            List: Deletion markers
        """
        markers = []

        deletion_keywords = [
            "delete", "deleted", "remove", "removed",
            "erase", "erased", "clear", "cleared"
        ]

        for msg in messages:
            text = msg.get('text', '').lower()

            for keyword in deletion_keywords:
                if keyword in text:
                    markers.append(f"Message contains deletion keyword: {keyword}")

        # Detect gaps in timestamps
        timestamps = []
        for msg in messages:
            ts = msg.get('timestamp')
            if ts:
                try:
                    if isinstance(ts, str):
                        timestamps.append(datetime.fromisoformat(ts))
                    elif isinstance(ts, (int, float)):
                        timestamps.append(datetime.fromtimestamp(ts))
                except (ValueError, TypeError):
                    pass

        if len(timestamps) > 1:
            timestamps.sort()
            time_diffs = [
                (timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]

            # Detect large gaps (potential deletions)
            large_gaps = sum(1 for t in time_diffs if t > 86400)  # > 24 hours
            if large_gaps > 0:
                markers.append(f"Large timestamp gaps detected ({large_gaps} gaps > 24 hours)")

        return markers

    def _identify_anomalies(self, messages: List[Dict]) -> List[str]:
        """Identify behavioral anomalies

        Args:
            messages: List of messages

        Returns:
            List: Anomaly descriptions
        """
        anomalies = []

        # Message length anomalies
        lengths = [len(m.get('text', '')) for m in messages]
        if lengths:
            avg_length = statistics.mean(lengths)
            std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0

            if std_dev > avg_length:
                anomalies.append("Highly variable message lengths (potential encryption or obfuscation)")

        # Sender ratio anomalies
        senders = {}
        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            senders[sender] = senders.get(sender, 0) + 1

        if len(senders) == 2:
            counts = list(senders.values())
            ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')

            if ratio > 5:
                anomalies.append("Highly imbalanced messaging ratio (one party dominates conversation)")

        return anomalies

    def _generate_behavioral_insights(
        self,
        analysis: DigitalFootprintAnalysis,
        messages: List[Dict]
    ) -> Dict[str, Any]:
        """Generate behavioral insights from patterns

        Args:
            analysis: Analysis results
            messages: Original messages

        Returns:
            Dict: Behavioral insights
        """
        insights = {}

        # Communication style insights
        if analysis.conversation_intensity > 0.7:
            insights["communication_style"] = "Very intense/rapid communication"
            insights["style_risk"] = "Potential obsessive or controlling behavior"
        elif analysis.conversation_intensity > 0.3:
            insights["communication_style"] = "Moderately active communication"
            insights["style_risk"] = "Normal engagement levels"
        else:
            insights["communication_style"] = "Sparse communication"
            insights["style_risk"] = "Possible avoidance or disconnection"

        # Privacy concerns
        if analysis.evidence_hiding_indicators:
            insights["privacy_concern"] = "Evidence hiding behavior detected"
        if analysis.privacy_seeking_indicators:
            insights["privacy_awareness"] = "Privacy-conscious behavior detected"

        # Platform behavior
        if analysis.platform_patterns.get("suspicious_switching"):
            insights["platform_behavior"] = "Frequent platform switching detected"
            insights["platform_risk"] = "Potential evasion or operational security concerns"

        # Response pattern insights
        response_time = analysis.response_time_analysis.get("average_response_time_seconds", 0)
        if response_time < 60:
            insights["response_pattern"] = "Very rapid response times"
            insights["response_risk"] = "Potential constant monitoring or proximity"
        elif response_time < 3600:
            insights["response_pattern"] = "Quick response times"
        else:
            insights["response_pattern"] = "Delayed response times"

        return insights

    def _calculate_risk_score(self, analysis: DigitalFootprintAnalysis) -> float:
        """Calculate overall digital footprint risk score

        Args:
            analysis: Analysis results

        Returns:
            float: Risk score (0-1)
        """
        risk = 0.0

        # Evidence hiding is highest risk indicator
        if analysis.evidence_hiding_indicators:
            risk += 0.3

        # Privacy seeking behavior
        if analysis.privacy_seeking_indicators:
            risk += 0.15

        # Conversation intensity contribution
        risk += analysis.conversation_intensity * 0.2

        # Platform switching
        if analysis.platform_patterns.get("suspicious_switching"):
            risk += 0.15

        # Pattern anomalies
        risk += (len(analysis.pattern_anomalies) * 0.1)

        # Deletion markers
        risk += (len(analysis.deletion_frequency_markers) * 0.05)

        return min(1.0, risk)

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score

        Args:
            score: Risk score (0-1)

        Returns:
            str: Risk level
        """
        for level in ["critical", "high", "moderate", "low"]:
            if score >= self.RISK_THRESHOLDS[level]:
                return level
        return "low"

    def analyze_temporal_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in messaging

        Args:
            messages: List of messages

        Returns:
            Dict: Temporal analysis
        """
        analysis = {
            "total_messages": len(messages),
            "timespan_days": 0,
            "messages_per_day": 0,
            "active_hours": [],
            "peak_activity_hour": None,
            "communication_consistency": "unknown"
        }

        if len(messages) < 2:
            return analysis

        # Extract timestamps
        timestamps = []
        for msg in messages:
            ts = msg.get('timestamp')
            if ts:
                try:
                    if isinstance(ts, str):
                        timestamps.append(datetime.fromisoformat(ts))
                    elif isinstance(ts, (int, float)):
                        timestamps.append(datetime.fromtimestamp(ts))
                except (ValueError, TypeError):
                    pass

        if len(timestamps) < 2:
            return analysis

        timestamps.sort()

        # Calculate timespan
        timespan = timestamps[-1] - timestamps[0]
        analysis["timespan_days"] = timespan.days

        if timespan.days > 0:
            analysis["messages_per_day"] = len(messages) / max(timespan.days, 1)

        # Analyze active hours
        hours = [ts.hour for ts in timestamps]
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        if hour_counts:
            analysis["active_hours"] = sorted(hour_counts.keys())
            analysis["peak_activity_hour"] = max(hour_counts, key=hour_counts.get)

        # Assess consistency
        if len(hour_counts) <= 2:
            analysis["communication_consistency"] = "highly_consistent"
        elif len(hour_counts) <= 5:
            analysis["communication_consistency"] = "moderately_consistent"
        else:
            analysis["communication_consistency"] = "irregular"

        return analysis
