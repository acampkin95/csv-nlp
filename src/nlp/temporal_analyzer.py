"""
Temporal Pattern Analysis Module
Detects escalation, pattern progression, and frequency changes over time
Requires timestamp validation before use
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TemporalWindow:
    """Time window with aggregated metrics"""
    window_start: datetime
    window_end: datetime
    message_count: int = 0
    messages: List[Dict] = field(default_factory=list)
    avg_risk: float = 0.0
    max_risk: float = 0.0
    sentiment_mean: float = 0.0
    sentiment_variance: float = 0.0
    pattern_count: int = 0
    unique_senders: int = 0


@dataclass
class EscalationEvent:
    """Detected escalation event"""
    timestamp: datetime
    message_index: int
    escalation_type: str  # risk, sentiment, pattern
    from_value: float
    to_value: float
    change_percent: float
    severity: str  # mild, moderate, severe
    description: str


@dataclass
class TemporalAnalysisResult:
    """Complete temporal analysis results"""
    is_escalating: bool = False
    escalation_score: float = 0.0
    trend_direction: str = "stable"  # improving, declining, stable, volatile
    trend_slope: float = 0.0

    # Risk progression
    risk_progression: List[Dict] = field(default_factory=list)
    escalation_events: List[EscalationEvent] = field(default_factory=list)

    # Pattern progression (grooming stages, etc.)
    stage_progression: List[Dict] = field(default_factory=list)
    is_progressing_through_stages: bool = False

    # Frequency analysis
    message_frequency: Dict[str, float] = field(default_factory=dict)
    frequency_increasing: bool = False
    frequency_change_percent: float = 0.0

    # Time windows
    windows: List[TemporalWindow] = field(default_factory=list)

    # Warnings and insights
    warnings: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


class TemporalAnalyzer:
    """Analyzes patterns that evolve over time"""

    # Escalation thresholds
    MILD_ESCALATION_THRESHOLD = 0.15  # 15% increase
    MODERATE_ESCALATION_THRESHOLD = 0.30  # 30% increase
    SEVERE_ESCALATION_THRESHOLD = 0.50  # 50% increase

    # Frequency change threshold
    FREQUENCY_INCREASE_THRESHOLD = 0.50  # 50% increase in message frequency

    # Default window size for aggregation
    DEFAULT_WINDOW_SIZE_HOURS = 24

    def __init__(self, window_size_hours: int = DEFAULT_WINDOW_SIZE_HOURS):
        """Initialize temporal analyzer

        Args:
            window_size_hours: Size of time windows for aggregation
        """
        self.window_size = timedelta(hours=window_size_hours)

    def analyze_temporal_patterns(self, messages: List[Dict[str, Any]]) -> TemporalAnalysisResult:
        """Analyze temporal patterns across conversation

        Args:
            messages: List of messages with timestamps and analysis results

        Returns:
            TemporalAnalysisResult: Temporal analysis results
        """
        result = TemporalAnalysisResult()

        if not messages:
            result.warnings.append("No messages provided")
            return result

        # Filter messages with timestamps
        messages_with_ts = [m for m in messages if m.get('timestamp')]

        if len(messages_with_ts) < 2:
            result.warnings.append("Insufficient messages with timestamps for temporal analysis")
            return result

        # Sort by timestamp
        sorted_messages = sorted(messages_with_ts, key=lambda x: x['timestamp'])

        # Detect escalation
        escalation_analysis = self._analyze_escalation(sorted_messages)
        result.is_escalating = escalation_analysis['is_escalating']
        result.escalation_score = escalation_analysis['score']
        result.trend_direction = escalation_analysis['trend']
        result.trend_slope = escalation_analysis['slope']
        result.risk_progression = escalation_analysis['progression']
        result.escalation_events = escalation_analysis['events']

        # Detect pattern/stage progression
        stage_analysis = self._analyze_stage_progression(sorted_messages)
        result.stage_progression = stage_analysis['stages']
        result.is_progressing_through_stages = stage_analysis['is_progressing']

        # Analyze message frequency
        frequency_analysis = self._analyze_frequency(sorted_messages)
        result.message_frequency = frequency_analysis['metrics']
        result.frequency_increasing = frequency_analysis['increasing']
        result.frequency_change_percent = frequency_analysis['change_percent']

        # Create time windows
        result.windows = self._create_time_windows(sorted_messages)

        # Generate insights and warnings
        result.warnings.extend(self._generate_warnings(result))
        result.insights.extend(self._generate_insights(result))

        return result

    def _analyze_escalation(self, messages: List[Dict]) -> Dict:
        """Analyze if risks are escalating over time

        Args:
            messages: Sorted messages with timestamps

        Returns:
            Dict with escalation analysis
        """
        analysis = {
            'is_escalating': False,
            'score': 0.0,
            'trend': 'stable',
            'slope': 0.0,
            'progression': [],
            'events': []
        }

        # Extract risk scores over time
        risk_scores = []
        timestamps = []

        for i, msg in enumerate(messages):
            # Try to extract risk score from various fields
            risk = self._extract_risk_score(msg)
            if risk is not None:
                risk_scores.append(risk)
                timestamps.append(msg['timestamp'])
                analysis['progression'].append({
                    'timestamp': msg['timestamp'],
                    'index': i,
                    'risk': risk,
                    'message': msg.get('text', '')[:100]
                })

        if len(risk_scores) < 3:
            return analysis

        # Calculate trend using simple linear regression
        n = len(risk_scores)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(risk_scores)

        numerator = sum((x[i] - x_mean) * (risk_scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            analysis['slope'] = slope

            # Determine trend
            if slope > 0.01:  # Increasing
                analysis['trend'] = 'escalating'
                analysis['is_escalating'] = True
            elif slope < -0.01:  # Decreasing
                analysis['trend'] = 'de-escalating'
            else:
                analysis['trend'] = 'stable'

            # Calculate escalation score (0-1)
            # Higher slope and higher final risk = higher escalation score
            final_risk = risk_scores[-1]
            analysis['score'] = min(1.0, abs(slope) * 5 + final_risk * 0.3)

        # Detect escalation events (significant jumps in risk)
        for i in range(1, len(risk_scores)):
            from_risk = risk_scores[i - 1]
            to_risk = risk_scores[i]
            change = to_risk - from_risk

            if from_risk > 0:
                change_percent = (change / from_risk) * 100
            else:
                change_percent = 0

            # Significant increase
            if change_percent >= self.MILD_ESCALATION_THRESHOLD * 100:
                severity = self._classify_escalation_severity(change_percent / 100)

                event = EscalationEvent(
                    timestamp=timestamps[i],
                    message_index=i,
                    escalation_type='risk',
                    from_value=from_risk,
                    to_value=to_risk,
                    change_percent=change_percent,
                    severity=severity,
                    description=f"Risk increased by {change_percent:.0f}% ({from_risk:.2f} → {to_risk:.2f})"
                )
                analysis['events'].append(event)

        return analysis

    def _analyze_stage_progression(self, messages: List[Dict]) -> Dict:
        """Analyze progression through stages (e.g., grooming stages)

        Args:
            messages: Sorted messages

        Returns:
            Dict with stage progression analysis
        """
        analysis = {
            'stages': [],
            'is_progressing': False
        }

        # Track detected stages
        stages_detected = []

        for i, msg in enumerate(messages):
            # Extract stage information from pattern analysis
            stage = self._extract_stage_info(msg)

            if stage:
                stages_detected.append({
                    'index': i,
                    'timestamp': msg.get('timestamp'),
                    'stage': stage['name'],
                    'stage_number': stage['number'],
                    'severity': stage.get('severity', 0.0)
                })

        analysis['stages'] = stages_detected

        # Check if stages are progressing (increasing stage numbers)
        if len(stages_detected) >= 2:
            stage_numbers = [s['stage_number'] for s in stages_detected]

            # Check for monotonic increase (or at least non-decreasing)
            is_progressing = all(
                stage_numbers[i] <= stage_numbers[i + 1]
                for i in range(len(stage_numbers) - 1)
            )

            analysis['is_progressing'] = is_progressing

        return analysis

    def _analyze_frequency(self, messages: List[Dict]) -> Dict:
        """Analyze message frequency changes over time

        Args:
            messages: Sorted messages with timestamps

        Returns:
            Dict with frequency analysis
        """
        analysis = {
            'metrics': {},
            'increasing': False,
            'change_percent': 0.0
        }

        if len(messages) < 10:  # Need sufficient messages
            return analysis

        # Calculate messages per day
        first_ts = messages[0]['timestamp']
        last_ts = messages[-1]['timestamp']
        total_days = (last_ts - first_ts).total_seconds() / 86400

        if total_days < 1:
            total_days = 1  # Minimum 1 day

        avg_msgs_per_day = len(messages) / total_days
        analysis['metrics']['avg_messages_per_day'] = avg_msgs_per_day
        analysis['metrics']['total_days'] = total_days

        # Split into early and late periods
        mid_point = len(messages) // 2
        early_msgs = messages[:mid_point]
        late_msgs = messages[mid_point:]

        # Calculate frequency for each period
        early_days = (early_msgs[-1]['timestamp'] - early_msgs[0]['timestamp']).total_seconds() / 86400 or 1
        late_days = (late_msgs[-1]['timestamp'] - late_msgs[0]['timestamp']).total_seconds() / 86400 or 1

        early_frequency = len(early_msgs) / early_days
        late_frequency = len(late_msgs) / late_days

        analysis['metrics']['early_frequency'] = early_frequency
        analysis['metrics']['late_frequency'] = late_frequency

        # Calculate change
        if early_frequency > 0:
            change = (late_frequency - early_frequency) / early_frequency
            analysis['change_percent'] = change * 100
            analysis['increasing'] = change > self.FREQUENCY_INCREASE_THRESHOLD

        return analysis

    def _create_time_windows(self, messages: List[Dict]) -> List[TemporalWindow]:
        """Create time windows with aggregated metrics

        Args:
            messages: Sorted messages with timestamps

        Returns:
            List of TemporalWindow objects
        """
        if not messages:
            return []

        windows = []
        first_ts = messages[0]['timestamp']
        last_ts = messages[-1]['timestamp']

        current_start = first_ts

        while current_start <= last_ts:
            current_end = current_start + self.window_size

            # Collect messages in this window
            window_messages = [
                m for m in messages
                if current_start <= m['timestamp'] < current_end
            ]

            if window_messages:
                window = TemporalWindow(
                    window_start=current_start,
                    window_end=current_end,
                    message_count=len(window_messages),
                    messages=window_messages
                )

                # Calculate metrics
                risks = [self._extract_risk_score(m) for m in window_messages if self._extract_risk_score(m) is not None]
                if risks:
                    window.avg_risk = statistics.mean(risks)
                    window.max_risk = max(risks)

                sentiments = [self._extract_sentiment(m) for m in window_messages if self._extract_sentiment(m) is not None]
                if sentiments:
                    window.sentiment_mean = statistics.mean(sentiments)
                    if len(sentiments) > 1:
                        window.sentiment_variance = statistics.variance(sentiments)

                # Count unique senders
                senders = set(m.get('sender', 'Unknown') for m in window_messages)
                window.unique_senders = len(senders)

                # Count patterns
                window.pattern_count = sum(
                    1 for m in window_messages
                    if m.get('pattern_analysis') or m.get('patterns')
                )

                windows.append(window)

            current_start = current_end

        return windows

    def _extract_risk_score(self, message: Dict) -> Optional[float]:
        """Extract risk score from message

        Args:
            message: Message dictionary

        Returns:
            Risk score (0-1) or None
        """
        # Try various field names
        if 'risk_score' in message:
            return float(message['risk_score'])

        if 'risk_analysis' in message and isinstance(message['risk_analysis'], dict):
            return float(message['risk_analysis'].get('overall_risk', 0))

        if 'overall_risk' in message:
            return float(message['overall_risk'])

        # Try to extract from analysis object
        if 'analysis' in message and hasattr(message['analysis'], 'overall_score'):
            return float(message['analysis'].overall_score)

        return None

    def _extract_sentiment(self, message: Dict) -> Optional[float]:
        """Extract sentiment score from message

        Args:
            message: Message dictionary

        Returns:
            Sentiment score (-1 to 1) or None
        """
        if 'sentiment' in message:
            return float(message['sentiment'])

        if 'sentiment_analysis' in message and isinstance(message['sentiment_analysis'], dict):
            return float(message['sentiment_analysis'].get('compound', 0))

        return None

    def _extract_stage_info(self, message: Dict) -> Optional[Dict]:
        """Extract stage information from message

        Args:
            message: Message dictionary

        Returns:
            Stage info dict or None
        """
        # Map stage names to numbers for progression tracking
        stage_map = {
            'trust_building': 1,
            'isolation': 2,
            'boundary_testing': 3,
            'normalization': 4,
            'desensitization': 5,
            'control': 6
        }

        # Try to extract primary concern or stage
        stage_name = None

        if 'pattern_analysis' in message and isinstance(message['pattern_analysis'], dict):
            stage_name = message['pattern_analysis'].get('primary_concern')

        if 'analysis' in message and hasattr(message['analysis'], 'primary_concern'):
            stage_name = message['analysis'].primary_concern

        if stage_name and stage_name in stage_map:
            return {
                'name': stage_name,
                'number': stage_map[stage_name],
                'severity': self._extract_risk_score(message) or 0.0
            }

        return None

    def _classify_escalation_severity(self, change_percent: float) -> str:
        """Classify escalation severity

        Args:
            change_percent: Percentage change (as decimal, e.g., 0.30 for 30%)

        Returns:
            Severity level string
        """
        if change_percent >= self.SEVERE_ESCALATION_THRESHOLD:
            return "severe"
        elif change_percent >= self.MODERATE_ESCALATION_THRESHOLD:
            return "moderate"
        else:
            return "mild"

    def _generate_warnings(self, result: TemporalAnalysisResult) -> List[str]:
        """Generate warnings based on temporal analysis

        Args:
            result: Analysis result

        Returns:
            List of warning strings
        """
        warnings = []

        # Escalation warnings
        if result.is_escalating:
            if result.escalation_score > 0.7:
                warnings.append("⚠️  CRITICAL: Rapid risk escalation detected")
            elif result.escalation_score > 0.5:
                warnings.append("⚠️  WARNING: Significant risk escalation detected")

        # Severe escalation events
        severe_events = [e for e in result.escalation_events if e.severity == "severe"]
        if severe_events:
            warnings.append(f"⚠️  {len(severe_events)} severe escalation event(s) detected")

        # Stage progression warnings
        if result.is_progressing_through_stages and len(result.stage_progression) >= 3:
            warnings.append("⚠️  WARNING: Progression through multiple concerning stages detected")

        # Frequency warnings
        if result.frequency_increasing and result.frequency_change_percent > 100:
            warnings.append(f"⚠️  Message frequency increased by {result.frequency_change_percent:.0f}%")

        return warnings

    def _generate_insights(self, result: TemporalAnalysisResult) -> List[str]:
        """Generate insights from temporal analysis

        Args:
            result: Analysis result

        Returns:
            List of insight strings
        """
        insights = []

        # Trend insights
        if result.trend_direction == "improving":
            insights.append("✅ Risk levels are decreasing over time")
        elif result.trend_direction == "de-escalating":
            insights.append("✅ Conversation shows de-escalation pattern")

        # Frequency insights
        if result.message_frequency:
            freq = result.message_frequency.get('avg_messages_per_day', 0)
            if freq > 0:
                insights.append(f"📊 Average message frequency: {freq:.1f} messages/day")

        # Window insights
        if result.windows:
            peak_window = max(result.windows, key=lambda w: w.max_risk)
            if peak_window.max_risk > 0.6:
                insights.append(
                    f"📍 Peak risk period: {peak_window.window_start.strftime('%Y-%m-%d %H:%M')} "
                    f"(risk: {peak_window.max_risk:.2f})"
                )

        return insights
