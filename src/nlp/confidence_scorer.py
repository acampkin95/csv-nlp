"""
Confidence Scoring System
Provides confidence metrics for all NLP detections and improves accuracy through
ensemble methods, context-awareness, and speaker profiling.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Confidence score for a detection"""
    overall_confidence: float  # 0-1
    method_confidence: Dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0  # Multiple methods agreeing
    context_boost: float = 0.0  # Boost from context
    baseline_deviation: float = 0.0  # How much it deviates from baseline
    factors: List[str] = field(default_factory=list)


@dataclass
class SpeakerBaseline:
    """Baseline behavior profile for a speaker"""
    speaker_name: str
    message_count: int = 0
    avg_sentiment: float = 0.0
    sentiment_variance: float = 0.0
    avg_message_length: float = 0.0
    typical_risk_level: float = 0.0
    common_patterns: List[str] = field(default_factory=list)
    communication_style: str = "neutral"
    updated_at: Optional[str] = None


class ConfidenceScorer:
    """Calculates confidence scores for NLP detections"""

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.75
    MEDIUM_CONFIDENCE = 0.50
    LOW_CONFIDENCE = 0.25

    # Agreement bonus
    AGREEMENT_BONUS = 0.15  # 15% boost when multiple methods agree

    def __init__(self):
        """Initialize confidence scorer"""
        self.speaker_baselines: Dict[str, SpeakerBaseline] = {}

    def calculate_detection_confidence(
        self,
        detection_value: float,
        method_name: str,
        supporting_evidence: Optional[List] = None,
        context: Optional[Dict] = None
    ) -> ConfidenceScore:
        """Calculate confidence for a single detection

        Args:
            detection_value: The detection score (0-1)
            method_name: Name of detection method
            supporting_evidence: List of supporting evidence items
            context: Optional context information

        Returns:
            ConfidenceScore: Confidence score object
        """
        confidence = ConfidenceScore(overall_confidence=0.0)

        # Base confidence from detection value
        base_confidence = abs(detection_value)
        confidence.method_confidence[method_name] = base_confidence

        factors = []

        # Evidence strength
        if supporting_evidence:
            evidence_count = len(supporting_evidence)
            if evidence_count >= 3:
                base_confidence += 0.10
                factors.append(f"{evidence_count} pieces of evidence")
            elif evidence_count >= 1:
                base_confidence += 0.05
                factors.append(f"{evidence_count} evidence item(s)")

        # Context boost
        if context:
            context_boost = self._calculate_context_boost(context)
            confidence.context_boost = context_boost
            base_confidence += context_boost
            if context_boost > 0:
                factors.append(f"Context boost: +{context_boost:.2f}")

        confidence.overall_confidence = min(1.0, base_confidence)
        confidence.factors = factors

        return confidence

    def calculate_ensemble_confidence(
        self,
        detections: Dict[str, float],
        supporting_data: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate ensemble confidence from multiple detection methods

        Args:
            detections: Dict of method_name -> detection_score
            supporting_data: Additional supporting data per method

        Returns:
            ConfidenceScore: Ensemble confidence score
        """
        if not detections:
            return ConfidenceScore(overall_confidence=0.0)

        confidence = ConfidenceScore(overall_confidence=0.0)
        confidence.method_confidence = detections.copy()

        # Calculate base confidence (average of all methods)
        base_confidence = statistics.mean(detections.values())

        # Calculate agreement score
        agreement = self._calculate_agreement(detections)
        confidence.agreement_score = agreement

        # Agreement bonus (when multiple methods agree)
        agreement_bonus = 0.0
        if agreement > 0.7:  # High agreement
            agreement_bonus = self.AGREEMENT_BONUS
            confidence.factors.append("High agreement across methods")
        elif agreement > 0.5:  # Moderate agreement
            agreement_bonus = self.AGREEMENT_BONUS * 0.5
            confidence.factors.append("Moderate agreement across methods")

        # Combine scores
        confidence.overall_confidence = min(1.0, base_confidence + agreement_bonus)

        return confidence

    def _calculate_agreement(self, detections: Dict[str, float]) -> float:
        """Calculate agreement score between multiple detection methods

        Args:
            detections: Dict of detection scores

        Returns:
            float: Agreement score (0-1)
        """
        if len(detections) < 2:
            return 1.0  # Single method = perfect "agreement"

        scores = list(detections.values())

        # Calculate variance (lower variance = higher agreement)
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0

        # Convert variance to agreement (0 variance = 1.0 agreement)
        # Max variance is ~0.25 for scores in [0,1]
        agreement = max(0.0, 1.0 - (variance * 4))

        return agreement

    def _calculate_context_boost(self, context: Dict) -> float:
        """Calculate confidence boost from context

        Args:
            context: Context information

        Returns:
            float: Confidence boost (0-0.2)
        """
        boost = 0.0

        # Consecutive pattern matches
        if context.get('consecutive_matches', 0) >= 2:
            boost += 0.10

        # Pattern in multiple recent messages
        if context.get('recent_pattern_count', 0) >= 3:
            boost += 0.05

        # Escalating pattern
        if context.get('is_escalating'):
            boost += 0.05

        return min(0.2, boost)

    def build_speaker_baseline(self, speaker_name: str, messages: List[Dict]) -> SpeakerBaseline:
        """Build baseline behavioral profile for a speaker

        Args:
            speaker_name: Speaker name
            messages: List of messages from this speaker

        Returns:
            SpeakerBaseline: Baseline profile
        """
        baseline = SpeakerBaseline(speaker_name=speaker_name)

        if not messages:
            return baseline

        baseline.message_count = len(messages)

        # Calculate sentiment baseline
        sentiments = []
        message_lengths = []
        risk_scores = []
        patterns_found = defaultdict(int)

        for msg in messages:
            # Extract sentiment
            if 'sentiment' in msg:
                sentiments.append(msg['sentiment'])
            elif 'sentiment_analysis' in msg:
                sent = msg['sentiment_analysis']
                if isinstance(sent, dict):
                    sentiments.append(sent.get('compound', 0))

            # Extract message length
            if 'text' in msg:
                message_lengths.append(len(msg['text']))

            # Extract risk score
            if 'risk_score' in msg:
                risk_scores.append(msg['risk_score'])

            # Extract patterns
            if 'patterns' in msg:
                for pattern in msg['patterns']:
                    patterns_found[pattern] += 1

        # Calculate statistics
        if sentiments:
            baseline.avg_sentiment = statistics.mean(sentiments)
            if len(sentiments) > 1:
                baseline.sentiment_variance = statistics.variance(sentiments)

        if message_lengths:
            baseline.avg_message_length = statistics.mean(message_lengths)

        if risk_scores:
            baseline.typical_risk_level = statistics.mean(risk_scores)

        # Most common patterns
        if patterns_found:
            baseline.common_patterns = [
                pattern for pattern, count in
                sorted(patterns_found.items(), key=lambda x: x[1], reverse=True)[:5]
            ]

        # Store baseline
        self.speaker_baselines[speaker_name] = baseline
        logger.debug(f"Built baseline for {speaker_name}: {baseline.message_count} messages")

        return baseline

    def detect_baseline_deviation(
        self,
        speaker_name: str,
        current_value: float,
        metric: str = 'risk'
    ) -> Tuple[bool, float, str]:
        """Detect if current value deviates significantly from speaker baseline

        Args:
            speaker_name: Speaker name
            current_value: Current metric value
            metric: Metric type ('risk', 'sentiment', etc.)

        Returns:
            Tuple of (is_anomaly, deviation_score, description)
        """
        baseline = self.speaker_baselines.get(speaker_name)

        if not baseline or baseline.message_count < 3:
            # Not enough data for baseline
            return False, 0.0, "Insufficient baseline data"

        # Get baseline value for metric
        if metric == 'risk':
            baseline_value = baseline.typical_risk_level
            threshold = 0.3  # 30% deviation
        elif metric == 'sentiment':
            baseline_value = baseline.avg_sentiment
            threshold = 0.4  # 40% deviation
        else:
            return False, 0.0, "Unknown metric"

        if baseline_value == 0:
            return False, 0.0, "No baseline established"

        # Calculate deviation
        deviation = abs(current_value - baseline_value) / (abs(baseline_value) + 0.1)

        is_anomaly = deviation > threshold

        if is_anomaly:
            direction = "higher" if current_value > baseline_value else "lower"
            description = f"{metric.capitalize()} {deviation*100:.0f}% {direction} than baseline"
        else:
            description = "Within normal range"

        return is_anomaly, deviation, description


class ContextAwareAnalyzer:
    """Provides context-aware analysis considering surrounding messages"""

    CONTEXT_WINDOW = 3  # Number of messages before/after to consider

    def __init__(self):
        """Initialize context-aware analyzer"""
        pass

    def analyze_with_context(
        self,
        message_index: int,
        messages: List[Dict],
        analysis_func: callable
    ) -> Tuple[Any, Dict]:
        """Analyze message with context from surrounding messages

        Args:
            message_index: Index of message to analyze
            messages: Full list of messages
            analysis_func: Analysis function to call

        Returns:
            Tuple of (analysis_result, context_info)
        """
        # Get context window
        start_idx = max(0, message_index - self.CONTEXT_WINDOW)
        end_idx = min(len(messages), message_index + self.CONTEXT_WINDOW + 1)

        context_messages = messages[start_idx:end_idx]
        current_message = messages[message_index]

        # Perform analysis with context
        result = analysis_func(current_message['text'])

        # Build context info
        context_info = {
            'context_size': len(context_messages),
            'position_in_conversation': message_index / len(messages),
            'same_sender_nearby': self._count_same_sender(
                messages, message_index, current_message.get('sender', '')
            ),
            'consecutive_matches': 0,
            'recent_pattern_count': 0
        }

        # Detect patterns in context
        if hasattr(result, 'primary_concern') and result.primary_concern:
            pattern = result.primary_concern

            # Count pattern in nearby messages
            pattern_count = 0
            consecutive = 0

            for i in range(start_idx, end_idx):
                if i == message_index:
                    continue

                msg = messages[i]
                if 'analysis' in msg and hasattr(msg['analysis'], 'primary_concern'):
                    if msg['analysis'].primary_concern == pattern:
                        pattern_count += 1

                        # Check if consecutive
                        if abs(i - message_index) == 1:
                            consecutive += 1

            context_info['recent_pattern_count'] = pattern_count
            context_info['consecutive_matches'] = consecutive

        return result, context_info

    def _count_same_sender(self, messages: List[Dict], current_index: int, sender: str) -> int:
        """Count messages from same sender in context window

        Args:
            messages: List of messages
            current_index: Current message index
            sender: Sender name

        Returns:
            int: Count of messages from same sender nearby
        """
        if not sender:
            return 0

        start = max(0, current_index - self.CONTEXT_WINDOW)
        end = min(len(messages), current_index + self.CONTEXT_WINDOW + 1)

        count = 0
        for i in range(start, end):
            if i != current_index and messages[i].get('sender') == sender:
                count += 1

        return count


def get_confidence_level(confidence: float) -> str:
    """Convert confidence score to level

    Args:
        confidence: Confidence score (0-1)

    Returns:
        str: Confidence level
    """
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.50:
        return "medium"
    elif confidence >= 0.25:
        return "low"
    else:
        return "very low"
