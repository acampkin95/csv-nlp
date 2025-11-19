"""
Power Dynamics Analyzer

Analyzes power relationships and dominance patterns in conversations.
Detects conversational control, assertiveness, compliance patterns, and authority structures.

Author: Message Processor Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class PowerDynamicsResult:
    """Results from power dynamics analysis."""

    # Overall metrics
    power_score: float  # -1 (submissive) to 1 (dominant)
    assertiveness_level: str  # passive, assertive, aggressive
    control_level: str  # low, moderate, high, excessive

    # Turn-taking metrics
    turn_count: int
    interruption_count: int
    topic_initiations: int
    topic_controls: int

    # Communication patterns
    question_ratio: float  # Questions asked / total messages
    imperative_ratio: float  # Commands / total messages
    hedging_ratio: float  # Hedging language / total messages

    # Dominance indicators
    dominance_patterns: List[str] = field(default_factory=list)
    submission_patterns: List[str] = field(default_factory=list)

    # Speaker comparison (for multi-speaker conversations)
    relative_power: Optional[Dict[str, float]] = None
    power_imbalance_score: float = 0.0

    # Evidence
    dominant_messages: List[Dict[str, Any]] = field(default_factory=list)
    submissive_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'power_score': self.power_score,
            'assertiveness_level': self.assertiveness_level,
            'control_level': self.control_level,
            'turn_count': self.turn_count,
            'interruption_count': self.interruption_count,
            'topic_initiations': self.topic_initiations,
            'topic_controls': self.topic_controls,
            'question_ratio': self.question_ratio,
            'imperative_ratio': self.imperative_ratio,
            'hedging_ratio': self.hedging_ratio,
            'dominance_patterns': self.dominance_patterns,
            'submission_patterns': self.submission_patterns,
            'relative_power': self.relative_power,
            'power_imbalance_score': self.power_imbalance_score,
            'dominant_message_count': len(self.dominant_messages),
            'submissive_message_count': len(self.submissive_messages),
            'confidence': self.confidence
        }


class PowerDynamicsAnalyzer:
    """
    Analyzes power dynamics and dominance patterns in conversations.

    Features:
    - Turn-taking and interruption analysis
    - Question vs statement ratio tracking
    - Imperative mood detection
    - Hedging language analysis
    - Assertiveness scoring
    - Topic control metrics
    - Multi-speaker power comparison
    """

    def __init__(self):
        """Initialize the power dynamics analyzer."""
        self._compile_patterns()
        logger.info("PowerDynamicsAnalyzer initialized")

    def _compile_patterns(self):
        """Compile regex patterns for detection."""

        # Imperative patterns (commands, demands)
        self.imperative_patterns = [
            re.compile(r'\b(do|don\'t|get|give|tell|show|stop|start|go|come|bring|take)\b', re.IGNORECASE),
            re.compile(r'\byou (need to|have to|must|should|better)\b', re.IGNORECASE),
            re.compile(r'\b(listen|hear me|pay attention)\b', re.IGNORECASE),
        ]

        # Hedging patterns (uncertainty, softening)
        self.hedging_patterns = [
            re.compile(r'\b(maybe|perhaps|possibly|probably|kind of|sort of|I think|I guess)\b', re.IGNORECASE),
            re.compile(r'\b(if you want|if you don\'t mind|could you|would you mind)\b', re.IGNORECASE),
            re.compile(r'\b(sorry|excuse me|pardon)\b', re.IGNORECASE),
        ]

        # Dominance patterns
        self.dominance_patterns = {
            'direct_commands': re.compile(r'\b(do it|do this|do that|just do)\b', re.IGNORECASE),
            'demands': re.compile(r'\b(I (want|need|expect|demand|require) you to)\b', re.IGNORECASE),
            'threats': re.compile(r'\b(or else|you better|if you don\'t)\b', re.IGNORECASE),
            'dismissiveness': re.compile(r'\b(I don\'t care|doesn\'t matter|whatever|who cares)\b', re.IGNORECASE),
            'interruption_markers': re.compile(r'\b(wait|hold on|let me (finish|talk|speak))\b', re.IGNORECASE),
            'topic_forcing': re.compile(r'\b(we need to talk about|listen to me|pay attention)\b', re.IGNORECASE),
            'absolute_statements': re.compile(r'\b(you (always|never)|every time|all the time)\b', re.IGNORECASE),
            'superiority': re.compile(r'\b(I know better|trust me|believe me|take my word)\b', re.IGNORECASE),
        }

        # Submission patterns
        self.submission_patterns = {
            'excessive_apologies': re.compile(r'\b(I\'m (so |really )?sorry|my bad|my fault)\b', re.IGNORECASE),
            'self_deprecation': re.compile(r'\b(I\'m (stupid|dumb|an idiot|useless|terrible))\b', re.IGNORECASE),
            'compliance': re.compile(r'\b(okay|fine|whatever you (want|say)|as you wish)\b', re.IGNORECASE),
            'permission_seeking': re.compile(r'\b(can I|may I|is it okay if|do you mind if)\b', re.IGNORECASE),
            'agreement': re.compile(r'\b(you\'re right|I agree|you know best)\b', re.IGNORECASE),
            'deference': re.compile(r'\b(up to you|your (choice|decision|call))\b', re.IGNORECASE),
        }

        # Question patterns
        self.question_pattern = re.compile(r'\?$|\b(who|what|when|where|why|how|can|could|would|should|is|are|do|does)\b.*\?', re.IGNORECASE)

    def analyze(self, messages: List[Dict[str, Any]]) -> PowerDynamicsResult:
        """
        Analyze power dynamics in conversation.

        Args:
            messages: List of message dictionaries with 'text', 'sender', 'timestamp'

        Returns:
            PowerDynamicsResult with power metrics
        """
        if not messages:
            return self._empty_result()

        # Analyze turn-taking patterns
        turn_metrics = self._analyze_turn_taking(messages)

        # Analyze linguistic patterns
        linguistic_metrics = self._analyze_linguistic_patterns(messages)

        # Analyze per-speaker power
        speaker_power = self._analyze_speaker_power(messages)

        # Calculate overall power score
        power_score = self._calculate_power_score(turn_metrics, linguistic_metrics)

        # Classify assertiveness and control
        assertiveness = self._classify_assertiveness(power_score, linguistic_metrics)
        control_level = self._classify_control_level(turn_metrics, linguistic_metrics)

        # Calculate confidence
        confidence = self._calculate_confidence(messages, linguistic_metrics)

        return PowerDynamicsResult(
            power_score=power_score,
            assertiveness_level=assertiveness,
            control_level=control_level,
            turn_count=turn_metrics['turn_count'],
            interruption_count=turn_metrics['interruptions'],
            topic_initiations=turn_metrics['topic_initiations'],
            topic_controls=turn_metrics['topic_controls'],
            question_ratio=linguistic_metrics['question_ratio'],
            imperative_ratio=linguistic_metrics['imperative_ratio'],
            hedging_ratio=linguistic_metrics['hedging_ratio'],
            dominance_patterns=linguistic_metrics['dominance_patterns'],
            submission_patterns=linguistic_metrics['submission_patterns'],
            relative_power=speaker_power['relative_power'],
            power_imbalance_score=speaker_power['imbalance_score'],
            dominant_messages=linguistic_metrics['dominant_messages'],
            submissive_messages=linguistic_metrics['submissive_messages'],
            confidence=confidence
        )

    def _analyze_turn_taking(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze turn-taking patterns."""
        turn_count = len(messages)
        interruptions = 0
        topic_initiations = 0
        topic_controls = 0

        prev_sender = None
        prev_time = None

        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')
            timestamp = msg.get('timestamp')

            # Check for interruption (quick response time)
            if prev_time and timestamp:
                time_diff = (timestamp - prev_time).total_seconds()
                if time_diff < 10 and sender != prev_sender:  # Quick interruption
                    interruptions += 1

            # Check for topic initiation (change of subject)
            if i > 0 and sender != prev_sender:
                if self._is_topic_change(text, messages[i-1].get('text', '')):
                    topic_initiations += 1

            # Check for topic control (forceful redirection)
            topic_forcing = sum(1 for pattern in self.dominance_patterns.values() if pattern.search(text))
            if topic_forcing > 0:
                topic_controls += 1

            prev_sender = sender
            prev_time = timestamp

        return {
            'turn_count': turn_count,
            'interruptions': interruptions,
            'topic_initiations': topic_initiations,
            'topic_controls': topic_controls
        }

    def _is_topic_change(self, current_text: str, previous_text: str) -> bool:
        """Detect if message represents topic change."""
        # Simple heuristic: look for topic change markers
        topic_change_markers = [
            'anyway', 'by the way', 'speaking of', 'changing the subject',
            'let\'s talk about', 'I wanted to ask', 'can we discuss'
        ]

        current_lower = current_text.lower()
        return any(marker in current_lower for marker in topic_change_markers)

    def _analyze_linguistic_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze linguistic patterns for power indicators."""
        total_messages = len(messages)

        question_count = 0
        imperative_count = 0
        hedging_count = 0

        dominance_patterns_found = []
        submission_patterns_found = []
        dominant_messages = []
        submissive_messages = []

        for msg in messages:
            text = msg.get('text', '')

            # Count questions
            if self.question_pattern.search(text):
                question_count += 1

            # Count imperatives
            if any(pattern.search(text) for pattern in self.imperative_patterns):
                imperative_count += 1

            # Count hedging
            if any(pattern.search(text) for pattern in self.hedging_patterns):
                hedging_count += 1

            # Detect dominance patterns
            for pattern_name, pattern in self.dominance_patterns.items():
                if pattern.search(text):
                    dominance_patterns_found.append(pattern_name)
                    dominant_messages.append({
                        'text': text,
                        'pattern': pattern_name,
                        'sender': msg.get('sender')
                    })

            # Detect submission patterns
            for pattern_name, pattern in self.submission_patterns.items():
                if pattern.search(text):
                    submission_patterns_found.append(pattern_name)
                    submissive_messages.append({
                        'text': text,
                        'pattern': pattern_name,
                        'sender': msg.get('sender')
                    })

        return {
            'question_ratio': question_count / total_messages if total_messages > 0 else 0.0,
            'imperative_ratio': imperative_count / total_messages if total_messages > 0 else 0.0,
            'hedging_ratio': hedging_count / total_messages if total_messages > 0 else 0.0,
            'dominance_patterns': list(set(dominance_patterns_found)),
            'submission_patterns': list(set(submission_patterns_found)),
            'dominant_messages': dominant_messages[:10],  # Limit to 10 examples
            'submissive_messages': submissive_messages[:10]
        }

    def _analyze_speaker_power(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze power dynamics between speakers."""
        speaker_metrics = {}

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')

            if sender not in speaker_metrics:
                speaker_metrics[sender] = {
                    'message_count': 0,
                    'dominance_score': 0,
                    'submission_score': 0
                }

            speaker_metrics[sender]['message_count'] += 1

            # Score dominance patterns
            for pattern in self.dominance_patterns.values():
                if pattern.search(text):
                    speaker_metrics[sender]['dominance_score'] += 1

            # Score submission patterns
            for pattern in self.submission_patterns.values():
                if pattern.search(text):
                    speaker_metrics[sender]['submission_score'] += 1

        # Calculate relative power
        relative_power = {}
        for speaker, metrics in speaker_metrics.items():
            power = (metrics['dominance_score'] - metrics['submission_score']) / max(metrics['message_count'], 1)
            relative_power[speaker] = round(power, 3)

        # Calculate power imbalance
        if len(relative_power) > 1:
            power_values = list(relative_power.values())
            imbalance = max(power_values) - min(power_values)
        else:
            imbalance = 0.0

        return {
            'relative_power': relative_power,
            'imbalance_score': round(imbalance, 3)
        }

    def _calculate_power_score(self, turn_metrics: Dict, linguistic_metrics: Dict) -> float:
        """Calculate overall power score."""
        # Power score components
        imperative_weight = 0.3
        dominance_weight = 0.3
        submission_weight = -0.2
        hedging_weight = -0.1
        topic_control_weight = 0.1

        score = 0.0

        # Add imperative contribution
        score += linguistic_metrics['imperative_ratio'] * imperative_weight

        # Add dominance pattern contribution
        dominance_factor = min(len(linguistic_metrics['dominance_patterns']) / 5.0, 1.0)
        score += dominance_factor * dominance_weight

        # Subtract submission contribution
        submission_factor = min(len(linguistic_metrics['submission_patterns']) / 5.0, 1.0)
        score += submission_factor * submission_weight

        # Subtract hedging contribution
        score += linguistic_metrics['hedging_ratio'] * hedging_weight

        # Add topic control contribution
        if turn_metrics['turn_count'] > 0:
            topic_control_factor = turn_metrics['topic_controls'] / turn_metrics['turn_count']
            score += topic_control_factor * topic_control_weight

        # Normalize to -1 to 1 range
        score = max(-1.0, min(1.0, score))

        return round(score, 3)

    def _classify_assertiveness(self, power_score: float, linguistic_metrics: Dict) -> str:
        """Classify assertiveness level."""
        # Consider both power score and linguistic patterns

        if power_score > 0.5:
            # High dominance, check for aggression
            if linguistic_metrics['imperative_ratio'] > 0.3:
                return 'aggressive'
            else:
                return 'assertive'
        elif power_score > 0:
            return 'assertive'
        elif power_score > -0.3:
            # Slightly submissive but not passive
            if linguistic_metrics['hedging_ratio'] < 0.3:
                return 'assertive'
            else:
                return 'passive-assertive'
        else:
            return 'passive'

    def _classify_control_level(self, turn_metrics: Dict, linguistic_metrics: Dict) -> str:
        """Classify level of conversational control."""
        control_score = 0

        # Topic control
        if turn_metrics['turn_count'] > 0:
            topic_control_ratio = turn_metrics['topic_controls'] / turn_metrics['turn_count']
            control_score += topic_control_ratio * 40

        # Dominance patterns
        control_score += min(len(linguistic_metrics['dominance_patterns']) * 10, 40)

        # Interruptions
        if turn_metrics['turn_count'] > 0:
            interruption_ratio = turn_metrics['interruptions'] / turn_metrics['turn_count']
            control_score += interruption_ratio * 20

        if control_score > 70:
            return 'excessive'
        elif control_score > 40:
            return 'high'
        elif control_score > 20:
            return 'moderate'
        else:
            return 'low'

    def _calculate_confidence(self, messages: List[Dict], linguistic_metrics: Dict) -> float:
        """Calculate confidence in analysis."""
        confidence = 0.5  # Base confidence

        # More messages = higher confidence
        if len(messages) > 50:
            confidence += 0.3
        elif len(messages) > 20:
            confidence += 0.2
        elif len(messages) > 10:
            confidence += 0.1

        # Clear patterns = higher confidence
        total_patterns = len(linguistic_metrics['dominance_patterns']) + len(linguistic_metrics['submission_patterns'])
        if total_patterns > 5:
            confidence += 0.2
        elif total_patterns > 2:
            confidence += 0.1

        return min(1.0, confidence)

    def _empty_result(self) -> PowerDynamicsResult:
        """Return empty result for edge cases."""
        return PowerDynamicsResult(
            power_score=0.0,
            assertiveness_level='unknown',
            control_level='unknown',
            turn_count=0,
            interruption_count=0,
            topic_initiations=0,
            topic_controls=0,
            question_ratio=0.0,
            imperative_ratio=0.0,
            hedging_ratio=0.0,
            confidence=0.0
        )


# Example usage
if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        {'sender': 'A', 'text': 'You need to tell me where you are right now.', 'timestamp': None},
        {'sender': 'B', 'text': 'I\'m sorry, I\'m just at the store...', 'timestamp': None},
        {'sender': 'A', 'text': 'I don\'t care. Get back here immediately.', 'timestamp': None},
        {'sender': 'B', 'text': 'Okay, I\'ll come home right away.', 'timestamp': None},
        {'sender': 'A', 'text': 'And don\'t make me wait. You always do this.', 'timestamp': None},
    ]

    analyzer = PowerDynamicsAnalyzer()
    result = analyzer.analyze(test_messages)

    print("Power Dynamics Analysis:")
    print(f"Power Score: {result.power_score}")
    print(f"Assertiveness: {result.assertiveness_level}")
    print(f"Control Level: {result.control_level}")
    print(f"Relative Power: {result.relative_power}")
    print(f"Power Imbalance: {result.power_imbalance_score}")
    print(f"Confidence: {result.confidence}")
