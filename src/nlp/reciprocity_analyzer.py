"""
Reciprocity and Balanced Relationship Analysis Module
Analyzes balance in question asking, emotional support exchange, self-disclosure,
interest reciprocation, and effort matching in relationships.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ReciprocityMetric:
    """Container for reciprocity analysis"""
    metric_type: str  # questions, support, disclosure, interest, effort
    participant_a_score: float
    participant_b_score: float
    imbalance_ratio: float  # >1 = A dominates, <1 = B dominates, 1.0 = balanced
    balance_level: str  # balanced, slightly_imbalanced, imbalanced, severely_imbalanced
    examples_a: List[str] = field(default_factory=list)
    examples_b: List[str] = field(default_factory=list)


@dataclass
class ReciprocityAnalysis:
    """Complete reciprocity analysis results"""
    conversation_turn_taking: float = 0.0  # 0-1, how balanced the conversation is
    question_reciprocity: float = 0.0  # 0-1, balanced question asking
    support_exchange_balance: float = 0.0  # 0-1, emotional support balance
    self_disclosure_balance: float = 0.0  # 0-1, sharing of personal info
    interest_reciprocation: float = 0.0  # 0-1, mutual interest
    effort_matching: float = 0.0  # 0-1, effort balance
    conversation_initiation_ratio: float = 0.0  # A's initiations / total
    response_quality_match: float = 0.0  # 0-1, response depth matching
    metrics: List[ReciprocityMetric] = field(default_factory=list)
    overall_reciprocity_score: float = 0.0  # 0-1, overall balance
    reciprocity_health: str = "unknown"  # healthy, imbalanced, severely_imbalanced
    dominant_participant: Optional[str] = None
    imbalance_type: str = ""  # "over-giving", "under-giving", "balanced"
    relationship_risk: float = 0.0  # 0-1, risk from imbalance


class ReciprocityAnalyzer:
    """Analyzes reciprocity and balance in relationships"""

    # Question patterns
    QUESTION_PATTERNS = [
        r"\b(what|where|when|why|how|who|which)\b.*\?",
        r"\bdo you\b.*\?",
        r"\bdid you\b.*\?",
        r"\bare you\b.*\?",
        r"\bwere you\b.*\?",
        r"\bhave you\b.*\?",
        r"\bcan you\b.*\?",
        r"\bwould you\b.*\?",
        r"\bcould you\b.*\?",
        r"\bwhat do you think\b",
        r"\bhow do you feel\b",
    ]

    # Emotional support patterns
    SUPPORT_GIVEN_PATTERNS = [
        r"\bi'm here for you\b",
        r"\bi understand\b",
        r"\bthat (must|sounds)\b.*\b(hard|difficult|painful|scary)\b",
        r"\bi'm sorry (you|that)\b",
        r"\byou can (do|handle|get through)\b.*\bthis\b",
        r"\bi believe in you\b",
        r"\byou're (strong|capable|resilient)\b",
        r"\blet me help\b",
        r"\byou're not alone\b",
        r"\bthis isn't your fault\b",
        r"\bi care about you\b",
        r"\byou deserve\b",
    ]

    SUPPORT_SEEKING_PATTERNS = [
        r"\bi (need|could use|want)\b.*\b(help|support|advice)\b",
        r"\bi'm (struggling|overwhelmed|stressed)\b",
        r"\bcan you (help|advise|listen)\b",
        r"\bi don't know what to do\b",
        r"\bwhat would you do\b",
        r"\bhow would you handle\b",
        r"\bcan i talk to you\b",
        r"\bi'm (scared|worried|anxious)\b",
        r"\bam i (overreacting|being reasonable)\b",
    ]

    # Self-disclosure patterns (personal information sharing)
    DISCLOSURE_PATTERNS = [
        r"\bi (feel|felt|am)\b",
        r"\bi (did|was|have been)\b",
        r"\bwhen i\b",
        r"\bmy (family|parents|childhood)\b",
        r"\bi (struggled|suffered|went through)\b",
        r"\bmy (dream|goal|fear|secret)\b",
        r"\bi (believe|think|value)\b",
        r"\bthis matters to me\b",
        r"\bmy insecurity\b",
        r"\bmy weakness\b",
        r"\bi (admitted|confessed|revealed)\b",
    ]

    # Interest reciprocation patterns
    INTEREST_PATTERNS = [
        r"\bthat (sounds|seems)\b.*\b(interesting|amazing|cool|fun)\b",
        r"\btell me (more|about|everything)\b",
        r"\bi want to know\b",
        r"\bwhat (happened|happens)\b.*\bnext\b",
        r"\bthat's (impressive|awesome|incredible)\b",
        r"\bi'd (love|like) to (hear|know|learn)\b",
        r"\byou're so (interesting|fascinating|talented)\b",
        r"\bhow did that make you feel\b",
        r"\bwhat was that like\b",
        r"\bfollow up on\b",
    ]

    # Effort patterns
    EFFORT_PATTERNS = [
        r"\bi (spent|took|made)\b.*\btime\b",
        r"\bi (went|traveled|drove)\b",
        r"\bi (prepared|planned|organized)\b",
        r"\bi (sacrificed|gave up)\b",
        r"\bi (worked hard|put in effort)\b",
        r"\bi (thought|researched|looked into)\b.*\bfor you\b",
        r"\bi (asked|suggested|recommended)\b",
        r"\bi (learned|studied)\b.*\b(for you|because of you)\b",
        r"\bi (invested|dedicated|committed)\b",
    ]

    # Conversational turn patterns
    OPENING_PATTERNS = [
        r"^(hey|hi|hello|so|anyway|by the way|speaking of)",
        r"^(i (want|need|have to) tell you)",
        r"^(guess what)",
    ]

    def __init__(self):
        """Initialize reciprocity analyzer"""
        self.compiled_patterns = self._compile_patterns()
        self.conversation_history: Dict[str, List[str]] = defaultdict(list)

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for efficiency"""
        return {
            'questions': [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS],
            'support_given': [re.compile(p, re.IGNORECASE) for p in self.SUPPORT_GIVEN_PATTERNS],
            'support_seeking': [re.compile(p, re.IGNORECASE) for p in self.SUPPORT_SEEKING_PATTERNS],
            'disclosure': [re.compile(p, re.IGNORECASE) for p in self.DISCLOSURE_PATTERNS],
            'interest': [re.compile(p, re.IGNORECASE) for p in self.INTEREST_PATTERNS],
            'effort': [re.compile(p, re.IGNORECASE) for p in self.EFFORT_PATTERNS],
            'opening': [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.OPENING_PATTERNS],
        }

    def analyze_reciprocity(
        self,
        messages_participant_a: List[str],
        messages_participant_b: List[str],
        participant_a_name: str = "Participant A",
        participant_b_name: str = "Participant B"
    ) -> ReciprocityAnalysis:
        """Analyze reciprocity between two participants

        Args:
            messages_participant_a: Messages from participant A
            messages_participant_b: Messages from participant B
            participant_a_name: Name of participant A
            participant_b_name: Name of participant B

        Returns:
            ReciprocityAnalysis: Complete analysis
        """
        analysis = ReciprocityAnalysis()

        # Combine messages for overall conversation analysis
        all_messages_a = "\n".join(messages_participant_a)
        all_messages_b = "\n".join(messages_participant_b)

        # Question reciprocity
        questions_a = self._count_pattern_matches(all_messages_a, 'questions')
        questions_b = self._count_pattern_matches(all_messages_b, 'questions')
        q_metric = self._calculate_reciprocity_metric(
            questions_a, questions_b,
            participant_a_name, participant_b_name,
            'questions'
        )
        analysis.metrics.append(q_metric)
        analysis.question_reciprocity = q_metric.balance_level == 'balanced' and 1.0 or 0.5

        # Support exchange balance
        support_given_a = self._count_pattern_matches(all_messages_a, 'support_given')
        support_seeking_a = self._count_pattern_matches(all_messages_a, 'support_seeking')
        support_given_b = self._count_pattern_matches(all_messages_b, 'support_given')
        support_seeking_b = self._count_pattern_matches(all_messages_b, 'support_seeking')

        support_metric = self._calculate_support_balance(
            support_given_a, support_seeking_a,
            support_given_b, support_seeking_b,
            participant_a_name, participant_b_name
        )
        analysis.metrics.append(support_metric)
        analysis.support_exchange_balance = 1.0 - abs(support_given_a - support_given_b) / \
                                            max(1, (support_given_a + support_given_b))

        # Self-disclosure balance
        disclosure_a = self._count_pattern_matches(all_messages_a, 'disclosure')
        disclosure_b = self._count_pattern_matches(all_messages_b, 'disclosure')
        disclosure_metric = self._calculate_reciprocity_metric(
            disclosure_a, disclosure_b,
            participant_a_name, participant_b_name,
            'disclosure'
        )
        analysis.metrics.append(disclosure_metric)
        analysis.self_disclosure_balance = disclosure_metric.imbalance_ratio

        # Interest reciprocation
        interest_a = self._count_pattern_matches(all_messages_a, 'interest')
        interest_b = self._count_pattern_matches(all_messages_b, 'interest')
        interest_metric = self._calculate_reciprocity_metric(
            interest_a, interest_b,
            participant_a_name, participant_b_name,
            'interest'
        )
        analysis.metrics.append(interest_metric)
        analysis.interest_reciprocation = 1.0 if interest_metric.balance_level == 'balanced' else 0.5

        # Effort matching
        effort_a = self._count_pattern_matches(all_messages_a, 'effort')
        effort_b = self._count_pattern_matches(all_messages_b, 'effort')
        effort_metric = self._calculate_reciprocity_metric(
            effort_a, effort_b,
            participant_a_name, participant_b_name,
            'effort'
        )
        analysis.metrics.append(effort_metric)
        analysis.effort_matching = 1.0 if effort_metric.balance_level == 'balanced' else 0.5

        # Conversation initiation ratio
        opening_a = sum(1 for msg in messages_participant_a if self._matches_any_pattern(msg, 'opening'))
        opening_b = sum(1 for msg in messages_participant_b if self._matches_any_pattern(msg, 'opening'))
        total_initiatives = opening_a + opening_b

        if total_initiatives > 0:
            analysis.conversation_initiation_ratio = opening_a / total_initiatives
        else:
            analysis.conversation_initiation_ratio = 0.5  # Equal if no clear initiations

        # Response quality assessment
        analysis.response_quality_match = self._assess_response_quality(
            messages_participant_a,
            messages_participant_b
        )

        # Calculate overall reciprocity score
        scores = [
            analysis.question_reciprocity,
            analysis.support_exchange_balance,
            analysis.self_disclosure_balance,
            analysis.interest_reciprocation,
            analysis.effort_matching,
            analysis.response_quality_match,
        ]

        analysis.overall_reciprocity_score = statistics.mean([s for s in scores if s > 0])

        # Determine reciprocity health
        analysis.reciprocity_health = self._assess_reciprocity_health(analysis)

        # Identify dominant participant
        if analysis.conversation_initiation_ratio > 0.6:
            analysis.dominant_participant = participant_a_name
        elif analysis.conversation_initiation_ratio < 0.4:
            analysis.dominant_participant = participant_b_name

        # Determine imbalance type
        analysis.imbalance_type = self._determine_imbalance_type(analysis)

        # Calculate relationship risk
        if analysis.reciprocity_health == "severely_imbalanced":
            analysis.relationship_risk = 0.8
        elif analysis.reciprocity_health == "imbalanced":
            analysis.relationship_risk = 0.5
        elif analysis.reciprocity_health == "slightly_imbalanced":
            analysis.relationship_risk = 0.2
        else:
            analysis.relationship_risk = 0.0

        return analysis

    def _count_pattern_matches(self, text: str, pattern_type: str) -> int:
        """Count matches for a pattern type"""
        if pattern_type not in self.compiled_patterns:
            return 0

        count = 0
        for pattern in self.compiled_patterns[pattern_type]:
            for _ in pattern.finditer(text):
                count += 1

        return count

    def _matches_any_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text matches any pattern of given type"""
        if pattern_type not in self.compiled_patterns:
            return False

        for pattern in self.compiled_patterns[pattern_type]:
            if pattern.search(text):
                return True

        return False

    def _calculate_reciprocity_metric(
        self,
        score_a: float,
        score_b: float,
        participant_a: str,
        participant_b: str,
        metric_type: str
    ) -> ReciprocityMetric:
        """Calculate reciprocity metric between two participants"""
        total = score_a + score_b

        if total == 0:
            imbalance_ratio = 1.0
        else:
            imbalance_ratio = max(score_a, score_b) / total if total > 0 else 1.0

        # Determine balance level
        if abs(imbalance_ratio - 0.5) < 0.15:
            balance_level = "balanced"
        elif abs(imbalance_ratio - 0.5) < 0.3:
            balance_level = "slightly_imbalanced"
        elif abs(imbalance_ratio - 0.5) < 0.4:
            balance_level = "imbalanced"
        else:
            balance_level = "severely_imbalanced"

        metric = ReciprocityMetric(
            metric_type=metric_type,
            participant_a_score=score_a,
            participant_b_score=score_b,
            imbalance_ratio=imbalance_ratio,
            balance_level=balance_level
        )

        return metric

    def _calculate_support_balance(
        self,
        given_a: int,
        seeking_a: int,
        given_b: int,
        seeking_b: int,
        participant_a: str,
        participant_b: str
    ) -> ReciprocityMetric:
        """Calculate support exchange balance"""
        # Support balance considers both giving and seeking
        # Someone seeking without giving, or giving without receiving, is imbalanced

        total_given = given_a + given_b
        total_seeking = seeking_a + seeking_b

        if total_given + total_seeking == 0:
            return ReciprocityMetric(
                metric_type="support",
                participant_a_score=0,
                participant_b_score=0,
                imbalance_ratio=1.0,
                balance_level="balanced"
            )

        # Calculate what each person contributes to support dynamic
        a_support_ratio = (given_a + seeking_a) / (total_given + total_seeking) if (total_given + total_seeking) > 0 else 0
        b_support_ratio = (given_b + seeking_b) / (total_given + total_seeking) if (total_given + total_seeking) > 0 else 0

        # Calculate imbalance in give/seek within each participant
        a_give_seek_balance = abs(given_a - seeking_a)
        b_give_seek_balance = abs(given_b - seeking_b)

        imbalance_ratio = max(a_give_seek_balance, b_give_seek_balance) / \
                         (a_give_seek_balance + b_give_seek_balance + 1)

        if imbalance_ratio < 0.2:
            balance_level = "balanced"
        elif imbalance_ratio < 0.35:
            balance_level = "slightly_imbalanced"
        elif imbalance_ratio < 0.5:
            balance_level = "imbalanced"
        else:
            balance_level = "severely_imbalanced"

        return ReciprocityMetric(
            metric_type="support",
            participant_a_score=given_a + seeking_a,
            participant_b_score=given_b + seeking_b,
            imbalance_ratio=imbalance_ratio,
            balance_level=balance_level
        )

    def _assess_response_quality(self, messages_a: List[str], messages_b: List[str]) -> float:
        """Assess if response quality is matched between participants"""
        avg_length_a = statistics.mean([len(msg) for msg in messages_a]) if messages_a else 0
        avg_length_b = statistics.mean([len(msg) for msg in messages_b]) if messages_b else 0

        if avg_length_a + avg_length_b == 0:
            return 0.5

        length_ratio = min(avg_length_a, avg_length_b) / max(avg_length_a, avg_length_b) if max(avg_length_a, avg_length_b) > 0 else 0

        # High ratio = well-matched response depth
        return length_ratio

    def _assess_reciprocity_health(self, analysis: ReciprocityAnalysis) -> str:
        """Assess overall reciprocity health"""
        if analysis.overall_reciprocity_score >= 0.75:
            return "healthy"
        elif analysis.overall_reciprocity_score >= 0.55:
            return "slightly_imbalanced"
        elif analysis.overall_reciprocity_score >= 0.35:
            return "imbalanced"
        else:
            return "severely_imbalanced"

    def _determine_imbalance_type(self, analysis: ReciprocityAnalysis) -> str:
        """Determine the type of imbalance"""
        if analysis.reciprocity_health == "healthy":
            return "balanced"

        # Check if any participant is consistently giving more
        giving_a = sum(1 for m in analysis.metrics if m.participant_a_score > m.participant_b_score)
        giving_b = sum(1 for m in analysis.metrics if m.participant_b_score > m.participant_a_score)

        if giving_a > giving_b:
            return "over-giving_participant_a"
        elif giving_b > giving_a:
            return "over-giving_participant_b"
        else:
            return "mixed_imbalance"

    def analyze_single_message_reciprocity(
        self,
        text: str,
        message_type: str = "unknown"
    ) -> Dict[str, float]:
        """Analyze reciprocity indicators in a single message

        Args:
            text: Text to analyze
            message_type: Type of message context

        Returns:
            Dict with reciprocity indicators
        """
        return {
            'questions_asked': self._count_pattern_matches(text, 'questions'),
            'support_given': self._count_pattern_matches(text, 'support_given'),
            'support_seeking': self._count_pattern_matches(text, 'support_seeking'),
            'self_disclosure': self._count_pattern_matches(text, 'disclosure'),
            'interest_shown': self._count_pattern_matches(text, 'interest'),
            'effort_mentioned': self._count_pattern_matches(text, 'effort'),
            'conversation_initiating': int(self._matches_any_pattern(text, 'opening')),
        }

    def track_reciprocity_over_time(
        self,
        messages_a: List[str],
        messages_b: List[str]
    ) -> Dict[str, Any]:
        """Track reciprocity patterns over multiple exchanges

        Args:
            messages_a: Messages from participant A
            messages_b: Messages from participant B

        Returns:
            Dict with reciprocity trend information
        """
        # Calculate reciprocity for each exchange pair
        scores = []

        min_len = min(len(messages_a), len(messages_b))
        for i in range(min_len):
            analysis = self.analyze_reciprocity(
                [messages_a[i]],
                [messages_b[i]]
            )
            scores.append(analysis.overall_reciprocity_score)

        if not scores:
            return {
                'trend': 'unknown',
                'average_score': 0.0,
                'volatility': 0.0,
                'improving': False
            }

        avg_score = statistics.mean(scores)
        volatility = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Determine if improving
        improving = False
        if len(scores) >= 2:
            first_half = statistics.mean(scores[:len(scores)//2])
            second_half = statistics.mean(scores[len(scores)//2:])
            improving = second_half > first_half

        return {
            'trend': 'improving' if improving else 'declining',
            'average_score': avg_score,
            'volatility': volatility,
            'improving': improving,
            'score_history': scores
        }
