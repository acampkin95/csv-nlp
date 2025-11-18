"""
Empath-based Psychological and Topical Analysis Module
Analyzes text across 200+ pre-validated categories including emotions, topics, and social dimensions.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

# Import model cache for performance optimization
from .model_cache import get_cache

# Third-party imports
try:
    from empath import Empath
except ImportError:
    Empath = None

logger = logging.getLogger(__name__)


@dataclass
class EmpathResult:
    """Container for Empath analysis results"""
    # All category scores (200+ categories)
    all_categories: Dict[str, float] = field(default_factory=dict)

    # Top categories by domain
    top_emotional: List[Tuple[str, float]] = field(default_factory=list)
    top_topical: List[Tuple[str, float]] = field(default_factory=list)
    top_social: List[Tuple[str, float]] = field(default_factory=list)
    top_risk: List[Tuple[str, float]] = field(default_factory=list)

    # Dominant category overall
    dominant_category: Optional[str] = None
    dominant_score: float = 0.0

    # Aggregate metrics
    emotional_intensity: float = 0.0
    risk_indicators: float = 0.0
    social_complexity: float = 0.0

    # Category counts
    active_categories: int = 0  # Categories with non-zero scores
    total_score: float = 0.0


@dataclass
class ConversationEmpath:
    """Empath analysis for entire conversation"""
    overall_themes: List[Tuple[str, float]] = field(default_factory=list)
    emotional_trajectory: str = "stable"  # escalating, de-escalating, stable, volatile
    conversation_topics: List[str] = field(default_factory=list)
    speaker_profiles: Dict[str, Dict] = field(default_factory=dict)
    theme_shifts: List[Dict] = field(default_factory=list)
    risk_progression: List[float] = field(default_factory=list)
    dominant_emotion: Optional[str] = None
    dominant_topic: Optional[str] = None


class EmpathAnalyzer:
    """Empath-based analyzer for psychological and topical dimensions"""

    # Emotional categories (based on Empath's lexicon)
    EMOTIONAL_CATEGORIES = [
        'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise',
        'anticipation', 'trust', 'love', 'hate', 'suffering', 'pain',
        'nervousness', 'aggression', 'cheerfulness', 'optimism', 'pride',
        'shame', 'envy', 'disappointment', 'confusion', 'horror'
    ]

    # Risk-related categories
    RISK_CATEGORIES = [
        'violence', 'crime', 'aggression', 'hate', 'weapon', 'kill',
        'death', 'torture', 'abuse', 'stealing', 'sexual', 'dominant',
        'submission', 'neglect', 'fear', 'horror', 'suffering', 'pain'
    ]

    # Social interaction categories
    SOCIAL_CATEGORIES = [
        'social_media', 'communication', 'meeting', 'friends', 'family',
        'children', 'emotional', 'affection', 'attractive', 'party',
        'negotiate', 'dispute', 'help', 'sympathy', 'trust', 'love'
    ]

    # Topical categories (subset of common conversation topics)
    TOPICAL_CATEGORIES = [
        'school', 'work', 'home', 'money', 'business', 'office',
        'shopping', 'payment', 'banking', 'health', 'medical', 'exercise',
        'sports', 'leisure', 'vacation', 'travel', 'technology', 'internet',
        'phone', 'computer', 'messaging', 'music', 'art', 'entertainment'
    ]

    # Thresholds
    RISK_THRESHOLD = 0.1  # Score above this indicates risk presence
    HIGH_INTENSITY_THRESHOLD = 0.3  # High emotional intensity

    def __init__(self):
        """Initialize Empath analyzer using cached lexicon for performance"""
        cache = get_cache()

        if Empath:
            # Use cached Empath lexicon (significant speedup)
            self.empath = cache.get_or_load('empath_lexicon', self._load_empath)
        else:
            self.empath = None
            logger.warning("Empath not installed. Install with: pip install empath")

    @staticmethod
    def _load_empath():
        """Load Empath lexicon (used by cache)"""
        return Empath()

    def analyze_text(self, text: str) -> EmpathResult:
        """Perform comprehensive Empath analysis on text

        Args:
            text: Text to analyze

        Returns:
            EmpathResult: Complete Empath analysis
        """
        result = EmpathResult()

        if not text or not self.empath:
            return result

        try:
            # Analyze with Empath (normalized scores)
            scores = self.empath.analyze(text, normalize=True)

            # Filter out zero scores for cleaner results
            result.all_categories = {
                category: score
                for category, score in scores.items()
                if score > 0
            }

            result.active_categories = len(result.all_categories)
            result.total_score = sum(result.all_categories.values())

            # Find dominant category
            if result.all_categories:
                dominant = max(result.all_categories.items(), key=lambda x: x[1])
                result.dominant_category = dominant[0]
                result.dominant_score = dominant[1]

            # Extract top categories by domain
            result.top_emotional = self._get_top_by_domain(
                result.all_categories,
                self.EMOTIONAL_CATEGORIES,
                top_n=5
            )

            result.top_risk = self._get_top_by_domain(
                result.all_categories,
                self.RISK_CATEGORIES,
                top_n=5
            )

            result.top_social = self._get_top_by_domain(
                result.all_categories,
                self.SOCIAL_CATEGORIES,
                top_n=5
            )

            result.top_topical = self._get_top_by_domain(
                result.all_categories,
                self.TOPICAL_CATEGORIES,
                top_n=5
            )

            # Calculate aggregate metrics
            result.emotional_intensity = self._calculate_emotional_intensity(result.all_categories)
            result.risk_indicators = self._calculate_risk_score(result.all_categories)
            result.social_complexity = self._calculate_social_complexity(result.all_categories)

        except Exception as e:
            logger.error(f"Empath analysis failed: {e}")

        return result

    def _get_top_by_domain(
        self,
        all_scores: Dict[str, float],
        domain_categories: List[str],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Extract top N categories from a specific domain

        Args:
            all_scores: All category scores
            domain_categories: List of categories in this domain
            top_n: Number of top categories to return

        Returns:
            List of (category, score) tuples, sorted by score descending
        """
        domain_scores = {
            cat: score
            for cat, score in all_scores.items()
            if cat in domain_categories and score > 0
        }

        sorted_scores = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def _calculate_emotional_intensity(self, scores: Dict[str, float]) -> float:
        """Calculate overall emotional intensity from Empath scores

        Args:
            scores: All category scores

        Returns:
            float: Emotional intensity (0-1)
        """
        emotional_scores = [
            scores.get(cat, 0)
            for cat in self.EMOTIONAL_CATEGORIES
        ]

        if not emotional_scores:
            return 0.0

        # Use sum of emotional categories, normalized
        total = sum(emotional_scores)
        # Cap at 1.0
        return min(total / len(emotional_scores), 1.0)

    def _calculate_risk_score(self, scores: Dict[str, float]) -> float:
        """Calculate aggregated risk score from risk-related categories

        Args:
            scores: All category scores

        Returns:
            float: Risk score (0-1)
        """
        risk_scores = [
            scores.get(cat, 0)
            for cat in self.RISK_CATEGORIES
        ]

        if not risk_scores:
            return 0.0

        # Use mean of active risk categories
        active_risks = [s for s in risk_scores if s > 0]
        if not active_risks:
            return 0.0

        return min(statistics.mean(active_risks), 1.0)

    def _calculate_social_complexity(self, scores: Dict[str, float]) -> float:
        """Calculate social interaction complexity

        Args:
            scores: All category scores

        Returns:
            float: Social complexity (0-1)
        """
        social_scores = [
            scores.get(cat, 0)
            for cat in self.SOCIAL_CATEGORIES
        ]

        if not social_scores:
            return 0.0

        # Number of active social categories indicates complexity
        active_social = sum(1 for s in social_scores if s > 0)
        complexity = active_social / len(self.SOCIAL_CATEGORIES)

        return min(complexity, 1.0)

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> ConversationEmpath:
        """Analyze Empath patterns across entire conversation

        Args:
            messages: List of messages with 'text', 'sender' keys

        Returns:
            ConversationEmpath: Conversation-level Empath analysis
        """
        conv_empath = ConversationEmpath()

        if not messages or not self.empath:
            return conv_empath

        # Analyze each message
        message_results = []
        speaker_data = {}
        all_categories = {}
        risk_scores = []

        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Analyze message
            result = self.analyze_text(text)
            message_results.append(result)
            risk_scores.append(result.risk_indicators)

            # Track by speaker
            if sender not in speaker_data:
                speaker_data[sender] = {
                    'category_totals': {},
                    'emotional_intensity': [],
                    'risk_scores': [],
                    'themes': []
                }

            # Aggregate speaker data
            speaker_data[sender]['emotional_intensity'].append(result.emotional_intensity)
            speaker_data[sender]['risk_scores'].append(result.risk_indicators)

            if result.dominant_category:
                speaker_data[sender]['themes'].append(result.dominant_category)

            # Aggregate all categories
            for cat, score in result.all_categories.items():
                all_categories[cat] = all_categories.get(cat, 0) + score
                speaker_data[sender]['category_totals'][cat] = \
                    speaker_data[sender]['category_totals'].get(cat, 0) + score

            # Detect theme shifts
            if i > 0:
                prev_result = message_results[i-1]
                if prev_result.dominant_category and result.dominant_category:
                    if prev_result.dominant_category != result.dominant_category:
                        conv_empath.theme_shifts.append({
                            'index': i,
                            'sender': sender,
                            'from_theme': prev_result.dominant_category,
                            'to_theme': result.dominant_category,
                            'magnitude': abs(result.dominant_score - prev_result.dominant_score)
                        })

        # Calculate overall themes (top 10)
        if all_categories:
            sorted_themes = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
            conv_empath.overall_themes = sorted_themes[:10]

            # Identify dominant emotion and topic
            emotional_themes = [(cat, score) for cat, score in sorted_themes
                               if cat in self.EMOTIONAL_CATEGORIES]
            if emotional_themes:
                conv_empath.dominant_emotion = emotional_themes[0][0]

            topical_themes = [(cat, score) for cat, score in sorted_themes
                             if cat in self.TOPICAL_CATEGORIES]
            if topical_themes:
                conv_empath.dominant_topic = topical_themes[0][0]

            # Extract conversation topics (top 5 topical categories)
            conv_empath.conversation_topics = [
                cat for cat, _ in sorted_themes[:5]
                if cat in self.TOPICAL_CATEGORIES
            ]

        # Determine emotional trajectory from risk scores
        if len(risk_scores) > 2:
            conv_empath.emotional_trajectory = self._determine_trajectory(risk_scores)
            conv_empath.risk_progression = risk_scores

        # Compile speaker profiles
        for sender, data in speaker_data.items():
            top_categories = sorted(
                data['category_totals'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            conv_empath.speaker_profiles[sender] = {
                'top_themes': [cat for cat, _ in top_categories],
                'avg_emotional_intensity': statistics.mean(data['emotional_intensity'])
                    if data['emotional_intensity'] else 0,
                'avg_risk_score': statistics.mean(data['risk_scores'])
                    if data['risk_scores'] else 0,
                'message_count': len(data['emotional_intensity']),
                'theme_diversity': len(data['category_totals'])
            }

        return conv_empath

    def _determine_trajectory(self, risk_scores: List[float]) -> str:
        """Determine emotional/risk trajectory over time

        Args:
            risk_scores: List of risk scores over time

        Returns:
            str: Trajectory type
        """
        if len(risk_scores) < 3:
            return "stable"

        # Split into thirds
        third = len(risk_scores) // 3
        first_third = statistics.mean(risk_scores[:third])
        middle_third = statistics.mean(risk_scores[third:2*third])
        last_third = statistics.mean(risk_scores[2*third:])

        # Check for consistent escalation or de-escalation
        if last_third > middle_third > first_third:
            if last_third - first_third > 0.2:  # Significant increase
                return "escalating"
        elif last_third < middle_third < first_third:
            if first_third - last_third > 0.2:  # Significant decrease
                return "de-escalating"

        # Check for volatility
        variance = statistics.variance(risk_scores)
        if variance > 0.1:  # High variance
            return "volatile"

        return "stable"

    def get_analysis_summary(self, result: EmpathResult) -> str:
        """Generate human-readable Empath analysis summary

        Args:
            result: Empath analysis result

        Returns:
            str: Summary text
        """
        if not result.all_categories:
            return "No significant themes detected"

        summary_parts = []

        # Dominant theme
        if result.dominant_category:
            summary_parts.append(f"Primary theme: {result.dominant_category}")

        # Top emotional categories
        if result.top_emotional:
            emotions = [cat for cat, _ in result.top_emotional[:3]]
            summary_parts.append(f"Emotions: {', '.join(emotions)}")

        # Risk indicators
        if result.risk_indicators > self.RISK_THRESHOLD:
            risk_level = "high" if result.risk_indicators > 0.3 else "moderate"
            summary_parts.append(f"{risk_level} risk indicators present")

        # Emotional intensity
        if result.emotional_intensity > self.HIGH_INTENSITY_THRESHOLD:
            summary_parts.append("high emotional intensity")

        return "; ".join(summary_parts)

    def get_risk_categories(self, result: EmpathResult) -> List[Tuple[str, float]]:
        """Extract risk-related categories with scores

        Args:
            result: Empath analysis result

        Returns:
            List of (category, score) tuples for risk categories
        """
        return [
            (cat, score)
            for cat, score in result.all_categories.items()
            if cat in self.RISK_CATEGORIES and score > self.RISK_THRESHOLD
        ]

    def compare_themes(
        self,
        result1: EmpathResult,
        result2: EmpathResult
    ) -> Dict[str, Any]:
        """Compare themes between two Empath results

        Args:
            result1: First Empath result
            result2: Second Empath result

        Returns:
            Dict: Comparison metrics
        """
        comparison = {
            'theme_shift': None,
            'emotional_intensity_change': result2.emotional_intensity - result1.emotional_intensity,
            'risk_change': result2.risk_indicators - result1.risk_indicators,
            'new_themes': [],
            'disappeared_themes': [],
            'shared_themes': []
        }

        # Theme shift
        if result1.dominant_category != result2.dominant_category:
            comparison['theme_shift'] = {
                'from': result1.dominant_category,
                'to': result2.dominant_category
            }

        # Category changes
        cats1 = set(result1.all_categories.keys())
        cats2 = set(result2.all_categories.keys())

        comparison['new_themes'] = list(cats2 - cats1)
        comparison['disappeared_themes'] = list(cats1 - cats2)
        comparison['shared_themes'] = list(cats1 & cats2)

        return comparison
