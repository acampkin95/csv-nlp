#!/usr/bin/env python3
"""
Attachment Style Analyzer Module
Analyzes attachment patterns based on attachment theory (Bowlby, Ainsworth, Hazan & Shaver).
Detects secure, anxious, avoidant, and disorganized attachment indicators with clinical accuracy.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AttachmentIndicator:
    """Represents a detected attachment indicator"""
    indicator_type: str  # secure, anxious, avoidant, disorganized
    attachment_dimension: str  # autonomy, closeness, trust, fear, dependency
    text_segment: str
    message_index: int
    confidence: float  # 0-1
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class AttachmentProfile:
    """Overall attachment style profile for a speaker"""
    speaker_name: str
    primary_style: str = "secure"  # secure, anxious, avoidant, disorganized, mixed
    style_scores: Dict[str, float] = field(default_factory=dict)  # 0-1 for each style

    # Specific dimensions
    anxiety_about_relationships: float = 0.0  # 0-1
    avoidance_of_intimacy: float = 0.0  # 0-1
    dependency_level: float = 0.0  # 0-1
    fear_of_abandonment: float = 0.0  # 0-1
    trust_in_others: float = 0.0  # 0-1
    self_reliance: float = 0.0  # 0-1

    # Pattern counts
    secure_indicators: int = 0
    anxious_indicators: int = 0
    avoidant_indicators: int = 0
    disorganized_indicators: int = 0

    # Trend analysis
    attachment_stability: str = "stable"  # stable, fluctuating, deteriorating
    relationship_satisfaction_proxy: float = 0.5  # 0-1


class AttachmentStyleAnalyzer:
    """
    Analyzes attachment styles using evidence-based patterns from attachment theory.
    Based on Mary Ainsworth's attachment classifications and contemporary research.
    """

    # Secure attachment patterns
    SECURE_PATTERNS = [
        r'\b(I\s+feel|I\s+am)\s+(comfortable|safe|secure|confident)\s+\w+',
        r'\b(we\s+can|we\s+work|we\s+talk)\s+\w+',
        r'\b(I\s+trust|I\s+believe\s+in)\s+\w+',
        r'\b(open|honest|direct)\s+\w+(communication|talk|conversation)',
        r'\b(I\s+can|I\s+am\s+able)\s+to\s+(be\s+vulnerable|share|express)',
        r'\b(I\s+value|I\s+appreciate)\s+(you|your|our)\b',
        r'\b(I\s+support|I\s+understand)\s+(you|your|their)',
        r'\b(secure|stable|healthy)\s+(relationship|bond|connection)',
    ]

    # Anxious attachment patterns
    ANXIOUS_PATTERNS = [
        r'\b(need|need\s+to|constantly|always\s+need)\s+\w+\s+(reassurance|confirmation|validation)',
        r'\b(are\s+you|do\s+you|will\s+you)\s+(still|really|truly)\s+(love|care)\s+(me|about\s+me)',
        r'\b(I\s+worry|I\s+am\s+worried|I\s+am\s+afraid)\s+(you\s+)?(will|might)\s+(leave|abandon|forget)',
        r'\b(what\s+if\s+you)\s+(leave|abandon|cheat|replace)',
        r'\b(don\'t|can\'t)\s+\w+\s+(without|without\s+you)',
        r'\b(desperate|clingy|needy|dependent)\b',
        r'\b(I\s+can\'t|I\s+won\'t|I\s+can\'t\s+live)\s+(without\s+)?you',
        r'\b(please\s+don\'t|don\'t\s+leave|stay\s+with)\s+me',
        r'\b(jealous|insecure|anxious)\s+\w+(about\s+)?(you|us|our)',
        r'\b(hypervigilant|monitoring|checking)\s+\w+(on\s+)?(you|your)',
    ]

    # Avoidant attachment patterns
    AVOIDANT_PATTERNS = [
        r'\b(don\'t\s+need|don\'t\s+want|not\s+interested\s+in)\s+\w+(relationship|closeness|intimacy)',
        r'\b(independent|self\s+sufficient|don\'t\s+rely)\s+\w+',
        r'\b(distance|space|alone|time\s+apart)\b',
        r'\b(emotional|feelings|vulnerability)\s+\w+\s+(difficult|hard|uncomfortable)',
        r'\b(I\s+prefer|I\s+like)\s+(being\s+alone|solitude|my\s+own)\b',
        r'\b(don\'t\s+want|not\s+comfortable\s+with)\s+(intimacy|closeness|affection)',
        r'\b(moving\s+on|moving\s+forward)\s+(without|alone)\b',
        r'\b(cold|distant|emotionally\s+unavailable)\b',
        r'\b(can\'t|won\'t|refuse\s+to)\s+(commit|settle\s+down|get\s+serious)',
        r'\b(independent|self\s+reliant|autonomous)\b',
    ]

    # Disorganized attachment patterns
    DISORGANIZED_PATTERNS = [
        r'\b(confused|mixed\s+feelings|don\'t\s+know)\s+\w+(feel|want|need)',
        r'\b(hot\s+and\s+cold|push\s+away|pull\s+close)\b',
        r'\b(love\s+and\s+hate|can\'t\s+live\s+with|can\'t\s+live\s+without)\b',
        r'\b(simultaneous|both|at\s+the\s+same\s+time)\s+(want|need|feel)\b',
        r'\b(contradictory|inconsistent|unpredictable)\s+\w+',
        r'\b(scared|frightened|terrified)\s+(of|by)\s+(intimacy|rejection|abandonment)',
        r'\b(cycle|pattern)\s+(of|repeating)\s+(push\s+pull|approach\s+avoid)',
        r'\b(trauma|abuse|betrayal)\s+\w+(affecting|impacting|influencing)',
        r'\b(trust\s+issues|can\'t\s+trust|don\'t\s+trust)\b',
    ]

    # Dependency language
    DEPENDENCY_PATTERNS = [
        r'\b(depend|rely|count)\s+(on\s+)?(you|him|her|them)',
        r'\b(without\s+)?you\s+(I\s+)?(can\'t|won\'t|don\'t)\s+\w+',
        r'\b(need\s+you|need\s+your)\s+\w+(support|help|approval|validation)',
        r'\b(lost|empty|incomplete)\s+\w+(without\s+)?you',
        r'\b(emotionally\s+dependent|needy|co\s+dependent)\b',
        r'\b(control|control\s+me|need\s+to\s+control)\b',
    ]

    # Fear of abandonment
    ABANDONMENT_FEAR_PATTERNS = [
        r'\b(afraid|fear|scared)\s+(of|that)\s+(you\s+)?(will\s+)?(leave|abandon|forget)',
        r'\b(terrified|petrified)\s+(of|by)\s+(abandonment|rejection|loss)',
        r'\b(please\s+don\'t|don\'t|never)\s+(leave|go|abandon|forget)\b',
        r'\b(what\s+if\s+you)\s+(left|abandoned|rejected)\s+me',
        r'\b(betrayal|betrayed|let\s+down)\b',
        r'\b(alone|lonely|abandoned|forgotten)\b',
        r'\b(when\s+you\s+leave|if\s+you\s+go|when\s+you\s+left)\b',
    ]

    # Intimacy avoidance
    INTIMACY_AVOIDANCE_PATTERNS = [
        r'\b(emotional\s+distance|keep\s+distance|maintain\s+distance)\b',
        r'\b(too\s+intimate|too\s+close|getting\s+too\s+close)\b',
        r'\b(need\s+space|need\s+breathing\s+room|need\s+to\s+breathe)\b',
        r'\b(vulnerable|exposed|exposed\s+to)\s+\w+',
        r'\b(share|sharing)\s+\w+(difficult|hard|impossible|uncomfortable)',
        r'\b(feelings|emotions|personal)\s+\w+(hard\s+to|difficult\s+to|won\'t)\s+(share|discuss)',
        r'\b(wall|barrier|defend|defend\s+myself)\b',
        r'\b(guard|guarded|protective)\s+\w+(my\s+)?(heart|feelings|emotions)',
    ]

    # Relationship satisfaction proxy patterns
    SATISFACTION_POSITIVE = [
        r'\b(happy|satisfied|content|fulfilled)\s+(in|with|about)\s+(our|this|the)\s+relationship',
        r'\b(grateful|thankful|appreciate)\s+\w+\b',
        r'\b(connected|bonded|close)\s+to\s+you',
        r'\b(support\s+each\s+other|work\s+together|team)\b',
    ]

    SATISFACTION_NEGATIVE = [
        r'\b(unhappy|unsatisfied|miserable)\s+(in|with)\s+(our|this|the)\s+relationship',
        r'\b(fed\s+up|tired|exhausted)\s+(with\s+)?you',
        r'\b(resentful|bitter|angry)\s+\w+\s+(you|your)',
        r'\b(mistake|error|wrong|bad)\s+\w+(relationship|choice)',
    ]

    def __init__(self):
        """Initialize attachment style analyzer"""
        logger.info("AttachmentStyleAnalyzer initialized")

    def analyze_attachment_styles(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze attachment styles across all messages in a conversation.

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict with attachment indicators and profiles per speaker
        """
        logger.info(f"Analyzing attachment styles in {len(messages)} messages")

        all_indicators: Dict[str, List[AttachmentIndicator]] = defaultdict(list)
        attachment_profiles: Dict[str, AttachmentProfile] = {}

        for idx, message in enumerate(messages):
            text = message.get('text', '')
            sender = message.get('sender', 'Unknown')

            if not text.strip():
                continue

            # Detect all attachment indicators
            indicators = self._detect_message_indicators(text, idx, sender)
            all_indicators[sender].extend(indicators)

        # Build profiles for each speaker
        for speaker, indicators in all_indicators.items():
            attachment_profiles[speaker] = self._build_attachment_profile(
                speaker, indicators
            )

        logger.info(
            f"Detected {sum(len(i) for i in all_indicators.values())} total attachment indicators"
        )

        return {
            'indicators_by_speaker': dict(all_indicators),
            'attachment_profiles': attachment_profiles,
            'summary': self._generate_summary(attachment_profiles),
        }

    def _detect_message_indicators(
        self, text: str, message_index: int, sender: str
    ) -> List[AttachmentIndicator]:
        """Detect all attachment indicators in a single message."""
        indicators: List[AttachmentIndicator] = []

        # Secure attachment
        secure_results = self._detect_secure_attachment(text, message_index)
        indicators.extend(secure_results)

        # Anxious attachment
        anxious_results = self._detect_anxious_attachment(text, message_index)
        indicators.extend(anxious_results)

        # Avoidant attachment
        avoidant_results = self._detect_avoidant_attachment(text, message_index)
        indicators.extend(avoidant_results)

        # Disorganized attachment
        disorganized_results = self._detect_disorganized_attachment(text, message_index)
        indicators.extend(disorganized_results)

        # Dependency language
        dependency_results = self._detect_dependency_language(text, message_index)
        indicators.extend(dependency_results)

        # Fear of abandonment
        abandonment_results = self._detect_abandonment_fear(text, message_index)
        indicators.extend(abandonment_results)

        # Intimacy avoidance
        intimacy_results = self._detect_intimacy_avoidance(text, message_index)
        indicators.extend(intimacy_results)

        return indicators

    def _detect_secure_attachment(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect secure attachment indicators."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.SECURE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    AttachmentIndicator(
                        indicator_type='secure',
                        attachment_dimension='trust',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.7,
                        supporting_evidence=['secure language pattern'],
                    )
                )

        return indicators

    def _detect_anxious_attachment(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect anxious attachment indicators."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.ANXIOUS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_attachment_confidence(match.group(), text, 'anxious')

                indicators.append(
                    AttachmentIndicator(
                        indicator_type='anxious',
                        attachment_dimension='fear_of_abandonment',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        supporting_evidence=['reassurance seeking', 'abandonment concerns'],
                    )
                )

        return indicators

    def _detect_avoidant_attachment(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect avoidant attachment indicators."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.AVOIDANT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_attachment_confidence(match.group(), text, 'avoidant')

                indicators.append(
                    AttachmentIndicator(
                        indicator_type='avoidant',
                        attachment_dimension='autonomy',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        supporting_evidence=['distance seeking', 'independence emphasis'],
                    )
                )

        return indicators

    def _detect_disorganized_attachment(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect disorganized attachment indicators."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.DISORGANIZED_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_attachment_confidence(match.group(), text, 'disorganized')

                indicators.append(
                    AttachmentIndicator(
                        indicator_type='disorganized',
                        attachment_dimension='trust',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        supporting_evidence=['conflicting needs', 'fear-based approach'],
                    )
                )

        return indicators

    def _detect_dependency_language(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect dependency language patterns."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.DEPENDENCY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    AttachmentIndicator(
                        indicator_type='anxious',
                        attachment_dimension='dependency',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.65,
                        supporting_evidence=['dependency expressed'],
                    )
                )

        return indicators

    def _detect_abandonment_fear(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect fear of abandonment patterns."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.ABANDONMENT_FEAR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_attachment_confidence(match.group(), text, 'abandonment_fear')

                indicators.append(
                    AttachmentIndicator(
                        indicator_type='anxious',
                        attachment_dimension='fear_of_abandonment',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        supporting_evidence=['abandonment fear', 'rejection sensitivity'],
                    )
                )

        return indicators

    def _detect_intimacy_avoidance(self, text: str, message_index: int) -> List[AttachmentIndicator]:
        """Detect intimacy avoidance patterns."""
        indicators: List[AttachmentIndicator] = []

        for pattern in self.INTIMACY_AVOIDANCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    AttachmentIndicator(
                        indicator_type='avoidant',
                        attachment_dimension='intimacy_avoidance',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.65,
                        supporting_evidence=['intimacy avoidance', 'emotional distance'],
                    )
                )

        return indicators

    def _calculate_attachment_confidence(self, pattern_text: str, full_text: str, attachment_type: str) -> float:
        """Calculate confidence for attachment indicator."""
        confidence = 0.6

        # Strengthening evidence
        strengtheners = {
            'anxious': ['always', 'need', 'please', 'desperate', 'can\'t live without'],
            'avoidant': ['never', 'don\'t want', 'independent', 'alone', 'distance'],
            'disorganized': ['confused', 'scared', 'both', 'contradictory'],
        }

        if attachment_type in strengtheners:
            found_strengtheners = sum(
                1 for s in strengtheners[attachment_type]
                if s.lower() in full_text.lower()
            )
            confidence += min(found_strengtheners * 0.05, 0.2)

        return min(confidence, 1.0)

    def _extract_context(self, text: str, position: int, context_length: int = 60) -> str:
        """Extract context around a match position."""
        start = max(0, position - context_length)
        end = min(len(text), position + context_length)
        return text[start:end].strip()

    def _build_attachment_profile(
        self, speaker_name: str, indicators: List[AttachmentIndicator]
    ) -> AttachmentProfile:
        """Build overall attachment profile for a speaker."""
        profile = AttachmentProfile(speaker_name=speaker_name)

        if not indicators:
            return profile

        # Count indicators by type
        type_counts = defaultdict(int)
        for indicator in indicators:
            type_counts[indicator.indicator_type] += 1

        profile.secure_indicators = type_counts.get('secure', 0)
        profile.anxious_indicators = type_counts.get('anxious', 0)
        profile.avoidant_indicators = type_counts.get('avoidant', 0)
        profile.disorganized_indicators = type_counts.get('disorganized', 0)

        total_indicators = len(indicators)

        # Calculate style scores (0-1)
        if total_indicators > 0:
            profile.style_scores['secure'] = profile.secure_indicators / total_indicators
            profile.style_scores['anxious'] = profile.anxious_indicators / total_indicators
            profile.style_scores['avoidant'] = profile.avoidant_indicators / total_indicators
            profile.style_scores['disorganized'] = profile.disorganized_indicators / total_indicators

        # Determine primary style
        if profile.style_scores:
            primary_style = max(profile.style_scores, key=profile.style_scores.get)
            primary_score = profile.style_scores[primary_style]

            if primary_score >= 0.4:
                profile.primary_style = primary_style
            else:
                # Check if multiple styles are present
                significant_styles = [
                    s for s, score in profile.style_scores.items() if score > 0.15
                ]
                profile.primary_style = 'mixed' if len(significant_styles) > 1 else 'secure'

        # Calculate specific dimensions
        profile.anxiety_about_relationships = profile.anxious_indicators / max(total_indicators, 1)
        profile.avoidance_of_intimacy = profile.avoidant_indicators / max(total_indicators, 1)
        profile.fear_of_abandonment = self._calculate_abandonment_fear_score(indicators)
        profile.dependency_level = self._calculate_dependency_score(indicators)
        profile.trust_in_others = profile.style_scores.get('secure', 0.5)
        profile.self_reliance = profile.avoidant_indicators / max(total_indicators, 1)

        # Relationship satisfaction
        profile.relationship_satisfaction_proxy = self._calculate_satisfaction(indicators)

        return profile

    def _calculate_abandonment_fear_score(self, indicators: List[AttachmentIndicator]) -> float:
        """Calculate fear of abandonment score."""
        abandonment_indicators = sum(
            1 for ind in indicators
            if ind.attachment_dimension == 'fear_of_abandonment'
        )
        return abandonment_indicators / max(len(indicators), 1)

    def _calculate_dependency_score(self, indicators: List[AttachmentIndicator]) -> float:
        """Calculate dependency level score."""
        dependency_indicators = sum(
            1 for ind in indicators
            if ind.attachment_dimension == 'dependency'
        )
        return dependency_indicators / max(len(indicators), 1)

    def _calculate_satisfaction(self, indicators: List[AttachmentIndicator]) -> float:
        """Calculate relationship satisfaction proxy score."""
        # Placeholder - would be based on positive vs negative language analysis
        anxious_count = sum(1 for ind in indicators if ind.indicator_type == 'anxious')
        avoidant_count = sum(1 for ind in indicators if ind.indicator_type == 'avoidant')
        disorganized_count = sum(1 for ind in indicators if ind.indicator_type == 'disorganized')

        total = len(indicators)
        if total == 0:
            return 0.5

        # Lower insecure attachment = higher satisfaction
        insecure_ratio = (anxious_count + avoidant_count + disorganized_count) / total
        return max(0.0, 1.0 - insecure_ratio)

    def _generate_summary(
        self, attachment_profiles: Dict[str, AttachmentProfile]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not attachment_profiles:
            return {
                'total_speakers_analyzed': 0,
                'primary_styles_distribution': {},
            }

        style_counts = defaultdict(int)
        for profile in attachment_profiles.values():
            style_counts[profile.primary_style] += 1

        avg_anxiety = statistics.mean(
            [p.anxiety_about_relationships for p in attachment_profiles.values()]
        ) if attachment_profiles else 0.0

        avg_avoidance = statistics.mean(
            [p.avoidance_of_intimacy for p in attachment_profiles.values()]
        ) if attachment_profiles else 0.0

        return {
            'total_speakers_analyzed': len(attachment_profiles),
            'primary_styles_distribution': dict(style_counts),
            'avg_relationship_anxiety': avg_anxiety,
            'avg_intimacy_avoidance': avg_avoidance,
            'secure_attachment_rate': (
                style_counts.get('secure', 0) / len(attachment_profiles)
                if attachment_profiles else 0.0
            ),
        }
