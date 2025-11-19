#!/usr/bin/env python3
"""
Cognitive Distortion Detector Module
Identifies cognitive distortions from CBT (Cognitive Behavioral Therapy) framework.
Detects patterns including all-or-nothing thinking, overgeneralization, catastrophizing,
and other distorted thinking patterns with clinical accuracy and confidence scoring.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class CognitiveDistortion:
    """Represents a detected cognitive distortion"""
    distortion_type: str  # all_or_nothing, overgeneralization, mental_filtering, etc.
    text_segment: str
    message_index: int
    confidence: float  # 0-1
    severity: str  # mild, moderate, severe
    supporting_evidence: List[str] = field(default_factory=list)
    triggering_words: List[str] = field(default_factory=list)


@dataclass
class DistortionProfile:
    """Overall cognitive distortion profile for a speaker"""
    speaker_name: str
    total_distortions: int = 0
    distortion_types: Dict[str, int] = field(default_factory=dict)
    avg_severity: str = "mild"
    distortion_trend: str = "stable"  # increasing, decreasing, stable
    cognitive_flexibility: float = 0.5  # 0-1, higher is more flexible
    black_white_thinking_prevalence: float = 0.0
    catastrophizing_score: float = 0.0
    personalization_score: float = 0.0


class CognitiveDistortionDetector:
    """
    Detects cognitive distortions using evidence-based patterns from CBT literature.
    Based on David D. Burns' work on cognitive distortions.
    """

    # All-or-nothing thinking patterns (binary, extreme thinking)
    ALL_OR_NOTHING_PATTERNS = [
        r'\b(always|never|forever|constantly|completely|totally|entirely|absolutely)\b',
        r'\b(all\s+\w+|no\s+\w+|everyone|nobody|nothing|everything)\b',
        r'\bI\s+(can\'t|won\'t|must|have\s+to)',
        r'\b(perfect|failure|success|disaster)\b',
        r'\b(either|or|black|white)\s+and\s+(white|black)',
        r'\byou\s+(always|never)\s+\w+\s+(me|my)',
    ]

    # Overgeneralization patterns (single event as pattern)
    OVERGENERALIZATION_PATTERNS = [
        r'\bif\s+\w+\s+\w+\b.*\b(then|means|shows|proves|so)\b',
        r'\b(once|one\s+time|that\s+time)\b.*\b(now|always|forever|never)\b',
        r'\bthis\s+(proves|shows|means|demonstrates)\s+\w+',
        r'\b(everyone|all|nobody)\b.*\b(always|never|everyone|all)\b',
        r'\byou\s+(always|never|all\s+the\s+time)\b',
        r'\bI\s+(can\'t|will\s+never|always|forever)\b',
    ]

    # Mental filtering patterns (focusing only on negatives)
    MENTAL_FILTERING_PATTERNS = [
        r'\b(but|however|still|yet|except)\b.*\b(bad|wrong|fail|terrible|awful|horrible|negative)\b',
        r'\bwhat\s+(about|if)\s+\w+\s+\w+',
        r'\bI\s+(only|just)\s+(see|notice|remember|focus)\s+\w+\s+(negative|bad|wrong)',
        r'\b(ignore|overlook|forget)\s+\w+\s+(good|positive|well)',
        r'\bwe\s+(never|don\'t|didn\'t)\s+\w+\s+(right|well|good)',
    ]

    # Catastrophizing patterns (worst case scenario)
    CATASTROPHIZING_PATTERNS = [
        r'\b(what\s+if|what\s+about|imagine|suppose)\s+\w+\s+(happens|occurred|did)',
        r'\b(will|would|could)\s+(fail|collapse|crash|break|fall\s+apart|ruin|destroy|end)',
        r'\b(worst|terrible|horrible|awful|catastrophic|disastrous)\b',
        r'\bthis\s+(will|would)\s+(destroy|ruin|end)\s+(everything|my|our|the)',
        r'\bI\s+(can\'t|won\'t)\s+(survive|handle|cope|manage|bear)\s+\w+',
        r'\b(life\s+is\s+over|I\'m\s+ruined|everything\s+is\s+over)\b',
    ]

    # Personalization patterns (taking responsibility for external events)
    PERSONALIZATION_PATTERNS = [
        r'\b(it\'s|it\s+is)\s+(my|your|his|her|their)\s+(fault|responsibility|blame)',
        r'\bI\s+(made|caused|did|am\s+responsible\s+for)\s+\w+\s+happen',
        r'\bthey\s+(\w+\s+)+because\s+(of\s+)?me',
        r'\bif\s+(I|you|he|she|they)\s+(\w+\s+)+wouldn\'t\s+happen',
        r'\bI\s+(should\s+have|shouldn\'t\s+have)\s+\w+\s+(then|so|otherwise)',
        r'\bthis\s+is\s+because\s+of\s+(my|your|his|her)\s+(actions|words|choice)',
    ]

    # Should/must statements (CBT rigid thinking)
    SHOULD_MUST_PATTERNS = [
        r'\b(should|must|have\s+to|ought\s+to|need\s+to|supposed\s+to)\b',
        r'\b(I|you|they|we|one)\s+(should|must|have\s+to|ought\s+to)\s+',
        r'\b(shouldn\'t|mustn\'t|can\'t|couldn\'t)\s+\w+',
        r'\bif\s+\w+\s+(should|must|ought)\b',
        r'\b(need\s+to|have\s+to|am\s+supposed\s+to)\s+\w+\s+\w+',
    ]

    # Emotional reasoning (feeling = fact)
    EMOTIONAL_REASONING_PATTERNS = [
        r'\bI\s+(feel|felt)\s+(like|that)\s+\w+\s+(is|are|means|proves)\b',
        r'\b(because|since)\s+I\s+(feel|felt|am|am\s+afraid)\b',
        r'\bI\s+(know|think|believe)\s+\w+\s+because\s+I\s+(feel|felt)',
        r'\bif\s+I\s+(feel|felt|am\s+afraid)\s+\w+\s+(then|means|shows)\b',
        r'\b(my\s+fear|my\s+feeling|my\s+emotion)\s+\w+\s+(means|proves|shows|is)\b',
    ]

    # Fortune-telling patterns (predicting the future negatively)
    FORTUNE_TELLING_PATTERNS = [
        r'\b(will|won\'t|never\s+will|always\s+will)\s+\w+\s+\w+\s+(in\s+the\s+)?future',
        r'\b(know|bet|guarantee|sure)\s+\w+\s+(will|won\'t)\s+\w+',
        r'\b(mark\s+my\s+words|you\'ll\s+see|I\s+know\s+what)\s+\w+\s+(will|won\'t)',
        r'\bthis\s+(will|won\'t)\s+\w+\s+\w+',
        r'\b(sooner|later|eventually)\s+\w+\s+(will|happens|occurs)',
    ]

    # Label-making (applying negative labels)
    LABEL_MAKING_PATTERNS = [
        r'\bI\s+am\s+a?\s+(failure|loser|idiot|fool|bad|horrible|terrible)\b',
        r'\byou\s+are\s+(a\s+)?(failure|loser|idiot|fool|bad|abusive|toxic)\b',
        r'\b(people\s+like|he\'s|she\'s|I\'m)\s+\w+\s+(bad|awful|terrible|horrible|damaged)\b',
        r'\b(labeled\s+as|seen\s+as|known\s+as)\s+(a\s+)?(failure|loser|bad)\b',
    ]

    # Mind-reading patterns (assuming others' thoughts)
    MIND_READING_PATTERNS = [
        r'\bthey\s+(think|know|believe|hate|judge|blame)\s+(me|that|I)',
        r'\b(know|bet|can\s+tell)\s+\w+\s+(thinks|believes|knows)\b',
        r'\bthey\s+(don\'t|can\'t)\s+\w+\s+(me|my|what|that)\b',
        r'\b(I\s+can\s+tell|it\'s\s+obvious|I\s+know)\s+\w+\s+\w+\s+(thinks|knows|feels)\b',
    ]

    # Severity thresholds
    SEVERE_MARKERS = ['always', 'never', 'forever', 'completely', 'destroyed', 'ruined']
    MODERATE_MARKERS = ['often', 'usually', 'probably', 'likely', 'might', 'seems']

    def __init__(self):
        """Initialize cognitive distortion detector"""
        logger.info("CognitiveDistortionDetector initialized")
        self.distortion_cache: Dict[str, List[CognitiveDistortion]] = defaultdict(list)

    def detect_distortions(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Detect cognitive distortions across all messages in a conversation.

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict with detected distortions and profiles per speaker
        """
        logger.info(f"Detecting cognitive distortions in {len(messages)} messages")

        all_distortions: Dict[str, List[CognitiveDistortion]] = defaultdict(list)
        distortion_profiles: Dict[str, DistortionProfile] = {}

        for idx, message in enumerate(messages):
            text = message.get('text', '')
            sender = message.get('sender', 'Unknown')

            if not text.strip():
                continue

            # Detect all distortion types
            distortions = self._detect_message_distortions(text, idx, sender)
            all_distortions[sender].extend(distortions)

        # Build profiles for each speaker
        for speaker, distortions in all_distortions.items():
            distortion_profiles[speaker] = self._build_distortion_profile(
                speaker, distortions
            )

        logger.info(
            f"Detected {sum(len(d) for d in all_distortions.values())} total distortions"
        )

        return {
            'distortions_by_speaker': dict(all_distortions),
            'distortion_profiles': distortion_profiles,
            'summary': self._generate_summary(distortion_profiles),
        }

    def _detect_message_distortions(
        self, text: str, message_index: int, sender: str
    ) -> List[CognitiveDistortion]:
        """Detect all distortions in a single message."""
        distortions: List[CognitiveDistortion] = []

        # All-or-nothing thinking
        all_nothing_results = self._detect_all_or_nothing(text, message_index)
        distortions.extend(all_nothing_results)

        # Overgeneralization
        overgeneralization_results = self._detect_overgeneralization(text, message_index)
        distortions.extend(overgeneralization_results)

        # Mental filtering
        mental_filtering_results = self._detect_mental_filtering(text, message_index)
        distortions.extend(mental_filtering_results)

        # Catastrophizing
        catastrophizing_results = self._detect_catastrophizing(text, message_index)
        distortions.extend(catastrophizing_results)

        # Personalization
        personalization_results = self._detect_personalization(text, message_index)
        distortions.extend(personalization_results)

        # Should/must statements
        should_must_results = self._detect_should_must(text, message_index)
        distortions.extend(should_must_results)

        # Emotional reasoning
        emotional_reasoning_results = self._detect_emotional_reasoning(
            text, message_index
        )
        distortions.extend(emotional_reasoning_results)

        # Fortune-telling
        fortune_telling_results = self._detect_fortune_telling(text, message_index)
        distortions.extend(fortune_telling_results)

        # Label-making
        label_making_results = self._detect_label_making(text, message_index)
        distortions.extend(label_making_results)

        # Mind-reading
        mind_reading_results = self._detect_mind_reading(text, message_index)
        distortions.extend(mind_reading_results)

        return distortions

    def _detect_all_or_nothing(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect all-or-nothing (black-and-white) thinking."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.ALL_OR_NOTHING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)
                severity = self._determine_severity(match.group())

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='all_or_nothing_thinking',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity=severity,
                        triggering_words=[match.group()],
                    )
                )

        return distortions

    def _detect_overgeneralization(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect overgeneralization patterns."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.OVERGENERALIZATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Additional check: look for time/quantity generalizations
                if self._is_strong_generalization(text, match):
                    confidence = self._calculate_pattern_confidence(match.group(), text)

                    distortions.append(
                        CognitiveDistortion(
                            distortion_type='overgeneralization',
                            text_segment=self._extract_context(text, match.start()),
                            message_index=message_index,
                            confidence=min(confidence + 0.1, 1.0),
                            severity='moderate',
                            triggering_words=[m.group() for m in re.finditer(r'\b\w+\b', match.group())],
                        )
                    )

        return distortions

    def _detect_mental_filtering(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect mental filtering (focus on negatives, ignore positives)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.MENTAL_FILTERING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='mental_filtering',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity='mild',
                        triggering_words=['but', 'however', 'still', 'yet'],
                    )
                )

        return distortions

    def _detect_catastrophizing(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect catastrophizing (worst-case scenario thinking)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.CATASTROPHIZING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)
                # Catastrophizing is typically moderate to severe
                severity = 'severe' if any(
                    m in match.group().lower() for m in self.SEVERE_MARKERS
                ) else 'moderate'

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='catastrophizing',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity=severity,
                        triggering_words=[match.group()],
                    )
                )

        return distortions

    def _detect_personalization(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect personalization (taking responsibility for external events)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.PERSONALIZATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='personalization',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity='moderate',
                        triggering_words=['my fault', 'my responsibility', 'because of me'],
                    )
                )

        return distortions

    def _detect_should_must(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect should/must statements (rigid thinking patterns)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.SHOULD_MUST_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)
                # Assess if this is a rigid expectation (higher severity)
                severity = 'moderate' if 'I should' in match.group() else 'mild'

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='should_must_statement',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity=severity,
                        triggering_words=['should', 'must', 'have to', 'ought to'],
                    )
                )

        return distortions

    def _detect_emotional_reasoning(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect emotional reasoning (feeling as fact)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.EMOTIONAL_REASONING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='emotional_reasoning',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity='moderate',
                        triggering_words=['feel', 'felt', 'because'],
                    )
                )

        return distortions

    def _detect_fortune_telling(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect fortune-telling (predicting future negatively)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.FORTUNE_TELLING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)
                severity = 'severe' if 'never will' in match.group().lower() else 'moderate'

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='fortune_telling',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity=severity,
                        triggering_words=['will', "won't", 'never', 'always'],
                    )
                )

        return distortions

    def _detect_label_making(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect label-making (applying negative labels)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.LABEL_MAKING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)
                severity = 'severe'  # Label-making is typically severe

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='label_making',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity=severity,
                        triggering_words=['failure', 'loser', 'bad', 'terrible', 'awful'],
                    )
                )

        return distortions

    def _detect_mind_reading(self, text: str, message_index: int) -> List[CognitiveDistortion]:
        """Detect mind-reading (assuming others' thoughts)."""
        distortions: List[CognitiveDistortion] = []

        for pattern in self.MIND_READING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_pattern_confidence(match.group(), text)

                distortions.append(
                    CognitiveDistortion(
                        distortion_type='mind_reading',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        severity='moderate',
                        triggering_words=['think', 'know', 'believe', 'judge'],
                    )
                )

        return distortions

    def _calculate_pattern_confidence(self, pattern_text: str, full_text: str) -> float:
        """Calculate confidence score for a pattern match."""
        confidence = 0.6  # Base confidence

        # Boost for multiple supporting words
        supporting_words = [
            'always', 'never', 'completely', 'totally', 'entirely',
            'definitely', 'certainly', 'absolutely', 'obviously'
        ]
        supporting_count = sum(
            1 for word in supporting_words
            if word.lower() in full_text.lower()
        )
        confidence += min(supporting_count * 0.05, 0.25)

        # Context boost if pattern is in first 30% of message (more emphasis)
        if full_text.find(pattern_text) < len(full_text) * 0.3:
            confidence += 0.05

        return min(confidence, 1.0)

    def _determine_severity(self, text: str) -> str:
        """Determine severity of distortion based on language intensity."""
        text_lower = text.lower()

        if any(m in text_lower for m in self.SEVERE_MARKERS):
            return 'severe'
        elif any(m in text_lower for m in self.MODERATE_MARKERS):
            return 'moderate'
        return 'mild'

    def _is_strong_generalization(self, text: str, match: re.Match) -> bool:
        """Check if a potential overgeneralization is strong."""
        context = self._extract_context(text, match.start())
        strong_generalizations = [
            'always', 'never', 'every time', 'forever', 'completely',
            'entirely', 'all the time'
        ]
        return any(g in context.lower() for g in strong_generalizations)

    def _extract_context(self, text: str, position: int, context_length: int = 60) -> str:
        """Extract context around a match position."""
        start = max(0, position - context_length)
        end = min(len(text), position + context_length)
        return text[start:end].strip()

    def _build_distortion_profile(
        self, speaker_name: str, distortions: List[CognitiveDistortion]
    ) -> DistortionProfile:
        """Build overall distortion profile for a speaker."""
        profile = DistortionProfile(speaker_name=speaker_name)

        if not distortions:
            return profile

        profile.total_distortions = len(distortions)

        # Count distortions by type
        type_counts: Dict[str, int] = defaultdict(int)
        severity_scores = {'mild': 1, 'moderate': 2, 'severe': 3}
        severity_values = []

        for distortion in distortions:
            type_counts[distortion.distortion_type] += 1
            severity_values.append(severity_scores.get(distortion.severity, 1))

        profile.distortion_types = dict(type_counts)

        # Average severity
        if severity_values:
            avg_severity_score = statistics.mean(severity_values)
            if avg_severity_score < 1.5:
                profile.avg_severity = 'mild'
            elif avg_severity_score < 2.5:
                profile.avg_severity = 'moderate'
            else:
                profile.avg_severity = 'severe'

        # Cognitive flexibility (inverse of distortion density)
        profile.cognitive_flexibility = max(
            0.0, 1.0 - (len(distortions) / (len(distortions) + 10))
        )

        # Specific prevalence scores
        profile.black_white_thinking_prevalence = type_counts.get(
            'all_or_nothing_thinking', 0
        ) / max(len(distortions), 1)
        profile.catastrophizing_score = type_counts.get(
            'catastrophizing', 0
        ) / max(len(distortions), 1)
        profile.personalization_score = type_counts.get(
            'personalization', 0
        ) / max(len(distortions), 1)

        return profile

    def _generate_summary(
        self, distortion_profiles: Dict[str, DistortionProfile]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not distortion_profiles:
            return {
                'total_speakers_analyzed': 0,
                'speakers_with_distortions': 0,
                'most_common_distortion': None,
                'avg_distortions_per_speaker': 0.0,
            }

        speakers_with_distortions = sum(
            1 for p in distortion_profiles.values() if p.total_distortions > 0
        )

        all_types: Dict[str, int] = defaultdict(int)
        for profile in distortion_profiles.values():
            for dtype, count in profile.distortion_types.items():
                all_types[dtype] += count

        total_distortions = sum(p.total_distortions for p in distortion_profiles.values())

        return {
            'total_speakers_analyzed': len(distortion_profiles),
            'speakers_with_distortions': speakers_with_distortions,
            'total_distortions': total_distortions,
            'most_common_distortion': max(all_types, key=all_types.get) if all_types else None,
            'distortion_distribution': dict(all_types),
            'avg_distortions_per_speaker': (
                total_distortions / len(distortion_profiles) if distortion_profiles else 0.0
            ),
        }
