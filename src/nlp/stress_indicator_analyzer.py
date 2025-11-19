#!/usr/bin/env python3
"""
Stress Indicator Analyzer Module
Identifies stress markers including stress vocabulary, urgency language, overwhelm indicators,
coping mechanisms, health mentions, and pressure from external sources.
Based on stress and burnout assessment literature.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class StressIndicator:
    """Represents a detected stress indicator"""
    indicator_type: str  # stress_vocabulary, urgency, overwhelm, coping, health, external_pressure, deadline_pressure
    stress_category: str  # work, relational, health, financial, time-related
    text_segment: str
    message_index: int
    confidence: float  # 0-1
    severity: str  # mild, moderate, severe
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class StressProfile:
    """Overall stress profile for a speaker"""
    speaker_name: str
    overall_stress_level: float = 0.0  # 0-1
    perceived_stress_severity: str = "none"  # none, mild, moderate, high, severe
    stress_source_diversity: int = 0  # How many different stress sources
    urgency_language_prevalence: float = 0.0  # 0-1
    overwhelm_level: float = 0.0  # 0-1
    coping_strategy_count: int = 0
    healthy_coping_attempts: int = 0
    unhealthy_coping_references: int = 0
    health_concern_mentions: int = 0
    sleep_issue_indicators: int = 0
    external_pressure_count: int = 0
    deadline_pressure_count: int = 0
    financial_stress_markers: int = 0
    relational_stress_markers: int = 0
    work_stress_markers: int = 0
    burnout_indicators: int = 0
    recovery_capacity: float = 0.5  # Estimated ability to recover
    support_seeking_behavior: float = 0.0


class StressIndicatorAnalyzer:
    """
    Analyzes stress indicators using evidence-based patterns.
    Based on stress theory, burnout literature (Maslach), and coping theory (Lazarus & Folkman).
    """

    # Stress vocabulary patterns
    STRESS_VOCABULARY_PATTERNS = [
        r'\b(stress|stressed|stressful|anxiety|anxious|pressure|pressured)\b',
        r'\b(overwhelming|overwhelmed|burden|burdened|overloaded|drowning)\b',
        r'\b(exhausted|exhaustion|burned\s+out|burnout|fatigue|fatigued)\b',
        r'\b(worried|worry|concerned|concerned|anxious|nervous|tense)\b',
        r'\b(chaos|chaotic|hectic|frantic|crazy|insane|mad)\b',
        r'\b(panic|panicking|panicked|panic\s+attack)\b',
        r'\b(impossible|too\s+much|can\'t\s+handle|can\'t\s+cope)\b',
    ]

    # Urgency language patterns
    URGENCY_PATTERNS = [
        r'\b(urgent|urgently|immediately|asap|right\s+now|this\s+second)\b',
        r'\b(rushing|rush|hurry|hurried|quick|quicksand|race\s+against)\b',
        r'\b(deadline|deadlines|due\s+soon|running\s+out\s+of\s+time)\b',
        r'\b(no\s+time|time\s+is\s+running\s+out|countdown|last\s+minute)\b',
        r'\b(critical|critical\s+situation|critical\s+timing)\b',
        r'\b(emergency|emergencies|crisis|urgent\s+crisis)\b',
        r'\b(can\'t\s+wait|can\'t\s+delay|must\s+be\s+now|must\s+happen\s+now)\b',
    ]

    # Overwhelm indicators
    OVERWHELM_PATTERNS = [
        r'\b(too\s+much|too\s+many|too\s+hard|too\s+difficult|too\s+complex)\b',
        r'\b(can\'t\s+handle|can\'t\s+cope|can\'t\s+manage|can\'t\s+deal)\b',
        r'\b(drowning|sinking|suffocating|drowning\s+in)\b',
        r'\b(all\s+at\s+once|everything\s+at\s+once|hit\s+me\s+all\s+at\s+once)\b',
        r'\b(lost\s+control|out\s+of\s+control|spinning|chaos)\b',
        r'\b(don\'t\s+know\s+where\s+to\s+start|so\s+much\s+to\s+do|endless\s+list)\b',
    ]

    # Coping mechanism references
    HEALTHY_COPING_PATTERNS = [
        r'\b(exercise|workout|gym|run|walk|yoga|meditation)\b',
        r'\b(talk\s+to|share\s+with|confide\s+in|support)\b',
        r'\b(take\s+a\s+break|rest|relax|self\s+care|treat\s+myself)\b',
        r'\b(breathing|breathe|deep\s+breath|deep\s+breaths)\b',
        r'\b(journaling|journal|write|writing|talk\s+to\s+therapist|therapy)\b',
        r'\b(hobby|hobbies|hobby|passion|interest)\b',
        r'\b(time\s+out|step\s+back|pause|slow\s+down)\b',
        r'\b(positive\s+thinking|reframe|perspective|silver\s+lining)\b',
    ]

    UNHEALTHY_COPING_PATTERNS = [
        r'\b(drinking|drink\s+to|alcohol|drunk|wine|beer)\b',
        r'\b(smoking|cigarette|drug|drugs|high|stoned)\b',
        r'\b(sleep\s+too\s+much|oversleep|binge|binge\s+eating|comfort\s+eat)\b',
        r'\b(isolate|isolation|withdraw|avoid\s+people)\b',
        r'\b(work\s+too\s+much|overwork|no\s+break|no\s+time\s+off)\b',
        r'\b(snap\s+at|yell|anger|lose\s+temper|aggressive)\b',
    ]

    # Sleep and health mentions
    SLEEP_HEALTH_PATTERNS = [
        r'\b(can\'t\s+sleep|insomnia|sleeping\s+issues|sleep\s+problem)\b',
        r'\b(sleep\s+deprived|sleep\s+deprivation|not\s+sleeping|no\s+sleep)\b',
        r'\b(nightmares|bad\s+dreams|waking\s+up|wake\s+up\s+at|wake\s+in)\b',
        r'\b(tired|exhausted|fatigue|fatigued|drowsy|can\'t\s+wake\s+up)\b',
        r'\b(headaches?|migraines|backache|back\s+pain|pain)\b',
        r'\b(sick|illness|ill|cold|flu|getting\s+sick)\b',
        r'\b(appetite|eating|eat\s+less|eat\s+more|not\s+eating)\b',
        r'\b(physical\s+health|health\s+issues|health\s+problems|medical)\b',
    ]

    # External pressure sources
    EXTERNAL_PRESSURE_PATTERNS = [
        r'\b(boss|manager|supervisor|work\s+demands|workload)\b',
        r'\b(family\s+pressure|family\s+expectations|parents?\s+want|spouse\s+wants)\b',
        r'\b(expectations|expectation|should|must|have\s+to|supposed\s+to)\b',
        r'\b(demands\s+on|demands\s+from|pressure\s+from|pressure\s+to)\b',
        r'\b(obligations|obligated|committed|commitment|commitments)\b',
        r'\b(expectations\s+are|expectations\s+of|expected\s+to|expected\s+that)\b',
    ]

    # Deadline and time pressure
    DEADLINE_PRESSURE_PATTERNS = [
        r'\b(deadline|deadlines|due\s+date|due\s+by)\b',
        r'\b(day\s+after\s+tomorrow|next\s+week|next\s+month|soon)\b',
        r'\b(running\s+out\s+of\s+time|time\s+is\s+running\s+out|countdown)\b',
        r'\b(in\s+\d+\s+(days?|weeks?|months?)|by\s+\w+day|before\s+\w+)\b',
        r'\b(last\s+minute|last\s+second|down\s+to\s+the\s+wire)\b',
        r'\b(tomorrow|this\s+week|this\s+month|this\s+year)\b',
    ]

    # Financial stress
    FINANCIAL_STRESS_PATTERNS = [
        r'\b(money|financial|finances|debt|bills|mortgage|rent)\b',
        r'\b(afford|can\'t\s+afford|expensive|cost|costs|paying)\b',
        r'\b(broke|broke|no\s+money|running\s+out\s+of\s+money|poor)\b',
        r'\b(financial\s+pressure|financial\s+stress|money\s+stress)\b',
    ]

    # Relational stress
    RELATIONAL_STRESS_PATTERNS = [
        r'\b(conflict|argument|fighting|fight|disagree|disagreement)\b',
        r'\b(relationship\s+issues|relationship\s+problems|relationship\s+stress)\b',
        r'\b(marriage|divorce|separation|custody|family\s+issues)\b',
        r'\b(difficult\s+person|difficult\s+relationship|toxic)\b',
        r'\b(communication\s+breakdown|can\'t\s+talk|talking\s+to\s+them)\b',
    ]

    # Work stress
    WORK_STRESS_PATTERNS = [
        r'\b(work\s+stress|work\s+pressure|workplace\s+stress|job\s+stress)\b',
        r'\b(boss|manager|coworker|colleague|team)\b',
        r'\b(workload|overwhelm\s+at\s+work|too\s+much\s+work|work\s+overload)\b',
        r'\b(performance|performance\s+review|evaluation|meeting\s+deadlines)\b',
        r'\b(project|projects|assignments|tasks|responsibility|responsibilities)\b',
    ]

    # Burnout indicators
    BURNOUT_PATTERNS = [
        r'\b(burned\s+out|burnout|burn\s+out|feeling\s+burned\s+out)\b',
        r'\b(cynical|cynicism|detached|depersonalization)\b',
        r'\b(no\s+longer\s+care|don\'t\s+care\s+anymore|stopped\s+caring)\b',
        r'\b(exhausted|exhaustion|drained|completely\s+drained)\b',
        r'\b(ineffective|ineffectual|can\'t\s+do|can\'t\s+accomplish)\b',
    ]

    # Support seeking
    SUPPORT_SEEKING_PATTERNS = [
        r'\b(help|need\s+help|asking\s+for\s+help|seek\s+help)\b',
        r'\b(support|need\s+support|seeking\s+support|emotional\s+support)\b',
        r'\b(therapist|therapy|counselor|counseling|doctor|medical)\b',
        r'\b(talk\s+to|share\s+with|confide\s+in|open\s+up\s+to)\b',
    ]

    def __init__(self):
        """Initialize stress indicator analyzer"""
        logger.info("StressIndicatorAnalyzer initialized")

    def analyze_stress_indicators(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze stress indicators across a conversation.

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict with detected stress indicators and profiles per speaker
        """
        logger.info(f"Analyzing stress indicators in {len(messages)} messages")

        all_indicators: Dict[str, List[StressIndicator]] = defaultdict(list)
        stress_profiles: Dict[str, StressProfile] = {}

        for idx, message in enumerate(messages):
            text = message.get('text', '')
            sender = message.get('sender', 'Unknown')

            if not text.strip():
                continue

            # Detect all stress indicators
            indicators = self._detect_message_indicators(text, idx, sender)
            all_indicators[sender].extend(indicators)

        # Build profiles for each speaker
        for speaker, indicators in all_indicators.items():
            stress_profiles[speaker] = self._build_stress_profile(speaker, indicators)

        logger.info(
            f"Detected {sum(len(i) for i in all_indicators.values())} total stress indicators"
        )

        return {
            'stress_indicators_by_speaker': dict(all_indicators),
            'stress_profiles': stress_profiles,
            'summary': self._generate_summary(stress_profiles),
        }

    def _detect_message_indicators(
        self, text: str, message_index: int, sender: str
    ) -> List[StressIndicator]:
        """Detect all stress indicators in a single message."""
        indicators: List[StressIndicator] = []

        # Stress vocabulary
        stress_vocab_results = self._detect_stress_vocabulary(text, message_index)
        indicators.extend(stress_vocab_results)

        # Urgency language
        urgency_results = self._detect_urgency_language(text, message_index)
        indicators.extend(urgency_results)

        # Overwhelm
        overwhelm_results = self._detect_overwhelm(text, message_index)
        indicators.extend(overwhelm_results)

        # Coping mechanisms
        coping_results = self._detect_coping_mechanisms(text, message_index)
        indicators.extend(coping_results)

        # Sleep/health mentions
        health_results = self._detect_health_mentions(text, message_index)
        indicators.extend(health_results)

        # External pressure
        pressure_results = self._detect_external_pressure(text, message_index)
        indicators.extend(pressure_results)

        # Deadline pressure
        deadline_results = self._detect_deadline_pressure(text, message_index)
        indicators.extend(deadline_results)

        # Financial stress
        financial_results = self._detect_financial_stress(text, message_index)
        indicators.extend(financial_results)

        # Relational stress
        relational_results = self._detect_relational_stress(text, message_index)
        indicators.extend(relational_results)

        # Work stress
        work_results = self._detect_work_stress(text, message_index)
        indicators.extend(work_results)

        # Burnout
        burnout_results = self._detect_burnout(text, message_index)
        indicators.extend(burnout_results)

        return indicators

    def _detect_stress_vocabulary(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect stress vocabulary patterns."""
        indicators: List[StressIndicator] = []

        for pattern in self.STRESS_VOCABULARY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                severity = self._determine_stress_severity(match.group())

                indicators.append(
                    StressIndicator(
                        indicator_type='stress_vocabulary',
                        stress_category='general',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.75,
                        severity=severity,
                        supporting_evidence=['stress language'],
                    )
                )

        return indicators

    def _detect_urgency_language(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect urgency language patterns."""
        indicators: List[StressIndicator] = []

        for pattern in self.URGENCY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='urgency',
                        stress_category='time-related',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.72,
                        severity='moderate',
                        supporting_evidence=['urgency language'],
                    )
                )

        return indicators

    def _detect_overwhelm(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect overwhelm indicators."""
        indicators: List[StressIndicator] = []

        for pattern in self.OVERWHELM_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='overwhelm',
                        stress_category='general',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.78,
                        severity='severe',
                        supporting_evidence=['overwhelm expressed'],
                    )
                )

        return indicators

    def _detect_coping_mechanisms(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect coping mechanism references."""
        indicators: List[StressIndicator] = []

        for pattern in self.HEALTHY_COPING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='coping',
                        stress_category='general',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.70,
                        severity='mild',
                        supporting_evidence=['healthy coping attempt'],
                    )
                )

        for pattern in self.UNHEALTHY_COPING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='coping',
                        stress_category='general',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.65,
                        severity='severe',
                        supporting_evidence=['unhealthy coping pattern'],
                    )
                )

        return indicators

    def _detect_health_mentions(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect sleep and health mentions."""
        indicators: List[StressIndicator] = []

        for pattern in self.SLEEP_HEALTH_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicator_type = 'health'
                stress_category = 'health'

                indicators.append(
                    StressIndicator(
                        indicator_type=indicator_type,
                        stress_category=stress_category,
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.72,
                        severity='moderate',
                        supporting_evidence=['health/sleep impact'],
                    )
                )

        return indicators

    def _detect_external_pressure(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect external pressure sources."""
        indicators: List[StressIndicator] = []

        for pattern in self.EXTERNAL_PRESSURE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='external_pressure',
                        stress_category='general',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.68,
                        severity='moderate',
                        supporting_evidence=['external pressure'],
                    )
                )

        return indicators

    def _detect_deadline_pressure(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect deadline and time pressure."""
        indicators: List[StressIndicator] = []

        for pattern in self.DEADLINE_PRESSURE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='deadline_pressure',
                        stress_category='time-related',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.75,
                        severity='moderate',
                        supporting_evidence=['deadline pressure'],
                    )
                )

        return indicators

    def _detect_financial_stress(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect financial stress markers."""
        indicators: List[StressIndicator] = []

        for pattern in self.FINANCIAL_STRESS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='external_pressure',
                        stress_category='financial',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.70,
                        severity='moderate',
                        supporting_evidence=['financial stress'],
                    )
                )

        return indicators

    def _detect_relational_stress(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect relational stress markers."""
        indicators: List[StressIndicator] = []

        for pattern in self.RELATIONAL_STRESS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='external_pressure',
                        stress_category='relational',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.72,
                        severity='moderate',
                        supporting_evidence=['relational conflict/stress'],
                    )
                )

        return indicators

    def _detect_work_stress(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect work stress markers."""
        indicators: List[StressIndicator] = []

        for pattern in self.WORK_STRESS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='external_pressure',
                        stress_category='work',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.73,
                        severity='moderate',
                        supporting_evidence=['work-related stress'],
                    )
                )

        return indicators

    def _detect_burnout(self, text: str, message_index: int) -> List[StressIndicator]:
        """Detect burnout indicators."""
        indicators: List[StressIndicator] = []

        for pattern in self.BURNOUT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    StressIndicator(
                        indicator_type='stress_vocabulary',
                        stress_category='work',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.80,
                        severity='severe',
                        supporting_evidence=['burnout indicators'],
                    )
                )

        return indicators

    def _determine_stress_severity(self, text: str) -> str:
        """Determine severity from stress language."""
        severe_terms = ['panic', 'panic attack', 'completely overwhelmed', 'burnout', 'exhausted']
        moderate_terms = ['stress', 'anxious', 'worried', 'pressure', 'overwhelmed']

        text_lower = text.lower()

        if any(term in text_lower for term in severe_terms):
            return 'severe'
        elif any(term in text_lower for term in moderate_terms):
            return 'moderate'
        return 'mild'

    def _extract_context(self, text: str, position: int, context_length: int = 60) -> str:
        """Extract context around a match position."""
        start = max(0, position - context_length)
        end = min(len(text), position + context_length)
        return text[start:end].strip()

    def _build_stress_profile(
        self, speaker_name: str, indicators: List[StressIndicator]
    ) -> StressProfile:
        """Build stress profile for a speaker."""
        profile = StressProfile(speaker_name=speaker_name)

        if not indicators:
            return profile

        # Count each type
        stress_types: Dict[str, int] = defaultdict(int)
        categories: Dict[str, int] = defaultdict(int)
        severity_scores = {'mild': 1, 'moderate': 2, 'severe': 3}
        severity_values = []

        for indicator in indicators:
            stress_types[indicator.indicator_type] += 1
            categories[indicator.stress_category] += 1
            severity_values.append(severity_scores.get(indicator.severity, 1))

        # Count specific markers
        profile.external_pressure_count = stress_types.get('external_pressure', 0)
        profile.deadline_pressure_count = stress_types.get('deadline_pressure', 0)
        profile.financial_stress_markers = sum(
            1 for ind in indicators if ind.stress_category == 'financial'
        )
        profile.relational_stress_markers = sum(
            1 for ind in indicators if ind.stress_category == 'relational'
        )
        profile.work_stress_markers = sum(
            1 for ind in indicators if ind.stress_category == 'work'
        )
        profile.health_concern_mentions = stress_types.get('health', 0)
        profile.sleep_issue_indicators = sum(
            1 for ind in indicators
            if 'sleep' in ind.text_segment.lower() or 'insomnia' in ind.text_segment.lower()
        )

        # Coping strategies
        profile.coping_strategy_count = stress_types.get('coping', 0)
        profile.healthy_coping_attempts = sum(
            1 for ind in indicators
            if ind.indicator_type == 'coping' and 'healthy' in str(ind.supporting_evidence).lower()
        )
        profile.unhealthy_coping_references = sum(
            1 for ind in indicators
            if ind.indicator_type == 'coping' and 'unhealthy' in str(ind.supporting_evidence).lower()
        )

        # Burnout
        profile.burnout_indicators = sum(
            1 for ind in indicators if 'burnout' in ind.text_segment.lower()
        )

        # Diversity of stress sources
        profile.stress_source_diversity = len(categories)

        # Overall stress level
        total = len(indicators)
        if total == 0:
            profile.overall_stress_level = 0.0
        else:
            # Weighted calculation
            overwhelm_weight = stress_types.get('overwhelm', 0) * 2
            stress_vocab_weight = stress_types.get('stress_vocabulary', 0)
            other_weight = total - stress_types.get('overwhelm', 0) - stress_types.get('stress_vocabulary', 0)

            profile.overall_stress_level = min(
                (overwhelm_weight + stress_vocab_weight + other_weight) / (total + 5), 1.0
            )

        # Perceived stress severity
        if profile.overall_stress_level < 0.2:
            profile.perceived_stress_severity = 'none'
        elif profile.overall_stress_level < 0.4:
            profile.perceived_stress_severity = 'mild'
        elif profile.overall_stress_level < 0.6:
            profile.perceived_stress_severity = 'moderate'
        elif profile.overall_stress_level < 0.8:
            profile.perceived_stress_severity = 'high'
        else:
            profile.perceived_stress_severity = 'severe'

        # Urgency prevalence
        profile.urgency_language_prevalence = stress_types.get('urgency', 0) / max(total, 1)

        # Overwhelm level
        profile.overwhelm_level = stress_types.get('overwhelm', 0) / max(total, 1)

        # Recovery capacity (inverse relationship to stress)
        profile.recovery_capacity = max(0.0, 1.0 - profile.overall_stress_level)

        # Support seeking behavior
        profile.support_seeking_behavior = 0.0  # Would require additional indicators

        return profile

    def _generate_summary(self, stress_profiles: Dict[str, StressProfile]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not stress_profiles:
            return {
                'total_speakers_analyzed': 0,
                'high_stress_speakers': 0,
                'avg_stress_level': 0.0,
            }

        stress_levels = [p.overall_stress_level for p in stress_profiles.values()]
        high_stress_count = sum(
            1 for p in stress_profiles.values()
            if p.overall_stress_level > 0.6
        )

        severity_distribution = defaultdict(int)
        for profile in stress_profiles.values():
            severity_distribution[profile.perceived_stress_severity] += 1

        avg_work_stress = statistics.mean(
            [p.work_stress_markers for p in stress_profiles.values()]
        ) if stress_profiles else 0.0

        avg_relational_stress = statistics.mean(
            [p.relational_stress_markers for p in stress_profiles.values()]
        ) if stress_profiles else 0.0

        coping_attempts = sum(
            p.healthy_coping_attempts + p.unhealthy_coping_references
            for p in stress_profiles.values()
        )

        burnout_count = sum(
            1 for p in stress_profiles.values()
            if p.burnout_indicators > 0
        )

        return {
            'total_speakers_analyzed': len(stress_profiles),
            'high_stress_speakers': high_stress_count,
            'avg_stress_level': statistics.mean(stress_levels) if stress_levels else 0.0,
            'perceived_stress_distribution': dict(severity_distribution),
            'avg_work_stress_markers': avg_work_stress,
            'avg_relational_stress_markers': avg_relational_stress,
            'total_coping_attempts': coping_attempts,
            'speakers_with_coping_strategies': sum(
                1 for p in stress_profiles.values()
                if p.coping_strategy_count > 0
            ),
            'speakers_showing_burnout_indicators': burnout_count,
            'sleep_issue_prevalence': sum(
                1 for p in stress_profiles.values()
                if p.sleep_issue_indicators > 0
            ),
        }
