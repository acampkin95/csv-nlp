#!/usr/bin/env python3
"""
Trauma Language Detector Module
Identifies trauma-related language patterns including dissociation, hypervigilance, PTSD
indicators, shame/guilt, self-blame, and trust issues. Uses trauma-informed care principles.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TraumaIndicator:
    """Represents a detected trauma language indicator"""
    indicator_type: str  # dissociation, hypervigilance, avoidance, re_experiencing, shame, self_blame, safety_concern, trust_issue
    symptom_category: str  # PTSD, Complex PTSD, or other trauma response
    text_segment: str
    message_index: int
    confidence: float  # 0-1
    clinical_significance: str  # mild, moderate, severe, critical
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class TraumaProfile:
    """Overall trauma response profile for a speaker"""
    speaker_name: str
    trauma_exposure_likelihood: float = 0.0  # 0-1
    dissociation_markers: int = 0
    hypervigilance_markers: int = 0
    avoidance_markers: int = 0
    re_experiencing_markers: int = 0
    shame_guilt_markers: int = 0
    self_blame_markers: int = 0
    safety_concern_markers: int = 0
    trust_issue_markers: int = 0

    # Composite indicators
    ptsd_symptom_severity: str = "none"  # none, mild, moderate, severe
    complex_trauma_indicators: int = 0
    dissociative_tendency: float = 0.0
    hypervigilance_level: float = 0.0
    avoidance_behavior_strength: float = 0.0
    emotional_numbing: float = 0.0
    safety_perception: float = 0.5  # 0-1, lower = less safe
    interpersonal_trust_level: float = 0.5  # 0-1, lower = less trust
    trauma_recovery_stage: str = "uncertain"  # denial, anger, bargaining, depression, acceptance


class TraumaLanguageDetector:
    """
    Detects trauma-related language patterns using evidence-based indicators.
    Based on DSM-5 PTSD criteria and Complex PTSD (C-PTSD) frameworks.
    """

    # Dissociation markers
    DISSOCIATION_PATTERNS = [
        r'\b(dissociate|dissociation|zoned\s+out|spacing\s+out|blank|detached)\b',
        r'\b(not\s+feeling|numb|empty|can\'t\s+feel|emotionally\s+dead|watching\s+myself)\b',
        r'\b(outside\s+of|looking\s+down|out\s+of\s+body|floating|separated\s+from)\b',
        r'\b(surreal|unreal|like\s+a\s+dream|foggy|hazy|not\s+present)\b',
        r'\b(can\'t\s+remember|memory\s+is\s+blank|blackout|lost\s+time)\b',
        r'\b(like\s+I\s+am\s+watching|watching\s+myself|as\s+if|autopilot)\b',
    ]

    # Hypervigilance patterns
    HYPERVIGILANCE_PATTERNS = [
        r'\b(always\s+watching|constantly\s+alert|on\s+guard|hypervigilant|scanning)\b',
        r'\b(jump\s+at|startle\s+at|reaction\s+to\s+sounds|react\s+to)\b',
        r'\b(alert\s+for|looking\s+for|expecting|anticipating)\s+(danger|threat|trouble)',
        r'\b(can\'t\s+relax|can\'t\s+let\s+down|always\s+on\s+edge|edge)\b',
        r'\b(suspicious\s+of|don\'t\s+trust|think\s+everyone)\b',
        r'\b(notice\s+everything|pay\s+attention\s+to|hyper\s+aware)\b',
        r'\b(exits|escape\s+routes|exits|threats|danger)\b',
    ]

    # Avoidance patterns (trauma-specific)
    AVOIDANCE_PATTERNS = [
        r'\b(avoid|avoiding|avoid\s+talking|avoid\s+thinking)\s+(about|discussing|memories)',
        r'\b(don\'t\s+want\s+to|refuse\s+to|won\'t)\s+(remember|talk|discuss|think)',
        r'\b(triggers?|triggered|avoid\s+situations?|situations?\s+remind)\b',
        r'\b(reminders|avoid\s+reminders|can\'t\s+go|can\'t\s+be|can\'t\s+hear)\b',
        r'\b(numb|numbness|emotional\s+numbing|shut\s+down|shut\s+off)\b',
        r'\b(block\s+out|push\s+away|suppress|repress)\b',
    ]

    # Re-experiencing patterns (PTSD flashbacks, nightmares)
    RE_EXPERIENCING_PATTERNS = [
        r'\b(flashback|flashbacks|replay|replaying|keep\s+seeing|haunted)\b',
        r'\b(nightmare|bad\s+dream|dream|night\s+terrors|waking\s+nightmares)\b',
        r'\b(comes\s+back|keeps\s+coming\s+back|can\'t\s+get\s+out|intrusive\s+thoughts?)\b',
        r'\b(relive|reliving|happening\s+again|like\s+it\s+was\s+yesterday)\b',
        r'\b(vivid|memories|haunting|can\'t\s+forget|won\'t\s+go\s+away)\b',
        r'\b(triggered|trigger|reminded\s+of|brings\s+back)\b',
    ]

    # Shame/guilt patterns
    SHAME_GUILT_PATTERNS = [
        r'\b(ashamed|shame|humiliated|degraded|worthless|defective|damaged|broken)\b',
        r'\b(guilty|guilt|wrong|bad|evil|dirty|contaminated|tainted)\b',
        r'\b(embarrassed|mortified|exposed|exposed\s+as|unworthy|not\s+good\s+enough)\b',
        r'\b(shameful|disgraceful|despicable|disgusting)\b',
        r'\b(I\s+am\s+(a\s+)?(failure|loser|bad|broken|damaged|worthless))\b',
    ]

    # Self-blame patterns
    SELF_BLAME_PATTERNS = [
        r'\b(my\s+fault|I\s+caused|I\s+made\s+it|I\s+let|I\s+allowed|I\s+should\s+have)\b',
        r'\b(I\s+could\s+have|I\s+should\'ve|if\s+I\s+had|why\s+didn\'t\s+I)\b',
        r'\b(blamed\s+myself|blame\s+myself|blaming\s+myself|self\s+blame)\b',
        r'\b(should\s+have\s+known|should\s+have\s+seen|should\s+have\s+prevented)\b',
        r'\b(responsible\s+for|caused\s+this|made\s+it\s+happen|brought\s+this)\b',
        r'\b(stupid\s+for|foolish\s+for|naive\s+for|dumb\s+for)\b',
    ]

    # Safety concern patterns
    SAFETY_PATTERNS = [
        r'\b(not\s+safe|unsafe|danger|dangerous|threat|threatened)\b',
        r'\b(can\'t\s+feel\s+safe|never\s+safe|afraid|scared|terrified)\b',
        r'\b(vulnerable|exposed|in\s+danger|at\s+risk)\b',
        r'\b(protect|protection|guard|guarding|defensive)\b',
        r'\b(escape|need\s+to\s+escape|can\'t\s+escape|trapped)\b',
        r'\b(helpless|powerless|no\s+control|can\'t\s+protect)\b',
    ]

    # Trust issues patterns
    TRUST_ISSUE_PATTERNS = [
        r'\b(don\'t\s+trust|can\'t\s+trust|won\'t\s+trust|trust\s+issues)\b',
        r'\b(betrayal|betrayed|let\s+down|disappointed)\b',
        r'\b(lied|liar|lie|lying|lie\s+to\s+me|dishonest)\b',
        r'\b(manipulation|manipulative|manipulated|being\s+played)\b',
        r'\b(ulterior\s+motive|hidden\s+agenda|too\s+good\s+to\s+be\s+true)\b',
        r'\b(fool\s+me|won\'t\s+fall\s+for|won\'t\s+be\s+fooled)\b',
        r'\b(test|testing\s+you|prove\s+yourself|prove\s+it)\b',
    ]

    # Emotional numbing patterns
    NUMBING_PATTERNS = [
        r'\b(numb|numbness|can\'t\s+feel|emotionally\s+dead|empty|hollow)\b',
        r'\b(no\s+longer\s+feel|stopped\s+feeling|can\'t\s+cry|can\'t\s+care)\b',
        r'\b(disconnected|detached|isolated|alone|withdrawn)\b',
        r'\b(shut\s+down|shutting\s+down|turned\s+off|turned\s+myself\s+off)\b',
    ]

    # Complex PTSD indicators (relational/interpersonal trauma)
    COMPLEX_TRAUMA_PATTERNS = [
        r'\b(ongoing|repeated|prolonged|chronic|ongoing\s+abuse)\b',
        r'\b(abuse|abused|abusive|exploitation|exploited)\b',
        r'\b(patterns?|cycle|keep\s+repeating|happen\s+again)\b',
        r'\b(relationship\s+problems|relational|interpersonal|difficult\s+relationships?)\b',
    ]

    def __init__(self):
        """Initialize trauma language detector"""
        logger.info("TraumaLanguageDetector initialized")

    def detect_trauma_language(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Detect trauma-related language patterns across a conversation.

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict with detected trauma indicators and profiles per speaker
        """
        logger.info(f"Detecting trauma language in {len(messages)} messages")

        all_indicators: Dict[str, List[TraumaIndicator]] = defaultdict(list)
        trauma_profiles: Dict[str, TraumaProfile] = {}

        for idx, message in enumerate(messages):
            text = message.get('text', '')
            sender = message.get('sender', 'Unknown')

            if not text.strip():
                continue

            # Detect all trauma indicators
            indicators = self._detect_message_indicators(text, idx, sender)
            all_indicators[sender].extend(indicators)

        # Build profiles for each speaker
        for speaker, indicators in all_indicators.items():
            trauma_profiles[speaker] = self._build_trauma_profile(speaker, indicators)

        logger.info(
            f"Detected {sum(len(i) for i in all_indicators.values())} total trauma indicators"
        )

        return {
            'trauma_indicators_by_speaker': dict(all_indicators),
            'trauma_profiles': trauma_profiles,
            'summary': self._generate_summary(trauma_profiles),
        }

    def _detect_message_indicators(
        self, text: str, message_index: int, sender: str
    ) -> List[TraumaIndicator]:
        """Detect all trauma indicators in a single message."""
        indicators: List[TraumaIndicator] = []

        # Dissociation
        dissociation_results = self._detect_dissociation(text, message_index)
        indicators.extend(dissociation_results)

        # Hypervigilance
        hypervigilance_results = self._detect_hypervigilance(text, message_index)
        indicators.extend(hypervigilance_results)

        # Avoidance
        avoidance_results = self._detect_avoidance(text, message_index)
        indicators.extend(avoidance_results)

        # Re-experiencing
        re_experiencing_results = self._detect_re_experiencing(text, message_index)
        indicators.extend(re_experiencing_results)

        # Shame/guilt
        shame_guilt_results = self._detect_shame_guilt(text, message_index)
        indicators.extend(shame_guilt_results)

        # Self-blame
        self_blame_results = self._detect_self_blame(text, message_index)
        indicators.extend(self_blame_results)

        # Safety concerns
        safety_results = self._detect_safety_concerns(text, message_index)
        indicators.extend(safety_results)

        # Trust issues
        trust_results = self._detect_trust_issues(text, message_index)
        indicators.extend(trust_results)

        # Emotional numbing
        numbing_results = self._detect_emotional_numbing(text, message_index)
        indicators.extend(numbing_results)

        # Complex trauma
        complex_results = self._detect_complex_trauma(text, message_index)
        indicators.extend(complex_results)

        return indicators

    def _detect_dissociation(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect dissociation markers."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.DISSOCIATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='dissociation',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.75,
                        clinical_significance='moderate',
                        supporting_evidence=['dissociative symptoms'],
                    )
                )

        return indicators

    def _detect_hypervigilance(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect hypervigilance markers."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.HYPERVIGILANCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = self._calculate_trauma_confidence(match.group(), text)

                indicators.append(
                    TraumaIndicator(
                        indicator_type='hypervigilance',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=confidence,
                        clinical_significance='moderate',
                        supporting_evidence=['heightened threat detection'],
                    )
                )

        return indicators

    def _detect_avoidance(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect avoidance patterns (trauma-specific)."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.AVOIDANCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='avoidance',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.70,
                        clinical_significance='moderate',
                        supporting_evidence=['trauma memory avoidance'],
                    )
                )

        return indicators

    def _detect_re_experiencing(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect re-experiencing indicators (flashbacks, nightmares)."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.RE_EXPERIENCING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                severity = 'severe' if 'flashback' in match.group().lower() else 'moderate'

                indicators.append(
                    TraumaIndicator(
                        indicator_type='re_experiencing',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.80,
                        clinical_significance=severity,
                        supporting_evidence=['intrusive memories', 'flashbacks/nightmares'],
                    )
                )

        return indicators

    def _detect_shame_guilt(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect shame and guilt expressions."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.SHAME_GUILT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='shame',
                        symptom_category='Complex PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.75,
                        clinical_significance='moderate',
                        supporting_evidence=['shame/guilt expression'],
                    )
                )

        return indicators

    def _detect_self_blame(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect self-blame language."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.SELF_BLAME_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='self_blame',
                        symptom_category='Complex PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.78,
                        clinical_significance='moderate',
                        supporting_evidence=['self-blame', 'self-attribution'],
                    )
                )

        return indicators

    def _detect_safety_concerns(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect safety concern indicators."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.SAFETY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                severity = 'severe' if 'trapped' in match.group().lower() else 'moderate'

                indicators.append(
                    TraumaIndicator(
                        indicator_type='safety_concern',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.72,
                        clinical_significance=severity,
                        supporting_evidence=['safety threat perception'],
                    )
                )

        return indicators

    def _detect_trust_issues(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect trust issue markers."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.TRUST_ISSUE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='trust_issue',
                        symptom_category='Complex PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.70,
                        clinical_significance='moderate',
                        supporting_evidence=['trust violation perception'],
                    )
                )

        return indicators

    def _detect_emotional_numbing(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect emotional numbing indicators."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.NUMBING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='avoidance',
                        symptom_category='PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.75,
                        clinical_significance='moderate',
                        supporting_evidence=['emotional numbing', 'dissociative avoidance'],
                    )
                )

        return indicators

    def _detect_complex_trauma(self, text: str, message_index: int) -> List[TraumaIndicator]:
        """Detect Complex PTSD indicators (relational trauma patterns)."""
        indicators: List[TraumaIndicator] = []

        for pattern in self.COMPLEX_TRAUMA_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    TraumaIndicator(
                        indicator_type='complex_trauma',
                        symptom_category='Complex PTSD',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        confidence=0.65,
                        clinical_significance='moderate',
                        supporting_evidence=['prolonged/repeated trauma', 'relational trauma'],
                    )
                )

        return indicators

    def _calculate_trauma_confidence(self, pattern_text: str, full_text: str) -> float:
        """Calculate confidence for trauma indicator."""
        confidence = 0.65

        # Strengthening markers
        intensity_markers = ['always', 'constantly', 'never', 'forever', 'can\'t']
        found_intensity = sum(
            1 for m in intensity_markers
            if m.lower() in full_text.lower()
        )
        confidence += min(found_intensity * 0.03, 0.15)

        return min(confidence, 1.0)

    def _extract_context(self, text: str, position: int, context_length: int = 60) -> str:
        """Extract context around a match position."""
        start = max(0, position - context_length)
        end = min(len(text), position + context_length)
        return text[start:end].strip()

    def _build_trauma_profile(
        self, speaker_name: str, indicators: List[TraumaIndicator]
    ) -> TraumaProfile:
        """Build trauma response profile for a speaker."""
        profile = TraumaProfile(speaker_name=speaker_name)

        if not indicators:
            return profile

        # Count each type of indicator
        for indicator in indicators:
            if indicator.indicator_type == 'dissociation':
                profile.dissociation_markers += 1
            elif indicator.indicator_type == 'hypervigilance':
                profile.hypervigilance_markers += 1
            elif indicator.indicator_type == 'avoidance':
                profile.avoidance_markers += 1
            elif indicator.indicator_type == 're_experiencing':
                profile.re_experiencing_markers += 1
            elif indicator.indicator_type == 'shame':
                profile.shame_guilt_markers += 1
            elif indicator.indicator_type == 'self_blame':
                profile.self_blame_markers += 1
            elif indicator.indicator_type == 'safety_concern':
                profile.safety_concern_markers += 1
            elif indicator.indicator_type == 'trust_issue':
                profile.trust_issue_markers += 1

        # Complex trauma count
        profile.complex_trauma_indicators = sum(
            1 for ind in indicators
            if ind.symptom_category == 'Complex PTSD'
        )

        # Calculate PTSD severity
        ptsd_count = profile.dissociation_markers + profile.hypervigilance_markers + \
                     profile.avoidance_markers + profile.re_experiencing_markers
        total_indicators = len(indicators)

        if ptsd_count == 0:
            profile.ptsd_symptom_severity = 'none'
        elif ptsd_count <= 2:
            profile.ptsd_symptom_severity = 'mild'
        elif ptsd_count <= 5:
            profile.ptsd_symptom_severity = 'moderate'
        else:
            profile.ptsd_symptom_severity = 'severe'

        # Calculate specific metrics
        profile.dissociative_tendency = profile.dissociation_markers / max(total_indicators, 1)
        profile.hypervigilance_level = profile.hypervigilance_markers / max(total_indicators, 1)
        profile.avoidance_behavior_strength = profile.avoidance_markers / max(total_indicators, 1)
        profile.emotional_numbing = sum(
            1 for ind in indicators if 'numbing' in str(ind.supporting_evidence).lower()
        ) / max(total_indicators, 1)

        # Safety and trust perception
        profile.safety_perception = max(0.0, 1.0 - (profile.safety_concern_markers / max(total_indicators, 5)))
        profile.interpersonal_trust_level = max(0.0, 1.0 - (profile.trust_issue_markers / max(total_indicators, 5)))

        # Trauma exposure likelihood
        if ptsd_count > 0 or profile.complex_trauma_indicators > 0:
            profile.trauma_exposure_likelihood = min(
                (ptsd_count + profile.complex_trauma_indicators) / (total_indicators + 5), 1.0
            )

        # Recovery stage assessment (simplified)
        if profile.ptsd_symptom_severity == 'severe':
            profile.trauma_recovery_stage = 'acute'
        elif profile.shame_guilt_markers > 2 or profile.self_blame_markers > 2:
            profile.trauma_recovery_stage = 'bargaining'
        elif profile.re_experiencing_markers > profile.avoidance_markers:
            profile.trauma_recovery_stage = 'anger'
        else:
            profile.trauma_recovery_stage = 'uncertain'

        return profile

    def _generate_summary(self, trauma_profiles: Dict[str, TraumaProfile]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not trauma_profiles:
            return {
                'total_speakers_analyzed': 0,
                'trauma_exposed_speakers': 0,
                'avg_trauma_exposure': 0.0,
            }

        exposed_speakers = sum(
            1 for p in trauma_profiles.values()
            if p.trauma_exposure_likelihood > 0.3
        )

        avg_exposure = statistics.mean(
            [p.trauma_exposure_likelihood for p in trauma_profiles.values()]
        ) if trauma_profiles else 0.0

        ptsd_severity_counts = defaultdict(int)
        for profile in trauma_profiles.values():
            ptsd_severity_counts[profile.ptsd_symptom_severity] += 1

        avg_hypervigilance = statistics.mean(
            [p.hypervigilance_level for p in trauma_profiles.values()]
        ) if trauma_profiles else 0.0

        avg_trust = statistics.mean(
            [p.interpersonal_trust_level for p in trauma_profiles.values()]
        ) if trauma_profiles else 0.0

        return {
            'total_speakers_analyzed': len(trauma_profiles),
            'trauma_exposed_speakers': exposed_speakers,
            'avg_trauma_exposure_likelihood': avg_exposure,
            'ptsd_severity_distribution': dict(ptsd_severity_counts),
            'avg_hypervigilance_level': avg_hypervigilance,
            'avg_interpersonal_trust_level': avg_trust,
            'complex_trauma_indicated': sum(
                1 for p in trauma_profiles.values()
                if p.complex_trauma_indicators > 0
            ),
        }
