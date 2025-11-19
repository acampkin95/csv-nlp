#!/usr/bin/env python3
"""
Emotional Regulation Tracker Module
Analyzes emotional regulation capacity, volatility, and coping patterns.
Based on emotion regulation theory (Gross & John) and affects theory.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class EmotionalEvent:
    """Represents an emotional event or shift"""
    event_type: str  # escalation, de_escalation, volatility, recovery, regulation_attempt
    emotion_level: float  # 0-1, where 1 is most intense
    emotional_tone: str  # positive, negative, neutral, mixed
    text_segment: str
    message_index: int
    regulation_strategy: Optional[str] = None
    effectiveness: Optional[str] = None  # effective, ineffective, unknown


@dataclass
class RegulationProfile:
    """Overall emotional regulation profile for a speaker"""
    speaker_name: str
    overall_volatility: float = 0.0  # 0-1, higher = more volatile
    mood_swing_frequency: int = 0
    emotional_baseline: float = 0.0  # 0-1 emotional intensity baseline
    peak_emotion_level: float = 0.0  # 0-1 maximum intensity
    trough_emotion_level: float = 0.0  # 0-1 minimum intensity
    recovery_capability: float = 0.5  # 0-1, ability to calm down
    self_soothing_competence: float = 0.0  # Evidence of self-soothing
    escalation_risk: float = 0.0  # 0-1 risk of further escalation
    flooding_indicators: int = 0
    total_regulation_attempts: int = 0
    successful_regulation_attempts: int = 0
    regulation_success_rate: float = 0.0
    coping_strategies: Dict[str, int] = field(default_factory=dict)
    emotional_flexibility: float = 0.5  # 0-1, ability to shift emotions


class EmotionalRegulationTracker:
    """
    Tracks emotional regulation patterns and capacity using evidence-based indicators.
    Based on Gross & John's process model of emotion regulation.
    """

    # Emotional volatility indicators
    VOLATILITY_PATTERNS = [
        r'\b(suddenly|abruptly|out\s+of\s+nowhere)\s+\w+\s+(angry|upset|sad|happy)',
        r'\b(mood\s+swings?|emotional\s+roller\s+coaster|ups\s+and\s+downs)\b',
        r'\b(one\s+minute|one\s+second)\s+\w+\s+(happy|sad|angry)',
        r'\b(changed|shifted)\s+\w+(quickly|drastically|suddenly)\b',
    ]

    # Self-soothing language patterns
    SELF_SOOTHING_PATTERNS = [
        r'\b(calm\s+down|take\s+a\s+breath|breathe|relax|collect\s+myself)\b',
        r'\b(deep\s+breath|breathing\s+exercise|meditation|yoga)\b',
        r'\b(walk|exercise|go\s+for\s+a|physical\s+activity)\b',
        r'\b(talk\s+to|share\s+with|confide\s+in)\s+\w+',
        r'\b(time\s+out|step\s+back|pause|take\s+time)\b',
        r'\b(help\s+myself|distract|focus\s+on|think\s+about)\b',
        r'\b(self\s+care|treat\s+myself|rest|sleep|eat)\b',
    ]

    # Escalation patterns (emotional intensity increasing)
    ESCALATION_PATTERNS = [
        r'\b(more\s+and\s+more|increasingly|getting\s+worse)\s+\w+\s+(angry|upset|frustrated)',
        r'\b(can\'t\s+take\s+it|can\'t\s+handle|can\'t\s+stand)\s+\w+',
        r'\b(furious|enraged|infuriated|livid|devastated|shattered)\b',
        r'\b(worse|worse\s+than)\s+\w+(ever|before|before)\b',
        r'\b(building\s+up|building\s+pressure|pressure\s+building)\b',
        r'\b(breaking\s+point|snapping|exploding|going\s+crazy)\b',
    ]

    # De-escalation attempts
    DE_ESCALATION_PATTERNS = [
        r'\b(I\'m\s+trying|I\'m\s+attempting)\s+to\s+(stay\s+calm|not\s+get|not\s+be)',
        r'\b(trying\s+to|trying|attempt)\s+(understand|see|listen|hear)',
        r'\b(slow\s+down|take\s+time|pause|wait|think)\b',
        r'\b(let\s+me|let\'s)\s+(think|talk|discuss|calm\s+down)',
        r'\b(I\s+get\s+it|I\s+understand|I\s+hear\s+you)\b',
        r'\b(sorry|apology|wrong|my\s+fault|I\s+was\s+wrong)\b',
        r'\b(let\'s\s+move\s+on|let\'s\s+forget|let\'s\s+try\s+again)\b',
    ]

    # Emotional flooding (overwhelm) indicators
    FLOODING_PATTERNS = [
        r'\b(overwhelm|overwhelmed|drowning|suffocating|lost)\b',
        r'\b(can\'t\s+think|can\'t\s+focus|can\'t\s+breathe|can\'t\s+cope)\b',
        r'\b(shutting\s+down|shutting\s+off|numb|empty|blank)\b',
        r'\b(too\s+much|too\s+many|too\s+hard|too\s+painful)\b',
        r'\b(breaking\s+down|breaking\s+apart|falling\s+apart|losing\s+it)\b',
        r'\b(all\s+at\s+once|everything\s+at\s+once|hit\s+me\s+all\s+at\s+once)\b',
    ]

    # Recovery patterns
    RECOVERY_PATTERNS = [
        r'\b(feel\s+better|feeling\s+better|better\s+now|improved|improving)\b',
        r'\b(moved\s+past|got\s+over|let\s+it\s+go|accepted|moved\s+forward)\b',
        r'\b(calm\s+now|calm\s+again|peaceful|content|at\s+peace)\b',
        r'\b(time\s+helped|helped\s+me|support\s+helped)\b',
        r'\b(perspective|see\s+things\s+differently|understand\s+now)\b',
    ]

    # Cognitive reappraisal (healthy coping)
    COGNITIVE_REAPPRAISAL_PATTERNS = [
        r'\b(maybe|perhaps|could\s+be|might\s+be)\s+\w+',
        r'\b(another\s+way\s+to\s+look\s+at|perspective|view)\b',
        r'\b(understand\s+why|see\s+their\s+point|understand\s+them)\b',
        r'\b(it\'s\s+not\s+that|it\'s\s+not\s+all)\b',
        r'\b(silver\s+lining|bright\s+side|positive\s+side)\b',
    ]

    # Rumination (unhealthy coping)
    RUMINATION_PATTERNS = [
        r'\b(can\'t\s+stop\s+thinking|can\'t\s+get|keep\s+thinking)\s+\w+',
        r'\b(replaying|replay|keep\s+replaying|going\s+over)\b',
        r'\b(stuck\s+on|fixated\s+on|obsessing)\b',
        r'\b(why\s+me|why\s+did|why\s+can\'t|what\s+if)\b',
        r'\b(always\s+thinking\s+about|can\'t\s+stop\s+thinking)\b',
    ]

    # Avoidance (unhealthy coping)
    AVOIDANCE_PATTERNS = [
        r'\b(don\'t\s+want\s+to|avoid|avoid\s+talking|don\'t\s+want\s+to\s+deal)\b',
        r'\b(not\s+dealing\s+with|pretend|ignore|block\s+out)\b',
        r'\b(don\'t\s+bring\s+it\s+up|let\'s\s+not\s+discuss|change\s+subject)\b',
    ]

    def __init__(self):
        """Initialize emotional regulation tracker"""
        logger.info("EmotionalRegulationTracker initialized")

    def track_emotional_regulation(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Track emotional regulation patterns across a conversation.

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict with emotional events, profiles, and analysis
        """
        logger.info(f"Tracking emotional regulation in {len(messages)} messages")

        all_events: Dict[str, List[EmotionalEvent]] = defaultdict(list)
        emotional_profiles: Dict[str, RegulationProfile] = {}
        emotion_trajectory: Dict[str, List[float]] = defaultdict(list)

        for idx, message in enumerate(messages):
            text = message.get('text', '')
            sender = message.get('sender', 'Unknown')

            if not text.strip():
                continue

            # Detect emotional events
            events = self._detect_emotional_events(text, idx, sender)
            all_events[sender].extend(events)

            # Track emotional intensity
            intensity = self._calculate_emotional_intensity(text)
            emotion_trajectory[sender].append(intensity)

        # Build profiles for each speaker
        for speaker in set(msg.get('sender', 'Unknown') for msg in messages):
            events = all_events.get(speaker, [])
            trajectory = emotion_trajectory.get(speaker, [])
            emotional_profiles[speaker] = self._build_regulation_profile(
                speaker, events, trajectory
            )

        logger.info(f"Detected {sum(len(e) for e in all_events.values())} total emotional events")

        return {
            'emotional_events_by_speaker': dict(all_events),
            'regulation_profiles': emotional_profiles,
            'emotion_trajectories': dict(emotion_trajectory),
            'summary': self._generate_summary(emotional_profiles),
        }

    def _detect_emotional_events(
        self, text: str, message_index: int, sender: str
    ) -> List[EmotionalEvent]:
        """Detect all emotional events in a single message."""
        events: List[EmotionalEvent] = []

        # Check for volatility
        volatility_events = self._detect_volatility(text, message_index)
        events.extend(volatility_events)

        # Check for escalation
        escalation_events = self._detect_escalation(text, message_index)
        events.extend(escalation_events)

        # Check for de-escalation
        de_escalation_events = self._detect_de_escalation(text, message_index)
        events.extend(de_escalation_events)

        # Check for flooding
        flooding_events = self._detect_flooding(text, message_index)
        events.extend(flooding_events)

        # Check for recovery
        recovery_events = self._detect_recovery(text, message_index)
        events.extend(recovery_events)

        # Check for self-soothing
        soothing_events = self._detect_self_soothing(text, message_index)
        events.extend(soothing_events)

        return events

    def _detect_volatility(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect emotional volatility (sudden shifts)."""
        events: List[EmotionalEvent] = []

        for pattern in self.VOLATILITY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append(
                    EmotionalEvent(
                        event_type='volatility',
                        emotion_level=0.7,
                        emotional_tone='mixed',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                    )
                )

        return events

    def _detect_escalation(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect emotional escalation."""
        events: List[EmotionalEvent] = []

        for pattern in self.ESCALATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                intensity = self._calculate_pattern_intensity(match.group())
                events.append(
                    EmotionalEvent(
                        event_type='escalation',
                        emotion_level=intensity,
                        emotional_tone='negative',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                    )
                )

        return events

    def _detect_de_escalation(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect de-escalation attempts."""
        events: List[EmotionalEvent] = []

        for pattern in self.DE_ESCALATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append(
                    EmotionalEvent(
                        event_type='de_escalation',
                        emotion_level=0.4,
                        emotional_tone='positive',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        regulation_strategy='de_escalation',
                        effectiveness='potential',
                    )
                )

        return events

    def _detect_flooding(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect emotional flooding (overwhelm)."""
        events: List[EmotionalEvent] = []

        for pattern in self.FLOODING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append(
                    EmotionalEvent(
                        event_type='flooding',
                        emotion_level=0.9,
                        emotional_tone='negative',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                    )
                )

        return events

    def _detect_recovery(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect emotional recovery."""
        events: List[EmotionalEvent] = []

        for pattern in self.RECOVERY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append(
                    EmotionalEvent(
                        event_type='recovery',
                        emotion_level=0.3,
                        emotional_tone='positive',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        regulation_strategy='recovery',
                        effectiveness='effective',
                    )
                )

        return events

    def _detect_self_soothing(self, text: str, message_index: int) -> List[EmotionalEvent]:
        """Detect self-soothing strategies."""
        events: List[EmotionalEvent] = []

        for pattern in self.SELF_SOOTHING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append(
                    EmotionalEvent(
                        event_type='regulation_attempt',
                        emotion_level=0.5,
                        emotional_tone='neutral',
                        text_segment=self._extract_context(text, match.start()),
                        message_index=message_index,
                        regulation_strategy='self_soothing',
                        effectiveness='potential',
                    )
                )

        return events

    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate overall emotional intensity of a message."""
        intensity = 0.5  # baseline

        # Negative intensity markers
        negative_markers = [
            ('furious', 0.95), ('devastating', 0.9), ('horrific', 0.88),
            ('terrible', 0.8), ('awful', 0.75), ('angry', 0.7),
            ('frustrated', 0.6), ('upset', 0.55)
        ]

        # Positive intensity markers
        positive_markers = [
            ('wonderful', 0.85), ('amazing', 0.8), ('excited', 0.75),
            ('happy', 0.65), ('glad', 0.55), ('content', 0.45)
        ]

        for marker, value in negative_markers:
            if marker.lower() in text.lower():
                intensity = max(intensity, value)

        for marker, value in positive_markers:
            if marker.lower() in text.lower():
                intensity = max(intensity, value)

        return min(intensity, 1.0)

    def _calculate_pattern_intensity(self, pattern_text: str) -> float:
        """Calculate intensity from a pattern match."""
        intensity = 0.6

        intense_words = ['furious', 'enraged', 'devastated', 'breaking', 'destroying']
        for word in intense_words:
            if word.lower() in pattern_text.lower():
                intensity = max(intensity, 0.85)

        return min(intensity, 1.0)

    def _extract_context(self, text: str, position: int, context_length: int = 60) -> str:
        """Extract context around a match position."""
        start = max(0, position - context_length)
        end = min(len(text), position + context_length)
        return text[start:end].strip()

    def _build_regulation_profile(
        self,
        speaker_name: str,
        events: List[EmotionalEvent],
        emotion_trajectory: List[float],
    ) -> RegulationProfile:
        """Build emotional regulation profile for a speaker."""
        profile = RegulationProfile(speaker_name=speaker_name)

        if not events and not emotion_trajectory:
            return profile

        # Calculate volatility
        if len(emotion_trajectory) > 1:
            profile.overall_volatility = statistics.stdev(emotion_trajectory) if len(emotion_trajectory) > 1 else 0.0
            profile.emotional_baseline = statistics.mean(emotion_trajectory)
            profile.peak_emotion_level = max(emotion_trajectory) if emotion_trajectory else 0.0
            profile.trough_emotion_level = min(emotion_trajectory) if emotion_trajectory else 0.0

        # Count mood swings
        mood_swings = 0
        for i in range(1, len(emotion_trajectory)):
            if emotion_trajectory[i] > emotion_trajectory[i-1] + 0.2:
                mood_swings += 1
            elif emotion_trajectory[i] < emotion_trajectory[i-1] - 0.2:
                mood_swings += 1
        profile.mood_swing_frequency = mood_swings

        # Analyze events
        escalation_count = sum(1 for e in events if e.event_type == 'escalation')
        de_escalation_count = sum(1 for e in events if e.event_type == 'de_escalation')
        soothing_count = sum(1 for e in events if e.regulation_strategy == 'self_soothing')
        flooding_count = sum(1 for e in events if e.event_type == 'flooding')
        recovery_count = sum(1 for e in events if e.event_type == 'recovery')

        profile.flooding_indicators = flooding_count
        profile.escalation_risk = min(escalation_count / max(len(events), 1), 1.0)
        profile.self_soothing_competence = soothing_count / max(len(events), 1)
        profile.total_regulation_attempts = de_escalation_count + soothing_count + recovery_count
        profile.successful_regulation_attempts = recovery_count + (de_escalation_count // 2)

        if profile.total_regulation_attempts > 0:
            profile.regulation_success_rate = (
                profile.successful_regulation_attempts / profile.total_regulation_attempts
            )

        # Recovery capability
        if escalation_count > 0:
            profile.recovery_capability = recovery_count / escalation_count if escalation_count > 0 else 0.5

        # Identify coping strategies
        coping_strategies: Dict[str, int] = defaultdict(int)
        for event in events:
            if event.regulation_strategy:
                coping_strategies[event.regulation_strategy] += 1

        # Detect cognitive reappraisal and rumination from text
        # This would require access to full messages - simplified for this implementation

        profile.coping_strategies = dict(coping_strategies)

        # Emotional flexibility (ability to vary emotional responses)
        emotional_variance = profile.overall_volatility
        profile.emotional_flexibility = min(0.7, emotional_variance)  # Capped at 0.7

        return profile

    def _generate_summary(self, regulation_profiles: Dict[str, RegulationProfile]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not regulation_profiles:
            return {
                'total_speakers_analyzed': 0,
                'avg_volatility': 0.0,
                'high_volatility_speakers': 0,
            }

        volatilities = [p.overall_volatility for p in regulation_profiles.values()]
        escalation_risks = [p.escalation_risk for p in regulation_profiles.values()]
        success_rates = [
            p.regulation_success_rate for p in regulation_profiles.values()
            if p.total_regulation_attempts > 0
        ]

        return {
            'total_speakers_analyzed': len(regulation_profiles),
            'avg_volatility': statistics.mean(volatilities) if volatilities else 0.0,
            'avg_escalation_risk': statistics.mean(escalation_risks) if escalation_risks else 0.0,
            'avg_regulation_success_rate': statistics.mean(success_rates) if success_rates else 0.0,
            'high_volatility_speakers': sum(1 for v in volatilities if v > 0.3),
            'speakers_with_regulation_attempts': sum(
                1 for p in regulation_profiles.values()
                if p.total_regulation_attempts > 0
            ),
        }
