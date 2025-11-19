"""
Love Bombing Detection Module
Identifies love bombing and idealization patterns including excessive flattery,
future-faking, intensity escalation, premature intimacy, and possessive language.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LoveBombMarker:
    """Container for love bombing pattern match"""
    category: str  # flattery, future_faking, intensity, premature_intimacy, possessiveness
    pattern: str
    matched_text: str
    severity: float  # 0-1
    confidence: float  # 0-1
    position: int
    context: str = ""


@dataclass
class LoveBombingAnalysis:
    """Complete love bombing analysis results"""
    markers: List[LoveBombMarker] = field(default_factory=list)
    excessive_flattery_score: float = 0.0
    future_faking_score: float = 0.0
    intensity_escalation_score: float = 0.0
    over_the_top_declarations: int = 0
    premature_intimacy_score: float = 0.0
    gift_gesture_mentions: int = 0
    possessive_language_score: float = 0.0
    timeline_pressure_score: float = 0.0
    overall_love_bomb_score: float = 0.0
    love_bomb_risk_level: str = "none"  # none, low, moderate, high, critical
    is_love_bombing: bool = False
    primary_tactics: List[str] = field(default_factory=list)
    escalation_pattern: bool = False
    red_flags: List[str] = field(default_factory=list)


class LoveBombingDetector:
    """Detects love bombing and idealization patterns"""

    # Excessive flattery patterns
    FLATTERY_PATTERNS = [
        r"\byou're (so|really|absolutely|incredibly)\b.*\b(beautiful|handsome|stunning|gorgeous|amazing|perfect|wonderful|special)\b",
        r"\byou're the most\b.*\b(beautiful|perfect|amazing|incredible)\b.*\bi've ever (met|seen)\b",
        r"\bi've never (met|seen|experienced)\b.*\banyone like you\b",
        r"\byou (complete|fulfill|make)\b.*\bme\b",
        r"\byou're everything i\b",
        r"\bi can't live without you\b",
        r"\byou're my (soulmate|twin flame|other half|destiny)\b",
        r"\byou're (flawless|perfect|without flaw)\b",
        r"\bi've never felt this way\b",
        r"\byou're unlike anyone\b",
        r"\byou're (too good|way too good) to be true\b",
    ]

    # Future-faking patterns (premature future promises)
    FUTURE_FAKING_PATTERNS = [
        r"\bwe're going to\b.*\b(marry|get married|have kids|build a life)\b",
        r"\blet's (move in|travel the world|buy a house|start a family)\b",
        r"\bi see us\b.*\b(forever|getting old together|raising kids)\b",
        r"\bwhen we get married\b",
        r"\bour (future|wedding|kids|house)\b",
        r"\bi can't wait to\b.*\b(spend my life|grow old)\b.*\bwith you\b",
        r"\byou'll (never have to|never need to)\b",
        r"\bwe're meant to be\b",
        r"\byou're the one i've been waiting for\b",
        r"\bwe'll be together forever\b",
        r"\blet's run away together\b",
    ]

    # Intensity escalation patterns
    INTENSITY_PATTERNS = [
        r"\bi'm (obsessed|crazy|head over heels)\b.*\bfor you\b",
        r"\bi (need|can't stop thinking about) you\b",
        r"\byou're all i (think about|care about|need)\b",
        r"\bi (love|adore) you so much\b",
        r"\bi'd do anything for you\b",
        r"\byou mean everything to me\b",
        r"\bi've never felt this (intense|strong)\b",
        r"\bi (can't eat|can't sleep)\b.*\b(without|thinking about)\b.*\byou\b",
        r"\byou're my (everything|world|life)\b",
        r"\bi'm (devoted|committed|dedicated)\b.*\bto you\b",
    ]

    # Over-the-top declarations
    OVER_TOP_PATTERNS = [
        r"\bi would (die|kill|sacrifice my life)\b.*\bfor you\b",
        r"\byou're the (one|answer|solution)\b",
        r"\bno one (has ever|will ever)\b.*\bmake me feel\b",
        r"\byou're my (reason to live|life purpose|meaning)\b",
        r"\bi worship you\b",
        r"\byou're a (goddess|god|angel|divine)\b",
        r"\bwithout you i'm (nothing|lost|broken)\b",
        r"\byou saved me\b",
        r"\bi'll never (leave|hurt|disappoint) you\b",
        r"\byou're (perfect for me|made for me|my destiny)\b",
    ]

    # Premature intimacy patterns
    PREMATURE_INTIMACY_PATTERNS = [
        r"\bi (love|trust) you (already|so soon|after such a short time)\b",
        r"\blet's (move in|sleep together|meet the family)\b.*\b(tonight|tomorrow|soon|immediately)\b",
        r"\bwe should (move in|get married|have a baby)\b.*\b(quickly|soon|immediately)\b",
        r"\bi've never told anyone this\b",
        r"\bi'm telling you things i\b.*\b(never|haven't)\b.*\btold anyone\b",
        r"\byou're the only one i\b",
        r"\bcan i (move in|share your life|be your everything)\b",
        r"\bwhat if we (spent all our time|only saw each other|were always together)\b",
        r"\bi'm ready to (commit|get married|have kids)\b",
    ]

    # Possessive language patterns
    POSSESSIVE_PATTERNS = [
        r"\byou're (mine|my property|belonging to me)\b",
        r"\bi don't want you (seeing|talking to) anyone else\b",
        r"\byou can't (see|talk to|spend time with)\b.*\b(anyone else|other people|friends|family)\b",
        r"\byou belong with me\b",
        r"\bno one else deserves you\b",
        r"\bi'll take care of everything\b",
        r"\byou don't need anyone but me\b",
        r"\byou're perfect for me and me alone\b",
        r"\bi'm the (only one|best thing)\b.*\b(for you|in your life)\b",
        r"\byou're (all mine|belong to me)\b",
    ]

    # Gift and gesture patterns
    GIFT_PATTERNS = [
        r"\bi (bought|got|got you|prepared)\b.*\b(gift|present|flowers|jewelry|ring|car|trip)\b",
        r"\bsurprise (gift|vacation|trip|dinner)\b",
        r"\bi spent (thousands|lots of money|my savings)\b.*\bon you\b",
        r"\bi (arranged|planned|booked)\b.*\b(trip|dinner|experience)\b",
        r"\bwant to (spoil|take care of) you\b",
        r"\bi'll (buy|get|give) you (anything|everything)\b",
        r"\bi (paid for|treating you to)\b",
    ]

    # Timeline pressure patterns
    TIMELINE_PATTERNS = [
        r"\bwe should (decide|commit|move in)\b.*\b(today|tomorrow|this week|immediately)\b",
        r"\bi need to (know|hear)\b.*\b(now|today|right away)\b",
        r"\btime is (running out|short)\b",
        r"\bwe can't (wait|take our time)\b",
        r"\bif you (love me|care about me)\b.*\b(prove it|show me)\b.*\b(now|immediately|today)\b",
        r"\bwe need to (lock this down|make this official)\b.*\b(now|urgently|immediately)\b",
        r"\byou need to (decide|choose|commit)\b.*\b(now|today|urgently)\b",
    ]

    def __init__(self):
        """Initialize love bombing detector"""
        self.compiled_patterns = self._compile_patterns()
        self.love_bomb_history: Dict[str, List[float]] = defaultdict(list)

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for efficiency"""
        return {
            'flattery': [re.compile(p, re.IGNORECASE) for p in self.FLATTERY_PATTERNS],
            'future_faking': [re.compile(p, re.IGNORECASE) for p in self.FUTURE_FAKING_PATTERNS],
            'intensity': [re.compile(p, re.IGNORECASE) for p in self.INTENSITY_PATTERNS],
            'over_top': [re.compile(p, re.IGNORECASE) for p in self.OVER_TOP_PATTERNS],
            'premature_intimacy': [re.compile(p, re.IGNORECASE) for p in self.PREMATURE_INTIMACY_PATTERNS],
            'possessive': [re.compile(p, re.IGNORECASE) for p in self.POSSESSIVE_PATTERNS],
            'gifts': [re.compile(p, re.IGNORECASE) for p in self.GIFT_PATTERNS],
            'timeline': [re.compile(p, re.IGNORECASE) for p in self.TIMELINE_PATTERNS],
        }

    def analyze_love_bombing(
        self,
        text: str,
        relationship_duration_days: int = 0,
        speaker_name: str = "unknown"
    ) -> LoveBombingAnalysis:
        """Analyze text for love bombing patterns

        Args:
            text: Text to analyze
            relationship_duration_days: Duration of relationship in days
            speaker_name: Name of the speaker

        Returns:
            LoveBombingAnalysis: Analysis results
        """
        analysis = LoveBombingAnalysis()

        # Detect flattery
        flattery_score = self._detect_flattery(text, analysis)
        analysis.excessive_flattery_score = flattery_score

        # Detect future-faking
        future_score = self._detect_future_faking(text, analysis, relationship_duration_days)
        analysis.future_faking_score = future_score

        # Detect intensity escalation
        intensity_score = self._detect_intensity(text, analysis)
        analysis.intensity_escalation_score = intensity_score

        # Detect over-the-top declarations
        over_top_count = self._detect_over_the_top(text, analysis)
        analysis.over_the_top_declarations = over_top_count

        # Detect premature intimacy
        intimacy_score = self._detect_premature_intimacy(text, analysis, relationship_duration_days)
        analysis.premature_intimacy_score = intimacy_score

        # Detect gift/gesture mentions
        gift_count = self._detect_gifts(text, analysis)
        analysis.gift_gesture_mentions = gift_count

        # Detect possessive language
        possessive_score = self._detect_possessiveness(text, analysis)
        analysis.possessive_language_score = possessive_score

        # Detect timeline pressure
        timeline_score = self._detect_timeline_pressure(text, analysis)
        analysis.timeline_pressure_score = timeline_score

        # Calculate overall love bombing score
        self._calculate_overall_score(analysis, relationship_duration_days)

        # Determine if love bombing
        analysis.is_love_bombing = analysis.overall_love_bomb_score > 0.6

        # Generate red flags
        analysis.red_flags = self._generate_red_flags(analysis, relationship_duration_days)

        # Track history
        self.love_bomb_history[speaker_name].append(analysis.overall_love_bomb_score)

        # Detect escalation pattern
        if len(self.love_bomb_history[speaker_name]) >= 3:
            recent_scores = self.love_bomb_history[speaker_name][-3:]
            analysis.escalation_pattern = recent_scores[-1] > recent_scores[0]

        return analysis

    def _detect_flattery(self, text: str, analysis: LoveBombingAnalysis) -> float:
        """Detect excessive flattery patterns"""
        score = 0.0

        for pattern in self.compiled_patterns['flattery']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.6
                confidence = 0.8

                marker = LoveBombMarker(
                    category='flattery',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_future_faking(
        self,
        text: str,
        analysis: LoveBombingAnalysis,
        relationship_duration_days: int
    ) -> float:
        """Detect future-faking patterns, especially early in relationship"""
        score = 0.0

        for pattern in self.compiled_patterns['future_faking']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                # Severity increases if early in relationship
                base_severity = 0.65
                if relationship_duration_days < 90:  # Less than 3 months
                    severity = min(1.0, base_severity + 0.2)
                else:
                    severity = base_severity

                confidence = 0.8

                marker = LoveBombMarker(
                    category='future_faking',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_intensity(self, text: str, analysis: LoveBombingAnalysis) -> float:
        """Detect intensity escalation patterns"""
        score = 0.0

        for pattern in self.compiled_patterns['intensity']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.7
                confidence = 0.75

                marker = LoveBombMarker(
                    category='intensity',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_over_the_top(self, text: str, analysis: LoveBombingAnalysis) -> int:
        """Detect over-the-top declarations"""
        count = 0

        for pattern in self.compiled_patterns['over_top']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                marker = LoveBombMarker(
                    category='over_the_top',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=0.75,
                    confidence=0.85,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                count += 1

        return count

    def _detect_premature_intimacy(
        self,
        text: str,
        analysis: LoveBombingAnalysis,
        relationship_duration_days: int
    ) -> float:
        """Detect premature intimacy markers"""
        score = 0.0

        for pattern in self.compiled_patterns['premature_intimacy']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                # Severity depends on relationship duration
                base_severity = 0.7
                if relationship_duration_days < 30:  # Less than a month
                    severity = min(1.0, base_severity + 0.25)
                elif relationship_duration_days < 90:
                    severity = base_severity
                else:
                    severity = max(0.2, base_severity - 0.2)

                confidence = 0.8

                marker = LoveBombMarker(
                    category='premature_intimacy',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_gifts(self, text: str, analysis: LoveBombingAnalysis) -> int:
        """Detect gift and gesture mentions"""
        count = 0

        for pattern in self.compiled_patterns['gifts']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                marker = LoveBombMarker(
                    category='gifts_gestures',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=0.4,  # Gifts alone aren't inherently bad
                    confidence=0.85,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                count += 1

        return count

    def _detect_possessiveness(self, text: str, analysis: LoveBombingAnalysis) -> float:
        """Detect possessive and controlling language"""
        score = 0.0

        for pattern in self.compiled_patterns['possessive']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.85  # Possessiveness is a red flag
                confidence = 0.85

                marker = LoveBombMarker(
                    category='possessiveness',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_timeline_pressure(self, text: str, analysis: LoveBombingAnalysis) -> float:
        """Detect pressure for rapid commitment or decisions"""
        score = 0.0

        for pattern in self.compiled_patterns['timeline']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.75  # Pressure is concerning
                confidence = 0.8

                marker = LoveBombMarker(
                    category='timeline_pressure',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _calculate_overall_score(
        self,
        analysis: LoveBombingAnalysis,
        relationship_duration_days: int
    ) -> None:
        """Calculate overall love bombing score"""
        # Weight different components
        weights = {
            'flattery': 0.15,
            'future_faking': 0.20,
            'intensity': 0.15,
            'over_top': 0.15,
            'premature_intimacy': 0.20,
            'possessive': 0.25,  # Possessiveness is most concerning
            'timeline': 0.15,
        }

        total_weighted = 0.0
        total_weight = 0.0

        total_weighted += analysis.excessive_flattery_score * weights['flattery']
        total_weighted += analysis.future_faking_score * weights['future_faking']
        total_weighted += analysis.intensity_escalation_score * weights['intensity']
        total_weighted += min(1.0, analysis.over_the_top_declarations / 3.0) * weights['over_top']
        total_weighted += analysis.premature_intimacy_score * weights['premature_intimacy']
        total_weighted += analysis.possessive_language_score * weights['possessive']
        total_weighted += analysis.timeline_pressure_score * weights['timeline']

        total_weight = sum(weights.values())

        analysis.overall_love_bomb_score = total_weighted / total_weight if total_weight > 0 else 0.0

        # Early relationship boost (more concerning if early)
        if relationship_duration_days < 30:
            analysis.overall_love_bomb_score = min(1.0, analysis.overall_love_bomb_score * 1.3)

        # Determine risk level
        if analysis.overall_love_bomb_score >= 0.8:
            analysis.love_bomb_risk_level = "critical"
        elif analysis.overall_love_bomb_score >= 0.6:
            analysis.love_bomb_risk_level = "high"
        elif analysis.overall_love_bomb_score >= 0.4:
            analysis.love_bomb_risk_level = "moderate"
        elif analysis.overall_love_bomb_score >= 0.2:
            analysis.love_bomb_risk_level = "low"
        else:
            analysis.love_bomb_risk_level = "none"

    def _generate_red_flags(
        self,
        analysis: LoveBombingAnalysis,
        relationship_duration_days: int
    ) -> List[str]:
        """Generate specific red flags based on analysis"""
        flags = []

        if analysis.excessive_flattery_score > 0.6:
            flags.append("Excessive, unrealistic flattery detected")

        if analysis.future_faking_score > 0.6 and relationship_duration_days < 90:
            flags.append("Premature talk of marriage, moving in, or long-term commitment")

        if analysis.intensity_escalation_score > 0.6 and relationship_duration_days < 60:
            flags.append("Intense declarations of love unusually early in relationship")

        if analysis.over_the_top_declarations > 2:
            flags.append(f"Multiple over-the-top declarations ({analysis.over_the_top_declarations})")

        if analysis.premature_intimacy_score > 0.6 and relationship_duration_days < 30:
            flags.append("Pushing for rapid physical/emotional intimacy")

        if analysis.possessive_language_score > 0.6:
            flags.append("Possessive or controlling language detected")

        if analysis.gift_gesture_mentions > 3:
            flags.append(f"Excessive gift-giving or grand gestures ({analysis.gift_gesture_mentions})")

        if analysis.timeline_pressure_score > 0.5:
            flags.append("Pressure for quick decisions or commitment")

        if analysis.escalation_pattern and len(self.love_bomb_history.get("unknown", [])) > 2:
            flags.append("Love bombing intensity escalating over time")

        return flags

    def _get_context(self, text: str, position: int, window: int = 30) -> str:
        """Get text context around a position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()

    def analyze_relationship_phase(
        self,
        messages: List[str],
        relationship_duration_days: int
    ) -> Dict[str, Any]:
        """Analyze if relationship is in love bombing phase

        Args:
            messages: List of messages to analyze
            relationship_duration_days: Relationship duration

        Returns:
            Dict with phase analysis
        """
        combined_text = "\n".join(messages)
        analysis = self.analyze_love_bombing(
            combined_text,
            relationship_duration_days
        )

        return {
            'in_love_bombing_phase': analysis.is_love_bombing,
            'overall_score': analysis.overall_love_bomb_score,
            'risk_level': analysis.love_bomb_risk_level,
            'red_flags': analysis.red_flags,
            'duration_days': relationship_duration_days,
            'expected_phase': self._get_expected_relationship_phase(relationship_duration_days),
            'phase_anomaly': analysis.overall_love_bomb_score > 0.5 and relationship_duration_days < 90,
        }

    def _get_expected_relationship_phase(self, days: int) -> str:
        """Get expected relationship phase based on duration"""
        if days < 30:
            return "early_dating"
        elif days < 90:
            return "honeymoon"
        elif days < 180:
            return "settling_in"
        elif days < 365:
            return "building"
        else:
            return "established"
