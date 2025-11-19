"""
Conflict Resolution and Gottman Method Analysis Module
Analyzes relationship conflict patterns including the Four Horsemen (criticism,
contempt, defensiveness, stonewalling), conflict initiation, resolution attempts,
and repair attempts based on John Gottman's relationship research.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ConflictMarker:
    """Container for conflict pattern match"""
    category: str  # criticism, contempt, defensiveness, stonewalling, repair, complaint
    pattern: str
    matched_text: str
    severity: float  # 0-1
    confidence: float  # 0-1
    position: int
    context: str = ""


@dataclass
class FourHorsemenIndicator:
    """Container for Four Horsemen of the Apocalypse indicator"""
    horseman: str  # criticism, contempt, defensiveness, stonewalling
    severity: float  # 0-1
    indicators: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, moderate, high, critical


@dataclass
class ConflictAnalysis:
    """Complete conflict analysis results"""
    conflict_markers: List[ConflictMarker] = field(default_factory=list)
    four_horsemen: List[FourHorsemenIndicator] = field(default_factory=list)
    conflict_initiation_detected: bool = False
    resolution_attempts: int = 0
    repair_attempts: int = 0
    stonewalling_score: float = 0.0
    defensiveness_score: float = 0.0
    contempt_score: float = 0.0
    criticism_score: float = 0.0
    overall_conflict_score: float = 0.0
    conflict_escalation_pattern: bool = False
    estimated_relationship_health: str = "unknown"  # healthy, at-risk, critical
    recommended_interventions: List[str] = field(default_factory=list)


class ConflictResolutionAnalyzer:
    """Analyzes relationship conflict patterns using Gottman method"""

    # Criticism patterns (expressed complaints vs attacks)
    CRITICISM_PATTERNS = [
        r"\byou always\b",
        r"\byou never\b",
        r"\byou can't\b",
        r"\byou don't\b",
        r"\byou're (so|such)\b",
        r"\byou're (incompetent|useless|hopeless|lazy|selfish)",
        r"\byou can't do anything right\b",
        r"\bwhy are you (so|always)\b",
        r"\byou should (know|understand|realize)\b",
        r"\byou're being\b.*\b(unreasonable|stupid|crazy|ridiculous)\b",
    ]

    # Contempt patterns (disgust, mockery, superiority)
    CONTEMPT_MARKERS = [
        r"\b(ugh|yuck|gag|disgusting|repulsive)\b",
        r"\byou're (pathetic|disgusting|beneath me|worthless)\b",
        r"eye.{0,5}rolling",
        r"\bthat's pathetic\b",
        r"\byou're such a\b.*\b(loser|jerk|idiot|joke)\b",
        r"\bi can't believe i\b.*\b(married|ended up with|chose)\b.*\byou\b",
        r"\byou disgust me\b",
        r"\b(mocking|sarcastic|contemptuous)\b.*tone",
        r"\bgoody two shoes\b",
        r"\byou act like\b.*\b(baby|child|idiot)\b",
    ]

    # Defensiveness patterns (counterattack, victim stance)
    DEFENSIVENESS_PATTERNS = [
        r"\bi didn't (do|say|mean)\b",
        r"\bthat's not (true|what happened|fair)\b",
        r"\byou're the (one|problem|issue|real culprit)\b",
        r"\byou're always blaming me\b",
        r"\bi was just\b",
        r"\bif you (had|weren't|didn't)\b",
        r"\byou started it\b",
        r"\bnot my fault\b",
        r"\bi can't help it\b",
        r"\byou're being (unfair|unreasonable|hypocritical)\b",
        r"\byou do the same thing\b",
    ]

    # Stonewalling patterns (withdrawal, silence, avoidance)
    STONEWALLING_PATTERNS = [
        r"\b(fine|whatever|okay|sure)\b.*\bwhatever\b",
        r"\bi don't want to talk about this\b",
        r"\bi'm done\b",
        r"\bi'm not discussing this\b",
        r"\b(silence|quiet|shut down)\b",
        r"\bi'm walking away\b",
        r"\bleave me alone\b",
        r"\bnot talking about it\b",
        r"\byou never listen anyway\b",
        r"\bwhat's the point\b",
    ]

    # Repair attempts (de-escalation, humor, softening)
    REPAIR_PATTERNS = [
        r"\bi'm sorry\b",
        r"\bcan we (talk|discuss|start over)\b",
        r"\bi didn't mean to (hurt|upset|offend)\b",
        r"\blet me try again\b",
        r"\bi appreciate\b",
        r"\bthank you for\b",
        r"\bi understand\b",
        r"\bthat was (my fault|wrong of me)\b",
        r"\blet's (take a break|calm down|start fresh)\b",
        r"\bi love you\b",
        r"\bthat was a dumb thing to say\b",
        r"\byou're right\b",
        r"\bi was wrong\b",
    ]

    # Complaint patterns (legitimate grievance vs attack)
    COMPLAINT_PATTERNS = [
        r"\bi feel\b.*\b(hurt|upset|disappointed|frustrated|ignored)\b",
        r"\bwhen you\b.*\bi feel\b",
        r"\bi would appreciate\b",
        r"\bcan you please\b",
        r"\bi need\b.*\bfrom you\b",
        r"\bits important to me\b",
        r"\bthis matters to me\b",
        r"\bi'm concerned about\b",
    ]

    # Resolution indicators (problem-solving, compromise)
    RESOLUTION_PATTERNS = [
        r"\blet's find a solution\b",
        r"\bwhat can we do\b",
        r"\bhow can we (fix|resolve|improve)\b",
        r"\blet's compromise\b",
        r"\bwhat would work for you\b",
        r"\bi'm willing to\b",
        r"\bwe can (try|figure out)\b",
        r"\blet's work together\b",
        r"\bhow do you (feel|think)\b.*\b(about|regarding)\b",
        r"\bi understand your (perspective|point|concern)\b",
    ]

    def __init__(self):
        """Initialize conflict resolution analyzer"""
        self.compiled_patterns = self._compile_patterns()
        self.conflict_history: Dict[str, List[float]] = defaultdict(list)

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for efficiency"""
        return {
            'criticism': [re.compile(p, re.IGNORECASE) for p in self.CRITICISM_PATTERNS],
            'contempt': [re.compile(p, re.IGNORECASE) for p in self.CONTEMPT_MARKERS],
            'defensiveness': [re.compile(p, re.IGNORECASE) for p in self.DEFENSIVENESS_PATTERNS],
            'stonewalling': [re.compile(p, re.IGNORECASE) for p in self.STONEWALLING_PATTERNS],
            'repair': [re.compile(p, re.IGNORECASE) for p in self.REPAIR_PATTERNS],
            'complaint': [re.compile(p, re.IGNORECASE) for p in self.COMPLAINT_PATTERNS],
            'resolution': [re.compile(p, re.IGNORECASE) for p in self.RESOLUTION_PATTERNS],
        }

    def analyze_conflict(self, text: str, speaker_name: str = "unknown") -> ConflictAnalysis:
        """Analyze text for conflict patterns

        Args:
            text: Text to analyze
            speaker_name: Name of the speaker

        Returns:
            ConflictAnalysis: Analysis results
        """
        analysis = ConflictAnalysis()

        # Detect criticism
        criticism_score = self._detect_criticism(text, analysis)
        analysis.criticism_score = criticism_score

        # Detect contempt
        contempt_score = self._detect_contempt(text, analysis)
        analysis.contempt_score = contempt_score

        # Detect defensiveness
        defensiveness_score = self._detect_defensiveness(text, analysis)
        analysis.defensiveness_score = defensiveness_score

        # Detect stonewalling
        stonewalling_score = self._detect_stonewalling(text, analysis)
        analysis.stonewalling_score = stonewalling_score

        # Detect repair attempts
        repair_count = self._detect_repair_attempts(text, analysis)
        analysis.repair_attempts = repair_count

        # Detect resolution attempts
        resolution_count = self._detect_resolution_attempts(text, analysis)
        analysis.resolution_attempts = resolution_count

        # Detect conflict initiation
        analysis.conflict_initiation_detected = self._is_conflict_initiation(text)

        # Calculate overall conflict score
        self._calculate_overall_scores(analysis, speaker_name)

        # Assess relationship health
        analysis.estimated_relationship_health = self._estimate_relationship_health(analysis)

        # Generate recommendations
        analysis.recommended_interventions = self._generate_interventions(analysis)

        return analysis

    def _detect_criticism(self, text: str, analysis: ConflictAnalysis) -> float:
        """Detect criticism patterns (attacks on character)"""
        score = 0.0
        matches = []

        for pattern in self.compiled_patterns['criticism']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.7  # Criticism is relatively serious
                confidence = 0.75

                marker = ConflictMarker(
                    category='criticism',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                matches.append(matched_text)
                score = max(score, severity * confidence)

        # Also check for complaint (legitimate grievance)
        complaints = sum(1 for p in self.compiled_patterns['complaint'] for _ in p.finditer(text))

        # If there's a complaint mixed in, lower the score (complaint is healthier than criticism)
        if complaints > 0:
            score *= 0.7

        return score

    def _detect_contempt(self, text: str, analysis: ConflictAnalysis) -> float:
        """Detect contempt markers (disgust, superiority, mockery)"""
        score = 0.0

        for pattern in self.compiled_patterns['contempt']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.9  # Contempt is the most predictive of divorce
                confidence = 0.8

                marker = ConflictMarker(
                    category='contempt',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_defensiveness(self, text: str, analysis: ConflictAnalysis) -> float:
        """Detect defensiveness patterns (counterattack, victim stance)"""
        score = 0.0

        for pattern in self.compiled_patterns['defensiveness']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.65  # Defensiveness is less harmful than contempt
                confidence = 0.75

                marker = ConflictMarker(
                    category='defensiveness',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_stonewalling(self, text: str, analysis: ConflictAnalysis) -> float:
        """Detect stonewalling patterns (withdrawal, avoidance, silence)"""
        score = 0.0

        for pattern in self.compiled_patterns['stonewalling']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]
                severity = 0.8  # Stonewalling is very damaging
                confidence = 0.8

                marker = ConflictMarker(
                    category='stonewalling',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=severity,
                    confidence=confidence,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                score = max(score, severity * confidence)

        return score

    def _detect_repair_attempts(self, text: str, analysis: ConflictAnalysis) -> int:
        """Detect repair attempt patterns (de-escalation, apologies)"""
        count = 0

        for pattern in self.compiled_patterns['repair']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                marker = ConflictMarker(
                    category='repair',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=0.0,  # Repairs are positive
                    confidence=0.8,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                count += 1

        return count

    def _detect_resolution_attempts(self, text: str, analysis: ConflictAnalysis) -> int:
        """Detect conflict resolution and problem-solving patterns"""
        count = 0

        for pattern in self.compiled_patterns['resolution']:
            for match in pattern.finditer(text):
                matched_text = text[match.start():match.end()]

                marker = ConflictMarker(
                    category='resolution',
                    pattern=pattern.pattern,
                    matched_text=matched_text,
                    severity=0.0,  # Resolution is positive
                    confidence=0.8,
                    position=match.start(),
                    context=self._get_context(text, match.start())
                )
                analysis.conflict_markers.append(marker)
                count += 1

        return count

    def _is_conflict_initiation(self, text: str) -> bool:
        """Determine if message initiates a conflict"""
        # Look for accusatory opening phrases
        conflict_starters = [
            r"^(you|why|how could)",
            r"\byou (never|always)\b",
            r"\bwhat's wrong with you\b",
            r"\bi can't believe\b",
            r"\byou're (the problem|impossible|impossible to live with)\b",
        ]

        for pattern in conflict_starters:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def _calculate_overall_scores(self, analysis: ConflictAnalysis, speaker_name: str) -> None:
        """Calculate overall conflict scores and Four Horsemen indicators"""
        # Calculate weighted conflict score
        four_horsemen_scores = {
            'criticism': analysis.criticism_score,
            'contempt': analysis.contempt_score,
            'defensiveness': analysis.defensiveness_score,
            'stonewalling': analysis.stonewalling_score,
        }

        # Create Four Horsemen indicators
        for horseman, score in four_horsemen_scores.items():
            if score > 0.0:
                indicator = FourHorsemenIndicator(
                    horseman=horseman,
                    severity=score,
                    indicators=[m.matched_text for m in analysis.conflict_markers if m.category == horseman],
                    evidence=[m.context for m in analysis.conflict_markers if m.category == horseman][:3],
                    risk_level=self._severity_to_risk(score)
                )
                analysis.four_horsemen.append(indicator)

        # Overall conflict score (weighted by Gottman research)
        weights = {
            'criticism': 0.2,
            'contempt': 0.4,  # Most predictive of divorce
            'defensiveness': 0.2,
            'stonewalling': 0.4,  # Also very damaging
        }

        weighted_sum = sum(
            four_horsemen_scores[horseman] * weights[horseman]
            for horseman in four_horsemen_scores
        )

        # Boost from repair attempts (negative adjustment)
        repair_boost = analysis.repair_attempts * 0.1
        resolution_boost = analysis.resolution_attempts * 0.15

        analysis.overall_conflict_score = max(0.0, weighted_sum - repair_boost - resolution_boost)

        # Track conflict history
        self.conflict_history[speaker_name].append(analysis.overall_conflict_score)

        # Detect escalation
        if len(self.conflict_history[speaker_name]) >= 2:
            recent_scores = self.conflict_history[speaker_name][-3:]
            if len(recent_scores) >= 2:
                analysis.conflict_escalation_pattern = recent_scores[-1] > recent_scores[0]

    def _estimate_relationship_health(self, analysis: ConflictAnalysis) -> str:
        """Estimate overall relationship health based on conflict analysis"""
        score = analysis.overall_conflict_score

        if score >= 0.7:
            return "critical"
        elif score >= 0.5:
            return "at-risk"
        elif score >= 0.3:
            return "strained"
        elif score > 0.0:
            return "healthy with conflicts"
        else:
            return "healthy"

    def _generate_interventions(self, analysis: ConflictAnalysis) -> List[str]:
        """Generate recommended interventions based on conflict analysis"""
        recommendations = []

        # High contempt = urgent intervention needed
        if analysis.contempt_score > 0.6:
            recommendations.append("URGENT: Contempt detected. Seek professional couples therapy immediately.")

        # Stonewalling = communication breakdown
        if analysis.stonewalling_score > 0.6:
            recommendations.append("Stonewalling detected. Establish ground rules for communication breaks.")
            recommendations.append("Practice taking structured breaks and returning to discussion.")

        # High criticism = address patterns
        if analysis.criticism_score > 0.6:
            recommendations.append("High criticism detected. Work on expressing complaints without attacks.")
            recommendations.append("Use 'I feel' statements instead of 'You always/never' statements.")

        # Defensiveness = perspective-taking
        if analysis.defensiveness_score > 0.6:
            recommendations.append("Defensiveness detected. Practice active listening and validation.")

        # Few repair attempts = build them
        if analysis.repair_attempts == 0 and analysis.overall_conflict_score > 0.3:
            recommendations.append("No repair attempts detected. Learn and practice repair techniques.")

        # Escalation pattern = urgent
        if analysis.conflict_escalation_pattern:
            recommendations.append("CRITICAL: Conflict escalation pattern detected. Seek professional help.")

        return recommendations

    def _severity_to_risk(self, severity: float) -> str:
        """Convert severity score to risk level"""
        if severity >= 0.8:
            return "critical"
        elif severity >= 0.6:
            return "high"
        elif severity >= 0.4:
            return "moderate"
        else:
            return "low"

    def _get_context(self, text: str, position: int, window: int = 30) -> str:
        """Get text context around a position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()

    def track_conflict_over_time(
        self,
        speaker_name: str,
        messages: List[str]
    ) -> Dict[str, Any]:
        """Track conflict patterns over multiple messages

        Args:
            speaker_name: Name of the speaker
            messages: List of messages from this speaker

        Returns:
            Dict with conflict tracking over time
        """
        scores = []

        for message in messages:
            analysis = self.analyze_conflict(message, speaker_name)
            scores.append(analysis.overall_conflict_score)

        if not scores:
            return {
                'trend': 'unknown',
                'average_score': 0.0,
                'peak_score': 0.0,
                'volatility': 0.0
            }

        avg_score = statistics.mean(scores)
        peak_score = max(scores)

        volatility = 0.0
        if len(scores) > 1:
            volatility = statistics.stdev(scores)

        # Determine trend
        if len(scores) >= 2:
            trend = "escalating" if scores[-1] > scores[0] else "de-escalating"
        else:
            trend = "stable"

        return {
            'trend': trend,
            'average_score': avg_score,
            'peak_score': peak_score,
            'volatility': volatility,
            'score_history': scores,
            'message_count': len(messages)
        }

    def identify_conflict_pattern(self, analysis: ConflictAnalysis) -> Optional[str]:
        """Identify the primary conflict pattern

        Args:
            analysis: ConflictAnalysis object

        Returns:
            str: Identified pattern or None
        """
        if not analysis.four_horsemen:
            return None

        # Sort by severity
        sorted_horsemen = sorted(
            analysis.four_horsemen,
            key=lambda h: h.severity,
            reverse=True
        )

        return sorted_horsemen[0].horseman if sorted_horsemen else None
