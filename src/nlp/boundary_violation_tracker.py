"""
Boundary Violation Tracker Module
Tracks request escalation, consent violations, pressure tactics, guilt-tripping,
and assessment of respect for boundaries.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

# Import model cache for pattern caching
from .model_cache import get_cache, load_patterns_file, compile_regex_patterns

logger = logging.getLogger(__name__)


@dataclass
class BoundaryViolation:
    """Container for a boundary violation pattern match"""
    category: str
    subcategory: str
    violation_type: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""
    escalation_level: int = 0  # 0-5 scale of escalation


@dataclass
class BoundaryAnalysis:
    """Complete boundary violation analysis results"""
    violations_found: List[BoundaryViolation] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    violation_count: int = 0
    unique_violations: int = 0
    violation_types: List[str] = field(default_factory=list)
    request_escalation_detected: bool = False
    consent_violations: bool = False
    pressure_tactics_used: bool = False
    guilt_tripping_detected: bool = False
    respect_for_no: float = 1.0  # 1.0 = full respect, 0.0 = no respect
    boundary_pushback_intensity: float = 0.0  # How aggressively boundaries are challenged
    refusal_response_pattern: str = "unknown"  # compliant, resistant, unclear


class BoundaryViolationTracker:
    """Tracks boundary violations and escalating requests"""

    # Violation categories
    VIOLATION_CATEGORIES = [
        "request_escalation",
        "consent_violation",
        "pressure_tactic",
        "guilt_tripping",
        "refusal_dismissal",
        "boundary_push",
        "emotional_manipulation"
    ]

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.15,
        "moderate": 0.35,
        "high": 0.60,
        "critical": 0.80
    }

    # Category weights
    CATEGORY_WEIGHTS = {
        "request_escalation": 1.3,
        "consent_violation": 1.6,
        "pressure_tactic": 1.2,
        "guilt_tripping": 1.1,
        "refusal_dismissal": 1.4,
        "boundary_push": 1.3,
        "emotional_manipulation": 1.0
    }

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize boundary violation tracker

        Args:
            patterns_file: Path to patterns JSON file
        """
        cache = get_cache()
        cache_key = f"boundary_patterns_{patterns_file or 'default'}"

        # Load patterns from cache (or file if not cached)
        self.patterns = cache.get_or_load(
            cache_key,
            load_patterns_file,
            patterns_file,
            'boundary'
        )

        # If patterns are empty, use defaults
        if not self.patterns:
            self.patterns = self._get_default_patterns()

        # Compile and cache patterns
        compiled_cache_key = f"boundary_compiled_{patterns_file or 'default'}"
        self.compiled_patterns = cache.get_or_load(
            compiled_cache_key,
            compile_regex_patterns,
            self.patterns
        )

    def _get_default_patterns(self) -> Dict:
        """Get default boundary violation patterns

        Returns:
            Dict: Default patterns
        """
        return {
            "request_escalation": [
                {"regex": r"\b(but|just|please|one (more|last|time)|come on)\b.*\b(can (you|we)|will (you|we)|let (me|us))\b", "severity": 0.7, "violation": "escalated_request", "description": "Persistent request repetition"},
                {"regex": r"\b(first|then|after that|once you|if you do)\b.*\b(promise|swear|agree)\b", "severity": 0.75, "violation": "incremental_escalation", "description": "Incremental request progression"},
                {"regex": r"\b(just (a|one|this|some)|not (much|that|a lot)|quick)\b", "severity": 0.65, "violation": "minimization_request", "description": "Request minimization"},
                {"regex": r"\b(only|just|only you|only for me|special)\b", "severity": 0.7, "violation": "exclusivity_request", "description": "Exclusivity-based requests"},
            ],
            "consent_violation": [
                {"regex": r"\b(without (your|my) (knowledge|permission)|behind (your|my) back|secretly|don't tell)\b", "severity": 0.85, "violation": "covert_action", "description": "Hidden or secret actions"},
                {"regex": r"\b(don't need permission|already decided|no choice|has to happen)\b", "severity": 0.88, "violation": "unilateral_decision", "description": "Unilateral decision-making"},
                {"regex": r"\b(touch|kiss|sex|intimate).{0,50}\b(without|no|don't want)\b", "severity": 0.92, "violation": "physical_violation", "description": "Physical contact violation"},
                {"regex": r"\b(share|post|tell|send).{0,50}\b(without (asking|permission)|don't (want|ask)|shouldn't)\b", "severity": 0.88, "violation": "information_violation", "description": "Information sharing violation"},
            ],
            "pressure_tactic": [
                {"regex": r"\b(everyone (does|will)|all your (friends|people))\b", "severity": 0.75, "violation": "social_pressure", "description": "Social pressure and normalization"},
                {"regex": r"\b(if you (don't|care)|unless you|come on|be cool)\b", "severity": 0.8, "violation": "conditional_pressure", "description": "Conditional pressure statements"},
                {"regex": r"\b(i (need|want) you to|you (have to|must|should))\b", "severity": 0.7, "violation": "obligation_creation", "description": "Creating false obligation"},
                {"regex": r"\b(now|immediately|today|right now|can't wait)\b", "severity": 0.78, "violation": "time_pressure", "description": "Time pressure and urgency"},
            ],
            "guilt_tripping": [
                {"regex": r"\b(after (all|everything) (i('ve| have)|we) done)\b", "severity": 0.8, "violation": "sacrifice_reminder", "description": "Reminder of sacrifices"},
                {"regex": r"\b(you('re| are) (so|being) ungrateful|so selfish|don't care about)\b", "severity": 0.82, "violation": "gratitude_demand", "description": "Demanding gratitude"},
                {"regex": r"\b(how could you|i can't believe you|after i)\b", "severity": 0.78, "violation": "disappointment_leverage", "description": "Leveraging disappointment"},
                {"regex": r"\b(owe me|i deserve|i've earned|you promised)\b", "severity": 0.75, "violation": "debt_creation", "description": "Creating sense of debt"},
            ],
            "refusal_dismissal": [
                {"regex": r"\b(no (big deal|problem)|why not|don't be (shy|difficult|annoying))\b", "severity": 0.8, "violation": "dismissal", "description": "Dismissing stated refusal"},
                {"regex": r"\b(you (don't|won't) (mean|actually) (it|that)|you'll change)\b", "severity": 0.82, "violation": "disbelief", "description": "Refusing to believe refusal"},
                {"regex": r"\b(you('re| are) being ((un)?reasonable|dramatic|overreacting))\b", "severity": 0.8, "violation": "invalidation", "description": "Invalidating refusal reasons"},
                {"regex": r"\b(later|next time|when you're ready|eventually)\b", "severity": 0.7, "violation": "delay_tactic", "description": "Attempting to delay acceptance"},
            ],
            "boundary_push": [
                {"regex": r"\b(test(ing)?|push(ing)?|see (what|if)|how (far|much)).{0,50}\b(boundary|limit|line)\b", "severity": 0.85, "violation": "deliberate_testing", "description": "Deliberate boundary testing"},
                {"regex": r"\b(i thought you meant|i thought you said|that's not what you meant)\b", "severity": 0.75, "violation": "reinterpretation", "description": "Reinterpreting boundaries"},
                {"regex": r"\b(didn't say you couldn't|only said no to|didn't prohibit)\b", "severity": 0.82, "violation": "narrow_interpretation", "description": "Narrow boundary interpretation"},
                {"regex": r"\b(one exception|this (time|once)|special circumstance)\b", "severity": 0.78, "violation": "exception_claim", "description": "Claiming exceptions to boundaries"},
            ],
            "emotional_manipulation": [
                {"regex": r"\b(if you (loved|cared|really) (me|about us))\b", "severity": 0.85, "violation": "love_contingency", "description": "Conditioning affection on compliance"},
                {"regex": r"\b(i('ll| will) (hurt|kill) myself|hurt|leave|die)\b.*\b(if you don't|unless)\b", "severity": 0.92, "violation": "self_harm_threat", "description": "Threatening self-harm"},
                {"regex": r"\b(i ('ll| will) (be upset|sad|cry|break))\b", "severity": 0.78, "violation": "emotional_reaction", "description": "Weaponizing emotional reactions"},
                {"regex": r"\b(you('re| are) the only one|only you (can|understand))\b", "severity": 0.72, "violation": "exclusivity_claim", "description": "Claiming exclusivity of support"},
            ]
        }

    def analyze_message(
        self,
        text: str,
        previous_messages: Optional[List[str]] = None,
        previous_refusals: Optional[List[Dict]] = None,
        context_before: str = ""
    ) -> BoundaryAnalysis:
        """Analyze a single message for boundary violations

        Args:
            text: Message text to analyze
            previous_messages: List of previous messages in conversation
            previous_refusals: List of previous refusals from recipient
            context_before: Previous context

        Returns:
            BoundaryAnalysis: Analysis results
        """
        analysis = BoundaryAnalysis()

        if not text:
            return analysis

        # Track if this is a refusal response
        refusal_words = ['no', 'don\'t', 'won\'t', 'can\'t', 'stop', 'please stop', 'leave me alone', 'not interested']
        is_refusal = any(word in text.lower() for word in refusal_words)

        # Check for previous refusals that might be ignored
        refusal_context = self._analyze_refusal_history(previous_refusals, text)

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []

            for pattern_tuple in patterns:
                # Pattern tuple: (regex, severity, violation, description)
                regex, severity, violation, description = pattern_tuple
                matches = list(regex.finditer(text))

                for match in matches:
                    escalation_level = self._calculate_escalation_level(
                        category, violation, refusal_context, len(previous_messages or [])
                    )

                    violation_obj = BoundaryViolation(
                        category="boundary",
                        subcategory=category,
                        violation_type=violation,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(match, text, severity),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description,
                        escalation_level=escalation_level
                    )
                    category_matches.append(violation_obj)
                    analysis.violations_found.append(violation_obj)

            # Calculate category score
            if category_matches:
                total_severity = sum(v.severity * v.confidence for v in category_matches)
                avg_severity = total_severity / len(category_matches)
                analysis.category_scores[category] = avg_severity * self.CATEGORY_WEIGHTS[category]

        # Identify behavior patterns
        analysis.request_escalation_detected = "request_escalation" in analysis.category_scores
        analysis.consent_violations = "consent_violation" in analysis.category_scores
        analysis.pressure_tactics_used = "pressure_tactic" in analysis.category_scores
        analysis.guilt_tripping_detected = "guilt_tripping" in analysis.category_scores

        # Calculate respect for "no"
        analysis.respect_for_no = self._calculate_respect_for_no(refusal_context, analysis)

        # Calculate boundary pushback intensity
        analysis.boundary_pushback_intensity = sum(
            v.escalation_level for v in analysis.violations_found
        ) / max(1, len(analysis.violations_found))

        # Calculate overall metrics
        analysis.violation_count = len(analysis.violations_found)
        analysis.unique_violations = len(set(v.violation_type for v in analysis.violations_found))
        analysis.violation_types = list(set(v.violation_type for v in analysis.violations_found))

        # Determine refusal response pattern
        if is_refusal:
            analysis.refusal_response_pattern = "compliant"
        elif analysis.category_scores:
            # High pushback after refusal = resistant pattern
            if refusal_context["previous_refusals"] > 0 and analysis.boundary_pushback_intensity > 0.7:
                analysis.refusal_response_pattern = "resistant"
            else:
                analysis.refusal_response_pattern = "unclear"

        # Calculate overall score
        if analysis.category_scores:
            analysis.overall_score = self._calculate_overall_score(analysis.category_scores)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_concern = max(analysis.category_scores, key=analysis.category_scores.get)

        return analysis

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for boundary violation patterns

        Args:
            messages: List of message dictionaries

        Returns:
            Dict: Comprehensive analysis results
        """
        results = {
            "per_message_analysis": [],
            "per_speaker_analysis": {},
            "conversation_progression": [],
            "overall_risk": "low",
            "risk_trajectory": "stable",
            "high_risk_messages": [],
            "refusal_patterns": [],
            "escalation_chain": [],
            "pattern_summary": {},
            "boundary_respect_assessment": {},
            "recommendations": [],
            "consent_concerns": []
        }

        speaker_messages = {}
        speaker_refusals = {}

        # First pass: identify refusals
        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '')

            refusal_words = ['no', 'don\'t', 'won\'t', 'can\'t', 'stop', 'please stop', 'leave me alone']
            is_refusal = any(word in text.lower() for word in refusal_words)

            if is_refusal:
                if sender not in speaker_refusals:
                    speaker_refusals[sender] = []
                speaker_refusals[sender].append({
                    "index": i,
                    "text": text,
                    "timestamp": i
                })

        # Second pass: analyze messages
        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Get previous messages from conversation
            previous_messages = [m.get('text', '') for m in messages[:i]]

            # Get previous refusals from OTHER party
            other_sender = None
            for m in messages[:i]:
                if m.get('sender') != sender:
                    other_sender = m.get('sender')
                    break

            previous_refusals = None
            if other_sender and other_sender in speaker_refusals:
                previous_refusals = speaker_refusals[other_sender]

            context_before = messages[i-1].get('text', '') if i > 0 else ""

            # Analyze message
            analysis = self.analyze_message(text, previous_messages, previous_refusals, context_before)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "analysis": analysis,
                "risk_level": analysis.risk_level,
                "boundary_pushback": analysis.boundary_pushback_intensity
            }
            results["per_message_analysis"].append(msg_result)

            # Track by speaker
            if sender not in speaker_messages:
                speaker_messages[sender] = []
            speaker_messages[sender].extend(analysis.violations_found)

            # Track high-risk messages
            if analysis.risk_level in ["high", "critical"]:
                results["high_risk_messages"].append({
                    "index": i,
                    "sender": sender,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "risk_level": analysis.risk_level,
                    "violations": analysis.violation_types,
                    "escalation_intensity": analysis.boundary_pushback_intensity
                })

        # Analyze per speaker
        for speaker, violations in speaker_messages.items():
            speaker_analysis = self._analyze_speaker_patterns(violations)
            results["per_speaker_analysis"][speaker] = speaker_analysis

        # Analyze progression
        results["conversation_progression"] = self._analyze_progression(results["per_message_analysis"])

        # Analyze escalation chain
        results["escalation_chain"] = self._analyze_escalation_chain(results["per_message_analysis"])

        # Identify refusal patterns
        results["refusal_patterns"] = self._analyze_refusal_patterns(results["per_message_analysis"], speaker_refusals)

        # Determine overall risk
        results["overall_risk"] = self._determine_conversation_risk(results)
        results["risk_trajectory"] = self._analyze_trajectory(results["per_message_analysis"])

        # Generate pattern summary
        results["pattern_summary"] = self._generate_pattern_summary(results["per_message_analysis"])

        # Assess boundary respect
        results["boundary_respect_assessment"] = self._assess_boundary_respect(results["per_message_analysis"], speaker_refusals)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Identify consent concerns
        results["consent_concerns"] = self._identify_consent_concerns(results)

        return results

    def _calculate_confidence(self, match: re.Match, text: str, severity: float) -> float:
        """Calculate confidence in violation detection

        Args:
            match: Regex match object
            text: Full message text
            severity: Base severity

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.75 if severity >= 0.8 else 0.65

        if len(text) < 20:
            confidence -= 0.1
        elif len(text) > 250:
            confidence += 0.05

        if match.start() == 0:
            confidence += 0.05

        return max(0.1, min(1.0, confidence))

    def _calculate_escalation_level(
        self,
        category: str,
        violation: str,
        refusal_context: Dict,
        conversation_length: int
    ) -> int:
        """Calculate escalation level (0-5 scale)

        Args:
            category: Violation category
            violation: Specific violation type
            refusal_context: Context of refusals
            conversation_length: Length of conversation

        Returns:
            int: Escalation level (0-5)
        """
        base_level = 2

        # If ignoring previous refusals, escalate
        if refusal_context["previous_refusals"] > 0:
            base_level += refusal_context["previous_refusals"]

        # If escalating requests, increase level
        if category == "request_escalation":
            base_level += 1

        # If using emotional manipulation, increase level
        if violation == "self_harm_threat":
            base_level += 2

        return min(5, base_level)

    def _analyze_refusal_history(self, previous_refusals: Optional[List[Dict]], current_text: str) -> Dict:
        """Analyze how refusals are being handled

        Args:
            previous_refusals: List of previous refusals
            current_text: Current message text

        Returns:
            Dict: Refusal context
        """
        if not previous_refusals:
            return {
                "previous_refusals": 0,
                "after_multiple_refusals": False,
                "ignoring_pattern": False
            }

        refusal_count = len(previous_refusals)
        after_multiple = refusal_count >= 2

        # Check if current text shows pattern of ignoring refusals
        ignoring_pattern = False
        if after_multiple:
            # Check for pushing after clear refusal
            push_words = ['but', 'however', 'still', 'anyway', 'however', 'despite', 'even so']
            ignoring_pattern = any(word in current_text.lower() for word in push_words)

        return {
            "previous_refusals": refusal_count,
            "after_multiple_refusals": after_multiple,
            "ignoring_pattern": ignoring_pattern
        }

    def _calculate_respect_for_no(self, refusal_context: Dict, analysis: BoundaryAnalysis) -> float:
        """Calculate respect for stated boundaries

        Args:
            refusal_context: Refusal context information
            analysis: Analysis results

        Returns:
            float: Respect score (1.0 = full respect, 0.0 = no respect)
        """
        if refusal_context["previous_refusals"] == 0:
            # No previous refusal to assess
            return 1.0

        # Start at full respect
        respect = 1.0

        # Deduct for pushing after refusal
        respect -= 0.2 * refusal_context["previous_refusals"]

        # Deduct for boundary violations after refusal
        if analysis.boundary_pushback_intensity > 0:
            respect -= (analysis.boundary_pushback_intensity * 0.3)

        # Deduct for emotional manipulation after refusal
        if analysis.guilt_tripping_detected or analysis.pressure_tactics_used:
            respect -= 0.2

        return max(0.0, min(1.0, respect))

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall boundary violation score

        Args:
            category_scores: Scores by category

        Returns:
            float: Overall score (0-1)
        """
        if not category_scores:
            return 0.0

        total_score = sum(category_scores.values())
        avg_score = total_score / len(category_scores)

        # Multiple violation types amplify score
        num_types = len(category_scores)
        if num_types >= 4:
            avg_score *= 1.30
        elif num_types >= 3:
            avg_score *= 1.15
        elif num_types >= 2:
            avg_score *= 1.05

        return min(1.0, avg_score)

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score

        Args:
            score: Risk score (0-1)

        Returns:
            str: Risk level
        """
        for level in ["critical", "high", "moderate", "low"]:
            if score >= self.RISK_THRESHOLDS[level]:
                return level
        return "low"

    def _analyze_speaker_patterns(self, violations: List[BoundaryViolation]) -> Dict[str, Any]:
        """Analyze patterns for a specific speaker

        Args:
            violations: List of violations found

        Returns:
            Dict: Speaker analysis
        """
        if not violations:
            return {
                "violation_count": 0,
                "risk_level": "low",
                "violation_types": []
            }

        category_counts = {}
        violation_types = set()

        for violation in violations:
            category_counts[violation.subcategory] = category_counts.get(violation.subcategory, 0) + 1
            violation_types.add(violation.violation_type)

        total_severity = sum(v.severity * v.confidence for v in violations)
        avg_severity = total_severity / len(violations) if violations else 0
        avg_escalation = sum(v.escalation_level for v in violations) / len(violations) if violations else 0

        return {
            "violation_count": len(violations),
            "unique_violations": len(set(v.violation_type for v in violations)),
            "risk_level": self._determine_risk_level(avg_severity),
            "categories": list(category_counts.keys()),
            "violation_types": list(violation_types),
            "average_severity": avg_severity,
            "average_escalation_level": avg_escalation
        }

    def _analyze_progression(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze boundary violation progression

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Progression events
        """
        progression = []
        previous_risk = "low"

        for i, msg_analysis in enumerate(message_analyses):
            analysis = msg_analysis["analysis"]
            current_risk = analysis.risk_level

            risk_levels = ["low", "moderate", "high", "critical"]
            if risk_levels.index(current_risk) > risk_levels.index(previous_risk):
                progression.append({
                    "type": "risk_escalation",
                    "message_index": i,
                    "from": previous_risk,
                    "to": current_risk,
                    "sender": msg_analysis["sender"]
                })

            # Track violation introduction
            if analysis.violations_found:
                progression.append({
                    "type": "violations_present",
                    "message_index": i,
                    "violation_count": len(analysis.violations_found),
                    "sender": msg_analysis["sender"]
                })

            previous_risk = current_risk

        return progression

    def _analyze_escalation_chain(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze chain of escalating requests

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Escalation chain events
        """
        chain = []
        current_chain = []

        for msg in message_analyses:
            analysis = msg["analysis"]

            if analysis.request_escalation_detected:
                current_chain.append({
                    "index": msg["index"],
                    "sender": msg["sender"],
                    "intensity": analysis.boundary_pushback_intensity
                })
            else:
                # Chain broken
                if len(current_chain) >= 2:
                    chain.append({
                        "chain_length": len(current_chain),
                        "messages": current_chain,
                        "start_index": current_chain[0]["index"],
                        "end_index": current_chain[-1]["index"]
                    })
                current_chain = []

        return chain

    def _analyze_refusal_patterns(
        self,
        message_analyses: List[Dict],
        speaker_refusals: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """Analyze patterns of refusal and response

        Args:
            message_analyses: List of per-message analyses
            speaker_refusals: Dictionary of refusals by speaker

        Returns:
            List: Refusal patterns
        """
        patterns = []

        for speaker, refusals in speaker_refusals.items():
            for refusal in refusals:
                refusal_idx = refusal["index"]

                # Look for responses after this refusal
                for msg in message_analyses:
                    if msg["index"] > refusal_idx and msg["sender"] != speaker:
                        analysis = msg["analysis"]

                        if analysis.violations_found:
                            patterns.append({
                                "refusal_index": refusal_idx,
                                "response_index": msg["index"],
                                "refuser": speaker,
                                "responder": msg["sender"],
                                "responder_violated_boundary": True,
                                "violation_types": analysis.violation_types,
                                "severity": analysis.overall_score
                            })
                            break

        return patterns

    def _analyze_trajectory(self, message_analyses: List[Dict]) -> str:
        """Analyze risk trajectory

        Args:
            message_analyses: List of per-message analyses

        Returns:
            str: Trajectory
        """
        if len(message_analyses) < 3:
            return "stable"

        risk_scores = []
        risk_levels = {"low": 0.1, "moderate": 0.35, "high": 0.6, "critical": 0.85}

        for msg in message_analyses:
            score = risk_levels.get(msg["analysis"].risk_level, 0.1)
            risk_scores.append(score)

        first_third = sum(risk_scores[:len(risk_scores)//3]) / max(1, len(risk_scores)//3)
        last_third = sum(risk_scores[-len(risk_scores)//3:]) / max(1, len(risk_scores)//3)

        if last_third > first_third * 1.5:
            return "escalating"
        elif last_third < first_third * 0.5:
            return "de-escalating"
        else:
            return "stable"

    def _determine_conversation_risk(self, results: Dict) -> str:
        """Determine overall conversation risk

        Args:
            results: Analysis results

        Returns:
            str: Risk level
        """
        high_risk_count = len(results["high_risk_messages"])
        refusal_violations = len(results["refusal_patterns"])

        speaker_risks = [
            analysis["risk_level"]
            for analysis in results["per_speaker_analysis"].values()
        ]

        if refusal_violations >= 3 or "critical" in speaker_risks:
            return "critical"
        elif high_risk_count >= 3 or "high" in speaker_risks:
            return "high"
        elif "moderate" in speaker_risks or high_risk_count >= 1:
            return "moderate"
        else:
            return "low"

    def _generate_pattern_summary(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate summary of boundary violations

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Pattern summary
        """
        all_violations = []
        category_counts = {}
        violation_type_counts = {}

        for msg in message_analyses:
            violations = msg["analysis"].violations_found
            all_violations.extend(violations)

            for violation in violations:
                category_counts[violation.subcategory] = category_counts.get(violation.subcategory, 0) + 1
                violation_type_counts[violation.violation_type] = violation_type_counts.get(violation.violation_type, 0) + 1

        return {
            "total_violations": len(all_violations),
            "unique_violation_types": len(set(v.violation_type for v in all_violations)),
            "category_distribution": category_counts,
            "violation_type_distribution": violation_type_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "most_common_violation": max(violation_type_counts, key=violation_type_counts.get) if violation_type_counts else None,
            "average_severity": sum(v.severity for v in all_violations) / len(all_violations) if all_violations else 0
        }

    def _assess_boundary_respect(
        self,
        message_analyses: List[Dict],
        speaker_refusals: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Assess overall boundary respect in conversation

        Args:
            message_analyses: List of per-message analyses
            speaker_refusals: Dictionary of refusals by speaker

        Returns:
            Dict: Boundary respect assessment
        """
        speaker_respect = {}

        for msg in message_analyses:
            analysis = msg["analysis"]
            sender = msg["sender"]

            if sender not in speaker_respect:
                speaker_respect[sender] = {
                    "respect_scores": [],
                    "violation_count": 0,
                    "escalation_intensity": []
                }

            speaker_respect[sender]["respect_scores"].append(analysis.respect_for_no)
            speaker_respect[sender]["violation_count"] += analysis.violation_count
            speaker_respect[sender]["escalation_intensity"].append(analysis.boundary_pushback_intensity)

        # Calculate averages
        for speaker, data in speaker_respect.items():
            data["average_respect"] = sum(data["respect_scores"]) / len(data["respect_scores"]) if data["respect_scores"] else 1.0
            data["average_escalation"] = sum(data["escalation_intensity"]) / len(data["escalation_intensity"]) if data["escalation_intensity"] else 0.0

        return speaker_respect

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        risk = results["overall_risk"]

        if risk == "critical":
            recommendations.append("CRITICAL: Severe and repeated boundary violations detected. Professional support essential.")
            recommendations.append("Clear boundaries need to be established and enforced with consequences.")
            recommendations.append("Consider safety planning and limit contact if necessary.")

        elif risk == "high":
            recommendations.append("HIGH RISK: Multiple boundary violations and escalating requests detected.")
            recommendations.append("Establish clear, firm boundaries and communicate them explicitly.")
            recommendations.append("Seek counseling or mediation to address relationship dynamics.")

        elif risk == "moderate":
            recommendations.append("MODERATE RISK: Some boundary violations detected.")
            recommendations.append("Communicate boundaries clearly and set consequences for violations.")
            recommendations.append("Monitor for escalation patterns.")

        else:
            recommendations.append("LOW RISK: Minimal boundary violations detected.")

        if results["escalation_chain"]:
            recommendations.append("Escalating request pattern detected. Address early in future interactions.")

        return recommendations

    def _identify_consent_concerns(self, results: Dict) -> List[str]:
        """Identify specific consent-related concerns

        Args:
            results: Analysis results

        Returns:
            List: Consent concerns
        """
        concerns = []

        pattern_summary = results["pattern_summary"]
        if pattern_summary.get("most_common_category") == "consent_violation":
            concerns.append("Direct consent violations detected - physical or informational boundaries crossed")

        if results["boundary_respect_assessment"]:
            for speaker, data in results["boundary_respect_assessment"].items():
                if data["average_respect"] < 0.5:
                    concerns.append(f"Low respect for stated boundaries from {speaker}")

        if results["refusal_patterns"]:
            concerns.append(f"Refusals ignored or not respected - {len(results['refusal_patterns'])} instances")

        return concerns
