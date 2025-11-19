"""
Self-Harm Indicator Detection Module
Comprehensive detection of self-harm indicators, suicidal ideation, and crisis signals.
Based on clinical research and suicide prevention frameworks (SAMHSA, CDC, AAS).
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
class SelfHarmPattern:
    """Container for a self-harm pattern match"""
    category: str
    subcategory: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""
    requires_immediate_action: bool = False


@dataclass
class SelfHarmAnalysis:
    """Complete self-harm analysis results"""
    patterns_found: List[SelfHarmPattern] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    pattern_count: int = 0
    unique_patterns: int = 0
    crisis_indicators: List[str] = field(default_factory=list)
    help_seeking_indicators: bool = False
    help_rejecting_indicators: bool = False
    immediate_danger: bool = False
    emergency_contact_required: bool = False


class SelfHarmIndicatorDetector:
    """Detects self-harm and suicidal ideation indicators in text messages"""

    # Risk categories
    HARM_CATEGORIES = [
        "direct_self_harm",
        "suicidal_ideation",
        "hopelessness_despair",
        "means_methods",
        "plan_indicators",
        "goodbye_messages",
        "help_seeking",
        "help_rejecting"
    ]

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.15,
        "moderate": 0.35,
        "high": 0.60,
        "critical": 0.80
    }

    # Category weights (based on clinical severity)
    CATEGORY_WEIGHTS = {
        "direct_self_harm": 1.5,
        "suicidal_ideation": 1.8,
        "hopelessness_despair": 1.2,
        "means_methods": 1.6,
        "plan_indicators": 1.7,
        "goodbye_messages": 1.9,
        "help_seeking": -0.5,  # Negative = protective
        "help_rejecting": 0.8
    }

    # Crisis resource information
    CRISIS_RESOURCES = {
        "988_suicide_lifeline": {
            "description": "National Suicide Prevention Lifeline (US)",
            "contact": "Call or text 988",
            "available": "24/7"
        },
        "crisis_text_line": {
            "description": "Crisis Text Line",
            "contact": "Text HOME to 741741",
            "available": "24/7"
        },
        "international_association": {
            "description": "International Association for Suicide Prevention",
            "contact": "https://www.iasp.info/resources/Crisis_Centres/",
            "available": "Variable by country"
        }
    }

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize self-harm detector with cached patterns

        Args:
            patterns_file: Path to patterns JSON file
        """
        cache = get_cache()
        cache_key = f"self_harm_patterns_{patterns_file or 'default'}"

        # Load patterns from cache (or file if not cached)
        self.patterns = cache.get_or_load(
            cache_key,
            load_patterns_file,
            patterns_file,
            'self_harm'
        )

        # If patterns are empty, use defaults
        if not self.patterns:
            self.patterns = self._get_default_patterns()

        # Compile and cache patterns
        compiled_cache_key = f"self_harm_compiled_{patterns_file or 'default'}"
        self.compiled_patterns = cache.get_or_load(
            compiled_cache_key,
            compile_regex_patterns,
            self.patterns
        )

    def _get_default_patterns(self) -> Dict:
        """Get default self-harm patterns based on clinical research

        Returns:
            Dict: Default patterns
        """
        return {
            "direct_self_harm": [
                {"regex": r"\b(cut|cutting|cuts|self-harm|self harm|hurt myself|injure)\b", "severity": 0.9, "description": "Direct self-harm references"},
                {"regex": r"\b(blade|knife|razor|sharp)\b.*\b(cut|harm|hurt)\b", "severity": 0.85, "description": "Instrument and harm combination"},
                {"regex": r"\bburn(ing|s|ed)?\b.*\b(myself|me)\b", "severity": 0.85, "description": "Self-burning references"},
                {"regex": r"\bban(ning|s|dage|daging)\b.*\b(wound|cut|mark)\b", "severity": 0.75, "description": "Wound management language"},
            ],
            "suicidal_ideation": [
                {"regex": r"\b(suicide|suicidal|kill myself|end (my )?life|take my life)\b", "severity": 0.95, "description": "Direct suicidal language"},
                {"regex": r"\b(want to die|wish i was dead|better off dead|kill (me|myself))\b", "severity": 0.9, "description": "Death wish language"},
                {"regex": r"\b(don't want to live|can't go on|no point (in|to)|not worth living)\b", "severity": 0.85, "description": "Life negation statements"},
                {"regex": r"\b(end (everything|it|this)|escape this|permanent solution)\b", "severity": 0.8, "description": "Escapism/ending references"},
            ],
            "hopelessness_despair": [
                {"regex": r"\b(hopeless|despair|desperat|worthless|useless|pointless)\b", "severity": 0.75, "description": "Hopelessness markers"},
                {"regex": r"\b(can't take (anymore|this|it)|breaking down|falling apart)\b", "severity": 0.7, "description": "Breakdown markers"},
                {"regex": r"\b(trapped|stuck|no way out|impossible|never get better)\b", "severity": 0.8, "description": "Entrapment language"},
                {"regex": r"\b(burden|mistake|failure|ashamed|shame|guilty)\b", "severity": 0.65, "description": "Self-blame and guilt markers"},
                {"regex": r"\b(alone|lonely|nobody cares|isolated)\b", "severity": 0.6, "description": "Isolation and loneliness markers"},
            ],
            "means_methods": [
                {"regex": r"\b(pills|medication|overdose|od|poison)\b", "severity": 0.88, "description": "Overdose method references"},
                {"regex": r"\b(gun|firearm|weapon|rope|noose|cord)\b", "severity": 0.9, "description": "Violent method references"},
                {"regex": r"\b(jump|building|bridge|height|fall)\b", "severity": 0.85, "description": "Jumping method references"},
                {"regex": r"\b(car|exhaust|carbon monoxide|gas)\b", "severity": 0.82, "description": "Vehicle/suffocation methods"},
                {"regex": r"\b(access|obtain|get|find).{1,50}\b(pills|gun|rope|method)\b", "severity": 0.85, "description": "Method seeking behavior"},
            ],
            "plan_indicators": [
                {"regex": r"\b(planned|plan to|going to|will|when i|if i|this week|this month)\b.*\b(kill|hurt|harm|die|end it)\b", "severity": 0.92, "description": "Specific planning language"},
                {"regex": r"\b(already|research|look(ed)?|figure(d)?|know how)\b.*\b(method|way to|how to)\b", "severity": 0.88, "description": "Preparation indicators"},
                {"regex": r"\b(last|final|goodbye|farewell|soon|time is running out)\b", "severity": 0.85, "description": "Temporal pressure language"},
                {"regex": r"\b(after|when|once|if i|before).{1,100}\b(note|letter|will|inheritance|goodbye)\b", "severity": 0.87, "description": "Afterlife planning language"},
            ],
            "goodbye_messages": [
                {"regex": r"\b(goodbye|bye|farewell|it's been|thanks for|i love you)\b.*\b(all|everyone|you|guys)\b", "severity": 0.9, "description": "Farewell patterns"},
                {"regex": r"\b(forgive me|sorry|i'm sorry|my fault|my mistake)\b", "severity": 0.7, "description": "Apology and blame patterns"},
                {"regex": r"\b(take care|take my|don't forget|remember me|miss me)\b", "severity": 0.8, "description": "Remembrance language"},
                {"regex": r"\b(thank you|grateful|appreciate|legacy|will|left you)\b", "severity": 0.75, "description": "Legacy-focused language"},
            ],
            "help_seeking": [
                {"regex": r"\b(help|save me|need help|please help|support|therapist|doctor|call|emergency)\b", "severity": -0.6, "description": "Help-seeking language"},
                {"regex": r"\b(can't do (it|this) alone|reaching out|talk(ing)?)\b", "severity": -0.5, "description": "Support-seeking indicators"},
                {"regex": r"\b(go to hospital|get help|treatment|medication|counseling)\b", "severity": -0.7, "description": "Professional help indicators"},
            ],
            "help_rejecting": [
                {"regex": r"\b(no one can help|beyond help|not worth (saving|helping)|too late)\b", "severity": 0.85, "description": "Help rejection markers"},
                {"regex": r"\b(don't bother|no point|won't help|tried everything|nothing works)\b", "severity": 0.8, "description": "Treatment hopelessness"},
                {"regex": r"\b(don't tell anyone|keep (it|this) secret|don't call)\b", "severity": 0.75, "description": "Help resistance language"},
            ]
        }

    def analyze_message(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
        speaker_history: Optional[List[Dict]] = None
    ) -> SelfHarmAnalysis:
        """Analyze a single message for self-harm indicators

        Args:
            text: Message text to analyze
            context_before: Previous message(s) for context
            context_after: Following message(s) for context
            speaker_history: Historical messages from this speaker

        Returns:
            SelfHarmAnalysis: Analysis results
        """
        analysis = SelfHarmAnalysis()

        if not text:
            return analysis

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []

            for regex, severity, description in patterns:
                matches = list(regex.finditer(text))

                for match in matches:
                    requires_action = severity >= 0.85
                    pattern = SelfHarmPattern(
                        category="self_harm",
                        subcategory=category,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(
                            match, text, context_before, context_after, severity
                        ),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description,
                        requires_immediate_action=requires_action
                    )
                    category_matches.append(pattern)
                    analysis.patterns_found.append(pattern)

            # Calculate category score
            if category_matches:
                total_severity = sum(p.severity * p.confidence for p in category_matches)
                avg_severity = total_severity / len(category_matches)
                analysis.category_scores[category] = avg_severity * self.CATEGORY_WEIGHTS[category]

        # Identify help-seeking vs help-rejecting
        analysis.help_seeking_indicators = "help_seeking" in analysis.category_scores
        analysis.help_rejecting_indicators = "help_rejecting" in analysis.category_scores

        # Calculate overall metrics
        analysis.pattern_count = len(analysis.patterns_found)
        analysis.unique_patterns = len(set(p.pattern for p in analysis.patterns_found))

        # Check for immediate danger indicators
        immediate_danger_patterns = [
            p for p in analysis.patterns_found
            if p.requires_immediate_action and p.severity >= 0.85
        ]
        analysis.immediate_danger = len(immediate_danger_patterns) >= 2
        analysis.emergency_contact_required = len(immediate_danger_patterns) >= 1 or any(
            score > self.RISK_THRESHOLDS["critical"]
            for score in analysis.category_scores.values()
        )

        # Calculate overall score
        if analysis.category_scores:
            analysis.overall_score = self._calculate_overall_score(analysis.category_scores)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_concern = max(analysis.category_scores, key=analysis.category_scores.get)

        # Identify crisis indicators
        analysis.crisis_indicators = self._identify_crisis_indicators(analysis)

        return analysis

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for self-harm escalation

        Args:
            messages: List of message dictionaries with 'text' and 'sender' keys

        Returns:
            Dict: Comprehensive analysis results
        """
        results = {
            "per_message_analysis": [],
            "per_speaker_analysis": {},
            "conversation_progression": [],
            "overall_risk": "low",
            "risk_trajectory": "stable",  # stable, escalating, de-escalating
            "high_risk_messages": [],
            "crisis_messages": [],
            "pattern_summary": {},
            "recommendations": [],
            "emergency_contacts": [],
            "resources": self.CRISIS_RESOURCES
        }

        speaker_patterns = {}

        # Analyze each message with context
        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Get context
            context_before = messages[i-1].get('text', '') if i > 0 else ""
            context_after = messages[i+1].get('text', '') if i < len(messages) - 1 else ""

            # Get speaker history
            speaker_history = [m for m in messages[:i] if m.get('sender') == sender]

            # Analyze message
            analysis = self.analyze_message(text, context_before, context_after, speaker_history)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "analysis": analysis,
                "risk_level": analysis.risk_level,
                "emergency_contact_required": analysis.emergency_contact_required
            }
            results["per_message_analysis"].append(msg_result)

            # Track by speaker
            if sender not in speaker_patterns:
                speaker_patterns[sender] = []
            speaker_patterns[sender].extend(analysis.patterns_found)

            # Track high-risk messages
            if analysis.risk_level in ["high", "critical"]:
                risk_data = {
                    "index": i,
                    "sender": sender,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "risk_level": analysis.risk_level,
                    "primary_concern": analysis.primary_concern
                }
                results["high_risk_messages"].append(risk_data)

            # Track crisis messages requiring immediate action
            if analysis.emergency_contact_required or analysis.immediate_danger:
                results["crisis_messages"].append({
                    "index": i,
                    "sender": sender,
                    "text": text,
                    "risk_level": analysis.risk_level,
                    "urgent": analysis.immediate_danger,
                    "crisis_indicators": analysis.crisis_indicators
                })

        # Analyze per speaker
        for speaker, patterns in speaker_patterns.items():
            speaker_analysis = self._analyze_speaker_patterns(patterns)
            results["per_speaker_analysis"][speaker] = speaker_analysis

        # Analyze conversation progression
        results["conversation_progression"] = self._analyze_progression(results["per_message_analysis"])

        # Determine overall risk and trajectory
        results["overall_risk"] = self._determine_conversation_risk(results)
        results["risk_trajectory"] = self._analyze_trajectory(results["per_message_analysis"])

        # Generate pattern summary
        results["pattern_summary"] = self._generate_pattern_summary(results["per_message_analysis"])

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Set emergency contacts if needed
        if results["overall_risk"] in ["high", "critical"]:
            results["emergency_contacts"] = self._get_emergency_contacts()

        return results

    def _calculate_confidence(
        self,
        match: re.Match,
        text: str,
        context_before: str,
        context_after: str,
        severity: float
    ) -> float:
        """Calculate confidence in pattern detection

        Args:
            match: Regex match object
            text: Full message text
            context_before: Previous context
            context_after: Following context
            severity: Base severity of pattern

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.65 if severity < 0 else 0.7

        # Adjust based on match position
        if match.start() == 0 or match.end() == len(text):
            confidence += 0.1

        # Adjust based on message length
        if len(text) < 20:
            confidence -= 0.1
        elif len(text) > 200:
            confidence += 0.1

        # Context from surrounding messages
        if context_before:
            crisis_words = ['help', 'save', 'please', 'afraid', 'scared']
            if any(word in context_before.lower() for word in crisis_words):
                confidence += 0.05 if severity >= 0 else 0.1

        if context_after:
            affirmation_words = ['yes', 'okay', 'alright', 'understand']
            if any(word in context_after.lower() for word in affirmation_words):
                confidence += 0.05 if severity >= 0 else 0.1

        return max(0.1, min(1.0, confidence))

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall self-harm risk score

        Args:
            category_scores: Scores by category

        Returns:
            float: Overall score (0-1)
        """
        if not category_scores:
            return 0.0

        # Filter out protective factors (help-seeking)
        positive_scores = {k: v for k, v in category_scores.items() if v > 0}
        negative_scores = {k: v for k, v in category_scores.items() if v <= 0}

        if not positive_scores:
            return 0.0

        # Calculate with protective factors
        risk_sum = sum(positive_scores.values())
        protection_sum = abs(sum(negative_scores.values()))

        overall = risk_sum / (risk_sum + protection_sum + 1)

        # Multiple high-risk categories amplify score
        high_risk_count = sum(1 for v in positive_scores.values() if v > 0.5)
        if high_risk_count >= 3:
            overall *= 1.3
        elif high_risk_count >= 2:
            overall *= 1.15

        return min(1.0, overall)

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

    def _analyze_speaker_patterns(self, patterns: List[SelfHarmPattern]) -> Dict[str, Any]:
        """Analyze patterns for a specific speaker

        Args:
            patterns: List of patterns found

        Returns:
            Dict: Speaker analysis
        """
        if not patterns:
            return {
                "pattern_count": 0,
                "risk_level": "low",
                "categories_detected": [],
                "has_critical_patterns": False,
                "help_seeking": False,
                "help_rejecting": False
            }

        # Count patterns by category
        category_counts = {}
        help_seeking = False
        help_rejecting = False

        for pattern in patterns:
            category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1
            if pattern.subcategory == "help_seeking":
                help_seeking = True
            elif pattern.subcategory == "help_rejecting":
                help_rejecting = True

        # Calculate risk
        total_severity = sum(p.severity * p.confidence for p in patterns)
        avg_severity = total_severity / len(patterns) if patterns else 0

        # Check for critical patterns
        has_critical = any(p.severity >= 0.85 for p in patterns)

        return {
            "pattern_count": len(patterns),
            "unique_patterns": len(set(p.pattern for p in patterns)),
            "risk_level": self._determine_risk_level(avg_severity),
            "categories_detected": list(category_counts.keys()),
            "category_counts": category_counts,
            "average_severity": avg_severity,
            "has_critical_patterns": has_critical,
            "help_seeking": help_seeking,
            "help_rejecting": help_rejecting,
            "most_severe_pattern": max(patterns, key=lambda p: p.severity) if patterns else None
        }

    def _analyze_progression(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze self-harm indicator progression over time

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Progression events
        """
        progression = []
        previous_risk = "low"
        escalations = 0

        for i, msg_analysis in enumerate(message_analyses):
            analysis = msg_analysis["analysis"]
            current_risk = analysis.risk_level

            # Track escalation
            risk_levels = ["low", "moderate", "high", "critical"]
            if risk_levels.index(current_risk) > risk_levels.index(previous_risk):
                escalations += 1
                progression.append({
                    "type": "escalation",
                    "from": previous_risk,
                    "to": current_risk,
                    "message_index": i,
                    "sender": msg_analysis["sender"],
                    "primary_concern": analysis.primary_concern
                })

            # Track crisis events
            if analysis.emergency_contact_required:
                progression.append({
                    "type": "crisis_event",
                    "message_index": i,
                    "sender": msg_analysis["sender"],
                    "severity": analysis.overall_score
                })

            previous_risk = current_risk

        return progression

    def _analyze_trajectory(self, message_analyses: List[Dict]) -> str:
        """Analyze risk trajectory over conversation

        Args:
            message_analyses: List of per-message analyses

        Returns:
            str: Trajectory
        """
        if len(message_analyses) < 3:
            return "stable"

        # Get risk scores over time
        risk_scores = []
        risk_levels = {"low": 0.1, "moderate": 0.35, "high": 0.6, "critical": 0.85}

        for msg in message_analyses:
            score = risk_levels.get(msg["analysis"].risk_level, 0.1)
            risk_scores.append(score)

        # Analyze trend
        first_third = sum(risk_scores[:len(risk_scores)//3]) / max(1, len(risk_scores)//3)
        last_third = sum(risk_scores[-len(risk_scores)//3:]) / max(1, len(risk_scores)//3)

        if last_third > first_third * 1.5:
            return "escalating"
        elif last_third < first_third * 0.5:
            return "de-escalating"
        else:
            return "stable"

    def _determine_conversation_risk(self, results: Dict) -> str:
        """Determine overall conversation risk level

        Args:
            results: Analysis results

        Returns:
            str: Risk level
        """
        crisis_count = len(results["crisis_messages"])
        high_risk_count = len(results["high_risk_messages"])

        # Check speaker risks
        speaker_risks = [
            analysis["risk_level"]
            for analysis in results["per_speaker_analysis"].values()
        ]

        if crisis_count >= 1 or "critical" in speaker_risks:
            return "critical"
        elif high_risk_count >= 3 or "high" in speaker_risks:
            return "high"
        elif "moderate" in speaker_risks or high_risk_count >= 1:
            return "moderate"
        else:
            return "low"

    def _generate_pattern_summary(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate summary of patterns found

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Pattern summary
        """
        all_patterns = []
        category_counts = {}

        for msg in message_analyses:
            patterns = msg["analysis"].patterns_found
            all_patterns.extend(patterns)

            for pattern in patterns:
                category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1

        return {
            "total_patterns": len(all_patterns),
            "unique_patterns": len(set(p.pattern for p in all_patterns)),
            "category_distribution": category_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "average_severity": sum(p.severity for p in all_patterns) / len(all_patterns) if all_patterns else 0,
            "critical_patterns": len([p for p in all_patterns if p.requires_immediate_action])
        }

    def _identify_crisis_indicators(self, analysis: SelfHarmAnalysis) -> List[str]:
        """Identify specific crisis indicators

        Args:
            analysis: Analysis results

        Returns:
            List: Crisis indicator descriptions
        """
        indicators = []

        for pattern in analysis.patterns_found:
            if pattern.requires_immediate_action:
                indicators.append(f"{pattern.subcategory}: {pattern.description}")

        return list(set(indicators))

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate safety recommendations

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        risk = results["overall_risk"]
        trajectory = results["risk_trajectory"]

        if risk == "critical":
            recommendations.append("CRITICAL: Immediate professional intervention required. Contact emergency services or crisis line immediately.")
            recommendations.append("Document all messages and keep evidence for potential professional assessment.")
            recommendations.append("Never leave the person alone if possible; maintain continuous support.")
            recommendations.append("Activate emergency protocols and involve trained mental health professionals.")

        elif risk == "high":
            recommendations.append("HIGH RISK: Strong self-harm indicators detected. Seek immediate professional mental health support.")
            recommendations.append("Contact a suicide prevention hotline or mental health crisis team.")
            recommendations.append("Avoid dismissing concerns or leaving the person isolated.")
            recommendations.append("Consider involving family members or trusted support system.")

        elif risk == "moderate":
            recommendations.append("MODERATE RISK: Concerning patterns detected. Encourage professional mental health support.")
            recommendations.append("Be supportive and non-judgmental. Listen without trying to fix.")
            recommendations.append("Share crisis resources and encourage their use.")

        else:
            recommendations.append("LOW RISK: No significant self-harm indicators detected.")

        # Trajectory-based
        if trajectory == "escalating":
            recommendations.append("WARNING: Risk level is increasing. Monitor closely and escalate support quickly.")
        elif trajectory == "de-escalating":
            recommendations.append("POSITIVE: Risk level is decreasing. Continue support and professional involvement.")

        # Pattern-specific
        pattern_summary = results["pattern_summary"]
        if pattern_summary["most_common_category"] == "hopelessness_despair":
            recommendations.append("Help instill hope through professional therapy and support systems.")
        elif pattern_summary["most_common_category"] == "help_rejecting":
            recommendations.append("Focus on building trust and demonstrating that help is available without judgment.")

        return recommendations

    def _get_emergency_contacts(self) -> List[Dict[str, str]]:
        """Get emergency contact information

        Returns:
            List: Emergency contact information
        """
        return [
            {
                "service": "National Suicide Prevention Lifeline (US)",
                "contact": "988 or 1-800-273-8255",
                "method": "Call or text",
                "available": "24/7"
            },
            {
                "service": "Crisis Text Line",
                "contact": "Text HOME to 741741",
                "method": "Text",
                "available": "24/7"
            },
            {
                "service": "Emergency Services",
                "contact": "911 (US), 999 (UK), 112 (EU)",
                "method": "Call",
                "available": "24/7"
            },
            {
                "service": "International Association for Suicide Prevention",
                "contact": "https://www.iasp.info/resources/Crisis_Centres/",
                "method": "Online directory",
                "available": "Variable by country"
            }
        ]
