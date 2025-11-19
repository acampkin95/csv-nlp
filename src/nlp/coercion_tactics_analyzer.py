"""
Coercion Tactics Analysis Module
Detects patterns of coercion, control, and abuse using the Duluth Model framework.
Identifies isolation, economic control, surveillance, threats, and intimidation tactics.
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
class CoercionPattern:
    """Container for a coercion pattern match"""
    category: str
    subcategory: str
    tactic_type: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""
    power_control_level: float = 0.0  # 0-1 scale


@dataclass
class CoercionAnalysis:
    """Complete coercion tactics analysis results"""
    patterns_found: List[CoercionPattern] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    pattern_count: int = 0
    unique_patterns: int = 0
    tactic_types_detected: List[str] = field(default_factory=list)
    power_control_intensity: float = 0.0
    isolation_indicators: bool = False
    economic_control: bool = False
    surveillance_detected: bool = False
    threat_patterns: bool = False
    intimidation_markers: bool = False
    duluth_model_wheels: List[str] = field(default_factory=list)


class CoercionTacticsAnalyzer:
    """Analyzes coercive and controlling behavior patterns"""

    # Duluth Model dimensions
    DULUTH_DIMENSIONS = [
        "physical_violence",
        "sexual_violence",
        "threats",
        "intimidation",
        "emotional_abuse",
        "isolation",
        "economic_abuse",
        "sexual_coercion",
        "male_privilege",
        "children_use",
        "monitoring",
        "control"
    ]

    # Tactic categories
    TACTIC_CATEGORIES = [
        "isolation",
        "economic_control",
        "surveillance",
        "threats",
        "intimidation",
        "emotional_abuse",
        "reproductive_coercion",
        "financial_exploitation"
    ]

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.20,
        "moderate": 0.40,
        "high": 0.65,
        "critical": 0.85
    }

    # Category weights (based on harm severity)
    CATEGORY_WEIGHTS = {
        "isolation": 1.3,
        "economic_control": 1.4,
        "surveillance": 1.2,
        "threats": 1.7,
        "intimidation": 1.5,
        "emotional_abuse": 1.1,
        "reproductive_coercion": 1.6,
        "financial_exploitation": 1.3
    }

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize coercion analyzer with cached patterns

        Args:
            patterns_file: Path to patterns JSON file
        """
        cache = get_cache()
        cache_key = f"coercion_patterns_{patterns_file or 'default'}"

        # Load patterns from cache (or file if not cached)
        self.patterns = cache.get_or_load(
            cache_key,
            load_patterns_file,
            patterns_file,
            'coercion'
        )

        # If patterns are empty, use defaults
        if not self.patterns:
            self.patterns = self._get_default_patterns()

        # Compile and cache patterns
        compiled_cache_key = f"coercion_compiled_{patterns_file or 'default'}"
        self.compiled_patterns = cache.get_or_load(
            compiled_cache_key,
            compile_regex_patterns,
            self.patterns
        )

    def _get_default_patterns(self) -> Dict:
        """Get default coercion patterns based on Duluth Model

        Returns:
            Dict: Default patterns
        """
        return {
            "isolation": [
                {"regex": r"\b(can't talk to|don't talk to|isolate|isolated|cut (you )?off)\b", "severity": 0.85, "tactic": "social_isolation", "description": "Isolation tactics"},
                {"regex": r"\b(your friends|your family).{0,30}\b(don't like|are bad|problem|jealous)\b", "severity": 0.8, "tactic": "relationship_sabotage", "description": "Undermining relationships"},
                {"regex": r"\b(stay away from|don't see|can't hang out with)\b", "severity": 0.85, "tactic": "association_restriction", "description": "Restricting associations"},
                {"regex": r"\b(only (i|me)|depend on (me|only me)|need (only|just) me)\b", "severity": 0.8, "tactic": "exclusivity", "description": "Creating dependency"},
                {"regex": r"\b(alone|lonely|trapped|stuck with me)\b", "severity": 0.75, "tactic": "emotional_trapping", "description": "Emotional imprisonment language"},
            ],
            "economic_control": [
                {"regex": r"\b(money|spend|job|work|income|earn).{0,40}\b(can't|won't|don't|shouldn't)\b", "severity": 0.9, "tactic": "income_control", "description": "Income restriction"},
                {"regex": r"\b(pay (me|for)|give (me|money)|financially responsible)\b", "severity": 0.85, "tactic": "financial_dependence", "description": "Financial dependence creation"},
                {"regex": r"\b(can't (afford|buy|have)|no (money|choice))\b", "severity": 0.8, "tactic": "economic_restriction", "description": "Economic restriction"},
                {"regex": r"\b(bills|rent|housing|support).{0,30}\b(need|owe|responsible)\b", "severity": 0.8, "tactic": "debt_leverage", "description": "Debt and obligation leverage"},
            ],
            "surveillance": [
                {"regex": r"\b(where (are|were) you|who (are|were) you with|what (are|were) you doing)\b", "severity": 0.8, "tactic": "location_monitoring", "description": "Location tracking questions"},
                {"regex": r"\b(check (your|my) phone|see your|look at your messages|track|gps|phone)\b", "severity": 0.9, "tactic": "digital_surveillance", "description": "Digital monitoring"},
                {"regex": r"\b(watching|saw you|followed|know where)\b", "severity": 0.85, "tactic": "physical_surveillance", "description": "Physical surveillance"},
                {"regex": r"\b(who texted|who called|check your|login to)\b", "severity": 0.88, "tactic": "communication_monitoring", "description": "Communication surveillance"},
            ],
            "threats": [
                {"regex": r"\b(i('ll| will) (hurt|harm|kill|leave you|take)|(threat|promise) to)\b", "severity": 0.95, "tactic": "direct_threat", "description": "Direct threat language"},
                {"regex": r"\b(if you (don't|leave|tell)|you'll (regret|pay|be sorry))\b", "severity": 0.9, "tactic": "conditional_threat", "description": "Conditional threats"},
                {"regex": r"\b(i know where|don't think i (can't|won't)|you think)\b.*\b(i'm serious|kidding)\b", "severity": 0.85, "tactic": "implied_threat", "description": "Implied threat language"},
                {"regex": r"\b(call (the cops|police)|social services|tell|report|expose)\b.*\b(i'll|threat)\b", "severity": 0.88, "tactic": "report_threat", "description": "Threatening to report"},
            ],
            "intimidation": [
                {"regex": r"\b(don't test me|watch (your )?self|careful what|shut up|be quiet)\b", "severity": 0.85, "tactic": "warning_language", "description": "Intimidating warnings"},
                {"regex": r"\b(i'm not (playing|joking)|serious|for real|dead serious)\b", "severity": 0.75, "tactic": "seriousness_emphasis", "description": "Emphasis of seriousness"},
                {"regex": r"\b(disrespect|respect me|show respect|attitude)\b", "severity": 0.8, "tactic": "respect_demand", "description": "Respect demands"},
                {"regex": r"\b(you'll see|wait till|just wait|remember this|you'll regret)\b", "severity": 0.8, "tactic": "future_punishment", "description": "Future punishment language"},
            ],
            "emotional_abuse": [
                {"regex": r"\b(stupid|idiot|worthless|useless|failure|weak)\b", "severity": 0.8, "tactic": "insult", "description": "Name-calling and insults"},
                {"regex": r"\b(you('re| are) (crazy|insane|paranoid|dramatic|emotional))\b", "severity": 0.75, "tactic": "gaslighting", "description": "Invalidation through gaslighting"},
                {"regex": r"\b(no one (likes|will ever|would)|i('m| am) the only one)\b", "severity": 0.8, "tactic": "social_diminishment", "description": "Social diminishment"},
                {"regex": r"\b(you (deserve|should|need) (this|me))\b", "severity": 0.75, "tactic": "deserved_blame", "description": "Blame and deservedness language"},
            ],
            "reproductive_coercion": [
                {"regex": r"\b(pregnant|pregnancy|baby|children|contraception|birth control)\b.*\b(stop|don't|can't|won't)\b", "severity": 0.92, "tactic": "pregnancy_control", "description": "Pregnancy/contraception control"},
                {"regex": r"\b(have (my )?baby|get pregnant|want kids)\b", "severity": 0.88, "tactic": "reproductive_demand", "description": "Reproductive demands"},
                {"regex": r"\b(abortion|miscarriage|child(ren)?).{0,40}\b(fault|your fault|your choice)\b", "severity": 0.9, "tactic": "reproductive_blame", "description": "Reproductive blame"},
            ],
            "financial_exploitation": [
                {"regex": r"\b(steal|theft|money|account|bank|credit card)\b", "severity": 0.88, "tactic": "direct_theft", "description": "Theft and stealing"},
                {"regex": r"\b(debt|loan|owe|borrow|pay back|interest)\b", "severity": 0.8, "tactic": "debt_trap", "description": "Debt trapping"},
                {"regex": r"\b(business|investment|buy|sell|property)\b.*\b(without|don't|permission)\b", "severity": 0.85, "tactic": "property_control", "description": "Property/business control"},
            ]
        }

    def analyze_message(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
        speaker_history: Optional[List[Dict]] = None
    ) -> CoercionAnalysis:
        """Analyze a single message for coercion tactics

        Args:
            text: Message text to analyze
            context_before: Previous message(s) for context
            context_after: Following message(s) for context
            speaker_history: Historical messages from this speaker

        Returns:
            CoercionAnalysis: Analysis results
        """
        analysis = CoercionAnalysis()

        if not text:
            return analysis

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []

            for pattern_tuple in patterns:
                # Pattern tuple: (regex, severity, tactic, description)
                regex, severity, tactic, description = pattern_tuple
                matches = list(regex.finditer(text))

                for match in matches:
                    power_control = self._calculate_power_control(category, severity)
                    pattern = CoercionPattern(
                        category="coercion",
                        subcategory=category,
                        tactic_type=tactic,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(match, text, severity),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description,
                        power_control_level=power_control
                    )
                    category_matches.append(pattern)
                    analysis.patterns_found.append(pattern)

            # Calculate category score
            if category_matches:
                total_severity = sum(p.severity * p.confidence for p in category_matches)
                avg_severity = total_severity / len(category_matches)
                analysis.category_scores[category] = avg_severity * self.CATEGORY_WEIGHTS[category]

        # Identify behavior patterns
        analysis.isolation_indicators = "isolation" in analysis.category_scores
        analysis.economic_control = "economic_control" in analysis.category_scores
        analysis.surveillance_detected = "surveillance" in analysis.category_scores
        analysis.threat_patterns = "threats" in analysis.category_scores
        analysis.intimidation_markers = "intimidation" in analysis.category_scores

        # Calculate overall metrics
        analysis.pattern_count = len(analysis.patterns_found)
        analysis.unique_patterns = len(set(p.pattern for p in analysis.patterns_found))

        # Identify tactic types
        tactic_types = set(p.tactic_type for p in analysis.patterns_found)
        analysis.tactic_types_detected = list(tactic_types)

        # Calculate power/control intensity
        analysis.power_control_intensity = sum(p.power_control_level * p.confidence for p in analysis.patterns_found) / max(1, len(analysis.patterns_found))

        # Identify Duluth Model wheels
        analysis.duluth_model_wheels = self._identify_duluth_wheels(analysis)

        # Calculate overall score
        if analysis.category_scores:
            analysis.overall_score = self._calculate_overall_score(analysis.category_scores)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_concern = max(analysis.category_scores, key=analysis.category_scores.get)

        return analysis

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for coercive patterns

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
            "coercion_pattern_summary": {},
            "duluth_analysis": {},
            "power_dynamic_assessment": {},
            "recommendations": [],
            "intervention_indicators": []
        }

        speaker_patterns = {}

        # Analyze each message
        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            context_before = messages[i-1].get('text', '') if i > 0 else ""
            context_after = messages[i+1].get('text', '') if i < len(messages) - 1 else ""
            speaker_history = [m for m in messages[:i] if m.get('sender') == sender]

            analysis = self.analyze_message(text, context_before, context_after, speaker_history)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "analysis": analysis,
                "risk_level": analysis.risk_level,
                "power_control_intensity": analysis.power_control_intensity
            }
            results["per_message_analysis"].append(msg_result)

            # Track by speaker
            if sender not in speaker_patterns:
                speaker_patterns[sender] = []
            speaker_patterns[sender].extend(analysis.patterns_found)

            # Track high-risk messages
            if analysis.risk_level in ["high", "critical"]:
                results["high_risk_messages"].append({
                    "index": i,
                    "sender": sender,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "risk_level": analysis.risk_level,
                    "tactics_detected": analysis.tactic_types_detected,
                    "power_control_intensity": analysis.power_control_intensity
                })

        # Analyze per speaker
        for speaker, patterns in speaker_patterns.items():
            speaker_analysis = self._analyze_speaker_patterns(patterns)
            results["per_speaker_analysis"][speaker] = speaker_analysis

        # Analyze progression
        results["conversation_progression"] = self._analyze_progression(results["per_message_analysis"])

        # Determine overall risk
        results["overall_risk"] = self._determine_conversation_risk(results)
        results["risk_trajectory"] = self._analyze_trajectory(results["per_message_analysis"])

        # Generate pattern summary
        results["coercion_pattern_summary"] = self._generate_pattern_summary(results["per_message_analysis"])

        # Analyze Duluth Model patterns
        results["duluth_analysis"] = self._analyze_duluth_patterns(results["per_message_analysis"])

        # Assess power dynamics
        results["power_dynamic_assessment"] = self._assess_power_dynamics(results["per_message_analysis"])

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Identify intervention indicators
        results["intervention_indicators"] = self._identify_intervention_needs(results)

        return results

    def _calculate_confidence(self, match: re.Match, text: str, severity: float) -> float:
        """Calculate confidence in pattern detection

        Args:
            match: Regex match object
            text: Full message text
            severity: Base severity of pattern

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

    def _calculate_power_control(self, category: str, severity: float) -> float:
        """Calculate power and control intensity

        Args:
            category: Tactic category
            severity: Base severity

        Returns:
            float: Power control level (0-1)
        """
        # Map category to power control intensity
        control_mapping = {
            "isolation": 0.85,
            "economic_control": 0.90,
            "surveillance": 0.80,
            "threats": 0.95,
            "intimidation": 0.85,
            "emotional_abuse": 0.75,
            "reproductive_coercion": 0.92,
            "financial_exploitation": 0.88
        }

        base_power = control_mapping.get(category, 0.7)
        return (base_power + severity) / 2

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall coercion risk score

        Args:
            category_scores: Scores by category

        Returns:
            float: Overall score (0-1)
        """
        if not category_scores:
            return 0.0

        # All scores are positive (no protective factors)
        total_score = sum(category_scores.values())
        avg_score = total_score / len(category_scores)

        # Multiple tactics amplify score
        num_tactics = len(category_scores)
        if num_tactics >= 4:
            avg_score *= 1.35
        elif num_tactics >= 3:
            avg_score *= 1.20
        elif num_tactics >= 2:
            avg_score *= 1.10

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

    def _identify_duluth_wheels(self, analysis: CoercionAnalysis) -> List[str]:
        """Identify Duluth Model power and control wheels

        Args:
            analysis: Analysis results

        Returns:
            List: Identified wheels
        """
        wheels = []

        if analysis.threat_patterns:
            wheels.append("threats")
        if analysis.intimidation_markers:
            wheels.append("intimidation")
        if analysis.isolation_indicators:
            wheels.append("isolation")
        if analysis.economic_control:
            wheels.append("economic_abuse")
        if analysis.surveillance_detected:
            wheels.append("monitoring_and_surveillance")
        if any(p.subcategory == "emotional_abuse" for p in analysis.patterns_found):
            wheels.append("emotional_abuse")

        return wheels

    def _analyze_speaker_patterns(self, patterns: List[CoercionPattern]) -> Dict[str, Any]:
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
                "tactics_used": [],
                "power_control_average": 0.0
            }

        category_counts = {}
        tactic_types = set()

        for pattern in patterns:
            category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1
            tactic_types.add(pattern.tactic_type)

        total_severity = sum(p.severity * p.confidence for p in patterns)
        avg_severity = total_severity / len(patterns) if patterns else 0
        power_control_avg = sum(p.power_control_level for p in patterns) / len(patterns) if patterns else 0

        return {
            "pattern_count": len(patterns),
            "unique_patterns": len(set(p.pattern for p in patterns)),
            "risk_level": self._determine_risk_level(avg_severity),
            "categories_used": list(category_counts.keys()),
            "tactics_used": list(tactic_types),
            "average_severity": avg_severity,
            "power_control_average": power_control_avg,
            "most_severe_pattern": max(patterns, key=lambda p: p.severity) if patterns else None
        }

    def _analyze_progression(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze coercion escalation over time

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Progression events
        """
        progression = []
        previous_risk = "low"
        previous_tactics = set()

        for i, msg_analysis in enumerate(message_analyses):
            analysis = msg_analysis["analysis"]
            current_risk = analysis.risk_level
            current_tactics = set(analysis.tactic_types_detected)

            # Track escalation
            risk_levels = ["low", "moderate", "high", "critical"]
            if risk_levels.index(current_risk) > risk_levels.index(previous_risk):
                progression.append({
                    "type": "risk_escalation",
                    "message_index": i,
                    "from": previous_risk,
                    "to": current_risk,
                    "sender": msg_analysis["sender"]
                })

            # New tactics introduced
            new_tactics = current_tactics - previous_tactics
            if new_tactics:
                progression.append({
                    "type": "new_tactic_introduced",
                    "message_index": i,
                    "tactics": list(new_tactics),
                    "sender": msg_analysis["sender"]
                })

            previous_risk = current_risk
            previous_tactics = current_tactics

        return progression

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
        risk_levels = {"low": 0.1, "moderate": 0.4, "high": 0.65, "critical": 0.85}

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
        """Determine overall conversation risk level

        Args:
            results: Analysis results

        Returns:
            str: Risk level
        """
        high_risk_count = len(results["high_risk_messages"])
        speaker_risks = [
            analysis["risk_level"]
            for analysis in results["per_speaker_analysis"].values()
        ]

        if "critical" in speaker_risks or high_risk_count >= 5:
            return "critical"
        elif "high" in speaker_risks or high_risk_count >= 3:
            return "high"
        elif "moderate" in speaker_risks or high_risk_count >= 1:
            return "moderate"
        else:
            return "low"

    def _generate_pattern_summary(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate summary of coercion patterns

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Pattern summary
        """
        all_patterns = []
        category_counts = {}
        tactic_counts = {}

        for msg in message_analyses:
            patterns = msg["analysis"].patterns_found
            all_patterns.extend(patterns)

            for pattern in patterns:
                category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1
                tactic_counts[pattern.tactic_type] = tactic_counts.get(pattern.tactic_type, 0) + 1

        return {
            "total_patterns": len(all_patterns),
            "unique_patterns": len(set(p.pattern for p in all_patterns)),
            "category_distribution": category_counts,
            "tactic_distribution": tactic_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "most_common_tactic": max(tactic_counts, key=tactic_counts.get) if tactic_counts else None,
            "average_severity": sum(p.severity for p in all_patterns) / len(all_patterns) if all_patterns else 0,
            "average_power_control": sum(p.power_control_level for p in all_patterns) / len(all_patterns) if all_patterns else 0
        }

    def _analyze_duluth_patterns(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns using Duluth Model framework

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Duluth Model analysis
        """
        all_wheels = []

        for msg in message_analyses:
            analysis = msg["analysis"]
            all_wheels.extend(analysis.duluth_model_wheels)

        wheel_counts = {}
        for wheel in all_wheels:
            wheel_counts[wheel] = wheel_counts.get(wheel, 0) + 1

        return {
            "wheels_present": list(set(all_wheels)),
            "wheel_frequency": wheel_counts,
            "total_wheels_detected": len(set(all_wheels)),
            "is_pattern_abuse": len(set(all_wheels)) >= 3
        }

    def _assess_power_dynamics(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Assess power dynamics in relationship

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Power dynamics assessment
        """
        power_scores_by_speaker = {}

        for msg in message_analyses:
            sender = msg["sender"]
            power_intensity = msg["analysis"].power_control_intensity

            if sender not in power_scores_by_speaker:
                power_scores_by_speaker[sender] = []
            power_scores_by_speaker[sender].append(power_intensity)

        # Calculate average power for each speaker
        speaker_power_avg = {
            speaker: sum(scores) / len(scores)
            for speaker, scores in power_scores_by_speaker.items()
        }

        # Identify dominant party
        if speaker_power_avg:
            dominant_speaker = max(speaker_power_avg, key=speaker_power_avg.get)
            dominant_power = speaker_power_avg[dominant_speaker]
        else:
            dominant_speaker = None
            dominant_power = 0.0

        return {
            "power_by_speaker": speaker_power_avg,
            "dominant_speaker": dominant_speaker,
            "dominant_power_level": dominant_power,
            "balanced_dynamic": max(speaker_power_avg.values() or [0]) < 0.4 if speaker_power_avg else True,
            "severe_imbalance": max(speaker_power_avg.values() or [0]) > 0.75 if speaker_power_avg else False
        }

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate intervention recommendations

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        risk = results["overall_risk"]

        if risk == "critical":
            recommendations.append("CRITICAL: Severe coercive and controlling behavior detected. Safety planning recommended.")
            recommendations.append("Contact domestic violence support services or law enforcement if there is immediate danger.")
            recommendations.append("Maintain safety by limiting contact with the abusive party.")

        elif risk == "high":
            recommendations.append("HIGH RISK: Multiple coercive tactics detected. Professional support strongly recommended.")
            recommendations.append("Consider contacting domestic violence hotlines or support organizations.")
            recommendations.append("Document all abusive communications for potential legal action.")

        elif risk == "moderate":
            recommendations.append("MODERATE RISK: Concerning controlling behaviors detected.")
            recommendations.append("Seek counseling or support to address unhealthy relationship dynamics.")
            recommendations.append("Set clear boundaries and consider professional mediation.")

        else:
            recommendations.append("LOW RISK: Minimal coercive patterns detected.")

        duluth = results["duluth_analysis"]
        if duluth["total_wheels_detected"] >= 3:
            recommendations.append("Pattern of abuse indicated. Professional intervention strongly recommended.")

        return recommendations

    def _identify_intervention_needs(self, results: Dict) -> List[str]:
        """Identify specific intervention needs

        Args:
            results: Analysis results

        Returns:
            List: Intervention indicators
        """
        indicators = []

        pattern_summary = results["coercion_pattern_summary"]
        if pattern_summary["most_common_category"] == "isolation":
            indicators.append("Isolation patterns require safety planning and support network building")
        if pattern_summary["most_common_category"] == "economic_control":
            indicators.append("Economic control requires financial counseling and independence planning")
        if "threats" in results["duluth_analysis"]["wheels_present"]:
            indicators.append("Threats present - safety planning essential")

        power_dynamics = results["power_dynamic_assessment"]
        if power_dynamics["severe_imbalance"]:
            indicators.append("Severe power imbalance - professional intervention urgent")

        return indicators
