"""
Grooming Detection Module for Message Processor
Implements research-backed pattern detection for grooming behaviors
with 6 categories and 20+ patterns based on academic literature.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroomingPattern:
    """Container for a grooming pattern match"""
    category: str
    subcategory: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""


@dataclass
class GroomingAnalysis:
    """Complete grooming analysis results"""
    patterns_found: List[GroomingPattern] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    pattern_count: int = 0
    unique_patterns: int = 0
    timeline_progression: List[str] = field(default_factory=list)


class GroomingDetector:
    """Detects grooming patterns in text messages"""

    # Grooming stage progression (typical order)
    GROOMING_STAGES = [
        "trust_building",
        "isolation",
        "boundary_testing",
        "normalization",
        "desensitization",
        "control"
    ]

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.6,
        "critical": 0.8
    }

    # Category weights (based on severity and research)
    CATEGORY_WEIGHTS = {
        "trust_building": 0.8,
        "isolation": 1.0,
        "boundary_testing": 1.2,
        "normalization": 1.0,
        "desensitization": 1.2,
        "control": 1.5
    }

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize grooming detector

        Args:
            patterns_file: Path to patterns JSON file
        """
        self.patterns = self._load_patterns(patterns_file)
        self.compiled_patterns = self._compile_patterns()

    def _load_patterns(self, patterns_file: Optional[str] = None) -> Dict:
        """Load grooming patterns from JSON file

        Args:
            patterns_file: Path to patterns file

        Returns:
            Dict: Patterns dictionary
        """
        if patterns_file is None:
            patterns_file = Path(__file__).parent / "patterns.json"
        else:
            patterns_file = Path(patterns_file)

        try:
            with open(patterns_file, 'r') as f:
                all_patterns = json.load(f)
                return all_patterns.get('grooming', {})
        except FileNotFoundError:
            logger.error(f"Patterns file not found: {patterns_file}")
            return self._get_default_patterns()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in patterns file: {e}")
            return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """Get default grooming patterns if file not found

        Returns:
            Dict: Default patterns
        """
        return {
            "trust_building": [
                {"regex": r"\byou can trust me\b", "severity": 0.6},
                {"regex": r"\bour (little )?secret\b", "severity": 0.8},
            ],
            "isolation": [
                {"regex": r"\b(they|nobody) (don't|doesn't) understand\b", "severity": 0.7},
                {"regex": r"\bdon't (tell|talk to) anyone\b", "severity": 0.8},
            ],
            "boundary_testing": [
                {"regex": r"\bno one (has to|needs to) know\b", "severity": 0.7},
                {"regex": r"\byou're mature enough\b", "severity": 0.6},
            ],
            "normalization": [
                {"regex": r"\bit's (normal|natural)\b", "severity": 0.5},
                {"regex": r"\beveryone does (it|this)\b", "severity": 0.5},
            ],
            "desensitization": [
                {"regex": r"\byou'll get used to it\b", "severity": 0.7},
                {"regex": r"\bdon't worry about\b", "severity": 0.4},
            ],
            "control": [
                {"regex": r"\byou (have to|must)\b", "severity": 0.5},
                {"regex": r"\byou owe me\b", "severity": 0.8},
            ]
        }

    def _compile_patterns(self) -> Dict[str, List[Tuple]]:
        """Compile regex patterns for efficiency

        Returns:
            Dict: Compiled patterns by category
        """
        compiled = {}

        for category, patterns in self.patterns.items():
            compiled[category] = []
            for pattern_dict in patterns:
                try:
                    regex = re.compile(pattern_dict['regex'], re.IGNORECASE)
                    severity = pattern_dict.get('severity', 0.5)
                    description = pattern_dict.get('description', '')
                    compiled[category].append((regex, severity, description))
                except re.error as e:
                    logger.error(f"Invalid regex in {category}: {pattern_dict['regex']} - {e}")

        return compiled

    def analyze_message(self, text: str, context_before: str = "", context_after: str = "") -> GroomingAnalysis:
        """Analyze a single message for grooming patterns

        Args:
            text: Message text to analyze
            context_before: Previous message(s) for context
            context_after: Following message(s) for context

        Returns:
            GroomingAnalysis: Analysis results
        """
        analysis = GroomingAnalysis()

        if not text:
            return analysis

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []

            for regex, severity, description in patterns:
                matches = list(regex.finditer(text))

                for match in matches:
                    pattern = GroomingPattern(
                        category="grooming",
                        subcategory=category,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(match, text, context_before, context_after),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description
                    )
                    category_matches.append(pattern)
                    analysis.patterns_found.append(pattern)

            # Calculate category score
            if category_matches:
                # Weighted average of severities
                total_severity = sum(p.severity * p.confidence for p in category_matches)
                avg_severity = total_severity / len(category_matches)
                analysis.category_scores[category] = avg_severity * self.CATEGORY_WEIGHTS[category]

        # Calculate overall metrics
        analysis.pattern_count = len(analysis.patterns_found)
        analysis.unique_patterns = len(set(p.pattern for p in analysis.patterns_found))

        # Calculate overall score
        if analysis.category_scores:
            analysis.overall_score = self._calculate_overall_score(analysis.category_scores)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_concern = max(analysis.category_scores, key=analysis.category_scores.get)

        # Determine stage progression
        analysis.timeline_progression = [cat for cat in self.GROOMING_STAGES if cat in analysis.category_scores]

        return analysis

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for grooming patterns and progression

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
            "pattern_summary": {},
            "recommendations": []
        }

        speaker_patterns = {}

        # Analyze each message with context
        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Get context
            context_before = messages[i-1].get('text', '') if i > 0 else ""
            context_after = messages[i+1].get('text', '') if i < len(messages) - 1 else ""

            # Analyze message
            analysis = self.analyze_message(text, context_before, context_after)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "analysis": analysis,
                "risk_level": analysis.risk_level
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
                    "primary_concern": analysis.primary_concern
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

        return results

    def _calculate_confidence(self, match: re.Match, text: str, context_before: str, context_after: str) -> float:
        """Calculate confidence in pattern detection

        Args:
            match: Regex match object
            text: Full message text
            context_before: Previous context
            context_after: Following context

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence

        # Adjust based on match position
        if match.start() == 0 or match.end() == len(text):
            confidence += 0.1  # Pattern at message boundary

        # Adjust based on context
        if context_before and any(word in context_before.lower() for word in ['no', 'stop', 'don\'t']):
            confidence -= 0.2  # Resistance in context

        if context_after and any(word in context_after.lower() for word in ['yes', 'ok', 'sure']):
            confidence += 0.1  # Acceptance in context

        # Adjust based on message length
        if len(text) < 20:
            confidence -= 0.1  # Very short message
        elif len(text) > 200:
            confidence += 0.1  # Longer, more elaborate message

        return max(0.1, min(1.0, confidence))

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall grooming risk score

        Args:
            category_scores: Scores by category

        Returns:
            float: Overall score (0-1)
        """
        if not category_scores:
            return 0.0

        # Weight later stages more heavily
        stage_multipliers = {
            "trust_building": 1.0,
            "isolation": 1.2,
            "boundary_testing": 1.4,
            "normalization": 1.6,
            "desensitization": 1.8,
            "control": 2.0
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for category, score in category_scores.items():
            multiplier = stage_multipliers.get(category, 1.0)
            weighted_sum += score * multiplier
            weight_total += multiplier

        overall = weighted_sum / weight_total if weight_total > 0 else 0

        # Apply non-linear scaling for multiple categories
        num_categories = len(category_scores)
        if num_categories >= 3:
            overall *= 1.2  # Multiple stages present
        if num_categories >= 5:
            overall *= 1.3  # Most stages present

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

    def _analyze_speaker_patterns(self, patterns: List[GroomingPattern]) -> Dict[str, Any]:
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
                "categories_used": [],
                "stage_progression": []
            }

        # Count patterns by category
        category_counts = {}
        for pattern in patterns:
            category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1

        # Calculate risk
        total_severity = sum(p.severity * p.confidence for p in patterns)
        avg_severity = total_severity / len(patterns) if patterns else 0

        # Check stage progression
        stages_present = [stage for stage in self.GROOMING_STAGES if stage in category_counts]

        return {
            "pattern_count": len(patterns),
            "unique_patterns": len(set(p.pattern for p in patterns)),
            "risk_level": self._determine_risk_level(avg_severity),
            "categories_used": list(category_counts.keys()),
            "category_counts": category_counts,
            "stage_progression": stages_present,
            "average_severity": avg_severity,
            "most_severe_pattern": max(patterns, key=lambda p: p.severity) if patterns else None
        }

    def _analyze_progression(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze grooming progression over time

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Progression events
        """
        progression = []
        current_stage = None
        stage_start = None

        for i, msg_analysis in enumerate(message_analyses):
            analysis = msg_analysis["analysis"]

            if analysis.primary_concern:
                if current_stage != analysis.primary_concern:
                    if current_stage:
                        # Stage transition
                        progression.append({
                            "type": "stage_transition",
                            "from_stage": current_stage,
                            "to_stage": analysis.primary_concern,
                            "message_index": i,
                            "sender": msg_analysis["sender"]
                        })

                    current_stage = analysis.primary_concern
                    stage_start = i

                # Check for stage escalation
                if current_stage and self.GROOMING_STAGES.index(current_stage) >= 3:
                    if i - stage_start == 0:  # New dangerous stage
                        progression.append({
                            "type": "escalation",
                            "stage": current_stage,
                            "message_index": i,
                            "sender": msg_analysis["sender"],
                            "severity": "high"
                        })

        return progression

    def _analyze_trajectory(self, message_analyses: List[Dict]) -> str:
        """Analyze risk trajectory over conversation

        Args:
            message_analyses: List of per-message analyses

        Returns:
            str: Trajectory (stable, escalating, de-escalating)
        """
        if len(message_analyses) < 3:
            return "stable"

        # Get risk scores over time
        risk_scores = []
        for msg in message_analyses:
            score = msg["analysis"].overall_score
            risk_scores.append(score)

        # Analyze trend (simple linear regression concept)
        first_third = sum(risk_scores[:len(risk_scores)//3]) / (len(risk_scores)//3)
        last_third = sum(risk_scores[-len(risk_scores)//3:]) / (len(risk_scores)//3)

        if last_third > first_third * 1.3:
            return "escalating"
        elif last_third < first_third * 0.7:
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
        # Check high-risk indicators
        high_risk_count = len(results["high_risk_messages"])
        trajectory = results["risk_trajectory"]

        # Check speaker risks
        speaker_risks = [analysis["risk_level"] for analysis in results["per_speaker_analysis"].values()]

        if "critical" in speaker_risks or high_risk_count >= 5:
            return "critical"
        elif "high" in speaker_risks or (high_risk_count >= 3 and trajectory == "escalating"):
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
            "average_severity": sum(p.severity for p in all_patterns) / len(all_patterns) if all_patterns else 0
        }

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate safety recommendations based on analysis

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        risk = results["overall_risk"]
        trajectory = results["risk_trajectory"]

        # Base recommendations by risk level
        if risk == "critical":
            recommendations.append("CRITICAL: Immediate intervention recommended. Consider involving trusted adults or authorities.")
            recommendations.append("Document all concerning messages and maintain evidence.")
            recommendations.append("Avoid being alone with the concerning individual.")

        elif risk == "high":
            recommendations.append("HIGH RISK: Significant grooming indicators detected. Seek support from trusted individuals.")
            recommendations.append("Consider limiting or ending contact with the concerning party.")
            recommendations.append("Keep records of all interactions for potential future reference.")

        elif risk == "moderate":
            recommendations.append("MODERATE RISK: Some concerning patterns detected. Maintain awareness and set clear boundaries.")
            recommendations.append("Consider discussing concerns with a trusted friend or counselor.")

        else:
            recommendations.append("LOW RISK: No significant grooming patterns detected in current conversation.")

        # Trajectory-based recommendations
        if trajectory == "escalating":
            recommendations.append("WARNING: Risk level is increasing over time. Early intervention recommended.")
        elif trajectory == "de-escalating":
            recommendations.append("POSITIVE: Risk indicators are decreasing, but maintain vigilance.")

        # Stage-specific recommendations
        pattern_summary = results["pattern_summary"]
        if pattern_summary["most_common_category"] == "isolation":
            recommendations.append("Maintain connections with friends and family. Resist attempts at isolation.")
        elif pattern_summary["most_common_category"] == "control":
            recommendations.append("Assert your autonomy. You have the right to make your own decisions.")
        elif pattern_summary["most_common_category"] == "boundary_testing":
            recommendations.append("Trust your instincts. If something feels uncomfortable, it's okay to say no.")

        return recommendations