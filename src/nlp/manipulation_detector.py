"""
Manipulation and Gaslighting Detection Module
Identifies psychological manipulation tactics including gaslighting, blame shifting,
emotional invalidation, guilt tripping, love bombing, and threats.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManipulationPattern:
    """Container for manipulation pattern match"""
    category: str
    subcategory: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""
    manipulation_type: str = ""  # gaslighting, guilt, threat, etc.


@dataclass
class ManipulationAnalysis:
    """Complete manipulation analysis results"""
    patterns_found: List[ManipulationPattern] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    manipulation_types: Dict[str, int] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"
    primary_tactic: Optional[str] = None
    emotional_harm_score: float = 0.0
    coercion_score: float = 0.0
    reality_distortion_score: float = 0.0


class ManipulationDetector:
    """Detects manipulation and gaslighting patterns"""

    # Manipulation tactics severity weights
    TACTIC_WEIGHTS = {
        "gaslighting": 1.5,      # Reality distortion is most harmful
        "threats": 1.4,           # Direct threats are severe
        "blame_shifting": 1.2,    # Responsibility manipulation
        "guilt_tripping": 1.1,    # Emotional manipulation
        "emotional_invalidation": 1.0,
        "love_bombing": 0.8       # Can be positive or manipulative
    }

    # Risk thresholds
    RISK_THRESHOLDS = {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.6,
        "critical": 0.8
    }

    # Emotional harm indicators
    EMOTIONAL_HARM_KEYWORDS = [
        "crazy", "insane", "paranoid", "stupid", "worthless",
        "pathetic", "useless", "dramatic", "ridiculous", "baby"
    ]

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize manipulation detector

        Args:
            patterns_file: Path to patterns JSON file
        """
        self.patterns = self._load_patterns(patterns_file)
        self.compiled_patterns = self._compile_patterns()

    def _load_patterns(self, patterns_file: Optional[str] = None) -> Dict:
        """Load manipulation patterns from JSON file"""
        if patterns_file is None:
            patterns_file = Path(__file__).parent / "patterns.json"
        else:
            patterns_file = Path(patterns_file)

        try:
            with open(patterns_file, 'r') as f:
                all_patterns = json.load(f)
                return all_patterns.get('manipulation', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading patterns: {e}")
            return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """Get default manipulation patterns"""
        return {
            "gaslighting": [
                {"regex": r"\byou('re| are) (crazy|insane)\b", "severity": 0.9},
                {"regex": r"\bthat never happened\b", "severity": 0.8},
            ],
            "blame_shifting": [
                {"regex": r"\byou made me\b", "severity": 0.7},
                {"regex": r"\bit's your fault\b", "severity": 0.8},
            ],
            "guilt_tripping": [
                {"regex": r"\bafter everything i've done\b", "severity": 0.6},
                {"regex": r"\byou're so ungrateful\b", "severity": 0.7},
            ],
            "threats": [
                {"regex": r"\bi'll (hurt|harm|kill)\b", "severity": 0.95},
                {"regex": r"\byou'll regret\b", "severity": 0.8},
            ]
        }

    def _compile_patterns(self) -> Dict[str, List[Tuple]]:
        """Compile regex patterns for efficiency"""
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

    def analyze_message(self, text: str, context: Optional[Dict] = None) -> ManipulationAnalysis:
        """Analyze message for manipulation patterns

        Args:
            text: Message text
            context: Optional context (speaker, previous messages, etc.)

        Returns:
            ManipulationAnalysis: Analysis results
        """
        analysis = ManipulationAnalysis()

        if not text:
            return analysis

        # Check each manipulation category
        for category, patterns in self.compiled_patterns.items():
            for regex, severity, description in patterns:
                matches = list(regex.finditer(text))

                for match in matches:
                    pattern = ManipulationPattern(
                        category="manipulation",
                        subcategory=category,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(match, text, category),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description,
                        manipulation_type=category
                    )
                    analysis.patterns_found.append(pattern)

                    # Track manipulation types
                    analysis.manipulation_types[category] = \
                        analysis.manipulation_types.get(category, 0) + 1

        # Calculate scores
        self._calculate_scores(analysis)

        # Assess emotional harm
        analysis.emotional_harm_score = self._assess_emotional_harm(text, analysis)

        # Assess coercion level
        analysis.coercion_score = self._assess_coercion(analysis)

        # Assess reality distortion
        analysis.reality_distortion_score = self._assess_reality_distortion(analysis)

        return analysis

    def _calculate_confidence(self, match: re.Match, text: str, category: str) -> float:
        """Calculate confidence in pattern detection

        Args:
            match: Regex match
            text: Full text
            category: Manipulation category

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence

        # Adjust for category-specific factors
        if category == "gaslighting":
            # Gaslighting is more confident with question marks
            if "?" in text:
                confidence += 0.1
            # Multiple negations increase confidence
            if text.count("not") > 1 or text.count("never") > 1:
                confidence += 0.15

        elif category == "threats":
            # Threats with future tense are more credible
            if any(word in text.lower() for word in ["will", "going to", "gonna"]):
                confidence += 0.2

        elif category == "guilt_tripping":
            # First person pronouns increase confidence
            if text.lower().count("i") > 2:
                confidence += 0.1

        # Adjust for intensity markers
        if any(word in text.lower() for word in ["always", "never", "every", "completely"]):
            confidence += 0.1

        # Caps lock indicates emotional intensity
        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_scores(self, analysis: ManipulationAnalysis):
        """Calculate various scores from found patterns"""
        if not analysis.patterns_found:
            return

        # Category scores
        category_patterns = {}
        for pattern in analysis.patterns_found:
            if pattern.subcategory not in category_patterns:
                category_patterns[pattern.subcategory] = []
            category_patterns[pattern.subcategory].append(pattern)

        for category, patterns in category_patterns.items():
            # Weighted average of severity and confidence
            total_score = sum(p.severity * p.confidence for p in patterns)
            avg_score = total_score / len(patterns)
            weight = self.TACTIC_WEIGHTS.get(category, 1.0)
            analysis.category_scores[category] = avg_score * weight

        # Overall score
        if analysis.category_scores:
            analysis.overall_score = sum(analysis.category_scores.values()) / len(analysis.category_scores)

            # Boost score for multiple tactics (compound manipulation)
            if len(analysis.category_scores) >= 3:
                analysis.overall_score *= 1.3

            analysis.overall_score = min(1.0, analysis.overall_score)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_tactic = max(analysis.category_scores, key=analysis.category_scores.get)

    def _assess_emotional_harm(self, text: str, analysis: ManipulationAnalysis) -> float:
        """Assess potential emotional harm level

        Args:
            text: Message text
            analysis: Current analysis

        Returns:
            float: Emotional harm score (0-1)
        """
        harm_score = 0.0
        text_lower = text.lower()

        # Check for harmful keywords
        harm_keyword_count = sum(1 for keyword in self.EMOTIONAL_HARM_KEYWORDS
                                 if keyword in text_lower)
        if harm_keyword_count > 0:
            harm_score += min(0.3 * harm_keyword_count, 0.6)

        # Gaslighting causes significant emotional harm
        if "gaslighting" in analysis.manipulation_types:
            harm_score += 0.3

        # Emotional invalidation
        if "emotional_invalidation" in analysis.manipulation_types:
            harm_score += 0.2

        # Personal attacks
        if any(phrase in text_lower for phrase in ["you always", "you never", "you're the problem"]):
            harm_score += 0.2

        return min(1.0, harm_score)

    def _assess_coercion(self, analysis: ManipulationAnalysis) -> float:
        """Assess coercion level

        Args:
            analysis: Current analysis

        Returns:
            float: Coercion score (0-1)
        """
        coercion_score = 0.0

        # Threats are highly coercive
        if "threats" in analysis.manipulation_types:
            threat_count = analysis.manipulation_types["threats"]
            coercion_score += min(0.3 * threat_count, 0.6)

        # Guilt tripping is moderately coercive
        if "guilt_tripping" in analysis.manipulation_types:
            coercion_score += 0.2

        # Blame shifting can be coercive
        if "blame_shifting" in analysis.manipulation_types:
            coercion_score += 0.15

        return min(1.0, coercion_score)

    def _assess_reality_distortion(self, analysis: ManipulationAnalysis) -> float:
        """Assess reality distortion level (gaslighting severity)

        Args:
            analysis: Current analysis

        Returns:
            float: Reality distortion score (0-1)
        """
        distortion_score = 0.0

        # Gaslighting is primary reality distortion
        if "gaslighting" in analysis.manipulation_types:
            gaslight_count = analysis.manipulation_types["gaslighting"]
            distortion_score += min(0.4 * gaslight_count, 0.8)

        # Blame shifting can distort responsibility perception
        if "blame_shifting" in analysis.manipulation_types:
            distortion_score += 0.15

        # Check for memory/perception attacks
        for pattern in analysis.patterns_found:
            if any(word in pattern.matched_text.lower()
                  for word in ["remember", "imagining", "confused", "wrong"]):
                distortion_score += 0.1
                break

        return min(1.0, distortion_score)

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        for level in ["critical", "high", "moderate", "low"]:
            if score >= self.RISK_THRESHOLDS[level]:
                return level
        return "low"

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for manipulation patterns

        Args:
            messages: List of messages with 'text', 'sender' keys

        Returns:
            Dict: Comprehensive analysis
        """
        results = {
            "per_message_analysis": [],
            "per_speaker_analysis": {},
            "manipulation_timeline": [],
            "escalation_points": [],
            "victim_responses": [],
            "overall_risk": "low",
            "dominant_tactics": [],
            "recommendations": []
        }

        speaker_tactics = {}
        victim_responses = {}

        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Analyze message
            analysis = self.analyze_message(text)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "analysis": analysis,
                "risk_level": analysis.risk_level
            }
            results["per_message_analysis"].append(msg_result)

            # Track by speaker
            if sender not in speaker_tactics:
                speaker_tactics[sender] = {
                    "tactics_used": {},
                    "total_patterns": 0,
                    "emotional_harm_caused": 0,
                    "coercion_level": 0
                }

            # Update speaker profile
            if analysis.patterns_found:
                speaker_tactics[sender]["total_patterns"] += len(analysis.patterns_found)
                speaker_tactics[sender]["emotional_harm_caused"] = max(
                    speaker_tactics[sender]["emotional_harm_caused"],
                    analysis.emotional_harm_score
                )
                speaker_tactics[sender]["coercion_level"] = max(
                    speaker_tactics[sender]["coercion_level"],
                    analysis.coercion_score
                )

                for tactic in analysis.manipulation_types:
                    speaker_tactics[sender]["tactics_used"][tactic] = \
                        speaker_tactics[sender]["tactics_used"].get(tactic, 0) + 1

            # Track victim responses (simplified - looks for submission patterns)
            if i > 0 and analysis.risk_level in ["high", "critical"]:
                prev_sender = messages[i-1].get('sender', 'Unknown')
                if prev_sender != sender:
                    # Check for submission/compliance in response
                    if any(phrase in text.lower() for phrase in
                          ["sorry", "you're right", "my fault", "i was wrong", "okay"]):
                        if prev_sender not in victim_responses:
                            victim_responses[prev_sender] = []
                        victim_responses[prev_sender].append({
                            "message_index": i,
                            "response_type": "submission",
                            "to_manipulation": analysis.primary_tactic
                        })

            # Track escalation points
            if analysis.risk_level in ["high", "critical"]:
                results["escalation_points"].append({
                    "index": i,
                    "sender": sender,
                    "tactic": analysis.primary_tactic,
                    "severity": analysis.overall_score
                })

        # Compile speaker analysis
        results["per_speaker_analysis"] = speaker_tactics
        results["victim_responses"] = victim_responses

        # Determine dominant tactics
        all_tactics = {}
        for speaker_data in speaker_tactics.values():
            for tactic, count in speaker_data["tactics_used"].items():
                all_tactics[tactic] = all_tactics.get(tactic, 0) + count

        if all_tactics:
            sorted_tactics = sorted(all_tactics.items(), key=lambda x: x[1], reverse=True)
            results["dominant_tactics"] = [t[0] for t in sorted_tactics[:3]]

        # Overall risk assessment
        results["overall_risk"] = self._assess_conversation_risk(results)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _assess_conversation_risk(self, results: Dict) -> str:
        """Assess overall conversation risk"""
        # Check for high-risk patterns
        escalation_count = len(results["escalation_points"])

        # Check speaker profiles
        max_harm = 0
        max_coercion = 0
        for speaker_data in results["per_speaker_analysis"].values():
            max_harm = max(max_harm, speaker_data["emotional_harm_caused"])
            max_coercion = max(max_coercion, speaker_data["coercion_level"])

        # Determine risk level
        if max_harm > 0.7 or max_coercion > 0.7 or escalation_count > 5:
            return "critical"
        elif max_harm > 0.5 or max_coercion > 0.5 or escalation_count > 3:
            return "high"
        elif max_harm > 0.3 or max_coercion > 0.3 or escalation_count > 1:
            return "moderate"
        else:
            return "low"

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        risk = results["overall_risk"]

        if risk == "critical":
            recommendations.append("CRITICAL: Severe manipulation detected. Professional support strongly recommended.")
            recommendations.append("Document all interactions for potential legal or therapeutic use.")
            recommendations.append("Consider immediate safety planning if threats are present.")

        elif risk == "high":
            recommendations.append("HIGH: Significant manipulation patterns detected.")
            recommendations.append("Consider seeking support from a counselor or therapist.")
            recommendations.append("Practice setting and maintaining firm boundaries.")

        elif risk == "moderate":
            recommendations.append("MODERATE: Some manipulation tactics observed.")
            recommendations.append("Learn to recognize and respond to manipulation tactics.")
            recommendations.append("Consider limiting exposure to manipulative individuals.")

        # Tactic-specific recommendations
        if "gaslighting" in results["dominant_tactics"]:
            recommendations.append("Keep a journal to validate your experiences and memories.")
            recommendations.append("Seek external validation from trusted friends or professionals.")

        if "threats" in results["dominant_tactics"]:
            recommendations.append("Take all threats seriously. Consider involving authorities if necessary.")
            recommendations.append("Create a safety plan with trusted contacts and safe locations.")

        if "guilt_tripping" in results["dominant_tactics"]:
            recommendations.append("Remember: You are not responsible for others' emotional reactions.")
            recommendations.append("Practice self-compassion and boundary setting.")

        return recommendations