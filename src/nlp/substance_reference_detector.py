"""
Substance Reference Detection Module
Detects drug/alcohol mentions, intoxication markers, substance seeking behavior,
and addiction-related language patterns.
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
class SubstancePattern:
    """Container for a substance reference pattern match"""
    category: str
    subcategory: str
    substance_type: str
    pattern: str
    matched_text: str
    severity: float
    confidence: float
    start_pos: int
    end_pos: int
    description: str = ""
    implies_active_use: bool = False


@dataclass
class SubstanceAnalysis:
    """Complete substance reference analysis results"""
    patterns_found: List[SubstancePattern] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    pattern_count: int = 0
    unique_patterns: int = 0
    substances_mentioned: List[str] = field(default_factory=list)
    substance_types: Dict[str, int] = field(default_factory=dict)
    active_use_indicators: bool = False
    seeking_behavior: bool = False
    recovery_indicators: bool = False
    relapse_risk: bool = False
    peer_pressure_detected: bool = False


class SubstanceReferenceDetector:
    """Detects substance-related references and behaviors"""

    # Risk categories
    SUBSTANCE_CATEGORIES = [
        "drug_mention",
        "alcohol_mention",
        "intoxication_markers",
        "substance_seeking",
        "peer_pressure",
        "recovery_language",
        "relapse_indicators",
        "culture_slang"
    ]

    # Substance types
    SUBSTANCE_TYPES = {
        "alcohol": ["alcohol", "beer", "wine", "liquor", "drunk", "drinking"],
        "cannabis": ["weed", "marijuana", "pot", "hash", "joint", "blunt"],
        "stimulants": ["cocaine", "meth", "amphetamine", "speed", "coke", "crack"],
        "opioids": ["heroin", "fentanyl", "oxycontin", "pills", "h", "dope"],
        "hallucinogens": ["lsd", "acid", "shrooms", "mushrooms", "ecstasy", "mdma"],
        "depressants": ["xanax", "valium", "benzodiazepine", "pills", "bars"],
        "other": ["drugs", "substance", "high", "trip", "fix"]
    }

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.15,
        "moderate": 0.35,
        "high": 0.60,
        "critical": 0.80
    }

    # Category weights
    CATEGORY_WEIGHTS = {
        "drug_mention": 0.9,
        "alcohol_mention": 0.6,
        "intoxication_markers": 1.0,
        "substance_seeking": 1.4,
        "peer_pressure": 1.2,
        "recovery_language": -0.8,  # Protective factor
        "relapse_indicators": 1.5,
        "culture_slang": 0.7
    }

    def __init__(self, patterns_file: Optional[str] = None):
        """Initialize substance detector with cached patterns

        Args:
            patterns_file: Path to patterns JSON file
        """
        cache = get_cache()
        cache_key = f"substance_patterns_{patterns_file or 'default'}"

        # Load patterns from cache (or file if not cached)
        self.patterns = cache.get_or_load(
            cache_key,
            load_patterns_file,
            patterns_file,
            'substance'
        )

        # If patterns are empty, use defaults
        if not self.patterns:
            self.patterns = self._get_default_patterns()

        # Compile and cache patterns
        compiled_cache_key = f"substance_compiled_{patterns_file or 'default'}"
        self.compiled_patterns = cache.get_or_load(
            compiled_cache_key,
            compile_regex_patterns,
            self.patterns
        )

    def _get_default_patterns(self) -> Dict:
        """Get default substance reference patterns

        Returns:
            Dict: Default patterns
        """
        return {
            "drug_mention": [
                {"regex": r"\b(drug|drugs|narcotics|controlled substance)\b", "severity": 0.7, "substance": "general", "description": "General drug references"},
                {"regex": r"\b(cocaine|coke|crack|powder|white)\b", "severity": 0.85, "substance": "stimulants", "description": "Cocaine references"},
                {"regex": r"\b(heroin|h|dope|junk|china white)\b", "severity": 0.9, "substance": "opioids", "description": "Heroin references"},
                {"regex": r"\b(meth|methamphetamine|crystal|ice|glass)\b", "severity": 0.88, "substance": "stimulants", "description": "Methamphetamine references"},
                {"regex": r"\b(lsd|acid|tabs|blotter)\b", "severity": 0.75, "substance": "hallucinogens", "description": "LSD references"},
                {"regex": r"\b(ecstasy|mdma|molly|e)\b", "severity": 0.78, "substance": "hallucinogens", "description": "MDMA/Ecstasy references"},
                {"regex": r"\b(marijuana|weed|pot|grass|joint|blunt)\b", "severity": 0.6, "substance": "cannabis", "description": "Cannabis references"},
                {"regex": r"\b(xanax|valium|benzos|bars|pills)\b", "severity": 0.75, "substance": "depressants", "description": "Depressant references"},
            ],
            "alcohol_mention": [
                {"regex": r"\b(alcohol|drinking|drank|drunk)\b", "severity": 0.5, "substance": "alcohol", "description": "General alcohol references"},
                {"regex": r"\b(beer|wine|liquor|vodka|whiskey|rum)\b", "severity": 0.4, "substance": "alcohol", "description": "Specific alcohol beverages"},
                {"regex": r"\b(shots|drinking games|pregame|party)\b", "severity": 0.5, "substance": "alcohol", "description": "Drinking context"},
                {"regex": r"\b(binge|blackout|wasted|hammered|smashed)\b", "severity": 0.6, "substance": "alcohol", "description": "Heavy drinking indicators"},
            ],
            "intoxication_markers": [
                {"regex": r"\b(high|stoned|buzzed|lit|wasted|trashed)\b", "severity": 0.7, "substance": "general", "description": "Intoxication state"},
                {"regex": r"\b(can't think|can't feel|numb|floating|trippy)\b", "severity": 0.75, "substance": "general", "description": "Intoxication effects"},
                {"regex": r"\b(coming down|crashing|crash)\b", "severity": 0.8, "substance": "general", "description": "Drug come-down"},
                {"regex": r"\b(feeling (it|good)|this stuff (is|works))\b", "severity": 0.65, "substance": "general", "description": "Substance effect affirmation"},
            ],
            "substance_seeking": [
                {"regex": r"\b(where (can i|to) get|find|buy|dealer|connect)\b", "severity": 0.9, "substance": "general", "description": "Seeking substances"},
                {"regex": r"\b(hook (me )?up|score|plug|source)\b", "severity": 0.85, "substance": "general", "description": "Seeking connection language"},
                {"regex": r"\b(anyone know|know someone|who has|who knows)\b.*\b(drugs?|dealer|supplier)\b", "severity": 0.88, "substance": "general", "description": "Asking for suppliers"},
                {"regex": r"\b(need (some|a)|want to|looking for)\b.*\b(drugs?|weed|coke|heroin|meth)\b", "severity": 0.9, "substance": "general", "description": "Explicit seeking behavior"},
            ],
            "peer_pressure": [
                {"regex": r"\b(come on|just (try|one)|don't be|be cool|everyone does)\b.*\b(drugs?|weed|drink)\b", "severity": 0.8, "substance": "general", "description": "Direct pressure to use"},
                {"regex": r"\b(you're (boring|lame|square)|no fun)\b", "severity": 0.7, "substance": "general", "description": "Social pressure language"},
                {"regex": r"\b(if you (don't|won't)|you'd better)\b.*\b(try|smoke|drink)\b", "severity": 0.85, "substance": "general", "description": "Conditional pressure"},
                {"regex": r"\b(everyone (is|will)|nobody's|all the (cool|smart))\b.*\b(using|doing|smoking)\b", "severity": 0.75, "substance": "general", "description": "Normalization pressure"},
            ],
            "recovery_language": [
                {"regex": r"\b(sober|clean|recovery|in recovery|quit|stopped using)\b", "severity": -0.7, "substance": "general", "description": "Recovery commitment"},
                {"regex": r"\b(aa|na|12 step|sponsor|support group|treatment)\b", "severity": -0.8, "substance": "general", "description": "Recovery program involvement"},
                {"regex": r"\b(working (on|through)|trying to quit|getting help|seeking treatment)\b", "severity": -0.6, "substance": "general", "description": "Recovery efforts"},
                {"regex": r"\b(proud|stronger|better|healthier|clearer)\b.*\b(since|without)\b", "severity": -0.7, "substance": "general", "description": "Recovery achievements"},
            ],
            "relapse_indicators": [
                {"regex": r"\b(relapse|relapsed|using again|back to|started again)\b", "severity": 0.9, "substance": "general", "description": "Direct relapse acknowledgment"},
                {"regex": r"\b(can't (stop|help)|why do i|it's back)\b", "severity": 0.8, "substance": "general", "description": "Loss of control markers"},
                {"regex": r"\b(breaking my (streak|sobriety)|so close|almost made it)\b", "severity": 0.85, "substance": "general", "description": "Near-relapse or recent use"},
                {"regex": r"\b(thinking about it|can't stop thinking|cravings|temptation)\b", "severity": 0.75, "substance": "general", "description": "Relapse triggers and cravings"},
            ],
            "culture_slang": [
                {"regex": r"\b(slanging|dealing|pushing|traffic(king)?)\b", "severity": 0.85, "substance": "general", "description": "Drug distribution language"},
                {"regex": r"\b(8 ball|quarter|half|key|brick|g)\b", "severity": 0.8, "substance": "general", "description": "Drug quantity slang"},
                {"regex": r"\b(clean|dirty|tested)\b", "severity": 0.65, "substance": "general", "description": "Drug quality testing language"},
                {"regex": r"\b(trip|roll|fixated on|chasing)\b", "severity": 0.6, "substance": "general", "description": "Drug experience slang"},
            ]
        }

    def analyze_message(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
        speaker_history: Optional[List[Dict]] = None
    ) -> SubstanceAnalysis:
        """Analyze a single message for substance references

        Args:
            text: Message text to analyze
            context_before: Previous message(s) for context
            context_after: Following message(s) for context
            speaker_history: Historical messages from this speaker

        Returns:
            SubstanceAnalysis: Analysis results
        """
        analysis = SubstanceAnalysis()

        if not text:
            return analysis

        # Track detected substances
        detected_substances = set()

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []

            for pattern_tuple in patterns:
                # Pattern tuple: (regex, severity, substance, description)
                regex, severity, substance, description = pattern_tuple
                matches = list(regex.finditer(text))

                for match in matches:
                    implies_use = category in ["intoxication_markers", "substance_seeking"]
                    pattern = SubstancePattern(
                        category="substance",
                        subcategory=category,
                        substance_type=substance,
                        pattern=regex.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        confidence=self._calculate_confidence(match, text, severity),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        description=description,
                        implies_active_use=implies_use
                    )
                    category_matches.append(pattern)
                    analysis.patterns_found.append(pattern)
                    detected_substances.add(substance)

            # Calculate category score
            if category_matches:
                total_severity = sum(p.severity * p.confidence for p in category_matches)
                avg_severity = total_severity / len(category_matches)
                analysis.category_scores[category] = avg_severity * self.CATEGORY_WEIGHTS[category]

        # Set detected substances
        analysis.substances_mentioned = list(detected_substances)

        # Count substances by type
        for pattern in analysis.patterns_found:
            substance = pattern.substance_type
            analysis.substance_types[substance] = analysis.substance_types.get(substance, 0) + 1

        # Identify behavior patterns
        analysis.active_use_indicators = any(
            p.implies_active_use for p in analysis.patterns_found
        )
        analysis.seeking_behavior = "substance_seeking" in analysis.category_scores
        analysis.recovery_indicators = "recovery_language" in analysis.category_scores
        analysis.relapse_risk = "relapse_indicators" in analysis.category_scores
        analysis.peer_pressure_detected = "peer_pressure" in analysis.category_scores

        # Calculate overall metrics
        analysis.pattern_count = len(analysis.patterns_found)
        analysis.unique_patterns = len(set(p.pattern for p in analysis.patterns_found))

        # Calculate overall score
        if analysis.category_scores:
            analysis.overall_score = self._calculate_overall_score(analysis.category_scores)
            analysis.risk_level = self._determine_risk_level(analysis.overall_score)
            analysis.primary_concern = max(analysis.category_scores, key=analysis.category_scores.get)

        return analysis

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze full conversation for substance-related patterns

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
            "substances_detected": set(),
            "seeking_incidents": [],
            "pattern_summary": {},
            "recommendations": [],
            "concerning_behaviors": []
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
                "risk_level": analysis.risk_level
            }
            results["per_message_analysis"].append(msg_result)

            # Track substances
            results["substances_detected"].update(analysis.substances_mentioned)

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
                    "substances": analysis.substances_mentioned,
                    "concerning_behaviors": self._identify_behaviors(analysis)
                })

            # Track seeking incidents
            if analysis.seeking_behavior:
                results["seeking_incidents"].append({
                    "index": i,
                    "sender": sender,
                    "substances": analysis.substances_mentioned,
                    "severity": analysis.overall_score
                })

            # Track concerning behaviors
            behaviors = self._identify_behaviors(analysis)
            if behaviors:
                results["concerning_behaviors"].extend(behaviors)

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
        results["pattern_summary"] = self._generate_pattern_summary(results["per_message_analysis"])

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Convert set to list for JSON serialization
        results["substances_detected"] = list(results["substances_detected"])

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
        confidence = 0.7 if severity >= 0 else 0.8

        # Adjust based on message length
        if len(text) < 20:
            confidence -= 0.1
        elif len(text) > 200:
            confidence += 0.05

        # Position boost
        if match.start() == 0:
            confidence += 0.05

        return max(0.1, min(1.0, confidence))

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall substance risk score

        Args:
            category_scores: Scores by category

        Returns:
            float: Overall score (0-1)
        """
        if not category_scores:
            return 0.0

        positive_scores = {k: v for k, v in category_scores.items() if v > 0}
        negative_scores = {k: v for k, v in category_scores.items() if v <= 0}

        if not positive_scores:
            return 0.0

        risk_sum = sum(positive_scores.values())
        protection_sum = abs(sum(negative_scores.values()))

        overall = risk_sum / (risk_sum + protection_sum + 1)

        # Multiple risk categories amplify score
        high_risk_count = sum(1 for v in positive_scores.values() if v > 0.5)
        if high_risk_count >= 3:
            overall *= 1.25
        elif high_risk_count >= 2:
            overall *= 1.1

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

    def _analyze_speaker_patterns(self, patterns: List[SubstancePattern]) -> Dict[str, Any]:
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
                "substances_mentioned": [],
                "active_use": False
            }

        category_counts = {}
        substances = set()

        for pattern in patterns:
            category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1
            substances.add(pattern.substance_type)

        total_severity = sum(p.severity * p.confidence for p in patterns)
        avg_severity = total_severity / len(patterns) if patterns else 0
        active_use = any(p.implies_active_use for p in patterns)

        return {
            "pattern_count": len(patterns),
            "unique_patterns": len(set(p.pattern for p in patterns)),
            "risk_level": self._determine_risk_level(avg_severity),
            "categories_detected": list(category_counts.keys()),
            "substances_mentioned": list(substances),
            "active_use": active_use,
            "average_severity": avg_severity,
            "most_severe_pattern": max(patterns, key=lambda p: p.severity) if patterns else None
        }

    def _analyze_progression(self, message_analyses: List[Dict]) -> List[Dict]:
        """Analyze substance reference progression over time

        Args:
            message_analyses: List of per-message analyses

        Returns:
            List: Progression events
        """
        progression = []
        previous_substances = set()

        for i, msg_analysis in enumerate(message_analyses):
            analysis = msg_analysis["analysis"]
            current_substances = set(analysis.substances_mentioned)

            # New substances introduced
            new_substances = current_substances - previous_substances
            if new_substances:
                progression.append({
                    "type": "new_substance",
                    "message_index": i,
                    "sender": msg_analysis["sender"],
                    "substances": list(new_substances)
                })

            # Escalation in risk
            if i > 0 and analysis.risk_level in ["high", "critical"]:
                progression.append({
                    "type": "risk_escalation",
                    "message_index": i,
                    "sender": msg_analysis["sender"],
                    "risk_level": analysis.risk_level
                })

            previous_substances = current_substances

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
        """Determine overall conversation risk level

        Args:
            results: Analysis results

        Returns:
            str: Risk level
        """
        seeking_count = len(results["seeking_incidents"])
        high_risk_count = len(results["high_risk_messages"])

        speaker_risks = [
            analysis["risk_level"]
            for analysis in results["per_speaker_analysis"].values()
        ]

        if seeking_count >= 3 or "critical" in speaker_risks:
            return "critical"
        elif high_risk_count >= 3 or "high" in speaker_risks:
            return "high"
        elif "moderate" in speaker_risks or high_risk_count >= 1:
            return "moderate"
        else:
            return "low"

    def _identify_behaviors(self, analysis: SubstanceAnalysis) -> List[str]:
        """Identify specific concerning behaviors

        Args:
            analysis: Analysis results

        Returns:
            List: Behavior descriptions
        """
        behaviors = []

        if analysis.seeking_behavior:
            behaviors.append("Active substance seeking behavior")
        if analysis.active_use_indicators:
            behaviors.append("Current intoxication or use indicators")
        if analysis.relapse_risk:
            behaviors.append("Relapse risk and craving indicators")
        if analysis.peer_pressure_detected:
            behaviors.append("Peer pressure to use substances")

        return behaviors

    def _generate_pattern_summary(self, message_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate summary of patterns found

        Args:
            message_analyses: List of per-message analyses

        Returns:
            Dict: Pattern summary
        """
        all_patterns = []
        category_counts = {}
        substance_counts = {}

        for msg in message_analyses:
            patterns = msg["analysis"].patterns_found
            all_patterns.extend(patterns)

            for pattern in patterns:
                category_counts[pattern.subcategory] = category_counts.get(pattern.subcategory, 0) + 1
                substance_counts[pattern.substance_type] = substance_counts.get(pattern.substance_type, 0) + 1

        return {
            "total_patterns": len(all_patterns),
            "unique_patterns": len(set(p.pattern for p in all_patterns)),
            "category_distribution": category_counts,
            "substance_distribution": substance_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "most_mentioned_substance": max(substance_counts, key=substance_counts.get) if substance_counts else None,
            "average_severity": sum(p.severity for p in all_patterns) / len(all_patterns) if all_patterns else 0
        }

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate substance-related recommendations

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        risk = results["overall_risk"]

        if risk == "critical":
            recommendations.append("CRITICAL: Active substance use and seeking behavior detected. Immediate intervention needed.")
            recommendations.append("Consider involving addiction specialists or substance abuse counseling services.")
            recommendations.append("Monitor for overdose risk, especially with opioid or stimulant use.")

        elif risk == "high":
            recommendations.append("HIGH RISK: Significant substance use patterns and concerning behaviors detected.")
            recommendations.append("Encourage participation in support groups or professional treatment programs.")
            recommendations.append("Provide addiction resources and treatment hotline information.")

        elif risk == "moderate":
            recommendations.append("MODERATE RISK: Substance references and some concerning patterns detected.")
            recommendations.append("Provide educational information about substance risks.")
            recommendations.append("Encourage open dialogue about substance use without judgment.")

        else:
            recommendations.append("LOW RISK: Minimal substance references detected.")

        # Specific recommendations
        pattern_summary = results["pattern_summary"]
        if results["seeking_incidents"]:
            recommendations.append("Monitor for escalating seeking behavior and potential access to substances.")
        if pattern_summary.get("most_mentioned_substance"):
            recommendations.append(f"Focus support on {pattern_summary['most_mentioned_substance']} use prevention.")

        return recommendations
