"""
Deception Markers Analysis Module
Detects linguistic indicators of deception including vagueness, distancing,
negation patterns, and absolutes based on forensic linguistics research.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DeceptionMarker:
    """Container for deception marker"""
    category: str  # vagueness, distancing, negation, absolutes
    indicator: str
    matched_text: str
    confidence: float
    position: int
    context: str = ""


@dataclass
class DeceptionAnalysis:
    """Deception analysis results"""
    markers: List[DeceptionMarker] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_deception_score: float = 0.0
    cognitive_load_indicator: float = 0.0
    statement_consistency_score: float = 1.0
    credibility_assessment: str = "credible"  # credible, questionable, deceptive
    linguistic_indicators: Dict[str, Any] = field(default_factory=dict)


class DeceptionAnalyzer:
    """Analyzes text for deception markers based on forensic linguistics"""

    # Linguistic indicators of deception (research-based)
    VAGUENESS_MARKERS = [
        r'\b(maybe|possibly|probably|perhaps|might)\b',
        r'\b(sort of|kind of|basically|essentially)\b',
        r'\b(around|approximately|roughly|about)\b',
        r'\b(i think|i believe|i suppose|i guess)\b',
        r'\bas far as i (know|remember|can tell)\b',
        r'\bto the best of my (knowledge|recollection)\b',
    ]

    DISTANCING_MARKERS = [
        r'\bthat (person|man|woman|guy|girl)\b',
        r'\bthe (individual|subject|party)\b',
        r'\b(one|someone|somebody) (might|would|could)\b',
        r'\bit (happened|occurred|took place)\b',
        # Using passive voice to distance from action
        r'was \w+ed by',
        r'were \w+ed by',
    ]

    NEGATION_PATTERNS = [
        r"\b(didn't|did not|don't|do not)\b",
        r"\b(wasn't|was not|weren't|were not)\b",
        r"\b(couldn't|could not|wouldn't|would not)\b",
        r"\b(haven't|have not|hasn't|has not)\b",
        r'\bnot my (fault|problem|responsibility|doing)\b',
        r'\bi had nothing to do with',
        r'\bnever\b(?! before| again)',  # Simple "never" (not "never before/again")
    ]

    ABSOLUTE_STATEMENTS = [
        r'\b(always|never|every|none|all|nothing)\b',
        r'\b(completely|totally|absolutely|entirely)\b',
        r'\b100%|one hundred percent\b',
        r'\bwithout a doubt\b',
        r'\bfor sure\b',
        r'\bdefinitely\b',
    ]

    # Credibility indicators (truth-telling patterns)
    CREDIBILITY_MARKERS = [
        r'\bspecifically\b',
        r'\bexactly\b',
        r'\b\d+:\d+\b',  # Specific times
        r'\b\d+ (minutes|hours|days)\b',  # Specific durations
        r'\b(first|then|after|before|during|while)\b',  # Temporal ordering
    ]

    # Cognitive load indicators (complexity of deception)
    COGNITIVE_LOAD_MARKERS = [
        r'\b(uh|um|er|ah)\b',  # Filled pauses
        r'\.{3,}',  # Ellipses indicating trailing off
        r'\b(\w+) \1\b',  # Word repetition
        r'[.!?]\s*[.!?]',  # Multiple punctuation
    ]

    def __init__(self):
        """Initialize deception analyzer"""
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile all regex patterns"""
        return {
            'vagueness': [re.compile(p, re.IGNORECASE) for p in self.VAGUENESS_MARKERS],
            'distancing': [re.compile(p, re.IGNORECASE) for p in self.DISTANCING_MARKERS],
            'negation': [re.compile(p, re.IGNORECASE) for p in self.NEGATION_PATTERNS],
            'absolutes': [re.compile(p, re.IGNORECASE) for p in self.ABSOLUTE_STATEMENTS],
            'credibility': [re.compile(p, re.IGNORECASE) for p in self.CREDIBILITY_MARKERS],
            'cognitive_load': [re.compile(p, re.IGNORECASE) for p in self.COGNITIVE_LOAD_MARKERS],
        }

    def analyze_message(self, text: str, baseline: Optional[Dict] = None) -> DeceptionAnalysis:
        """Analyze single message for deception markers

        Args:
            text: Message text
            baseline: Optional baseline statistics for comparison

        Returns:
            DeceptionAnalysis: Analysis results
        """
        analysis = DeceptionAnalysis()

        if not text:
            return analysis

        # Analyze each deception category
        for category in ['vagueness', 'distancing', 'negation', 'absolutes']:
            markers = self._find_markers(text, category)
            analysis.markers.extend(markers)

            if markers:
                # Calculate category score
                analysis.category_scores[category] = self._calculate_category_score(
                    markers, text, category
                )

        # Analyze credibility indicators (inverse relationship)
        credibility_markers = self._find_markers(text, 'credibility')
        credibility_score = len(credibility_markers) / max(len(text.split()), 1) * 10

        # Analyze cognitive load
        cognitive_markers = self._find_markers(text, 'cognitive_load')
        analysis.cognitive_load_indicator = len(cognitive_markers) / max(len(text.split()), 1) * 10

        # Calculate linguistic indicators
        analysis.linguistic_indicators = self._calculate_linguistic_indicators(text)

        # Calculate overall deception score
        analysis.overall_deception_score = self._calculate_overall_score(
            analysis.category_scores,
            credibility_score,
            analysis.cognitive_load_indicator,
            analysis.linguistic_indicators
        )

        # Determine credibility assessment
        analysis.credibility_assessment = self._assess_credibility(
            analysis.overall_deception_score
        )

        # Compare with baseline if provided
        if baseline:
            analysis.statement_consistency_score = self._calculate_consistency(
                analysis.linguistic_indicators,
                baseline
            )

        return analysis

    def _find_markers(self, text: str, category: str) -> List[DeceptionMarker]:
        """Find deception markers in text

        Args:
            text: Text to analyze
            category: Marker category

        Returns:
            List[DeceptionMarker]: Found markers
        """
        markers = []
        patterns = self.compiled_patterns.get(category, [])

        for pattern in patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                marker = DeceptionMarker(
                    category=category,
                    indicator=pattern.pattern,
                    matched_text=match.group(),
                    confidence=self._calculate_marker_confidence(match, text, category),
                    position=match.start(),
                    context=context
                )
                markers.append(marker)

        return markers

    def _calculate_marker_confidence(self, match: re.Match, text: str, category: str) -> float:
        """Calculate confidence in deception marker

        Args:
            match: Regex match
            text: Full text
            category: Marker category

        Returns:
            float: Confidence (0-1)
        """
        confidence = 0.6  # Base confidence

        # Adjust based on category
        if category == 'vagueness':
            # Multiple vagueness markers increase confidence
            vague_count = sum(1 for p in self.compiled_patterns['vagueness']
                            if p.search(text))
            if vague_count > 3:
                confidence += 0.2

        elif category == 'distancing':
            # Third person references when first person expected
            if text.count('I') < text.count('he') + text.count('she') + text.count('they'):
                confidence += 0.15

        elif category == 'negation':
            # Excessive negation is suspicious
            negation_density = text.lower().count('not') + text.lower().count("n't")
            if negation_density > len(text.split()) * 0.1:  # More than 10% negation
                confidence += 0.2

        elif category == 'absolutes':
            # Absolutes with emotional language
            if any(emotion in text.lower() for emotion in ['angry', 'upset', 'hate', 'love']):
                confidence += 0.1

        return min(1.0, confidence)

    def _calculate_category_score(self, markers: List[DeceptionMarker], text: str, category: str) -> float:
        """Calculate deception score for category

        Args:
            markers: List of markers found
            text: Full text
            category: Category name

        Returns:
            float: Category score (0-1)
        """
        if not markers:
            return 0.0

        # Calculate marker density
        word_count = len(text.split())
        marker_density = len(markers) / max(word_count, 1)

        # Weight by average confidence
        avg_confidence = sum(m.confidence for m in markers) / len(markers)

        # Category-specific adjustments
        category_weight = {
            'vagueness': 1.0,
            'distancing': 1.2,  # Distancing is strong indicator
            'negation': 0.9,
            'absolutes': 0.8,  # Absolutes can be truthful emphasis too
        }.get(category, 1.0)

        score = marker_density * avg_confidence * category_weight * 10
        return min(1.0, score)

    def _calculate_linguistic_indicators(self, text: str) -> Dict[str, Any]:
        """Calculate various linguistic indicators

        Args:
            text: Text to analyze

        Returns:
            Dict: Linguistic indicators
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        indicators = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'pronoun_usage': {},
            'tense_distribution': {},
            'lexical_diversity': 0.0,
            'self_references': 0,
            'other_references': 0,
        }

        # Pronoun analysis
        pronouns = {
            'first_singular': ['i', 'me', 'my', 'mine', 'myself'],
            'first_plural': ['we', 'us', 'our', 'ours', 'ourselves'],
            'second': ['you', 'your', 'yours', 'yourself'],
            'third': ['he', 'she', 'they', 'him', 'her', 'them'],
        }

        for category, pronoun_list in pronouns.items():
            count = sum(1 for word in words if word.lower() in pronoun_list)
            indicators['pronoun_usage'][category] = count / max(len(words), 1)

        indicators['self_references'] = (
            indicators['pronoun_usage'].get('first_singular', 0) * len(words)
        )
        indicators['other_references'] = (
            indicators['pronoun_usage'].get('third', 0) * len(words)
        )

        # Tense analysis (simplified)
        past_markers = ['was', 'were', 'did', 'had', 'went', 'said', 'told']
        present_markers = ['is', 'are', 'am', 'do', 'does', 'have', 'has']
        future_markers = ['will', 'shall', 'going to', 'gonna']

        words_lower = [w.lower() for w in words]
        indicators['tense_distribution']['past'] = sum(1 for w in words_lower if w in past_markers)
        indicators['tense_distribution']['present'] = sum(1 for w in words_lower if w in present_markers)
        indicators['tense_distribution']['future'] = sum(1 for w in words_lower if w in future_markers)

        # Lexical diversity (unique words / total words)
        unique_words = len(set(words_lower))
        indicators['lexical_diversity'] = unique_words / max(len(words), 1)

        return indicators

    def _calculate_overall_score(self, category_scores: Dict[str, float],
                                credibility_score: float,
                                cognitive_load: float,
                                linguistic_indicators: Dict) -> float:
        """Calculate overall deception score

        Args:
            category_scores: Scores by category
            credibility_score: Credibility indicator score
            cognitive_load: Cognitive load indicator
            linguistic_indicators: Linguistic metrics

        Returns:
            float: Overall deception score (0-1)
        """
        score = 0.0

        # Weight category scores
        if category_scores:
            weighted_sum = (
                category_scores.get('vagueness', 0) * 0.25 +
                category_scores.get('distancing', 0) * 0.3 +
                category_scores.get('negation', 0) * 0.2 +
                category_scores.get('absolutes', 0) * 0.15
            )
            score += weighted_sum

        # Add cognitive load component
        score += cognitive_load * 0.1

        # Subtract credibility indicators (they suggest truthfulness)
        score -= credibility_score * 0.2

        # Adjust for linguistic indicators
        # Low self-reference can indicate deception
        self_ref_ratio = linguistic_indicators.get('self_references', 0) / \
                        max(linguistic_indicators.get('word_count', 1), 1)
        if self_ref_ratio < 0.01:  # Very low self-reference
            score += 0.1

        # Low lexical diversity can indicate cognitive load
        if linguistic_indicators.get('lexical_diversity', 1) < 0.3:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _assess_credibility(self, deception_score: float) -> str:
        """Assess credibility based on deception score

        Args:
            deception_score: Overall deception score

        Returns:
            str: Credibility assessment
        """
        if deception_score < 0.3:
            return "credible"
        elif deception_score < 0.6:
            return "questionable"
        else:
            return "deceptive"

    def _calculate_consistency(self, current_indicators: Dict, baseline: Dict) -> float:
        """Calculate consistency with baseline

        Args:
            current_indicators: Current linguistic indicators
            baseline: Baseline indicators

        Returns:
            float: Consistency score (0-1, higher is more consistent)
        """
        consistency = 1.0

        # Compare key metrics
        metrics_to_compare = [
            'avg_sentence_length',
            'lexical_diversity',
            ('pronoun_usage', 'first_singular'),
        ]

        for metric in metrics_to_compare:
            if isinstance(metric, tuple):
                current_val = current_indicators.get(metric[0], {}).get(metric[1], 0)
                baseline_val = baseline.get(metric[0], {}).get(metric[1], 0)
            else:
                current_val = current_indicators.get(metric, 0)
                baseline_val = baseline.get(metric, 0)

            if baseline_val > 0:
                deviation = abs(current_val - baseline_val) / baseline_val
                consistency -= deviation * 0.2

        return max(0.0, consistency)

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation for deception patterns

        Args:
            messages: List of messages

        Returns:
            Dict: Comprehensive deception analysis
        """
        results = {
            "per_message_analysis": [],
            "per_speaker_baseline": {},
            "per_speaker_deception": {},
            "inconsistency_flags": [],
            "high_deception_messages": [],
            "overall_credibility": "credible",
            "deception_patterns": {},
            "recommendations": []
        }

        speaker_messages = {}
        speaker_indicators = {}

        # First pass: establish baselines
        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '')

            if sender not in speaker_messages:
                speaker_messages[sender] = []
                speaker_indicators[sender] = []

            speaker_messages[sender].append(text)

            # Calculate indicators
            analysis = self.analyze_message(text)
            speaker_indicators[sender].append(analysis.linguistic_indicators)

        # Calculate speaker baselines
        for sender, indicators_list in speaker_indicators.items():
            if indicators_list:
                baseline = self._calculate_speaker_baseline(indicators_list)
                results["per_speaker_baseline"][sender] = baseline

        # Second pass: analyze with baselines
        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '')

            baseline = results["per_speaker_baseline"].get(sender)
            analysis = self.analyze_message(text, baseline)

            msg_result = {
                "index": i,
                "sender": sender,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "analysis": analysis
            }
            results["per_message_analysis"].append(msg_result)

            # Track high deception messages
            if analysis.overall_deception_score > 0.6:
                results["high_deception_messages"].append({
                    "index": i,
                    "sender": sender,
                    "score": analysis.overall_deception_score,
                    "primary_indicators": list(analysis.category_scores.keys())
                })

            # Track inconsistencies
            if baseline and analysis.statement_consistency_score < 0.5:
                results["inconsistency_flags"].append({
                    "index": i,
                    "sender": sender,
                    "consistency_score": analysis.statement_consistency_score,
                    "deviation_from_baseline": "significant"
                })

            # Update speaker deception profile
            if sender not in results["per_speaker_deception"]:
                results["per_speaker_deception"][sender] = {
                    "avg_deception_score": 0,
                    "max_deception_score": 0,
                    "deceptive_message_count": 0,
                    "primary_deception_style": None
                }

            speaker_profile = results["per_speaker_deception"][sender]
            speaker_profile["max_deception_score"] = max(
                speaker_profile["max_deception_score"],
                analysis.overall_deception_score
            )
            if analysis.overall_deception_score > 0.5:
                speaker_profile["deceptive_message_count"] += 1

        # Calculate averages and determine patterns
        for sender in results["per_speaker_deception"]:
            profile = results["per_speaker_deception"][sender]
            sender_analyses = [m["analysis"] for m in results["per_message_analysis"]
                             if m["sender"] == sender]

            if sender_analyses:
                avg_score = sum(a.overall_deception_score for a in sender_analyses) / len(sender_analyses)
                profile["avg_deception_score"] = avg_score

                # Determine primary deception style
                all_categories = Counter()
                for analysis in sender_analyses:
                    all_categories.update(analysis.category_scores.keys())

                if all_categories:
                    profile["primary_deception_style"] = all_categories.most_common(1)[0][0]

        # Determine overall credibility
        all_scores = [m["analysis"].overall_deception_score
                     for m in results["per_message_analysis"]]
        if all_scores:
            avg_deception = sum(all_scores) / len(all_scores)
            if avg_deception < 0.3:
                results["overall_credibility"] = "credible"
            elif avg_deception < 0.6:
                results["overall_credibility"] = "questionable"
            else:
                results["overall_credibility"] = "deceptive"

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _calculate_speaker_baseline(self, indicators_list: List[Dict]) -> Dict:
        """Calculate baseline indicators for speaker

        Args:
            indicators_list: List of linguistic indicators

        Returns:
            Dict: Baseline indicators
        """
        baseline = {}

        if not indicators_list:
            return baseline

        # Average numeric metrics
        numeric_metrics = ['word_count', 'sentence_count', 'avg_sentence_length',
                          'lexical_diversity', 'self_references', 'other_references']

        for metric in numeric_metrics:
            values = [ind.get(metric, 0) for ind in indicators_list]
            baseline[metric] = sum(values) / len(values) if values else 0

        # Average pronoun usage
        baseline['pronoun_usage'] = {}
        pronoun_categories = ['first_singular', 'first_plural', 'second', 'third']
        for category in pronoun_categories:
            values = [ind.get('pronoun_usage', {}).get(category, 0) for ind in indicators_list]
            baseline['pronoun_usage'][category] = sum(values) / len(values) if values else 0

        return baseline

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on deception analysis

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        credibility = results["overall_credibility"]

        if credibility == "deceptive":
            recommendations.append("HIGH DECEPTION: Multiple linguistic markers of deception detected.")
            recommendations.append("Verify claims with independent sources.")
            recommendations.append("Document inconsistencies for future reference.")
            recommendations.append("Consider requesting specific details and documentation.")

        elif credibility == "questionable":
            recommendations.append("QUESTIONABLE: Some deception indicators present.")
            recommendations.append("Ask follow-up questions for clarification.")
            recommendations.append("Pay attention to consistency across statements.")
            recommendations.append("Look for corroborating evidence.")

        else:
            recommendations.append("CREDIBLE: No significant deception indicators detected.")
            recommendations.append("Continue normal communication patterns.")

        # Check for inconsistencies
        if results["inconsistency_flags"]:
            recommendations.append("NOTE: Significant inconsistencies detected in communication style.")
            recommendations.append("This may indicate stress, deception, or emotional state changes.")

        # Speaker-specific recommendations
        for sender, profile in results["per_speaker_deception"].items():
            if profile["avg_deception_score"] > 0.6:
                style = profile.get("primary_deception_style", "unknown")
                recommendations.append(f"Speaker '{sender}' shows high deception markers, primarily: {style}")

        return recommendations