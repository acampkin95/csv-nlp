#!/usr/bin/env python3
"""
Advanced Analysis Passes (Passes 16-20)

High-confidence system passes for enhanced accuracy and validation.

Pass 16: Language Pattern Analysis
Pass 17: Cross-Validation
Pass 18: Pattern Correlation
Pass 19: Anomaly Detection
Pass 20: Final Confidence Assessment
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import statistics

from .base_pass import BasePass, PassGroup
from .confidence_framework import Evidence, Finding, get_evidence_aggregator

logger = logging.getLogger(__name__)


class Pass16_LanguagePatternAnalysis(BasePass):
    """
    Pass 16: Language Pattern Analysis

    Analyzes linguistic patterns including:
    - Vocabulary sophistication
    - Sentence structure complexity
    - Formality levels
    - Language consistency
    - Communication style shifts
    """

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=16,
            pass_name="Language Pattern Analysis",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['sentiment_analysis', 'person_identification']
        )

    def _execute_pass(
        self,
        messages: List[Dict],
        sentiment_results: Dict,
        person_identification: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute language pattern analysis"""
        logger.info(f"  Analyzing language patterns across {len(messages)} messages")

        # Analyze vocabulary
        vocabulary_analysis = self._analyze_vocabulary(messages)

        # Analyze sentence complexity
        complexity_analysis = self._analyze_complexity(messages)

        # Analyze formality
        formality_analysis = self._analyze_formality(messages)

        # Detect style shifts
        style_shifts = self._detect_style_shifts(messages)

        # Calculate confidence
        confidence = self._calculate_confidence(
            vocabulary_analysis,
            complexity_analysis,
            formality_analysis
        )

        # Create evidence
        if style_shifts:
            evidence_aggregator = get_evidence_aggregator()
            for shift in style_shifts[:3]:  # Top 3 shifts
                evidence = Evidence(
                    source_pass=16,
                    source_name="Language Pattern Analysis",
                    finding_type="language_shift",
                    description=f"Significant style shift detected at message {shift['index']}",
                    confidence=shift['confidence'],
                    supporting_data=shift
                )
                evidence_aggregator.add_evidence(evidence)

        result = {
            'vocabulary': vocabulary_analysis,
            'complexity': complexity_analysis,
            'formality': formality_analysis,
            'style_shifts': style_shifts,
            'overall_confidence': confidence,
            'patterns_detected': len(style_shifts)
        }

        print(f"  Language Patterns Detected: {len(style_shifts)}, Confidence: {confidence:.2f}")
        return result

    def _analyze_vocabulary(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze vocabulary usage"""
        all_words = []
        word_lengths = []

        for msg in messages:
            text = msg.get('text', '').lower()
            words = text.split()
            all_words.extend(words)
            word_lengths.extend(len(w) for w in words)

        unique_words = set(all_words)

        return {
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'vocabulary_diversity': len(unique_words) / len(all_words) if all_words else 0,
            'avg_word_length': statistics.mean(word_lengths) if word_lengths else 0,
            'long_words_ratio': sum(1 for w in all_words if len(w) > 6) / len(all_words) if all_words else 0
        }

    def _analyze_complexity(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze sentence complexity"""
        sentence_lengths = []
        word_counts = []

        for msg in messages:
            text = msg.get('text', '')
            # Simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_lengths.extend(len(s) for s in sentences)

            words = text.split()
            word_counts.append(len(words))

        return {
            'avg_sentence_length': statistics.mean(sentence_lengths) if sentence_lengths else 0,
            'avg_words_per_message': statistics.mean(word_counts) if word_counts else 0,
            'complexity_score': (
                statistics.mean(sentence_lengths) * 0.6 +
                statistics.mean(word_counts) * 0.4
            ) if sentence_lengths and word_counts else 0
        }

    def _analyze_formality(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze formality level"""
        informal_markers = ['lol', 'omg', 'wtf', 'lmao', 'tbh', 'imo', '!!!', '???']
        formal_markers = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']

        informal_count = 0
        formal_count = 0

        for msg in messages:
            text = msg.get('text', '').lower()
            informal_count += sum(marker in text for marker in informal_markers)
            formal_count += sum(marker in text for marker in formal_markers)

        total = informal_count + formal_count

        return {
            'informal_score': informal_count / len(messages) if messages else 0,
            'formal_score': formal_count / len(messages) if messages else 0,
            'formality_ratio': (
                (formal_count - informal_count) / total
                if total > 0 else 0
            )
        }

    def _detect_style_shifts(self, messages: List[Dict]) -> List[Dict]:
        """Detect significant shifts in communication style"""
        shifts = []

        # Analyze in windows
        window_size = 5
        for i in range(window_size, len(messages) - window_size):
            before = messages[i - window_size:i]
            after = messages[i:i + window_size]

            # Compare vocabulary diversity
            before_vocab = self._analyze_vocabulary(before)
            after_vocab = self._analyze_vocabulary(after)

            vocab_diff = abs(
                before_vocab['vocabulary_diversity'] -
                after_vocab['vocabulary_diversity']
            )

            # Significant shift threshold
            if vocab_diff > 0.15:
                shifts.append({
                    'index': i,
                    'type': 'vocabulary_shift',
                    'magnitude': vocab_diff,
                    'confidence': min(vocab_diff * 3, 1.0),
                    'before_diversity': before_vocab['vocabulary_diversity'],
                    'after_diversity': after_vocab['vocabulary_diversity']
                })

        return sorted(shifts, key=lambda x: x['magnitude'], reverse=True)

    def _calculate_confidence(self, vocab: Dict, complexity: Dict, formality: Dict) -> float:
        """Calculate overall confidence in analysis"""
        # Base confidence on data quantity
        word_count = vocab.get('total_words', 0)

        if word_count < 50:
            return 0.30
        elif word_count < 200:
            return 0.50
        elif word_count < 500:
            return 0.70
        else:
            return 0.85

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback result"""
        return {
            'vocabulary': {},
            'complexity': {},
            'formality': {},
            'style_shifts': [],
            'overall_confidence': 0.0,
            'error': 'Language pattern analysis failed'
        }


class Pass17_CrossValidation(BasePass):
    """
    Pass 17: Cross-Validation

    Validates findings across multiple passes to increase confidence.
    Identifies consistent vs. contradictory evidence.
    """

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=17,
            pass_name="Cross-Validation",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=[
                'grooming_detection',
                'manipulation_detection',
                'deception_analysis',
                'gaslighting_detection',
                'risk_assessment'
            ]
        )

    def _execute_pass(
        self,
        grooming_results: Dict,
        manipulation_results: Dict,
        deception_results: Dict,
        gaslighting_results: Dict,
        risk_assessment: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute cross-validation"""
        logger.info("  Cross-validating results from behavioral passes")

        # Collect findings from all passes
        findings = {
            'grooming': grooming_results,
            'manipulation': manipulation_results,
            'deception': deception_results,
            'gaslighting': gaslighting_results,
            'risk': risk_assessment
        }

        # Validate consistency
        consistency_score = self._validate_consistency(findings)

        # Identify corroborating evidence
        corroboration = self._find_corroboration(findings)

        # Identify contradictions
        contradictions = self._find_contradictions(findings)

        # Calculate overall validation confidence
        validation_confidence = self._calculate_validation_confidence(
            consistency_score,
            corroboration,
            contradictions
        )

        # Add evidence for validated findings
        evidence_aggregator = get_evidence_aggregator()
        for finding in corroboration:
            evidence = Evidence(
                source_pass=17,
                source_name="Cross-Validation",
                finding_type=finding['type'],
                description=f"Finding validated across {finding['source_count']} passes",
                confidence=finding['confidence'],
                supporting_data=finding
            )
            evidence_aggregator.add_evidence(evidence)

        result = {
            'consistency_score': consistency_score,
            'corroborating_findings': corroboration,
            'contradictions': contradictions,
            'validation_confidence': validation_confidence,
            'validated_count': len(corroboration),
            'contradiction_count': len(contradictions)
        }

        print(f"  Validated Findings: {len(corroboration)}, "
              f"Contradictions: {len(contradictions)}, "
              f"Confidence: {validation_confidence:.2f}")

        return result

    def _validate_consistency(self, findings: Dict[str, Dict]) -> float:
        """Validate consistency across findings"""
        # Check if risk levels align across passes
        risk_levels = []

        for pass_name, results in findings.items():
            if isinstance(results, dict):
                risk = results.get('overall_risk') or results.get('risk_level')
                if risk:
                    risk_levels.append(risk)

        if not risk_levels:
            return 0.5

        # Calculate consistency
        most_common = Counter(risk_levels).most_common(1)[0]
        consistency = most_common[1] / len(risk_levels)

        return consistency

    def _find_corroboration(self, findings: Dict[str, Dict]) -> List[Dict]:
        """Find corroborating evidence across passes"""
        corroboration = []

        # Check for high-risk findings across multiple passes
        high_risk_passes = []
        for pass_name, results in findings.items():
            if isinstance(results, dict):
                risk = results.get('overall_risk') or results.get('risk_level')
                if risk in ['high', 'critical']:
                    high_risk_passes.append(pass_name)

        if len(high_risk_passes) >= 2:
            corroboration.append({
                'type': 'high_risk_consensus',
                'source_count': len(high_risk_passes),
                'sources': high_risk_passes,
                'confidence': min(0.95, 0.6 + len(high_risk_passes) * 0.15)
            })

        return corroboration

    def _find_contradictions(self, findings: Dict[str, Dict]) -> List[Dict]:
        """Find contradictory findings"""
        contradictions = []

        # Example: Check if sentiment is positive but risk is high
        # This is a simplified example - real implementation would be more sophisticated

        return contradictions

    def _calculate_validation_confidence(
        self,
        consistency: float,
        corroboration: List,
        contradictions: List
    ) -> float:
        """Calculate overall validation confidence"""
        base_confidence = consistency * 0.6

        # Boost for corroboration
        corroboration_boost = min(0.3, len(corroboration) * 0.1)

        # Penalty for contradictions
        contradiction_penalty = min(0.2, len(contradictions) * 0.05)

        return max(0.0, min(1.0, base_confidence + corroboration_boost - contradiction_penalty))

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback result"""
        return {
            'consistency_score': 0.5,
            'corroborating_findings': [],
            'contradictions': [],
            'validation_confidence': 0.5,
            'error': 'Cross-validation failed'
        }


class Pass18_PatternCorrelation(BasePass):
    """
    Pass 18: Pattern Correlation Analysis

    Identifies correlations between different detected patterns.
    Helps establish causal relationships and identify complex behaviors.
    """

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=18,
            pass_name="Pattern Correlation",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=[
                'emotional_dynamics',
                'grooming_detection',
                'manipulation_detection',
                'timeline_analysis'
            ]
        )

    def _execute_pass(
        self,
        emotional_dynamics: Dict,
        grooming_results: Dict,
        manipulation_results: Dict,
        timeline_analysis: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute pattern correlation analysis"""
        logger.info("  Analyzing pattern correlations")

        # Analyze temporal correlations
        temporal_correlations = self._analyze_temporal_correlations(
            timeline_analysis,
            emotional_dynamics
        )

        # Analyze behavioral correlations
        behavioral_correlations = self._analyze_behavioral_correlations(
            grooming_results,
            manipulation_results
        )

        # Identify pattern sequences
        pattern_sequences = self._identify_pattern_sequences(
            temporal_correlations,
            behavioral_correlations
        )

        # Calculate confidence
        correlation_confidence = self._calculate_correlation_confidence(
            temporal_correlations,
            behavioral_correlations,
            pattern_sequences
        )

        result = {
            'temporal_correlations': temporal_correlations,
            'behavioral_correlations': behavioral_correlations,
            'pattern_sequences': pattern_sequences,
            'correlation_confidence': correlation_confidence,
            'significant_correlations': len(pattern_sequences)
        }

        print(f"  Pattern Correlations Found: {len(pattern_sequences)}, "
              f"Confidence: {correlation_confidence:.2f}")

        return result

    def _analyze_temporal_correlations(
        self,
        timeline: Dict,
        emotional: Dict
    ) -> List[Dict]:
        """Analyze temporal correlations"""
        correlations = []

        # Check if emotion shifts correlate with timeline events
        emotion_shifts = emotional.get('emotion_shifts', [])
        timeline_points = timeline.get('timeline_points', [])

        for shift in emotion_shifts:
            # Find nearby timeline events
            shift_index = shift.get('index', 0)
            nearby_events = [
                tp for tp in timeline_points
                if abs(tp.get('index', 0) - shift_index) <= 3
            ]

            if nearby_events:
                correlations.append({
                    'type': 'emotion_timeline_correlation',
                    'shift_index': shift_index,
                    'magnitude': shift.get('magnitude', 0),
                    'nearby_events': len(nearby_events),
                    'confidence': 0.7
                })

        return correlations

    def _analyze_behavioral_correlations(
        self,
        grooming: Dict,
        manipulation: Dict
    ) -> List[Dict]:
        """Analyze behavioral correlations"""
        correlations = []

        # Check if grooming and manipulation patterns co-occur
        grooming_risk = grooming.get('overall_risk', 'low')
        manipulation_risk = manipulation.get('overall_risk', 'low')

        if grooming_risk in ['high', 'critical'] and manipulation_risk in ['high', 'critical']:
            correlations.append({
                'type': 'grooming_manipulation_correlation',
                'grooming_risk': grooming_risk,
                'manipulation_risk': manipulation_risk,
                'confidence': 0.85,
                'severity': 'high'
            })

        return correlations

    def _identify_pattern_sequences(
        self,
        temporal: List[Dict],
        behavioral: List[Dict]
    ) -> List[Dict]:
        """Identify significant pattern sequences"""
        sequences = []

        # Combine high-confidence correlations
        all_correlations = temporal + behavioral
        high_confidence = [c for c in all_correlations if c.get('confidence', 0) >= 0.7]

        for correlation in high_confidence:
            sequences.append({
                'sequence_type': correlation.get('type'),
                'confidence': correlation.get('confidence'),
                'details': correlation
            })

        return sequences

    def _calculate_correlation_confidence(
        self,
        temporal: List,
        behavioral: List,
        sequences: List
    ) -> float:
        """Calculate overall correlation confidence"""
        total_correlations = len(temporal) + len(behavioral)

        if total_correlations == 0:
            return 0.5

        # More correlations = higher confidence
        base_confidence = min(0.9, 0.5 + total_correlations * 0.05)

        # Boost if we have both types
        if temporal and behavioral:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback result"""
        return {
            'temporal_correlations': [],
            'behavioral_correlations': [],
            'pattern_sequences': [],
            'correlation_confidence': 0.5,
            'error': 'Pattern correlation analysis failed'
        }


class Pass19_AnomalyDetection(BasePass):
    """
    Pass 19: Statistical Anomaly Detection

    Detects statistical anomalies in conversation patterns that may indicate:
    - Unusual communication patterns
    - Outlier behaviors
    - Suspicious timing patterns
    - Atypical responses
    """

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=19,
            pass_name="Anomaly Detection",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['sentiment_analysis', 'timeline_analysis']
        )

    def _execute_pass(
        self,
        messages: List[Dict],
        sentiment_results: Dict,
        timeline_analysis: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute anomaly detection"""
        logger.info(f"  Detecting anomalies in {len(messages)} messages")

        # Detect message length anomalies
        length_anomalies = self._detect_length_anomalies(messages)

        # Detect sentiment anomalies
        sentiment_anomalies = self._detect_sentiment_anomalies(sentiment_results)

        # Detect timing anomalies
        timing_anomalies = self._detect_timing_anomalies(messages)

        # Calculate anomaly score
        anomaly_score, anomaly_confidence = self._calculate_anomaly_score(
            length_anomalies,
            sentiment_anomalies,
            timing_anomalies
        )

        result = {
            'length_anomalies': length_anomalies,
            'sentiment_anomalies': sentiment_anomalies,
            'timing_anomalies': timing_anomalies,
            'anomaly_score': anomaly_score,
            'anomaly_confidence': anomaly_confidence,
            'total_anomalies': len(length_anomalies) + len(sentiment_anomalies) + len(timing_anomalies)
        }

        print(f"  Anomalies Detected: {result['total_anomalies']}, "
              f"Score: {anomaly_score:.2f}, Confidence: {anomaly_confidence:.2f}")

        return result

    def _detect_length_anomalies(self, messages: List[Dict]) -> List[Dict]:
        """Detect anomalous message lengths"""
        lengths = [len(msg.get('text', '')) for msg in messages]

        if len(lengths) < 10:
            return []

        mean_length = statistics.mean(lengths)
        stdev_length = statistics.stdev(lengths) if len(lengths) > 1 else 0

        anomalies = []
        threshold = 2.5  # Standard deviations

        for i, length in enumerate(lengths):
            if stdev_length > 0:
                z_score = abs((length - mean_length) / stdev_length)
                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'type': 'length',
                        'value': length,
                        'mean': mean_length,
                        'z_score': z_score,
                        'confidence': min(0.9, z_score / 5)
                    })

        return anomalies

    def _detect_sentiment_anomalies(self, sentiment_results: Dict) -> List[Dict]:
        """Detect anomalous sentiment values"""
        sentiments = sentiment_results.get('per_message', [])

        if not sentiments:
            return []

        sentiment_values = [
            s.combined_sentiment if hasattr(s, 'combined_sentiment') else 0
            for s in sentiments if s is not None
        ]

        if len(sentiment_values) < 10:
            return []

        mean_sentiment = statistics.mean(sentiment_values)
        stdev_sentiment = statistics.stdev(sentiment_values) if len(sentiment_values) > 1 else 0

        anomalies = []
        threshold = 2.0

        for i, value in enumerate(sentiment_values):
            if stdev_sentiment > 0:
                z_score = abs((value - mean_sentiment) / stdev_sentiment)
                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'type': 'sentiment',
                        'value': value,
                        'mean': mean_sentiment,
                        'z_score': z_score,
                        'confidence': min(0.85, z_score / 4)
                    })

        return anomalies

    def _detect_timing_anomalies(self, messages: List[Dict]) -> List[Dict]:
        """Detect anomalous timing patterns"""
        # Simplified - would need actual timestamps
        return []

    def _calculate_anomaly_score(
        self,
        length_anomalies: List,
        sentiment_anomalies: List,
        timing_anomalies: List
    ) -> Tuple[float, float]:
        """Calculate overall anomaly score and confidence"""
        total_anomalies = len(length_anomalies) + len(sentiment_anomalies) + len(timing_anomalies)

        # Anomaly score based on count and severity
        anomaly_score = min(1.0, total_anomalies * 0.1)

        # Confidence based on statistical significance
        if total_anomalies == 0:
            confidence = 0.5
        else:
            avg_confidence = statistics.mean(
                [a.get('confidence', 0.5) for a in length_anomalies + sentiment_anomalies + timing_anomalies]
            )
            confidence = avg_confidence

        return anomaly_score, confidence

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback result"""
        return {
            'length_anomalies': [],
            'sentiment_anomalies': [],
            'timing_anomalies': [],
            'anomaly_score': 0.0,
            'anomaly_confidence': 0.5,
            'total_anomalies': 0,
            'error': 'Anomaly detection failed'
        }


class Pass20_FinalConfidenceAssessment(BasePass):
    """
    Pass 20: Final Confidence Assessment

    Aggregates all evidence and generates final high-confidence assessment.
    Provides overall risk level with confidence scores and detailed evidence.
    """

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=20,
            pass_name="Final Confidence Assessment",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=[
                'risk_assessment',
                'cross_validation',
                'pattern_correlation',
                'anomaly_detection',
                'intervention_recommendations'
            ]
        )

    def _execute_pass(
        self,
        risk_assessment: Dict,
        cross_validation: Dict,
        pattern_correlation: Dict,
        anomaly_detection: Dict,
        intervention_recommendations: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute final confidence assessment"""
        logger.info("  Generating final high-confidence assessment")

        # Get evidence aggregator
        evidence_aggregator = get_evidence_aggregator()

        # Generate confidence report
        confidence_report = evidence_aggregator.generate_confidence_report()

        # Calculate final risk level
        final_risk_level, final_confidence = self._calculate_final_assessment(
            risk_assessment,
            cross_validation,
            pattern_correlation,
            anomaly_detection,
            confidence_report
        )

        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(evidence_aggregator)

        # Generate recommendations with confidence
        final_recommendations = self._generate_final_recommendations(
            final_risk_level,
            final_confidence,
            intervention_recommendations,
            evidence_summary
        )

        result = {
            'final_risk_level': final_risk_level,
            'overall_confidence': final_confidence,
            'confidence_level': self._classify_confidence(final_confidence),
            'evidence_report': confidence_report,
            'evidence_summary': evidence_summary,
            'recommendations': final_recommendations,
            'high_confidence_findings': confidence_report.get('high_confidence_findings', []),
            'assessment_quality': self._assess_quality(final_confidence, evidence_summary)
        }

        print(f"  Final Assessment: {final_risk_level} (Confidence: {final_confidence:.1%})")
        print(f"  Evidence Items: {confidence_report.get('total_evidence_items', 0)}")
        print(f"  High-Confidence Findings: {len(confidence_report.get('high_confidence_findings', []))}")

        return result

    def _calculate_final_assessment(
        self,
        risk: Dict,
        validation: Dict,
        correlation: Dict,
        anomaly: Dict,
        confidence_report: Dict
    ) -> Tuple[str, float]:
        """Calculate final risk level and confidence"""
        # Get base risk level
        base_risk = risk.get('overall_risk_assessment', {}).get('risk_level', 'unknown')

        # Adjust based on validation
        validation_conf = validation.get('validation_confidence', 0.5)

        # Adjust based on anomalies
        anomaly_score = anomaly.get('anomaly_score', 0.0)

        # Get overall confidence from evidence
        evidence_conf = confidence_report.get('overall_confidence', 0.5)

        # Calculate final confidence (weighted average)
        final_confidence = (
            evidence_conf * 0.4 +
            validation_conf * 0.3 +
            (1.0 - anomaly_score) * 0.2 +
            correlation.get('correlation_confidence', 0.5) * 0.1
        )

        return base_risk, final_confidence

    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= 0.90:
            return "VERY HIGH"
        elif confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.50:
            return "MEDIUM"
        elif confidence >= 0.25:
            return "LOW"
        else:
            return "VERY LOW"

    def _generate_evidence_summary(self, aggregator) -> Dict[str, Any]:
        """Generate summary of evidence"""
        return {
            'total_evidence_items': sum(
                len(evidence) for evidence in aggregator.evidence_by_type.values()
            ),
            'evidence_types': list(aggregator.evidence_by_type.keys()),
            'findings_count': len(aggregator.findings),
            'high_confidence_count': sum(
                1 for f in aggregator.findings if f.confidence >= 0.75
            )
        }

    def _generate_final_recommendations(
        self,
        risk_level: str,
        confidence: float,
        intervention: Dict,
        evidence: Dict
    ) -> List[Dict]:
        """Generate final recommendations with confidence levels"""
        recommendations = []

        base_recommendations = intervention.get('recommendations', [])

        for rec in base_recommendations:
            if isinstance(rec, str):
                recommendations.append({
                    'recommendation': rec,
                    'confidence': confidence,
                    'priority': 'high' if risk_level in ['high', 'critical'] else 'medium'
                })

        # Add evidence-based recommendations
        if evidence['high_confidence_count'] >= 3:
            recommendations.append({
                'recommendation': 'Seek professional assessment - multiple high-confidence indicators detected',
                'confidence': 0.90,
                'priority': 'critical'
            })

        return recommendations

    def _assess_quality(self, confidence: float, evidence: Dict) -> str:
        """Assess overall quality of the assessment"""
        evidence_count = evidence.get('total_evidence_items', 0)

        if confidence >= 0.80 and evidence_count >= 10:
            return "EXCELLENT"
        elif confidence >= 0.70 and evidence_count >= 5:
            return "GOOD"
        elif confidence >= 0.50:
            return "FAIR"
        else:
            return "LIMITED"

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback result"""
        return {
            'final_risk_level': 'unknown',
            'overall_confidence': 0.0,
            'confidence_level': 'UNKNOWN',
            'evidence_report': {},
            'evidence_summary': {},
            'recommendations': [],
            'error': 'Final confidence assessment failed'
        }
