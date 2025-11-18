#!/usr/bin/env python3
"""
Confidence Framework

Framework for tracking confidence scores and evidence across all passes.
Enables high-confidence system with evidence-based conclusions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level classifications"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"            # 75-89%
    MEDIUM = "medium"        # 50-74%
    LOW = "low"              # 25-49%
    VERY_LOW = "very_low"    # 0-24%
    UNKNOWN = "unknown"      # No data


@dataclass
class Evidence:
    """Evidence item supporting a finding"""
    source_pass: int
    source_name: str
    finding_type: str
    description: str
    confidence: float  # 0.0 - 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        return f"<Evidence pass={self.source_pass} type={self.finding_type} conf={self.confidence:.2f}>"


@dataclass
class Finding:
    """A finding with confidence score and evidence"""
    finding_id: str
    finding_type: str
    description: str
    severity: str  # critical, high, medium, low
    confidence: float  # 0.0 - 1.0
    evidence_items: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification"""
        if self.confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.25:
            return ConfidenceLevel.LOW
        elif self.confidence > 0:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.UNKNOWN

    @property
    def evidence_count(self) -> int:
        """Get number of supporting evidence items"""
        return len(self.evidence_items)

    def add_evidence(self, evidence: Evidence):
        """Add supporting evidence"""
        self.evidence_items.append(evidence)
        # Recalculate confidence based on multiple evidence
        self._recalculate_confidence()

    def _recalculate_confidence(self):
        """Recalculate confidence based on all evidence"""
        if not self.evidence_items:
            return

        # Multiple evidence increases confidence
        # Use weighted average with diminishing returns
        total_weight = 0.0
        weighted_conf = 0.0

        for i, evidence in enumerate(self.evidence_items):
            # Each additional piece of evidence has less weight
            weight = 1.0 / (i + 1)
            total_weight += weight
            weighted_conf += evidence.confidence * weight

        if total_weight > 0:
            base_conf = weighted_conf / total_weight

            # Bonus for multiple corroborating evidence (up to 20% boost)
            evidence_boost = min(0.20, len(self.evidence_items) * 0.05)

            self.confidence = min(1.0, base_conf + evidence_boost)


class ConfidenceScorer:
    """
    Confidence scoring system for pass results.

    Calculates confidence scores based on:
    - Data quality
    - Evidence quantity
    - Cross-validation
    - Historical accuracy
    """

    def __init__(self):
        """Initialize confidence scorer"""
        self.historical_accuracy: Dict[int, float] = {}  # Pass number -> accuracy
        self.baseline_confidence: Dict[str, float] = {
            # Default confidence by finding type
            'grooming': 0.70,
            'manipulation': 0.65,
            'deception': 0.60,
            'gaslighting': 0.75,
            'risk_assessment': 0.80,
            'sentiment': 0.85,
            'person_identification': 0.90,
        }

    def score_finding(
        self,
        finding_type: str,
        supporting_evidence: List[Evidence],
        data_quality: float = 1.0,
        cross_validated: bool = False
    ) -> float:
        """
        Calculate confidence score for a finding.

        Args:
            finding_type: Type of finding
            supporting_evidence: List of evidence supporting the finding
            data_quality: Quality of input data (0.0-1.0)
            cross_validated: Whether finding was cross-validated

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Start with baseline confidence
        base_confidence = self.baseline_confidence.get(finding_type, 0.50)

        # Adjust for data quality
        confidence = base_confidence * data_quality

        # Adjust for evidence count
        if supporting_evidence:
            # More evidence increases confidence (logarithmic)
            import math
            evidence_factor = 1.0 + (0.3 * math.log1p(len(supporting_evidence)))
            confidence *= min(evidence_factor, 1.5)  # Max 50% boost

        # Boost if cross-validated
        if cross_validated:
            confidence *= 1.15  # 15% boost

        # Cap at 1.0
        return min(1.0, confidence)

    def aggregate_confidence(
        self,
        findings: List[Finding],
        aggregation_method: str = 'weighted_average'
    ) -> float:
        """
        Aggregate confidence from multiple findings.

        Args:
            findings: List of findings
            aggregation_method: How to aggregate ('weighted_average', 'min', 'max')

        Returns:
            float: Aggregated confidence
        """
        if not findings:
            return 0.0

        if aggregation_method == 'weighted_average':
            # Weight by severity
            severity_weights = {
                'critical': 1.5,
                'high': 1.2,
                'medium': 1.0,
                'low': 0.8
            }

            total_weight = 0.0
            weighted_sum = 0.0

            for finding in findings:
                weight = severity_weights.get(finding.severity, 1.0)
                total_weight += weight
                weighted_sum += finding.confidence * weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        elif aggregation_method == 'min':
            return min(f.confidence for f in findings)

        elif aggregation_method == 'max':
            return max(f.confidence for f in findings)

        else:
            # Simple average
            return sum(f.confidence for f in findings) / len(findings)

    def update_historical_accuracy(self, pass_number: int, accuracy: float):
        """
        Update historical accuracy for a pass.

        Args:
            pass_number: Pass number
            accuracy: Measured accuracy (0.0-1.0)
        """
        if pass_number not in self.historical_accuracy:
            self.historical_accuracy[pass_number] = accuracy
        else:
            # Exponential moving average
            alpha = 0.3  # Weight for new data
            self.historical_accuracy[pass_number] = (
                alpha * accuracy + (1 - alpha) * self.historical_accuracy[pass_number]
            )


class EvidenceAggregator:
    """
    Aggregates evidence from multiple passes to form high-confidence conclusions.
    """

    def __init__(self):
        """Initialize evidence aggregator"""
        self.evidence_by_type: Dict[str, List[Evidence]] = {}
        self.findings: List[Finding] = []
        self.confidence_scorer = ConfidenceScorer()

    def add_evidence(self, evidence: Evidence):
        """
        Add evidence item.

        Args:
            evidence: Evidence to add
        """
        finding_type = evidence.finding_type

        if finding_type not in self.evidence_by_type:
            self.evidence_by_type[finding_type] = []

        self.evidence_by_type[finding_type].append(evidence)
        logger.debug(f"Added evidence: {evidence}")

    def create_finding(
        self,
        finding_id: str,
        finding_type: str,
        description: str,
        severity: str,
        evidence_items: Optional[List[Evidence]] = None
    ) -> Finding:
        """
        Create a finding with evidence.

        Args:
            finding_id: Unique finding ID
            finding_type: Type of finding
            description: Description
            severity: Severity level
            evidence_items: Optional initial evidence

        Returns:
            Finding: Created finding
        """
        # Calculate initial confidence
        evidence_items = evidence_items or []
        confidence = self.confidence_scorer.score_finding(
            finding_type,
            evidence_items,
            data_quality=1.0,
            cross_validated=False
        )

        finding = Finding(
            finding_id=finding_id,
            finding_type=finding_type,
            description=description,
            severity=severity,
            confidence=confidence,
            evidence_items=evidence_items
        )

        self.findings.append(finding)
        return finding

    def get_evidence_for_type(self, finding_type: str) -> List[Evidence]:
        """Get all evidence for a specific finding type"""
        return self.evidence_by_type.get(finding_type, [])

    def get_corroborating_evidence(
        self,
        finding_type: str,
        min_confidence: float = 0.5
    ) -> List[Evidence]:
        """
        Get evidence that corroborates across multiple passes.

        Args:
            finding_type: Type of finding
            min_confidence: Minimum confidence threshold

        Returns:
            List of high-confidence corroborating evidence
        """
        evidence = self.get_evidence_for_type(finding_type)

        # Filter by confidence
        high_conf_evidence = [e for e in evidence if e.confidence >= min_confidence]

        # Group by source pass
        by_pass: Dict[int, List[Evidence]] = {}
        for e in high_conf_evidence:
            if e.source_pass not in by_pass:
                by_pass[e.source_pass] = []
            by_pass[e.source_pass].append(e)

        # Return evidence that appears in multiple passes
        if len(by_pass) >= 2:
            return high_conf_evidence
        else:
            return []

    def generate_confidence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive confidence report.

        Returns:
            Dict with confidence analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_evidence_items': sum(len(e) for e in self.evidence_by_type.values()),
            'evidence_by_type': {
                ftype: len(evidence)
                for ftype, evidence in self.evidence_by_type.items()
            },
            'total_findings': len(self.findings),
            'findings_by_confidence': {
                level.value: 0 for level in ConfidenceLevel
            },
            'findings_by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'high_confidence_findings': [],
            'low_confidence_findings': [],
            'overall_confidence': 0.0
        }

        # Analyze findings
        for finding in self.findings:
            # Count by confidence level
            report['findings_by_confidence'][finding.confidence_level.value] += 1

            # Count by severity
            report['findings_by_severity'][finding.severity] += 1

            # Track high/low confidence findings
            if finding.confidence >= 0.75:
                report['high_confidence_findings'].append({
                    'id': finding.finding_id,
                    'type': finding.finding_type,
                    'confidence': finding.confidence,
                    'evidence_count': finding.evidence_count
                })
            elif finding.confidence < 0.50:
                report['low_confidence_findings'].append({
                    'id': finding.finding_id,
                    'type': finding.finding_type,
                    'confidence': finding.confidence,
                    'evidence_count': finding.evidence_count
                })

        # Calculate overall confidence
        if self.findings:
            report['overall_confidence'] = self.confidence_scorer.aggregate_confidence(
                self.findings,
                aggregation_method='weighted_average'
            )

        return report

    def clear(self):
        """Clear all evidence and findings"""
        self.evidence_by_type = {}
        self.findings = []


# Global evidence aggregator instance
_global_evidence_aggregator = EvidenceAggregator()


def get_evidence_aggregator() -> EvidenceAggregator:
    """Get global evidence aggregator instance"""
    return _global_evidence_aggregator
