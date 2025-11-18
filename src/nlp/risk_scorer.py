"""
Behavioral Risk Scoring System
Comprehensive risk assessment combining grooming, manipulation, deception, and hostility indicators.
Provides multi-dimensional risk scores and actionable recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics
from datetime import datetime

# Import confidence scoring
from .confidence_scorer import ConfidenceScorer, get_confidence_level

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Container for comprehensive risk assessment"""
    # Component scores (0-1 scale)
    grooming_risk: float = 0.0
    manipulation_risk: float = 0.0
    deception_risk: float = 0.0
    hostility_risk: float = 0.0

    # Aggregate scores
    overall_risk: float = 0.0
    weighted_risk: float = 0.0

    # Risk categorization
    risk_level: str = "low"  # low, moderate, high, critical
    primary_concern: Optional[str] = None
    secondary_concerns: List[str] = field(default_factory=list)

    # Behavioral indicators
    escalation_risk: float = 0.0
    recidivism_risk: float = 0.0
    immediate_danger: bool = False

    # Confidence and reliability
    assessment_confidence: float = 0.0
    data_completeness: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    intervention_priority: str = "routine"  # routine, urgent, emergency


@dataclass
class SpeakerRiskProfile:
    """Risk profile for individual speaker"""
    speaker_id: str = ""
    risk_scores: RiskAssessment = field(default_factory=RiskAssessment)
    pattern_history: List[Dict] = field(default_factory=list)
    risk_trajectory: str = "stable"  # improving, stable, worsening
    behavioral_patterns: Dict[str, int] = field(default_factory=dict)
    interaction_risks: Dict[str, float] = field(default_factory=dict)  # Risk with specific others


class BehavioralRiskScorer:
    """Comprehensive behavioral risk assessment system"""

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.6,
        "critical": 0.8
    }

    # Component weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "grooming": 0.3,
        "manipulation": 0.3,
        "hostility": 0.2,
        "deception": 0.2
    }

    # Escalation indicators
    ESCALATION_PATTERNS = [
        "increasing_severity",
        "decreasing_intervals",
        "expanding_tactics",
        "resistance_to_boundaries",
        "punishment_for_compliance_failure"
    ]

    # Immediate danger indicators
    DANGER_KEYWORDS = [
        "kill", "suicide", "hurt", "harm", "weapon", "gun", "knife",
        "police", "emergency", "help me", "save me", "trapped"
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize risk scorer

        Args:
            weights: Custom risk component weights
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.confidence_scorer = ConfidenceScorer()
        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Risk weights sum to {total}, normalizing to 1.0")
            # Normalize weights
            self.weights = {k: v/total for k, v in self.weights.items()}

    def assess_risk(self,
                   grooming_analysis: Optional[Any] = None,
                   manipulation_analysis: Optional[Any] = None,
                   deception_analysis: Optional[Any] = None,
                   sentiment_analysis: Optional[Any] = None,
                   intent_analysis: Optional[Any] = None,
                   message_text: Optional[str] = None) -> RiskAssessment:
        """Perform comprehensive risk assessment

        Args:
            grooming_analysis: Grooming detection results
            manipulation_analysis: Manipulation detection results
            deception_analysis: Deception analysis results
            sentiment_analysis: Sentiment analysis results
            intent_analysis: Intent classification results
            message_text: Raw message text for danger keyword detection

        Returns:
            RiskAssessment: Comprehensive risk assessment
        """
        assessment = RiskAssessment()

        # Calculate component risks
        if grooming_analysis:
            assessment.grooming_risk = self._calculate_grooming_risk(grooming_analysis)

        if manipulation_analysis:
            assessment.manipulation_risk = self._calculate_manipulation_risk(manipulation_analysis)

        if deception_analysis:
            assessment.deception_risk = self._calculate_deception_risk(deception_analysis)

        # Calculate hostility risk from sentiment and intent
        assessment.hostility_risk = self._calculate_hostility_risk(
            sentiment_analysis, intent_analysis
        )

        # Check for immediate danger
        if message_text:
            assessment.immediate_danger = self._check_immediate_danger(message_text)

        # Calculate weighted overall risk
        assessment.weighted_risk = self._calculate_weighted_risk(assessment)

        # Calculate simple average for comparison
        components = [
            assessment.grooming_risk,
            assessment.manipulation_risk,
            assessment.deception_risk,
            assessment.hostility_risk
        ]
        non_zero_components = [c for c in components if c > 0]
        assessment.overall_risk = statistics.mean(non_zero_components) if non_zero_components else 0

        # Use the higher of weighted and simple average
        assessment.overall_risk = max(assessment.weighted_risk, assessment.overall_risk)

        # Determine risk level
        assessment.risk_level = self._determine_risk_level(assessment.overall_risk)

        # Identify primary and secondary concerns
        self._identify_concerns(assessment)

        # Calculate escalation risk
        assessment.escalation_risk = self._calculate_escalation_risk(assessment)

        # Calculate confidence
        assessment.assessment_confidence = self._calculate_confidence(assessment)

        # Determine intervention priority
        assessment.intervention_priority = self._determine_intervention_priority(assessment)

        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(assessment)

        # Suggest resources
        assessment.resources = self._suggest_resources(assessment)

        return assessment

    def _calculate_grooming_risk(self, analysis: Any) -> float:
        """Calculate grooming risk score

        Args:
            analysis: Grooming analysis object

        Returns:
            float: Grooming risk (0-1)
        """
        if not analysis:
            return 0.0

        # Base risk on overall score from grooming detector
        base_risk = analysis.overall_score if hasattr(analysis, 'overall_score') else 0

        # Adjust for stage progression
        if hasattr(analysis, 'timeline_progression'):
            stages = len(analysis.timeline_progression)
            if stages >= 4:  # Multiple stages present
                base_risk *= 1.3
            elif stages >= 2:
                base_risk *= 1.1

        # Adjust for pattern count
        if hasattr(analysis, 'pattern_count'):
            if analysis.pattern_count > 10:
                base_risk *= 1.2

        return min(1.0, base_risk)

    def _calculate_manipulation_risk(self, analysis: Any) -> float:
        """Calculate manipulation risk score

        Args:
            analysis: Manipulation analysis object

        Returns:
            float: Manipulation risk (0-1)
        """
        if not analysis:
            return 0.0

        risk = 0.0

        # Base risk on overall score
        if hasattr(analysis, 'overall_score'):
            risk = analysis.overall_score

        # Factor in emotional harm
        if hasattr(analysis, 'emotional_harm_score'):
            risk = max(risk, analysis.emotional_harm_score * 0.8)

        # Factor in coercion
        if hasattr(analysis, 'coercion_score'):
            risk = max(risk, analysis.coercion_score * 0.9)

        # Factor in reality distortion (gaslighting)
        if hasattr(analysis, 'reality_distortion_score'):
            risk = max(risk, analysis.reality_distortion_score)

        return min(1.0, risk)

    def _calculate_deception_risk(self, analysis: Any) -> float:
        """Calculate deception risk score

        Args:
            analysis: Deception analysis object

        Returns:
            float: Deception risk (0-1)
        """
        if not analysis:
            return 0.0

        # Base risk on overall deception score
        risk = analysis.overall_deception_score if hasattr(analysis, 'overall_deception_score') else 0

        # Adjust based on credibility assessment
        if hasattr(analysis, 'credibility_assessment'):
            if analysis.credibility_assessment == "deceptive":
                risk = max(risk, 0.7)
            elif analysis.credibility_assessment == "questionable":
                risk = max(risk, 0.4)

        return min(1.0, risk)

    def _calculate_hostility_risk(self, sentiment: Any, intent: Any) -> float:
        """Calculate hostility risk from sentiment and intent

        Args:
            sentiment: Sentiment analysis results
            intent: Intent classification results

        Returns:
            float: Hostility risk (0-1)
        """
        risk = 0.0

        # Factor in negative sentiment
        if sentiment and hasattr(sentiment, 'combined_sentiment'):
            if sentiment.combined_sentiment < -0.5:
                risk += 0.4
            elif sentiment.combined_sentiment < -0.2:
                risk += 0.2

            # Check for anger emotion
            if hasattr(sentiment, 'emotions') and 'anger' in sentiment.emotions:
                risk += sentiment.emotions['anger'] * 0.3

        # Factor in hostile intent
        if intent:
            if hasattr(intent, 'primary_intent'):
                if intent.primary_intent == "conflictive":
                    risk += 0.3
                elif intent.primary_intent in ["coercive", "controlling"]:
                    risk += 0.4

            # Check communication style
            if hasattr(intent, 'communication_style'):
                if intent.communication_style == "aggressive":
                    risk += 0.3
                elif intent.communication_style == "passive_aggressive":
                    risk += 0.2

        return min(1.0, risk)

    def _check_immediate_danger(self, text: str) -> bool:
        """Check for immediate danger keywords

        Args:
            text: Message text

        Returns:
            bool: True if immediate danger detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.DANGER_KEYWORDS)

    def _calculate_weighted_risk(self, assessment: RiskAssessment) -> float:
        """Calculate weighted overall risk

        Args:
            assessment: Risk assessment

        Returns:
            float: Weighted risk score
        """
        weighted_sum = (
            assessment.grooming_risk * self.weights.get("grooming", 0.25) +
            assessment.manipulation_risk * self.weights.get("manipulation", 0.25) +
            assessment.deception_risk * self.weights.get("deception", 0.25) +
            assessment.hostility_risk * self.weights.get("hostility", 0.25)
        )

        # Apply multipliers for compound risks
        active_components = sum([
            1 for risk in [
                assessment.grooming_risk,
                assessment.manipulation_risk,
                assessment.deception_risk,
                assessment.hostility_risk
            ] if risk > 0.3
        ])

        if active_components >= 3:
            weighted_sum *= 1.2  # Multiple risk factors present
        elif active_components >= 2:
            weighted_sum *= 1.1

        return min(1.0, weighted_sum)

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score

        Args:
            risk_score: Overall risk score

        Returns:
            str: Risk level
        """
        for level in ["critical", "high", "moderate", "low"]:
            if risk_score >= self.RISK_THRESHOLDS[level]:
                return level
        return "low"

    def _identify_concerns(self, assessment: RiskAssessment):
        """Identify primary and secondary concerns

        Args:
            assessment: Risk assessment
        """
        concerns = [
            ("grooming", assessment.grooming_risk),
            ("manipulation", assessment.manipulation_risk),
            ("deception", assessment.deception_risk),
            ("hostility", assessment.hostility_risk)
        ]

        # Sort by risk level
        concerns.sort(key=lambda x: x[1], reverse=True)

        # Identify significant concerns (above threshold)
        significant = [c for c in concerns if c[1] > 0.3]

        if significant:
            assessment.primary_concern = significant[0][0]
            if len(significant) > 1:
                assessment.secondary_concerns = [c[0] for c in significant[1:]]

    def _calculate_escalation_risk(self, assessment: RiskAssessment) -> float:
        """Calculate risk of escalation

        Args:
            assessment: Risk assessment

        Returns:
            float: Escalation risk (0-1)
        """
        risk = 0.0

        # High overall risk suggests escalation potential
        if assessment.overall_risk > 0.6:
            risk += 0.3

        # Multiple concerns suggest complexity and escalation potential
        if len(assessment.secondary_concerns) >= 2:
            risk += 0.2

        # Specific concerning combinations
        if assessment.grooming_risk > 0.5 and assessment.manipulation_risk > 0.5:
            risk += 0.3  # Grooming + manipulation is concerning

        if assessment.hostility_risk > 0.5 and assessment.manipulation_risk > 0.5:
            risk += 0.2  # Hostility + manipulation suggests control

        # Immediate danger is highest escalation risk
        if assessment.immediate_danger:
            risk = max(risk, 0.9)

        return min(1.0, risk)

    def _calculate_confidence(self, assessment: RiskAssessment) -> float:
        """Calculate confidence in assessment using ensemble confidence scoring

        Args:
            assessment: Risk assessment

        Returns:
            float: Confidence (0-1)
        """
        # Build detection dictionary for ensemble confidence
        detections = {}

        if assessment.grooming_risk > 0:
            detections['grooming'] = assessment.grooming_risk
        if assessment.manipulation_risk > 0:
            detections['manipulation'] = assessment.manipulation_risk
        if assessment.deception_risk > 0:
            detections['deception'] = assessment.deception_risk
        if assessment.hostility_risk > 0:
            detections['hostility'] = assessment.hostility_risk

        if not detections:
            return 0.0

        # Use ensemble confidence scoring
        confidence_score = self.confidence_scorer.calculate_ensemble_confidence(detections)

        # Calculate data completeness (how many risk components were analyzed)
        total_components = 4  # grooming, manipulation, deception, hostility
        analyzed_components = len(detections)
        assessment.data_completeness = analyzed_components / total_components

        return confidence_score.overall_confidence

    def _determine_intervention_priority(self, assessment: RiskAssessment) -> str:
        """Determine intervention priority

        Args:
            assessment: Risk assessment

        Returns:
            str: Priority level
        """
        if assessment.immediate_danger:
            return "emergency"

        if assessment.risk_level == "critical" or assessment.escalation_risk > 0.7:
            return "urgent"

        if assessment.risk_level == "high":
            return "priority"

        return "routine"

    def _generate_recommendations(self, assessment: RiskAssessment) -> List[str]:
        """Generate actionable recommendations

        Args:
            assessment: Risk assessment

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        # Priority-based recommendations
        if assessment.intervention_priority == "emergency":
            recommendations.append("EMERGENCY: Immediate intervention required.")
            recommendations.append("Contact emergency services if in immediate danger.")
            recommendations.append("Reach out to crisis hotline for immediate support.")

        elif assessment.intervention_priority == "urgent":
            recommendations.append("URGENT: High-risk situation detected.")
            recommendations.append("Seek professional support within 24-48 hours.")
            recommendations.append("Document all concerning interactions.")
            recommendations.append("Create a safety plan with trusted contacts.")

        elif assessment.intervention_priority == "priority":
            recommendations.append("PRIORITY: Significant risk factors present.")
            recommendations.append("Schedule consultation with mental health professional.")
            recommendations.append("Establish clear boundaries in communication.")

        # Concern-specific recommendations
        if assessment.primary_concern == "grooming":
            recommendations.append("Grooming patterns detected: Maintain connections with trusted friends/family.")
            recommendations.append("Be cautious about sharing personal information.")
            recommendations.append("Trust your instincts if something feels wrong.")

        elif assessment.primary_concern == "manipulation":
            recommendations.append("Manipulation tactics identified: Keep records of interactions.")
            recommendations.append("Seek external validation for your experiences.")
            recommendations.append("Consider limiting contact with manipulative individuals.")

        elif assessment.primary_concern == "hostility":
            recommendations.append("High hostility detected: Prioritize your safety.")
            recommendations.append("Avoid engaging in escalating arguments.")
            recommendations.append("Consider involving a mediator or counselor.")

        elif assessment.primary_concern == "deception":
            recommendations.append("Deception markers found: Verify important claims independently.")
            recommendations.append("Document inconsistencies for future reference.")

        # General safety recommendations
        if assessment.overall_risk > 0.5:
            recommendations.append("Consider creating a safety plan.")
            recommendations.append("Keep evidence of concerning behavior.")
            recommendations.append("Don't hesitate to seek help when needed.")

        return recommendations

    def _suggest_resources(self, assessment: RiskAssessment) -> List[str]:
        """Suggest relevant resources

        Args:
            assessment: Risk assessment

        Returns:
            List[str]: Resource suggestions
        """
        resources = []

        # Crisis resources for high-risk situations
        if assessment.intervention_priority in ["emergency", "urgent"]:
            resources.append("National Suicide Prevention Lifeline: 988")
            resources.append("Crisis Text Line: Text HOME to 741741")
            resources.append("National Domestic Violence Hotline: 1-800-799-7233")

        # Concern-specific resources
        if assessment.primary_concern == "grooming":
            resources.append("RAINN National Sexual Assault Hotline: 1-800-656-4673")
            resources.append("Childhelp National Child Abuse Hotline: 1-800-422-4453")

        elif assessment.primary_concern == "manipulation":
            resources.append("National Dating Abuse Helpline: 1-866-331-9474")
            resources.append("Psychology Today Therapist Directory: psychologytoday.com")

        # General resources
        if assessment.overall_risk > 0.3:
            resources.append("SAMHSA National Helpline: 1-800-662-4357")
            resources.append("Local mental health services: Call 211 for information")

        return resources

    def analyze_conversation_risks(self, messages: List[Dict[str, Any]],
                                  analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risks across entire conversation

        Args:
            messages: List of messages
            analyses: Dictionary containing various analysis results

        Returns:
            Dict: Comprehensive conversation risk analysis
        """
        results = {
            "overall_risk_assessment": None,
            "per_speaker_risks": {},
            "risk_timeline": [],
            "risk_trajectory": "stable",
            "interaction_risks": {},
            "high_risk_episodes": [],
            "cumulative_harm_score": 0.0,
            "recovery_indicators": [],
            "system_recommendations": [],
            "monitoring_priorities": []
        }

        # Track risks over time
        risk_scores = []
        speaker_risks = {}

        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'Unknown')

            # Get relevant analyses for this message
            msg_analyses = self._extract_message_analyses(i, analyses)

            # Assess risk for this message
            assessment = self.assess_risk(
                grooming_analysis=msg_analyses.get('grooming'),
                manipulation_analysis=msg_analyses.get('manipulation'),
                deception_analysis=msg_analyses.get('deception'),
                sentiment_analysis=msg_analyses.get('sentiment'),
                intent_analysis=msg_analyses.get('intent'),
                message_text=msg.get('text', '')
            )

            risk_scores.append(assessment.overall_risk)

            # Track by speaker
            if sender not in speaker_risks:
                speaker_risks[sender] = []
            speaker_risks[sender].append(assessment)

            # Track high-risk episodes
            if assessment.risk_level in ["high", "critical"]:
                results["high_risk_episodes"].append({
                    "index": i,
                    "sender": sender,
                    "risk_level": assessment.risk_level,
                    "primary_concern": assessment.primary_concern,
                    "immediate_danger": assessment.immediate_danger
                })

        # Calculate overall risk assessment
        if risk_scores:
            overall = RiskAssessment()
            overall.overall_risk = statistics.mean(risk_scores)
            overall.risk_level = self._determine_risk_level(overall.overall_risk)

            # Find primary concern across conversation
            all_concerns = {}
            for assessments in speaker_risks.values():
                for a in assessments:
                    if a.primary_concern:
                        all_concerns[a.primary_concern] = all_concerns.get(a.primary_concern, 0) + 1

            if all_concerns:
                overall.primary_concern = max(all_concerns.items(), key=lambda x: x[1])[0]

            results["overall_risk_assessment"] = overall

        # Analyze per-speaker risks
        for sender, assessments in speaker_risks.items():
            if assessments:
                speaker_profile = SpeakerRiskProfile()
                speaker_profile.speaker_id = sender

                # Average risk scores
                speaker_profile.risk_scores.grooming_risk = statistics.mean(
                    [a.grooming_risk for a in assessments]
                )
                speaker_profile.risk_scores.manipulation_risk = statistics.mean(
                    [a.manipulation_risk for a in assessments]
                )
                speaker_profile.risk_scores.deception_risk = statistics.mean(
                    [a.deception_risk for a in assessments]
                )
                speaker_profile.risk_scores.hostility_risk = statistics.mean(
                    [a.hostility_risk for a in assessments]
                )
                speaker_profile.risk_scores.overall_risk = statistics.mean(
                    [a.overall_risk for a in assessments]
                )

                speaker_profile.risk_scores.risk_level = self._determine_risk_level(
                    speaker_profile.risk_scores.overall_risk
                )

                # Determine trajectory
                if len(assessments) >= 3:
                    recent = statistics.mean([a.overall_risk for a in assessments[-3:]])
                    earlier = statistics.mean([a.overall_risk for a in assessments[:3]])

                    if recent > earlier * 1.2:
                        speaker_profile.risk_trajectory = "worsening"
                    elif recent < earlier * 0.8:
                        speaker_profile.risk_trajectory = "improving"

                results["per_speaker_risks"][sender] = speaker_profile

        # Determine conversation risk trajectory
        if len(risk_scores) >= 3:
            recent_third = statistics.mean(risk_scores[-len(risk_scores)//3:])
            early_third = statistics.mean(risk_scores[:len(risk_scores)//3])

            if recent_third > early_third * 1.2:
                results["risk_trajectory"] = "escalating"
            elif recent_third < early_third * 0.8:
                results["risk_trajectory"] = "de-escalating"

        # Generate system recommendations
        results["system_recommendations"] = self._generate_system_recommendations(results)

        return results

    def _extract_message_analyses(self, index: int, analyses: Dict) -> Dict:
        """Extract relevant analyses for a specific message

        Args:
            index: Message index
            analyses: All analyses

        Returns:
            Dict: Relevant analyses for this message
        """
        extracted = {}

        # Extract from various analysis results
        # (This is simplified - actual implementation would map to specific message indices)

        return extracted

    def _generate_system_recommendations(self, results: Dict) -> List[str]:
        """Generate system-level recommendations

        Args:
            results: Conversation risk analysis

        Returns:
            List[str]: System recommendations
        """
        recommendations = []

        # Overall risk recommendations
        if results.get("overall_risk_assessment"):
            overall = results["overall_risk_assessment"]

            if overall.risk_level == "critical":
                recommendations.append("CRITICAL RISK: Immediate professional intervention recommended.")
                recommendations.append("Consider involving appropriate authorities if safety is at risk.")

            elif overall.risk_level == "high":
                recommendations.append("HIGH RISK: Professional support strongly advised.")
                recommendations.append("Implement safety measures and support systems.")

        # Trajectory-based recommendations
        if results.get("risk_trajectory") == "escalating":
            recommendations.append("WARNING: Risk levels are increasing over time.")
            recommendations.append("Early intervention recommended to prevent further escalation.")

        # Speaker-specific recommendations
        for sender, profile in results.get("per_speaker_risks", {}).items():
            if profile.risk_scores.risk_level in ["critical", "high"]:
                recommendations.append(f"Monitor interactions with {sender} closely.")

                if profile.risk_trajectory == "worsening":
                    recommendations.append(f"Risk from {sender} is increasing - consider boundaries.")

        return recommendations