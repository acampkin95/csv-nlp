#!/usr/bin/env python3
"""
Communication Analysis Passes (Passes 7-8)

Pass 7: Intent classification and conversation dynamics
Pass 8: Comprehensive behavioral risk assessment
"""

import logging
from typing import Dict, List, Any

from .base_pass import BasePass, PassGroup
from utils.performance import ProgressTracker

logger = logging.getLogger(__name__)


class Pass7_IntentClassification(BasePass):
    """Pass 7: Intent classification and conversation dynamics"""

    def __init__(self, intent_classifier, cache_manager=None):
        super().__init__(
            pass_number=7,
            pass_name="Intent Classification",
            pass_group=PassGroup.COMMUNICATION,
            cache_manager=cache_manager,
            dependencies=[]
        )
        self.intent_classifier = intent_classifier

    def _execute_pass(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute intent classification"""
        logger.info(f"  Classifying intents for {len(messages)} messages")
        result = self.intent_classifier.analyze_conversation_intents(messages)
        dynamic = result.get('conversation_dynamic', 'neutral')
        print(f"  Conversation Dynamic: {dynamic}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for intent classification failure"""
        return {
            'conversation_dynamic': 'unknown',
            'error': 'Intent classification failed'
        }


class Pass8_RiskAssessment(BasePass):
    """Pass 8: Comprehensive behavioral risk assessment"""

    def __init__(self, risk_scorer, cache_manager=None):
        super().__init__(
            pass_number=8,
            pass_name="Risk Assessment",
            pass_group=PassGroup.COMMUNICATION,
            cache_manager=cache_manager,
            dependencies=['grooming_detection', 'manipulation_detection',
                          'deception_analysis', 'intent_classification']
        )
        self.risk_scorer = risk_scorer

    def _execute_pass(self, messages: List[Dict], analyses: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute risk assessment with progress tracking"""
        per_message_risks = []
        progress = ProgressTracker(len(messages), desc="  Risk assessment", log_interval=20)

        for i, msg in enumerate(messages):
            try:
                risk = self.risk_scorer.assess_risk(
                    message_text=msg.get('text', '')
                )
                per_message_risks.append(risk)
            except Exception as e:
                logger.warning(f"  Risk assessment failed for message {i}: {e}")
                per_message_risks.append(None)

            progress.update(1)

        progress.finish()

        # Filter out None values for risk calculation
        valid_risks = [r for r in per_message_risks if r is not None]

        if valid_risks:
            avg_risk = sum(r.overall_risk for r in valid_risks) / len(valid_risks)
            max_risk = max(r.overall_risk for r in valid_risks)

            if max_risk > 0.8 or avg_risk > 0.6:
                risk_level = 'critical'
            elif max_risk > 0.6 or avg_risk > 0.4:
                risk_level = 'high'
            elif max_risk > 0.4 or avg_risk > 0.2:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
        else:
            risk_level = 'unknown'
            avg_risk = 0
            max_risk = 0

        print(f"  Overall Risk Level: {risk_level}")

        return {
            'per_message_risks': per_message_risks,
            'overall_risk_assessment': {
                'risk_level': risk_level,
                'average_risk': avg_risk,
                'max_risk': max_risk
            }
        }

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for risk assessment failure"""
        return {
            'per_message_risks': [],
            'overall_risk_assessment': {
                'risk_level': 'unknown',
                'average_risk': 0,
                'max_risk': 0
            },
            'error': 'Risk assessment failed'
        }
