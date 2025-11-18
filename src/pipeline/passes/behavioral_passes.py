#!/usr/bin/env python3
"""
Behavioral Detection Passes (Passes 4-6)

Pass 4: Grooming pattern detection
Pass 5: Manipulation and escalation detection
Pass 6: Deception markers and credibility assessment

Note: These passes can run in parallel as they are independent.
"""

import logging
from typing import Dict, List, Any

from .base_pass import BasePass, PassGroup

logger = logging.getLogger(__name__)


class Pass4_GroomingDetection(BasePass):
    """Pass 4: Grooming pattern detection"""

    def __init__(self, grooming_detector, cache_manager=None):
        super().__init__(
            pass_number=4,
            pass_name="Grooming Detection",
            pass_group=PassGroup.BEHAVIORAL,
            cache_manager=cache_manager,
            dependencies=[]  # Independent, can run in parallel with 5-6
        )
        self.grooming_detector = grooming_detector

    def _execute_pass(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute grooming detection"""
        logger.info(f"  Analyzing {len(messages)} messages for grooming patterns")
        result = self.grooming_detector.analyze_conversation(messages)
        risk_level = result.get('overall_risk', 'low')
        print(f"  Grooming Risk: {risk_level}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for grooming detection failure"""
        return {
            'overall_risk': 'unknown',
            'error': 'Grooming detection failed'
        }


class Pass5_ManipulationDetection(BasePass):
    """Pass 5: Manipulation and escalation detection"""

    def __init__(self, manipulation_detector, cache_manager=None):
        super().__init__(
            pass_number=5,
            pass_name="Manipulation Detection",
            pass_group=PassGroup.BEHAVIORAL,
            cache_manager=cache_manager,
            dependencies=[]  # Independent, can run in parallel with 4, 6
        )
        self.manipulation_detector = manipulation_detector

    def _execute_pass(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute manipulation detection"""
        logger.info(f"  Analyzing {len(messages)} messages for manipulation patterns")
        result = self.manipulation_detector.analyze_conversation(messages)
        risk_level = result.get('overall_risk', 'low')
        print(f"  Manipulation Risk: {risk_level}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for manipulation detection failure"""
        return {
            'overall_risk': 'unknown',
            'error': 'Manipulation detection failed'
        }


class Pass6_DeceptionAnalysis(BasePass):
    """Pass 6: Deception markers and credibility assessment"""

    def __init__(self, deception_analyzer, cache_manager=None):
        super().__init__(
            pass_number=6,
            pass_name="Deception Analysis",
            pass_group=PassGroup.BEHAVIORAL,
            cache_manager=cache_manager,
            dependencies=[]  # Independent, can run in parallel with 4, 5
        )
        self.deception_analyzer = deception_analyzer

    def _execute_pass(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute deception analysis"""
        logger.info(f"  Analyzing {len(messages)} messages for deception markers")
        result = self.deception_analyzer.analyze_conversation(messages)
        credibility = result.get('overall_credibility', 'unknown')
        print(f"  Credibility Assessment: {credibility}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for deception analysis failure"""
        return {
            'overall_credibility': 'unknown',
            'error': 'Deception analysis failed'
        }
