#!/usr/bin/env python3
"""
Person-Centric Analysis Passes (Passes 11-15)

Pass 11: Person identification and role classification
Pass 12: Interaction mapping and relationship structure
Pass 13: Gaslighting-specific detection
Pass 14: Relationship dynamics and power analysis
Pass 15: Intervention recommendations and case formulation

Note: Passes 12-14 can run in parallel after Pass 11 completes.
"""

import logging
from typing import Dict, List, Any
import pandas as pd

from .base_pass import BasePass, PassGroup

logger = logging.getLogger(__name__)


class Pass11_PersonIdentification(BasePass):
    """Pass 11: Person identification and role classification"""

    def __init__(self, person_analyzer, cache_manager=None):
        super().__init__(
            pass_number=11,
            pass_name="Person Identification",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=[]
        )
        self.person_analyzer = person_analyzer

    def _execute_pass(self, messages: List[Dict], df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute person identification"""
        logger.info(f"  Identifying persons in conversation with {len(messages)} messages")
        result = self.person_analyzer.identify_persons_in_conversation(messages, df)
        print(f"  Persons Identified: {len(result.get('persons', []))}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for person identification failure"""
        return {
            'persons': [],
            'error': 'Person identification failed'
        }


class Pass12_InteractionMapping(BasePass):
    """Pass 12: Interaction mapping and relationship structure"""

    def __init__(self, person_analyzer, cache_manager=None):
        super().__init__(
            pass_number=12,
            pass_name="Interaction Mapping",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['person_identification']  # Depends on pass 11
        )
        self.person_analyzer = person_analyzer

    def _execute_pass(self, messages: List[Dict], person_identification: Dict, **kwargs) -> Dict[str, Any]:
        """Execute interaction mapping"""
        result = self.person_analyzer.extract_interaction_patterns(
            messages, person_identification.get('persons', [])
        )
        print(f"  Interaction Patterns Mapped: {len(result.get('interactions', []))}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for interaction mapping failure"""
        return {
            'interactions': [],
            'error': 'Interaction mapping failed'
        }


class Pass13_GaslightingDetection(BasePass):
    """Pass 13: Gaslighting-specific detection"""

    def __init__(self, person_analyzer, cache_manager=None):
        super().__init__(
            pass_number=13,
            pass_name="Gaslighting Detection",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['person_identification', 'manipulation_detection']  # Depends on passes 11 and 5
        )
        self.person_analyzer = person_analyzer

    def _execute_pass(
        self,
        messages: List[Dict],
        person_identification: Dict,
        manipulation_results: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute gaslighting detection"""
        result = self.person_analyzer.detect_gaslighting_patterns(
            messages, person_identification.get('persons', []), manipulation_results
        )
        risk_level = result.get('gaslighting_risk', 'low')
        print(f"  Gaslighting Risk: {risk_level}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for gaslighting detection failure"""
        return {
            'gaslighting_risk': 'unknown',
            'error': 'Gaslighting detection failed'
        }


class Pass14_RelationshipAnalysis(BasePass):
    """Pass 14: Relationship dynamics and power analysis"""

    def __init__(self, person_analyzer, cache_manager=None):
        super().__init__(
            pass_number=14,
            pass_name="Relationship Analysis",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['person_identification']  # Depends on pass 11
        )
        self.person_analyzer = person_analyzer

    def _execute_pass(
        self,
        messages: List[Dict],
        person_identification: Dict,
        interaction_mapping: Dict = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute relationship analysis"""
        result = self.person_analyzer.assess_relationship_dynamics(
            messages, person_identification.get('persons', [])
        )
        print(f"  Relationship Dynamics Analyzed")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for relationship analysis failure"""
        return {
            'error': 'Relationship analysis failed'
        }


class Pass15_InterventionRecommendations(BasePass):
    """Pass 15: Intervention recommendations and case formulation"""

    def __init__(self, person_analyzer, cache_manager=None):
        super().__init__(
            pass_number=15,
            pass_name="Intervention Recommendations",
            pass_group=PassGroup.PERSON_CENTRIC,
            cache_manager=cache_manager,
            dependencies=['risk_assessment', 'person_identification',
                          'relationship_analysis', 'gaslighting_detection']
        )
        self.person_analyzer = person_analyzer

    def _execute_pass(
        self,
        risk_assessment: Dict,
        person_identification: Dict,
        relationship_analysis: Dict,
        gaslighting_detection: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute intervention recommendations"""
        result = self.person_analyzer.generate_intervention_recommendations(
            risk_assessment, person_identification, relationship_analysis, gaslighting_detection
        )
        print(f"  Intervention Recommendations Generated: {len(result.get('recommendations', []))}")
        return result

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for intervention recommendations failure"""
        return {
            'recommendations': [],
            'error': 'Intervention recommendations failed'
        }
