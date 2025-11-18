#!/usr/bin/env python3
"""
Pass Factory

Factory for creating and configuring pipeline passes with all dependencies.
Simplifies pass instantiation and provides centralized configuration.
"""

import logging
from typing import Dict, Any, Optional, List

from .base_pass import BasePass, PassGroup
from .pass_registry import PassRegistry

# Import all pass classes
from .normalization_passes import (
    Pass1_DataValidation,
    Pass2_SentimentAnalysis,
    Pass3_EmotionalDynamics
)
from .behavioral_passes import (
    Pass4_GroomingDetection,
    Pass5_ManipulationDetection,
    Pass6_DeceptionAnalysis
)
from .communication_passes import (
    Pass7_IntentClassification,
    Pass8_RiskAssessment
)
from .timeline_passes import (
    Pass9_TimelineAnalysis,
    Pass10_ContextualInsights
)
from .person_passes import (
    Pass11_PersonIdentification,
    Pass12_InteractionMapping,
    Pass13_GaslightingDetection,
    Pass14_RelationshipAnalysis,
    Pass15_InterventionRecommendations
)

logger = logging.getLogger(__name__)


class PassFactory:
    """
    Factory for creating pipeline passes with proper configuration.

    Handles:
    - Pass instantiation with dependencies
    - Configuration injection
    - Registry population
    - Parallel group assignment
    """

    def __init__(
        self,
        csv_validator,
        sentiment_analyzer,
        grooming_detector,
        manipulation_detector,
        deception_analyzer,
        intent_classifier,
        risk_scorer,
        person_analyzer,
        cache_manager=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pass factory with all required analyzers.

        Args:
            csv_validator: CSV validator instance
            sentiment_analyzer: Sentiment analysis engine
            grooming_detector: Grooming detection engine
            manipulation_detector: Manipulation detection engine
            deception_analyzer: Deception analysis engine
            intent_classifier: Intent classification engine
            risk_scorer: Risk scoring engine
            person_analyzer: Person analysis engine
            cache_manager: Optional cache manager for result caching
            config: Optional configuration dict for pass customization
        """
        self.csv_validator = csv_validator
        self.sentiment_analyzer = sentiment_analyzer
        self.grooming_detector = grooming_detector
        self.manipulation_detector = manipulation_detector
        self.deception_analyzer = deception_analyzer
        self.intent_classifier = intent_classifier
        self.risk_scorer = risk_scorer
        self.person_analyzer = person_analyzer
        self.cache_manager = cache_manager
        self.config = config or {}

    def create_registry(self, enabled_passes: Optional[List[int]] = None) -> PassRegistry:
        """
        Create a fully populated pass registry.

        Args:
            enabled_passes: Optional list of pass numbers to enable (None = all)

        Returns:
            PassRegistry with all configured passes
        """
        registry = PassRegistry()

        # Define all passes with their parallel groups
        pass_definitions = [
            (1, self._create_pass1, None),
            (2, self._create_pass2, None),
            (3, self._create_pass3, None),
            (4, self._create_pass4, 1),  # Parallel group 1
            (5, self._create_pass5, 1),  # Parallel group 1
            (6, self._create_pass6, 1),  # Parallel group 1
            (7, self._create_pass7, None),
            (8, self._create_pass8, None),
            (9, self._create_pass9, None),
            (10, self._create_pass10, None),
            (11, self._create_pass11, None),
            (12, self._create_pass12, 2),  # Parallel group 2
            (13, self._create_pass13, 2),  # Parallel group 2
            (14, self._create_pass14, 2),  # Parallel group 2
            (15, self._create_pass15, None),
        ]

        # Register enabled passes
        for pass_num, creator_func, parallel_group in pass_definitions:
            if enabled_passes is None or pass_num in enabled_passes:
                pass_instance = creator_func()
                registry.register(pass_instance, parallel_group=parallel_group)
                logger.debug(f"Registered {pass_instance}")

        # Validate dependencies
        if not registry.validate_dependencies():
            raise ValueError("Pass dependencies validation failed")

        logger.info(f"Created registry with {len(registry)} passes")
        return registry

    # Pass creator methods
    def _create_pass1(self) -> BasePass:
        """Create Pass 1: Data Validation"""
        return Pass1_DataValidation(
            csv_validator=self.csv_validator,
            cache_manager=self.cache_manager
        )

    def _create_pass2(self) -> BasePass:
        """Create Pass 2: Sentiment Analysis"""
        return Pass2_SentimentAnalysis(
            sentiment_analyzer=self.sentiment_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass3(self) -> BasePass:
        """Create Pass 3: Emotional Dynamics"""
        return Pass3_EmotionalDynamics(
            cache_manager=self.cache_manager
        )

    def _create_pass4(self) -> BasePass:
        """Create Pass 4: Grooming Detection"""
        return Pass4_GroomingDetection(
            grooming_detector=self.grooming_detector,
            cache_manager=self.cache_manager
        )

    def _create_pass5(self) -> BasePass:
        """Create Pass 5: Manipulation Detection"""
        return Pass5_ManipulationDetection(
            manipulation_detector=self.manipulation_detector,
            cache_manager=self.cache_manager
        )

    def _create_pass6(self) -> BasePass:
        """Create Pass 6: Deception Analysis"""
        return Pass6_DeceptionAnalysis(
            deception_analyzer=self.deception_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass7(self) -> BasePass:
        """Create Pass 7: Intent Classification"""
        return Pass7_IntentClassification(
            intent_classifier=self.intent_classifier,
            cache_manager=self.cache_manager
        )

    def _create_pass8(self) -> BasePass:
        """Create Pass 8: Risk Assessment"""
        return Pass8_RiskAssessment(
            risk_scorer=self.risk_scorer,
            cache_manager=self.cache_manager
        )

    def _create_pass9(self) -> BasePass:
        """Create Pass 9: Timeline Analysis"""
        return Pass9_TimelineAnalysis(
            cache_manager=self.cache_manager
        )

    def _create_pass10(self) -> BasePass:
        """Create Pass 10: Contextual Insights"""
        return Pass10_ContextualInsights(
            cache_manager=self.cache_manager
        )

    def _create_pass11(self) -> BasePass:
        """Create Pass 11: Person Identification"""
        return Pass11_PersonIdentification(
            person_analyzer=self.person_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass12(self) -> BasePass:
        """Create Pass 12: Interaction Mapping"""
        return Pass12_InteractionMapping(
            person_analyzer=self.person_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass13(self) -> BasePass:
        """Create Pass 13: Gaslighting Detection"""
        return Pass13_GaslightingDetection(
            person_analyzer=self.person_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass14(self) -> BasePass:
        """Create Pass 14: Relationship Analysis"""
        return Pass14_RelationshipAnalysis(
            person_analyzer=self.person_analyzer,
            cache_manager=self.cache_manager
        )

    def _create_pass15(self) -> BasePass:
        """Create Pass 15: Intervention Recommendations"""
        return Pass15_InterventionRecommendations(
            person_analyzer=self.person_analyzer,
            cache_manager=self.cache_manager
        )

    def create_custom_registry(self, pass_configs: List[Dict[str, Any]]) -> PassRegistry:
        """
        Create a custom registry with specific pass configurations.

        Args:
            pass_configs: List of dicts with 'pass_number', 'enabled', 'parallel_group'

        Returns:
            PassRegistry with custom configuration

        Example:
            pass_configs = [
                {'pass_number': 1, 'enabled': True},
                {'pass_number': 2, 'enabled': True},
                {'pass_number': 3, 'enabled': False},  # Skip pass 3
                # ...
            ]
        """
        registry = PassRegistry()

        for config in pass_configs:
            pass_num = config['pass_number']
            enabled = config.get('enabled', True)
            parallel_group = config.get('parallel_group')

            if not enabled:
                continue

            # Get creator function
            creator_func = getattr(self, f'_create_pass{pass_num}', None)
            if creator_func is None:
                logger.warning(f"No creator function for pass {pass_num}")
                continue

            pass_instance = creator_func()
            registry.register(pass_instance, parallel_group=parallel_group)

        if not registry.validate_dependencies():
            raise ValueError("Custom pass dependencies validation failed")

        return registry

    def get_pass_info(self) -> Dict[str, Any]:
        """
        Get information about all available passes.

        Returns:
            Dict with pass information
        """
        return {
            'total_passes': 15,
            'groups': {
                'normalization': [1, 2, 3],
                'behavioral': [4, 5, 6],
                'communication': [7, 8],
                'timeline': [9, 10],
                'person_centric': [11, 12, 13, 14, 15]
            },
            'parallel_groups': {
                1: [4, 5, 6],  # Behavioral passes
                2: [12, 13, 14]  # Person analysis passes
            },
            'dependencies': {
                3: [2],
                8: [4, 5, 6, 7],
                9: [8],
                10: [2, 9],
                12: [11],
                13: [5, 11],
                14: [11],
                15: [8, 11, 13, 14]
            }
        }
