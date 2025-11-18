"""
Pipeline Passes Package

Modular pass implementations for the 15-pass unified analysis pipeline.

Pass Groups:
- normalization_passes: Passes 1-3 (Data validation, sentiment, emotional dynamics)
- behavioral_passes: Passes 4-6 (Grooming, manipulation, deception detection)
- communication_passes: Passes 7-8 (Intent classification, risk assessment)
- timeline_passes: Passes 9-10 (Timeline analysis, contextual insights)
- person_passes: Passes 11-15 (Person-centric analysis and interventions)
"""

from .base_pass import BasePass, PassResult, PassGroup
from .pass_registry import PassRegistry, PassMetadata

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

__all__ = [
    # Base classes
    'BasePass',
    'PassResult',
    'PassGroup',
    'PassRegistry',
    'PassMetadata',

    # Normalization passes
    'Pass1_DataValidation',
    'Pass2_SentimentAnalysis',
    'Pass3_EmotionalDynamics',

    # Behavioral passes
    'Pass4_GroomingDetection',
    'Pass5_ManipulationDetection',
    'Pass6_DeceptionAnalysis',

    # Communication passes
    'Pass7_IntentClassification',
    'Pass8_RiskAssessment',

    # Timeline passes
    'Pass9_TimelineAnalysis',
    'Pass10_ContextualInsights',

    # Person passes
    'Pass11_PersonIdentification',
    'Pass12_InteractionMapping',
    'Pass13_GaslightingDetection',
    'Pass14_RelationshipAnalysis',
    'Pass15_InterventionRecommendations',
]
