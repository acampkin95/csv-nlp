"""
Pipeline Passes Package

Modular pass implementations for the 15-pass unified analysis pipeline.

Pass Groups:
- normalization_passes: Passes 1-3 (Data validation, sentiment, emotional dynamics)
- behavioral_passes: Passes 4-6 (Grooming, manipulation, deception detection)
- communication_passes: Passes 7-8 (Intent classification, risk assessment)
- timeline_passes: Passes 9-10 (Timeline analysis, contextual insights)
- person_passes: Passes 11-15 (Person-centric analysis and interventions)

Advanced Features:
- Pass Factory: Easy pass instantiation and registry creation
- Metrics System: Comprehensive execution metrics and statistics
- Hooks System: Before/after execution hooks with conditional execution
- Profiler: Detailed performance profiling (CPU, memory, I/O)
- Configuration: Flexible pass configuration with presets
- Visualizer: Execution timelines, graphs, and HTML reports
- Comparator: Result comparison and regression testing
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

# Import advanced features
from .pass_factory import PassFactory
from .pass_metrics import (
    PassMetricsCollector,
    PassExecutionMetrics,
    PipelineMetrics,
    get_metrics_collector
)
from .pass_hooks import (
    PassHookManager,
    HookType,
    HookContext,
    ConditionalExecutionManager,
    get_hook_manager
)
from .pass_profiler import (
    PassProfiler,
    ProfileData,
    get_profiler
)
from .pass_config import (
    PassConfig,
    PipelineConfig,
    PassConfigManager,
    get_config_manager
)
from .pass_visualizer import (
    PassVisualizer,
    get_visualizer
)
from .pass_comparator import (
    PassResultComparator,
    get_comparator
)
from .confidence_framework import (
    ConfidenceLevel,
    Evidence,
    Finding,
    ConfidenceScorer,
    EvidenceAggregator,
    get_evidence_aggregator
)
from .advanced_passes import (
    Pass16_LanguagePatternAnalysis,
    Pass17_CrossValidation,
    Pass18_PatternCorrelation,
    Pass19_AnomalyDetection,
    Pass20_FinalConfidenceAssessment
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

    # Advanced features
    'PassFactory',
    'PassMetricsCollector',
    'PassExecutionMetrics',
    'PipelineMetrics',
    'get_metrics_collector',
    'PassHookManager',
    'HookType',
    'HookContext',
    'ConditionalExecutionManager',
    'get_hook_manager',
    'PassProfiler',
    'ProfileData',
    'get_profiler',
    'PassConfig',
    'PipelineConfig',
    'PassConfigManager',
    'get_config_manager',
    'PassVisualizer',
    'get_visualizer',
    'PassResultComparator',
    'get_comparator',

    # Confidence framework
    'ConfidenceLevel',
    'Evidence',
    'Finding',
    'ConfidenceScorer',
    'EvidenceAggregator',
    'get_evidence_aggregator',

    # Advanced passes (16-20)
    'Pass16_LanguagePatternAnalysis',
    'Pass17_CrossValidation',
    'Pass18_PatternCorrelation',
    'Pass19_AnomalyDetection',
    'Pass20_FinalConfidenceAssessment',
]
