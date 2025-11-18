# High-Confidence Analysis System (20-Pass Pipeline)

## Overview

The High-Confidence Analysis System extends the modular 15-pass pipeline to a comprehensive 20-pass system with evidence-based confidence scoring. This system provides quantifiable confidence metrics for all analysis results, backed by cross-validation and multi-source evidence aggregation.

## Key Features

### 1. Evidence-Based Confidence Scoring
- **Multiple Confidence Levels**: VERY_HIGH (90-100%), HIGH (75-89%), MEDIUM (50-74%), LOW (25-49%), VERY_LOW (0-24%)
- **Evidence Tracking**: Each finding is backed by traceable evidence from multiple passes
- **Automatic Confidence Calculation**: Confidence scores are calculated based on:
  - Number of evidence items supporting the finding
  - Data quality metrics
  - Cross-validation results
  - Source reliability

### 2. Five New Advanced Passes (16-20)

#### Pass 16: Language Pattern Analysis
**Purpose**: Deep linguistic analysis of communication patterns

**Analyzes**:
- Vocabulary sophistication (unique words, rare words, lexical diversity)
- Sentence complexity (avg length, word length, complexity score)
- Formality levels (formal vs informal language patterns)
- Communication style shifts (detecting changes in writing style)

**Dependencies**: Pass 2 (Sentiment), Pass 11 (Person Identification)

**Output**:
```python
{
    'vocabulary_analysis': {
        'unique_word_ratio': float,  # 0.0-1.0
        'rare_words_ratio': float,   # Percentage of uncommon words
        'lexical_diversity': float   # Type-token ratio
    },
    'complexity_analysis': {
        'avg_sentence_length': float,
        'avg_word_length': float,
        'complexity_score': float    # 0.0-1.0
    },
    'formality_analysis': {
        'formality_score': float,    # 0.0-1.0 (0=informal, 1=formal)
        'formality_level': str       # 'very_informal', 'informal', 'neutral', 'formal', 'very_formal'
    },
    'style_shifts': [
        {
            'message_index': int,
            'shift_type': str,       # 'vocabulary', 'complexity', 'formality'
            'shift_magnitude': float,
            'evidence': Evidence
        }
    ]
}
```

#### Pass 17: Cross-Validation
**Purpose**: Validate findings across multiple independent passes

**Validates**:
- Grooming detection results (Pass 4)
- Manipulation detection results (Pass 5)
- Deception analysis results (Pass 6)
- Gaslighting detection results (Pass 13)

**Dependencies**: Pass 4, 5, 6, 13

**Output**:
```python
{
    'consistency_score': float,      # 0.0-1.0 overall consistency
    'corroboration': [
        {
            'finding_type': str,
            'supporting_passes': List[int],
            'confidence': float,
            'evidence': Evidence
        }
    ],
    'contradictions': [
        {
            'finding_type': str,
            'conflicting_passes': List[int],
            'description': str
        }
    ],
    'validated_findings': int,
    'total_findings': int
}
```

#### Pass 18: Pattern Correlation
**Purpose**: Identify correlations between different pattern types

**Analyzes**:
- Temporal correlations (e.g., grooming patterns coinciding with emotional volatility)
- Behavioral correlations (manipulation tactics co-occurring with deception)
- Pattern sequences (escalation patterns, typical attack sequences)

**Dependencies**: Pass 3 (Emotional Dynamics), Pass 4 (Grooming), Pass 9 (Timeline)

**Output**:
```python
{
    'temporal_correlations': [
        {
            'pattern1': str,
            'pattern2': str,
            'correlation_strength': float,  # 0.0-1.0
            'time_lag': float,              # In hours/days
            'occurrences': int
        }
    ],
    'behavioral_correlations': [
        {
            'behavior1': str,
            'behavior2': str,
            'co_occurrence_rate': float,
            'significance': str
        }
    ],
    'pattern_sequences': [
        {
            'sequence': List[str],
            'frequency': int,
            'confidence': float
        }
    ]
}
```

#### Pass 19: Anomaly Detection
**Purpose**: Statistical detection of unusual patterns and outliers

**Detects**:
- Message length anomalies (unusually long/short messages)
- Sentiment anomalies (extreme or unexpected sentiment)
- Timing anomalies (unusual message timing patterns)

**Method**: Z-score based statistical analysis with configurable thresholds

**Dependencies**: Pass 2 (Sentiment), Pass 9 (Timeline)

**Output**:
```python
{
    'length_anomalies': [
        {
            'message_index': int,
            'length': int,
            'z_score': float,
            'severity': str  # 'moderate', 'high', 'severe'
        }
    ],
    'sentiment_anomalies': [
        {
            'message_index': int,
            'sentiment_score': float,
            'z_score': float,
            'severity': str
        }
    ],
    'timing_anomalies': [
        {
            'message_index': int,
            'time_gap': float,  # In hours
            'z_score': float,
            'severity': str
        }
    ],
    'anomaly_score': float,      # 0.0-1.0 overall anomaly severity
    'confidence': float          # Confidence in anomaly detection
}
```

#### Pass 20: Final Confidence Assessment
**Purpose**: Generate final high-confidence assessment with complete evidence summary

**Aggregates**:
- All evidence from passes 1-19
- Cross-validation results
- Pattern correlations
- Anomaly detection results
- Risk assessment (Pass 8)

**Output**:
```python
{
    'final_risk_level': str,         # 'minimal', 'low', 'moderate', 'high', 'critical'
    'final_confidence': float,       # 0.0-1.0 confidence in assessment
    'confidence_level': str,         # 'very_high', 'high', 'medium', 'low', 'very_low'
    'evidence_summary': {
        'total_evidence_items': int,
        'high_confidence_items': int,
        'evidence_by_type': Dict[str, int],
        'evidence_by_source': Dict[int, int]
    },
    'confidence_report': {
        'findings_by_confidence': {
            'very_high': int,
            'high': int,
            'medium': int,
            'low': int,
            'very_low': int
        },
        'average_confidence': float,
        'cross_validated_findings': int
    },
    'final_recommendations': [
        {
            'priority': str,         # 'immediate', 'high', 'medium', 'low'
            'action': str,
            'evidence': List[Evidence],
            'confidence': float
        }
    ]
}
```

## Confidence Framework

### Evidence Class
```python
@dataclass
class Evidence:
    source_pass: int              # Which pass generated this evidence
    source_name: str              # Pass name
    finding_type: str             # Type of finding
    description: str              # Human-readable description
    confidence: float             # 0.0-1.0 confidence score
    timestamp: str                # When evidence was collected
    supporting_data: Dict[str, Any]  # Additional data supporting the evidence
```

### Finding Class
```python
@dataclass
class Finding:
    finding_id: str               # Unique identifier
    finding_type: str             # Type of finding
    description: str              # Description
    severity: str                 # 'low', 'medium', 'high', 'critical'
    confidence: float             # Overall confidence (auto-calculated)
    confidence_level: ConfidenceLevel  # Enum level
    evidence_items: List[Evidence]     # Supporting evidence
    cross_validated: bool         # Whether cross-validated

    def add_evidence(self, evidence: Evidence):
        """Add evidence and recalculate confidence"""
        self.evidence_items.append(evidence)
        self._recalculate_confidence()
```

### ConfidenceScorer
```python
class ConfidenceScorer:
    def score_finding(
        self,
        finding_type: str,
        supporting_evidence: List[Evidence],
        data_quality: float = 1.0,
        cross_validated: bool = False
    ) -> float:
        """
        Calculate confidence score based on:
        - Number of evidence items (more = higher confidence)
        - Average confidence of evidence
        - Data quality
        - Cross-validation bonus (+10% if validated)
        """

    def aggregate_confidence(
        self,
        findings: List[Finding],
        aggregation_method: str = 'weighted_average'
    ) -> float:
        """
        Aggregate confidence from multiple findings.
        Methods: 'weighted_average', 'conservative', 'optimistic'
        """
```

### EvidenceAggregator
```python
class EvidenceAggregator:
    def add_evidence(self, evidence: Evidence):
        """Collect evidence from any pass"""

    def create_finding(
        self,
        finding_id: str,
        finding_type: str,
        description: str,
        severity: str
    ) -> Finding:
        """Create finding with relevant evidence"""

    def get_findings_by_confidence(
        self,
        min_confidence: float = 0.0
    ) -> List[Finding]:
        """Get findings above confidence threshold"""

    def generate_confidence_report(self) -> Dict[str, Any]:
        """Generate comprehensive confidence analysis"""
```

## Usage Examples

### Example 1: Using the 20-Pass System
```python
from src.pipeline.passes import PassFactory, get_evidence_aggregator

# Initialize factory with all analyzers
factory = PassFactory(
    csv_validator=csv_validator,
    sentiment_analyzer=sentiment_analyzer,
    grooming_detector=grooming_detector,
    manipulation_detector=manipulation_detector,
    deception_analyzer=deception_analyzer,
    intent_classifier=intent_classifier,
    risk_scorer=risk_scorer,
    person_analyzer=person_analyzer,
    cache_manager=cache_manager
)

# Create registry with all 20 passes
registry = factory.create_registry()

# Execute all passes
results = registry.execute_all(
    messages=messages,
    file_path=file_path
)

# Get final confidence assessment (Pass 20)
final_assessment = results[20]
print(f"Risk Level: {final_assessment.data['final_risk_level']}")
print(f"Confidence: {final_assessment.data['final_confidence']:.2%}")
print(f"Confidence Level: {final_assessment.data['confidence_level']}")

# Get evidence aggregator
aggregator = get_evidence_aggregator()
confidence_report = aggregator.generate_confidence_report()
print(f"Total Evidence Items: {confidence_report['total_evidence_items']}")
print(f"High Confidence Findings: {confidence_report['high_confidence_findings']}")
```

### Example 2: Creating Custom Findings with Evidence
```python
from src.pipeline.passes import get_evidence_aggregator, Evidence, ConfidenceLevel

# Get global aggregator
aggregator = get_evidence_aggregator()

# Add evidence from a pass
evidence = Evidence(
    source_pass=4,
    source_name="Grooming Detection",
    finding_type="grooming_pattern",
    description="Detected isolation tactic in message #42",
    confidence=0.85,
    supporting_data={'message_id': 42, 'tactic': 'isolation'}
)
aggregator.add_evidence(evidence)

# Create finding with evidence
finding = aggregator.create_finding(
    finding_id="grooming_001",
    finding_type="grooming_pattern",
    description="Multiple grooming tactics detected",
    severity="high"
)

print(f"Finding Confidence: {finding.confidence:.2%}")
print(f"Confidence Level: {finding.confidence_level.value}")
print(f"Evidence Count: {len(finding.evidence_items)}")
```

### Example 3: Enabling Only High-Confidence Passes
```python
# Enable passes 1-20 (full high-confidence system)
enabled_passes = list(range(1, 21))
registry = factory.create_registry(enabled_passes=enabled_passes)

# Or enable specific analysis groups
registry = factory.create_registry(enabled_passes=[
    1, 2,          # Core validation and sentiment
    4, 5, 6,       # Behavioral analysis
    8,             # Risk assessment
    16, 17, 18, 19, 20  # Advanced high-confidence analysis
])
```

### Example 4: Filtering by Confidence Level
```python
from src.pipeline.passes import get_evidence_aggregator, ConfidenceLevel

aggregator = get_evidence_aggregator()

# Get only high-confidence findings
high_confidence = aggregator.get_findings_by_confidence(min_confidence=0.75)
print(f"High Confidence Findings: {len(high_confidence)}")

# Get findings by confidence level
very_high = [f for f in aggregator.findings.values()
             if f.confidence_level == ConfidenceLevel.VERY_HIGH]
print(f"Very High Confidence Findings: {len(very_high)}")
```

## Architecture

### Pass Dependencies
```
Pass 1: Data Validation
Pass 2: Sentiment Analysis
Pass 3: Emotional Dynamics ← [2]
Pass 4: Grooming Detection
Pass 5: Manipulation Detection
Pass 6: Deception Analysis
Pass 7: Intent Classification
Pass 8: Risk Assessment ← [4, 5, 6, 7]
Pass 9: Timeline Analysis ← [8]
Pass 10: Contextual Insights ← [2, 9]
Pass 11: Person Identification
Pass 12: Interaction Mapping ← [11]
Pass 13: Gaslighting Detection ← [5, 11]
Pass 14: Relationship Analysis ← [11]
Pass 15: Intervention Recommendations ← [8, 11, 13, 14]
Pass 16: Language Pattern Analysis ← [2, 11]
Pass 17: Cross-Validation ← [4, 5, 6, 13]
Pass 18: Pattern Correlation ← [3, 4, 9]
Pass 19: Anomaly Detection ← [2, 9]
Pass 20: Final Confidence Assessment ← [8, 17, 18]
```

### Parallel Execution Groups
- **Group 1**: Passes 4, 5, 6 (Behavioral analysis - can run in parallel)
- **Group 2**: Passes 12, 13, 14 (Person-centric analysis - can run in parallel)
- **Group 3**: Passes 18, 19 (Pattern correlation and anomaly detection - can run in parallel)

## Performance Considerations

### Memory Usage
The evidence aggregator maintains all evidence in memory. For large datasets:
- Consider periodic cleanup of low-confidence evidence
- Use evidence filtering to keep only relevant items
- Enable pass result caching to avoid recomputation

### Execution Time
The 20-pass system adds approximately 15-20% overhead compared to the 15-pass system:
- Pass 16 (Language Analysis): ~5% overhead
- Pass 17 (Cross-Validation): ~3% overhead
- Pass 18 (Pattern Correlation): ~4% overhead
- Pass 19 (Anomaly Detection): ~2% overhead
- Pass 20 (Final Assessment): ~3% overhead

### Optimization Tips
1. **Use caching**: Enable pass result caching for repeated analyses
2. **Parallel execution**: Leverage parallel groups (1, 2, 3)
3. **Selective passes**: Only enable passes 16-20 when high confidence is needed
4. **Evidence pruning**: Clear evidence aggregator between analyses

## Configuration

### Enabling High-Confidence System
```python
from src.pipeline.passes import PassConfig, PipelineConfig, get_config_manager

config_manager = get_config_manager()

# Create high-confidence preset
config = config_manager.create_preset_config('thorough')
config.enabled_passes = list(range(1, 21))  # Enable all 20 passes
config.profiling_enabled = True
config.metrics_enabled = True

# Enable all passes
for i in range(1, 21):
    config.set_pass_enabled(i, True)

# Save configuration
config_manager.configs['high_confidence'] = config
config_manager.save_to_file('high_confidence', 'configs/high_confidence.json')
```

### Environment Variables
```bash
# Enable all passes
export PIPELINE_ENABLED_PASSES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# Enable profiling for performance analysis
export PIPELINE_PROFILING_ENABLED=true

# Enable metrics collection
export PIPELINE_METRICS_ENABLED=true
```

## Migration from 15-Pass to 20-Pass System

### Backward Compatibility
The 20-pass system is **fully backward compatible** with the 15-pass system:
- Existing code using passes 1-15 continues to work unchanged
- New passes (16-20) are optional and can be enabled selectively
- Evidence framework is opt-in (no impact if not used)

### Migration Steps
1. **Update imports** (optional):
   ```python
   from src.pipeline.passes import (
       get_evidence_aggregator,
       ConfidenceLevel,
       Pass16_LanguagePatternAnalysis,
       Pass17_CrossValidation,
       Pass18_PatternCorrelation,
       Pass19_AnomalyDetection,
       Pass20_FinalConfidenceAssessment
   )
   ```

2. **Enable new passes** (optional):
   ```python
   registry = factory.create_registry(
       enabled_passes=list(range(1, 21))  # Instead of range(1, 16)
   )
   ```

3. **Use evidence framework** (optional):
   ```python
   aggregator = get_evidence_aggregator()
   confidence_report = aggregator.generate_confidence_report()
   ```

## Testing

### Unit Tests
```python
import unittest
from src.pipeline.passes import (
    Evidence, Finding, ConfidenceLevel,
    ConfidenceScorer, EvidenceAggregator
)

class TestConfidenceFramework(unittest.TestCase):
    def test_evidence_creation(self):
        evidence = Evidence(
            source_pass=4,
            source_name="Test",
            finding_type="test",
            description="Test evidence",
            confidence=0.8,
            supporting_data={}
        )
        self.assertEqual(evidence.confidence, 0.8)

    def test_finding_confidence_calculation(self):
        finding = Finding(
            finding_id="test",
            finding_type="test",
            description="Test",
            severity="medium",
            confidence=0.0,
            evidence_items=[]
        )

        # Add evidence
        finding.add_evidence(Evidence(...))
        finding.add_evidence(Evidence(...))

        # Confidence should be recalculated
        self.assertGreater(finding.confidence, 0.0)
```

### Integration Tests
```python
def test_20_pass_system():
    """Test complete 20-pass execution"""
    factory = PassFactory(...)
    registry = factory.create_registry(enabled_passes=list(range(1, 21)))

    results = registry.execute_all(messages=test_messages)

    # Verify all 20 passes executed
    assert len(results) == 20

    # Verify final assessment
    final = results[20]
    assert 'final_confidence' in final.data
    assert 'evidence_summary' in final.data

    # Verify evidence aggregation
    aggregator = get_evidence_aggregator()
    report = aggregator.generate_confidence_report()
    assert report['total_evidence_items'] > 0
```

## Best Practices

1. **Always use evidence tracking**: Add evidence for significant findings
2. **Cross-validate important findings**: Use Pass 17 to validate critical detections
3. **Monitor confidence levels**: Pay attention to confidence scores
4. **Use Pass 20 for final decisions**: Always check the final confidence assessment
5. **Clear evidence between analyses**: Call `aggregator.clear()` between different datasets
6. **Profile performance**: Enable profiling to identify bottlenecks
7. **Validate configurations**: Use `config_manager.validate_config()` before execution

## Future Enhancements

- **Machine learning confidence models**: Train models to predict confidence more accurately
- **Adaptive thresholds**: Dynamically adjust confidence thresholds based on data characteristics
- **Evidence visualization**: Generate visual evidence graphs
- **Confidence explainability**: Detailed explanations for confidence scores
- **Real-time confidence updates**: Stream confidence updates during analysis
- **Multi-language support**: Extend language analysis to multiple languages

## Support and Documentation

- **Modularization Guide**: See `MODULARIZATION_GUIDE.md` for pipeline architecture
- **Performance Report**: See `PERFORMANCE_OPTIMIZATION_REPORT.md` for optimization details
- **Security Report**: See `SECURITY_IMPLEMENTATION_REPORT.md` for security features

## Changelog

### Version 2.0 (Current)
- Added 5 new passes (16-20) for high-confidence analysis
- Implemented evidence-based confidence framework
- Added cross-validation pass
- Implemented pattern correlation analysis
- Added statistical anomaly detection
- Created final confidence assessment pass
- Extended factory to support 20 passes
- Updated documentation

### Version 1.0
- Initial 15-pass modular system
- 5 advanced features (factory, metrics, hooks, profiler, config, visualizer, comparator)
- Performance optimizations
- Security hardening
