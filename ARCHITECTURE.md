# Unified 15-Pass Pipeline Architecture

## System Overview

The unified pipeline represents a complete integration of two analytical frameworks:
1. **Original 10-Pass System:** Message analysis with behavioral pattern detection
2. **ppl_int 5-Pass System:** Person-centric analysis from ppl_int

The result is a comprehensive 15-pass psychological analysis system for text conversations.

---

## Module Structure

### Core Pipeline Module: `src/pipeline/unified_processor.py`

**Class: `UnifiedProcessor`**
- Main orchestrator for all 15 passes
- Manages module initialization and coordination
- Coordinates data flow between passes
- Handles result aggregation and export

**Class: `UnifiedAnalysisResult`**
- Dataclass containing all 15-pass results
- Structured output with clear separation of concerns
- Supports JSON serialization for export
- Backward-compatible fields

**Key Methods:**
```python
process_file(input_file, output_dir) -> UnifiedAnalysisResult
  - Executes complete 15-pass pipeline
  - Returns comprehensive analysis result

_pass_1_validate_and_normalize(input_file)
_pass_2_sentiment_analysis(messages)
_pass_3_emotional_dynamics(messages, sentiment_results)
_pass_4_grooming_detection(messages)
_pass_5_manipulation_detection(messages)
_pass_6_deception_analysis(messages)
_pass_7_intent_classification(messages)
_pass_8_risk_assessment(messages, analyses)
_pass_9_timeline_analysis(messages, risk_assessment)
_pass_10_contextual_insights(messages, sentiment_results, timeline_analysis)
_pass_11_person_identification(messages, df)
_pass_12_interaction_mapping(messages, person_identification)
_pass_13_gaslighting_detection(messages, person_identification, manipulation_results)
_pass_14_relationship_analysis(messages, person_identification, interaction_mapping)
_pass_15_intervention_recommendations(...)
```

---

### Person Analysis Module: `src/nlp/person_analyzer.py`

**Class: `PersonAnalyzer`**
- Implements passes 11-15
- Person-centric analysis functionality
- Relationship dynamics assessment

**Key Methods:**

```python
identify_persons_in_conversation(messages, df) -> Dict
  - Pass 11: Identify and classify persons
  - Returns: person list with roles and characteristics

extract_interaction_patterns(messages, persons) -> Dict
  - Pass 12: Map interactions and relationships
  - Returns: directed interactions and network structure

detect_gaslighting_patterns(messages, persons, manipulation_results) -> Dict
  - Pass 13: Detect gaslighting indicators
  - Returns: 5-category gaslighting analysis with perpetrator/victim IDs

assess_relationship_dynamics(messages, persons) -> Dict
  - Pass 14: Analyze relationship quality and power
  - Returns: comprehensive relationship assessment

generate_intervention_recommendations(...) -> Dict
  - Pass 15: Generate clinical recommendations
  - Returns: prioritized intervention strategies
```

**Data Classes:**
```python
@dataclass
class Person:
  - name: str
  - aliases: List[str]
  - message_count: int
  - role: str
  - characteristics: Dict

@dataclass
class Interaction:
  - speaker: str
  - recipient: str
  - message_indices: List[int]
  - interaction_type: str
  - power_direction: Optional[str]
```

---

## Integration Points

### 1. Existing NLP Modules Integration

The unified pipeline leverages existing modules:

```
UnifiedProcessor
    ├─ SentimentAnalyzer (Pass 2-3)
    ├─ GroomingDetector (Pass 4)
    ├─ ManipulationDetector (Pass 5)
    ├─ DeceptionAnalyzer (Pass 6)
    ├─ IntentClassifier (Pass 7)
    ├─ BehavioralRiskScorer (Pass 8)
    └─ PersonAnalyzer (Passes 11-15)
```

Each module is independently initialized and called with consistent message format:
```python
messages: List[Dict] = [
    {
        'index': int,
        'text': str,
        'sender': str,
        'timestamp': str,
        'date': str,
        'time': str,
        ... (additional fields from CSV)
    }
]
```

### 2. Database Integration

**For PostgreSQL:**
```python
class UnifiedEnhancedMessageProcessor(UnifiedProcessor):
  - Extends UnifiedProcessor
  - Adds PostgreSQL adapter
  - Uses PostgreSQLAdapter for persistence
```

**Database Artifacts:**
- Messages stored in import-specific tables
- Patterns indexed by analysis_run_id
- Analysis runs tracked with metadata
- Queryable results for historical analysis

### 3. Configuration System Integration

Uses existing `ConfigManager`:
```python
config = ConfigManager().load_config("preset_name")
config.nlp.enable_grooming_detection = True/False
config.nlp.risk_weight_grooming = 0.25
config.database.enable_caching = True/False
```

Supports presets:
- `quick_analysis` - Fast, essential only
- `deep_analysis` - All passes enabled
- `clinical_report` - Therapeutic focus
- `legal_report` - Forensic focus

---

## Data Flow Architecture

### Input Normalization (Pass 1)
```
CSV File
    ↓
CSVValidator.validate_file()
    ↓
DataFrame
    ↓
_dataframe_to_messages()
    ↓
Normalized Message List
```

### Sentiment Pipeline (Passes 2-3)
```
Messages
    ↓
SentimentAnalyzer
    ├─ analyze_text() → per-message sentiment
    ├─ analyze_conversation() → conversation-level analysis
    └─ (VADER, TextBlob, NRCLex engines)
    ↓
SentimentResult + ConversationSentiment
    ↓
_pass_3_emotional_dynamics()
    ├─ Calculate volatility
    ├─ Detect emotion shifts
    └─ Assess consistency
    ↓
Emotional Dynamics Dict
```

### Behavioral Pattern Detection (Passes 4-6)
```
Messages
    ├─ GroomingDetector.analyze_conversation()
    │   └─ Pattern matching → Grooming Results
    │
    ├─ ManipulationDetector.analyze_conversation()
    │   └─ Tactic identification → Manipulation Results
    │
    └─ DeceptionAnalyzer.analyze_conversation()
        └─ Credibility assessment → Deception Results

All Results → Aggregation for Pass 8
```

### Person-Centric Analysis (Passes 11-15)
```
Messages + DataFrame
    ↓
[Pass 11] identify_persons_in_conversation()
    ├─ Count messages per sender
    ├─ Classify roles
    ├─ Extract characteristics
    └─ → Person Identification
    ↓
[Pass 12] extract_interaction_patterns()
    ├─ Map directed interactions
    ├─ Analyze network structure
    ├─ Assess communication balance
    └─ → Interaction Mapping
    ↓
[Pass 13] detect_gaslighting_patterns()
    ├─ 5-category phrase matching
    ├─ Perpetrator identification
    ├─ Victim identification
    └─ → Gaslighting Detection
    ↓
[Pass 14] assess_relationship_dynamics()
    ├─ Power dynamic analysis
    ├─ Control pattern detection
    ├─ Emotional pattern tracking
    └─ → Relationship Analysis
    ↓
[Pass 15] generate_intervention_recommendations()
    ├─ Risk-based recommendations
    ├─ Case formulation
    ├─ Resource identification
    └─ → Intervention Recommendations
```

---

## Result Aggregation

### Per-Pass Results Structure

Each pass generates a dictionary result:

```python
results = {
    'data_validation': Dict,           # Pass 1
    'sentiment_results': Dict,         # Pass 2
    'emotional_dynamics': Dict,        # Pass 3
    'grooming_results': Dict,          # Pass 4
    'manipulation_results': Dict,      # Pass 5
    'deception_results': Dict,         # Pass 6
    'intent_results': Dict,            # Pass 7
    'risk_assessment': Dict,           # Pass 8
    'timeline_analysis': Dict,         # Pass 9
    'contextual_insights': Dict,       # Pass 10
    'person_identification': Dict,     # Pass 11
    'interaction_mapping': Dict,       # Pass 12
    'gaslighting_detection': Dict,     # Pass 13
    'relationship_analysis': Dict,     # Pass 14
    'intervention_recommendations': Dict # Pass 15
}
```

### Aggregation Methods

```python
_aggregate_concerns() -> List[str]
  - Consolidates all concerns from all passes
  - Removes duplicates
  - Prioritizes by severity

_aggregate_recommendations() -> List[str]
  - Combines recommendations from all passes
  - Focuses on actionable items
  - Includes resource references
```

### Export Formats

**JSON Export:**
- Complete unmodified results from all passes
- Preserves dataclass objects (converted to dicts)
- Timestamp included
- Suitable for archival and re-analysis

**CSV Summary Export:**
- One row per pass
- Pass number, description, status
- Key metrics truncated for viewing
- Suitable for quick review

**PostgreSQL Storage:**
- Messages stored in dedicated tables
- Patterns indexed and queryable
- Analysis runs tracked
- Suitable for longitudinal studies

---

## Key Design Decisions

### 1. Sequential Pass Execution
- Passes execute in order (1 → 15)
- Each pass can use results from previous passes
- Allows for dependency chains without circular references
- Simplifies debugging and result tracking

### 2. Modular NLP Integration
- Existing NLP modules kept unchanged
- New PersonAnalyzer module for passes 11-15
- Consistent message format for all modules
- Minimal coupling between modules

### 3. Backward Compatibility
- Legacy 10-pass system still available
- Same configuration system
- Same export formats
- Command-line switches for both modes

### 4. Person-Centric Enhancement
- Built on top of existing behavioral analysis
- Adds relationship and role perspective
- Enhances clinical utility
- Enables victim/perpetrator identification

### 5. Flexible Output
- Multiple export formats
- Optional PostgreSQL persistence
- Extensible architecture for new exporters
- Support for both clinical and forensic use

---

## Performance Optimization

### Parallelization Opportunities
```python
# Potential parallelization points:
1. Per-message sentiment analysis (currently sequential)
2. NLP module execution (currently sequential)
3. Multiple file processing (batch mode)
```

### Memory Management
```python
# Current approach:
- Keep all messages in memory (optimal for < 10K messages)
- Streaming option available for large datasets
- Database caching for repeated analyses

# Optimization for large files:
- Process in chunks
- Use generators instead of lists
- Stream to database
```

### Database Query Optimization
```python
# Current indexes:
- analysis_run_id (primary)
- message_id (secondary)
- pattern_type (tertiary)
- timestamp (for time-series queries)
```

---

## Error Handling Strategy

### Validation Level
```python
Pass 1: Validates input CSV
  → If invalid, fail immediately with error details
```

### Module Failure Handling
```python
Each NLP module wrapped in try-except
  → If module unavailable, continue with empty results
  → Log warnings but don't halt pipeline
```

### Result Validation
```python
Before export, validate:
  - All required fields present
  - No circular references
  - JSON-serializable format
```

---

## Testing Strategy

### Unit Tests
```
test_unified_processor.py
  - Test each pass individually
  - Test pass combinations
  - Test error scenarios

test_person_analyzer.py
  - Test person identification
  - Test interaction mapping
  - Test gaslighting detection
  - Test relationship analysis
```

### Integration Tests
```
test_integration.py
  - Test full pipeline
  - Test with real CSV data
  - Test with PostgreSQL backend
  - Test with SQLite backend
```

### Regression Tests
```
- Ensure 10-pass legacy system still works
- Compare unified results with legacy for passes 1-10
- Verify backward compatibility
```

---

## Extensibility Points

### Adding New Passes
1. Implement analysis function in appropriate module
2. Add method to UnifiedProcessor (_pass_N_description)
3. Call in process_file() at appropriate position
4. Add result to UnifiedAnalysisResult dataclass
5. Update documentation

### Adding New Export Formats
1. Implement exporter class with standard interface
2. Add to _export_results() method
3. Register format in format mapping
4. Update documentation

### Adding New NLP Modules
1. Implement module following existing patterns
2. Accept List[Dict] of messages
3. Return Dict of results
4. Initialize in UnifiedProcessor.__init__()
5. Call at appropriate pass

---

## Deployment Considerations

### Development Environment
- SQLite backend (local storage)
- All passes enabled
- Verbose logging
- Fast feedback loop

### Production Environment
- PostgreSQL backend (central storage)
- Selective pass enabling based on use case
- Performance monitoring
- Archival strategy

### Clinical Deployment
- All passes enabled
- Clinical report preset
- Therapist review workflow
- HIPAA compliance (if applicable)

### Forensic Deployment
- All passes enabled
- Legal report preset
- Chain of custody tracking
- Detailed documentation

---

## Version Control Strategy

### Current Implementation
- Version 1.0: Initial unified pipeline
- Fully backward compatible
- No breaking changes to existing API

### Future Versions
- Version 2.0: Performance optimizations
- Version 3.0: Additional passes (projected)
- Version 4.0: Advanced analytics (projected)

---

## Security and Privacy

### Data Handling
- No data transmission during processing (local execution)
- Optional PostgreSQL persistence (secure connection recommended)
- Results can be encrypted before storage
- No external API calls required

### Credential Management
- PostgreSQL credentials in config file (should be in environment vars in production)
- Recommend: Use environment variables for credentials
- Consider: Encryption of stored credentials

### Audit Trail
- All analysis runs logged to database
- Timestamp of each analysis
- Configuration used for each run
- Results linked to analysis_run_id

---

## Documentation

### User Documentation
- PIPELINE_DOCUMENTATION.md (this file)
- Command-line examples
- Python API examples
- Configuration guide

### Developer Documentation
- ARCHITECTURE.md (this file)
- Module docstrings
- Function docstrings
- Type hints for IDE support

### Clinical Documentation
- Clinical use cases
- Interpretation guide
- Limitations and caveats
- Ethical considerations

---

## References

### Original Components
- 10-Pass Message Processor (Dev-Root/src/pipeline/)
- ppl_int 5-Pass Processor (Data Store/ppl_int/core/)

### External Libraries
- VADER: github.com/cjhutto/vaderSentiment
- TextBlob: github.com/sloria/TextBlob
- NRCLex: github.com/pjrvs/nrclex
- NLTK: github.com/nltk/nltk

### Related Standards
- VADER Sentiment Analysis
- TextBlob Sentiment API
- NRC Emotion Lexicon
- Standard psychological assessment frameworks
