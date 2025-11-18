# 15-Pass Unified Analysis Pipeline
## Comprehensive Message Analysis with Person-Centric Integration

### Overview

The 15-pass unified pipeline integrates the original 10-pass message analysis system with 5 additional passes focused on person-centric analysis. This creates a comprehensive analytical framework for detecting harmful patterns, relationship dynamics, and psychological concerns in conversations.

**Total Processing Passes: 15**
- **Passes 1-3:** Data Normalization & Sentiment Analysis
- **Passes 4-6:** Behavioral Pattern Detection
- **Passes 7-8:** Communication Analysis & Risk Assessment
- **Passes 9-10:** Timeline & Context Analysis
- **Passes 11-15:** Person-Centric Analysis (NEW)

---

## Pipeline Architecture

### Module Locations

```
src/
├── pipeline/
│   ├── unified_processor.py      # 15-pass unified pipeline orchestrator
│   └── message_processor.py      # Legacy 10-pass processor (backward compatible)
├── nlp/
│   ├── person_analyzer.py        # Person-centric analysis (NEW)
│   ├── sentiment_analyzer.py     # Sentiment analysis
│   ├── grooming_detector.py      # Grooming pattern detection
│   ├── manipulation_detector.py  # Manipulation pattern detection
│   ├── deception_analyzer.py     # Deception marker analysis
│   ├── intent_classifier.py      # Intent classification
│   └── risk_scorer.py            # Behavioral risk scoring
└── ... (other modules)
```

---

## Detailed Pass Descriptions

### PASSES 1-3: Data Normalization & Sentiment Analysis

#### Pass 1: CSV Validation and Data Normalization
**Purpose:** Ensure data quality and consistency before analysis

**Key Functions:**
- Validate CSV format, encoding, and structure
- Normalize message formatting (capitalization, timestamps)
- Identify and standardize sender names
- Check for missing or malformed data
- Auto-correct common formatting issues

**Output:**
```python
{
    'is_valid': bool,
    'encoding': str,
    'rows': int,
    'columns': int,
    'warnings': [list of validation warnings]
}
```

#### Pass 2: Sentiment Analysis
**Purpose:** Analyze emotional tone and sentiment across messages

**Key Functions:**
- VADER sentiment intensity analysis
- TextBlob polarity and subjectivity scoring
- NRCLex emotion classification
- Per-message and conversation-level aggregation
- Emotional intensity assessment

**Output:**
```python
{
    'per_message': [SentimentResult objects],
    'conversation': {
        'overall_sentiment': float (-1 to 1),
        'sentiment_trajectory': str,  # improving, declining, stable, volatile
        'emotional_volatility': float,
        'dominant_emotions': [list],
        'speaker_sentiments': {speaker: sentiment_data}
    }
}
```

#### Pass 3: Emotional Dynamics and Volatility
**Purpose:** Assess emotional consistency and shifts in conversation

**Key Functions:**
- Calculate sentiment volatility (standard deviation)
- Detect significant emotion shifts between messages
- Identify emotional escalation patterns
- Assess emotional consistency per speaker

**Output:**
```python
{
    'volatility': float,
    'sentiments': [list of sentiment scores],
    'emotion_shifts': [
        {
            'index': int,
            'from': float,
            'to': float,
            'magnitude': float
        }
    ]
}
```

---

### PASSES 4-6: Behavioral Pattern Detection

#### Pass 4: Grooming Pattern Detection
**Purpose:** Identify grooming-related behaviors and trust-building tactics

**Key Functions:**
- Pattern matching for grooming phrases
- Trust-building sequence analysis
- Victim isolation patterns
- Desensitization markers
- Age-related targeting (if applicable)

**Output:**
```python
{
    'overall_risk': str,  # low, moderate, high, critical
    'high_risk_messages': [list of concerning messages],
    'patterns_detected': [list of pattern types],
    'recommendations': [list of recommended actions]
}
```

#### Pass 5: Manipulation and Escalation Detection
**Purpose:** Identify manipulation tactics and escalation patterns

**Key Functions:**
- Detect gaslighting language patterns
- Identify blame-shifting tactics
- Escalation point analysis
- Coercion and threat detection
- Control pattern identification

**Output:**
```python
{
    'overall_risk': str,  # low, moderate, high, critical
    'escalation_points': [list of escalation indices],
    'tactics': [list of identified tactics],
    'severity_trend': str  # increasing, stable, decreasing
}
```

#### Pass 6: Deception Markers Analysis
**Purpose:** Assess credibility and identify deceptive language patterns

**Key Functions:**
- Pronoun usage analysis
- Verb tense inconsistency detection
- Evasive language identification
- Story consistency assessment
- Baseline deviation analysis

**Output:**
```python
{
    'overall_credibility': str,  # credible, questionable, deceptive
    'deception_markers': [list of markers],
    'confidence': float,
    'concern_areas': [list of areas]
}
```

---

### PASSES 7-8: Communication Analysis

#### Pass 7: Intent Classification
**Purpose:** Determine underlying communication intent

**Key Functions:**
- Conversational intent classification
- Dynamic assessment (cooperative, adversarial, neutral)
- Subtext analysis
- Hidden agenda detection
- Communication goal identification

**Output:**
```python
{
    'conversation_dynamic': str,  # cooperative, adversarial, mixed, neutral
    'per_message_intents': [list of intent classifications],
    'primary_intent': str,
    'secondary_intents': [list]
}
```

#### Pass 8: Behavioral Risk Scoring
**Purpose:** Comprehensive risk assessment combining all behavioral analyses

**Key Functions:**
- Per-message risk scoring
- Weighted risk aggregation
- Risk trajectory analysis
- Concern prioritization
- Intervention recommendation generation

**Output:**
```python
{
    'per_message_risks': [list of risk scores],
    'overall_risk_assessment': {
        'risk_level': str,  # low, moderate, high, critical
        'average_risk': float,
        'max_risk': float,
        'primary_concern': str,
        'recommendations': [list]
    }
}
```

---

### PASSES 9-10: Timeline & Context Analysis

#### Pass 9: Timeline Reconstruction and Pattern Sequencing
**Purpose:** Reconstruct conversation timeline and identify pattern sequences

**Key Functions:**
- Temporal ordering of events
- Pattern sequence identification
- Duration estimation
- Key moment extraction
- Chronological insight generation

**Output:**
```python
{
    'timeline_points': [
        {
            'index': int,
            'sender': str,
            'timestamp': str,
            'text': str
        }
    ],
    'conversation_duration': str,
    'pattern_sequences': [list of pattern sequences]
}
```

#### Pass 10: Contextual Insights and Conversation Flow
**Purpose:** Generate contextual understanding of conversation dynamics

**Key Functions:**
- Conversation flow analysis
- Context-aware interpretation
- Environmental factor consideration
- Communication style matching/mismatching
- Contextual risk adjustment

**Output:**
```python
{
    'insights': [list of contextual insights],
    'conversation_flow': str,  # simple, moderate, complex
    'context_factors': [list of relevant factors]
}
```

---

### PASSES 11-15: Person-Centric Analysis (NEW)

#### Pass 11: Person Identification and Role Classification
**Purpose:** Identify all individuals in conversation and classify their roles

**Key Functions:**
- Speaker identification and name standardization
- Alias detection and consolidation
- Role assignment (initiator, responder, victim, perpetrator, etc.)
- Speaker characteristic analysis
- Conversation type classification (dyadic, group, etc.)

**Output:**
```python
{
    'persons': [
        {
            'name': str,
            'aliases': [list],
            'message_count': int,
            'role': str,  # primary_initiator, primary_responder, participant, etc.
            'message_indices': [list],
            'characteristics': {
                'average_message_length': float,
                'aggression_score': float,
                'communication_style': str
            }
        }
    ],
    'person_map': {name: person_data},
    'total_speakers': int,
    'conversation_type': str,  # dyadic, small_group, large_group
    'primary_dyad': (str, str) or None
}
```

#### Pass 12: Interaction Mapping and Relationship Structure
**Purpose:** Map directed interactions and relationship structures between people

**Key Functions:**
- Directed interaction tracking (who talks to whom)
- Interaction type classification
- Network structure analysis
- Communication balance assessment
- Power flow identification

**Output:**
```python
{
    'interactions': [
        {
            'speaker': str,
            'recipient': str,
            'message_index': int,
            'type': str  # accusatory, defensive, questioning, cooperative, neutral
        }
    ],
    'interaction_matrix': {
        'speaker->recipient': {
            'count': int,
            'indices': [list],
            'types': [list]
        }
    },
    'pattern_analysis': {
        'asymmetrical_communication': dict,
        'accusation_patterns': [list],
        'defensive_patterns': [list]
    },
    'network_structure': {
        'centrality': {speaker: float},
        'connectivity': int
    },
    'communication_balance': {
        'speaker_distribution': dict,
        'imbalance_score': float,
        'is_balanced': bool
    }
}
```

#### Pass 13: Gaslighting-Specific Detection
**Purpose:** Identify and analyze gaslighting-specific manipulation patterns

**Key Functions:**
- Gaslighting indicator detection (5 categories):
  - Reality denial ("that didn't happen")
  - Blame shifting ("you caused this")
  - Trivializing ("you're overreacting")
  - Diverting ("why are you bringing up")
  - Countering ("you misunderstood")
- Perpetrator identification
- Victim identification
- Risk level assessment

**Output:**
```python
{
    'gaslighting_indicators': {
        'reality_denial': [list of instances],
        'blame_shifting': [list of instances],
        'trivializing': [list of instances],
        'diverting': [list of instances],
        'countering': [list of instances]
    },
    'total_indicators': int,
    'gaslighting_risk': str,  # low, moderate, high, critical
    'high_risk_instances': [list],
    'perpetrators': [(name, count), ...],
    'victims': [(name, count), ...]
}
```

#### Pass 14: Relationship Dynamics and Power Analysis
**Purpose:** Assess relationship quality, power structures, and control patterns

**Key Functions:**
- Power dynamic analysis (control statement detection)
- Emotional pattern analysis per person
- Dependency dynamic assessment
- Control pattern identification (isolation, emotional, financial, decision)
- Relationship type classification

**Output:**
```python
{
    'power_dynamics': {
        'control_statements': {speaker: int},
        'power_imbalance': bool,
        'severity': str,  # none, high, severe
        'dominant_person': str or None
    },
    'emotional_patterns': {
        'person_name': {
            'anger': int,
            'sadness': int,
            'fear': int,
            'love': int
        }
    },
    'dependency_dynamics': {
        'dependency_counts': dict,
        'has_dependency_dynamics': bool,
        'dependent_person': str or None
    },
    'control_patterns': {
        'isolation_attempts': [list],
        'emotional_control': [list],
        'financial_control': [list],
        'decision_control': [list]
    },
    'relationship_type': str,  # controlling, imbalanced, balanced
    'relationship_quality': str,  # poor, unhealthy, fair
    'power_imbalance': bool,
    'power_imbalance_severity': str
}
```

#### Pass 15: Intervention Recommendations and Case Formulation
**Purpose:** Generate clinical recommendations and comprehensive case formulation

**Key Functions:**
- Risk-based recommendation generation
- Gaslighting-specific intervention suggestions
- Power imbalance remediation strategies
- Resource identification
- Follow-up action planning
- Clinical impression formulation

**Output:**
```python
{
    'recommendations': [list of recommendations],
    'intervention_priority': str,  # routine, urgent
    'case_formulation': {
        'presentation': str,
        'primary_concerns': [list],
        'relationship_dynamics': str,
        'power_structure': str
    },
    'resources': [list of resource links/contacts],
    'follow_up_actions': [list of recommended follow-up steps],
    'summary': {
        'key_findings': [list],
        'clinical_impressions': str
    }
}
```

---

## Usage Guide

### Running the Unified Pipeline

#### Command Line Usage

**15-Pass Unified Pipeline:**
```bash
python message_processor.py input.csv --unified -o Reports/
```

**Legacy 10-Pass Pipeline:**
```bash
python message_processor.py input.csv -o Reports/
```

**With SQLite Backend:**
```bash
python message_processor.py input.csv --unified --use-sqlite -o Reports/
```

**Disable Specific Analyses:**
```bash
python message_processor.py input.csv --unified --no-grooming --no-deception -o Reports/
```

**Verbose Output:**
```bash
python message_processor.py input.csv --unified -v -o Reports/
```

### Python API Usage

#### Using the Unified Processor Directly

```python
from src.pipeline.unified_processor import UnifiedProcessor
from src.config.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("deep_analysis")

# Create processor
processor = UnifiedProcessor(config)

# Process file
result = processor.process_file("conversations.csv", output_dir="Reports")

# Access results
print(f"Risk Level: {result.overall_risk_level}")
print(f"Persons Identified: {result.person_identification['total_speakers']}")
print(f"Gaslighting Risk: {result.gaslighting_detection['gaslighting_risk']}")
print(f"Recommendations: {result.recommendations}")
```

#### Using the Enhanced Processor with PostgreSQL

```python
from message_processor import UnifiedEnhancedMessageProcessor
from src.config.config_manager import ConfigManager

config = ConfigManager().load_config()
processor = UnifiedEnhancedMessageProcessor(config, use_postgresql=True)
result = processor.process_csv_file("conversations.csv", "Reports")

print(f"Analysis ID: {result['analysis_run_id']}")
print(f"Stored in PostgreSQL: acdev.host/messagestore")
```

---

## Output Files

### JSON Output
**File:** `unified_analysis_{run_id}_{timestamp}.json`

Complete analysis results including all 15 passes with:
- Data validation results
- Sentiment analysis data
- Behavioral patterns detected
- Risk assessments
- Person identification and roles
- Interaction mapping
- Gaslighting indicators
- Relationship analysis
- Intervention recommendations

### CSV Summary
**File:** `unified_analysis_{run_id}_{timestamp}_summary.csv`

Summary table with:
- Pass number and description
- Completion status
- Key metrics for each pass

### PostgreSQL Storage
When using PostgreSQL backend:
- Messages stored in dedicated import tables
- Patterns indexed and queryable
- Analysis runs tracked with metadata
- Results retrievable for comparison

---

## Data Flow Diagram

```
Input CSV
    ↓
[Pass 1] Validation & Normalization
    ↓
[Pass 2-3] Sentiment & Emotional Analysis
    ↓
[Pass 4-6] Behavioral Pattern Detection
    ↓
[Pass 7-8] Communication & Risk Analysis
    ↓
[Pass 9-10] Timeline & Context Analysis
    ↓
[Pass 11-15] Person-Centric Analysis
    ├─ [11] Person Identification
    ├─ [12] Interaction Mapping
    ├─ [13] Gaslighting Detection
    ├─ [14] Relationship Dynamics
    └─ [15] Intervention Recommendations
    ↓
Result Aggregation & Export
    ├─ JSON Report
    ├─ CSV Summary
    └─ PostgreSQL Storage
```

---

## Key Features of the Unified Pipeline

### 1. Comprehensive Person-Centric Analysis
- Identifies all individuals in conversation
- Assigns roles (victim, perpetrator, mediator, etc.)
- Tracks individual characteristics and communication patterns
- Maps relationship structures

### 2. Advanced Gaslighting Detection
- 5-category gaslighting indicator framework
- Perpetrator and victim identification
- Evidence-based risk assessment
- Clinical recommendations

### 3. Relationship Dynamics Assessment
- Power imbalance detection
- Control pattern analysis (isolation, emotional, financial, decision)
- Emotional pattern tracking
- Dependency dynamic assessment

### 4. Backward Compatibility
- Legacy 10-pass pipeline still available
- Same configuration system
- Database compatibility
- Command-line switches for both modes

### 5. Flexible Output Options
- JSON for machine processing
- CSV for spreadsheet analysis
- PostgreSQL for longitudinal studies
- Extensible export framework

---

## Clinical and Research Applications

### Clinical Use Cases
1. **Psychotherapy Assessment:** Identify patterns for therapy
2. **Abuse Assessment:** Detect abusive relationship patterns
3. **Custody Evaluations:** Assess parent-child communication
4. **Forensic Analysis:** Document concerning patterns
5. **Group Dynamics:** Analyze group conversation patterns

### Research Applications
1. **Relationship Studies:** Longitudinal pattern analysis
2. **Psychological Research:** Data collection for studies
3. **Communication Research:** Analyze communication patterns
4. **Intervention Studies:** Pre/post assessment
5. **Training Data:** Create annotated datasets

---

## Configuration

### Default Configuration Presets

**quick_analysis:**
- Minimal processing
- Fast turnaround
- Essential risk assessment only

**deep_analysis:**
- All 15 passes enabled
- Comprehensive person-centric analysis
- Detailed report generation

**clinical_report:**
- Clinical-focused output
- Therapeutic recommendations
- Professional language

**legal_report:**
- Evidence-focused output
- Timeline reconstruction
- Documented findings

---

## Troubleshooting

### Common Issues

**Issue:** Import errors for NLP modules
```bash
# Solution: Install required dependencies
pip install vaderSentiment textblob nrclex nltk
```

**Issue:** PostgreSQL connection failure
```bash
# Solution: Use SQLite backend temporarily
python message_processor.py input.csv --use-sqlite --unified
```

**Issue:** Missing timestamps in data
```bash
# Solution: Pipeline handles missing timestamps gracefully
# Uses message order as timeline alternative
```

---

## Performance Considerations

### Processing Speed
- **10-50 messages:** < 1 second
- **50-100 messages:** 1-2 seconds
- **100-500 messages:** 2-5 seconds
- **500+ messages:** 5-30 seconds (depending on system)

### Memory Usage
- Base overhead: ~50MB
- Per 100 messages: ~10-20MB
- With PostgreSQL: Additional network overhead

### Optimization Tips
1. Use `--no-grooming`, `--no-manipulation`, `--no-deception` to skip unnecessary passes
2. Batch process multiple files
3. Use SQLite for local testing, PostgreSQL for production
4. Enable multiprocessing for large files (workers parameter)

---

## Version History

### Version 1.0 (Current)
- Initial release of unified 15-pass pipeline
- Integration of person-centric analysis
- Full backward compatibility with 10-pass system
- PostgreSQL and SQLite support

---

## Support and Documentation

- **Configuration Guide:** See `src/config/config_manager.py`
- **NLP Modules:** See individual module docstrings
- **API Reference:** See `src/pipeline/unified_processor.py`
- **Example Usage:** See main section of `message_processor.py`

---

## License and Attribution

This pipeline integrates:
- Original 10-pass analysis system
- ppl_int 5-pass person-centric analysis (Data Store)
- VADER, TextBlob, and NRCLex libraries for sentiment analysis
- PostgreSQL and SQLite backends

All components work together to provide comprehensive psychological analysis of text conversations.
