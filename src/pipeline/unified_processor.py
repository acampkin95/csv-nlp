#!/usr/bin/env python3
"""
Unified 15-Pass Analysis Pipeline
Integrates 10-pass existing pipeline with 5-pass person-centric analysis from ppl_int.

OPTIMIZED VERSION with:
- Lazy loading of NLP modules
- Result caching between passes
- Batch processing
- Progress tracking
- Improved error handling

PASS STRUCTURE:
=================
Passes 1-3: Data Normalization & Sentiment
  Pass 1: CSV validation and data normalization
  Pass 2: Sentiment analysis (VADER, TextBlob, NRCLex)
  Pass 3: Emotional dynamics and volatility

Passes 4-6: Behavioral Pattern Detection
  Pass 4: Grooming pattern detection
  Pass 5: Manipulation and escalation tactics
  Pass 6: Deception markers and credibility assessment

Passes 7-8: Communication Analysis
  Pass 7: Intent classification and conversation dynamics
  Pass 8: Behavioral risk scoring and aggregation

Passes 9-10: Timeline & Context Analysis
  Pass 9: Timeline reconstruction and pattern sequencing
  Pass 10: Contextual insights and conversation flow

Passes 11-15: Person-Centric Analysis (NEW from ppl_int)
  Pass 11: Person identification and role classification
  Pass 12: Interaction mapping and relationship structure
  Pass 13: Gaslighting-specific detection
  Pass 14: Relationship dynamics and power analysis
  Pass 15: Intervention recommendations and case formulation
"""

import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import DatabaseAdapter
from validation.csv_validator import CSVValidator
from config.config_manager import ConfigManager, Configuration

# Import performance utilities
from utils.performance import (
    LazyLoader, LRUCache, BatchProcessor, ProgressTracker,
    timed_operation, get_result_cache, memoize
)
from utils.batch_optimizer import MessageBatchOptimizer, PassResultsCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalysisResult:
    """Complete 15-pass analysis result"""
    # Metadata
    analysis_run_id: int
    input_file: str
    message_count: int
    speaker_count: int
    processing_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Pass 1-3: Data Normalization & Sentiment
    data_validation: Dict[str, Any] = field(default_factory=dict)
    sentiment_results: Dict[str, Any] = field(default_factory=dict)
    emotional_dynamics: Dict[str, Any] = field(default_factory=dict)

    # Pass 4-6: Behavioral Patterns
    grooming_results: Dict[str, Any] = field(default_factory=dict)
    manipulation_results: Dict[str, Any] = field(default_factory=dict)
    deception_results: Dict[str, Any] = field(default_factory=dict)

    # Pass 7-8: Communication Analysis
    intent_results: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

    # Pass 9-10: Timeline & Context
    timeline_analysis: Dict[str, Any] = field(default_factory=dict)
    contextual_insights: Dict[str, Any] = field(default_factory=dict)

    # Pass 11-15: Person-Centric Analysis
    person_identification: Dict[str, Any] = field(default_factory=dict)
    interaction_mapping: Dict[str, Any] = field(default_factory=dict)
    gaslighting_detection: Dict[str, Any] = field(default_factory=dict)
    relationship_analysis: Dict[str, Any] = field(default_factory=dict)
    intervention_recommendations: Dict[str, Any] = field(default_factory=dict)

    # Aggregated results
    overall_risk_level: str = "unknown"
    primary_concerns: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Export paths
    json_output: Optional[str] = None
    csv_output: Optional[str] = None
    pdf_output: Optional[str] = None


class UnifiedProcessor:
    """15-pass unified analysis pipeline with optimizations"""

    def __init__(self, config: Configuration):
        """Initialize unified processor with lazy loading

        Args:
            config: Configuration object
        """
        self.config = config
        self.db = DatabaseAdapter(config.database.path)
        self.csv_validator = CSVValidator(auto_correct=True)

        # Initialize caches
        self.result_cache = LRUCache(maxsize=512)
        self.pass_cache = PassResultsCache()  # Cache results between passes

        # Initialize batch processor for optimized message processing
        self.batch_optimizer = MessageBatchOptimizer(batch_size=500)
        self.batch_processor = BatchProcessor(batch_size=1000)

        # Lazy load NLP modules (loaded only when needed)
        logger.info("Setting up lazy loaders for NLP modules...")

        self._sentiment_analyzer = LazyLoader(lambda: self._load_sentiment_analyzer())
        self._grooming_detector = LazyLoader(lambda: self._load_grooming_detector())
        self._manipulation_detector = LazyLoader(lambda: self._load_manipulation_detector())
        self._deception_analyzer = LazyLoader(lambda: self._load_deception_analyzer())
        self._intent_classifier = LazyLoader(lambda: self._load_intent_classifier())
        self._risk_scorer = LazyLoader(lambda: self._load_risk_scorer())
        self._person_analyzer = LazyLoader(lambda: self._load_person_analyzer())

        logger.info("Lazy loaders configured (modules will load on first use)")

    def _load_sentiment_analyzer(self):
        """Lazy load sentiment analyzer"""
        from nlp.sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer()

    def _load_grooming_detector(self):
        """Lazy load grooming detector"""
        from nlp.grooming_detector import GroomingDetector
        return GroomingDetector()

    def _load_manipulation_detector(self):
        """Lazy load manipulation detector"""
        from nlp.manipulation_detector import ManipulationDetector
        return ManipulationDetector()

    def _load_deception_analyzer(self):
        """Lazy load deception analyzer"""
        from nlp.deception_analyzer import DeceptionAnalyzer
        return DeceptionAnalyzer()

    def _load_intent_classifier(self):
        """Lazy load intent classifier"""
        from nlp.intent_classifier import IntentClassifier
        return IntentClassifier()

    def _load_risk_scorer(self):
        """Lazy load risk scorer"""
        from nlp.risk_scorer import BehavioralRiskScorer
        return BehavioralRiskScorer(
            weights={
                "grooming": self.config.nlp.risk_weight_grooming,
                "manipulation": self.config.nlp.risk_weight_manipulation,
                "hostility": self.config.nlp.risk_weight_hostility,
                "deception": self.config.nlp.risk_weight_deception
            }
        )

    def _load_person_analyzer(self):
        """Lazy load person analyzer"""
        from nlp.person_analyzer import PersonAnalyzer
        return PersonAnalyzer()

    @property
    def sentiment_analyzer(self):
        """Get sentiment analyzer (lazy loaded)"""
        return self._sentiment_analyzer.get()

    @property
    def grooming_detector(self):
        """Get grooming detector (lazy loaded)"""
        return self._grooming_detector.get()

    @property
    def manipulation_detector(self):
        """Get manipulation detector (lazy loaded)"""
        return self._manipulation_detector.get()

    @property
    def deception_analyzer(self):
        """Get deception analyzer (lazy loaded)"""
        return self._deception_analyzer.get()

    @property
    def intent_classifier(self):
        """Get intent classifier (lazy loaded)"""
        return self._intent_classifier.get()

    @property
    def risk_scorer(self):
        """Get risk scorer (lazy loaded)"""
        return self._risk_scorer.get()

    @property
    def person_analyzer(self):
        """Get person analyzer (lazy loaded)"""
        return self._person_analyzer.get()

    def get_loading_stats(self) -> Dict[str, Any]:
        """Get statistics on which modules are loaded

        Returns:
            Dict with loading status for each module
        """
        loaded_modules = {
            'sentiment_analyzer': self._sentiment_analyzer.is_loaded,
            'grooming_detector': self._grooming_detector.is_loaded,
            'manipulation_detector': self._manipulation_detector.is_loaded,
            'deception_analyzer': self._deception_analyzer.is_loaded,
            'intent_classifier': self._intent_classifier.is_loaded,
            'risk_scorer': self._risk_scorer.is_loaded,
            'person_analyzer': self._person_analyzer.is_loaded,
        }

        loading_times = {}
        for name, loader in [
            ('sentiment_analyzer', self._sentiment_analyzer),
            ('grooming_detector', self._grooming_detector),
            ('manipulation_detector', self._manipulation_detector),
            ('deception_analyzer', self._deception_analyzer),
            ('intent_classifier', self._intent_classifier),
            ('risk_scorer', self._risk_scorer),
            ('person_analyzer', self._person_analyzer),
        ]:
            if loader.is_loaded and loader.loading_time is not None:
                loading_times[name] = f"{loader.loading_time:.2f}s"

        return {
            'modules_loaded': loaded_modules,
            'loading_times': loading_times,
            'total_loaded': sum(loaded_modules.values()),
            'cache_stats': self.pass_cache.get_cache_stats()
        }

    @timed_operation("15-pass unified pipeline")
    def process_file(self, input_file: str, output_dir: Optional[str] = None) -> UnifiedAnalysisResult:
        """Execute complete 15-pass pipeline with optimizations

        Args:
            input_file: Path to input CSV file
            output_dir: Optional output directory for results

        Returns:
            UnifiedAnalysisResult: Complete analysis results
        """
        start_time = time.time()
        logger.info(f"Starting optimized 15-pass pipeline on {input_file}")
        print("\n" + "="*70)
        print("15-PASS UNIFIED ANALYSIS PIPELINE (OPTIMIZED)")
        print("="*70)

        # Clear pass cache for new file
        self.pass_cache.clear()

        # Create analysis run
        run_id = self.db.create_analysis_run(input_file, self.config.to_dict())

        try:
            # ===================================================================
            # PASSES 1-3: DATA NORMALIZATION & SENTIMENT
            # ===================================================================
            print("\n[PASSES 1-3] Data Normalization & Sentiment Analysis")
            print("-" * 70)

            # Pass 1: CSV Validation and Data Normalization
            logger.info("Pass 1: CSV validation and data normalization")
            data_validation, df = self._pass_1_validate_and_normalize(input_file)
            messages = self._dataframe_to_messages(df)

            if self.config.database.enable_caching:
                self.db.insert_messages_batch(messages)

            # Pass 2: Sentiment Analysis
            logger.info("Pass 2: Sentiment analysis")
            sentiment_results = self._pass_2_sentiment_analysis(messages)

            # Pass 3: Emotional Dynamics
            logger.info("Pass 3: Emotional dynamics and volatility")
            emotional_dynamics = self._pass_3_emotional_dynamics(messages, sentiment_results)

            # ===================================================================
            # PASSES 4-6: BEHAVIORAL PATTERN DETECTION (PARALLEL EXECUTION)
            # ===================================================================
            print("\n[PASSES 4-6] Behavioral Pattern Detection (Parallel)")
            print("-" * 70)

            # Passes 4-6 are independent and can run in parallel for 3x speedup
            grooming_results = {}
            manipulation_results = {}
            deception_results = {}

            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time

            start_time = time.time()
            detection_tasks = []

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit parallel tasks
                if self.config.nlp.enable_grooming_detection:
                    logger.info("Pass 4: Grooming pattern detection (parallel)")
                    detection_tasks.append(
                        executor.submit(self._pass_4_grooming_detection, messages)
                    )

                if self.config.nlp.enable_manipulation_detection:
                    logger.info("Pass 5: Manipulation detection (parallel)")
                    detection_tasks.append(
                        executor.submit(self._pass_5_manipulation_detection, messages)
                    )

                if self.config.nlp.enable_deception_markers:
                    logger.info("Pass 6: Deception analysis (parallel)")
                    detection_tasks.append(
                        executor.submit(self._pass_6_deception_analysis, messages)
                    )

                # Collect results as they complete
                results_map = {}
                for future in as_completed(detection_tasks):
                    result = future.result()
                    # Identify which pass this is based on keys in result
                    if 'grooming_risk' in str(result) or 'stage_progression' in str(result):
                        grooming_results = result
                    elif 'manipulation_tactics' in str(result) or 'escalation_points' in str(result):
                        manipulation_results = result
                    elif 'credibility' in str(result) or 'deception_markers' in str(result):
                        deception_results = result

            parallel_time = time.time() - start_time
            logger.info(f"Parallel detection completed in {parallel_time:.2f}s")

            # ===================================================================
            # PASSES 7-8: COMMUNICATION ANALYSIS
            # ===================================================================
            print("\n[PASSES 7-8] Communication Analysis")
            print("-" * 70)

            # Pass 7: Intent Classification
            logger.info("Pass 7: Intent classification")
            intent_results = {}
            if self.config.nlp.enable_intent_classification:
                intent_results = self._pass_7_intent_classification(messages)

            # Pass 8: Risk Assessment
            logger.info("Pass 8: Behavioral risk scoring")
            risk_assessment = self._pass_8_risk_assessment(
                messages,
                {
                    'sentiment': sentiment_results,
                    'grooming': grooming_results,
                    'manipulation': manipulation_results,
                    'deception': deception_results,
                    'intent': intent_results
                }
            )

            # ===================================================================
            # PASSES 9-10: TIMELINE & CONTEXT ANALYSIS
            # ===================================================================
            print("\n[PASSES 9-10] Timeline & Context Analysis")
            print("-" * 70)

            # Pass 9: Timeline Reconstruction
            logger.info("Pass 9: Timeline reconstruction and pattern sequencing")
            timeline_analysis = self._pass_9_timeline_analysis(messages, risk_assessment)

            # Pass 10: Contextual Insights
            logger.info("Pass 10: Contextual insights and conversation flow")
            contextual_insights = self._pass_10_contextual_insights(
                messages, sentiment_results, timeline_analysis
            )

            # ===================================================================
            # PASSES 11-15: PERSON-CENTRIC ANALYSIS
            # ===================================================================
            print("\n[PASSES 11-15] Person-Centric Analysis")
            print("-" * 70)

            # Pass 11: Person Identification (must run first)
            logger.info("Pass 11: Person identification and role classification")
            person_identification = self._pass_11_person_identification(messages, df)

            # Passes 12-14 can run in parallel after person identification
            logger.info("Passes 12-14: Running parallel analysis")
            interaction_mapping = {}
            gaslighting_detection = {}
            relationship_analysis = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit parallel tasks
                future_interaction = executor.submit(
                    self._pass_12_interaction_mapping,
                    messages, person_identification.get('persons', [])
                )

                future_gaslighting = executor.submit(
                    self._pass_13_gaslighting_detection,
                    messages, person_identification.get('persons', []), manipulation_results
                )

                future_relationship = executor.submit(
                    self._pass_14_relationship_analysis,
                    messages, person_identification.get('persons', []), {}  # Will update after interaction
                )

                # Collect results
                interaction_mapping = future_interaction.result()
                gaslighting_detection = future_gaslighting.result()
                relationship_analysis = future_relationship.result()

                logger.info("Parallel person-centric analysis completed")

            # Pass 15: Intervention Recommendations
            logger.info("Pass 15: Intervention recommendations and case formulation")
            intervention_recommendations = self._pass_15_intervention_recommendations(
                risk_assessment,
                person_identification,
                relationship_analysis,
                gaslighting_detection
            )

            # ===================================================================
            # FINALIZATION
            # ===================================================================
            print("\n[FINALIZATION] Aggregating Results")
            print("-" * 70)

            # Store patterns
            self._store_patterns(run_id, messages, risk_assessment)

            # Export results
            export_paths = self._export_results(
                run_id,
                {
                    'data_validation': data_validation,
                    'sentiment': sentiment_results,
                    'emotional_dynamics': emotional_dynamics,
                    'grooming': grooming_results,
                    'manipulation': manipulation_results,
                    'deception': deception_results,
                    'intent': intent_results,
                    'risk': risk_assessment,
                    'timeline': timeline_analysis,
                    'contextual_insights': contextual_insights,
                    'person_identification': person_identification,
                    'interaction_mapping': interaction_mapping,
                    'gaslighting': gaslighting_detection,
                    'relationship': relationship_analysis,
                    'intervention': intervention_recommendations
                },
                output_dir
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Aggregate findings
            overall_risk = risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown')
            primary_concerns = self._aggregate_concerns(
                risk_assessment,
                grooming_results,
                manipulation_results,
                gaslighting_detection,
                relationship_analysis
            )
            recommendations = self._aggregate_recommendations(
                risk_assessment,
                intervention_recommendations
            )

            # Update analysis run
            self.db.update_analysis_run(
                run_id,
                status="completed",
                duration=processing_time,
                results={
                    'risk_level': overall_risk,
                    'concerns': primary_concerns,
                    'recommendations': recommendations
                }
            )

            # Create result object
            result = UnifiedAnalysisResult(
                analysis_run_id=run_id,
                input_file=input_file,
                message_count=len(messages),
                speaker_count=len(df['sender'].unique()),
                processing_time=processing_time,
                data_validation=data_validation,
                sentiment_results=sentiment_results,
                emotional_dynamics=emotional_dynamics,
                grooming_results=grooming_results,
                manipulation_results=manipulation_results,
                deception_results=deception_results,
                intent_results=intent_results,
                risk_assessment=risk_assessment,
                timeline_analysis=timeline_analysis,
                contextual_insights=contextual_insights,
                person_identification=person_identification,
                interaction_mapping=interaction_mapping,
                gaslighting_detection=gaslighting_detection,
                relationship_analysis=relationship_analysis,
                intervention_recommendations=intervention_recommendations,
                overall_risk_level=overall_risk,
                primary_concerns=primary_concerns,
                recommendations=recommendations,
                json_output=export_paths.get('json'),
                csv_output=export_paths.get('csv')
            )

            print("\n" + "="*70)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*70)
            logger.info(f"Processing completed in {processing_time:.2f} seconds")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.db.update_analysis_run(
                run_id,
                status="failed",
                error_message=str(e)
            )
            raise

    # PASS IMPLEMENTATIONS
    # =====================================================================

    def _pass_1_validate_and_normalize(self, input_file: str) -> Tuple[Dict, pd.DataFrame]:
        """Pass 1: CSV validation and data normalization"""
        validation_result, df = self.csv_validator.validate_file(input_file)

        if not validation_result.is_valid:
            raise ValueError(f"CSV validation failed: {validation_result.errors}")

        print(f"  CSV Validated: {len(df)} messages, {len(df.get('sender', df.get('Sender Name', [])).unique())} speakers")

        return {
            'is_valid': validation_result.is_valid,
            'encoding': validation_result.encoding,
            'rows': len(df),
            'columns': len(df.columns),
            'warnings': validation_result.warnings if validation_result.warnings else []
        }, df

    def _pass_2_sentiment_analysis(self, messages: List[Dict]) -> Dict[str, Any]:
        """Pass 2: Sentiment analysis using multiple engines with batch processing"""
        # Check cache first
        if self.pass_cache.has_pass_result('sentiment'):
            logger.info("  Using cached sentiment results")
            return self.pass_cache.get_pass_result('sentiment')

        try:
            message_sentiments = []
            progress = ProgressTracker(len(messages), desc="  Sentiment analysis", log_interval=20)

            # Process in batches with progress tracking
            for i, msg in enumerate(messages):
                try:
                    sentiment = self.sentiment_analyzer.analyze_text(msg.get('text', ''))
                    message_sentiments.append(sentiment)
                except Exception as e:
                    logger.warning(f"  Sentiment analysis failed for message {i}: {e}")
                    message_sentiments.append(None)

                progress.update(1)

            progress.finish()

            conversation_sentiment = self.sentiment_analyzer.analyze_conversation(messages)

            overall = conversation_sentiment.overall_sentiment if hasattr(conversation_sentiment, 'overall_sentiment') else 0.0
            print(f"  Overall Sentiment: {overall:.2f}, Trajectory: {getattr(conversation_sentiment, 'sentiment_trajectory', 'N/A')}")

            result = {
                'per_message': message_sentiments,
                'conversation': conversation_sentiment
            }

            # Cache result
            self.pass_cache.cache_pass_result('sentiment', result)
            return result

        except Exception as e:
            logger.error(f"  Sentiment analysis pass failed: {e}")
            return {'per_message': [], 'conversation': None, 'error': str(e)}

    def _pass_3_emotional_dynamics(self, messages: List[Dict], sentiment_results: Dict) -> Dict[str, Any]:
        """Pass 3: Emotional dynamics and volatility assessment"""
        # Check cache first
        if self.pass_cache.has_pass_result('emotional_dynamics'):
            logger.info("  Using cached emotional dynamics results")
            return self.pass_cache.get_pass_result('emotional_dynamics')

        try:
            if not sentiment_results.get('per_message'):
                return {'error': 'No sentiment data available'}

            sentiments = [s.combined_sentiment if hasattr(s, 'combined_sentiment') else 0.0
                          for s in sentiment_results['per_message'] if s is not None]

            if len(sentiments) < 2:
                volatility = 0.0
            else:
                import statistics
                volatility = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0

            print(f"  Emotional Volatility: {volatility:.2f}")

            result = {
                'volatility': volatility,
                'sentiments': sentiments,
                'emotion_shifts': self._detect_emotion_shifts(sentiments)
            }

            # Cache result with dependency on sentiment
            self.pass_cache.cache_pass_result('emotional_dynamics', result, dependencies=['sentiment'])
            return result

        except Exception as e:
            logger.error(f"  Emotional dynamics analysis failed: {e}")
            return {'error': str(e)}

    def _pass_4_grooming_detection(self, messages: List[Dict]) -> Dict[str, Any]:
        """Pass 4: Grooming pattern detection with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('grooming'):
            logger.info("  Using cached grooming detection results")
            return self.pass_cache.get_pass_result('grooming')

        try:
            logger.info(f"  Analyzing {len(messages)} messages for grooming patterns")
            result = self.grooming_detector.analyze_conversation(messages)
            risk_level = result.get('overall_risk', 'low')
            print(f"  Grooming Risk: {risk_level}")

            # Cache result
            self.pass_cache.cache_pass_result('grooming', result)
            return result

        except Exception as e:
            logger.error(f"  Grooming detection failed: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}

    def _pass_5_manipulation_detection(self, messages: List[Dict]) -> Dict[str, Any]:
        """Pass 5: Manipulation and escalation detection with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('manipulation'):
            logger.info("  Using cached manipulation detection results")
            return self.pass_cache.get_pass_result('manipulation')

        try:
            logger.info(f"  Analyzing {len(messages)} messages for manipulation patterns")
            result = self.manipulation_detector.analyze_conversation(messages)
            risk_level = result.get('overall_risk', 'low')
            print(f"  Manipulation Risk: {risk_level}")

            # Cache result
            self.pass_cache.cache_pass_result('manipulation', result)
            return result

        except Exception as e:
            logger.error(f"  Manipulation detection failed: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}

    def _pass_6_deception_analysis(self, messages: List[Dict]) -> Dict[str, Any]:
        """Pass 6: Deception markers analysis with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('deception'):
            logger.info("  Using cached deception analysis results")
            return self.pass_cache.get_pass_result('deception')

        try:
            logger.info(f"  Analyzing {len(messages)} messages for deception markers")
            result = self.deception_analyzer.analyze_conversation(messages)
            credibility = result.get('overall_credibility', 'unknown')
            print(f"  Credibility Assessment: {credibility}")

            # Cache result
            self.pass_cache.cache_pass_result('deception', result)
            return result

        except Exception as e:
            logger.error(f"  Deception analysis failed: {e}")
            return {'overall_credibility': 'unknown', 'error': str(e)}

    def _pass_7_intent_classification(self, messages: List[Dict]) -> Dict[str, Any]:
        """Pass 7: Intent classification with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('intent'):
            logger.info("  Using cached intent classification results")
            return self.pass_cache.get_pass_result('intent')

        try:
            logger.info(f"  Classifying intents for {len(messages)} messages")
            result = self.intent_classifier.analyze_conversation_intents(messages)
            dynamic = result.get('conversation_dynamic', 'neutral')
            print(f"  Conversation Dynamic: {dynamic}")

            # Cache result
            self.pass_cache.cache_pass_result('intent', result)
            return result

        except Exception as e:
            logger.error(f"  Intent classification failed: {e}")
            return {'conversation_dynamic': 'unknown', 'error': str(e)}

    def _pass_8_risk_assessment(self, messages: List[Dict], analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Pass 8: Comprehensive risk assessment with progress tracking"""
        # Check cache first
        if self.pass_cache.has_pass_result('risk'):
            logger.info("  Using cached risk assessment results")
            return self.pass_cache.get_pass_result('risk')

        try:
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

            result = {
                'per_message_risks': per_message_risks,
                'overall_risk_assessment': {
                    'risk_level': risk_level,
                    'average_risk': avg_risk,
                    'max_risk': max_risk
                }
            }

            # Cache result with dependencies
            self.pass_cache.cache_pass_result('risk', result,
                dependencies=['grooming', 'manipulation', 'deception', 'intent'])
            return result

        except Exception as e:
            logger.error(f"  Risk assessment failed: {e}")
            return {
                'per_message_risks': [],
                'overall_risk_assessment': {
                    'risk_level': 'unknown',
                    'average_risk': 0,
                    'max_risk': 0
                },
                'error': str(e)
            }

    def _pass_9_timeline_analysis(self, messages: List[Dict], risk_assessment: Dict) -> Dict[str, Any]:
        """Pass 9: Timeline reconstruction and pattern sequencing with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('timeline'):
            logger.info("  Using cached timeline analysis results")
            return self.pass_cache.get_pass_result('timeline')

        try:
            timeline_points = []

            for i, msg in enumerate(messages):
                try:
                    timeline_points.append({
                        'index': i,
                        'sender': msg.get('sender', 'Unknown'),
                        'timestamp': msg.get('timestamp', msg.get('date', '')),
                        'text': msg.get('text', '')[:100] + '...' if len(msg.get('text', '')) > 100 else msg.get('text', '')
                    })
                except Exception as e:
                    logger.warning(f"  Timeline point extraction failed for message {i}: {e}")

            print(f"  Timeline Points Extracted: {len(timeline_points)}")

            result = {
                'timeline_points': timeline_points,
                'conversation_duration': self._estimate_duration(messages),
                'pattern_sequences': self._identify_pattern_sequences(timeline_points, risk_assessment)
            }

            # Cache result with dependency on risk
            self.pass_cache.cache_pass_result('timeline', result, dependencies=['risk'])
            return result

        except Exception as e:
            logger.error(f"  Timeline analysis failed: {e}")
            return {'timeline_points': [], 'error': str(e)}

    def _pass_10_contextual_insights(self, messages: List[Dict], sentiment_results: Dict, timeline_analysis: Dict) -> Dict[str, Any]:
        """Pass 10: Contextual insights and conversation flow with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('contextual'):
            logger.info("  Using cached contextual insights results")
            return self.pass_cache.get_pass_result('contextual')

        try:
            insights = []

            if timeline_analysis.get('conversation_duration'):
                insights.append(f"Conversation spanning {timeline_analysis['conversation_duration']}")

            print(f"  Contextual Insights Generated: {len(insights)}")

            result = {
                'insights': insights,
                'conversation_flow': 'complex' if len(messages) > 50 else 'moderate' if len(messages) > 20 else 'simple'
            }

            # Cache result with dependencies
            self.pass_cache.cache_pass_result('contextual', result,
                dependencies=['sentiment', 'timeline'])
            return result

        except Exception as e:
            logger.error(f"  Contextual insights analysis failed: {e}")
            return {'insights': [], 'error': str(e)}

    def _pass_11_person_identification(self, messages: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
        """Pass 11: Person identification and role classification with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('person_id'):
            logger.info("  Using cached person identification results")
            return self.pass_cache.get_pass_result('person_id')

        try:
            logger.info(f"  Identifying persons in conversation with {len(messages)} messages")
            result = self.person_analyzer.identify_persons_in_conversation(messages, df)
            print(f"  Persons Identified: {len(result.get('persons', []))}")

            # Cache result
            self.pass_cache.cache_pass_result('person_id', result)
            return result

        except Exception as e:
            logger.error(f"  Person identification failed: {e}")
            return {'persons': [], 'error': str(e)}

    def _pass_12_interaction_mapping(self, messages: List[Dict], person_identification: Dict) -> Dict[str, Any]:
        """Pass 12: Interaction mapping and relationship structure with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('interaction'):
            logger.info("  Using cached interaction mapping results")
            return self.pass_cache.get_pass_result('interaction')

        try:
            result = self.person_analyzer.extract_interaction_patterns(
                messages, person_identification.get('persons', [])
            )
            print(f"  Interaction Patterns Mapped: {len(result.get('interactions', []))}")

            # Cache result with dependency on person identification
            self.pass_cache.cache_pass_result('interaction', result, dependencies=['person_id'])
            return result

        except Exception as e:
            logger.error(f"  Interaction mapping failed: {e}")
            return {'interactions': [], 'error': str(e)}

    def _pass_13_gaslighting_detection(self, messages: List[Dict], person_identification: Dict,
                                       manipulation_results: Dict) -> Dict[str, Any]:
        """Pass 13: Gaslighting-specific detection with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('gaslighting'):
            logger.info("  Using cached gaslighting detection results")
            return self.pass_cache.get_pass_result('gaslighting')

        try:
            result = self.person_analyzer.detect_gaslighting_patterns(
                messages, person_identification.get('persons', []), manipulation_results
            )
            risk_level = result.get('gaslighting_risk', 'low')
            print(f"  Gaslighting Risk: {risk_level}")

            # Cache result with dependencies
            self.pass_cache.cache_pass_result('gaslighting', result,
                dependencies=['person_id', 'manipulation'])
            return result

        except Exception as e:
            logger.error(f"  Gaslighting detection failed: {e}")
            return {'gaslighting_risk': 'unknown', 'error': str(e)}

    def _pass_14_relationship_analysis(self, messages: List[Dict], person_identification: Dict,
                                       interaction_mapping: Dict) -> Dict[str, Any]:
        """Pass 14: Relationship dynamics and power analysis with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('relationship'):
            logger.info("  Using cached relationship analysis results")
            return self.pass_cache.get_pass_result('relationship')

        try:
            result = self.person_analyzer.assess_relationship_dynamics(
                messages, person_identification.get('persons', [])
            )
            print(f"  Relationship Dynamics Analyzed")

            # Cache result with dependency on person identification
            self.pass_cache.cache_pass_result('relationship', result, dependencies=['person_id'])
            return result

        except Exception as e:
            logger.error(f"  Relationship analysis failed: {e}")
            return {'error': str(e)}

    def _pass_15_intervention_recommendations(self, risk_assessment: Dict, person_identification: Dict,
                                               relationship_analysis: Dict, gaslighting_detection: Dict) -> Dict[str, Any]:
        """Pass 15: Intervention recommendations and case formulation with error recovery"""
        # Check cache first
        if self.pass_cache.has_pass_result('intervention'):
            logger.info("  Using cached intervention recommendations results")
            return self.pass_cache.get_pass_result('intervention')

        try:
            result = self.person_analyzer.generate_intervention_recommendations(
                risk_assessment, person_identification, relationship_analysis, gaslighting_detection
            )
            print(f"  Intervention Recommendations Generated: {len(result.get('recommendations', []))}")

            # Cache result with dependencies
            self.pass_cache.cache_pass_result('intervention', result,
                dependencies=['risk', 'person_id', 'relationship', 'gaslighting'])
            return result

        except Exception as e:
            logger.error(f"  Intervention recommendations failed: {e}")
            return {'recommendations': [], 'error': str(e)}

    # UTILITY METHODS
    # =====================================================================

    def _dataframe_to_messages(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to message list"""
        messages = []
        for idx, row in df.iterrows():
            msg = {
                'index': idx,
                'text': str(row.get('text', row.get('Text', row.get('Message', '')))),
                'sender': str(row.get('sender', row.get('Sender Name', row.get('Sender', 'Unknown')))),
                'timestamp': row.get('timestamp', row.get('date', row.get('Date', ''))),
                'date': row.get('date', row.get('Date', '')),
                'time': row.get('time', row.get('Time', ''))
            }

            for col in df.columns:
                if col not in msg:
                    msg[col] = row[col]

            messages.append(msg)

        return messages

    def _detect_emotion_shifts(self, sentiments: List[float]) -> List[Dict]:
        """Detect significant emotion shifts in conversation"""
        shifts = []
        threshold = 0.3

        for i in range(1, len(sentiments)):
            shift = abs(sentiments[i] - sentiments[i-1])
            if shift > threshold:
                shifts.append({
                    'index': i,
                    'from': sentiments[i-1],
                    'to': sentiments[i],
                    'magnitude': shift
                })

        return shifts

    def _estimate_duration(self, messages: List[Dict]) -> str:
        """Estimate conversation duration"""
        if len(messages) < 2:
            return "unknown"

        first = messages[0].get('timestamp', messages[0].get('date', ''))
        last = messages[-1].get('timestamp', messages[-1].get('date', ''))

        if first and last and first != last:
            return f"{first} to {last}"
        elif len(messages) > 100:
            return "extended"
        elif len(messages) > 50:
            return "moderate"
        else:
            return "brief"

    def _identify_pattern_sequences(self, timeline_points: List[Dict], risk_assessment: Dict) -> List[Dict]:
        """Identify sequences of concerning patterns"""
        sequences = []
        return sequences  # Stub for expansion

    def _store_patterns(self, run_id: int, messages: List[Dict], risk_assessment: Dict):
        """Store detected patterns in database with batch optimization"""
        try:
            patterns = []

            # Build patterns list in batches to avoid memory issues
            for i, risk in enumerate(risk_assessment.get('per_message_risks', [])):
                try:
                    if risk is not None and hasattr(risk, 'primary_concern') and risk.primary_concern:
                        patterns.append({
                            'analysis_run_id': run_id,
                            'message_id': i,
                            'pattern_type': risk.primary_concern,
                            'severity': risk.overall_risk if hasattr(risk, 'overall_risk') else 0.5,
                            'confidence': risk.assessment_confidence if hasattr(risk, 'assessment_confidence') else 0.7
                        })
                except Exception as e:
                    logger.warning(f"Failed to extract pattern from message {i}: {e}")

            if patterns:
                # Use batch processing for large pattern sets
                if len(patterns) > 1000:
                    logger.info(f"Storing {len(patterns)} patterns in batches...")
                    batch_size = 500
                    for i in range(0, len(patterns), batch_size):
                        batch = patterns[i:i + batch_size]
                        self.db.insert_patterns_batch(batch)
                    logger.info(f"Stored {len(patterns)} patterns in database (batched)")
                else:
                    self.db.insert_patterns_batch(patterns)
                    logger.info(f"Stored {len(patterns)} patterns in database")

        except Exception as e:
            logger.error(f"Failed to store patterns in database: {e}")

    def _aggregate_concerns(self, *results) -> List[str]:
        """Aggregate all primary concerns from all passes"""
        concerns = []

        for result in results:
            if isinstance(result, dict):
                if result.get('overall_risk') in ['high', 'critical']:
                    concerns.append(result.get('primary_concern', 'Unknown risk'))
                if result.get('gaslighting_risk') in ['high', 'critical']:
                    concerns.append("Gaslighting patterns detected")
                if result.get('power_imbalance'):
                    concerns.append("Significant power imbalance detected")

        return list(set(concerns))  # Remove duplicates

    def _aggregate_recommendations(self, risk_assessment: Dict, intervention_recommendations: Dict) -> List[str]:
        """Aggregate recommendations from all passes"""
        recommendations = set()

        if risk_assessment.get('overall_risk_assessment', {}).get('risk_level') in ['high', 'critical']:
            recommendations.add("Seek professional support immediately")
            recommendations.add("Document all concerning interactions")

        if intervention_recommendations.get('recommendations'):
            recommendations.update(intervention_recommendations['recommendations'])

        return list(recommendations)

    def _export_results(self, run_id: int, results: Dict, output_dir: Optional[str]) -> Dict[str, str]:
        """Export 15-pass analysis results"""
        if not output_dir:
            output_dir = "Reports"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"unified_analysis_{run_id}_{timestamp}"

        export_paths = {}

        # Export JSON
        json_path = output_path / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json_safe = self._make_json_serializable(results)
            json.dump(json_safe, f, indent=2)
        export_paths['json'] = str(json_path)
        logger.info(f"Exported JSON to {json_path}")

        # Export CSV summary
        csv_path = output_path / f"{base_name}_summary.csv"
        self._export_summary_csv(results, csv_path)
        export_paths['csv'] = str(csv_path)
        logger.info(f"Exported CSV to {csv_path}")

        return export_paths

    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)

    def _export_summary_csv(self, results: Dict, csv_path: Path):
        """Export summary to CSV"""
        summary_data = []

        # Add pass results summary
        passes = [
            ('Pass 1', 'Data Validation', results.get('data_validation', {})),
            ('Pass 2-3', 'Sentiment & Emotional Dynamics', results.get('sentiment', {})),
            ('Pass 4', 'Grooming Detection', results.get('grooming', {})),
            ('Pass 5', 'Manipulation Detection', results.get('manipulation', {})),
            ('Pass 6', 'Deception Analysis', results.get('deception', {})),
            ('Pass 7', 'Intent Classification', results.get('intent', {})),
            ('Pass 8', 'Risk Assessment', results.get('risk', {})),
            ('Pass 9-10', 'Timeline & Context', results.get('timeline', {})),
            ('Pass 11', 'Person Identification', results.get('person_identification', {})),
            ('Pass 12', 'Interaction Mapping', results.get('interaction_mapping', {})),
            ('Pass 13', 'Gaslighting Detection', results.get('gaslighting', {})),
            ('Pass 14', 'Relationship Analysis', results.get('relationship', {})),
            ('Pass 15', 'Intervention Recommendations', results.get('intervention', {}))
        ]

        for pass_name, description, data in passes:
            summary_data.append({
                'Pass': pass_name,
                'Description': description,
                'Status': 'Completed' if data else 'Skipped',
                'Details': json.dumps(data)[:100] if data else 'N/A'
            })

        if summary_data:
            pd.DataFrame(summary_data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    print("Unified 15-Pass Pipeline Module")
    print("Use with: from unified_processor import UnifiedProcessor, UnifiedAnalysisResult")
