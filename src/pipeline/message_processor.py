#!/usr/bin/env python3
"""
Message Processor - Main Pipeline
Orchestrates multi-pass analysis combining all NLP modules for comprehensive message analysis.
Refactored from original multi_pass_chat_analysis.py with modular architecture.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, asdict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from db.database import DatabaseAdapter, Message, AnalysisRun, Pattern
from validation.csv_validator import CSVValidator
from validation.timestamp_validator import TimestampValidator
from config.config_manager import ConfigManager, Configuration
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.grooming_detector import GroomingDetector
from nlp.manipulation_detector import ManipulationDetector
from nlp.deception_analyzer import DeceptionAnalyzer
from nlp.intent_classifier import IntentClassifier
from nlp.risk_scorer import BehavioralRiskScorer
from nlp.temporal_analyzer import TemporalAnalyzer
from cache.analysis_cache import AnalysisResultCache, create_analysis_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Container for complete processing results"""
    analysis_run_id: int
    input_file: str
    message_count: int
    speaker_count: int
    processing_time: float

    # Analysis results
    sentiment_results: Dict[str, Any]
    grooming_results: Dict[str, Any]
    manipulation_results: Dict[str, Any]
    deception_results: Dict[str, Any]
    intent_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    temporal_results: Optional[Dict[str, Any]] = None

    # Aggregated insights
    overall_risk_level: str
    primary_concerns: List[str]
    key_findings: List[str]
    recommendations: List[str]

    # Cache performance
    cache_hit: bool = False
    cache_stats: Optional[Dict[str, Any]] = None

    # Export paths
    json_output: Optional[str] = None
    csv_output: Optional[str] = None
    pdf_output: Optional[str] = None


class MessageProcessor:
    """Main processing pipeline for message analysis"""

    def __init__(self, config: Configuration, enable_cache: bool = True):
        """Initialize processor with configuration

        Args:
            config: Configuration object
            enable_cache: Enable Redis analysis caching (default: True)
        """
        self.config = config

        # Initialize database
        self.db = DatabaseAdapter(config.database.path)

        # Initialize validators
        self.csv_validator = CSVValidator(auto_correct=True)
        self.timestamp_validator = TimestampValidator()

        # Initialize NLP modules
        logger.info("Initializing NLP modules...")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.grooming_detector = GroomingDetector()
        self.manipulation_detector = ManipulationDetector()
        self.deception_analyzer = DeceptionAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.risk_scorer = BehavioralRiskScorer(
            weights={
                "grooming": config.nlp.risk_weight_grooming,
                "manipulation": config.nlp.risk_weight_manipulation,
                "hostility": config.nlp.risk_weight_hostility,
                "deception": config.nlp.risk_weight_deception
            }
        )

        # Initialize temporal analyzer
        self.temporal_analyzer = TemporalAnalyzer(window_size_hours=24)

        # Initialize analysis result cache
        if enable_cache:
            try:
                self.analysis_cache = create_analysis_cache()
                if self.analysis_cache.redis.enabled:
                    logger.info("✅ Analysis result caching enabled (Redis)")
                else:
                    logger.warning("⚠️  Redis unavailable - caching disabled")
                    self.analysis_cache = None
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
                self.analysis_cache = None
        else:
            self.analysis_cache = None
            logger.info("Analysis caching disabled")

        # Processing statistics
        self.stats = {
            "messages_processed": 0,
            "patterns_detected": 0,
            "processing_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def process_file(self, input_file: str, output_dir: Optional[str] = None, use_cache: bool = True) -> ProcessingResult:
        """Process a CSV file through complete analysis pipeline

        Args:
            input_file: Path to input CSV file
            output_dir: Optional output directory for results
            use_cache: Use cached results if available (default: True)

        Returns:
            ProcessingResult: Complete analysis results
        """
        start_time = time.time()
        logger.info(f"Starting processing of {input_file}")

        # Check cache first (2-3x speedup for re-analysis)
        if use_cache and self.analysis_cache:
            logger.info("Checking analysis cache...")
            cached_result = self.analysis_cache.get_cached_analysis(
                input_file,
                self.config.to_dict()
            )

            if cached_result and 'results' in cached_result:
                logger.info("✅ Using cached analysis results!")
                self.stats['cache_hits'] += 1

                # Return cached result (reconstruct ProcessingResult)
                return self._reconstruct_result_from_cache(
                    cached_result['results'],
                    input_file,
                    output_dir
                )
            else:
                logger.info("Cache miss - performing full analysis")
                self.stats['cache_misses'] += 1

        # Create analysis run
        run_id = self.db.create_analysis_run(input_file, self.config.to_dict())

        try:
            # Pass 0: Validate and load data
            logger.info("Pass 0: Data validation and loading")
            validation_result, df = self.csv_validator.validate_file(input_file)

            if not validation_result.is_valid:
                raise ValueError(f"CSV validation failed: {validation_result.errors}")

            logger.info(f"Loaded {len(df)} messages from {len(df['sender'].unique())} speakers")

            # Convert to message list for processing
            messages = self._dataframe_to_messages(df)

            # Store messages in database if caching enabled
            if self.config.database.enable_caching:
                logger.info("Caching messages to database...")
                self.db.insert_messages_batch(messages)

            # Pass 1: Sentiment Analysis
            logger.info("Pass 1: Sentiment analysis")
            sentiment_results = self._process_sentiment(messages)

            # Pass 2: Grooming Detection
            logger.info("Pass 2: Grooming pattern detection")
            grooming_results = {}
            if self.config.nlp.enable_grooming_detection:
                grooming_results = self.grooming_detector.analyze_conversation(messages)

            # Pass 3: Manipulation Detection
            logger.info("Pass 3: Manipulation and gaslighting detection")
            manipulation_results = {}
            if self.config.nlp.enable_manipulation_detection:
                manipulation_results = self.manipulation_detector.analyze_conversation(messages)

            # Pass 4: Deception Analysis
            logger.info("Pass 4: Deception markers analysis")
            deception_results = {}
            if self.config.nlp.enable_deception_markers:
                deception_results = self.deception_analyzer.analyze_conversation(messages)

            # Pass 5: Intent Classification
            logger.info("Pass 5: Intent classification")
            intent_results = {}
            if self.config.nlp.enable_intent_classification:
                intent_results = self.intent_classifier.analyze_conversation_intents(messages)

            # Pass 6: Risk Assessment
            logger.info("Pass 6: Comprehensive risk assessment")
            risk_assessment = self._perform_risk_assessment(
                messages,
                {
                    'sentiment': sentiment_results,
                    'grooming': grooming_results,
                    'manipulation': manipulation_results,
                    'deception': deception_results,
                    'intent': intent_results
                }
            )

            # Pass 7: Temporal Analysis (with timestamp validation)
            logger.info("Pass 7: Temporal pattern analysis")
            temporal_results = self._perform_temporal_analysis(messages, risk_assessment)

            # Pass 8: Pattern Storage
            logger.info("Pass 8: Pattern storage and indexing")
            self._store_patterns(run_id, messages, risk_assessment)

            # Pass 9: Generate Insights
            logger.info("Pass 9: Generating insights and recommendations")
            insights = self._generate_insights(
                sentiment_results,
                grooming_results,
                manipulation_results,
                deception_results,
                intent_results,
                risk_assessment,
                temporal_results
            )

            # Pass 10: Export Results
            logger.info("Pass 10: Exporting results")
            export_paths = self._export_results(
                run_id,
                {
                    'sentiment': sentiment_results,
                    'grooming': grooming_results,
                    'manipulation': manipulation_results,
                    'deception': deception_results,
                    'intent': intent_results,
                    'risk': risk_assessment,
                    'temporal': temporal_results,
                    'insights': insights
                },
                output_dir
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update analysis run
            self.db.update_analysis_run(
                run_id,
                status="completed",
                duration=processing_time,
                results=insights
            )

            # Create result object
            result = ProcessingResult(
                analysis_run_id=run_id,
                input_file=input_file,
                message_count=len(messages),
                speaker_count=len(df['sender'].unique()),
                processing_time=processing_time,
                sentiment_results=sentiment_results,
                grooming_results=grooming_results,
                manipulation_results=manipulation_results,
                deception_results=deception_results,
                intent_results=intent_results,
                risk_assessment=risk_assessment,
                temporal_results=temporal_results,
                overall_risk_level=risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown'),
                primary_concerns=insights.get('primary_concerns', []),
                key_findings=insights.get('key_findings', []),
                recommendations=insights.get('recommendations', []),
                cache_hit=False,
                cache_stats=self.analysis_cache.get_stats() if self.analysis_cache else None,
                json_output=export_paths.get('json'),
                csv_output=export_paths.get('csv'),
                pdf_output=export_paths.get('pdf')
            )

            # Cache the complete analysis results for future re-use
            if self.analysis_cache:
                logger.info("Caching analysis results...")
                self.analysis_cache.cache_analysis(
                    input_file,
                    self.config.to_dict(),
                    {
                        'analysis_run_id': run_id,
                        'input_file': input_file,
                        'message_count': len(messages),
                        'speaker_count': len(df['sender'].unique()),
                        'processing_time': processing_time,
                        'sentiment_results': sentiment_results,
                        'grooming_results': grooming_results,
                        'manipulation_results': manipulation_results,
                        'deception_results': deception_results,
                        'intent_results': intent_results,
                        'risk_assessment': risk_assessment,
                        'temporal_results': temporal_results,
                        'overall_risk_level': risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown'),
                        'primary_concerns': insights.get('primary_concerns', []),
                        'key_findings': insights.get('key_findings', []),
                        'recommendations': insights.get('recommendations', []),
                    }
                )

            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.db.update_analysis_run(
                run_id,
                status="failed",
                error_message=str(e)
            )
            raise

    def _reconstruct_result_from_cache(self, cached_data: Dict, input_file: str,
                                        output_dir: Optional[str]) -> ProcessingResult:
        """Reconstruct ProcessingResult from cached data

        Args:
            cached_data: Cached analysis results
            input_file: Input file path
            output_dir: Output directory

        Returns:
            ProcessingResult: Reconstructed result object
        """
        # Re-export if output_dir specified
        export_paths = {}
        if output_dir:
            export_paths = self._export_results(
                cached_data.get('analysis_run_id', 0),
                {
                    'sentiment': cached_data.get('sentiment_results', {}),
                    'grooming': cached_data.get('grooming_results', {}),
                    'manipulation': cached_data.get('manipulation_results', {}),
                    'deception': cached_data.get('deception_results', {}),
                    'intent': cached_data.get('intent_results', {}),
                    'risk': cached_data.get('risk_assessment', {}),
                    'temporal': cached_data.get('temporal_results', {}),
                    'insights': {
                        'primary_concerns': cached_data.get('primary_concerns', []),
                        'key_findings': cached_data.get('key_findings', []),
                        'recommendations': cached_data.get('recommendations', [])
                    }
                },
                output_dir
            )

        return ProcessingResult(
            analysis_run_id=cached_data.get('analysis_run_id', 0),
            input_file=input_file,
            message_count=cached_data.get('message_count', 0),
            speaker_count=cached_data.get('speaker_count', 0),
            processing_time=0.001,  # Cache retrieval is nearly instant
            sentiment_results=cached_data.get('sentiment_results', {}),
            grooming_results=cached_data.get('grooming_results', {}),
            manipulation_results=cached_data.get('manipulation_results', {}),
            deception_results=cached_data.get('deception_results', {}),
            intent_results=cached_data.get('intent_results', {}),
            risk_assessment=cached_data.get('risk_assessment', {}),
            temporal_results=cached_data.get('temporal_results'),
            overall_risk_level=cached_data.get('overall_risk_level', 'unknown'),
            primary_concerns=cached_data.get('primary_concerns', []),
            key_findings=cached_data.get('key_findings', []),
            recommendations=cached_data.get('recommendations', []),
            cache_hit=True,
            cache_stats=self.analysis_cache.get_stats() if self.analysis_cache else None,
            json_output=export_paths.get('json'),
            csv_output=export_paths.get('csv'),
            pdf_output=export_paths.get('pdf')
        )

    def _perform_temporal_analysis(self, messages: List[Dict],
                                   risk_assessment: Dict) -> Dict[str, Any]:
        """Perform temporal pattern analysis with timestamp validation

        Args:
            messages: List of messages
            risk_assessment: Risk assessment results

        Returns:
            Dict: Temporal analysis results
        """
        temporal_results = {
            'enabled': False,
            'validation': None,
            'analysis': None
        }

        try:
            # First, validate timestamps
            validation = self.timestamp_validator.validate_timestamps(messages)
            temporal_results['validation'] = validation.__dict__

            if validation.can_use_temporal_analysis:
                logger.info("✅ Timestamps valid - performing temporal analysis")
                logger.info(f"   Coverage: {validation.coverage_percentage:.1f}%")
                logger.info(f"   Timespan: {validation.timespan_days:.1f} days")

                # Enrich messages with risk scores for temporal analysis
                enriched_messages = []
                for i, msg in enumerate(messages):
                    enriched_msg = msg.copy()

                    # Add timestamp from date/time fields
                    if 'date' in msg and 'time' in msg:
                        try:
                            from datetime import datetime
                            date_str = f"{msg['date']} {msg['time']}"
                            enriched_msg['timestamp'] = pd.to_datetime(date_str)
                        except:
                            pass

                    # Add risk score from risk assessment
                    if i < len(risk_assessment.get('per_message_risks', [])):
                        risk = risk_assessment['per_message_risks'][i]
                        enriched_msg['risk_score'] = getattr(risk, 'overall_risk', 0)
                        enriched_msg['risk_analysis'] = {'overall_risk': getattr(risk, 'overall_risk', 0)}

                    enriched_messages.append(enriched_msg)

                # Perform temporal analysis
                temporal_analysis = self.temporal_analyzer.analyze_temporal_patterns(enriched_messages)
                temporal_results['analysis'] = temporal_analysis.__dict__
                temporal_results['enabled'] = True

                # Log findings
                if temporal_analysis.is_escalating:
                    logger.warning(f"⚠️  ESCALATION DETECTED: Score {temporal_analysis.escalation_score:.2f}")
                if temporal_analysis.frequency_increasing:
                    logger.warning(f"⚠️  Frequency increased by {temporal_analysis.frequency_change_percent:.0f}%")

            else:
                logger.warning("⚠️  Timestamps insufficient for temporal analysis")
                logger.info(f"   Issues: {len(validation.issues)}")
                for issue in validation.issues[:3]:
                    logger.info(f"     - {issue}")

        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            temporal_results['error'] = str(e)

        return temporal_results

    def _dataframe_to_messages(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert dataframe to message list

        Args:
            df: Pandas dataframe

        Returns:
            List of message dictionaries
        """
        messages = []

        for idx, row in df.iterrows():
            msg = {
                'index': idx,
                'text': str(row.get('text', '')),
                'sender': str(row.get('sender', 'Unknown')),
                'date': row.get('date'),
                'time': row.get('time'),
                'Type': row.get('Type'),
                'Service': row.get('Service'),
                'Recipients': row.get('Recipients'),
                'Attachment': row.get('Attachment')
            }

            # Add any additional columns
            for col in df.columns:
                if col not in msg:
                    msg[col] = row[col]

            messages.append(msg)

        return messages

    def _process_sentiment(self, messages: List[Dict]) -> Dict[str, Any]:
        """Process sentiment analysis for all messages

        Args:
            messages: List of messages

        Returns:
            Dict: Sentiment analysis results
        """
        # Use multiprocessing for parallel processing if configured
        if self.config.analysis.workers > 1:
            return self._process_sentiment_parallel(messages)

        # Sequential processing
        message_sentiments = []
        for msg in messages:
            sentiment = self.sentiment_analyzer.analyze_text(msg['text'])
            message_sentiments.append(sentiment)

        # Analyze conversation-level sentiment
        conversation_sentiment = self.sentiment_analyzer.analyze_conversation(messages)

        return {
            'per_message': message_sentiments,
            'conversation': conversation_sentiment
        }

    def _process_sentiment_parallel(self, messages: List[Dict]) -> Dict[str, Any]:
        """Process sentiment in parallel

        Args:
            messages: List of messages

        Returns:
            Dict: Sentiment results
        """
        # Extract texts for parallel processing
        texts = [msg['text'] for msg in messages]

        # Process in parallel
        with Pool(self.config.analysis.workers) as pool:
            message_sentiments = pool.map(self.sentiment_analyzer.analyze_text, texts)

        # Analyze conversation-level sentiment
        conversation_sentiment = self.sentiment_analyzer.analyze_conversation(messages)

        return {
            'per_message': message_sentiments,
            'conversation': conversation_sentiment
        }

    def _perform_risk_assessment(self, messages: List[Dict], analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment

        Args:
            messages: List of messages
            analyses: Dictionary of all analysis results

        Returns:
            Dict: Risk assessment results
        """
        # Extract per-message analyses
        per_message_risks = []

        for i, msg in enumerate(messages):
            # Get relevant analyses for this message
            msg_grooming = None
            msg_manipulation = None
            msg_deception = None
            msg_sentiment = None
            msg_intent = None

            # Extract from analysis results
            if 'grooming' in analyses and 'per_message_analysis' in analyses['grooming']:
                if i < len(analyses['grooming']['per_message_analysis']):
                    msg_grooming = analyses['grooming']['per_message_analysis'][i].get('analysis')

            if 'manipulation' in analyses and 'per_message_analysis' in analyses['manipulation']:
                if i < len(analyses['manipulation']['per_message_analysis']):
                    msg_manipulation = analyses['manipulation']['per_message_analysis'][i].get('analysis')

            if 'deception' in analyses and 'per_message_analysis' in analyses['deception']:
                if i < len(analyses['deception']['per_message_analysis']):
                    msg_deception = analyses['deception']['per_message_analysis'][i].get('analysis')

            if 'sentiment' in analyses and 'per_message' in analyses['sentiment']:
                if i < len(analyses['sentiment']['per_message']):
                    msg_sentiment = analyses['sentiment']['per_message'][i]

            if 'intent' in analyses and 'per_message_intents' in analyses['intent']:
                if i < len(analyses['intent']['per_message_intents']):
                    msg_intent = analyses['intent']['per_message_intents'][i]

            # Assess risk for this message
            risk = self.risk_scorer.assess_risk(
                grooming_analysis=msg_grooming,
                manipulation_analysis=msg_manipulation,
                deception_analysis=msg_deception,
                sentiment_analysis=msg_sentiment,
                intent_analysis=msg_intent,
                message_text=msg['text']
            )

            per_message_risks.append(risk)

        # Analyze conversation-level risks
        conversation_risks = self.risk_scorer.analyze_conversation_risks(messages, analyses)

        return {
            'per_message_risks': per_message_risks,
            'conversation_risks': conversation_risks,
            'overall_risk_assessment': conversation_risks.get('overall_risk_assessment')
        }

    def _store_patterns(self, run_id: int, messages: List[Dict], risk_assessment: Dict):
        """Store detected patterns in database

        Args:
            run_id: Analysis run ID
            messages: List of messages
            risk_assessment: Risk assessment results
        """
        patterns = []

        # Extract patterns from per-message risks
        for i, msg in enumerate(messages):
            if i < len(risk_assessment.get('per_message_risks', [])):
                risk = risk_assessment['per_message_risks'][i]

                if risk.primary_concern:
                    pattern = Pattern(
                        analysis_run_id=run_id,
                        message_id=i,  # Using index as message_id for now
                        pattern_type=risk.primary_concern,
                        pattern_subtype=None,
                        severity=risk.overall_risk,
                        confidence=risk.assessment_confidence
                    )
                    patterns.append(pattern)

        # Store patterns in batch
        if patterns:
            self.db.insert_patterns_batch(patterns)
            logger.info(f"Stored {len(patterns)} patterns in database")

    def _generate_insights(self, sentiment: Dict, grooming: Dict, manipulation: Dict,
                          deception: Dict, intent: Dict, risk: Dict, temporal: Dict = None) -> Dict[str, Any]:
        """Generate key insights and recommendations

        Args:
            Various analysis results including temporal

        Returns:
            Dict: Insights and recommendations
        """
        insights = {
            'primary_concerns': [],
            'key_findings': [],
            'recommendations': [],
            'summary': {}
        }

        # Identify primary concerns
        if risk.get('overall_risk_assessment'):
            overall_risk = risk['overall_risk_assessment']
            if overall_risk.get('primary_concern'):
                insights['primary_concerns'].append(overall_risk['primary_concern'])

            if overall_risk.get('risk_level') in ['high', 'critical']:
                insights['primary_concerns'].append(f"{overall_risk['risk_level']} risk level")

        # Key findings from sentiment
        if sentiment.get('conversation'):
            conv_sentiment = sentiment['conversation']
            if conv_sentiment.get('sentiment_trajectory'):
                insights['key_findings'].append(
                    f"Sentiment trajectory: {conv_sentiment['sentiment_trajectory']}"
                )
            if conv_sentiment.get('emotional_volatility', 0) > 0.5:
                insights['key_findings'].append("High emotional volatility detected")

        # Key findings from grooming
        if grooming.get('overall_risk'):
            insights['key_findings'].append(f"Grooming risk: {grooming['overall_risk']}")

        # Key findings from manipulation
        if manipulation.get('overall_risk'):
            insights['key_findings'].append(f"Manipulation risk: {manipulation['overall_risk']}")

        # Key findings from deception
        if deception.get('overall_credibility'):
            insights['key_findings'].append(f"Credibility assessment: {deception['overall_credibility']}")

        # Key findings from intent
        if intent.get('conversation_dynamic'):
            insights['key_findings'].append(f"Conversation dynamic: {intent['conversation_dynamic']}")

        # Key findings from temporal analysis
        if temporal and temporal.get('enabled') and temporal.get('analysis'):
            temp_analysis = temporal['analysis']

            if temp_analysis.get('is_escalating'):
                insights['key_findings'].append(
                    f"⚠️  Risk escalation detected (score: {temp_analysis.get('escalation_score', 0):.2f})"
                )
                insights['primary_concerns'].append("Risk escalation over time")

            if temp_analysis.get('frequency_increasing'):
                insights['key_findings'].append(
                    f"Message frequency increased by {temp_analysis.get('frequency_change_percent', 0):.0f}%"
                )

            if temp_analysis.get('is_progressing_through_stages'):
                insights['key_findings'].append("Progression through concerning behavioral stages detected")

            # Add temporal warnings
            if temp_analysis.get('warnings'):
                insights['recommendations'].extend(temp_analysis['warnings'][:3])

        # Compile recommendations
        all_recommendations = set()

        # Add risk recommendations
        if risk.get('overall_risk_assessment', {}).get('recommendations'):
            all_recommendations.update(risk['overall_risk_assessment']['recommendations'])

        # Add grooming recommendations
        if grooming.get('recommendations'):
            all_recommendations.update(grooming['recommendations'])

        # Add manipulation recommendations
        if manipulation.get('recommendations'):
            all_recommendations.update(manipulation['recommendations'])

        # Add intent recommendations
        if intent.get('recommendations'):
            all_recommendations.update(intent['recommendations'])

        insights['recommendations'] = list(all_recommendations)

        # Generate summary
        insights['summary'] = {
            'risk_level': risk.get('overall_risk_assessment', {}).get('risk_level', 'unknown'),
            'primary_concern': risk.get('overall_risk_assessment', {}).get('primary_concern'),
            'intervention_priority': risk.get('overall_risk_assessment', {}).get('intervention_priority', 'routine'),
            'key_findings_count': len(insights['key_findings']),
            'recommendations_count': len(insights['recommendations'])
        }

        return insights

    def _export_results(self, run_id: int, results: Dict, output_dir: Optional[str]) -> Dict[str, str]:
        """Export results to various formats

        Args:
            run_id: Analysis run ID
            results: All analysis results
            output_dir: Output directory

        Returns:
            Dict: Paths to exported files
        """
        if not output_dir:
            output_dir = "Reports"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"analysis_{run_id}_{timestamp}"

        export_paths = {}

        # Export JSON
        json_path = output_path / f"{base_name}.json"
        with open(json_path, 'w') as f:
            # Convert any dataclass objects to dicts
            json_safe = self._make_json_serializable(results)
            json.dump(json_safe, f, indent=2)
        export_paths['json'] = str(json_path)
        logger.info(f"Exported JSON to {json_path}")

        # Export CSV (timeline/summary)
        csv_path = output_path / f"{base_name}_timeline.csv"
        self._export_timeline_csv(results, csv_path)
        export_paths['csv'] = str(csv_path)
        logger.info(f"Exported CSV to {csv_path}")

        # Note: PDF export would be implemented with ReportLab in Phase 4

        return export_paths

    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable

        Args:
            obj: Object to serialize

        Returns:
            JSON-safe object
        """
        if hasattr(obj, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        else:
            return obj

    def _export_timeline_csv(self, results: Dict, csv_path: Path):
        """Export timeline data to CSV

        Args:
            results: Analysis results
            csv_path: Output CSV path
        """
        timeline_data = []

        # Extract sentiment timeline if available
        if 'sentiment' in results and 'per_message' in results['sentiment']:
            for i, sentiment in enumerate(results['sentiment']['per_message']):
                timeline_data.append({
                    'message_index': i,
                    'sentiment': sentiment.combined_sentiment if hasattr(sentiment, 'combined_sentiment') else 0,
                    'subjectivity': sentiment.textblob_subjectivity if hasattr(sentiment, 'textblob_subjectivity') else 0
                })

        if timeline_data:
            df = pd.DataFrame(timeline_data)
            df.to_csv(csv_path, index=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Message Processor - Comprehensive chat analysis system"
    )

    parser.add_argument(
        "input_file",
        help="Path to input CSV file containing messages"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for results (default: Reports)",
        default="Reports"
    )

    parser.add_argument(
        "-c", "--config",
        help="Configuration file or preset name",
        default=None
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers",
        default=4
    )

    parser.add_argument(
        "--no-grooming",
        action="store_true",
        help="Disable grooming detection"
    )

    parser.add_argument(
        "--no-manipulation",
        action="store_true",
        help="Disable manipulation detection"
    )

    parser.add_argument(
        "--no-deception",
        action="store_true",
        help="Disable deception analysis"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)

    # Apply command-line overrides
    if args.workers:
        config.analysis.workers = args.workers

    if args.no_grooming:
        config.nlp.enable_grooming_detection = False

    if args.no_manipulation:
        config.nlp.enable_manipulation_detection = False

    if args.no_deception:
        config.nlp.enable_deception_markers = False

    # Create processor
    processor = MessageProcessor(config)

    try:
        # Process file
        result = processor.process_file(args.input_file, args.output)

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Messages processed: {result.message_count}")
        print(f"Speakers identified: {result.speaker_count}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Overall risk level: {result.overall_risk_level}")

        if result.primary_concerns:
            print(f"\nPrimary concerns:")
            for concern in result.primary_concerns:
                print(f"  • {concern}")

        if result.key_findings:
            print(f"\nKey findings:")
            for finding in result.key_findings[:5]:
                print(f"  • {finding}")

        if result.recommendations:
            print(f"\nTop recommendations:")
            for rec in result.recommendations[:3]:
                print(f"  • {rec}")

        print(f"\nResults exported to:")
        print(f"  • JSON: {result.json_output}")
        print(f"  • CSV: {result.csv_output}")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())