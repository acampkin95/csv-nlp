#!/usr/bin/env python3
"""
Message Processor - Main Entry Point
Comprehensive chat analysis system with PostgreSQL backend
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
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__) / "src"))

from src.db.postgresql_adapter import PostgreSQLAdapter, DatabaseConfig
from src.validation.csv_validator import CSVValidator
from src.config.config_manager import ConfigManager, Configuration
from src.pipeline.message_processor import MessageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMessageProcessor(MessageProcessor):
    """Enhanced processor with PostgreSQL integration"""

    def __init__(self, config: Configuration, use_postgresql: bool = True):
        """Initialize enhanced processor

        Args:
            config: Configuration object
            use_postgresql: Whether to use PostgreSQL (True) or SQLite (False)
        """
        self.config = config
        self.use_postgresql = use_postgresql

        if use_postgresql:
            # Initialize PostgreSQL adapter
            db_config = DatabaseConfig(
                host="acdev.host",
                database="messagestore",
                user="msgprocess",
                password="DHifde93jes9dk"
            )
            self.db = PostgreSQLAdapter(db_config)
            logger.info("Using PostgreSQL backend at acdev.host")
        else:
            # Fall back to SQLite
            from src.db.database import DatabaseAdapter
            self.db = DatabaseAdapter(config.database.path)
            logger.info("Using SQLite backend")

        # Initialize validators and NLP modules
        self.csv_validator = CSVValidator(auto_correct=True)
        self._init_nlp_modules()

        self.stats = {
            "messages_processed": 0,
            "patterns_detected": 0,
            "processing_time": 0
        }

    def _init_nlp_modules(self):
        """Initialize NLP analysis modules"""
        try:
            from src.nlp.sentiment_analyzer import SentimentAnalyzer
            from src.nlp.grooming_detector import GroomingDetector
            from src.nlp.manipulation_detector import ManipulationDetector
            from src.nlp.deception_analyzer import DeceptionAnalyzer
            from src.nlp.intent_classifier import IntentClassifier
            from src.nlp.risk_scorer import BehavioralRiskScorer

            logger.info("Initializing NLP modules...")
            self.sentiment_analyzer = SentimentAnalyzer()
            self.grooming_detector = GroomingDetector()
            self.manipulation_detector = ManipulationDetector()
            self.deception_analyzer = DeceptionAnalyzer()
            self.intent_classifier = IntentClassifier()
            self.risk_scorer = BehavioralRiskScorer(
                weights={
                    "grooming": self.config.nlp.risk_weight_grooming,
                    "manipulation": self.config.nlp.risk_weight_manipulation,
                    "hostility": self.config.nlp.risk_weight_hostility,
                    "deception": self.config.nlp.risk_weight_deception
                }
            )
            logger.info("NLP modules initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import NLP modules: {e}")
            logger.info("Some analysis features may be unavailable")

    def process_csv_file(self, input_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a CSV file with full pipeline

        Args:
            input_file: Path to CSV file
            output_dir: Optional output directory

        Returns:
            Dict: Processing results
        """
        start_time = time.time()
        logger.info(f"Starting processing of {input_file}")

        try:
            # ========================================
            # PASS 0: Validate and Load CSV
            # ========================================
            logger.info("PASS 0: CSV validation and loading")
            validation_result, df = self.csv_validator.validate_file(input_file)

            if not validation_result.is_valid:
                raise ValueError(f"CSV validation failed: {validation_result.errors}")

            print(f"\n📊 CSV Validation Results:")
            print(f"  • Encoding: {validation_result.encoding}")
            print(f"  • Rows: {len(df)}")
            print(f"  • Columns: {len(df.columns)}")
            print(f"  • Speakers: {len(df.get('sender', df.get('Sender Name', [])).unique())}")

            if validation_result.warnings:
                print(f"  ⚠️  Warnings: {len(validation_result.warnings)}")
                for warning in validation_result.warnings[:3]:
                    print(f"    - {warning}")

            # ========================================
            # PASS 1: Database Import (PostgreSQL)
            # ========================================
            if self.use_postgresql:
                logger.info("PASS 1: Importing to PostgreSQL")
                session_id = self.db.create_csv_import_session(
                    filename=Path(input_file).name,
                    df=df
                )
                print(f"  • CSV imported with session ID: {session_id}")
                print(f"  • Created dedicated table for data integrity")

                # Create analysis run
                analysis_run_id = self.db.create_analysis_run(
                    csv_session_id=session_id,
                    config=self.config.to_dict()
                )
                print(f"  • Analysis run ID: {analysis_run_id}")

                # Get messages from database
                messages = self.db.get_messages(csv_session_id=session_id)
            else:
                # Use in-memory processing
                messages = self._dataframe_to_messages(df)
                analysis_run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # ========================================
            # PASS 2: Sentiment Analysis
            # ========================================
            logger.info("PASS 2: Sentiment analysis")
            print(f"\n🎭 Analyzing sentiment...")
            sentiment_results = self._analyze_sentiment(messages)
            print(f"  • Overall sentiment: {sentiment_results.get('overall_sentiment', 'N/A'):.2f}")
            print(f"  • Trajectory: {sentiment_results.get('trajectory', 'N/A')}")

            # ========================================
            # PASS 3: Pattern Detection
            # ========================================
            logger.info("PASS 3: Pattern detection")
            print(f"\n🔍 Detecting behavioral patterns...")

            # Grooming detection
            grooming_results = {}
            if self.config.nlp.enable_grooming_detection:
                grooming_results = self._analyze_grooming(messages)
                print(f"  • Grooming risk: {grooming_results.get('overall_risk', 'N/A')}")

            # Manipulation detection
            manipulation_results = {}
            if self.config.nlp.enable_manipulation_detection:
                manipulation_results = self._analyze_manipulation(messages)
                print(f"  • Manipulation risk: {manipulation_results.get('overall_risk', 'N/A')}")

            # Deception analysis
            deception_results = {}
            if self.config.nlp.enable_deception_markers:
                deception_results = self._analyze_deception(messages)
                print(f"  • Credibility: {deception_results.get('overall_credibility', 'N/A')}")

            # ========================================
            # PASS 4: Intent Classification
            # ========================================
            logger.info("PASS 4: Intent classification")
            intent_results = {}
            if self.config.nlp.enable_intent_classification:
                intent_results = self._analyze_intent(messages)
                print(f"  • Conversation dynamic: {intent_results.get('conversation_dynamic', 'N/A')}")

            # ========================================
            # PASS 5: Risk Assessment
            # ========================================
            logger.info("PASS 5: Comprehensive risk assessment")
            print(f"\n⚠️  Assessing behavioral risks...")
            risk_assessment = self._assess_risk(
                messages,
                sentiment_results,
                grooming_results,
                manipulation_results,
                deception_results,
                intent_results
            )

            overall_risk = risk_assessment.get('overall_risk_level', 'unknown')
            print(f"  • Overall risk: {overall_risk.upper()}")

            if risk_assessment.get('primary_concerns'):
                print(f"  • Primary concerns:")
                for concern in risk_assessment['primary_concerns'][:3]:
                    print(f"    - {concern}")

            # ========================================
            # PASS 6: Store Results (PostgreSQL)
            # ========================================
            if self.use_postgresql:
                logger.info("PASS 6: Storing analysis results")
                self._store_results(
                    analysis_run_id,
                    risk_assessment,
                    grooming_results,
                    manipulation_results,
                    deception_results
                )

            # ========================================
            # PASS 7: Generate Reports
            # ========================================
            logger.info("PASS 7: Generating reports")
            print(f"\n📄 Generating reports...")
            export_paths = self._export_results(
                analysis_run_id,
                {
                    'sentiment': sentiment_results,
                    'grooming': grooming_results,
                    'manipulation': manipulation_results,
                    'deception': deception_results,
                    'intent': intent_results,
                    'risk': risk_assessment
                },
                output_dir
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update analysis run if using PostgreSQL
            if self.use_postgresql:
                self.db.update_analysis_run(
                    analysis_run_id,
                    status='completed',
                    duration=processing_time,
                    results=risk_assessment
                )

            # ========================================
            # Generate Summary
            # ========================================
            summary = {
                'analysis_run_id': analysis_run_id,
                'input_file': input_file,
                'message_count': len(messages),
                'processing_time': processing_time,
                'overall_risk_level': overall_risk,
                'primary_concerns': risk_assessment.get('primary_concerns', []),
                'recommendations': risk_assessment.get('recommendations', []),
                'export_paths': export_paths
            }

            return summary

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if self.use_postgresql and 'analysis_run_id' in locals():
                self.db.update_analysis_run(
                    analysis_run_id,
                    status='failed',
                    error_message=str(e)
                )
            raise

    def _dataframe_to_messages(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to message list"""
        messages = []
        for idx, row in df.iterrows():
            messages.append({
                'id': str(idx),
                'text': str(row.get('text', row.get('Text', row.get('Message', '')))),
                'sender': str(row.get('sender', row.get('Sender Name', row.get('Sender', 'Unknown')))),
                'timestamp': row.get('timestamp', row.get('Date', '')),
                **row.to_dict()
            })
        return messages

    def _analyze_sentiment(self, messages: List[Dict]) -> Dict:
        """Analyze sentiment across messages"""
        if not hasattr(self, 'sentiment_analyzer'):
            return {}

        conversation_sentiment = self.sentiment_analyzer.analyze_conversation(
            [{'text': m.get('text', ''), 'sender': m.get('sender', '')} for m in messages]
        )

        return {
            'overall_sentiment': conversation_sentiment.overall_sentiment,
            'trajectory': conversation_sentiment.sentiment_trajectory,
            'volatility': conversation_sentiment.emotional_volatility,
            'dominant_emotions': conversation_sentiment.dominant_emotions,
            'speaker_sentiments': conversation_sentiment.speaker_sentiments
        }

    def _analyze_grooming(self, messages: List[Dict]) -> Dict:
        """Analyze grooming patterns"""
        if not hasattr(self, 'grooming_detector'):
            return {}

        return self.grooming_detector.analyze_conversation(
            [{'text': m.get('text', ''), 'sender': m.get('sender', '')} for m in messages]
        )

    def _analyze_manipulation(self, messages: List[Dict]) -> Dict:
        """Analyze manipulation patterns"""
        if not hasattr(self, 'manipulation_detector'):
            return {}

        return self.manipulation_detector.analyze_conversation(
            [{'text': m.get('text', ''), 'sender': m.get('sender', '')} for m in messages]
        )

    def _analyze_deception(self, messages: List[Dict]) -> Dict:
        """Analyze deception markers"""
        if not hasattr(self, 'deception_analyzer'):
            return {}

        return self.deception_analyzer.analyze_conversation(
            [{'text': m.get('text', ''), 'sender': m.get('sender', '')} for m in messages]
        )

    def _analyze_intent(self, messages: List[Dict]) -> Dict:
        """Analyze communication intent"""
        if not hasattr(self, 'intent_classifier'):
            return {}

        return self.intent_classifier.analyze_conversation_intents(
            [{'text': m.get('text', ''), 'sender': m.get('sender', '')} for m in messages]
        )

    def _assess_risk(self, messages, sentiment, grooming, manipulation, deception, intent) -> Dict:
        """Perform comprehensive risk assessment"""
        if not hasattr(self, 'risk_scorer'):
            return {'overall_risk_level': 'unknown', 'primary_concerns': [], 'recommendations': []}

        # Assess individual message risks
        message_risks = []
        for msg in messages:
            risk = self.risk_scorer.assess_risk(
                message_text=msg.get('text', '')
            )
            message_risks.append(risk)

        # Calculate overall risk
        if message_risks:
            avg_risk = sum(r.overall_risk for r in message_risks) / len(message_risks)
            max_risk = max(r.overall_risk for r in message_risks)

            # Determine risk level
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

        # Collect primary concerns
        primary_concerns = []
        if grooming.get('overall_risk') in ['high', 'critical']:
            primary_concerns.append(f"Grooming patterns detected ({grooming['overall_risk']} risk)")
        if manipulation.get('overall_risk') in ['high', 'critical']:
            primary_concerns.append(f"Manipulation tactics identified ({manipulation['overall_risk']} risk)")
        if deception.get('overall_credibility') == 'deceptive':
            primary_concerns.append("High deception markers present")
        if intent.get('conversation_dynamic') == 'adversarial':
            primary_concerns.append("Adversarial conversation dynamic")

        # Generate recommendations
        recommendations = []
        if risk_level in ['critical', 'high']:
            recommendations.append("Seek professional support or intervention")
            recommendations.append("Document all concerning interactions")
            recommendations.append("Consider safety planning if threats present")
        elif risk_level == 'moderate':
            recommendations.append("Monitor situation closely")
            recommendations.append("Set clear boundaries")
            recommendations.append("Consider seeking guidance from a counselor")

        return {
            'overall_risk_level': risk_level,
            'primary_concerns': primary_concerns,
            'recommendations': recommendations,
            'message_risks': message_risks
        }

    def _store_results(self, analysis_run_id, risk_assessment, grooming, manipulation, deception):
        """Store analysis results in PostgreSQL"""
        if not self.use_postgresql:
            return

        # Store patterns
        patterns = []

        # Add grooming patterns
        if grooming.get('high_risk_messages'):
            for msg in grooming['high_risk_messages']:
                patterns.append({
                    'analysis_run_id': analysis_run_id,
                    'message_id': msg.get('index'),
                    'pattern_category': 'grooming',
                    'pattern_type': msg.get('primary_concern'),
                    'severity': 0.8,
                    'confidence': 0.7
                })

        # Add manipulation patterns
        if manipulation.get('escalation_points'):
            for point in manipulation['escalation_points']:
                patterns.append({
                    'analysis_run_id': analysis_run_id,
                    'message_id': point.get('index'),
                    'pattern_category': 'manipulation',
                    'pattern_type': point.get('tactic'),
                    'severity': point.get('severity', 0.7),
                    'confidence': 0.7
                })

        if patterns:
            self.db.insert_patterns_batch(patterns)

        # Store risk assessment
        self.db.save_risk_assessment({
            'analysis_run_id': analysis_run_id,
            'overall_risk': risk_assessment.get('overall_risk_level'),
            'risk_level': risk_assessment.get('overall_risk_level'),
            'primary_concern': risk_assessment.get('primary_concerns', [''])[0] if risk_assessment.get('primary_concerns') else None,
            'recommendations': risk_assessment.get('recommendations', [])
        })

    def _export_results(self, analysis_run_id, results, output_dir):
        """Export results to files"""
        if not output_dir:
            output_dir = "Reports"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"analysis_{timestamp}"

        export_paths = {}

        # Export JSON
        json_path = output_path / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)
        export_paths['json'] = str(json_path)

        # Export summary CSV
        csv_path = output_path / f"{base_name}_summary.csv"
        self._export_summary_csv(results, csv_path)
        export_paths['csv'] = str(csv_path)

        logger.info(f"Results exported to {output_dir}")
        return export_paths

    def _make_json_serializable(self, obj):
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

    def _export_summary_csv(self, results, csv_path):
        """Export summary to CSV"""
        summary_data = []

        # Add risk summary
        if 'risk' in results:
            summary_data.append({
                'Category': 'Overall Risk',
                'Value': results['risk'].get('overall_risk_level', 'unknown'),
                'Details': ', '.join(results['risk'].get('primary_concerns', []))[:100]
            })

        # Add sentiment summary
        if 'sentiment' in results:
            summary_data.append({
                'Category': 'Sentiment',
                'Value': f"{results['sentiment'].get('overall_sentiment', 0):.2f}",
                'Details': results['sentiment'].get('trajectory', '')
            })

        # Add other summaries
        for category in ['grooming', 'manipulation', 'deception', 'intent']:
            if category in results and results[category]:
                risk_level = results[category].get('overall_risk', results[category].get('overall_credibility', 'N/A'))
                summary_data.append({
                    'Category': category.capitalize(),
                    'Value': risk_level,
                    'Details': ''
                })

        if summary_data:
            pd.DataFrame(summary_data).to_csv(csv_path, index=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Message Processor - Comprehensive chat analysis with PostgreSQL backend"
    )

    parser.add_argument(
        "input_file",
        help="Path to CSV file containing messages"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for reports (default: Reports)",
        default="Reports"
    )

    parser.add_argument(
        "-c", "--config",
        help="Configuration file or preset (quick_analysis, deep_analysis, clinical_report, legal_report)",
        default=None
    )

    parser.add_argument(
        "--use-sqlite",
        action="store_true",
        help="Use SQLite instead of PostgreSQL"
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
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Suppress some INFO messages for cleaner output
        logging.getLogger('src').setLevel(logging.WARNING)

    print("=" * 60)
    print("MESSAGE PROCESSOR - PSYCHOLOGICAL ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output}")
    print(f"Backend: {'PostgreSQL (acdev.host)' if not args.use_sqlite else 'SQLite (local)'}")
    print("=" * 60)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)

    # Apply command-line overrides
    if args.no_grooming:
        config.nlp.enable_grooming_detection = False
    if args.no_manipulation:
        config.nlp.enable_manipulation_detection = False
    if args.no_deception:
        config.nlp.enable_deception_markers = False

    # Create processor
    processor = EnhancedMessageProcessor(
        config,
        use_postgresql=not args.use_sqlite
    )

    try:
        # Process file
        result = processor.process_csv_file(args.input_file, args.output)

        # Print final summary
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Messages processed: {result['message_count']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Overall risk level: {result['overall_risk_level'].upper()}")

        if result['primary_concerns']:
            print(f"\n🚨 Primary Concerns:")
            for concern in result['primary_concerns']:
                print(f"  • {concern}")

        if result['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"  • {rec}")

        print(f"\n📁 Results exported to:")
        for file_type, path in result['export_paths'].items():
            print(f"  • {file_type.upper()}: {path}")

        if not args.use_sqlite:
            print(f"\n🗄️  Data stored in PostgreSQL:")
            print(f"  • Analysis ID: {result['analysis_run_id']}")
            print(f"  • Database: acdev.host/messagestore")
            print(f"  • All messages and patterns preserved for future reference")

        return 0

    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {args.input_file}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Clean up database connection
        if hasattr(processor, 'db'):
            if hasattr(processor.db, 'close'):
                processor.db.close()


if __name__ == "__main__":
    sys.exit(main())