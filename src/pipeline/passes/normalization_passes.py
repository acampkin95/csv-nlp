#!/usr/bin/env python3
"""
Normalization Passes (Passes 1-3)

Pass 1: CSV validation and data normalization
Pass 2: Sentiment analysis (VADER, TextBlob, NRCLex)
Pass 3: Emotional dynamics and volatility assessment
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd

from .base_pass import BasePass, PassGroup
from utils.performance import ProgressTracker

logger = logging.getLogger(__name__)


class Pass1_DataValidation(BasePass):
    """Pass 1: CSV validation and data normalization"""

    def __init__(self, csv_validator, cache_manager=None):
        super().__init__(
            pass_number=1,
            pass_name="Data Validation",
            pass_group=PassGroup.NORMALIZATION,
            cache_manager=cache_manager,
            dependencies=[]
        )
        self.csv_validator = csv_validator

    def _execute_pass(self, input_file: str, **kwargs) -> Dict[str, Any]:
        """Execute CSV validation"""
        validation_result, df = self.csv_validator.validate_file(input_file)

        if not validation_result.is_valid:
            raise ValueError(f"CSV validation failed: {validation_result.errors}")

        print(f"  CSV Validated: {len(df)} messages, "
              f"{len(df.get('sender', df.get('Sender Name', [])).unique())} speakers")

        return {
            'is_valid': validation_result.is_valid,
            'encoding': validation_result.encoding,
            'rows': len(df),
            'columns': len(df.columns),
            'warnings': validation_result.warnings if validation_result.warnings else [],
            'dataframe': df  # Pass dataframe to next passes
        }

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for validation failure"""
        return {
            'is_valid': False,
            'error': 'Validation failed',
            'dataframe': None
        }


class Pass2_SentimentAnalysis(BasePass):
    """Pass 2: Sentiment analysis using multiple engines"""

    def __init__(self, sentiment_analyzer, cache_manager=None):
        super().__init__(
            pass_number=2,
            pass_name="Sentiment Analysis",
            pass_group=PassGroup.NORMALIZATION,
            cache_manager=cache_manager,
            dependencies=[]  # Doesn't depend on pass 1 results, but needs messages
        )
        self.sentiment_analyzer = sentiment_analyzer

    def _execute_pass(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute sentiment analysis with progress tracking"""
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

        overall = (conversation_sentiment.overall_sentiment
                   if hasattr(conversation_sentiment, 'overall_sentiment') else 0.0)
        trajectory = getattr(conversation_sentiment, 'sentiment_trajectory', 'N/A')

        print(f"  Overall Sentiment: {overall:.2f}, Trajectory: {trajectory}")

        return {
            'per_message': message_sentiments,
            'conversation': conversation_sentiment
        }

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for sentiment analysis failure"""
        return {
            'per_message': [],
            'conversation': None,
            'error': 'Sentiment analysis failed'
        }


class Pass3_EmotionalDynamics(BasePass):
    """Pass 3: Emotional dynamics and volatility assessment"""

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=3,
            pass_name="Emotional Dynamics",
            pass_group=PassGroup.NORMALIZATION,
            cache_manager=cache_manager,
            dependencies=['sentiment_analysis']  # Depends on pass 2
        )

    def _execute_pass(self, sentiment_results: Dict, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute emotional dynamics analysis"""
        if not sentiment_results.get('per_message'):
            return {'error': 'No sentiment data available'}

        sentiments = [
            s.combined_sentiment if hasattr(s, 'combined_sentiment') else 0.0
            for s in sentiment_results['per_message'] if s is not None
        ]

        if len(sentiments) < 2:
            volatility = 0.0
        else:
            import statistics
            volatility = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0

        print(f"  Emotional Volatility: {volatility:.2f}")

        emotion_shifts = self._detect_emotion_shifts(sentiments)

        return {
            'volatility': volatility,
            'sentiments': sentiments,
            'emotion_shifts': emotion_shifts
        }

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

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for emotional dynamics failure"""
        return {
            'volatility': 0.0,
            'sentiments': [],
            'emotion_shifts': [],
            'error': 'Emotional dynamics analysis failed'
        }
