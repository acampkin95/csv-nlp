"""
Enhanced Sentiment Analysis Module
Integrates VADER, TextBlob, and NLTK for comprehensive sentiment and subjectivity analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

# Third-party imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    logger.warning("VADER not installed. Install with: pip install vaderSentiment")
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except ImportError:
    logger.warning("TextBlob not installed. Install with: pip install textblob")
    TextBlob = None

try:
    from nrclex import NRCLex
except ImportError:
    logger.warning("NRCLex not installed. Install with: pip install nrclex")
    NRCLex = None

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    # VADER scores
    vader_compound: float = 0.0
    vader_positive: float = 0.0
    vader_negative: float = 0.0
    vader_neutral: float = 0.0

    # TextBlob scores
    textblob_polarity: float = 0.0  # -1 to 1
    textblob_subjectivity: float = 0.0  # 0 to 1

    # NRCLex emotions
    emotions: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: Optional[str] = None

    # Combined metrics
    combined_sentiment: float = 0.0
    sentiment_label: str = "neutral"  # positive, negative, neutral
    confidence: float = 0.0

    # Subjectivity assessment
    subjectivity_label: str = "objective"  # objective, subjective, mixed

    # Emotional intensity
    emotional_intensity: float = 0.0


@dataclass
class ConversationSentiment:
    """Sentiment analysis for entire conversation"""
    overall_sentiment: float = 0.0
    sentiment_trajectory: str = "stable"  # improving, declining, stable, volatile
    emotional_volatility: float = 0.0
    dominant_emotions: List[str] = field(default_factory=list)
    speaker_sentiments: Dict[str, Dict] = field(default_factory=dict)
    sentiment_shifts: List[Dict] = field(default_factory=list)
    peak_positive: Tuple[int, float] = (0, 0.0)
    peak_negative: Tuple[int, float] = (0, 0.0)


class SentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple engines"""

    # Sentiment thresholds
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1

    # Subjectivity thresholds
    OBJECTIVE_THRESHOLD = 0.3
    SUBJECTIVE_THRESHOLD = 0.7

    # Emotion categories
    BASIC_EMOTIONS = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
    COMPLEX_EMOTIONS = ['trust', 'anticipation', 'love', 'optimism', 'pessimism', 'contempt']

    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        self.has_textblob = TextBlob is not None
        self.has_nrclex = NRCLex is not None

        if not any([self.vader, self.has_textblob]):
            logger.error("No sentiment analysis libraries available. Please install required packages.")

    def analyze_text(self, text: str) -> SentimentResult:
        """Perform comprehensive sentiment analysis on text

        Args:
            text: Text to analyze

        Returns:
            SentimentResult: Complete sentiment analysis
        """
        result = SentimentResult()

        if not text:
            return result

        # VADER analysis
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            result.vader_compound = vader_scores['compound']
            result.vader_positive = vader_scores['pos']
            result.vader_negative = vader_scores['neg']
            result.vader_neutral = vader_scores['neu']

        # TextBlob analysis
        if self.has_textblob:
            try:
                blob = TextBlob(text)
                result.textblob_polarity = blob.sentiment.polarity
                result.textblob_subjectivity = blob.sentiment.subjectivity
            except Exception as e:
                logger.error(f"TextBlob analysis failed: {e}")

        # NRCLex emotion analysis
        if self.has_nrclex:
            try:
                emotion_analyzer = NRCLex(text)
                # Get raw emotion frequencies
                raw_emotions = emotion_analyzer.affect_frequencies

                # Normalize and filter emotions
                total = sum(raw_emotions.values())
                if total > 0:
                    result.emotions = {
                        emotion: freq/total
                        for emotion, freq in raw_emotions.items()
                        if freq > 0
                    }

                    # Find dominant emotion
                    if result.emotions:
                        result.dominant_emotion = max(
                            result.emotions.items(),
                            key=lambda x: x[1]
                        )[0]
            except Exception as e:
                logger.error(f"NRCLex analysis failed: {e}")

        # Calculate combined sentiment (weighted average)
        sentiment_scores = []
        weights = []

        if self.vader:
            sentiment_scores.append(result.vader_compound)
            weights.append(0.5)  # VADER is very reliable

        if self.has_textblob:
            sentiment_scores.append(result.textblob_polarity)
            weights.append(0.3)  # TextBlob is good but less specialized

        if sentiment_scores:
            result.combined_sentiment = sum(s * w for s, w in zip(sentiment_scores, weights)) / sum(weights)

        # Determine sentiment label
        if result.combined_sentiment > self.POSITIVE_THRESHOLD:
            result.sentiment_label = "positive"
        elif result.combined_sentiment < self.NEGATIVE_THRESHOLD:
            result.sentiment_label = "negative"
        else:
            result.sentiment_label = "neutral"

        # Calculate confidence (agreement between analyzers)
        if len(sentiment_scores) > 1:
            variance = statistics.variance(sentiment_scores) if len(sentiment_scores) > 1 else 0
            result.confidence = max(0, 1 - (variance * 2))  # Lower variance = higher confidence
        else:
            result.confidence = 0.7  # Default confidence with single analyzer

        # Determine subjectivity label
        if self.has_textblob:
            if result.textblob_subjectivity < self.OBJECTIVE_THRESHOLD:
                result.subjectivity_label = "objective"
            elif result.textblob_subjectivity > self.SUBJECTIVE_THRESHOLD:
                result.subjectivity_label = "subjective"
            else:
                result.subjectivity_label = "mixed"

        # Calculate emotional intensity
        result.emotional_intensity = self._calculate_emotional_intensity(result)

        return result

    def _calculate_emotional_intensity(self, result: SentimentResult) -> float:
        """Calculate overall emotional intensity

        Args:
            result: Sentiment result

        Returns:
            float: Emotional intensity (0-1)
        """
        intensity = 0.0

        # Factor 1: Sentiment extremity
        sentiment_extremity = abs(result.combined_sentiment)
        intensity += sentiment_extremity * 0.3

        # Factor 2: Subjectivity (more subjective = more emotional)
        if self.has_textblob:
            intensity += result.textblob_subjectivity * 0.3

        # Factor 3: Emotion presence
        if result.emotions:
            # Strong emotions increase intensity
            strong_emotions = ['anger', 'fear', 'disgust', 'joy', 'love']
            strong_emotion_score = sum(
                result.emotions.get(emotion, 0)
                for emotion in strong_emotions
            )
            intensity += min(strong_emotion_score * 0.4, 0.4)

        return min(intensity, 1.0)

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> ConversationSentiment:
        """Analyze sentiment across entire conversation

        Args:
            messages: List of messages with 'text', 'sender' keys

        Returns:
            ConversationSentiment: Conversation-level sentiment analysis
        """
        conv_sentiment = ConversationSentiment()

        if not messages:
            return conv_sentiment

        # Analyze each message
        message_sentiments = []
        speaker_data = {}
        all_emotions = {}

        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Analyze message
            sentiment = self.analyze_text(text)
            message_sentiments.append(sentiment.combined_sentiment)

            # Track by speaker
            if sender not in speaker_data:
                speaker_data[sender] = {
                    'sentiments': [],
                    'emotions': {},
                    'subjectivity': []
                }

            speaker_data[sender]['sentiments'].append(sentiment.combined_sentiment)
            speaker_data[sender]['subjectivity'].append(sentiment.textblob_subjectivity)

            # Aggregate emotions
            for emotion, score in sentiment.emotions.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + score
                speaker_data[sender]['emotions'][emotion] = \
                    speaker_data[sender]['emotions'].get(emotion, 0) + score

            # Track peaks
            if sentiment.combined_sentiment > conv_sentiment.peak_positive[1]:
                conv_sentiment.peak_positive = (i, sentiment.combined_sentiment)

            if sentiment.combined_sentiment < conv_sentiment.peak_negative[1]:
                conv_sentiment.peak_negative = (i, sentiment.combined_sentiment)

            # Detect sentiment shifts
            if i > 0:
                prev_sentiment = message_sentiments[i-1]
                shift = sentiment.combined_sentiment - prev_sentiment

                if abs(shift) > 0.5:  # Significant shift
                    conv_sentiment.sentiment_shifts.append({
                        'index': i,
                        'sender': sender,
                        'shift_magnitude': shift,
                        'from_sentiment': prev_sentiment,
                        'to_sentiment': sentiment.combined_sentiment,
                        'type': 'positive_shift' if shift > 0 else 'negative_shift'
                    })

        # Calculate overall metrics
        if message_sentiments:
            conv_sentiment.overall_sentiment = statistics.mean(message_sentiments)

            # Calculate volatility (standard deviation of sentiments)
            if len(message_sentiments) > 1:
                conv_sentiment.emotional_volatility = statistics.stdev(message_sentiments)

            # Determine trajectory
            conv_sentiment.sentiment_trajectory = self._determine_trajectory(message_sentiments)

        # Find dominant emotions
        if all_emotions:
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
            conv_sentiment.dominant_emotions = [e[0] for e in sorted_emotions[:3]]

        # Compile speaker sentiments
        for sender, data in speaker_data.items():
            conv_sentiment.speaker_sentiments[sender] = {
                'average_sentiment': statistics.mean(data['sentiments']) if data['sentiments'] else 0,
                'sentiment_variance': statistics.variance(data['sentiments']) if len(data['sentiments']) > 1 else 0,
                'average_subjectivity': statistics.mean(data['subjectivity']) if data['subjectivity'] else 0,
                'dominant_emotion': max(data['emotions'].items(), key=lambda x: x[1])[0] if data['emotions'] else None,
                'message_count': len(data['sentiments'])
            }

        return conv_sentiment

    def _determine_trajectory(self, sentiments: List[float]) -> str:
        """Determine sentiment trajectory over time

        Args:
            sentiments: List of sentiment scores

        Returns:
            str: Trajectory type
        """
        if len(sentiments) < 3:
            return "stable"

        # Split into thirds
        third = len(sentiments) // 3
        first_third = statistics.mean(sentiments[:third])
        middle_third = statistics.mean(sentiments[third:2*third])
        last_third = statistics.mean(sentiments[2*third:])

        # Check for consistent improvement or decline
        if last_third > middle_third > first_third:
            return "improving"
        elif last_third < middle_third < first_third:
            return "declining"

        # Check for volatility
        variance = statistics.variance(sentiments)
        if variance > 0.25:  # High variance
            return "volatile"

        return "stable"

    def get_sentiment_summary(self, result: SentimentResult) -> str:
        """Generate human-readable sentiment summary

        Args:
            result: Sentiment result

        Returns:
            str: Summary text
        """
        summary_parts = []

        # Overall sentiment
        intensity_desc = "strongly" if abs(result.combined_sentiment) > 0.5 else "moderately"
        if abs(result.combined_sentiment) < 0.1:
            intensity_desc = "slightly"

        summary_parts.append(f"{intensity_desc} {result.sentiment_label}")

        # Subjectivity
        if self.has_textblob:
            summary_parts.append(f"{result.subjectivity_label}")

        # Dominant emotion
        if result.dominant_emotion:
            summary_parts.append(f"primarily expressing {result.dominant_emotion}")

        # Emotional intensity
        if result.emotional_intensity > 0.7:
            summary_parts.append("highly emotional")
        elif result.emotional_intensity > 0.4:
            summary_parts.append("moderately emotional")

        return ", ".join(summary_parts)

    def compare_sentiments(self, sentiment1: SentimentResult, sentiment2: SentimentResult) -> Dict[str, Any]:
        """Compare two sentiment results

        Args:
            sentiment1: First sentiment result
            sentiment2: Second sentiment result

        Returns:
            Dict: Comparison metrics
        """
        comparison = {
            'sentiment_shift': sentiment2.combined_sentiment - sentiment1.combined_sentiment,
            'polarity_change': None,
            'subjectivity_shift': sentiment2.textblob_subjectivity - sentiment1.textblob_subjectivity,
            'emotional_intensity_change': sentiment2.emotional_intensity - sentiment1.emotional_intensity,
            'emotion_changes': {}
        }

        # Determine polarity change
        if sentiment1.sentiment_label != sentiment2.sentiment_label:
            comparison['polarity_change'] = f"{sentiment1.sentiment_label} → {sentiment2.sentiment_label}"

        # Calculate emotion changes
        all_emotions = set(sentiment1.emotions.keys()) | set(sentiment2.emotions.keys())
        for emotion in all_emotions:
            old_val = sentiment1.emotions.get(emotion, 0)
            new_val = sentiment2.emotions.get(emotion, 0)
            change = new_val - old_val

            if abs(change) > 0.1:  # Significant change
                comparison['emotion_changes'][emotion] = change

        return comparison