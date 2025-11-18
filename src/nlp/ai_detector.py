"""
AI-Generated Text Detection Module
Detects AI-generated content in messages using multiple detection methods.

Note: User requested "scalpel-ai" but no such library exists.
Implementation uses aidetector library with heuristic fallback.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import Counter

# Import model cache for performance optimization
from .model_cache import get_cache

# Third-party imports - aidetector for ML-based detection
try:
    from aidetector.tokenization import get_tokenizer
    from aidetector.inference import check_input
    from aidetector.aidetectorclass import AiDetector
except ImportError:
    AiDetector = None
    get_tokenizer = None
    check_input = None

logger = logging.getLogger(__name__)


@dataclass
class AIDetectionResult:
    """Container for AI detection results"""
    # Detection scores
    is_ai_generated: bool = False
    ai_confidence: float = 0.0  # 0-1 confidence score
    detection_method: str = "none"  # aidetector, heuristic, none

    # Heuristic indicators
    repetition_score: float = 0.0  # Higher = more repetitive
    formality_score: float = 0.0  # Higher = more formal
    complexity_score: float = 0.0  # Higher = more complex
    pattern_score: float = 0.0  # AI writing patterns detected

    # Detailed indicators
    ai_indicators: List[str] = field(default_factory=list)
    human_indicators: List[str] = field(default_factory=list)

    # Flags for reporting
    flag_for_review: bool = False
    confidence_level: str = "unknown"  # high, medium, low, unknown


@dataclass
class ConversationAIDetection:
    """AI detection analysis for entire conversation"""
    overall_ai_likelihood: float = 0.0  # 0-1 overall score
    messages_flagged: int = 0
    total_messages: int = 0
    ai_percentage: float = 0.0

    # Per-speaker analysis
    speaker_ai_scores: Dict[str, float] = field(default_factory=dict)
    speakers_flagged: List[str] = field(default_factory=list)

    # Patterns
    ai_message_indices: List[int] = field(default_factory=list)
    consecutive_ai_messages: int = 0

    # High confidence detections
    high_confidence_ai: List[Dict] = field(default_factory=list)


class AIContentDetector:
    """Detector for AI-generated content with multiple detection methods"""

    # AI writing pattern indicators
    AI_PATTERNS = [
        r'\bas an AI(?: language model)?\b',
        r'\bI (?:don\'t|cannot|can\'t) (?:have|provide|access)\b',
        r'\bI apologize,? but\b',
        r'\bit\'s important to note that\b',
        r'\bplease note that\b',
        r'\bin conclusion\b',
        r'\bto summarize\b',
        r'\bfurthermore\b',
        r'\bmoreover\b',
        r'\bconversely\b',
        r'\bnevertheless\b',
        r'\bnotwithstanding\b',
    ]

    # Overly formal phrases common in AI
    FORMAL_PHRASES = [
        'it is important to',
        'one must consider',
        'it should be noted',
        'with regard to',
        'in terms of',
        'with respect to',
        'it is worth noting',
        'it is essential to',
        'one should',
        'it would be prudent',
    ]

    # Thresholds
    AI_CONFIDENCE_THRESHOLD = 0.7  # High confidence AI
    HEURISTIC_THRESHOLD = 0.6  # Heuristic detection threshold
    FLAG_FOR_REVIEW_THRESHOLD = 0.5  # Lower threshold for flagging

    def __init__(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None):
        """Initialize AI content detector

        Args:
            model_path: Path to trained aidetector model (optional)
            vocab_path: Path to vocabulary file (optional)
        """
        self.model_path = model_path
        self.vocab_path = vocab_path

        # Check if aidetector is available
        self.has_aidetector = AiDetector is not None

        if not self.has_aidetector:
            logger.warning(
                "aidetector library not installed. Using heuristic-based detection only. "
                "Install with: pip install aidetector"
            )
            logger.info(
                "Note: User requested 'scalpel-ai' but no such library exists. "
                "Using aidetector as alternative AI text detection solution."
            )

        # Initialize aidetector if available and paths provided
        self.detector = None
        if self.has_aidetector and model_path and vocab_path:
            try:
                self.detector = AiDetector(model_path, vocab_path)
                logger.info("✅ AI detector initialized with custom model")
            except Exception as e:
                logger.warning(f"Failed to initialize custom AI detector model: {e}")
                logger.info("Falling back to heuristic-based detection")

        # Compile AI patterns for faster matching
        cache = get_cache()
        self.compiled_ai_patterns = cache.get_or_load(
            'ai_detection_patterns',
            self._compile_patterns
        )

    @staticmethod
    def _compile_patterns():
        """Compile regex patterns for AI detection"""
        return [re.compile(pattern, re.IGNORECASE) for pattern in AIContentDetector.AI_PATTERNS]

    def detect_ai_content(self, text: str) -> AIDetectionResult:
        """Detect if text is AI-generated

        Args:
            text: Text to analyze

        Returns:
            AIDetectionResult: Detection results with confidence scores
        """
        result = AIDetectionResult()

        if not text or len(text.strip()) < 20:
            # Too short to reliably detect
            result.detection_method = "skipped"
            return result

        # Try ML-based detection first (if available)
        if self.has_aidetector and self.detector:
            try:
                prediction = check_input(text, self.detector, get_tokenizer())
                if prediction == 1:
                    result.is_ai_generated = True
                    result.ai_confidence = 0.85  # aidetector doesn't return confidence, use high default
                    result.detection_method = "aidetector"
                    result.confidence_level = "high"
                    result.ai_indicators.append("ML model detected AI-generated content")
                else:
                    result.detection_method = "aidetector"
                    result.confidence_level = "high"
            except Exception as e:
                logger.debug(f"AI detector inference failed: {e}")

        # Always run heuristic analysis for additional context
        heuristic_score, indicators = self._heuristic_detection(text)
        result.pattern_score = heuristic_score

        # If ML detection not available, use heuristics as primary method
        if result.detection_method == "none":
            result.detection_method = "heuristic"
            if heuristic_score >= self.HEURISTIC_THRESHOLD:
                result.is_ai_generated = True
                result.ai_confidence = heuristic_score
                result.confidence_level = self._get_confidence_level(heuristic_score)
                result.ai_indicators.extend(indicators)
            else:
                result.ai_confidence = heuristic_score
                result.confidence_level = "low"
                if heuristic_score > 0.3:
                    result.ai_indicators.extend(indicators)
                    result.human_indicators.append("No strong AI patterns detected")
                else:
                    result.human_indicators.append("Natural conversational style")
                    result.human_indicators.append("Low formality and repetition")

        # Calculate component scores
        result.repetition_score = self._calculate_repetition(text)
        result.formality_score = self._calculate_formality(text)
        result.complexity_score = self._calculate_complexity(text)

        # Flag for review if confidence is medium or higher
        if result.ai_confidence >= self.FLAG_FOR_REVIEW_THRESHOLD:
            result.flag_for_review = True

        return result

    def _heuristic_detection(self, text: str) -> Tuple[float, List[str]]:
        """Perform heuristic-based AI detection

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score 0-1, list of indicators)
        """
        indicators = []
        score = 0.0
        weights_sum = 0.0

        text_lower = text.lower()

        # Check for explicit AI patterns (high weight)
        pattern_matches = sum(1 for pattern in self.compiled_ai_patterns if pattern.search(text))
        if pattern_matches > 0:
            pattern_weight = min(pattern_matches * 0.3, 0.6)
            score += pattern_weight
            weights_sum += 0.6
            indicators.append(f"Contains {pattern_matches} AI self-reference pattern(s)")
        else:
            weights_sum += 0.6

        # Check for overly formal phrases (medium weight)
        formal_count = sum(1 for phrase in self.FORMAL_PHRASES if phrase in text_lower)
        if formal_count >= 2:
            formal_weight = min(formal_count * 0.1, 0.3)
            score += formal_weight
            weights_sum += 0.3
            indicators.append(f"Contains {formal_count} overly formal phrase(s)")
        else:
            weights_sum += 0.3

        # Check for repetition (low weight)
        repetition = self._calculate_repetition(text)
        if repetition > 0.3:
            score += repetition * 0.1
            weights_sum += 0.1
            indicators.append(f"High word repetition ({repetition:.0%})")
        else:
            weights_sum += 0.1

        # Normalize score
        if weights_sum > 0:
            score = score / weights_sum

        return score, indicators

    def _calculate_repetition(self, text: str) -> float:
        """Calculate word repetition score

        Args:
            text: Text to analyze

        Returns:
            float: Repetition score 0-1
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 10:
            return 0.0

        # Count word frequencies
        word_counts = Counter(words)

        # Calculate repetition (exclude very common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                       'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as',
                       'it', 'that', 'this', 'i', 'you', 'he', 'she', 'they', 'we'}

        meaningful_words = [w for w in words if w not in common_words and len(w) > 3]
        if len(meaningful_words) < 5:
            return 0.0

        meaningful_counts = Counter(meaningful_words)
        repeated_words = sum(1 for count in meaningful_counts.values() if count > 2)

        repetition_score = repeated_words / len(set(meaningful_words)) if meaningful_words else 0.0
        return min(repetition_score, 1.0)

    def _calculate_formality(self, text: str) -> float:
        """Calculate text formality score

        Args:
            text: Text to analyze

        Returns:
            float: Formality score 0-1
        """
        text_lower = text.lower()

        # Count formal indicators
        formal_count = sum(1 for phrase in self.FORMAL_PHRASES if phrase in text_lower)

        # Count informal indicators (contractions, casual language)
        informal_patterns = [
            r'\b(?:don\'t|can\'t|won\'t|shouldn\'t|wouldn\'t|couldn\'t)\b',
            r'\b(?:gonna|wanna|gotta|yeah|nope|yep)\b',
            r'\b(?:lol|lmao|omg|wtf|btw|imo|tbh)\b',
        ]
        informal_count = sum(1 for pattern in informal_patterns
                            if re.search(pattern, text_lower))

        # Calculate formality (high formal count + low informal count = high formality)
        if formal_count + informal_count == 0:
            return 0.5  # Neutral

        formality = formal_count / (formal_count + informal_count + 1)
        return min(formality, 1.0)

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score

        Args:
            text: Text to analyze

        Returns:
            float: Complexity score 0-1
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Normalize (15-30 words is typical, >30 is complex)
        length_complexity = min(avg_sentence_length / 30, 1.0)

        # Count complex words (>3 syllables, rough estimate)
        words = re.findall(r'\b\w+\b', text.lower())
        complex_words = sum(1 for w in words if len(w) > 10)  # Rough proxy for syllables

        word_complexity = complex_words / len(words) if words else 0.0

        # Combined complexity
        complexity = (length_complexity * 0.6 + word_complexity * 0.4)
        return min(complexity, 1.0)

    def _get_confidence_level(self, score: float) -> str:
        """Convert confidence score to level

        Args:
            score: Confidence score 0-1

        Returns:
            str: Confidence level
        """
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.3:
            return "low"
        else:
            return "very_low"

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> ConversationAIDetection:
        """Analyze AI content usage across conversation

        Args:
            messages: List of messages with 'text', 'sender' keys

        Returns:
            ConversationAIDetection: Conversation-level AI detection analysis
        """
        conv_detection = ConversationAIDetection()
        conv_detection.total_messages = len(messages)

        if not messages:
            return conv_detection

        # Analyze each message
        speaker_ai_counts = {}
        speaker_message_counts = {}
        consecutive_count = 0
        max_consecutive = 0

        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Detect AI content
            result = self.detect_ai_content(text)

            # Track speaker statistics
            if sender not in speaker_ai_counts:
                speaker_ai_counts[sender] = 0
                speaker_message_counts[sender] = 0

            speaker_message_counts[sender] += 1

            if result.is_ai_generated or result.flag_for_review:
                conv_detection.messages_flagged += 1
                conv_detection.ai_message_indices.append(i)
                speaker_ai_counts[sender] += 1

                # Track consecutive AI messages
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)

                # Track high confidence detections
                if result.ai_confidence >= self.AI_CONFIDENCE_THRESHOLD:
                    conv_detection.high_confidence_ai.append({
                        'message_index': i,
                        'sender': sender,
                        'confidence': result.ai_confidence,
                        'method': result.detection_method,
                        'indicators': result.ai_indicators
                    })
            else:
                consecutive_count = 0

        conv_detection.consecutive_ai_messages = max_consecutive

        # Calculate per-speaker AI scores
        for speaker in speaker_message_counts:
            if speaker_message_counts[speaker] > 0:
                ai_ratio = speaker_ai_counts[speaker] / speaker_message_counts[speaker]
                conv_detection.speaker_ai_scores[speaker] = ai_ratio

                # Flag speakers with >30% AI content
                if ai_ratio > 0.3 and speaker_ai_counts[speaker] >= 2:
                    conv_detection.speakers_flagged.append(speaker)

        # Calculate overall metrics
        if conv_detection.total_messages > 0:
            conv_detection.ai_percentage = (
                conv_detection.messages_flagged / conv_detection.total_messages * 100
            )
            conv_detection.overall_ai_likelihood = (
                conv_detection.messages_flagged / conv_detection.total_messages
            )

        return conv_detection

    def get_detection_summary(self, result: AIDetectionResult) -> str:
        """Generate human-readable detection summary

        Args:
            result: AI detection result

        Returns:
            str: Summary text
        """
        if result.detection_method == "none" or result.detection_method == "skipped":
            return "No AI detection performed"

        summary_parts = []

        # Detection verdict
        if result.is_ai_generated:
            summary_parts.append(f"⚠️  AI-generated content detected ({result.confidence_level} confidence)")
        else:
            summary_parts.append(f"Likely human-authored ({result.confidence_level} confidence)")

        # Detection method
        summary_parts.append(f"Method: {result.detection_method}")

        # Key indicators
        if result.ai_indicators:
            summary_parts.append(f"AI indicators: {', '.join(result.ai_indicators[:2])}")

        return "; ".join(summary_parts)
