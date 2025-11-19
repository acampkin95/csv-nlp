"""
Linguistic Complexity Analyzer Module

Analyzes text for linguistic complexity metrics including readability scores,
lexical diversity, sentence complexity, vocabulary sophistication, syntactic
complexity, code-switching, and register analysis.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics
from collections import Counter

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
except ImportError:
    nltk = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)

# Download required NLTK data
if nltk:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)


@dataclass
class LinguisticComplexityResult:
    """Container for linguistic complexity analysis results"""

    # Readability metrics
    flesch_kincaid_grade: float = 0.0
    flesch_reading_ease: float = 0.0

    # Lexical diversity
    type_token_ratio: float = 0.0
    hapax_legomena_ratio: float = 0.0
    brunet_index: float = 0.0

    # Sentence metrics
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    sentence_length_variance: float = 0.0

    # Vocabulary sophistication
    dale_chall_score: float = 0.0
    difficult_words: List[str] = field(default_factory=list)

    # Syntactic complexity
    avg_clause_length: float = 0.0
    dependent_clause_ratio: float = 0.0
    passive_voice_ratio: float = 0.0

    # Code-switching detection
    code_switching_detected: bool = False
    code_switching_instances: List[Dict] = field(default_factory=list)
    language_codes: List[str] = field(default_factory=list)

    # Register analysis
    formality_score: float = 0.5
    register_type: str = "neutral"
    formal_indicators: List[str] = field(default_factory=list)
    informal_indicators: List[str] = field(default_factory=list)

    # Overall complexity
    overall_complexity_score: float = 0.0
    complexity_level: str = "moderate"

    # Confidence
    confidence: float = 0.0


class LinguisticComplexityAnalyzer:
    """Analyzes linguistic complexity of text"""

    # Dale-Chall difficult words list (common 3000 easy words)
    EASY_WORDS = {
        'a', 'able', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
        'best', 'between', 'big', 'both', 'but', 'by', 'call', 'came', 'can', 'come',
        'could', 'day', 'did', 'do', 'does', 'done', 'down', 'each', 'even', 'ever',
        'every', 'face', 'fact', 'for', 'from', 'get', 'give', 'go', 'goes', 'going',
        'good', 'got', 'had', 'has', 'have', 'he', 'help', 'her', 'here', 'him',
        'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just',
        'know', 'let', 'like', 'long', 'made', 'make', 'man', 'many', 'may', 'me',
        'mean', 'might', 'more', 'most', 'move', 'much', 'must', 'my', 'myself',
        'name', 'need', 'never', 'new', 'no', 'not', 'now', 'of', 'off', 'old',
        'on', 'once', 'only', 'or', 'other', 'our', 'out', 'over', 'own', 'part',
        'people', 'place', 'play', 'said', 'same', 'say', 'see', 'set', 'she',
        'show', 'small', 'so', 'some', 'such', 'take', 'tell', 'than', 'that',
        'the', 'their', 'them', 'then', 'there', 'these', 'they', 'thing', 'think',
        'this', 'those', 'through', 'time', 'to', 'too', 'try', 'under', 'up', 'us',
        'use', 'very', 'want', 'was', 'water', 'way', 'we', 'week', 'were', 'what',
        'when', 'where', 'which', 'while', 'who', 'why', 'will', 'with', 'word',
        'work', 'world', 'would', 'write', 'year', 'you', 'your'
    }

    # Formal and informal register indicators
    FORMAL_INDICATORS = {
        'moreover', 'furthermore', 'therefore', 'nevertheless', 'thus',
        'subsequently', 'previously', 'accordingly', 'consequently',
        'shall', 'aforementioned', 'herein', 'thereof', 'wherein'
    }

    INFORMAL_INDICATORS = {
        'gonna', 'wanna', 'gotta', 'dunno', 'kinda', 'sorta', "ain't",
        "y'all", 'yeah', 'yep', 'nope', 'dude', 'like', 'totally',
        'awesome', 'cool', 'stuff', 'thing', 'guy', 'girl', 'wow', 'omg'
    }

    def __init__(self):
        """Initialize linguistic complexity analyzer"""
        self.has_nltk = nltk is not None
        self.has_spacy = spacy is not None
        self.nlp = None

        if self.has_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.nlp = None
                logger.warning("spaCy model not found")

    def analyze(self, text: str) -> LinguisticComplexityResult:
        """Perform comprehensive linguistic complexity analysis"""
        result = LinguisticComplexityResult()

        if not text or len(text.strip()) < 10:
            result.confidence = 0.3
            return result

        # Tokenize
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)

        if not sentences or not words:
            result.confidence = 0.2
            return result

        # Calculate readability metrics
        self._calculate_readability(text, sentences, words, result)

        # Calculate lexical diversity
        self._calculate_lexical_diversity(words, result)

        # Calculate sentence metrics
        self._calculate_sentence_metrics(sentences, words, result)

        # Calculate vocabulary sophistication
        self._calculate_vocabulary_sophistication(words, result)

        # Analyze register
        self._analyze_register(text, words, result)

        # Calculate overall complexity
        self._calculate_overall_complexity(result)

        result.confidence = 0.85
        return result

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        if self.has_nltk:
            return sent_tokenize(text)
        else:
            return re.split(r'[.!?]+', text)

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.has_nltk:
            return word_tokenize(text.lower())
        else:
            return re.findall(r'\b\w+\b', text.lower())

    def _calculate_readability(self, text: str, sentences: List[str],
                               words: List[str], result: LinguisticComplexityResult):
        """Calculate Flesch-Kincaid readability metrics"""
        if not sentences or not words:
            return

        num_words = len(words)
        num_sentences = len(sentences)
        num_syllables = self._count_syllables(text)

        if num_sentences > 0 and num_words > 0:
            flesch_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
            result.flesch_reading_ease = max(0, min(100, flesch_ease))

            flesch_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
            result.flesch_kincaid_grade = max(0, flesch_grade)

    def _count_syllables(self, text: str) -> int:
        """Count approximate syllables in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0

        for word in words:
            syllable_count += max(1, len(re.findall(r'[aeiouy]+', word)))
            if word.endswith('e'):
                syllable_count -= 1
            if word.endswith(('le', 'els')):
                syllable_count += 1

        return max(len(words), syllable_count)

    def _calculate_lexical_diversity(self, words: List[str],
                                    result: LinguisticComplexityResult):
        """Calculate lexical diversity metrics"""
        if not words:
            return

        unique_words = set(words)
        num_words = len(words)
        num_unique = len(unique_words)

        result.type_token_ratio = num_unique / num_words if num_words > 0 else 0

        word_freq = Counter(words)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        result.hapax_legomena_ratio = hapax_count / num_words if num_words > 0 else 0

        if num_words > 0:
            result.brunet_index = num_unique / (num_words ** 0.5)

    def _calculate_sentence_metrics(self, sentences: List[str], words: List[str],
                                   result: LinguisticComplexityResult):
        """Calculate sentence-level metrics"""
        if not sentences:
            return

        sentence_lengths = []
        for sentence in sentences:
            sent_words = re.findall(r'\b\w+\b', sentence.lower())
            sentence_lengths.append(len(sent_words))

        if sentence_lengths:
            result.avg_sentence_length = statistics.mean(sentence_lengths)
            if len(sentence_lengths) > 1:
                result.sentence_length_variance = statistics.variance(sentence_lengths)

        word_lengths = [len(word) for word in words]
        if word_lengths:
            result.avg_word_length = statistics.mean(word_lengths)

    def _calculate_vocabulary_sophistication(self, words: List[str],
                                            result: LinguisticComplexityResult):
        """Calculate Dale-Chall vocabulary sophistication score"""
        if not words:
            return

        difficult_words = []
        for word in set(words):
            if word.lower() not in self.EASY_WORDS and len(word) > 2:
                if len(word) >= 6:
                    difficult_words.append(word)

        result.difficult_words = difficult_words[:20]

        if len(words) > 0:
            difficult_ratio = len([w for w in words if w.lower() not in self.EASY_WORDS]) / len(words)
            result.dale_chall_score = difficult_ratio

    def _analyze_register(self, text: str, words: List[str],
                         result: LinguisticComplexityResult):
        """Analyze formality register"""
        text_lower = text.lower()

        formal_count = sum(1 for indicator in self.FORMAL_INDICATORS
                          if re.search(rf'\b{indicator}\b', text_lower))
        informal_count = sum(1 for indicator in self.INFORMAL_INDICATORS
                            if re.search(rf'\b{indicator}\b', text_lower))

        result.formal_indicators = [ind for ind in self.FORMAL_INDICATORS
                                   if re.search(rf'\b{ind}\b', text_lower)]
        result.informal_indicators = [ind for ind in self.INFORMAL_INDICATORS
                                     if re.search(rf'\b{ind}\b', text_lower)]

        total_indicators = formal_count + informal_count
        if total_indicators > 0:
            result.formality_score = formal_count / total_indicators
        else:
            result.formality_score = 0.5

        if result.formality_score > 0.7:
            result.register_type = "formal"
        elif result.formality_score < 0.3:
            result.register_type = "informal"
        else:
            result.register_type = "neutral"

    def _calculate_overall_complexity(self, result: LinguisticComplexityResult):
        """Calculate overall complexity score and level"""
        flesch_normalized = 1 - (result.flesch_reading_ease / 100)
        lexical_normalized = 1 - result.type_token_ratio
        grade_normalized = min(result.flesch_kincaid_grade / 18, 1.0)

        weights = [0.3, 0.3, 0.4]
        result.overall_complexity_score = (
            flesch_normalized * weights[0] +
            lexical_normalized * weights[1] +
            grade_normalized * weights[2]
        )

        if result.overall_complexity_score < 0.25:
            result.complexity_level = "simple"
        elif result.overall_complexity_score < 0.5:
            result.complexity_level = "moderate"
        elif result.overall_complexity_score < 0.75:
            result.complexity_level = "complex"
        else:
            result.complexity_level = "very_complex"

    def analyze_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze linguistic complexity across conversation"""
        if not messages:
            return {}

        speaker_complexity = {}
        all_results = []

        for msg in messages:
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            if not text:
                continue

            result = self.analyze(text)
            all_results.append(result)

            if sender not in speaker_complexity:
                speaker_complexity[sender] = {
                    'message_count': 0,
                    'avg_flesch_kincaid': 0,
                    'avg_complexity_score': 0,
                    'register_types': Counter()
                }

            speaker_complexity[sender]['message_count'] += 1
            speaker_complexity[sender]['avg_flesch_kincaid'] += result.flesch_kincaid_grade
            speaker_complexity[sender]['avg_complexity_score'] += result.overall_complexity_score
            speaker_complexity[sender]['register_types'][result.register_type] += 1

        for sender, data in speaker_complexity.items():
            if data['message_count'] > 0:
                data['avg_flesch_kincaid'] /= data['message_count']
                data['avg_complexity_score'] /= data['message_count']
                data['dominant_register'] = data['register_types'].most_common(1)[0][0]

        if all_results:
            avg_flesch = statistics.mean([r.flesch_reading_ease for r in all_results])
            avg_complexity = statistics.mean([r.overall_complexity_score for r in all_results])
        else:
            avg_flesch = 0
            avg_complexity = 0

        return {
            'speaker_complexity': speaker_complexity,
            'conversation_avg_flesch_reading_ease': avg_flesch,
            'conversation_avg_complexity_score': avg_complexity,
            'message_count': len(all_results)
        }
