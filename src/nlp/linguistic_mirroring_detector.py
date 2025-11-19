"""
Linguistic Mirroring Detector

Analyzes linguistic mirroring patterns in conversations, including vocabulary convergence,
syntax mimicry, phrase repetition, communication style adaptation, and formality matching.
Detects when speakers adopt each other's linguistic patterns.

Author: Message Processor Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class LinguisticMirroringResult:
    """Results from linguistic mirroring analysis."""

    # Overall metrics
    mirroring_score: float  # 0.0 (no mirroring) to 1.0 (high mirroring)
    mirroring_level: str  # none, subtle, moderate, strong

    # Vocabulary metrics
    vocabulary_convergence: float  # How much speakers adopt each other's vocab
    shared_vocabulary: List[str] = field(default_factory=list)
    unique_vocabulary_adoption: int = 0  # Unique words adopted from other speaker

    # Syntax and structure
    syntax_mimicry_score: float = 0.0
    sentence_structure_similarity: float = 0.0
    punctuation_mimicry: float = 0.0

    # Phrase patterns
    phrase_repetition_instances: int = 0
    shared_phrases: List[str] = field(default_factory=list)
    phrase_echo_ratio: float = 0.0

    # Communication style
    style_adaptation_score: float = 0.0
    formality_convergence: float = 0.0  # How much formality levels converge
    tone_similarity: float = 0.0

    # Emoji/Emoticon patterns
    emoji_mimicry: float = 0.0
    emoticon_mimicry: float = 0.0
    shared_emoticons: List[str] = field(default_factory=list)

    # Slang and informal patterns
    slang_adoption_score: float = 0.0
    slang_terms_adopted: List[str] = field(default_factory=list)

    # Evidence and details
    mirroring_evidence: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Per-speaker metrics
    speaker_mirroring_index: Dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'mirroring_score': self.mirroring_score,
            'mirroring_level': self.mirroring_level,
            'vocabulary_convergence': self.vocabulary_convergence,
            'shared_vocabulary_count': len(self.shared_vocabulary),
            'unique_vocabulary_adoption': self.unique_vocabulary_adoption,
            'syntax_mimicry_score': self.syntax_mimicry_score,
            'sentence_structure_similarity': self.sentence_structure_similarity,
            'punctuation_mimicry': self.punctuation_mimicry,
            'phrase_repetition_instances': self.phrase_repetition_instances,
            'shared_phrases_count': len(self.shared_phrases),
            'phrase_echo_ratio': self.phrase_echo_ratio,
            'style_adaptation_score': self.style_adaptation_score,
            'formality_convergence': self.formality_convergence,
            'tone_similarity': self.tone_similarity,
            'emoji_mimicry': self.emoji_mimicry,
            'emoticon_mimicry': self.emoticon_mimicry,
            'slang_adoption_score': self.slang_adoption_score,
            'speaker_mirroring_index': self.speaker_mirroring_index,
            'evidence_count': len(self.mirroring_evidence),
            'confidence': self.confidence
        }


class LinguisticMirroringDetector:
    """
    Detects linguistic mirroring patterns in conversations.

    Features:
    - Vocabulary convergence analysis
    - Syntax mimicry detection
    - Phrase repetition tracking
    - Communication style adaptation measurement
    - Emoji/emoticon copying detection
    - Slang adoption patterns
    - Formality matching analysis
    """

    def __init__(self):
        """Initialize the linguistic mirroring detector."""
        self._compile_patterns()
        logger.info("LinguisticMirroringDetector initialized")

    def _compile_patterns(self):
        """Compile regex patterns for detection."""

        # Emoticon patterns
        self.emoticon_patterns = {
            'positive': [r':[\)-]', r';\)', r'\(-?:', r'\^_\^'],
            'negative': [r':[(\-]', r':\/', r'DX', r':-\('],
            'neutral': [r':\|', r'-_-', r'\.\.\.'],
        }

        # Common emojis (simplified regex patterns)
        self.emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]|[🙂😊😢😡❤️👍👎]')

        # Slang patterns (common informal terms)
        self.slang_patterns = {
            'contemporary': [
                r'\b(gonna|gotta|wanna|dunno|kinda|sorta|ain\'t)\b',
                r'\b(like|literally|totally|awesome|cool|dude|bro)\b',
                r'\b(lol|lmao|omg|wtf|asap)\b',
            ],
            'casual': [
                r'\b(yeah|yep|nope|nah|uh huh|uh uh)\b',
                r'\b(guy|girl|thing|stuff|whatever)\b',
                r'\b(you know|i mean|basically)\b',
            ]
        }

        # Formality indicators
        self.formal_markers = [
            r'\b(therefore|furthermore|moreover|nevertheless|indeed)\b',
            r'\b(pursuant|hereby|thus|whence|henceforth)\b',
            r'\b(would you be so kind|kindly|respectfully)\b',
        ]

        self.informal_markers = [
            r'\b(hey|yo|btw|imo|fyi|tbh)\b',
            r'\b(can\'t|won\'t|don\'t|isn\'t|aren\'t)\b',
            r'\b(gotta|gonna|wanna|kinda)\b',
        ]

        # Sentence structure patterns
        self.sentence_start_patterns = {
            'question': r'^(who|what|when|where|why|how|is|are|do|does|can|could|would)',
            'imperative': r'^(do|don\'t|get|give|tell|show|stop|start)',
            'interrogative_tag': r'(,\s*(right|yeah|don\'t you|isn\'t it|isn\'t there))\s*\?$',
            'exclamatory': r'!$',
        }

        # Punctuation patterns
        self.punctuation_patterns = {
            'multiple_punctuation': r'[!?]{2,}',
            'ellipsis': r'\.{2,}',
            'dash_usage': r'-{2,}',
            'parenthetical': r'\([^)]*\)',
        }

    def analyze(self, messages: List[Dict[str, Any]]) -> LinguisticMirroringResult:
        """
        Analyze linguistic mirroring in conversation.

        Args:
            messages: List of message dictionaries with 'text', 'sender', 'timestamp'

        Returns:
            LinguisticMirroringResult with mirroring metrics
        """
        if not messages or len(messages) < 2:
            return self._empty_result()

        # Get unique speakers
        speakers = list(set(msg.get('sender', 'unknown') for msg in messages))
        if len(speakers) < 2:
            return self._empty_result()

        # Analyze vocabulary convergence
        vocab_metrics = self._analyze_vocabulary_convergence(messages, speakers)

        # Analyze syntax mimicry
        syntax_metrics = self._analyze_syntax_mimicry(messages, speakers)

        # Analyze phrase repetition
        phrase_metrics = self._analyze_phrase_repetition(messages, speakers)

        # Analyze style adaptation
        style_metrics = self._analyze_style_adaptation(messages, speakers)

        # Analyze emoji/emoticon patterns
        emoji_metrics = self._analyze_emoji_patterns(messages, speakers)

        # Analyze slang adoption
        slang_metrics = self._analyze_slang_adoption(messages, speakers)

        # Calculate overall mirroring score
        overall_score = self._calculate_overall_mirroring_score(
            vocab_metrics, syntax_metrics, phrase_metrics, style_metrics
        )

        # Calculate per-speaker mirroring index
        speaker_mirroring = self._calculate_speaker_mirroring_index(messages, speakers)

        # Create adaptation timeline
        timeline = self._create_adaptation_timeline(messages)

        # Determine mirroring level
        mirroring_level = self._classify_mirroring_level(overall_score)

        # Calculate confidence
        confidence = self._calculate_confidence(messages, vocab_metrics)

        return LinguisticMirroringResult(
            mirroring_score=overall_score,
            mirroring_level=mirroring_level,
            vocabulary_convergence=vocab_metrics['convergence'],
            shared_vocabulary=vocab_metrics['shared_words'],
            unique_vocabulary_adoption=vocab_metrics['unique_adoption'],
            syntax_mimicry_score=syntax_metrics['mimicry_score'],
            sentence_structure_similarity=syntax_metrics['structure_similarity'],
            punctuation_mimicry=syntax_metrics['punctuation_similarity'],
            phrase_repetition_instances=phrase_metrics['repetition_count'],
            shared_phrases=phrase_metrics['shared_phrases'],
            phrase_echo_ratio=phrase_metrics['echo_ratio'],
            style_adaptation_score=style_metrics['adaptation_score'],
            formality_convergence=style_metrics['formality_convergence'],
            tone_similarity=style_metrics['tone_similarity'],
            emoji_mimicry=emoji_metrics['emoji_similarity'],
            emoticon_mimicry=emoji_metrics['emoticon_similarity'],
            shared_emoticons=emoji_metrics['shared_emoticons'],
            slang_adoption_score=slang_metrics['adoption_score'],
            slang_terms_adopted=slang_metrics['adopted_terms'],
            mirroring_evidence=self._extract_evidence(
                messages, vocab_metrics, phrase_metrics, emoji_metrics
            ),
            adaptation_timeline=timeline,
            speaker_mirroring_index=speaker_mirroring,
            confidence=confidence
        )

    def _analyze_vocabulary_convergence(self, messages: List[Dict[str, Any]],
                                       speakers: List[str]) -> Dict[str, Any]:
        """Analyze vocabulary adoption between speakers."""
        if len(speakers) < 2:
            return {
                'convergence': 0.0,
                'shared_words': [],
                'unique_adoption': 0
            }

        # Get vocabularies per speaker
        speaker_vocabs = {}
        speaker_messages = {}

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '').lower()

            if sender not in speaker_messages:
                speaker_messages[sender] = []
            speaker_messages[sender].append(text)

            words = set(re.findall(r'\b\w+\b', text))
            if sender not in speaker_vocabs:
                speaker_vocabs[sender] = set()
            speaker_vocabs[sender].update(words)

        # Calculate vocabulary introduced at different points
        adoption_count = 0
        shared_words = []

        # For each speaker, track which words from other speakers they use
        for i, speaker in enumerate(speakers):
            own_words = speaker_vocabs.get(speaker, set())
            other_speakers = [s for j, s in enumerate(speakers) if j != i]

            for other in other_speakers:
                other_words = speaker_vocabs.get(other, set())
                # Words that appear in both vocabularies
                shared = own_words & other_words
                shared_words.extend(list(shared))

        # Remove duplicates
        shared_words = list(set(shared_words))

        # Calculate convergence score
        if all(speaker_vocabs.get(s) for s in speakers):
            vocab_sizes = [len(speaker_vocabs.get(s, set())) for s in speakers]
            avg_vocab = sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 1
            convergence = len(shared_words) / avg_vocab if avg_vocab > 0 else 0.0
            convergence = min(1.0, convergence)
        else:
            convergence = 0.0

        return {
            'convergence': convergence,
            'shared_words': shared_words[:20],  # Top 20 shared words
            'unique_adoption': len(shared_words),
            'speaker_vocabs': speaker_vocabs
        }

    def _analyze_syntax_mimicry(self, messages: List[Dict[str, Any]],
                               speakers: List[str]) -> Dict[str, Any]:
        """Analyze syntax and sentence structure mimicry."""
        if len(speakers) < 2:
            return {
                'mimicry_score': 0.0,
                'structure_similarity': 0.0,
                'punctuation_similarity': 0.0
            }

        speaker_structures = {}
        speaker_punctuation = {}

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')

            if sender not in speaker_structures:
                speaker_structures[sender] = []
                speaker_punctuation[sender] = []

            # Classify sentence structure
            structure = self._classify_sentence_structure(text)
            speaker_structures[sender].append(structure)

            # Analyze punctuation
            punct_pattern = self._analyze_punctuation(text)
            speaker_punctuation[sender].append(punct_pattern)

        # Calculate structure similarity
        structure_similarity = 0.0
        if len(speakers) >= 2:
            speaker_a = speakers[0]
            speaker_b = speakers[1] if len(speakers) > 1 else speakers[0]

            structs_a = speaker_structures.get(speaker_a, [])
            structs_b = speaker_structures.get(speaker_b, [])

            if structs_a and structs_b:
                # Count matching structures
                matches = sum(1 for s_a, s_b in zip(structs_a, structs_b) if s_a == s_b)
                structure_similarity = matches / max(len(structs_a), len(structs_b))

        # Calculate punctuation similarity
        punctuation_similarity = 0.0
        if len(speakers) >= 2:
            punct_a = speaker_punctuation.get(speakers[0], [])
            punct_b = speaker_punctuation.get(speakers[1], [])

            if punct_a and punct_b:
                matches = sum(1 for p_a, p_b in zip(punct_a, punct_b) if p_a == p_b)
                punctuation_similarity = matches / max(len(punct_a), len(punct_b))

        # Overall mimicry score
        mimicry_score = (structure_similarity + punctuation_similarity) / 2

        return {
            'mimicry_score': mimicry_score,
            'structure_similarity': structure_similarity,
            'punctuation_similarity': punctuation_similarity
        }

    def _analyze_phrase_repetition(self, messages: List[Dict[str, Any]],
                                  speakers: List[str]) -> Dict[str, Any]:
        """Analyze phrase repetition and echoing."""
        phrase_occurrences = Counter()
        speaker_phrases = {}
        shared_phrases = []
        echo_count = 0

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '').lower()

            if sender not in speaker_phrases:
                speaker_phrases[sender] = set()

            # Extract phrases (2-4 word sequences)
            words = re.findall(r'\b\w+\b', text)
            for i in range(len(words) - 1):
                for length in range(2, min(5, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+length])
                    phrase_occurrences[phrase] += 1
                    speaker_phrases[sender].add(phrase)

        # Find shared phrases between speakers
        if len(speakers) >= 2:
            phrases_by_speaker = [speaker_phrases.get(s, set()) for s in speakers]
            shared = set(phrases_by_speaker[0])
            for phrases in phrases_by_speaker[1:]:
                shared &= phrases
            shared_phrases = list(shared)[:15]

        # Calculate echo ratio (repeated phrases within same or next message)
        prev_text = None
        for msg in messages:
            text = msg.get('text', '').lower()
            if prev_text:
                words_current = set(re.findall(r'\b\w+\b', text))
                words_prev = set(re.findall(r'\b\w+\b', prev_text))
                if words_current & words_prev:
                    echo_count += 1
            prev_text = text

        echo_ratio = echo_count / len(messages) if messages else 0.0

        return {
            'repetition_count': len(phrase_occurrences),
            'shared_phrases': shared_phrases,
            'echo_ratio': min(1.0, echo_ratio)
        }

    def _analyze_style_adaptation(self, messages: List[Dict[str, Any]],
                                 speakers: List[str]) -> Dict[str, Any]:
        """Analyze communication style and formality adaptation."""
        speaker_styles = {}
        speaker_formality = {}

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')

            if sender not in speaker_styles:
                speaker_styles[sender] = []
                speaker_formality[sender] = []

            # Analyze formality
            formality_score = self._calculate_formality_score(text)
            speaker_formality[sender].append(formality_score)

            # Analyze style markers
            style = self._analyze_communication_style(text)
            speaker_styles[sender].append(style)

        # Calculate formality convergence
        formality_convergence = 0.0
        if len(speakers) >= 2:
            avg_formality = {}
            for speaker in speakers:
                scores = speaker_formality.get(speaker, [])
                avg_formality[speaker] = sum(scores) / len(scores) if scores else 0.5

            # Convergence = 1 - (difference between average formality levels)
            formality_values = list(avg_formality.values())
            if formality_values:
                formality_range = max(formality_values) - min(formality_values)
                formality_convergence = 1.0 - min(formality_range, 1.0)

        # Calculate style similarity
        style_adaptation = 0.0
        if len(speakers) >= 2:
            styles_a = speaker_styles.get(speakers[0], [])
            styles_b = speaker_styles.get(speakers[1], [])

            if styles_a and styles_b:
                matches = sum(1 for s_a, s_b in zip(styles_a, styles_b) if s_a == s_b)
                style_adaptation = matches / max(len(styles_a), len(styles_b))

        # Tone similarity (based on punctuation and word choice)
        tone_similarity = 0.0
        if len(speakers) >= 2:
            tone_a = self._analyze_tone(messages, speakers[0])
            tone_b = self._analyze_tone(messages, speakers[1])
            tone_similarity = 1.0 - abs(tone_a - tone_b)

        return {
            'adaptation_score': style_adaptation,
            'formality_convergence': max(0.0, formality_convergence),
            'tone_similarity': max(0.0, tone_similarity)
        }

    def _analyze_emoji_patterns(self, messages: List[Dict[str, Any]],
                               speakers: List[str]) -> Dict[str, Any]:
        """Analyze emoji and emoticon usage patterns."""
        speaker_emojis = {}
        speaker_emoticons = {}
        all_emojis = []
        all_emoticons = []

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')

            if sender not in speaker_emojis:
                speaker_emojis[sender] = []
                speaker_emoticons[sender] = []

            # Find emojis
            emojis = self.emoji_pattern.findall(text)
            speaker_emojis[sender].extend(emojis)
            all_emojis.extend(emojis)

            # Find emoticons
            emoticons = self._find_emoticons(text)
            speaker_emoticons[sender].extend(emoticons)
            all_emoticons.extend(emoticons)

        # Calculate emoji similarity
        emoji_similarity = 0.0
        if len(speakers) >= 2:
            emojis_a = set(speaker_emojis.get(speakers[0], []))
            emojis_b = set(speaker_emojis.get(speakers[1], []))

            if emojis_a or emojis_b:
                union_size = len(emojis_a | emojis_b)
                if union_size > 0:
                    emoji_similarity = len(emojis_a & emojis_b) / union_size

        # Calculate emoticon similarity
        emoticon_similarity = 0.0
        shared_emoticons = []
        if len(speakers) >= 2:
            emos_a = set(speaker_emoticons.get(speakers[0], []))
            emos_b = set(speaker_emoticons.get(speakers[1], []))

            if emos_a or emos_b:
                shared_emoticons = list(emos_a & emos_b)
                union_size = len(emos_a | emos_b)
                if union_size > 0:
                    emoticon_similarity = len(emos_a & emos_b) / union_size

        return {
            'emoji_similarity': emoji_similarity,
            'emoticon_similarity': emoticon_similarity,
            'shared_emoticons': shared_emoticons
        }

    def _analyze_slang_adoption(self, messages: List[Dict[str, Any]],
                               speakers: List[str]) -> Dict[str, Any]:
        """Analyze slang and informal language adoption."""
        speaker_slang = {}
        adopted_slang = []

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '').lower()

            if sender not in speaker_slang:
                speaker_slang[sender] = set()

            # Find slang terms
            for category, patterns in self.slang_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    speaker_slang[sender].update(matches)

        # Find slang adoption (same slang in multiple speakers)
        if len(speakers) >= 2:
            slang_sets = [speaker_slang.get(s, set()) for s in speakers]
            adopted_slang = list(set.intersection(*[set(s) for s in slang_sets if s]))

        # Calculate adoption score
        all_slang = set()
        for slang_set in speaker_slang.values():
            all_slang.update(slang_set)

        adoption_score = 0.0
        if all_slang:
            # Score based on how many speakers use similar slang
            adoption_score = len(adopted_slang) / len(all_slang) if all_slang else 0.0

        return {
            'adoption_score': min(1.0, adoption_score),
            'adopted_terms': adopted_slang[:10]
        }

    def _classify_sentence_structure(self, text: str) -> str:
        """Classify sentence structure type."""
        if re.search(r'\?$', text.strip()):
            return 'question'
        elif re.search(r'^(do|don\'t|get|give|tell)\b', text.lower()):
            return 'imperative'
        elif re.search(r'!$', text.strip()):
            return 'exclamatory'
        else:
            return 'declarative'

    def _analyze_punctuation(self, text: str) -> str:
        """Analyze punctuation pattern."""
        if re.search(r'[!?]{2,}', text):
            return 'emphatic'
        elif re.search(r'\.{2,}', text):
            return 'trailing'
        elif re.search(r'[!?]', text):
            return 'standard'
        else:
            return 'none'

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality level of text."""
        formal_count = sum(1 for marker in self.formal_markers
                          if re.search(marker, text, re.IGNORECASE))
        informal_count = sum(1 for marker in self.informal_markers
                            if re.search(marker, text, re.IGNORECASE))

        total = formal_count + informal_count
        if total == 0:
            return 0.5

        return formal_count / total

    def _analyze_communication_style(self, text: str) -> str:
        """Analyze communication style."""
        if re.search(r'[!?]{2,}|wow|amazing|great|love', text, re.IGNORECASE):
            return 'enthusiastic'
        elif re.search(r'sorry|excuse|please|thank', text, re.IGNORECASE):
            return 'polite'
        elif re.search(r'but|however|yet|though', text, re.IGNORECASE):
            return 'cautious'
        else:
            return 'neutral'

    def _analyze_tone(self, messages: List[Dict[str, Any]], speaker: str) -> float:
        """Analyze overall tone for a speaker (0=negative, 1=positive)."""
        tone_scores = []
        positive_words = r'\b(great|good|love|happy|awesome|wonderful|excellent)\b'
        negative_words = r'\b(hate|bad|terrible|awful|horrible|sad|angry)\b'

        for msg in messages:
            if msg.get('sender') == speaker:
                text = msg.get('text', '')
                pos_count = len(re.findall(positive_words, text, re.IGNORECASE))
                neg_count = len(re.findall(negative_words, text, re.IGNORECASE))

                if pos_count + neg_count > 0:
                    tone = pos_count / (pos_count + neg_count)
                else:
                    tone = 0.5
                tone_scores.append(tone)

        return sum(tone_scores) / len(tone_scores) if tone_scores else 0.5

    def _find_emoticons(self, text: str) -> List[str]:
        """Find emoticons in text."""
        emoticons = []
        for category, patterns in self.emoticon_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                emoticons.extend(matches)
        return emoticons

    def _calculate_overall_mirroring_score(self, vocab_metrics: Dict,
                                          syntax_metrics: Dict,
                                          phrase_metrics: Dict,
                                          style_metrics: Dict) -> float:
        """Calculate overall mirroring score."""
        weights = {
            'vocabulary': 0.25,
            'syntax': 0.25,
            'phrase': 0.20,
            'style': 0.30
        }

        score = (
            vocab_metrics['convergence'] * weights['vocabulary'] +
            syntax_metrics['mimicry_score'] * weights['syntax'] +
            phrase_metrics['echo_ratio'] * weights['phrase'] +
            style_metrics['adaptation_score'] * weights['style']
        )

        return round(min(1.0, score), 3)

    def _classify_mirroring_level(self, score: float) -> str:
        """Classify mirroring level."""
        if score < 0.2:
            return 'none'
        elif score < 0.4:
            return 'subtle'
        elif score < 0.7:
            return 'moderate'
        else:
            return 'strong'

    def _calculate_speaker_mirroring_index(self, messages: List[Dict[str, Any]],
                                          speakers: List[str]) -> Dict[str, float]:
        """Calculate per-speaker mirroring index."""
        speaker_index = {}

        for speaker in speakers:
            # Count patterns this speaker mirrors from others
            pattern_score = 0
            message_count = 0

            for msg in messages:
                if msg.get('sender') == speaker:
                    message_count += 1
                    text = msg.get('text', '')

                    # Count various mirroring indicators
                    if re.search(r'[!?]{2,}', text):
                        pattern_score += 0.2
                    if re.search(r'\.\.\.$', text):
                        pattern_score += 0.1
                    if len(re.findall(r'\b(yeah|yep|nope)\b', text)) > 0:
                        pattern_score += 0.15

            if message_count > 0:
                index = min(1.0, pattern_score / (message_count * 0.5))
            else:
                index = 0.0

            speaker_index[speaker] = round(index, 3)

        return speaker_index

    def _create_adaptation_timeline(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create timeline of adaptation patterns."""
        timeline = []

        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '')

            # Detect adaptation indicators
            adaptations = []

            if i > 0:
                prev_text = messages[i-1].get('text', '')

                # Vocabulary adoption
                prev_words = set(re.findall(r'\b\w+\b', prev_text.lower()))
                curr_words = set(re.findall(r'\b\w+\b', text.lower()))
                shared = prev_words & curr_words

                if len(shared) > 3:
                    adaptations.append('vocabulary_adoption')

                # Punctuation mimicry
                if re.search(r'[!?]{2,}', prev_text) and re.search(r'[!?]{2,}', text):
                    adaptations.append('punctuation_mimicry')

                # Emoji/emoticon usage
                if self.emoji_pattern.search(prev_text) and self.emoji_pattern.search(text):
                    adaptations.append('emoji_adoption')

            if adaptations:
                timeline.append({
                    'message_index': i,
                    'sender': sender,
                    'adaptations': adaptations
                })

        return timeline

    def _extract_evidence(self, messages: List[Dict[str, Any]],
                         vocab_metrics: Dict,
                         phrase_metrics: Dict,
                         emoji_metrics: Dict) -> List[Dict[str, Any]]:
        """Extract evidence of mirroring."""
        evidence = []

        # Vocabulary evidence
        if vocab_metrics['shared_words']:
            evidence.append({
                'type': 'vocabulary_convergence',
                'examples': vocab_metrics['shared_words'][:5]
            })

        # Phrase evidence
        if phrase_metrics['shared_phrases']:
            evidence.append({
                'type': 'phrase_repetition',
                'examples': phrase_metrics['shared_phrases'][:5]
            })

        # Emoji evidence
        if emoji_metrics['shared_emoticons']:
            evidence.append({
                'type': 'emoticon_usage',
                'examples': emoji_metrics['shared_emoticons'][:5]
            })

        return evidence

    def _calculate_confidence(self, messages: List[Dict], vocab_metrics: Dict) -> float:
        """Calculate confidence in analysis."""
        confidence = 0.5  # Base confidence

        # More messages = higher confidence
        if len(messages) > 50:
            confidence += 0.3
        elif len(messages) > 20:
            confidence += 0.2
        elif len(messages) > 10:
            confidence += 0.1

        # Clear mirroring patterns = higher confidence
        if vocab_metrics['unique_adoption'] > 5:
            confidence += 0.2
        elif vocab_metrics['unique_adoption'] > 2:
            confidence += 0.1

        return min(1.0, confidence)

    def _empty_result(self) -> LinguisticMirroringResult:
        """Return empty result for edge cases."""
        return LinguisticMirroringResult(
            mirroring_score=0.0,
            mirroring_level='none',
            vocabulary_convergence=0.0,
            confidence=0.0
        )


# Example usage
if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        {'sender': 'A', 'text': 'Hey! I love this coffee shop. It\'s amazing!', 'timestamp': None},
        {'sender': 'B', 'text': 'Yeah, I love it too! The vibe is great!', 'timestamp': None},
        {'sender': 'A', 'text': 'Totally! And the staff is so awesome.', 'timestamp': None},
        {'sender': 'B', 'text': 'Awesome is right! We should come here more often.', 'timestamp': None},
        {'sender': 'A', 'text': 'Definitely! Same time next week?', 'timestamp': None},
        {'sender': 'B', 'text': 'Sounds good! I love hanging out here.', 'timestamp': None},
    ]

    detector = LinguisticMirroringDetector()
    result = detector.analyze(test_messages)

    print("Linguistic Mirroring Analysis:")
    print(f"Mirroring Score: {result.mirroring_score}")
    print(f"Mirroring Level: {result.mirroring_level}")
    print(f"Vocabulary Convergence: {result.vocabulary_convergence}")
    print(f"Shared Vocabulary: {result.shared_vocabulary[:10]}")
    print(f"Syntax Mimicry: {result.syntax_mimicry_score}")
    print(f"Style Adaptation: {result.style_adaptation_score}")
    print(f"Shared Phrases: {result.shared_phrases}")
    print(f"Confidence: {result.confidence}")
