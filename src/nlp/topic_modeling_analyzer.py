"""
Topic Modeling Analyzer

Analyzes topics in conversations using keyword frequency and TF-IDF approach.
Detects topic shifts, persistence, avoidance patterns, and conversation steering.
No external dependencies - uses only Python standard library.

Author: Message Processor Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
import re
from collections import Counter
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """Represents a detected topic."""
    name: str
    keywords: List[str]
    frequency: int
    first_mention_index: int
    last_mention_index: int
    speaker_mentions: Dict[str, int] = field(default_factory=dict)
    persistence_score: float = 0.0
    relevance_score: float = 0.0


@dataclass
class TopicShift:
    """Represents a topic shift/change."""
    message_index: int
    from_topic: str
    to_topic: str
    shift_type: str  # natural, forced, abandoned
    initiating_speaker: str


@dataclass
class TopicModelingResult:
    """Results from topic modeling analysis."""

    # Topics
    topics: List[Topic] = field(default_factory=list)
    dominant_topic: Optional[str] = None
    topic_count: int = 0
    topic_diversity: float = 0.0  # Measure of topic variety

    # Topic shifts
    topic_shifts: List[TopicShift] = field(default_factory=list)
    shift_count: int = 0
    shift_frequency: float = 0.0  # Shifts per message

    # Topic persistence
    topic_persistence_scores: Dict[str, float] = field(default_factory=dict)
    average_topic_duration: float = 0.0  # Messages per topic
    longest_topic_duration: int = 0

    # Topic avoidance
    avoided_topics: List[str] = field(default_factory=list)
    quick_abandonment_count: int = 0  # Topics dropped after few mentions

    # Conversation steering
    steering_detected: bool = False
    steering_intensity: float = 0.0  # 0 (none) to 1 (extreme)
    steering_initiators: Dict[str, int] = field(default_factory=dict)

    # Topic appropriateness
    topic_appropriateness_score: float = 0.0
    off_topic_indicators: List[str] = field(default_factory=list)
    topic_consistency: float = 0.0  # How consistently on-topic conversation is

    # Taboo topics
    taboo_topics_detected: List[str] = field(default_factory=list)
    sensitive_content: bool = False

    # Speaker analysis
    speaker_topic_preferences: Dict[str, Dict[str, int]] = field(default_factory=dict)
    topic_control_by_speaker: Dict[str, float] = field(default_factory=dict)

    # Timeline
    topic_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'topic_count': self.topic_count,
            'topic_diversity': self.topic_diversity,
            'dominant_topic': self.dominant_topic,
            'shift_count': self.shift_count,
            'shift_frequency': self.shift_frequency,
            'average_topic_duration': self.average_topic_duration,
            'longest_topic_duration': self.longest_topic_duration,
            'quick_abandonment_count': self.quick_abandonment_count,
            'steering_detected': self.steering_detected,
            'steering_intensity': self.steering_intensity,
            'topic_appropriateness_score': self.topic_appropriateness_score,
            'topic_consistency': self.topic_consistency,
            'taboo_topics_count': len(self.taboo_topics_detected),
            'sensitive_content': self.sensitive_content,
            'confidence': self.confidence
        }


class TopicModelingAnalyzer:
    """
    Analyzes topics in conversations.

    Features:
    - Topic extraction using keyword frequency
    - Topic shift detection
    - Topic persistence tracking
    - Subject avoidance pattern detection
    - Conversation steering detection
    - Topic appropriateness scoring
    - Taboo topic identification
    """

    def __init__(self):
        """Initialize the topic modeling analyzer."""
        self._compile_patterns()
        self._load_topic_keywords()
        logger.info("TopicModelingAnalyzer initialized")

    def _compile_patterns(self):
        """Compile regex patterns for detection."""

        # Stop words (common words to exclude)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why',
            'how', 'if', 'as', 'by', 'with', 'from', 'about', 'into', 'through',
            'during', 'so', 'than', 'very', 'just', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'me', 'him', 'her', 'us', 'them'
        }

        # Topic shift indicators
        self.topic_shift_patterns = {
            'natural': [
                r'\b(anyway|by the way|speaking of|on that note)\b',
                r'\b(so about|let\'s talk about|here\'s the thing)\b',
                r'\b(that reminds me of|which brings me to)\b',
            ],
            'forced': [
                r'\b(but we need to discuss|look, the real issue|actually,)\b',
                r'\b(forget that, what i really|more importantly|let\'s be honest)\b',
                r'\b(enough about that, let\'s|stop right there|you know what matters)\b',
            ],
            'abandoned': [
                r'\b(never mind|forget i said that|doesn\'t matter)\b',
                r'\b(let\'s just drop it|not important|whatever|forget it)\b',
            ]
        }

        # Off-topic patterns
        self.off_topic_patterns = [
            r'\b(off the subject|off topic|random but|just saying)\b',
            r'\b(this is random|completely unrelated)\b',
        ]

        # Taboo/sensitive topic keywords
        self.taboo_keywords = {
            'financial': ['money', 'debt', 'bankruptcy', 'poor', 'rich', 'salary', 'wage'],
            'health': ['disease', 'illness', 'condition', 'medication', 'hospital', 'cancer'],
            'family': ['divorce', 'abuse', 'affair', 'custody', 'inheritance'],
            'political': ['election', 'vote', 'candidate', 'government', 'policy', 'law'],
            'religious': ['god', 'faith', 'belief', 'religion', 'church', 'prayer'],
            'intimate': ['sex', 'relationship', 'romantic', 'attraction', 'partner'],
            'controversial': ['racism', 'discrimination', 'violence', 'drugs', 'crime']
        }

        # Steering patterns (controlling topic changes)
        self.steering_patterns = [
            r'\b(we\'re not talking about that|i don\'t want to discuss)\b',
            r'\b(stop bringing up|shut up about|no more talk of)\b',
            r'\b(let\'s move on from|can we please talk about something else)\b',
        ]

    def _load_topic_keywords(self):
        """Load common topic keywords."""
        self.topic_keywords = {
            'work': ['work', 'job', 'boss', 'office', 'meeting', 'project', 'deadline', 'colleague'],
            'relationships': ['relationship', 'boyfriend', 'girlfriend', 'partner', 'love', 'dating', 'couple'],
            'family': ['family', 'parent', 'mother', 'father', 'sibling', 'brother', 'sister', 'child'],
            'health': ['health', 'doctor', 'sick', 'illness', 'exercise', 'diet', 'hospital'],
            'entertainment': ['movie', 'music', 'show', 'watch', 'listen', 'tv', 'film', 'series'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'play', 'sport', 'match'],
            'travel': ['trip', 'travel', 'vacation', 'visit', 'hotel', 'flight', 'destination', 'beach'],
            'food': ['eat', 'food', 'restaurant', 'cook', 'meal', 'dinner', 'lunch', 'breakfast'],
            'education': ['school', 'class', 'study', 'learning', 'teacher', 'student', 'exam', 'grade'],
            'politics': ['politics', 'government', 'election', 'vote', 'candidate', 'policy', 'law'],
            'weather': ['weather', 'rain', 'snow', 'cold', 'hot', 'sunny', 'temperature'],
        }

    def analyze(self, messages: List[Dict[str, Any]]) -> TopicModelingResult:
        """
        Analyze topics in conversation.

        Args:
            messages: List of message dictionaries with 'text', 'sender', 'timestamp'

        Returns:
            TopicModelingResult with topic analysis
        """
        if not messages:
            return self._empty_result()

        # Extract keywords and identify topics
        keywords_by_message = self._extract_keywords(messages)
        topics = self._identify_topics(messages, keywords_by_message)

        if not topics:
            return self._empty_result()

        # Assign topics to messages
        message_topics = self._assign_topics_to_messages(messages, topics, keywords_by_message)

        # Analyze topic shifts
        topic_shifts = self._analyze_topic_shifts(messages, message_topics)

        # Analyze topic persistence
        persistence_scores = self._analyze_topic_persistence(message_topics, topics)
        for topic in topics:
            topic.persistence_score = persistence_scores.get(topic.name, 0.0)

        # Detect topic avoidance
        avoided_topics, quick_abandonment = self._detect_topic_avoidance(message_topics)

        # Detect conversation steering
        steering_info = self._detect_conversation_steering(messages, topic_shifts)

        # Calculate topic appropriateness
        appropriateness = self._calculate_topic_appropriateness(messages, keywords_by_message)

        # Identify taboo topics
        taboo_topics = self._identify_taboo_topics(messages, keywords_by_message)

        # Analyze speaker topic preferences
        speaker_preferences = self._analyze_speaker_preferences(messages, message_topics)

        # Calculate topic control
        topic_control = self._calculate_topic_control(messages, topics, speaker_preferences)

        # Create timeline
        timeline = self._create_topic_timeline(messages, message_topics)

        # Calculate metrics
        dominant_topic = max(topics, key=lambda t: t.frequency).name if topics else None
        topic_diversity = self._calculate_diversity(topics)
        shift_frequency = len(topic_shifts) / len(messages) if messages else 0.0
        avg_duration = self._calculate_avg_duration(message_topics, topics)

        # Calculate confidence
        confidence = self._calculate_confidence(messages, topics)

        return TopicModelingResult(
            topics=topics,
            dominant_topic=dominant_topic,
            topic_count=len(topics),
            topic_diversity=topic_diversity,
            topic_shifts=topic_shifts,
            shift_count=len(topic_shifts),
            shift_frequency=shift_frequency,
            topic_persistence_scores=persistence_scores,
            average_topic_duration=avg_duration,
            longest_topic_duration=self._get_longest_duration(message_topics),
            avoided_topics=avoided_topics,
            quick_abandonment_count=quick_abandonment,
            steering_detected=steering_info['detected'],
            steering_intensity=steering_info['intensity'],
            steering_initiators=steering_info['initiators'],
            topic_appropriateness_score=appropriateness['score'],
            off_topic_indicators=appropriateness['off_topic'],
            topic_consistency=appropriateness['consistency'],
            taboo_topics_detected=taboo_topics,
            sensitive_content=len(taboo_topics) > 0,
            speaker_topic_preferences=speaker_preferences,
            topic_control_by_speaker=topic_control,
            topic_timeline=timeline,
            confidence=confidence
        )

    def _extract_keywords(self, messages: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Extract keywords from messages."""
        keywords_by_message = {}

        for i, msg in enumerate(messages):
            text = msg.get('text', '').lower()
            # Extract words, remove punctuation
            words = re.findall(r'\b\w+\b', text)
            # Remove stop words
            keywords = [w for w in words if w not in self.stop_words and len(w) > 3]
            keywords_by_message[i] = keywords

        return keywords_by_message

    def _identify_topics(self, messages: List[Dict[str, Any]],
                        keywords_by_message: Dict[int, List[str]]) -> List[Topic]:
        """Identify topics using keyword matching."""
        topics = {}

        # Count keyword occurrences
        all_keywords = []
        for keywords in keywords_by_message.values():
            all_keywords.extend(keywords)

        keyword_freq = Counter(all_keywords)

        # Match to predefined topics
        topic_scores = {}
        for topic_name, keywords in self.topic_keywords.items():
            score = sum(keyword_freq.get(kw, 0) for kw in keywords)
            if score > 0:
                topic_scores[topic_name] = score

        # Create Topic objects
        topic_objects = []
        for topic_name, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True):
            # Find first and last mention
            first_mention = None
            last_mention = None

            for msg_idx, keywords in keywords_by_message.items():
                if any(kw in keywords for kw in self.topic_keywords[topic_name]):
                    if first_mention is None:
                        first_mention = msg_idx
                    last_mention = msg_idx

            if first_mention is not None:
                topic = Topic(
                    name=topic_name,
                    keywords=self.topic_keywords[topic_name],
                    frequency=score,
                    first_mention_index=first_mention,
                    last_mention_index=last_mention
                )
                topic_objects.append(topic)

        return topic_objects[:10]  # Limit to 10 topics

    def _assign_topics_to_messages(self, messages: List[Dict[str, Any]],
                                   topics: List[Topic],
                                   keywords_by_message: Dict[int, List[str]]) -> Dict[int, str]:
        """Assign dominant topic to each message."""
        message_topics = {}

        for msg_idx, keywords in keywords_by_message.items():
            best_topic = None
            best_score = 0

            for topic in topics:
                score = sum(1 for kw in keywords if kw in topic.keywords)
                if score > best_score:
                    best_score = score
                    best_topic = topic.name

            if best_topic:
                message_topics[msg_idx] = best_topic

                # Update speaker mentions
                sender = messages[msg_idx].get('sender', 'unknown')
                for topic in topics:
                    if topic.name == best_topic:
                        if sender not in topic.speaker_mentions:
                            topic.speaker_mentions[sender] = 0
                        topic.speaker_mentions[sender] += 1

        return message_topics

    def _analyze_topic_shifts(self, messages: List[Dict[str, Any]],
                             message_topics: Dict[int, str]) -> List[TopicShift]:
        """Analyze topic shifts and changes."""
        shifts = []
        prev_topic = None
        prev_index = 0

        for msg_idx in sorted(message_topics.keys()):
            current_topic = message_topics.get(msg_idx)

            if current_topic and prev_topic and current_topic != prev_topic:
                sender = messages[msg_idx].get('sender', 'unknown')

                # Determine shift type
                text = messages[msg_idx].get('text', '').lower()
                shift_type = 'natural'

                for stype, patterns in self.topic_shift_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            shift_type = stype
                            break

                shift = TopicShift(
                    message_index=msg_idx,
                    from_topic=prev_topic,
                    to_topic=current_topic,
                    shift_type=shift_type,
                    initiating_speaker=sender
                )
                shifts.append(shift)

            if current_topic:
                prev_topic = current_topic
                prev_index = msg_idx

        return shifts

    def _analyze_topic_persistence(self, message_topics: Dict[int, str],
                                  topics: List[Topic]) -> Dict[str, float]:
        """Analyze how long topics persist."""
        persistence = {}

        for topic in topics:
            # Find continuous blocks of this topic
            blocks = []
            current_block = []

            for msg_idx in sorted(message_topics.keys()):
                if message_topics.get(msg_idx) == topic.name:
                    current_block.append(msg_idx)
                else:
                    if current_block:
                        blocks.append(len(current_block))
                        current_block = []

            if current_block:
                blocks.append(len(current_block))

            # Calculate persistence score
            if blocks:
                avg_block_size = sum(blocks) / len(blocks)
                max_block_size = max(blocks)
                # Score based on longest and average block size
                persistence[topic.name] = min(1.0, max_block_size / 20.0)
            else:
                persistence[topic.name] = 0.0

        return persistence

    def _detect_topic_avoidance(self, message_topics: Dict[int, str]) -> Tuple[List[str], int]:
        """Detect avoided and quickly abandoned topics."""
        topic_durations = {}

        for msg_idx, topic in message_topics.items():
            if topic not in topic_durations:
                topic_durations[topic] = 0
            topic_durations[topic] += 1

        # Topics discussed for very few messages (< 3 mentions)
        avoided_topics = [topic for topic, count in topic_durations.items() if count < 2]

        # Topics that appear, disappear, then don't return (quick abandonment)
        quick_abandonment = 0
        topic_indices = {}
        for msg_idx in sorted(message_topics.keys()):
            topic = message_topics.get(msg_idx)
            if topic:
                if topic not in topic_indices:
                    topic_indices[topic] = []
                topic_indices[topic].append(msg_idx)

        for topic, indices in topic_indices.items():
            if len(indices) > 1:
                # Check for gaps
                gaps = []
                for i in range(len(indices) - 1):
                    gap = indices[i+1] - indices[i]
                    gaps.append(gap)

                # If there are large gaps and few total mentions, it's abandoned
                if max(gaps) > 5 and len(indices) < 5:
                    quick_abandonment += 1

        return avoided_topics, quick_abandonment

    def _detect_conversation_steering(self, messages: List[Dict[str, Any]],
                                     topic_shifts: List[TopicShift]) -> Dict[str, Any]:
        """Detect conversation steering patterns."""
        steering_found = False
        intensity = 0.0
        initiators = Counter()

        # Count forced topic shifts
        forced_shifts = [s for s in topic_shifts if s.shift_type == 'forced']
        if forced_shifts:
            steering_found = True
            intensity = min(1.0, len(forced_shifts) / len(messages)) if messages else 0.0

            for shift in forced_shifts:
                initiators[shift.initiating_speaker] += 1

        # Look for explicit steering language
        for msg in messages:
            text = msg.get('text', '').lower()
            for pattern in self.steering_patterns:
                if re.search(pattern, text):
                    steering_found = True
                    intensity = min(1.0, intensity + 0.2)
                    initiators[msg.get('sender', 'unknown')] += 1

        return {
            'detected': steering_found,
            'intensity': intensity,
            'initiators': dict(initiators)
        }

    def _calculate_topic_appropriateness(self, messages: List[Dict[str, Any]],
                                        keywords_by_message: Dict[int, List[str]]) -> Dict[str, Any]:
        """Calculate topic appropriateness and consistency."""
        off_topic_count = 0
        off_topic_indicators = []

        for i, msg in enumerate(messages):
            text = msg.get('text', '').lower()

            if re.search(r'\b(off topic|random|unrelated)\b', text):
                off_topic_count += 1
                off_topic_indicators.append(text[:50])

        # Consistency based on keyword distribution
        consistency = 0.0
        if messages:
            # Measure how well keywords are distributed across messages
            keyword_counts = [len(kws) for kws in keywords_by_message.values()]
            if keyword_counts:
                avg_keywords = sum(keyword_counts) / len(keyword_counts)
                # Lower variance = higher consistency
                variance = sum((k - avg_keywords) ** 2 for k in keyword_counts) / len(keyword_counts)
                consistency = 1.0 / (1.0 + (variance ** 0.5))

        appropriateness_score = max(0.0, 1.0 - (off_topic_count / len(messages))) if messages else 0.5

        return {
            'score': appropriateness_score,
            'off_topic': off_topic_indicators[:5],
            'consistency': consistency
        }

    def _identify_taboo_topics(self, messages: List[Dict[str, Any]],
                              keywords_by_message: Dict[int, List[str]]) -> List[str]:
        """Identify taboo or sensitive topics."""
        detected_taboo = set()

        for category, keywords in self.taboo_keywords.items():
            for msg_idx, msg_keywords in keywords_by_message.items():
                if any(kw in msg_keywords for kw in keywords):
                    detected_taboo.add(category)

        return list(detected_taboo)

    def _analyze_speaker_preferences(self, messages: List[Dict[str, Any]],
                                    message_topics: Dict[int, str]) -> Dict[str, Dict[str, int]]:
        """Analyze topic preferences by speaker."""
        preferences = {}

        for msg_idx, topic in message_topics.items():
            sender = messages[msg_idx].get('sender', 'unknown')
            if sender not in preferences:
                preferences[sender] = {}
            if topic not in preferences[sender]:
                preferences[sender][topic] = 0
            preferences[sender][topic] += 1

        return preferences

    def _calculate_topic_control(self, messages: List[Dict[str, Any]],
                                topics: List[Topic],
                                preferences: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Calculate topic control by speaker."""
        control = {}

        total_mentions = sum(topic.frequency for topic in topics)
        if total_mentions == 0:
            return control

        for speaker, topic_counts in preferences.items():
            speaker_total = sum(topic_counts.values())
            # Control = proportion of conversations they initiated
            control[speaker] = min(1.0, speaker_total / total_mentions)

        return control

    def _create_topic_timeline(self, messages: List[Dict[str, Any]],
                              message_topics: Dict[int, str]) -> List[Dict[str, Any]]:
        """Create timeline of topic changes."""
        timeline = []
        prev_topic = None

        for msg_idx in sorted(message_topics.keys()):
            topic = message_topics.get(msg_idx)
            if topic != prev_topic:
                timeline.append({
                    'message_index': msg_idx,
                    'topic': topic,
                    'sender': messages[msg_idx].get('sender', 'unknown')
                })
                prev_topic = topic

        return timeline

    def _calculate_diversity(self, topics: List[Topic]) -> float:
        """Calculate topic diversity."""
        if not topics:
            return 0.0

        total_frequency = sum(topic.frequency for topic in topics)
        if total_frequency == 0:
            return 0.0

        # Shannon entropy of topic distribution
        entropy = 0.0
        for topic in topics:
            if topic.frequency > 0:
                p = topic.frequency / total_frequency
                entropy -= p * math.log(p)

        # Normalize by max entropy
        max_entropy = math.log(len(topics))
        if max_entropy > 0:
            diversity = entropy / max_entropy
        else:
            diversity = 0.0

        return min(1.0, diversity)

    def _calculate_avg_duration(self, message_topics: Dict[int, str],
                               topics: List[Topic]) -> float:
        """Calculate average topic duration."""
        if not topics or not message_topics:
            return 0.0

        durations = []
        for topic in topics:
            # Find continuous blocks
            indices = [idx for idx, t in message_topics.items() if t == topic.name]
            if indices:
                indices.sort()
                block_lengths = []
                current_length = 1

                for i in range(len(indices) - 1):
                    if indices[i+1] - indices[i] == 1:
                        current_length += 1
                    else:
                        block_lengths.append(current_length)
                        current_length = 1
                block_lengths.append(current_length)

                avg_length = sum(block_lengths) / len(block_lengths)
                durations.append(avg_length)

        return sum(durations) / len(durations) if durations else 0.0

    def _get_longest_duration(self, message_topics: Dict[int, str]) -> int:
        """Get longest consecutive messages on same topic."""
        if not message_topics:
            return 0

        max_duration = 1
        current_duration = 1
        prev_topic = None

        for msg_idx in sorted(message_topics.keys()):
            topic = message_topics.get(msg_idx)
            if topic == prev_topic:
                current_duration += 1
            else:
                max_duration = max(max_duration, current_duration)
                current_duration = 1
            prev_topic = topic

        return max_duration

    def _calculate_confidence(self, messages: List[Dict], topics: List[Topic]) -> float:
        """Calculate confidence in analysis."""
        confidence = 0.5

        # More messages = higher confidence
        if len(messages) > 50:
            confidence += 0.3
        elif len(messages) > 20:
            confidence += 0.2
        elif len(messages) > 10:
            confidence += 0.1

        # Clear topics = higher confidence
        if topics and topics[0].frequency > 5:
            confidence += 0.2

        return min(1.0, confidence)

    def _empty_result(self) -> TopicModelingResult:
        """Return empty result for edge cases."""
        return TopicModelingResult(confidence=0.0)


# Example usage
if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        {'sender': 'A', 'text': 'So I had this crazy thing happen at work today.', 'timestamp': None},
        {'sender': 'B', 'text': 'Oh yeah? What happened at the office?', 'timestamp': None},
        {'sender': 'A', 'text': 'My boss was being really difficult during the meeting.', 'timestamp': None},
        {'sender': 'B', 'text': 'That sounds stressful. Work can be tough.', 'timestamp': None},
        {'sender': 'A', 'text': 'Anyway, speaking of difficult things, how\'s your health been?', 'timestamp': None},
        {'sender': 'B', 'text': 'Pretty good! Been exercising more and eating better.', 'timestamp': None},
        {'sender': 'A', 'text': 'That\'s great! I should really get back to the gym.', 'timestamp': None},
        {'sender': 'B', 'text': 'You should! We could go together sometime.', 'timestamp': None},
    ]

    analyzer = TopicModelingAnalyzer()
    result = analyzer.analyze(test_messages)

    print("Topic Modeling Analysis:")
    print(f"Topics Detected: {result.topic_count}")
    print(f"Dominant Topic: {result.dominant_topic}")
    print(f"Topic Diversity: {result.topic_diversity:.3f}")
    print(f"Topic Shifts: {result.shift_count}")
    print(f"Shift Frequency: {result.shift_frequency:.3f}")
    print(f"Average Topic Duration: {result.average_topic_duration:.1f}")
    print(f"Steering Detected: {result.steering_detected}")
    print(f"Topic Appropriateness: {result.topic_appropriateness_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
