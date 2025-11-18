"""
Intent Classification System
Classifies message intent into categories: neutral, supportive, conflictive, coercive, controlling
Based on linguistic patterns and context analysis.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

# Import model cache for pattern caching
from .model_cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class IntentClassification:
    """Container for intent classification results"""
    primary_intent: str = "neutral"  # Main intent category
    intent_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    linguistic_markers: List[str] = field(default_factory=list)
    communication_style: str = "neutral"  # assertive, passive, aggressive, passive-aggressive
    action_orientation: str = "none"  # requesting, offering, commanding, questioning
    emotional_tone: str = "neutral"  # warm, cold, hostile, caring


class IntentClassifier:
    """Classifies communication intent based on linguistic patterns"""

    # Intent patterns
    INTENT_PATTERNS = {
        'supportive': {
            'patterns': [
                r'\bi (understand|hear|see) (you|what|how)',
                r'\b(i\'m|i am) (here|sorry|proud)',
                r'\byou (can|will|are) (do|make|get) (it|this|through)',
                r'\b(let me|i can|i\'ll) help',
                r'\bthat (must be|sounds) (hard|difficult|tough)',
                r'\byou\'re (doing|handling) (great|well|amazing)',
                r'\bi believe in you',
                r'\btake your time',
                r'\bit\'s (okay|ok|alright)',
                r'\b(thank you|thanks) for',
            ],
            'weight': 1.2
        },
        'conflictive': {
            'patterns': [
                r'\byou (always|never) \w+',
                r'\bwhy (don\'t|can\'t|won\'t) you',
                r'\byou\'re (wrong|mistaken|lying)',
                r'\bthat\'s (stupid|ridiculous|absurd)',
                r'\bi (hate|can\'t stand|despise)',
                r'\byou (disgust|annoy|irritate) me',
                r'\b(shut up|go away|leave me alone)',
                r'\byou\'re (an idiot|stupid|pathetic)',
                r'\bi don\'t care',
                r'\bwhatever\b',
            ],
            'weight': 1.3
        },
        'coercive': {
            'patterns': [
                r'\byou (have to|must|need to)',
                r'\b(do it|do this|do that) (now|immediately)',
                r'\bif you don\'t .* i (will|\'ll)',
                r'\byou (better|should) \w+',
                r'\bi (demand|insist|require)',
                r'\byou owe (me|it)',
                r'\bdon\'t make me',
                r'\byou\'ll (regret|be sorry)',
                r'\bi\'m warning you',
                r'\blast chance',
            ],
            'weight': 1.5
        },
        'controlling': {
            'patterns': [
                r'\b(don\'t|do not) (talk to|see|go)',
                r'\byou (can\'t|cannot|may not)',
                r'\bi (forbid|prohibit|won\'t allow)',
                r'\bask (my |for )?permission',
                r'\bi decide (what|when|where|who)',
                r'\byou need my (approval|permission)',
                r'\bonly if i say',
                r'\bi\'ll (check|monitor|watch)',
                r'\bshow me your',
                r'\bgive me your (phone|password)',
            ],
            'weight': 1.6
        },
        'neutral': {
            'patterns': [
                r'\b(okay|ok|sure|yes|no)\b',
                r'\bi (think|believe|feel)',
                r'\b(maybe|perhaps|possibly)',
                r'\bthat\'s (interesting|nice|good)',
                r'\bi (see|understand)',
                r'\b(how|what|when|where|why) (is|are|do|does)',
            ],
            'weight': 0.8
        }
    }

    # Communication style indicators
    STYLE_PATTERNS = {
        'assertive': [
            r'\bi (need|want|prefer|would like)',
            r'\bi (feel|think|believe) that',
            r'\bin my opinion',
            r'\bi (agree|disagree)',
            r'\blet\'s (discuss|talk about)',
        ],
        'passive': [
            r'\b(sorry|excuse me) for',
            r'\bif (it\'s|that\'s) (ok|okay|alright)',
            r'\bi (guess|suppose|maybe)',
            r'\b(don\'t worry|never mind)',
            r'\bwhatever you (want|say|think)',
        ],
        'aggressive': [
            r'\byou (idiot|moron|fool)',
            r'\bshut (up|your mouth)',
            r'\bi don\'t give a',
            r'\bget (lost|out|away)',
            r'\b(stupid|dumb|pathetic)',
        ],
        'passive_aggressive': [
            r'\bfine\b(?! and)',  # "Fine" alone, not "fine and..."
            r'\bwhatever\b',
            r'\bif you say so',
            r'\bi\'m not (mad|angry|upset)',
            r'\bdo what you want',
            r'\bsure, (but|though)',
        ]
    }

    # Action orientation patterns
    ACTION_PATTERNS = {
        'requesting': [
            r'\b(can|could|would|will) you',
            r'\b(please|kindly) \w+',
            r'\bi (need|want) you to',
            r'\bwould you mind',
        ],
        'offering': [
            r'\b(can|may|shall) i',
            r'\blet me',
            r'\bi (can|could|will) \w+ for you',
            r'\bwould you like me to',
        ],
        'commanding': [
            r'^[A-Z][a-z]+ (it|this|that|me)',  # Starts with imperative verb
            r'\b(do|don\'t|stop|start) \w+ing',
            r'\byou (will|must|shall)',
            r'\b(immediately|now|right away)',
        ],
        'questioning': [
            r'^(who|what|when|where|why|how)',
            r'\?$',  # Ends with question mark
            r'\b(is|are|do|does|did|can|could|would|will) (you|it|this|that)',
        ]
    }

    def __init__(self):
        """Initialize intent classifier with cached patterns"""
        cache = get_cache()
        # Cache compiled patterns for faster initialization
        self.compiled_patterns = cache.get_or_load(
            'intent_compiled_patterns',
            self._compile_patterns
        )

    def _compile_patterns(self) -> Dict[str, Dict]:
        """Compile regex patterns for efficiency"""
        compiled = {}

        # Compile intent patterns
        for intent, data in self.INTENT_PATTERNS.items():
            compiled[intent] = {
                'patterns': [re.compile(p, re.IGNORECASE) for p in data['patterns']],
                'weight': data['weight']
            }

        # Compile style patterns
        compiled['styles'] = {}
        for style, patterns in self.STYLE_PATTERNS.items():
            compiled['styles'][style] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Compile action patterns
        compiled['actions'] = {}
        for action, patterns in self.ACTION_PATTERNS.items():
            compiled['actions'][action] = [re.compile(p, re.IGNORECASE) for p in patterns]

        return compiled

    def classify_intent(self, text: str, context: Optional[Dict] = None) -> IntentClassification:
        """Classify the intent of a message

        Args:
            text: Message text
            context: Optional context (previous messages, sender info, etc.)

        Returns:
            IntentClassification: Classification results
        """
        classification = IntentClassification()

        if not text:
            return classification

        # Score each intent category
        intent_scores = {}
        matched_patterns = []

        for intent, data in self.compiled_patterns.items():
            if intent in ['styles', 'actions']:
                continue

            score = 0
            matches = 0

            for pattern in data['patterns']:
                if pattern.search(text):
                    matches += 1
                    matched_patterns.append(pattern.pattern)

            if matches > 0:
                # Calculate score based on matches and weight
                base_score = matches / len(data['patterns'])
                score = base_score * data['weight']
                intent_scores[intent] = score

        # Normalize scores
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                intent_scores = {k: v/max_score for k, v in intent_scores.items()}

        classification.intent_scores = intent_scores

        # Determine primary intent
        if intent_scores:
            primary = max(intent_scores.items(), key=lambda x: x[1])
            classification.primary_intent = primary[0]
            classification.confidence = primary[1]
        else:
            classification.primary_intent = "neutral"
            classification.confidence = 0.5

        # Store matched patterns
        classification.linguistic_markers = matched_patterns

        # Classify communication style
        classification.communication_style = self._classify_style(text)

        # Classify action orientation
        classification.action_orientation = self._classify_action(text)

        # Determine emotional tone
        classification.emotional_tone = self._determine_emotional_tone(
            text, classification.primary_intent, classification.communication_style
        )

        # Adjust confidence based on context
        if context:
            classification.confidence = self._adjust_confidence_with_context(
                classification.confidence, context
            )

        return classification

    def _classify_style(self, text: str) -> str:
        """Classify communication style

        Args:
            text: Message text

        Returns:
            str: Communication style
        """
        style_scores = {}

        for style, patterns in self.compiled_patterns['styles'].items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                style_scores[style] = matches

        if style_scores:
            return max(style_scores.items(), key=lambda x: x[1])[0]

        return "neutral"

    def _classify_action(self, text: str) -> str:
        """Classify action orientation

        Args:
            text: Message text

        Returns:
            str: Action orientation
        """
        action_scores = {}

        for action, patterns in self.compiled_patterns['actions'].items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                action_scores[action] = matches

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]

        return "none"

    def _determine_emotional_tone(self, text: str, intent: str, style: str) -> str:
        """Determine emotional tone of message

        Args:
            text: Message text
            intent: Primary intent
            style: Communication style

        Returns:
            str: Emotional tone
        """
        # Map intent and style to emotional tone
        if intent == "supportive":
            return "warm"
        elif intent == "conflictive" or style == "aggressive":
            return "hostile"
        elif intent == "coercive" or intent == "controlling":
            return "cold"
        elif style == "passive":
            return "withdrawn"

        # Check for caring indicators
        caring_patterns = [r'\blove\b', r'\bcare about', r'\bworried about', r'\bconcerned']
        if any(re.search(p, text, re.IGNORECASE) for p in caring_patterns):
            return "caring"

        return "neutral"

    def _adjust_confidence_with_context(self, confidence: float, context: Dict) -> float:
        """Adjust confidence based on context

        Args:
            confidence: Initial confidence
            context: Context information

        Returns:
            float: Adjusted confidence
        """
        adjusted = confidence

        # Check for pattern consistency
        if 'previous_intent' in context:
            if context['previous_intent'] == context.get('current_intent'):
                adjusted += 0.1  # Consistent intent increases confidence

        # Check for sender history
        if 'sender_history' in context:
            history = context['sender_history']
            if 'typical_intent' in history:
                if history['typical_intent'] == context.get('current_intent'):
                    adjusted += 0.05  # Matches typical behavior

        return min(1.0, adjusted)

    def analyze_conversation_intents(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze intents across a conversation

        Args:
            messages: List of messages

        Returns:
            Dict: Conversation intent analysis
        """
        results = {
            "per_message_intents": [],
            "per_speaker_intents": {},
            "intent_distribution": {},
            "intent_transitions": [],
            "conflict_episodes": [],
            "supportive_episodes": [],
            "dominant_intent": "neutral",
            "conversation_dynamic": "neutral",  # collaborative, adversarial, mixed
            "recommendations": []
        }

        # Track intent sequences
        intent_sequence = []
        speaker_intents = {}

        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            sender = msg.get('sender', 'Unknown')

            # Create context
            context = {}
            if i > 0:
                context['previous_intent'] = intent_sequence[-1] if intent_sequence else None

            # Classify intent
            classification = self.classify_intent(text, context)

            # Store results
            msg_result = {
                "index": i,
                "sender": sender,
                "intent": classification.primary_intent,
                "confidence": classification.confidence,
                "style": classification.communication_style,
                "action": classification.action_orientation,
                "tone": classification.emotional_tone
            }
            results["per_message_intents"].append(msg_result)

            # Track sequence
            intent_sequence.append(classification.primary_intent)

            # Track by speaker
            if sender not in speaker_intents:
                speaker_intents[sender] = []
            speaker_intents[sender].append(classification.primary_intent)

            # Track intent transitions
            if i > 0:
                prev_intent = intent_sequence[-2]
                curr_intent = intent_sequence[-1]
                if prev_intent != curr_intent:
                    results["intent_transitions"].append({
                        "index": i,
                        "from": prev_intent,
                        "to": curr_intent,
                        "sender": sender
                    })

            # Detect conflict episodes
            if classification.primary_intent in ["conflictive", "coercive", "controlling"]:
                if not results["conflict_episodes"] or \
                   results["conflict_episodes"][-1]["end_index"] < i - 1:
                    # Start new conflict episode
                    results["conflict_episodes"].append({
                        "start_index": i,
                        "end_index": i,
                        "participants": [sender],
                        "intensity": classification.confidence
                    })
                else:
                    # Extend current conflict episode
                    episode = results["conflict_episodes"][-1]
                    episode["end_index"] = i
                    if sender not in episode["participants"]:
                        episode["participants"].append(sender)
                    episode["intensity"] = max(episode["intensity"], classification.confidence)

            # Detect supportive episodes
            if classification.primary_intent == "supportive":
                if not results["supportive_episodes"] or \
                   results["supportive_episodes"][-1]["end_index"] < i - 1:
                    results["supportive_episodes"].append({
                        "start_index": i,
                        "end_index": i,
                        "participants": [sender]
                    })
                else:
                    episode = results["supportive_episodes"][-1]
                    episode["end_index"] = i
                    if sender not in episode["participants"]:
                        episode["participants"].append(sender)

        # Calculate intent distribution
        for intent in intent_sequence:
            results["intent_distribution"][intent] = \
                results["intent_distribution"].get(intent, 0) + 1

        # Determine dominant intent
        if results["intent_distribution"]:
            results["dominant_intent"] = max(
                results["intent_distribution"].items(),
                key=lambda x: x[1]
            )[0]

        # Analyze per-speaker intents
        for sender, intents in speaker_intents.items():
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            dominant = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else "neutral"

            results["per_speaker_intents"][sender] = {
                "dominant_intent": dominant,
                "intent_distribution": intent_counts,
                "message_count": len(intents),
                "conflict_ratio": (intent_counts.get("conflictive", 0) +
                                 intent_counts.get("coercive", 0) +
                                 intent_counts.get("controlling", 0)) / len(intents) if intents else 0,
                "support_ratio": intent_counts.get("supportive", 0) / len(intents) if intents else 0
            }

        # Determine conversation dynamic
        results["conversation_dynamic"] = self._determine_dynamic(results)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _determine_dynamic(self, results: Dict) -> str:
        """Determine overall conversation dynamic

        Args:
            results: Analysis results

        Returns:
            str: Conversation dynamic
        """
        conflict_count = sum(
            results["intent_distribution"].get(intent, 0)
            for intent in ["conflictive", "coercive", "controlling"]
        )
        support_count = results["intent_distribution"].get("supportive", 0)
        total = sum(results["intent_distribution"].values())

        if total == 0:
            return "neutral"

        conflict_ratio = conflict_count / total
        support_ratio = support_count / total

        if conflict_ratio > 0.4:
            return "adversarial"
        elif support_ratio > 0.3:
            return "collaborative"
        elif conflict_ratio > 0.2 and support_ratio > 0.2:
            return "mixed"
        else:
            return "neutral"

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on intent analysis

        Args:
            results: Analysis results

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        dynamic = results["conversation_dynamic"]

        if dynamic == "adversarial":
            recommendations.append("HIGH CONFLICT: Conversation shows adversarial dynamic.")
            recommendations.append("Consider mediation or taking a break from the conversation.")
            recommendations.append("Focus on 'I' statements and avoid accusatory language.")

        elif dynamic == "collaborative":
            recommendations.append("POSITIVE: Conversation shows collaborative dynamic.")
            recommendations.append("Continue building on this supportive communication.")

        elif dynamic == "mixed":
            recommendations.append("MIXED: Conversation alternates between conflict and support.")
            recommendations.append("Try to identify triggers for conflict episodes.")
            recommendations.append("Build on supportive moments to improve overall dynamic.")

        # Check for specific concerning patterns
        for speaker, data in results["per_speaker_intents"].items():
            if data["conflict_ratio"] > 0.5:
                recommendations.append(f"Speaker '{speaker}' shows high conflict orientation ({data['conflict_ratio']:.1%}).")

            if data["dominant_intent"] in ["coercive", "controlling"]:
                recommendations.append(f"WARNING: Speaker '{speaker}' shows {data['dominant_intent']} behavior patterns.")

        # Episode-based recommendations
        if len(results["conflict_episodes"]) > 3:
            recommendations.append("Multiple conflict episodes detected. Consider underlying issues.")

        return recommendations