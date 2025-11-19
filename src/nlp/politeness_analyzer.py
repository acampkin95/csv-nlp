"""
Politeness Analyzer

Analyzes politeness patterns in conversations including politeness markers,
formality levels, request strategies, face-threatening acts, and respect indicators.
Uses linguistic markers to detect politeness and social distance patterns.

Author: Message Processor Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolitenessResult:
    """Results from politeness analysis."""

    # Overall metrics
    politeness_score: float = 0.5  # 0 (impolite) to 1 (very polite)
    politeness_level: str = 'neutral'  # very_impolite, impolite, neutral, polite, very_polite
    formality_level: float = 0.5  # 0 (very informal) to 1 (very formal)
    formality_classification: str = 'neutral'  # very_informal, informal, neutral, formal, very_formal

    # Politeness markers
    politeness_markers_found: List[str] = field(default_factory=list)
    polite_marker_count: int = 0
    impolite_marker_count: int = 0

    # Request strategies
    request_count: int = 0
    direct_request_ratio: float = 0.0
    indirect_request_ratio: float = 0.0
    polite_request_ratio: float = 0.0

    # Face-threatening acts
    face_threatening_acts_detected: bool = False
    face_threatening_act_count: int = 0
    face_threat_examples: List[str] = field(default_factory=list)

    # Apology and gratitude
    apology_count: int = 0
    gratitude_count: int = 0
    apology_to_message_ratio: float = 0.0
    gratitude_to_message_ratio: float = 0.0

    # Social distance
    social_distance_markers: List[str] = field(default_factory=list)
    social_distance_score: float = 0.0  # 0 (close) to 1 (distant)
    intimacy_level: str = 'semi_formal'  # formal, semi_formal, semi_intimate, intimate

    # Respect indicators
    respect_markers_count: int = 0
    disrespect_markers_count: int = 0
    net_respect_score: float = 0.0  # -1 (disrespectful) to 1 (respectful)

    # Hedging and mitigation
    hedging_phrases_count: int = 0
    mitigation_strategies: List[str] = field(default_factory=list)
    hedging_ratio: float = 0.0

    # Per-speaker metrics
    speaker_politeness: Dict[str, float] = field(default_factory=dict)
    speaker_formality: Dict[str, float] = field(default_factory=dict)

    # Reciprocal patterns
    politeness_imbalance: float = 0.0  # Difference in politeness between speakers
    formality_mismatch: bool = False

    # Evidence
    detailed_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'politeness_score': self.politeness_score,
            'politeness_level': self.politeness_level,
            'formality_level': self.formality_level,
            'formality_classification': self.formality_classification,
            'polite_marker_count': self.polite_marker_count,
            'impolite_marker_count': self.impolite_marker_count,
            'request_count': self.request_count,
            'direct_request_ratio': self.direct_request_ratio,
            'indirect_request_ratio': self.indirect_request_ratio,
            'apology_count': self.apology_count,
            'gratitude_count': self.gratitude_count,
            'face_threatening_acts_detected': self.face_threatening_acts_detected,
            'face_threatening_act_count': self.face_threatening_act_count,
            'social_distance_score': self.social_distance_score,
            'intimacy_level': self.intimacy_level,
            'respect_score': self.net_respect_score,
            'hedging_ratio': self.hedging_ratio,
            'politeness_imbalance': self.politeness_imbalance,
            'formality_mismatch': self.formality_mismatch,
            'confidence': self.confidence
        }


class PolitenessAnalyzer:
    """
    Analyzes politeness patterns in conversations.

    Features:
    - Politeness marker detection
    - Formality level tracking
    - Request strategy analysis
    - Face-threatening act detection
    - Apology/gratitude pattern tracking
    - Social distance indicators
    - Respect vs disrespect markers
    - Hedging and mitigation analysis
    """

    def __init__(self):
        """Initialize the politeness analyzer."""
        self._compile_patterns()
        logger.info("PolitenessAnalyzer initialized")

    def _compile_patterns(self):
        """Compile regex patterns for detection."""

        # Politeness markers (polite expressions)
        self.polite_markers = {
            'explicit_politeness': [
                r'\bplease\b', r'\bthank you\b', r'\bthanks\b', r'\bthank\b',
                r'\bexcuse me\b', r'\bpardon me\b', r'\bsorry\b', r'\bi\'m sorry\b',
                r'\bkindly\b', r'\bif you would\b', r'\bwould you mind\b',
            ],
            'indirect_requests': [
                r'\bcould you\b', r'\bwould you\b', r'\bmight you\b',
                r'\bif you don\'t mind\b', r'\bif it\'s not too much\b',
                r'\bwould you be so kind\b', r'\bwould you possibly\b',
            ],
            'deference': [
                r'\byour opinion\b', r'\bi think\b', r'\bin my view\b',
                r'\bas i see it\b', r'\bif you agree\b', r'\bwhat do you think\b',
            ]
        }

        # Impolite markers
        self.impolite_markers = {
            'direct_rudeness': [
                r'\bstop\b', r'\bshut up\b', r'\bshut your mouth\b',
                r'\bi don\'t care\b', r'\bdone\b', r'\bno way\b',
            ],
            'disrespect': [
                r'\bstupid\b', r'\bidiot\b', r'\bdumb\b', r'\bfool\b',
                r'\byou\'re wrong\b', r'\bthat\'s ridiculous\b',
            ],
            'dismissal': [
                r'\bwhatever\b', r'\bnever mind\b', r'\bfine\b',
                r'\byou wouldn\'t understand\b', r'\bnot your concern\b',
            ]
        }

        # Formal language markers
        self.formal_markers = [
            r'\b(furthermore|moreover|however|nevertheless)\b',
            r'\b(pursuant to|in accordance with|aforementioned)\b',
            r'\b(shall|henceforth|whereby|herein)\b',
            r'\b(respectfully|sincerely|regards)\b',
            r'\b(would you be so kind|kindly|I would appreciate)\b',
        ]

        # Informal language markers
        self.informal_markers = [
            r'\b(gonna|gotta|wanna|dunno|kinda|sorta)\b',
            r'\b(yeah|yep|nope|nah|cool|awesome|dude|bro)\b',
            r'\b(omg|lol|lmao|wtf|asap)\b',
            r'\bain\'t\b', r'\bcan\'t', r'\bdon\'t\b',
        ]

        # Request types
        self.direct_request_patterns = [
            r'\b(do|get|tell|show|give|make|go|come)\s+\w+',
            r'\byou (must|have to|need to|will|should)\b',
        ]

        self.indirect_request_patterns = [
            r'\bcould you\b', r'\bwould you\b', r'\bmight you\b',
            r'\bwould it be possible\b', r'\bdo you think you could\b',
        ]

        self.polite_request_patterns = [
            r'\bif you don\'t mind\b', r'\bif it\'s not too much trouble\b',
            r'\bwould you be so kind\b', r'\bwould you possibly\b',
        ]

        # Face-threatening acts (criticism, disagreement, requests)
        self.face_threat_patterns = {
            'criticism': [
                r'\bthat\'s (wrong|bad|stupid|ridiculous)\b',
                r'\byou (always|never) \w+', r'\byou didn\'t\b',
            ],
            'disagreement': [
                r'\bi disagree\b', r'\bthat\'s not right\b',
                r'\byou\'re wrong\b', r'\bi don\'t think so\b',
            ],
            'demands': [
                r'\byou (have to|must|will)\b',
                r'\b(do it|do this|do that) (now|immediately)\b',
            ],
            'accusations': [
                r'\byou (lied|didn\'t tell|hid|concealed)\b',
                r'\byou\'re (lying|being dishonest)\b',
            ]
        }

        # Apology patterns
        self.apology_patterns = [
            r'\bi\'m sorry\b', r'\bi apologize\b', r'\bmy apologies\b',
            r'\bsorry about\b', r'\bsorry for\b', r'\bpardon me\b',
        ]

        # Gratitude patterns
        self.gratitude_patterns = [
            r'\bthank you\b', r'\bthanks\b', r'\bthank\b',
            r'\bi appreciate\b', r'\bvery kind of you\b',
            r'\bthanks for\b', r'\bthank you for\b',
        ]

        # Social distance markers
        self.formal_address_patterns = [
            r'\bMr\.|Mrs\.|Ms\.|Dr\.|Prof\.',
            r'\byour (sir|madam|honor)',
        ]

        self.informal_address_patterns = [
            r'\bhey\b', r'\bhey there\b', r'\bfriend\b', r'\bbuddy\b',
            r'\bdude\b', r'\bguy\b', r'\bhun\b', r'\bdear\b',
        ]

        # Respect markers
        self.respect_markers = [
            r'\bi respect\b', r'\bi admire\b', r'\byou\'re amazing\b',
            r'\byou\'re great\b', r'\b(well done|great job|excellent work)\b',
        ]

        # Disrespect markers
        self.disrespect_markers = [
            r'\byou\'re (stupid|dumb|idiot|fool)\b',
            r'\byou\'re not (smart|good|capable)\b',
            r'\bi don\'t respect\b', r'\byou\'re pathetic\b',
        ]

        # Hedging patterns
        self.hedging_patterns = [
            r'\b(maybe|perhaps|possibly|arguably|sort of|kind of)\b',
            r'\b(in a sense|to some extent|somewhat|relatively)\b',
            r'\b(i think|i believe|it seems|it appears)\b',
            r'\b(i could be wrong|correct me if|unless i\'m mistaken)\b',
        ]

        # Mitigation strategies
        self.mitigation_patterns = {
            'minimization': [
                r'\b(just|only|merely|simply)\b',
            ],
            'softening': [
                r'\b(a bit|a little|somewhat|kind of|sort of)\b',
            ],
            'appreciation': [
                r'\bthank you\b', r'\bi appreciate\b',
            ],
            'justification': [
                r'\bbecause\b', r'\bsince\b', r'\bas\b',
            ]
        }

    def analyze(self, messages: List[Dict[str, Any]]) -> PolitenessResult:
        """
        Analyze politeness in conversation.

        Args:
            messages: List of message dictionaries with 'text', 'sender', 'timestamp'

        Returns:
            PolitenessResult with politeness metrics
        """
        if not messages:
            return self._empty_result()

        # Analyze politeness markers
        marker_metrics = self._analyze_politeness_markers(messages)

        # Analyze formality
        formality_metrics = self._analyze_formality(messages)

        # Analyze request strategies
        request_metrics = self._analyze_requests(messages)

        # Analyze face-threatening acts
        fta_metrics = self._analyze_face_threats(messages)

        # Analyze apologies and gratitude
        ag_metrics = self._analyze_apologies_gratitude(messages)

        # Analyze social distance
        distance_metrics = self._analyze_social_distance(messages)

        # Analyze respect indicators
        respect_metrics = self._analyze_respect(messages)

        # Analyze hedging
        hedging_metrics = self._analyze_hedging(messages)

        # Calculate per-speaker metrics
        speaker_politeness = self._calculate_speaker_politeness(messages, marker_metrics)
        speaker_formality = self._calculate_speaker_formality(messages, formality_metrics)

        # Calculate overall politeness score
        politeness_score = self._calculate_overall_politeness(marker_metrics, fta_metrics, ag_metrics)

        # Calculate politeness level
        politeness_level = self._classify_politeness(politeness_score)

        # Calculate formality level
        avg_formality = sum(formality_metrics.values()) / len(formality_metrics) if formality_metrics else 0.5
        formality_level = self._classify_formality(avg_formality)

        # Detect imbalances
        politeness_imbalance = self._calculate_politeness_imbalance(speaker_politeness)
        formality_mismatch = self._detect_formality_mismatch(speaker_formality)

        # Calculate confidence
        confidence = self._calculate_confidence(messages, marker_metrics)

        return PolitenessResult(
            politeness_score=politeness_score,
            politeness_level=politeness_level,
            formality_level=avg_formality,
            formality_classification=formality_level,
            polite_marker_count=marker_metrics['polite_count'],
            impolite_marker_count=marker_metrics['impolite_count'],
            politeness_markers_found=marker_metrics['markers'],
            request_count=request_metrics['total_requests'],
            direct_request_ratio=request_metrics['direct_ratio'],
            indirect_request_ratio=request_metrics['indirect_ratio'],
            polite_request_ratio=request_metrics['polite_ratio'],
            face_threatening_acts_detected=fta_metrics['detected'],
            face_threatening_act_count=fta_metrics['count'],
            face_threat_examples=fta_metrics['examples'],
            apology_count=ag_metrics['apologies'],
            gratitude_count=ag_metrics['gratitude'],
            apology_to_message_ratio=ag_metrics['apology_ratio'],
            gratitude_to_message_ratio=ag_metrics['gratitude_ratio'],
            social_distance_markers=distance_metrics['markers'],
            social_distance_score=distance_metrics['distance_score'],
            intimacy_level=distance_metrics['intimacy_level'],
            respect_markers_count=respect_metrics['respect_count'],
            disrespect_markers_count=respect_metrics['disrespect_count'],
            net_respect_score=respect_metrics['net_score'],
            hedging_phrases_count=hedging_metrics['hedging_count'],
            mitigation_strategies=hedging_metrics['mitigation_types'],
            hedging_ratio=hedging_metrics['hedging_ratio'],
            speaker_politeness=speaker_politeness,
            speaker_formality=speaker_formality,
            politeness_imbalance=politeness_imbalance,
            formality_mismatch=formality_mismatch,
            detailed_findings=self._generate_findings(
                marker_metrics, request_metrics, fta_metrics, ag_metrics
            ),
            confidence=confidence
        )

    def _analyze_politeness_markers(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze politeness markers in messages."""
        polite_count = 0
        impolite_count = 0
        markers_found = []

        for msg in messages:
            text = msg.get('text', '').lower()

            # Count polite markers
            for category, patterns in self.polite_markers.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        polite_count += 1
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        markers_found.extend(matches)

            # Count impolite markers
            for category, patterns in self.impolite_markers.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        impolite_count += 1

        return {
            'polite_count': polite_count,
            'impolite_count': impolite_count,
            'markers': list(set(markers_found))[:15]  # Unique markers
        }

    def _analyze_formality(self, messages: List[Dict[str, Any]]) -> Dict[int, float]:
        """Analyze formality level in each message."""
        formality_scores = {}

        for i, msg in enumerate(messages):
            text = msg.get('text', '').lower()

            # Count formal and informal markers
            formal_count = sum(1 for pattern in self.formal_markers
                              if re.search(pattern, text, re.IGNORECASE))
            informal_count = sum(1 for pattern in self.informal_markers
                                if re.search(pattern, text, re.IGNORECASE))

            # Calculate formality score
            total_markers = formal_count + informal_count
            if total_markers > 0:
                formality_score = formal_count / total_markers
            else:
                # Default to neutral
                formality_score = 0.5

            formality_scores[i] = formality_score

        return formality_scores

    def _analyze_requests(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze request strategies."""
        direct_requests = 0
        indirect_requests = 0
        polite_requests = 0
        total_requests = 0

        for msg in messages:
            text = msg.get('text', '')

            # Check for requests
            is_direct = any(re.search(pattern, text, re.IGNORECASE)
                          for pattern in self.direct_request_patterns)
            is_indirect = any(re.search(pattern, text, re.IGNORECASE)
                            for pattern in self.indirect_request_patterns)
            is_polite = any(re.search(pattern, text, re.IGNORECASE)
                          for pattern in self.polite_request_patterns)

            if is_direct or is_indirect or is_polite:
                total_requests += 1

            if is_direct:
                direct_requests += 1
            if is_indirect:
                indirect_requests += 1
            if is_polite:
                polite_requests += 1

        # Calculate ratios
        if total_requests > 0:
            direct_ratio = direct_requests / total_requests
            indirect_ratio = indirect_requests / total_requests
            polite_ratio = polite_requests / total_requests
        else:
            direct_ratio = indirect_ratio = polite_ratio = 0.0

        return {
            'total_requests': total_requests,
            'direct_ratio': direct_ratio,
            'indirect_ratio': indirect_ratio,
            'polite_ratio': polite_ratio
        }

    def _analyze_face_threats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze face-threatening acts."""
        fta_count = 0
        fta_detected = False
        examples = []

        for msg in messages:
            text = msg.get('text', '')

            for category, patterns in self.face_threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        fta_count += 1
                        fta_detected = True
                        examples.append(text[:50])
                        break

        return {
            'detected': fta_detected,
            'count': fta_count,
            'examples': examples[:5]  # Limit to 5 examples
        }

    def _analyze_apologies_gratitude(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze apologies and gratitude expressions."""
        apology_count = 0
        gratitude_count = 0

        for msg in messages:
            text = msg.get('text', '').lower()

            apologies = sum(1 for pattern in self.apology_patterns
                          if re.search(pattern, text, re.IGNORECASE))
            gratitudes = sum(1 for pattern in self.gratitude_patterns
                           if re.search(pattern, text, re.IGNORECASE))

            apology_count += apologies
            gratitude_count += gratitudes

        # Calculate ratios
        if messages:
            apology_ratio = apology_count / len(messages)
            gratitude_ratio = gratitude_count / len(messages)
        else:
            apology_ratio = gratitude_ratio = 0.0

        return {
            'apologies': apology_count,
            'gratitude': gratitude_count,
            'apology_ratio': apology_ratio,
            'gratitude_ratio': gratitude_ratio
        }

    def _analyze_social_distance(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze social distance indicators."""
        formal_address = 0
        informal_address = 0
        distance_markers = []

        for msg in messages:
            text = msg.get('text', '')

            # Check for formal address
            if any(re.search(pattern, text) for pattern in self.formal_address_patterns):
                formal_address += 1
                distance_markers.append('formal_address')

            # Check for informal address
            if any(re.search(pattern, text, re.IGNORECASE)
                   for pattern in self.informal_address_patterns):
                informal_address += 1
                distance_markers.append('informal_address')

        # Calculate social distance score (0=close, 1=distant)
        total_address = formal_address + informal_address
        if total_address > 0:
            distance_score = formal_address / total_address
        else:
            distance_score = 0.5  # Neutral

        # Classify intimacy level
        if distance_score > 0.7:
            intimacy_level = 'formal'
        elif distance_score > 0.5:
            intimacy_level = 'semi_formal'
        elif distance_score > 0.3:
            intimacy_level = 'semi_intimate'
        else:
            intimacy_level = 'intimate'

        return {
            'distance_score': distance_score,
            'intimacy_level': intimacy_level,
            'markers': list(set(distance_markers))[:10]
        }

    def _analyze_respect(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze respect indicators."""
        respect_count = 0
        disrespect_count = 0

        for msg in messages:
            text = msg.get('text', '')

            respect_count += sum(1 for pattern in self.respect_markers
                                if re.search(pattern, text, re.IGNORECASE))
            disrespect_count += sum(1 for pattern in self.disrespect_markers
                                   if re.search(pattern, text, re.IGNORECASE))

        # Calculate net respect score (-1 to 1)
        total_respect = respect_count + disrespect_count
        if total_respect > 0:
            net_score = (respect_count - disrespect_count) / total_respect
        else:
            net_score = 0.0

        return {
            'respect_count': respect_count,
            'disrespect_count': disrespect_count,
            'net_score': net_score
        }

    def _analyze_hedging(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze hedging and mitigation strategies."""
        hedging_count = 0
        mitigation_types = []

        for msg in messages:
            text = msg.get('text', '')

            # Count hedging phrases
            for pattern in self.hedging_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    hedging_count += 1

            # Identify mitigation strategies
            for strategy, patterns in self.mitigation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        mitigation_types.append(strategy)
                        break

        # Calculate hedging ratio
        if messages:
            hedging_ratio = hedging_count / len(messages)
        else:
            hedging_ratio = 0.0

        return {
            'hedging_count': hedging_count,
            'hedging_ratio': hedging_ratio,
            'mitigation_types': list(set(mitigation_types))
        }

    def _calculate_speaker_politeness(self, messages: List[Dict[str, Any]],
                                     marker_metrics: Dict) -> Dict[str, float]:
        """Calculate per-speaker politeness score."""
        speaker_politeness = {}

        for msg in messages:
            sender = msg.get('sender', 'unknown')
            text = msg.get('text', '').lower()

            if sender not in speaker_politeness:
                speaker_politeness[sender] = {'polite': 0, 'impolite': 0, 'messages': 0}

            speaker_politeness[sender]['messages'] += 1

            # Count markers
            for category, patterns in self.polite_markers.items():
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    speaker_politeness[sender]['polite'] += 1

            for category, patterns in self.impolite_markers.items():
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    speaker_politeness[sender]['impolite'] += 1

        # Calculate scores
        result = {}
        for speaker, counts in speaker_politeness.items():
            polite_score = (counts['polite'] - counts['impolite']) / max(counts['messages'], 1)
            # Normalize to 0-1
            result[speaker] = max(0.0, min(1.0, (polite_score + 1.0) / 2.0))

        return result

    def _calculate_speaker_formality(self, messages: List[Dict[str, Any]],
                                    formality_scores: Dict) -> Dict[str, float]:
        """Calculate per-speaker formality score."""
        speaker_formality = {}

        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'unknown')
            if sender not in speaker_formality:
                speaker_formality[sender] = []

            if i in formality_scores:
                speaker_formality[sender].append(formality_scores[i])

        # Average formality per speaker
        result = {}
        for speaker, scores in speaker_formality.items():
            if scores:
                result[speaker] = sum(scores) / len(scores)
            else:
                result[speaker] = 0.5

        return result

    def _calculate_overall_politeness(self, marker_metrics: Dict,
                                     fta_metrics: Dict,
                                     ag_metrics: Dict) -> float:
        """Calculate overall politeness score."""
        # Weighted combination of factors
        polite_factor = marker_metrics['polite_count'] * 0.1  # Weight per marker
        impolite_penalty = marker_metrics['impolite_count'] * 0.15
        fta_penalty = fta_metrics['count'] * 0.1
        apology_bonus = ag_metrics['apologies'] * 0.05
        gratitude_bonus = ag_metrics['gratitude'] * 0.05

        score = max(0.0, (polite_factor + apology_bonus + gratitude_bonus - impolite_penalty - fta_penalty))

        # Normalize to 0-1 range
        score = min(1.0, score / 2.0)

        return score

    def _classify_politeness(self, score: float) -> str:
        """Classify politeness level."""
        if score < 0.2:
            return 'very_impolite'
        elif score < 0.4:
            return 'impolite'
        elif score < 0.6:
            return 'neutral'
        elif score < 0.8:
            return 'polite'
        else:
            return 'very_polite'

    def _classify_formality(self, formality_level: float) -> str:
        """Classify formality level."""
        if formality_level < 0.2:
            return 'very_informal'
        elif formality_level < 0.4:
            return 'informal'
        elif formality_level < 0.6:
            return 'neutral'
        elif formality_level < 0.8:
            return 'formal'
        else:
            return 'very_formal'

    def _calculate_politeness_imbalance(self, speaker_politeness: Dict[str, float]) -> float:
        """Calculate imbalance in politeness between speakers."""
        if not speaker_politeness or len(speaker_politeness) < 2:
            return 0.0

        scores = list(speaker_politeness.values())
        imbalance = max(scores) - min(scores)

        return round(imbalance, 3)

    def _detect_formality_mismatch(self, speaker_formality: Dict[str, float]) -> bool:
        """Detect if speakers have mismatched formality levels."""
        if not speaker_formality or len(speaker_formality) < 2:
            return False

        scores = list(speaker_formality.values())
        # Mismatch if difference > 0.4
        return max(scores) - min(scores) > 0.4

    def _generate_findings(self, marker_metrics: Dict, request_metrics: Dict,
                          fta_metrics: Dict, ag_metrics: Dict) -> List[Dict[str, Any]]:
        """Generate detailed findings."""
        findings = []

        if marker_metrics['polite_count'] > 0:
            findings.append({
                'finding': 'Politeness markers detected',
                'details': f"Found {marker_metrics['polite_count']} polite expressions"
            })

        if request_metrics['total_requests'] > 0:
            findings.append({
                'finding': 'Request strategies analysis',
                'details': f"Direct requests: {request_metrics['direct_ratio']:.1%}, Indirect: {request_metrics['indirect_ratio']:.1%}"
            })

        if fta_metrics['detected']:
            findings.append({
                'finding': 'Face-threatening acts detected',
                'details': f"Found {fta_metrics['count']} potential FTA instances"
            })

        if ag_metrics['apologies'] > 0:
            findings.append({
                'finding': 'Apologies',
                'details': f"Found {ag_metrics['apologies']} apology instances"
            })

        if ag_metrics['gratitude'] > 0:
            findings.append({
                'finding': 'Gratitude expressions',
                'details': f"Found {ag_metrics['gratitude']} gratitude instances"
            })

        return findings

    def _calculate_confidence(self, messages: List[Dict], marker_metrics: Dict) -> float:
        """Calculate confidence in analysis."""
        confidence = 0.5

        # More messages = higher confidence
        if len(messages) > 50:
            confidence += 0.3
        elif len(messages) > 20:
            confidence += 0.2
        elif len(messages) > 10:
            confidence += 0.1

        # Clear politeness patterns = higher confidence
        total_markers = marker_metrics['polite_count'] + marker_metrics['impolite_count']
        if total_markers > 10:
            confidence += 0.2
        elif total_markers > 3:
            confidence += 0.1

        return min(1.0, confidence)

    def _empty_result(self) -> PolitenessResult:
        """Return empty result for edge cases."""
        return PolitenessResult(
            politeness_score=0.5,
            politeness_level='neutral',
            formality_level=0.5,
            formality_classification='neutral',
            confidence=0.0
        )


# Example usage
if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        {'sender': 'A', 'text': 'Could you please help me with this project?', 'timestamp': None},
        {'sender': 'B', 'text': 'Of course! I\'d be happy to help.', 'timestamp': None},
        {'sender': 'A', 'text': 'Thank you so much, I really appreciate it.', 'timestamp': None},
        {'sender': 'B', 'text': 'No problem at all. If you need anything else, just let me know.', 'timestamp': None},
        {'sender': 'A', 'text': 'I\'m sorry to bother you, but one more thing...', 'timestamp': None},
        {'sender': 'B', 'text': 'You\'re not bothering me at all. What is it?', 'timestamp': None},
    ]

    analyzer = PolitenessAnalyzer()
    result = analyzer.analyze(test_messages)

    print("Politeness Analysis:")
    print(f"Politeness Score: {result.politeness_score:.3f}")
    print(f"Politeness Level: {result.politeness_level}")
    print(f"Formality Level: {result.formality_level:.3f}")
    print(f"Formality Classification: {result.formality_classification}")
    print(f"Polite Markers: {result.polite_marker_count}")
    print(f"Request Count: {result.request_count}")
    print(f"Apology Count: {result.apology_count}")
    print(f"Gratitude Count: {result.gratitude_count}")
    print(f"Respect Score: {result.net_respect_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
