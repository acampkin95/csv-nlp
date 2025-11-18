#!/usr/bin/env python3
"""
Person-Centric Analysis Module
Implements passes 11-15 of the unified pipeline:
- Person identification and role classification
- Interaction mapping and relationship structure
- Gaslighting-specific detection
- Relationship dynamics and power analysis
- Intervention recommendations and case formulation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class Person:
    """Represents a person in the conversation"""
    name: str
    aliases: List[str] = field(default_factory=list)
    message_count: int = 0
    role: str = "participant"  # initiator, responder, victim, perpetrator, etc.
    characteristics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    """Represents an interaction between two people"""
    speaker: str
    recipient: str
    message_indices: List[int] = field(default_factory=list)
    interaction_type: str = "generic"  # direct, indirect, accusatory, defensive, etc.
    power_direction: Optional[str] = None  # up, down, equal


class PersonAnalyzer:
    """Analyzes person-centric aspects of conversations"""

    def __init__(self):
        """Initialize person analyzer"""
        logger.info("PersonAnalyzer initialized")

    def identify_persons_in_conversation(self, messages: List[Dict], df: Optional[Any] = None) -> Dict[str, Any]:
        """
        Pass 11: Identify and classify persons in the conversation

        Args:
            messages: List of message dictionaries
            df: Optional DataFrame for additional context

        Returns:
            Dict containing identified persons and their roles
        """
        logger.info("Pass 11: Identifying persons in conversation")

        persons = {}
        sender_counts = defaultdict(int)
        message_indices = defaultdict(list)

        # Count messages per sender
        for idx, msg in enumerate(messages):
            sender = msg.get('sender', 'Unknown')
            sender_counts[sender] += 1
            message_indices[sender].append(idx)

        # Classify persons by message volume and roles
        sorted_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)

        for rank, (sender, count) in enumerate(sorted_senders):
            # Determine role based on activity
            if rank == 0:
                role = "primary_initiator"
            elif rank == 1:
                role = "primary_responder"
            else:
                role = "participant"

            # Additional role assessment
            role = self._assess_speaker_role(sender, messages, message_indices[sender], role)

            persons[sender] = {
                'name': sender,
                'aliases': self._extract_aliases(sender),
                'message_count': count,
                'role': role,
                'message_indices': message_indices[sender],
                'characteristics': self._analyze_speaker_characteristics(
                    sender, messages, message_indices[sender]
                )
            }

        logger.info(f"Identified {len(persons)} persons with roles: {[p['role'] for p in persons.values()]}")

        return {
            'persons': list(persons.values()),
            'person_map': persons,
            'total_speakers': len(persons),
            'conversation_type': self._classify_conversation_type(persons),
            'primary_dyad': self._identify_primary_dyad(persons)
        }

    def extract_interaction_patterns(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """
        Pass 12: Map interactions and relationship structures

        Args:
            messages: List of message dictionaries
            persons: List of identified persons

        Returns:
            Dict containing interaction mapping and structure
        """
        logger.info("Pass 12: Extracting interaction patterns")

        interactions = defaultdict(lambda: {'count': 0, 'indices': [], 'types': []})
        directed_interactions = []

        person_names = [p['name'] for p in persons]

        for idx, msg in enumerate(messages):
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '').lower()

            # Direct mentions
            for person in person_names:
                if person != sender:
                    if self._mentions_person(text, person):
                        key = f"{sender}->{person}"
                        interactions[key]['count'] += 1
                        interactions[key]['indices'].append(idx)
                        interaction_type = self._classify_interaction_type(text, sender, person)
                        interactions[key]['types'].append(interaction_type)

                        directed_interactions.append({
                            'speaker': sender,
                            'recipient': person,
                            'message_index': idx,
                            'type': interaction_type
                        })

        # Analyze interaction patterns
        pattern_analysis = self._analyze_interaction_patterns(directed_interactions, person_names)

        logger.info(f"Mapped {len(directed_interactions)} directed interactions")

        return {
            'interactions': directed_interactions,
            'interaction_matrix': dict(interactions),
            'pattern_analysis': pattern_analysis,
            'network_structure': self._analyze_network_structure(directed_interactions, person_names),
            'communication_balance': self._assess_communication_balance(directed_interactions, person_names)
        }

    def detect_gaslighting_patterns(self, messages: List[Dict], persons: List[Dict],
                                    manipulation_results: Dict) -> Dict[str, Any]:
        """
        Pass 13: Detect gaslighting-specific patterns

        Args:
            messages: List of message dictionaries
            persons: List of identified persons
            manipulation_results: Results from manipulation detection pass

        Returns:
            Dict containing gaslighting detection results
        """
        logger.info("Pass 13: Detecting gaslighting patterns")

        gaslighting_indicators = {
            'reality_denial': [],
            'blame_shifting': [],
            'trivializing': [],
            'diverting': [],
            'countering': []
        }

        gaslighting_phrases = {
            'reality_denial': [
                "that didn't happen",
                "you're making it up",
                "you're crazy",
                "you're being paranoid",
                "that never happened"
            ],
            'blame_shifting': [
                "you made me do it",
                "your fault",
                "if you hadn't",
                "you caused this",
                "it's because of you"
            ],
            'trivializing': [
                "you're too sensitive",
                "you can't take a joke",
                "don't be so dramatic",
                "you're overreacting",
                "it's not that bad"
            ],
            'diverting': [
                "why are you bringing up",
                "that's not important",
                "let's talk about what you",
                "what about when you",
                "you always"
            ],
            'countering': [
                "you're wrong",
                "that's not true",
                "you don't remember it right",
                "you misunderstood",
                "that's not what happened"
            ]
        }

        for idx, msg in enumerate(messages):
            text = msg.get('text', '').lower()

            for category, phrases in gaslighting_phrases.items():
                for phrase in phrases:
                    if phrase in text:
                        gaslighting_indicators[category].append({
                            'message_index': idx,
                            'speaker': msg.get('sender', 'Unknown'),
                            'phrase': phrase,
                            'context': msg.get('text', '')[:200]
                        })

        # Calculate gaslighting risk
        total_indicators = sum(len(v) for v in gaslighting_indicators.values())
        gaslighting_risk = self._assess_gaslighting_risk(total_indicators, len(messages))

        logger.info(f"Detected {total_indicators} gaslighting indicators (Risk: {gaslighting_risk})")

        return {
            'gaslighting_indicators': gaslighting_indicators,
            'total_indicators': total_indicators,
            'gaslighting_risk': gaslighting_risk,
            'high_risk_instances': [
                i for indicators in gaslighting_indicators.values()
                for i in indicators
            ],
            'perpetrators': self._identify_gaslighting_perpetrators(gaslighting_indicators),
            'victims': self._identify_gaslighting_victims(gaslighting_indicators, messages)
        }

    def assess_relationship_dynamics(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """
        Pass 14: Analyze relationship dynamics and power structures

        Args:
            messages: List of message dictionaries
            persons: List of identified persons

        Returns:
            Dict containing relationship analysis
        """
        logger.info("Pass 14: Assessing relationship dynamics")

        # Analyze power dynamics
        power_analysis = self._analyze_power_dynamics(messages, persons)

        # Analyze emotional patterns
        emotional_patterns = self._analyze_emotional_patterns(messages, persons)

        # Assess dependency dynamics
        dependency = self._assess_dependency_dynamics(messages, persons)

        # Identify control patterns
        control_patterns = self._identify_control_patterns(messages, persons)

        logger.info(f"Power imbalance detected: {power_analysis.get('power_imbalance', False)}")

        return {
            'power_dynamics': power_analysis,
            'emotional_patterns': emotional_patterns,
            'dependency_dynamics': dependency,
            'control_patterns': control_patterns,
            'relationship_type': self._classify_relationship_type(power_analysis, control_patterns),
            'relationship_quality': self._assess_relationship_quality(
                messages, power_analysis, control_patterns
            ),
            'power_imbalance': power_analysis.get('power_imbalance', False),
            'power_imbalance_severity': power_analysis.get('severity', 'none')
        }

    def generate_intervention_recommendations(self, risk_assessment: Dict, person_identification: Dict,
                                               relationship_analysis: Dict, gaslighting_detection: Dict) -> Dict[str, Any]:
        """
        Pass 15: Generate intervention recommendations and case formulation

        Args:
            risk_assessment: Results from risk assessment pass
            person_identification: Results from person identification pass
            relationship_analysis: Results from relationship analysis pass
            gaslighting_detection: Results from gaslighting detection pass

        Returns:
            Dict containing intervention recommendations
        """
        logger.info("Pass 15: Generating intervention recommendations")

        recommendations = []
        case_formulation = {}
        intervention_priority = "routine"

        # Risk-based recommendations
        risk_level = risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown')

        if risk_level in ['critical', 'high']:
            recommendations.extend([
                "Seek immediate professional mental health support",
                "Consult with a trauma-informed therapist",
                "Document all concerning interactions",
                "Consider safety planning if threats are present",
                "Establish clear boundaries with potentially harmful individuals"
            ])
            intervention_priority = "urgent"

        # Gaslighting-specific recommendations
        if gaslighting_detection.get('gaslighting_risk') in ['high', 'critical']:
            recommendations.extend([
                "Seek support from a therapist experienced with gaslighting",
                "Build a support network outside the relationship",
                "Practice reality-checking with trusted individuals",
                "Keep detailed records of events and interactions",
                "Develop self-validation techniques"
            ])
            intervention_priority = "urgent"

        # Power imbalance recommendations
        if relationship_analysis.get('power_imbalance'):
            severity = relationship_analysis.get('power_imbalance_severity', 'none')
            if severity in ['high', 'severe']:
                recommendations.extend([
                    "Consider professional mediation or counseling",
                    "Establish role clarity and boundaries",
                    "Develop assertiveness and communication skills",
                    "Explore relationship restructuring options"
                ])

        # Formulate case summary
        case_formulation = self._formulate_case(
            risk_assessment, person_identification, relationship_analysis, gaslighting_detection
        )

        logger.info(f"Generated {len(recommendations)} recommendations with priority: {intervention_priority}")

        return {
            'recommendations': list(set(recommendations)),  # Remove duplicates
            'intervention_priority': intervention_priority,
            'case_formulation': case_formulation,
            'resources': self._generate_resources(risk_level, gaslighting_detection),
            'follow_up_actions': self._identify_follow_up_actions(
                risk_level, relationship_analysis, gaslighting_detection
            ),
            'summary': {
                'key_findings': self._summarize_findings(
                    risk_assessment, person_identification, relationship_analysis, gaslighting_detection
                ),
                'clinical_impressions': self._generate_clinical_impressions(
                    risk_assessment, relationship_analysis, gaslighting_detection
                )
            }
        }

    # INTERNAL HELPER METHODS
    # =====================================================================

    def _assess_speaker_role(self, sender: str, messages: List[Dict], indices: List[int], default_role: str) -> str:
        """Assess speaker role based on message patterns"""
        if not indices:
            return default_role

        messages_subset = [messages[i] for i in indices if i < len(messages)]

        # Analyze message characteristics
        accusatory_count = sum(1 for msg in messages_subset
                              if any(w in msg.get('text', '').lower()
                                    for w in ['you always', 'you never', 'you made', 'your fault']))

        defensive_count = sum(1 for msg in messages_subset
                             if any(w in msg.get('text', '').lower()
                                   for w in ["i didn't", "that's not true", "no i didn't", "you're wrong"]))

        if accusatory_count > len(messages_subset) * 0.3:
            return "perpetrator"
        elif defensive_count > len(messages_subset) * 0.3:
            return "victim"

        return default_role

    def _extract_aliases(self, name: str) -> List[str]:
        """Extract possible aliases from person name"""
        aliases = []

        # Handle common name variations
        if ' ' in name:
            first_name = name.split()[0]
            aliases.append(first_name)

        return aliases

    def _analyze_speaker_characteristics(self, sender: str, messages: List[Dict], indices: List[int]) -> Dict[str, Any]:
        """Analyze characteristics of a speaker"""
        if not indices:
            return {}

        messages_subset = [messages[i] for i in indices if i < len(messages)]

        # Calculate message length average
        avg_length = sum(len(msg.get('text', '')) for msg in messages_subset) / len(messages_subset) if messages_subset else 0

        # Analyze tone
        aggressive_words = ['always', 'never', 'hate', 'disgusting', 'stupid', 'idiotic']
        aggressive_count = sum(
            1 for msg in messages_subset
            for word in aggressive_words
            if word in msg.get('text', '').lower()
        )

        return {
            'average_message_length': avg_length,
            'aggression_score': aggressive_count / len(messages_subset) if messages_subset else 0,
            'communication_style': 'aggressive' if aggressive_count > len(messages_subset) * 0.2 else 'neutral'
        }

    def _classify_conversation_type(self, persons: Dict) -> str:
        """Classify conversation type (dyadic, group, etc.)"""
        count = len(persons)

        if count == 2:
            return "dyadic"
        elif count <= 4:
            return "small_group"
        else:
            return "large_group"

    def _identify_primary_dyad(self, persons: Dict) -> Optional[Tuple[str, str]]:
        """Identify primary dyad in conversation"""
        if len(persons) < 2:
            return None

        sorted_persons = sorted(persons.items(), key=lambda x: x[1]['message_count'], reverse=True)

        if len(sorted_persons) >= 2:
            return (sorted_persons[0][0], sorted_persons[1][0])

        return None

    def _mentions_person(self, text: str, person: str) -> bool:
        """Check if text mentions a specific person"""
        # Simple mention detection
        person_lower = person.lower()
        first_name = person_lower.split()[0] if ' ' in person_lower else person_lower

        return person_lower in text or first_name in text or f"@{first_name}" in text

    def _classify_interaction_type(self, text: str, speaker: str, recipient: str) -> str:
        """Classify type of interaction"""
        text_lower = text.lower()

        if any(w in text_lower for w in ['you always', 'you never', 'you made', 'your fault']):
            return "accusatory"
        elif any(w in text_lower for w in ["i didn't", "that's not true", "no i didn't"]):
            return "defensive"
        elif any(w in text_lower for w in ['?']):
            return "questioning"
        elif any(w in text_lower for w in ['please', 'thank you', 'sorry']):
            return "cooperative"

        return "neutral"

    def _analyze_interaction_patterns(self, directed_interactions: List[Dict], persons: List[str]) -> Dict[str, Any]:
        """Analyze patterns in directed interactions"""
        patterns = {
            'asymmetrical_communication': self._detect_asymmetrical_communication(directed_interactions),
            'accusation_patterns': self._detect_accusation_patterns(directed_interactions),
            'defensive_patterns': self._detect_defensive_patterns(directed_interactions)
        }

        return patterns

    def _detect_asymmetrical_communication(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Detect asymmetrical communication patterns"""
        speaker_to_counts = defaultdict(int)
        speaker_from_counts = defaultdict(int)

        for interaction in interactions:
            speaker_to_counts[interaction['speaker']] += 1
            speaker_from_counts[interaction['recipient']] += 1

        asymmetry = {}
        for speaker in speaker_to_counts:
            to_count = speaker_to_counts[speaker]
            from_count = speaker_from_counts.get(speaker, 0)

            if to_count > 0 and from_count > 0:
                asymmetry[speaker] = {
                    'ratio': to_count / from_count,
                    'asymmetrical': abs(to_count - from_count) > 5
                }

        return asymmetry

    def _detect_accusation_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Detect accusatory patterns"""
        accusations = [i for i in interactions if i.get('type') == 'accusatory']
        return accusations

    def _detect_defensive_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Detect defensive patterns"""
        defenses = [i for i in interactions if i.get('type') == 'defensive']
        return defenses

    def _analyze_network_structure(self, interactions: List[Dict], persons: List[str]) -> Dict[str, Any]:
        """Analyze network structure of interactions"""
        # Simplified network analysis
        return {
            'centrality': self._calculate_centrality(interactions, persons),
            'connectivity': len(set(i['speaker'] for i in interactions))
        }

    def _calculate_centrality(self, interactions: List[Dict], persons: List[str]) -> Dict[str, float]:
        """Calculate centrality of each person in interaction network"""
        centrality = {person: 0 for person in persons}

        for interaction in interactions:
            centrality[interaction['speaker']] += 1
            centrality[interaction['recipient']] += 0.5

        total = sum(centrality.values())
        if total > 0:
            centrality = {k: v/total for k, v in centrality.items()}

        return centrality

    def _assess_communication_balance(self, interactions: List[Dict], persons: List[str]) -> Dict[str, Any]:
        """Assess balance of communication"""
        speaker_counts = Counter(i['speaker'] for i in interactions)
        recipient_counts = Counter(i['recipient'] for i in interactions)

        imbalance_score = max(speaker_counts.values()) - min(speaker_counts.values()) if speaker_counts else 0

        return {
            'speaker_distribution': dict(speaker_counts),
            'recipient_distribution': dict(recipient_counts),
            'imbalance_score': imbalance_score,
            'is_balanced': imbalance_score < 5
        }

    def _assess_gaslighting_risk(self, total_indicators: int, message_count: int) -> str:
        """Assess gaslighting risk level"""
        if message_count == 0:
            return "low"

        indicator_ratio = total_indicators / message_count

        if indicator_ratio > 0.2:
            return "critical"
        elif indicator_ratio > 0.1:
            return "high"
        elif indicator_ratio > 0.05:
            return "moderate"

        return "low"

    def _identify_gaslighting_perpetrators(self, gaslighting_indicators: Dict) -> List[str]:
        """Identify likely gaslighting perpetrators"""
        perpetrators = defaultdict(int)

        for category, indicators in gaslighting_indicators.items():
            for indicator in indicators:
                perpetrators[indicator['speaker']] += 1

        return sorted(perpetrators.items(), key=lambda x: x[1], reverse=True)

    def _identify_gaslighting_victims(self, gaslighting_indicators: Dict, messages: List[Dict]) -> List[str]:
        """Identify likely gaslighting victims"""
        victims = defaultdict(int)

        for category, indicators in gaslighting_indicators.items():
            for indicator in indicators:
                # Look for defensive responses after gaslighting attempts
                msg_idx = indicator['message_index']
                if msg_idx + 1 < len(messages):
                    responder = messages[msg_idx + 1].get('sender', 'Unknown')
                    if responder != indicator['speaker']:
                        victims[responder] += 1

        return sorted(victims.items(), key=lambda x: x[1], reverse=True)

    def _analyze_power_dynamics(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """Analyze power dynamics in conversation"""
        person_names = {p['name']: p for p in persons}

        # Count control statements per person
        control_statements = defaultdict(int)

        control_words = ['should', 'must', 'have to', 'need to', 'you have to', 'you must', 'you should']

        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '').lower()

            for word in control_words:
                if word in text:
                    control_statements[sender] += 1

        # Identify power imbalance
        power_imbalance = False
        severity = 'none'

        if control_statements:
            max_controls = max(control_statements.values())
            if max_controls > len(messages) * 0.05:
                power_imbalance = True
                severity = 'severe' if max_controls > len(messages) * 0.1 else 'high'

        return {
            'control_statements': dict(control_statements),
            'power_imbalance': power_imbalance,
            'severity': severity,
            'dominant_person': max(control_statements, key=control_statements.get) if control_statements else None
        }

    def _analyze_emotional_patterns(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns per person"""
        emotional_patterns = {}

        for person in persons:
            person_name = person['name']
            person_messages = [m for m in messages if m.get('sender') == person_name]

            emotional_words = {
                'anger': ['angry', 'furious', 'hate', 'disgusting'],
                'sadness': ['sad', 'depressed', 'lonely', 'miserable'],
                'fear': ['afraid', 'scared', 'terrified', 'anxious'],
                'love': ['love', 'adore', 'cherish', 'beautiful']
            }

            emotion_scores = {emotion: 0 for emotion in emotional_words}

            for msg in person_messages:
                text = msg.get('text', '').lower()
                for emotion, words in emotional_words.items():
                    for word in words:
                        if word in text:
                            emotion_scores[emotion] += 1

            emotional_patterns[person_name] = emotion_scores

        return emotional_patterns

    def _assess_dependency_dynamics(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """Assess dependency dynamics in relationship"""
        dependency_words = ['need', 'depend on', 'can\'t live without', 'what would i do', 'i need you']
        dependency_counts = defaultdict(int)

        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            text = msg.get('text', '').lower()

            for word in dependency_words:
                if word in text:
                    dependency_counts[sender] += 1

        return {
            'dependency_counts': dict(dependency_counts),
            'has_dependency_dynamics': sum(dependency_counts.values()) > 0,
            'dependent_person': max(dependency_counts, key=dependency_counts.get) if dependency_counts else None
        }

    def _identify_control_patterns(self, messages: List[Dict], persons: List[Dict]) -> Dict[str, Any]:
        """Identify control patterns in conversation"""
        control_patterns = {
            'isolation_attempts': [],
            'emotional_control': [],
            'financial_control': [],
            'decision_control': []
        }

        isolation_phrases = ['don\'t talk to', 'don\'t see', 'can\'t hang out', 'don\'t spend time']
        emotional_phrases = ['you\'re too sensitive', 'you\'re crazy', 'you\'re overreacting']
        financial_phrases = ['spend money', 'how much did you spend', 'stop spending']
        decision_phrases = ['you can\'t', 'you shouldn\'t', 'i won\'t let you']

        for idx, msg in enumerate(messages):
            text = msg.get('text', '').lower()

            for phrase in isolation_phrases:
                if phrase in text:
                    control_patterns['isolation_attempts'].append({'index': idx, 'speaker': msg.get('sender')})

            for phrase in emotional_phrases:
                if phrase in text:
                    control_patterns['emotional_control'].append({'index': idx, 'speaker': msg.get('sender')})

            for phrase in financial_phrases:
                if phrase in text:
                    control_patterns['financial_control'].append({'index': idx, 'speaker': msg.get('sender')})

            for phrase in decision_phrases:
                if phrase in text:
                    control_patterns['decision_control'].append({'index': idx, 'speaker': msg.get('sender')})

        return control_patterns

    def _classify_relationship_type(self, power_analysis: Dict, control_patterns: Dict) -> str:
        """Classify relationship type based on analysis"""
        total_control = sum(len(v) for v in control_patterns.values())

        if power_analysis.get('power_imbalance') and total_control > 5:
            return "controlling"
        elif power_analysis.get('power_imbalance'):
            return "imbalanced"
        else:
            return "balanced"

    def _assess_relationship_quality(self, messages: List[Dict], power_analysis: Dict, control_patterns: Dict) -> str:
        """Assess overall relationship quality"""
        negative_words = ['hate', 'disgusting', 'stupid', 'idiot', 'useless']
        positive_words = ['love', 'appreciate', 'grateful', 'wonderful', 'caring']

        negative_count = sum(1 for msg in messages
                            for word in negative_words
                            if word in msg.get('text', '').lower())

        positive_count = sum(1 for msg in messages
                            for word in positive_words
                            if word in msg.get('text', '').lower())

        if negative_count > positive_count * 2:
            return "poor"
        elif power_analysis.get('power_imbalance'):
            return "unhealthy"
        else:
            return "fair"

    def _formulate_case(self, risk_assessment: Dict, person_identification: Dict,
                        relationship_analysis: Dict, gaslighting_detection: Dict) -> Dict[str, Any]:
        """Formulate comprehensive case summary"""
        return {
            'presentation': f"Conversation between {person_identification.get('total_speakers', 'multiple')} individuals",
            'primary_concerns': [
                risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown'),
                relationship_analysis.get('relationship_type', 'unknown'),
                gaslighting_detection.get('gaslighting_risk', 'low')
            ],
            'relationship_dynamics': relationship_analysis.get('relationship_type', 'unknown'),
            'power_structure': 'imbalanced' if relationship_analysis.get('power_imbalance') else 'balanced'
        }

    def _generate_resources(self, risk_level: str, gaslighting_detection: Dict) -> List[str]:
        """Generate resource recommendations"""
        resources = [
            "National Domestic Violence Hotline: 1-800-799-7233",
            "Crisis Text Line: Text HOME to 741741",
            "Psychology Today Therapist Finder: psychologytoday.com"
        ]

        if gaslighting_detection.get('gaslighting_risk') in ['high', 'critical']:
            resources.append("Gaslighting Support Groups: www.rainn.org")

        if risk_level in ['critical', 'high']:
            resources.append("Emergency Services: 911")

        return resources

    def _identify_follow_up_actions(self, risk_level: str, relationship_analysis: Dict,
                                     gaslighting_detection: Dict) -> List[str]:
        """Identify recommended follow-up actions"""
        actions = []

        if risk_level in ['critical', 'high']:
            actions.append("Schedule urgent follow-up appointment")
            actions.append("Assess for immediate safety concerns")

        if relationship_analysis.get('power_imbalance'):
            actions.append("Refer to relationship counselor")

        if gaslighting_detection.get('gaslighting_risk') in ['high', 'critical']:
            actions.append("Provide psychoeducation about gaslighting")
            actions.append("Refer to trauma-informed therapist")

        return actions

    def _summarize_findings(self, risk_assessment: Dict, person_identification: Dict,
                           relationship_analysis: Dict, gaslighting_detection: Dict) -> List[str]:
        """Summarize key findings from analysis"""
        findings = []

        risk_level = risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown')
        findings.append(f"Overall Risk Level: {risk_level}")

        findings.append(f"Speakers Identified: {person_identification.get('total_speakers', 'unknown')}")

        if relationship_analysis.get('power_imbalance'):
            findings.append(f"Power Imbalance: {relationship_analysis.get('power_imbalance_severity', 'unknown')}")

        if gaslighting_detection.get('gaslighting_risk') in ['high', 'critical']:
            findings.append(f"Gaslighting Indicators: {gaslighting_detection.get('total_indicators', 0)}")

        return findings

    def _generate_clinical_impressions(self, risk_assessment: Dict, relationship_analysis: Dict,
                                       gaslighting_detection: Dict) -> str:
        """Generate clinical impressions summary"""
        impressions = []

        risk_level = risk_assessment.get('overall_risk_assessment', {}).get('risk_level', 'unknown')
        impressions.append(f"Clinical presentation suggests {risk_level} level of concern.")

        if relationship_analysis.get('power_imbalance'):
            impressions.append("Significant power imbalance present in relationship.")

        if gaslighting_detection.get('gaslighting_risk') in ['high', 'critical']:
            impressions.append("Clear gaslighting patterns present in communication.")

        return " ".join(impressions)


if __name__ == "__main__":
    print("Person Analysis Module")
    print("Use with: from person_analyzer import PersonAnalyzer")
