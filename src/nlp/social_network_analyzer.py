"""
Social Network and Support System Analysis Module
Analyzes third-party mentions, isolation from others, social support network mapping,
friend/family reference patterns, social activities, loneliness indicators, and
network expansion/contraction trends.
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SocialContact:
    """Container for tracked social contact"""
    name: str
    relationship: str  # friend, family, coworker, acquaintance
    mention_count: int = 0
    mention_sentiment: float = 0.0  # -1 to 1
    last_mentioned: str = ""
    context_positive: int = 0
    context_negative: int = 0
    context_neutral: int = 0


@dataclass
class SocialNetworkMarker:
    """Container for social network pattern match"""
    marker_type: str  # mention, activity, isolation, support, loneliness
    detected_item: str
    relationship_type: Optional[str] = None
    sentiment: float = 0.0
    matched_text: str = ""
    position: int = 0
    context: str = ""


@dataclass
class SocialNetworkAnalysis:
    """Complete social network analysis results"""
    social_contacts: List[SocialContact] = field(default_factory=list)
    markers: List[SocialNetworkMarker] = field(default_factory=list)
    network_size_estimate: int = 0  # Estimated number of contacts
    isolation_score: float = 0.0  # 0-1, higher = more isolated
    social_support_quality: str = "unknown"  # strong, moderate, weak, insufficient
    friend_family_ratio: float = 0.5  # Friend/total ratio
    social_activities_mentioned: int = 0
    loneliness_indicators: int = 0
    family_connection_level: str = "unknown"  # close, normal, distant, estranged
    friend_connection_level: str = "unknown"  # close, normal, distant, none
    network_trend: str = "stable"  # expanding, contracting, stable
    contact_diversity: float = 0.0  # 0-1, diversity of relationship types
    support_system_strength: float = 0.0  # 0-1
    isolation_risk: str = "low"  # low, moderate, high, critical
    loneliness_risk: str = "low"  # low, moderate, high, critical
    social_health: str = "unknown"  # healthy, at-risk, critical
    vulnerable_to_manipulation: bool = False  # Is isolation reducing critical thinking?
    recommendations: List[str] = field(default_factory=list)


class SocialNetworkAnalyzer:
    """Analyzes social networks and support systems"""

    # Third-party mention patterns
    MENTION_PATTERNS = {
        'friend': [
            r'\b(my friend|my best friend|my best mate|my bff|my bestie)\b',
            r'\b(friend of mine|friends with)\b',
            r'\b(buddy|pal|mate|homie)\b',
            r'\bwe (went|hung out|partied|played)\b',
            r'\bwith my (friends|crew|squad|gang)\b',
        ],
        'family': [
            r'\b(my mom|my dad|my mother|my father|my parents)\b',
            r'\b(my brother|my sister|my sibling)\b',
            r'\b(my grandpa|my grandma|my grandfather|my grandmother)\b',
            r'\b(my aunt|my uncle|my cousin)\b',
            r'\b(my son|my daughter|my kids|my children)\b',
            r'\b(my spouse|my husband|my wife|my partner)\b',
            r'\b(my family|our family)\b',
        ],
        'coworker': [
            r'\b(my coworker|my colleague|my boss|my manager)\b',
            r'\b(at work|with my team|office|workplace)\b',
            r'\b(colleague of mine|work friend)\b',
        ],
        'acquaintance': [
            r'\b(someone i know|acquaintance|person i know)\b',
            r'\b(guy|girl|person)\b.*\b(from|i met)\b',
        ],
    }

    # Social activities
    SOCIAL_ACTIVITY_PATTERNS = [
        r'\b(went to|attended|went on)\b.*\b(party|concert|festival|event|gathering)\b',
        r'\b(hung out|spent time|got together)\b.*\bwith\b',
        r'\b(dinner|lunch|breakfast)\b.*\b(with|together)\b',
        r'\b(weekend|holiday|vacation)\b.*\b(with|family|friends)\b',
        r'\b(movie|games|sports|hiking|beach)\b.*\b(with|together)\b',
        r'\b(date|dating)\b',
        r'\b(club|bar|restaurant)\b',
        r'\b(traveled|trip|journey)\b.*\bwith\b',
        r'\b(wedding|birthday|celebration)\b',
        r'\b(church|mosque|synagogue|temple)\b',
    ]

    # Isolation indicators
    ISOLATION_PATTERNS = [
        r"\bi (don't have|don't see|can't see)\b.*\b(friends|people|anyone)\b",
        r"\b(alone|by myself|on my own)\b.*\b(all the time|always|most of the time)\b",
        r"\bno (one|body) (talks to|cares about|knows about|understands)\b.*\bme\b",
        r"\bi (don't go|can't go)\b.*\b(out|anywhere|to events)\b",
        r"\b(isolated|lonely|cut off|secluded)\b",
        r"\bi (stopped|stopped hanging out|lost touch)\b.*\bwith\b",
        r"\b(pushed away|distanced from) my\b.*\b(friends|family)\b",
        r"\bi'm (stuck|trapped|alone)\b",
        r"\b(nobody calls|nobody visits)\b.*\bme\b",
    ]

    # Support seeking patterns
    SUPPORT_PATTERNS = [
        r"\bi (can|could) talk to\b.*\b(about|with)\b",
        r"\b(someone|people) (listens|understand|cares|helps)\b",
        r"\b(my support system|support network|safety net)\b",
        r"\b(i can count on|i can rely on)\b",
        r"\b(there for me|has my back|supports me)\b",
        r"\b(called|texted|reached out)\b.*\b(to|for help|for support)\b",
    ]

    # Loneliness indicators
    LONELINESS_PATTERNS = [
        r"\b(feel|feeling)\b.*\b(lonely|isolated|alone|left out)\b",
        r"\b(nobody|no one) (wants|likes|cares about)\b.*\bme\b",
        r"\bno (one|body) (understands|gets)\b.*\bme\b",
        r"\b(unwanted|unimportant|invisible)\b",
        r"\bi (miss|miss having)\b.*\b(friends|family|connection|people)\b",
        r"\b(friendless|love-less|unloved)\b",
        r"\bi (sit|stay)\b.*\b(alone|by myself)\b.*\b(all day|all night|every day)\b",
        r"\b(nobody visits|nobody calls|nobody texts)\b",
        r"\b(so alone|so lonely|so isolated)\b",
    ]

    # Network contraction patterns
    CONTRACTION_PATTERNS = [
        r"\bi (stopped|quit|lost|ended)\b.*\b(friendship|contact|touch|communication)\b",
        r"\bwe (don't|stopped)\b.*\b(talk|speak|hang out|see each other)\b",
        r"\b(cut off|cut ties with|distance from)\b",
        r"\b(people (avoid|ignore) me|friendship (ended|faded))\b",
        r"\bi (let go|moved on|left behind)\b",
    ]

    # Network expansion patterns
    EXPANSION_PATTERNS = [
        r"\bi (met|made|found)\b.*\b(new friend|new people|new connection)\b",
        r"\bstarted (going to|attending|joining)\b",
        r"\b(joined|signed up for|started)\b.*\b(group|club|class|team)\b",
        r"\bmet someone (new|interesting|amazing)\b",
        r"\b(new friendship|new relationship|new circle)\b",
    ]

    def __init__(self):
        """Initialize social network analyzer"""
        self.compiled_patterns = self._compile_patterns()
        self.contact_database: Dict[str, SocialContact] = {}

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for efficiency"""
        patterns = {}

        for rel_type, pattern_list in self.MENTION_PATTERNS.items():
            patterns[f'mention_{rel_type}'] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]

        patterns['activities'] = [
            re.compile(p, re.IGNORECASE) for p in self.SOCIAL_ACTIVITY_PATTERNS
        ]
        patterns['isolation'] = [
            re.compile(p, re.IGNORECASE) for p in self.ISOLATION_PATTERNS
        ]
        patterns['support'] = [
            re.compile(p, re.IGNORECASE) for p in self.SUPPORT_PATTERNS
        ]
        patterns['loneliness'] = [
            re.compile(p, re.IGNORECASE) for p in self.LONELINESS_PATTERNS
        ]
        patterns['contraction'] = [
            re.compile(p, re.IGNORECASE) for p in self.CONTRACTION_PATTERNS
        ]
        patterns['expansion'] = [
            re.compile(p, re.IGNORECASE) for p in self.EXPANSION_PATTERNS
        ]

        return patterns

    def analyze_social_network(self, text: str, speaker_name: str = "unknown") -> SocialNetworkAnalysis:
        """Analyze social network from text

        Args:
            text: Text to analyze
            speaker_name: Name of the speaker

        Returns:
            SocialNetworkAnalysis: Complete analysis
        """
        analysis = SocialNetworkAnalysis()

        # Detect social contacts mentioned
        contacts = self._detect_social_contacts(text)
        analysis.social_contacts = contacts
        analysis.network_size_estimate = len(set(c.name for c in contacts))

        # Detect social activities
        activities = sum(1 for p in self.compiled_patterns['activities'] for _ in p.finditer(text))
        analysis.social_activities_mentioned = activities

        # Calculate isolation score
        isolation_score = self._calculate_isolation_score(text)
        analysis.isolation_score = isolation_score

        # Detect loneliness indicators
        loneliness_count = sum(1 for p in self.compiled_patterns['loneliness'] for _ in p.finditer(text))
        analysis.loneliness_indicators = loneliness_count

        # Analyze relationship types
        friend_count = sum(1 for c in contacts if c.relationship == 'friend')
        family_count = sum(1 for c in contacts if c.relationship == 'family')
        total_contacts = len(contacts)

        if total_contacts > 0:
            analysis.friend_family_ratio = friend_count / total_contacts
        else:
            analysis.friend_family_ratio = 0.5

        # Assess connection levels
        analysis.family_connection_level = self._assess_connection_level(family_count, text, 'family')
        analysis.friend_connection_level = self._assess_connection_level(friend_count, text, 'friend')

        # Detect network trends
        contraction_count = sum(1 for p in self.compiled_patterns['contraction'] for _ in p.finditer(text))
        expansion_count = sum(1 for p in self.compiled_patterns['expansion'] for _ in p.finditer(text))

        if expansion_count > contraction_count:
            analysis.network_trend = "expanding"
        elif contraction_count > expansion_count:
            analysis.network_trend = "contracting"
        else:
            analysis.network_trend = "stable"

        # Calculate contact diversity
        if contacts:
            relationship_types = Counter(c.relationship for c in contacts)
            unique_types = len(relationship_types)
            max_possible = len(self.MENTION_PATTERNS)
            analysis.contact_diversity = unique_types / max_possible
        else:
            analysis.contact_diversity = 0.0

        # Calculate support system strength
        support_mentions = sum(1 for p in self.compiled_patterns['support'] for _ in p.finditer(text))
        analysis.support_system_strength = min(1.0, support_mentions / max(1, total_contacts))

        # Assess social support quality
        if total_contacts >= 3 and support_mentions > 0:
            analysis.social_support_quality = "strong"
        elif total_contacts >= 2 or support_mentions > 0:
            analysis.social_support_quality = "moderate"
        elif total_contacts >= 1:
            analysis.social_support_quality = "weak"
        else:
            analysis.social_support_quality = "insufficient"

        # Assess isolation risk
        if analysis.isolation_score > 0.7:
            analysis.isolation_risk = "critical"
        elif analysis.isolation_score > 0.5:
            analysis.isolation_risk = "high"
        elif analysis.isolation_score > 0.3:
            analysis.isolation_risk = "moderate"
        else:
            analysis.isolation_risk = "low"

        # Assess loneliness risk
        if loneliness_count > 5:
            analysis.loneliness_risk = "critical"
        elif loneliness_count > 3:
            analysis.loneliness_risk = "high"
        elif loneliness_count > 1:
            analysis.loneliness_risk = "moderate"
        else:
            analysis.loneliness_risk = "low"

        # Overall social health
        analysis.social_health = self._assess_social_health(analysis)

        # Check vulnerability to manipulation
        analysis.vulnerable_to_manipulation = self._check_manipulation_vulnerability(analysis)

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        return analysis

    def _detect_social_contacts(self, text: str) -> List[SocialContact]:
        """Detect and track social contacts mentioned"""
        contacts = []
        seen_names = set()

        for rel_type in self.MENTION_PATTERNS.keys():
            pattern_key = f'mention_{rel_type}'
            if pattern_key not in self.compiled_patterns:
                continue

            for pattern in self.compiled_patterns[pattern_key]:
                for match in pattern.finditer(text):
                    matched_text = match.group()

                    # Extract potential name (simple heuristic)
                    # This is a simplified version - real implementation would be more sophisticated
                    name = self._extract_contact_name(matched_text, rel_type)

                    if name and name not in seen_names:
                        contact = SocialContact(
                            name=name,
                            relationship=rel_type,
                            mention_count=1,
                            last_mentioned=matched_text
                        )
                        contacts.append(contact)
                        seen_names.add(name)

                        # Store in database
                        self.contact_database[name] = contact
                    elif name in seen_names:
                        # Update existing contact
                        for c in contacts:
                            if c.name == name:
                                c.mention_count += 1
                                break

        return contacts

    def _extract_contact_name(self, matched_text: str, rel_type: str) -> Optional[str]:
        """Extract contact name from matched text"""
        # Very simplified - real implementation would use NER
        # This is a placeholder for proper name extraction

        # Look for capitalized words after the pattern
        words = matched_text.split()

        # Common name patterns
        if 'name' in matched_text.lower():
            # e.g., "my friend John" -> John
            for i, word in enumerate(words):
                if i > 0 and word[0].isupper():
                    return word

        # Default: use relationship type as generic name
        return f"Contact ({rel_type})"

    def _calculate_isolation_score(self, text: str) -> float:
        """Calculate isolation score based on patterns"""
        isolation_matches = sum(1 for p in self.compiled_patterns['isolation'] for _ in p.finditer(text))
        support_matches = sum(1 for p in self.compiled_patterns['support'] for _ in p.finditer(text))

        # Normalize by text length
        text_length = len(text.split())
        if text_length == 0:
            return 0.0

        # Higher isolation patterns + lower support = higher isolation score
        isolation_density = isolation_matches / text_length
        support_density = support_matches / text_length

        isolation_score = max(0.0, isolation_density - support_density)
        return min(1.0, isolation_score * 10)  # Scale to 0-1 range

    def _assess_connection_level(self, count: int, text: str, conn_type: str) -> str:
        """Assess level of connection in a category"""
        if count == 0:
            return "none"

        # Look for sentiment indicators
        positive_terms = [
            r'\b(love|wonderful|amazing|great|good|best|cherish|treasure)\b',
            r'\bso (close|tight|connected|bonded)\b',
        ]

        negative_terms = [
            r'\b(hate|terrible|awful|horrible|regret|resent)\b',
            r'\b(distant|cold|toxic|abusive|controlling)\b',
        ]

        positive_count = sum(1 for p in positive_terms for _ in re.finditer(p, text, re.IGNORECASE))
        negative_count = sum(1 for p in negative_terms for _ in re.finditer(p, text, re.IGNORECASE))

        if positive_count > negative_count:
            return "close"
        elif negative_count > positive_count:
            return "estranged"
        else:
            return "normal" if count > 1 else "distant"

    def _assess_social_health(self, analysis: SocialNetworkAnalysis) -> str:
        """Assess overall social health"""
        health_factors = []

        # Network size
        if analysis.network_size_estimate >= 5:
            health_factors.append(1)
        elif analysis.network_size_estimate >= 3:
            health_factors.append(0.7)
        else:
            health_factors.append(0.3)

        # Support quality
        if analysis.social_support_quality == "strong":
            health_factors.append(1)
        elif analysis.social_support_quality == "moderate":
            health_factors.append(0.6)
        else:
            health_factors.append(0.2)

        # Isolation level
        health_factors.append(1 - analysis.isolation_score)

        # Loneliness
        loneliness_factor = max(0, 1 - (analysis.loneliness_indicators / 10))
        health_factors.append(loneliness_factor)

        # Network trend
        if analysis.network_trend == "expanding":
            health_factors.append(1)
        elif analysis.network_trend == "stable":
            health_factors.append(0.7)
        else:
            health_factors.append(0.4)

        average_health = statistics.mean(health_factors)

        if average_health >= 0.75:
            return "healthy"
        elif average_health >= 0.5:
            return "at-risk"
        else:
            return "critical"

    def _check_manipulation_vulnerability(self, analysis: SocialNetworkAnalysis) -> bool:
        """Check if isolation increases vulnerability to manipulation"""
        # High isolation + weak support + no diverse contacts = vulnerable
        if (analysis.isolation_score > 0.6 and
            analysis.social_support_quality in ["weak", "insufficient"] and
            analysis.contact_diversity < 0.3):
            return True

        return False

    def _generate_recommendations(self, analysis: SocialNetworkAnalysis) -> List[str]:
        """Generate recommendations based on social network analysis"""
        recommendations = []

        if analysis.isolation_risk == "critical":
            recommendations.append("CRITICAL: Severe isolation detected. Seek professional support immediately.")
            recommendations.append("Consider reaching out to family, friends, or counseling services.")

        if analysis.loneliness_risk == "critical":
            recommendations.append("CRITICAL: Severe loneliness indicators present.")
            recommendations.append("Connect with support groups or mental health professional.")

        if analysis.social_support_quality == "insufficient":
            recommendations.append("Develop and strengthen support network.")
            recommendations.append("Engage in community activities or groups matching your interests.")

        if analysis.network_trend == "contracting":
            recommendations.append("Social network appears to be shrinking.")
            recommendations.append("Initiate contact with existing friends or make new connections.")

        if analysis.vulnerable_to_manipulation:
            recommendations.append("ALERT: High isolation with weak support network.")
            recommendations.append("Be cautious of relationships offering immediate deep connection.")
            recommendations.append("Maintain diverse social connections for perspective.")

        if analysis.contact_diversity < 0.3:
            recommendations.append("Expand social circle to include diverse relationship types.")
            recommendations.append("Balance friendships with family, coworkers, and community.")

        return recommendations

    def analyze_network_over_time(
        self,
        messages: List[str]
    ) -> Dict[str, Any]:
        """Analyze how social network changes over multiple messages

        Args:
            messages: List of messages over time

        Returns:
            Dict with trend information
        """
        analyses = [self.analyze_social_network(msg) for msg in messages]

        if not analyses:
            return {
                'trend': 'unknown',
                'network_size_trend': 'stable',
                'isolation_trend': 'stable',
                'average_network_size': 0,
                'average_isolation': 0.0,
            }

        # Calculate trends
        network_sizes = [a.network_size_estimate for a in analyses]
        isolation_scores = [a.isolation_score for a in analyses]

        avg_network = statistics.mean(network_sizes) if network_sizes else 0
        avg_isolation = statistics.mean(isolation_scores) if isolation_scores else 0.0

        # Network size trend
        if len(network_sizes) > 1:
            network_trend = "expanding" if network_sizes[-1] > network_sizes[0] else "contracting"
        else:
            network_trend = "stable"

        # Isolation trend
        if len(isolation_scores) > 1:
            isolation_trend = "increasing" if isolation_scores[-1] > isolation_scores[0] else "decreasing"
        else:
            isolation_trend = "stable"

        return {
            'network_size_trend': network_trend,
            'isolation_trend': isolation_trend,
            'average_network_size': avg_network,
            'average_isolation': avg_isolation,
            'volatility': statistics.stdev(network_sizes) if len(network_sizes) > 1 else 0.0,
            'analyses_count': len(analyses),
        }

    def map_relationship_network(self, text: str) -> Dict[str, Any]:
        """Create a map of relationships mentioned

        Args:
            text: Text to analyze

        Returns:
            Dict representing relationship network
        """
        analysis = self.analyze_social_network(text)

        network_map = {
            'nodes': [],
            'edges': [],
            'statistics': {
                'total_contacts': len(analysis.social_contacts),
                'friends': sum(1 for c in analysis.social_contacts if c.relationship == 'friend'),
                'family': sum(1 for c in analysis.social_contacts if c.relationship == 'family'),
                'coworkers': sum(1 for c in analysis.social_contacts if c.relationship == 'coworker'),
                'isolation_score': analysis.isolation_score,
                'support_strength': analysis.support_system_strength,
            }
        }

        # Add nodes (contacts)
        for contact in analysis.social_contacts:
            network_map['nodes'].append({
                'id': contact.name,
                'relationship': contact.relationship,
                'mentions': contact.mention_count,
                'sentiment': contact.mention_sentiment,
            })

        # For edges, we'd normally detect co-mentions or interactions
        # Simplified version here
        network_map['edges'] = []

        return network_map
