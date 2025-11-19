"""
Cultural Context and Multilingual Analysis Module
Analyzes language detection, cultural idioms, slang/dialect, age-appropriate language,
cultural references, code-switching patterns, and formality norms.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """Container for language detection result"""
    language: str  # en, es, fr, de, etc.
    language_name: str  # English, Spanish, French, etc.
    confidence: float  # 0-1
    detected_text_ratio: float  # percentage of text detected
    alternative_languages: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class CulturalMarker:
    """Container for cultural element detected"""
    marker_type: str  # idiom, slang, cultural_reference, code_switch
    detected_item: str
    language: str
    cultural_origin: str
    matched_text: str
    position: int
    confidence: float


@dataclass
class CulturalAnalysis:
    """Complete cultural context analysis results"""
    primary_language: LanguageDetection = field(default_factory=lambda: LanguageDetection("unknown", "Unknown", 0.0, 0.0))
    secondary_languages: List[LanguageDetection] = field(default_factory=list)
    cultural_markers: List[CulturalMarker] = field(default_factory=list)
    idiom_count: int = 0
    slang_count: int = 0
    cultural_references: int = 0
    code_switching_instances: int = 0
    age_group: str = "unknown"  # teen, young_adult, adult, senior
    estimated_age_range: Tuple[int, int] = (0, 0)
    formality_level: str = "neutral"  # formal, neutral, casual, very_casual
    dialect_detected: str = "standard"  # standard, regional, non_native
    cultural_appropriateness: str = "appropriate"  # appropriate, potentially_offensive, offensive
    multilingual_fluency: str = "unknown"  # monolingual, bilingual, multilingual
    tone_cultural_fit: bool = True  # Does tone match cultural norms?
    recommendations: List[str] = field(default_factory=list)


class CulturalContextAnalyzer:
    """Analyzes cultural context and multilingual patterns in communication"""

    # Language indicators (common words/patterns in different languages)
    LANGUAGE_INDICATORS = {
        'en': ['the', 'and', 'to', 'you', 'is', 'that', 'in', 'for', 'a', 'of'],
        'es': ['el', 'la', 'y', 'de', 'que', 'en', 'un', 'una', 'por', 'con'],
        'fr': ['le', 'de', 'et', 'un', 'une', 'la', 'que', 'à', 'en', 'pour'],
        'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
        'it': ['il', 'di', 'e', 'in', 'da', 'lo', 'per', 'un', 'una', 'è'],
        'pt': ['o', 'de', 'e', 'a', 'para', 'em', 'um', 'uma', 'por', 'que'],
        'ru': ['в', 'и', 'на', 'что', 'я', 'не', 'с', 'он', 'она', 'это'],
        'zh': ['的', '一', '是', '在', '人', '了', '有', '和', '人', '这'],
        'ja': ['は', 'を', 'に', 'の', 'て', 'が', 'た', 'で', 'も', 'から'],
        'ar': ['في', 'هذا', 'من', 'أن', 'على', 'هو', 'الذي', 'إلى', 'وقد', 'ما'],
    }

    # English idioms
    ENGLISH_IDIOMS = [
        (r'\bbreak the ice\b', 'English'),
        (r'\bpiece of cake\b', 'English'),
        (r'\braining cats and dogs\b', 'English'),
        (r'\bspill the beans\b', 'English'),
        (r'\bundercovers cop\b', 'English'),
        (r'\blevel playing field\b', 'English'),
        (r'\bcall it a day\b', 'English'),
        (r'\bblow off steam\b', 'English'),
        (r'\bsave face\b', 'English'),
        (r'\bwearing your heart on your sleeve\b', 'English'),
        (r'\bunder the weather\b', 'English'),
        (r'\bis the cherry on top\b', 'English'),
    ]

    # Spanish idioms
    SPANISH_IDIOMS = [
        (r'\blluvia de ideas\b', 'Spanish'),
        (r'\ben la novena\b', 'Spanish'),
        (r'\btomar el pelo\b', 'Spanish'),
        (r'\blanzar la toalla\b', 'Spanish'),
    ]

    # Slang patterns (informal language)
    SLANG_PATTERNS = {
        'English': [
            (r'\b(gonna|wanna|gotta)\b', 'contracted_slang'),
            (r'\b(lol|omg|wtf|tbh)\b', 'acronym_slang'),
            (r'\b(hey|dude|bro|mate|guy)\b', 'informal_address'),
            (r'\byeah|nah\b', 'informal_agreement'),
            (r'\b(ain\'t|isn\'t|haven\'t)\b', 'informal_negation'),
            (r'\b(super|mega|totally|literally)\b', 'intensifier'),
            (r'\b(dunno|kinda|sorta)\b', 'informal_qualifier'),
        ],
        'Spanish': [
            (r'\b(tío|tía|boludo|che)\b', 'regional_address'),
            (r'\bwey|mande|órale\b', 'spanish_slang'),
        ],
    }

    # Age-related language patterns
    AGE_PATTERNS = {
        'teen': [
            r'\b(lol|omg|wtf|tbh|fml)\b',
            r'\b(so|like|totally|literally)\b.*\b(way|much|epic)\b',
            r'\b(my parents|my friends|school)\b',
            r'\b(hashtag|trending|viral)\b',
            r'\b(crush|dating|dating app)\b',
        ],
        'young_adult': [
            r'\b(job|career|college|university)\b',
            r'\b(apartment|rent|bills|student loans)\b',
            r'\b(dating|relationship|serious)\b',
            r'\b(work|boss|coworkers)\b',
        ],
        'adult': [
            r'\b(kids|children|family|marriage)\b',
            r'\b(house|mortgage|bills|insurance)\b',
            r'\b(career|promotion|leadership)\b',
            r'\b(retirement|savings|investment)\b',
        ],
        'senior': [
            r'\b(grandchildren|retirement|pension)\b',
            r'\b(health|doctor|medication)\b',
            r'\b(back in my day|when i was your age)\b',
            r'\b(remember when|years ago)\b',
        ],
    }

    # Formality indicators
    FORMAL_PATTERNS = [
        r'\b(sincerely|respectfully|cordially)\b',
        r'\b(regarding|pertaining to|in accordance)\b',
        r'\b(consequently|furthermore|nevertheless)\b',
        r'\bDear (Sir|Madam|Mr|Ms|Dr)\b',
        r'\b(we kindly request|we would appreciate)\b',
        r'\b(hereby|thus|thereof)\b',
    ]

    CASUAL_PATTERNS = [
        r'\b(hey|hi|yo|wassup|wussup)\b',
        r'\b(gonna|wanna|gotta)\b',
        r'\b(lol|haha|rofl)\b',
        r'\b(dude|man|bro|girl|guy)\b',
        r'\b(totally|super|really|so)\b',
        r'\b(ain\'t|dunno|sorta|kinda)\b',
    ]

    # Cultural references (examples)
    CULTURAL_REFERENCES = {
        'American': [
            r'\b(Hollywood|Silicon Valley|Super Bowl|Fourth of July)\b',
            r'\b(Trump|Biden|Congress|White House)\b',
            r'\b(Marvel|Netflix|Disney|Apple)\b',
        ],
        'British': [
            r'\b(Parliament|London|Westminster|Crown)\b',
            r'\b(BBC|The Guardian|Brexit)\b',
            r'\b(royals|the Queen|Prince)\b',
        ],
        'Anime/Japanese': [
            r'\b(anime|manga|kawaii|otaku)\b',
            r'\b(sushi|sakura|samurai|ninja)\b',
        ],
        'Indian': [
            r'\b(Bollywood|cricket|Taj Mahal|Diwali)\b',
            r'\b(curry|yoga|Hinduism|Buddhism)\b',
        ],
    }

    # Code-switching indicators
    CODE_SWITCH_PATTERNS = [
        r'(\bcommoncode_marker\b)',  # Placeholder for detection
        r'([\u0100-\u017F]|[\u0370-\u03FF]|[\u0600-\u06FF])',  # Non-Latin scripts
    ]

    def __init__(self):
        """Initialize cultural context analyzer"""
        self.compiled_patterns = self._compile_patterns()
        self.language_model: Dict[str, List[str]] = self.LANGUAGE_INDICATORS

    def _compile_patterns(self) -> Dict[str, List]:
        """Compile regex patterns for efficiency"""
        return {
            'english_idioms': [re.compile(p, re.IGNORECASE) for p, _ in self.ENGLISH_IDIOMS],
            'spanish_idioms': [re.compile(p, re.IGNORECASE) for p, _ in self.SPANISH_IDIOMS],
            'formal': [re.compile(p, re.IGNORECASE) for p in self.FORMAL_PATTERNS],
            'casual': [re.compile(p, re.IGNORECASE) for p in self.CASUAL_PATTERNS],
        }

    def analyze_cultural_context(self, text: str) -> CulturalAnalysis:
        """Analyze cultural and multilingual context of text

        Args:
            text: Text to analyze

        Returns:
            CulturalAnalysis: Complete analysis
        """
        analysis = CulturalAnalysis()

        # Detect language
        primary_language = self._detect_language(text)
        analysis.primary_language = primary_language

        # Detect secondary languages (for code-switching)
        secondary = self._detect_code_switching(text)
        analysis.secondary_languages = secondary

        # Update multilingual fluency
        if len(secondary) > 1:
            analysis.multilingual_fluency = "multilingual"
        elif len(secondary) == 1:
            analysis.multilingual_fluency = "bilingual"
        else:
            analysis.multilingual_fluency = "monolingual"

        # Detect cultural markers
        markers = self._detect_cultural_markers(text)
        analysis.cultural_markers = markers

        # Count different types
        analysis.idiom_count = sum(1 for m in markers if m.marker_type == 'idiom')
        analysis.slang_count = sum(1 for m in markers if m.marker_type == 'slang')
        analysis.cultural_references = sum(1 for m in markers if m.marker_type == 'cultural_reference')
        analysis.code_switching_instances = len(secondary)

        # Detect age group
        age_group, age_range = self._detect_age_group(text)
        analysis.age_group = age_group
        analysis.estimated_age_range = age_range

        # Detect formality level
        formality = self._detect_formality(text)
        analysis.formality_level = formality

        # Detect dialect
        dialect = self._detect_dialect(text, primary_language.language)
        analysis.dialect_detected = dialect

        # Assess cultural appropriateness
        appropriateness = self._assess_cultural_appropriateness(text, markers)
        analysis.cultural_appropriateness = appropriateness

        # Check tone cultural fit
        analysis.tone_cultural_fit = self._check_tone_cultural_fit(
            text, analysis.formality_level, analysis.age_group
        )

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        return analysis

    def _detect_language(self, text: str) -> LanguageDetection:
        """Detect primary language of text"""
        text_lower = text.lower()
        words = text_lower.split()

        language_scores = defaultdict(float)

        for lang, keywords in self.language_model.items():
            matches = sum(1 for word in words if word in keywords)
            if matches > 0:
                language_scores[lang] = matches / len(words)

        if not language_scores:
            return LanguageDetection("unknown", "Unknown", 0.0, 0.0)

        # Get top language
        top_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[top_lang]

        lang_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
        }

        # Get alternatives
        alternatives = [
            (lang, score) for lang, score in sorted(
                language_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[1:3]
        ]

        return LanguageDetection(
            language=top_lang,
            language_name=lang_names.get(top_lang, top_lang),
            confidence=min(1.0, confidence),
            detected_text_ratio=confidence,
            alternative_languages=alternatives
        )

    def _detect_code_switching(self, text: str) -> List[LanguageDetection]:
        """Detect code-switching (mixing multiple languages)"""
        # Split text into sentences/chunks
        chunks = text.split('. ')
        secondary_languages = []

        for chunk in chunks:
            if len(chunk.strip()) < 5:
                continue

            lang = self._detect_language(chunk)
            if lang.language != 'unknown' and lang.language not in [l.language for l in secondary_languages]:
                secondary_languages.append(lang)

        # Return only secondary languages (remove duplicates of primary)
        return secondary_languages[1:] if len(secondary_languages) > 1 else []

    def _detect_cultural_markers(self, text: str) -> List[CulturalMarker]:
        """Detect idioms, slang, and cultural references"""
        markers = []

        # Detect idioms
        for pattern, lang in self.ENGLISH_IDIOMS + self.SPANISH_IDIOMS:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                marker = CulturalMarker(
                    marker_type='idiom',
                    detected_item=match.group(),
                    language=lang,
                    cultural_origin=lang,
                    matched_text=match.group(),
                    position=match.start(),
                    confidence=0.85
                )
                markers.append(marker)

        # Detect slang
        for lang, slang_list in self.SLANG_PATTERNS.items():
            for pattern, slang_type in slang_list:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    marker = CulturalMarker(
                        marker_type='slang',
                        detected_item=match.group(),
                        language=lang,
                        cultural_origin=slang_type,
                        matched_text=match.group(),
                        position=match.start(),
                        confidence=0.8
                    )
                    markers.append(marker)

        # Detect cultural references
        for culture, references in self.CULTURAL_REFERENCES.items():
            for pattern in references:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    marker = CulturalMarker(
                        marker_type='cultural_reference',
                        detected_item=match.group(),
                        language='multiple',
                        cultural_origin=culture,
                        matched_text=match.group(),
                        position=match.start(),
                        confidence=0.8
                    )
                    markers.append(marker)

        return markers

    def _detect_age_group(self, text: str) -> Tuple[str, Tuple[int, int]]:
        """Detect likely age group based on language patterns"""
        scores = defaultdict(int)

        for age_group, patterns in self.AGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[age_group] += 1

        if not scores:
            return "unknown", (0, 0)

        detected_group = max(scores, key=scores.get)

        age_ranges = {
            'teen': (13, 19),
            'young_adult': (20, 35),
            'adult': (36, 55),
            'senior': (56, 120),
        }

        return detected_group, age_ranges.get(detected_group, (0, 0))

    def _detect_formality(self, text: str) -> str:
        """Detect formality level of communication"""
        formal_score = sum(1 for p in self.compiled_patterns['formal'] for _ in p.finditer(text))
        casual_score = sum(1 for p in self.compiled_patterns['casual'] for _ in p.finditer(text))

        total = formal_score + casual_score

        if total == 0:
            return "neutral"

        if formal_score > casual_score * 2:
            return "formal"
        elif casual_score > formal_score * 2:
            return "very_casual"
        elif casual_score > formal_score:
            return "casual"
        else:
            return "neutral"

    def _detect_dialect(self, text: str, primary_language: str) -> str:
        """Detect dialect or regional variation"""
        # Look for regional markers
        regional_markers = {
            'American': [r'\b(y\'all|ain\'t|gonna|wanna)\b', r'\b(color|favorite|center)\b'],
            'British': [r'\b(bloke|mate|loo|lorry)\b', r'\b(colour|favourite|centre)\b'],
            'Australian': [r'\b(mate|g\'day|arvo|brekkie)\b'],
            'Indian': [r'\b(da|la|isn\'t it|no)\b.*$'],
        }

        for dialect, patterns in regional_markers.items():
            matches = sum(1 for p in patterns for _ in re.finditer(p, text, re.IGNORECASE | re.MULTILINE))
            if matches > 0:
                return dialect

        # Check for non-native English markers
        if primary_language == 'en':
            non_native_markers = [
                r'\b(the)\s+(school|university)',
                r'\bdo\s+not\s+(like|want)',
            ]
            if any(re.search(p, text, re.IGNORECASE) for p in non_native_markers):
                return "non_native"

        return "standard"

    def _assess_cultural_appropriateness(
        self,
        text: str,
        markers: List[CulturalMarker]
    ) -> str:
        """Assess cultural appropriateness of communication"""
        # Check for potentially offensive terms
        offensive_terms = [
            r'\b(crap|damn|hell)\b',  # Mild
            r'\b(f\*ck|ass|bitch)\b',  # Strong
            r'\b(n-word|slur|derogatory)\b',  # Highly offensive
        ]

        strong_count = 0
        mild_count = 0

        for term in offensive_terms:
            if re.search(term, text, re.IGNORECASE):
                if term == offensive_terms[-1]:
                    return "offensive"
                else:
                    strong_count += 1

        if strong_count > 2:
            return "potentially_offensive"
        elif strong_count > 0:
            return "potentially_offensive"

        return "appropriate"

    def _check_tone_cultural_fit(
        self,
        text: str,
        formality: str,
        age_group: str
    ) -> bool:
        """Check if tone matches expected cultural norms"""
        # Very formal but teen language = misfit
        if formality == "formal" and age_group == "teen":
            if sum(1 for p in self.compiled_patterns['casual'] for _ in p.finditer(text)) > 3:
                return False

        # Very casual but formal context = misfit
        if formality == "very_casual" and age_group == "senior":
            return False

        return True

    def _generate_recommendations(self, analysis: CulturalAnalysis) -> List[str]:
        """Generate recommendations based on cultural analysis"""
        recommendations = []

        if analysis.cultural_appropriateness == "offensive":
            recommendations.append("CRITICAL: Offensive language detected. Review for appropriateness.")

        if analysis.cultural_appropriateness == "potentially_offensive":
            recommendations.append("Potentially offensive language detected. Consider more inclusive wording.")

        if analysis.code_switching_instances > 2:
            recommendations.append("Multiple language switching detected. Ensure context is clear for all audiences.")

        if not analysis.tone_cultural_fit:
            recommendations.append(f"Tone mismatch detected for age group: {analysis.age_group}")

        if analysis.dialect_detected == "non_native" and analysis.multilingual_fluency == "monolingual":
            recommendations.append("Non-native English patterns detected. Consider clarity improvements.")

        if analysis.formality_level == "very_casual" and analysis.primary_language.detected_text_ratio > 0.7:
            recommendations.append("Very casual tone detected in formal context. Consider more professional language.")

        return recommendations
