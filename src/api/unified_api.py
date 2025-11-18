"""
Unified API Module for Message Processor
Merges ppl_int FastAPI features with Flask webapp
Provides RESTful endpoints for:
- Person CRUD operations
- Interaction tracking
- Relationship timeline analysis
- Risk assessment
- Real-time WebSocket updates
"""

from flask import Blueprint, request, jsonify, session
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid
import logging
from functools import wraps

logger = logging.getLogger(__name__)


# ==========================================
# Data Models / Schemas
# ==========================================

class PersonProfile:
    """Person entity with extended profile"""

    def __init__(self, person_id: str, name: str, phone: Optional[str] = None,
                 email: Optional[str] = None, metadata: Optional[Dict] = None):
        self.id = person_id
        self.name = name
        self.phone = phone
        self.email = email
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.interaction_count = 0
        self.risk_level = 'low'
        self.last_interaction = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'email': self.email,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'interaction_count': self.interaction_count,
            'risk_level': self.risk_level,
            'last_interaction': self.last_interaction
        }

    @staticmethod
    def from_dict(data: Dict) -> 'PersonProfile':
        """Create from dictionary"""
        p = PersonProfile(
            person_id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            phone=data.get('phone'),
            email=data.get('email'),
            metadata=data.get('metadata', {})
        )
        p.interaction_count = data.get('interaction_count', 0)
        p.risk_level = data.get('risk_level', 'low')
        p.last_interaction = data.get('last_interaction')
        p.created_at = data.get('created_at', p.created_at)
        p.updated_at = data.get('updated_at', p.updated_at)
        return p


class Interaction:
    """Interaction record between two persons"""

    def __init__(self, interaction_id: str, person1_id: str, person2_id: str,
                 interaction_type: str, content: str, timestamp: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        self.id = interaction_id
        self.person1_id = person1_id
        self.person2_id = person2_id
        self.interaction_type = interaction_type  # message, call, meeting, etc.
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
        self.sentiment = 0.0
        self.risk_score = 0.0
        self.flags = []

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'person1_id': self.person1_id,
            'person2_id': self.person2_id,
            'interaction_type': self.interaction_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'sentiment': self.sentiment,
            'risk_score': self.risk_score,
            'flags': self.flags
        }


class RelationshipTimeline:
    """Timeline of interactions between two persons"""

    def __init__(self, person1_id: str, person2_id: str):
        self.person1_id = person1_id
        self.person2_id = person2_id
        self.interactions: List[Interaction] = []
        self.first_interaction = None
        self.last_interaction = None
        self.total_interactions = 0
        self.relationship_status = 'new'
        self.overall_risk = 'low'
        self.timeline_summary = {}

    def add_interaction(self, interaction: Interaction):
        """Add interaction to timeline"""
        self.interactions.append(interaction)
        self.total_interactions += 1

        if not self.first_interaction:
            self.first_interaction = interaction.timestamp
        self.last_interaction = interaction.timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'person1_id': self.person1_id,
            'person2_id': self.person2_id,
            'interactions': [i.to_dict() for i in self.interactions],
            'first_interaction': self.first_interaction,
            'last_interaction': self.last_interaction,
            'total_interactions': self.total_interactions,
            'relationship_status': self.relationship_status,
            'overall_risk': self.overall_risk,
            'timeline_summary': self.timeline_summary
        }


class RiskAssessment:
    """Risk assessment for a person"""

    def __init__(self, person_id: str, assessment_type: str = 'comprehensive'):
        self.person_id = person_id
        self.assessment_type = assessment_type
        self.timestamp = datetime.now().isoformat()
        self.grooming_risk = 0.0
        self.manipulation_risk = 0.0
        self.deception_risk = 0.0
        self.hostility_risk = 0.0
        self.escalation_risk = 0.0
        self.overall_risk = 0.0
        self.risk_level = 'low'
        self.primary_concerns = []
        self.behavioral_indicators = {}
        self.recommendations = []
        self.confidence = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'person_id': self.person_id,
            'assessment_type': self.assessment_type,
            'timestamp': self.timestamp,
            'grooming_risk': self.grooming_risk,
            'manipulation_risk': self.manipulation_risk,
            'deception_risk': self.deception_risk,
            'hostility_risk': self.hostility_risk,
            'escalation_risk': self.escalation_risk,
            'overall_risk': self.overall_risk,
            'risk_level': self.risk_level,
            'primary_concerns': self.primary_concerns,
            'behavioral_indicators': self.behavioral_indicators,
            'recommendations': self.recommendations,
            'confidence': self.confidence
        }


# ==========================================
# Manager Classes
# ==========================================

class PersonManager:
    """Manage person CRUD operations"""

    def __init__(self, db, cache):
        """
        Initialize PersonManager

        Args:
            db: Database adapter (PostgreSQLAdapter)
            cache: Redis cache instance
        """
        self.db = db
        self.cache = cache
        self.persons: Dict[str, PersonProfile] = {}

    def create_person(self, name: str, phone: Optional[str] = None,
                     email: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Create a new person

        Args:
            name: Person's name
            phone: Optional phone number
            email: Optional email
            metadata: Optional metadata dictionary

        Returns:
            str: Person ID
        """
        person_id = str(uuid.uuid4())
        person = PersonProfile(person_id, name, phone, email, metadata)

        # Store in memory
        self.persons[person_id] = person

        # Cache it
        cache_key = f"person:{person_id}"
        self.cache.create_session(cache_key, person.to_dict())

        # Store in database
        try:
            speaker_id = self.db.create_speaker(name, phone)
            person.metadata['speaker_id'] = str(speaker_id)
            self._update_person_cache(person)
        except Exception as e:
            logger.error(f"Failed to create speaker in database: {e}")

        logger.info(f"Created person: {person_id} ({name})")
        return person_id

    def get_person(self, person_id: str) -> Optional[PersonProfile]:
        """
        Get person by ID

        Args:
            person_id: Person ID

        Returns:
            Optional[PersonProfile]: Person profile or None
        """
        # Try cache first
        cache_key = f"person:{person_id}"
        cached_data = self.cache.get_session(cache_key)

        if cached_data:
            return PersonProfile.from_dict(cached_data)

        # Try memory
        if person_id in self.persons:
            return self.persons[person_id]

        return None

    def update_person(self, person_id: str, updates: Dict) -> bool:
        """
        Update person information

        Args:
            person_id: Person ID
            updates: Dictionary of updates

        Returns:
            bool: Success status
        """
        person = self.get_person(person_id)
        if not person:
            return False

        # Apply updates
        if 'name' in updates:
            person.name = updates['name']
        if 'phone' in updates:
            person.phone = updates['phone']
        if 'email' in updates:
            person.email = updates['email']
        if 'metadata' in updates:
            person.metadata.update(updates['metadata'])

        person.updated_at = datetime.now().isoformat()

        # Update memory
        self.persons[person_id] = person

        # Update cache
        self._update_person_cache(person)

        logger.info(f"Updated person: {person_id}")
        return True

    def delete_person(self, person_id: str) -> bool:
        """
        Delete a person

        Args:
            person_id: Person ID

        Returns:
            bool: Success status
        """
        # Remove from memory
        if person_id in self.persons:
            del self.persons[person_id]

        # Remove from cache
        cache_key = f"person:{person_id}"
        self.cache.delete_session(cache_key)

        logger.info(f"Deleted person: {person_id}")
        return True

    def list_persons(self) -> List[PersonProfile]:
        """
        List all persons

        Returns:
            List[PersonProfile]: List of person profiles
        """
        return list(self.persons.values())

    def _update_person_cache(self, person: PersonProfile):
        """Update person in cache"""
        cache_key = f"person:{person.id}"
        self.cache.create_session(cache_key, person.to_dict())


class InteractionTracker:
    """Track interactions between persons"""

    def __init__(self, db, cache):
        """
        Initialize InteractionTracker

        Args:
            db: Database adapter
            cache: Redis cache instance
        """
        self.db = db
        self.cache = cache
        self.interactions: Dict[str, Interaction] = {}
        self.person_manager = None

    def set_person_manager(self, person_manager: PersonManager):
        """Set reference to person manager"""
        self.person_manager = person_manager

    def record_interaction(self, person1_id: str, person2_id: str,
                          interaction_type: str, content: str,
                          timestamp: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> str:
        """
        Record an interaction between two persons

        Args:
            person1_id: First person ID
            person2_id: Second person ID
            interaction_type: Type of interaction (message, call, etc.)
            content: Interaction content/text
            timestamp: Optional timestamp
            metadata: Optional metadata

        Returns:
            str: Interaction ID
        """
        interaction_id = str(uuid.uuid4())
        interaction = Interaction(
            interaction_id, person1_id, person2_id,
            interaction_type, content, timestamp, metadata
        )

        # Store in memory
        self.interactions[interaction_id] = interaction

        # Cache it
        cache_key = f"interaction:{interaction_id}"
        self.cache.create_session(cache_key, interaction.to_dict())

        # Update person interaction counts
        if self.person_manager:
            p1 = self.person_manager.get_person(person1_id)
            if p1:
                p1.interaction_count += 1
                p1.last_interaction = interaction.timestamp
                self.person_manager._update_person_cache(p1)

            p2 = self.person_manager.get_person(person2_id)
            if p2:
                p2.interaction_count += 1
                p2.last_interaction = interaction.timestamp
                self.person_manager._update_person_cache(p2)

        # Store message in database
        try:
            self.db.update_message_analysis(
                str(uuid.uuid4()),
                'interaction',
                interaction.to_dict()
            )
        except Exception as e:
            logger.error(f"Failed to store interaction in database: {e}")

        logger.info(f"Recorded interaction: {interaction_id}")
        return interaction_id

    def get_interaction(self, interaction_id: str) -> Optional[Interaction]:
        """
        Get interaction by ID

        Args:
            interaction_id: Interaction ID

        Returns:
            Optional[Interaction]: Interaction or None
        """
        # Try cache first
        cache_key = f"interaction:{interaction_id}"
        cached_data = self.cache.get_session(cache_key)

        if cached_data:
            inter = Interaction(**cached_data)
            return inter

        # Try memory
        if interaction_id in self.interactions:
            return self.interactions[interaction_id]

        return None

    def get_person_interactions(self, person_id: str, limit: Optional[int] = None) -> List[Interaction]:
        """
        Get all interactions for a person

        Args:
            person_id: Person ID
            limit: Optional limit on results

        Returns:
            List[Interaction]: List of interactions
        """
        interactions = [
            inter for inter in self.interactions.values()
            if inter.person1_id == person_id or inter.person2_id == person_id
        ]

        # Sort by timestamp descending
        interactions.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            interactions = interactions[:limit]

        return interactions


class RelationshipAnalyzer:
    """Analyze relationships between persons"""

    def __init__(self, db, cache, interaction_tracker: InteractionTracker):
        """
        Initialize RelationshipAnalyzer

        Args:
            db: Database adapter
            cache: Redis cache instance
            interaction_tracker: Interaction tracker instance
        """
        self.db = db
        self.cache = cache
        self.interaction_tracker = interaction_tracker
        self.timelines: Dict[str, RelationshipTimeline] = {}

    def get_relationship_timeline(self, person1_id: str, person2_id: str) -> RelationshipTimeline:
        """
        Get timeline of interactions between two persons

        Args:
            person1_id: First person ID
            person2_id: Second person ID

        Returns:
            RelationshipTimeline: Relationship timeline
        """
        # Create cache key (sorted to ensure consistency)
        sorted_ids = tuple(sorted([person1_id, person2_id]))
        cache_key = f"timeline:{sorted_ids[0]}:{sorted_ids[1]}"

        # Try cache first
        cached_data = self.cache.get_session(cache_key)
        if cached_data:
            timeline = RelationshipTimeline(person1_id, person2_id)
            timeline.total_interactions = cached_data.get('total_interactions', 0)
            timeline.first_interaction = cached_data.get('first_interaction')
            timeline.last_interaction = cached_data.get('last_interaction')
            timeline.relationship_status = cached_data.get('relationship_status', 'new')
            timeline.overall_risk = cached_data.get('overall_risk', 'low')
            timeline.timeline_summary = cached_data.get('timeline_summary', {})
            return timeline

        # Build from interactions
        timeline = RelationshipTimeline(person1_id, person2_id)
        all_interactions = self.interaction_tracker.interactions.values()

        for interaction in all_interactions:
            # Check if interaction involves both persons
            if ((interaction.person1_id == person1_id and interaction.person2_id == person2_id) or
                (interaction.person1_id == person2_id and interaction.person2_id == person1_id)):
                timeline.add_interaction(interaction)

        # Analyze timeline characteristics
        self._analyze_timeline(timeline)

        # Cache it
        self.cache.create_session(cache_key, timeline.to_dict())

        self.timelines[f"{person1_id}:{person2_id}"] = timeline
        return timeline

    def _analyze_timeline(self, timeline: RelationshipTimeline):
        """Analyze timeline characteristics"""
        if timeline.total_interactions == 0:
            timeline.relationship_status = 'no_interaction'
            timeline.overall_risk = 'unknown'
            return

        # Calculate relationship status
        if timeline.total_interactions <= 5:
            timeline.relationship_status = 'new'
        elif timeline.total_interactions <= 20:
            timeline.relationship_status = 'developing'
        else:
            timeline.relationship_status = 'established'

        # Calculate overall risk
        risk_scores = [i.risk_score for i in timeline.interactions if i.risk_score > 0]

        if not risk_scores:
            timeline.overall_risk = 'low'
        else:
            avg_risk = sum(risk_scores) / len(risk_scores)
            if avg_risk > 0.7:
                timeline.overall_risk = 'high'
            elif avg_risk > 0.4:
                timeline.overall_risk = 'medium'
            else:
                timeline.overall_risk = 'low'

        # Build timeline summary
        timeline.timeline_summary = {
            'total_interactions': timeline.total_interactions,
            'date_range': f"{timeline.first_interaction} to {timeline.last_interaction}",
            'interaction_types': self._count_interaction_types(timeline),
            'average_sentiment': self._calculate_avg_sentiment(timeline),
            'risk_events': sum(1 for i in timeline.interactions if i.risk_score > 0.5)
        }

    def _count_interaction_types(self, timeline: RelationshipTimeline) -> Dict[str, int]:
        """Count interactions by type"""
        types = {}
        for interaction in timeline.interactions:
            types[interaction.interaction_type] = types.get(interaction.interaction_type, 0) + 1
        return types

    def _calculate_avg_sentiment(self, timeline: RelationshipTimeline) -> float:
        """Calculate average sentiment"""
        sentiments = [i.sentiment for i in timeline.interactions if i.sentiment != 0]
        if not sentiments:
            return 0.0
        return sum(sentiments) / len(sentiments)


class RiskAssessmentEngine:
    """Generate risk assessments for persons"""

    def __init__(self, db, cache):
        """
        Initialize RiskAssessmentEngine

        Args:
            db: Database adapter
            cache: Redis cache instance
        """
        self.db = db
        self.cache = cache

    def assess_person_risk(self, person_id: str, interaction_history: Optional[List[Interaction]] = None) -> RiskAssessment:
        """
        Generate risk assessment for a person

        Args:
            person_id: Person ID
            interaction_history: Optional list of interactions

        Returns:
            RiskAssessment: Risk assessment
        """
        assessment = RiskAssessment(person_id)

        if not interaction_history:
            interaction_history = []

        # Analyze interactions for risk indicators
        for interaction in interaction_history:
            # This would integrate with actual risk analyzers
            # For now, use placeholder logic
            self._score_interaction(interaction, assessment)

        # Calculate overall risk
        self._calculate_overall_risk(assessment)

        # Generate recommendations
        self._generate_recommendations(assessment)

        # Store in database
        try:
            self.db.save_risk_assessment(assessment.to_dict())
        except Exception as e:
            logger.error(f"Failed to save risk assessment: {e}")

        # Cache it
        cache_key = f"risk_assessment:{person_id}"
        self.cache.create_session(cache_key, assessment.to_dict())

        return assessment

    def _score_interaction(self, interaction: Interaction, assessment: RiskAssessment):
        """Score interaction for risk"""
        # This would use actual NLP/analysis engines
        # Placeholder implementation
        if 'threat' in interaction.content.lower():
            assessment.hostility_risk += 0.2
        if 'manipulate' in interaction.content.lower():
            assessment.manipulation_risk += 0.2
        if 'groom' in interaction.content.lower():
            assessment.grooming_risk += 0.2

    def _calculate_overall_risk(self, assessment: RiskAssessment):
        """Calculate overall risk score"""
        scores = [
            assessment.grooming_risk,
            assessment.manipulation_risk,
            assessment.deception_risk,
            assessment.hostility_risk,
            assessment.escalation_risk
        ]

        assessment.overall_risk = sum(scores) / len(scores)
        assessment.confidence = 0.6  # Placeholder

        # Determine risk level
        if assessment.overall_risk > 0.7:
            assessment.risk_level = 'critical'
        elif assessment.overall_risk > 0.5:
            assessment.risk_level = 'high'
        elif assessment.overall_risk > 0.3:
            assessment.risk_level = 'medium'
        else:
            assessment.risk_level = 'low'

    def _generate_recommendations(self, assessment: RiskAssessment):
        """Generate recommendations based on assessment"""
        if assessment.risk_level == 'critical':
            assessment.recommendations.append('Immediate intervention required')
            assessment.recommendations.append('Review all communications')
        elif assessment.risk_level == 'high':
            assessment.recommendations.append('Close monitoring recommended')
            assessment.recommendations.append('Escalate to appropriate authorities')
        elif assessment.risk_level == 'medium':
            assessment.recommendations.append('Document all interactions')
            assessment.recommendations.append('Regular monitoring suggested')

        if assessment.grooming_risk > 0.5:
            assessment.primary_concerns.append('Potential grooming behavior')
        if assessment.manipulation_risk > 0.5:
            assessment.primary_concerns.append('Manipulation tactics detected')
        if assessment.hostility_risk > 0.5:
            assessment.primary_concerns.append('Hostile behavior patterns')


# ==========================================
# Flask Blueprint Creation
# ==========================================

def create_api_blueprint(db, cache, interaction_tracker=None, person_manager=None, risk_engine=None):
    """
    Create Flask API blueprint with unified endpoints

    Args:
        db: Database adapter instance
        cache: Redis cache instance
        interaction_tracker: Optional InteractionTracker instance
        person_manager: Optional PersonManager instance
        risk_engine: Optional RiskAssessmentEngine instance

    Returns:
        Blueprint: Flask blueprint with all API endpoints
    """
    api = Blueprint('api', __name__, url_prefix='/api')

    # Initialize managers if not provided
    if not person_manager:
        person_manager = PersonManager(db, cache)
    if not interaction_tracker:
        interaction_tracker = InteractionTracker(db, cache)
        interaction_tracker.set_person_manager(person_manager)
    if not risk_engine:
        risk_engine = RiskAssessmentEngine(db, cache)

    relationship_analyzer = RelationshipAnalyzer(db, cache, interaction_tracker)

    # ==========================================
    # Person CRUD Endpoints
    # ==========================================

    @api.route('/persons', methods=['POST'])
    def create_person_endpoint():
        """Create a new person"""
        try:
            data = request.get_json()

            if not data or 'name' not in data:
                return jsonify({'error': 'Name is required'}), 400

            person_id = person_manager.create_person(
                name=data['name'],
                phone=data.get('phone'),
                email=data.get('email'),
                metadata=data.get('metadata', {})
            )

            person = person_manager.get_person(person_id)
            return jsonify({
                'success': True,
                'person': person.to_dict()
            }), 201

        except Exception as e:
            logger.error(f"Error creating person: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/persons', methods=['GET'])
    def list_persons_endpoint():
        """List all persons"""
        try:
            persons = person_manager.list_persons()
            return jsonify({
                'success': True,
                'count': len(persons),
                'persons': [p.to_dict() for p in persons]
            }), 200

        except Exception as e:
            logger.error(f"Error listing persons: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/persons/<person_id>', methods=['GET'])
    def get_person_endpoint(person_id):
        """Get person by ID"""
        try:
            person = person_manager.get_person(person_id)

            if not person:
                return jsonify({'error': 'Person not found'}), 404

            return jsonify({
                'success': True,
                'person': person.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error getting person: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/persons/<person_id>', methods=['PUT'])
    def update_person_endpoint(person_id):
        """Update person information"""
        try:
            data = request.get_json()

            if not data:
                return jsonify({'error': 'No data provided'}), 400

            success = person_manager.update_person(person_id, data)

            if not success:
                return jsonify({'error': 'Person not found'}), 404

            person = person_manager.get_person(person_id)
            return jsonify({
                'success': True,
                'person': person.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error updating person: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/persons/<person_id>', methods=['DELETE'])
    def delete_person_endpoint(person_id):
        """Delete a person"""
        try:
            success = person_manager.delete_person(person_id)

            if not success:
                return jsonify({'error': 'Person not found'}), 404

            return jsonify({'success': True}), 200

        except Exception as e:
            logger.error(f"Error deleting person: {e}")
            return jsonify({'error': str(e)}), 500

    # ==========================================
    # Interaction Tracking Endpoints
    # ==========================================

    @api.route('/interactions', methods=['POST'])
    def record_interaction_endpoint():
        """Record an interaction between two persons"""
        try:
            data = request.get_json()

            required_fields = ['person1_id', 'person2_id', 'interaction_type', 'content']
            if not data or not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400

            interaction_id = interaction_tracker.record_interaction(
                person1_id=data['person1_id'],
                person2_id=data['person2_id'],
                interaction_type=data['interaction_type'],
                content=data['content'],
                timestamp=data.get('timestamp'),
                metadata=data.get('metadata', {})
            )

            interaction = interaction_tracker.get_interaction(interaction_id)
            return jsonify({
                'success': True,
                'interaction': interaction.to_dict()
            }), 201

        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/interactions/<interaction_id>', methods=['GET'])
    def get_interaction_endpoint(interaction_id):
        """Get interaction by ID"""
        try:
            interaction = interaction_tracker.get_interaction(interaction_id)

            if not interaction:
                return jsonify({'error': 'Interaction not found'}), 404

            return jsonify({
                'success': True,
                'interaction': interaction.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error getting interaction: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/persons/<person_id>/interactions', methods=['GET'])
    def get_person_interactions_endpoint(person_id):
        """Get all interactions for a person"""
        try:
            limit = request.args.get('limit', type=int)
            interactions = interaction_tracker.get_person_interactions(person_id, limit)

            return jsonify({
                'success': True,
                'person_id': person_id,
                'count': len(interactions),
                'interactions': [i.to_dict() for i in interactions]
            }), 200

        except Exception as e:
            logger.error(f"Error getting interactions: {e}")
            return jsonify({'error': str(e)}), 500

    # ==========================================
    # Relationship Timeline Endpoints
    # ==========================================

    @api.route('/timeline/<person1_id>/<person2_id>', methods=['GET'])
    def get_relationship_timeline_endpoint(person1_id, person2_id):
        """Get relationship timeline between two persons"""
        try:
            timeline = relationship_analyzer.get_relationship_timeline(person1_id, person2_id)

            return jsonify({
                'success': True,
                'timeline': timeline.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return jsonify({'error': str(e)}), 500

    # ==========================================
    # Risk Assessment Endpoints
    # ==========================================

    @api.route('/risk-assessment/<person_id>', methods=['GET'])
    def get_risk_assessment_endpoint(person_id):
        """Get risk assessment for a person"""
        try:
            # Get interaction history
            interactions = interaction_tracker.get_person_interactions(person_id)

            # Generate assessment
            assessment = risk_engine.assess_person_risk(person_id, interactions)

            return jsonify({
                'success': True,
                'assessment': assessment.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error getting risk assessment: {e}")
            return jsonify({'error': str(e)}), 500

    @api.route('/risk-assessment/<person_id>/recompute', methods=['POST'])
    def recompute_risk_assessment_endpoint(person_id):
        """Recompute risk assessment for a person"""
        try:
            # Get interaction history
            interactions = interaction_tracker.get_person_interactions(person_id)

            # Generate fresh assessment
            assessment = risk_engine.assess_person_risk(person_id, interactions)

            return jsonify({
                'success': True,
                'assessment': assessment.to_dict()
            }), 200

        except Exception as e:
            logger.error(f"Error recomputing risk assessment: {e}")
            return jsonify({'error': str(e)}), 500

    # ==========================================
    # Utility Endpoints
    # ==========================================

    @api.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }), 200

    @api.route('/stats', methods=['GET'])
    def get_stats():
        """Get API statistics"""
        return jsonify({
            'success': True,
            'statistics': {
                'total_persons': len(person_manager.list_persons()),
                'total_interactions': len(interaction_tracker.interactions),
                'total_relationships': len(relationship_analyzer.timelines)
            }
        }), 200

    return api, person_manager, interaction_tracker, relationship_analyzer, risk_engine
