# Person Management - Backend Implementation Guide

## Overview

This guide provides the backend implementation requirements for the Person Management UI. The frontend components have been created and are ready to integrate with backend API endpoints.

## Required API Endpoints

### Person Management Endpoints

#### 1. List All Persons
```
GET /api/persons
Content-Type: application/json

Response:
{
    "persons": [
        {
            "id": "person_123",
            "name": "John Doe",
            "first_name": "John",
            "last_name": "Doe",
            "email": "john@example.com",
            "phone": "+1234567890",
            "status": "active",
            "risk_level": "high",
            "risk_score": 75,
            "risk_notes": "Assessment notes",
            "notes": "Additional notes",
            "tags": "tag1,tag2",
            "interaction_count": 42,
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-18T10:30:00Z"
        },
        ...
    ],
    "total": 10,
    "page": 1,
    "per_page": 50
}

Status: 200 OK
```

#### 2. Create New Person
```
POST /api/persons
Content-Type: application/json

Request Body:
{
    "first_name": "Jane",
    "last_name": "Smith",
    "email": "jane@example.com",
    "phone": "+9876543210",
    "status": "active",
    "risk_level": "moderate",
    "risk_score": 50,
    "risk_notes": "Initial assessment",
    "notes": "New person entry",
    "tags": "monitored,active"
}

Response:
{
    "id": "person_456",
    "name": "Jane Smith",
    "first_name": "Jane",
    "last_name": "Smith",
    "email": "jane@example.com",
    "phone": "+9876543210",
    "status": "active",
    "risk_level": "moderate",
    "risk_score": 50,
    "risk_notes": "Initial assessment",
    "notes": "New person entry",
    "tags": "monitored,active",
    "interaction_count": 0,
    "created_at": "2024-01-18T10:35:00Z",
    "updated_at": "2024-01-18T10:35:00Z"
}

Status: 201 Created
```

#### 3. Get Person Details
```
GET /api/persons/<person_id>
Content-Type: application/json

Response: (Same as person object above)

Status: 200 OK / 404 Not Found
```

#### 4. Update Person
```
PUT /api/persons/<person_id>
Content-Type: application/json

Request Body:
{
    "first_name": "Jane",
    "last_name": "Smith",
    "email": "jane.smith@example.com",
    "status": "active",
    "risk_level": "high",
    "risk_score": 80,
    "risk_notes": "Updated assessment",
    "notes": "Updated information",
    "tags": "flagged,high-risk"
}

Response: (Updated person object)

Status: 200 OK / 404 Not Found / 400 Bad Request
```

#### 5. Delete Person
```
DELETE /api/persons/<person_id>
Content-Type: application/json

Response:
{
    "success": true,
    "message": "Person deleted successfully",
    "person_id": "person_456"
}

Status: 200 OK / 404 Not Found
```

---

### Interaction Endpoints

#### 1. List Interactions
```
GET /api/interactions?person_id=<id>&type=<type>&from_date=<date>&to_date=<date>
Content-Type: application/json

Query Parameters:
- person_id: (optional) Filter by person
- type: (optional) message, call, meeting, incident
- from_date: (optional) ISO date string
- to_date: (optional) ISO date string

Response:
{
    "interactions": [
        {
            "id": "interaction_789",
            "person_id": "person_123",
            "related_person_id": "person_456",
            "type": "message",
            "description": "Message content",
            "date": "2024-01-18T10:30:00Z",
            "risk_level": "moderate",
            "created_at": "2024-01-18T10:35:00Z"
        },
        ...
    ],
    "total": 100,
    "page": 1
}

Status: 200 OK
```

#### 2. Create Interaction
```
POST /api/interactions
Content-Type: application/json

Request Body:
{
    "person_id": "person_123",
    "related_person_id": "person_456",
    "type": "message",
    "description": "Interaction details",
    "date": "2024-01-18T10:30:00Z",
    "risk_level": "low"
}

Response:
{
    "id": "interaction_790",
    "person_id": "person_123",
    "related_person_id": "person_456",
    "type": "message",
    "description": "Interaction details",
    "date": "2024-01-18T10:30:00Z",
    "risk_level": "low",
    "created_at": "2024-01-18T10:35:00Z"
}

Status: 201 Created
```

---

### Report & Escalation Endpoints

#### 1. Generate Intervention Report
```
POST /api/reports/intervention/<person_id>
Content-Type: application/json

Response:
{
    "success": true,
    "report_id": "report_123",
    "file_url": "/files/reports/intervention_person_123.pdf",
    "generated_at": "2024-01-18T10:35:00Z"
}

Status: 200 OK / 404 Not Found
```

#### 2. Escalate Case
```
POST /api/cases/escalate/<person_id>
Content-Type: application/json

Request Body (optional):
{
    "reason": "Escalation reason",
    "priority": "urgent",
    "assignee": "manager_id"
}

Response:
{
    "success": true,
    "case_id": "case_456",
    "status": "escalated",
    "escalated_at": "2024-01-18T10:35:00Z"
}

Status: 200 OK / 404 Not Found
```

---

## WebSocket Implementation

### Person Updates WebSocket

```
Connection: ws://localhost:5000/ws/persons
or
wss://localhost:5000/ws/persons (for HTTPS)
```

#### Message Types

**Person Updated**
```json
{
    "type": "person_updated",
    "person": {
        "id": "person_123",
        "name": "John Doe",
        "email": "john@example.com",
        "status": "active",
        "risk_level": "high",
        "risk_score": 75,
        "interaction_count": 50,
        "updated_at": "2024-01-18T10:35:00Z"
    }
}
```

**Person Deleted**
```json
{
    "type": "person_deleted",
    "person_id": "person_123"
}
```

### Interaction Updates WebSocket

```
Connection: ws://localhost:5000/ws/interactions
or
wss://localhost:5000/ws/interactions (for HTTPS)
```

#### Message Types

**New Interaction**
```json
{
    "type": "new_interaction",
    "person_id": "person_123",
    "interaction": {
        "id": "interaction_791",
        "person_id": "person_123",
        "related_person_id": "person_456",
        "type": "message",
        "description": "New interaction",
        "date": "2024-01-18T10:35:00Z",
        "risk_level": "moderate"
    }
}
```

---

## Backend Implementation Examples

### Flask Implementation

```python
from flask import Blueprint, jsonify, request
from datetime import datetime
import json

# Create blueprint
persons_bp = Blueprint('persons', __name__, url_prefix='/api/persons')

# Sample database/storage (replace with actual DB)
persons_db = {}

@persons_bp.route('', methods=['GET'])
def list_persons():
    """List all persons"""
    persons = list(persons_db.values())
    return jsonify({
        'persons': persons,
        'total': len(persons),
        'page': 1,
        'per_page': 50
    }), 200

@persons_bp.route('', methods=['POST'])
def create_person():
    """Create new person"""
    data = request.get_json()

    # Validation
    required_fields = ['first_name', 'last_name', 'email', 'risk_level']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    person_id = f"person_{len(persons_db) + 1}"

    person = {
        'id': person_id,
        'name': f"{data['first_name']} {data['last_name']}",
        'first_name': data['first_name'],
        'last_name': data['last_name'],
        'email': data['email'],
        'phone': data.get('phone', ''),
        'status': data.get('status', 'active'),
        'risk_level': data['risk_level'],
        'risk_score': data.get('risk_score', 0),
        'risk_notes': data.get('risk_notes', ''),
        'notes': data.get('notes', ''),
        'tags': data.get('tags', ''),
        'interaction_count': 0,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }

    persons_db[person_id] = person

    # Broadcast to WebSocket clients
    broadcast_person_update('person_updated', person)

    return jsonify(person), 201

@persons_bp.route('/<person_id>', methods=['GET'])
def get_person(person_id):
    """Get person details"""
    person = persons_db.get(person_id)
    if not person:
        return jsonify({'error': 'Person not found'}), 404
    return jsonify(person), 200

@persons_bp.route('/<person_id>', methods=['PUT'])
def update_person(person_id):
    """Update person"""
    person = persons_db.get(person_id)
    if not person:
        return jsonify({'error': 'Person not found'}), 404

    data = request.get_json()

    # Update fields
    updateable_fields = [
        'first_name', 'last_name', 'email', 'phone', 'status',
        'risk_level', 'risk_score', 'risk_notes', 'notes', 'tags'
    ]

    for field in updateable_fields:
        if field in data:
            person[field] = data[field]

    # Update name if first_name or last_name changed
    if 'first_name' in data or 'last_name' in data:
        person['name'] = f"{person['first_name']} {person['last_name']}"

    person['updated_at'] = datetime.utcnow().isoformat() + 'Z'

    # Broadcast update
    broadcast_person_update('person_updated', person)

    return jsonify(person), 200

@persons_bp.route('/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete person"""
    if person_id not in persons_db:
        return jsonify({'error': 'Person not found'}), 404

    del persons_db[person_id]

    # Broadcast deletion
    broadcast_person_update('person_deleted', {'person_id': person_id})

    return jsonify({
        'success': True,
        'message': 'Person deleted successfully',
        'person_id': person_id
    }), 200

# Register blueprint
app.register_blueprint(persons_bp)
```

### WebSocket Handler (Flask-SocketIO)

```python
from flask_socketio import SocketIO, emit, join_room, leave_room

socketio = SocketIO(app, cors_allowed_origins="*")

# Track connected clients
connected_clients = {
    'persons': set(),
    'interactions': set()
}

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f'Client connected: {request.sid}')

@socketio.on('join_persons')
def join_persons_room():
    """Join persons update room"""
    join_room('persons')
    connected_clients['persons'].add(request.sid)
    emit('status', {'status': 'connected'})

@socketio.on('join_interactions')
def join_interactions_room():
    """Join interactions update room"""
    join_room('interactions')
    connected_clients['interactions'].add(request.sid)
    emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle disconnection"""
    connected_clients['persons'].discard(request.sid)
    connected_clients['interactions'].discard(request.sid)
    print(f'Client disconnected: {request.sid}')

def broadcast_person_update(message_type, data):
    """Broadcast person update to all connected clients"""
    socketio.emit(message_type, {
        'type': message_type,
        'person': data if message_type == 'person_updated' else None,
        'person_id': data.get('person_id') if message_type == 'person_deleted' else None
    }, room='persons')

def broadcast_interaction_update(person_id, interaction):
    """Broadcast interaction update to all connected clients"""
    socketio.emit('new_interaction', {
        'type': 'new_interaction',
        'person_id': person_id,
        'interaction': interaction
    }, room='interactions')
```

---

## Database Schema

### Persons Table

```sql
CREATE TABLE persons (
    id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',
    risk_level VARCHAR(20) NOT NULL,
    risk_score INTEGER DEFAULT 0,
    risk_notes TEXT,
    notes TEXT,
    tags TEXT,
    interaction_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX (email),
    INDEX (risk_level),
    INDEX (status),
    INDEX (created_at)
);
```

### Interactions Table

```sql
CREATE TABLE interactions (
    id VARCHAR(50) PRIMARY KEY,
    person_id VARCHAR(50) NOT NULL,
    related_person_id VARCHAR(50),
    type VARCHAR(50) NOT NULL,
    description TEXT,
    date TIMESTAMP NOT NULL,
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY (related_person_id) REFERENCES persons(id) ON DELETE SET NULL,
    INDEX (person_id),
    INDEX (date),
    INDEX (type),
    INDEX (risk_level)
);
```

---

## Integration Steps

### 1. Setup Routes
- Add person endpoints to Flask/FastAPI app
- Add interaction endpoints
- Configure error handling

### 2. Setup WebSocket
- Install and configure WebSocket library (Socket.IO/WebSockets)
- Create connection handlers
- Implement broadcast functions

### 3. Database
- Create person and interaction tables
- Add indexes for performance
- Configure relationships

### 4. Testing
- Test each endpoint with curl/Postman
- Test WebSocket connections
- Test data persistence

### 5. Frontend Integration
- Ensure API URLs match backend routes
- Test form submissions
- Verify real-time updates

---

## Risk Level Classification

Implement risk classification based on:

```python
def calculate_risk_level(risk_score):
    """Calculate risk level from score"""
    if risk_score >= 80:
        return "critical"
    elif risk_score >= 60:
        return "high"
    elif risk_score >= 40:
        return "moderate"
    else:
        return "low"

def assess_risk_score(person_data, interactions):
    """Calculate risk score from person data and interactions"""
    score = 0

    # Base risk factors
    if person_data.get('status') == 'flagged':
        score += 20

    # Interaction-based factors
    interaction_count = len(interactions)
    if interaction_count > 20:
        score += 10
    if interaction_count > 50:
        score += 20

    # Risk level escalation
    high_risk_interactions = len([i for i in interactions if i.get('risk_level') == 'high'])
    critical_risk_interactions = len([i for i in interactions if i.get('risk_level') == 'critical'])

    score += high_risk_interactions * 5
    score += critical_risk_interactions * 15

    return min(100, max(0, score))
```

---

## Intervention Recommendation Engine

```python
def generate_intervention_recommendations(person, interactions):
    """Generate intervention recommendations"""
    recommendations = []
    risk_level = person.get('risk_level', 'low')

    if risk_level == 'critical':
        recommendations.append({
            'title': 'Immediate Safety Assessment Required',
            'description': 'Person classified as critical risk. Schedule immediate safety assessment.',
            'priority': 'urgent'
        })
        recommendations.append({
            'title': 'Escalate to Senior Management',
            'description': 'Critical risk cases must be escalated to senior management.',
            'priority': 'urgent'
        })

    if risk_level in ['critical', 'high']:
        recommendations.append({
            'title': 'Increase Monitoring Frequency',
            'description': 'Implement daily monitoring and interaction review.',
            'priority': 'urgent'
        })

    if len(interactions) > 20:
        recommendations.append({
            'title': 'High Interaction Volume',
            'description': f'{len(interactions)} interactions detected. Review patterns for concerning behaviors.',
            'priority': 'important'
        })

    return recommendations
```

---

## Testing Checklist

- [ ] Create person endpoint returns 201
- [ ] List persons endpoint returns all persons
- [ ] Get person endpoint returns correct person
- [ ] Update person endpoint modifies data
- [ ] Delete person endpoint removes person
- [ ] List interactions with filters works
- [ ] Create interaction endpoint works
- [ ] WebSocket connection establishes
- [ ] Person update broadcasts to clients
- [ ] Interaction update broadcasts to clients
- [ ] Report generation creates PDF
- [ ] Case escalation marks case correctly

---

## Performance Optimization

### Indexing Strategy
```sql
-- Add indexes for frequently queried fields
CREATE INDEX idx_persons_risk_level ON persons(risk_level);
CREATE INDEX idx_persons_status ON persons(status);
CREATE INDEX idx_persons_email ON persons(email);
CREATE INDEX idx_interactions_person_id ON interactions(person_id);
CREATE INDEX idx_interactions_date ON interactions(date);
```

### Query Optimization
```python
# Use pagination
def list_persons(page=1, per_page=50):
    offset = (page - 1) * per_page
    return db.query(Person).offset(offset).limit(per_page).all()

# Cache frequently accessed data
from functools import lru_cache

@lru_cache(maxsize=100)
def get_person_with_interactions(person_id):
    person = db.query(Person).get(person_id)
    interactions = db.query(Interaction).filter_by(person_id=person_id).all()
    return (person, interactions)
```

---

## Deployment Considerations

1. **Environment Variables**
   - Store database credentials in `.env`
   - Configure WebSocket secure mode (WSS)
   - Set CORS origins

2. **Database Migration**
   - Use Alembic for schema migrations
   - Version control for schema changes

3. **API Documentation**
   - Generate OpenAPI/Swagger docs
   - Document all endpoints
   - Include example requests/responses

4. **Monitoring**
   - Log all API requests
   - Monitor WebSocket connections
   - Track error rates

---

## Support

For implementation questions, refer to:
- Frontend documentation: `PERSON_MANAGEMENT_UI.md`
- API specifications above
- WebSocket implementation examples

**Created**: 2024-01-18
**Version**: 1.0
