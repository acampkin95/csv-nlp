# Unified API Documentation
## Message Processor Flask Webapp with ppl_int FastAPI Features

### Overview

This document describes the unified REST API endpoints for the Message Processor Flask web application. The API provides comprehensive person management, interaction tracking, relationship analysis, and risk assessment capabilities merged from the ppl_int backend.

### Base URL
```
http://localhost:5000/api
```

### Authentication
Currently, all endpoints are accessible without authentication. In production, implement OAuth2/JWT authentication.

---

## Person Management Endpoints

### Create Person
**POST** `/api/persons`

Create a new person profile in the system.

#### Request Body
```json
{
  "name": "John Doe",
  "phone": "+1-555-0123",
  "email": "john.doe@example.com",
  "metadata": {
    "department": "Sales",
    "region": "West"
  }
}
```

#### Response (201 Created)
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "John Doe",
    "phone": "+1-555-0123",
    "email": "john.doe@example.com",
    "metadata": {
      "department": "Sales",
      "region": "West"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T10:30:00.000000",
    "interaction_count": 0,
    "risk_level": "low",
    "last_interaction": null
  }
}
```

#### Error Responses
- **400 Bad Request**: Missing required 'name' field
- **500 Internal Server Error**: Database or cache operation failed

---

### Get Person
**GET** `/api/persons/<person_id>`

Retrieve a specific person's profile.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Response (200 OK)
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "John Doe",
    "phone": "+1-555-0123",
    "email": "john.doe@example.com",
    "metadata": {
      "department": "Sales",
      "region": "West"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T10:30:00.000000",
    "interaction_count": 5,
    "risk_level": "low",
    "last_interaction": "2024-11-18T12:45:00.000000"
  }
}
```

#### Error Responses
- **404 Not Found**: Person not found
- **500 Internal Server Error**: Database error

---

### Update Person
**PUT** `/api/persons/<person_id>`

Update an existing person's information.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Request Body
```json
{
  "name": "Jane Doe",
  "phone": "+1-555-0456",
  "email": "jane.doe@example.com",
  "metadata": {
    "department": "Marketing",
    "region": "East"
  }
}
```

#### Response (200 OK)
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Jane Doe",
    "phone": "+1-555-0456",
    "email": "jane.doe@example.com",
    "metadata": {
      "department": "Marketing",
      "region": "East"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T14:00:00.000000",
    "interaction_count": 5,
    "risk_level": "low",
    "last_interaction": "2024-11-18T12:45:00.000000"
  }
}
```

#### Error Responses
- **400 Bad Request**: No data provided
- **404 Not Found**: Person not found
- **500 Internal Server Error**: Database error

---

### List All Persons
**GET** `/api/persons`

Retrieve all persons in the system.

#### Response (200 OK)
```json
{
  "success": true,
  "count": 2,
  "persons": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "John Doe",
      "phone": "+1-555-0123",
      "email": "john.doe@example.com",
      "metadata": {},
      "created_at": "2024-11-18T10:30:00.000000",
      "updated_at": "2024-11-18T10:30:00.000000",
      "interaction_count": 5,
      "risk_level": "low",
      "last_interaction": "2024-11-18T12:45:00.000000"
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "name": "Jane Doe",
      "phone": "+1-555-0456",
      "email": "jane.doe@example.com",
      "metadata": {},
      "created_at": "2024-11-18T11:00:00.000000",
      "updated_at": "2024-11-18T14:00:00.000000",
      "interaction_count": 3,
      "risk_level": "low",
      "last_interaction": "2024-11-18T13:30:00.000000"
    }
  ]
}
```

#### Error Responses
- **500 Internal Server Error**: Database error

---

### Delete Person
**DELETE** `/api/persons/<person_id>`

Remove a person from the system.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Response (200 OK)
```json
{
  "success": true
}
```

#### Error Responses
- **404 Not Found**: Person not found
- **500 Internal Server Error**: Database error

---

## Interaction Tracking Endpoints

### Record Interaction
**POST** `/api/interactions`

Record a new interaction between two persons.

#### Request Body
```json
{
  "person1_id": "550e8400-e29b-41d4-a716-446655440000",
  "person2_id": "660e8400-e29b-41d4-a716-446655440001",
  "interaction_type": "message",
  "content": "Hey, how are you doing?",
  "timestamp": "2024-11-18T12:45:00.000000",
  "metadata": {
    "platform": "WhatsApp",
    "language": "en"
  }
}
```

#### Response (201 Created)
```json
{
  "success": true,
  "interaction": {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "person1_id": "550e8400-e29b-41d4-a716-446655440000",
    "person2_id": "660e8400-e29b-41d4-a716-446655440001",
    "interaction_type": "message",
    "content": "Hey, how are you doing?",
    "timestamp": "2024-11-18T12:45:00.000000",
    "metadata": {
      "platform": "WhatsApp",
      "language": "en"
    },
    "sentiment": 0.0,
    "risk_score": 0.0,
    "flags": []
  }
}
```

#### Error Responses
- **400 Bad Request**: Missing required fields (person1_id, person2_id, interaction_type, content)
- **500 Internal Server Error**: Database or cache error

---

### Get Interaction
**GET** `/api/interactions/<interaction_id>`

Retrieve details of a specific interaction.

#### URL Parameters
- `interaction_id` (string, required): UUID of the interaction

#### Response (200 OK)
```json
{
  "success": true,
  "interaction": {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "person1_id": "550e8400-e29b-41d4-a716-446655440000",
    "person2_id": "660e8400-e29b-41d4-a716-446655440001",
    "interaction_type": "message",
    "content": "Hey, how are you doing?",
    "timestamp": "2024-11-18T12:45:00.000000",
    "metadata": {
      "platform": "WhatsApp",
      "language": "en"
    },
    "sentiment": 0.15,
    "risk_score": 0.1,
    "flags": []
  }
}
```

#### Error Responses
- **404 Not Found**: Interaction not found
- **500 Internal Server Error**: Database error

---

### Get Person's Interactions
**GET** `/api/persons/<person_id>/interactions`

Retrieve all interactions for a specific person.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Query Parameters
- `limit` (integer, optional): Maximum number of interactions to return (default: all)

#### Response (200 OK)
```json
{
  "success": true,
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "count": 3,
  "interactions": [
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "person1_id": "550e8400-e29b-41d4-a716-446655440000",
      "person2_id": "660e8400-e29b-41d4-a716-446655440001",
      "interaction_type": "message",
      "content": "Hey, how are you doing?",
      "timestamp": "2024-11-18T12:45:00.000000",
      "metadata": {},
      "sentiment": 0.15,
      "risk_score": 0.1,
      "flags": []
    },
    {
      "id": "880e8400-e29b-41d4-a716-446655440003",
      "person1_id": "550e8400-e29b-41d4-a716-446655440000",
      "person2_id": "770e8400-e29b-41d4-a716-446655440002",
      "interaction_type": "call",
      "content": "Voice call 5 minutes",
      "timestamp": "2024-11-18T11:30:00.000000",
      "metadata": {},
      "sentiment": 0.0,
      "risk_score": 0.0,
      "flags": []
    }
  ]
}
```

#### Error Responses
- **500 Internal Server Error**: Database error

---

## Relationship Analysis Endpoints

### Get Relationship Timeline
**GET** `/api/timeline/<person1_id>/<person2_id>`

Retrieve the complete timeline of interactions between two persons.

#### URL Parameters
- `person1_id` (string, required): UUID of the first person
- `person2_id` (string, required): UUID of the second person

#### Response (200 OK)
```json
{
  "success": true,
  "timeline": {
    "person1_id": "550e8400-e29b-41d4-a716-446655440000",
    "person2_id": "660e8400-e29b-41d4-a716-446655440001",
    "interactions": [
      {
        "id": "770e8400-e29b-41d4-a716-446655440002",
        "person1_id": "550e8400-e29b-41d4-a716-446655440000",
        "person2_id": "660e8400-e29b-41d4-a716-446655440001",
        "interaction_type": "message",
        "content": "Hey, how are you doing?",
        "timestamp": "2024-11-18T12:45:00.000000",
        "metadata": {},
        "sentiment": 0.15,
        "risk_score": 0.1,
        "flags": []
      }
    ],
    "first_interaction": "2024-11-18T09:00:00.000000",
    "last_interaction": "2024-11-18T14:30:00.000000",
    "total_interactions": 5,
    "relationship_status": "developing",
    "overall_risk": "low",
    "timeline_summary": {
      "total_interactions": 5,
      "date_range": "2024-11-18T09:00:00.000000 to 2024-11-18T14:30:00.000000",
      "interaction_types": {
        "message": 4,
        "call": 1
      },
      "average_sentiment": 0.08,
      "risk_events": 0
    }
  }
}
```

#### Relationship Status Values
- `no_interaction`: No recorded interactions
- `new`: 1-5 interactions
- `developing`: 6-20 interactions
- `established`: 20+ interactions

#### Overall Risk Values
- `unknown`: Insufficient data
- `low`: Average risk score < 0.3
- `medium`: Average risk score 0.3-0.7
- `high`: Average risk score > 0.7

#### Error Responses
- **500 Internal Server Error**: Database error

---

## Risk Assessment Endpoints

### Get Risk Assessment
**GET** `/api/risk-assessment/<person_id>`

Generate or retrieve a risk assessment for a person.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Response (200 OK)
```json
{
  "success": true,
  "assessment": {
    "person_id": "550e8400-e29b-41d4-a716-446655440000",
    "assessment_type": "comprehensive",
    "timestamp": "2024-11-18T15:00:00.000000",
    "grooming_risk": 0.0,
    "manipulation_risk": 0.15,
    "deception_risk": 0.05,
    "hostility_risk": 0.0,
    "escalation_risk": 0.0,
    "overall_risk": 0.04,
    "risk_level": "low",
    "primary_concerns": [],
    "behavioral_indicators": {},
    "recommendations": [
      "Document all interactions",
      "Regular monitoring suggested"
    ],
    "confidence": 0.6
  }
}
```

#### Risk Level Values
- `low`: Overall risk < 0.3
- `medium`: Overall risk 0.3-0.5
- `high`: Overall risk 0.5-0.7
- `critical`: Overall risk >= 0.7

#### Risk Score Components
- `grooming_risk` (0.0-1.0): Likelihood of grooming behavior
- `manipulation_risk` (0.0-1.0): Likelihood of manipulation tactics
- `deception_risk` (0.0-1.0): Likelihood of dishonesty/deception
- `hostility_risk` (0.0-1.0): Likelihood of hostile behavior
- `escalation_risk` (0.0-1.0): Likelihood of escalating behavior
- `overall_risk` (0.0-1.0): Aggregate risk score
- `confidence` (0.0-1.0): Confidence in the assessment

#### Error Responses
- **500 Internal Server Error**: Assessment generation error

---

### Recompute Risk Assessment
**POST** `/api/risk-assessment/<person_id>/recompute`

Force a fresh computation of risk assessment based on current data.

#### URL Parameters
- `person_id` (string, required): UUID of the person

#### Response (200 OK)
```json
{
  "success": true,
  "assessment": {
    "person_id": "550e8400-e29b-41d4-a716-446655440000",
    "assessment_type": "comprehensive",
    "timestamp": "2024-11-18T15:30:00.000000",
    "grooming_risk": 0.0,
    "manipulation_risk": 0.15,
    "deception_risk": 0.05,
    "hostility_risk": 0.0,
    "escalation_risk": 0.0,
    "overall_risk": 0.04,
    "risk_level": "low",
    "primary_concerns": [],
    "behavioral_indicators": {},
    "recommendations": [
      "Document all interactions",
      "Regular monitoring suggested"
    ],
    "confidence": 0.6
  }
}
```

#### Error Responses
- **500 Internal Server Error**: Assessment computation error

---

## Utility Endpoints

### Health Check
**GET** `/api/health`

Check API health and availability.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "timestamp": "2024-11-18T15:45:00.000000"
}
```

---

### Get API Statistics
**GET** `/api/stats`

Retrieve current API statistics.

#### Response (200 OK)
```json
{
  "success": true,
  "statistics": {
    "total_persons": 25,
    "total_interactions": 150,
    "total_relationships": 42
  }
}
```

---

## Caching Strategy

The unified API leverages Redis caching for optimal performance:

### Cache TTLs (Time To Live)
- **Person Profiles**: 1 hour (3600 seconds)
- **Interactions**: 2 hours (7200 seconds)
- **Relationship Timelines**: 30 minutes (1800 seconds)
- **Risk Assessments**: 1 hour (3600 seconds)
- **Feature Extraction**: 24 hours (86400 seconds)

### Cache Key Patterns
```
msgproc:person:{person_id}
msgproc:interaction:{interaction_id}
msgproc:person_interactions:{person_id}
msgproc:timeline:{person1_id}:{person2_id}
msgproc:risk_assessment:{person_id}
```

---

## Integration with Existing Systems

### Database Integration
All endpoints integrate with:
- **PostgreSQL**: Primary data store via `PostgreSQLAdapter`
- **Redis Cache**: High-performance caching layer
- **Message Processor**: NLP/analysis engines for risk scoring

### Project Management Integration
Persons and interactions can be associated with analysis projects:
```python
project_manager.add_person_to_project(project_id, person_id, person_data)
project_manager.add_interaction_to_project(project_id, interaction_id, interaction_data)
```

---

## Error Handling

All endpoints follow standard HTTP status codes:

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | OK | Successful GET/PUT |
| 201 | Created | Successful POST |
| 400 | Bad Request | Invalid input |
| 404 | Not Found | Resource doesn't exist |
| 500 | Internal Server Error | Server-side error |

---

## Rate Limiting (Future)

Currently no rate limiting is implemented. In production, add:
- Per-user rate limits (e.g., 100 requests/minute)
- Per-endpoint throttling
- IP-based blocking for suspicious activity

---

## Security Considerations

### Current Limitations
- No authentication/authorization
- No input validation
- No rate limiting
- Exposed database credentials (should use environment variables)

### Recommended Improvements
1. Implement JWT/OAuth2 authentication
2. Add input validation and sanitization
3. Implement request rate limiting
4. Use parameterized queries (already implemented)
5. Add CORS configuration
6. Encrypt sensitive data at rest
7. Audit logging for sensitive operations

---

## WebSocket Support (Future)

The architecture supports real-time WebSocket updates for:
- Live interaction tracking
- Real-time risk assessment updates
- Bi-directional person profile synchronization

---

## Example Usage

### Create a person and track interactions
```bash
# Create person 1
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# Create person 2
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob", "email": "bob@example.com"}'

# Record interaction
curl -X POST http://localhost:5000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "PERSON_1_ID",
    "person2_id": "PERSON_2_ID",
    "interaction_type": "message",
    "content": "Hello Bob, how are you?"
  }'

# Get relationship timeline
curl http://localhost:5000/api/timeline/PERSON_1_ID/PERSON_2_ID

# Get risk assessment
curl http://localhost:5000/api/risk-assessment/PERSON_1_ID
```

---

## Conclusion

This unified API provides a comprehensive interface for person management, interaction tracking, relationship analysis, and risk assessment. It seamlessly integrates ppl_int FastAPI features with the existing Flask webapp while maintaining backward compatibility and leveraging Redis caching for high-performance operations.
