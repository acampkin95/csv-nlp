# API Testing Guide

This guide demonstrates how to test the unified API endpoints using various tools and methods.

---

## Prerequisites

- Flask webapp running on `http://localhost:5000`
- Redis running on `localhost:6379`
- PostgreSQL running and accessible
- cURL or Postman for testing

---

## Starting the Application

```bash
cd /Users/alex/Projects/Dev/Projects/Message\ Processor/Dev-Root

# Set environment variables
export FLASK_ENV=development
export REDIS_HOST=localhost
export REDIS_PORT=6379
export POSTGRES_HOST=acdev.host
export POSTGRES_DB=messagestore
export POSTGRES_USER=msgprocess
export POSTGRES_PASSWORD=DHifde93jes9dk

# Run the application
python webapp.py
```

---

## Quick Health Check

```bash
# Check API health
curl -X GET http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-18T10:30:00.000000"
}
```

---

## Person Management Tests

### Test 1: Create First Person

```bash
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson",
    "phone": "+1-555-0001",
    "email": "alice.johnson@example.com",
    "metadata": {
      "department": "Engineering",
      "location": "San Francisco"
    }
  }'
```

**Expected Response** (201 Created):
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Alice Johnson",
    "phone": "+1-555-0001",
    "email": "alice.johnson@example.com",
    "metadata": {
      "department": "Engineering",
      "location": "San Francisco"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T10:30:00.000000",
    "interaction_count": 0,
    "risk_level": "low",
    "last_interaction": null
  }
}
```

**Save the person ID** for subsequent tests:
```bash
PERSON_1_ID="550e8400-e29b-41d4-a716-446655440000"
```

### Test 2: Create Second Person

```bash
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Bob Smith",
    "phone": "+1-555-0002",
    "email": "bob.smith@example.com",
    "metadata": {
      "department": "Marketing",
      "location": "New York"
    }
  }'
```

**Save the person ID**:
```bash
PERSON_2_ID="660e8400-e29b-41d4-a716-446655440001"
```

### Test 3: Get Person Details

```bash
curl -X GET http://localhost:5000/api/persons/$PERSON_1_ID
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Alice Johnson",
    "phone": "+1-555-0001",
    "email": "alice.johnson@example.com",
    "metadata": {
      "department": "Engineering",
      "location": "San Francisco"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T10:30:00.000000",
    "interaction_count": 0,
    "risk_level": "low",
    "last_interaction": null
  }
}
```

### Test 4: List All Persons

```bash
curl -X GET http://localhost:5000/api/persons
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "count": 2,
  "persons": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Alice Johnson",
      ...
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "name": "Bob Smith",
      ...
    }
  ]
}
```

### Test 5: Update Person

```bash
curl -X PUT http://localhost:5000/api/persons/$PERSON_1_ID \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson",
    "phone": "+1-555-0001-updated",
    "metadata": {
      "department": "Engineering",
      "location": "San Francisco",
      "role": "Senior Engineer"
    }
  }'
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "person": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Alice Johnson",
    "phone": "+1-555-0001-updated",
    "email": "alice.johnson@example.com",
    "metadata": {
      "department": "Engineering",
      "location": "San Francisco",
      "role": "Senior Engineer"
    },
    "created_at": "2024-11-18T10:30:00.000000",
    "updated_at": "2024-11-18T10:30:05.000000",
    "interaction_count": 0,
    "risk_level": "low",
    "last_interaction": null
  }
}
```

---

## Interaction Tracking Tests

### Test 6: Record First Interaction

```bash
curl -X POST http://localhost:5000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "'$PERSON_1_ID'",
    "person2_id": "'$PERSON_2_ID'",
    "interaction_type": "message",
    "content": "Hey Bob, how are you doing today?",
    "timestamp": "2024-11-18T10:00:00.000000",
    "metadata": {
      "platform": "WhatsApp",
      "message_length": 36
    }
  }'
```

**Expected Response** (201 Created):
```json
{
  "success": true,
  "interaction": {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "person1_id": "550e8400-e29b-41d4-a716-446655440000",
    "person2_id": "660e8400-e29b-41d4-a716-446655440001",
    "interaction_type": "message",
    "content": "Hey Bob, how are you doing today?",
    "timestamp": "2024-11-18T10:00:00.000000",
    "metadata": {
      "platform": "WhatsApp",
      "message_length": 36
    },
    "sentiment": 0.0,
    "risk_score": 0.0,
    "flags": []
  }
}
```

**Save the interaction ID**:
```bash
INTERACTION_1_ID="770e8400-e29b-41d4-a716-446655440002"
```

### Test 7: Record More Interactions

```bash
# Interaction 2: Call
curl -X POST http://localhost:5000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "'$PERSON_1_ID'",
    "person2_id": "'$PERSON_2_ID'",
    "interaction_type": "call",
    "content": "Voice call - 10 minutes",
    "timestamp": "2024-11-18T11:30:00.000000",
    "metadata": {
      "platform": "Phone",
      "duration_minutes": 10
    }
  }'

# Interaction 3: Message
curl -X POST http://localhost:5000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "'$PERSON_2_ID'",
    "person2_id": "'$PERSON_1_ID'",
    "interaction_type": "message",
    "content": "I am good thanks! How about you?",
    "timestamp": "2024-11-18T11:35:00.000000",
    "metadata": {
      "platform": "WhatsApp"
    }
  }'
```

### Test 8: Get Interaction Details

```bash
curl -X GET http://localhost:5000/api/interactions/$INTERACTION_1_ID
```

### Test 9: Get Person's Interactions

```bash
curl -X GET "http://localhost:5000/api/persons/$PERSON_1_ID/interactions?limit=10"
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "count": 2,
  "interactions": [
    {
      "id": "880e8400-e29b-41d4-a716-446655440003",
      "person1_id": "660e8400-e29b-41d4-a716-446655440001",
      "person2_id": "550e8400-e29b-41d4-a716-446655440000",
      "interaction_type": "message",
      "content": "I am good thanks! How about you?",
      "timestamp": "2024-11-18T11:35:00.000000",
      "metadata": {"platform": "WhatsApp"},
      "sentiment": 0.0,
      "risk_score": 0.0,
      "flags": []
    },
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "person1_id": "550e8400-e29b-41d4-a716-446655440000",
      "person2_id": "660e8400-e29b-41d4-a716-446655440001",
      "interaction_type": "message",
      "content": "Hey Bob, how are you doing today?",
      "timestamp": "2024-11-18T10:00:00.000000",
      "metadata": {"platform": "WhatsApp", "message_length": 36},
      "sentiment": 0.0,
      "risk_score": 0.0,
      "flags": []
    }
  ]
}
```

---

## Relationship Analysis Tests

### Test 10: Get Relationship Timeline

```bash
curl -X GET http://localhost:5000/api/timeline/$PERSON_1_ID/$PERSON_2_ID
```

**Expected Response** (200 OK):
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
        "content": "Hey Bob, how are you doing today?",
        "timestamp": "2024-11-18T10:00:00.000000",
        "metadata": {"platform": "WhatsApp", "message_length": 36},
        "sentiment": 0.0,
        "risk_score": 0.0,
        "flags": []
      }
    ],
    "first_interaction": "2024-11-18T10:00:00.000000",
    "last_interaction": "2024-11-18T11:35:00.000000",
    "total_interactions": 3,
    "relationship_status": "new",
    "overall_risk": "low",
    "timeline_summary": {
      "total_interactions": 3,
      "date_range": "2024-11-18T10:00:00.000000 to 2024-11-18T11:35:00.000000",
      "interaction_types": {
        "message": 2,
        "call": 1
      },
      "average_sentiment": 0.0,
      "risk_events": 0
    }
  }
}
```

---

## Risk Assessment Tests

### Test 11: Get Risk Assessment

```bash
curl -X GET http://localhost:5000/api/risk-assessment/$PERSON_1_ID
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "assessment": {
    "person_id": "550e8400-e29b-41d4-a716-446655440000",
    "assessment_type": "comprehensive",
    "timestamp": "2024-11-18T12:00:00.000000",
    "grooming_risk": 0.0,
    "manipulation_risk": 0.0,
    "deception_risk": 0.0,
    "hostility_risk": 0.0,
    "escalation_risk": 0.0,
    "overall_risk": 0.0,
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

### Test 12: Recompute Risk Assessment

```bash
curl -X POST http://localhost:5000/api/risk-assessment/$PERSON_1_ID/recompute \
  -H "Content-Type: application/json"
```

---

## Utility Tests

### Test 13: Get API Statistics

```bash
curl -X GET http://localhost:5000/api/stats
```

**Expected Response** (200 OK):
```json
{
  "success": true,
  "statistics": {
    "total_persons": 2,
    "total_interactions": 3,
    "total_relationships": 1
  }
}
```

---

## Cache Verification Tests

### Test 14: Verify Cache Hit (Person Profile)

First request (cache miss):
```bash
time curl -X GET http://localhost:5000/api/persons/$PERSON_1_ID
# Should take ~100-200ms
```

Second request (cache hit):
```bash
time curl -X GET http://localhost:5000/api/persons/$PERSON_1_ID
# Should take ~5-10ms
```

### Test 15: Cache Invalidation (Update)

Update person:
```bash
curl -X PUT http://localhost:5000/api/persons/$PERSON_1_ID \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"updated": true}}'
```

Get person (should have fresh data):
```bash
curl -X GET http://localhost:5000/api/persons/$PERSON_1_ID
```

---

## Error Handling Tests

### Test 16: Test 404 Not Found

```bash
curl -X GET http://localhost:5000/api/persons/invalid-id
```

**Expected Response** (404 Not Found):
```json
{
  "error": "Person not found"
}
```

### Test 17: Test 400 Bad Request

```bash
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

**Expected Response** (400 Bad Request):
```json
{
  "error": "Name is required"
}
```

### Test 18: Test Missing Interaction Fields

```bash
curl -X POST http://localhost:5000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "'$PERSON_1_ID'",
    "interaction_type": "message"
  }'
```

**Expected Response** (400 Bad Request):
```json
{
  "error": "Missing required fields"
}
```

---

## Batch Testing Script

Create a file `test_api.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:5000/api"

echo "=== Testing Unified API ==="
echo

# Test health
echo "1. Testing health endpoint..."
curl -s $BASE_URL/health | jq '.'
echo

# Create person 1
echo "2. Creating person 1..."
RESPONSE=$(curl -s -X POST $BASE_URL/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "Test User 1"}')
PERSON_1_ID=$(echo $RESPONSE | jq -r '.person.id')
echo "Person 1 ID: $PERSON_1_ID"
echo

# Create person 2
echo "3. Creating person 2..."
RESPONSE=$(curl -s -X POST $BASE_URL/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "Test User 2"}')
PERSON_2_ID=$(echo $RESPONSE | jq -r '.person.id')
echo "Person 2 ID: $PERSON_2_ID"
echo

# Record interaction
echo "4. Recording interaction..."
curl -s -X POST $BASE_URL/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "person1_id": "'$PERSON_1_ID'",
    "person2_id": "'$PERSON_2_ID'",
    "interaction_type": "message",
    "content": "Test message"
  }' | jq '.'
echo

# Get timeline
echo "5. Getting relationship timeline..."
curl -s $BASE_URL/timeline/$PERSON_1_ID/$PERSON_2_ID | jq '.'
echo

# Get risk assessment
echo "6. Getting risk assessment..."
curl -s $BASE_URL/risk-assessment/$PERSON_1_ID | jq '.'
echo

# Get stats
echo "7. Getting API statistics..."
curl -s $BASE_URL/stats | jq '.'
echo

echo "=== All tests completed ==="
```

Run the script:
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## Performance Testing

### Using Apache Bench

```bash
# Test person endpoint (100 requests, 10 concurrent)
ab -n 100 -c 10 http://localhost:5000/api/persons

# Test risk assessment (50 requests, 5 concurrent)
ab -n 50 -c 5 http://localhost:5000/api/risk-assessment/$PERSON_1_ID
```

### Using wrk

```bash
# Install wrk (if not installed)
# On macOS: brew install wrk
# On Linux: apt-get install wrk

# Run 30 second load test with 4 threads and 100 connections
wrk -t4 -c100 -d30s http://localhost:5000/api/persons
```

---

## Postman Collection

Import this as a Postman collection:

```json
{
  "info": {
    "name": "Message Processor Unified API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/api/health"
      }
    },
    {
      "name": "Create Person",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/persons",
        "body": {
          "mode": "raw",
          "raw": "{\"name\": \"Test User\"}"
        }
      }
    },
    {
      "name": "List Persons",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/api/persons"
      }
    },
    {
      "name": "Get Person",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/api/persons/{{person_id}}"
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:5000"
    },
    {
      "key": "person_id",
      "value": ""
    }
  ]
}
```

---

## Debugging Tips

### Enable Flask Debug Logging

```python
# In webapp.py
logging.basicConfig(level=logging.DEBUG)
logger.debug("API request received")
```

### Monitor Redis Cache

```bash
# Redis CLI
redis-cli

# Monitor all keys
KEYS msgproc:*

# Get cache stats
INFO stats

# Monitor in real-time
MONITOR
```

### Check PostgreSQL

```bash
# Connect to database
psql -h acdev.host -U msgprocess -d messagestore

# Check tables
\dt

# Check data
SELECT * FROM speakers;
SELECT * FROM messages_master LIMIT 10;
SELECT * FROM risk_assessments LIMIT 10;
```

---

## Conclusion

This testing guide covers:
1. All major API endpoints
2. Common workflows
3. Error handling scenarios
4. Cache behavior verification
5. Performance testing techniques
6. Debugging strategies

Use these tests to verify the API is working correctly in your environment.
