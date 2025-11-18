"""
Redis Caching Module for Message Processor
Provides high-performance caching for analysis results, feature extraction, and session management
"""

import redis
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, List
from datetime import timedelta
from dataclasses import asdict, is_dataclass
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


class RedisCache:
    """High-performance Redis cache for message analysis"""

    # Cache TTL (Time To Live) in seconds
    DEFAULT_TTL = 3600  # 1 hour
    FEATURE_EXTRACTION_TTL = 86400  # 24 hours
    ANALYSIS_RESULTS_TTL = 7200  # 2 hours
    SESSION_TTL = 86400  # 24 hours

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: Optional[str] = None, decode_responses: bool = False):
        """Initialize Redis cache connection

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number (0-15)
            password: Redis password if authentication required
            decode_responses: Whether to decode responses to strings
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            self.enabled = True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Cache disabled.")
            self.enabled = False
            self.client = None

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create cache key

        Args:
            prefix: Key prefix (e.g., 'sentiment', 'grooming', 'session')
            identifier: Unique identifier (often a hash)

        Returns:
            str: Formatted cache key
        """
        return f"msgproc:{prefix}:{identifier}"

    def _hash_data(self, data: Any) -> str:
        """Create hash of data for cache key

        Args:
            data: Data to hash

        Returns:
            str: SHA256 hash
        """
        # Convert to JSON for consistent hashing
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage

        Args:
            data: Data to serialize

        Returns:
            bytes: Serialized data
        """
        # Handle dataclasses
        if is_dataclass(data):
            data = asdict(data)

        # Use pickle for complex objects
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from cache

        Args:
            data: Serialized data

        Returns:
            Any: Deserialized data
        """
        return pickle.loads(data)

    # ==========================================
    # Feature Extraction Caching
    # ==========================================

    def cache_feature_extraction(self, text: str, features: Dict, ttl: Optional[int] = None) -> bool:
        """Cache feature extraction results

        Args:
            text: Original text
            features: Extracted features
            ttl: Time to live in seconds

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        try:
            text_hash = self._hash_data(text)
            key = self._make_key('features', text_hash)
            data = self._serialize(features)

            ttl = ttl or self.FEATURE_EXTRACTION_TTL
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Failed to cache features: {e}")
            return False

    def get_cached_features(self, text: str) -> Optional[Dict]:
        """Get cached feature extraction results

        Args:
            text: Original text

        Returns:
            Optional[Dict]: Cached features or None
        """
        if not self.enabled:
            return None

        try:
            text_hash = self._hash_data(text)
            key = self._make_key('features', text_hash)
            data = self.client.get(key)

            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached features: {e}")
            return None

    # ==========================================
    # Analysis Results Caching
    # ==========================================

    def cache_analysis(self, analysis_type: str, message_data: Any, result: Any,
                      ttl: Optional[int] = None) -> bool:
        """Cache analysis results

        Args:
            analysis_type: Type of analysis (sentiment, grooming, manipulation, etc.)
            message_data: Input message data (text or dict)
            result: Analysis result
            ttl: Time to live in seconds

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        try:
            data_hash = self._hash_data(message_data)
            key = self._make_key(f'analysis:{analysis_type}', data_hash)
            data = self._serialize(result)

            ttl = ttl or self.ANALYSIS_RESULTS_TTL
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Failed to cache analysis: {e}")
            return False

    def get_cached_analysis(self, analysis_type: str, message_data: Any) -> Optional[Any]:
        """Get cached analysis results

        Args:
            analysis_type: Type of analysis
            message_data: Input message data

        Returns:
            Optional[Any]: Cached result or None
        """
        if not self.enabled:
            return None

        try:
            data_hash = self._hash_data(message_data)
            key = self._make_key(f'analysis:{analysis_type}', data_hash)
            data = self.client.get(key)

            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached analysis: {e}")
            return None

    # ==========================================
    # Batch Caching
    # ==========================================

    def cache_batch_analysis(self, analysis_type: str, messages: List[Dict],
                            results: List[Any], ttl: Optional[int] = None) -> int:
        """Cache multiple analysis results at once

        Args:
            analysis_type: Type of analysis
            messages: List of message dictionaries
            results: List of analysis results
            ttl: Time to live in seconds

        Returns:
            int: Number of successfully cached items
        """
        if not self.enabled or len(messages) != len(results):
            return 0

        cached_count = 0
        pipeline = self.client.pipeline()
        ttl = ttl or self.ANALYSIS_RESULTS_TTL

        try:
            for msg, result in zip(messages, results):
                msg_hash = self._hash_data(msg.get('text', msg))
                key = self._make_key(f'analysis:{analysis_type}', msg_hash)
                data = self._serialize(result)
                pipeline.setex(key, ttl, data)
                cached_count += 1

            pipeline.execute()
            logger.info(f"Batch cached {cached_count} {analysis_type} results")
            return cached_count
        except Exception as e:
            logger.error(f"Failed to batch cache: {e}")
            return 0

    def get_batch_analysis(self, analysis_type: str, messages: List[Dict]) -> List[Optional[Any]]:
        """Get multiple cached analysis results

        Args:
            analysis_type: Type of analysis
            messages: List of message dictionaries

        Returns:
            List[Optional[Any]]: List of cached results (None for cache misses)
        """
        if not self.enabled:
            return [None] * len(messages)

        try:
            pipeline = self.client.pipeline()
            keys = []

            for msg in messages:
                msg_hash = self._hash_data(msg.get('text', msg))
                key = self._make_key(f'analysis:{analysis_type}', msg_hash)
                keys.append(key)
                pipeline.get(key)

            raw_results = pipeline.execute()
            results = []

            for data in raw_results:
                if data:
                    results.append(self._deserialize(data))
                else:
                    results.append(None)

            cache_hits = sum(1 for r in results if r is not None)
            logger.info(f"Batch cache: {cache_hits}/{len(messages)} hits for {analysis_type}")
            return results
        except Exception as e:
            logger.error(f"Failed to batch get cache: {e}")
            return [None] * len(messages)

    # ==========================================
    # Session Management
    # ==========================================

    def create_session(self, session_id: str, data: Dict, ttl: Optional[int] = None) -> bool:
        """Create or update a session

        Args:
            session_id: Unique session identifier
            data: Session data
            ttl: Time to live in seconds

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        try:
            key = self._make_key('session', session_id)
            serialized = self._serialize(data)
            ttl = ttl or self.SESSION_TTL

            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data

        Args:
            session_id: Session identifier

        Returns:
            Optional[Dict]: Session data or None
        """
        if not self.enabled:
            return None

        try:
            key = self._make_key('session', session_id)
            data = self.client.get(key)

            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def update_session(self, session_id: str, updates: Dict) -> bool:
        """Update session data

        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply

        Returns:
            bool: Success status
        """
        session_data = self.get_session(session_id)
        if session_data:
            session_data.update(updates)
            return self.create_session(session_id, session_data)
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session

        Args:
            session_id: Session identifier

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        try:
            key = self._make_key('session', session_id)
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    # ==========================================
    # Cache Statistics
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict: Cache statistics
        """
        if not self.enabled:
            return {'enabled': False}

        try:
            info = self.client.info()
            return {
                'enabled': True,
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_keys': self.client.dbsize(),
                'uptime_days': info.get('uptime_in_days'),
                'hit_rate': self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'enabled': True, 'error': str(e)}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate

        Args:
            info: Redis INFO output

        Returns:
            float: Hit rate percentage
        """
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses

        if total > 0:
            return (hits / total) * 100
        return 0.0

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries

        Args:
            pattern: Key pattern to match (e.g., 'msgproc:analysis:*')
                    If None, clears entire database

        Returns:
            int: Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            if pattern:
                keys = self.client.keys(pattern)
                if keys:
                    deleted = self.client.delete(*keys)
                    logger.info(f"Cleared {deleted} cache entries matching '{pattern}'")
                    return deleted
                return 0
            else:
                self.client.flushdb()
                logger.info("Cleared entire cache database")
                return -1  # Unknown count
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0


# ==========================================
# Decorator for Automatic Caching
# ==========================================

def cached_analysis(cache: RedisCache, analysis_type: str, ttl: Optional[int] = None):
    """Decorator for automatic caching of analysis functions

    Args:
        cache: RedisCache instance
        analysis_type: Type of analysis
        ttl: Time to live in seconds

    Example:
        @cached_analysis(cache, 'sentiment', ttl=3600)
        def analyze_sentiment(text):
            # ... analysis code ...
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(text, *args, **kwargs):
            # Try to get from cache
            cached_result = cache.get_cached_analysis(analysis_type, text)
            if cached_result is not None:
                logger.debug(f"Cache hit for {analysis_type}: {text[:50]}...")
                return cached_result

            # Cache miss - perform analysis
            logger.debug(f"Cache miss for {analysis_type}: {text[:50]}...")
            result = func(text, *args, **kwargs)

            # Store in cache
            cache.cache_analysis(analysis_type, text, result, ttl)

            return result
        return wrapper
    return decorator


# ==========================================
# Performance Monitoring
# ==========================================

class CachePerformanceMonitor:
    """Monitor cache performance metrics"""

    def __init__(self, cache: RedisCache):
        """Initialize monitor

        Args:
            cache: RedisCache instance
        """
        self.cache = cache
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'total_time': 0.0
        }

    def record_hit(self, elapsed_time: float):
        """Record cache hit"""
        self.stats['hits'] += 1
        self.stats['total_time'] += elapsed_time

    def record_miss(self, elapsed_time: float):
        """Record cache miss"""
        self.stats['misses'] += 1
        self.stats['total_time'] += elapsed_time

    def record_error(self):
        """Record cache error"""
        self.stats['errors'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics

        Returns:
            Dict: Performance metrics
        """
        total_requests = self.stats['hits'] + self.stats['misses']

        return {
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'errors': self.stats['errors'],
            'hit_rate': (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0,
            'avg_response_time': (self.stats['total_time'] / total_requests) if total_requests > 0 else 0
        }

    def reset(self):
        """Reset statistics"""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'total_time': 0.0
        }
