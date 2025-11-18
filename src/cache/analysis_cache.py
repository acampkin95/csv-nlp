"""
Enhanced Analysis Result Caching
Provides full-analysis result caching with CSV hash + config-based invalidation
Enables instant results for re-analysis of same data
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class AnalysisResultCache:
    """Smart caching for complete analysis results"""

    # Full analysis cache TTL (2 hours - configurable)
    FULL_ANALYSIS_TTL = 7200

    def __init__(self, redis_cache: RedisCache):
        """Initialize analysis result cache

        Args:
            redis_cache: RedisCache instance
        """
        self.redis = redis_cache
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_saves': 0
        }

    def calculate_csv_hash(self, csv_path: str) -> str:
        """Calculate hash of CSV file content

        Args:
            csv_path: Path to CSV file

        Returns:
            str: SHA256 hash of file content
        """
        try:
            # Read CSV and calculate hash
            df = pd.read_csv(csv_path)

            # Convert to consistent string representation
            csv_content = df.to_csv(index=False)

            # Calculate hash
            file_hash = hashlib.sha256(csv_content.encode()).hexdigest()

            logger.debug(f"Calculated CSV hash: {file_hash[:16]}...")
            return file_hash

        except Exception as e:
            logger.error(f"Failed to calculate CSV hash: {e}")
            # Return path hash as fallback
            return hashlib.sha256(csv_path.encode()).hexdigest()

    def calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of analysis configuration

        Args:
            config: Configuration dictionary

        Returns:
            str: SHA256 hash of configuration
        """
        try:
            # Sort keys for consistent hashing
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()

            logger.debug(f"Calculated config hash: {config_hash[:16]}...")
            return config_hash

        except Exception as e:
            logger.error(f"Failed to calculate config hash: {e}")
            return "default"

    def get_cache_key(self, csv_hash: str, config_hash: str) -> str:
        """Generate cache key from CSV + config hashes

        Args:
            csv_hash: CSV content hash
            config_hash: Configuration hash

        Returns:
            str: Cache key
        """
        combined = f"{csv_hash}:{config_hash}"
        cache_key_hash = hashlib.sha256(combined.encode()).hexdigest()
        return f"full_analysis:{cache_key_hash}"

    def get_cached_analysis(self, csv_path: str, config: Dict[str, Any]) -> Optional[Dict]:
        """Get cached analysis result if available

        Args:
            csv_path: Path to CSV file
            config: Analysis configuration

        Returns:
            Optional[Dict]: Cached analysis result or None
        """
        if not self.redis.enabled:
            return None

        try:
            # Calculate hashes
            csv_hash = self.calculate_csv_hash(csv_path)
            config_hash = self.calculate_config_hash(config)

            # Get cache key
            cache_key = self.get_cache_key(csv_hash, config_hash)

            # Try to get from cache
            cached = self.redis.get_session(cache_key)

            if cached:
                self.stats['cache_hits'] += 1
                logger.info(f"✅ CACHE HIT: Full analysis for {Path(csv_path).name}")
                logger.info(f"   Key: {cache_key[:32]}...")
                return cached
            else:
                self.stats['cache_misses'] += 1
                logger.info(f"❌ CACHE MISS: {Path(csv_path).name}")
                return None

        except Exception as e:
            logger.error(f"Error getting cached analysis: {e}")
            return None

    def cache_analysis(self, csv_path: str, config: Dict[str, Any],
                      results: Dict, ttl: Optional[int] = None):
        """Cache complete analysis results

        Args:
            csv_path: Path to CSV file
            config: Analysis configuration
            results: Complete analysis results
            ttl: Time to live in seconds (default: 2 hours)
        """
        if not self.redis.enabled:
            return

        try:
            # Calculate hashes
            csv_hash = self.calculate_csv_hash(csv_path)
            config_hash = self.calculate_config_hash(config)

            # Get cache key
            cache_key = self.get_cache_key(csv_hash, config_hash)

            # Add metadata
            cache_data = {
                'csv_path': csv_path,
                'csv_hash': csv_hash,
                'config_hash': config_hash,
                'results': results,
                'cached_at': pd.Timestamp.now().isoformat()
            }

            # Cache with TTL
            ttl = ttl or self.FULL_ANALYSIS_TTL
            success = self.redis.create_session(cache_key, cache_data, ttl=ttl)

            if success:
                self.stats['cache_saves'] += 1
                logger.info(f"💾 CACHED: Full analysis for {Path(csv_path).name}")
                logger.info(f"   Key: {cache_key[:32]}...")
                logger.info(f"   TTL: {ttl / 3600:.1f} hours")
            else:
                logger.warning("Failed to cache analysis results")

        except Exception as e:
            logger.error(f"Error caching analysis: {e}")

    def invalidate_cache(self, csv_path: Optional[str] = None, pattern: Optional[str] = None):
        """Invalidate cached analysis results

        Args:
            csv_path: Specific CSV file to invalidate (invalidates all configs)
            pattern: Pattern to match for bulk invalidation
        """
        if not self.redis.enabled:
            return

        try:
            if csv_path:
                # Invalidate all analyses for this CSV
                csv_hash = self.calculate_csv_hash(csv_path)
                # Use wildcard pattern to match all config variations
                pattern = f"msgproc:session:full_analysis:{csv_hash}*"

            if pattern:
                deleted = self.redis.clear_cache(pattern)
                logger.info(f"🗑️  Invalidated {deleted} cached analyses")
            else:
                logger.warning("No CSV path or pattern provided for invalidation")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict: Cache statistics
        """
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_saves': self.stats['cache_saves'],
            'total_requests': total_requests,
            'hit_rate_percent': f"{hit_rate:.1f}%",
            'estimated_time_saved_minutes': self.stats['cache_hits'] * 0.5  # Assuming 30s per analysis
        }

    def print_stats(self):
        """Print cache statistics to console"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("ANALYSIS CACHE STATISTICS")
        print("="*60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits:     {stats['cache_hits']} ({stats['hit_rate_percent']})")
        print(f"Cache Misses:   {stats['cache_misses']}")
        print(f"Cache Saves:    {stats['cache_saves']}")
        print(f"Time Saved:     ~{stats['estimated_time_saved_minutes']:.1f} minutes")
        print("="*60)


def create_analysis_cache(redis_host: str = 'localhost', redis_port: int = 6379) -> AnalysisResultCache:
    """Factory function to create analysis cache

    Args:
        redis_host: Redis server host
        redis_port: Redis server port

    Returns:
        AnalysisResultCache: Configured cache instance
    """
    redis_cache = RedisCache(host=redis_host, port=redis_port)
    return AnalysisResultCache(redis_cache)
