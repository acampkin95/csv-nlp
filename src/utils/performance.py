#!/usr/bin/env python3
"""
Performance Optimization Module
Provides caching, lazy loading, and batch processing utilities for the message processor
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LRUCache:
    """Thread-safe Least Recently Used cache with size limit"""

    def __init__(self, maxsize: int = 128):
        """Initialize LRU cache

        Args:
            maxsize: Maximum number of items to cache
        """
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.maxsize:
                # Remove oldest
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict with cache stats
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }


class LazyLoader:
    """Lazy loader for heavy dependencies"""

    def __init__(self, load_func: Callable[[], T]):
        """Initialize lazy loader

        Args:
            load_func: Function to call to load the module/resource
        """
        self._load_func = load_func
        self._module: Optional[T] = None
        self._loading_time: Optional[float] = None

    def get(self) -> T:
        """Get the loaded module (loads if not already loaded)

        Returns:
            Loaded module/resource
        """
        if self._module is None:
            start_time = time.time()
            logger.info(f"Lazy loading: {self._load_func.__name__}")
            self._module = self._load_func()
            self._loading_time = time.time() - start_time
            logger.info(f"Loaded in {self._loading_time:.2f}s")
        return cast(T, self._module)

    @property
    def is_loaded(self) -> bool:
        """Check if module is loaded

        Returns:
            True if already loaded
        """
        return self._module is not None

    @property
    def loading_time(self) -> Optional[float]:
        """Get loading time in seconds

        Returns:
            Loading time or None if not loaded yet
        """
        return self._loading_time


def memoize(maxsize: int = 128):
    """Memoization decorator with LRU cache

    Args:
        maxsize: Maximum cache size

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Calculate and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        # Attach cache stats method
        wrapper.cache_stats = cache.get_stats  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore

        return wrapper
    return decorator


class BatchProcessor:
    """Batch processor for efficient bulk operations"""

    def __init__(self, batch_size: int = 1000):
        """Initialize batch processor

        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size

    def process_in_batches(
        self,
        items: list,
        process_func: Callable[[list], Any],
        desc: str = "Processing"
    ) -> list:
        """Process items in batches

        Args:
            items: List of items to process
            process_func: Function to process each batch
            desc: Description for logging

        Returns:
            List of results from all batches
        """
        results = []
        total = len(items)
        num_batches = (total + self.batch_size - 1) // self.batch_size

        logger.info(f"{desc}: {total} items in {num_batches} batches of {self.batch_size}")

        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            start_time = time.time()
            batch_results = process_func(batch)
            elapsed = time.time() - start_time

            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])

            logger.debug(
                f"Batch {batch_num}/{num_batches}: {len(batch)} items in {elapsed:.2f}s "
                f"({len(batch)/elapsed:.1f} items/sec)"
            )

        return results


class ProgressTracker:
    """Progress tracker for long-running operations"""

    def __init__(self, total: int, desc: str = "Processing", log_interval: int = 10):
        """Initialize progress tracker

        Args:
            total: Total number of items
            desc: Operation description
            log_interval: Log every N percent
        """
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.last_log_percent = 0

    def update(self, n: int = 1) -> None:
        """Update progress

        Args:
            n: Number of items completed
        """
        self.current += n
        percent = (self.current / self.total * 100) if self.total > 0 else 0

        # Log at intervals
        if percent - self.last_log_percent >= self.log_interval:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0

            logger.info(
                f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%) | "
                f"Rate: {rate:.1f} items/s | ETA: {eta:.1f}s"
            )
            self.last_log_percent = percent

    def finish(self) -> float:
        """Mark as finished and return total time

        Returns:
            Total elapsed time in seconds
        """
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.desc}: Complete! {self.total} items in {elapsed:.2f}s "
            f"({rate:.1f} items/s)"
        )
        return elapsed


class TimedCache:
    """Time-based cache that expires entries"""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize timed cache

        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Dict[str, tuple[Any, datetime]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            else:
                # Expired, remove
                del self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache with current timestamp

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, datetime.now())

    def clear_expired(self) -> int:
        """Remove expired entries

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            k for k, (_, ts) in self.cache.items()
            if now - ts >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)


def timed_operation(operation_name: str):
    """Decorator to time and log operations

    Args:
        operation_name: Name of the operation for logging

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting: {operation_name}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Completed: {operation_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed: {operation_name} after {elapsed:.2f}s: {e}")
                raise

        return wrapper
    return decorator


# Global caches for cross-module use
_global_result_cache = LRUCache(maxsize=256)
_global_text_cache = TimedCache(ttl_seconds=3600)  # 1 hour


def get_result_cache() -> LRUCache:
    """Get global result cache

    Returns:
        Global LRU cache instance
    """
    return _global_result_cache


def get_text_cache() -> TimedCache:
    """Get global text cache

    Returns:
        Global timed cache instance
    """
    return _global_text_cache
