#!/usr/bin/env python3
"""
Batch Processing Optimization Helpers
Provides specialized batch processing for message analysis operations
"""

import logging
from typing import List, Dict, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageBatchOptimizer:
    """Optimizes batch processing of messages for NLP analysis"""

    def __init__(self, batch_size: int = 500):
        """Initialize batch optimizer

        Args:
            batch_size: Number of messages per batch
        """
        self.batch_size = batch_size

    def process_messages_in_batches(
        self,
        messages: List[Dict[str, Any]],
        analyzers: Dict[str, Callable],
        desc: str = "messages"
    ) -> List[Dict[str, Any]]:
        """Process messages in optimized batches

        Args:
            messages: List of message dictionaries
            analyzers: Dict of analyzer functions to apply
            desc: Description for logging

        Returns:
            List of messages with analysis results added
        """
        total = len(messages)
        num_batches = (total + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing {total} {desc} in {num_batches} batches of {self.batch_size}")

        results = []
        for i in range(0, total, self.batch_size):
            batch = messages[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            # Apply all analyzers to batch
            for analyzer_name, analyzer_func in analyzers.items():
                try:
                    for msg in batch:
                        if 'text' in msg and msg['text']:
                            msg[analyzer_name] = analyzer_func(msg['text'])
                except Exception as e:
                    logger.warning(f"Analyzer {analyzer_name} failed on batch {batch_num}: {e}")

            results.extend(batch)

            if batch_num % 10 == 0 or batch_num == num_batches:
                logger.info(f"Processed batch {batch_num}/{num_batches}")

        return results

    def aggregate_by_speaker(
        self,
        messages: List[Dict[str, Any]],
        agg_fields: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate analysis results by speaker

        Args:
            messages: List of analyzed messages
            agg_fields: Fields to aggregate

        Returns:
            Dict mapping speaker to aggregated results
        """
        speaker_data = defaultdict(lambda: defaultdict(list))

        for msg in messages:
            speaker = msg.get('sender', 'unknown')
            for field in agg_fields:
                if field in msg and msg[field] is not None:
                    speaker_data[speaker][field].append(msg[field])

        # Calculate statistics
        speaker_stats = {}
        for speaker, data in speaker_data.items():
            stats = {}
            for field, values in data.items():
                if values and isinstance(values[0], (int, float)):
                    stats[field] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
                else:
                    stats[field] = {
                        'count': len(values),
                        'unique': len(set(str(v) for v in values))
                    }
            speaker_stats[speaker] = stats

        return speaker_stats


class PassResultsCache:
    """Cache manager for pipeline pass results"""

    def __init__(self):
        """Initialize cache"""
        self.cache: Dict[str, Any] = {}
        self.dependencies: Dict[str, List[str]] = {}

    def cache_pass_result(self, pass_name: str, result: Any, dependencies: List[str] = None) -> None:
        """Cache result of a pipeline pass

        Args:
            pass_name: Name of the pass
            result: Result to cache
            dependencies: List of pass names this depends on
        """
        self.cache[pass_name] = result
        if dependencies:
            self.dependencies[pass_name] = dependencies
        logger.debug(f"Cached result for pass: {pass_name}")

    def get_pass_result(self, pass_name: str) -> Any:
        """Get cached pass result

        Args:
            pass_name: Name of the pass

        Returns:
            Cached result or None
        """
        return self.cache.get(pass_name)

    def has_pass_result(self, pass_name: str) -> bool:
        """Check if pass result is cached

        Args:
            pass_name: Name of the pass

        Returns:
            True if cached
        """
        return pass_name in self.cache

    def invalidate_dependent_passes(self, pass_name: str) -> None:
        """Invalidate passes that depend on this pass

        Args:
            pass_name: Name of the pass that changed
        """
        to_invalidate = []
        for dependent, deps in self.dependencies.items():
            if pass_name in deps:
                to_invalidate.append(dependent)

        for pass_to_invalidate in to_invalidate:
            if pass_to_invalidate in self.cache:
                del self.cache[pass_to_invalidate]
                logger.debug(f"Invalidated dependent pass: {pass_to_invalidate}")
                # Recursively invalidate
                self.invalidate_dependent_passes(pass_to_invalidate)

    def clear(self) -> None:
        """Clear all cached results"""
        self.cache.clear()
        self.dependencies.clear()
        logger.debug("Cleared pass results cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict with cache stats
        """
        return {
            'cached_passes': list(self.cache.keys()),
            'num_cached': len(self.cache),
            'dependencies': dict(self.dependencies)
        }
