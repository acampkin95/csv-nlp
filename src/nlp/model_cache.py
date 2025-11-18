"""
Global NLP Model Cache
Thread-safe singleton cache for expensive NLP model initialization.
Provides 5-10 second speedup per analysis by caching models in memory.
"""

import threading
import logging
from typing import Dict, Any, Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe singleton cache for NLP models and compiled patterns"""

    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _init_lock = threading.Lock()  # Separate lock for model initialization

    def __new__(cls):
        """Implement singleton pattern with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    logger.info("Initialized ModelCache singleton")
        return cls._instance

    def get_or_load(self, model_name: str, loader_func: Callable, *args, **kwargs) -> Any:
        """Get cached model or load if not present

        Args:
            model_name: Unique identifier for the model
            loader_func: Function to call if model not cached
            *args: Arguments to pass to loader_func
            **kwargs: Keyword arguments to pass to loader_func

        Returns:
            The cached or newly loaded model
        """
        # Fast path: check if model already cached (no lock needed for read)
        if model_name in self._models:
            logger.debug(f"Cache HIT: {model_name}")
            return self._models[model_name]

        # Slow path: model not cached, need to load it
        with self._init_lock:
            # Double-check after acquiring lock (another thread might have loaded it)
            if model_name in self._models:
                logger.debug(f"Cache HIT (after lock): {model_name}")
                return self._models[model_name]

            # Load the model
            logger.info(f"Cache MISS: Loading {model_name}...")
            try:
                model = loader_func(*args, **kwargs)
                self._models[model_name] = model
                logger.info(f"Successfully cached {model_name}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache without loading

        Args:
            model_name: Model identifier

        Returns:
            Cached model or None if not found
        """
        return self._models.get(model_name)

    def clear(self, model_name: Optional[str] = None):
        """Clear cached models

        Args:
            model_name: Specific model to clear, or None to clear all
        """
        with self._init_lock:
            if model_name:
                if model_name in self._models:
                    del self._models[model_name]
                    logger.info(f"Cleared {model_name} from cache")
            else:
                self._models.clear()
                logger.info("Cleared all models from cache")

    def list_cached_models(self) -> list:
        """List all cached model names

        Returns:
            List of cached model names
        """
        return list(self._models.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict with cache statistics
        """
        return {
            'cached_models': self.list_cached_models(),
            'model_count': len(self._models),
            'memory_efficient': True  # Models shared across all instances
        }


# Convenience functions for common NLP models

def load_vader_analyzer():
    """Load VADER sentiment analyzer"""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.error("VADER not installed. Install with: pip install vaderSentiment")
        return None


def load_patterns_file(patterns_file: Optional[str] = None, pattern_type: str = 'grooming') -> Dict:
    """Load patterns from JSON file

    Args:
        patterns_file: Path to patterns JSON file
        pattern_type: Type of patterns to load (grooming, manipulation, etc.)

    Returns:
        Dict of patterns
    """
    import json

    if patterns_file is None:
        # Use default patterns file in same directory
        patterns_file = Path(__file__).parent / "patterns.json"
    else:
        patterns_file = Path(patterns_file)

    try:
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)
            return all_patterns.get(pattern_type, {})
    except FileNotFoundError:
        logger.warning(f"Patterns file not found: {patterns_file}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in patterns file: {e}")
        return {}


def compile_regex_patterns(patterns_dict: Dict) -> Dict:
    """Compile regex patterns for efficiency

    Args:
        patterns_dict: Dictionary of pattern categories and patterns

    Returns:
        Dictionary with compiled regex patterns
    """
    import re

    compiled = {}

    for category, patterns in patterns_dict.items():
        compiled[category] = []

        # Handle different pattern formats
        if isinstance(patterns, list):
            for pattern_item in patterns:
                try:
                    if isinstance(pattern_item, dict) and 'regex' in pattern_item:
                        # Format: {"regex": "...", "severity": 0.8}
                        regex = re.compile(pattern_item['regex'], re.IGNORECASE)
                        severity = pattern_item.get('severity', 0.5)
                        description = pattern_item.get('description', '')
                        compiled[category].append((regex, severity, description))
                    elif isinstance(pattern_item, str):
                        # Format: raw regex string
                        regex = re.compile(pattern_item, re.IGNORECASE)
                        compiled[category].append((regex, 0.5, ''))
                except re.error as e:
                    logger.error(f"Invalid regex in {category}: {pattern_item} - {e}")

    return compiled


# Global cache instance
_cache = ModelCache()


def get_cache() -> ModelCache:
    """Get global model cache instance

    Returns:
        ModelCache singleton instance
    """
    return _cache
