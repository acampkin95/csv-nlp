#!/usr/bin/env python3
"""
Base Pass Class

Abstract base class for all pipeline passes with standard functionality:
- Error recovery
- Result caching
- Progress tracking
- Logging
- Dependency management
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PassGroup(Enum):
    """Pass group classification"""
    NORMALIZATION = "normalization"  # Passes 1-3
    BEHAVIORAL = "behavioral"  # Passes 4-6
    COMMUNICATION = "communication"  # Passes 7-8
    TIMELINE = "timeline"  # Passes 9-10
    PERSON_CENTRIC = "person_centric"  # Passes 11-15


@dataclass
class PassResult:
    """Standard pass result container"""
    pass_number: int
    pass_name: str
    data: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    cached: bool = False


class BasePass(ABC):
    """
    Abstract base class for all pipeline passes.

    Provides standard functionality:
    - Error recovery with graceful degradation
    - Result caching with dependency tracking
    - Progress tracking for long operations
    - Structured logging
    - Standard interface
    """

    def __init__(
        self,
        pass_number: int,
        pass_name: str,
        pass_group: PassGroup,
        cache_manager: Optional[Any] = None,
        dependencies: Optional[List[str]] = None
    ):
        """
        Initialize base pass.

        Args:
            pass_number: Pass number (1-15)
            pass_name: Human-readable pass name
            pass_group: Pass group classification
            cache_manager: Optional cache manager for result caching
            dependencies: Optional list of pass names this pass depends on
        """
        self.pass_number = pass_number
        self.pass_name = pass_name
        self.pass_group = pass_group
        self.cache_manager = cache_manager
        self.dependencies = dependencies or []
        self.logger = logging.getLogger(f"{__name__}.{pass_name}")

    @property
    def cache_key(self) -> str:
        """Get cache key for this pass"""
        return self.pass_name.lower().replace(' ', '_').replace('-', '_')

    def execute(self, **kwargs) -> PassResult:
        """
        Execute the pass with standard error handling and caching.

        Args:
            **kwargs: Pass-specific arguments

        Returns:
            PassResult: Standardized pass result
        """
        import time

        # Check cache first
        if self.cache_manager and self.cache_manager.has_pass_result(self.cache_key):
            self.logger.info(f"  Using cached {self.pass_name} results")
            cached_data = self.cache_manager.get_pass_result(self.cache_key)
            return PassResult(
                pass_number=self.pass_number,
                pass_name=self.pass_name,
                data=cached_data,
                success=True,
                cached=True
            )

        # Execute pass
        start_time = time.time()
        self.logger.info(f"Executing {self.pass_name}...")

        try:
            # Call the specific pass implementation
            result_data = self._execute_pass(**kwargs)
            execution_time = time.time() - start_time

            # Cache result if cache manager available
            if self.cache_manager:
                self.cache_manager.cache_pass_result(
                    self.cache_key,
                    result_data,
                    dependencies=self.dependencies
                )

            self.logger.info(f"  {self.pass_name} completed in {execution_time:.2f}s")

            return PassResult(
                pass_number=self.pass_number,
                pass_name=self.pass_name,
                data=result_data,
                success=True,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"  {self.pass_name} failed: {e}", exc_info=True)

            # Return fallback result
            fallback_data = self._get_fallback_result()

            return PassResult(
                pass_number=self.pass_number,
                pass_name=self.pass_name,
                data=fallback_data,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    @abstractmethod
    def _execute_pass(self, **kwargs) -> Dict[str, Any]:
        """
        Actual pass implementation - must be overridden by subclasses.

        Args:
            **kwargs: Pass-specific arguments

        Returns:
            Dict containing pass results
        """
        pass

    def _get_fallback_result(self) -> Dict[str, Any]:
        """
        Get fallback result for when pass fails.

        Override this in subclasses to provide sensible defaults.

        Returns:
            Dict with fallback/empty results
        """
        return {'error': 'Pass execution failed'}

    def validate_dependencies(self, available_results: Dict[str, Any]) -> bool:
        """
        Validate that all dependencies are met.

        Args:
            available_results: Dict of available results from prior passes

        Returns:
            bool: True if all dependencies are met
        """
        for dep in self.dependencies:
            if dep not in available_results:
                self.logger.warning(f"Missing dependency: {dep}")
                return False
            if isinstance(available_results[dep], dict) and 'error' in available_results[dep]:
                self.logger.warning(f"Dependency {dep} has errors")
                return False
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} pass={self.pass_number} name='{self.pass_name}'>"
