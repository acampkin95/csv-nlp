#!/usr/bin/env python3
"""
Pass Registry

Manages pass registration, execution order, and dependency resolution.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base_pass import BasePass, PassResult, PassGroup

logger = logging.getLogger(__name__)


@dataclass
class PassMetadata:
    """Metadata about a registered pass"""
    pass_instance: BasePass
    pass_number: int
    pass_name: str
    pass_group: PassGroup
    dependencies: List[str] = field(default_factory=list)
    parallel_group: Optional[int] = None  # Passes with same parallel_group can run in parallel


class PassRegistry:
    """
    Registry for managing pipeline passes.

    Handles:
    - Pass registration
    - Execution order
    - Dependency resolution
    - Parallel execution groups
    """

    def __init__(self):
        """Initialize empty registry"""
        self._passes: Dict[int, PassMetadata] = {}
        self._pass_by_name: Dict[str, PassMetadata] = {}
        self._parallel_groups: Dict[int, List[PassMetadata]] = {}

    def register(
        self,
        pass_instance: BasePass,
        parallel_group: Optional[int] = None
    ) -> None:
        """
        Register a pass in the registry.

        Args:
            pass_instance: The pass instance to register
            parallel_group: Optional parallel group ID for concurrent execution
        """
        metadata = PassMetadata(
            pass_instance=pass_instance,
            pass_number=pass_instance.pass_number,
            pass_name=pass_instance.pass_name,
            pass_group=pass_instance.pass_group,
            dependencies=pass_instance.dependencies,
            parallel_group=parallel_group
        )

        self._passes[pass_instance.pass_number] = metadata
        self._pass_by_name[pass_instance.pass_name] = metadata

        # Track parallel groups
        if parallel_group is not None:
            if parallel_group not in self._parallel_groups:
                self._parallel_groups[parallel_group] = []
            self._parallel_groups[parallel_group].append(metadata)

        logger.debug(f"Registered {pass_instance}")

    def get_pass(self, pass_number: int) -> Optional[BasePass]:
        """Get pass by number"""
        metadata = self._passes.get(pass_number)
        return metadata.pass_instance if metadata else None

    def get_pass_by_name(self, pass_name: str) -> Optional[BasePass]:
        """Get pass by name"""
        metadata = self._pass_by_name.get(pass_name)
        return metadata.pass_instance if metadata else None

    def get_execution_order(self) -> List[int]:
        """
        Get pass numbers in execution order.

        Returns:
            List of pass numbers sorted by execution order
        """
        return sorted(self._passes.keys())

    def get_parallel_groups(self) -> Dict[int, List[PassMetadata]]:
        """
        Get passes grouped by parallel execution group.

        Returns:
            Dict mapping parallel group ID to list of passes
        """
        return self._parallel_groups.copy()

    def validate_dependencies(self) -> bool:
        """
        Validate that all pass dependencies are met.

        Returns:
            bool: True if all dependencies are valid
        """
        all_pass_names = set(self._pass_by_name.keys())

        for metadata in self._passes.values():
            for dep in metadata.dependencies:
                if dep not in all_pass_names:
                    logger.error(f"Pass '{metadata.pass_name}' depends on unknown pass '{dep}'")
                    return False

                # Check that dependency comes before this pass
                dep_metadata = self._pass_by_name[dep]
                if dep_metadata.pass_number >= metadata.pass_number:
                    logger.error(
                        f"Pass '{metadata.pass_name}' (#{metadata.pass_number}) "
                        f"depends on '{dep}' (#{dep_metadata.pass_number}) which comes after it"
                    )
                    return False

        logger.info("All pass dependencies validated successfully")
        return True

    def get_passes_by_group(self, group: PassGroup) -> List[PassMetadata]:
        """
        Get all passes in a specific group.

        Args:
            group: The PassGroup to filter by

        Returns:
            List of PassMetadata for passes in the group
        """
        return [
            metadata for metadata in self._passes.values()
            if metadata.pass_group == group
        ]

    def execute_all(self, **kwargs) -> Dict[int, PassResult]:
        """
        Execute all registered passes in order.

        Args:
            **kwargs: Arguments to pass to each pass

        Returns:
            Dict mapping pass number to PassResult
        """
        results: Dict[int, PassResult] = {}
        available_results: Dict[str, Any] = {}

        for pass_num in self.get_execution_order():
            metadata = self._passes[pass_num]
            pass_instance = metadata.pass_instance

            logger.info(f"\n[PASS {pass_num}] {metadata.pass_name}")
            logger.info("-" * 70)

            # Validate dependencies
            if not pass_instance.validate_dependencies(available_results):
                logger.warning(f"Skipping {metadata.pass_name} due to missing dependencies")
                results[pass_num] = PassResult(
                    pass_number=pass_num,
                    pass_name=metadata.pass_name,
                    data={},
                    success=False,
                    error="Missing dependencies"
                )
                continue

            # Execute pass
            result = pass_instance.execute(**kwargs)
            results[pass_num] = result

            # Make result available for dependent passes
            available_results[pass_instance.cache_key] = result.data

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with statistics about registered passes
        """
        group_counts = {}
        for group in PassGroup:
            group_counts[group.value] = len(self.get_passes_by_group(group))

        return {
            'total_passes': len(self._passes),
            'passes_by_group': group_counts,
            'parallel_groups': len(self._parallel_groups),
            'passes_with_dependencies': sum(
                1 for m in self._passes.values() if m.dependencies
            )
        }

    def __len__(self) -> int:
        """Get number of registered passes"""
        return len(self._passes)

    def __repr__(self) -> str:
        return f"<PassRegistry passes={len(self._passes)} groups={len(self._parallel_groups)}>"
