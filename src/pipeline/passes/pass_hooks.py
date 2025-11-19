#!/usr/bin/env python3
"""
Pass Execution Hooks

Hook system for injecting custom logic before/after pass execution.
Enables conditional execution, logging, validation, and custom processing.
"""

import logging
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .base_pass import BasePass, PassResult

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Hook execution types"""
    BEFORE_PASS = "before_pass"
    AFTER_PASS = "after_pass"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ON_SKIP = "on_skip"


@dataclass
class HookContext:
    """Context provided to hooks"""
    pass_instance: BasePass
    pass_number: int
    pass_name: str
    kwargs: Dict[str, Any]
    result: Optional[PassResult] = None
    should_skip: bool = False
    skip_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PassHookManager:
    """
    Manages execution hooks for pipeline passes.

    Features:
    - Before/after pass execution hooks
    - Success/failure specific hooks
    - Conditional execution based on hook results
    - Hook chaining
    - Global and pass-specific hooks
    """

    def __init__(self):
        """Initialize hook manager"""
        self.global_hooks: Dict[HookType, List[Callable]] = {
            hook_type: [] for hook_type in HookType
        }
        self.pass_specific_hooks: Dict[int, Dict[HookType, List[Callable]]] = {}

    def register_global_hook(self, hook_type: HookType, hook_func: Callable):
        """
        Register a global hook that applies to all passes.

        Args:
            hook_type: Type of hook
            hook_func: Hook function with signature: func(context: HookContext) -> Optional[bool]
                      Should return True to continue, False to skip pass

        Example:
            def log_before_pass(context):
                print(f"Executing {context.pass_name}")
                return True  # Continue execution

            manager.register_global_hook(HookType.BEFORE_PASS, log_before_pass)
        """
        self.global_hooks[hook_type].append(hook_func)
        logger.debug(f"Registered global hook for {hook_type.value}")

    def register_pass_hook(
        self,
        pass_number: int,
        hook_type: HookType,
        hook_func: Callable
    ):
        """
        Register a hook for a specific pass.

        Args:
            pass_number: Pass number to hook into
            hook_type: Type of hook
            hook_func: Hook function
        """
        if pass_number not in self.pass_specific_hooks:
            self.pass_specific_hooks[pass_number] = {
                hook_type: [] for hook_type in HookType
            }

        self.pass_specific_hooks[pass_number][hook_type].append(hook_func)
        logger.debug(f"Registered hook for pass {pass_number}, type {hook_type.value}")

    def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext
    ) -> bool:
        """
        Execute all hooks of a specific type.

        Args:
            hook_type: Type of hooks to execute
            context: Hook context

        Returns:
            bool: True to continue, False to skip
        """
        # Execute global hooks
        for hook_func in self.global_hooks[hook_type]:
            try:
                result = hook_func(context)
                # If hook returns False, skip execution
                if result is False:
                    context.should_skip = True
                    if not context.skip_reason:
                        context.skip_reason = f"Skipped by {hook_type.value} hook"
                    return False
            except Exception as e:
                logger.error(f"Error in global {hook_type.value} hook: {e}", exc_info=True)

        # Execute pass-specific hooks
        if context.pass_number in self.pass_specific_hooks:
            for hook_func in self.pass_specific_hooks[context.pass_number][hook_type]:
                try:
                    result = hook_func(context)
                    if result is False:
                        context.should_skip = True
                        if not context.skip_reason:
                            context.skip_reason = f"Skipped by pass-specific {hook_type.value} hook"
                        return False
                except Exception as e:
                    logger.error(
                        f"Error in pass {context.pass_number} {hook_type.value} hook: {e}",
                        exc_info=True
                    )

        return True

    def clear_hooks(self, pass_number: Optional[int] = None):
        """
        Clear hooks.

        Args:
            pass_number: If specified, clear only hooks for this pass.
                        If None, clear all hooks.
        """
        if pass_number is None:
            self.global_hooks = {hook_type: [] for hook_type in HookType}
            self.pass_specific_hooks = {}
            logger.info("Cleared all hooks")
        else:
            if pass_number in self.pass_specific_hooks:
                del self.pass_specific_hooks[pass_number]
                logger.info(f"Cleared hooks for pass {pass_number}")

    def get_hook_count(self, pass_number: Optional[int] = None) -> Dict[str, int]:
        """
        Get count of registered hooks.

        Args:
            pass_number: Optional pass number to filter

        Returns:
            Dict with hook counts by type
        """
        if pass_number is None:
            return {
                hook_type.value: len(hooks)
                for hook_type, hooks in self.global_hooks.items()
            }
        else:
            if pass_number not in self.pass_specific_hooks:
                return {hook_type.value: 0 for hook_type in HookType}

            return {
                hook_type.value: len(hooks)
                for hook_type, hooks in self.pass_specific_hooks[pass_number].items()
            }


# Feature 1: Conditional Pass Execution
class ConditionalExecutionManager:
    """
    Manages conditional pass execution based on rules.

    Allows skipping passes based on:
    - Previous pass results
    - Data characteristics
    - Custom conditions
    - Configuration
    """

    def __init__(self, hook_manager: PassHookManager):
        """
        Initialize conditional execution manager.

        Args:
            hook_manager: Hook manager to integrate with
        """
        self.hook_manager = hook_manager
        self.conditions: Dict[int, List[Callable]] = {}

    def add_condition(self, pass_number: int, condition_func: Callable):
        """
        Add a condition for pass execution.

        Args:
            pass_number: Pass number
            condition_func: Function that returns True to execute, False to skip
                           Signature: func(context: HookContext) -> bool

        Example:
            # Only run grooming detection if messages > 10
            def require_min_messages(context):
                messages = context.kwargs.get('messages', [])
                if len(messages) < 10:
                    context.skip_reason = "Too few messages for grooming detection"
                    return False
                return True

            manager.add_condition(4, require_min_messages)
        """
        if pass_number not in self.conditions:
            self.conditions[pass_number] = []

        self.conditions[pass_number].append(condition_func)

        # Register as a before_pass hook
        self.hook_manager.register_pass_hook(
            pass_number,
            HookType.BEFORE_PASS,
            condition_func
        )

        logger.info(f"Added condition for pass {pass_number}")

    def add_dependency_condition(
        self,
        pass_number: int,
        required_passes: List[int],
        require_success: bool = True
    ):
        """
        Add condition that requires other passes to succeed.

        Args:
            pass_number: Pass number
            required_passes: List of pass numbers that must complete
            require_success: Whether to require those passes to succeed
        """
        def dependency_check(context: HookContext) -> bool:
            # This would need access to previous results
            # Implementation depends on how results are passed
            return True  # Placeholder

        self.add_condition(pass_number, dependency_check)

    def add_data_condition(
        self,
        pass_number: int,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        required_fields: Optional[List[str]] = None
    ):
        """
        Add condition based on data characteristics.

        Args:
            pass_number: Pass number
            min_items: Minimum number of items required
            max_items: Maximum number of items allowed
            required_fields: Required fields in data
        """
        def data_check(context: HookContext) -> bool:
            messages = context.kwargs.get('messages', [])

            if min_items is not None and len(messages) < min_items:
                context.skip_reason = f"Insufficient items: {len(messages)} < {min_items}"
                return False

            if max_items is not None and len(messages) > max_items:
                context.skip_reason = f"Too many items: {len(messages)} > {max_items}"
                return False

            if required_fields:
                for field in required_fields:
                    if field not in context.kwargs:
                        context.skip_reason = f"Missing required field: {field}"
                        return False

            return True

        self.add_condition(pass_number, data_check)

    def clear_conditions(self, pass_number: Optional[int] = None):
        """Clear conditions for a pass or all passes"""
        if pass_number is None:
            self.conditions = {}
            logger.info("Cleared all conditions")
        else:
            if pass_number in self.conditions:
                del self.conditions[pass_number]
                logger.info(f"Cleared conditions for pass {pass_number}")


# Global hook manager instance
_global_hook_manager = PassHookManager()


def get_hook_manager() -> PassHookManager:
    """Get global hook manager instance"""
    return _global_hook_manager
