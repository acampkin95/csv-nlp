#!/usr/bin/env python3
"""
Pass Configuration System (Feature 3)

Flexible configuration system for pipeline passes.
Supports JSON/YAML configs, environment variables, and runtime overrides.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PassConfig:
    """Configuration for a single pass"""
    pass_number: int
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    cache_enabled: bool = True
    parallel_group: Optional[int] = None
    priority: int = 0  # Higher priority passes run first within a group
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PipelineConfig:
    """Configuration for entire pipeline"""
    pipeline_name: str = "default"
    enabled_passes: List[int] = field(default_factory=lambda: list(range(1, 16)))
    pass_configs: Dict[int, PassConfig] = field(default_factory=dict)
    global_timeout_seconds: Optional[int] = None
    fail_fast: bool = False  # Stop on first error
    max_parallel_passes: int = 3
    cache_enabled: bool = True
    profiling_enabled: bool = False
    metrics_enabled: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def get_pass_config(self, pass_number: int) -> PassConfig:
        """
        Get configuration for a specific pass.

        Args:
            pass_number: Pass number

        Returns:
            PassConfig for the pass
        """
        if pass_number not in self.pass_configs:
            # Create default config
            self.pass_configs[pass_number] = PassConfig(pass_number=pass_number)

        return self.pass_configs[pass_number]

    def set_pass_enabled(self, pass_number: int, enabled: bool):
        """Enable or disable a pass"""
        config = self.get_pass_config(pass_number)
        config.enabled = enabled

        if enabled and pass_number not in self.enabled_passes:
            self.enabled_passes.append(pass_number)
        elif not enabled and pass_number in self.enabled_passes:
            self.enabled_passes.remove(pass_number)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pipeline_name': self.pipeline_name,
            'enabled_passes': self.enabled_passes,
            'pass_configs': {
                num: config.to_dict()
                for num, config in self.pass_configs.items()
            },
            'global_timeout_seconds': self.global_timeout_seconds,
            'fail_fast': self.fail_fast,
            'max_parallel_passes': self.max_parallel_passes,
            'cache_enabled': self.cache_enabled,
            'profiling_enabled': self.profiling_enabled,
            'metrics_enabled': self.metrics_enabled,
            'custom_settings': self.custom_settings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary"""
        pass_configs = {
            int(num): PassConfig.from_dict(config)
            for num, config in data.get('pass_configs', {}).items()
        }

        return cls(
            pipeline_name=data.get('pipeline_name', 'default'),
            enabled_passes=data.get('enabled_passes', list(range(1, 16))),
            pass_configs=pass_configs,
            global_timeout_seconds=data.get('global_timeout_seconds'),
            fail_fast=data.get('fail_fast', False),
            max_parallel_passes=data.get('max_parallel_passes', 3),
            cache_enabled=data.get('cache_enabled', True),
            profiling_enabled=data.get('profiling_enabled', False),
            metrics_enabled=data.get('metrics_enabled', True),
            custom_settings=data.get('custom_settings', {})
        )


class PassConfigManager:
    """
    Manages pipeline configurations.

    Features:
    - Load/save configs from JSON/YAML
    - Environment variable overrides
    - Runtime configuration updates
    - Multiple named configurations
    - Configuration validation
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir or os.getcwd())
        self.configs: Dict[str, PipelineConfig] = {}
        self.current_config_name: Optional[str] = None

    def create_config(self, name: str = "default") -> PipelineConfig:
        """
        Create a new configuration.

        Args:
            name: Configuration name

        Returns:
            PipelineConfig
        """
        config = PipelineConfig(pipeline_name=name)
        self.configs[name] = config
        if self.current_config_name is None:
            self.current_config_name = name
        return config

    def load_from_file(self, filepath: str) -> PipelineConfig:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to config file

        Returns:
            PipelineConfig
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            if filepath.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

        config = PipelineConfig.from_dict(data)
        self.configs[config.pipeline_name] = config

        logger.info(f"Loaded configuration '{config.pipeline_name}' from {filepath}")
        return config

    def save_to_file(self, config_name: str, filepath: str):
        """
        Save configuration to JSON file.

        Args:
            config_name: Name of configuration to save
            filepath: Output file path
        """
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")

        config = self.configs[config_name]
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved configuration '{config_name}' to {filepath}")

    def get_config(self, name: Optional[str] = None) -> PipelineConfig:
        """
        Get configuration by name.

        Args:
            name: Configuration name (None = current)

        Returns:
            PipelineConfig
        """
        if name is None:
            name = self.current_config_name

        if name is None or name not in self.configs:
            # Create default config
            return self.create_config("default")

        return self.configs[name]

    def set_current_config(self, name: str):
        """Set current active configuration"""
        if name not in self.configs:
            raise ValueError(f"Configuration '{name}' not found")

        self.current_config_name = name
        logger.info(f"Set current configuration to '{name}'")

    def apply_env_overrides(self, config: PipelineConfig):
        """
        Apply environment variable overrides to configuration.

        Environment variables:
        - PIPELINE_CACHE_ENABLED=true/false
        - PIPELINE_PROFILING_ENABLED=true/false
        - PIPELINE_METRICS_ENABLED=true/false
        - PIPELINE_FAIL_FAST=true/false
        - PIPELINE_GLOBAL_TIMEOUT=seconds
        - PASS_{N}_ENABLED=true/false
        - PASS_{N}_TIMEOUT=seconds

        Args:
            config: Configuration to modify
        """
        # Global settings
        if 'PIPELINE_CACHE_ENABLED' in os.environ:
            config.cache_enabled = os.environ['PIPELINE_CACHE_ENABLED'].lower() == 'true'

        if 'PIPELINE_PROFILING_ENABLED' in os.environ:
            config.profiling_enabled = os.environ['PIPELINE_PROFILING_ENABLED'].lower() == 'true'

        if 'PIPELINE_METRICS_ENABLED' in os.environ:
            config.metrics_enabled = os.environ['PIPELINE_METRICS_ENABLED'].lower() == 'true'

        if 'PIPELINE_FAIL_FAST' in os.environ:
            config.fail_fast = os.environ['PIPELINE_FAIL_FAST'].lower() == 'true'

        if 'PIPELINE_GLOBAL_TIMEOUT' in os.environ:
            config.global_timeout_seconds = int(os.environ['PIPELINE_GLOBAL_TIMEOUT'])

        # Per-pass settings
        for pass_num in range(1, 16):
            enabled_var = f'PASS_{pass_num}_ENABLED'
            if enabled_var in os.environ:
                config.set_pass_enabled(
                    pass_num,
                    os.environ[enabled_var].lower() == 'true'
                )

            timeout_var = f'PASS_{pass_num}_TIMEOUT'
            if timeout_var in os.environ:
                pass_config = config.get_pass_config(pass_num)
                pass_config.timeout_seconds = int(os.environ[timeout_var])

        logger.info("Applied environment variable overrides to configuration")

    def create_preset_config(self, preset: str) -> PipelineConfig:
        """
        Create a preset configuration.

        Args:
            preset: Preset name ('fast', 'thorough', 'minimal', 'custom')

        Returns:
            PipelineConfig
        """
        if preset == 'fast':
            # Skip slower passes, use caching aggressively
            config = PipelineConfig(
                pipeline_name='fast',
                enabled_passes=[1, 2, 4, 5, 8, 11],
                cache_enabled=True,
                profiling_enabled=False
            )

        elif preset == 'thorough':
            # All passes enabled, no skipping
            config = PipelineConfig(
                pipeline_name='thorough',
                enabled_passes=list(range(1, 16)),
                cache_enabled=False,  # Force re-computation
                profiling_enabled=True,
                fail_fast=False
            )

        elif preset == 'minimal':
            # Only essential passes
            config = PipelineConfig(
                pipeline_name='minimal',
                enabled_passes=[1, 2, 8],  # Validation, sentiment, risk
                cache_enabled=True,
                profiling_enabled=False
            )

        else:
            raise ValueError(f"Unknown preset: {preset}")

        self.configs[config.pipeline_name] = config
        return config

    def validate_config(self, config: PipelineConfig) -> List[str]:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for invalid pass numbers
        for pass_num in config.enabled_passes:
            if pass_num < 1 or pass_num > 15:
                errors.append(f"Invalid pass number: {pass_num}")

        # Check for dependency issues
        dependencies = {
            3: [2],
            8: [4, 5, 6, 7],
            9: [8],
            10: [2, 9],
            12: [11],
            13: [5, 11],
            14: [11],
            15: [8, 11, 13, 14]
        }

        for pass_num, deps in dependencies.items():
            if pass_num in config.enabled_passes:
                for dep in deps:
                    if dep not in config.enabled_passes:
                        errors.append(
                            f"Pass {pass_num} requires pass {dep} which is not enabled"
                        )

        return errors

    def list_configs(self) -> List[str]:
        """List all available configurations"""
        return list(self.configs.keys())


# Global config manager instance
_global_config_manager = PassConfigManager()


def get_config_manager() -> PassConfigManager:
    """Get global config manager instance"""
    return _global_config_manager
