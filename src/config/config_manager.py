"""
Configuration Manager for Message Processor
Handles loading, validation, and management of JSON configuration files
with support for presets, defaults, and runtime overrides.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Analysis configuration settings"""
    workers: int = 4
    deduplication: bool = True
    timeline_bin_size: str = "day"  # hour, day, week, month
    top_n_results: int = 10
    psychological_analysis: str = "each"  # each, aggregated, none
    cache_features: bool = True
    batch_size: int = 100


@dataclass
class NLPConfig:
    """NLP processing configuration"""
    # Basic features
    enable_vader: bool = True
    enable_nrclex: bool = True
    enable_nltk: bool = True

    # Advanced features
    enable_grooming_detection: bool = True
    enable_manipulation_detection: bool = True
    enable_deception_markers: bool = True
    enable_intent_classification: bool = True
    enable_textblob: bool = True

    # Topic modeling
    topic_modeling_enabled: bool = True
    topic_modeling_num_topics: int = 15
    topic_modeling_algorithm: str = "lda"  # lda, nmf

    # Risk scoring weights
    risk_weight_grooming: float = 0.3
    risk_weight_manipulation: float = 0.3
    risk_weight_hostility: float = 0.2
    risk_weight_deception: float = 0.2

    # Risk thresholds
    risk_threshold_low: float = 0.3
    risk_threshold_moderate: float = 0.5
    risk_threshold_high: float = 0.7
    risk_threshold_critical: float = 0.85

    # Pattern detection thresholds
    min_pattern_confidence: float = 0.5
    min_pattern_severity: float = 0.3


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    dpi: int = 300
    format: str = "png"  # png, svg, pdf
    color_scheme: str = "clinical"  # clinical, colorful, grayscale
    figure_width: int = 12
    figure_height: int = 8
    show_grid: bool = True
    show_legend: bool = True


@dataclass
class PDFConfig:
    """PDF report configuration"""
    template: str = "clinical"  # clinical, legal, personal, research
    include_visualizations: bool = True
    include_citations: bool = True
    include_raw_data: bool = False
    font_size: int = 10
    font_family: str = "Helvetica"
    page_size: str = "A4"  # A4, Letter, Legal
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0
    })


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "data/analysis.db"
    enable_caching: bool = True
    connection_timeout: int = 30
    vacuum_on_close: bool = False
    backup_before_analysis: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file: str = "logs/analysis.log"
    rotation: str = "daily"  # daily, size, none
    max_bytes: int = 10485760  # 10MB for size-based rotation
    backup_count: int = 7
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Configuration:
    """Complete configuration container"""
    version: str = "2.0"
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Configuration':
        """Create configuration from dictionary"""
        config = cls()

        # Parse each section
        if 'analysis' in data:
            config.analysis = AnalysisConfig(**data['analysis'])
        if 'nlp' in data:
            # Handle nested dictionaries in NLP config
            nlp_data = data['nlp'].copy()

            # Flatten topic modeling settings
            if 'topic_modeling' in nlp_data:
                tm = nlp_data.pop('topic_modeling')
                nlp_data['topic_modeling_enabled'] = tm.get('enabled', True)
                nlp_data['topic_modeling_num_topics'] = tm.get('num_topics', 15)
                nlp_data['topic_modeling_algorithm'] = tm.get('algorithm', 'lda')

            # Flatten risk scoring settings
            if 'risk_scoring' in nlp_data:
                rs = nlp_data.pop('risk_scoring')
                if 'weights' in rs:
                    for key, value in rs['weights'].items():
                        nlp_data[f'risk_weight_{key}'] = value
                if 'thresholds' in rs:
                    for key, value in rs['thresholds'].items():
                        nlp_data[f'risk_threshold_{key}'] = value

            config.nlp = NLPConfig(**nlp_data)

        if 'visualization' in data:
            config.visualization = VisualizationConfig(**data['visualization'])
        if 'pdf' in data:
            config.pdf = PDFConfig(**data['pdf'])
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])

        return config


class ConfigManager:
    """Manages configuration loading, saving, and presets"""

    DEFAULT_CONFIG_DIR = Path("config")
    DEFAULT_CONFIG_FILE = "default.json"
    PRESETS_DIR = "presets"

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir = self.config_dir / self.PRESETS_DIR
        self.presets_dir.mkdir(exist_ok=True)

        # Create default configuration if it doesn't exist
        self._ensure_default_config()

        # Create preset configurations
        self._create_presets()

    def _ensure_default_config(self):
        """Ensure default configuration exists"""
        default_path = self.config_dir / self.DEFAULT_CONFIG_FILE

        if not default_path.exists():
            default_config = Configuration()
            self.save_config(default_config, default_path)
            logger.info(f"Created default configuration at {default_path}")

    def _create_presets(self):
        """Create preset configurations"""
        presets = {
            "quick_analysis": self._create_quick_analysis_preset(),
            "deep_analysis": self._create_deep_analysis_preset(),
            "clinical_report": self._create_clinical_preset(),
            "legal_report": self._create_legal_preset(),
            "research": self._create_research_preset(),
        }

        for name, config in presets.items():
            preset_path = self.presets_dir / f"{name}.json"
            if not preset_path.exists():
                self.save_config(config, preset_path)
                logger.info(f"Created preset '{name}' at {preset_path}")

    def _create_quick_analysis_preset(self) -> Configuration:
        """Create quick analysis preset (fast, basic features)"""
        config = Configuration()
        config.analysis.workers = 8
        config.analysis.psychological_analysis = "none"
        config.nlp.enable_grooming_detection = False
        config.nlp.enable_manipulation_detection = False
        config.nlp.enable_deception_markers = False
        config.nlp.topic_modeling_enabled = False
        config.visualization.dpi = 150
        config.pdf.include_visualizations = False
        return config

    def _create_deep_analysis_preset(self) -> Configuration:
        """Create deep analysis preset (thorough, all features)"""
        config = Configuration()
        config.analysis.workers = 4
        config.analysis.psychological_analysis = "each"
        config.analysis.top_n_results = 20
        config.nlp.topic_modeling_num_topics = 20
        config.nlp.min_pattern_confidence = 0.3
        config.visualization.dpi = 300
        config.pdf.include_raw_data = True
        return config

    def _create_clinical_preset(self) -> Configuration:
        """Create clinical report preset"""
        config = Configuration()
        config.pdf.template = "clinical"
        config.pdf.include_citations = True
        config.nlp.risk_threshold_critical = 0.8
        config.visualization.color_scheme = "clinical"
        return config

    def _create_legal_preset(self) -> Configuration:
        """Create legal report preset"""
        config = Configuration()
        config.pdf.template = "legal"
        config.pdf.include_raw_data = True
        config.pdf.page_size = "Legal"
        config.analysis.deduplication = False  # Keep all evidence
        config.database.backup_before_analysis = True
        return config

    def _create_research_preset(self) -> Configuration:
        """Create research preset"""
        config = Configuration()
        config.pdf.template = "research"
        config.pdf.include_citations = True
        config.nlp.topic_modeling_enabled = True
        config.nlp.topic_modeling_num_topics = 25
        config.analysis.psychological_analysis = "each"
        config.visualization.format = "svg"  # Vector for publications
        return config

    def load_config(self, config_path: Optional[str] = None) -> Configuration:
        """Load configuration from file

        Args:
            config_path: Path to configuration file (or preset name)

        Returns:
            Configuration: Loaded configuration
        """
        if config_path is None:
            config_path = self.config_dir / self.DEFAULT_CONFIG_FILE
        else:
            config_path = Path(config_path)

            # Check if it's a preset name
            if not config_path.exists() and not config_path.suffix:
                preset_path = self.presets_dir / f"{config_path}.json"
                if preset_path.exists():
                    config_path = preset_path
                else:
                    raise FileNotFoundError(f"Configuration file or preset not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                config = Configuration.from_dict(data)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return Configuration()

    def save_config(self, config: Configuration, config_path: Optional[str] = None):
        """Save configuration to file

        Args:
            config: Configuration to save
            config_path: Path to save to (defaults to default config)
        """
        if config_path is None:
            config_path = self.config_dir / self.DEFAULT_CONFIG_FILE
        else:
            config_path = Path(config_path)

        try:
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
                logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    def merge_configs(self, base: Configuration, override: Dict[str, Any]) -> Configuration:
        """Merge configuration with overrides

        Args:
            base: Base configuration
            override: Dictionary of overrides

        Returns:
            Configuration: Merged configuration
        """
        # Convert base to dictionary
        config_dict = base.to_dict()

        # Deep merge with overrides
        config_dict = self._deep_merge(config_dict, override)

        # Create new configuration
        return Configuration.from_dict(config_dict)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Dict: Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def list_presets(self) -> List[str]:
        """List available presets

        Returns:
            List[str]: List of preset names
        """
        presets = []
        for preset_file in self.presets_dir.glob("*.json"):
            presets.append(preset_file.stem)
        return sorted(presets)

    def validate_config(self, config: Configuration) -> List[str]:
        """Validate configuration for issues

        Args:
            config: Configuration to validate

        Returns:
            List[str]: List of validation issues (empty if valid)
        """
        issues = []

        # Validate workers
        if config.analysis.workers < 1 or config.analysis.workers > 32:
            issues.append(f"Invalid worker count: {config.analysis.workers} (must be 1-32)")

        # Validate bin size
        valid_bins = ["hour", "day", "week", "month"]
        if config.analysis.timeline_bin_size not in valid_bins:
            issues.append(f"Invalid bin size: {config.analysis.timeline_bin_size}")

        # Validate risk weights sum to 1.0
        weight_sum = (
            config.nlp.risk_weight_grooming +
            config.nlp.risk_weight_manipulation +
            config.nlp.risk_weight_hostility +
            config.nlp.risk_weight_deception
        )
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(f"Risk weights must sum to 1.0 (current: {weight_sum})")

        # Validate thresholds
        thresholds = [
            config.nlp.risk_threshold_low,
            config.nlp.risk_threshold_moderate,
            config.nlp.risk_threshold_high,
            config.nlp.risk_threshold_critical
        ]
        if not all(0 <= t <= 1 for t in thresholds):
            issues.append("Risk thresholds must be between 0 and 1")
        if thresholds != sorted(thresholds):
            issues.append("Risk thresholds must be in ascending order")

        # Validate DPI
        if config.visualization.dpi < 72 or config.visualization.dpi > 600:
            issues.append(f"Invalid DPI: {config.visualization.dpi} (must be 72-600)")

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.level not in valid_levels:
            issues.append(f"Invalid log level: {config.logging.level}")

        return issues


def get_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Configuration:
    """Convenience function to get configuration

    Args:
        config_path: Path to config file or preset name
        overrides: Dictionary of overrides

    Returns:
        Configuration: Loaded and validated configuration
    """
    manager = ConfigManager()
    config = manager.load_config(config_path)

    if overrides:
        config = manager.merge_configs(config, overrides)

    issues = manager.validate_config(config)
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")

    return config