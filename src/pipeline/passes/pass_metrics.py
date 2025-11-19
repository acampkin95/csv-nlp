#!/usr/bin/env python3
"""
Pass Metrics System

Comprehensive metrics collection for pipeline pass execution.
Tracks performance, success rates, errors, and resource usage.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PassExecutionMetrics:
    """Metrics for a single pass execution"""
    pass_number: int
    pass_name: str
    start_time: float
    end_time: Optional[float] = None
    execution_time: float = 0.0
    success: bool = True
    cached: bool = False
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    items_processed: int = 0
    items_failed: int = 0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class PipelineMetrics:
    """Aggregate metrics for entire pipeline execution"""
    pipeline_id: str
    start_time: float
    end_time: Optional[float] = None
    total_execution_time: float = 0.0
    pass_metrics: Dict[int, PassExecutionMetrics] = field(default_factory=dict)
    total_passes: int = 0
    successful_passes: int = 0
    failed_passes: int = 0
    cached_passes: int = 0
    total_items_processed: int = 0
    total_items_failed: int = 0


class PassMetricsCollector:
    """
    Collects and aggregates metrics for pass execution.

    Features:
    - Real-time metrics collection
    - Historical metrics storage
    - Performance statistics
    - Error tracking
    - Resource usage monitoring
    """

    def __init__(self):
        """Initialize metrics collector"""
        self.current_pipeline: Optional[PipelineMetrics] = None
        self.historical_pipelines: List[PipelineMetrics] = []
        self.pass_statistics: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0,
                'errors': []
            }
        )

    def start_pipeline(self, pipeline_id: str = None) -> str:
        """
        Start tracking a new pipeline execution.

        Args:
            pipeline_id: Optional pipeline identifier

        Returns:
            str: Pipeline ID
        """
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_pipeline = PipelineMetrics(
            pipeline_id=pipeline_id,
            start_time=time.time()
        )

        logger.info(f"Started metrics collection for pipeline: {pipeline_id}")
        return pipeline_id

    def end_pipeline(self):
        """End tracking current pipeline"""
        if self.current_pipeline is None:
            logger.warning("No active pipeline to end")
            return

        self.current_pipeline.end_time = time.time()
        self.current_pipeline.total_execution_time = (
            self.current_pipeline.end_time - self.current_pipeline.start_time
        )

        # Calculate aggregate metrics
        self.current_pipeline.total_passes = len(self.current_pipeline.pass_metrics)
        self.current_pipeline.successful_passes = sum(
            1 for m in self.current_pipeline.pass_metrics.values() if m.success
        )
        self.current_pipeline.failed_passes = sum(
            1 for m in self.current_pipeline.pass_metrics.values() if not m.success
        )
        self.current_pipeline.cached_passes = sum(
            1 for m in self.current_pipeline.pass_metrics.values() if m.cached
        )
        self.current_pipeline.total_items_processed = sum(
            m.items_processed for m in self.current_pipeline.pass_metrics.values()
        )
        self.current_pipeline.total_items_failed = sum(
            m.items_failed for m in self.current_pipeline.pass_metrics.values()
        )

        # Store in history
        self.historical_pipelines.append(self.current_pipeline)

        logger.info(f"Ended metrics collection for pipeline: {self.current_pipeline.pipeline_id}")
        logger.info(f"  Total time: {self.current_pipeline.total_execution_time:.2f}s")
        logger.info(f"  Passes: {self.current_pipeline.successful_passes}/{self.current_pipeline.total_passes} successful")

    def start_pass(self, pass_number: int, pass_name: str) -> PassExecutionMetrics:
        """
        Start tracking a pass execution.

        Args:
            pass_number: Pass number
            pass_name: Pass name

        Returns:
            PassExecutionMetrics: Metrics object for this pass
        """
        if self.current_pipeline is None:
            logger.warning("No active pipeline, starting one automatically")
            self.start_pipeline()

        metrics = PassExecutionMetrics(
            pass_number=pass_number,
            pass_name=pass_name,
            start_time=time.time()
        )

        self.current_pipeline.pass_metrics[pass_number] = metrics
        return metrics

    def end_pass(
        self,
        pass_number: int,
        success: bool = True,
        cached: bool = False,
        error: Optional[str] = None,
        warnings: List[str] = None,
        items_processed: int = 0,
        items_failed: int = 0
    ):
        """
        End tracking a pass execution.

        Args:
            pass_number: Pass number
            success: Whether pass succeeded
            cached: Whether result was cached
            error: Optional error message
            warnings: Optional list of warnings
            items_processed: Number of items processed
            items_failed: Number of items that failed
        """
        if self.current_pipeline is None or pass_number not in self.current_pipeline.pass_metrics:
            logger.warning(f"No active metrics for pass {pass_number}")
            return

        metrics = self.current_pipeline.pass_metrics[pass_number]
        metrics.end_time = time.time()
        metrics.execution_time = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.cached = cached
        metrics.error = error
        metrics.warnings = warnings or []
        metrics.items_processed = items_processed
        metrics.items_failed = items_failed

        # Update pass statistics
        self._update_pass_statistics(metrics)

    def _update_pass_statistics(self, metrics: PassExecutionMetrics):
        """Update historical statistics for a pass"""
        stats = self.pass_statistics[metrics.pass_number]

        stats['total_executions'] += 1
        if metrics.success:
            stats['successful_executions'] += 1
        else:
            stats['failed_executions'] += 1
            stats['error_count'] += 1
            if metrics.error:
                stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': metrics.error
                })

        # Update timing statistics
        if not metrics.cached:  # Don't include cached executions in timing stats
            stats['total_execution_time'] += metrics.execution_time
            stats['min_execution_time'] = min(stats['min_execution_time'], metrics.execution_time)
            stats['max_execution_time'] = max(stats['max_execution_time'], metrics.execution_time)

            # Calculate average
            uncached_count = stats['total_executions'] - stats.get('cached_count', 0)
            if uncached_count > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / uncached_count

        # Track cache statistics
        if metrics.cached:
            stats['cached_count'] = stats.get('cached_count', 0) + 1

        if stats['total_executions'] > 0:
            stats['cache_hit_rate'] = stats.get('cached_count', 0) / stats['total_executions']

    def get_current_pipeline_metrics(self) -> Optional[PipelineMetrics]:
        """Get metrics for current pipeline"""
        return self.current_pipeline

    def get_pass_metrics(self, pass_number: int) -> Optional[PassExecutionMetrics]:
        """Get metrics for a specific pass in current pipeline"""
        if self.current_pipeline is None:
            return None
        return self.current_pipeline.pass_metrics.get(pass_number)

    def get_pass_statistics(self, pass_number: int) -> Dict[str, Any]:
        """Get historical statistics for a pass"""
        return dict(self.pass_statistics.get(pass_number, {}))

    def get_all_pass_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get historical statistics for all passes"""
        return {
            pass_num: dict(stats)
            for pass_num, stats in self.pass_statistics.items()
        }

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of current pipeline execution.

        Returns:
            Dict with pipeline summary
        """
        if self.current_pipeline is None:
            return {}

        return {
            'pipeline_id': self.current_pipeline.pipeline_id,
            'execution_time': self.current_pipeline.total_execution_time,
            'total_passes': self.current_pipeline.total_passes,
            'successful_passes': self.current_pipeline.successful_passes,
            'failed_passes': self.current_pipeline.failed_passes,
            'cached_passes': self.current_pipeline.cached_passes,
            'success_rate': (
                self.current_pipeline.successful_passes / self.current_pipeline.total_passes
                if self.current_pipeline.total_passes > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.current_pipeline.cached_passes / self.current_pipeline.total_passes
                if self.current_pipeline.total_passes > 0 else 0.0
            ),
            'items_processed': self.current_pipeline.total_items_processed,
            'items_failed': self.current_pipeline.total_items_failed,
            'pass_breakdown': {
                pass_num: {
                    'name': metrics.pass_name,
                    'execution_time': metrics.execution_time,
                    'success': metrics.success,
                    'cached': metrics.cached,
                    'items_processed': metrics.items_processed
                }
                for pass_num, metrics in self.current_pipeline.pass_metrics.items()
            }
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dict with performance statistics
        """
        total_pipelines = len(self.historical_pipelines)
        if total_pipelines == 0:
            return {'message': 'No pipeline executions recorded'}

        total_time = sum(p.total_execution_time for p in self.historical_pipelines)
        avg_time = total_time / total_pipelines

        return {
            'total_pipelines_executed': total_pipelines,
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'fastest_execution': min(p.total_execution_time for p in self.historical_pipelines),
            'slowest_execution': max(p.total_execution_time for p in self.historical_pipelines),
            'pass_statistics': self.get_all_pass_statistics(),
            'recent_pipelines': [
                {
                    'id': p.pipeline_id,
                    'execution_time': p.total_execution_time,
                    'success_rate': p.successful_passes / p.total_passes if p.total_passes > 0 else 0
                }
                for p in self.historical_pipelines[-10:]  # Last 10 pipelines
            ]
        }

    def export_metrics(self, format: str = 'dict') -> Any:
        """
        Export metrics in specified format.

        Args:
            format: Export format ('dict', 'json')

        Returns:
            Metrics in requested format
        """
        data = {
            'current_pipeline': self.get_pipeline_summary(),
            'pass_statistics': self.get_all_pass_statistics(),
            'performance_report': self.get_performance_report()
        }

        if format == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            return data

    def reset_statistics(self):
        """Reset all historical statistics"""
        self.current_pipeline = None
        self.historical_pipelines = []
        self.pass_statistics = defaultdict(
            lambda: {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0,
                'errors': []
            }
        )
        logger.info("Reset all metrics statistics")


# Global metrics collector instance
_global_metrics_collector = PassMetricsCollector()


def get_metrics_collector() -> PassMetricsCollector:
    """Get global metrics collector instance"""
    return _global_metrics_collector
