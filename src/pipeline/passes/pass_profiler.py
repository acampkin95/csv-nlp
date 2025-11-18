#!/usr/bin/env python3
"""
Pass Profiler (Feature 2)

Comprehensive profiling and benchmarking for pipeline passes.
Tracks CPU, memory, I/O, and provides detailed performance analysis.
"""

import logging
import time
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ProfileData:
    """Profile data for a single pass execution"""
    pass_number: int
    pass_name: str
    start_time: float
    end_time: float
    execution_time: float

    # CPU metrics
    cpu_percent_start: float
    cpu_percent_end: float
    cpu_time_user: float
    cpu_time_system: float

    # Memory metrics
    memory_mb_start: float
    memory_mb_end: float
    memory_mb_peak: float
    memory_mb_delta: float

    # I/O metrics (if available)
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0

    # Process metrics
    thread_count: int = 0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PassProfiler:
    """
    Profiler for detailed pass performance analysis.

    Features:
    - CPU usage tracking
    - Memory usage tracking
    - I/O monitoring
    - Thread count tracking
    - Custom metric support
    - Comparative analysis
    - Performance reports
    """

    def __init__(self):
        """Initialize profiler"""
        self.process = psutil.Process(os.getpid())
        self.profiles: List[ProfileData] = []
        self.current_profile: Optional[ProfileData] = None
        self.enabled = True

    def start_profiling(self, pass_number: int, pass_name: str) -> ProfileData:
        """
        Start profiling a pass.

        Args:
            pass_number: Pass number
            pass_name: Pass name

        Returns:
            ProfileData: Profile data object
        """
        if not self.enabled:
            return None

        # Get initial metrics
        cpu_times = self.process.cpu_times()
        memory_info = self.process.memory_info()

        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0.0

        self.current_profile = ProfileData(
            pass_number=pass_number,
            pass_name=pass_name,
            start_time=time.time(),
            end_time=0.0,
            execution_time=0.0,
            cpu_percent_start=cpu_percent,
            cpu_percent_end=0.0,
            cpu_time_user=cpu_times.user,
            cpu_time_system=cpu_times.system,
            memory_mb_start=memory_info.rss / 1024 / 1024,
            memory_mb_end=0.0,
            memory_mb_peak=memory_info.rss / 1024 / 1024,
            memory_mb_delta=0.0,
            thread_count=self.process.num_threads()
        )

        return self.current_profile

    def end_profiling(self):
        """End profiling current pass"""
        if not self.enabled or self.current_profile is None:
            return

        # Get final metrics
        cpu_times = self.process.cpu_times()
        memory_info = self.process.memory_info()

        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0.0

        self.current_profile.end_time = time.time()
        self.current_profile.execution_time = (
            self.current_profile.end_time - self.current_profile.start_time
        )
        self.current_profile.cpu_percent_end = cpu_percent
        self.current_profile.memory_mb_end = memory_info.rss / 1024 / 1024
        self.current_profile.memory_mb_delta = (
            self.current_profile.memory_mb_end - self.current_profile.memory_mb_start
        )

        # Track I/O if available
        try:
            io_counters = self.process.io_counters()
            # Note: These are cumulative, so would need to track delta
            self.current_profile.io_read_mb = io_counters.read_bytes / 1024 / 1024
            self.current_profile.io_write_mb = io_counters.write_bytes / 1024 / 1024
        except:
            pass  # I/O counters not available on all platforms

        # Store profile
        self.profiles.append(self.current_profile)
        self.current_profile = None

    def add_custom_metric(self, key: str, value: Any):
        """
        Add a custom metric to current profile.

        Args:
            key: Metric name
            value: Metric value
        """
        if self.current_profile is not None:
            self.current_profile.custom_metrics[key] = value

    def get_pass_profile(self, pass_number: int) -> Optional[ProfileData]:
        """
        Get most recent profile for a pass.

        Args:
            pass_number: Pass number

        Returns:
            ProfileData or None
        """
        for profile in reversed(self.profiles):
            if profile.pass_number == pass_number:
                return profile
        return None

    def get_all_profiles(self) -> List[ProfileData]:
        """Get all recorded profiles"""
        return self.profiles.copy()

    def get_comparative_report(self, pass_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate comparative performance report.

        Args:
            pass_numbers: Optional list of pass numbers to compare

        Returns:
            Dict with comparative analysis
        """
        if pass_numbers is None:
            profiles = self.profiles
        else:
            profiles = [p for p in self.profiles if p.pass_number in pass_numbers]

        if not profiles:
            return {'message': 'No profiles available'}

        # Aggregate by pass number
        pass_aggregates = {}
        for profile in profiles:
            if profile.pass_number not in pass_aggregates:
                pass_aggregates[profile.pass_number] = {
                    'pass_name': profile.pass_name,
                    'executions': [],
                    'avg_time': 0.0,
                    'avg_memory_delta': 0.0,
                    'avg_cpu_percent': 0.0
                }

            pass_aggregates[profile.pass_number]['executions'].append(profile)

        # Calculate averages
        for pass_num, data in pass_aggregates.items():
            execs = data['executions']
            data['count'] = len(execs)
            data['avg_time'] = sum(p.execution_time for p in execs) / len(execs)
            data['avg_memory_delta'] = sum(p.memory_mb_delta for p in execs) / len(execs)
            data['avg_cpu_percent'] = sum((p.cpu_percent_start + p.cpu_percent_end) / 2 for p in execs) / len(execs)
            data['min_time'] = min(p.execution_time for p in execs)
            data['max_time'] = max(p.execution_time for p in execs)

            # Remove executions list from output
            del data['executions']

        # Sort by execution time
        sorted_passes = sorted(
            pass_aggregates.items(),
            key=lambda x: x[1]['avg_time'],
            reverse=True
        )

        return {
            'total_profiles': len(profiles),
            'passes_analyzed': len(pass_aggregates),
            'slowest_passes': [
                {
                    'pass_number': num,
                    'pass_name': data['pass_name'],
                    'avg_time': data['avg_time'],
                    'executions': data['count']
                }
                for num, data in sorted_passes[:5]
            ],
            'memory_intensive_passes': sorted(
                [
                    {
                        'pass_number': num,
                        'pass_name': data['pass_name'],
                        'avg_memory_delta_mb': data['avg_memory_delta'],
                        'executions': data['count']
                    }
                    for num, data in pass_aggregates.items()
                ],
                key=lambda x: x['avg_memory_delta_mb'],
                reverse=True
            )[:5],
            'pass_details': pass_aggregates
        }

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Analyze pipeline bottlenecks.

        Returns:
            Dict with bottleneck analysis
        """
        if not self.profiles:
            return {'message': 'No profiles available'}

        # Find slowest individual execution
        slowest = max(self.profiles, key=lambda p: p.execution_time)

        # Find highest memory usage
        highest_memory = max(self.profiles, key=lambda p: p.memory_mb_delta)

        # Calculate total time by pass
        pass_times = {}
        for profile in self.profiles:
            if profile.pass_number not in pass_times:
                pass_times[profile.pass_number] = {
                    'name': profile.pass_name,
                    'total_time': 0.0,
                    'count': 0
                }
            pass_times[profile.pass_number]['total_time'] += profile.execution_time
            pass_times[profile.pass_number]['count'] += 1

        # Find pass consuming most total time
        if pass_times:
            most_time_consuming = max(
                pass_times.items(),
                key=lambda x: x[1]['total_time']
            )
        else:
            most_time_consuming = None

        return {
            'slowest_execution': {
                'pass_number': slowest.pass_number,
                'pass_name': slowest.pass_name,
                'execution_time': slowest.execution_time,
                'timestamp': datetime.fromtimestamp(slowest.start_time).isoformat()
            },
            'highest_memory_usage': {
                'pass_number': highest_memory.pass_number,
                'pass_name': highest_memory.pass_name,
                'memory_delta_mb': highest_memory.memory_mb_delta,
                'timestamp': datetime.fromtimestamp(highest_memory.start_time).isoformat()
            },
            'most_time_consuming_pass': {
                'pass_number': most_time_consuming[0],
                'pass_name': most_time_consuming[1]['name'],
                'total_time': most_time_consuming[1]['total_time'],
                'average_time': most_time_consuming[1]['total_time'] / most_time_consuming[1]['count'],
                'executions': most_time_consuming[1]['count']
            } if most_time_consuming else None
        }

    def export_profiles(self, filepath: str):
        """
        Export profiles to JSON file.

        Args:
            filepath: Output file path
        """
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_profiles': len(self.profiles),
            'profiles': [
                {
                    'pass_number': p.pass_number,
                    'pass_name': p.pass_name,
                    'timestamp': datetime.fromtimestamp(p.start_time).isoformat(),
                    'execution_time': p.execution_time,
                    'memory_mb_delta': p.memory_mb_delta,
                    'cpu_percent_avg': (p.cpu_percent_start + p.cpu_percent_end) / 2,
                    'custom_metrics': p.custom_metrics
                }
                for p in self.profiles
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.profiles)} profiles to {filepath}")

    def clear_profiles(self):
        """Clear all recorded profiles"""
        self.profiles = []
        self.current_profile = None
        logger.info("Cleared all profiles")

    def enable(self):
        """Enable profiling"""
        self.enabled = True

    def disable(self):
        """Disable profiling"""
        self.enabled = False


# Global profiler instance
_global_profiler = PassProfiler()


def get_profiler() -> PassProfiler:
    """Get global profiler instance"""
    return _global_profiler
