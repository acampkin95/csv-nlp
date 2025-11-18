#!/usr/bin/env python3
"""
Pass Result Comparator (Feature 5)

Tool for comparing and diffing pipeline execution results.
Useful for regression testing, A/B testing, and validating changes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from difflib import unified_diff

logger = logging.getLogger(__name__)


class PassResultComparator:
    """
    Comparator for pipeline execution results.

    Features:
    - Compare two pipeline executions
    - Detect differences in pass results
    - Performance regression detection
    - Result validation
    - Change reporting
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize comparator.

        Args:
            tolerance: Tolerance for numeric comparisons (default: 1%)
        """
        self.tolerance = tolerance

    def compare_pipelines(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two complete pipeline executions.

        Args:
            baseline: Baseline execution results
            current: Current execution results

        Returns:
            Dict with comparison results
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'overall_diff': {},
            'pass_diffs': {},
            'regressions': [],
            'improvements': [],
            'identical_passes': [],
            'summary': {}
        }

        # Compare overall metrics
        comparison['overall_diff'] = self._compare_metrics(
            baseline.get('overall_metrics', {}),
            current.get('overall_metrics', {})
        )

        # Compare individual passes
        baseline_passes = baseline.get('pass_results', {})
        current_passes = current.get('pass_results', {})

        all_pass_nums = set(baseline_passes.keys()) | set(current_passes.keys())

        for pass_num in sorted(all_pass_nums):
            baseline_result = baseline_passes.get(pass_num, {})
            current_result = current_passes.get(pass_num, {})

            diff = self._compare_pass_results(pass_num, baseline_result, current_result)
            comparison['pass_diffs'][pass_num] = diff

            # Categorize differences
            if diff['identical']:
                comparison['identical_passes'].append(pass_num)
            elif diff['performance_regression']:
                comparison['regressions'].append({
                    'pass_number': pass_num,
                    'baseline_time': baseline_result.get('execution_time', 0),
                    'current_time': current_result.get('execution_time', 0),
                    'regression_percent': diff.get('time_diff_percent', 0)
                })
            elif diff['performance_improvement']:
                comparison['improvements'].append({
                    'pass_number': pass_num,
                    'baseline_time': baseline_result.get('execution_time', 0),
                    'current_time': current_result.get('execution_time', 0),
                    'improvement_percent': abs(diff.get('time_diff_percent', 0))
                })

        # Generate summary
        comparison['summary'] = {
            'total_passes_compared': len(all_pass_nums),
            'identical_passes': len(comparison['identical_passes']),
            'passes_with_differences': len(comparison['pass_diffs']) - len(comparison['identical_passes']),
            'performance_regressions': len(comparison['regressions']),
            'performance_improvements': len(comparison['improvements']),
            'overall_performance_change': self._calculate_overall_performance_change(baseline, current)
        }

        return comparison

    def _compare_pass_results(
        self,
        pass_num: int,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare results from a single pass.

        Args:
            pass_num: Pass number
            baseline: Baseline results
            current: Current results

        Returns:
            Dict with differences
        """
        diff = {
            'pass_number': pass_num,
            'identical': False,
            'missing_in_baseline': False,
            'missing_in_current': False,
            'data_differences': [],
            'performance_regression': False,
            'performance_improvement': False,
            'time_diff_percent': 0.0
        }

        # Check if pass exists in both
        if not baseline and current:
            diff['missing_in_baseline'] = True
            return diff
        elif baseline and not current:
            diff['missing_in_current'] = True
            return diff
        elif not baseline and not current:
            diff['identical'] = True
            return diff

        # Compare execution times
        baseline_time = baseline.get('execution_time', 0)
        current_time = current.get('execution_time', 0)

        if baseline_time > 0:
            time_diff = ((current_time - baseline_time) / baseline_time) * 100
            diff['time_diff_percent'] = time_diff

            if time_diff > self.tolerance * 100:
                diff['performance_regression'] = True
            elif time_diff < -self.tolerance * 100:
                diff['performance_improvement'] = True

        # Compare data (simplified - would need deep comparison)
        baseline_data = baseline.get('data', {})
        current_data = current.get('data', {})

        # Compare keys
        baseline_keys = set(baseline_data.keys())
        current_keys = set(current_data.keys())

        missing_keys = baseline_keys - current_keys
        new_keys = current_keys - baseline_keys

        if missing_keys:
            diff['data_differences'].append(f"Missing keys: {missing_keys}")
        if new_keys:
            diff['data_differences'].append(f"New keys: {new_keys}")

        # Check if results are identical
        if not diff['data_differences'] and abs(diff['time_diff_percent']) < self.tolerance * 100:
            diff['identical'] = True

        return diff

    def _compare_metrics(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare overall metrics"""
        return {
            'execution_time_diff': current.get('execution_time', 0) - baseline.get('execution_time', 0),
            'success_rate_diff': current.get('success_rate', 0) - baseline.get('success_rate', 0),
            'cache_hit_rate_diff': current.get('cache_hit_rate', 0) - baseline.get('cache_hit_rate', 0)
        }

    def _calculate_overall_performance_change(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> float:
        """Calculate overall performance change percentage"""
        baseline_time = baseline.get('overall_metrics', {}).get('execution_time', 0)
        current_time = current.get('overall_metrics', {}).get('execution_time', 0)

        if baseline_time > 0:
            return ((current_time - baseline_time) / baseline_time) * 100
        return 0.0

    def generate_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison: Comparison results from compare_pipelines

        Returns:
            str: Formatted report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PIPELINE COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {comparison['timestamp']}")
        lines.append("")

        # Summary
        summary = comparison['summary']
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  Total Passes Compared: {summary['total_passes_compared']}")
        lines.append(f"  Identical Passes: {summary['identical_passes']}")
        lines.append(f"  Passes with Differences: {summary['passes_with_differences']}")
        lines.append(f"  Performance Regressions: {summary['performance_regressions']}")
        lines.append(f"  Performance Improvements: {summary['performance_improvements']}")
        lines.append(f"  Overall Performance Change: {summary['overall_performance_change']:+.2f}%")
        lines.append("")

        # Regressions
        if comparison['regressions']:
            lines.append("PERFORMANCE REGRESSIONS")
            lines.append("-" * 80)
            for reg in comparison['regressions']:
                lines.append(
                    f"  Pass {reg['pass_number']}: "
                    f"{reg['baseline_time']:.2f}s → {reg['current_time']:.2f}s "
                    f"({reg['regression_percent']:+.1f}%)"
                )
            lines.append("")

        # Improvements
        if comparison['improvements']:
            lines.append("PERFORMANCE IMPROVEMENTS")
            lines.append("-" * 80)
            for imp in comparison['improvements']:
                lines.append(
                    f"  Pass {imp['pass_number']}: "
                    f"{imp['baseline_time']:.2f}s → {imp['current_time']:.2f}s "
                    f"(-{imp['improvement_percent']:.1f}%)"
                )
            lines.append("")

        # Pass-by-pass differences
        lines.append("DETAILED PASS DIFFERENCES")
        lines.append("-" * 80)
        for pass_num, diff in sorted(comparison['pass_diffs'].items()):
            if diff['identical']:
                status = "✓ IDENTICAL"
            elif diff['missing_in_baseline']:
                status = "⊕ NEW"
            elif diff['missing_in_current']:
                status = "⊖ REMOVED"
            elif diff['performance_regression']:
                status = "⚠ REGRESSION"
            elif diff['performance_improvement']:
                status = "↑ IMPROVED"
            else:
                status = "≠ DIFFERENT"

            lines.append(f"  Pass {pass_num:2d}: {status}")

            if diff['data_differences']:
                for data_diff in diff['data_differences']:
                    lines.append(f"    - {data_diff}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_baseline(self, results: Dict[str, Any], filepath: str):
        """
        Save results as baseline for future comparisons.

        Args:
            results: Pipeline execution results
            filepath: Path to save baseline
        """
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

        with open(filepath, 'w') as f:
            json.dump(baseline, f, indent=2, default=str)

        logger.info(f"Saved baseline to {filepath}")

    def load_baseline(self, filepath: str) -> Dict[str, Any]:
        """
        Load baseline from file.

        Args:
            filepath: Path to baseline file

        Returns:
            Dict with baseline results
        """
        with open(filepath, 'r') as f:
            baseline = json.load(f)

        logger.info(f"Loaded baseline from {filepath}")
        return baseline.get('results', {})

    def validate_regression(
        self,
        comparison: Dict[str, Any],
        max_regression_percent: float = 10.0
    ) -> Tuple[bool, List[str]]:
        """
        Validate that there are no significant regressions.

        Args:
            comparison: Comparison results
            max_regression_percent: Maximum acceptable regression (%)

        Returns:
            Tuple of (valid, errors)
        """
        errors = []

        # Check overall performance
        overall_change = comparison['summary']['overall_performance_change']
        if overall_change > max_regression_percent:
            errors.append(
                f"Overall performance regression: {overall_change:.1f}% "
                f"(threshold: {max_regression_percent:.1f}%)"
            )

        # Check individual regressions
        for reg in comparison['regressions']:
            if reg['regression_percent'] > max_regression_percent:
                errors.append(
                    f"Pass {reg['pass_number']} regression: {reg['regression_percent']:.1f}% "
                    f"(threshold: {max_regression_percent:.1f}%)"
                )

        return (len(errors) == 0, errors)


# Global comparator instance
_global_comparator = PassResultComparator()


def get_comparator() -> PassResultComparator:
    """Get global comparator instance"""
    return _global_comparator
