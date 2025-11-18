#!/usr/bin/env python3
"""
Pass Execution Visualizer (Feature 4)

Visualization and reporting tools for pipeline execution.
Generates execution timelines, dependency graphs, and performance charts.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PassVisualizer:
    """
    Visualizer for pass execution and performance.

    Features:
    - ASCII execution timeline
    - Dependency graph visualization
    - Performance comparison charts
    - Success/failure reports
    - HTML report generation
    """

    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'success': '\033[92m',  # Green
            'failure': '\033[91m',  # Red
            'warning': '\033[93m',  # Yellow
            'info': '\033[94m',     # Blue
            'reset': '\033[0m'
        }

    def generate_execution_timeline(
        self,
        pass_results: Dict[int, Any],
        show_timing: bool = True
    ) -> str:
        """
        Generate ASCII timeline of pass execution.

        Args:
            pass_results: Dict mapping pass number to results
            show_timing: Whether to show execution times

        Returns:
            str: ASCII timeline
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PIPELINE EXECUTION TIMELINE")
        lines.append("=" * 80)
        lines.append("")

        for pass_num in sorted(pass_results.keys()):
            result = pass_results[pass_num]

            # Determine status symbol and color
            if hasattr(result, 'success'):
                if result.success:
                    symbol = "✓"
                    color = self.colors['success']
                elif result.cached:
                    symbol = "◉"
                    color = self.colors['info']
                else:
                    symbol = "✗"
                    color = self.colors['failure']
            else:
                symbol = "?"
                color = self.colors['warning']

            # Build timeline entry
            entry = f"[Pass {pass_num:2d}] {symbol} "

            if hasattr(result, 'pass_name'):
                entry += f"{result.pass_name}"
            else:
                entry += f"Unknown Pass"

            if show_timing and hasattr(result, 'execution_time'):
                entry += f" ({result.execution_time:.2f}s)"

            if hasattr(result, 'cached') and result.cached:
                entry += " [CACHED]"

            lines.append(f"{color}{entry}{self.colors['reset']}")

            # Show error if present
            if hasattr(result, 'error') and result.error:
                lines.append(f"    └─ Error: {result.error}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_dependency_graph(self) -> str:
        """
        Generate ASCII dependency graph.

        Returns:
            str: ASCII dependency graph
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PASS DEPENDENCY GRAPH")
        lines.append("=" * 80)
        lines.append("")

        # Define dependencies
        dependencies = {
            1: [],
            2: [],
            3: [2],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [4, 5, 6, 7],
            9: [8],
            10: [2, 9],
            11: [],
            12: [11],
            13: [5, 11],
            14: [11],
            15: [8, 11, 13, 14]
        }

        pass_names = {
            1: "Data Validation",
            2: "Sentiment Analysis",
            3: "Emotional Dynamics",
            4: "Grooming Detection",
            5: "Manipulation Detection",
            6: "Deception Analysis",
            7: "Intent Classification",
            8: "Risk Assessment",
            9: "Timeline Analysis",
            10: "Contextual Insights",
            11: "Person Identification",
            12: "Interaction Mapping",
            13: "Gaslighting Detection",
            14: "Relationship Analysis",
            15: "Intervention Recommendations"
        }

        for pass_num in sorted(dependencies.keys()):
            deps = dependencies[pass_num]
            name = pass_names.get(pass_num, f"Pass {pass_num}")

            if deps:
                dep_str = ", ".join(str(d) for d in deps)
                lines.append(f"[{pass_num:2d}] {name:30s} ← depends on: {dep_str}")
            else:
                lines.append(f"[{pass_num:2d}] {name:30s} (no dependencies)")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_performance_chart(
        self,
        pass_metrics: Dict[int, float],
        title: str = "Pass Execution Times"
    ) -> str:
        """
        Generate ASCII bar chart of performance.

        Args:
            pass_metrics: Dict mapping pass number to execution time
            title: Chart title

        Returns:
            str: ASCII bar chart
        """
        if not pass_metrics:
            return "No metrics available"

        lines = []
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        lines.append("")

        # Find max value for scaling
        max_value = max(pass_metrics.values())
        max_bar_length = 50

        for pass_num in sorted(pass_metrics.keys()):
            value = pass_metrics[pass_num]

            # Calculate bar length
            if max_value > 0:
                bar_length = int((value / max_value) * max_bar_length)
            else:
                bar_length = 0

            # Create bar
            bar = "█" * bar_length

            lines.append(f"Pass {pass_num:2d} | {bar} {value:.2f}s")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_summary_report(
        self,
        pipeline_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate summary report of pipeline execution.

        Args:
            pipeline_metrics: Pipeline metrics dict

        Returns:
            str: Summary report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PIPELINE EXECUTION SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Overall metrics
        lines.append("Overall Metrics:")
        lines.append(f"  Pipeline ID: {pipeline_metrics.get('pipeline_id', 'N/A')}")
        lines.append(f"  Total Execution Time: {pipeline_metrics.get('execution_time', 0):.2f}s")
        lines.append(f"  Total Passes: {pipeline_metrics.get('total_passes', 0)}")
        lines.append(f"  Successful: {pipeline_metrics.get('successful_passes', 0)}")
        lines.append(f"  Failed: {pipeline_metrics.get('failed_passes', 0)}")
        lines.append(f"  Cached: {pipeline_metrics.get('cached_passes', 0)}")
        lines.append(f"  Success Rate: {pipeline_metrics.get('success_rate', 0) * 100:.1f}%")
        lines.append(f"  Cache Hit Rate: {pipeline_metrics.get('cache_hit_rate', 0) * 100:.1f}%")
        lines.append("")

        # Pass breakdown
        if 'pass_breakdown' in pipeline_metrics:
            lines.append("Pass Breakdown:")
            for pass_num, details in sorted(pipeline_metrics['pass_breakdown'].items()):
                status = "✓" if details.get('success', False) else "✗"
                cached = " [CACHED]" if details.get('cached', False) else ""
                time_str = f"{details.get('execution_time', 0):.2f}s"
                lines.append(
                    f"  [{pass_num:2d}] {status} {details.get('name', 'Unknown'):30s} "
                    f"{time_str:8s}{cached}"
                )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_html_report(
        self,
        pipeline_metrics: Dict[str, Any],
        output_file: str
    ):
        """
        Generate HTML report.

        Args:
            pipeline_metrics: Pipeline metrics
            output_file: Output HTML file path
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Execution Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .success {{
            color: #4CAF50;
        }}
        .failure {{
            color: #f44336;
        }}
        .cached {{
            color: #2196F3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipeline Execution Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Execution Time</div>
                <div class="metric-value">{pipeline_metrics.get('execution_time', 0):.2f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Passes</div>
                <div class="metric-value">{pipeline_metrics.get('total_passes', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{pipeline_metrics.get('success_rate', 0) * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value">{pipeline_metrics.get('cache_hit_rate', 0) * 100:.1f}%</div>
            </div>
        </div>

        <h2>Pass Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Pass #</th>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""

        if 'pass_breakdown' in pipeline_metrics:
            for pass_num, details in sorted(pipeline_metrics['pass_breakdown'].items()):
                status_class = "success" if details.get('success', False) else "failure"
                status_text = "✓ Success" if details.get('success', False) else "✗ Failed"
                cached_text = " (Cached)" if details.get('cached', False) else ""

                html += f"""
                <tr>
                    <td>{pass_num}</td>
                    <td>{details.get('name', 'Unknown')}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{details.get('execution_time', 0):.2f}s{cached_text}</td>
                    <td>{details.get('items_processed', 0)} items</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html)

        logger.info(f"Generated HTML report: {output_file}")


# Global visualizer instance
_global_visualizer = PassVisualizer()


def get_visualizer() -> PassVisualizer:
    """Get global visualizer instance"""
    return _global_visualizer
