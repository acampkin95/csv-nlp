#!/usr/bin/env python3
"""
Timeline & Context Analysis Passes (Passes 9-10)

Pass 9: Timeline reconstruction and pattern sequencing
Pass 10: Contextual insights and conversation flow
"""

import logging
from typing import Dict, List, Any

from .base_pass import BasePass, PassGroup

logger = logging.getLogger(__name__)


class Pass9_TimelineAnalysis(BasePass):
    """Pass 9: Timeline reconstruction and pattern sequencing"""

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=9,
            pass_name="Timeline Analysis",
            pass_group=PassGroup.TIMELINE,
            cache_manager=cache_manager,
            dependencies=['risk_assessment']
        )

    def _execute_pass(self, messages: List[Dict], risk_assessment: Dict, **kwargs) -> Dict[str, Any]:
        """Execute timeline analysis"""
        timeline_points = []

        for i, msg in enumerate(messages):
            try:
                text = msg.get('text', '')
                truncated_text = text[:100] + '...' if len(text) > 100 else text

                timeline_points.append({
                    'index': i,
                    'sender': msg.get('sender', 'Unknown'),
                    'timestamp': msg.get('timestamp', msg.get('date', '')),
                    'text': truncated_text
                })
            except Exception as e:
                logger.warning(f"  Timeline point extraction failed for message {i}: {e}")

        print(f"  Timeline Points Extracted: {len(timeline_points)}")

        return {
            'timeline_points': timeline_points,
            'conversation_duration': self._estimate_duration(messages),
            'pattern_sequences': self._identify_pattern_sequences(timeline_points, risk_assessment)
        }

    def _estimate_duration(self, messages: List[Dict]) -> str:
        """Estimate conversation duration"""
        if len(messages) < 2:
            return "unknown"

        first = messages[0].get('timestamp', messages[0].get('date', ''))
        last = messages[-1].get('timestamp', messages[-1].get('date', ''))

        if first and last and first != last:
            return f"{first} to {last}"
        elif len(messages) > 100:
            return "extended"
        elif len(messages) > 50:
            return "moderate"
        else:
            return "brief"

    def _identify_pattern_sequences(self, timeline_points: List[Dict], risk_assessment: Dict) -> List[Dict]:
        """Identify sequences of concerning patterns"""
        sequences = []
        # Stub for expansion - can be enhanced with pattern detection logic
        return sequences

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for timeline analysis failure"""
        return {
            'timeline_points': [],
            'conversation_duration': 'unknown',
            'pattern_sequences': [],
            'error': 'Timeline analysis failed'
        }


class Pass10_ContextualInsights(BasePass):
    """Pass 10: Contextual insights and conversation flow"""

    def __init__(self, cache_manager=None):
        super().__init__(
            pass_number=10,
            pass_name="Contextual Insights",
            pass_group=PassGroup.TIMELINE,
            cache_manager=cache_manager,
            dependencies=['sentiment_analysis', 'timeline_analysis']
        )

    def _execute_pass(
        self,
        messages: List[Dict],
        sentiment_results: Dict,
        timeline_analysis: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute contextual insights analysis"""
        insights = []

        if timeline_analysis.get('conversation_duration'):
            insights.append(f"Conversation spanning {timeline_analysis['conversation_duration']}")

        # Determine conversation flow complexity
        if len(messages) > 50:
            flow = 'complex'
        elif len(messages) > 20:
            flow = 'moderate'
        else:
            flow = 'simple'

        print(f"  Contextual Insights Generated: {len(insights)}")

        return {
            'insights': insights,
            'conversation_flow': flow
        }

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Fallback for contextual insights failure"""
        return {
            'insights': [],
            'conversation_flow': 'unknown',
            'error': 'Contextual insights analysis failed'
        }
