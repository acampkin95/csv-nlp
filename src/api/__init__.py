"""
API Module for Message Processor
Provides unified REST and WebSocket endpoints for the Flask webapp
"""

from .unified_api import (
    create_api_blueprint,
    PersonManager,
    InteractionTracker,
    RelationshipAnalyzer,
)

__all__ = [
    'create_api_blueprint',
    'PersonManager',
    'InteractionTracker',
    'RelationshipAnalyzer',
]
