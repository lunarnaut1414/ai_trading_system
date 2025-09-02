"""
Database models for the AI Trading System.

This module exports all database models and the Base class for SQLAlchemy.
"""

from .base import Base, TimestampMixin

# Import all models - using try/except to handle missing files gracefully
try:
    from .trades import Trade
except ImportError:
    Trade = None

try:
    from .portfolio import Portfolio, Position
except ImportError:
    Portfolio = None
    Position = None

try:
    from .signals import Signal
except ImportError:
    Signal = None

try:
    from .agents import AgentAction, AgentDecision
except ImportError:
    AgentAction = None
    AgentDecision = None

try:
    from .alerts import Alert
except ImportError:
    Alert = None

try:
    from .performance import PerformanceMetric
except ImportError:
    PerformanceMetric = None

# Export all models and Base
__all__ = [
    'Base',
    'TimestampMixin',
    'Trade',
    'Position',
    'Portfolio',
    'Signal',
    'AgentAction',
    'AgentDecision',
    'Alert',
    'PerformanceMetric'
]
