"""
Agent-related models for the database.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from .base import Base, TimestampMixin
from datetime import datetime


class AgentAction(Base, TimestampMixin):
    __tablename__ = 'agent_actions'
    
    id = Column(Integer, primary_key=True)
    agent_name = Column(String(50), nullable=False)
    action_type = Column(String(50), nullable=False)
    details = Column(Text)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


class AgentDecision(Base, TimestampMixin):
    __tablename__ = 'agent_decisions'
    
    id = Column(Integer, primary_key=True)
    agent_name = Column(String(50), nullable=False)
    decision_type = Column(String(50))
    details = Column(Text)
    confidence = Column(Float)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
