"""
Signal model for the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime
from .base import Base, TimestampMixin
from datetime import datetime


class Signal(Base, TimestampMixin):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    signal_type = Column(String(20), nullable=False)
    strength = Column(Float)
    source = Column(String(50))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
